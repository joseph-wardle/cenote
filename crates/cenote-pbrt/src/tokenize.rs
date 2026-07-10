//! The pbrt lexer. A `.pbrt` file is a flat stream of four token shapes —
//! quoted strings, `[`, `]`, and bare atoms (numbers, directive keywords,
//! `true`/`false`) — separated by whitespace, with `#` comments running to
//! end of line. Strings carry no escape sequences; `[` and `]` are
//! self-delimiting. Tokens come out as spans into the source (an imported
//! scene can carry megabytes of inline geometry, so the lexer allocates
//! nothing per token) with the line number every parse diagnostic needs.

use cenote::{Error, Result};

/// One token: a byte range into the source, whether it was quoted (the
/// range excludes the quotes), and the 1-based line it started on.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Span {
    start: usize,
    end: usize,
    /// True for `"quoted strings"` — the parser needs the distinction to
    /// tell a parameter declaration from a directive keyword.
    pub quoted: bool,
    /// 1-based source line, for diagnostics.
    pub line: u32,
}

/// A tokenizer over one file's source, owned so an include stack can hold
/// any number of them alive at once.
pub(crate) struct Tokenizer {
    source: String,
    offset: usize,
    line: u32,
}

impl Tokenizer {
    pub fn new(source: String) -> Self {
        Self {
            source,
            offset: 0,
            line: 1,
        }
    }

    /// The token's text (for quoted tokens, the content between the
    /// quotes).
    pub fn text(&self, span: Span) -> &str {
        &self.source[span.start..span.end]
    }

    /// The next token, or `None` at end of input.
    ///
    /// # Errors
    ///
    /// [`Error::SceneFormat`] on an unterminated string.
    pub fn next_span(&mut self) -> Result<Option<Span>> {
        let bytes = self.source.as_bytes();
        // Skip whitespace and comments, counting lines.
        while let Some(&byte) = bytes.get(self.offset) {
            match byte {
                b'\n' => {
                    self.line += 1;
                    self.offset += 1;
                }
                byte if byte.is_ascii_whitespace() => self.offset += 1,
                b'#' => {
                    while bytes.get(self.offset).is_some_and(|&byte| byte != b'\n') {
                        self.offset += 1;
                    }
                }
                _ => break,
            }
        }
        let Some(&byte) = bytes.get(self.offset) else {
            return Ok(None);
        };
        let line = self.line;
        let start = self.offset;
        let span = match byte {
            b'[' | b']' => {
                self.offset += 1;
                Span {
                    start,
                    end: self.offset,
                    quoted: false,
                    line,
                }
            }
            b'"' => {
                self.offset += 1;
                let content = self.offset;
                loop {
                    match bytes.get(self.offset) {
                        Some(b'"') => break,
                        Some(b'\n') => {
                            self.line += 1;
                            self.offset += 1;
                        }
                        Some(_) => self.offset += 1,
                        None => {
                            return Err(Error::SceneFormat(format!(
                                "line {line}: unterminated string"
                            )));
                        }
                    }
                }
                let end = self.offset;
                self.offset += 1;
                Span {
                    start: content,
                    end,
                    quoted: true,
                    line,
                }
            }
            _ => {
                while bytes.get(self.offset).is_some_and(|&byte| {
                    !byte.is_ascii_whitespace() && !matches!(byte, b'[' | b']' | b'"' | b'#')
                }) {
                    self.offset += 1;
                }
                Span {
                    start,
                    end: self.offset,
                    quoted: false,
                    line,
                }
            }
        };
        Ok(Some(span))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_tokens(source: &str) -> Vec<(String, bool, u32)> {
        let mut tokenizer = Tokenizer::new(source.to_owned());
        let mut tokens = Vec::new();
        while let Some(span) = tokenizer.next_span().expect("tokenizes") {
            tokens.push((tokenizer.text(span).to_owned(), span.quoted, span.line));
        }
        tokens
    }

    #[test]
    fn the_four_shapes_tokenize_with_lines() {
        let tokens = all_tokens(
            "# a comment\nShape \"trianglemesh\" # trailing\n  \"integer indices\" [0 1 2]\n",
        );
        let expected = [
            ("Shape", false, 2),
            ("trianglemesh", true, 2),
            ("integer indices", true, 3),
            ("[", false, 3),
            ("0", false, 3),
            ("1", false, 3),
            ("2", false, 3),
            ("]", false, 3),
        ];
        assert_eq!(tokens.len(), expected.len());
        for (token, (text, quoted, line)) in tokens.iter().zip(expected) {
            assert_eq!(token, &(text.to_owned(), quoted, line));
        }
    }

    #[test]
    fn brackets_and_quotes_self_delimit() {
        let tokens = all_tokens("Transform[1 2]\"a b\"[3]");
        let texts: Vec<&str> = tokens.iter().map(|(text, ..)| text.as_str()).collect();
        assert_eq!(
            texts,
            ["Transform", "[", "1", "2", "]", "a b", "[", "3", "]"]
        );
        assert!(tokens[5].1, "the string is marked quoted");
    }

    #[test]
    fn an_unterminated_string_is_an_error_with_its_line() {
        let mut tokenizer = Tokenizer::new("Translate 1 2 3\n\"oops".to_owned());
        for _ in 0..4 {
            tokenizer.next_span().expect("tokenizes").expect("token");
        }
        let error = tokenizer.next_span().unwrap_err();
        assert!(error.to_string().contains("line 2"), "{error}");
        assert!(error.to_string().contains("unterminated"), "{error}");
    }

    #[test]
    fn empty_and_comment_only_input_ends_cleanly() {
        assert!(all_tokens("").is_empty());
        assert!(all_tokens("# nothing\n# at all").is_empty());
    }
}
