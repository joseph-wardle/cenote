//! The pbrt parser: tokens in, directives out. The grammar is flat —
//! there is no nesting below a directive — but each directive has a fixed
//! positional shape (bare, N numbers, N names, or names + parameter list)
//! that must be known to parse it, so the arity table here *is* the
//! grammar. `Include`/`Import` splice their file's directives into the
//! stream through a stack of tokenizers, with every relative path
//! resolved against the top-level scene file's directory, exactly as
//! pbrt's own `SetSearchDirectory` does.
//!
//! Parameters (`"float roughness" [0.1]`) parse into typed values, and
//! every access marks its parameter consumed — the semantics layer warns
//! about whatever a directive left unconsumed, so a parameter this
//! importer doesn't understand is never silently dropped.

use std::cell::Cell;
use std::path::{Path, PathBuf};

use cenote::{Error, Result};

use crate::tokenize::{Span, Tokenizer};

/// Include depth at which a cycle is assumed. pbrt scenes nest two or
/// three deep (scene → geometry → materials).
const MAX_INCLUDE_DEPTH: usize = 32;

/// One parsed directive: keyword, its positional arguments, and its
/// parameter list.
#[derive(Debug)]
pub(crate) struct Directive {
    pub keyword: String,
    /// Positional names (a shape or material type, a texture's
    /// name/type/class triple, an object name).
    pub names: Vec<String>,
    /// Positional numbers (`Translate`'s three, `Transform`'s sixteen).
    pub numbers: Vec<f64>,
    pub params: Params,
    /// `file:line`, for diagnostics.
    pub location: String,
}

/// A parameter's values: pbrt lists are homogeneous.
#[derive(Debug)]
pub(crate) enum Value {
    Numbers(Vec<f64>),
    Strings(Vec<String>),
    Bools(Vec<bool>),
}

/// One `"type name" value…` parameter.
#[derive(Debug)]
pub(crate) struct Param {
    /// The declared type slug (`float`, `rgb`, `texture`, …), kept
    /// verbatim: slots that accept several types dispatch on it.
    pub ty: String,
    pub name: String,
    pub value: Value,
    pub location: String,
    consumed: Cell<bool>,
}

impl Param {
    /// The values as numbers.
    ///
    /// # Errors
    ///
    /// [`Error::SceneFormat`] when the values are strings or bools.
    pub fn as_floats(&self) -> Result<&[f64]> {
        match &self.value {
            Value::Numbers(numbers) => Ok(numbers),
            _ => Err(Error::SceneFormat(format!(
                "{}: parameter \"{}\" needs numeric values",
                self.location, self.name
            ))),
        }
    }

    /// The single string value.
    ///
    /// # Errors
    ///
    /// [`Error::SceneFormat`] when the value is not exactly one string.
    pub fn as_string(&self) -> Result<&str> {
        match &self.value {
            Value::Strings(strings) if strings.len() == 1 => Ok(&strings[0]),
            _ => Err(Error::SceneFormat(format!(
                "{}: parameter \"{}\" needs one string value",
                self.location, self.name
            ))),
        }
    }
}

/// A directive's parameter list, with consumption tracking.
#[derive(Debug, Default)]
pub(crate) struct Params(Vec<Param>);

impl Params {
    /// Look up `name`, checking its declared type against `types`, and
    /// mark it consumed. `None` when absent.
    ///
    /// # Errors
    ///
    /// [`Error::SceneFormat`] when the parameter exists with a type
    /// outside `types` — a typed mismatch renders wrong, so it refuses
    /// rather than warns.
    pub fn take(&self, name: &str, types: &[&str]) -> Result<Option<&Param>> {
        let Some(param) = self.0.iter().find(|param| param.name == name) else {
            return Ok(None);
        };
        param.consumed.set(true);
        if !types.contains(&param.ty.as_str()) {
            return Err(Error::SceneFormat(format!(
                "{}: parameter \"{name}\" is declared \"{}\", expected one of {types:?}",
                param.location, param.ty
            )));
        }
        Ok(Some(param))
    }

    /// A single float (an `integer` converts).
    ///
    /// # Errors
    ///
    /// [`Error::SceneFormat`] on a type or arity mismatch.
    pub fn float(&self, name: &str) -> Result<Option<f32>> {
        let Some(param) = self.take(name, &["float", "integer"])? else {
            return Ok(None);
        };
        match param.as_floats()? {
            [value] => Ok(Some(*value as f32)),
            _ => Err(Error::SceneFormat(format!(
                "{}: parameter \"{name}\" needs one value",
                param.location
            ))),
        }
    }

    /// A single integer.
    ///
    /// # Errors
    ///
    /// [`Error::SceneFormat`] on a type or arity mismatch.
    pub fn integer(&self, name: &str) -> Result<Option<i64>> {
        let Some(param) = self.take(name, &["integer"])? else {
            return Ok(None);
        };
        match param.as_floats()? {
            [value] => Ok(Some(*value as i64)),
            _ => Err(Error::SceneFormat(format!(
                "{}: parameter \"{name}\" needs one value",
                param.location
            ))),
        }
    }

    /// A single string.
    ///
    /// # Errors
    ///
    /// [`Error::SceneFormat`] on a type or arity mismatch.
    pub fn string(&self, name: &str) -> Result<Option<&str>> {
        let Some(param) = self.take(name, &["string", "texture"])? else {
            return Ok(None);
        };
        param.as_string().map(Some)
    }

    /// A single bool. Accepts pbrt-v3's quoted `"true"` spelling too.
    ///
    /// # Errors
    ///
    /// [`Error::SceneFormat`] on a type or arity mismatch.
    pub fn boolean(&self, name: &str) -> Result<Option<bool>> {
        let Some(param) = self.take(name, &["bool"])? else {
            return Ok(None);
        };
        match &param.value {
            Value::Bools(bools) if bools.len() == 1 => Ok(Some(bools[0])),
            Value::Strings(strings) if strings.len() == 1 && strings[0] == "true" => Ok(Some(true)),
            Value::Strings(strings) if strings.len() == 1 && strings[0] == "false" => {
                Ok(Some(false))
            }
            _ => Err(Error::SceneFormat(format!(
                "{}: parameter \"{name}\" needs one boolean value",
                param.location
            ))),
        }
    }

    /// Report every parameter nothing consumed — the "silence never means
    /// handled" backstop, called once per directive by the semantics
    /// layer.
    pub fn warn_unused(&self, context: &str, mut warn: impl FnMut(String)) {
        for param in &self.0 {
            if !param.consumed.get() {
                warn(format!(
                    "{}: {context}: parameter \"{}\" ({}) is not supported — ignored",
                    param.location, param.name, param.ty
                ));
            }
        }
    }
}

/// How many positional arguments a directive takes, and whether a
/// parameter list follows. This table is the pbrt grammar.
enum Arity {
    /// The keyword alone.
    Bare,
    /// Exactly this many numbers (brackets optional).
    Numbers(usize),
    /// Exactly this many names.
    Names(usize),
    /// Names, then a parameter list.
    NamesParams(usize),
}

fn arity(keyword: &str) -> Option<Arity> {
    Some(match keyword {
        "WorldBegin" | "WorldEnd" | "AttributeBegin" | "AttributeEnd" | "TransformBegin"
        | "TransformEnd" | "ObjectEnd" | "ReverseOrientation" | "Identity" => Arity::Bare,
        "Translate" | "Scale" => Arity::Numbers(3),
        "Rotate" => Arity::Numbers(4),
        "LookAt" => Arity::Numbers(9),
        "Transform" | "ConcatTransform" => Arity::Numbers(16),
        "TransformTimes" => Arity::Numbers(2),
        "Include" | "Import" | "ObjectBegin" | "ObjectInstance" | "NamedMaterial"
        | "CoordinateSystem" | "CoordSysTransform" | "ColorSpace" | "ActiveTransform" => {
            Arity::Names(1)
        }
        "MediumInterface" => Arity::Names(2),
        "Texture" => Arity::NamesParams(3),
        "Shape" | "Material" | "MakeNamedMaterial" | "MakeNamedMedium" | "LightSource"
        | "AreaLightSource" | "Camera" | "Film" | "Sampler" | "Integrator" | "PixelFilter"
        | "Accelerator" | "Option" | "Attribute" => Arity::NamesParams(1),
        _ => return None,
    })
}

/// One open file on the include stack.
struct SourceFile {
    path: PathBuf,
    tokenizer: Tokenizer,
    peeked: Option<Span>,
}

impl SourceFile {
    fn open(path: PathBuf) -> Result<Self> {
        let source = std::fs::read_to_string(&path).map_err(|error| {
            Error::SceneFormat(format!("can't read \"{}\": {error}", path.display()))
        })?;
        Ok(Self {
            path,
            tokenizer: Tokenizer::new(source),
            peeked: None,
        })
    }
}

/// The directive stream over a scene file and everything it includes.
pub(crate) struct Parser {
    stack: Vec<SourceFile>,
    /// The top-level scene file's directory: what every relative path in
    /// the scene resolves against (pbrt's search-directory rule).
    search_dir: PathBuf,
}

impl Parser {
    /// Open the top-level scene file.
    ///
    /// # Errors
    ///
    /// [`Error::SceneFormat`] when the file can't be read.
    pub fn open(path: &Path) -> Result<Self> {
        let absolute = std::path::absolute(path).map_err(|error| {
            Error::SceneFormat(format!("can't resolve \"{}\": {error}", path.display()))
        })?;
        let search_dir = absolute
            .parent()
            .unwrap_or_else(|| Path::new("/"))
            .to_owned();
        Ok(Self {
            stack: vec![SourceFile::open(absolute)?],
            search_dir,
        })
    }

    /// Resolve a path the pbrt way: relative means relative to the
    /// top-level scene file's directory, even from inside an include.
    pub fn resolve(&self, path: &str) -> PathBuf {
        let path = Path::new(path);
        if path.is_absolute() {
            path.to_owned()
        } else {
            self.search_dir.join(path)
        }
    }

    /// The next token, popping finished include files.
    fn next(&mut self) -> Result<Option<Span>> {
        loop {
            let Some(file) = self.stack.last_mut() else {
                return Ok(None);
            };
            if let Some(span) = file.peeked.take() {
                return Ok(Some(span));
            }
            match file.tokenizer.next_span() {
                Ok(Some(span)) => return Ok(Some(span)),
                Ok(None) => {
                    self.stack.pop();
                }
                Err(error) => return Err(self.contextualize(error)),
            }
        }
    }

    fn peek(&mut self) -> Result<Option<Span>> {
        let span = self.next()?;
        if let (Some(span), Some(file)) = (span, self.stack.last_mut()) {
            file.peeked = Some(span);
        }
        Ok(span)
    }

    /// The current file (valid while a token from it is in hand).
    fn file(&self) -> &SourceFile {
        self.stack.last().expect("a token implies an open file")
    }

    fn text(&self, span: Span) -> &str {
        self.file().tokenizer.text(span)
    }

    fn location(&self, span: Span) -> String {
        format!("{}:{}", self.file().path.display(), span.line)
    }

    /// Prefix a tokenizer error (which only knows its line) with the
    /// current file.
    fn contextualize(&self, error: Error) -> Error {
        match error {
            Error::SceneFormat(message) => {
                Error::SceneFormat(format!("{}:{message}", self.file().path.display()))
            }
            other => other,
        }
    }

    fn error(&self, span: Span, message: &str) -> Error {
        Error::SceneFormat(format!(
            "{}: {message} (at \"{}\")",
            self.location(span),
            self.text(span)
        ))
    }

    /// The next directive, with `Include`/`Import` spliced transparently.
    ///
    /// # Errors
    ///
    /// [`Error::SceneFormat`] naming file and line: an unknown directive,
    /// a malformed positional argument, or a malformed parameter.
    pub fn next_directive(&mut self) -> Result<Option<Directive>> {
        loop {
            let Some(span) = self.next()? else {
                return Ok(None);
            };
            if span.quoted || matches!(self.text(span), "[" | "]") {
                return Err(self.error(span, "expected a directive keyword"));
            }
            let keyword = self.text(span).to_owned();
            let location = self.location(span);
            let Some(arity) = arity(&keyword) else {
                return Err(self.error(span, "unknown directive"));
            };
            let mut directive = Directive {
                keyword,
                names: Vec::new(),
                numbers: Vec::new(),
                params: Params::default(),
                location,
            };
            match arity {
                Arity::Bare => {}
                Arity::Numbers(count) => self.parse_numbers(&mut directive, count)?,
                Arity::Names(count) => self.parse_names(&mut directive, count)?,
                Arity::NamesParams(count) => {
                    self.parse_names(&mut directive, count)?;
                    self.parse_params(&mut directive)?;
                }
            }
            // Includes splice: push the file and hand back its first
            // directive on the next loop.
            if matches!(directive.keyword.as_str(), "Include" | "Import") {
                if self.stack.len() >= MAX_INCLUDE_DEPTH {
                    return Err(Error::SceneFormat(format!(
                        "{}: includes nest deeper than {MAX_INCLUDE_DEPTH} — a cycle?",
                        directive.location
                    )));
                }
                let path = self.resolve(&directive.names[0]);
                self.stack.push(SourceFile::open(path)?);
                continue;
            }
            return Ok(Some(directive));
        }
    }

    /// `count` numbers, with one optional level of brackets around them.
    fn parse_numbers(&mut self, directive: &mut Directive, count: usize) -> Result<()> {
        let bracketed = match self.peek()? {
            Some(span) if !span.quoted && self.text(span) == "[" => {
                self.next()?;
                true
            }
            _ => false,
        };
        for _ in 0..count {
            let span = self.next()?.ok_or_else(|| eof(directive, "more numbers"))?;
            let number = self
                .text(span)
                .parse()
                .map_err(|_| self.error(span, &format!("{} expects numbers", directive.keyword)))?;
            directive.numbers.push(number);
        }
        if bracketed {
            let span = self.next()?.ok_or_else(|| eof(directive, "\"]\""))?;
            if span.quoted || self.text(span) != "]" {
                return Err(self.error(span, "expected \"]\""));
            }
        }
        Ok(())
    }

    /// `count` names — quoted for everything but `ActiveTransform`'s bare
    /// keyword argument, so a bare atom is accepted too.
    fn parse_names(&mut self, directive: &mut Directive, count: usize) -> Result<()> {
        for _ in 0..count {
            let span = self.next()?.ok_or_else(|| eof(directive, "a name"))?;
            if !span.quoted && matches!(self.text(span), "[" | "]") {
                return Err(self.error(span, "expected a name"));
            }
            directive.names.push(self.text(span).to_owned());
        }
        Ok(())
    }

    /// Parameters run until the next token is no longer a `"type name"`
    /// declaration (every declaration is a quoted string with whitespace,
    /// which nothing else in the grammar is).
    fn parse_params(&mut self, directive: &mut Directive) -> Result<()> {
        loop {
            let Some(span) = self.peek()? else {
                return Ok(());
            };
            if !span.quoted || !self.text(span).contains(char::is_whitespace) {
                return Ok(());
            }
            self.next()?;
            let location = self.location(span);
            let declaration = self.text(span);
            let mut words = declaration.split_ascii_whitespace();
            let (Some(ty), Some(name), None) = (words.next(), words.next(), words.next()) else {
                return Err(self.error(span, "expected a \"type name\" declaration"));
            };
            let (ty, name) = (ty.to_owned(), name.to_owned());
            let value = self.parse_value(directive, &name)?;
            directive.params.0.push(Param {
                ty,
                name,
                value,
                location,
                consumed: Cell::new(false),
            });
        }
    }

    /// One parameter's value: a bracketed homogeneous list or a single
    /// bare value.
    fn parse_value(&mut self, directive: &Directive, name: &str) -> Result<Value> {
        let first = self
            .next()?
            .ok_or_else(|| eof(directive, "a parameter value"))?;
        let bracketed = !first.quoted && self.text(first) == "[";
        let mut value = None;
        let mut push = |parser: &Self, span: Span| -> Result<()> {
            let text = parser.text(span);
            let value = value.get_or_insert_with(|| match (span.quoted, text) {
                (true, _) => Value::Strings(Vec::new()),
                (false, "true" | "false") => Value::Bools(Vec::new()),
                (false, _) => Value::Numbers(Vec::new()),
            });
            match value {
                Value::Strings(strings) if span.quoted => strings.push(text.to_owned()),
                Value::Bools(bools) if matches!(text, "true" | "false") && !span.quoted => {
                    bools.push(text == "true");
                }
                Value::Numbers(numbers) if !span.quoted => {
                    numbers.push(text.parse().map_err(|_| {
                        parser.error(span, &format!("parameter \"{name}\" expects numbers"))
                    })?);
                }
                _ => return Err(parser.error(span, "mixed value types in one parameter")),
            }
            Ok(())
        };
        if bracketed {
            loop {
                let span = self.next()?.ok_or_else(|| eof(directive, "\"]\""))?;
                if !span.quoted && self.text(span) == "]" {
                    break;
                }
                if !span.quoted && self.text(span) == "[" {
                    return Err(self.error(span, "nested brackets"));
                }
                push(self, span)?;
            }
        } else {
            push(self, first)?;
        }
        Ok(value.unwrap_or(Value::Numbers(Vec::new())))
    }
}

/// The end-of-input error for a directive still expecting arguments.
fn eof(directive: &Directive, expected: &str) -> Error {
    Error::SceneFormat(format!(
        "{}: {} ends at end of file, expected {expected}",
        directive.location, directive.keyword
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write `files` into a fresh fixture directory and parse the first
    /// one to completion.
    fn parse_files(test: &str, files: &[(&str, &str)]) -> Result<Vec<Directive>> {
        let dir = std::env::temp_dir().join(format!("cenote-pbrt-{test}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("fixture dir");
        for (name, source) in files {
            std::fs::write(dir.join(name), source).expect("write fixture");
        }
        let result = (|| {
            let mut parser = Parser::open(&dir.join(files[0].0))?;
            let mut directives = Vec::new();
            while let Some(directive) = parser.next_directive()? {
                directives.push(directive);
            }
            Ok(directives)
        })();
        std::fs::remove_dir_all(&dir).ok();
        result
    }

    fn parse_source(test: &str, source: &str) -> Result<Vec<Directive>> {
        parse_files(test, &[("scene.pbrt", source)])
    }

    #[test]
    fn every_arity_shape_parses() {
        let directives = parse_source(
            "arities",
            r#"
            LookAt 3 4 1.5  .5 .5 0  0 0 1
            Camera "perspective" "float fov" 45
            WorldBegin
            AttributeBegin
            Translate 0 0 -1
            Transform [1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1]
            Material "diffuse" "rgb reflectance" [0.2 0.4 0.6] "bool twosided" true
            Texture "checks" "spectrum" "checkerboard"
                "float uscale" [8] "rgb tex1" [.1 .1 .1]
            Shape "sphere" "float radius" 1
            AttributeEnd
            "#,
        )
        .expect("parses");
        let keywords: Vec<&str> = directives
            .iter()
            .map(|directive| directive.keyword.as_str())
            .collect();
        assert_eq!(
            keywords,
            [
                "LookAt",
                "Camera",
                "WorldBegin",
                "AttributeBegin",
                "Translate",
                "Transform",
                "Material",
                "Texture",
                "Shape",
                "AttributeEnd"
            ]
        );
        assert_eq!(directives[0].numbers.len(), 9);
        assert_eq!(directives[5].numbers.len(), 16);
        assert_eq!(directives[7].names, ["checks", "spectrum", "checkerboard"]);

        let material = &directives[6];
        assert_eq!(material.names, ["diffuse"]);
        let reflectance = material
            .params
            .take("reflectance", &["rgb"])
            .expect("typed")
            .expect("present");
        assert_eq!(reflectance.as_floats().expect("numeric"), [0.2, 0.4, 0.6]);
        assert_eq!(
            material.params.boolean("twosided").expect("typed"),
            Some(true)
        );
        let camera = &directives[1];
        assert_eq!(camera.params.float("fov").expect("typed"), Some(45.0));
    }

    #[test]
    fn includes_splice_and_resolve_against_the_scene_dir() {
        let directives = parse_files(
            "includes",
            &[
                (
                    "scene.pbrt",
                    "WorldBegin\nInclude \"geometry.pbrt\"\nAttributeEnd\n",
                ),
                (
                    "geometry.pbrt",
                    "AttributeBegin\nShape \"sphere\" \"float radius\" 2\n",
                ),
            ],
        )
        .expect("parses");
        let keywords: Vec<&str> = directives
            .iter()
            .map(|directive| directive.keyword.as_str())
            .collect();
        // The include's directives land in the stream in place; parsing
        // continues in the outer file afterward.
        assert_eq!(
            keywords,
            ["WorldBegin", "AttributeBegin", "Shape", "AttributeEnd"]
        );
    }

    #[test]
    fn unconsumed_parameters_surface_by_name() {
        let directives = parse_source(
            "unused",
            "Material \"diffuse\" \"rgb reflectance\" [1 1 1] \"float sheen\" 0.5\n",
        )
        .expect("parses");
        let params = &directives[0].params;
        params
            .take("reflectance", &["rgb"])
            .expect("typed")
            .expect("present");
        let mut warnings = Vec::new();
        params.warn_unused("material \"diffuse\"", |warning| warnings.push(warning));
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("sheen"), "{}", warnings[0]);
        assert!(
            warnings[0].contains("scene.pbrt:1"),
            "location: {}",
            warnings[0]
        );
    }

    #[test]
    fn type_mismatches_are_errors_not_warnings() {
        let directives = parse_source(
            "mismatch",
            "Material \"diffuse\" \"string reflectance\" \"red\"\n",
        )
        .expect("parses");
        let error = directives[0]
            .params
            .take("reflectance", &["rgb"])
            .unwrap_err();
        assert!(error.to_string().contains("reflectance"), "{error}");
        assert!(error.to_string().contains("string"), "{error}");
    }

    #[test]
    fn malformed_scenes_fail_with_file_and_line() {
        let error = parse_source("unknown", "WorldBegin\nFrobnicate 1 2\n").unwrap_err();
        assert!(error.to_string().contains("unknown directive"), "{error}");
        assert!(error.to_string().contains("scene.pbrt:2"), "{error}");

        let error = parse_source("short", "Translate 1 2\n").unwrap_err();
        assert!(error.to_string().contains("end of file"), "{error}");

        let error =
            parse_source("mixed", "Shape \"sphere\" \"float radius\" [1 \"two\"]\n").unwrap_err();
        assert!(error.to_string().contains("mixed"), "{error}");

        let error = parse_source("missing", "Include \"nowhere.pbrt\"\n").unwrap_err();
        assert!(error.to_string().contains("nowhere.pbrt"), "{error}");
    }

    #[test]
    fn include_cycles_are_cut_off() {
        let error =
            parse_files("cycle", &[("scene.pbrt", "Include \"scene.pbrt\"\n")]).unwrap_err();
        assert!(error.to_string().contains("cycle"), "{error}");
    }
}
