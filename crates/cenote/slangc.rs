// The one `slangc` invocation. This file is `include!`d by both compile
// paths — `build.rs` (embed at build time) and `src/shaders.rs` (hot reload
// at run time) — so the flags and failure handling cannot drift.

/// Flags shared by every kernel compile. `-fvk-use-entrypoint-name` keeps the
/// Slang entry-point name in the SPIR-V instead of renaming it to `main`,
/// which matters once wavefront stages share a module (M1).
pub const SLANGC_ARGS: &[&str] = &["-target", "spirv", "-fvk-use-entrypoint-name"];

/// Compile `src` to SPIR-V at `dst` via the `slangc` subprocess.
///
/// # Errors
///
/// The compiler's diagnostics if `src` doesn't compile — or, if `slangc`
/// couldn't run at all, what went wrong launching it.
pub fn run_slangc(src: &std::path::Path, dst: &std::path::Path) -> Result<(), String> {
    let output = std::process::Command::new("slangc")
        .args(SLANGC_ARGS)
        .arg(src)
        .arg("-o")
        .arg(dst)
        .output();
    let output = match output {
        Ok(output) => output,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(
                "`slangc` not found on PATH — install a Slang release (see README quickstart)"
                    .to_owned(),
            );
        }
        Err(e) => return Err(format!("failed to run slangc: {e}")),
    };
    if output.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).into_owned())
    }
}
