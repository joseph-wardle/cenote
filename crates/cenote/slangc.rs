// The one `slangc` invocation. This file is `include!`d by both compile
// paths — `build.rs` (embed at build time) and `src/shaders.rs` (hot reload
// at run time) — so the flags and failure handling cannot drift.

/// Every kernel under `shaders/`, one compute entry point each, named after
/// its file. The build embeds them in this order and hot reload recompiles
/// them in this order. Not listed: shared modules (`pathstate.slang`,
/// `rng.slang`, `openpbr.slang`, `lights.slang`, `scene.slang`) compile
/// into their importers, and test-only fixtures (`rng_test.slang`) compile
/// at test time.
pub const KERNELS: [&str; 7] = [
    "raygen",
    "intersect",
    "shade_miss",
    "shade_surface",
    "trace_shadow",
    "accumulate",
    "tonemap",
];

/// Flags shared by every kernel compile. `-fvk-use-entrypoint-name` keeps the
/// Slang entry-point name in the SPIR-V instead of renaming it to `main` —
/// load-bearing now that wavefront stages share the `pathstate` module.
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
