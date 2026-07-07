//! Compiles every kernel in `shaders/` to SPIR-V with `slangc`, into `OUT_DIR`,
//! where `src/shaders.rs` embeds the results via `include_bytes!`.
//!
//! Runtime hot-reload (m0-plan §4 step 8) must invoke the same binary with the
//! same flags (decision D-004); when it lands, `SLANGC_ARGS` moves somewhere
//! both compile paths can share.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Kernel modules under `shaders/`, one entry point each, named after the file.
const KERNELS: &[&str] = &["primary"];

/// Flags shared by every kernel compile. `-fvk-use-entrypoint-name` keeps the
/// Slang entry-point name in the SPIR-V instead of renaming it to `main`,
/// which matters once wavefront stages share a module (M1).
const SLANGC_ARGS: &[&str] = &["-target", "spirv", "-fvk-use-entrypoint-name"];

fn main() {
    println!("cargo::rerun-if-changed=shaders");
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("cargo sets OUT_DIR"));
    for kernel in KERNELS {
        compile_kernel(kernel, &out_dir);
    }
}

fn compile_kernel(name: &str, out_dir: &Path) {
    let src = format!("shaders/{name}.slang");
    let dst = out_dir.join(format!("{name}.spv"));
    let output = match Command::new("slangc")
        .args(SLANGC_ARGS)
        .arg(&src)
        .arg("-o")
        .arg(&dst)
        .output()
    {
        Ok(output) => output,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            panic!("`slangc` not found on PATH — install a Slang release (see README quickstart)")
        }
        Err(e) => panic!("failed to run slangc: {e}"),
    };
    assert!(
        output.status.success(),
        "slangc failed for {src}:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
