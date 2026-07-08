//! Compiles every kernel in `shaders/` to SPIR-V with `slangc`, into `OUT_DIR`,
//! where `src/shaders.rs` embeds the results via `include_bytes!`.
//!
//! The invocation and the kernel list live in `slangc.rs`, shared with the
//! runtime hot-reload path so neither can drift.

use std::env;
use std::path::{Path, PathBuf};

include!("slangc.rs");

fn main() {
    println!("cargo::rerun-if-changed=shaders");
    println!("cargo::rerun-if-changed=slangc.rs");
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("cargo sets OUT_DIR"));
    for kernel in KERNELS {
        let src = format!("shaders/{kernel}.slang");
        let dst = out_dir.join(format!("{kernel}.spv"));
        if let Err(message) = run_slangc(Path::new(&src), &dst) {
            panic!("slangc failed for {src}:\n{message}");
        }
    }
}
