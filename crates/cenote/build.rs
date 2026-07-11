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
    // Linking libOpenImageDenoise (C++) makes ld extract ISPC archive
    // members it otherwise skips, and those need the C++ runtime the
    // shared library carries only transitively — name it explicitly.
    // Twice, because placement decides survival under --as-needed: the
    // link-lib propagates to downstream binaries (where this crate's
    // native libs land after the archives that need them), while this
    // crate's own test binaries put root-crate libs first — discarded —
    // so they also get it appended as a trailing linker arg.
    if env::var_os("CARGO_FEATURE_DENOISE").is_some()
        && env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux")
    {
        println!("cargo::rustc-link-lib=stdc++");
        println!("cargo::rustc-link-arg=-lstdc++");
    }
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("cargo sets OUT_DIR"));
    for kernel in KERNELS {
        let src = format!("shaders/{kernel}.slang");
        let dst = out_dir.join(format!("{kernel}.spv"));
        if let Err(message) = run_slangc(Path::new(&src), &dst) {
            panic!("slangc failed for {src}:\n{message}");
        }
    }
}
