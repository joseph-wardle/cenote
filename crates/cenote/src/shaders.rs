//! Compiled GPU kernels, embedded at build time.
//!
//! `build.rs` runs `slangc` over every kernel in `shaders/` and this module
//! embeds the resulting SPIR-V, so binaries are self-contained — no shader
//! files needed at run time. Runtime hot-reload (m0-plan §4 step 8) layers on
//! top: recompile the same sources through the same binary with the same
//! flags (decision D-004), swap pipelines only when compilation succeeds.

/// SPIR-V for the primary-visibility kernel (`shaders/primary.slang`).
/// Entry point: `primary`.
///
/// A byte slice with no alignment guarantee — convert to `u32` words at
/// pipeline-creation time (e.g. `ash::util::read_spv`).
pub const PRIMARY_SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/primary.spv"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_primary_kernel_is_spirv() {
        // SPIR-V words are little-endian; a valid module opens with the magic
        // number. This catches a broken/empty slangc invocation in GPU-less CI.
        let magic = u32::from_le_bytes(PRIMARY_SPIRV[..4].try_into().unwrap());
        assert_eq!(magic, 0x0723_0203);
    }
}
