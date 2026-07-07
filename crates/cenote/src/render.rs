//! Frame rendering: allocate the output buffer, dispatch the primary kernel,
//! read the pixels back. Orchestration only — all Vulkan stays behind
//! [`crate::gpu`] (decision D-005).
//!
//! M0 renders exactly one frame per call, blocking until it's done (D-007).
//! M1's progressive accumulation loop replaces this function, not the
//! modules it calls.

use ash::vk;
use bytemuck::{Pod, Zeroable};

use crate::error::Result;
use crate::gpu::{Context, MemoryLocation};
use crate::shaders;

/// Workgroup width/height — must match `[numthreads(8, 8, 1)]` in
/// `shaders/primary.slang`.
const WORKGROUP_SIZE: u32 = 8;

/// Push constants for the primary kernel. Mirrors `struct Params` in
/// `shaders/primary.slang` field-for-field (D-006: one struct at the top of
/// the kernel names everything it reads).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Device address of the output pixel buffer (`float4*` on the GPU side).
    pixels: vk::DeviceAddress,
    width: u32,
    height: u32,
}

/// Render one `width`×`height` frame and return it as row-major RGBA `f32`
/// with pixel (0, 0) top-left — the crate-wide image convention.
///
/// # Errors
///
/// Any [`crate::Error`] from buffer creation, pipeline creation, or
/// submission.
///
/// # Panics
///
/// On a zero-sized target — callers validate their inputs, so this is a
/// programmer bug (D-010).
pub fn render(gpu: &Context, width: u32, height: u32) -> Result<Vec<f32>> {
    assert!(width > 0 && height > 0, "zero-sized render target");

    let size = u64::from(width) * u64::from(height) * 4 * size_of::<f32>() as u64;
    let pixels = gpu.create_buffer(
        "render.pixels",
        size,
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_SRC,
        MemoryLocation::GpuOnly,
    )?;
    let pipeline = gpu.create_compute_pipeline(
        shaders::PRIMARY_SPIRV,
        shaders::PRIMARY_ENTRY,
        size_of::<Params>() as u32,
    )?;

    let params = Params {
        pixels: pixels.device_address(),
        width,
        height,
    };
    gpu.dispatch(
        &pipeline,
        bytemuck::bytes_of(&params),
        [
            width.div_ceil(WORKGROUP_SIZE),
            height.div_ceil(WORKGROUP_SIZE),
            1,
        ],
    )?;

    // pod_collect_to_vec rather than cast_slice: the downloaded bytes carry
    // no alignment guarantee.
    Ok(bytemuck::pod_collect_to_vec(&gpu.download_buffer(&pixels)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pixel(pixels: &[f32], width: u32, x: u32, y: u32) -> &[f32] {
        let idx = ((y * width + x) * 4) as usize;
        &pixels[idx..idx + 4]
    }

    /// The step-5 checkpoint: the fill kernel's UV gradient arrives on the
    /// host intact. Power-of-two dimensions make the shader's UV divisions
    /// exact, so the comparison is bitwise.
    #[test]
    fn dispatch_produces_uv_gradient() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let (width, height) = (64, 32);
        let pixels = render(&gpu, width, height).expect("render");
        assert_eq!(pixels.len(), (width * height * 4) as usize);
        assert_eq!(pixel(&pixels, width, 0, 0), [0.0, 0.0, 0.25, 1.0]);
        assert_eq!(pixel(&pixels, width, 32, 16), [0.5, 0.5, 0.25, 1.0]);
        assert_eq!(
            pixel(&pixels, width, 63, 31),
            [63.0 / 64.0, 31.0 / 32.0, 0.25, 1.0]
        );
    }

    /// Dimensions that aren't a multiple of the workgroup size exercise the
    /// kernel's bounds guard: partial workgroups must still write every
    /// in-bounds pixel (blue = 0.25, alpha = 1 everywhere) without tripping
    /// validation on the ragged edge.
    #[test]
    fn ragged_dimensions_cover_every_pixel() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let (width, height) = (33, 17);
        let pixels = render(&gpu, width, height).expect("render");
        for chunk in pixels.chunks_exact(4) {
            assert_eq!(chunk[2..], [0.25, 1.0]);
        }
    }
}
