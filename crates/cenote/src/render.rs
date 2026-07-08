//! Frame rendering: allocate the output buffer, dispatch the primary kernel
//! against the scene, read the pixels back. Orchestration only — all Vulkan
//! stays behind [`crate::gpu`].
//!
//! The [`Renderer`] owns the primary pipeline so hot reload can swap in a
//! recompiled kernel between frames. M0 renders exactly one frame per call,
//! blocking until it's done. M1's progressive accumulation loop replaces
//! [`Renderer::render`], not the modules it calls.

use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use crate::error::Result;
use crate::gpu::{ComputePipeline, Context, MemoryLocation};
use crate::scene::Scene;
use crate::shaders;

/// Workgroup width/height — must match `[numthreads(8, 8, 1)]` in
/// `shaders/primary.slang`.
const WORKGROUP_SIZE: u32 = 8;

/// Push constants for the primary kernel. Mirrors `struct Params` in
/// `shaders/primary.slang` field-for-field — one struct at the top of the
/// kernel names everything it reads. The scalars after each `Vec3` sit in
/// what std430 would otherwise spend on padding — field order is layout.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Device address of the output pixel buffer (`float4*` on the GPU side).
    pixels: vk::DeviceAddress,
    /// Device address of the scene's geometry lookup table.
    geometry: vk::DeviceAddress,
    camera_position: Vec3,
    width: u32,
    camera_right: Vec3,
    height: u32,
    camera_up: Vec3,
    _pad0: f32,
    camera_forward: Vec3,
    _pad1: f32,
}

/// The primary-visibility pipeline, ready to render frames. Created from the
/// embedded kernel; [`Renderer::reload`] swaps in hot-reloaded SPIR-V.
pub struct Renderer {
    pipeline: ComputePipeline,
}

impl Renderer {
    /// Create the renderer from the embedded primary kernel.
    ///
    /// # Errors
    ///
    /// Any [`crate::Error`] from pipeline creation.
    pub fn new(gpu: &Context) -> Result<Self> {
        Ok(Self {
            pipeline: create_pipeline(gpu, shaders::PRIMARY_SPIRV)?,
        })
    }

    /// Swap in a recompiled primary kernel; if pipeline creation fails, the
    /// current pipeline stays live. The entry-point name and the
    /// push-constant layout are pinned by the embedded build — hot reload
    /// covers kernel *body* edits; changing `Params` needs a `cargo build`.
    ///
    /// # Errors
    ///
    /// Any [`crate::Error`] from pipeline creation.
    pub fn reload(&mut self, gpu: &Context, spirv: &[u8]) -> Result<()> {
        self.pipeline = create_pipeline(gpu, spirv)?;
        Ok(())
    }

    /// Render one `width`×`height` frame of `scene` and return it as
    /// row-major RGBA `f32` with pixel (0, 0) top-left — the crate-wide
    /// image convention. Hits shade as the geometric normal mapped to color
    /// (0.5·n + 0.5), misses as black.
    ///
    /// # Errors
    ///
    /// Any [`crate::Error`] from buffer creation or submission.
    ///
    /// # Panics
    ///
    /// On a zero-sized target — callers validate their inputs, so this is a
    /// programmer bug.
    pub fn render(
        &self,
        gpu: &Context,
        scene: &Scene,
        width: u32,
        height: u32,
    ) -> Result<Vec<f32>> {
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

        let basis = scene.camera().basis(width as f32 / height as f32);
        let params = Params {
            pixels: pixels.device_address(),
            geometry: scene.geometry().device_address(),
            camera_position: scene.camera().position,
            width,
            camera_right: basis.right,
            height,
            camera_up: basis.up,
            _pad0: 0.0,
            camera_forward: basis.forward,
            _pad1: 0.0,
        };
        gpu.dispatch(
            &self.pipeline,
            scene.tlas(),
            bytemuck::bytes_of(&params),
            [
                width.div_ceil(WORKGROUP_SIZE),
                height.div_ceil(WORKGROUP_SIZE),
                1,
            ],
        )?;

        // pod_collect_to_vec rather than cast_slice: the downloaded bytes
        // carry no alignment guarantee.
        Ok(bytemuck::pod_collect_to_vec(&gpu.download_buffer(&pixels)?))
    }
}

fn create_pipeline(gpu: &Context, spirv: &[u8]) -> Result<ComputePipeline> {
    gpu.create_compute_pipeline(spirv, shaders::PRIMARY_ENTRY, size_of::<Params>() as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pixel(pixels: &[f32], width: u32, x: u32, y: u32) -> &[f32] {
        let idx = ((y * width + x) * 4) as usize;
        &pixels[idx..idx + 4]
    }

    /// The demo image shows the sphere and plane as normals, sky as black.
    /// Three probes pin the scene's known features:
    ///
    /// - top-left is sky — an exact miss color;
    /// - the image center looks straight at the sphere, so the hit facet's
    ///   normal points back at the camera (≈ +Z → blue-dominant);
    /// - bottom-center lands on the ground plane, whose geometric normal is
    ///   exactly +Y → color (0.5, 1, 0.5).
    #[test]
    fn demo_image_shows_normals_against_black_sky() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let scene = Scene::demo(&gpu).expect("demo scene");
        let renderer = Renderer::new(&gpu).expect("renderer");
        let (width, height) = (128, 128);
        let pixels = renderer
            .render(&gpu, &scene, width, height)
            .expect("render");

        assert_eq!(pixel(&pixels, width, 0, 0), [0.0, 0.0, 0.0, 1.0]);

        let center = pixel(&pixels, width, 64, 64);
        assert!(
            center[2] > 0.85,
            "sphere facet should face the camera, got {center:?}"
        );
        assert_eq!(center[3..], [1.0]);

        let bottom = pixel(&pixels, width, 64, 127);
        for (channel, expected) in bottom.iter().zip([0.5, 1.0, 0.5, 1.0]) {
            assert!(
                (channel - expected).abs() < 1e-3,
                "plane should shade as its +Y normal, got {bottom:?}"
            );
        }
    }

    /// Dimensions that aren't a multiple of the workgroup size exercise the
    /// kernel's bounds guard: partial workgroups must still write every
    /// in-bounds pixel (hit or miss, alpha is 1) without tripping validation
    /// on the ragged edge.
    #[test]
    fn ragged_dimensions_cover_every_pixel() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let scene = Scene::demo(&gpu).expect("demo scene");
        let renderer = Renderer::new(&gpu).expect("renderer");
        let pixels = renderer.render(&gpu, &scene, 33, 17).expect("render");
        for chunk in pixels.chunks_exact(4) {
            assert_eq!(chunk[3..], [1.0]);
        }
    }

    /// The hot-reload swap end to end, minus the file watch: recompile the
    /// unmodified kernel through the runtime `slangc` path, swap it in, and
    /// require a pixel-identical frame — same source, same compiler, same
    /// flags must mean the same image.
    #[test]
    fn reloaded_kernel_renders_identically() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let scene = Scene::demo(&gpu).expect("demo scene");
        let mut renderer = Renderer::new(&gpu).expect("renderer");
        let before = renderer.render(&gpu, &scene, 64, 64).expect("render");

        let spirv = shaders::recompile_primary().expect("recompile");
        renderer.reload(&gpu, &spirv).expect("reload");
        let after = renderer.render(&gpu, &scene, 64, 64).expect("render");

        assert_eq!(before, after);
    }
}
