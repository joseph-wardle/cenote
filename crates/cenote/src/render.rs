//! Frame orchestration: dispatch kernels against the scene and manage the
//! film. Orchestration only — all Vulkan stays behind [`crate::gpu`].
//!
//! Two paths share the primary kernel:
//!
//! - **One-shot** ([`Renderer::render`]): allocate a buffer, render one
//!   frame, read the linear pixels back — the CLI and test path.
//! - **Progressive** ([`Renderer::accumulate`] + [`Renderer::tonemap`]):
//!   each `accumulate` traces one sample into the [`Film`]'s running sums;
//!   `tonemap` averages, exposes, and applies the ACES display transform
//!   into the RGBA8 display buffer that [`crate::gpu::Presenter::present`]
//!   shows. The viewer's redraw loop is one of each per frame.
//!
//! The M0 primary kernel is deterministic, so accumulating it refines
//! nothing yet — the progressive path proves the display plumbing in
//! isolation (M1 build step 4) until the wavefront engine (step 5) and
//! sample jitter (step 6) make every sample count.

use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use crate::error::Result;
use crate::gpu::{Bindings, Buffer, ComputePipeline, Context, MemoryLocation};
use crate::scene::Scene;
use crate::shaders;

/// Workgroup width/height — must match `[numthreads(8, 8, 1)]` in every
/// kernel under `shaders/`.
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

/// Push constants for the accumulation kernel; mirrors `struct Params` in
/// `shaders/accumulate.slang`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AccumulateParams {
    /// Device address of the new sample (`float4*`).
    frame: vk::DeviceAddress,
    /// Device address of the film's running sums (`float4*`).
    sum: vk::DeviceAddress,
    width: u32,
    height: u32,
    /// Bool: overwrite the sums instead of adding — the first sample after
    /// a reset is the clear.
    reset: u32,
    _pad0: u32,
}

/// Push constants for the tonemap kernel; mirrors `struct Params` in
/// `shaders/tonemap.slang`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TonemapParams {
    /// Device address of the film's running sums (`float4*`).
    sum: vk::DeviceAddress,
    /// Device address of the packed RGBA8 display buffer (`uint*`).
    display: vk::DeviceAddress,
    width: u32,
    height: u32,
    /// `exp2(exposure stops) / sample count` — average and exposure folded
    /// into one multiply.
    scale: f32,
    _pad0: f32,
}

/// Progressive accumulation state for one render-target size: per-pixel
/// linear RGBA f32 sums, the sample the current wave writes, and the
/// tonemapped RGBA8 frame the presenter shows. The sample count lives on
/// the host — it is uniform across pixels by construction.
///
/// Sized at creation; a resize means a new `Film`. A view change means
/// [`Film::reset`].
pub struct Film {
    /// One sample's radiance, written by the primary kernel each wave and
    /// consumed by the accumulation kernel.
    sample: Buffer,
    /// The running sums. `TRANSFER_SRC` so the accumulated image can be
    /// read back (tests now, batch EXR output in later steps).
    sum: Buffer,
    /// The tonemap kernel's output: packed RGBA8, sRGB-encoded — exactly
    /// what [`crate::gpu::Presenter::present`] expects.
    display: Buffer,
    width: u32,
    height: u32,
    samples: u32,
}

impl Film {
    /// Create a film for `width`×`height` renders. Starts empty: the first
    /// [`Renderer::accumulate`] initializes the sums, so no clear pass runs.
    ///
    /// # Errors
    ///
    /// Any [`crate::Error`] from buffer creation.
    ///
    /// # Panics
    ///
    /// On zero dimensions — callers validate their inputs, so this is a
    /// programmer bug.
    pub fn new(gpu: &Context, width: u32, height: u32) -> Result<Self> {
        assert!(width > 0 && height > 0, "zero-sized film");
        let texels = u64::from(width) * u64::from(height);
        let storage =
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        Ok(Self {
            sample: gpu.create_buffer(
                "film.sample",
                texels * 16,
                storage,
                MemoryLocation::GpuOnly,
            )?,
            sum: gpu.create_buffer(
                "film.sum",
                texels * 16,
                storage | vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::GpuOnly,
            )?,
            display: gpu.create_buffer(
                "film.display",
                texels * 4,
                storage | vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::GpuOnly,
            )?,
            width,
            height,
            samples: 0,
        })
    }

    /// Start over (the view changed): the next sample overwrites the sums
    /// instead of adding, so nothing needs clearing now.
    pub fn reset(&mut self) {
        self.samples = 0;
    }

    /// Samples accumulated since creation or the last [`Film::reset`].
    #[must_use]
    pub fn samples(&self) -> u32 {
        self.samples
    }

    /// The tonemapped frame, valid after [`Renderer::tonemap`] — hand it to
    /// [`crate::gpu::Presenter::present`].
    #[must_use]
    pub fn display(&self) -> &Buffer {
        &self.display
    }

    /// Width in pixels.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }
}

/// The kernel pipelines, ready to render frames. Created from the embedded
/// kernels; [`Renderer::reload`] swaps in hot-reloaded SPIR-V.
pub struct Renderer {
    primary: ComputePipeline,
    accumulate: ComputePipeline,
    tonemap: ComputePipeline,
}

impl Renderer {
    /// Create the renderer from the embedded kernels.
    ///
    /// # Errors
    ///
    /// Any [`crate::Error`] from pipeline creation.
    pub fn new(gpu: &Context) -> Result<Self> {
        Ok(Self {
            primary: create_primary_pipeline(gpu, shaders::PRIMARY_SPIRV)?,
            accumulate: gpu.create_compute_pipeline(
                shaders::ACCUMULATE_SPIRV,
                shaders::ACCUMULATE_ENTRY,
                size_of::<AccumulateParams>() as u32,
                Bindings::None,
            )?,
            tonemap: gpu.create_compute_pipeline(
                shaders::TONEMAP_SPIRV,
                shaders::TONEMAP_ENTRY,
                size_of::<TonemapParams>() as u32,
                Bindings::None,
            )?,
        })
    }

    /// Swap in a recompiled primary kernel; if pipeline creation fails, the
    /// current pipeline stays live. The entry-point name and the
    /// push-constant layout are pinned by the embedded build — hot reload
    /// covers kernel *body* edits; changing `Params` needs a `cargo build`.
    /// The accumulate and tonemap kernels aren't reloadable yet; the
    /// wavefront engine's kernel registry (M1 step 5) generalizes this.
    ///
    /// # Errors
    ///
    /// Any [`crate::Error`] from pipeline creation.
    pub fn reload(&mut self, gpu: &Context, spirv: &[u8]) -> Result<()> {
        self.primary = create_primary_pipeline(gpu, spirv)?;
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
        let pixels = self.render_to_buffer(gpu, scene, width, height)?;
        // pod_collect_to_vec rather than cast_slice: the downloaded bytes
        // carry no alignment guarantee.
        Ok(bytemuck::pod_collect_to_vec(&gpu.download_buffer(&pixels)?))
    }

    /// [`Renderer::render`], minus the readback: the frame stays in the
    /// returned GPU buffer.
    ///
    /// # Errors
    ///
    /// Any [`crate::Error`] from buffer creation or submission.
    ///
    /// # Panics
    ///
    /// On a zero-sized target — callers validate their inputs, so this is a
    /// programmer bug.
    pub fn render_to_buffer(
        &self,
        gpu: &Context,
        scene: &Scene,
        width: u32,
        height: u32,
    ) -> Result<Buffer> {
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
        self.trace(gpu, scene, &pixels, width, height)?;
        Ok(pixels)
    }

    /// Trace one sample of `scene` and add it to `film`'s sums (the first
    /// sample after creation or a reset overwrites them). One blocking wave:
    /// primary kernel into the film's sample buffer, then the accumulation
    /// kernel — with its unconditional NaN/Inf guard — into the sums.
    ///
    /// # Errors
    ///
    /// Any [`crate::Error`] from submission.
    pub fn accumulate(&self, gpu: &Context, scene: &Scene, film: &mut Film) -> Result<()> {
        self.trace(gpu, scene, &film.sample, film.width, film.height)?;
        self.add_sample(gpu, film)?;
        film.samples += 1;
        Ok(())
    }

    /// Tonemap `film`'s accumulated average into its display buffer:
    /// exposure (in stops), the ACES display transform, sRGB encode, RGBA8
    /// pack — everything [`crate::gpu::Presenter::present`] needs.
    ///
    /// # Errors
    ///
    /// Any [`crate::Error`] from submission.
    ///
    /// # Panics
    ///
    /// If the film has no samples — there is no average to show, so calling
    /// order is a programmer bug.
    pub fn tonemap(&self, gpu: &Context, film: &Film, exposure: f32) -> Result<()> {
        assert!(film.samples > 0, "tonemapping an empty film");
        let params = TonemapParams {
            sum: film.sum.device_address(),
            display: film.display.device_address(),
            width: film.width,
            height: film.height,
            scale: exposure.exp2() / film.samples as f32,
            _pad0: 0.0,
        };
        gpu.dispatch(
            &self.tonemap,
            None,
            bytemuck::bytes_of(&params),
            workgroups(film.width, film.height),
        )
    }

    /// Dispatch the primary kernel: one frame of `scene` into `pixels`.
    fn trace(
        &self,
        gpu: &Context,
        scene: &Scene,
        pixels: &Buffer,
        width: u32,
        height: u32,
    ) -> Result<()> {
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
            &self.primary,
            Some(scene.tlas()),
            bytemuck::bytes_of(&params),
            workgroups(width, height),
        )
    }

    /// Dispatch the accumulation kernel: `film.sample` into `film.sum`,
    /// overwriting when the film is empty.
    fn add_sample(&self, gpu: &Context, film: &Film) -> Result<()> {
        let params = AccumulateParams {
            frame: film.sample.device_address(),
            sum: film.sum.device_address(),
            width: film.width,
            height: film.height,
            reset: u32::from(film.samples == 0),
            _pad0: 0,
        };
        gpu.dispatch(
            &self.accumulate,
            None,
            bytemuck::bytes_of(&params),
            workgroups(film.width, film.height),
        )
    }
}

fn create_primary_pipeline(gpu: &Context, spirv: &[u8]) -> Result<ComputePipeline> {
    gpu.create_compute_pipeline(
        spirv,
        shaders::PRIMARY_ENTRY,
        size_of::<Params>() as u32,
        Bindings::Tlas,
    )
}

/// 2D dispatch covering every pixel of a `width`×`height` target.
fn workgroups(width: u32, height: u32) -> [u32; 3] {
    [
        width.div_ceil(WORKGROUP_SIZE),
        height.div_ceil(WORKGROUP_SIZE),
        1,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pixel(pixels: &[f32], width: u32, x: u32, y: u32) -> &[f32] {
        let idx = ((y * width + x) * 4) as usize;
        &pixels[idx..idx + 4]
    }

    fn download_f32(gpu: &Context, buffer: &Buffer) -> Vec<f32> {
        bytemuck::pod_collect_to_vec(&gpu.download_buffer(buffer).expect("download"))
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

    /// Accumulating the deterministic M0 kernel N times must sum to N× a
    /// single frame — the film adds, it doesn't average or overwrite.
    #[test]
    fn accumulation_sums_identical_samples() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let scene = Scene::demo(&gpu).expect("demo scene");
        let renderer = Renderer::new(&gpu).expect("renderer");
        let mut film = Film::new(&gpu, 64, 64).expect("film");
        for _ in 0..3 {
            renderer
                .accumulate(&gpu, &scene, &mut film)
                .expect("accumulate");
        }
        assert_eq!(film.samples(), 3);

        let single = renderer.render(&gpu, &scene, 64, 64).expect("render");
        let sum = download_f32(&gpu, &film.sum);
        for (accumulated, one) in sum.iter().zip(&single) {
            assert!(
                (accumulated - 3.0 * one).abs() < 1e-5,
                "sum {accumulated} should be 3 × {one}"
            );
        }
    }

    /// After a reset, the next sample overwrites the stale sums — that *is*
    /// the clear pass. With the deterministic kernel the result must be
    /// bitwise identical to a fresh single frame.
    #[test]
    fn reset_restarts_the_accumulation() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let scene = Scene::demo(&gpu).expect("demo scene");
        let renderer = Renderer::new(&gpu).expect("renderer");
        let mut film = Film::new(&gpu, 64, 64).expect("film");
        for _ in 0..2 {
            renderer
                .accumulate(&gpu, &scene, &mut film)
                .expect("accumulate");
        }
        film.reset();
        renderer
            .accumulate(&gpu, &scene, &mut film)
            .expect("accumulate");
        assert_eq!(film.samples(), 1);

        let single = renderer.render(&gpu, &scene, 64, 64).expect("render");
        assert_eq!(download_f32(&gpu, &film.sum), single);
    }

    /// The accumulation kernel's finite guard: a NaN or Inf in any channel
    /// drops that pixel's whole contribution — on the overwrite path and
    /// the additive path alike — while clean pixels land untouched.
    #[test]
    fn non_finite_contributions_are_dropped() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let renderer = Renderer::new(&gpu).expect("renderer");
        let mut film = Film::new(&gpu, 4, 1).expect("film");
        let poisoned: [f32; 16] = [
            f32::NAN,
            0.5,
            0.5,
            1.0, // NaN red
            0.5,
            f32::INFINITY,
            0.5,
            1.0, // Inf green
            0.5,
            0.5,
            0.5,
            f32::NEG_INFINITY, // -Inf alpha
            0.25,
            0.5,
            0.75,
            1.0, // clean
        ];
        // Swap in a hand-poisoned sample; the usual writer (the primary
        // kernel) can't produce one.
        film.sample = gpu
            .upload_buffer(
                "film.sample.poisoned",
                bytemuck::bytes_of(&poisoned),
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            )
            .expect("upload");

        renderer.add_sample(&gpu, &film).expect("overwrite path");
        let expected_once = [
            0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, //
            0.25, 0.5, 0.75, 1.0,
        ];
        assert_eq!(download_f32(&gpu, &film.sum), expected_once);

        film.samples = 1;
        renderer.add_sample(&gpu, &film).expect("additive path");
        let doubled: Vec<f32> = expected_once.iter().map(|value| 2.0 * value).collect();
        assert_eq!(download_f32(&gpu, &film.sum), doubled);
    }

    /// The tonemap kernel against a CPU mirror of the same transform:
    /// average + exposure scale, `ACEScg` → Rec.709, Hill's ACES fit, sRGB
    /// encode, RGBA8 pack. A transposed matrix or wrong constant shows up
    /// as more than the ±1 quantization step allowed here.
    #[test]
    fn tonemap_matches_the_cpu_reference() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let renderer = Renderer::new(&gpu).expect("renderer");
        let mut film = Film::new(&gpu, 6, 1).expect("film");
        let sums: [f32; 24] = [
            0.0, 0.0, 0.0, 2.0, // black stays black
            0.36, 0.36, 0.36, 2.0, // mid grey (0.18 after ÷ samples)
            2.0, 2.0, 2.0, 2.0, // white
            20.0, 4.0, 1.0, 2.0, // hot highlight, compressed not clipped
            -1.0, 0.4, 0.4, 2.0, // negative clamps to zero, not garbage
            2.0, 0.2, 0.1, 2.0, // saturated red
        ];
        film.sum = gpu
            .upload_buffer(
                "film.sum.synthetic",
                bytemuck::bytes_of(&sums),
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            )
            .expect("upload");
        film.samples = 2;
        let exposure = 0.5;
        renderer.tonemap(&gpu, &film, exposure).expect("tonemap");

        let display = gpu.download_buffer(&film.display).expect("download");
        let scale = exposure.exp2() / 2.0;
        for (index, texel) in display.chunks_exact(4).enumerate() {
            let sum = &sums[index * 4..index * 4 + 3];
            let rgb = aces_display([sum[0] * scale, sum[1] * scale, sum[2] * scale]);
            for channel in 0..3 {
                let expected = (srgb_encode(rgb[channel]) * 255.0).round() as i16;
                let difference = (i16::from(texel[channel]) - expected).abs();
                assert!(
                    difference <= 1,
                    "pixel {index} channel {channel}: GPU {} vs CPU {expected}",
                    texel[channel]
                );
            }
            assert_eq!(texel[3], 255, "display frames are opaque");
        }
    }

    // -- CPU mirror of shaders/tonemap.slang, same constants, same order --

    fn multiply(matrix: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
        std::array::from_fn(|row| {
            matrix[row][0] * v[0] + matrix[row][1] * v[1] + matrix[row][2] * v[2]
        })
    }

    fn aces_display(acescg: [f32; 3]) -> [f32; 3] {
        const SRGB_FROM_ACESCG: [[f32; 3]; 3] = [
            [1.705_051, -0.621_792_1, -0.083_258_87],
            [-0.130_256_42, 1.140_804_7, -0.010_548_319],
            [-0.024_003_357, -0.128_968_98, 1.152_972_3],
        ];
        const ACES_INPUT: [[f32; 3]; 3] = [
            [0.59719, 0.35458, 0.04823],
            [0.07600, 0.90834, 0.01566],
            [0.02840, 0.13383, 0.83777],
        ];
        const ACES_OUTPUT: [[f32; 3]; 3] = [
            [1.60475, -0.53108, -0.07367],
            [-0.10208, 1.10813, -0.00605],
            [-0.00327, -0.07276, 1.07602],
        ];
        let v = multiply(&ACES_INPUT, multiply(&SRGB_FROM_ACESCG, acescg));
        let fitted = v.map(|v| {
            (v * (v + 0.024_578_6) - 0.000_090_537) / (v * (0.983_729 * v + 0.432_951) + 0.238_081)
        });
        multiply(&ACES_OUTPUT, fitted).map(|channel| channel.clamp(0.0, 1.0))
    }

    fn srgb_encode(channel: f32) -> f32 {
        if channel <= 0.003_130_8 {
            12.92 * channel
        } else {
            1.055 * channel.powf(1.0 / 2.4) - 0.055
        }
    }
}
