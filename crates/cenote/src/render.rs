//! Frame orchestration: drive the wavefront engine against the scene and
//! manage the film. Orchestration only — Vulkan stays behind [`crate::gpu`],
//! tracing behind [`crate::wavefront`].
//!
//! Two paths share the engine:
//!
//! - **One-shot** ([`Renderer::render`]): allocate a buffer, trace one
//!   wave, read the linear pixels back — the test and hot-reload-probe
//!   path.
//! - **Progressive** ([`Renderer::accumulate`]): each call traces one
//!   sample into the [`Film`]'s running sums. The viewer instead calls
//!   [`Renderer::accumulate_and_tonemap`], folding that trace, the
//!   accumulate, and the tonemap — average, exposure, the ACES display
//!   transform, into the RGBA8 buffer [`crate::gpu::Presenter::present`]
//!   shows — into a single submission; the CLI accumulates to a fixed
//!   sample count and reads the linear average back with [`Film::average`],
//!   writing it as the batch EXR. Batch output and the viewer's converged
//!   image are the same estimator by construction — they share the film.
//!
//! Every sample is a full path-traced estimate — jittered camera ray,
//! MIS-weighted direct light sampling at every bounce (quad lights and the
//! importance-sampled environment), `OpenPBR` bounces — keyed by the
//! film's sample count, so accumulation converges toward the true render:
//! edges anti-alias, noise settles into soft shadows, color bleed, and
//! contact darkening.

use ash::vk;
use bytemuck::{Pod, Zeroable};

use crate::error::Result;
use crate::gpu::{Bindings, Buffer, ComputePipeline, Context, MemoryLocation, Pass};
use crate::scene::Scene;
use crate::shaders::Kernels;
use crate::wavefront::{LightSampling, Wavefront};

/// Workgroup width/height — must match `[numthreads(8, 8, 1)]` in the film
/// kernels (`accumulate.slang`, `tonemap.slang`). Named apart from the
/// wavefront's 1D `WORKGROUP_SIZE` (`wavefront.rs`), which is a different
/// value governing a different kernel family.
const FILM_WORKGROUP_SIZE: u32 = 8;

/// Push constants for the accumulation kernel; mirrors `struct Params` in
/// `shaders/accumulate.slang`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AccumulateParams {
    /// Device address of the new sample (`float4*`).
    sample: vk::DeviceAddress,
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
    /// One sample's radiance, written by the wavefront's shading kernels
    /// each wave and consumed by the accumulation kernel.
    sample: Buffer,
    /// The running sums. `TRANSFER_SRC` so the accumulated image can be
    /// read back — [`Film::average`] and the tests.
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
            // TRANSFER_DST: each wave starts by zero-filling its target.
            sample: gpu.create_buffer(
                "film.sample",
                texels * 16,
                storage | vk::BufferUsageFlags::TRANSFER_DST,
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

    /// Read back the accumulated average — linear `ACEScg` RGBA, row-major,
    /// pixel (0, 0) top-left — the image the batch CLI writes. Each
    /// channel is its sum divided by the sample count, so alpha comes out
    /// exactly 1 and a one-sample average is bit-identical to the sample.
    ///
    /// # Errors
    ///
    /// Any [`crate::Error`] from the readback.
    ///
    /// # Panics
    ///
    /// If the film has no samples — there is no average yet, so calling
    /// order is a programmer bug.
    pub fn average(&self, gpu: &Context) -> Result<Vec<f32>> {
        assert!(self.samples > 0, "averaging an empty film");
        let sums: Vec<f32> = bytemuck::pod_collect_to_vec(&gpu.download_buffer(&self.sum)?);
        Ok(sums.iter().map(|sum| sum / self.samples as f32).collect())
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

/// The renderer: the wavefront engine plus the film kernels, ready to
/// render frames. Created from the embedded kernels; [`Renderer::reload`]
/// swaps in a recompiled set.
pub struct Renderer {
    wavefront: Wavefront,
    accumulate: ComputePipeline,
    tonemap: ComputePipeline,
    /// The path-length cap the wavefront was built with, kept so
    /// [`Renderer::reload`] rebuilds an identical engine.
    max_bounces: u32,
}

impl Renderer {
    /// Create the renderer from the embedded kernels, at the default
    /// path-length cap.
    ///
    /// # Errors
    ///
    /// Any [`crate::Error`] from pipeline or buffer creation.
    pub fn new(gpu: &Context) -> Result<Self> {
        Self::with_max_bounces(gpu, Wavefront::DEFAULT_MAX_BOUNCES)
    }

    /// [`Renderer::new`] with an explicit path-length cap — the CLI's
    /// `--depth`.
    ///
    /// # Errors
    ///
    /// Any [`crate::Error`] from pipeline or buffer creation.
    ///
    /// # Panics
    ///
    /// On zero bounces — callers validate their inputs, so this is a
    /// programmer bug.
    pub fn with_max_bounces(gpu: &Context, max_bounces: u32) -> Result<Self> {
        Self::from_kernels(gpu, &Kernels::embedded(), max_bounces)
    }

    /// Build every pipeline from `kernels` — the constructors with the
    /// embedded set, [`Renderer::reload`] with a recompiled one.
    fn from_kernels(gpu: &Context, kernels: &Kernels, max_bounces: u32) -> Result<Self> {
        Ok(Self {
            wavefront: Wavefront::new(
                gpu,
                kernels,
                Wavefront::DEFAULT_CAPACITY,
                max_bounces,
                LightSampling::Mis,
            )?,
            accumulate: gpu.create_compute_pipeline(
                &kernels.accumulate.spirv,
                kernels.accumulate.entry,
                size_of::<AccumulateParams>() as u32,
                Bindings::None,
            )?,
            tonemap: gpu.create_compute_pipeline(
                &kernels.tonemap.spirv,
                kernels.tonemap.entry,
                size_of::<TonemapParams>() as u32,
                Bindings::None,
            )?,
            max_bounces,
        })
    }

    /// Swap in a recompiled kernel set; if any pipeline fails to build, the
    /// current renderer stays live untouched. Entry-point names and
    /// push-constant layouts are pinned by the embedded build — hot reload
    /// covers kernel *body* edits; changing a params struct or the
    /// path-state schema needs a `cargo build`.
    ///
    /// # Errors
    ///
    /// Any [`crate::Error`] from pipeline or buffer creation.
    pub fn reload(&mut self, gpu: &Context, kernels: &Kernels) -> Result<()> {
        *self = Self::from_kernels(gpu, kernels, self.max_bounces)?;
        Ok(())
    }

    /// Render one `width`×`height` frame of `scene` — sample 0 of every
    /// pixel's sequence, a single path-traced estimate per pixel — and
    /// return it as row-major RGBA `f32` with pixel (0, 0) top-left, the
    /// crate-wide image convention.
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
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
        )?;
        self.wavefront
            .trace(gpu, scene, &pixels, width, height, 0)?;
        Ok(pixels)
    }

    /// Trace the film's next sample of `scene` and add it to its sums (the
    /// first sample after creation or a reset overwrites them). One
    /// submission: the wave — at sample index [`Film::samples`], so a reset
    /// replays the exact same sequence — into the film's sample buffer, then
    /// the accumulation kernel, with its unconditional NaN/Inf guard, folded
    /// into the same fence, into the sums.
    ///
    /// # Errors
    ///
    /// Any [`crate::Error`] from submission.
    pub fn accumulate(&self, gpu: &Context, scene: &Scene, film: &mut Film) -> Result<()> {
        let accumulate = accumulate_params(film);
        self.wavefront.trace_then(
            gpu,
            scene,
            &film.sample,
            film.width,
            film.height,
            film.samples,
            &[self.accumulate_pass(&accumulate)],
        )?;
        film.samples += 1;
        Ok(())
    }

    /// [`Renderer::accumulate`] with the tonemap folded in: trace the next
    /// sample, add it to the sums, and tonemap the new average into the
    /// display buffer — all one fence-waited submission, the viewer's
    /// per-frame path. `exposure` (stops) applies as in [`Renderer::tonemap`];
    /// the average divides by the count *including* this sample, since its
    /// accumulate lands in the same submission the tonemap reads.
    ///
    /// # Errors
    ///
    /// Any [`crate::Error`] from submission.
    pub fn accumulate_and_tonemap(
        &self,
        gpu: &Context,
        scene: &Scene,
        film: &mut Film,
        exposure: f32,
    ) -> Result<()> {
        let accumulate = accumulate_params(film);
        let tonemap = tonemap_params(film, exposure, film.samples + 1);
        self.wavefront.trace_then(
            gpu,
            scene,
            &film.sample,
            film.width,
            film.height,
            film.samples,
            &[
                self.accumulate_pass(&accumulate),
                self.tonemap_pass(&tonemap),
            ],
        )?;
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
        let params = tonemap_params(film, exposure, film.samples);
        gpu.dispatch(
            &self.tonemap,
            None,
            bytemuck::bytes_of(&params),
            workgroups(film.width, film.height),
        )
    }

    /// The accumulation dispatch as a [`Pass`], so it can ride the wave's
    /// submission (see [`Renderer::accumulate`]) or run on its own.
    fn accumulate_pass<'a>(&'a self, params: &'a AccumulateParams) -> Pass<'a> {
        Pass::Dispatch {
            pipeline: &self.accumulate,
            scene: None,
            push_constants: bytemuck::bytes_of(params),
            group_counts: workgroups(params.width, params.height),
        }
    }

    /// The tonemap dispatch as a [`Pass`], like [`Renderer::accumulate_pass`].
    fn tonemap_pass<'a>(&'a self, params: &'a TonemapParams) -> Pass<'a> {
        Pass::Dispatch {
            pipeline: &self.tonemap,
            scene: None,
            push_constants: bytemuck::bytes_of(params),
            group_counts: workgroups(params.width, params.height),
        }
    }
}

/// The accumulation kernel's push constants: `film.sample` into `film.sum`,
/// overwriting when the film is empty.
fn accumulate_params(film: &Film) -> AccumulateParams {
    AccumulateParams {
        sample: film.sample.device_address(),
        sum: film.sum.device_address(),
        width: film.width,
        height: film.height,
        reset: u32::from(film.samples == 0),
        _pad0: 0,
    }
}

/// The tonemap kernel's push constants: `film.sum`, averaged over `samples`
/// and exposure-scaled, into `film.display`. `samples` is a parameter
/// because the folded path ([`Renderer::accumulate_and_tonemap`]) divides by
/// the count *after* its own accumulate lands, one past [`Film::samples`].
fn tonemap_params(film: &Film, exposure: f32, samples: u32) -> TonemapParams {
    TonemapParams {
        sum: film.sum.device_address(),
        display: film.display.device_address(),
        width: film.width,
        height: film.height,
        scale: exposure.exp2() / samples as f32,
        _pad0: 0.0,
    }
}

/// 2D dispatch covering every pixel of a `width`×`height` target.
fn workgroups(width: u32, height: u32) -> [u32; 3] {
    [
        width.div_ceil(FILM_WORKGROUP_SIZE),
        height.div_ceil(FILM_WORKGROUP_SIZE),
        1,
    ]
}

#[cfg(test)]
mod tests {
    use glam::{Mat4, Vec3};

    use super::*;
    use crate::environment::Environment;
    use crate::material::Material;
    use crate::scene::{Camera, Object, ground_plane};

    fn pixel(pixels: &[f32], width: u32, x: u32, y: u32) -> &[f32] {
        let idx = ((y * width + x) * 4) as usize;
        &pixels[idx..idx + 4]
    }

    fn download_f32(gpu: &Context, buffer: &Buffer) -> Vec<f32> {
        bytemuck::pod_collect_to_vec(&gpu.download_buffer(buffer).expect("download"))
    }

    /// Accumulate `samples` waves of `scene` into a fresh `size`×`size`
    /// film and return the raw per-pixel RGBA sums.
    fn accumulate_sum(
        gpu: &Context,
        renderer: &Renderer,
        scene: &Scene,
        size: u32,
        samples: u32,
    ) -> Vec<f32> {
        let mut film = Film::new(gpu, size, size).expect("film");
        for _ in 0..samples {
            renderer
                .accumulate(gpu, scene, &mut film)
                .expect("accumulate");
        }
        download_f32(gpu, &film.sum)
    }

    /// A furnace scene: one big plane of the given material, scaled by
    /// `scale` and centered at `center`, under a half-intensity gray sky,
    /// with the camera just above looking obliquely down (the basis
    /// forbids straight down) so every camera ray lands on it. A path hits
    /// the plane, scatters upward, and escapes — so for a white material
    /// the expected pixel value is exactly the sky radiance (the
    /// energy-preservation property the EON and compensated-GGX lobes are
    /// built around), and for a pure Lambert surface every individual
    /// sample equals albedo × sky.
    fn furnace_scene(gpu: &Context, material: Material, center: Vec3, scale: f32) -> Scene {
        let object = Object {
            mesh: ground_plane(5.0),
            transform: Mat4::from_translation(center) * Mat4::from_scale(Vec3::splat(scale)),
            material,
        };
        let camera = Camera {
            position: center + Vec3::new(0.0, scale, 0.0),
            look_at: center + Vec3::new(0.0, 0.0, -scale),
            vfov_degrees: 40.0,
        };
        Scene::new(
            gpu,
            &[object],
            camera,
            &Environment::constant(Vec3::splat(0.5)),
        )
        .expect("furnace scene")
    }

    /// Accumulate `samples` waves through a BSDF-only engine and return
    /// the per-pixel RGBA sums. The exactness furnace tests below use this
    /// mode deliberately: single-strategy Lambert estimates are pointwise
    /// exact (every sample equals albedo × sky), while next-event + MIS
    /// estimates the same integral with per-sample variance — unbiased,
    /// but no longer a tight per-pixel assertion. Strategy agreement is
    /// the MIS-agreement tests' job, over in `wavefront.rs`.
    fn bsdf_only_sum(gpu: &Context, scene: &Scene, size: u32, samples: u32) -> Vec<f32> {
        let wavefront = Wavefront::new(
            gpu,
            &Kernels::embedded(),
            Wavefront::DEFAULT_CAPACITY,
            Wavefront::DEFAULT_MAX_BOUNCES,
            LightSampling::BsdfOnly,
        )
        .expect("wavefront");
        let radiance = gpu
            .create_buffer(
                "test.radiance",
                u64::from(size) * u64::from(size) * 16,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuOnly,
            )
            .expect("radiance buffer");
        let mut sum = vec![0.0_f32; (size * size * 4) as usize];
        for sample in 0..samples {
            wavefront
                .trace(gpu, scene, &radiance, size, size, sample)
                .expect("trace");
            for (total, value) in sum.iter_mut().zip(download_f32(gpu, &radiance)) {
                *total += value;
            }
        }
        sum
    }

    /// Probe the demo image's invariants. Every pixel finishes exactly
    /// once per wave (alpha 1, finite, non-negative), nearly the whole
    /// frame is lit under the daytime HDRI — at 1 spp a pixel goes black
    /// only when Russian roulette kills its path with every next-event
    /// connection occluded, which is rare — and the top-left pixel is open
    /// sky bright enough to be daytime. (The exact-background probe lives
    /// with the constant-sky scene in `wavefront.rs`; the demo's HDRI
    /// background varies per direction.)
    #[test]
    fn demo_image_is_sky_lit() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let scene = Scene::demo(&gpu).expect("demo scene");
        let renderer = Renderer::new(&gpu).expect("renderer");
        let (width, height) = (128, 128);
        let pixels = renderer
            .render(&gpu, &scene, width, height)
            .expect("render");

        let mut lit = 0;
        for chunk in pixels.chunks_exact(4) {
            assert_eq!(chunk[3..], [1.0], "a pixel was skipped: {chunk:?}");
            assert!(
                chunk[..3].iter().all(|c| c.is_finite() && *c >= 0.0),
                "non-finite or negative radiance: {chunk:?}"
            );
            if chunk[..3].iter().sum::<f32>() > 0.0 {
                lit += 1;
            }
        }
        assert!(
            lit > (width * height * 9 / 10) as usize,
            "most of the frame should be lit, got {lit} pixels"
        );
        assert!(
            pixel(&pixels, width, 0, 0)[..3].iter().sum::<f32>() > 0.5,
            "the top-left pixel should be open daytime sky"
        );
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

    /// The diffuse white furnace (the step-7 checkpoint): an albedo-1 EON
    /// plane under a uniform sky must reflect exactly the sky radiance —
    /// energy lost or gained anywhere in the estimator (a dropped
    /// multiple-scattering lobe, a wrong pdf, a biased roulette) shifts the
    /// result. At roughness 0 the lobe is Lambert and, BSDF-only, *every
    /// sample of every pixel* equals the sky exactly, so the bound is
    /// tight; at roughness 1 the per-sample value is stochastic (and the
    /// albedo fit itself is only good to ~4e-4), so the mean over the full
    /// MIS renderer carries the assertion.
    #[test]
    fn diffuse_furnace_closes() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let renderer = Renderer::new(&gpu).expect("renderer");
        let sky = 0.5;

        let lambert = furnace_scene(&gpu, Material::matte(Vec3::ONE, 0.0), Vec3::ZERO, 1.0);
        let sum = bsdf_only_sum(&gpu, &lambert, 32, 4);
        for chunk in sum.chunks_exact(4) {
            for channel in &chunk[..3] {
                let value = channel / 4.0;
                assert!(
                    (value - sky).abs() < 1e-3,
                    "Lambert furnace leaked: {value} vs {sky}"
                );
            }
        }

        let rough = furnace_scene(&gpu, Material::matte(Vec3::ONE, 1.0), Vec3::ZERO, 1.0);
        let samples = 64;
        let sum = accumulate_sum(&gpu, &renderer, &rough, 32, samples);
        let mean =
            sum.chunks_exact(4).map(|chunk| chunk[0]).sum::<f32>() / (32.0 * 32.0 * samples as f32);
        assert!(
            (mean - sky).abs() < 0.005,
            "rough furnace leaked: mean {mean} vs {sky}"
        );
    }

    /// The spawn-point offsets hold at scene scale — the property the van
    /// Antwerpen rigorous error bounds exist for. A half-albedo Lambert
    /// furnace, with the plane pushed 10⁴ m from the origin and scaled
    /// 1000×, where hit reconstruction error reaches millimeters: every
    /// sample must still be albedo × sky exactly. A bounce ray that
    /// self-intersects the plane it just left multiplies in another albedo
    /// factor and fails the bound loudly. (An albedo-1 furnace can't see
    /// this — spurious extra bounces cost it no energy — which is why this
    /// one is gray. BSDF-only, for the same per-sample exactness as the
    /// Lambert furnace above.)
    #[test]
    fn ray_offsets_hold_at_scene_scale() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let scene = furnace_scene(
            &gpu,
            Material::matte(Vec3::splat(0.5), 0.0),
            Vec3::new(1e4, 0.0, 1e4),
            1e3,
        );
        let sum = bsdf_only_sum(&gpu, &scene, 32, 4);
        let expected = 0.5 * 0.5; // albedo × sky
        for chunk in sum.chunks_exact(4) {
            for channel in &chunk[..3] {
                let value = channel / 4.0;
                assert!(
                    (value - expected).abs() < 1e-3,
                    "self-intersection at scale: {value} vs {expected}"
                );
            }
        }
    }

    /// The step-9 checkpoint's teeth: the white-furnace matrix over the
    /// `OpenPBR` M1 lobe set. A white material of any construction must
    /// return exactly the sky's radiance — single-scatter GGX *fails this
    /// by design* (it loses up to half its energy at roughness 1), so this
    /// pins the Turquin compensation, the regenerated albedo fits, and the
    /// albedo-scaling layering all at once. The tolerance is the fits'
    /// measured residual (CPU-validated at ≤ 0.6% over these very
    /// configurations) plus sampling noise.
    #[test]
    fn openpbr_furnace_matrix() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let renderer = Renderer::new(&gpu).expect("renderer");
        let white = Vec3::ONE;
        let configs = [
            ("metal r=0.05", Material::metal(white, 0.05)),
            ("metal r=0.5", Material::metal(white, 0.5)),
            ("metal r=1.0", Material::metal(white, 1.0)),
            ("glossy-diffuse r=0.05", Material::glossy(white, 0.0, 0.05)),
            ("glossy-diffuse r=0.5", Material::glossy(white, 0.0, 0.5)),
            (
                "glossy-diffuse r=1.0, rough base",
                Material::glossy(white, 1.0, 1.0),
            ),
            (
                "half metal",
                Material::glossy(white, 0.0, 0.5).with_metalness(0.5),
            ),
        ];
        let (sky, samples) = (0.5, 64);
        for (label, material) in configs {
            let scene = furnace_scene(&gpu, material, Vec3::ZERO, 1.0);
            let sum = accumulate_sum(&gpu, &renderer, &scene, 32, samples);
            let mean = sum.chunks_exact(4).map(|chunk| chunk[0]).sum::<f32>()
                / (32.0 * 32.0 * samples as f32);
            assert!(
                (mean - sky).abs() / sky < 0.015,
                "{label}: furnace leaked, mean {mean} vs {sky}"
            );
        }
    }

    /// First global illumination, made mechanical: sky light bounces off
    /// a terracotta sphere onto a gray floor, so floor pixels beside the
    /// sphere pick up a red cast that the far floor corner doesn't. Both
    /// probes are the same neutral material — the difference is purely
    /// bounced light. (A dedicated scene, not the demo: the probe
    /// positions pin this geometry.)
    #[test]
    fn indirect_light_bleeds_color() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let objects = [
            Object {
                mesh: crate::scene::icosphere(2),
                transform: Mat4::from_translation(Vec3::Y),
                material: Material::matte(
                    crate::color::acescg_from_rec709(Vec3::new(0.7, 0.22, 0.08)),
                    0.6,
                ),
            },
            Object {
                mesh: ground_plane(5.0),
                transform: Mat4::IDENTITY,
                material: Material::matte(crate::color::acescg_from_rec709(Vec3::splat(0.65)), 0.1),
            },
        ];
        let camera = Camera {
            position: Vec3::new(0.0, 1.8, 5.0),
            look_at: Vec3::new(0.0, 1.0, 0.0),
            vfov_degrees: 40.0,
        };
        let scene =
            Scene::new(&gpu, &objects, camera, &Environment::constant(Vec3::ONE)).expect("scene");
        let renderer = Renderer::new(&gpu).expect("renderer");
        let size = 64;
        let sum = accumulate_sum(&gpu, &renderer, &scene, size, 32);

        // Mean red/blue ratio over a 3×3 patch — single accumulated pixels
        // are still noisy at 32 samples.
        let redness = |x: u32, y: u32| {
            let (mut red, mut blue) = (0.0, 0.0);
            for dy in 0..3 {
                for dx in 0..3 {
                    let probe = pixel(&sum, size, x + dx, y + dy);
                    red += probe[0];
                    blue += probe[2];
                }
            }
            red / blue
        };
        // The sphere (image center, radius ≈ 18 px at 64²) meets the floor
        // around y = 50; the corner patch sees almost none of it.
        let near = redness(30, 53);
        let far = redness(2, 60);
        assert!(
            near > far * 1.05,
            "no red bleed beside the sphere: near {near} vs far {far}"
        );
    }

    /// The hot-reload swap end to end, minus the file watch: recompile the
    /// unmodified kernel set through the runtime `slangc` path, swap it in,
    /// and require a pixel-identical frame — same source, same compiler,
    /// same flags must mean the same image.
    #[test]
    fn reloaded_kernels_render_identically() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let scene = Scene::demo(&gpu).expect("demo scene");
        let mut renderer = Renderer::new(&gpu).expect("renderer");
        let before = renderer.render(&gpu, &scene, 64, 64).expect("render");

        let kernels = Kernels::recompile().expect("recompile");
        renderer.reload(&gpu, &kernels).expect("reload");
        let after = renderer.render(&gpu, &scene, 64, 64).expect("render");

        assert_eq!(before, after);
    }

    /// Two renders of the same scene must agree bit for bit — the
    /// charter's replay guarantee, made mechanical. This is the check that
    /// pins the wavefront's determinism rule: queue push order varies from
    /// run to run, so any radiance write that isn't pixel-owned (or any
    /// atomic accumulation) shows up here as flickering low bits.
    #[test]
    fn rendering_is_bitwise_deterministic() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let scene = Scene::demo(&gpu).expect("demo scene");
        let renderer = Renderer::new(&gpu).expect("renderer");
        let first = renderer.render(&gpu, &scene, 128, 128).expect("render");
        let second = renderer.render(&gpu, &scene, 128, 128).expect("render");
        assert_eq!(first, second);
    }

    /// The film adds each wave's sample — and consecutive samples genuinely
    /// differ now that raygen jitters. Rebuild the expected sums from
    /// individually traced samples 0..3: the CPU adds in the same order as
    /// the three accumulation dispatches (one `f32` add per wave), so
    /// agreement is bitwise.
    #[test]
    fn accumulation_adds_distinct_samples() {
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

        let sample = |index: u32| -> Vec<f32> {
            let radiance = gpu
                .create_buffer(
                    "test.sample",
                    64 * 64 * 16,
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                        | vk::BufferUsageFlags::TRANSFER_SRC
                        | vk::BufferUsageFlags::TRANSFER_DST,
                    MemoryLocation::GpuOnly,
                )
                .expect("radiance buffer");
            renderer
                .wavefront
                .trace(&gpu, &scene, &radiance, 64, 64, index)
                .expect("trace");
            download_f32(&gpu, &radiance)
        };
        let (s0, s1, s2) = (sample(0), sample(1), sample(2));
        assert_ne!(s0, s1, "jitter must vary from sample to sample");

        let expected: Vec<f32> = s0
            .iter()
            .zip(&s1)
            .zip(&s2)
            .map(|((a, b), c)| a + b + c)
            .collect();
        assert_eq!(download_f32(&gpu, &film.sum), expected);

        // The batch readback is those sums divided by the count — the same
        // f32 division on both sides, so agreement is again bitwise.
        let average: Vec<f32> = expected.iter().map(|sum| sum / 3.0).collect();
        assert_eq!(film.average(&gpu).expect("average"), average);
    }

    /// After a reset, the next sample overwrites the stale sums — that *is*
    /// the clear pass. And a reset restarts the sample sequence at index 0,
    /// so the result must be bitwise identical to a fresh single frame.
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

    /// The viewer's folded path — trace, accumulate, and tonemap in one
    /// submission — must produce the exact same display buffer as running
    /// the three as separate fence-waited submissions: a barrier between
    /// passes orders the writes a fence would. Both films trace the same
    /// sample sequence, and the fold divides by the count *including* its
    /// own sample, so the average and the tonemap match bit for bit.
    #[test]
    fn folded_frame_matches_separate_passes() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let scene = Scene::demo(&gpu).expect("demo scene");
        let renderer = Renderer::new(&gpu).expect("renderer");
        let exposure = 0.5;

        // Separate: three accumulations, then a standalone tonemap.
        let mut separate = Film::new(&gpu, 64, 64).expect("film");
        for _ in 0..3 {
            renderer
                .accumulate(&gpu, &scene, &mut separate)
                .expect("accumulate");
        }
        renderer
            .tonemap(&gpu, &separate, exposure)
            .expect("tonemap");
        let expected = gpu.download_buffer(separate.display()).expect("download");

        // Folded: the third sample's accumulate and the tonemap ride the wave.
        let mut folded = Film::new(&gpu, 64, 64).expect("film");
        for _ in 0..2 {
            renderer
                .accumulate(&gpu, &scene, &mut folded)
                .expect("accumulate");
        }
        renderer
            .accumulate_and_tonemap(&gpu, &scene, &mut folded, exposure)
            .expect("accumulate and tonemap");
        assert_eq!(folded.samples(), 3);
        assert_eq!(
            gpu.download_buffer(folded.display()).expect("download"),
            expected
        );
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

        // Drive the accumulation kernel directly — the same pass the render
        // paths fold into the wave, here submitted alone against a poisoned
        // sample the primary kernel could never produce.
        let overwrite = accumulate_params(&film);
        gpu.submit_passes(&[renderer.accumulate_pass(&overwrite)])
            .expect("overwrite path");
        let expected_once = [
            0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, //
            0.25, 0.5, 0.75, 1.0,
        ];
        assert_eq!(download_f32(&gpu, &film.sum), expected_once);

        film.samples = 1;
        let additive = accumulate_params(&film);
        gpu.submit_passes(&[renderer.accumulate_pass(&additive)])
            .expect("additive path");
        let doubled: Vec<f32> = expected_once.iter().map(|value| 2.0 * value).collect();
        assert_eq!(download_f32(&gpu, &film.sum), doubled);
    }

    /// The tonemap kernel against a CPU mirror of the same transform:
    /// average + exposure scale, `ACEScg` → `Rec.709`, Hill's ACES fit, sRGB
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
