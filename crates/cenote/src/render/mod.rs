//! Frame orchestration: drive the wavefront engine against the scene and
//! manage the film. Orchestration only — Vulkan stays behind [`crate::gpu`],
//! tracing behind [`crate::wavefront`].
//!
//! The estimator ends at a *linear average*, and the view transform is a
//! separate, downstream step — the split every production renderer draws
//! between the render buffer and its color pipeline:
//!
//! - **One-shot** ([`Renderer::render`]): allocate a buffer, trace one
//!   wave, read the linear pixels back — the test and hot-reload-probe
//!   path.
//! - **Progressive** ([`Renderer::accumulate`]): each call traces one
//!   sample into the [`Film`]'s running sums. [`Renderer::resolve`] then
//!   divides those sums by the sample count into a caller-owned linear
//!   average — the estimator's current best image. The CLI resolves on the
//!   host with [`Film::average`] and writes the batch EXR; the [`Session`]
//!   resolves on the GPU into a published frame and hands it to a consumer's
//!   [`Tonemap`] view transform. Batch output and the viewer's converged
//!   image are the same estimator by construction — they share the film.
//!
//! [`Tonemap`] is the other half of that split: exposure, the ACES display
//! transform, and the sRGB pack that turn a linear average into the frame
//! the presenter blits. The viewer owns one and drives it each frame; the
//! CLI never touches it, since EXR output stays linear.
//!
//! Every sample is a full path-traced estimate — jittered camera ray,
//! MIS-weighted direct light sampling at every bounce (quad lights and the
//! importance-sampled environment), `OpenPBR` bounces — keyed by the
//! film's sample count, so accumulation converges toward the true render:
//! edges anti-alias, noise settles into soft shadows, color bleed, and
//! contact darkening.
//!
//! [`Session`] wraps this progressive path in a render thread, so the viewer
//! and the future Hydra delegate consume published frames without pacing the
//! renderer to their own refresh — the actor that decouples the render loop.

mod session;
mod tonemap;

pub use session::{Frame, Session};
pub use tonemap::Tonemap;

use ash::vk;
use bytemuck::{Pod, Zeroable};

use crate::error::Result;
use crate::gpu::{Bindings, Buffer, ComputePipeline, Context, MemoryLocation, Pass};
use crate::scene::Scene;
use crate::shaders::Kernels;
use crate::wavefront::{LightSampling, Wavefront};

/// Workgroup width/height — must match `[numthreads(8, 8, 1)]` in the film
/// kernels (`accumulate.slang`, `resolve.slang`, `tonemap.slang`). Named
/// apart from the wavefront's 1D `WORKGROUP_SIZE` (`wavefront.rs`), which is
/// a different value governing a different kernel family.
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

/// Push constants for the resolve kernel; mirrors `struct Params` in
/// `shaders/resolve.slang`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ResolveParams {
    /// Device address of the film's running sums (`float4*`).
    sum: vk::DeviceAddress,
    /// Device address of the linear average target (`float4*`).
    average: vk::DeviceAddress,
    width: u32,
    height: u32,
    /// The sample count to divide by, as an `f32`. The host
    /// [`Film::average`] divides by the same count, so the two averages
    /// agree to a few ULP (GPU division is only approximately rounded).
    samples: f32,
    _pad0: f32,
}

/// Progressive accumulation state for one render-target size: per-pixel
/// linear RGBA f32 sums and the sample the current wave writes. The sample
/// count lives on the host — it is uniform across pixels by construction.
///
/// The resolved average — the sums divided by the count — is written into a
/// caller-owned buffer ([`Renderer::resolve`]) rather than held here, so the
/// [`Session`] can double-buffer its published frames while the film keeps
/// accumulating into these sums.
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
    resolve: ComputePipeline,
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
            resolve: gpu.create_compute_pipeline(
                &kernels.resolve.spirv,
                kernels.resolve.entry,
                size_of::<ResolveParams>() as u32,
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

    /// Resolve `film`'s running sums into `target` as a linear average: one
    /// dispatch dividing each pixel's sum by the sample count. `target` is the
    /// caller's — the [`Session`] rotates through a pair of them so it can
    /// publish one frame while the film keeps accumulating. Separate from
    /// [`Renderer::accumulate`] on purpose, too: the render thread accumulates
    /// flat out and resolves only when it publishes, so resolving must not
    /// ride every sample.
    ///
    /// # Errors
    ///
    /// Any [`crate::Error`] from submission.
    ///
    /// # Panics
    ///
    /// If the film has no samples — there is no average to resolve, so
    /// calling order is a programmer bug — or if `target` is smaller than the
    /// film's `width`×`height` RGBA f32 texels.
    pub fn resolve(&self, gpu: &Context, film: &Film, target: &Buffer) -> Result<()> {
        assert!(film.samples > 0, "resolving an empty film");
        assert!(
            target.size() >= u64::from(film.width) * u64::from(film.height) * 16,
            "resolve target is smaller than the film"
        );
        let params = resolve_params(film, target);
        gpu.dispatch(
            &self.resolve,
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

/// The resolve kernel's push constants: `film.sum` divided by the sample
/// count into `target`.
fn resolve_params(film: &Film, target: &Buffer) -> ResolveParams {
    ResolveParams {
        sum: film.sum.device_address(),
        average: target.device_address(),
        width: film.width,
        height: film.height,
        samples: film.samples as f32,
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
            up: Vec3::Y,
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
            up: Vec3::Y,
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

    /// The GPU resolve must land the same average as the host
    /// [`Film::average`] readback — same sums, same divisor. GPU division is
    /// only correctly rounded to a couple of ULP (Vulkan's precision floor),
    /// so the two agree to floating-point noise, not bit for bit; a real bug
    /// (wrong divisor, transposed indices) misses by far more than that.
    /// This is what lets the viewer and the CLI claim to show the same image.
    #[test]
    fn resolve_matches_host_average() {
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
        let target = gpu
            .create_buffer(
                "test.average",
                64 * 64 * 16,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::GpuOnly,
            )
            .expect("average buffer");
        renderer.resolve(&gpu, &film, &target).expect("resolve");
        let gpu_average = download_f32(&gpu, &target);
        let host_average = film.average(&gpu).expect("host average");
        for (gpu, host) in gpu_average.iter().zip(&host_average) {
            assert!(
                (gpu - host).abs() <= 1e-5 * host.abs().max(1.0),
                "resolve diverged from the host average: {gpu} vs {host}"
            );
        }
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
}
