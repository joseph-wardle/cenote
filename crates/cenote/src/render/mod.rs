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
//!   host with [`Film::beauty_average`] and writes the batch EXR; the [`Session`]
//!   resolves on the GPU into a published frame and hands it to a consumer's
//!   [`Tonemap`] view transform. Batch output and the viewer's converged
//!   image are the same estimator by construction — they share the film.
//!
//! [`Tonemap`] is the other half of that split: exposure, the ACES display
//! transform, and the sRGB pack that turn a linear average into the frame
//! the presenter blits. The viewer owns one and drives it each frame; the
//! CLI never touches it, since EXR output stays linear.
//!
//! The film carries four buffers, not one: beauty plus the AOVs — the
//! denoiser's albedo and normal guides (with their specular pass-through:
//! mirrors record what they show) and first-hit depth. All four share the
//! accumulate/resolve path and the pixel-owned determinism invariant; the
//! CLI writes them as one multi-layer EXR, and OIDN consumes the guides.
//!
//! Every sample is a full path-traced estimate — jittered camera ray,
//! MIS-weighted direct light sampling at every bounce (emissive geometry,
//! delta lights, and the importance-sampled environment), `OpenPBR`
//! bounces — keyed by the
//! film's sample count, so accumulation converges toward the true render:
//! edges anti-alias, noise settles into soft shadows, color bleed, and
//! contact darkening.
//!
//! [`Session`] wraps this progressive path in a render thread, so the viewer
//! and a future scene-graph delegate consume published frames without pacing
//! the renderer to their own refresh — the actor that decouples the render loop.

mod film;
mod session;
mod tonemap;

pub use film::{Film, FilmAverages};
pub use session::{AutoStop, Frame, Session};
pub use tonemap::Tonemap;

use ash::vk;
use bytemuck::{Pod, Zeroable};

use crate::error::Result;
use crate::gpu::{Bindings, Buffer, ComputePipeline, Context, MemoryLocation, Pass, PassTimer};
use crate::scene::Scene;
use crate::shaders::Kernels;
use crate::stats::PassTimings;
use crate::wavefront::{LightSampling, Wavefront};

/// Workgroup width/height — must match `[numthreads(8, 8, 1)]` in the film
/// kernels (`accumulate.slang`, `resolve.slang`, `tonemap.slang`). Named
/// apart from the wavefront's 1D `WORKGROUP_SIZE` (`wavefront.rs`), which is
/// a different value governing a different kernel family.
const FILM_WORKGROUP_SIZE: u32 = 8;

/// Size of the film's auto-stop tally (step 6b): one `u32` the accumulate
/// kernel atomically counts converged pixels into. Shared by the film's
/// allocation and the per-sample zero-fill.
pub(super) const CONVERGED_COUNT_BYTES: u64 = 4;

/// Push constants for the accumulation kernel; mirrors `struct Params` in
/// `shaders/accumulate.slang` — one sample/sum address pair per film
/// buffer: beauty and the three AOVs.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AccumulateParams {
    /// Device address of the new beauty sample (`float4*`).
    sample: vk::DeviceAddress,
    /// Device address of the film's running beauty sums (`float4*`).
    sum: vk::DeviceAddress,
    albedo_sample: vk::DeviceAddress,
    albedo_sum: vk::DeviceAddress,
    normal_sample: vk::DeviceAddress,
    normal_sum: vk::DeviceAddress,
    /// The depth pair is `float*` — one channel per pixel.
    depth_sample: vk::DeviceAddress,
    depth_sum: vk::DeviceAddress,
    /// The variance substrate's second moment (`float*`) — Σ luminance² of
    /// the guarded beauty sample. No sample pair: it is derived in-kernel
    /// from the beauty sample, since the first moment is already the beauty
    /// sum (luminance is linear).
    moment2_sum: vk::DeviceAddress,
    /// The auto-stop convergence tally (`Atomic<uint>*`, step 6b): pixels whose
    /// relative estimator standard error fell below `noise_threshold` this
    /// sample. Zero-filled before the kernel each accumulate, so it is a fresh
    /// snapshot; read back by [`Film::converged_fraction`].
    converged_count: vk::DeviceAddress,
    width: u32,
    height: u32,
    /// Bool: overwrite the sums instead of adding — the first sample after
    /// a reset is the clear.
    reset: u32,
    /// Sample count after this contribution (N) — the variance divisor the
    /// noise metric reads.
    samples: u32,
    /// Below this N the noise metric is untrusted and no pixel is counted.
    min_samples: u32,
    /// Relative std-error below which a pixel counts as converged.
    noise_threshold: f32,
    /// Luminance floor in the relative-error denominator (near-black pixels).
    noise_floor: f32,
    _pad0: u32,
}

/// Push constants for the resolve kernel; mirrors `struct Params` in
/// `shaders/resolve.slang` — one sum/average address pair per film buffer.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ResolveParams {
    /// Device address of the film's running beauty sums (`float4*`).
    sum: vk::DeviceAddress,
    /// Device address of the linear beauty average target (`float4*`).
    average: vk::DeviceAddress,
    albedo_sum: vk::DeviceAddress,
    albedo_average: vk::DeviceAddress,
    normal_sum: vk::DeviceAddress,
    normal_average: vk::DeviceAddress,
    /// The depth pair is `float*` — one channel per pixel.
    depth_sum: vk::DeviceAddress,
    depth_average: vk::DeviceAddress,
    width: u32,
    height: u32,
    /// The sample count to divide by, as an `f32`. The host
    /// [`Film::averages`] divides by the same count, so the two averages
    /// agree to a few ULP (GPU division is only approximately rounded).
    samples: f32,
    _pad0: f32,
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
    /// Relative std-error below which the accumulate kernel counts a pixel as
    /// converged (M3 step 6b/6c) — the auto-stop metric's threshold. Defaults to
    /// [`Renderer::NOISE_THRESHOLD`]; the CLI `--noise-threshold` and the session's
    /// convergence-idle set it via [`Renderer::set_noise_threshold`]. Preserved
    /// across [`Renderer::reload`].
    noise_threshold: f32,
}

impl Renderer {
    /// Auto-stop noise threshold (M3 step 6b): a pixel counts as converged once
    /// its relative estimator standard error — `sqrt(Var/N) / max(mean L, floor)`
    /// — falls below this. 1% relative error is a perceptually tight default; the
    /// interactive `--noise-threshold` (step 6c) overrides it.
    pub const NOISE_THRESHOLD: f32 = 0.01;

    /// Luminance floor in the relative-error denominator, so a near-black pixel
    /// (mean luminance → 0) doesn't read as infinitely noisy and stall the stop.
    pub const NOISE_FLOOR: f32 = 1e-3;

    /// Samples before the noise metric is trusted: below it a handful of
    /// samples make `Var` meaningless, which `sqrt(Var/N)` assumes away.
    pub const CONVERGENCE_MIN_SAMPLES: u32 = 16;

    /// Fraction of pixels that must be converged for the global auto-stop to fire
    /// (step 6b). Held short of 1 so a few slow firefly pixels can't hold the whole
    /// render open forever; the consumer (step 6c) reads
    /// [`Film::converged_fraction`] against it.
    pub const CONVERGENCE_TARGET: f32 = 0.98;

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
            noise_threshold: Self::NOISE_THRESHOLD,
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
        let noise_threshold = self.noise_threshold;
        *self = Self::from_kernels(gpu, kernels, self.max_bounces)?;
        // A body edit must not silently drop the view's auto-stop threshold.
        self.noise_threshold = noise_threshold;
        Ok(())
    }

    /// Set the auto-stop noise threshold (M3 step 6c): the relative estimator
    /// standard error below which the accumulate kernel counts a pixel converged.
    /// Changes what [`Film::converged_fraction`] measures on the next
    /// [`Renderer::accumulate`]; beauty is untouched, so no reset is needed. The
    /// CLI `--noise-threshold` and the session's convergence-idle drive it.
    ///
    /// # Panics
    ///
    /// On a non-positive or non-finite threshold — callers validate their inputs,
    /// so this is a programmer bug.
    pub fn set_noise_threshold(&mut self, threshold: f32) {
        assert!(
            threshold > 0.0 && threshold.is_finite(),
            "noise threshold must be positive and finite"
        );
        self.noise_threshold = threshold;
    }

    /// The current auto-stop noise threshold.
    #[must_use]
    pub fn noise_threshold(&self) -> f32 {
        self.noise_threshold
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
        assert!(width > 0 && height > 0, "zero-sized render target");
        let size = u64::from(width) * u64::from(height) * 4 * size_of::<f32>() as u64;
        let pixels = gpu.create_buffer(
            // Staging for the trip back to the host, so it buckets with
            // `download.staging` rather than with the film it carries.
            "download.pixels",
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
        )?;
        self.wavefront
            .trace(gpu, scene, &pixels, width, height, 0)?;
        // pod_collect_to_vec rather than cast_slice: the downloaded bytes
        // carry no alignment guarantee.
        Ok(bytemuck::pod_collect_to_vec(&gpu.download_buffer(&pixels)?))
    }

    /// Trace the film's next sample of `scene` and add it to its sums (the
    /// first sample after creation or a reset overwrites them). One
    /// submission: the wave — at sample index [`Film::samples`], so a reset
    /// replays the exact same sequence — into the film's sample buffers
    /// (beauty and the three AOVs), then the accumulation kernel, with its
    /// unconditional NaN/Inf guard, folded into the same fence, into the
    /// sums.
    ///
    /// # Errors
    ///
    /// Any [`crate::Error`] from submission.
    pub fn accumulate(&self, gpu: &Context, scene: &Scene, film: &mut Film) -> Result<()> {
        self.accumulate_timed(gpu, scene, film, None).map(|_| ())
    }

    /// [`Renderer::accumulate`], reporting what the sample cost on the
    /// device — and, on the frames the timer resolves, what each kernel
    /// cost. Without a `timer` the submission is unchanged and the timings
    /// come back empty — the two are the same code path, so a timed render
    /// and an untimed one cannot diverge in what they draw.
    ///
    /// # Errors
    ///
    /// As [`Renderer::accumulate`].
    pub fn accumulate_timed(
        &self,
        gpu: &Context,
        scene: &Scene,
        film: &mut Film,
        timer: Option<&mut PassTimer>,
    ) -> Result<PassTimings> {
        let accumulate = self.accumulate_params(film);
        let timings = self.wavefront.trace_then(
            gpu,
            scene,
            &film.beauty.sample,
            film.width,
            film.height,
            film.samples,
            Some(&film.aov_targets()),
            // Zero the auto-stop tally (step 6b) before the kernel folds this
            // sample's converged pixels into it — a fresh per-sample snapshot,
            // ordered ahead of the accumulate by submit_passes' barrier.
            &[
                Pass::Fill {
                    buffer: &film.converged,
                    offset: 0,
                    size: CONVERGED_COUNT_BYTES,
                    value: 0,
                },
                self.accumulate_pass(&accumulate),
            ],
            timer,
        )?;
        film.samples += 1;
        Ok(timings)
    }

    /// Resolve `film`'s running sums into `targets` as linear averages: one
    /// dispatch dividing each pixel's sums — beauty and the three AOVs — by
    /// the sample count. The targets are the caller's — the [`Session`]
    /// rotates through a pair of them so it can publish one frame while the
    /// film keeps accumulating. Separate from [`Renderer::accumulate`] on
    /// purpose, too: the render thread accumulates flat out and resolves
    /// only when it publishes, so resolving must not ride every sample.
    ///
    /// # Errors
    ///
    /// Any [`crate::Error`] from submission.
    ///
    /// # Panics
    ///
    /// If the film has no samples — there is no average to resolve, so
    /// calling order is a programmer bug — or if any target is smaller than
    /// the film's `width`×`height` at its texel size.
    pub fn resolve(&self, gpu: &Context, film: &Film, targets: &ResolveTargets) -> Result<()> {
        assert!(film.samples > 0, "resolving an empty film");
        let texels = u64::from(film.width) * u64::from(film.height);
        for (target, texel) in [
            (targets.beauty, 16),
            (targets.albedo, 16),
            (targets.normal, 16),
            (targets.depth, 4),
        ] {
            assert!(
                target.size() >= texels * texel,
                "a resolve target is smaller than the film"
            );
        }
        let params = resolve_params(film, targets);
        gpu.dispatch(
            &self.resolve,
            None,
            bytemuck::bytes_of(&params),
            workgroups(film.width, film.height),
        )
    }

    /// The accumulation kernel's push constants: each film buffer's sample into
    /// its sums (overwriting when the film is empty), plus this renderer's live
    /// auto-stop threshold, which the CLI/session may have moved off the default.
    fn accumulate_params(&self, film: &Film) -> AccumulateParams {
        AccumulateParams {
            sample: film.beauty.sample.device_address(),
            sum: film.beauty.sum.device_address(),
            albedo_sample: film.albedo.sample.device_address(),
            albedo_sum: film.albedo.sum.device_address(),
            normal_sample: film.normal.sample.device_address(),
            normal_sum: film.normal.sum.device_address(),
            depth_sample: film.depth.sample.device_address(),
            depth_sum: film.depth.sum.device_address(),
            moment2_sum: film.moment2.device_address(),
            converged_count: film.converged.device_address(),
            width: film.width,
            height: film.height,
            reset: u32::from(film.samples() == 0),
            // The count *after* this sample lands (reset makes it the first): the
            // divisor the kernel's variance and standard error read.
            samples: film.samples() + 1,
            min_samples: Renderer::CONVERGENCE_MIN_SAMPLES,
            noise_threshold: self.noise_threshold,
            noise_floor: Renderer::NOISE_FLOOR,
            _pad0: 0,
        }
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

/// The caller-owned buffers one [`Renderer::resolve`] writes: the film's
/// four linear averages, each in its accumulation buffer's own layout
/// (RGBA f32 quads; `depth` one f32 per pixel).
pub struct ResolveTargets<'a> {
    /// Linear `ACEScg` radiance, RGBA f32.
    pub beauty: &'a Buffer,
    /// The denoiser albedo guide, RGBA f32.
    pub albedo: &'a Buffer,
    /// The denoiser normal guide, RGBA f32.
    pub normal: &'a Buffer,
    /// Camera-plane z at the first hit, one f32 per pixel.
    pub depth: &'a Buffer,
}

/// The resolve kernel's push constants: each film buffer's sums divided by
/// the sample count into its target.
fn resolve_params(film: &Film, targets: &ResolveTargets) -> ResolveParams {
    ResolveParams {
        sum: film.beauty.sum.device_address(),
        average: targets.beauty.device_address(),
        albedo_sum: film.albedo.sum.device_address(),
        albedo_average: targets.albedo.device_address(),
        normal_sum: film.normal.sum.device_address(),
        normal_average: targets.normal.device_address(),
        depth_sum: film.depth.sum.device_address(),
        depth_average: targets.depth.device_address(),
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
mod tests;
