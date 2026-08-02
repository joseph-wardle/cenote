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
use crate::wavefront::{
    LightSampling, Reproject, ReprojectCamera, RestirInputs, TemporalReuse, Wavefront,
};

pub use crate::wavefront::{DebugView, RenderMode};

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
// The estimator switches are independent knobs on one pipeline, not a state:
// every combination is a configuration someone renders on purpose, so an enum
// would have to enumerate the product and a state machine would model
// transitions that don't exist.
#[expect(
    clippy::struct_excessive_bools,
    reason = "independent estimator toggles, not a state to machine"
)]
pub struct Renderer {
    wavefront: Wavefront,
    accumulate: ComputePipeline,
    resolve: ComputePipeline,
    /// The path-length cap the wavefront was built with, kept so
    /// [`Renderer::reload`] rebuilds an identical engine.
    max_bounces: u32,
    /// Which estimator owns the primary hit's direct lighting — the path
    /// tracer, or `ReSTIR`-DI. Set by [`Renderer::set_render_mode`] and read
    /// each [`Renderer::accumulate`]; the front-ends flip it (the viewer's
    /// toggle, the CLI's `--restir`). Preserved across [`Renderer::reload`].
    render_mode: RenderMode,
    /// The D-092 debug view `restir_resolve` false-colours, or
    /// [`DebugView::Off`]. Meaningful only in [`RenderMode::Restir`]. Preserved
    /// across [`Renderer::reload`].
    debug_view: DebugView,
    /// Whether `ReSTIR` folds in spatial neighbours (M3 step 4). On by default —
    /// it is the estimator's variance reduction — and meaningful only in
    /// [`RenderMode::Restir`]; off runs single-frame RIS (the step-3 path), which
    /// the unbiasedness gate flips to check that both converge to the same image.
    /// Preserved across [`Renderer::reload`].
    spatial_reuse: bool,
    /// Whether `ReSTIR` reuses last frame's reservoirs (M3 step 5). On by
    /// default and meaningful only in [`RenderMode::Restir`]; when on, candidates
    /// write `cand`, `restir_temporal` folds `cand` + the frame-end-swapped `prev`
    /// into `curr` (M-capped, unshadowed), and the ping-pong carries the history
    /// across a camera move — the warm-start (steps 5b–5c, D-094). Step 5c reads
    /// `prev` at the *reprojected* pixel: each frame the renderer records the
    /// camera and writes a reprojection block, so next frame's shading points map
    /// back through last frame's camera and a disocclusion gate drops history the
    /// camera moved off. Preserved across [`Renderer::reload`].
    temporal_reuse: bool,
    /// Whether `ReSTIR` shades the spatial pass's control-variate lane — the
    /// candidate mean blended with every neighbour's running colour estimate
    /// (`ReSTCV`, M6 step 6b-i; the §6.3 vector-weight shading of 6a was this
    /// estimator's zero-CV shallow end) — rather than the survivor alone. On
    /// by default: it is the estimator's colour-noise fix and costs no rays.
    /// Off is survivor-only shading, the zero-CV degenerate (D-130) the A/B
    /// gates compare against. Meaningful only in [`RenderMode::Restir`] with
    /// spatial reuse on (spatial-off configs shade the survivor regardless).
    /// Preserved across [`Renderer::reload`].
    cv_shading: bool,
    /// Whether a restart frame may be drawn cheap (M7 step 7a) — on by default;
    /// off pins M flat, the pre-7a renderer. One of the conjuncts in
    /// [`Renderer::restir_candidates`], which is where the rest of the story is.
    /// Preserved across [`Renderer::reload`].
    cheap_restart: bool,
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

    /// Samples before the noise metric is trusted: below it a handful of samples
    /// make `Var` meaningless, and in `ReSTIR` the temporal history has not yet
    /// decayed to independent frames (D-094), which `sqrt(Var/N)` assumes. Set to
    /// the temporal decay window ([`Wavefront::RESTIR_TEMPORAL_DECAY_FRAMES`]), the
    /// frame the handoff completes.
    pub const CONVERGENCE_MIN_SAMPLES: u32 = Wavefront::RESTIR_TEMPORAL_DECAY_FRAMES;

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
            render_mode: RenderMode::PathTracer,
            debug_view: DebugView::Off,
            spatial_reuse: true,
            temporal_reuse: true,
            cv_shading: true,
            cheap_restart: true,
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
        let (
            render_mode,
            debug_view,
            spatial_reuse,
            temporal_reuse,
            cv_shading,
            cheap_restart,
            noise_threshold,
        ) = (
            self.render_mode,
            self.debug_view,
            self.spatial_reuse,
            self.temporal_reuse,
            self.cv_shading,
            self.cheap_restart,
            self.noise_threshold,
        );
        *self = Self::from_kernels(gpu, kernels, self.max_bounces)?;
        // A body edit must not silently drop the view's estimator choices.
        self.render_mode = render_mode;
        self.debug_view = debug_view;
        self.spatial_reuse = spatial_reuse;
        self.temporal_reuse = temporal_reuse;
        self.cv_shading = cv_shading;
        self.cheap_restart = cheap_restart;
        self.noise_threshold = noise_threshold;
        Ok(())
    }

    /// Choose the estimator for the primary hit's direct lighting — the path
    /// tracer, or `ReSTIR`-DI (D-088). Takes effect on the next
    /// [`Renderer::accumulate`]; the caller resets its film to switch cleanly.
    pub fn set_render_mode(&mut self, mode: RenderMode) {
        self.render_mode = mode;
    }

    /// The current estimator.
    #[must_use]
    pub fn render_mode(&self) -> RenderMode {
        self.render_mode
    }

    /// Choose the D-092 debug view `restir_resolve` false-colours into the
    /// debug buffer (or [`DebugView::Off`]). Meaningful only in
    /// [`RenderMode::Restir`]; takes effect on the next
    /// [`Renderer::accumulate`].
    pub fn set_debug_view(&mut self, view: DebugView) {
        self.debug_view = view;
    }

    /// The current debug view.
    #[must_use]
    pub fn debug_view(&self) -> DebugView {
        self.debug_view
    }

    /// Toggle `ReSTIR` spatial reuse (M3 step 4). On by default; off is the
    /// single-frame-RIS path the unbiasedness gate compares against. Meaningful
    /// only in [`RenderMode::Restir`]; takes effect on the next
    /// [`Renderer::accumulate`], so the caller resets its film to switch cleanly.
    pub fn set_spatial_reuse(&mut self, enabled: bool) {
        self.spatial_reuse = enabled;
    }

    /// Whether spatial reuse is on.
    #[must_use]
    pub fn spatial_reuse(&self) -> bool {
        self.spatial_reuse
    }

    /// Toggle `ReSTIR` temporal reuse (M3 step 5): reusing last frame's
    /// reservoirs across a camera move (the warm-start). On by default and
    /// meaningful only in [`RenderMode::Restir`]; off is the fresh-RNG
    /// spatial-only path the D-085 correctness anchor converges against. Takes
    /// effect on the next [`Renderer::accumulate`], so the caller resets its
    /// film to switch cleanly.
    pub fn set_temporal_reuse(&mut self, enabled: bool) {
        self.temporal_reuse = enabled;
    }

    /// Whether temporal reuse is on.
    #[must_use]
    pub fn temporal_reuse(&self) -> bool {
        self.temporal_reuse
    }

    /// Toggle control-variate shading (`ReSTCV`, M6 step 6b-i): shade the
    /// spatial pass's CV lane — the candidate mean plus the per-neighbour
    /// control-variate terms — rather than the survivor alone. On by default;
    /// off is survivor-only shading, the zero-CV degenerate (D-130) the A/B
    /// gates flip to. Meaningful only in [`RenderMode::Restir`] with spatial
    /// reuse on; takes effect on the next [`Renderer::accumulate`], so the
    /// caller resets its film to switch cleanly.
    pub fn set_cv_shading(&mut self, enabled: bool) {
        self.cv_shading = enabled;
    }

    /// Whether control-variate shading is on.
    #[must_use]
    pub fn cv_shading(&self) -> bool {
        self.cv_shading
    }

    /// Toggle the cheap restart frame (M7 step 7a) — on by default, and where a
    /// moving camera's whole frame time is. Off is the pre-7a renderer, the arm
    /// the saving is measured against. Takes effect on the next
    /// [`Renderer::accumulate`], so the caller resets its film to switch
    /// cleanly. See [`Renderer::restir_candidates`] for what it does.
    pub fn set_cheap_restart(&mut self, enabled: bool) {
        self.cheap_restart = enabled;
    }

    /// Whether the cheap restart frame is on.
    #[must_use]
    pub fn cheap_restart(&self) -> bool {
        self.cheap_restart
    }

    /// [`Renderer::restir_candidates`] for a caller that does not know the mode
    /// — `None` in [`RenderMode::PathTracer`], which has no reservoirs to fill.
    ///
    /// Reported per frame as [`crate::stats::Frame::candidates`], where it names
    /// which arm produced a measurement. Takes the film rather than a sample
    /// count so a caller cannot report an M the renderer did not use.
    #[must_use]
    pub fn candidate_count(&self, film: &Film) -> Option<u32> {
        match self.render_mode {
            RenderMode::PathTracer => None,
            RenderMode::Restir => Some(self.restir_candidates(film)),
        }
    }

    /// The initial-RIS candidate count M the next sample into `film` is drawn
    /// with (M7 step 7a): the cheap one on a frame that restarts accumulation
    /// onto a live temporal history, the full count everywhere else.
    ///
    /// Every conjunct is the premise, not a precaution. One candidate suffices
    /// on a restart *because history already supplies the confidence a full
    /// sweep would buy*, so the saving fires only where there is history being
    /// read: reuse switched on, and reservoirs holding a committed frame. A cold
    /// film has none — the opening sample of a batch render or a fresh viewport
    /// pays in full, and in doing so commits the full-strength reservoir every
    /// frame after it leans on.
    ///
    /// **Only sample 0, and only it can be.** A moving camera resets the film
    /// every frame while the reservoirs survive, so sample 0 against a warm
    /// history is the only frame it ever draws — the entire saving — and there
    /// the film's average is that one frame, so leaning on history correlates
    /// nothing. From sample 1 the average holds the frames history is made of,
    /// and leaning on it correlates the very samples being averaged. Measured on
    /// the many-lights reuse gate: climbing to the full count over the temporal
    /// decay window instead of stepping there cost 43% more error at 8 spp,
    /// erasing `ReSTIR`'s whole margin over brute force. Annealing that window
    /// to zero is what *decorrelates* a settling still (D-094).
    ///
    /// Unbiased either way — M only ever trades variance for cost.
    fn restir_candidates(&self, film: &Film) -> u32 {
        let warm_restart = self.cheap_restart
            && self.temporal_reuse
            && film.samples == 0
            && film.restir_history_is_warm();
        if warm_restart {
            Wavefront::RESTIR_RESTART_CANDIDATES
        } else {
            Wavefront::RESTIR_CANDIDATES
        }
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
        // In ReSTIR mode the reservoir stages own bounce 0's direct lighting;
        // allocate this frame's reservoir (and, for a debug view, the debug
        // buffer) lazily, so the path-tracer default carries neither.
        let restir = match self.render_mode {
            RenderMode::PathTracer => None,
            RenderMode::Restir => {
                let debug = self.debug_view != DebugView::Off;
                film.ensure_restir(gpu, debug)?;
                // Temporal reuse reprojects through last frame's camera, so build
                // this frame's reprojection block (previous camera + the current
                // G-buffer addresses) and host-write it before the wave. Reads of
                // the addresses (Copy `u64`) release their borrows before the
                // mutable write.
                if self.temporal_reuse {
                    let reproject = Reproject::new(
                        film.prev_reproject(),
                        film.view().gbuffer_prev().device_address(),
                        film.view().gbuffer_curr().device_address(),
                        film.width,
                        film.height,
                    );
                    film.write_reproject(bytemuck::bytes_of(&reproject));
                }
                Some(RestirInputs {
                    reservoir: film.view().curr(),
                    temporal: self.temporal_reuse.then(|| TemporalReuse {
                        cand: film.view().cand(),
                        prev: film.view().prev(),
                        reproject: film.view().reproject(),
                        // The decay ramp reads samples-since-reset (the wave's
                        // sample index = `film.samples`) on the GPU; the host only
                        // hands it the window. Held-camera handoff to spatial-only
                        // convergence (D-094, step 5d).
                        decay_frames: Wavefront::RESTIR_TEMPORAL_DECAY_FRAMES,
                        // The epoch gate's host half: `prev` carries the build it
                        // was rendered against (recorded at the last swap); an
                        // edit since then bumped `Scene::epoch`, and the mismatch
                        // tells temporal to drop indirect history this frame.
                        prev_same_scene: film.view().prev_epoch() == scene.epoch(),
                    }),
                    scratch: self.spatial_reuse.then(|| film.view().scratch()),
                    // Resolved on the host (M7 step 7a), so the kernel sees only
                    // a push constant's value change.
                    candidates: self.restir_candidates(film),
                    cv_shading: self.cv_shading,
                    debug: debug.then(|| film.debug()),
                    debug_view: self.debug_view,
                })
            }
        };
        let timings = self.wavefront.trace_then(
            gpu,
            scene,
            &film.beauty.sample,
            film.width,
            film.height,
            film.samples,
            Some(&film.aov_targets()),
            restir.as_ref(),
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
        // Frame end: wind the temporal ping-pong so next frame's `prev` is this
        // frame's committed `curr`, which `restir_temporal` folds into next
        // frame's candidates (the warm-start). Only in ReSTIR mode with temporal
        // reuse on; a no-op before the first ReSTIR wave built the state.
        if self.render_mode == RenderMode::Restir && self.temporal_reuse {
            film.swap_reservoirs(scene.epoch());
            // Record this frame's camera as next frame's reprojection source. The
            // pinhole basis (raygen's, before any aperture scale), since
            // reprojection places the real world hit, not a focal point.
            let basis = scene
                .camera()
                .basis(film.width as f32 / film.height as f32);
            film.set_prev_reproject(ReprojectCamera {
                position: scene.camera().position,
                right: basis.right,
                up: basis.up,
                forward: basis.forward,
            });
        }
        Ok(timings)
    }

    /// This frame's `ReSTIR` debug false-colour buffer, once at least one
    /// [`Renderer::accumulate`] has run with a [`DebugView`] selected — the
    /// render thread copies it into the published frame. `None` until then.
    #[must_use]
    pub fn debug_buffer<'f>(&self, film: &'f Film) -> Option<&'f Buffer> {
        (self.render_mode == RenderMode::Restir && self.debug_view != DebugView::Off)
            .then(|| film.debug())
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
