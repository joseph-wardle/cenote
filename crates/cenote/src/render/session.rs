//! The render loop as an actor: a dedicated thread that accumulates as fast
//! as the GPU allows, so a consumer's display refresh never paces the
//! renderer. The viewer is the first consumer; a scene-graph delegate could
//! be a second — the concurrency lives here, once, not in each of them. This is
//! the shape Cycles, `MoonRay`, and Karma all use: the path tracer runs on its
//! own thread and the UI *peeks* at its output.
//!
//! Four lanes cross the thread boundary, each its own short-lived lock:
//!
//! - **Inputs in** — [`RenderInputs`] (camera, target size, a `generation`
//!   counter, a running flag) behind a mutex, latest-wins. The viewer writes
//!   the latest camera or size; the render thread snapshots the whole struct
//!   once per sample. Exposure is *not* here: it belongs to the consumer's
//!   view transform, downstream of the published frame.
//! - **Edits in** — queued [`ChangeSet`]s ([`Session::apply`] overlays a
//!   patch, [`Session::replace`] swaps in a whole scene — the file-reload
//!   shape, where objects the new set lacks retire). Edits merge in arrival
//!   order and land at the next wave boundary: the thread applies them to
//!   its description, re-preps exactly what the dirty state names, and
//!   restarts accumulation from sample 0 — the industry consensus
//!   (`MoonRay` restarts on any edit).
//! - **Frames out** — the resolved **linear** average, published behind a
//!   second mutex. The render thread resolves into whichever of its two
//!   frame buffers is free and hands over an [`Arc`] to it; the viewer takes
//!   the latest and tonemaps it. The lock spans only the pointer hand-off,
//!   never a GPU submit — the heavy accumulate runs lock-free.
//! - **Faults out** — a rejected edit (invalid change-set, or a description
//!   this build can't render) is *not* a render-thread failure: the thread
//!   posts it for [`Session::take_edit_error`], keeps rendering its last
//!   good scene, and retries the pending re-prep after the next applied
//!   edit. Only device faults end the thread, surfacing via
//!   [`Session::check`].
//!
//! Riding beside the lanes is the session **epoch** — a count of the
//! picture-changing verbs accepted so far, stamped onto every published
//! frame at the wave boundary that incorporated them. It lets a consumer at
//! the far end of a throttled, double-buffered pipe tell "converged, and it
//! includes everything I sent" from "converged, but an older picture"
//! without reaching into the renderer (M4 step 2, D-113). See
//! [`Lanes::epoch`] for the counting rules.
//!
//! Two frame buffers, not a triple-buffered mailbox: the render thread
//! resolves only into a buffer no one else references (a strong-count of one
//! means "in the pool alone"), and if both are busy it simply skips that
//! publish and keeps accumulating. So a slow consumer can never see a buffer
//! torn by an in-flight resolve, and the renderer never blocks on the
//! consumer. The strong count is a sound "free" test only because every
//! consumer submission blocks: a [`Frame`] drops strictly after the GPU work
//! that read its buffer completed. The pre-M3 timeline-pacing pass, which
//! removes those blocking fences, must revisit this reuse protocol with
//! them.
//!
//! A render-thread failure is not swallowed. Its own errors — a GPU call
//! failing mid-loop — ride back through the join as an ordinary `Err`; an
//! actual panic on that thread comes back too. [`Session::check`] lets the
//! consumer reap a thread that has ended early and surface the fault, rather
//! than spin forever on a renderer that will post no more frames; the join in
//! [`Session::drop`] is the backstop at shutdown.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use ash::vk;

use super::{DebugView, Film, RenderMode, Renderer, ResolveTargets};
use crate::error::{Error, Result};
use crate::gpu::{Buffer, Context, MemoryLocation, Pass, PassTimer};
use crate::scene::changeset::{ChangeSet, Dirty, Kind};
use crate::scene::description::SceneDescription;
use crate::scene::{Camera, Scene};
use crate::stats::{self, Recorder, Stats};

/// The shortest gap between published frames, early in an accumulation. The
/// render thread accumulates flat out but resolves and publishes at most this
/// often — resolving every sample would burn GPU time a consumer can't display
/// faster than its refresh anyway. Set just under a 60 Hz frame so a vsync'd
/// viewer always finds a fresh frame waiting. See [`publish_interval`].
const PUBLISH_INTERVAL_MIN: Duration = Duration::from_millis(15);

/// The longest gap between published frames, once an accumulation has slowed.
/// A converging image improves as ~1/N, so late samples move it imperceptibly;
/// backing the publish rate off to a few per second there saves resolves the
/// consumer can't see anyway. See [`publish_interval`].
const PUBLISH_INTERVAL_MAX: Duration = Duration::from_millis(250);

/// Sample count over which the publish gap widens by one [`PUBLISH_INTERVAL_MIN`]
/// step (see [`publish_interval`]) — the knee where per-sample change has fallen
/// enough that a slower publish rate is invisible.
const PUBLISH_INTERVAL_STEP: u32 = 64;

/// How long the render thread sleeps when there is nothing to draw (a
/// minimized, zero-area window) or when a settled render has parked, before
/// re-reading its inputs — long enough not to spin, short enough to wake
/// promptly when the window returns or the view changes.
const IDLE_NAP: Duration = Duration::from_millis(16);

/// How often the render thread re-reads device memory. The allocations move
/// only when the scene or the target does, so a per-wave read would be
/// precision theatre over numbers that did not change.
const MEMORY_SAMPLE_INTERVAL: Duration = Duration::from_secs(1);

/// When the render thread stops accumulating and parks — a settled render must
/// not pin the GPU forever (M3 step 6c, the D-089 interactivity bundle).
/// `max_samples` is the hard cap every render obeys; `noise_threshold`, when
/// set, is an additional early stop once [`Renderer::CONVERGENCE_TARGET`] of the
/// pixels have reached that relative estimator standard error. A parked thread
/// wakes and re-accumulates the moment any input restarts the film.
#[derive(Clone, Copy)]
pub struct AutoStop {
    /// The sample count at which accumulation stops regardless of convergence —
    /// the backstop the viewer always sets, so an idle view releases the GPU.
    pub max_samples: u32,
    /// The convergence early-out threshold, or `None` to stop only at
    /// `max_samples`. Passed through to [`Renderer::set_noise_threshold`], so it
    /// is also what [`Film::converged_fraction`] measures against.
    pub noise_threshold: Option<f32>,
}

/// Grow the publish gap with the sample count: publish every frame while the
/// image is changing fast, then back off toward [`PUBLISH_INTERVAL_MAX`] as it
/// converges (improvement ~1/N, so late publishes carry vanishing new detail).
/// Linear in the sample count — one [`PUBLISH_INTERVAL_MIN`] step per
/// [`PUBLISH_INTERVAL_STEP`] samples — clamped at both ends.
fn publish_interval(samples: u32) -> Duration {
    (PUBLISH_INTERVAL_MIN * (samples / PUBLISH_INTERVAL_STEP).max(1)).min(PUBLISH_INTERVAL_MAX)
}

/// What the viewer feeds the render thread, latest-wins, snapshotted once per
/// sample. No exposure: that is the consumer's view transform, applied
/// downstream of the published frame.
// Latest-wins scalars on one snapshot, not a state: the estimator toggles are
// independent of each other and of `running`, and every combination is one
// someone asks for on purpose.
#[expect(
    clippy::struct_excessive_bools,
    reason = "independent latest-wins inputs, not a state to machine"
)]
#[derive(Clone, Copy)]
struct RenderInputs {
    /// The view to render. Applied to the scene when `generation` changes.
    camera: Camera,
    /// The render-target size in physical pixels; a change means a new film.
    size: (u32, u32),
    /// Bumped on every view change. When it moves, the render thread adopts
    /// the new camera and restarts accumulation — the threaded equivalent of
    /// the single-threaded viewer's `Film::reset`.
    generation: u64,
    /// Which estimator owns the primary hit's direct lighting. The viewer's
    /// `ReSTIR` toggle writes it; the render thread adopts a change and
    /// restarts accumulation (the two estimators converge to the same image,
    /// so this keeps an A/B switch crisp rather than mixing them).
    render_mode: RenderMode,
    /// The D-092 debug view to false-colour, or [`DebugView::Off`]. Flows
    /// straight into each sample — the debug surface is single-shot and live,
    /// so a change needs no accumulation restart.
    debug_view: DebugView,
    /// Whether `ReSTIR` folds in spatial neighbours (M3 step 4). Meaningful only
    /// in [`RenderMode::Restir`]; like `render_mode`, a change restarts
    /// accumulation so the on/off images (both the same integral) never mix — the
    /// D-090 unbiasedness gate the viewer drives.
    spatial_reuse: bool,
    /// Whether `ReSTIR` warm-starts from the previous frame's reservoirs (M3
    /// step 5). Meaningful only in [`RenderMode::Restir`]; a change restarts
    /// accumulation like the other estimator switches. On a held camera the
    /// decay ramp (D-094) anneals temporal off regardless, so on/off converge to
    /// the same still — this toggle is for watching the warm-start live.
    temporal_reuse: bool,
    /// Whether a restart frame may be drawn cheap (M7 step 7a). Meaningful only
    /// in [`RenderMode::Restir`]; a change restarts accumulation like the other
    /// estimator switches, so a flip never blends two candidate policies into
    /// one image. What it changes is visible only while the camera is moving,
    /// which no still render reproduces — this is the toggle a person judges.
    cheap_restart: bool,
    /// Hard cap on accumulated samples (M3 step 6c). At this count the render
    /// thread parks — stops accumulating and idles — until an input restarts the
    /// film, so a settled view stops pinning the GPU.
    max_samples: u32,
    /// Convergence early-out threshold, or `None` to park only at `max_samples`.
    /// When set, the thread also parks once [`Renderer::CONVERGENCE_TARGET`] of
    /// pixels reach it; adopted into the renderer, so it is what the accumulate
    /// kernel counts and [`Film::converged_fraction`] reads.
    noise_threshold: Option<f32>,
    /// Cleared to stop the thread; checked at the top of every iteration.
    running: bool,
}

/// One queued scene edit — the two verbs a change-set can arrive as.
enum SceneEdit {
    /// Overlay onto the current description ([`SceneDescription::apply`]).
    Apply(ChangeSet),
    /// The set describes the whole scene from empty; the description
    /// becomes it, diffing for dirt ([`SceneDescription::replace`]).
    Replace(ChangeSet),
}

/// The four lanes between a consumer and the render thread — one shared
/// allocation, each lane its own short-lived lock.
struct Lanes {
    inputs: Mutex<RenderInputs>,
    edits: Mutex<Vec<SceneEdit>>,
    /// The latest rejected edit, kept until the consumer takes it. A newer
    /// rejection replaces an untaken older one — the consumer polling once
    /// a frame sees the freshest fault, and the log carries the history.
    edit_error: Mutex<Option<Error>>,
    published: Mutex<Option<Frame>>,
    /// The session epoch (D-113): a count of the picture-changing verbs —
    /// [`Session::apply`], [`Session::replace`], [`Session::set_camera`],
    /// [`Session::resize`] — each bumping it *after* placing its payload.
    /// The viewer toggles do not count: they restart accumulation anyway,
    /// so their fresh samples already say "new picture". The render thread
    /// reads it at each wave boundary before draining, and stamps the value
    /// on every frame it then publishes: a drained edit is incorporated
    /// whether it applied or was rejected, so a rejection can never wedge
    /// a consumer waiting on its epoch.
    epoch: AtomicU64,
}

/// One publish slot's buffers: the film's four resolved linear averages,
/// rotated as a unit so a frame's beauty and its guides always come from
/// the same resolve. `TRANSFER_SRC` on each: the denoise pass is a host
/// copy (OIDN has no Vulkan device), and the tests read them back.
struct FrameBuffers {
    beauty: Buffer,
    albedo: Buffer,
    normal: Buffer,
    depth: Buffer,
    /// The D-092 debug false-colour, copied from the film's single-shot debug
    /// buffer at publish. Untouched (and unread) unless a [`DebugView`] is
    /// active; the viewer presents it in place of `beauty` when it is.
    debug: Buffer,
}

/// A published frame: the estimator's current best image as a **linear**
/// average — plus its AOVs (the denoiser guides and first-hit depth, from
/// the same resolve) and the metadata a consumer needs to tonemap and
/// present it without reaching back into the renderer. The buffers are
/// shared by [`Arc`] so the render thread can tell — by its strong count —
/// when the consumer has let go and the slot is free to resolve into again.
pub struct Frame {
    buffers: Arc<FrameBuffers>,
    width: u32,
    height: u32,
    /// Samples in the average, for the spp readout.
    samples: u32,
    /// Everything [`crate::stats`] knows as of this publish. Metadata
    /// beside the pixels, never mixed into them: a consumer that only wants
    /// the image reads exactly the bytes it always did.
    stats: Stats,
    /// The session epoch at the wave boundary this frame's accumulation
    /// last crossed — see [`Frame::epoch`].
    epoch: u64,
}

impl Frame {
    /// The linear `ACEScg` beauty average, ready for a [`super::Tonemap`]
    /// to read.
    #[must_use]
    pub fn beauty(&self) -> &Buffer {
        &self.buffers.beauty
    }

    /// The denoiser albedo guide — linear RGBA f32, alpha unused.
    #[must_use]
    pub fn albedo(&self) -> &Buffer {
        &self.buffers.albedo
    }

    /// The denoiser normal guide — world-space shading normals, post
    /// normal-map, RGBA f32 (averaged unnormalized; alpha unused).
    #[must_use]
    pub fn normal(&self) -> &Buffer {
        &self.buffers.normal
    }

    /// Camera-plane z at the first hit, one f32 per pixel; +∞ where every
    /// sample missed.
    #[must_use]
    pub fn depth(&self) -> &Buffer {
        &self.buffers.depth
    }

    /// The `ReSTIR` debug false-colour (RGBA f32) — meaningful only when the
    /// frame was published with a [`DebugView`] active. Already display-ready:
    /// present it through the tonemap's passthrough, not the tone curve.
    #[must_use]
    pub fn debug(&self) -> &Buffer {
        &self.buffers.debug
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

    /// Samples accumulated into this average.
    #[must_use]
    pub fn samples(&self) -> u32 {
        self.samples
    }

    /// Everything measured as of this publish: the sample's per-kernel GPU
    /// time, the interactivity marks, and where the memory went. The one
    /// source every consumer reads — see [`crate::stats`].
    #[must_use]
    pub fn stats(&self) -> &Stats {
        &self.stats
    }

    /// Everything enqueued while [`Session::epoch`] read at most this value
    /// is incorporated in this picture — applied or rejected (D-113). A
    /// consumer that reads the session epoch after its last verb can tell a
    /// settled frame of the *edited* scene from a settled frame of the old
    /// one: the first frame with `frame.epoch() >= that value` has it all.
    #[must_use]
    pub fn epoch(&self) -> u64 {
        self.epoch
    }
}

/// Owns the render thread and the lanes across to it. Dropping it stops
/// the thread and joins, so every GPU resource the thread holds is released
/// before the shared [`Context`] is.
pub struct Session {
    lanes: Arc<Lanes>,
    thread: Option<JoinHandle<Result<()>>>,
}

impl Session {
    /// Spawn the render thread. It takes ownership of `description`,
    /// `scene` (its prepped residency), `renderer`, and a [`Context`]
    /// handle, and starts accumulating `camera` at `width`×`height`
    /// immediately; the first [`Session::take_frame`] to return `Some`
    /// marks the first frame ready. `auto_stop` bounds accumulation: the
    /// thread parks at its sample cap (and optional convergence threshold)
    /// so a settled view releases the GPU.
    ///
    /// # Panics
    ///
    /// If the OS refuses to spawn the render thread — an environment failure
    /// at startup, not something a caller can recover from here.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "distinct owned resources handed to the render thread at startup; \
                  grouping them into a struct would only rename the argument list"
    )]
    pub fn new(
        gpu: Arc<Context>,
        description: SceneDescription,
        scene: Scene,
        renderer: Renderer,
        camera: Camera,
        width: u32,
        height: u32,
        auto_stop: AutoStop,
    ) -> Self {
        let lanes = Arc::new(Lanes {
            inputs: Mutex::new(RenderInputs {
                camera,
                size: (width, height),
                generation: 0,
                render_mode: RenderMode::PathTracer,
                debug_view: DebugView::Off,
                spatial_reuse: true,
                temporal_reuse: true,
                cheap_restart: true,
                max_samples: auto_stop.max_samples,
                noise_threshold: auto_stop.noise_threshold,
                running: true,
            }),
            edits: Mutex::new(Vec::new()),
            edit_error: Mutex::new(None),
            published: Mutex::new(None),
            epoch: AtomicU64::new(0),
        });
        let thread = {
            let lanes = Arc::clone(&lanes);
            std::thread::Builder::new()
                .name("cenote-render".into())
                .spawn(move || render_loop(&gpu, description, scene, renderer, &lanes))
                .expect("spawning the render thread")
        };
        Self {
            lanes,
            thread: Some(thread),
        }
    }

    /// Point the render at a new view — the viewer's orbit control calls this
    /// each time the camera moves. Bumps the generation so the render thread
    /// restarts accumulation from the new pose, and the epoch so the frames
    /// that show it are identifiable ([`Frame::epoch`]).
    ///
    /// # Panics
    ///
    /// If the render thread panicked while holding the input lock — a bug on
    /// that thread, surfaced here rather than silently ignored.
    pub fn set_camera(&self, camera: Camera) {
        {
            let mut inputs = self.lanes.inputs.lock().expect("inputs mutex poisoned");
            inputs.camera = camera;
            inputs.generation += 1;
        }
        self.lanes.epoch.fetch_add(1, Ordering::Release);
    }

    /// Switch the primary-hit estimator — the viewer's `ReSTIR` toggle. The
    /// render thread adopts the change at the next wave boundary and restarts
    /// accumulation, so an A/B flip shows one estimator at a time.
    ///
    /// # Panics
    ///
    /// If the render thread panicked while holding the input lock.
    pub fn set_render_mode(&self, mode: RenderMode) {
        self.lanes
            .inputs
            .lock()
            .expect("inputs mutex poisoned")
            .render_mode = mode;
    }

    /// Select the D-092 debug view the render thread false-colours (or
    /// [`DebugView::Off`]). The debug surface is single-shot, so the change is
    /// live on the next published frame without restarting accumulation.
    ///
    /// # Panics
    ///
    /// If the render thread panicked while holding the input lock.
    pub fn set_debug_view(&self, view: DebugView) {
        self.lanes
            .inputs
            .lock()
            .expect("inputs mutex poisoned")
            .debug_view = view;
    }

    /// Toggle `ReSTIR` spatial reuse (M3 step 4). Meaningful only in
    /// [`RenderMode::Restir`]; the render thread adopts a change and restarts
    /// accumulation, so the on/off images stay unmixed — the interactive form of
    /// the D-090 unbiasedness gate.
    ///
    /// # Panics
    ///
    /// If the render thread panicked while holding the input lock.
    pub fn set_spatial_reuse(&self, enabled: bool) {
        self.lanes
            .inputs
            .lock()
            .expect("inputs mutex poisoned")
            .spatial_reuse = enabled;
    }

    /// Toggle `ReSTIR` temporal reuse — the warm-start from the previous frame's
    /// reservoirs (M3 step 5). Meaningful only in [`RenderMode::Restir`]; the
    /// render thread adopts a change and restarts accumulation, so an on/off
    /// comparison never mixes the two (both converge to the same still, since the
    /// decay ramp hands temporal off on a held camera regardless — D-094).
    ///
    /// # Panics
    ///
    /// If the render thread panicked while holding the input lock.
    pub fn set_temporal_reuse(&self, enabled: bool) {
        self.lanes
            .inputs
            .lock()
            .expect("inputs mutex poisoned")
            .temporal_reuse = enabled;
    }

    /// Toggle the cheap restart frame (M7 step 7a): a restart drawn thin,
    /// leaning on the history the reset kept. Meaningful only in
    /// [`RenderMode::Restir`]; the render thread adopts a change and restarts
    /// accumulation, so a flip never mixes two candidate policies into one
    /// image.
    ///
    /// # Panics
    ///
    /// If the render thread panicked while holding the input lock.
    pub fn set_cheap_restart(&self, enabled: bool) {
        self.lanes
            .inputs
            .lock()
            .expect("inputs mutex poisoned")
            .cheap_restart = enabled;
    }

    /// Note a new render-target size; the render thread rebuilds its film to
    /// match on the next sample. Bumps the epoch — even for a size the render
    /// already has, so a consumer waiting on the verb's frame never wedges.
    ///
    /// # Panics
    ///
    /// If the render thread panicked while holding the input lock.
    pub fn resize(&self, width: u32, height: u32) {
        self.lanes
            .inputs
            .lock()
            .expect("inputs mutex poisoned")
            .size = (width, height);
        self.lanes.epoch.fetch_add(1, Ordering::Release);
    }

    /// Queue a change-set to overlay onto the scene — the lookdev shape.
    /// Edits merge in arrival order and land at the next wave boundary:
    /// stop, apply, re-prep what the edit dirtied, restart accumulation.
    /// A rejected set leaves the scene untouched and surfaces through
    /// [`Session::take_edit_error`]. Bumps the epoch after queueing, so the
    /// frame that incorporates this edit — applied or rejected — is
    /// identifiable ([`Frame::epoch`]).
    ///
    /// # Panics
    ///
    /// If the render thread panicked while holding the edit lock.
    pub fn apply(&self, set: ChangeSet) {
        self.lanes
            .edits
            .lock()
            .expect("edits mutex poisoned")
            .push(SceneEdit::Apply(set));
        self.lanes.epoch.fetch_add(1, Ordering::Release);
    }

    /// Queue a whole-scene replacement — the file-reload shape: `set`
    /// describes the entire scene from empty, and objects it no longer
    /// contains are removed, retiring their GPU residency. Unchanged
    /// objects re-prep nothing, so re-saving an untouched file is free.
    /// Rejections behave as in [`Session::apply`], and the epoch bumps the
    /// same way.
    ///
    /// # Panics
    ///
    /// If the render thread panicked while holding the edit lock.
    pub fn replace(&self, set: ChangeSet) {
        self.lanes
            .edits
            .lock()
            .expect("edits mutex poisoned")
            .push(SceneEdit::Replace(set));
        self.lanes.epoch.fetch_add(1, Ordering::Release);
    }

    /// Take the latest rejected edit, if one hasn't been taken yet. The
    /// render thread keeps rendering its previous scene through a
    /// rejection — this is how a consumer learns the edit didn't land.
    ///
    /// # Panics
    ///
    /// If the render thread panicked while holding the edit-error lock.
    #[must_use]
    pub fn take_edit_error(&self) -> Option<Error> {
        self.lanes
            .edit_error
            .lock()
            .expect("edit-error mutex poisoned")
            .take()
    }

    /// Take the latest published frame, if the render thread has posted a new
    /// one since the last take. `None` means no fresh frame — the consumer
    /// keeps showing the one it already holds.
    ///
    /// # Panics
    ///
    /// If the render thread panicked while holding the publish lock.
    #[must_use]
    pub fn take_frame(&self) -> Option<Frame> {
        self.lanes
            .published
            .lock()
            .expect("published mutex poisoned")
            .take()
    }

    /// The session epoch (D-113): how many picture-changing verbs —
    /// [`apply`](Self::apply), [`replace`](Self::replace),
    /// [`set_camera`](Self::set_camera), [`resize`](Self::resize) — have been
    /// accepted so far. Read it after a verb and hold onto it: the first
    /// frame with [`Frame::epoch`] at or past it incorporates that verb and
    /// everything before it, so "settled *and* current" is one comparison,
    /// with no round-trip into the renderer.
    #[must_use]
    pub fn epoch(&self) -> u64 {
        self.lanes.epoch.load(Ordering::Acquire)
    }

    /// Surface a render-thread failure to the consumer. While the thread runs
    /// this is `Ok(())`; once it has ended early — a GPU error returned from
    /// the loop, or a panic — it joins the thread and returns that, so the
    /// viewer can exit reporting the fault instead of spinning on a renderer
    /// that will publish no more frames. Idempotent: once it has reaped the
    /// thread, later calls are `Ok(())`.
    ///
    /// The loop returns `Ok` only when asked to stop (which is [`Drop`]'s
    /// job), so a thread found finished here has always failed.
    ///
    /// # Errors
    ///
    /// The [`crate::Error`] the render loop returned, or
    /// [`crate::Error::RenderThreadPanicked`] if it panicked.
    pub fn check(&mut self) -> Result<()> {
        // Join only once the thread has actually ended, so this never blocks.
        if self.thread.as_ref().is_some_and(JoinHandle::is_finished)
            && let Some(thread) = self.thread.take()
        {
            return join_render_thread(thread);
        }
        Ok(())
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // Signal stop. A poisoned lock means the thread panicked mid-flight
        // holding it; recover the guard rather than panicking again here in a
        // Drop, since the join below is what surfaces that panic.
        self.lanes
            .inputs
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .running = false;
        if let Some(thread) = self.thread.take() {
            // Join so the thread's Renderer, Scene, Film, and Context handle
            // are dropped here — before this crate's owner drops the Context,
            // which checks that nothing outlives it. `check` normally reaps a
            // failed thread and hands the error to the viewer; if it died in
            // the gap before shutdown there is no caller left to return to,
            // so a leftover error is logged as the last word.
            if let Err(err) = join_render_thread(thread) {
                log::error!("render thread ended with an error: {err}");
            }
        }
    }
}

/// Join the render thread and flatten its outcome: an error the loop returned
/// passes straight through, while a panic becomes an
/// [`Error::RenderThreadPanicked`] carrying whatever message the panic left.
fn join_render_thread(thread: JoinHandle<Result<()>>) -> Result<()> {
    match thread.join() {
        Ok(result) => result,
        Err(panic) => {
            // A panic payload is usually the `&str` or `String` passed to
            // `panic!`; anything else we can only name generically.
            let message = panic
                .downcast_ref::<&str>()
                .map(|s| (*s).to_owned())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "the panic payload was not a string".to_owned());
            Err(Error::RenderThreadPanicked(message))
        }
    }
}

/// The render thread's body: accumulate `scene` into a film sized to the
/// latest inputs, folding queued edits in at wave boundaries and publishing
/// a resolved average on the throttle. Returns when the running flag clears,
/// or early on the first device fault.
#[expect(
    clippy::too_many_lines,
    reason = "one wave of the loop, in the order the waves run; splitting it \
              would scatter the applied-state bookkeeping across helpers"
)]
fn render_loop(
    gpu: &Context,
    mut description: SceneDescription,
    mut scene: Scene,
    mut renderer: Renderer,
    lanes: &Lanes,
) -> Result<()> {
    log::debug!("render thread started");
    // The render target: the film and its pair of publish buffers, sized
    // together and rebuilt together when the requested size changes.
    // `applied_generation` tracks which view is in the scene, so a bump
    // restarts accumulation.
    let mut target: Option<(Film, [Arc<FrameBuffers>; 2])> = None;
    let mut applied_size = (0, 0);
    let mut applied_generation = 0;
    // Which estimator is in the renderer, so a viewer toggle restarts
    // accumulation on the switch (matching `Session::new`'s default).
    let mut applied_render_mode = RenderMode::PathTracer;
    let mut applied_spatial_reuse = true;
    let mut applied_temporal_reuse = true;
    let mut applied_cheap_restart = true;
    // The auto-stop threshold currently in the renderer (M3 step 6c). `None` =
    // the renderer's default; a change is adopted without a reset, since it only
    // moves what counts as converged, not the beauty.
    let mut applied_noise_threshold: Option<f32> = None;
    let mut last_publish: Option<Instant> = None;
    // The epoch stamped on the most recent successful publish, so a parked
    // thread can see the counter move past its settled frame and republish.
    let mut published_epoch = 0;
    // The measurement spine (see `crate::stats`). The recorder carries the
    // running state across waves — so the frame forced out when the render
    // parks still reports the last real sample — and the timer stamps the
    // pass boundaries of the waves it samples. A device without queue
    // timestamps hands back `None` and the loop runs exactly as it did
    // before, reporting wall-clock alone.
    let mut recorder = Recorder::new();
    let mut timer = gpu.create_pass_timer(PassTimer::WAVE_CAPACITY)?;
    let mut last_memory_sample: Option<Instant> = None;
    // The film's sample count as of the previous wave; a drop means
    // something restarted accumulation.
    let mut last_samples = 0;
    // Whether accumulation has settled (hit the sample cap or converged) and the
    // thread is idling. Cleared whenever a reset below zeroes the film.
    let mut parked = false;
    // Dirt whose re-prep was rejected (this build can't render the edited
    // description). It survives here so the *next* applied edit retries the
    // whole backlog — nothing goes silently stale.
    let mut stale = Dirty::default();

    loop {
        // The wave's epoch, read *before* the inputs snapshot and the edit
        // drain. A verb bumps the counter only after placing its payload, so
        // everything counted here is visible to this wave — the stamp can
        // undercount a racing verb, never claim one it missed (D-113).
        let epoch = lanes.epoch.load(Ordering::Acquire);
        let input = *lanes.inputs.lock().expect("inputs mutex poisoned");
        if !input.running {
            log::debug!("render thread stopping");
            return Ok(());
        }
        let (width, height) = input.size;
        if width == 0 || height == 0 {
            // Minimized: nothing to render until the window comes back.
            // Edits queue meanwhile and land with the first visible wave.
            std::thread::sleep(IDLE_NAP);
            continue;
        }

        // A resize restarts by building a fresh (empty) film and publish
        // buffers, adopting the latest view with them.
        if input.size != applied_size {
            log::debug!("film rebuilt at {width}×{height}");
            target = Some((
                Film::new(gpu, width, height)?,
                publish_buffers(gpu, width, height)?,
            ));
            *scene.camera_mut() = input.camera;
            applied_size = input.size;
            applied_generation = input.generation;
            last_publish = None;
        }
        let (film, frames) = target.as_mut().expect("sized by the resize branch above");
        // Queued edits land here, at the wave boundary: stop, apply,
        // re-prep, restart accumulation from sample 0.
        if apply_edits(gpu, lanes, &mut description, &mut scene, &mut stale)? {
            film.reset();
            last_publish = None;
        }
        // A plain view change resets the existing film instead.
        if input.generation != applied_generation {
            log::debug!("camera adopted; accumulation restarts");
            *scene.camera_mut() = input.camera;
            film.reset();
            applied_generation = input.generation;
        }
        // The estimator switch restarts accumulation (the two converge to the
        // same image, so this keeps an A/B flip from mixing them). The debug
        // view is single-shot and live, so it just flows into the next sample.
        if input.render_mode != applied_render_mode {
            log::debug!("render mode adopted; accumulation restarts");
            renderer.set_render_mode(input.render_mode);
            applied_render_mode = input.render_mode;
            film.reset();
            last_publish = None;
        }
        // The spatial-reuse toggle is the other estimator switch, restarted the
        // same way so an on/off comparison never mixes the two.
        if input.spatial_reuse != applied_spatial_reuse {
            log::debug!("spatial reuse adopted; accumulation restarts");
            renderer.set_spatial_reuse(input.spatial_reuse);
            applied_spatial_reuse = input.spatial_reuse;
            film.reset();
            last_publish = None;
        }
        // The temporal-reuse toggle restarts the same way. Note the film reset
        // (accumulation from sample 0) is also the decay clock's reset, so
        // flipping temporal on restarts its warm-start from a fresh ramp — the
        // on state is visibly the warm-start, the off state its absence.
        if input.temporal_reuse != applied_temporal_reuse {
            log::debug!("temporal reuse adopted; accumulation restarts");
            renderer.set_temporal_reuse(input.temporal_reuse);
            applied_temporal_reuse = input.temporal_reuse;
            film.reset();
            last_publish = None;
        }
        // The cheap restart restarts too: it only ever acts on sample 0, so a
        // flip mid-accumulation would land on nothing until the next reset, and
        // the A/B would silently compare an arm against itself.
        if input.cheap_restart != applied_cheap_restart {
            log::debug!("cheap restart adopted; accumulation restarts");
            renderer.set_cheap_restart(input.cheap_restart);
            applied_cheap_restart = input.cheap_restart;
            film.reset();
            last_publish = None;
        }
        // Every restart above rewound the film's sample count, and that is
        // the one signal all of them share — a resize, an edit, a camera
        // move, either reuse toggle, the cheap restart. (It is also the signal
        // the cheap restart reads, which is why one cheap frame covers all of
        // them and not just camera motion.) Re-arming the interactivity marks off
        // it keeps the measurement honest without seven call sites that would
        // drift apart the first time an eighth reset is added — as one just was.
        if film.samples() < last_samples {
            recorder.restart();
        }
        last_samples = film.samples();
        renderer.set_debug_view(input.debug_view);
        // The auto-stop threshold changes only what the accumulate kernel counts
        // as converged, so it is adopted without a reset — the per-sample count
        // self-heals on the next sample. `None` restores the renderer default.
        if input.noise_threshold != applied_noise_threshold {
            renderer.set_noise_threshold(
                input.noise_threshold.unwrap_or(Renderer::NOISE_THRESHOLD),
            );
            applied_noise_threshold = input.noise_threshold;
        }
        // Any reset above zeroed the film, which wakes a parked render.
        if film.samples() == 0 {
            parked = false;
        }
        // Parked: a settled render idles here, until one of the resets above
        // zeroes the film and clears the flag. One duty remains: a verb that
        // restarts nothing — a visual no-op edit, a rejection, a same-size
        // resize — still moved the epoch, and a consumer may be waiting on a
        // frame that carries it, so the settled image goes out again under
        // the fresh stamp (same picture, one resolve). Both slots busy just
        // retries on the next nap.
        if parked {
            if epoch > published_epoch
                && publish(gpu, &renderer, film, frames, lanes, &recorder, epoch)?
            {
                published_epoch = epoch;
            }
            std::thread::sleep(IDLE_NAP);
            continue;
        }

        // Stop accumulating once the film has hit its sample cap or (with a
        // threshold set) converged — a settled render must not pin the GPU. The
        // cap is checked first, so the `converged_fraction` readback runs only
        // when a threshold is set and the cap has not already fired.
        let complete = film.samples() >= input.max_samples
            || auto_stopped(gpu, film, input.noise_threshold)?;
        if complete {
            // Force the settled image out once (past the throttle) so the
            // converged frame is definitely the latest, then park. If both slots
            // are busy the publish is retried on the next tick.
            if publish(gpu, &renderer, film, frames, lanes, &recorder, epoch)? {
                parked = true;
                published_epoch = epoch;
            }
            std::thread::sleep(IDLE_NAP);
            continue;
        }

        // Every frame is bracketed; one in `PassTimer::BREAKDOWN_INTERVAL`
        // is also resolved kernel by kernel.
        let started = Instant::now();
        // Read before the wave, not after: the candidate count is picked from
        // the film's state going in, and `accumulate_timed` has moved it on by
        // the time it returns.
        let candidates = renderer.candidate_count(film);
        let passes = renderer.accumulate_timed(gpu, &scene, film, timer.as_mut())?;
        // `stats::Frame` is the measurement of a sample; the `Frame` this
        // module publishes is the pixels. Spelled out so the two never read
        // as one.
        recorder.record(stats::Frame {
            cpu: started.elapsed(),
            passes,
            size: (film.width(), film.height()),
            samples: film.samples(),
            candidates,
        });
        if last_memory_sample.is_none_or(|at| at.elapsed() >= MEMORY_SAMPLE_INTERVAL) {
            recorder.memory(gpu.memory());
            last_memory_sample = Some(Instant::now());
        }

        // Publish on the throttle — which widens as the image converges (see
        // `publish_interval`) — but only into a buffer no consumer still holds.
        // If both are busy, skip: the next tick catches up, and the renderer
        // never waits on the consumer.
        if last_publish.is_none_or(|at| at.elapsed() >= publish_interval(film.samples()))
            && publish(gpu, &renderer, film, frames, lanes, &recorder, epoch)?
        {
            last_publish = Some(Instant::now());
            published_epoch = epoch;
        }
    }
}

/// Resolve the film's current average into a free publish slot and post it for
/// the consumer — stamped with `epoch`, the wave boundary this accumulation
/// last crossed (D-113) — returning whether a slot was free. Both slots busy
/// means the consumer still holds the last two frames — the caller retries
/// next tick and the renderer never blocks. Lifts the debug false-colour into the slot too
/// when a [`DebugView`] is active (`accumulate` allocated the film's buffer then).
fn publish(
    gpu: &Context,
    renderer: &Renderer,
    film: &Film,
    frames: &[Arc<FrameBuffers>; 2],
    lanes: &Lanes,
    recorder: &Recorder,
    epoch: u64,
) -> Result<bool> {
    let Some(free) = frames.iter().find(|frame| Arc::strong_count(frame) == 1) else {
        return Ok(false);
    };
    renderer.resolve(
        gpu,
        film,
        &ResolveTargets {
            beauty: &free.beauty,
            albedo: &free.albedo,
            normal: &free.normal,
            depth: &free.depth,
        },
    )?;
    // The debug surface isn't a film accumulation channel — it's a single-shot
    // buffer the wave just wrote — so it's lifted into the frame slot by a plain
    // copy rather than a resolve.
    if let Some(debug) = renderer.debug_buffer(film) {
        gpu.submit_passes(&[Pass::CopyBuffer {
            src: debug,
            dst: &free.debug,
            size: u64::from(film.width()) * u64::from(film.height()) * 16,
        }])?;
    }
    let frame = Frame {
        buffers: Arc::clone(free),
        width: film.width(),
        height: film.height(),
        samples: film.samples(),
        stats: recorder.stats(),
        epoch,
    };
    *lanes.published.lock().expect("published mutex poisoned") = Some(frame);
    Ok(true)
}

/// Whether an auto-stop threshold is set and the film has reached it — the
/// convergence-idle stop. Below [`Renderer::CONVERGENCE_MIN_SAMPLES`] the metric
/// is untrusted (D-094), so it never stops there; above it, it reads the film's
/// converged fraction (a 4-byte device readback) against
/// [`Renderer::CONVERGENCE_TARGET`]. `None` never stops — the cap alone bounds it.
fn auto_stopped(gpu: &Context, film: &Film, threshold: Option<f32>) -> Result<bool> {
    if threshold.is_none() || film.samples() < Renderer::CONVERGENCE_MIN_SAMPLES {
        return Ok(false);
    }
    Ok(film.converged_fraction(gpu)? >= Renderer::CONVERGENCE_TARGET)
}

/// Drain and apply the queued edits, re-prepping what they dirtied. True
/// means the visible scene changed and accumulation must restart. A
/// rejected change-set or re-prep posts to the edit-error lane and keeps
/// the previous scene; the dirt it left in `stale` retries after the next
/// edit that applies. Only device faults return `Err`.
fn apply_edits(
    gpu: &Context,
    lanes: &Lanes,
    description: &mut SceneDescription,
    scene: &mut Scene,
    stale: &mut Dirty,
) -> Result<bool> {
    let edits = std::mem::take(&mut *lanes.edits.lock().expect("edits mutex poisoned"));
    if edits.is_empty() {
        return Ok(false);
    }
    let mut applied = false;
    for edit in edits {
        let result = match edit {
            SceneEdit::Apply(set) => description.apply(&set),
            SceneEdit::Replace(set) => {
                let mut fresh = SceneDescription::new();
                fresh.apply(&set).map(|()| description.replace(fresh))
            }
        };
        match result {
            Ok(()) => applied = true,
            Err(error) => post_edit_error(lanes, error),
        }
    }
    stale.merge(description.take_dirty());
    if !applied || stale.is_empty() {
        return Ok(false);
    }
    // Settings carry no residency, so a settings-only edit must not throw
    // away the accumulated image.
    let visual = stale
        .changed
        .iter()
        .chain(&stale.removed)
        .any(|(kind, _)| *kind != Kind::Settings);
    match scene.update(gpu, description, stale) {
        Ok(()) => {
            log::debug!("scene edits applied; accumulation restarts");
            *stale = Dirty::default();
            Ok(visual)
        }
        // This build can't render the edited description; the previous
        // residency keeps rendering and `stale` holds the backlog.
        Err(error @ Error::Scene(_)) => {
            post_edit_error(lanes, error);
            Ok(false)
        }
        Err(fatal) => Err(fatal),
    }
}

/// Post a rejected edit for the consumer, latest-wins.
fn post_edit_error(lanes: &Lanes, error: Error) {
    log::debug!("scene edit rejected: {error}");
    *lanes.edit_error.lock().expect("edit-error mutex poisoned") = Some(error);
}

/// The pair of publish slots — the double-buffer the render thread rotates
/// through — each the film's four full-frame linear averages: what the
/// resolve kernel writes and a consumer reads by device address.
fn publish_buffers(gpu: &Context, width: u32, height: u32) -> Result<[Arc<FrameBuffers>; 2]> {
    let texels = u64::from(width) * u64::from(height);
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
        | vk::BufferUsageFlags::TRANSFER_SRC;
    let buffer = |name: &str, bytes: u64| -> Result<Buffer> {
        gpu.create_buffer(name, bytes, usage, MemoryLocation::GpuOnly)
    };
    let slot = || -> Result<Arc<FrameBuffers>> {
        Ok(Arc::new(FrameBuffers {
            beauty: buffer("session.frame", texels * 16)?,
            albedo: buffer("session.frame.albedo", texels * 16)?,
            normal: buffer("session.frame.normal", texels * 16)?,
            depth: buffer("session.frame.depth", texels * 4)?,
            // Also a copy target for the film's debug buffer at publish.
            debug: gpu.create_buffer(
                "session.frame.debug",
                texels * 16,
                usage | vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuOnly,
            )?,
        }))
    };
    Ok([slot()?, slot()?])
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use glam::Vec3;

    use super::*;
    use crate::scene::changeset::{MaterialPatch, Op};
    use crate::scene::description::Texturable;

    /// A demo session that never parks — the sample cap is effectively
    /// unbounded, so the accumulation-and-edit tests see samples climb freely.
    fn demo_session(gpu: &Arc<Context>, size: u32) -> Session {
        demo_session_with(
            gpu,
            size,
            AutoStop {
                max_samples: u32::MAX,
                noise_threshold: None,
            },
        )
    }

    /// A demo session with an explicit auto-stop policy: the description, its
    /// prepped scene, and the thread already accumulating.
    fn demo_session_with(gpu: &Arc<Context>, size: u32, auto_stop: AutoStop) -> Session {
        let mut description = SceneDescription::new();
        description.apply(&ChangeSet::demo()).expect("demo applies");
        let scene = Scene::prep(gpu, &mut description).expect("demo preps");
        let camera = *scene.camera();
        let renderer = Renderer::new(gpu).expect("renderer");
        Session::new(
            Arc::clone(gpu),
            description,
            scene,
            renderer,
            camera,
            size,
            size,
            auto_stop,
        )
    }

    /// The render thread runs and publishes: spin one up on the demo scene,
    /// and it must post a frame at the requested size with samples on it. This
    /// is the whole actor end to end — spawn, snapshot inputs, accumulate,
    /// resolve, publish — that a single-threaded test can't exercise.
    #[test]
    fn session_publishes_accumulating_frames() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let gpu = Arc::new(gpu);
        let session = demo_session(&gpu, 64);

        // Wait for the first publish, then for a later one — the sample count
        // must climb, proving the thread keeps accumulating, not just resolves
        // one frame.
        let first = wait_for_frame(&session);
        assert_eq!((first.width(), first.height()), (64, 64));
        assert!(first.samples() > 0, "first frame has no samples");
        let later = wait_for_frame(&session);
        assert!(
            later.samples() > first.samples(),
            "accumulation stalled: {} then {}",
            first.samples(),
            later.samples()
        );

        // The frame carries its AOVs, resolved alongside the beauty: the
        // demo's albedo guide is nowhere black (lit surfaces and a white-
        // albedo sky), and every buffer is full-frame.
        assert_eq!(later.albedo().size(), 64 * 64 * 16);
        assert_eq!(later.normal().size(), 64 * 64 * 16);
        assert_eq!(later.depth().size(), 64 * 64 * 4);
        let albedo: Vec<f32> = bytemuck::pod_collect_to_vec(
            &gpu.download_buffer(later.albedo())
                .expect("download albedo"),
        );
        assert!(
            albedo
                .chunks_exact(4)
                .all(|texel| texel[..3].iter().sum::<f32>() > 0.0),
            "the demo's albedo guide should be lit everywhere"
        );
    }

    /// The edit channel end to end: a queued material edit lands at a wave
    /// boundary and restarts accumulation — the sample counter, which only
    /// ever climbs otherwise, must drop back and start over.
    #[test]
    fn an_edit_restarts_accumulation() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let gpu = Arc::new(gpu);
        let session = demo_session(&gpu, 64);
        let mut high = 0;
        while high < 3 {
            high = high.max(wait_for_frame(&session).samples());
        }

        session.apply(ChangeSet {
            ops: vec![Op::Material(Box::new(MaterialPatch {
                base_color: Some(Texturable::Constant([0.9, 0.1, 0.1])),
                ..MaterialPatch::new("floor")
            }))],
        });
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let frame = wait_for_frame(&session);
            if frame.samples() < high {
                break; // accumulation restarted from the edited scene
            }
            high = high.max(frame.samples());
            assert!(Instant::now() < deadline, "the edit never landed");
        }
        assert!(session.take_edit_error().is_none());
    }

    /// A rejected edit surfaces without stopping the renderer: the fault
    /// arrives on the edit-error lane while frames keep flowing.
    #[test]
    fn a_rejected_edit_surfaces_and_rendering_continues() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let gpu = Arc::new(gpu);
        let mut session = demo_session(&gpu, 64);
        wait_for_frame(&session);

        session.apply(ChangeSet {
            ops: vec![Op::Remove(Kind::Material, "no-such-material".into())],
        });
        let deadline = Instant::now() + Duration::from_secs(10);
        let error = loop {
            if let Some(error) = session.take_edit_error() {
                break error;
            }
            assert!(Instant::now() < deadline, "the rejection never surfaced");
            std::thread::sleep(Duration::from_millis(2));
        };
        assert!(error.to_string().contains("no-such-material"), "{error}");
        // Still alive and still accumulating the previous scene.
        session.check().expect("render thread survives a rejection");
        let a = wait_for_frame(&session).samples();
        let b = wait_for_frame(&session).samples();
        assert!(b > a, "rendering stalled after a rejected edit");
    }

    /// The sample cap parks the render thread (M3 step 6c): with a low
    /// `max_samples`, accumulation climbs to the cap and stops there — the count
    /// never overshoots it, and once settled the thread idles instead of
    /// publishing on, so a converged view stops pinning the GPU.
    #[test]
    fn the_sample_cap_parks_accumulation() {
        const CAP: u32 = 8;
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let gpu = Arc::new(gpu);
        let mut session = demo_session_with(
            &gpu,
            32,
            AutoStop {
                max_samples: CAP,
                noise_threshold: None,
            },
        );

        // Accumulation reaches the cap without ever overshooting it.
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let frame = wait_for_frame(&session);
            assert!(
                frame.samples() <= CAP,
                "accumulated past the cap: {}",
                frame.samples()
            );
            if frame.samples() == CAP {
                break;
            }
            assert!(Instant::now() < deadline, "never reached the cap");
        }

        // Parked: drain any queued frame, then confirm the idle thread publishes
        // nothing further — and is still alive, not crashed into the cap.
        while session.take_frame().is_some() {}
        std::thread::sleep(Duration::from_millis(200));
        session.check().expect("render thread survives parking");
        if let Some(frame) = session.take_frame() {
            assert_eq!(frame.samples(), CAP, "resumed past the cap while parked");
        }
    }

    /// The picture-changing verbs each bump the session epoch once, and the
    /// viewer toggles don't: a toggle restarts accumulation anyway, so its
    /// fresh sample count already says "new picture" — only the verbs a
    /// remote consumer acknowledges need the stamp (D-113).
    #[test]
    fn picture_changing_verbs_bump_the_epoch() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let gpu = Arc::new(gpu);
        let session = demo_session(&gpu, 32);
        assert_eq!(session.epoch(), 0);

        session.set_camera(Camera {
            position: Vec3::new(0.0, 1.0, 4.0),
            look_at: Vec3::ZERO,
            up: Vec3::Y,
            vfov_degrees: 45.0,
            lens: None,
        });
        assert_eq!(session.epoch(), 1);
        // Even a resize to the size the render already has counts: a consumer
        // waiting on the verb's frame must never wedge.
        session.resize(32, 32);
        assert_eq!(session.epoch(), 2);
        session.apply(ChangeSet::default());
        assert_eq!(session.epoch(), 3);
        session.replace(ChangeSet::demo());
        assert_eq!(session.epoch(), 4);

        session.set_render_mode(RenderMode::Restir);
        session.set_debug_view(DebugView::Off);
        session.set_spatial_reuse(false);
        session.set_temporal_reuse(false);
        assert_eq!(session.epoch(), 4, "viewer toggles must not bump the epoch");
    }

    /// An applied edit's stamp reaches the published frame: accumulation
    /// crosses a wave boundary that drained the edit, and every frame it
    /// publishes from there carries the moved epoch (D-113).
    #[test]
    fn an_applied_edit_advances_the_frame_epoch() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let gpu = Arc::new(gpu);
        let session = demo_session(&gpu, 64);
        let first = wait_for_frame(&session);
        assert_eq!(first.epoch(), 0, "no verbs yet, so no stamp");

        session.apply(ChangeSet {
            ops: vec![Op::Material(Box::new(MaterialPatch {
                base_color: Some(Texturable::Constant([0.9, 0.1, 0.1])),
                ..MaterialPatch::new("floor")
            }))],
        });
        wait_for(&session, "a frame carrying the edit's epoch", |frame| {
            frame.epoch() >= 1
        });
        assert!(session.take_edit_error().is_none());
    }

    /// The delivery guarantee behind the epoch: an edit that changes nothing
    /// visible (the equality gate leaves no dirt, so nothing restarts and no
    /// new picture is coming) still gets a frame carrying its stamp — the
    /// parked thread republishes the settled image, where otherwise a
    /// consumer waiting on that epoch would wait forever (D-113).
    #[test]
    fn a_noop_edit_republishes_the_parked_frame() {
        const CAP: u32 = 8;
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let gpu = Arc::new(gpu);
        let session = demo_session_with(
            &gpu,
            32,
            AutoStop {
                max_samples: CAP,
                noise_threshold: None,
            },
        );
        wait_for(&session, "the settled frame", |frame| frame.samples() == CAP);

        // A real edit first: the scene changes, resettles at the cap, and the
        // settled frame carries the edit's stamp.
        let patch = ChangeSet {
            ops: vec![Op::Material(Box::new(MaterialPatch {
                base_color: Some(Texturable::Constant([0.9, 0.1, 0.1])),
                ..MaterialPatch::new("floor")
            }))],
        };
        session.apply(patch.clone());
        wait_for(&session, "the edited scene settled", |frame| {
            frame.samples() == CAP && frame.epoch() >= 1
        });

        // The identical patch again: a genuine visual no-op. Only the
        // republish can carry the fresh stamp out — and it must arrive with
        // the settled sample count, not a restarted accumulation.
        session.apply(patch);
        wait_for(&session, "the republished settled frame", |frame| {
            frame.samples() == CAP && frame.epoch() >= 2
        });
        assert!(session.take_edit_error().is_none());
    }

    /// A rejected edit counts as incorporated: the drain moves the stamp past
    /// it even though the scene is untouched, so a rejection can never wedge
    /// a consumer waiting on its epoch — the settled image returns under the
    /// fresh stamp and the fault rides the error lane as usual (D-113).
    #[test]
    fn a_rejected_edit_still_advances_the_frame_epoch() {
        const CAP: u32 = 8;
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let gpu = Arc::new(gpu);
        let session = demo_session_with(
            &gpu,
            32,
            AutoStop {
                max_samples: CAP,
                noise_threshold: None,
            },
        );
        wait_for(&session, "the settled frame", |frame| frame.samples() == CAP);

        session.apply(ChangeSet {
            ops: vec![Op::Remove(Kind::Material, "no-such-material".into())],
        });
        wait_for(&session, "the rejection's epoch on a settled frame", |frame| {
            frame.samples() == CAP && frame.epoch() >= 1
        });
        // The rejection was posted at the same wave boundary, before the
        // republish, so it is already waiting on the error lane.
        let error = session
            .take_edit_error()
            .expect("the rejection surfaces alongside the republish");
        assert!(error.to_string().contains("no-such-material"), "{error}");
    }

    /// The publish gap widens with the sample count and clamps at both ends
    /// (M3 step 6c): every frame early, a few per second once converging.
    #[test]
    fn publish_interval_grows_and_clamps() {
        // Below the first step it holds at the floor; a later step is strictly
        // wider; and it is monotonic and bounded across a full sweep.
        assert_eq!(publish_interval(0), PUBLISH_INTERVAL_MIN);
        assert_eq!(publish_interval(PUBLISH_INTERVAL_STEP), PUBLISH_INTERVAL_MIN);
        assert!(publish_interval(2 * PUBLISH_INTERVAL_STEP) > PUBLISH_INTERVAL_MIN);
        let mut last = Duration::ZERO;
        for samples in (0..20_000).step_by(37) {
            let interval = publish_interval(samples);
            assert!(interval >= last, "publish interval dipped at {samples}");
            assert!((PUBLISH_INTERVAL_MIN..=PUBLISH_INTERVAL_MAX).contains(&interval));
            last = interval;
        }
        assert_eq!(publish_interval(u32::MAX), PUBLISH_INTERVAL_MAX);
    }

    /// Poll `take_frame` until one appears, with a generous timeout so a slow
    /// machine doesn't flake — the render thread posts its first frame within
    /// milliseconds on any real GPU.
    fn wait_for_frame(session: &Session) -> Frame {
        wait_for(session, "a published frame", |_| true)
    }

    /// Poll `take_frame` until a frame satisfies `pred`, on the same generous
    /// deadline as [`wait_for_frame`]; `what` names the wait in the timeout
    /// message. Non-matching frames are consumed — each test states its full
    /// condition rather than assuming an earlier wait left frames behind.
    fn wait_for(session: &Session, what: &str, pred: impl Fn(&Frame) -> bool) -> Frame {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(frame) = session.take_frame()
                && pred(&frame)
            {
                return frame;
            }
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            std::thread::sleep(Duration::from_millis(2));
        }
    }
}
