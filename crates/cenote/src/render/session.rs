//! The render loop as an actor: a dedicated thread that accumulates as fast
//! as the GPU allows, so a consumer's display refresh never paces the
//! renderer. The viewer is the first consumer; a scene-graph delegate could
//! be a second — the concurrency lives here, once, not in each of them.
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
//!   restarts accumulation from sample 0.
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
//! without reaching into the renderer. See
//! [`Lanes::epoch`] for the counting rules.
//!
//! Beside the epoch rides a timestamp, for the other half of the same
//! question. Every verb records when it happened, and the wave boundary that
//! acts on it measures its latency from *there* rather than from the
//! boundary — so the wait for the boundary, which can be a whole sample and
//! is invisible from inside the renderer, is part of what a click costs
//! rather than something in front of it. What the wave then does with the
//! verb it reports as [`stats::Phases`]. See [`Restart`].
//!
//! Two frame buffers, not a triple-buffered mailbox: the render thread
//! resolves only into a buffer no one else references (a strong-count of one
//! means "in the pool alone"), and if both are busy it simply skips that
//! publish and keeps accumulating. So a slow consumer can never see a buffer
//! torn by an in-flight resolve, and the renderer never blocks on the
//! consumer. The strong count is a sound "free" test only because every
//! consumer submission blocks: a [`Frame`] drops strictly after the GPU work
//! that read its buffer completed. Anything that removes those blocking
//! fences — timeline-semaphore pacing — has to revisit this protocol.
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

use super::{Film, Renderer, ResolveTargets};
use crate::error::{Error, Result};
use crate::gpu::{Buffer, Context, MemoryLocation, PassTimer};
use crate::scene::changeset::{ChangeSet, Dirty, Kind};
use crate::scene::description::{SceneDescription, Settings};
use crate::scene::{Camera, Scene};
use crate::stats::{self, Phases, Recorder, Stats};

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

/// How long after the last change the picture still counts as unsettled —
/// long enough to outlast the gap between two moves of a travelling mouse,
/// or between two edits of a dragged slider. A hold rather than a flag from
/// the consumer because there is no verb meaning "stopped": a mouse wheel
/// has no release event, so a dolly would need a timeout beside its flag
/// anyway, and neither does the edit API.
const CHANGE_HOLD: Duration = Duration::from_millis(50);

/// The most an unsettled picture will divide its resolution by. Past a
/// quarter of the window's width a blit stops reading as a soft image and
/// starts reading as a broken one.
const PREVIEW_DIVISOR_MAX: u32 = 4;

/// How often the render thread re-reads device memory. The allocations move
/// only when the scene or the target does, so a per-wave read would be
/// precision theatre over numbers that did not change.
const MEMORY_SAMPLE_INTERVAL: Duration = Duration::from_secs(1);

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
    /// The frame time a changing picture aims at, or `None` — the default —
    /// to always render at full resolution. See
    /// [`Session::set_preview_target`].
    preview_target: Option<Duration>,
    /// When the verb that last wrote this struct happened — the origin the
    /// latency of a camera move or a resize is measured from. Latest-wins
    /// like the rest of the struct: a move that lands while the previous
    /// one is still waiting for the wave boundary supersedes it, and it is
    /// the newer one the person is waiting on.
    stamped: Instant,
    /// Cleared to stop the thread; checked at the top of every iteration.
    running: bool,
}

/// One queued scene edit — the two verbs a change-set can arrive as, each
/// carrying when it was queued so the wait for the next wave boundary lands
/// inside the latency it caused rather than in front of it.
enum SceneEdit {
    /// Overlay onto the current description ([`SceneDescription::apply`]).
    Apply { at: Instant, set: ChangeSet },
    /// The set describes the whole scene from empty; the description
    /// becomes it, diffing for dirt ([`SceneDescription::replace`]).
    Replace { at: Instant, set: ChangeSet },
}

impl SceneEdit {
    /// When this edit was queued.
    fn at(&self) -> Instant {
        match self {
            SceneEdit::Apply { at, .. } | SceneEdit::Replace { at, .. } => *at,
        }
    }
}

/// A restart and what it cost to get to: the verb behind it, and the phases
/// of the work the render thread did before the first sample could start.
///
/// One per wave at most. Every reset the loop can take produces one — an
/// edit drain fills the prep phases, a camera move or a resize leaves them
/// zero because it does none of that work, and both carry the origin that
/// makes the wait for this boundary part of the measurement.
struct Restart {
    /// When the verb happened. Handed to
    /// [`Recorder::restart`](crate::stats::Recorder::restart) as the origin.
    origin: Instant,
    /// What has been clocked so far. The sample and the remainder are the
    /// recorder's to fill once the mark lands.
    phases: Phases,
}

impl Restart {
    /// A restart with a verb behind it but no prep to report: a camera move
    /// or a resize, whose whole cost before the sample is the wait for this
    /// wave boundary.
    fn waited(origin: Instant) -> Self {
        Self {
            origin,
            phases: Phases {
                before: origin.elapsed(),
                ..Phases::default()
            },
        }
    }
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
    /// The session epoch: a count of the picture-changing verbs —
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
    /// Whether this frame is the render's final answer — see
    /// [`Frame::converged`].
    converged: bool,
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

    /// Whether the render that produced this frame is done: it spent its
    /// sample budget, or reached the convergence threshold the settings
    /// asked for. The session's own verdict, decided by the same test that
    /// parks the render thread — a consumer waiting for a still frame must
    /// read this and never re-derive it from [`Frame::samples`], which
    /// cannot see an early stop and would wait forever for a count the
    /// render has already decided not to reach.
    #[must_use]
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// Everything measured as of this publish: the sample's per-kernel GPU
    /// time, the interactivity marks, and where the memory went. The one
    /// source every consumer reads — see [`crate::stats`].
    #[must_use]
    pub fn stats(&self) -> &Stats {
        &self.stats
    }

    /// Everything enqueued while [`Session::epoch`] read at most this value
    /// is incorporated in this picture — applied or rejected. A
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
    /// The frame time to hand [`Session::set_preview_target`] unless there is
    /// a reason to want another: one vsync at 60 Hz.
    pub const PREVIEW_TARGET: Duration = Duration::from_millis(16);

    /// Spawn the render thread. It takes ownership of `description`,
    /// `scene` (its prepped residency), `renderer`, and a [`Context`]
    /// handle, and starts accumulating `camera` at `width`×`height`
    /// immediately; the first [`Session::take_frame`] to return `Some`
    /// marks the first frame ready.
    ///
    /// What bounds accumulation is the description's own
    /// [`Settings`](crate::scene::description::Settings) — the sample
    /// budget, the optional convergence threshold, and the path-length cap
    /// — re-read at every wave boundary. A host with a policy of its own
    /// authors it into the scene like any other value; there is no second
    /// channel for it, so an edit and the settings it carries can never
    /// disagree about which render is running.
    ///
    /// `load` is where the time before this call went, as
    /// [`Scene::prep_timed`] reports it. The render thread has no way to
    /// measure a load it arrived after, so it is handed the breakdown
    /// instead; pass [`Phases::default`] when there was no load to speak of
    /// and the startup mark should stay unattributed.
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
        load: Phases,
    ) -> Self {
        let lanes = Arc::new(Lanes {
            inputs: Mutex::new(RenderInputs {
                camera,
                size: (width, height),
                generation: 0,
                preview_target: None,
                stamped: Instant::now(),
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
                .spawn(move || render_loop(&gpu, description, scene, renderer, &lanes, load))
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
    pub fn set_camera(&self, camera: Camera) {
        {
            let mut inputs = self.lanes.inputs.lock().expect("inputs mutex poisoned");
            inputs.camera = camera;
            inputs.generation += 1;
            inputs.stamped = Instant::now();
        }
        self.lanes.epoch.fetch_add(1, Ordering::Release);
    }

    /// Note a new render-target size; the render thread rebuilds its film to
    /// match on the next sample. Bumps the epoch — even for a size the render
    /// already has, so a consumer waiting on the verb's frame never wedges.
    pub fn resize(&self, width: u32, height: u32) {
        {
            let mut inputs = self.lanes.inputs.lock().expect("inputs mutex poisoned");
            inputs.size = (width, height);
            inputs.stamped = Instant::now();
        }
        self.lanes.epoch.fetch_add(1, Ordering::Release);
    }

    /// Trade resolution for cadence while the picture is changing: a render
    /// that restarted within the last `CHANGE_HOLD` — a camera move, a scene
    /// edit — draws a smaller rectangle aimed at `target`, which a consumer
    /// scales up to its window, and returns to full resolution once the
    /// changes stop. So it shortens the wait for the first frame after a
    /// single edit as well as the frame time through a drag.
    ///
    /// A resize arms the hold like the others, but its first sample is always
    /// full size and a window being dragged larger gets no reduction at all:
    /// the divisor divides a measured cost, a new rectangle invalidates that
    /// cost, and every wave of a drag-resize brings another new rectangle.
    ///
    /// Off by default, because a reduced frame is a *smaller* frame — its
    /// size is [`Frame::width`] and [`Frame::height`], and a consumer that
    /// assumes every frame fills its window would drop or misread it. Not a
    /// picture-changing verb: no epoch bump, no restart, nothing to wait on.
    pub fn set_preview_target(&self, target: Option<Duration>) {
        self.lanes
            .inputs
            .lock()
            .expect("inputs mutex poisoned")
            .preview_target = target;
    }

    /// Queue a change-set to overlay onto the scene — the lookdev shape.
    /// Edits merge in arrival order and land at the next wave boundary:
    /// stop, apply, re-prep what the edit dirtied, restart accumulation.
    /// A rejected set leaves the scene untouched and surfaces through
    /// [`Session::take_edit_error`]. Bumps the epoch after queueing, so the
    /// frame that incorporates this edit — applied or rejected — is
    /// identifiable ([`Frame::epoch`]).
    pub fn apply(&self, set: ChangeSet) {
        self.lanes
            .edits
            .lock()
            .expect("edits mutex poisoned")
            .push(SceneEdit::Apply {
                at: Instant::now(),
                set,
            });
        self.lanes.epoch.fetch_add(1, Ordering::Release);
    }

    /// Queue a whole-scene replacement — the file-reload shape: `set`
    /// describes the entire scene from empty, and objects it no longer
    /// contains are removed, retiring their GPU residency. Unchanged
    /// objects re-prep nothing, so re-saving an untouched file is free.
    /// Rejections behave as in [`Session::apply`], and the epoch bumps the
    /// same way.
    pub fn replace(&self, set: ChangeSet) {
        self.lanes
            .edits
            .lock()
            .expect("edits mutex poisoned")
            .push(SceneEdit::Replace {
                at: Instant::now(),
                set,
            });
        self.lanes.epoch.fetch_add(1, Ordering::Release);
    }

    /// Take the latest rejected edit, if one hasn't been taken yet. The
    /// render thread keeps rendering its previous scene through a
    /// rejection — this is how a consumer learns the edit didn't land.
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
    #[must_use]
    pub fn take_frame(&self) -> Option<Frame> {
        self.lanes
            .published
            .lock()
            .expect("published mutex poisoned")
            .take()
    }

    /// The session epoch: how many picture-changing verbs —
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
    reason = "one wave's decision sequence, in the order the decisions must happen: \
              each step is guarded by state the next one reads, and carving out a \
              middle would hand the piece half a dozen mutable bindings and hide \
              the ordering that is the whole content of the loop"
)]
fn render_loop(
    gpu: &Context,
    mut description: SceneDescription,
    mut scene: Scene,
    mut renderer: Renderer,
    lanes: &Lanes,
    load: Phases,
) -> Result<()> {
    log::debug!("render thread started");
    // The render target: the film and its pair of publish buffers, sized
    // together and rebuilt together when the requested size changes.
    // `applied_generation` tracks which view is in the scene, so a bump
    // restarts accumulation.
    let mut target: Option<(Film, [Arc<FrameBuffers>; 2])> = None;
    let mut applied_size = (0, 0);
    let mut applied_generation = 0;
    // The stopping rule currently in force, as the last wave resolved it
    // from the description. Held rather than re-read per use so a change
    // is a comparison: the budget and the threshold both wake a parked
    // render, and the threshold is also pushed into the renderer.
    let mut applied_controls = Controls::default();
    let mut last_publish: Option<Instant> = None;
    // The whole of the preview state: the picture is still changing if this
    // is inside `CHANGE_HOLD`. Stamped by the one re-arm below, so every
    // rewind arms it — a camera move, an edit, a resize alike.
    let mut last_change: Option<Instant> = None;
    // What a sample costs at full resolution — the reference the preview
    // divisor is measured against.
    let mut full_sample: Option<Duration> = None;
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
    // The load happened before this thread existed, so the breakdown for it
    // is handed in rather than measured here; it closes against
    // time-to-first-ray when the first sample lands.
    recorder.attribute_load(load);
    let mut timer = gpu.create_pass_timer(PassTimer::WAVE_CAPACITY)?;
    let mut last_memory_sample: Option<Instant> = None;
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
        // undercount a racing verb, never claim one it missed.
        let epoch = lanes.epoch.load(Ordering::Acquire);
        let input = *lanes.inputs.lock().expect("inputs mutex poisoned");
        // The restart this wave is about to make, if any: filled by whichever
        // branch below rewinds the film, and consumed by the one place that
        // re-arms the interactivity marks.
        let mut restart: Option<Restart> = None;
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
            // A sample's cost is a cost per this many pixels.
            full_sample = None;
            restart = Some(Restart::waited(input.stamped));
        }
        let (film, frames) = target.as_mut().expect("sized by the resize branch above");
        // Queued edits land here, at the wave boundary: stop, apply,
        // re-prep, restart accumulation from sample 0.
        if let Some(edits) = apply_edits(gpu, lanes, &mut description, &mut scene, &mut stale)? {
            film.reset();
            last_publish = None;
            // An edit that arrived in the same wave as a resize supersedes
            // it: the resize did no prep, so the edit's breakdown is the one
            // with something to say about where the time went.
            restart = Some(edits);
        }
        // A plain view change resets the existing film instead.
        if input.generation != applied_generation {
            log::debug!("camera adopted; accumulation restarts");
            *scene.camera_mut() = input.camera;
            film.reset();
            last_publish = None;
            applied_generation = input.generation;
            restart.get_or_insert_with(|| Restart::waited(input.stamped));
        }
        // The verb every branch above left behind is what re-arms the
        // interactivity marks — one place, not a call site per reset. Not
        // the film's sample count: two restarts in *consecutive* waves both
        // leave it at zero, so the second would be dropped and the mark
        // would report the first move for the rest of a drag.
        if let Some(Restart { origin, phases }) = restart {
            recorder.restart(origin, phases);
            // And the same rewind arms the preview — each one is a wait for
            // a new first frame and the rectangle below shortens them alike,
            // so there is nothing to sort by kind here. Stamped now rather
            // than at the verb, unlike the mark: the mark measures what a
            // person waited for, while the hold asks whether the picture is
            // still changing, which starts when the film is actually
            // rewound. Stamping the verb instead measured inert — a scene
            // heavy enough to want a divisor above 1 preps for longer than
            // the hold, so the preview was never drawn on exactly the scenes
            // that need one.
            last_change = Some(Instant::now());
        }
        // How long to render, and how far — resolved from the description
        // at every boundary, because a settings edit lands in it like any
        // other value and there is no second channel to disagree with.
        let controls = controls_of(&description);
        if controls != applied_controls {
            // Depth is the one setting that changes the estimator rather
            // than the finish line, so samples traced under the old cap
            // cannot stay in the average. `apply_edits` already rewound the
            // film for the ordinary case, and raised the restart with the
            // edit's own origin; this rewinds it again for the one case that
            // misses — an edit whose re-prep was rejected leaves the
            // description ahead of the residency, and the estimator has to
            // follow the cap the renderer is actually tracing at.
            if controls.max_bounces != renderer.max_bounces() {
                renderer.set_max_bounces(controls.max_bounces);
                film.reset();
                last_publish = None;
            }
            // The threshold moves what the accumulate kernel counts as
            // converged, not the beauty, so it is adopted into the running
            // average — the per-sample count self-heals on the next sample.
            // `None` restores the renderer default, which `auto_stopped`
            // then never consults.
            renderer.set_noise_threshold(
                controls
                    .noise_threshold
                    .unwrap_or(Renderer::NOISE_THRESHOLD),
            );
            // And any of the three wakes a settled render. The budget and
            // the threshold cause no reset of their own, and `parked` is
            // otherwise cleared only by one — so without this a raised
            // budget would sit unread until something else happened to
            // rewind the picture.
            parked = false;
            applied_controls = controls;
        }
        // Resolution follows the picture: smaller while it is still
        // changing, full size once the hold expires. Deliberately below the
        // re-arm above, being the one rewind with no verb behind it — arming
        // the interactivity marks here would report a latency nobody waited
        // for and overwrite the mark of the change still being answered.
        let unsettled = last_change.is_some_and(|at| at.elapsed() < CHANGE_HOLD);
        let rectangle = match input.preview_target {
            Some(target) if unsettled => divided(input.size, preview_divisor(full_sample, target)),
            _ => input.size,
        };
        if (film.width(), film.height()) != rectangle {
            log::debug!("rendering at {}×{}", rectangle.0, rectangle.1);
            film.rescale(rectangle.0, rectangle.1);
            last_publish = None;
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
                && publish(gpu, &renderer, film, frames, lanes, &recorder, epoch, true)?
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
        let complete = film.samples() >= applied_controls.max_samples
            || auto_stopped(gpu, film, applied_controls.noise_threshold)?;
        if complete {
            // Force the settled image out once (past the throttle) so the
            // converged frame is definitely the latest, then park. If both slots
            // are busy the publish is retried on the next tick.
            if publish(gpu, &renderer, film, frames, lanes, &recorder, epoch, true)? {
                parked = true;
                published_epoch = epoch;
            }
            std::thread::sleep(IDLE_NAP);
            continue;
        }

        // Every frame is bracketed; one in `PassTimer::BREAKDOWN_INTERVAL`
        // is also resolved kernel by kernel.
        let started = Instant::now();
        let passes = renderer.accumulate_timed(gpu, &scene, film, timer.as_mut())?;
        let cpu = started.elapsed();
        // Read once for the two things that care whether this sample is the
        // window's own resolution: the readable mark, which a reduced sample
        // must not close, and the divisor's reference below.
        let full_size = (film.width(), film.height()) == input.size;
        // `stats::Frame` is the measurement of a sample; the `Frame` this
        // module publishes is the pixels. Spelled out so the two never read
        // as one.
        recorder.record(stats::Frame {
            cpu,
            passes,
            size: (film.width(), film.height()),
            samples: film.samples(),
            preview: !full_size,
        });
        // Only full-resolution samples set the reference, so the divisor
        // divides the still image measured before the drag and cannot chase
        // its own output.
        if full_size {
            full_sample = Some(cpu);
        }
        if last_memory_sample.is_none_or(|at| at.elapsed() >= MEMORY_SAMPLE_INTERVAL) {
            recorder.memory(gpu.memory());
            last_memory_sample = Some(Instant::now());
        }

        // Publish on the throttle — which widens as the image converges (see
        // `publish_interval`) — but only into a buffer no consumer still holds.
        // If both are busy, skip: the next tick catches up, and the renderer
        // never waits on the consumer.
        if last_publish.is_none_or(|at| at.elapsed() >= publish_interval(film.samples()))
            && publish(gpu, &renderer, film, frames, lanes, &recorder, epoch, false)?
        {
            last_publish = Some(Instant::now());
            published_epoch = epoch;
        }
    }
}

/// The smallest divisor that brings a sample costing `full_sample` inside
/// `target`, capped at [`PREVIEW_DIVISOR_MAX`]. Divides by 1 when the scene
/// is already fast enough, or when nothing has been measured yet.
///
/// Powers of two only: cost is near-linear in pixel count, so the
/// arithmetic would support any real number, but a divisor drifting with
/// every measurement changes the softness of the image mid-drag and reads
/// as the render malfunctioning.
fn preview_divisor(full_sample: Option<Duration>, target: Duration) -> u32 {
    let Some(full_sample) = full_sample else {
        return 1;
    };
    let mut divisor = 1;
    // By area: dividing each axis by `d` leaves 1/d² of the pixels.
    // Multiplying the target rather than dividing the sample keeps this in
    // integers and cannot round a duration to zero.
    while divisor < PREVIEW_DIVISOR_MAX && full_sample > target * (divisor * divisor) {
        divisor *= 2;
    }
    divisor
}

/// A render size divided by `divisor`, never below a single pixel — a
/// window can legitimately be a few pixels across, and a zero-sized film is
/// a panic.
fn divided((width, height): (u32, u32), divisor: u32) -> (u32, u32) {
    ((width / divisor).max(1), (height / divisor).max(1))
}

/// Resolve the film's current average into a free publish slot and post it for
/// the consumer — stamped with `epoch`, the wave boundary this accumulation
/// last crossed, and with `converged` saying whether the render is done —
/// returning whether a slot was free. Both slots busy
/// means the consumer still holds the last two frames — the caller retries
/// next tick and the renderer never blocks.
#[expect(
    clippy::too_many_arguments,
    reason = "the resolve's inputs and the two stamps the frame carries; a struct \
              would only rename them"
)]
fn publish(
    gpu: &Context,
    renderer: &Renderer,
    film: &Film,
    frames: &[Arc<FrameBuffers>; 2],
    lanes: &Lanes,
    recorder: &Recorder,
    epoch: u64,
    converged: bool,
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
    let frame = Frame {
        buffers: Arc::clone(free),
        width: film.width(),
        height: film.height(),
        samples: film.samples(),
        converged,
        stats: recorder.stats(),
        epoch,
    };
    *lanes.published.lock().expect("published mutex poisoned") = Some(frame);
    Ok(true)
}

/// The three settings the render loop itself acts on, resolved from the
/// description at a wave boundary. The rest of
/// [`Settings`](crate::scene::description::Settings) belongs to other
/// stages — resolution follows the render target, the global medium is
/// residency — and never reaches here.
#[derive(Clone, Copy, PartialEq)]
struct Controls {
    /// The sample budget: accumulation parks here whatever the noise says.
    max_samples: u32,
    /// The convergence early-out, or `None` to spend the whole budget.
    noise_threshold: Option<f32>,
    /// The path-length cap. Unlike the other two this changes the
    /// estimator rather than the finish line, so a change to it restarts
    /// accumulation — see [`apply_edits`].
    max_bounces: u32,
}

impl Default for Controls {
    /// What an unauthored scene renders under, so the session's starting
    /// assumption is the same one every other consumer of the format makes.
    fn default() -> Self {
        Self::from(&Settings::default())
    }
}

impl From<&Settings> for Controls {
    fn from(settings: &Settings) -> Self {
        Self {
            max_samples: settings.spp,
            noise_threshold: settings.noise_threshold,
            max_bounces: settings.max_bounces,
        }
    }
}

/// The description's render controls — from the one settings object prep
/// guarantees. A description without one falls back to the defaults rather
/// than panicking: a missing singleton is prep's error to raise, and the
/// render thread has a previous scene to keep drawing.
fn controls_of(description: &SceneDescription) -> Controls {
    description
        .settings()
        .values()
        .next()
        .map_or_else(Controls::default, Controls::from)
}

/// Whether an auto-stop threshold is set and the film has reached it — the
/// convergence-idle stop. Below [`Renderer::CONVERGENCE_MIN_SAMPLES`] the metric
/// is untrusted, so it never stops there; above it, it reads the film's
/// converged fraction (a 4-byte device readback) against
/// [`Renderer::CONVERGENCE_TARGET`]. `None` never stops — the cap alone bounds it.
fn auto_stopped(gpu: &Context, film: &Film, threshold: Option<f32>) -> Result<bool> {
    if threshold.is_none() || film.samples() < Renderer::CONVERGENCE_MIN_SAMPLES {
        return Ok(false);
    }
    Ok(film.converged_fraction(gpu)? >= Renderer::CONVERGENCE_TARGET)
}

/// Drain and apply the queued edits, re-prepping what they dirtied. A
/// [`Restart`] means the visible scene changed and accumulation must restart,
/// and carries where the drain's time went; `None` means nothing visible
/// happened and the accumulation stands. A rejected change-set or re-prep
/// posts to the edit-error lane and keeps the previous scene; the dirt it
/// left in `stale` retries after the next edit that applies. Only device
/// faults return `Err`.
///
/// The batch's origin is its *oldest* edit — the longest anyone in it
/// waited. A burst that lands on one boundary is one restart and one
/// measurement, with `batched` saying how many verbs it covered, because
/// that is what actually happened: they cost one re-prep between them.
fn apply_edits(
    gpu: &Context,
    lanes: &Lanes,
    description: &mut SceneDescription,
    scene: &mut Scene,
    stale: &mut Dirty,
) -> Result<Option<Restart>> {
    let edits = std::mem::take(&mut *lanes.edits.lock().expect("edits mutex poisoned"));
    let Some(origin) = edits.iter().map(SceneEdit::at).min() else {
        return Ok(None);
    };
    // Read before the drain, compared after: the one settings field whose
    // change invalidates the accumulated image.
    let depth_before = controls_of(description).max_bounces;
    let batched = u32::try_from(edits.len()).unwrap_or(u32::MAX);
    let before = origin.elapsed();
    let applying = Instant::now();
    let mut applied = false;
    for edit in edits {
        let result = match edit {
            SceneEdit::Apply { set, .. } => description.apply(&set),
            SceneEdit::Replace { set, .. } => {
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
    let apply = applying.elapsed();
    if !applied || stale.is_empty() {
        return Ok(None);
    }
    // Settings carry no residency, so a settings-only edit must not throw
    // away the accumulated image — with one exception. The path-length cap
    // is part of the estimator: samples traced under two different caps are
    // estimates of two different integrals, and averaging them together
    // would converge to neither. The budget and the noise threshold only
    // move the finish line, and are adopted mid-render.
    let visual = stale
        .changed
        .iter()
        .chain(&stale.removed)
        .any(|(kind, _)| *kind != Kind::Settings)
        || controls_of(description).max_bounces != depth_before;
    match scene.update(gpu, description, stale) {
        Ok(phases) => {
            log::debug!("scene edits applied; accumulation restarts");
            *stale = Dirty::default();
            Ok(visual.then_some(Restart {
                origin,
                phases: Phases {
                    before,
                    apply,
                    batched,
                    ..phases
                },
            }))
        }
        // This build can't render the edited description; the previous
        // residency keeps rendering and `stale` holds the backlog.
        Err(error @ Error::Scene(_)) => {
            post_edit_error(lanes, error);
            Ok(None)
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
        }))
    };
    Ok([slot()?, slot()?])
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use glam::Vec3;

    use super::*;
    use crate::scene::changeset::{MaterialPatch, Op, SettingsPatch};
    use crate::scene::description::Texturable;

    /// A demo session that never parks — the sample budget is effectively
    /// unbounded, so the accumulation-and-edit tests see samples climb freely.
    fn demo_session(gpu: &Arc<Context>, size: u32) -> Session {
        demo_session_with(gpu, size, u32::MAX)
    }

    /// A demo session bounded by `max_samples`: the description, its prepped
    /// scene, and the thread already accumulating. The budget is authored
    /// into the scene, which is the session's only channel for it — the demo
    /// names its settings object `main`.
    fn demo_session_with(gpu: &Arc<Context>, size: u32, max_samples: u32) -> Session {
        demo_session_with_settings(gpu, size, max_samples, None)
    }

    /// [`demo_session_with`] with a convergence threshold as well — the early
    /// stop, which no sample count implies.
    fn demo_session_with_settings(
        gpu: &Arc<Context>,
        size: u32,
        max_samples: u32,
        noise_threshold: Option<f32>,
    ) -> Session {
        let mut description = SceneDescription::new();
        description.apply(&ChangeSet::demo()).expect("demo applies");
        description
            .apply(&ChangeSet {
                ops: vec![Op::Settings(SettingsPatch {
                    spp: Some(max_samples),
                    noise_threshold: Some(noise_threshold),
                    ..SettingsPatch::new("main")
                })],
            })
            .expect("a budget applies");
        let (scene, load) = Scene::prep_timed(gpu, &mut description).expect("demo preps");
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
            load,
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

    /// An edit's latency arrives broken down, and the breakdown accounts for
    /// the mark exactly. This is the end of the wire the phases exist for:
    /// the origin is the `apply` call rather than the wave boundary that
    /// drained it, so the wait for that boundary is inside the number a
    /// person feels instead of in front of it.
    ///
    /// The `tables` assertion is the standing claim about the update path,
    /// pinned here so it stays a measurement: a material edit touches no
    /// instance, and the instance tables rebuild for it anyway.
    #[test]
    fn an_edit_reports_where_its_latency_went() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let gpu = Arc::new(gpu);
        let session = demo_session(&gpu, 64);
        wait_for_frame(&session);

        session.apply(ChangeSet {
            ops: vec![Op::Material(Box::new(MaterialPatch {
                base_color: Some(Texturable::Constant([0.9, 0.1, 0.1])),
                ..MaterialPatch::new("floor")
            }))],
        });
        let frame = wait_for(&session, "a frame carrying the edit's breakdown", |frame| {
            frame.stats().interactivity.interaction.is_some()
        });

        let interactivity = frame.stats().interactivity;
        let phases = interactivity.interaction.expect("waited for it");
        assert_eq!(
            Some(phases.total()),
            interactivity.to_first_pixel,
            "the phases are the mark, exactly: {:?}",
            phases.named()
        );
        assert_eq!(phases.batched, 1, "one edit drained");
        assert!(phases.apply > Duration::ZERO, "the description was patched");
        assert!(
            phases.tables > Duration::ZERO,
            "a material edit rebuilds the instance tables it did not touch"
        );
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

    /// The sample cap parks the render thread: with a low
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
        let mut session = demo_session_with(&gpu, 32, CAP);

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

    /// A frame carries the session's own verdict on whether the render is
    /// done, and the noise threshold is why it has to. With an unbounded
    /// budget the early stop is the *only* thing that can finish this render,
    /// so a consumer re-deriving the flag from the sample count — comparing
    /// it against a budget the render already decided not to reach — would
    /// wait forever.
    #[test]
    fn a_frame_reports_the_early_stop_the_sample_count_cannot() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let gpu = Arc::new(gpu);
        // A threshold of 1 — the loosest the schema allows — so the stop
        // fires the moment the metric is trusted rather than depending on how
        // fast the demo scene actually converges.
        let session = demo_session_with_settings(&gpu, 32, u32::MAX, Some(1.0));

        // Before the metric is trusted nothing is converged, whatever the
        // threshold says.
        let early = wait_for(&session, "an unconverged frame", |frame| {
            frame.samples() < Renderer::CONVERGENCE_MIN_SAMPLES
        });
        assert!(!early.converged(), "converged before the metric is trusted");

        let settled = wait_for(&session, "the early stop", Frame::converged);
        assert!(
            settled.samples() >= Renderer::CONVERGENCE_MIN_SAMPLES,
            "stopped on an untrusted metric at {} samples",
            settled.samples()
        );
        assert!(
            settled.samples() < u32::MAX,
            "the budget cannot be what stopped this render"
        );
    }

    /// The two halves of the per-field rule, in one settled render. Raising
    /// the budget must wake the parked thread — nothing else rewinds the film
    /// for it, so without the unpark the new budget would sit unread — and it
    /// must do so *without* discarding the samples already accumulated, since
    /// a finish line is not part of the estimator. Deepening the path is the
    /// other half: that one restarts, because samples traced under two caps
    /// estimate two different integrals.
    #[test]
    fn the_budget_resumes_the_render_and_the_depth_restarts_it() {
        const CAP: u32 = 8;
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let gpu = Arc::new(gpu);
        let session = demo_session_with(&gpu, 32, CAP);
        wait_for(&session, "the settled frame", |frame| {
            frame.samples() == CAP && frame.converged()
        });

        // A raised budget: accumulation continues from where it parked.
        session.apply(ChangeSet {
            ops: vec![Op::Settings(SettingsPatch {
                spp: Some(CAP * 2),
                ..SettingsPatch::new("main")
            })],
        });
        wait_for(&session, "the resumed render", |frame| {
            frame.samples() > CAP
        });
        wait_for(&session, "the higher budget settling", |frame| {
            frame.samples() == CAP * 2 && frame.converged()
        });

        // A new depth: the film rewinds, so the count must be seen below the
        // budget again before it climbs back.
        session.apply(ChangeSet {
            ops: vec![Op::Settings(SettingsPatch {
                max_bounces: Some(3),
                ..SettingsPatch::new("main")
            })],
        });
        wait_for(&session, "the restarted accumulation", |frame| {
            frame.samples() < CAP * 2
        });
        assert!(session.take_edit_error().is_none());
    }

    /// The picture-changing verbs each bump the session epoch once, and the
    /// viewer toggles don't: a toggle restarts accumulation anyway, so its
    /// fresh sample count already says "new picture" — only the verbs a
    /// remote consumer acknowledges need the stamp.
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
    }

    /// An applied edit's stamp reaches the published frame: accumulation
    /// crosses a wave boundary that drained the edit, and every frame it
    /// publishes from there carries the moved epoch.
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
    /// consumer waiting on that epoch would wait forever.
    #[test]
    fn a_noop_edit_republishes_the_parked_frame() {
        const CAP: u32 = 8;
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let gpu = Arc::new(gpu);
        let session = demo_session_with(&gpu, 32, CAP);
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
    /// fresh stamp and the fault rides the error lane as usual.
    #[test]
    fn a_rejected_edit_still_advances_the_frame_epoch() {
        const CAP: u32 = 8;
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let gpu = Arc::new(gpu);
        let session = demo_session_with(&gpu, 32, CAP);
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

    /// Walk a session through the three preview states, `nudge` supplying
    /// whichever verb the calling test is about: no target (a change reduces
    /// nothing — the default `cenote-server` relies on), target and
    /// unsettled (the smaller rectangle), and changes stopped (full size
    /// once the hold expires). One body for both verbs because there is one
    /// mechanism: what the hold is armed by is the only difference between
    /// them.
    ///
    /// Sizes rather than frames, all the way through: the publish pool is
    /// two deep and the render thread cannot resolve into a slot a consumer
    /// still holds, so a test that keeps two frames alive stops the renderer
    /// it is waiting on.
    ///
    /// The target is a nanosecond rather than [`Session::PREVIEW_TARGET`]
    /// because a 64×64 demo scene samples so far inside 16 ms that its
    /// honest divisor is 1; a target nothing can meet pins the reduction at
    /// [`PREVIEW_DIVISOR_MAX`] on any machine. The verbs come in a loop
    /// because the hold is wall-clock: one of them on a slow machine could
    /// expire before a frame answering it is published. Waiting out a
    /// full-resolution sample first is what gives the divisor a cost to
    /// divide — until then it is 1, which is why startup needs no special
    /// case.
    fn preview_follows(gpu: Context, mut nudge: impl FnMut(&Session)) {
        const SIZE: u32 = 64;
        let session = demo_session(&Arc::new(gpu), SIZE);
        let size = |frame: &Frame| (frame.width(), frame.height());
        assert_eq!(size(&wait_for_frame(&session)), (SIZE, SIZE));

        for _ in 0..8 {
            nudge(&session);
            if let Some(frame) = session.take_frame() {
                assert_eq!(size(&frame), (SIZE, SIZE), "reduced without being asked");
            }
            std::thread::sleep(Duration::from_millis(2));
        }

        session.set_preview_target(Some(Duration::from_nanos(1)));
        let deadline = Instant::now() + Duration::from_secs(10);
        let reduced = loop {
            nudge(&session);
            if let Some(frame) = session.take_frame()
                && size(&frame) != (SIZE, SIZE)
            {
                break size(&frame);
            }
            assert!(
                Instant::now() < deadline,
                "the change never dropped resolution"
            );
            std::thread::sleep(Duration::from_millis(2));
        };
        assert_eq!(
            reduced,
            (SIZE / PREVIEW_DIVISOR_MAX, SIZE / PREVIEW_DIVISOR_MAX)
        );

        let full = wait_for(&session, "full size once the changes stop", |frame| {
            size(frame) == (SIZE, SIZE)
        });
        assert!(full.samples() > 0);
    }

    /// The camera half: a view being dragged renders reduced and recovers.
    #[test]
    fn a_moving_view_renders_reduced_and_recovers() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let mut view = Camera {
            position: Vec3::new(0.0, 1.0, 4.0),
            look_at: Vec3::ZERO,
            up: Vec3::Y,
            vfov_degrees: 45.0,
            lens: None,
        };
        preview_follows(gpu, |session| {
            view.position.x += 0.01;
            session.set_camera(view);
        });
    }

    /// The edit half: the camera never moves and the same reduction follows
    /// a *scene edit*, the hold being armed by the rewind rather than by the
    /// verb behind it.
    ///
    /// The patch alternates its colour because an edit that changes nothing
    /// visible restarts nothing, and every pass of the loop needs a restart.
    ///
    /// The rejection check leads rather than follows: a rejected patch
    /// restarts nothing, so it would surface as the helper's "never dropped
    /// resolution" deadline and name the wrong culprit. Checked before each
    /// patch, the previous one's rejection fails the test where it happened.
    #[test]
    fn an_edit_renders_reduced_and_recovers() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let mut hot = false;
        preview_follows(gpu, |session| {
            if let Some(error) = session.take_edit_error() {
                panic!("the patch was rejected: {error}");
            }
            hot = !hot;
            let colour = if hot { [0.9, 0.1, 0.1] } else { [0.1, 0.1, 0.9] };
            session.apply(ChangeSet {
                ops: vec![Op::Material(Box::new(MaterialPatch {
                    base_color: Some(Texturable::Constant(colour)),
                    ..MaterialPatch::new("floor")
                }))],
            });
        });
    }

    /// A window can legitimately be a few pixels across, and a zero-sized
    /// film is a panic.
    #[test]
    fn a_reduced_rectangle_keeps_at_least_one_pixel() {
        assert_eq!(divided((1280, 720), 2), (640, 360));
        assert_eq!(divided((3, 3), 2), (1, 1));
        assert_eq!(divided((1, 1), 2), (1, 1));
    }

    /// How many halvings the sample needs to fit the target, by area: each
    /// one buys a factor of four.
    #[test]
    fn the_preview_divisor_is_the_halving_that_fits_the_target() {
        let target = Duration::from_millis(16);
        let sample = |ms| Some(Duration::from_millis(ms));
        // Inside the target, and exactly on it: no reduction.
        assert_eq!(preview_divisor(sample(8), target), 1);
        assert_eq!(preview_divisor(sample(16), target), 1);
        // One halving covers up to 4× the target — sanmiguel's measured
        // 29 ms sits here — and each boundary belongs to the cheaper
        // divisor.
        assert_eq!(preview_divisor(sample(17), target), 2);
        assert_eq!(preview_divisor(sample(29), target), 2);
        assert_eq!(preview_divisor(sample(64), target), 2);
        assert_eq!(preview_divisor(sample(65), target), 4);
        // The cap holds however heavy the scene.
        assert_eq!(preview_divisor(sample(1_000), target), 4);
        // Nothing measured yet: full resolution rather than a guess.
        assert_eq!(preview_divisor(None, target), 1);
    }

    /// The publish gap widens with the sample count and clamps at both ends
    ///: every frame early, a few per second once converging.
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
