//! The render loop as an actor: a dedicated thread that accumulates as fast
//! as the GPU allows, so a consumer's display refresh never paces the
//! renderer. The viewer is the first consumer; the M2 Hydra delegate will be
//! a second — the concurrency lives here, once, not in each of them. This is
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

use std::sync::{Arc, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use ash::vk;

use super::{Film, Renderer, ResolveTargets};
use crate::error::{Error, Result};
use crate::gpu::{Buffer, Context, MemoryLocation};
use crate::scene::changeset::{ChangeSet, Dirty, Kind};
use crate::scene::description::SceneDescription;
use crate::scene::{Camera, Scene};

/// The shortest gap between published frames. The render thread accumulates
/// flat out but resolves and publishes at most this often — resolving every
/// sample would burn GPU time a consumer can't display faster than its
/// refresh anyway. Set just under a 60 Hz frame so a vsync'd viewer always
/// finds a fresh frame waiting.
const PUBLISH_INTERVAL: Duration = Duration::from_millis(15);

/// How long the render thread sleeps when there is nothing to draw (a
/// minimized, zero-area window) before re-reading its inputs — long enough
/// not to spin, short enough to wake promptly when the window returns.
const IDLE_NAP: Duration = Duration::from_millis(16);

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
    /// Wall-clock of the sample that preceded this publish — the render
    /// thread's own timing, for the viewer's stats panel.
    sample_time: Duration,
}

impl Frame {
    /// The linear `ACEScg` average, ready for a [`super::Tonemap`] to read.
    #[must_use]
    pub fn image(&self) -> &Buffer {
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

    /// Wall-clock of the sample before this publish — the render thread's own
    /// measurement, for the stats panel.
    #[must_use]
    pub fn sample_time(&self) -> Duration {
        self.sample_time
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
    /// marks the first frame ready.
    ///
    /// # Panics
    ///
    /// If the OS refuses to spawn the render thread — an environment failure
    /// at startup, not something a caller can recover from here.
    #[must_use]
    pub fn new(
        gpu: Arc<Context>,
        description: SceneDescription,
        scene: Scene,
        renderer: Renderer,
        camera: Camera,
        width: u32,
        height: u32,
    ) -> Self {
        let lanes = Arc::new(Lanes {
            inputs: Mutex::new(RenderInputs {
                camera,
                size: (width, height),
                generation: 0,
                running: true,
            }),
            edits: Mutex::new(Vec::new()),
            edit_error: Mutex::new(None),
            published: Mutex::new(None),
        });
        let thread = {
            let lanes = Arc::clone(&lanes);
            std::thread::Builder::new()
                .name("cenote-render".into())
                .spawn(move || render_loop(&gpu, description, scene, &renderer, &lanes))
                .expect("spawning the render thread")
        };
        Self {
            lanes,
            thread: Some(thread),
        }
    }

    /// Point the render at a new view — the viewer's orbit control calls this
    /// each time the camera moves. Bumps the generation so the render thread
    /// restarts accumulation from the new pose.
    ///
    /// # Panics
    ///
    /// If the render thread panicked while holding the input lock — a bug on
    /// that thread, surfaced here rather than silently ignored.
    pub fn set_camera(&self, camera: Camera) {
        let mut inputs = self.lanes.inputs.lock().expect("inputs mutex poisoned");
        inputs.camera = camera;
        inputs.generation += 1;
    }

    /// Note a new render-target size; the render thread rebuilds its film to
    /// match on the next sample.
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
    }

    /// Queue a change-set to overlay onto the scene — the lookdev shape.
    /// Edits merge in arrival order and land at the next wave boundary:
    /// stop, apply, re-prep what the edit dirtied, restart accumulation.
    /// A rejected set leaves the scene untouched and surfaces through
    /// [`Session::take_edit_error`].
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
    }

    /// Queue a whole-scene replacement — the file-reload shape: `set`
    /// describes the entire scene from empty, and objects it no longer
    /// contains are removed, retiring their GPU residency. Unchanged
    /// objects re-prep nothing, so re-saving an untouched file is free.
    /// Rejections behave as in [`Session::apply`].
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
fn render_loop(
    gpu: &Context,
    mut description: SceneDescription,
    mut scene: Scene,
    renderer: &Renderer,
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
    let mut last_publish: Option<Instant> = None;
    // Dirt whose re-prep was rejected (this build can't render the edited
    // description). It survives here so the *next* applied edit retries the
    // whole backlog — nothing goes silently stale.
    let mut stale = Dirty::default();

    loop {
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

        let started = Instant::now();
        renderer.accumulate(gpu, &scene, film)?;
        let sample_time = started.elapsed();

        // Publish on the throttle, but only into a buffer no consumer still
        // holds. If both are busy, skip: the next tick catches up, and the
        // renderer never waits on the consumer.
        if last_publish.is_none_or(|at| at.elapsed() >= PUBLISH_INTERVAL)
            && let Some(free) = frames.iter().find(|frame| Arc::strong_count(frame) == 1)
        {
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
                width,
                height,
                samples: film.samples(),
                sample_time,
            };
            *lanes.published.lock().expect("published mutex poisoned") = Some(frame);
            last_publish = Some(Instant::now());
        }
    }
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
        }))
    };
    Ok([slot()?, slot()?])
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use crate::scene::changeset::{MaterialPatch, Op};
    use crate::scene::description::Texturable;

    /// A demo session: the description, its prepped scene, and the thread
    /// already accumulating.
    fn demo_session(gpu: &Arc<Context>, size: u32) -> Session {
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

    /// Poll `take_frame` until one appears, with a generous timeout so a slow
    /// machine doesn't flake — the render thread posts its first frame within
    /// milliseconds on any real GPU.
    fn wait_for_frame(session: &Session) -> Frame {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(frame) = session.take_frame() {
                return frame;
            }
            assert!(Instant::now() < deadline, "no frame published in time");
            std::thread::sleep(Duration::from_millis(2));
        }
    }
}
