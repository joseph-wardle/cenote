//! The render loop as an actor: a dedicated thread that accumulates as fast
//! as the GPU allows, so a consumer's display refresh never paces the
//! renderer. The viewer is the first consumer; the M2 Hydra delegate will be
//! a second — the concurrency lives here, once, not in each of them. This is
//! the shape Cycles, `MoonRay`, and Karma all use: the path tracer runs on its
//! own thread and the UI *peeks* at its output.
//!
//! Two lanes cross the thread boundary, each its own short-lived lock:
//!
//! - **Inputs in** — [`RenderInputs`] (camera, target size, a `generation`
//!   counter, a running flag) behind a mutex, latest-wins. The viewer writes
//!   the latest camera or size; the render thread snapshots the whole struct
//!   once per sample. Exposure is *not* here: it belongs to the consumer's
//!   view transform, downstream of the published frame.
//! - **Frames out** — the resolved **linear** average, published behind a
//!   second mutex. The render thread resolves into whichever of its two
//!   frame buffers is free and hands over an [`Arc`] to it; the viewer takes
//!   the latest and tonemaps it. The lock spans only the pointer hand-off,
//!   never a GPU submit — the heavy accumulate runs lock-free.
//!
//! Two frame buffers, not a triple-buffered mailbox: the render thread
//! resolves only into a buffer no one else references (a strong-count of one
//! means "in the pool alone"), and if both are busy it simply skips that
//! publish and keeps accumulating. So a slow consumer can never see a buffer
//! torn by an in-flight resolve, and the renderer never blocks on the
//! consumer.
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

use super::{Film, Renderer};
use crate::error::{Error, Result};
use crate::gpu::{Buffer, Context, MemoryLocation};
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

/// A published frame: the estimator's current best image as a **linear**
/// average, plus the metadata a consumer needs to tonemap and present it
/// without reaching back into the renderer. The buffer is shared by [`Arc`]
/// so the render thread can tell — by its strong count — when the consumer
/// has let go and the buffer is free to resolve into again.
pub struct Frame {
    image: Arc<Buffer>,
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
        &self.image
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

/// Owns the render thread and the two lanes across to it. Dropping it stops
/// the thread and joins, so every GPU resource the thread holds is released
/// before the shared [`Context`] is.
pub struct Session {
    inputs: Arc<Mutex<RenderInputs>>,
    published: Arc<Mutex<Option<Frame>>>,
    thread: Option<JoinHandle<Result<()>>>,
}

impl Session {
    /// Spawn the render thread. It takes ownership of `scene`, `renderer`, and
    /// a `Context` handle, and starts accumulating `camera` at
    /// `width`×`height` immediately; the first [`Session::peek`] to return
    /// `Some` marks the first frame ready.
    ///
    /// # Panics
    ///
    /// If the OS refuses to spawn the render thread — an environment failure
    /// at startup, not something a caller can recover from here.
    #[must_use]
    pub fn new(
        gpu: Arc<Context>,
        scene: Scene,
        renderer: Renderer,
        camera: Camera,
        width: u32,
        height: u32,
    ) -> Self {
        let inputs = Arc::new(Mutex::new(RenderInputs {
            camera,
            size: (width, height),
            generation: 0,
            running: true,
        }));
        let published = Arc::new(Mutex::new(None));
        let thread = {
            let inputs = Arc::clone(&inputs);
            let published = Arc::clone(&published);
            std::thread::Builder::new()
                .name("cenote-render".into())
                .spawn(move || render_loop(&gpu, scene, &renderer, &inputs, &published))
                .expect("spawning the render thread")
        };
        Self {
            inputs,
            published,
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
        let mut inputs = self.inputs.lock().expect("inputs mutex poisoned");
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
        self.inputs.lock().expect("inputs mutex poisoned").size = (width, height);
    }

    /// Take the latest published frame, if the render thread has posted a new
    /// one since the last peek. `None` means no fresh frame — the consumer
    /// keeps showing the one it already holds.
    ///
    /// # Panics
    ///
    /// If the render thread panicked while holding the publish lock.
    #[must_use]
    pub fn peek(&self) -> Option<Frame> {
        self.published
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
        self.inputs
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
                .unwrap_or_else(|| "render thread panicked".to_owned());
            Err(Error::RenderThreadPanicked(message))
        }
    }
}

/// The render thread's body: accumulate `scene` into a film sized to the
/// latest inputs, publishing a resolved average on the throttle. Returns when
/// the running flag clears, or early on the first GPU error.
fn render_loop(
    gpu: &Context,
    mut scene: Scene,
    renderer: &Renderer,
    inputs: &Mutex<RenderInputs>,
    published: &Mutex<Option<Frame>>,
) -> Result<()> {
    // The film and its pair of publish buffers, both sized to `applied_size`;
    // rebuilt together when the requested size changes. `applied_generation`
    // tracks which view is in the scene, so a bump restarts accumulation.
    let mut film: Option<Film> = None;
    let mut frames: Vec<Arc<Buffer>> = Vec::new();
    let mut applied_size = (0, 0);
    let mut applied_generation = 0;
    let mut last_publish: Option<Instant> = None;

    loop {
        let input = *inputs.lock().expect("inputs mutex poisoned");
        if !input.running {
            return Ok(());
        }
        let (width, height) = input.size;
        if width == 0 || height == 0 {
            // Minimized: nothing to render until the window comes back.
            std::thread::sleep(IDLE_NAP);
            continue;
        }

        // Match the film and publish buffers to the requested size, and adopt
        // the latest view whenever it changed — a resize restarts by building
        // a fresh (empty) film; a plain view change resets the existing one.
        if input.size != applied_size {
            film = Some(Film::new(gpu, width, height)?);
            frames = new_frames(gpu, width, height)?;
            *scene.camera_mut() = input.camera;
            applied_size = input.size;
            applied_generation = input.generation;
            last_publish = None;
        } else if input.generation != applied_generation {
            *scene.camera_mut() = input.camera;
            film.as_mut().expect("film exists once sized").reset();
            applied_generation = input.generation;
        }
        let film = film.as_mut().expect("film exists once sized");

        let started = Instant::now();
        renderer.accumulate(gpu, &scene, film)?;
        let sample_time = started.elapsed();

        // Publish on the throttle, but only into a buffer no consumer still
        // holds. If both are busy, skip: the next tick catches up, and the
        // renderer never waits on the consumer.
        if last_publish.is_none_or(|at| at.elapsed() >= PUBLISH_INTERVAL)
            && let Some(free) = frames.iter().find(|frame| Arc::strong_count(frame) == 1)
        {
            renderer.resolve(gpu, film, free)?;
            let frame = Frame {
                image: Arc::clone(free),
                width,
                height,
                samples: film.samples(),
                sample_time,
            };
            *published.lock().expect("published mutex poisoned") = Some(frame);
            last_publish = Some(Instant::now());
        }
    }
}

/// The pair of publish buffers, each a full-frame linear RGBA f32 average —
/// what the resolve kernel writes and a consumer's tonemap reads by device
/// address.
fn new_frames(gpu: &Context, width: u32, height: u32) -> Result<Vec<Arc<Buffer>>> {
    let bytes = u64::from(width) * u64::from(height) * 16;
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
    (0..2)
        .map(|_| {
            Ok(Arc::new(gpu.create_buffer(
                "session.frame",
                bytes,
                usage,
                MemoryLocation::GpuOnly,
            )?))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

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
        let scene = Scene::demo(&gpu).expect("demo scene");
        let camera = *scene.camera();
        let renderer = Renderer::new(&gpu).expect("renderer");
        let session = Session::new(Arc::clone(&gpu), scene, renderer, camera, 64, 64);

        // Wait for the first publish, then for a later one — the sample count
        // must climb, proving the thread keeps accumulating, not just resolves
        // one frame.
        let first = wait_for_frame(&session);
        assert_eq!((first.width(), first.height()), (64, 64));
        assert!(first.samples() > 0, "first frame has no samples");
        let later = wait_for_frame(&session);
        assert!(
            later.samples() >= first.samples(),
            "accumulation stalled: {} then {}",
            first.samples(),
            later.samples()
        );
    }

    /// Poll `peek` until a frame appears, with a generous timeout so a slow
    /// machine doesn't flake — the render thread posts its first frame within
    /// milliseconds on any real GPU.
    fn wait_for_frame(session: &Session) -> Frame {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(frame) = session.peek() {
                return frame;
            }
            assert!(Instant::now() < deadline, "no frame published in time");
            std::thread::sleep(Duration::from_millis(2));
        }
    }
}
