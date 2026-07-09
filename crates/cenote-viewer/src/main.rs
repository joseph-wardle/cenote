//! Interactive viewer: the render live in a window, under an orbit camera,
//! progressively accumulated, with an egui stats/controls overlay. Every
//! sample is a full path-traced estimate of the `OpenPBR` lobe mix (EON
//! diffuse, energy-compensated GGX conductor and dielectric specular)
//! under MIS-weighted direct light sampling of the quad lights and the
//! importance-sampled HDRI environment, so the image starts noisy and
//! visibly converges as the spp counter climbs.
//!
//! The render loop runs on its own thread — a [`cenote::render::Session`] —
//! accumulating as fast as the GPU allows, unpaced by the display. The viewer
//! is a thin consumer: it feeds the session camera and size changes, *peeks*
//! at the latest published linear frame, tonemaps it (live exposure), and
//! presents. Each redraw requests the next, so vsync paces the *display*
//! while the renderer runs free behind it. Camera motion and resizes are just
//! inputs the session picks up; it restarts or rebuilds accordingly.

mod camera;
mod ui;

use std::sync::Arc;
use std::time::Instant;

use anyhow::Context as _;
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition};
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::raw_window_handle::{HasDisplayHandle as _, HasWindowHandle as _};
use winit::window::{Window, WindowId};

use crate::camera::OrbitCamera;
use crate::ui::{FrameStats, Gui};

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let event_loop = EventLoop::new()?;
    // Sleep between events — redraws happen on request, not on a timer.
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = App::default();
    event_loop.run_app(&mut app)?;
    app.error.map_or(Ok(()), Err)
}

/// The winit application shell: the [`Viewer`] is absent until `resumed`
/// hands us a window, and a failure anywhere parks its error here for
/// `main` to report after the loop unwinds.
#[derive(Default)]
struct App {
    viewer: Option<Viewer>,
    error: Option<anyhow::Error>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Guard against re-entry: on desktop `resumed` fires once, but the
        // contract allows suspend/resume cycles.
        if self.viewer.is_some() {
            return;
        }
        match Viewer::new(event_loop) {
            Ok(viewer) => self.viewer = Some(viewer),
            Err(err) => {
                self.error = Some(err);
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        if matches!(event, WindowEvent::CloseRequested) {
            // Tear the viewer down now, while the event loop — and with it
            // the display-server connection — is still alive: the exiting
            // `run_app` drops the loop before `main` drops `App`.
            self.viewer = None;
            event_loop.exit();
            return;
        }
        let Some(viewer) = self.viewer.as_mut() else {
            return;
        };
        if let Err(err) = viewer.handle(&event) {
            self.error = Some(err);
            event_loop.exit();
        }
    }
}

/// The live viewer: window, GPU, the render session, UI, and the input state
/// driving the orbit camera.
struct Viewer {
    // Field order is drop order, and the `Session` goes first: joining the
    // render thread stops its queue submits before the `Presenter`'s
    // teardown waits for the device to idle — the two would otherwise race
    // the one queue. Then the surface-owning `Presenter` and the `Gui` (whose
    // clipboard handle talks to the display server) drop before the window,
    // and the shared `Context` drops last, once every buffer that borrows its
    // allocator — including the frame we still hold — has been freed.
    session: cenote::render::Session,
    presenter: cenote::gpu::Presenter,
    /// The view transform: a published linear average → the displayed frame,
    /// at the panel's exposure. The consumer half of the estimator/view
    /// split, owned here and never by the renderer.
    tonemap: cenote::render::Tonemap,
    /// The frame currently on screen, held across redraws so an exposure drag
    /// re-tonemaps it even when the render thread hasn't posted a new one.
    /// Its `Arc` also keeps that publish buffer out of the render thread's
    /// reuse pool while we display it.
    frame: Option<cenote::render::Frame>,
    gpu: Arc<cenote::gpu::Context>,
    gui: Gui,
    window: Window,
    camera: OrbitCamera,
    stats: FrameStats,
    /// Left mouse button held (and not claimed by the UI) — cursor motion
    /// orbits.
    orbiting: bool,
    /// Cursor position at the last `CursorMoved`, for drag deltas.
    cursor: Option<PhysicalPosition<f64>>,
}

impl Viewer {
    fn new(event_loop: &ActiveEventLoop) -> anyhow::Result<Self> {
        let window = event_loop.create_window(
            Window::default_attributes()
                .with_title("cenote")
                .with_inner_size(LogicalSize::new(1280.0, 720.0)),
        )?;
        let display = window.display_handle()?.as_raw();
        let gpu = Arc::new(cenote::gpu::Context::presentable(display)?);
        let scene = cenote::scene::Scene::demo(&gpu)?;
        let camera = OrbitCamera::framing(scene.camera());
        let renderer = cenote::render::Renderer::new(&gpu)?;
        let tonemap = cenote::render::Tonemap::new(&gpu)?;
        let size = window.inner_size();
        let presenter = gpu.create_presenter(
            display,
            window.window_handle()?.as_raw(),
            size.width,
            size.height,
        )?;
        let gui = Gui::new(&window);
        // The session takes the scene and renderer onto its own thread and
        // starts accumulating the initial view at once.
        let session = cenote::render::Session::new(
            Arc::clone(&gpu),
            scene,
            renderer,
            camera.camera(),
            size.width,
            size.height,
        );
        // Not every platform sends an initial redraw request unprompted.
        window.request_redraw();
        Ok(Self {
            session,
            presenter,
            tonemap,
            frame: None,
            gpu,
            gui,
            window,
            camera,
            stats: FrameStats::default(),
            orbiting: false,
            cursor: None,
        })
    }

    fn handle(&mut self, event: &WindowEvent) -> anyhow::Result<()> {
        // egui sees every event first: only it knows whether the pointer is
        // on the UI, and `consumed` keeps such events off the camera. Its
        // repaint requests need no handling — the redraw loop is continuous.
        let response = self.gui.on_window_event(&self.window, event);
        match *event {
            WindowEvent::Resized(size) => {
                self.presenter.resize(size.width, size.height);
                self.session.resize(size.width, size.height);
                self.window.request_redraw();
            }
            WindowEvent::MouseInput {
                button: MouseButton::Left,
                state,
                ..
            } => {
                // A press on the UI belongs to egui; a release always ends
                // the orbit, or a drag let go over the panel never would.
                self.orbiting = state == ElementState::Pressed && !response.consumed;
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.orbiting
                    && !response.consumed
                    && let Some(last) = self.cursor
                {
                    self.camera
                        .orbit((position.x - last.x) as f32, (position.y - last.y) as f32);
                    self.view_changed();
                }
                self.cursor = Some(position);
            }
            WindowEvent::CursorLeft { .. } => self.cursor = None,
            WindowEvent::MouseWheel { delta, .. } if !response.consumed => {
                let notches = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    // Trackpads report pixels; ~50 px feels like one notch.
                    MouseScrollDelta::PixelDelta(position) => position.y as f32 / 50.0,
                };
                self.camera.dolly(notches);
                self.view_changed();
            }
            WindowEvent::RedrawRequested => self.redraw()?,
            _ => {}
        }
        Ok(())
    }

    /// The camera moved: hand the new view to the render thread, which
    /// restarts accumulation from it.
    fn view_changed(&mut self) {
        self.session.set_camera(self.camera.camera());
        self.window.request_redraw();
    }

    /// One display frame: peek the render thread's latest linear average (or
    /// keep the one we hold), tonemap it at the panel's exposure, run the UI,
    /// present, and request the next redraw. The renderer accumulates
    /// independently; this loop just shows its most recent output, paced by
    /// the FIFO present.
    fn redraw(&mut self) -> anyhow::Result<()> {
        let size = self.window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(()); // minimized; the resize that restores us redraws
        }

        // The UI runs first so its exposure is current for this frame's
        // tonemap and an exposure drag lands this very frame. The stats it
        // shows are the previous frame's — one frame stale, imperceptible.
        let gui_frame = self
            .gui
            .run(&self.window, self.gpu.device_summary(), &self.stats);

        // Take a fresher frame if the render thread posted one; otherwise
        // re-show the one we hold (so exposure still tracks). Nothing yet —
        // the very first frames — just pumps the loop.
        if let Some(frame) = self.session.peek() {
            self.frame = Some(frame);
        }
        let Some(frame) = &self.frame else {
            self.window.request_redraw();
            return Ok(());
        };

        self.tonemap
            .apply(
                &self.gpu,
                frame.image(),
                frame.width(),
                frame.height(),
                self.gui.exposure(),
            )
            .context("tonemapping the frame")?;
        self.stats.sample = frame.sample_time();
        self.stats.size = (frame.width(), frame.height());
        self.stats.samples = frame.samples();

        self.window.pre_present_notify();
        let started = Instant::now();
        self.presenter
            .present(
                self.tonemap.display(),
                frame.width(),
                frame.height(),
                Some(&gui_frame),
            )
            .context("presenting the frame")?;
        self.stats.display = started.elapsed();

        // Peek again next vblank: the render thread runs ahead of us, so
        // there is almost always a newer frame waiting.
        self.window.request_redraw();
        Ok(())
    }
}
