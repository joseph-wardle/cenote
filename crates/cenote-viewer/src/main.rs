//! Interactive viewer: the render live in a window, under an orbit camera,
//! progressively accumulated, with an egui stats/controls overlay. Every
//! sample is a full path-traced estimate of the `OpenPBR` lobe mix (EON
//! diffuse, energy-compensated GGX conductor and dielectric specular)
//! under MIS-weighted direct light sampling of the quad lights and the
//! importance-sampled HDRI environment, so the image starts noisy and
//! visibly converges as the spp counter climbs.
//!
//! Single-threaded, and self-scheduling once visible: every redraw
//! accumulates one sample into the film, tonemaps (live exposure), presents,
//! and requests the next redraw — vsync paces the loop, and the spp counter
//! climbs forever. Camera motion resets the film; a resize replaces it.

mod camera;
mod ui;

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

/// The live viewer: window, GPU, scene, UI, and the input state driving the
/// orbit camera.
struct Viewer {
    // Field order is drop order: GPU resources go down before their
    // `Context`, and the surface-owning `Presenter` — like the `Gui`, whose
    // clipboard handle talks to the display server — before the window.
    presenter: cenote::gpu::Presenter,
    renderer: cenote::render::Renderer,
    scene: cenote::scene::Scene,
    /// The accumulation target, created at window size by the first redraw
    /// and replaced whenever the window size stops matching.
    film: Option<cenote::render::Film>,
    gpu: cenote::gpu::Context,
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
        let gpu = cenote::gpu::Context::presentable(display)?;
        let scene = cenote::scene::Scene::demo(&gpu)?;
        let camera = OrbitCamera::framing(scene.camera());
        let renderer = cenote::render::Renderer::new(&gpu)?;
        let size = window.inner_size();
        let presenter = gpu.create_presenter(
            display,
            window.window_handle()?.as_raw(),
            size.width,
            size.height,
        )?;
        let gui = Gui::new(&window);
        // Not every platform sends an initial redraw request unprompted.
        window.request_redraw();
        Ok(Self {
            presenter,
            renderer,
            scene,
            film: None,
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

    /// The camera moved: the film's accumulated samples no longer match the
    /// view, so the next sample starts a fresh average.
    fn view_changed(&mut self) {
        if let Some(film) = &mut self.film {
            film.reset();
        }
        self.window.request_redraw();
    }

    /// One frame: accumulate a sample into the film (replacing the film if
    /// the window size changed), tonemap at the panel's exposure, run the
    /// UI, present, and request the next redraw — accumulation never stops.
    fn redraw(&mut self) -> anyhow::Result<()> {
        let size = self.window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(()); // minimized; the resize that restores us redraws
        }
        let stale = |film: &cenote::render::Film| {
            film.width() != size.width || film.height() != size.height
        };
        if self.film.as_ref().is_none_or(stale) {
            self.film = Some(
                cenote::render::Film::new(&self.gpu, size.width, size.height)
                    .context("creating the film")?,
            );
        }
        let film = self.film.as_mut().expect("created just above");

        *self.scene.camera_mut() = self.camera.camera();
        let started = Instant::now();
        self.renderer
            .accumulate(&self.gpu, &self.scene, film)
            .context("accumulating a sample")?;
        self.stats.sample = started.elapsed();
        self.stats.size = (size.width, size.height);
        self.stats.samples = film.samples();

        // The UI runs before the tonemap so an exposure drag lands in this
        // very frame.
        let gui_frame = self
            .gui
            .run(&self.window, self.gpu.device_summary(), &self.stats);
        self.window.pre_present_notify();
        let started = Instant::now();
        self.renderer
            .tonemap(&self.gpu, film, self.gui.exposure())
            .context("tonemapping the film")?;
        self.presenter
            .present(
                film.display(),
                film.width(),
                film.height(),
                Some(&gui_frame),
            )
            .context("presenting the frame")?;
        self.stats.display = started.elapsed();

        // Progressive accumulation: there is always a next sample. FIFO
        // (vsync) presents pace this loop at the refresh rate.
        self.window.request_redraw();
        Ok(())
    }
}
