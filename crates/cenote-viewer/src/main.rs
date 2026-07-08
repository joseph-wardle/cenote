//! Interactive viewer: the render live in a window, under an orbit camera,
//! with an egui stats/controls overlay. M1 build steps 2–3 — the M0 primary
//! kernel drives it until the wavefront engine replaces that kernel
//! (step 5).
//!
//! Single-threaded and event-driven (D-030): the loop sleeps until input.
//! Camera motion invalidates the cached frame and requests a redraw — one
//! blocking render at window size; UI-only redraws (hover, slider drags)
//! re-present the cached frame with fresh UI blended on top. Progressive
//! accumulation across redraws arrives with build step 4.

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
    /// The last traced frame, kept so UI-only redraws re-present instead of
    /// re-tracing. `None` when the view changed and a render is due.
    frame: Option<CachedFrame>,
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

/// A rendered frame staying on the GPU, with the size it was traced at.
struct CachedFrame {
    pixels: cenote::gpu::Buffer,
    width: u32,
    height: u32,
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
        // on the UI, and `consumed` keeps such events off the camera.
        let response = self.gui.on_window_event(&self.window, event);
        if response.repaint {
            self.window.request_redraw();
        }
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

    /// The camera moved: the cached frame no longer matches the view.
    fn view_changed(&mut self) {
        self.frame = None;
        self.window.request_redraw();
    }

    /// One frame: re-trace the scene if the view or window changed (else
    /// keep the cached render), run the UI, present both together.
    fn redraw(&mut self) -> anyhow::Result<()> {
        let size = self.window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(()); // minimized
        }
        let stale = |frame: &CachedFrame| frame.width != size.width || frame.height != size.height;
        if self.frame.as_ref().is_none_or(stale) {
            *self.scene.camera_mut() = self.camera.camera();
            let started = Instant::now();
            let pixels = self
                .renderer
                .render_to_buffer(&self.gpu, &self.scene, size.width, size.height)
                .context("rendering the frame")?;
            self.stats.render = started.elapsed();
            self.stats.render_size = (size.width, size.height);
            self.frame = Some(CachedFrame {
                pixels,
                width: size.width,
                height: size.height,
            });
        }

        let (gui_frame, repaint) =
            self.gui
                .run(&self.window, self.gpu.device_summary(), &self.stats);
        let frame = self.frame.as_ref().expect("rendered just above");
        self.window.pre_present_notify();
        let started = Instant::now();
        self.presenter
            .present(&frame.pixels, frame.width, frame.height, Some(&gui_frame))
            .context("presenting the frame")?;
        self.stats.present = started.elapsed();
        if repaint {
            self.window.request_redraw();
        }
        Ok(())
    }
}
