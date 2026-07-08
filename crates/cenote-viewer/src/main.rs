//! Interactive viewer: the render live in a window, under an orbit camera.
//! M1 build step 2 — the M0 primary kernel drives it until the wavefront
//! engine replaces that kernel (step 5); the egui overlay lands in step 3.
//!
//! Single-threaded and event-driven (D-030): the loop sleeps until input,
//! camera motion requests a redraw, and each redraw is one blocking
//! render-then-present at window size. Progressive accumulation across
//! redraws arrives with build step 4.

mod camera;

use anyhow::Context as _;
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition};
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::raw_window_handle::{HasDisplayHandle as _, HasWindowHandle as _};
use winit::window::{Window, WindowId};

use crate::camera::OrbitCamera;

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

/// The live viewer: window, GPU, scene, and the input state driving the
/// orbit camera.
struct Viewer {
    // Field order is drop order: GPU resources go down before their
    // `Context`, and the surface-owning `Presenter` before the window it
    // draws to.
    presenter: cenote::gpu::Presenter,
    renderer: cenote::render::Renderer,
    scene: cenote::scene::Scene,
    gpu: cenote::gpu::Context,
    window: Window,
    camera: OrbitCamera,
    /// Left mouse button held — cursor motion orbits.
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
        // Not every platform sends an initial redraw request unprompted.
        window.request_redraw();
        Ok(Self {
            presenter,
            renderer,
            scene,
            gpu,
            window,
            camera,
            orbiting: false,
            cursor: None,
        })
    }

    fn handle(&mut self, event: &WindowEvent) -> anyhow::Result<()> {
        match *event {
            WindowEvent::Resized(size) => {
                self.presenter.resize(size.width, size.height);
                self.window.request_redraw();
            }
            WindowEvent::MouseInput {
                button: MouseButton::Left,
                state,
                ..
            } => self.orbiting = state == ElementState::Pressed,
            WindowEvent::CursorMoved { position, .. } => {
                if self.orbiting
                    && let Some(last) = self.cursor
                {
                    self.camera
                        .orbit((position.x - last.x) as f32, (position.y - last.y) as f32);
                    self.window.request_redraw();
                }
                self.cursor = Some(position);
            }
            WindowEvent::CursorLeft { .. } => self.cursor = None,
            WindowEvent::MouseWheel { delta, .. } => {
                let notches = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    // Trackpads report pixels; ~50 px feels like one notch.
                    MouseScrollDelta::PixelDelta(position) => position.y as f32 / 50.0,
                };
                self.camera.dolly(notches);
                self.window.request_redraw();
            }
            WindowEvent::RedrawRequested => self.redraw()?,
            _ => {}
        }
        Ok(())
    }

    /// One blocking frame at window size: move the scene camera to the
    /// orbit position, render, present.
    fn redraw(&mut self) -> anyhow::Result<()> {
        let size = self.window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(()); // minimized
        }
        *self.scene.camera_mut() = self.camera.camera();
        let pixels = self
            .renderer
            .render_to_buffer(&self.gpu, &self.scene, size.width, size.height)
            .context("rendering the frame")?;
        self.window.pre_present_notify();
        self.presenter
            .present(&pixels, size.width, size.height)
            .context("presenting the frame")?;
        Ok(())
    }
}
