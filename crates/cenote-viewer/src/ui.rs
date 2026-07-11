//! The overlay UI: device and frame stats, the live exposure control, and
//! the lookdev material panel (in [`crate::lookdev`]).
//!
//! This is the egui half of the overlay — input translation, layout,
//! tessellation. The Vulkan half lives behind the core's `gpu` quarantine
//! and receives our output as a [`GuiFrame`].

use std::time::Duration;

use cenote::gpu::GuiFrame;
use cenote::scene::changeset::MaterialPatch;
use cenote::scene::description::SceneDescription;
use winit::event::WindowEvent;
use winit::window::Window;

use crate::lookdev::Lookdev;

/// Numbers the panel displays, measured by the redraw loop.
#[derive(Default)]
pub struct FrameStats {
    /// The render thread's last sample — trace plus film accumulate, timed on
    /// that thread — and the size it rendered at. The viewer's own tonemap and
    /// present are not in here; the present is the `display` line below.
    pub sample: Duration,
    pub size: (u32, u32),
    /// The last present.
    pub display: Duration,
    /// Samples in the film's average so far.
    pub samples: u32,
}

/// The egui context/winit bridge and the panel's widget state.
pub struct Gui {
    state: egui_winit::State,
    /// Exposure in stops, applied by the tonemap kernel.
    exposure: f32,
    /// Show the OIDN-denoised view instead of the raw average.
    #[cfg(feature = "denoise")]
    denoise: bool,
    /// The material inspector — its own window, driven each frame off the
    /// scene replica the viewer passes to [`Gui::run`].
    lookdev: Lookdev,
}

impl Gui {
    pub fn new(window: &Window) -> Self {
        let context = egui::Context::default();
        let state = egui_winit::State::new(
            context,
            egui::ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            window.theme(),
            None,
        );
        Self {
            state,
            exposure: 0.0,
            #[cfg(feature = "denoise")]
            denoise: false,
            lookdev: Lookdev::default(),
        }
    }

    /// Exposure in stops, for [`cenote::render::Tonemap::apply`].
    pub fn exposure(&self) -> f32 {
        self.exposure
    }

    /// Whether the panel's denoise toggle is on.
    #[cfg(feature = "denoise")]
    pub fn denoise(&self) -> bool {
        self.denoise
    }

    /// Feed a window event to egui. `consumed` in the response means the UI
    /// claimed it (pointer over a panel, widget being dragged) and it must
    /// not also drive the camera.
    pub fn on_window_event(
        &mut self,
        window: &Window,
        event: &WindowEvent,
    ) -> egui_winit::EventResponse {
        self.state.on_window_event(window, event)
    }

    /// Run one UI frame and tessellate it for the presenter. Returns the
    /// frame plus any material edit the lookdev panel produced — the target
    /// material's name and a patch of its values, for the caller to apply to
    /// both the scene replica and the render session. No repaint signal
    /// comes back: the viewer accumulates continuously, so every frame is
    /// followed by another.
    pub fn run(
        &mut self,
        window: &Window,
        device: &str,
        stats: &FrameStats,
        description: &SceneDescription,
    ) -> (GuiFrame, Option<(String, MaterialPatch)>) {
        let input = self.state.take_egui_input(window);
        // Clone the (cheap, shared-reference) context so the closure can
        // borrow `self`'s widget state while `self.state` stays untouched.
        let context = self.state.egui_ctx().clone();
        let mut edit = None;
        let output = context.run(input, |context| {
            self.panel(context, device, stats);
            edit = self.lookdev.show(context, description);
        });
        self.state
            .handle_platform_output(window, output.platform_output);

        let primitives = context.tessellate(output.shapes, output.pixels_per_point);
        let frame = GuiFrame {
            pixels_per_point: output.pixels_per_point,
            primitives,
            textures_delta: output.textures_delta,
        };
        (frame, edit)
    }

    fn panel(&mut self, context: &egui::Context, device: &str, stats: &FrameStats) {
        egui::Window::new("cenote")
            .default_pos([12.0, 12.0])
            .resizable(false)
            .show(context, |ui| {
                ui.label(egui::RichText::new(device).small());
                let millis = |duration: Duration| duration.as_secs_f64() * 1000.0;
                ui.monospace(format!(
                    "sample  {:>6.2} ms  ({}×{})",
                    millis(stats.sample),
                    stats.size.0,
                    stats.size.1,
                ));
                ui.monospace(format!("display {:>6.2} ms", millis(stats.display)));
                ui.monospace(format!("spp     {}", stats.samples));

                ui.separator();
                ui.add(egui::Slider::new(&mut self.exposure, -4.0..=4.0).text("exposure"));
                #[cfg(feature = "denoise")]
                ui.checkbox(&mut self.denoise, "denoise");
            });
    }
}
