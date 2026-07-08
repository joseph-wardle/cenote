//! The overlay UI: device and frame stats, plus placeholder sliders that
//! stake out where the real controls land (exposure with the tonemap
//! kernel in step 4, material parameters with the GGX lobes in step 9).
//!
//! This is the egui half of the overlay — input translation, layout,
//! tessellation. The Vulkan half lives behind the core's `gpu` quarantine
//! and receives our output as a [`GuiFrame`].

use std::time::Duration;

use cenote::gpu::GuiFrame;
use winit::event::WindowEvent;
use winit::window::Window;

/// Timings the panel displays, measured by the redraw loop. `render` keeps
/// its last value across UI-only redraws that re-present a cached frame.
#[derive(Default)]
pub struct FrameStats {
    /// The last scene render (dispatch + fence wait), and its target size.
    pub render: Duration,
    pub render_size: (u32, u32),
    /// The last present (UI pass, blit, fence wait).
    pub present: Duration,
}

/// The egui context/winit bridge and the panel's widget state.
pub struct Gui {
    state: egui_winit::State,
    /// Placeholder — becomes the tonemap kernel's exposure in step 4.
    exposure: f32,
    /// Placeholder — becomes `OpenPBR` material parameters in step 9.
    roughness: f32,
    metalness: f32,
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
            roughness: 0.5,
            metalness: 0.0,
        }
    }

    /// Feed a window event to egui. `consumed` in the response means the UI
    /// claimed it (pointer over a panel, widget being dragged) and it must
    /// not also drive the camera; `repaint` means the UI wants a redraw.
    pub fn on_window_event(
        &mut self,
        window: &Window,
        event: &WindowEvent,
    ) -> egui_winit::EventResponse {
        self.state.on_window_event(window, event)
    }

    /// Run one UI frame and tessellate it for the presenter. The `bool` is
    /// egui asking for an immediate repaint (mid-animation).
    pub fn run(&mut self, window: &Window, device: &str, stats: &FrameStats) -> (GuiFrame, bool) {
        let input = self.state.take_egui_input(window);
        // Clone the (cheap, shared-reference) context so the closure can
        // borrow `self`'s widget state while `self.state` stays untouched.
        let context = self.state.egui_ctx().clone();
        let output = context.run(input, |context| self.panel(context, device, stats));
        self.state
            .handle_platform_output(window, output.platform_output);

        let primitives = context.tessellate(output.shapes, output.pixels_per_point);
        let repaint = output
            .viewport_output
            .get(&egui::ViewportId::ROOT)
            .is_some_and(|viewport| viewport.repaint_delay.is_zero());
        (
            GuiFrame {
                pixels_per_point: output.pixels_per_point,
                primitives,
                textures_delta: output.textures_delta,
            },
            repaint,
        )
    }

    fn panel(&mut self, context: &egui::Context, device: &str, stats: &FrameStats) {
        egui::Window::new("cenote")
            .default_pos([12.0, 12.0])
            .resizable(false)
            .show(context, |ui| {
                ui.label(egui::RichText::new(device).small());
                let millis = |duration: Duration| duration.as_secs_f64() * 1000.0;
                ui.monospace(format!(
                    "render  {:>6.2} ms  ({}×{})",
                    millis(stats.render),
                    stats.render_size.0,
                    stats.render_size.1,
                ));
                ui.monospace(format!("present {:>6.2} ms", millis(stats.present)));

                ui.separator();
                ui.add(egui::Slider::new(&mut self.exposure, -4.0..=4.0).text("exposure"));
                ui.add(egui::Slider::new(&mut self.roughness, 0.0..=1.0).text("roughness"));
                ui.add(egui::Slider::new(&mut self.metalness, 0.0..=1.0).text("metalness"));
                ui.label(
                    egui::RichText::new(
                        "placeholders — exposure goes live with the tonemap kernel \
                         (step 4), material with the GGX lobes (step 9)",
                    )
                    .small()
                    .weak(),
                );
            });
    }
}
