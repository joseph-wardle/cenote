//! The overlay UI: device and frame stats, the live exposure control, and
//! placeholder material sliders that stake out where the real parameters
//! land (with the GGX lobes in step 9).
//!
//! This is the egui half of the overlay — input translation, layout,
//! tessellation. The Vulkan half lives behind the core's `gpu` quarantine
//! and receives our output as a [`GuiFrame`].

use std::time::Duration;

use cenote::gpu::GuiFrame;
use winit::event::WindowEvent;
use winit::window::Window;

/// Numbers the panel displays, measured by the redraw loop.
#[derive(Default)]
pub struct FrameStats {
    /// The last accumulation wave (primary trace + film add, fence-waited),
    /// and the size it rendered at.
    pub sample: Duration,
    pub size: (u32, u32),
    /// The last display pass: tonemap through present.
    pub display: Duration,
    /// Samples in the film's average so far.
    pub samples: u32,
}

/// The egui context/winit bridge and the panel's widget state.
pub struct Gui {
    state: egui_winit::State,
    /// Exposure in stops, applied by the tonemap kernel.
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

    /// Exposure in stops, for [`cenote::render::Renderer::tonemap`].
    pub fn exposure(&self) -> f32 {
        self.exposure
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

    /// Run one UI frame and tessellate it for the presenter. No repaint
    /// signal comes back: the viewer accumulates continuously, so every
    /// frame is followed by another.
    pub fn run(&mut self, window: &Window, device: &str, stats: &FrameStats) -> GuiFrame {
        let input = self.state.take_egui_input(window);
        // Clone the (cheap, shared-reference) context so the closure can
        // borrow `self`'s widget state while `self.state` stays untouched.
        let context = self.state.egui_ctx().clone();
        let output = context.run(input, |context| self.panel(context, device, stats));
        self.state
            .handle_platform_output(window, output.platform_output);

        let primitives = context.tessellate(output.shapes, output.pixels_per_point);
        GuiFrame {
            pixels_per_point: output.pixels_per_point,
            primitives,
            textures_delta: output.textures_delta,
        }
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

                ui.separator();
                ui.add(egui::Slider::new(&mut self.roughness, 0.0..=1.0).text("roughness"));
                ui.add(egui::Slider::new(&mut self.metalness, 0.0..=1.0).text("metalness"));
                ui.label(
                    egui::RichText::new(
                        "placeholders — material goes live with the GGX lobes (step 9)",
                    )
                    .small()
                    .weak(),
                );
            });
    }
}
