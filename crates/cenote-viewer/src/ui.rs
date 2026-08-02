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
use cenote::stats::{Bound, READABLE_SAMPLES, Stats};
use winit::event::WindowEvent;
use winit::window::Window;

use crate::lookdev::Lookdev;

/// Numbers the panel displays.
///
/// Almost all of it is the renderer's own [`Stats`], taken verbatim off the
/// published frame — the viewer measures nothing the renderer already
/// measures, so the overlay and a headless report can never disagree. The
/// one number that is genuinely the viewer's own is `display`: the render
/// thread has no idea what a present costs.
#[derive(Default)]
pub struct FrameStats {
    /// What the renderer measured, as of the last frame we took.
    pub render: Stats,
    /// The last present — tonemap and blit, on this thread.
    pub display: Duration,
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
                stats_lines(ui, stats);

                ui.separator();
                ui.add(egui::Slider::new(&mut self.exposure, -4.0..=4.0).text("exposure"));
                #[cfg(feature = "denoise")]
                ui.checkbox(&mut self.denoise, "denoise");
            });
    }
}

/// Milliseconds, the one unit every time in this panel is shown in.
fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

/// Bytes, in the largest unit that keeps the number readable.
fn bytes(count: u64) -> String {
    // A display string: the low bits of a gigabyte are not the point.
    let value = count as f64;
    for (unit, scale) in [("GiB", 1u64 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)] {
        let scale = scale as f64;
        if value >= scale {
            return format!("{:>6.1} {unit}", value / scale);
        }
    }
    format!("{count:>6} B  ")
}

/// The measurement block: four always-visible lines, then the detail behind
/// collapsing headers.
///
/// The split is the editorial claim. A frame time, its spread, what the GPU
/// actually did, and how far along the render is — those answer *is it
/// fast* and *is it done*, and they are worth permanent screen space. The
/// per-kernel breakdown, the startup marks, and the memory buckets answer
/// *why*, which is a question you ask on purpose. Behind a header they are
/// one click away and zero pixels of noise until then.
fn stats_lines(ui: &mut egui::Ui, stats: &FrameStats) {
    let frame = &stats.render.frame;
    ui.monospace(format!(
        "frame   {:>6.2} ms  ({}×{})",
        millis(stats.render.smoothed.median),
        frame.size.0,
        frame.size.1,
    ));
    // The median is what the frame *usually* costs; the p95 beside it is
    // the hitch. One number without the other reads as smoother than the
    // render is.
    ui.monospace(format!(
        "  p95   {:>6.2} ms   display {:>5.2} ms",
        millis(stats.render.smoothed.p95),
        millis(stats.display),
    ));
    // The verdict, with both numbers that produced it: a summed GPU time
    // well under the frame is the signature of a wave spending more on
    // launching kernels than on running them.
    match frame.bound() {
        Bound::Unknown => ui.monospace("  gpu        —  (no timestamps)"),
        bound => ui.monospace(format!("  gpu   {:>6.2} ms   {bound}", millis(frame.gpu()))),
    };
    ui.monospace(format!("spp     {:>6}", frame.samples));

    if frame.passes.has_breakdown() {
        egui::CollapsingHeader::new("kernels")
            .id_salt("stats.kernels")
            .show(ui, |ui| {
                for pass in &frame.passes {
                    ui.monospace(format!(
                        "{:<22}{:>6.3} ms ×{}",
                        pass.label,
                        millis(pass.gpu),
                        pass.calls,
                    ));
                }
            });
    }

    let marks = &stats.render.interactivity;
    egui::CollapsingHeader::new("latency")
        .id_salt("stats.latency")
        .show(ui, |ui| {
            let mark = |ui: &mut egui::Ui, name: &str, value: Option<Duration>| {
                ui.monospace(match value {
                    Some(value) => format!("{name:<14}{:>8.1} ms", millis(value)),
                    None => format!("{name:<14}{:>8}   ", "—"),
                });
            };
            mark(ui, "first ray", marks.to_first_ray);
            mark(ui, "first pixel", marks.to_first_pixel);
            mark(ui, &format!("{READABLE_SAMPLES} spp"), marks.to_readable);
        });

    let memory = &stats.render.memory;
    let headline = match memory.budget {
        Some(budget) => format!("memory {} /{}", bytes(memory.total()), bytes(budget)),
        None => format!("memory {}", bytes(memory.total())),
    };
    egui::CollapsingHeader::new(headline)
        .id_salt("stats.memory")
        .show(ui, |ui| {
            for (name, value) in memory.buckets() {
                ui.monospace(format!("{name:<12}{}", bytes(value)));
            }
        });
}
