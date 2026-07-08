//! The GPU side of the viewer's egui overlay: `egui-ash-renderer` behind
//! the quarantine.
//!
//! The split with `cenote-viewer` follows the crate boundary: the viewer
//! runs egui — input, layout, tessellation — and hands the result here as a
//! [`GuiFrame`]; this module turns it into Vulkan work inside the
//! presenter's frame. UI logic never sees a `vk` handle, GPU code never
//! sees a widget.

use std::sync::{Arc, Mutex};

use ash::vk;
use egui_ash_renderer::{DynamicRendering, Options, Renderer};
use gpu_allocator::vulkan::Allocator;

use crate::error::Result;

/// One frame of UI, ready to draw over a rendered image: what
/// `egui::Context::run` plus `tessellate` produce, bundled for
/// [`crate::gpu::Presenter::present`].
pub struct GuiFrame {
    /// Physical pixels per logical point — the `HiDPI` scale the UI was
    /// laid out at.
    pub pixels_per_point: f32,
    /// Tessellated UI geometry, in paint order.
    pub primitives: Vec<egui::ClippedPrimitive>,
    /// Font-atlas and image changes to apply around this frame. Deltas are
    /// incremental: each must be applied exactly once, even for frames that
    /// end up not shown.
    pub textures_delta: egui::TexturesDelta,
}

/// The egui mesh renderer, targeting one fixed color format (the
/// presenter's swapchain format, which is chosen once and never changes).
pub(super) struct OverlayRenderer {
    renderer: Renderer,
}

impl OverlayRenderer {
    /// Create the renderer: one pipeline drawing egui meshes into a
    /// `format` color attachment via dynamic rendering, buffers and font
    /// atlas allocated from the shared `allocator`.
    pub(super) fn new(
        allocator: Arc<Mutex<Allocator>>,
        device: ash::Device,
        format: vk::Format,
    ) -> Result<Self> {
        let renderer = Renderer::with_gpu_allocator(
            allocator,
            device,
            DynamicRendering {
                color_attachment_format: format,
                depth_attachment_format: None,
            },
            Options {
                // Matches the presenter's pacing: fence-waited, one frame in
                // flight, so one vertex/index buffer set suffices.
                in_flight_frames: 1,
                enable_depth_test: false,
                enable_depth_write: false,
                // An sRGB attachment encodes in fixed function, so the
                // fragment shader must output linear.
                srgb_framebuffer: is_srgb(format),
            },
        )?;
        Ok(Self { renderer })
    }

    /// Apply a frame's texture additions and updates. Runs its own blocking
    /// upload submission (allocated from `pool`), so call it outside command
    /// recording, before the frame that samples the textures.
    pub(super) fn upload_textures(
        &mut self,
        queue: vk::Queue,
        pool: vk::CommandPool,
        delta: &egui::TexturesDelta,
    ) -> Result<()> {
        self.renderer.set_textures(queue, pool, &delta.set)?;
        Ok(())
    }

    /// Free the textures a frame retired. Only call once the last submission
    /// that sampled them has finished — for the presenter, after its fence
    /// wait.
    pub(super) fn free_textures(&mut self, delta: &egui::TexturesDelta) -> Result<()> {
        self.renderer.free_textures(&delta.free)?;
        Ok(())
    }

    /// Record `gui`'s draws into `cmd`. The caller has begun dynamic
    /// rendering on a color attachment covering `extent`.
    pub(super) fn draw(
        &mut self,
        cmd: vk::CommandBuffer,
        extent: vk::Extent2D,
        gui: &GuiFrame,
    ) -> Result<()> {
        self.renderer
            .cmd_draw(cmd, extent, gui.pixels_per_point, &gui.primitives)?;
        Ok(())
    }
}

/// Whether `format` is one of the sRGB-encoded swapchain formats the
/// presenter selects (its non-sRGB fallback path reaches here too).
fn is_srgb(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::B8G8R8A8_SRGB | vk::Format::R8G8B8A8_SRGB
    )
}
