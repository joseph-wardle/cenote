//! Window presentation: the surface, the swapchain, and the transfer path
//! that puts a rendered frame on screen.
//!
//! [`Presenter`] shows linear RGBA f32 pixel buffers: copy into an RGBA32F
//! transfer image, then one blit onto the acquired swapchain image — the
//! blit scales to the window and encodes to the swapchain's sRGB format in
//! fixed function, so displaying a frame needs no shader. The tonemap
//! kernel (M1 build step 4) will replace the copy leg by writing the
//! transfer image directly; the blit-and-present tail stays.
//!
//! Pacing matches the crate's blocking-submit model: one frame in flight,
//! fence-waited before [`Presenter::present`] returns. Timeline-semaphore
//! pacing (D-022) arrives with the wavefront loop, not here.

use std::slice;
use std::sync::{Arc, Mutex};

use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use crate::error::{Error, Result};
use crate::gpu::{Buffer, Context, MemoryLocation};

/// Format of the transfer image — matches the pixel buffers' row-major
/// RGBA f32 layout, so the buffer→image copy is a straight memcpy.
const TRANSFER_FORMAT: vk::Format = vk::Format::R32G32B32A32_SFLOAT;

/// The whole color plane of a single-mip, single-layer image — every image
/// this module touches.
const COLOR_LAYER: vk::ImageSubresourceLayers = vk::ImageSubresourceLayers {
    aspect_mask: vk::ImageAspectFlags::COLOR,
    mip_level: 0,
    base_array_layer: 0,
    layer_count: 1,
};

/// Owns a window's surface and swapchain and shows rendered frames on it.
///
/// Created by [`Context::create_presenter`]; like every `gpu` resource, drop
/// it before the `Context` — and before the window whose handles it was
/// created from.
pub struct Presenter {
    surface_loader: ash::khr::surface::Instance,
    swapchain_loader: ash::khr::swapchain::Device,
    surface: vk::SurfaceKHR,
    /// Null while the window is zero-area (minimized).
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    /// One per swapchain image, indexed by acquired image index. Per-image
    /// rather than per-frame: a present may still be waiting on image *i*'s
    /// semaphore when the next frame starts, but never once *i* has been
    /// re-acquired — so indexing by image makes reuse safe.
    render_finished: Vec<vk::Semaphore>,
    extent: vk::Extent2D,
    /// The window's current inner size; the swapchain is rebuilt to match
    /// lazily, on the next present.
    desired_extent: vk::Extent2D,
    dirty: bool,
    /// Signaled by acquire, waited by the frame's submit. Reusable each
    /// frame because the fence wait proves the previous wait executed.
    image_acquired: vk::Semaphore,
    frame_done: vk::Fence,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    transfer: Option<TransferImage>,
    physical_device: vk::PhysicalDevice,
    queue: vk::Queue,
    device: ash::Device,
    allocator: Arc<Mutex<Allocator>>,
}

/// The RGBA32F image frames pass through between pixel buffer and swapchain,
/// sized to the *render*, not the window — the blit rescales.
struct TransferImage {
    image: vk::Image,
    allocation: Allocation,
    width: u32,
    height: u32,
}

impl Context {
    /// Create a [`Presenter`] for the window behind `window`/`display`.
    /// `width`/`height` is the window's current inner size in physical
    /// pixels; keep it current via [`Presenter::resize`].
    ///
    /// # Errors
    ///
    /// [`crate::Error::Vulkan`] if surface or swapchain creation fails;
    /// [`crate::Error::NoCapableGpu`] if the selected device cannot present
    /// to this surface.
    ///
    /// # Panics
    ///
    /// If `self` was not created with [`Context::presentable`] — the
    /// required extensions aren't enabled, a programmer bug.
    pub fn create_presenter(
        &self,
        display: RawDisplayHandle,
        window: RawWindowHandle,
        width: u32,
        height: u32,
    ) -> Result<Presenter> {
        assert!(
            self.presentable,
            "presentation needs a Context::presentable context"
        );
        let surface = unsafe {
            ash_window::create_surface(&self.entry, &self.instance, display, window, None)?
        };
        let surface_loader = ash::khr::surface::Instance::new(&self.entry, &self.instance);

        // Selection couldn't check this — surfaces outlive it conceptually
        // (no window existed yet) — so verify here. On every desktop driver
        // the universal graphics+compute family presents.
        let support = unsafe {
            surface_loader.get_physical_device_surface_support(
                self.physical_device,
                self.queue_family_index,
                surface,
            )
        };
        match support {
            Ok(true) => {}
            not_supported => {
                unsafe { surface_loader.destroy_surface(surface, None) };
                return Err(match not_supported {
                    Ok(_) => Error::NoCapableGpu(
                        "  selected device: its compute queue family cannot present to this window\n"
                            .into(),
                    ),
                    Err(err) => err.into(),
                });
            }
        }

        let mut presenter = Presenter {
            surface_loader,
            swapchain_loader: ash::khr::swapchain::Device::new(&self.instance, &self.device),
            surface,
            swapchain: vk::SwapchainKHR::null(),
            images: Vec::new(),
            render_finished: Vec::new(),
            extent: vk::Extent2D::default(),
            desired_extent: vk::Extent2D { width, height },
            dirty: false,
            image_acquired: vk::Semaphore::null(),
            frame_done: vk::Fence::null(),
            pool: vk::CommandPool::null(),
            cmd: vk::CommandBuffer::null(),
            transfer: None,
            physical_device: self.physical_device,
            queue: self.queue,
            device: self.device.clone(),
            allocator: self.allocator_handle(),
        };
        // From here, failure rolls back through Presenter's Drop, which
        // tolerates the null handles of whatever wasn't reached (Vulkan
        // destroy calls accept null).
        presenter.create_frame_resources(self.queue_family_index)?;
        presenter.recreate_swapchain()?;
        Ok(presenter)
    }
}

impl Presenter {
    /// Note the window's new inner size (physical pixels). The swapchain is
    /// rebuilt to match on the next [`Presenter::present`].
    pub fn resize(&mut self, width: u32, height: u32) {
        self.desired_extent = vk::Extent2D { width, height };
        self.dirty = true;
    }

    /// Show a frame: copy the `width`×`height` linear RGBA f32 `pixels`
    /// buffer into the transfer image, blit it across the whole swapchain
    /// image (bilinear rescale + sRGB encode), and present. Blocks until the
    /// GPU work finishes — one frame in flight, like every submit in the
    /// crate. A zero-area (minimized) window makes this a no-op.
    ///
    /// `pixels` needs `TRANSFER_SRC` usage and an already-completed writer,
    /// which every blocking dispatch in this crate guarantees.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Vulkan`] if a swapchain, submission, or present call
    /// fails. An out-of-date swapchain is not an error: the frame is skipped
    /// and the swapchain rebuilt for the next one.
    ///
    /// # Panics
    ///
    /// If `pixels` is smaller than `width`×`height` RGBA f32 texels.
    pub fn present(&mut self, pixels: &Buffer, width: u32, height: u32) -> Result<()> {
        assert!(
            pixels.size() >= u64::from(width) * u64::from(height) * 16,
            "pixel buffer is smaller than its stated dimensions"
        );
        if self.dirty {
            self.recreate_swapchain()?;
        }
        if self.extent.width == 0 || self.extent.height == 0 {
            return Ok(());
        }
        self.ensure_transfer_image(width, height)?;

        // Acquire, with one rebuild-and-retry for a stale swapchain. A
        // failed acquire leaves `image_acquired` unsignaled, so retrying
        // with the same semaphore is sound.
        let mut retried = false;
        let (index, suboptimal) = loop {
            let acquired = unsafe {
                self.swapchain_loader.acquire_next_image(
                    self.swapchain,
                    u64::MAX,
                    self.image_acquired,
                    vk::Fence::null(),
                )
            };
            match acquired {
                Ok(pair) => break pair,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) if !retried => {
                    retried = true;
                    self.recreate_swapchain()?;
                    if self.extent.width == 0 || self.extent.height == 0 {
                        return Ok(());
                    }
                }
                Err(err) => return Err(err.into()),
            }
        };

        self.record(pixels, width, height, self.images[index as usize])?;

        let wait = vk::SemaphoreSubmitInfo::default()
            .semaphore(self.image_acquired)
            // The first swapchain-image access below is its BLIT-stage
            // layout transition, so waiting at BLIT is exact.
            .stage_mask(vk::PipelineStageFlags2::BLIT);
        let signal = vk::SemaphoreSubmitInfo::default()
            .semaphore(self.render_finished[index as usize])
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
        let cmd = vk::CommandBufferSubmitInfo::default().command_buffer(self.cmd);
        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(slice::from_ref(&wait))
            .command_buffer_infos(slice::from_ref(&cmd))
            .signal_semaphore_infos(slice::from_ref(&signal));
        unsafe {
            self.device
                .queue_submit2(self.queue, slice::from_ref(&submit), self.frame_done)?;
        }

        let semaphores = [self.render_finished[index as usize]];
        let swapchains = [self.swapchain];
        let indices = [index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&semaphores)
            .swapchains(&swapchains)
            .image_indices(&indices);
        let presented = unsafe {
            self.swapchain_loader
                .queue_present(self.queue, &present_info)
        };

        // One frame in flight: this frame's GPU work ends before we return.
        unsafe {
            self.device
                .wait_for_fences(slice::from_ref(&self.frame_done), true, u64::MAX)?;
            self.device
                .reset_fences(slice::from_ref(&self.frame_done))?;
        }

        match presented {
            // Suboptimal (from acquire or present) still showed the frame;
            // rebuild to match the window before the next one.
            Ok(this_suboptimal) => {
                self.dirty |= suboptimal || this_suboptimal;
                Ok(())
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.dirty = true;
                Ok(())
            }
            Err(err) => Err(err.into()),
        }
    }

    /// Record this frame's commands: buffer → transfer image → blit onto
    /// `target` → ready to present.
    fn record(&self, pixels: &Buffer, width: u32, height: u32, target: vk::Image) -> Result<()> {
        let transfer = self
            .transfer
            .as_ref()
            .expect("present() ensured the transfer image")
            .image;
        let device = &self.device;
        unsafe {
            device.reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device.begin_command_buffer(self.cmd, &begin)?;

            // The transfer image holds last frame — discard (UNDEFINED) and
            // make it a copy target. The pixel buffer itself needs no
            // barrier: its writer's fence already made the writes available.
            image_barrier(
                device,
                self.cmd,
                transfer,
                (
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                ),
                (vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE),
                (
                    vk::PipelineStageFlags2::COPY,
                    vk::AccessFlags2::TRANSFER_WRITE,
                ),
            );

            let region = vk::BufferImageCopy::default()
                .image_subresource(COLOR_LAYER)
                .image_extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                });
            device.cmd_copy_buffer_to_image(
                self.cmd,
                pixels.handle(),
                transfer,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                slice::from_ref(&region),
            );

            image_barrier(
                device,
                self.cmd,
                transfer,
                (
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                ),
                (
                    vk::PipelineStageFlags2::COPY,
                    vk::AccessFlags2::TRANSFER_WRITE,
                ),
                (
                    vk::PipelineStageFlags2::BLIT,
                    vk::AccessFlags2::TRANSFER_READ,
                ),
            );
            // The swapchain image: whatever it held is being replaced
            // wholesale (UNDEFINED). Source stage BLIT chains this
            // transition after the submit's acquire-semaphore wait.
            image_barrier(
                device,
                self.cmd,
                target,
                (
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                ),
                (vk::PipelineStageFlags2::BLIT, vk::AccessFlags2::NONE),
                (
                    vk::PipelineStageFlags2::BLIT,
                    vk::AccessFlags2::TRANSFER_WRITE,
                ),
            );

            blit_whole_image(
                device,
                self.cmd,
                (transfer, width, height),
                (target, self.extent.width, self.extent.height),
            );

            // Hand the image to the presentation engine; the destination
            // side of the dependency is the signal semaphore, not a stage.
            image_barrier(
                device,
                self.cmd,
                target,
                (
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::PRESENT_SRC_KHR,
                ),
                (
                    vk::PipelineStageFlags2::BLIT,
                    vk::AccessFlags2::TRANSFER_WRITE,
                ),
                (vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE),
            );
            device.end_command_buffer(self.cmd)?;
        }
        Ok(())
    }

    fn create_frame_resources(&mut self, queue_family_index: u32) -> Result<()> {
        let device = &self.device;
        unsafe {
            let pool_info = vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                .queue_family_index(queue_family_index);
            self.pool = device.create_command_pool(&pool_info, None)?;
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            self.cmd = device.allocate_command_buffers(&alloc_info)?[0];
            self.image_acquired =
                device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
            self.frame_done = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
        }
        Ok(())
    }

    /// (Re)build the swapchain at the window's current size. A zero-area
    /// window (minimized) leaves the swapchain absent — [`Presenter::present`]
    /// no-ops — until a resize brings it back.
    fn recreate_swapchain(&mut self) -> Result<()> {
        // Idle before destroying: strictly, the presentation engine's use of
        // the old swapchain and semaphores isn't covered by wait_idle — the
        // real fix is VK_EXT_swapchain_maintenance1 — but every driver
        // tolerates this, and it's the ecosystem-standard shape.
        unsafe {
            self.device.device_wait_idle()?;
            self.destroy_swapchain();
        }
        self.dirty = false;

        let capabilities = unsafe {
            self.surface_loader
                .get_physical_device_surface_capabilities(self.physical_device, self.surface)?
        };
        self.extent = if capabilities.current_extent.width == u32::MAX {
            // The surface takes its size from the swapchain (Wayland): use
            // the window's, within the surface's limits.
            vk::Extent2D {
                width: self.desired_extent.width.clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: self.desired_extent.height.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            }
        } else {
            capabilities.current_extent
        };
        if self.extent.width == 0 || self.extent.height == 0 {
            return Ok(());
        }
        if !capabilities
            .supported_usage_flags
            .contains(vk::ImageUsageFlags::TRANSFER_DST)
        {
            return Err(Error::NoCapableGpu(
                "  selected device: swapchain images cannot be blit targets\n".into(),
            ));
        }

        let formats = unsafe {
            self.surface_loader
                .get_physical_device_surface_formats(self.physical_device, self.surface)?
        };
        // An sRGB format makes the blit's fixed-function encode correct for
        // the linear values we render. Every desktop driver offers one; the
        // fallback would merely display too dark, not break.
        let format = formats
            .iter()
            .copied()
            .find(|f| {
                (f.format == vk::Format::B8G8R8A8_SRGB || f.format == vk::Format::R8G8B8A8_SRGB)
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or(formats[0]);

        // One more than the minimum so acquire rarely blocks; 0 means "no
        // maximum".
        let image_count = if capabilities.max_image_count == 0 {
            capabilities.min_image_count + 1
        } else {
            (capabilities.min_image_count + 1).min(capabilities.max_image_count)
        };
        let composite = [
            vk::CompositeAlphaFlagsKHR::OPAQUE,
            vk::CompositeAlphaFlagsKHR::INHERIT,
            vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
        ]
        .into_iter()
        .find(|&flag| capabilities.supported_composite_alpha.contains(flag))
        .unwrap_or(vk::CompositeAlphaFlagsKHR::OPAQUE);

        let info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface)
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(self.extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(composite)
            // FIFO is the one mode Vulkan guarantees — and it's vsync,
            // which suits an event-driven viewer.
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true);
        unsafe {
            self.swapchain = self.swapchain_loader.create_swapchain(&info, None)?;
            self.images = self.swapchain_loader.get_swapchain_images(self.swapchain)?;
            for _ in 0..self.images.len() {
                self.render_finished.push(
                    self.device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?,
                );
            }
        }
        Ok(())
    }

    /// Destroy the swapchain and its per-image semaphores; tolerates an
    /// absent swapchain. Caller has made the device idle.
    unsafe fn destroy_swapchain(&mut self) {
        unsafe {
            for semaphore in self.render_finished.drain(..) {
                self.device.destroy_semaphore(semaphore, None);
            }
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }
        self.swapchain = vk::SwapchainKHR::null();
        self.images.clear();
        self.extent = vk::Extent2D::default();
    }

    /// Make the transfer image match the render size, recreating on change.
    /// The previous frame's fence wait already ran, so an old image is idle.
    fn ensure_transfer_image(&mut self, width: u32, height: u32) -> Result<()> {
        if self
            .transfer
            .as_ref()
            .is_some_and(|t| t.width == width && t.height == height)
        {
            return Ok(());
        }
        self.destroy_transfer_image();

        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(TRANSFER_FORMAT)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { self.device.create_image(&info, None)? };

        let requirements = unsafe { self.device.get_image_memory_requirements(image) };
        let allocated = self
            .allocator
            .lock()
            .expect("allocator mutex poisoned")
            .allocate(&AllocationCreateDesc {
                name: "present.transfer",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            });
        let allocation = match allocated {
            Ok(allocation) => allocation,
            Err(err) => {
                unsafe { self.device.destroy_image(image, None) };
                return Err(err.into());
            }
        };
        let bound = unsafe {
            self.device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
        };
        if let Err(err) = bound {
            unsafe { self.device.destroy_image(image, None) };
            free_allocation(&self.allocator, allocation);
            return Err(err.into());
        }

        self.transfer = Some(TransferImage {
            image,
            allocation,
            width,
            height,
        });
        Ok(())
    }

    fn destroy_transfer_image(&mut self) {
        if let Some(transfer) = self.transfer.take() {
            unsafe { self.device.destroy_image(transfer.image, None) };
            free_allocation(&self.allocator, transfer.allocation);
        }
    }
}

impl Drop for Presenter {
    fn drop(&mut self) {
        unsafe {
            // The last presented frame may still be in flight.
            self.device.device_wait_idle().ok();
            self.destroy_swapchain();
            self.device.destroy_semaphore(self.image_acquired, None);
            self.device.destroy_fence(self.frame_done, None);
            self.device.destroy_command_pool(self.pool, None);
        }
        self.destroy_transfer_image();
        unsafe { self.surface_loader.destroy_surface(self.surface, None) };
    }
}

fn free_allocation(allocator: &Arc<Mutex<Allocator>>, allocation: Allocation) {
    match allocator.lock() {
        Ok(mut allocator) => {
            if let Err(err) = allocator.free(allocation) {
                log::error!("failed to free transfer-image allocation: {err}");
            }
        }
        Err(_) => log::error!("allocator mutex poisoned — leaking transfer-image allocation"),
    }
}

/// Record a blit of all of `src` onto all of `dst` (each given with its
/// width and height): bilinear rescale plus the formats' conversion — the
/// sRGB encode, when `dst` is a swapchain image. Layouts are the transfer
/// ones the caller's barriers established.
fn blit_whole_image(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    (src, src_width, src_height): (vk::Image, u32, u32),
    (dst, dst_width, dst_height): (vk::Image, u32, u32),
) {
    let corner = |width: u32, height: u32| vk::Offset3D {
        x: width.cast_signed(),
        y: height.cast_signed(),
        z: 1,
    };
    let blit = vk::ImageBlit2::default()
        .src_subresource(COLOR_LAYER)
        .src_offsets([vk::Offset3D::default(), corner(src_width, src_height)])
        .dst_subresource(COLOR_LAYER)
        .dst_offsets([vk::Offset3D::default(), corner(dst_width, dst_height)]);
    let info = vk::BlitImageInfo2::default()
        .src_image(src)
        .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .dst_image(dst)
        .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .regions(slice::from_ref(&blit))
        .filter(vk::Filter::LINEAR);
    unsafe { device.cmd_blit_image2(cmd, &info) };
}

/// Record one full-image layout transition, `(old, new)` layouts ordered
/// between the `(stage, access)` source and destination scopes.
fn image_barrier(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    (old_layout, new_layout): (vk::ImageLayout, vk::ImageLayout),
    (src_stage, src_access): (vk::PipelineStageFlags2, vk::AccessFlags2),
    (dst_stage, dst_access): (vk::PipelineStageFlags2, vk::AccessFlags2),
) {
    let barrier = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    let info = vk::DependencyInfo::default().image_memory_barriers(slice::from_ref(&barrier));
    unsafe { device.cmd_pipeline_barrier2(cmd, &info) };
}
