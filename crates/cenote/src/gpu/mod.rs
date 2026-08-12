//! The unsafe-Vulkan quarantine.
//!
//! [`Context`] owns instance→device bring-up: validation wiring, physical
//! device selection against the ray-tracing baseline, one compute queue, and
//! the memory allocator. Code outside `gpu` never touches raw `vk` handles
//! or writes `unsafe`. Bring-up lives in `init`; buffers, sampled images,
//! one-shot submits, compute pipelines, acceleration structures, window
//! presentation, and the viewer's egui overlay pass in the other submodules.
//!
//! There is no backend abstraction here and there never will be — Cenote is
//! single-backend by design: a reader who knows Vulkan should be reading
//! Vulkan.

use std::mem::ManuallyDrop;
use std::sync::{Arc, Mutex};

use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use raw_window_handle::RawDisplayHandle;

use crate::error::Result;

mod accel;
mod buffer;
mod image;
mod init;
mod ledger;
mod overlay;
mod pipeline;
mod present;
mod submit;
mod timing;
mod upload;

pub use accel::{AccelerationStructure, TlasInstance};
pub use buffer::{Buffer, MemoryLocation};
pub use image::SampledImage;
pub use overlay::GuiFrame;
pub use pipeline::{
    Bindings, ComputePipeline, MAX_SCENE_TEXTURES, SceneBindings, TABLE_PLANES, TABLE_VOLUMES,
};
pub use present::Presenter;
pub use submit::Pass;
pub use timing::PassTimer;
pub use upload::Upload;

use init::DebugMessenger;
use ledger::Ledger;
use submit::Queue;

/// An initialized Vulkan device ready for compute dispatch.
///
/// Owns everything from the instance down to the allocator; dropping it
/// tears the stack down in reverse order after waiting for the device to
/// go idle.
pub struct Context {
    // Shared with every Buffer so they can free themselves on drop; the
    // Context's reference is released in Drop before `device` because
    // gpu-allocator frees device memory. Buffers must not outlive the
    // Context (checked with a strong-count log in Drop).
    allocator: ManuallyDrop<Arc<Mutex<Allocator>>>,
    /// Live bytes per [`crate::stats::Memory`] bucket, shared with every
    /// resource that owns an allocation.
    ledger: Arc<Ledger>,
    device: ash::Device,
    // Extension function table for VK_KHR_acceleration_structure; plain
    // function pointers, nothing to destroy.
    accel_loader: ash::khr::acceleration_structure::Device,
    // The one queue, lock-guarded so the render and present threads take
    // turns submitting to it — see [`submit::Queue`].
    queue: Queue,
    queue_family_index: u32,
    physical_device: vk::PhysicalDevice,
    device_type: vk::PhysicalDeviceType,
    /// `VkPhysicalDeviceIDProperties::deviceUUID` — read once here so the
    /// denoiser can open the same physical GPU. See [`Context::device_uuid`].
    device_uuid: [u8; 16],
    /// The device's minimum alignment for acceleration-structure build
    /// scratch — a constant, read once at bring-up rather than once per
    /// build. See [`accel::scratch_alignment`].
    scratch_alignment: vk::DeviceSize,
    /// Nanoseconds a timestamp tick represents, and how many of the 64 bits
    /// this queue family actually fills — everything [`PassTimer`] needs to
    /// turn query results into durations. Zero either means no timestamps.
    timestamp_period: f32,
    timestamp_valid_bits: u32,
    /// The device-local heap the memory buckets are spending against — a
    /// constant of the device, so it is read once here and never again. See
    /// [`ledger`], whose "reading it is five relaxed loads" depends on it.
    device_local_heap: Option<u64>,
    /// Created via [`Context::presentable`], i.e. the surface and swapchain
    /// extensions are enabled and [`Context::create_presenter`] may be called.
    presentable: bool,
    /// `VK_KHR_external_memory_fd` is enabled, so
    /// [`Context::create_exported_buffer`] and [`Context::export_fd`] work.
    external_memory_fd: bool,
    /// `CENOTE_PIPELINE_STATS` was set *and* the device supports
    /// `VK_KHR_pipeline_executable_properties`, so pipelines are created
    /// with statistics capture and log what the driver reports. The single
    /// source of truth — pipeline creation must not re-read the environment,
    /// because capture without the feature enabled is invalid usage.
    pipeline_stats: bool,
    summary: String,
    debug: Option<DebugMessenger>,
    instance: ash::Instance,
    // Read by presenter surface creation; must outlive `instance` regardless,
    // since dropping the Entry unloads libvulkan.
    entry: ash::Entry,
}

impl Context {
    /// Bring up Vulkan: load the loader, create an instance (with the
    /// Khronos validation layer and a `log`-routed debug messenger in debug
    /// builds), select the most capable physical device, and create the
    /// device, compute queue, and allocator.
    ///
    /// Device preference: discrete > integrated > everything else. A device
    /// qualifies only if it offers Vulkan 1.3, the ray-tracing extensions,
    /// and the feature baseline (ray query, acceleration structures, buffer
    /// device address, descriptor indexing) — `init.rs` holds the exact
    /// lists.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Loader`] if libvulkan is missing,
    /// [`crate::Error::NoCapableGpu`] with a per-device report if nothing
    /// qualifies, [`crate::Error::Vulkan`] / [`crate::Error::Allocation`] if
    /// bring-up calls fail.
    pub fn new() -> Result<Self> {
        Self::bring_up(None)
    }

    /// As [`Context::new`], but present-capable: enables the instance
    /// extensions for `display`'s windowing protocol plus `VK_KHR_swapchain`
    /// on the device, and requires them during device selection. Pair with
    /// [`Context::create_presenter`] once a window exists.
    ///
    /// # Errors
    ///
    /// As [`Context::new`], plus [`crate::Error::Vulkan`] if Vulkan cannot
    /// present on this platform's display protocol.
    pub fn presentable(display: RawDisplayHandle) -> Result<Self> {
        Self::bring_up(Some(display))
    }

    fn bring_up(display: Option<RawDisplayHandle>) -> Result<Self> {
        let entry = unsafe { ash::Entry::load() }?;
        let (instance, debug_utils_enabled) = init::create_instance(&entry, display)?;
        // From here on, failure must unwind what the constructor built so far.
        match Self::init_with_instance(&entry, &instance, debug_utils_enabled, display.is_some()) {
            Ok(context) => Ok(context),
            Err(err) => {
                unsafe { instance.destroy_instance(None) };
                Err(err)
            }
        }
    }

    fn init_with_instance(
        entry: &ash::Entry,
        instance: &ash::Instance,
        debug_utils_enabled: bool,
        presentable: bool,
    ) -> Result<Self> {
        let debug = debug_utils_enabled
            .then(|| init::create_debug_messenger(entry, instance))
            .transpose()?;

        let (physical_device, properties) = init::select_physical_device(instance, presentable)?;
        let queue_family_index = init::compute_queue_family(instance, physical_device)
            .expect("selection already verified a compute queue family");
        // How well this queue can tell the time — read once here so the
        // timing module never has to reach back to the instance.
        let timestamp_valid_bits = unsafe {
            instance.get_physical_device_queue_family_properties(physical_device)
                [queue_family_index as usize]
                .timestamp_valid_bits
        };
        let device_local_heap = ledger::device_local_heap(instance, physical_device);
        let summary = init::describe_device(instance, physical_device, &properties);
        log::info!("selected {summary}");

        let (device, pipeline_stats, external_memory_fd) =
            init::create_device(instance, physical_device, queue_family_index, presentable)?;
        let accel_loader = ash::khr::acceleration_structure::Device::new(instance, &device);
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
            buffer_device_address: true,
            allocation_sizes: gpu_allocator::AllocationSizes::default(),
        });
        let allocator = match allocator {
            Ok(allocator) => allocator,
            Err(err) => {
                unsafe { device.destroy_device(None) };
                return Err(err.into());
            }
        };

        Ok(Self {
            allocator: ManuallyDrop::new(Arc::new(Mutex::new(allocator))),
            ledger: Arc::default(),
            device,
            accel_loader,
            queue: Queue::new(queue),
            queue_family_index,
            physical_device,
            device_type: properties.device_type,
            device_uuid: init::device_uuid(instance, physical_device),
            scratch_alignment: accel::scratch_alignment(instance, physical_device),
            timestamp_period: properties.limits.timestamp_period,
            timestamp_valid_bits,
            device_local_heap,
            presentable,
            external_memory_fd,
            pipeline_stats,
            summary,
            debug,
            instance: instance.clone(),
            entry: entry.clone(),
        })
    }

    /// One-line human-readable description of the selected device
    /// (name, type, driver, Vulkan version).
    #[must_use]
    pub fn device_summary(&self) -> &str {
        &self.summary
    }

    /// The selected device's hardware class.
    #[must_use]
    pub fn device_type(&self) -> vk::PhysicalDeviceType {
        self.device_type
    }

    /// The selected device's UUID, as every API on the machine spells it —
    /// what [`crate::denoise::Denoiser::new`] hands OIDN so both land on
    /// one GPU. Not a raw handle, so it crosses the `gpu` boundary.
    #[must_use]
    pub fn device_uuid(&self) -> [u8; 16] {
        self.device_uuid
    }

    /// Whether this device can share buffer memory with another API through
    /// a POSIX fd (`VK_KHR_external_memory_fd`) — the denoiser's zero-copy
    /// path. False on drivers without it, and everywhere fds are not how
    /// memory is shared.
    #[must_use]
    pub fn external_memory_fd(&self) -> bool {
        self.external_memory_fd
    }

    // The raw-handle accessors below are `pub(super)`: the quarantine —
    // code outside `gpu` never touches raw `vk` handles — is enforced by
    // the compiler, not by convention.

    /// The logical device. Handles derived from it must not outlive `self`.
    #[must_use]
    pub(super) fn device(&self) -> &ash::Device {
        &self.device
    }

    /// Family index the queue belongs to.
    #[must_use]
    pub(super) fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }

    /// The selected physical device.
    #[must_use]
    pub(super) fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    /// The instance, for extension loaders and physical-device queries.
    #[must_use]
    pub(super) fn instance(&self) -> &ash::Instance {
        &self.instance
    }

    /// A clone of the shared allocator handle, for resources that free
    /// themselves on drop.
    fn allocator_handle(&self) -> Arc<Mutex<Allocator>> {
        Arc::clone(&self.allocator)
    }

    /// A clone of the shared queue handle, for the presenter's own
    /// submissions.
    fn queue_handle(&self) -> Queue {
        self.queue.clone()
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            if Arc::strong_count(&self.allocator) > 1 {
                log::error!("GPU resources outlive their Context — teardown order is now wrong");
            }
            // The allocator frees device memory, so it goes first.
            ManuallyDrop::drop(&mut self.allocator);
            self.device.destroy_device(None);
            // Messenger (via its own Drop) strictly before the instance.
            drop(self.debug.take());
            self.instance.destroy_instance(None);
        }
    }
}

/// GPU-gated test entry point: `Some(context)` on machines with a capable
/// GPU, `None` (test passes vacuously, with a note on stderr) everywhere
/// else, so plain `cargo test` works on GPU-less CI.
#[cfg(test)]
pub(crate) fn test_context() -> Option<Context> {
    // Surface validation-messenger output in tests: run e.g.
    // `RUST_LOG=warn cargo test -- --nocapture` to see it.
    let _ = env_logger::builder().is_test(true).try_init();
    match Context::new() {
        Ok(context) => Some(context),
        Err(err) => {
            eprintln!("skipping: no capable GPU here ({err})");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Selection must never pick a software rasterizer. Not vacuous: Mesa's
    /// lavapipe implements ray query and passes every capability check, so
    /// only the explicit device-type rejection keeps it out. Skips cleanly
    /// where bring-up fails entirely, e.g. GPU-less CI.
    #[test]
    fn selection_rejects_software_devices() {
        let Some(context) = test_context() else {
            return;
        };
        assert_ne!(context.device_type(), vk::PhysicalDeviceType::CPU);
        assert!(!context.device_summary().contains("llvmpipe"));
    }
}
