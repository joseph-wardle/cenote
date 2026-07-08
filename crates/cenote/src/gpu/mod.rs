//! The unsafe-Vulkan quarantine.
//!
//! [`Context`] owns instance→device bring-up: validation wiring, physical
//! device selection against the ray-tracing baseline, one compute queue, and
//! the memory allocator. Code outside `gpu` never touches raw `vk` handles
//! or writes `unsafe`. Bring-up lives in `init`; buffers, one-shot submits,
//! compute pipelines, and acceleration structures in the other submodules.
//!
//! There is no backend abstraction here and there never will be — Cenote is
//! single-backend by design: a reader who knows Vulkan should be reading
//! Vulkan.

use std::mem::ManuallyDrop;
use std::sync::{Arc, Mutex};

use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use crate::error::Result;

mod accel;
mod buffer;
mod init;
mod pipeline;
mod submit;

pub use accel::{AccelerationStructure, TlasInstance};
pub use buffer::{Buffer, MemoryLocation};
pub use pipeline::ComputePipeline;

use init::DebugMessenger;

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
    device: ash::Device,
    // Extension function table for VK_KHR_acceleration_structure; plain
    // function pointers, nothing to destroy.
    accel_loader: ash::khr::acceleration_structure::Device,
    queue: vk::Queue,
    queue_family_index: u32,
    physical_device: vk::PhysicalDevice,
    device_type: vk::PhysicalDeviceType,
    summary: String,
    debug: Option<DebugMessenger>,
    instance: ash::Instance,
    // Never read, but must outlive `instance`: dropping the Entry unloads
    // libvulkan.
    _entry: ash::Entry,
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
        let entry = unsafe { ash::Entry::load() }?;
        let (instance, debug_utils_enabled) = init::create_instance(&entry)?;
        // From here on, failure must unwind what the constructor built so far.
        match Self::init_with_instance(&entry, &instance, debug_utils_enabled) {
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
    ) -> Result<Self> {
        let debug = debug_utils_enabled
            .then(|| init::create_debug_messenger(entry, instance))
            .transpose()?;

        let (physical_device, properties) = init::select_physical_device(instance)?;
        let queue_family_index = init::compute_queue_family(instance, physical_device)
            .expect("selection already verified a compute queue family");
        let summary = init::describe_device(instance, physical_device, &properties);
        log::info!("selected {summary}");

        let device = init::create_device(instance, physical_device, queue_family_index)?;
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
            device,
            accel_loader,
            queue,
            queue_family_index,
            physical_device,
            device_type: properties.device_type,
            summary,
            debug,
            instance: instance.clone(),
            _entry: entry.clone(),
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

    /// The logical device. Handles derived from it must not outlive `self`.
    #[must_use]
    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    /// The one compute queue — every submission in the crate goes through it.
    #[must_use]
    pub fn queue(&self) -> vk::Queue {
        self.queue
    }

    /// Family index [`Self::queue`] belongs to.
    #[must_use]
    pub fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }

    /// The selected physical device.
    #[must_use]
    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    /// A clone of the shared allocator handle, for resources that free
    /// themselves on drop.
    fn allocator_handle(&self) -> Arc<Mutex<Allocator>> {
        Arc::clone(&self.allocator)
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
