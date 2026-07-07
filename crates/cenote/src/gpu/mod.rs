//! The unsafe-Vulkan quarantine (decision D-005).
//!
//! [`Context`] owns instance→device bring-up: validation wiring, physical
//! device selection against the ray-tracing baseline, one compute queue, and
//! the memory allocator. Code outside `gpu` never touches raw `vk` handles
//! or writes `unsafe`. Buffers, one-shot submits, compute pipelines, and
//! acceleration structures live in submodules.
//!
//! There is no backend abstraction here and there never will be (charter
//! non-goal): a reader who knows Vulkan should be reading Vulkan.

use std::ffi::{CStr, c_void};
use std::mem::ManuallyDrop;
use std::sync::{Arc, Mutex};

use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use crate::error::{Error, Result};

mod accel;
mod buffer;
mod pipeline;
mod submit;

pub use accel::{AccelerationStructure, TlasInstance};
pub use buffer::{Buffer, MemoryLocation};
pub use pipeline::ComputePipeline;

/// Baseline API version. Vulkan 1.3 makes `synchronization2` mandatory and
/// carries descriptor indexing + buffer device address in core, so the
/// extension list below stays short.
const MIN_API_VERSION: u32 = vk::API_VERSION_1_3;

/// Device extensions every capable GPU must offer (D-015). Descriptor
/// indexing and buffer device address are core-1.3 *features*, checked
/// separately in [`missing_requirements`].
const REQUIRED_EXTENSIONS: &[&CStr] = &[
    ash::khr::acceleration_structure::NAME,
    // Required by VK_KHR_acceleration_structure even though we never build
    // acceleration structures on the host.
    ash::khr::deferred_host_operations::NAME,
    ash::khr::ray_query::NAME,
];

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
    accel: ash::khr::acceleration_structure::Device,
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

struct DebugMessenger {
    loader: ash::ext::debug_utils::Instance,
    messenger: vk::DebugUtilsMessengerEXT,
}

impl Drop for DebugMessenger {
    // Self-destroying so every exit path — constructor unwind included —
    // tears it down before the instance (VUID-vkDestroyInstance-00629).
    fn drop(&mut self) {
        unsafe {
            self.loader
                .destroy_debug_utils_messenger(self.messenger, None);
        }
    }
}

impl Context {
    /// Bring up Vulkan: load the loader, create an instance (with the
    /// Khronos validation layer and a `log`-routed debug messenger in debug
    /// builds, per D-015), select the most capable physical device, and
    /// create the device, compute queue, and allocator.
    ///
    /// Device preference: discrete > integrated > everything else. A device
    /// qualifies only if it meets [`MIN_API_VERSION`], [`REQUIRED_EXTENSIONS`],
    /// and the feature baseline (ray query, acceleration structures, buffer
    /// device address, descriptor indexing).
    ///
    /// # Errors
    ///
    /// [`Error::Loader`] if libvulkan is missing, [`Error::NoCapableGpu`]
    /// with a per-device report if nothing qualifies, [`Error::Vulkan`] /
    /// [`Error::Allocation`] if bring-up calls fail.
    pub fn new() -> Result<Self> {
        let entry = unsafe { ash::Entry::load() }?;
        let instance = create_instance(&entry)?;
        // From here on, failure must unwind what the constructor built so far.
        match Self::init_with_instance(&entry, &instance) {
            Ok(context) => Ok(context),
            Err(err) => {
                unsafe { instance.destroy_instance(None) };
                Err(err)
            }
        }
    }

    fn init_with_instance(entry: &ash::Entry, instance: &ash::Instance) -> Result<Self> {
        let debug = create_debug_messenger(entry, instance)?;

        let (physical_device, properties) = select_physical_device(instance)?;
        let queue_family_index = compute_queue_family(instance, physical_device)
            .expect("selection already verified a compute queue family");
        let summary = describe_device(instance, physical_device, &properties);
        log::info!("selected {summary}");

        let device = create_device(instance, physical_device, queue_family_index)?;
        let accel = ash::khr::acceleration_structure::Device::new(instance, &device);
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
            accel,
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

    /// The one compute queue (decision D-007: single queue, blocking submits).
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

fn create_instance(entry: &ash::Entry) -> Result<ash::Instance> {
    let app_info = vk::ApplicationInfo::default()
        .application_name(c"cenote")
        .api_version(MIN_API_VERSION);

    // Validation is a debug-build concern (D-015). Both the layer and the
    // debug-utils extension are optional at run time: their absence is
    // logged, never fatal.
    let mut layers: Vec<*const i8> = Vec::new();
    let mut extensions: Vec<*const i8> = Vec::new();
    if cfg!(debug_assertions) {
        const VALIDATION: &CStr = c"VK_LAYER_KHRONOS_validation";
        let available = unsafe { entry.enumerate_instance_layer_properties()? };
        if available
            .iter()
            .any(|l| l.layer_name_as_c_str() == Ok(VALIDATION))
        {
            layers.push(VALIDATION.as_ptr());
        } else {
            log::warn!("debug build but {VALIDATION:?} not installed — validation off");
        }

        let available = unsafe { entry.enumerate_instance_extension_properties(None)? };
        if available
            .iter()
            .any(|e| e.extension_name_as_c_str() == Ok(ash::ext::debug_utils::NAME))
        {
            extensions.push(ash::ext::debug_utils::NAME.as_ptr());
        }
    }

    let create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions);
    Ok(unsafe { entry.create_instance(&create_info, None)? })
}

/// Create the debug messenger when the instance was created with
/// `VK_EXT_debug_utils` (debug builds only). Returns `None` when the
/// extension is absent, which `create_instance` already logged.
fn create_debug_messenger(
    entry: &ash::Entry,
    instance: &ash::Instance,
) -> Result<Option<DebugMessenger>> {
    if !cfg!(debug_assertions) {
        return Ok(None);
    }
    let available = unsafe { entry.enumerate_instance_extension_properties(None)? };
    if !available
        .iter()
        .any(|e| e.extension_name_as_c_str() == Ok(ash::ext::debug_utils::NAME))
    {
        return Ok(None);
    }

    let loader = ash::ext::debug_utils::Instance::new(entry, instance);
    let create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(debug_callback));
    let messenger = unsafe { loader.create_debug_utils_messenger(&create_info, None)? };
    Ok(Some(DebugMessenger { loader, messenger }))
}

/// Route validation messages into `log` so they interleave with our own
/// output and obey `RUST_LOG` filtering.
unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut c_void,
) -> vk::Bool32 {
    let message = unsafe {
        (*data)
            .message_as_c_str()
            .map_or_else(|| "<no message>".into(), CStr::to_string_lossy)
    };
    match severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => log::error!("[vulkan] {message}"),
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => log::warn!("[vulkan] {message}"),
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => log::debug!("[vulkan] {message}"),
        _ => log::trace!("[vulkan] {message}"),
    }
    vk::FALSE
}

/// Pick the most capable physical device, or report why every device was
/// rejected. Preference among capable devices: discrete > integrated > rest.
fn select_physical_device(
    instance: &ash::Instance,
) -> Result<(vk::PhysicalDevice, vk::PhysicalDeviceProperties)> {
    let mut rejections = String::new();
    let mut best: Option<(u32, vk::PhysicalDevice, vk::PhysicalDeviceProperties)> = None;

    for device in unsafe { instance.enumerate_physical_devices()? } {
        let properties = unsafe { instance.get_physical_device_properties(device) };
        let name = properties
            .device_name_as_c_str()
            .map_or_else(|_| "<unnamed>".into(), CStr::to_string_lossy);

        let missing = missing_requirements(instance, device, &properties);
        if missing.is_empty() {
            log::debug!("{name}: capable");
            let rank = device_type_rank(properties.device_type);
            if best
                .as_ref()
                .is_none_or(|(best_rank, ..)| rank < *best_rank)
            {
                best = Some((rank, device, properties));
            }
        } else {
            use std::fmt::Write;
            log::debug!("{name}: missing {}", missing.join(", "));
            let _ = writeln!(rejections, "  {name}: missing {}", missing.join(", "));
        }
    }

    match best {
        Some((_, device, properties)) => Ok((device, properties)),
        None if rejections.is_empty() => Err(Error::NoCapableGpu(
            "  no Vulkan devices enumerated\n".into(),
        )),
        None => Err(Error::NoCapableGpu(rejections)),
    }
}

/// Everything the selected device must have (D-015), reported as a list of
/// human-readable lacks — empty means capable.
fn missing_requirements(
    instance: &ash::Instance,
    device: vk::PhysicalDevice,
    properties: &vk::PhysicalDeviceProperties,
) -> Vec<String> {
    let mut missing = Vec::new();

    // Software rasterizers (lavapipe) genuinely implement ray query these
    // days, so capability checks alone don't exclude them — but a software
    // path tracer is out of identity (D-016): reject by device type.
    if properties.device_type == vk::PhysicalDeviceType::CPU {
        missing.push("hardware (software rasterizers are rejected)".to_owned());
    }

    if properties.api_version < MIN_API_VERSION {
        missing.push(format!(
            "Vulkan 1.3 (has {}.{})",
            vk::api_version_major(properties.api_version),
            vk::api_version_minor(properties.api_version)
        ));
        // Feature structs below assume a 1.2+ device; don't query them.
        return missing;
    }

    let extensions =
        unsafe { instance.enumerate_device_extension_properties(device) }.unwrap_or_default();
    let has_extension = |name: &CStr| {
        extensions
            .iter()
            .any(|e| e.extension_name_as_c_str() == Ok(name))
    };
    for &required in REQUIRED_EXTENSIONS {
        if !has_extension(required) {
            missing.push(required.to_string_lossy().into_owned());
        }
    }

    // Only chain feature structs whose extension exists — querying features
    // of an unsupported extension is invalid usage.
    let mut vk12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut accel = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default();
    let mut ray_query = vk::PhysicalDeviceRayQueryFeaturesKHR::default();
    let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut vk12);
    if has_extension(ash::khr::acceleration_structure::NAME) {
        features = features.push_next(&mut accel);
    }
    if has_extension(ash::khr::ray_query::NAME) {
        features = features.push_next(&mut ray_query);
    }
    unsafe { instance.get_physical_device_features2(device, &mut features) };

    let feature_checks = [
        (accel.acceleration_structure, "accelerationStructure"),
        (ray_query.ray_query, "rayQuery"),
        (vk12.buffer_device_address, "bufferDeviceAddress"),
        (vk12.descriptor_indexing, "descriptorIndexing"),
        (vk12.runtime_descriptor_array, "runtimeDescriptorArray"),
        (
            vk12.descriptor_binding_partially_bound,
            "descriptorBindingPartiallyBound",
        ),
        (
            vk12.descriptor_binding_variable_descriptor_count,
            "descriptorBindingVariableDescriptorCount",
        ),
        (
            vk12.descriptor_binding_sampled_image_update_after_bind,
            "descriptorBindingSampledImageUpdateAfterBind",
        ),
        (
            vk12.shader_sampled_image_array_non_uniform_indexing,
            "shaderSampledImageArrayNonUniformIndexing",
        ),
    ];
    for (supported, name) in feature_checks {
        if supported != vk::TRUE {
            missing.push(name.to_owned());
        }
    }

    if compute_queue_family(instance, device).is_none() {
        missing.push("a compute queue family".to_owned());
    }

    missing
}

fn compute_queue_family(instance: &ash::Instance, device: vk::PhysicalDevice) -> Option<u32> {
    let families = unsafe { instance.get_physical_device_queue_family_properties(device) };
    families
        .iter()
        .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
        .map(|index| index as u32)
}

fn device_type_rank(device_type: vk::PhysicalDeviceType) -> u32 {
    match device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => 0,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
        _ => 3,
    }
}

fn create_device(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
) -> Result<ash::Device> {
    let priorities = [1.0_f32];
    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&priorities);

    // Enable exactly the baseline that selection verified. Descriptor
    // indexing is enabled-but-unused until the bindless texture table
    // arrives in M2 (decision D-006).
    let mut vk12 = vk::PhysicalDeviceVulkan12Features::default()
        .buffer_device_address(true)
        .descriptor_indexing(true)
        .runtime_descriptor_array(true)
        .descriptor_binding_partially_bound(true)
        .descriptor_binding_variable_descriptor_count(true)
        .descriptor_binding_sampled_image_update_after_bind(true)
        .shader_sampled_image_array_non_uniform_indexing(true);
    let mut vk13 = vk::PhysicalDeviceVulkan13Features::default().synchronization2(true);
    let mut accel =
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default().acceleration_structure(true);
    let mut ray_query = vk::PhysicalDeviceRayQueryFeaturesKHR::default().ray_query(true);
    let mut features = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut vk12)
        .push_next(&mut vk13)
        .push_next(&mut accel)
        .push_next(&mut ray_query);

    let extensions: Vec<*const i8> = REQUIRED_EXTENSIONS.iter().map(|e| e.as_ptr()).collect();
    let create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(std::slice::from_ref(&queue_info))
        .enabled_extension_names(&extensions)
        .push_next(&mut features);

    Ok(unsafe { instance.create_device(physical_device, &create_info, None)? })
}

fn describe_device(
    instance: &ash::Instance,
    device: vk::PhysicalDevice,
    properties: &vk::PhysicalDeviceProperties,
) -> String {
    let name = properties
        .device_name_as_c_str()
        .map_or_else(|_| "<unnamed>".into(), CStr::to_string_lossy);
    let device_type = match properties.device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => "discrete GPU",
        vk::PhysicalDeviceType::INTEGRATED_GPU => "integrated GPU",
        vk::PhysicalDeviceType::VIRTUAL_GPU => "virtual GPU",
        vk::PhysicalDeviceType::CPU => "CPU",
        _ => "other",
    };

    // Driver name/info strings are core 1.2, which selection guaranteed.
    let mut driver = vk::PhysicalDeviceDriverProperties::default();
    let mut properties2 = vk::PhysicalDeviceProperties2::default().push_next(&mut driver);
    unsafe { instance.get_physical_device_properties2(device, &mut properties2) };
    let driver_name = driver
        .driver_name_as_c_str()
        .map_or_else(|_| "unknown driver".into(), CStr::to_string_lossy);
    let driver_info = driver
        .driver_info_as_c_str()
        .map_or_else(|_| "".into(), CStr::to_string_lossy);

    format!(
        "{name} ({device_type}, {driver_name} {driver_info}, Vulkan {}.{})",
        vk::api_version_major(properties.api_version),
        vk::api_version_minor(properties.api_version),
    )
}

/// GPU-gated test entry point (decision D-009): `Some(context)` on machines
/// with a capable GPU, `None` (test passes vacuously, with a note on stderr)
/// everywhere else, so plain `cargo test` works on GPU-less CI.
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
    /// only the explicit device-type rejection (D-016) keeps it out. Skips
    /// cleanly where bring-up fails entirely, e.g. GPU-less CI (D-009).
    #[test]
    fn selection_rejects_software_devices() {
        let Some(context) = test_context() else {
            return;
        };
        assert_ne!(context.device_type(), vk::PhysicalDeviceType::CPU);
        assert!(!context.device_summary().contains("llvmpipe"));
    }
}
