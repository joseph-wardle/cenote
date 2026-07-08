//! Instance and device bring-up: validation wiring, selection of the most
//! capable ray-tracing GPU, and logical-device creation.
//!
//! Selection and creation must agree: [`missing_requirements`] verifies
//! exactly the extensions and features [`create_device`] enables. When one
//! list grows, grow the other.

use std::ffi::{CStr, c_void};

use ash::vk;

use crate::error::{Error, Result};

/// Baseline API version. Vulkan 1.3 makes `synchronization2` mandatory and
/// carries descriptor indexing + buffer device address in core, so the
/// extension list below stays short.
const MIN_API_VERSION: u32 = vk::API_VERSION_1_3;

/// Device extensions every capable GPU must offer. Descriptor indexing and
/// buffer device address are core-1.3 *features*, checked separately in
/// [`missing_requirements`].
const REQUIRED_EXTENSIONS: &[&CStr] = &[
    ash::khr::acceleration_structure::NAME,
    // Required by VK_KHR_acceleration_structure even though we never build
    // acceleration structures on the host.
    ash::khr::deferred_host_operations::NAME,
    ash::khr::ray_query::NAME,
];

pub(super) struct DebugMessenger {
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

/// Create the instance. Debug builds add the Khronos validation layer and
/// the debug-utils extension when present — their absence is logged, never
/// fatal. The returned flag reports whether debug-utils was enabled, i.e.
/// whether [`create_debug_messenger`] may be called.
pub(super) fn create_instance(entry: &ash::Entry) -> Result<(ash::Instance, bool)> {
    let app_info = vk::ApplicationInfo::default()
        .application_name(c"cenote")
        .api_version(MIN_API_VERSION);

    let mut layers: Vec<*const i8> = Vec::new();
    let mut extensions: Vec<*const i8> = Vec::new();
    let mut debug_utils_enabled = false;
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
            debug_utils_enabled = true;
        }
    }

    let create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions);
    let instance = unsafe { entry.create_instance(&create_info, None)? };
    Ok((instance, debug_utils_enabled))
}

/// Create the messenger that routes validation output into `log`. Only call
/// when [`create_instance`] enabled `VK_EXT_debug_utils`.
pub(super) fn create_debug_messenger(
    entry: &ash::Entry,
    instance: &ash::Instance,
) -> Result<DebugMessenger> {
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
    Ok(DebugMessenger { loader, messenger })
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
pub(super) fn select_physical_device(
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

/// Everything the selected device must have, reported as a list of
/// human-readable lacks — empty means capable. Verifies exactly what
/// [`create_device`] enables.
fn missing_requirements(
    instance: &ash::Instance,
    device: vk::PhysicalDevice,
    properties: &vk::PhysicalDeviceProperties,
) -> Vec<String> {
    let mut missing = Vec::new();

    // Software rasterizers (lavapipe) genuinely implement ray query these
    // days, so capability checks alone don't exclude them — but a renderer
    // betting on extreme single-GPU performance must not silently "work" on
    // one: reject by device type.
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

    // One entry per feature `create_device` enables — keep in lockstep.
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

pub(super) fn compute_queue_family(
    instance: &ash::Instance,
    device: vk::PhysicalDevice,
) -> Option<u32> {
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

/// Create the logical device with one compute queue.
pub(super) fn create_device(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
) -> Result<ash::Device> {
    let priorities = [1.0_f32];
    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&priorities);

    // Enable exactly the baseline `missing_requirements` verified — keep in
    // lockstep. Descriptor indexing is enabled-but-unused until the bindless
    // texture table arrives (M2).
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

/// One-line human-readable description of a device (name, type, driver,
/// Vulkan version).
pub(super) fn describe_device(
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
