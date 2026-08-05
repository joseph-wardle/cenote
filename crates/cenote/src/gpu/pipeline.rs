//! Compute pipelines and their descriptor sets.
//!
//! A [`ComputePipeline`] is one kernel compiled from SPIR-V plus its layout
//! and — for kernels that read the scene's TLAS or its textures — the one
//! descriptor set that carries them. Kernels reach every buffer through
//! device addresses in a single push-constant struct; the only resources
//! that cannot be addresses are the scene TLAS and the sampled images
//! (filtered reads need real descriptors), so a kernel that touches any
//! declares [`Bindings::Scene`] and carries set 0 — binding 0 the TLAS,
//! binding 1 the environment, and binding 2 the bindless material-texture
//! array (partially bound: a scene binds only as many as it holds) —
//! written at submission time. Kernels that only chew buffers
//! ([`Bindings::None`]) have no descriptors at all.
//!
//! Running a pipeline lives next door in `submit.rs`: [`Context::dispatch`]
//! for one, [`Context::submit_passes`] for a wave's stage chain.

use std::ffi::CStr;
use std::slice;

use ash::vk;

use crate::error::Result;
use crate::gpu::{AccelerationStructure, Context, SampledImage};

/// The descriptor bindings a kernel needs. Buffers travel as device
/// addresses in push constants, so the only question is whether the kernel
/// touches the resources that must be descriptors — the TLAS, the
/// environment texture, the bindless material-texture array, and the
/// closure's lookup tables. One shared layout for all of them keeps the
/// binding model a single small set; a kernel that statically uses only
/// one is fine, Vulkan only requires that what it *uses* is bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bindings {
    /// Push constants only — no descriptor set.
    None,
    /// Set 0 — binding 0: the scene TLAS; binding 1: the environment
    /// texture; binding 2: the bindless material-texture array; bindings
    /// 3 and 4: the closure's 2D and 3D lookup tables. All written at
    /// dispatch time.
    Scene,
}

/// Capacity of the bindless texture array (binding 2). A fixed layout-time
/// bound — scenes bind only what they hold, the rest stays unwritten
/// under `PARTIALLY_BOUND` — sized far above any corpus scene while
/// staying well inside every ray-tracing GPU's per-stage sampled-image
/// limit.
pub const MAX_SCENE_TEXTURES: u32 = 1024;

/// Entries in the closure's 2D lookup-table array (binding 3) — the length
/// `tables.rs` declares its list at, so the two cannot drift. Every slot is
/// always written, hence no partial binding.
pub const TABLE_PLANES: u32 = 7;

/// Entries in the closure's 3D lookup-table array (binding 4), on the
/// same terms as [`TABLE_PLANES`].
pub const TABLE_VOLUMES: u32 = 3;

/// The scene resources a [`Bindings::Scene`] dispatch binds.
#[derive(Clone, Copy)]
pub struct SceneBindings<'a> {
    /// The scene TLAS (binding 0).
    pub tlas: &'a AccelerationStructure,
    /// The environment texture (binding 1).
    pub environment: &'a SampledImage,
    /// The material textures (binding 2), in bindless-index order — the
    /// order material records index. At most [`MAX_SCENE_TEXTURES`].
    pub textures: &'a [vk::DescriptorImageInfo],
    /// The 2D lookup tables (binding 3), [`TABLE_PLANES`] of them.
    pub table_planes: &'a [vk::DescriptorImageInfo],
    /// The 3D lookup tables (binding 4), [`TABLE_VOLUMES`] of them.
    pub table_volumes: &'a [vk::DescriptorImageInfo],
}

/// A compute pipeline plus its layout and (for scene-resource kernels) its
/// descriptor set, destroyed on drop (before the [`Context`], like every
/// `gpu` resource).
pub struct ComputePipeline {
    // `pub(super)`: the dispatch and pass-recording machinery in `submit.rs`
    // reads these, still inside the `gpu` quarantine.
    pub(super) handle: vk::Pipeline,
    pub(super) layout: vk::PipelineLayout,
    /// Present iff created with [`Bindings::Scene`].
    pub(super) scene: Option<SceneDescriptors>,
    pub(super) push_constant_size: u32,
    /// The kernel's entry-point name, which is also the name it reports
    /// itself under in [`crate::stats`] — see [`crate::gpu::Pass::label`].
    pub(super) label: &'static str,
    device: ash::Device,
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.handle, None);
            self.device.destroy_pipeline_layout(self.layout, None);
            if let Some(scene) = &self.scene {
                scene.destroy(&self.device);
            }
        }
    }
}

/// The scene descriptor set under construction: layout, pool, and the one
/// allocated set. Plain handles — ownership passes to the [`ComputePipeline`]
/// on success, to [`SceneDescriptors::destroy`] on failure.
pub(super) struct SceneDescriptors {
    set_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    /// Bound at dispatch time by `submit.rs`.
    pub(super) set: vk::DescriptorSet,
}

impl SceneDescriptors {
    /// Tear down after a failed pipeline build. The set itself is
    /// pool-allocated: destroying the pool frees it.
    unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

impl Context {
    /// Create a compute pipeline from SPIR-V bytes (embedded or hot-reloaded
    /// — both are `slangc` output). `entry` names the kernel entry point;
    /// `push_constant_size` is the byte size of the kernel's push-constant
    /// struct, enforced again at dispatch time; `bindings` says whether the
    /// kernel touches the scene's descriptor resources (TLAS, environment).
    ///
    /// `entry` is `'static` because it doubles as the pipeline's timing
    /// label ([`crate::gpu::Pass::label`]) — every caller already hands over
    /// a literal or a [`crate::shaders::Kernel`]'s own name.
    pub fn create_compute_pipeline(
        &self,
        spirv: &[u8],
        entry: &'static CStr,
        push_constant_size: u32,
        bindings: Bindings,
    ) -> Result<ComputePipeline> {
        assert!(
            push_constant_size > 0 && push_constant_size.is_multiple_of(4),
            "push-constant size must be a non-zero multiple of 4"
        );
        let words = ash::util::read_spv(&mut std::io::Cursor::new(spirv))
            .expect("kernel bytes are not valid SPIR-V");

        let device = self.device();
        let module_info = vk::ShaderModuleCreateInfo::default().code(&words);
        let module = unsafe { device.create_shader_module(&module_info, None)? };

        // The module is only an input to pipeline creation — destroyed on
        // success and failure alike.
        let label = entry.to_str().expect("kernel entry name is not UTF-8");
        let result = self
            .create_descriptors_and_pipeline(module, entry, push_constant_size, bindings, label);
        unsafe { device.destroy_shader_module(module, None) };
        result
    }

    fn create_descriptors_and_pipeline(
        &self,
        module: vk::ShaderModule,
        entry: &CStr,
        push_constant_size: u32,
        bindings: Bindings,
        label: &'static str,
    ) -> Result<ComputePipeline> {
        let scene = match bindings {
            Bindings::None => None,
            Bindings::Scene => Some(self.create_scene_descriptors()?),
        };
        match self.create_layout_and_pipeline(module, entry, push_constant_size, scene.as_ref()) {
            Ok((handle, layout)) => Ok(ComputePipeline {
                handle,
                layout,
                scene,
                push_constant_size,
                label,
                device: self.device().clone(),
            }),
            Err(err) => {
                if let Some(scene) = &scene {
                    unsafe { scene.destroy(self.device()) };
                }
                Err(err)
            }
        }
    }

    /// Create the binding model's single descriptor set: binding 0 = the
    /// scene TLAS, binding 1 = the environment texture, binding 2 = the
    /// bindless texture array, bindings 3 and 4 = the closure's 2D and 3D
    /// lookup tables. Contents are written at dispatch time; the texture
    /// array is partially bound — a scene writes only the slots it holds,
    /// and kernels index nothing past them.
    fn create_scene_descriptors(&self) -> Result<SceneDescriptors> {
        let device = self.device();
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(MAX_SCENE_TEXTURES)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(TABLE_PLANES)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(TABLE_VOLUMES)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let binding_flags = [
            vk::DescriptorBindingFlags::empty(),
            vk::DescriptorBindingFlags::empty(),
            vk::DescriptorBindingFlags::PARTIALLY_BOUND,
            vk::DescriptorBindingFlags::empty(),
            vk::DescriptorBindingFlags::empty(),
        ];
        let mut flags_info =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&bindings)
            .push_next(&mut flags_info);
        let set_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None)? };

        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1 + MAX_SCENE_TEXTURES + TABLE_PLANES + TABLE_VOLUMES),
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        let pool = unsafe { device.create_descriptor_pool(&pool_info, None) };
        let pool = match pool {
            Ok(pool) => pool,
            Err(err) => {
                unsafe { device.destroy_descriptor_set_layout(set_layout, None) };
                return Err(err.into());
            }
        };

        // `destroy` only touches the pool and layout, so the struct — and
        // its cleanup — can exist before the set does.
        let mut descriptors = SceneDescriptors {
            set_layout,
            pool,
            set: vk::DescriptorSet::null(),
        };
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(slice::from_ref(&set_layout));
        match unsafe { device.allocate_descriptor_sets(&alloc_info) } {
            Ok(sets) => {
                descriptors.set = sets[0];
                Ok(descriptors)
            }
            Err(err) => {
                unsafe { descriptors.destroy(device) };
                Err(err.into())
            }
        }
    }

    fn create_layout_and_pipeline(
        &self,
        module: vk::ShaderModule,
        entry: &CStr,
        push_constant_size: u32,
        descriptors: Option<&SceneDescriptors>,
    ) -> Result<(vk::Pipeline, vk::PipelineLayout)> {
        let device = self.device();
        let range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(push_constant_size);
        let set_layouts = descriptors.map_or(&[][..], |scene| slice::from_ref(&scene.set_layout));
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(set_layouts)
            .push_constant_ranges(slice::from_ref(&range));
        let layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(entry);
        // Set when init.rs enabled pipelineExecutableInfo; the capture flag
        // is invalid usage without it, hence the flag and not the env var.
        let mut info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);
        if self.pipeline_stats {
            info = info.flags(vk::PipelineCreateFlags::CAPTURE_STATISTICS_KHR);
        }
        let pipelines = unsafe {
            device.create_compute_pipelines(vk::PipelineCache::null(), slice::from_ref(&info), None)
        };

        match pipelines {
            Ok(pipelines) => {
                if self.pipeline_stats {
                    self.log_pipeline_statistics(pipelines[0], entry);
                }
                Ok((pipelines[0], layout))
            }
            Err((pipelines, err)) => {
                unsafe {
                    for pipeline in pipelines.into_iter().filter(|p| *p != vk::Pipeline::null()) {
                        device.destroy_pipeline(pipeline, None);
                    }
                    device.destroy_pipeline_layout(layout, None);
                }
                Err(err.into())
            }
        }
    }

    /// Log every executable statistic the driver reports for a pipeline
    /// created with `CAPTURE_STATISTICS_KHR`, register counts above all.
    /// Values are logged verbatim as the driver reports them; deriving
    /// occupancy from registers is the analyst's step, not this one.
    fn log_pipeline_statistics(&self, pipeline: vk::Pipeline, entry: &CStr) {
        use std::fmt::Write as _;
        let loader = ash::khr::pipeline_executable_properties::Device::new(
            &self.instance,
            self.device(),
        );
        let pipeline_info = vk::PipelineInfoKHR::default().pipeline(pipeline);
        let executables =
            match unsafe { loader.get_pipeline_executable_properties(&pipeline_info) } {
                Ok(executables) => executables,
                Err(err) => {
                    log::warn!("pipeline stats unavailable for {entry:?}: {err}");
                    return;
                }
            };
        for (index, executable) in executables.iter().enumerate() {
            let exec_info = vk::PipelineExecutableInfoKHR::default()
                .pipeline(pipeline)
                .executable_index(index as u32);
            let Ok(stats) = (unsafe { loader.get_pipeline_executable_statistics(&exec_info) })
            else {
                continue;
            };
            let name = executable
                .name_as_c_str()
                .map_or_else(|_| "<exec>".into(), CStr::to_string_lossy);
            let mut line = format!("pipeline stats {entry:?} [{name}]");
            for stat in &stats {
                let stat_name = stat
                    .name_as_c_str()
                    .map_or_else(|_| "<stat>".into(), CStr::to_string_lossy);
                let value = match stat.format {
                    vk::PipelineExecutableStatisticFormatKHR::BOOL32 => {
                        format!("{}", unsafe { stat.value.b32 } != 0)
                    }
                    vk::PipelineExecutableStatisticFormatKHR::INT64 => {
                        format!("{}", unsafe { stat.value.i64 })
                    }
                    vk::PipelineExecutableStatisticFormatKHR::UINT64 => {
                        format!("{}", unsafe { stat.value.u64 })
                    }
                    _ => format!("{}", unsafe { stat.value.f64 }),
                };
                let _ = write!(line, "; {stat_name} = {value}");
            }
            log::info!("{line}");
        }
    }
}
