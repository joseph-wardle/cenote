//! Compute pipelines and their descriptor sets.
//!
//! A [`ComputePipeline`] is one kernel compiled from SPIR-V plus its layout
//! and — for kernels that read the scene's TLAS or environment texture —
//! the one descriptor set that carries them. Kernels reach every buffer
//! through device addresses in a single push-constant struct; the only
//! resources that cannot be addresses are the scene TLAS and the
//! environment texture (filtered reads need a real sampled image), so a
//! kernel that touches either declares [`Bindings::Scene`] and carries set
//! 0 — binding 0 the TLAS, binding 1 the environment — written at
//! submission time. Kernels that only chew buffers ([`Bindings::None`])
//! have no descriptors at all.
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
/// touches the two resources that must be descriptors — the TLAS and the
/// environment texture. One shared layout for both keeps the binding model
/// a single small set; a kernel that statically uses only one of them is
/// fine, Vulkan only requires that what it *uses* is bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bindings {
    /// Push constants only — no descriptor set.
    None,
    /// Set 0 — binding 0: the scene TLAS; binding 1: the environment
    /// texture. Both written at dispatch time.
    Scene,
}

/// The scene resources a [`Bindings::Scene`] dispatch binds.
#[derive(Clone, Copy)]
pub struct SceneBindings<'a> {
    /// The scene TLAS (binding 0).
    pub tlas: &'a AccelerationStructure,
    /// The environment texture (binding 1).
    pub environment: &'a SampledImage,
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
    /// # Errors
    ///
    /// [`crate::Error::Vulkan`] if shader-module, descriptor, layout, or
    /// pipeline creation fails.
    ///
    /// # Panics
    ///
    /// If `spirv` is not valid SPIR-V or `push_constant_size` is not a
    /// non-zero multiple of 4 — programmer bugs upstream of any GPU work.
    pub fn create_compute_pipeline(
        &self,
        spirv: &[u8],
        entry: &CStr,
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
        let result =
            self.create_descriptors_and_pipeline(module, entry, push_constant_size, bindings);
        unsafe { device.destroy_shader_module(module, None) };
        result
    }

    fn create_descriptors_and_pipeline(
        &self,
        module: vk::ShaderModule,
        entry: &CStr,
        push_constant_size: u32,
        bindings: Bindings,
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
    /// scene TLAS, binding 1 = the environment texture. Contents are
    /// written at dispatch time.
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
        ];
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let set_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None)? };

        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1),
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
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);
        let pipelines = unsafe {
            device.create_compute_pipelines(vk::PipelineCache::null(), slice::from_ref(&info), None)
        };

        match pipelines {
            Ok(pipelines) => Ok((pipelines[0], layout)),
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
}
