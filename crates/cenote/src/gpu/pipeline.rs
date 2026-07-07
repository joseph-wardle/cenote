//! Compute pipelines and blocking dispatch.
//!
//! Pipelines follow the BDA-first binding model (D-006): kernels reach every
//! buffer through device addresses in a single push-constant struct. The one
//! resource that cannot be an address is the scene TLAS, so every pipeline
//! carries the binding model's single descriptor set — set 0, binding 0 —
//! and [`Context::dispatch`] writes the TLAS into it.

use std::ffi::CStr;
use std::slice;

use ash::vk;

use crate::error::Result;
use crate::gpu::{AccelerationStructure, Context};

/// A compute pipeline plus its layout and TLAS descriptor set, destroyed on
/// drop (before the [`Context`], like every `gpu` resource).
pub struct ComputePipeline {
    handle: vk::Pipeline,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    /// The binding model's one descriptor set; freed with `pool`.
    set: vk::DescriptorSet,
    push_constant_size: u32,
    device: ash::Device,
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.handle, None);
            self.device.destroy_pipeline_layout(self.layout, None);
            self.device.destroy_descriptor_pool(self.pool, None);
            self.device
                .destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

/// The TLAS descriptor set under construction: layout, pool, and the one
/// allocated set. Plain handles — ownership passes to the [`ComputePipeline`]
/// on success, back to the caller's cleanup on failure.
struct TlasDescriptors {
    set_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
}

impl Context {
    /// Create a compute pipeline from SPIR-V bytes (embedded or hot-reloaded —
    /// both paths produce `slangc` output, decision D-004). `entry` names the
    /// kernel entry point; `push_constant_size` is the byte size of the
    /// kernel's push-constant struct, enforced again at dispatch time.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Vulkan`] if shader-module, descriptor, layout, or
    /// pipeline creation fails.
    ///
    /// # Panics
    ///
    /// If `spirv` is not valid SPIR-V or `push_constant_size` is not a
    /// non-zero multiple of 4 — both are compile-pipeline or programmer bugs
    /// (D-010), not environment failures.
    pub fn create_compute_pipeline(
        &self,
        spirv: &[u8],
        entry: &CStr,
        push_constant_size: u32,
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
        // success and failure alike, so the rest funnels through one point.
        let result = self.create_with_module(module, entry, push_constant_size);
        unsafe { device.destroy_shader_module(module, None) };
        result
    }

    fn create_with_module(
        &self,
        module: vk::ShaderModule,
        entry: &CStr,
        push_constant_size: u32,
    ) -> Result<ComputePipeline> {
        let descriptors = self.create_tlas_descriptors()?;
        // From here every failure must destroy the descriptor objects:
        // funnel through one exit point.
        match self.create_with_descriptors(module, entry, push_constant_size, &descriptors) {
            Ok(pipeline) => Ok(pipeline),
            Err(err) => {
                let device = self.device();
                unsafe {
                    device.destroy_descriptor_pool(descriptors.pool, None);
                    device.destroy_descriptor_set_layout(descriptors.set_layout, None);
                }
                Err(err)
            }
        }
    }

    /// Create the single descriptor set of the binding model (D-006):
    /// binding 0 = the scene TLAS. Its contents are written at dispatch time.
    fn create_tlas_descriptors(&self) -> Result<TlasDescriptors> {
        let device = self.device();
        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);
        let layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(slice::from_ref(&binding));
        let set_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None)? };

        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .descriptor_count(1);
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(slice::from_ref(&pool_size));
        let pool = unsafe { device.create_descriptor_pool(&pool_info, None) };
        let pool = match pool {
            Ok(pool) => pool,
            Err(err) => {
                unsafe { device.destroy_descriptor_set_layout(set_layout, None) };
                return Err(err.into());
            }
        };

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(slice::from_ref(&set_layout));
        match unsafe { device.allocate_descriptor_sets(&alloc_info) } {
            Ok(sets) => Ok(TlasDescriptors {
                set_layout,
                pool,
                set: sets[0],
            }),
            Err(err) => {
                unsafe {
                    device.destroy_descriptor_pool(pool, None);
                    device.destroy_descriptor_set_layout(set_layout, None);
                }
                Err(err.into())
            }
        }
    }

    fn create_with_descriptors(
        &self,
        module: vk::ShaderModule,
        entry: &CStr,
        push_constant_size: u32,
        descriptors: &TlasDescriptors,
    ) -> Result<ComputePipeline> {
        let device = self.device();
        let range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(push_constant_size);
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(slice::from_ref(&descriptors.set_layout))
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
            Ok(pipelines) => Ok(ComputePipeline {
                handle: pipelines[0],
                layout,
                set_layout: descriptors.set_layout,
                pool: descriptors.pool,
                set: descriptors.set,
                push_constant_size,
                device: device.clone(),
            }),
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

    /// Bind `pipeline` with `tlas` in its descriptor set, set the push
    /// constants, dispatch `group_counts` workgroups, and block until the
    /// GPU finishes (D-007). The fence wait makes the kernel's writes
    /// available, so a subsequent [`Context::download_buffer`] needs no
    /// barrier.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Vulkan`] if submission fails.
    ///
    /// # Panics
    ///
    /// If `push_constants` doesn't match the size the pipeline was created
    /// with — a programmer bug (D-010): the bytes would silently misalign
    /// with the kernel's view of them.
    pub fn dispatch(
        &self,
        pipeline: &ComputePipeline,
        tlas: &AccelerationStructure,
        push_constants: &[u8],
        group_counts: [u32; 3],
    ) -> Result<()> {
        assert_eq!(
            push_constants.len() as u32,
            pipeline.push_constant_size,
            "push constants don't match the pipeline's declared size"
        );
        // (Re)writing the set before recording is safe: blocking submits
        // (D-007) mean it is never in flight here.
        let handles = [tlas.handle()];
        let mut tlas_write = vk::WriteDescriptorSetAccelerationStructureKHR::default()
            .acceleration_structures(&handles);
        let write = vk::WriteDescriptorSet::default()
            .dst_set(pipeline.set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            // Not inferred from the extension struct: without this the write
            // is a zero-descriptor no-op.
            .descriptor_count(1)
            .push_next(&mut tlas_write);
        unsafe {
            self.device()
                .update_descriptor_sets(slice::from_ref(&write), &[]);
        }

        self.submit_once(|device, cmd| unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline.handle);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.layout,
                0,
                slice::from_ref(&pipeline.set),
                &[],
            );
            device.cmd_push_constants(
                cmd,
                pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_constants,
            );
            device.cmd_dispatch(cmd, group_counts[0], group_counts[1], group_counts[2]);
        })
    }
}
