//! Compute pipelines and blocking dispatch.
//!
//! Pipelines follow the BDA-first binding model (D-006): kernels reach every
//! buffer through device addresses in a single push-constant struct, so M0
//! pipelines carry no descriptor sets at all. The TLAS descriptor set — the
//! one resource that cannot be an address — joins in m0-plan step 7.

use std::ffi::CStr;

use ash::vk;

use crate::error::Result;
use crate::gpu::Context;

/// A compute pipeline plus its layout, destroyed on drop (before the
/// [`Context`], like every `gpu` resource).
pub struct ComputePipeline {
    handle: vk::Pipeline,
    layout: vk::PipelineLayout,
    push_constant_size: u32,
    device: ash::Device,
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.handle, None);
            self.device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

impl Context {
    /// Create a compute pipeline from SPIR-V bytes (embedded or hot-reloaded —
    /// both paths produce `slangc` output, decision D-004). `entry` names the
    /// kernel entry point; `push_constant_size` is the byte size of the
    /// kernel's push-constant struct, enforced again at dispatch time.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Vulkan`] if shader-module, layout, or pipeline
    /// creation fails.
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
        let device = self.device();
        let range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(push_constant_size);
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .push_constant_ranges(std::slice::from_ref(&range));
        let layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(entry);
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);
        let pipelines = unsafe {
            device.create_compute_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&info),
                None,
            )
        };

        match pipelines {
            Ok(pipelines) => Ok(ComputePipeline {
                handle: pipelines[0],
                layout,
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

    /// Bind `pipeline`, set its push constants, dispatch `group_counts`
    /// workgroups, and block until the GPU finishes (D-007). The fence wait
    /// makes the kernel's writes available, so a subsequent
    /// [`Context::download_buffer`] needs no barrier.
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
        push_constants: &[u8],
        group_counts: [u32; 3],
    ) -> Result<()> {
        assert_eq!(
            push_constants.len() as u32,
            pipeline.push_constant_size,
            "push constants don't match the pipeline's declared size"
        );
        self.submit_once(|device, cmd| unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline.handle);
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
