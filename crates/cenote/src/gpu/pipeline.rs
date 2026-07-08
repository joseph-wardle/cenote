//! Compute pipelines and blocking dispatch.
//!
//! Kernels reach every buffer through device addresses in a single
//! push-constant struct. The one resource that cannot be an address is the
//! scene TLAS, so kernels that trace rays declare [`Bindings::Tlas`] and
//! carry the binding model's single descriptor set — set 0, binding 0 —
//! which [`Context::dispatch`] writes the TLAS into. Kernels that only chew
//! buffers ([`Bindings::None`]) have no descriptors at all.

use std::ffi::CStr;
use std::slice;

use ash::vk;

use crate::error::Result;
use crate::gpu::{AccelerationStructure, Context};

/// The descriptor bindings a kernel needs. Buffers travel as device
/// addresses in push constants, so the only question is whether the kernel
/// traces rays — the TLAS is the one resource that must be a descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bindings {
    /// Push constants only — no descriptor set.
    None,
    /// Set 0, binding 0: the scene TLAS, written at dispatch time.
    Tlas,
}

/// A compute pipeline plus its layout and (for ray-tracing kernels) TLAS
/// descriptor set, destroyed on drop (before the [`Context`], like every
/// `gpu` resource).
pub struct ComputePipeline {
    handle: vk::Pipeline,
    layout: vk::PipelineLayout,
    /// Present iff created with [`Bindings::Tlas`].
    tlas: Option<TlasDescriptors>,
    push_constant_size: u32,
    device: ash::Device,
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.handle, None);
            self.device.destroy_pipeline_layout(self.layout, None);
            if let Some(tlas) = &self.tlas {
                tlas.destroy(&self.device);
            }
        }
    }
}

/// The TLAS descriptor set under construction: layout, pool, and the one
/// allocated set. Plain handles — ownership passes to the [`ComputePipeline`]
/// on success, to [`TlasDescriptors::destroy`] on failure.
struct TlasDescriptors {
    set_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
}

impl TlasDescriptors {
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
    /// kernel traces rays and therefore needs the TLAS descriptor.
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
        let tlas = match bindings {
            Bindings::None => None,
            Bindings::Tlas => Some(self.create_tlas_descriptors()?),
        };
        match self.create_layout_and_pipeline(module, entry, push_constant_size, tlas.as_ref()) {
            Ok((handle, layout)) => Ok(ComputePipeline {
                handle,
                layout,
                tlas,
                push_constant_size,
                device: self.device().clone(),
            }),
            Err(err) => {
                if let Some(tlas) = &tlas {
                    unsafe { tlas.destroy(self.device()) };
                }
                Err(err)
            }
        }
    }

    /// Create the binding model's single descriptor set: binding 0 = the
    /// scene TLAS. Its contents are written at dispatch time.
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

        // `destroy` only touches the pool and layout, so the struct — and
        // its cleanup — can exist before the set does.
        let mut descriptors = TlasDescriptors {
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
        descriptors: Option<&TlasDescriptors>,
    ) -> Result<(vk::Pipeline, vk::PipelineLayout)> {
        let device = self.device();
        let range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(push_constant_size);
        let set_layouts = descriptors.map_or(&[][..], |tlas| slice::from_ref(&tlas.set_layout));
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

    /// Bind `pipeline` (with `tlas` written into its descriptor set, for
    /// ray-tracing kernels), set the push constants, dispatch `group_counts`
    /// workgroups, and block until the GPU finishes. The fence wait makes
    /// the kernel's writes available, so a subsequent
    /// [`Context::download_buffer`] needs no barrier.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Vulkan`] if submission fails.
    ///
    /// # Panics
    ///
    /// If `push_constants` doesn't match the size the pipeline was created
    /// with (the bytes would silently misalign with the kernel's view of
    /// them), or if `tlas` doesn't match the pipeline's [`Bindings`].
    pub fn dispatch(
        &self,
        pipeline: &ComputePipeline,
        tlas: Option<&AccelerationStructure>,
        push_constants: &[u8],
        group_counts: [u32; 3],
    ) -> Result<()> {
        assert_eq!(
            push_constants.len() as u32,
            pipeline.push_constant_size,
            "push constants don't match the pipeline's declared size"
        );
        assert_eq!(
            tlas.is_some(),
            pipeline.tlas.is_some(),
            "TLAS argument doesn't match the pipeline's declared bindings"
        );
        if let (Some(tlas), Some(descriptors)) = (tlas, &pipeline.tlas) {
            // (Re)writing the set before recording is safe: blocking submits
            // mean it is never in flight here.
            let handles = [tlas.handle()];
            let mut tlas_write = vk::WriteDescriptorSetAccelerationStructureKHR::default()
                .acceleration_structures(&handles);
            let write = vk::WriteDescriptorSet::default()
                .dst_set(descriptors.set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                // Not inferred from the extension struct: without this the
                // write is a zero-descriptor no-op.
                .descriptor_count(1)
                .push_next(&mut tlas_write);
            unsafe {
                self.device()
                    .update_descriptor_sets(slice::from_ref(&write), &[]);
            }
        }

        self.submit_once(|device, cmd| unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline.handle);
            if let Some(descriptors) = &pipeline.tlas {
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline.layout,
                    0,
                    slice::from_ref(&descriptors.set),
                    &[],
                );
            }
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
