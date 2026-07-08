//! Compute pipelines and blocking dispatch — single kernels or multi-pass
//! waves.
//!
//! Kernels reach every buffer through device addresses in a single
//! push-constant struct. The resources that cannot be addresses are the
//! scene TLAS and the environment texture (filtered reads need a real
//! sampled image), so kernels that touch either declare
//! [`Bindings::Scene`] and carry the binding model's single descriptor set
//! — set 0: binding 0 the TLAS, binding 1 the environment — written at
//! submission time. Kernels that only chew buffers ([`Bindings::None`])
//! have no descriptors at all.
//!
//! [`Context::submit_passes`] records a sequence of [`Pass`]es — buffer
//! fills, direct and indirect dispatches — into one blocking submission,
//! with a full barrier between passes: the wavefront engine's stage chain,
//! where each stage's workgroup count is a number the previous stage wrote.

use std::ffi::CStr;
use std::slice;

use ash::vk;

use crate::error::Result;
use crate::gpu::{AccelerationStructure, Buffer, Context, SampledImage};

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
    handle: vk::Pipeline,
    layout: vk::PipelineLayout,
    /// Present iff created with [`Bindings::Scene`].
    scene: Option<SceneDescriptors>,
    push_constant_size: u32,
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
struct SceneDescriptors {
    set_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
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

    /// Bind `pipeline` (with `scene`'s resources written into its
    /// descriptor set, for kernels that declared them), set the push
    /// constants, dispatch `group_counts` workgroups, and block until the
    /// GPU finishes. The fence wait makes the kernel's writes available, so
    /// a subsequent [`Context::download_buffer`] needs no barrier.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Vulkan`] if submission fails.
    ///
    /// # Panics
    ///
    /// As [`Context::submit_passes`].
    pub fn dispatch(
        &self,
        pipeline: &ComputePipeline,
        scene: Option<SceneBindings>,
        push_constants: &[u8],
        group_counts: [u32; 3],
    ) -> Result<()> {
        self.submit_passes(&[Pass::Dispatch {
            pipeline,
            scene,
            push_constants,
            group_counts,
        }])
    }

    /// Record `passes` in order into one command buffer, submit it, and
    /// block until the GPU finishes. A full memory barrier sits between
    /// consecutive passes, so each pass sees every prior pass's writes —
    /// including indirect dispatches reading workgroup counts a previous
    /// pass wrote. (Full flushes between stages are the simple-and-correct
    /// baseline; overlapping independent stages is a measured optimization
    /// for later.) The fence wait makes all writes available, so a
    /// subsequent [`Context::download_buffer`] needs no barrier.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Vulkan`] if submission fails.
    ///
    /// # Panics
    ///
    /// On programmer bugs, all checked before anything is recorded: push
    /// constants not matching a pipeline's declared size, a scene argument
    /// not matching a pipeline's [`Bindings`], the same pipeline given two
    /// different scenes (it has one descriptor set, written once per
    /// submission), or a fill that is misaligned or out of bounds.
    pub fn submit_passes(&self, passes: &[Pass]) -> Result<()> {
        for pass in passes {
            self.validate_and_write_descriptors(pass, passes);
        }
        self.submit_once(|device, cmd| {
            for (index, pass) in passes.iter().enumerate() {
                if index > 0 {
                    barrier_between_passes(device, cmd);
                }
                record_pass(device, cmd, pass);
            }
        })
    }

    /// The pre-recording half of [`Context::submit_passes`]: assert the
    /// pass is well-formed and write the scene descriptors for dispatches
    /// that carry them. Writing before recording is safe — blocking submits
    /// mean no set is ever in flight here.
    fn validate_and_write_descriptors(&self, pass: &Pass, passes: &[Pass]) {
        let (pipeline, scene, push_constants) = match *pass {
            Pass::Fill {
                buffer,
                offset,
                size,
                value: _,
            } => {
                assert!(
                    offset.is_multiple_of(4) && size > 0 && size.is_multiple_of(4),
                    "fill offset and size must be non-zero multiples of 4"
                );
                assert!(
                    offset + size <= buffer.size(),
                    "fill reaches past the end of the buffer"
                );
                return;
            }
            Pass::Dispatch {
                pipeline,
                scene,
                push_constants,
                group_counts: _,
            } => (pipeline, scene, push_constants),
            Pass::DispatchIndirect {
                pipeline,
                scene,
                push_constants,
                args,
                offset,
            } => {
                assert!(
                    offset.is_multiple_of(4) && offset + 12 <= args.size(),
                    "indirect args must be 4-byte aligned and inside the buffer"
                );
                (pipeline, scene, push_constants)
            }
        };
        assert_eq!(
            push_constants.len() as u32,
            pipeline.push_constant_size,
            "push constants don't match the pipeline's declared size"
        );
        assert_eq!(
            scene.is_some(),
            pipeline.scene.is_some(),
            "scene argument doesn't match the pipeline's declared bindings"
        );
        let Some(scene) = scene else {
            return;
        };
        assert!(
            passes
                .iter()
                .filter_map(|other| match *other {
                    Pass::Dispatch {
                        pipeline: p,
                        scene: s,
                        ..
                    }
                    | Pass::DispatchIndirect {
                        pipeline: p,
                        scene: s,
                        ..
                    } if std::ptr::eq(p, pipeline) => s,
                    _ => None,
                })
                .all(|other| {
                    other.tlas.handle() == scene.tlas.handle()
                        && std::ptr::eq(other.environment, scene.environment)
                }),
            "one pipeline, two scenes — its single descriptor set can hold only one"
        );

        let descriptors = pipeline.scene.as_ref().expect("checked against bindings");
        let handles = [scene.tlas.handle()];
        let mut tlas_write = vk::WriteDescriptorSetAccelerationStructureKHR::default()
            .acceleration_structures(&handles);
        let image_info = scene.environment.descriptor();
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(descriptors.set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                // Not inferred from the extension struct: without this the
                // write is a zero-descriptor no-op.
                .descriptor_count(1)
                .push_next(&mut tlas_write),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptors.set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(slice::from_ref(&image_info)),
        ];
        unsafe {
            self.device().update_descriptor_sets(&writes, &[]);
        }
    }
}

/// One step of a [`Context::submit_passes`] submission.
pub enum Pass<'a> {
    /// Overwrite a byte range with a repeated `u32` (`vkCmdFillBuffer`) —
    /// how a wave resets queue counters without touching the host.
    Fill {
        /// Target buffer (needs `TRANSFER_DST` usage).
        buffer: &'a Buffer,
        /// First byte to fill; a multiple of 4.
        offset: u64,
        /// Bytes to fill; a non-zero multiple of 4.
        size: u64,
        /// The `u32` repeated across the range.
        value: u32,
    },
    /// A compute dispatch with host-chosen workgroup counts.
    Dispatch {
        /// The pipeline to run.
        pipeline: &'a ComputePipeline,
        /// The scene resources, iff the pipeline declared [`Bindings::Scene`].
        scene: Option<SceneBindings<'a>>,
        /// Exactly the pipeline's declared push-constant size.
        push_constants: &'a [u8],
        /// Workgroups along x, y, z.
        group_counts: [u32; 3],
    },
    /// A compute dispatch whose workgroup counts the GPU reads from `args`
    /// at `offset` at execution time — how a stage sized by the previous
    /// stage's output dispatches with no readback.
    DispatchIndirect {
        /// The pipeline to run.
        pipeline: &'a ComputePipeline,
        /// The scene resources, iff the pipeline declared [`Bindings::Scene`].
        scene: Option<SceneBindings<'a>>,
        /// Exactly the pipeline's declared push-constant size.
        push_constants: &'a [u8],
        /// Where the counts live (needs `INDIRECT_BUFFER` usage).
        args: &'a Buffer,
        /// Byte offset of the `VkDispatchIndirectCommand` (three `u32`s:
        /// workgroups along x, y, z) inside `args`; a multiple of 4.
        offset: u64,
    },
}

fn record_pass(device: &ash::Device, cmd: vk::CommandBuffer, pass: &Pass) {
    match *pass {
        Pass::Fill {
            buffer,
            offset,
            size,
            value,
        } => unsafe {
            device.cmd_fill_buffer(cmd, buffer.handle(), offset, size, value);
        },
        Pass::Dispatch {
            pipeline,
            push_constants,
            group_counts,
            ..
        } => unsafe {
            bind_and_push(device, cmd, pipeline, push_constants);
            device.cmd_dispatch(cmd, group_counts[0], group_counts[1], group_counts[2]);
        },
        Pass::DispatchIndirect {
            pipeline,
            push_constants,
            args,
            offset,
            ..
        } => unsafe {
            bind_and_push(device, cmd, pipeline, push_constants);
            device.cmd_dispatch_indirect(cmd, args.handle(), offset);
        },
    }
}

unsafe fn bind_and_push(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    pipeline: &ComputePipeline,
    push_constants: &[u8],
) {
    unsafe {
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline.handle);
        if let Some(descriptors) = &pipeline.scene {
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
    }
}

/// Everything before, visible to everything after: compute and transfer
/// writes flushed to compute reads/writes, transfer writes, and indirect-
/// command reads. One barrier shape for every pass boundary keeps the wave
/// obviously correct.
fn barrier_between_passes(device: &ash::Device, cmd: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier2::default()
        .src_stage_mask(
            vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::ALL_TRANSFER,
        )
        .src_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::TRANSFER_WRITE)
        .dst_stage_mask(
            vk::PipelineStageFlags2::COMPUTE_SHADER
                | vk::PipelineStageFlags2::DRAW_INDIRECT
                | vk::PipelineStageFlags2::ALL_TRANSFER,
        )
        .dst_access_mask(
            vk::AccessFlags2::SHADER_READ
                | vk::AccessFlags2::SHADER_WRITE
                | vk::AccessFlags2::INDIRECT_COMMAND_READ
                | vk::AccessFlags2::TRANSFER_WRITE,
        );
    let info = vk::DependencyInfo::default().memory_barriers(slice::from_ref(&barrier));
    unsafe { device.cmd_pipeline_barrier2(cmd, &info) };
}
