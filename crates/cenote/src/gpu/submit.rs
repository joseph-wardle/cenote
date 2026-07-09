//! Command submission: record work into a command buffer, run it on the
//! compute queue, block on a fence. Two entry points, both blocking — the
//! crate keeps one submission in flight and has no timeline-semaphore
//! pacing yet (that arrives with the wavefront render loop, not here):
//!
//! - [`Context::submit_once`] — one transient command buffer for a single
//!   recorded job: uploads, readbacks, acceleration-structure builds.
//! - [`Context::submit_passes`] — a sequence of [`Pass`]es (buffer fills,
//!   direct and indirect dispatches) in one submission, a full memory
//!   barrier between each: the wavefront engine's stage chain, where every
//!   stage's workgroup count is a number the previous stage wrote.
//!
//! Cross-submission memory visibility is free with this shape: the fence
//! signal makes all device writes available, so the next upload, dispatch,
//! or readback needs no extra barrier.

use std::slice;

use ash::vk;

use crate::error::Result;
use crate::gpu::{Buffer, ComputePipeline, Context, SceneBindings};

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
        /// The scene resources, iff the pipeline declared [`crate::gpu::Bindings::Scene`].
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
        /// The scene resources, iff the pipeline declared [`crate::gpu::Bindings::Scene`].
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

impl Context {
    /// Record commands with `record` into a fresh transient command buffer,
    /// submit it on the compute queue, and block until the GPU finishes.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Vulkan`] if pool/buffer creation, submission, or the
    /// fence wait fails.
    pub fn submit_once<F>(&self, record: F) -> Result<()>
    where
        F: FnOnce(&ash::Device, vk::CommandBuffer),
    {
        let device = self.device();
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(self.queue_family_index());
        let pool = unsafe { device.create_command_pool(&pool_info, None)? };

        // Everything after pool creation funnels through one cleanup point:
        // destroying the pool frees the command buffer with it.
        let result = self.record_and_submit(pool, record);
        unsafe { device.destroy_command_pool(pool, None) };
        result
    }

    fn record_and_submit<F>(&self, pool: vk::CommandPool, record: F) -> Result<()>
    where
        F: FnOnce(&ash::Device, vk::CommandBuffer),
    {
        let device = self.device();
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe { device.allocate_command_buffers(&alloc_info)? }[0];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            device.begin_command_buffer(command_buffer, &begin_info)?;
            record(device, command_buffer);
            device.end_command_buffer(command_buffer)?;
        }

        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? };
        let buffers = [command_buffer];
        let submit_info = vk::SubmitInfo::default().command_buffers(&buffers);
        let result = unsafe {
            device
                .queue_submit(self.queue(), &[submit_info], fence)
                .and_then(|()| device.wait_for_fences(&[fence], true, u64::MAX))
        };
        unsafe { device.destroy_fence(fence, None) };
        Ok(result?)
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
    /// not matching a pipeline's [`crate::gpu::Bindings`], the same pipeline
    /// given two different scenes (it has one descriptor set, written once
    /// per submission), or a fill that is misaligned or out of bounds.
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
