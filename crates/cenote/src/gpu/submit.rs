//! Command submission: record work into a command buffer, run it on the
//! compute queue, block on a fence. Two entry points, both blocking — each
//! submitting thread keeps one submission in flight and waits its own fence
//! (timeline-semaphore pacing is deferred to the measured pre-M3
//! performance pass):
//!
//! - [`Context::submit_once`] — one transient command buffer for a single
//!   recorded job: uploads, readbacks, acceleration-structure builds.
//! - [`Context::submit_passes`] — a sequence of [`Pass`]es (buffer fills,
//!   direct and indirect dispatches) in one submission, a full memory
//!   barrier between each: the wavefront engine's stage chain, where every
//!   stage's workgroup count is a number the previous stage wrote. Its
//!   [`Context::submit_passes_timed`] form brackets the submission with a
//!   [`PassTimer`], and on the frames that ask for one stamps every span
//!   boundary inside it; the two are one function, so an instrumented wave
//!   cannot drift from an ordinary one.
//!
//! Cross-submission memory visibility is free with this shape: the fence
//! signal makes all device writes available, so the next upload, dispatch,
//! or readback needs no extra barrier.
//!
//! The one queue every submission funnels through is wrapped in [`Queue`],
//! whose lock is where the render loop's traces and the presenter's blits
//! take turns — Vulkan requires submission to a queue to be externally
//! synchronized, and the presenter's pre-rebuild device-idle wait takes the
//! same lock for the same reason.

use std::slice;
use std::sync::{Arc, Mutex};

use ash::prelude::VkResult;
use ash::vk;

use crate::error::Result;
use crate::gpu::timing::Detail;
use crate::gpu::{Buffer, ComputePipeline, Context, PassTimer, SceneBindings};
use crate::stats::PassTimings;

/// The device's single queue, wrapped so Vulkan's external-synchronization
/// rule for submission is enforced by the type rather than by a comment.
///
/// `vk::Queue` is `Sync`, so the compiler would let two threads submit at
/// once — which Vulkan forbids. Once the render loop traces on its own thread
/// while the presenter blits on another, every submission must take this
/// lock. It is held *only* around the submit call, never across the fence
/// wait that follows: waiting under it would stall the other thread for a
/// whole GPU frame. The one deliberate exception is
/// [`Queue::wait_device_idle`]: `vkDeviceWaitIdle` needs the same external
/// synchronization as submission, and idling the device *is* the point, so it
/// holds the lock across the wait.
///
/// Cloned rather than borrowed — the [`Context`] and its [`Presenter`] each
/// hold a handle to the same lock, exactly as they share the allocator.
#[derive(Clone)]
pub(super) struct Queue {
    queue: Arc<Mutex<vk::Queue>>,
}

impl Queue {
    /// Wrap the device's queue handle.
    pub(super) fn new(queue: vk::Queue) -> Self {
        Self {
            queue: Arc::new(Mutex::new(queue)),
        }
    }

    /// Submit `submits`, signaling `fence` on completion. Locks only for the
    /// submit; wait on `fence` after this returns, with the lock released.
    pub(super) fn submit(
        &self,
        device: &ash::Device,
        submits: &[vk::SubmitInfo],
        fence: vk::Fence,
    ) -> VkResult<()> {
        let queue = self.queue.lock().expect("queue mutex poisoned");
        unsafe { device.queue_submit(*queue, submits, fence) }
    }

    /// As [`Queue::submit`], for the synchronization2 submission the presenter
    /// records.
    pub(super) fn submit2(
        &self,
        device: &ash::Device,
        submits: &[vk::SubmitInfo2],
        fence: vk::Fence,
    ) -> VkResult<()> {
        let queue = self.queue.lock().expect("queue mutex poisoned");
        unsafe { device.queue_submit2(*queue, submits, fence) }
    }

    /// Wait for the device to finish every outstanding submission, holding
    /// the queue lock across the wait. `vkDeviceWaitIdle` requires every queue
    /// be externally synchronized just as submission does, so the presenter —
    /// which idles the device before rebuilding its swapchain — must fence out
    /// the render thread's submits for the wait's duration. Unlike
    /// [`Queue::submit`] the lock deliberately spans the whole wait; the render
    /// thread's next submit merely waits its (brief) turn, which the resize
    /// that triggers this can afford.
    pub(super) fn wait_device_idle(&self, device: &ash::Device) -> VkResult<()> {
        let _guard = self.queue.lock().expect("queue mutex poisoned");
        unsafe { device.device_wait_idle() }
    }

    /// Present through `swapchain`. Locks only for the present call; the
    /// returned bool is the swapchain's suboptimal flag.
    pub(super) fn present(
        &self,
        swapchain: &ash::khr::swapchain::Device,
        present_info: &vk::PresentInfoKHR,
    ) -> VkResult<bool> {
        let queue = self.queue.lock().expect("queue mutex poisoned");
        unsafe { swapchain.queue_present(*queue, present_info) }
    }

    /// Run `f` holding the queue lock, for a submission buried inside a
    /// dependency we don't record ourselves — the egui texture upload submits
    /// *and* fence-waits internally. Unlike [`Queue::submit`], the lock spans
    /// all of `f`, wait included, so this is for rare, small uploads only.
    pub(super) fn locked<T>(&self, f: impl FnOnce(vk::Queue) -> T) -> T {
        let queue = self.queue.lock().expect("queue mutex poisoned");
        f(*queue)
    }
}

/// One step of a [`Context::submit_passes`] submission. `Copy` so a caller
/// can append its own passes to a recorded list and submit them together —
/// how the film's accumulate and tonemap ride the wave's one submission.
#[derive(Clone, Copy)]
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
    /// Copy a byte range from one buffer to another (`vkCmdCopyBuffer`) — how
    /// a single-shot buffer is lifted somewhere else without a resolve pass.
    CopyBuffer {
        /// Source (needs `TRANSFER_SRC` usage).
        src: &'a Buffer,
        /// Destination (needs `TRANSFER_DST` usage).
        dst: &'a Buffer,
        /// Bytes to copy, from offset 0 of each; must fit both buffers.
        size: u64,
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

impl Pass<'_> {
    /// The name this pass reports itself under in [`crate::stats`].
    ///
    /// A dispatch borrows its kernel's entry-point name, which is already
    /// unique per kernel and already `'static` — so the pass-ID registry is
    /// the kernel list itself, with nothing to keep in sync. Split a kernel
    /// in two and the two halves show up under their own names the moment
    /// they exist; there is no table to remember to update.
    ///
    /// Fills and copies get one bucket apiece rather than one per buffer: a
    /// wave's dozen fills are noise taken singly and a real line item taken
    /// together, and which buffer was zeroed is not the question anyone
    /// reading a stats line is asking.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match *self {
            Pass::Fill { .. } => "fill",
            Pass::CopyBuffer { .. } => "copy",
            Pass::Dispatch { pipeline, .. } | Pass::DispatchIndirect { pipeline, .. } => {
                pipeline.label
            }
        }
    }
}

/// A submission's spans — maximal runs of consecutive passes under one
/// label — as the label and how many passes ran under it.
///
/// This is the unit [`PassTimer`] brackets, and the reason is that
/// [`PassTimings`] folds by label: a stamp between two consecutive passes
/// that share one buys a number the fold sums straight back together, at
/// the price of a real timestamp write. The wave has plenty of such runs —
/// three queue-clearing fills open every bounce, and up to seven open the
/// whole submission — so the saving is the difference between timing what
/// is reported and timing what happens to be recorded.
pub(super) fn spans<'a>(passes: &'a [Pass<'_>]) -> impl Iterator<Item = (&'static str, u32)> + 'a {
    passes
        .chunk_by(|left, right| left.label() == right.label())
        .map(|run| {
            (
                run[0].label(),
                u32::try_from(run.len()).unwrap_or(u32::MAX),
            )
        })
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
        // Submit under the queue lock, then wait with it released — a fence
        // wait held across the lock would stall the other thread's submits.
        let result = self
            .queue
            .submit(device, &[submit_info], fence)
            .and_then(|()| unsafe { device.wait_for_fences(&[fence], true, u64::MAX) });
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
        self.submit_passes_timed(passes, None).map(|_| ())
    }

    /// [`Context::submit_passes`], timed when a `timer` is given.
    ///
    /// Without one this *is* [`Context::submit_passes`] — not a parallel
    /// implementation of it but the same code with nothing extra recorded,
    /// which is the only way a timed render and an untimed one are
    /// guaranteed to submit the same work.
    ///
    /// With one, the timer decides how finely: every submission is
    /// bracketed at its outer boundaries for its total, and one in
    /// [`PassTimer::BREAKDOWN_INTERVAL`] is also stamped span by span. The
    /// breakdown carries one entry per *distinct* pass label, in the order
    /// the labels first appear: a wave that dispatches `intersect` once per
    /// bounce reports one `intersect` line with the summed time and the
    /// call count, which survives a kernel being split or a bounce count
    /// changing. The timer's pool grows to fit a submission of more spans
    /// than it has seen, so a deep-bounce scene is measured like any other
    /// rather than silently going dark.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Vulkan`] if submission or the query readback fails.
    ///
    /// # Panics
    ///
    /// As [`Context::submit_passes`].
    pub fn submit_passes_timed(
        &self,
        passes: &[Pass],
        timer: Option<&mut PassTimer>,
    ) -> Result<PassTimings> {
        for pass in passes {
            self.validate_and_write_descriptors(pass, passes);
        }
        // Whether this submission is timed, and how finely, is settled once
        // — here. An empty one has no boundaries worth stamping, so it is
        // not. Asking again further down is how a stamp and a readback come
        // to disagree about which queries were written.
        let mut timer = timer.filter(|_| !passes.is_empty());
        let detail = timer.as_deref_mut().map(PassTimer::detail);
        let intervals = detail.map_or(0, |detail| detail.intervals(passes));
        if let Some(timer) = timer.as_deref_mut() {
            // A total spans one interval, which every pool has room for
            // already, so this only ever grows for a breakdown.
            timer.grow_to(intervals)?;
        }
        // Recording only needs the timer to read, which lets the mutable
        // borrow resume for the readback once the closure is done.
        let stamps = timer.as_deref();
        let breakdown = detail == Some(Detail::Breakdown);
        self.submit_once(|device, cmd| {
            if let Some(timer) = stamps {
                timer.reset(cmd, intervals + 1);
            }
            // Passes are recorded one at a time and stamped a span at a
            // time: the barrier goes between every consecutive pair, the
            // timestamp only where the label changes. See [`spans`].
            let mut span = 0;
            let mut opening = true;
            for (index, pass) in passes.iter().enumerate() {
                if index > 0 {
                    barrier_between_passes(device, cmd);
                    opening = pass.label() != passes[index - 1].label();
                    if opening {
                        span += 1;
                    }
                }
                // Between the barrier and the pass, where the GPU is
                // already known to be quiet. A total stamps the opening
                // boundary and nothing else until the closing one.
                if let Some(timer) = stamps.filter(|_| opening && (breakdown || index == 0)) {
                    timer.stamp(cmd, span);
                }
                record_pass(device, cmd, pass);
            }
            // The closing boundary needs no barrier of its own: the stamp
            // waits on `ALL_COMMANDS`, which is everything the submission
            // could still be doing. Adding one would put a command in the
            // instrumented submission that the ordinary one does not have
            // — cost charged to the very frames being measured.
            if let Some(timer) = stamps {
                timer.stamp(cmd, intervals);
            }
        })?;
        match timer.zip(detail) {
            Some((timer, detail)) => timer.read(passes, detail),
            None => Ok(PassTimings::default()),
        }
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
            Pass::CopyBuffer { src, dst, size } => {
                assert!(
                    size <= src.size() && size <= dst.size(),
                    "copy reaches past the end of a buffer"
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
            scene.textures.len() <= crate::gpu::MAX_SCENE_TEXTURES as usize,
            "scene binds {} textures, the bindless table holds {}",
            scene.textures.len(),
            crate::gpu::MAX_SCENE_TEXTURES
        );
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
                        && std::ptr::eq(other.textures, scene.textures)
                }),
            "one pipeline, two scenes — its single descriptor set can hold only one"
        );
        self.write_scene_descriptors(pipeline, scene);
    }

    /// Write one validated [`SceneBindings`] into a pipeline's descriptor
    /// set: the TLAS at binding 0, the environment at binding 1, the
    /// bindless texture table at binding 2 — only as far as the scene
    /// fills it, since the binding is partially bound and kernels never
    /// index past what the material records name.
    fn write_scene_descriptors(&self, pipeline: &ComputePipeline, scene: SceneBindings) {
        let descriptors = pipeline.scene.as_ref().expect("checked against bindings");
        let handles = [scene.tlas.handle()];
        let mut tlas_write = vk::WriteDescriptorSetAccelerationStructureKHR::default()
            .acceleration_structures(&handles);
        let image_info = scene.environment.descriptor();
        let mut writes = vec![
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
        if !scene.textures.is_empty() {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptors.set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(scene.textures),
            );
        }
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
        Pass::CopyBuffer { src, dst, size } => unsafe {
            let region = vk::BufferCopy::default().size(size);
            device.cmd_copy_buffer(cmd, src.handle(), dst.handle(), &[region]);
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
