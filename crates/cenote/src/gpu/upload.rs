//! Batched residency: many buffer uploads and BLAS builds, few submits.
//!
//! A blocking submit costs ~0.4 ms on this machine whether it wraps
//! eighteen kilobytes of copy or nothing at all — the submit is the cost,
//! and the bytes beside it are free. A mesh needs four buffers and a BLAS
//! build, so a scene uploaded one resource at a time pays five of those per
//! mesh, which is most of a large scene's time to first ray. [`Upload`]
//! pays them per *chunk* instead: allocate every resource eagerly (which is
//! free) and queue only the GPU work, flushing when a chunk's worth of
//! staging and scratch has piled up. [`Context::upload_buffer`] is this on
//! a batch of one.
//!
//! Two properties are load-bearing and easy to lose:
//!
//! - **A flush blocks.** This batches submissions; it does not pipeline
//!   them. Prep's phase timings are exact host `Instant`s *because* every
//!   GPU call it makes has finished when it returns — let a submission
//!   outlive its phase and the numbers quietly redistribute themselves
//!   between `upload`, `tlas`, and the first sample. It is also what lets a
//!   flush free its staging: nothing is still reading it.
//! - **`Drop` releases, never submits.** No flush is ever in flight, so
//!   unwinding out of a half-queued load is unconditionally safe — it
//!   frees staging and scratch and abandons work that never started. Only
//!   [`Upload::finish`] submits, and it returns a `Result` rather than
//!   hiding a device fault in a destructor.
//!
//! Everything [`Upload`] hands back is complete on return: a [`Buffer`]'s
//! device address is valid at creation, and so is an
//! [`AccelerationStructure`]'s. Only the *contents* wait for the flush.

use ash::vk;

use crate::error::Result;
use crate::gpu::accel::{self, BuildJob};
use crate::gpu::{AccelerationStructure, Buffer, Context, MemoryLocation};

/// Bytes of staging plus scratch that trigger a flush, and so the bound on
/// an upload's transient memory (plus whichever resource crossed it).
///
/// A throughput knob, not a correctness one: any value uploads the same
/// bytes. Large enough that the per-submit constant disappears — a scene of
/// any size lands in single-digit chunks — and small enough to stay a
/// fraction of the residency being built.
const CHUNK_BYTES: vk::DeviceSize = 64 << 20;

/// One queued `vkCmdCopyBuffer`, as raw handles.
///
/// The destination is *borrowed* by handle: the [`Buffer`] it names was
/// handed to the caller, who must keep it alive until [`Upload::finish`].
struct Copy {
    src: vk::Buffer,
    dst: vk::Buffer,
    size: vk::DeviceSize,
}

/// A batched upload in progress: many buffer uploads and BLAS builds,
/// one submit per chunk instead of one per resource.
///
/// Queue work with [`Upload::buffer`] and [`Upload::blas`], then call
/// [`Upload::finish`]. Dropping one instead abandons whatever has not been
/// flushed yet — safe, and silent, because that is what an error unwinding
/// out of a half-built scene needs.
///
/// Everything handed out must outlive [`Upload::finish`], and a BLAS must
/// be queued *after* the buffers it reads: a flush between the two would
/// run the build before the copies.
pub struct Upload<'a> {
    gpu: &'a Context,
    /// Reset and re-recorded per flush, rather than created per submit —
    /// pool churn is part of what a submit costs.
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    /// Host-visible sources, alive until the flush that reads them returns.
    staging: Vec<Buffer>,
    copies: Vec<Copy>,
    builds: Vec<BuildJob>,
    /// Staging plus scratch currently queued — what `threshold` bounds.
    pending: vk::DeviceSize,
    threshold: vk::DeviceSize,
}

impl Context {
    /// Open a batched upload against this device. See [`Upload`].
    ///
    /// # Errors
    ///
    /// [`Error::Vulkan`](crate::Error) if the command pool, command buffer,
    /// or fence can't be created.
    pub fn upload(&self) -> Result<Upload<'_>> {
        Upload::new(self, CHUNK_BYTES)
    }
}

impl<'a> Upload<'a> {
    /// `threshold` is [`CHUNK_BYTES`] everywhere but the tests, which set it
    /// small enough to cross a chunk boundary without 64 MiB of data.
    fn new(gpu: &'a Context, threshold: vk::DeviceSize) -> Result<Self> {
        let device = gpu.device();
        let pool_info =
            vk::CommandPoolCreateInfo::default().queue_family_index(gpu.queue_family_index());
        let pool = unsafe { device.create_command_pool(&pool_info, None)? };
        // Past pool creation, failure has something to clean up.
        match Self::allocate(gpu, pool) {
            Ok((cmd, fence)) => Ok(Self {
                gpu,
                pool,
                cmd,
                fence,
                staging: Vec::new(),
                copies: Vec::new(),
                builds: Vec::new(),
                pending: 0,
                threshold,
            }),
            Err(err) => {
                unsafe { device.destroy_command_pool(pool, None) };
                Err(err)
            }
        }
    }

    fn allocate(gpu: &Context, pool: vk::CommandPool) -> Result<(vk::CommandBuffer, vk::Fence)> {
        let device = gpu.device();
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // Freed with the pool, so it needs no cleanup arm of its own.
        let cmd = unsafe { device.allocate_command_buffers(&alloc_info)? }[0];
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? };
        Ok((cmd, fence))
    }

    /// Create a device-local buffer holding `data` and queue the copy that
    /// fills it. `TRANSFER_DST` is added to `usage` automatically.
    ///
    /// The returned buffer is fully formed — size, usage, device address —
    /// but reads as undefined until [`Upload::finish`] returns, and must
    /// outlive that call. `name` picks the memory bucket, exactly as in
    /// [`Context::create_buffer`].
    ///
    /// # Errors
    ///
    /// [`Error::Vulkan`](crate::Error) or
    /// [`Error::Allocation`](crate::Error) from creating either buffer, or
    /// from the flush this call may trigger.
    pub fn buffer(
        &mut self,
        name: &str,
        data: &[u8],
        usage: vk::BufferUsageFlags,
    ) -> Result<Buffer> {
        let size = data.len() as vk::DeviceSize;
        let staging = self.gpu.staging_buffer(&format!("{name}.staging"), data)?;
        let buffer = self.gpu.create_buffer(
            name,
            size,
            usage | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
        )?;
        self.copies.push(Copy {
            src: staging.handle(),
            dst: buffer.handle(),
            size,
        });
        self.staging.push(staging);
        self.pending += size;
        self.flush_if_full()?;
        Ok(buffer)
    }

    /// Create a BLAS over `triangle_count` triangles and queue its build —
    /// [`Context::build_blas`], deferred.
    ///
    /// The structure is traceable once [`Upload::finish`] returns, and must
    /// outlive that call. Its geometry buffers may come from
    /// [`Upload::buffer`] on this same [`Upload`], and must have been
    /// queued before this call: within a chunk a barrier orders the copies
    /// before the builds, across chunks the flush's fence does.
    ///
    /// # Errors
    ///
    /// As [`Upload::buffer`], plus a failure to create the structure
    /// itself.
    pub fn blas(
        &mut self,
        name: &str,
        vertices: &Buffer,
        vertex_count: u32,
        indices: &Buffer,
        triangle_count: u32,
    ) -> Result<AccelerationStructure> {
        let (structure, job) =
            self.gpu
                .create_blas(name, vertices, vertex_count, indices, triangle_count)?;
        self.pending += job.scratch_size();
        self.builds.push(job);
        self.flush_if_full()?;
        Ok(structure)
    }

    /// Run everything still queued and block until the device has finished
    /// it. Every buffer and structure this upload handed out is complete
    /// when this returns.
    ///
    /// # Errors
    ///
    /// [`Error::Vulkan`](crate::Error) from recording, submitting, or
    /// waiting — a device fault, which ends the render either way.
    pub fn finish(mut self) -> Result<()> {
        self.flush()
    }

    /// Flush iff the queue has grown past the chunk threshold. Tested after
    /// queueing rather than before, so a single resource larger than the
    /// threshold still makes progress instead of wedging the load.
    fn flush_if_full(&mut self) -> Result<()> {
        if self.pending >= self.threshold {
            self.flush()
        } else {
            Ok(())
        }
    }

    fn flush(&mut self) -> Result<()> {
        if self.copies.is_empty() && self.builds.is_empty() {
            return Ok(());
        }
        let copies = std::mem::take(&mut self.copies);
        let builds = std::mem::take(&mut self.builds);
        let result = self.submit_chunk(&copies, &builds);
        // On the error path too: the submit blocked, so nothing is still
        // reading this chunk's staging or scratch, and a failing load should
        // not hold a chunk of host-visible memory while its caller unwinds.
        self.staging.clear();
        drop(builds);
        self.pending = 0;
        result
    }

    /// Record one chunk — every copy, then every build — into the reused
    /// command buffer, submit it, and wait.
    fn submit_chunk(&self, copies: &[Copy], builds: &[BuildJob]) -> Result<()> {
        let device = self.gpu.device();
        unsafe {
            device.reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())?;
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device.begin_command_buffer(self.cmd, &begin_info)?;
            for copy in copies {
                let region = vk::BufferCopy::default().size(copy.size);
                device.cmd_copy_buffer(self.cmd, copy.src, copy.dst, &[region]);
            }
            if !builds.is_empty() {
                // The one barrier a chunk needs: these builds read vertex and
                // index buffers the copies above wrote.
                if !copies.is_empty() {
                    barrier_before_builds(device, self.cmd);
                }
                accel::record_builds(&self.gpu.accel_loader, self.cmd, builds);
            }
            device.end_command_buffer(self.cmd)?;
            device.reset_fences(&[self.fence])?;
        }

        let buffers = [self.cmd];
        let submit_info = vk::SubmitInfo::default().command_buffers(&buffers);
        // Submit under the queue lock, wait with it released — the rule
        // every submission in `gpu` follows; see [`super::submit`]. The wait
        // is also what lets the next flush reset this pool and command
        // buffer instead of allocating new ones.
        self.gpu.queue.submit(device, &[submit_info], self.fence)?;
        unsafe { device.wait_for_fences(&[self.fence], true, u64::MAX)? };
        Ok(())
    }
}

impl Drop for Upload<'_> {
    fn drop(&mut self) {
        // No submission is ever in flight here — every flush waited — so
        // there is nothing to synchronize against before tearing down.
        // Staging, scratch, and any unflushed work go with the fields.
        unsafe {
            self.gpu.device().destroy_fence(self.fence, None);
            self.gpu.device().destroy_command_pool(self.pool, None);
        }
    }
}

/// Transfer writes visible to acceleration-structure builds.
///
/// `SHADER_READ` is the bit that matters and the easy one to omit: a
/// build's *inputs* — the vertex and index buffers these copies just wrote
/// — are read as shader data, not as acceleration structures.
/// `ACCELERATION_STRUCTURE_READ` covers only the structures a build reads,
/// which for a BLAS is nothing. Omitting it surfaces under synchronization
/// validation as a `READ_AFTER_WRITE` hazard on "vertex data".
fn barrier_before_builds(device: &ash::Device, cmd: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
        .dst_access_mask(
            vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
        );
    let info = vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&barrier));
    unsafe { device.cmd_pipeline_barrier2(cmd, &info) };
}

#[cfg(test)]
mod tests {
    use super::*;

    const PATTERN_BYTES: usize = 3072;
    const TEST_THRESHOLD: vk::DeviceSize = 4096;

    /// Bytes survive a batched upload **across a chunk boundary**, and the
    /// queue never holds more than the threshold plus the one resource that
    /// crossed it.
    ///
    /// The threshold is injected rather than met with 64 MiB of test data:
    /// what is under test is what happens at a boundary, not where the
    /// boundary happens to be. Six buffers over a 4 KiB threshold force
    /// several flushes, each carrying a distinct pattern, so a copy recorded
    /// into the wrong chunk — or a staging buffer freed while its copy was
    /// still queued — shows up as the wrong bytes rather than as nothing.
    #[test]
    fn a_batched_upload_survives_its_chunk_boundaries() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let patterns: Vec<Vec<u8>> = (0..6u8)
            .map(|index| {
                (0..u8::MAX)
                    .map(|byte| byte ^ index)
                    .cycle()
                    .take(PATTERN_BYTES)
                    .collect()
            })
            .collect();

        let mut upload = Upload::new(&gpu, TEST_THRESHOLD).expect("upload");
        let buffers: Vec<Buffer> = patterns
            .iter()
            .map(|data| {
                let buffer = upload
                    .buffer("test.chunked", data, vk::BufferUsageFlags::TRANSFER_SRC)
                    .expect("queue");
                // The gate fired if it needed to, so what is still queued —
                // and so the transient memory a chunk can hold, past the
                // resource that crossed the line — stays under the
                // threshold.
                assert!(
                    upload.pending < TEST_THRESHOLD,
                    "{} bytes still queued, past the {TEST_THRESHOLD}-byte threshold",
                    upload.pending
                );
                buffer
            })
            .collect();
        upload.finish().expect("finish");

        for (index, (buffer, data)) in buffers.iter().zip(&patterns).enumerate() {
            assert_eq!(
                &gpu.download_buffer(buffer).expect("download"),
                data,
                "buffer {index} came back wrong"
            );
        }
    }

    /// A BLAS built through the batcher traces like one built through
    /// [`Context::build_blas`]: the barrier between a chunk's copies and
    /// its builds is what makes the geometry readable, and nothing else
    /// would catch its absence.
    ///
    /// The TLAS at the end is a build that *reads* the BLAS, so an unbuilt
    /// one fails here rather than silently rendering nothing.
    #[test]
    fn a_batched_blas_is_traceable_when_finish_returns() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let positions: [f32; 12] = [
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,
        ];
        let triangles: [u32; 6] = [0, 1, 2, 0, 2, 3];
        let usage = vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;

        let mut upload = gpu.upload().expect("upload");
        let vertices = upload
            .buffer("test.blas.vertices", bytemuck::cast_slice(&positions), usage)
            .expect("vertices");
        let indices = upload
            .buffer("test.blas.indices", bytemuck::cast_slice(&triangles), usage)
            .expect("indices");
        let blas = upload
            .blas("test.blas", &vertices, 4, &indices, 2)
            .expect("blas");
        upload.finish().expect("finish");

        gpu.build_tlas(
            "test.blas.tlas",
            &[crate::gpu::TlasInstance {
                blas: &blas,
                transform: glam::Mat4::IDENTITY,
                custom_index: 0,
                mask: 0xFF,
                opaque: true,
            }],
        )
        .expect("a TLAS over the batched BLAS");
    }

    /// Dropping an upload with work still queued is safe and silent: the
    /// half-built scene an error unwinds out of must not submit, must not
    /// leak, and must not need the caller to remember anything.
    #[test]
    fn dropping_an_unfinished_upload_abandons_its_work() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let mut upload = gpu.upload().expect("upload");
        let buffer = upload
            .buffer(
                "test.abandoned",
                &[7u8; 1024],
                vk::BufferUsageFlags::TRANSFER_SRC,
            )
            .expect("queue");
        drop(upload);
        // The buffer outlives the upload that made it, and is still a
        // perfectly good (if undefined) buffer — only its contents never
        // arrived.
        assert_eq!(buffer.size(), 1024);
    }
}
