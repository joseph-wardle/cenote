//! One-shot command submission: record, submit, block on a fence.
//! M0's workload is strictly sequential, so there is no frames-in-
//! flight machinery and no semaphores — the fence wait is the sync model.
//! Cross-submission memory visibility is free with this shape: fence signal
//! makes all device writes available, and the next queue submission makes
//! available memory visible.
//!
//! M1's stage scheduler will replace this for the render loop; uploads and
//! readbacks keep using it.

use ash::vk;

use crate::error::Result;
use crate::gpu::Context;

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
}
