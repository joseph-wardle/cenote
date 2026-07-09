//! RAII GPU buffers and the staging upload/readback paths.
//!
//! [`Buffer`] owns its `vk::Buffer` plus allocation and frees both on drop
//! (buffers must be dropped before their [`Context`]). Device-local data
//! moves through transient staging buffers and the blocking one-shot submit
//! — no persistent staging ring until something needs one.

use std::mem::ManuallyDrop;
use std::sync::{Arc, Mutex};

use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};

use crate::error::Result;
use crate::gpu::Context;

pub use gpu_allocator::MemoryLocation;

/// A `vk::Buffer` bound to its memory, freed on drop.
pub struct Buffer {
    handle: vk::Buffer,
    allocation: ManuallyDrop<Allocation>,
    size: vk::DeviceSize,
    /// Set iff created with `SHADER_DEVICE_ADDRESS` usage.
    address: Option<vk::DeviceAddress>,
    device: ash::Device,
    allocator: Arc<Mutex<Allocator>>,
}

impl Buffer {
    /// The raw handle, for recording commands against. Stays inside `gpu`:
    /// the quarantine boundary.
    #[must_use]
    pub(super) fn handle(&self) -> vk::Buffer {
        self.handle
    }

    /// Size in bytes.
    #[must_use]
    pub fn size(&self) -> vk::DeviceSize {
        self.size
    }

    /// The buffer's GPU address, for kernels that reach it through a
    /// push-constant pointer.
    ///
    /// # Panics
    ///
    /// If the buffer was created without `SHADER_DEVICE_ADDRESS` usage —
    /// a programmer bug, not an environment failure.
    #[must_use]
    pub fn device_address(&self) -> vk::DeviceAddress {
        self.address
            .expect("buffer was created without SHADER_DEVICE_ADDRESS usage")
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe { self.device.destroy_buffer(self.handle, None) };
        let allocation = unsafe { ManuallyDrop::take(&mut self.allocation) };
        free_allocation(&self.allocator, allocation, "buffer");
    }
}

/// Return an allocation to the shared allocator, logging rather than
/// panicking on failure — a `Drop` path can't propagate an error, and a
/// leak is preferable to unwinding through it. `what` names the resource in
/// those logs. Shared by every `gpu` type that frees an allocation on drop.
pub(super) fn free_allocation(
    allocator: &Arc<Mutex<Allocator>>,
    allocation: Allocation,
    what: &str,
) {
    match allocator.lock() {
        Ok(mut allocator) => {
            if let Err(err) = allocator.free(allocation) {
                log::error!("failed to free {what} allocation: {err}");
            }
        }
        Err(_) => log::error!("allocator mutex poisoned — leaking {what} allocation"),
    }
}

impl Context {
    /// Create a buffer of `size` bytes. `name` labels the allocation in
    /// gpu-allocator's bookkeeping and leak reports.
    ///
    /// # Errors
    ///
    /// [`Error::Vulkan`] on buffer creation/bind failure, [`Error::Allocation`]
    /// if memory can't be allocated.
    pub fn create_buffer(
        &self,
        name: &str,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> Result<Buffer> {
        let device = self.device();
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { device.create_buffer(&info, None)? };

        let result = self.allocate_and_bind(name, buffer, location);
        match result {
            Ok(allocation) => {
                let address = usage
                    .contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
                    .then(|| {
                        let info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
                        unsafe { device.get_buffer_device_address(&info) }
                    });
                Ok(Buffer {
                    handle: buffer,
                    allocation: ManuallyDrop::new(allocation),
                    size,
                    address,
                    device: device.clone(),
                    allocator: self.allocator_handle(),
                })
            }
            Err(err) => {
                unsafe { device.destroy_buffer(buffer, None) };
                Err(err)
            }
        }
    }

    fn allocate_and_bind(
        &self,
        name: &str,
        buffer: vk::Buffer,
        location: MemoryLocation,
    ) -> Result<Allocation> {
        let device = self.device();
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let allocation = self
            .allocator_handle()
            .lock()
            // Poison means another thread already panicked mid-allocation —
            // a programmer bug, so panicking (not Err) is the honest shape.
            .expect("allocator mutex poisoned")
            .allocate(&AllocationCreateDesc {
                name,
                requirements,
                location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })?;
        unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())? };
        Ok(allocation)
    }

    /// Create a device-local buffer holding `data`, moved through a transient
    /// staging buffer. `TRANSFER_DST` is added to `usage` automatically.
    ///
    /// # Errors
    ///
    /// As [`Context::create_buffer`], plus [`Error::Vulkan`] from the copy
    /// submission.
    ///
    /// # Panics
    ///
    /// Only if the allocator breaks its contract that `CpuToGpu` memory is
    /// host-mapped — a bug, not an environment failure.
    pub fn upload_buffer(
        &self,
        name: &str,
        data: &[u8],
        usage: vk::BufferUsageFlags,
    ) -> Result<Buffer> {
        let size = data.len() as vk::DeviceSize;
        let staging = self.staging_buffer(&format!("{name}.staging"), data)?;
        let buffer = self.create_buffer(
            name,
            size,
            usage | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
        )?;
        self.copy_buffer(&staging, &buffer, size)?;
        Ok(buffer)
    }

    /// A transient `CpuToGpu` staging buffer pre-filled with `data` — the
    /// front half of every upload, buffer and image alike.
    pub(super) fn staging_buffer(&self, name: &str, data: &[u8]) -> Result<Buffer> {
        let mut staging = self.create_buffer(
            name,
            data.len() as vk::DeviceSize,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        )?;
        staging
            .allocation
            .mapped_slice_mut()
            .expect("CpuToGpu memory is always mapped")[..data.len()]
            .copy_from_slice(data);
        Ok(staging)
    }

    /// Read a buffer's full contents back to the host through a transient
    /// staging buffer. The source must have `TRANSFER_SRC` usage.
    ///
    /// # Errors
    ///
    /// As [`Context::create_buffer`], plus [`Error::Vulkan`] from the copy
    /// submission.
    ///
    /// # Panics
    ///
    /// Only if the allocator breaks its contract that `GpuToCpu` memory is
    /// host-mapped — a bug, not an environment failure.
    pub fn download_buffer(&self, buffer: &Buffer) -> Result<Vec<u8>> {
        let staging = self.create_buffer(
            "download.staging",
            buffer.size(),
            vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuToCpu,
        )?;
        self.copy_buffer(buffer, &staging, buffer.size())?;
        // The mapped slice spans the whole allocation, which the allocator
        // may pad past the requested size — return exactly the buffer.
        Ok(staging
            .allocation
            .mapped_slice()
            .expect("GpuToCpu memory is always mapped")[..buffer.size() as usize]
            .to_vec())
    }

    fn copy_buffer(&self, src: &Buffer, dst: &Buffer, size: vk::DeviceSize) -> Result<()> {
        self.submit_once(|device, cmd| {
            let region = vk::BufferCopy::default().size(size);
            unsafe { device.cmd_copy_buffer(cmd, src.handle(), dst.handle(), &[region]) };
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bytes survive host → device-local → host.
    #[test]
    fn buffer_upload_download_round_trip() {
        let Some(context) = crate::gpu::test_context() else {
            return;
        };
        let data: Vec<u8> = (0..u8::MAX).cycle().take(64 * 1024).collect();
        let buffer = context
            .upload_buffer("roundtrip", &data, vk::BufferUsageFlags::TRANSFER_SRC)
            .expect("upload");
        let readback = context.download_buffer(&buffer).expect("download");
        assert_eq!(data, readback);
    }
}
