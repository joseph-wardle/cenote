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
use crate::gpu::ledger::{Bucket, Ledger};

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
    /// The memory ledger and this buffer's bucket in it, so the bytes are
    /// counted out on drop exactly as they were counted in.
    ledger: Arc<Ledger>,
    bucket: Bucket,
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
    #[must_use]
    pub fn device_address(&self) -> vk::DeviceAddress {
        self.address
            .expect("buffer was created without SHADER_DEVICE_ADDRESS usage")
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        self.ledger.remove(self.bucket, self.size);
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
    /// gpu-allocator's bookkeeping and leak reports — **and picks its
    /// memory bucket**, so it is load-bearing, not decorative: see
    /// [`super::ledger`] for the dotted prefixes and what each one means.
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
                let ledger = self.ledger_handle();
                let bucket = ledger.add(name, size);
                Ok(Buffer {
                    handle: buffer,
                    allocation: ManuallyDrop::new(allocation),
                    size,
                    address,
                    device: device.clone(),
                    allocator: self.allocator_handle(),
                    ledger,
                    bucket,
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
    /// staging buffer and blocking until it arrives. `TRANSFER_DST` is added
    /// to `usage` automatically.
    ///
    /// One buffer per submit: for many at once, open an [`Upload`](super::Upload)
    /// directly, which is this on a batch.
    pub fn upload_buffer(
        &self,
        name: &str,
        data: &[u8],
        usage: vk::BufferUsageFlags,
    ) -> Result<Buffer> {
        let mut upload = self.upload()?;
        let buffer = upload.buffer(name, data, usage)?;
        upload.finish()?;
        Ok(buffer)
    }

    /// A transient `CpuToGpu` staging buffer pre-filled with `data` — the
    /// front half of every upload, buffer and image alike.
    pub(super) fn staging_buffer(&self, name: &str, data: &[u8]) -> Result<Buffer> {
        // Vulkan forbids zero-sized buffers; callers with possibly-empty
        // data pad to one unread record rather than skipping the upload.
        assert!(!data.is_empty(), "cannot upload an empty buffer ({name})");
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
    pub fn download_buffer(&self, buffer: &Buffer) -> Result<Vec<u8>> {
        let staging = self.create_buffer(
            "download.staging",
            buffer.size(),
            vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuToCpu,
        )?;
        self.submit_once(|device, cmd| {
            let region = vk::BufferCopy::default().size(buffer.size());
            unsafe { device.cmd_copy_buffer(cmd, buffer.handle(), staging.handle(), &[region]) };
        })?;
        // The mapped slice spans the whole allocation, which the allocator
        // may pad past the requested size — return exactly the buffer.
        Ok(staging
            .allocation
            .mapped_slice()
            .expect("GpuToCpu memory is always mapped")[..buffer.size() as usize]
            .to_vec())
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
