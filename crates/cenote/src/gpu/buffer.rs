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
    backing: Backing,
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

/// The memory behind a [`Buffer`].
enum Backing {
    /// A gpu-allocator suballocation — every ordinary buffer.
    Managed(ManuallyDrop<Allocation>),
    /// A dedicated allocation another API may import
    /// ([`Context::create_exported_buffer`]). `bytes` is the allocation's
    /// full size — what the importer must be told, since it may exceed what
    /// was asked for.
    Exported {
        memory: vk::DeviceMemory,
        bytes: vk::DeviceSize,
    },
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

    /// Whether another API may import this buffer's memory — created by
    /// [`Context::create_exported_buffer`], handed out by
    /// [`Context::export_fd`].
    #[must_use]
    pub fn exportable(&self) -> bool {
        matches!(self.backing, Backing::Exported { .. })
    }

    /// The buffer's bytes as the host sees them — host-visible memory
    /// ([`MemoryLocation::CpuToGpu`] or [`MemoryLocation::GpuToCpu`]) only.
    /// Exactly [`Buffer::size`] bytes: the allocator may pad past what was
    /// asked for, and that tail belongs to no one.
    #[must_use]
    pub fn mapped(&self) -> &[u8] {
        &self
            .allocation()
            .mapped_slice()
            .expect("buffer is not host-visible")[..self.size as usize]
    }

    /// [`Buffer::mapped`], to write into. The GPU must not be reading the
    /// buffer meanwhile — every submit in this module blocks, so "after the
    /// copy returned" is enough.
    #[must_use]
    pub fn mapped_mut(&mut self) -> &mut [u8] {
        let size = self.size as usize;
        let Backing::Managed(allocation) = &mut self.backing else {
            panic!("buffer is not host-visible");
        };
        &mut allocation
            .mapped_slice_mut()
            .expect("buffer is not host-visible")[..size]
    }

    /// The gpu-allocator allocation — exported buffers have none.
    fn allocation(&self) -> &Allocation {
        let Backing::Managed(allocation) = &self.backing else {
            panic!("buffer is not host-visible");
        };
        allocation
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        self.ledger.remove(self.bucket, self.size);
        unsafe { self.device.destroy_buffer(self.handle, None) };
        match &mut self.backing {
            Backing::Managed(allocation) => {
                let allocation = unsafe { ManuallyDrop::take(allocation) };
                free_allocation(&self.allocator, allocation, "buffer");
            }
            Backing::Exported { memory, .. } => unsafe { self.device.free_memory(*memory, None) },
        }
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
                    backing: Backing::Managed(ManuallyDrop::new(allocation)),
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
        staging.mapped_mut()[..data.len()].copy_from_slice(data);
        Ok(staging)
    }

    /// Copy the leading bytes of each `(from, to)` pair, all in one blocking
    /// submit. Sources need `TRANSFER_SRC` usage and destinations
    /// `TRANSFER_DST`; a length past either end is a panic, not a truncation.
    ///
    /// One submit rather than one per copy because a submit costs about as
    /// much as a small copy does.
    pub fn copy_buffers(&self, copies: &[(&Buffer, &Buffer, vk::DeviceSize)]) -> Result<()> {
        for (from, to, bytes) in copies {
            assert!(
                *bytes <= from.size() && *bytes <= to.size(),
                "copy of {bytes} bytes past the end of a buffer"
            );
        }
        self.submit_once(|device, cmd| {
            for (from, to, bytes) in copies {
                let region = vk::BufferCopy::default().size(*bytes);
                unsafe { device.cmd_copy_buffer(cmd, from.handle(), to.handle(), &[region]) };
            }
        })
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
        self.copy_buffers(&[(buffer, &staging, buffer.size())])?;
        Ok(staging.mapped().to_vec())
    }

    /// A device-local buffer whose memory another API on this GPU can
    /// import ([`Context::export_fd`]) — a dedicated allocation, since
    /// exported memory cannot be suballocated. Requires
    /// [`Context::external_memory_fd`]; a driver may still refuse, so
    /// callers fall back to [`Context::create_buffer`].
    ///
    /// # Errors
    ///
    /// [`crate::Error::Vulkan`] if creation, allocation, or binding fails.
    pub fn create_exported_buffer(
        &self,
        name: &str,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> Result<Buffer> {
        let device = self.device();
        let mut external = vk::ExternalMemoryBufferCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut external);
        let buffer = unsafe { device.create_buffer(&info, None)? };
        match self.allocate_exported(buffer) {
            Ok((memory, bytes)) => {
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
                    backing: Backing::Exported { memory, bytes },
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

    fn allocate_exported(&self, buffer: vk::Buffer) -> Result<(vk::DeviceMemory, vk::DeviceSize)> {
        let device = self.device();
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let properties = unsafe {
            self.instance()
                .get_physical_device_memory_properties(self.physical_device())
        };
        let index = (0..properties.memory_type_count)
            .find(|&i| {
                requirements.memory_type_bits & (1 << i) != 0
                    && properties.memory_types[i as usize]
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .ok_or(crate::Error::Vulkan(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY))?;
        let mut flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let mut export = vk::ExportMemoryAllocateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
        let info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(index)
            .push_next(&mut flags)
            .push_next(&mut export)
            .push_next(&mut dedicated);
        let memory = unsafe { device.allocate_memory(&info, None)? };
        if let Err(err) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
            unsafe { device.free_memory(memory, None) };
            return Err(err.into());
        }
        Ok((memory, requirements.size))
    }

    /// A POSIX fd another API imports to reach `buffer`'s memory, plus the
    /// allocation's full byte size, which the importer must be told exactly.
    /// The fd is the caller's to close — importing it usually does.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Vulkan`] if the driver refuses the export.
    ///
    /// # Panics
    ///
    /// If `buffer` was not created by [`Context::create_exported_buffer`].
    pub fn export_fd(&self, buffer: &Buffer) -> Result<(i32, vk::DeviceSize)> {
        let Backing::Exported { memory, bytes } = buffer.backing else {
            panic!("buffer was not created exportable");
        };
        let loader = ash::khr::external_memory_fd::Device::new(self.instance(), self.device());
        let info = vk::MemoryGetFdInfoKHR::default()
            .memory(memory)
            .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        let fd = unsafe { loader.get_memory_fd(&info)? };
        Ok((fd, bytes))
    }

    /// Hand ownership of `buffers` to the external queue family. After the
    /// submit fences — this blocks until it has — another API on this GPU
    /// may read and write them, and every Vulkan write before the call is
    /// visible to it.
    pub fn release_to_external(&self, buffers: &[&Buffer]) -> Result<()> {
        self.external_ownership(buffers, false)
    }

    /// Take ownership of `buffers` back from the external queue family,
    /// making the other API's writes visible to everything after. The other
    /// API's work must have finished before the call.
    pub fn acquire_from_external(&self, buffers: &[&Buffer]) -> Result<()> {
        self.external_ownership(buffers, true)
    }

    fn external_ownership(&self, buffers: &[&Buffer], acquire: bool) -> Result<()> {
        let family = self.queue_family_index();
        self.submit_once(|device, cmd| {
            let barriers: Vec<vk::BufferMemoryBarrier2> = buffers
                .iter()
                .map(|buffer| {
                    // Only the half the operation defines: the spec ignores
                    // the destination scope of a release and the source
                    // scope of an acquire.
                    let barrier = vk::BufferMemoryBarrier2::default()
                        .buffer(buffer.handle())
                        .offset(0)
                        .size(vk::WHOLE_SIZE);
                    if acquire {
                        barrier
                            .src_queue_family_index(vk::QUEUE_FAMILY_EXTERNAL)
                            .dst_queue_family_index(family)
                            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                            .dst_access_mask(
                                vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
                            )
                    } else {
                        barrier
                            .src_queue_family_index(family)
                            .dst_queue_family_index(vk::QUEUE_FAMILY_EXTERNAL)
                            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                    }
                })
                .collect();
            let info = vk::DependencyInfo::default().buffer_memory_barriers(&barriers);
            unsafe { device.cmd_pipeline_barrier2(cmd, &info) };
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
