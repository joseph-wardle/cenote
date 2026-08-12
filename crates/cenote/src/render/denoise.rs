//! The publish path's denoise pass: OIDN over a resolved frame, in place.
//!
//! Where the driver allows it, the filter runs **in the frame's own
//! memory**: the publish slots are allocated exportable, OIDN imports each
//! one once, and a publish is two queue-ownership barriers around the
//! filter — nothing is copied anywhere. Otherwise the pixels take a host
//! round trip through persistent staging: three planes down in one submit,
//! filtered from the mapped bytes, the colour copied back. Staging grows to
//! the largest frame seen and is never given back, so a preview rectangle
//! rides in the front of the full-size allocation.

use std::time::{Duration, Instant};

use ash::vk;

use super::ResolveTargets;
use crate::denoise::{Denoiser, Imported, Quality};
use crate::error::Result;
use crate::gpu::{Buffer, Context, MemoryLocation};

/// The denoiser and its route to the frames, held across publishes by the
/// render thread.
pub(super) struct DenoisePass {
    denoiser: Denoiser,
    mode: Mode,
    /// What the last filter cost, and the rectangle it cost that over — see
    /// [`DenoisePass::cost`].
    last: Option<((u32, u32), Duration)>,
}

/// How the pixels reach OIDN.
enum Mode {
    /// The publish slots' own memory, imported once per slot — the filter
    /// runs where the resolve wrote.
    Shared([Option<Planes>; 2]),
    /// Host staging between the frame and OIDN's device buffers — the
    /// fallback when memory cannot be shared (the CPU device, a driver
    /// without fd export, an import the device refused).
    Staged(Box<Staging>),
}

/// The staging planes of [`Mode::Staged`].
struct Staging {
    /// The colour plane both ways: filled from the frame's beauty,
    /// filtered in place, copied back. `TRANSFER_SRC` for that reason.
    color: Buffer,
    albedo: Buffer,
    normal: Buffer,
    /// Bytes each staging buffer holds; zero until the first frame.
    capacity: vk::DeviceSize,
}

/// One publish slot's planes, shared with OIDN.
struct Planes {
    beauty: Imported,
    albedo: Imported,
    normal: Imported,
}

impl Planes {
    fn import(denoiser: &Denoiser, gpu: &Context, targets: &ResolveTargets) -> Result<Self> {
        let plane = |buffer: &Buffer| -> Result<Imported> {
            if !buffer.exportable() {
                // The exported allocation was refused at frame build time.
                return Err(crate::Error::Denoise(
                    "the frame's memory is not exportable".to_owned(),
                ));
            }
            let (fd, bytes) = gpu.export_fd(buffer)?;
            denoiser.import(fd, bytes as usize)
        };
        Ok(Self {
            beauty: plane(targets.beauty)?,
            albedo: plane(targets.albedo)?,
            normal: plane(targets.normal)?,
        })
    }
}

impl DenoisePass {
    /// Open the denoiser on the GPU the renderer is rendering on.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Denoise`] if OIDN has no usable device at all.
    pub(super) fn new(gpu: &Context) -> Result<Self> {
        let denoiser = Denoiser::new(gpu.device_uuid())?;
        let mode = if gpu.external_memory_fd() && denoiser.imports_memory() {
            log::debug!("denoising shares the frames' memory");
            Mode::Shared([None, None])
        } else {
            log::debug!("denoising through staging copies");
            staged(gpu)?
        };
        Ok(Self {
            denoiser,
            mode,
            last: None,
        })
    }

    /// What a filter over a `size` rectangle costs, if one has run at that
    /// size.
    ///
    /// The render loop divides a *frame* by the preview divisor, and with
    /// denoising on the filter is part of one; but the wave that renders the
    /// reference sample is not always a wave the publish gate lets filter, so
    /// the number is read back rather than returned. Keyed by size because
    /// that is what it depends on — the cost of the drag's quarter-resolution
    /// filter says nothing about the full-resolution frame the divisor is
    /// measured against.
    pub(super) fn cost(&self, size: (u32, u32)) -> Option<Duration> {
        self.last.and_then(|(at, cost)| (at == size).then_some(cost))
    }

    /// The publish slots were rebuilt: let go of the old memory, so the next
    /// publish imports the new. Without this a fresh import could land on a
    /// recycled pointer or, worse, the filter could keep reading freed
    /// frames.
    pub(super) fn retarget(&mut self) {
        if let Mode::Shared(slots) = &mut self.mode {
            *slots = [None, None];
            self.denoiser.invalidate();
        }
    }

    /// Filter the `width`×`height` image at the front of `targets` — publish
    /// slot `slot` — leaving the result in `targets.beauty`, and report what
    /// it cost. The guides are read, not written.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Vulkan`] / [`crate::Error::Allocation`] if the copies
    /// or the staging fail, [`crate::Error::Denoise`] if the filter does.
    pub(super) fn run(
        &mut self,
        gpu: &Context,
        targets: &ResolveTargets,
        slot: usize,
        width: u32,
        height: u32,
        quality: Quality,
    ) -> Result<Duration> {
        let started = Instant::now();
        // First touch of a slot imports it — or demotes the pass for good:
        // a driver that refuses one import will refuse them all.
        let mut import_failed = false;
        if let Mode::Shared(slots) = &mut self.mode
            && slots[slot].is_none()
        {
            match Planes::import(&self.denoiser, gpu, targets) {
                Ok(planes) => slots[slot] = Some(planes),
                Err(error) => {
                    log::warn!(
                        "cannot share frame memory with OIDN — denoising through \
                         staging copies: {error}"
                    );
                    import_failed = true;
                }
            }
        }
        if import_failed {
            self.denoiser.invalidate();
            self.mode = staged(gpu)?;
        }
        match &mut self.mode {
            Mode::Shared(slots) => {
                let planes = slots[slot].as_ref().expect("imported above");
                let shared = [targets.beauty, targets.albedo, targets.normal];
                gpu.release_to_external(&shared)?;
                self.denoiser.denoise_imported(
                    width,
                    height,
                    quality,
                    &planes.beauty,
                    &planes.albedo,
                    &planes.normal,
                )?;
                gpu.acquire_from_external(&shared)?;
            }
            Mode::Staged(staging) => {
                let Staging {
                    color,
                    albedo,
                    normal,
                    capacity,
                } = staging.as_mut();
                let bytes = vk::DeviceSize::from(width) * vk::DeviceSize::from(height) * 16;
                if bytes > *capacity {
                    *color = plane(gpu, "denoise.color", bytes, true)?;
                    *albedo = plane(gpu, "denoise.albedo", bytes, false)?;
                    *normal = plane(gpu, "denoise.normal", bytes, false)?;
                    *capacity = bytes;
                }
                gpu.copy_buffers(&[
                    (targets.beauty, color, bytes),
                    (targets.albedo, albedo, bytes),
                    (targets.normal, normal, bytes),
                ])?;
                // Mapped Vulkan memory is aligned to the buffer's memory
                // requirements, which no implementation reports below a word.
                let floats = bytes as usize / size_of::<f32>();
                self.denoiser.denoise(
                    width,
                    height,
                    quality,
                    &mut bytemuck::cast_slice_mut(color.mapped_mut())[..floats],
                    &bytemuck::cast_slice(albedo.mapped())[..floats],
                    &bytemuck::cast_slice(normal.mapped())[..floats],
                )?;
                gpu.copy_buffers(&[(&*color, targets.beauty, bytes)])?;
            }
        }
        let cost = started.elapsed();
        self.last = Some(((width, height), cost));
        Ok(cost)
    }
}

/// The staging fallback, before any frame has named a size — Vulkan forbids
/// zero-sized buffers, so the placeholders are one byte.
fn staged(gpu: &Context) -> Result<Mode> {
    Ok(Mode::Staged(Box::new(Staging {
        color: plane(gpu, "denoise.staging", 1, false)?,
        albedo: plane(gpu, "denoise.staging", 1, false)?,
        normal: plane(gpu, "denoise.staging", 1, false)?,
        capacity: 0,
    })))
}

/// A host-visible staging plane. `readable` also gives it `TRANSFER_SRC`,
/// for the one plane that goes back to the device.
fn plane(gpu: &Context, name: &str, bytes: vk::DeviceSize, readable: bool) -> Result<Buffer> {
    let mut usage = vk::BufferUsageFlags::TRANSFER_DST;
    if readable {
        usage |= vk::BufferUsageFlags::TRANSFER_SRC;
    }
    gpu.create_buffer(name, bytes, usage, MemoryLocation::GpuToCpu)
}
