//! OIDN denoising of the film's linear averages.
//!
//! Two ways in. [`Denoiser::denoise`] filters host slices through OIDN's
//! own device buffers — the CLI's path, and the fallback. [`Imported`]
//! memory shared from Vulkan lets [`Denoiser::denoise_imported`] filter the
//! renderer's buffers where they lie, no copies at all — the render
//! session's path (`render::denoise` owns the Vulkan half).
//!
//! Three choices here are not the obvious ones:
//!
//! * **The device is opened by the Vulkan GPU's UUID**, not by type. On a
//!   two-GPU machine "a CUDA device" need not be the one Vulkan chose, and
//!   naming a backend would strand every non-NVIDIA card on the CPU —
//!   which is ~40× slower, the difference between a filter and a stall.
//! * **The buffers are `OIDN_STORAGE_DEVICE`.** `oidnNewBuffer`'s default
//!   is managed memory *or pinned host where managed is unsupported*, and
//!   the pinned-host path measures 568 ms against 25.7 at 1440p. Device
//!   storage is supported everywhere and has no such cliff.
//! * **The C API, not the `oidn` crate's wrapper.** The film's texels are
//!   RGBA and OIDN's images are RGB; only [`sys::oidnSetFilterImage`]
//!   takes the 16-byte pixel stride that reads them as they are — the
//!   wrapper's tight stride would force a host repack. Binding `color` and
//!   `output` to one buffer filters in place, which is what leaves alpha
//!   untouched.
//!
//! The guides are not declared `cleanAux`: OIDN's prescribed prefilter is
//! a further pass over each, and the default weights are trained for noisy
//! guides while ours are near-clean — the safe side of that mismatch.

use crate::{Error, Result};
use oidn::sys;

/// Filter quality, mirroring OIDN's knob: at 1440p the filter runs 10.1 ms
/// at `Fast` against 19.9 at `High`. Switching costs one ~12 ms recommit
/// the first time a tier is used and nothing thereafter, so a session may
/// move between them freely.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quality {
    /// Cheapest — for frames that will be replaced within one sample.
    Fast,
    /// Interactive — the viewer's preview cadence.
    Balanced,
    /// Final frame — the CLI's batch output.
    High,
}

impl Quality {
    /// OIDN's enumerant for this tier.
    fn enumerant(self) -> i32 {
        match self {
            Self::Fast => sys::OIDNQuality_OIDN_QUALITY_FAST,
            Self::Balanced => sys::OIDNQuality_OIDN_QUALITY_BALANCED,
            Self::High => sys::OIDNQuality_OIDN_QUALITY_HIGH,
        }
    }
}

/// An `OpenImageDenoise` device, one RT filter, and the device buffers
/// behind it, all reused across frames.
pub struct Denoiser {
    device: sys::OIDNDevice,
    filter: sys::OIDNFilter,
    /// Opened on the GPU (by UUID) rather than the CPU fallback — the
    /// precondition for sharing memory with Vulkan.
    on_gpu: bool,
    /// Input *and* output: see the module doc on in-place filtering.
    color: sys::OIDNBuffer,
    albedo: sys::OIDNBuffer,
    normal: sys::OIDNBuffer,
    /// Bytes each buffer holds. Grown to the largest frame seen and never
    /// shrunk — binding a smaller image onto a larger buffer is free and
    /// filters at the smaller image's cost, while reallocating is 6 ms a
    /// side. Zero means nothing is allocated yet.
    capacity: usize,
    /// What the filter is committed for — rectangle, quality, and the
    /// buffers its images point at — so an unchanged frame skips the
    /// commit.
    committed: Option<(u32, u32, Quality, [sys::OIDNBuffer; 3])>,
}

/// Another API's memory, imported for [`Denoiser::denoise_imported`] to
/// filter in place. Released on drop; OIDN keeps its device alive.
pub struct Imported(sys::OIDNBuffer);

impl Drop for Imported {
    fn drop(&mut self) {
        unsafe { sys::oidnReleaseBuffer(self.0) };
    }
}

impl Denoiser {
    /// Open OIDN on the GPU with this UUID — [`crate::gpu::Context::device_uuid`],
    /// so the filter runs where the renderer does. Falls back to the CPU
    /// device, loudly: landing there means an OIDN install without the
    /// module for this GPU, and it is ~40× slower.
    ///
    /// # Errors
    ///
    /// [`Error::Denoise`] when no device comes up at all.
    pub fn new(gpu_uuid: [u8; 16]) -> Result<Self> {
        let mut device = unsafe { sys::oidnNewDeviceByUUID(gpu_uuid.as_ptr().cast()) };
        let on_gpu = !device.is_null();
        if !on_gpu {
            log::warn!(
                "no OIDN device for this GPU — denoising on the CPU, ~40× slower; \
                 install an official OIDN release (see the README)"
            );
            device = unsafe { sys::oidnNewDevice(sys::OIDNDeviceType_OIDN_DEVICE_TYPE_CPU) };
            if device.is_null() {
                return Err(Error::Denoise("OIDN has no usable device".to_owned()));
            }
        }
        unsafe { sys::oidnCommitDevice(device) };
        let filter = unsafe { sys::oidnNewFilter(device, c"RT".as_ptr()) };
        // HDR linear input, set once: the film's averages are radiance, and
        // no view transform has touched them. (`srgb` is LDR-only, so under
        // `hdr` its default is the only meaning it has.)
        unsafe { sys::oidnSetFilterBool(filter, c"hdr".as_ptr(), true) };
        let denoiser = Self {
            device,
            filter,
            on_gpu,
            color: std::ptr::null_mut(),
            albedo: std::ptr::null_mut(),
            normal: std::ptr::null_mut(),
            capacity: 0,
            committed: None,
        };
        denoiser.check()?;
        Ok(denoiser)
    }

    /// Whether [`Denoiser::import`] can work at all: the device is the GPU
    /// itself and it accepts POSIX-fd memory. A `false` costs the caller a
    /// staging round trip, never correctness.
    #[must_use]
    pub fn imports_memory(&self) -> bool {
        const OPAQUE_FD: i32 =
            sys::OIDNExternalMemoryTypeFlag_OIDN_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_FD;
        self.on_gpu
            && unsafe { sys::oidnGetDeviceInt(self.device, c"externalMemoryTypes".as_ptr()) }
                & OPAQUE_FD
                != 0
    }

    /// Import `bytes` of another API's memory through the POSIX fd `fd` —
    /// the full allocation, not a slice of it. A successful import owns the
    /// fd; on failure it is closed here.
    ///
    /// # Errors
    ///
    /// [`Error::Denoise`] when the device refuses the import.
    pub fn import(&self, fd: i32, bytes: usize) -> Result<Imported> {
        let buffer = unsafe {
            sys::oidnNewSharedBufferFromFD(
                self.device,
                sys::OIDNExternalMemoryTypeFlag_OIDN_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_FD,
                fd,
                bytes,
            )
        };
        if buffer.is_null() {
            #[cfg(unix)]
            {
                use std::os::fd::FromRawFd;
                drop(unsafe { std::os::fd::OwnedFd::from_raw_fd(fd) });
            }
            self.check()?;
            return Err(Error::Denoise("OIDN refused the imported memory".to_owned()));
        }
        self.check()?;
        Ok(Imported(buffer))
    }

    /// Denoise a `width`×`height` frame **in place in another API's
    /// memory** — RGBA f32 planes laid out exactly as [`Denoiser::denoise`]
    /// takes them, already imported. The caller owns synchronization: the
    /// producer's writes must be visible before the call, and this returns
    /// only when the filter has finished.
    ///
    /// # Errors
    ///
    /// [`Error::Denoise`] when the filter setup or execution fails.
    pub fn denoise_imported(
        &mut self,
        width: u32,
        height: u32,
        quality: Quality,
        beauty: &Imported,
        albedo: &Imported,
        normal: &Imported,
    ) -> Result<()> {
        self.commit(width, height, quality, [beauty.0, albedo.0, normal.0])?;
        unsafe { sys::oidnExecuteFilter(self.filter) };
        self.check()
    }

    /// Forget what the filter is committed to. Callers that release
    /// [`Imported`] memory must call this: a fresh import can land on a
    /// recycled pointer, and the stale commit would pass for current.
    pub fn invalidate(&mut self) {
        self.committed = None;
    }

    /// Denoise one frame of linear RGBA averages **in place**, guided by
    /// the film's albedo and normal AOVs. Alpha survives untouched — OIDN
    /// filters colour only, and it rides the round trip in the lane the
    /// stride skips.
    ///
    /// # Errors
    ///
    /// [`Error::Denoise`] carrying OIDN's diagnostic when the filter setup
    /// or execution fails.
    pub fn denoise(
        &mut self,
        width: u32,
        height: u32,
        quality: Quality,
        beauty: &mut [f32],
        albedo: &[f32],
        normal: &[f32],
    ) -> Result<()> {
        let texels = width as usize * height as usize;
        assert_eq!(beauty.len(), texels * 4, "beauty must be RGBA per pixel");
        assert_eq!(albedo.len(), texels * 4, "albedo must be RGBA per pixel");
        assert_eq!(normal.len(), texels * 4, "normal must be RGBA per pixel");

        let bytes = texels * 4 * size_of::<f32>();
        if bytes > self.capacity {
            self.grow(bytes)?;
        }
        self.commit(width, height, quality, [self.color, self.albedo, self.normal])?;
        unsafe {
            // Whole RGBA slices at memcpy speed; the filter reads three of
            // every four floats on the device, where the stride is free.
            sys::oidnWriteBuffer(self.color, 0, bytes, beauty.as_ptr().cast());
            sys::oidnWriteBuffer(self.albedo, 0, bytes, albedo.as_ptr().cast());
            sys::oidnWriteBuffer(self.normal, 0, bytes, normal.as_ptr().cast());
            sys::oidnExecuteFilter(self.filter);
            sys::oidnReadBuffer(self.color, 0, bytes, beauty.as_mut_ptr().cast());
        }
        self.check()
    }

    /// Cut buffers for `bytes` each, releasing the smaller ones they
    /// replace. The filter's images point at the old buffers afterwards,
    /// so this invalidates the commit.
    fn grow(&mut self, bytes: usize) -> Result<()> {
        self.release_buffers();
        self.committed = None;
        let cut = || unsafe {
            sys::oidnNewBufferWithStorage(self.device, bytes, sys::OIDNStorage_OIDN_STORAGE_DEVICE)
        };
        (self.color, self.albedo, self.normal) = (cut(), cut(), cut());
        if self.color.is_null() || self.albedo.is_null() || self.normal.is_null() {
            self.check()?;
            return Err(Error::Denoise(format!(
                "OIDN refused {bytes} bytes of device memory"
            )));
        }
        self.capacity = bytes;
        self.check()
    }

    /// Point the filter at this rectangle of `[color, albedo, normal]`, at
    /// this quality — in place, so `color` is the output too. Skips the
    /// work when nothing changed.
    fn commit(
        &mut self,
        width: u32,
        height: u32,
        quality: Quality,
        buffers: [sys::OIDNBuffer; 3],
    ) -> Result<()> {
        if self.committed == Some((width, height, quality, buffers)) {
            return Ok(());
        }
        let [color, albedo, normal] = buffers;
        for (name, buffer) in [
            (c"color", color),
            (c"output", color),
            (c"albedo", albedo),
            (c"normal", normal),
        ] {
            unsafe {
                sys::oidnSetFilterImage(
                    self.filter,
                    name.as_ptr(),
                    buffer,
                    sys::OIDNFormat_OIDN_FORMAT_FLOAT3,
                    width as usize,
                    height as usize,
                    0,
                    4 * size_of::<f32>(),
                    0,
                );
            }
        }
        unsafe {
            sys::oidnSetFilterInt(self.filter, c"quality".as_ptr(), quality.enumerant());
            sys::oidnCommitFilter(self.filter);
        }
        self.check()?;
        self.committed = Some((width, height, quality, buffers));
        Ok(())
    }

    /// Drain OIDN's error queue into [`Error::Denoise`].
    fn check(&self) -> Result<()> {
        let mut message = std::ptr::null();
        let code = unsafe { sys::oidnGetDeviceError(self.device, &raw mut message) };
        if code == sys::OIDNError_OIDN_ERROR_NONE {
            return Ok(());
        }
        let text = if message.is_null() {
            format!("OIDN error {code}")
        } else {
            unsafe { std::ffi::CStr::from_ptr(message) }
                .to_string_lossy()
                .into_owned()
        };
        Err(Error::Denoise(text))
    }

    /// Release the buffers, if any are cut. `grow` calls this before
    /// replacing them, and it is the first thing [`Drop`] does — a buffer
    /// outliving its filter is undefined.
    fn release_buffers(&mut self) {
        if self.capacity > 0 {
            unsafe {
                sys::oidnReleaseBuffer(self.color);
                sys::oidnReleaseBuffer(self.albedo);
                sys::oidnReleaseBuffer(self.normal);
            }
            self.capacity = 0;
        }
    }
}

impl Drop for Denoiser {
    fn drop(&mut self) {
        self.release_buffers();
        unsafe {
            // Null when `new` errored between the two creations.
            if !self.filter.is_null() {
                sys::oidnReleaseFilter(self.filter);
            }
            sys::oidnReleaseDevice(self.device);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// No GPU has an all-zero UUID, so this takes the CPU fallback — the
    /// one path that runs everywhere, CI's GPU-less runner included.
    const NO_SUCH_GPU: [u8; 16] = [0; 16];

    /// A flat field under deterministic per-texel noise comes back nearly
    /// flat — variance collapses — while alpha rides through bit-exact.
    #[test]
    fn denoising_a_noisy_flat_field_removes_the_noise() {
        let mut denoiser = Denoiser::new(NO_SUCH_GPU).expect("OIDN device");
        let (width, height) = (64u32, 64u32);
        let texels = (width * height) as usize;
        // Mean-preserving hash noise around 0.5; alpha carries a ramp the
        // filter must not touch.
        let noisy: Vec<f32> = (0..texels)
            .flat_map(|i| {
                let hash = |k: usize| {
                    let bucket = ((i * 3 + k) * 2_654_435_761) % 1024;
                    0.5 + 0.4 * (bucket as f32 / 1024.0 - 0.5)
                };
                [hash(0), hash(1), hash(2), i as f32]
            })
            .collect();
        let albedo: Vec<f32> = std::iter::repeat_n([0.8, 0.8, 0.8, 1.0], texels)
            .flatten()
            .collect();
        let normal: Vec<f32> = std::iter::repeat_n([0.0, 0.0, 1.0, 1.0], texels)
            .flatten()
            .collect();

        let mut out = noisy.clone();
        denoiser
            .denoise(width, height, Quality::High, &mut out, &albedo, &normal)
            .expect("denoise");

        assert!(out.iter().all(|value| value.is_finite()));
        let variance = |image: &[f32]| {
            let luma: Vec<f32> = image
                .chunks_exact(4)
                .map(|texel| (texel[0] + texel[1] + texel[2]) / 3.0)
                .collect();
            let mean = luma.iter().sum::<f32>() / luma.len() as f32;
            luma.iter().map(|l| (l - mean) * (l - mean)).sum::<f32>() / luma.len() as f32
        };
        let (noisy_var, out_var) = (variance(&noisy), variance(&out));
        assert!(
            out_var < noisy_var / 20.0,
            "variance {noisy_var} only fell to {out_var}"
        );
        assert!(
            out.chunks_exact(4)
                .zip(noisy.chunks_exact(4))
                .all(|(o, n)| o[3].to_bits() == n[3].to_bits()),
            "alpha must pass through untouched"
        );
    }

    /// One denoiser filters a growing and then a shrinking frame. Growing
    /// re-cuts the buffers; shrinking must reuse them and still filter the
    /// smaller rectangle correctly, which is the whole point of never
    /// giving capacity back.
    #[test]
    fn frames_of_every_size_reuse_one_set_of_buffers() {
        let mut denoiser = Denoiser::new(NO_SUCH_GPU).expect("OIDN device");
        for (width, height) in [(32u32, 24u32), (64, 48), (32, 24), (24, 32)] {
            let texels = (width * height) as usize;
            // A hard step down the middle: a filter that had the wrong
            // rectangle would smear it or leave the far half untouched.
            let mut image: Vec<f32> = (0..texels)
                .flat_map(|i| {
                    let bright = if (i % width as usize) < width as usize / 2 {
                        0.8
                    } else {
                        0.2
                    };
                    let jitter = 0.1 * ((i % 7) as f32 / 7.0 - 0.5);
                    [bright + jitter, bright, bright - jitter, 1.0]
                })
                .collect();
            let guide: Vec<f32> = std::iter::repeat_n([0.5, 0.5, 0.5, 1.0], texels)
                .flatten()
                .collect();
            denoiser
                .denoise(width, height, Quality::Balanced, &mut image, &guide, &guide)
                .expect("denoise");
            assert!(image.iter().all(|value| value.is_finite()));
            let column = |x: usize| {
                let sum: f32 = (0..height as usize)
                    .map(|y| image[(y * width as usize + x) * 4])
                    .sum();
                sum / height as f32
            };
            assert!(
                column(width as usize / 4) > column(3 * width as usize / 4) + 0.4,
                "{width}×{height} lost the step, so the filter had the wrong rectangle"
            );
        }
        assert!(
            denoiser.capacity >= 64 * 48 * 16,
            "capacity must not shrink below the largest frame seen"
        );
    }

    /// The renderer's own GPU opens by its Vulkan UUID — the path every
    /// real run takes, and the one a UUID mismatch would silently drop off
    /// onto the CPU.
    #[test]
    fn the_renderer_s_gpu_opens_by_its_vulkan_uuid() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let mut denoiser = Denoiser::new(gpu.device_uuid()).expect("OIDN device");
        let texels = 32 * 32;
        let mut image: Vec<f32> = (0..texels)
            .flat_map(|i| [0.5 + 0.4 * ((i % 5) as f32 / 5.0 - 0.5), 0.5, 0.5, 1.0])
            .collect();
        let guide: Vec<f32> = std::iter::repeat_n([0.5, 0.5, 0.5, 1.0], texels)
            .flatten()
            .collect();
        denoiser
            .denoise(32, 32, Quality::Fast, &mut image, &guide, &guide)
            .expect("denoise");
        assert!(image.iter().all(|value| value.is_finite()));
    }
}
