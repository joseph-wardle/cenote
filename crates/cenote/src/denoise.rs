//! OIDN denoising of the film's linear averages, by host copy.
//!
//! `OpenImageDenoise` has no Vulkan device: zero-copy interop would need
//! exported `VkDeviceMemory` and cross-API synchronization it has no seam
//! for yet, so the film's averages travel through host memory instead —
//! the caller reads them back, this module filters them, and the result
//! goes wherever the caller needs it (an EXR, a re-uploaded display
//! buffer). At 720p the copies cost milliseconds against a filter that
//! costs hundreds, so the simple interop is the right trade.
//!
//! The filter consumes the film's albedo and normal AOVs as guides. They
//! are *not* declared noise-free (`cleanAux`): OIDN's prescribed guide
//! prefilter — each auxiliary image denoised through its own RT filter —
//! is not expressible through the `oidn` crate, which always binds an
//! input as the color image (and its `clean_aux` setter misspells the
//! OIDN parameter name, so the flag never reaches the filter either
//! way). The default weights are trained for noisy guides; ours are
//! near-clean, which is the safe side of that mismatch.

use crate::{Error, Result};

/// Filter quality, mirroring OIDN's knob. The CPU device treats both
/// alike; GPU devices trade quality for speed at `Balanced`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quality {
    /// Interactive — the viewer's preview cadence.
    Balanced,
    /// Final frame — the CLI's batch output.
    High,
}

/// An `OpenImageDenoise` device plus the packing scratch reused across
/// frames. One `Denoiser` serves any image size; the filter itself is
/// rebuilt per call, which costs ~10% of a 720p filter run.
pub struct Denoiser {
    device: oidn::Device,
    /// The film publishes RGBA texels; OIDN reads tightly packed RGB.
    color: Vec<f32>,
    albedo: Vec<f32>,
    normal: Vec<f32>,
}

impl Denoiser {
    /// Open the fastest OIDN device available (the CPU, unless OIDN's GPU
    /// device runtimes are installed).
    ///
    /// # Errors
    ///
    /// [`Error::Denoise`] when no device comes up — a build against a
    /// library whose device runtimes are missing.
    pub fn new() -> Result<Self> {
        let device = oidn::Device::new();
        if let Err((_, message)) = device.get_error() {
            return Err(Error::Denoise(message));
        }
        Ok(Self {
            device,
            color: Vec::new(),
            albedo: Vec::new(),
            normal: Vec::new(),
        })
    }

    /// Denoise one frame of linear RGBA averages, guided by the film's
    /// albedo and normal AOVs. Returns the filtered beauty as RGBA with
    /// the input's alpha untouched (OIDN filters color only).
    ///
    /// # Errors
    ///
    /// [`Error::Denoise`] carrying OIDN's diagnostic when the filter
    /// setup or execution fails.
    ///
    /// # Panics
    ///
    /// When a slice isn't `width × height` RGBA texels.
    pub fn denoise(
        &mut self,
        width: u32,
        height: u32,
        quality: Quality,
        beauty: &[f32],
        albedo: &[f32],
        normal: &[f32],
    ) -> Result<Vec<f32>> {
        let texels = width as usize * height as usize;
        assert_eq!(beauty.len(), texels * 4, "beauty must be RGBA per pixel");
        assert_eq!(albedo.len(), texels * 4, "albedo must be RGBA per pixel");
        assert_eq!(normal.len(), texels * 4, "normal must be RGBA per pixel");

        pack_rgb(&mut self.color, beauty);
        pack_rgb(&mut self.albedo, albedo);
        pack_rgb(&mut self.normal, normal);

        let mut filtered = vec![0.0f32; texels * 3];
        let mut filter = oidn::RayTracing::new(&self.device);
        filter
            .srgb(false)
            .hdr(true)
            .filter_quality(match quality {
                Quality::Balanced => oidn::Quality::Balanced,
                Quality::High => oidn::Quality::High,
            })
            .albedo_normal(&self.albedo, &self.normal)
            .image_dimensions(width as usize, height as usize);
        filter
            .filter(&self.color, &mut filtered)
            .map_err(|error| Error::Denoise(format!("{error:?}")))?;
        if let Err((_, message)) = self.device.get_error() {
            return Err(Error::Denoise(message));
        }

        Ok(filtered
            .chunks_exact(3)
            .zip(beauty.chunks_exact(4))
            .flat_map(|(rgb, rgba)| [rgb[0], rgb[1], rgb[2], rgba[3]])
            .collect())
    }
}

/// Drop the alpha lane: RGBA texels to the tightly packed RGB OIDN reads.
fn pack_rgb(dst: &mut Vec<f32>, src: &[f32]) {
    dst.clear();
    dst.extend(
        src.chunks_exact(4)
            .flat_map(|texel| texel[..3].iter().copied()),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat field under deterministic per-texel noise comes back nearly
    /// flat — variance collapses — while alpha rides through bit-exact.
    /// Runs on the CPU device: no GPU, so CI exercises the real filter.
    #[test]
    fn denoising_a_noisy_flat_field_removes_the_noise() {
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

        let mut denoiser = Denoiser::new().expect("OIDN device");
        let out = denoiser
            .denoise(width, height, Quality::High, &noisy, &albedo, &normal)
            .expect("denoise");

        assert_eq!(out.len(), noisy.len());
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
}
