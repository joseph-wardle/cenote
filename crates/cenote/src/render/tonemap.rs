//! The consumer's view transform: exposure, the ACES display transform, the
//! sRGB transfer curve, and a pack to the RGBA8 bytes the presenter blits —
//! turning a [`Film`](super::Film)'s resolved linear average into a
//! displayable frame.
//!
//! This is the consumer half of the estimator/view split (the [`super`]
//! module doc draws the whole picture): the tonemap runs *downstream* of the
//! published frame — the same place Hydra puts color correction, after the
//! render buffer. The viewer owns one permanently and drives it each frame;
//! the CLI never builds one, since batch EXRs stay linear `ACEScg`.

use ash::vk;
use bytemuck::{Pod, Zeroable};

use super::workgroups;
use crate::error::Result;
use crate::gpu::{Bindings, Buffer, ComputePipeline, Context, MemoryLocation};
use crate::shaders::Kernels;

/// Push constants for the tonemap kernel; mirrors `struct Params` in
/// `shaders/tonemap.slang`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TonemapParams {
    /// Device address of the film's resolved linear average (`float4*`).
    average: vk::DeviceAddress,
    /// Device address of the packed RGBA8 display buffer (`uint*`).
    display: vk::DeviceAddress,
    width: u32,
    height: u32,
    /// `exp2(exposure stops)` — the resolve kernel already averaged.
    exposure_scale: f32,
    _pad0: f32,
}

/// The view-transform pipeline plus its output: exposure (in stops), the
/// ACES display transform, the sRGB encode, and the RGBA8 pack, landing in
/// a lazily sized display buffer for the presenter to blit.
pub struct Tonemap {
    pipeline: ComputePipeline,
    /// The RGBA8 output, sized to the last [`Tonemap::apply`] and grown when
    /// a larger frame arrives — the pipeline itself is size-independent, so
    /// only this buffer tracks the window.
    display: Option<Buffer>,
}

impl Tonemap {
    /// Build the view transform from the embedded tonemap kernel. The
    /// display buffer is allocated lazily by [`Tonemap::apply`], since its
    /// size is the frame's, not known here.
    ///
    /// # Errors
    ///
    /// Any [`crate::Error`] from pipeline creation.
    pub fn new(gpu: &Context) -> Result<Self> {
        let kernels = Kernels::embedded();
        Ok(Self {
            pipeline: gpu.create_compute_pipeline(
                &kernels.tonemap.spirv,
                kernels.tonemap.entry,
                size_of::<TonemapParams>() as u32,
                Bindings::None,
            )?,
            display: None,
        })
    }

    /// Tonemap a `width`×`height` linear `average` into the display buffer:
    /// exposure (in stops), the ACES display transform, sRGB encode, RGBA8
    /// pack — everything [`crate::gpu::Presenter::present`] needs. Read the
    /// result with [`Tonemap::display`].
    ///
    /// # Errors
    ///
    /// Any [`crate::Error`] from buffer creation or submission.
    ///
    /// # Panics
    ///
    /// On a zero-sized frame, or an `average` buffer smaller than
    /// `width`×`height` RGBA f32 texels — callers validate their inputs, so
    /// both are programmer bugs.
    pub fn apply(
        &mut self,
        gpu: &Context,
        average: &Buffer,
        width: u32,
        height: u32,
        exposure: f32,
    ) -> Result<()> {
        assert!(width > 0 && height > 0, "zero-sized tonemap");
        assert!(
            average.size() >= u64::from(width) * u64::from(height) * 16,
            "average buffer is smaller than the frame"
        );
        let display = self.ensure_display(gpu, width, height)?.device_address();
        let params = TonemapParams {
            average: average.device_address(),
            display,
            width,
            height,
            exposure_scale: exposure.exp2(),
            _pad0: 0.0,
        };
        gpu.dispatch(
            &self.pipeline,
            None,
            bytemuck::bytes_of(&params),
            workgroups(width, height),
        )
    }

    /// The last tonemapped frame — packed RGBA8, sRGB-encoded, exactly what
    /// [`crate::gpu::Presenter::present`] expects. Hand it over right after
    /// [`Tonemap::apply`].
    ///
    /// # Panics
    ///
    /// Before the first [`Tonemap::apply`], when no frame exists yet.
    #[must_use]
    pub fn display(&self) -> &Buffer {
        self.display.as_ref().expect("apply has not run yet")
    }

    /// Upload a host frame of linear RGBA averages as a buffer
    /// [`Tonemap::apply`] can read — the road back onto the GPU for a
    /// frame that left it to be processed on the host, like the viewer's
    /// denoised beauty.
    ///
    /// # Errors
    ///
    /// Any [`crate::Error`] from buffer creation or the staging copy.
    pub fn upload_average(gpu: &Context, name: &str, texels: &[f32]) -> Result<Buffer> {
        gpu.upload_buffer(
            name,
            bytemuck::cast_slice(texels),
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        )
    }

    /// The display buffer sized for `width`×`height`, allocating or growing
    /// it when the frame outgrows it. The pipeline is size-independent, so
    /// this lazy buffer is all that tracks the window.
    fn ensure_display(&mut self, gpu: &Context, width: u32, height: u32) -> Result<&Buffer> {
        let bytes = u64::from(width) * u64::from(height) * 4;
        let stale = self.display.as_ref().is_none_or(|d| d.size() < bytes);
        if stale {
            self.display = Some(gpu.create_buffer(
                "tonemap.display",
                bytes,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::GpuOnly,
            )?);
        }
        Ok(self.display.as_ref().expect("just created above"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tonemap kernel against a CPU mirror of the same transform:
    /// exposure scale, `ACEScg` → `Rec.709`, Hill's ACES fit, sRGB encode,
    /// RGBA8 pack. It reads a resolved linear average now, so the fixture is
    /// the average directly (no ÷ samples here — that is the resolve
    /// kernel's job). A transposed matrix or wrong constant shows up as more
    /// than the ±1 quantization step allowed here.
    #[test]
    fn tonemap_matches_the_cpu_reference() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let mut tonemap = Tonemap::new(&gpu).expect("tonemap");
        let averages: [f32; 24] = [
            0.0, 0.0, 0.0, 1.0, // black stays black
            0.18, 0.18, 0.18, 1.0, // mid grey
            1.0, 1.0, 1.0, 1.0, // white
            10.0, 2.0, 0.5, 1.0, // hot highlight, compressed not clipped
            -0.5, 0.2, 0.2, 1.0, // negative clamps to zero, not garbage
            1.0, 0.1, 0.05, 1.0, // saturated red
        ];
        let average =
            Tonemap::upload_average(&gpu, "tonemap.average.synthetic", &averages).expect("upload");
        let exposure = 0.5;
        tonemap
            .apply(&gpu, &average, 6, 1, exposure)
            .expect("tonemap");

        let display = gpu.download_buffer(tonemap.display()).expect("download");
        let scale = exposure.exp2();
        for (index, texel) in display.chunks_exact(4).enumerate() {
            let avg = &averages[index * 4..index * 4 + 3];
            let rgb = aces_display([avg[0] * scale, avg[1] * scale, avg[2] * scale]);
            for channel in 0..3 {
                let expected = (srgb_encode(rgb[channel]) * 255.0).round() as i16;
                let difference = (i16::from(texel[channel]) - expected).abs();
                assert!(
                    difference <= 1,
                    "pixel {index} channel {channel}: GPU {} vs CPU {expected}",
                    texel[channel]
                );
            }
            assert_eq!(texel[3], 255, "display frames are opaque");
        }
    }

    // -- CPU mirror of shaders/tonemap.slang, same constants, same order --

    fn multiply(matrix: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
        std::array::from_fn(|row| {
            matrix[row][0] * v[0] + matrix[row][1] * v[1] + matrix[row][2] * v[2]
        })
    }

    fn aces_display(acescg: [f32; 3]) -> [f32; 3] {
        const SRGB_FROM_ACESCG: [[f32; 3]; 3] = [
            [1.705_051, -0.621_792_1, -0.083_258_87],
            [-0.130_256_42, 1.140_804_7, -0.010_548_319],
            [-0.024_003_357, -0.128_968_98, 1.152_972_3],
        ];
        const ACES_INPUT: [[f32; 3]; 3] = [
            [0.59719, 0.35458, 0.04823],
            [0.07600, 0.90834, 0.01566],
            [0.02840, 0.13383, 0.83777],
        ];
        const ACES_OUTPUT: [[f32; 3]; 3] = [
            [1.60475, -0.53108, -0.07367],
            [-0.10208, 1.10813, -0.00605],
            [-0.00327, -0.07276, 1.07602],
        ];
        let v = multiply(&ACES_INPUT, multiply(&SRGB_FROM_ACESCG, acescg));
        let fitted = v.map(|v| {
            (v * (v + 0.024_578_6) - 0.000_090_537) / (v * (0.983_729 * v + 0.432_951) + 0.238_081)
        });
        multiply(&ACES_OUTPUT, fitted).map(|channel| channel.clamp(0.0, 1.0))
    }

    fn srgb_encode(channel: f32) -> f32 {
        if channel <= 0.003_130_8 {
            12.92 * channel
        } else {
            1.055 * channel.powf(1.0 / 2.4) - 0.055
        }
    }
}
