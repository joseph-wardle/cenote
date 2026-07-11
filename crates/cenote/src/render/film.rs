//! The film: progressive accumulation state for one render-target size.
//! Four pixel-owned buffers — beauty plus the denoiser's albedo and normal
//! guides and first-hit depth — each a sample/sum pair a wave writes into
//! and the accumulation kernel folds together, so the bitwise-determinism
//! invariant covers them all. The sample count lives on the host, uniform
//! across pixels and buffers by construction.
//!
//! The renderer in [`super`] drives these buffers (`accumulate`, `resolve`);
//! the film only allocates them, resets, and reads them back. Resolved
//! averages are written into caller-owned targets rather than held here, so
//! the [`Session`](super::Session) can double-buffer its published frames
//! while the film keeps accumulating into the sums.

use ash::vk;

use crate::error::Result;
use crate::gpu::{Buffer, Context, MemoryLocation};
use crate::wavefront::{AovTargets, upload_aov_table};

/// One film buffer's accumulation pair: the per-pixel target a wave writes
/// its sample into (`TRANSFER_DST`: each wave starts by zero-filling it),
/// and the running sums the accumulation kernel folds it into
/// (`TRANSFER_SRC`: the accumulated image reads back — [`Film::averages`]
/// and the tests).
pub(super) struct Accumulation {
    pub(super) sample: Buffer,
    pub(super) sum: Buffer,
}

impl Accumulation {
    fn new(gpu: &Context, name: &str, bytes: u64) -> Result<Self> {
        let storage =
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        Ok(Self {
            sample: gpu.create_buffer(
                &format!("{name}.sample"),
                bytes,
                storage | vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuOnly,
            )?,
            sum: gpu.create_buffer(
                &format!("{name}.sum"),
                bytes,
                storage | vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::GpuOnly,
            )?,
        })
    }
}

/// Progressive accumulation state for one render-target size: per-pixel
/// linear f32 sums and the samples the current wave writes — beauty plus
/// the three AOVs (the denoiser's albedo and normal guides and first-hit
/// depth), each its own pixel-owned pair so the bitwise-determinism
/// invariant covers them all. The sample count lives on the host — it is
/// uniform across pixels and buffers by construction.
///
/// The resolved averages — the sums divided by the count — are written into
/// caller-owned buffers ([`Renderer::resolve`](super::Renderer::resolve))
/// rather than held here, so the [`Session`](super::Session) can
/// double-buffer its published frames while the film keeps accumulating
/// into these sums.
///
/// Sized at creation; a resize means a new `Film`. A view change means
/// [`Film::reset`].
pub struct Film {
    /// One sample's radiance, RGBA f32.
    pub(super) beauty: Accumulation,
    /// The denoiser albedo guide, RGBA f32 (alpha unused).
    pub(super) albedo: Accumulation,
    /// The denoiser normal guide — world-space shading normals, post
    /// normal-map — RGBA f32 (alpha unused).
    pub(super) normal: Accumulation,
    /// Camera-plane z at the first hit, f32; +∞ on miss.
    pub(super) depth: Accumulation,
    /// The guides' per-pixel feature-throughput scratch, alive within one
    /// wave — see `AovTable` in `shaders/pathstate.slang`. Reached only by
    /// GPU address through that table; held here for its lifetime.
    #[expect(dead_code, reason = "reached only by GPU address, via aov_table")]
    guide: Buffer,
    /// The uploaded table the shading kernels reach the four buffers
    /// above through.
    aov_table: Buffer,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) samples: u32,
}

impl Film {
    /// Create a film for `width`×`height` renders. Starts empty: the first
    /// [`Renderer::accumulate`](super::Renderer::accumulate) initializes the
    /// sums, so no clear pass runs.
    ///
    /// # Errors
    ///
    /// Any [`crate::Error`] from buffer creation.
    ///
    /// # Panics
    ///
    /// On zero dimensions — callers validate their inputs, so this is a
    /// programmer bug.
    pub fn new(gpu: &Context, width: u32, height: u32) -> Result<Self> {
        assert!(width > 0 && height > 0, "zero-sized film");
        let texels = u64::from(width) * u64::from(height);
        let albedo = Accumulation::new(gpu, "film.albedo", texels * 16)?;
        let normal = Accumulation::new(gpu, "film.normal", texels * 16)?;
        let depth = Accumulation::new(gpu, "film.depth", texels * 4)?;
        let guide = gpu.create_buffer(
            "film.aov.guide",
            texels * 16,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::GpuOnly,
        )?;
        let aov_table =
            upload_aov_table(gpu, &albedo.sample, &normal.sample, &depth.sample, &guide)?;
        Ok(Self {
            beauty: Accumulation::new(gpu, "film", texels * 16)?,
            albedo,
            normal,
            depth,
            guide,
            aov_table,
            width,
            height,
            samples: 0,
        })
    }

    /// The wave-facing halves of the AOV buffers, for
    /// [`Wavefront::trace_then`](crate::wavefront::Wavefront::trace_then).
    pub(super) fn aov_targets(&self) -> AovTargets<'_> {
        AovTargets {
            albedo: &self.albedo.sample,
            normal: &self.normal.sample,
            depth: &self.depth.sample,
            table: &self.aov_table,
        }
    }

    /// Start over (the view changed): the next sample overwrites the sums
    /// instead of adding, so nothing needs clearing now.
    pub fn reset(&mut self) {
        self.samples = 0;
    }

    /// Samples accumulated since creation or the last [`Film::reset`].
    #[must_use]
    pub fn samples(&self) -> u32 {
        self.samples
    }

    /// Read back the accumulated beauty average — linear `ACEScg` RGBA,
    /// row-major, pixel (0, 0) top-left. Each channel is its sum divided
    /// by the sample count, so alpha comes out exactly 1 and a one-sample
    /// average is bit-identical to the sample.
    ///
    /// # Errors
    ///
    /// Any [`crate::Error`] from the readback.
    ///
    /// # Panics
    ///
    /// If the film has no samples — there is no average yet, so calling
    /// order is a programmer bug.
    pub fn beauty_average(&self, gpu: &Context) -> Result<Vec<f32>> {
        assert!(self.samples > 0, "averaging an empty film");
        self.averaged(gpu, &self.beauty)
    }

    /// Read back every accumulated average — the beauty of
    /// [`Film::beauty_average`] plus the three AOVs, all in the same
    /// row-major layout (RGBA quads except depth, one `f32` per pixel) —
    /// what the batch CLI writes as one multi-layer EXR.
    ///
    /// # Errors
    ///
    /// Any [`crate::Error`] from the readbacks.
    ///
    /// # Panics
    ///
    /// If the film has no samples — there is no average yet, so calling
    /// order is a programmer bug.
    pub fn averages(&self, gpu: &Context) -> Result<FilmAverages> {
        assert!(self.samples > 0, "averaging an empty film");
        Ok(FilmAverages {
            beauty: self.averaged(gpu, &self.beauty)?,
            albedo: self.averaged(gpu, &self.albedo)?,
            normal: self.averaged(gpu, &self.normal)?,
            depth: self.averaged(gpu, &self.depth)?,
        })
    }

    /// One buffer's sums, downloaded and divided by the sample count.
    fn averaged(&self, gpu: &Context, accumulation: &Accumulation) -> Result<Vec<f32>> {
        let sums: Vec<f32> = bytemuck::pod_collect_to_vec(&gpu.download_buffer(&accumulation.sum)?);
        Ok(sums.iter().map(|sum| sum / self.samples as f32).collect())
    }

    /// Width in pixels.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }
}

/// Every accumulated average of a [`Film`], read back to the host
/// ([`Film::averages`]): row-major, pixel (0, 0) top-left; RGBA `f32`
/// quads except `depth`, one `f32` per pixel (+∞ where every sample
/// missed). `albedo` and `normal` are the denoiser guides; `normal` is
/// the world-space shading normal, averaged unnormalized.
pub struct FilmAverages {
    /// Linear `ACEScg` radiance, RGBA (alpha exactly 1).
    pub beauty: Vec<f32>,
    /// The denoiser albedo guide, RGBA (alpha unused).
    pub albedo: Vec<f32>,
    /// The denoiser normal guide, RGBA (alpha unused).
    pub normal: Vec<f32>,
    /// Camera-plane z at the first hit, one `f32` per pixel.
    pub depth: Vec<f32>,
}
