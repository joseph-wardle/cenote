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

/// Relative luminance of a linear `ACEScg` (AP1) colour — the CIE Y row of
/// the AP1→XYZ matrix. Identical to `luminance` in `accumulate.slang`, so the
/// host's standard-error readback weighs brightness exactly as the kernel that
/// filled the second-moment buffer did.
const LUMINANCE_AP1: [f32; 3] = [0.272_228_72, 0.674_081_74, 0.053_689_517];
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
    /// The variance substrate: Σ luminance² of the guarded beauty sample, one
    /// `f32` per pixel (M3 step 6a). A lone sum — no wave-written sample pair —
    /// because the accumulate kernel derives it from the beauty sample it
    /// already reads; the first moment is the beauty sum itself (luminance is a
    /// linear functional of RGB). Together they give the per-pixel sample
    /// variance `mean(L²) − mean(L)²`, and so the estimator standard error
    /// `sqrt(Var / N)` — [`Film::standard_error`]. Overwritten on the first
    /// sample after a reset like the sums, so it needs no separate clear.
    pub(super) moment2: Buffer,
    /// The auto-stop tally (M3 step 6b): a single `u32` the accumulate kernel
    /// atomically counts each sample's converged pixels into — those whose
    /// relative estimator standard error fell below the noise threshold. Zeroed
    /// before every accumulate (so it is a fresh snapshot, not a running total)
    /// and read back by [`Film::converged_fraction`], which the global auto-stop
    /// policy compares against [`Renderer::CONVERGENCE_TARGET`]. One `u32`
    /// keeps the per-frame convergence check a 4-byte readback instead of the
    /// whole variance field.
    pub(super) converged: Buffer,
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
        // Just the sum: the kernel derives Σ luminance² from the beauty sample,
        // so there is no wave-written sample half. STORAGE (kernel writes it) +
        // TRANSFER_SRC (the standard-error readback).
        let moment2 = gpu.create_buffer(
            "film.moment2",
            texels * 4,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::GpuOnly,
        )?;
        // The auto-stop tally: one u32. STORAGE (the kernel atomically counts
        // into it) + TRANSFER_DST (zero-filled before each accumulate) +
        // TRANSFER_SRC (the converged-fraction readback).
        let converged = gpu.create_buffer(
            "film.converged",
            super::CONVERGED_COUNT_BYTES,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::GpuOnly,
        )?;
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
            moment2,
            converged,
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

    /// Read back the per-pixel estimator standard error of beauty luminance —
    /// `sqrt(Var / N)`, row-major, one `f32` per pixel — the raw convergence
    /// metric the auto-stop policy (step 6b) and the validation harness (step
    /// 7) read. `Var = mean(L²) − mean(L)²` is the per-sample luminance
    /// variance, from the second-moment buffer (`mean(L²)`) and the beauty sum
    /// (`mean(L)`, since luminance is linear); `N` is the sample count.
    ///
    /// The `sqrt(Var / N)` standard-error reading assumes independent samples,
    /// which only holds once the temporal decay has handed off (D-094); the
    /// substrate accumulates from frame 0 regardless, so the *consumer* — not
    /// this raw readback — is responsible for ignoring it until then. Var is
    /// clamped at zero against the rounding that can drive `mean(L²) − mean(L)²`
    /// a hair negative on a noise-free pixel.
    ///
    /// # Errors
    ///
    /// Any [`crate::Error`] from the readbacks.
    ///
    /// # Panics
    ///
    /// If the film has no samples — there is no variance yet, so calling order
    /// is a programmer bug.
    pub fn standard_error(&self, gpu: &Context) -> Result<Vec<f32>> {
        assert!(self.samples > 0, "no samples to measure variance over");
        let n = self.samples as f32;
        let beauty: Vec<f32> =
            bytemuck::pod_collect_to_vec(&gpu.download_buffer(&self.beauty.sum)?);
        let moment2: Vec<f32> =
            bytemuck::pod_collect_to_vec(&gpu.download_buffer(&self.moment2)?);
        Ok(moment2
            .iter()
            .zip(beauty.chunks_exact(4))
            .map(|(sum_l2, rgba)| {
                let mean_l = (rgba[0] * LUMINANCE_AP1[0]
                    + rgba[1] * LUMINANCE_AP1[1]
                    + rgba[2] * LUMINANCE_AP1[2])
                    / n;
                let mean_l2 = sum_l2 / n;
                let var = (mean_l2 - mean_l * mean_l).max(0.0);
                (var / n).sqrt()
            })
            .collect())
    }

    /// The fraction of pixels the last [`Renderer::accumulate`](super::Renderer::accumulate)
    /// found converged — relative estimator standard error below the noise
    /// threshold — as its cheap global auto-stop signal (M3 step 6b). The
    /// accumulate kernel counts them into a single `u32` (a 4-byte readback, not
    /// the whole variance field), zeroed before every sample so this is a fresh
    /// snapshot. The global auto-stop policy stops once it crosses
    /// [`Renderer::CONVERGENCE_TARGET`](super::Renderer::CONVERGENCE_TARGET).
    ///
    /// Exactly `0` until the sample count passes
    /// [`Renderer::CONVERGENCE_MIN_SAMPLES`](super::Renderer::CONVERGENCE_MIN_SAMPLES):
    /// the kernel counts no pixel below that floor, where the variance
    /// estimate is untrusted.
    ///
    /// # Errors
    ///
    /// Any [`crate::Error`] from the readback.
    ///
    /// # Panics
    ///
    /// If the film has no samples — the tally reflects an accumulate that has
    /// not run — so calling order is a programmer bug.
    pub fn converged_fraction(&self, gpu: &Context) -> Result<f32> {
        assert!(self.samples > 0, "no accumulate has filled the converged tally");
        let tally: Vec<u32> =
            bytemuck::pod_collect_to_vec(&gpu.download_buffer(&self.converged)?);
        let total = self.width * self.height;
        Ok(tally[0] as f32 / total as f32)
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
