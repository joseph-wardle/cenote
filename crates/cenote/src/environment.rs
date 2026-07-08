//! The environment light on the host: an equirect radiance image and the
//! sampling tables built from it at prep. The GPU sees the image as the
//! binding model's one sampled texture and the tables as plain buffers
//! behind the scene table; `struct Environment` in
//! `shaders/environment.slang` consumes both and mirrors the table layout
//! built here.
//!
//! The sampling scheme is the classic marginal/conditional split over the
//! image: pick a row from a CDF over row sums, then a column from that
//! row's CDF, each weighted by luminance × sin θ (the equirect area
//! factor). Two prep-time details keep the estimator honest:
//!
//! - **Support dilation.** The kernel reads radiance *bilinearly*, so a
//!   black texel next to a bright one still contributes light over its
//!   footprint. Sampling weights are therefore the 3×3 neighborhood
//!   *maximum* of luminance (wrapping horizontally, clamping vertically —
//!   the sampler's own address modes), which makes the sampling pdf's
//!   support cover the bilinear reconstruction's exactly. Without this,
//!   radiance along zero/nonzero texel boundaries is unreachable by
//!   next-event sampling and the single-strategy estimators disagree.
//! - **A separate pdf table.** The pdf of a direction is its texel's
//!   normalized weight — stored directly rather than recovered as a CDF
//!   difference, which for dim texels under a bright sun cancels
//!   catastrophically in `f32`.

use glam::Vec3;

use crate::color::{acescg_from_rec709, luminance};
use crate::error::Result;

/// An equirect environment: radiance in `ACEScg`, by direction. Row 0 is
/// straight up (+Y), the bottom row straight down, and the horizontal
/// center of the image faces −Z — the direction the crate's cameras
/// conventionally look, so an HDRI's centered subject lands behind the
/// scene.
pub struct Environment {
    width: u32,
    height: u32,
    /// Tightly packed row-major RGBA (alpha 1), ready to upload.
    texels: Vec<f32>,
}

impl Environment {
    /// The same radiance in every direction — the furnace tests'
    /// environment, and the constant-sky look, as a 1×1 image through the
    /// one environment code path (no second estimator to keep honest).
    #[must_use]
    pub fn constant(radiance: Vec3) -> Self {
        Self::equirect(1, 1, vec![radiance.x, radiance.y, radiance.z, 1.0])
    }

    /// An equirect environment from raw RGBA texels, already in `ACEScg` —
    /// procedural skies, and tests that pin exact radiance per texel.
    ///
    /// # Panics
    ///
    /// On zero dimensions or a texel count that doesn't match them —
    /// programmer bugs.
    #[must_use]
    pub fn equirect(width: u32, height: u32, texels: Vec<f32>) -> Self {
        assert!(width > 0 && height > 0, "zero-sized environment");
        assert_eq!(
            texels.len() as u64,
            u64::from(width) * u64::from(height) * 4,
            "texel count doesn't match environment dimensions"
        );
        Self {
            width,
            height,
            texels,
        }
    }

    /// Load an equirect EXR (e.g. via `include_bytes!`). Pixels are taken
    /// as linear `Rec.709` and converted to `ACEScg` here — load is this
    /// texture's IDT, matching how authored colors enter at prep. Radiance
    /// is sanitized while it's cheap: negatives (out-of-gamut conversion
    /// residue) clamp to zero, non-finite texels drop to black.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Image`] if `bytes` don't decode as an EXR.
    pub fn from_equirect_exr(bytes: &[u8]) -> Result<Self> {
        let (width, height, mut texels) = crate::output::read_exr_bytes(bytes)?;
        for texel in texels.chunks_exact_mut(4) {
            let rgb = Vec3::new(texel[0], texel[1], texel[2]);
            let rgb = if rgb.is_finite() { rgb } else { Vec3::ZERO };
            let acescg = acescg_from_rec709(rgb).max(Vec3::ZERO);
            texel.copy_from_slice(&[acescg.x, acescg.y, acescg.z, 1.0]);
        }
        Ok(Self {
            width,
            height,
            texels,
        })
    }

    /// Image width in texels.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Image height in texels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// The RGBA texels, for the image upload.
    #[must_use]
    pub fn texels(&self) -> &[f32] {
        &self.texels
    }

    /// Build the sampling tables. All accumulation runs in `f64` and lands
    /// in the `f32` the kernels read.
    pub(crate) fn tables(&self) -> Tables {
        let (w, h) = (self.width as usize, self.height as usize);
        let lum: Vec<f64> = self
            .texels
            .chunks_exact(4)
            .map(|t| f64::from(luminance(Vec3::new(t[0], t[1], t[2]))))
            .collect();

        // Sampling weights: dilated luminance × the row's equirect area
        // factor (sin θ at the row center).
        let mut weight = vec![0.0_f64; w * h];
        for row in 0..h {
            let sin_theta = ((row as f64 + 0.5) / h as f64 * std::f64::consts::PI).sin();
            for col in 0..w {
                let mut peak = 0.0_f64;
                for dr in [-1_isize, 0, 1] {
                    let r = row.saturating_add_signed(dr).min(h - 1);
                    for dc in [-1_isize, 0, 1] {
                        let c = (col.cast_signed() + dc)
                            .rem_euclid(w.cast_signed())
                            .cast_unsigned();
                        peak = peak.max(lum[r * w + c]);
                    }
                }
                weight[row * w + col] = peak * sin_theta;
            }
        }
        let total: f64 = weight.iter().sum();

        // Marginal CDF over rows, conditional CDF within each row. Zero
        // rows leave a flat (never-selected) span; the final entries are
        // pinned to exactly 1 so a search can't fall off the end.
        let mut marginal = vec![0.0_f32; h + 1];
        let mut running = 0.0_f64;
        for row in 0..h {
            running += weight[row * w..(row + 1) * w].iter().sum::<f64>();
            marginal[row + 1] = if total > 0.0 {
                (running / total) as f32
            } else {
                0.0
            };
        }
        marginal[h] = 1.0;

        let mut conditional = vec![0.0_f32; h * (w + 1)];
        for row in 0..h {
            let row_sum: f64 = weight[row * w..(row + 1) * w].iter().sum();
            let cdf = &mut conditional[row * (w + 1)..(row + 1) * (w + 1)];
            let mut running = 0.0_f64;
            for col in 0..w {
                running += weight[row * w + col];
                cdf[col + 1] = if row_sum > 0.0 {
                    (running / row_sum) as f32
                } else {
                    0.0
                };
            }
            cdf[w] = 1.0;
        }

        // The pdf table: each texel's weight normalized to a joint density
        // over the unit (u, v) square (mean 1 by construction). The kernel
        // divides by the solid-angle Jacobian 2π² sin θ at lookup.
        let scale = if total > 0.0 {
            (w * h) as f64 / total
        } else {
            0.0
        };
        let pdfs: Vec<f32> = weight.iter().map(|&value| (value * scale) as f32).collect();

        // Emitted power ∝ the luminance integral over the sphere (exact
        // per-row solid angles, undilated — dilation is a sampling device,
        // not energy). Weighs the environment against the quad lights.
        let mut power = 0.0_f64;
        for row in 0..h {
            let theta0 = row as f64 / h as f64 * std::f64::consts::PI;
            let theta1 = (row + 1) as f64 / h as f64 * std::f64::consts::PI;
            let row_solid_angle =
                2.0 * std::f64::consts::PI / w as f64 * (theta0.cos() - theta1.cos());
            power += row_solid_angle * lum[row * w..(row + 1) * w].iter().sum::<f64>();
        }

        Tables {
            marginal,
            conditional,
            pdfs,
            power,
        }
    }
}

/// The finished sampling tables, in the layout the kernels index. Mirrors
/// what `struct Environment` in `shaders/environment.slang` reads.
pub(crate) struct Tables {
    /// CDF over rows; `height + 1` entries, 0 first, exactly 1 last.
    pub marginal: Vec<f32>,
    /// Per-row CDF over columns; `height` runs of `width + 1` entries.
    pub conditional: Vec<f32>,
    /// Per-texel joint density over the unit (u, v) square; `height × width`.
    pub pdfs: Vec<f32>,
    /// Luminance integral over the sphere — the selection weight against
    /// the quad lights.
    pub power: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(width: u32, height: u32, lum_at: impl Fn(u32, u32) -> f32) -> Environment {
        let mut texels = Vec::new();
        for row in 0..height {
            for col in 0..width {
                let y = lum_at(col, row);
                texels.extend_from_slice(&[y, y, y, 1.0]);
            }
        }
        Environment {
            width,
            height,
            texels,
        }
    }

    /// CDFs are proper distributions: monotone, 0 to exactly 1, and the
    /// pdf table is a unit-mean density over the (u, v) square — the
    /// normalization the kernel's solid-angle conversion builds on.
    #[test]
    #[expect(clippy::float_cmp, reason = "the CDF endpoints are pinned exactly")]
    fn tables_are_normalized_distributions() {
        let env = image(8, 4, |col, row| (1 + col + row * 8) as f32 * 0.1);
        let tables = env.tables();
        assert_eq!(tables.marginal.first(), Some(&0.0));
        assert_eq!(tables.marginal.last(), Some(&1.0));
        assert!(tables.marginal.windows(2).all(|w| w[0] <= w[1]));
        for row in 0..4 {
            let cdf = &tables.conditional[row * 9..(row + 1) * 9];
            assert_eq!(cdf[0], 0.0);
            assert_eq!(cdf[8], 1.0);
            assert!(cdf.windows(2).all(|w| w[0] <= w[1]));
        }
        let mean = tables.pdfs.iter().map(|&p| f64::from(p)).sum::<f64>() / 32.0;
        assert!((mean - 1.0).abs() < 1e-6, "pdf table mean {mean}");
    }

    /// A single bright texel: its whole 3×3 neighborhood — wrapping across
    /// the horizontal seam, clamping at the top edge — must carry sampling
    /// mass, because bilinear reads bleed the texel's radiance into it.
    /// Everything outside stays at zero. This is the dilation that keeps
    /// zero-radiance texels next to bright ones reachable.
    #[test]
    fn support_covers_the_bilinear_footprint() {
        let env = image(8, 4, |col, row| f32::from(col == 0 && row == 0));
        let tables = env.tables();
        for row in 0..4_usize {
            for col in 0..8_usize {
                let in_footprint = row <= 1 && [7, 0, 1].contains(&col);
                assert_eq!(
                    tables.pdfs[row * 8 + col] > 0.0,
                    in_footprint,
                    "texel ({col}, {row})"
                );
            }
        }
    }

    /// A black environment degenerates cleanly: no sampling mass anywhere,
    /// zero power (so scene prep gives it selection probability 0), and
    /// the CDFs still end at 1 so a stray search stays in bounds.
    #[test]
    #[expect(clippy::float_cmp, reason = "zero mass is exact, not approximate")]
    fn black_environment_has_no_mass() {
        let tables = Environment::constant(Vec3::ZERO).tables();
        assert_eq!(tables.power, 0.0);
        assert!(tables.pdfs.iter().all(|&p| p == 0.0));
        assert_eq!(tables.marginal.last(), Some(&1.0));
    }

    /// The power heuristic's input is exact for the case with a closed
    /// form: a constant environment's luminance integral is 4π × luminance,
    /// at any resolution (the per-row solid angles telescope).
    #[test]
    fn constant_environment_power_is_exact() {
        for (w, h) in [(1, 1), (8, 4), (13, 7)] {
            let env = image(w, h, |_, _| 0.5);
            let want = 4.0 * std::f64::consts::PI * f64::from(luminance(Vec3::splat(0.5)));
            assert!(
                (env.tables().power - want).abs() < 1e-9,
                "{w}×{h}: {} vs {want}",
                env.tables().power
            );
        }
    }

    /// The demo asset round-trips through the loader: right size, finite,
    /// non-negative, and bright enough to light a scene.
    #[test]
    fn demo_asset_loads() {
        let env =
            Environment::from_equirect_exr(include_bytes!("../assets/kloofendal_puresky.exr"))
                .expect("demo HDRI decodes");
        assert_eq!((env.width(), env.height()), (512, 256));
        assert!(env.texels().iter().all(|v| v.is_finite() && *v >= 0.0));
        assert!(env.tables().power > 1.0, "a daytime sky is not this dark");
    }
}
