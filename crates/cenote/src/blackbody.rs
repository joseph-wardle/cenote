//! Temperature → emitted color: Planck's law integrated against the CIE
//! 1931 observer, baked once into the table the tracker reads at every
//! emissive collision.
//!
//! This module is the *only* definition of what a temperature emits. The
//! shader reads [`table`]'s samples and interpolates them; it derives
//! nothing, so there is no second spelling of the physics to drift from
//! this one. (`cenote-pbrt`'s `blackbody_rec709` is not a second one: it
//! is chromaticity at unit luminance for *every* temperature, which is
//! pbrt's own convention for blackbody emitters and exists to import them
//! faithfully, not to say what a medium radiates.)
//!
//! The scale is physical, not normalized per temperature: a blackbody's
//! visible-band radiance climbs steeply with T, and keeping that climb is
//! what makes a fire's core read as hotter than its edge without the
//! author ramping anything. The one constant is the divisor —
//! [`REFERENCE_KELVIN`] emits unit luminance — so a medium's authored
//! emission scale is "how bright, in units of a 6500 K body".
//!
//! Samples are stored as base-2 logarithms. Visible-band radiance climbs
//! like `exp(−c/T)`, which no lerp between neighbours can follow but its
//! logarithm is nearly straight; stored linearly the same table is wrong
//! by 1.4 % at 806 K where stored logarithmically it is wrong by under
//! 0.1 %. Below about 250 K the samples reach `f32`'s floor and read as
//! [`LOG_FLOOR`] — 500 K below the coolest glow an eye can see, so the
//! clipped range costs nothing.

use ash::vk;
use glam::{DVec3, Vec3};

use crate::error::Result;
use crate::gpu::{Buffer, Context};

/// Samples in the table, spread over `[0, MAX_KELVIN]` — grid point `k`
/// stands at `k · MAX_KELVIN / (SIZE − 1)`, which is what
/// `shaders/scene.slang`'s reader inverts.
///
/// The count is what the interpolation error asks for, not a round number:
/// `log2` radiance curves like `−c/T`, so a lerp across a step `h` is
/// wrong by about `h²·c/(4T³)` — under half a percent at 800 K here, and
/// seven percent at 256 samples.
const SIZE: usize = 2048;

/// The hottest temperature the table carries; hotter authored values read
/// as this one. Twice a blue supergiant's surface, and far past anything
/// combustion reaches.
const MAX_KELVIN: f32 = 20_000.0;

/// Whose radiance the table divides by, so that its luminance is 1 there.
const REFERENCE_KELVIN: f64 = 6500.0;

/// Stands in for log2(0) in the table — an exponent no `exp2` can return
/// anything but zero for, and finite, so interpolating against it stays
/// arithmetic rather than a NaN.
const LOG_FLOOR: f32 = -1000.0;

/// Planck's first radiation constant for spectral radiance, `2hc²`
/// (W·m²/sr), and his second, `hc/k` (m·K) — CODATA 2018.
const C1L: f64 = 1.191_042_972_397_188e-16;
const C2: f64 = 1.438_776_877e-2;

/// The visible band the CIE observer is defined over, and the step the
/// integral below walks it in — metres.
const LAMBDA_MIN: f64 = 360e-9;
const LAMBDA_MAX: f64 = 830e-9;
const LAMBDA_STEP: f64 = 1e-9;

/// Planck's law: the spectral radiance of a blackbody at `kelvin`, per
/// metre of wavelength. Zero at and below absolute zero, and zero rather
/// than infinite wherever the exponential overflows — which is the same
/// answer, a cold body emitting nothing in the visible.
fn planck(lambda: f64, kelvin: f64) -> f64 {
    let x = C2 / (lambda * kelvin);
    C1L / (lambda.powi(5) * x.exp_m1())
}

/// The CIE 1931 2° color-matching functions, as the multi-lobe Gaussian
/// fits of Wyman, Sloan and Shirley, "Simple Analytic Approximations to
/// the CIE XYZ Color Matching Functions" (JCGT 2:2, 2013) — within about
/// one percent of the tabulated observer, which is finer than the
/// difference between one published observer and the next. `lambda` is in
/// nanometres, the units the fits are stated in.
fn cie_1931(nanometres: f64) -> DVec3 {
    // Each lobe is a Gaussian whose width differs either side of its peak.
    let lobe = |peak: f64, below: f64, above: f64| {
        let t = (nanometres - peak) * if nanometres < peak { below } else { above };
        (-0.5 * t * t).exp()
    };
    DVec3::new(
        0.362 * lobe(442.0, 0.0624, 0.0374) + 1.056 * lobe(599.8, 0.0264, 0.0323)
            - 0.065 * lobe(501.1, 0.0490, 0.0382),
        0.821 * lobe(568.8, 0.0213, 0.0247) + 0.286 * lobe(530.9, 0.0613, 0.0322),
        1.217 * lobe(437.0, 0.0845, 0.0278) + 0.681 * lobe(459.0, 0.0385, 0.0725),
    )
}

/// Tristimulus values of a blackbody at `kelvin`, in arbitrary but
/// consistent units — the integral the normalization below divides out.
fn tristimulus(kelvin: f64) -> DVec3 {
    let mut xyz = DVec3::ZERO;
    let mut lambda = LAMBDA_MIN;
    while lambda <= LAMBDA_MAX {
        xyz += cie_1931(lambda * 1e9) * planck(lambda, kelvin) * LAMBDA_STEP;
        lambda += LAMBDA_STEP;
    }
    xyz
}

/// What a blackbody at `kelvin` emits, in `ACEScg`, scaled so that
/// [`REFERENCE_KELVIN`] has luminance 1. The reference the table is baked
/// from and the gates compare against.
#[must_use]
pub fn radiance(kelvin: f64) -> Vec3 {
    let norm = tristimulus(REFERENCE_KELVIN).y;
    let xyz = tristimulus(kelvin) / norm;
    (crate::color::acescg_from_xyz() * xyz.as_vec3()).max(Vec3::ZERO)
}

/// The table the GPU reads: `SIZE` samples of log2 [`radiance`], one
/// `float4` each (the fourth lane is padding — a `float3` array would make
/// the shader's two neighbour loads straddle its stride).
#[must_use]
pub fn table() -> Vec<[f32; 4]> {
    (0..SIZE)
        .map(|k| {
            let kelvin = f64::from(MAX_KELVIN) * k as f64 / (SIZE - 1) as f64;
            let rgb = radiance(kelvin);
            let log2 = |x: f32| if x > 0.0 { x.log2() } else { LOG_FLOOR };
            [log2(rgb.x), log2(rgb.y), log2(rgb.z), 0.0]
        })
        .collect()
}

/// Upload [`table`] for the scene table to point at.
///
/// # Errors
///
/// GPU errors from allocation or upload.
pub fn upload(gpu: &Context) -> Result<Buffer> {
    gpu.upload_buffer(
        "scene.blackbody",
        bytemuck::cast_slice(&table()),
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `scene.slang`'s reader, in Rust: bound, scale onto the grid, lerp
    /// the logarithms, exponentiate. The render gate is what pins the two
    /// against each other; this exists so the tests below measure the
    /// error the shader will actually make — including on the inputs a
    /// temperature field can hold that a Kelvin cannot.
    fn sample(table: &[[f32; 4]], kelvin: f32) -> Vec3 {
        let bounded = if kelvin > 0.0 { kelvin.min(MAX_KELVIN) } else { 0.0 };
        let x = bounded * ((SIZE - 1) as f32 / MAX_KELVIN);
        let lo = (x as usize).min(SIZE - 1);
        let hi = (lo + 1).min(SIZE - 1);
        let f = x - lo as f32;
        Vec3::from_slice(&table[lo][..3])
            .lerp(Vec3::from_slice(&table[hi][..3]), f)
            .exp2()
    }

    /// The normalization, which is the table's one free constant.
    #[test]
    fn the_reference_temperature_emits_unit_luminance() {
        let y = crate::color::luminance(radiance(REFERENCE_KELVIN));
        assert!((y - 1.0).abs() < 1e-4, "{y}");
    }

    /// Interpolating stored logarithms recovers the integral itself across
    /// the range fire and filaments occupy — the claim that justifies a
    /// table at all rather than an in-shader fit, and the claim that decides
    /// how many samples it needs.
    ///
    /// Sampled at cell *midpoints*, where interpolation is at its worst: the
    /// grid points themselves are exact by construction, and anything nearer
    /// one would flatter whatever curve the table stored.
    #[test]
    fn the_table_recovers_planck_between_its_samples() {
        let table = table();
        let step = f64::from(MAX_KELVIN) / (SIZE - 1) as f64;
        for cell in [82, 126, 194, 307, 665, 1137, 2046] {
            let kelvin = (f64::from(cell) + 0.5) * step;
            let got = sample(&table, kelvin as f32);
            let want = radiance(kelvin);
            for channel in 0..3 {
                let error = f64::from((got[channel] - want[channel]).abs() / want[channel]);
                assert!(error < 0.005, "{kelvin} K channel {channel}: {got} vs {want}");
            }
        }
    }

    /// Color runs red at the bottom of the range and blue at the top,
    /// monotonically — the chromaticity check a per-channel magnitude
    /// comparison cannot make.
    #[test]
    fn colour_cools_from_red_to_blue() {
        let mut previous = f32::INFINITY;
        for step in 1..40 {
            let rgb = radiance(f64::from(step) * 500.0);
            let ratio = rgb.x / rgb.z;
            assert!(ratio < previous, "{} K: {rgb}", step * 500);
            previous = ratio;
        }
    }

    /// Nothing radiates at absolute zero, and the floor that stands in for
    /// its logarithm exponentiates back to nothing rather than to a
    /// denormal or a NaN.
    #[test]
    fn absolute_zero_emits_nothing() {
        assert_eq!(radiance(0.0), Vec3::ZERO);
        assert_eq!(sample(&table(), 0.0), Vec3::ZERO);
    }

    /// A temperature field is authored data run through a scale and an
    /// offset, so the reader is handed values no thermometer would
    /// produce. None of them may reach outside the table: a negative or a
    /// NaN Kelvin emits nothing, and everything past the top sample reads
    /// as that sample rather than off the end of the buffer.
    #[test]
    fn temperatures_no_thermometer_would_read_stay_inside_the_table() {
        let table = table();
        for cold in [-1.0, -1e30, f32::NEG_INFINITY, f32::NAN] {
            assert_eq!(sample(&table, cold), Vec3::ZERO, "{cold} K");
        }
        let top = sample(&table, MAX_KELVIN);
        for hot in [MAX_KELVIN + 1.0, 1e30, f32::INFINITY] {
            assert_eq!(sample(&table, hot), top, "{hot} K");
        }
        // And the top sample is the integral there, not an extrapolation
        // the clamp landed on.
        let want = radiance(f64::from(MAX_KELVIN));
        assert!(top.abs_diff_eq(want, 1e-3 * want.max_element()), "{top} vs {want}");
    }
}
