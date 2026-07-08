//! Color-space prep. Colors are authored in linear `Rec.709` (everyday sRGB
//! primaries); the render core works exclusively in `ACEScg` (AP1 primaries,
//! D60 white). The conversion happens once, here, when a scene is prepared
//! — never in kernels — and the tonemap kernel's display transform carries
//! the matching `ACEScg` → sRGB matrix back out.

use glam::{Mat3, Vec3};

/// Linear `Rec.709` → `ACEScg`, Bradford-adapted D65 → D60: the exact inverse
/// of the tonemap kernel's `SRGB_FROM_ACESCG` (`shaders/tonemap.slang`),
/// so an authored color round-trips through the pipeline. Agrees with the
/// published ACES matrices to float precision.
const ACESCG_FROM_REC709: Mat3 = Mat3::from_cols(
    Vec3::new(0.613_097_4, 0.070_193_73, 0.020_615_594),
    Vec3::new(0.339_523_14, 0.916_353_9, 0.109_569_78),
    Vec3::new(0.047_379_45, 0.013_452_399, 0.869_814_66),
);

/// An authored linear `Rec.709` color, expressed in `ACEScg`.
#[must_use]
pub fn acescg_from_rec709(rec709: Vec3) -> Vec3 {
    ACESCG_FROM_REC709 * rec709
}

/// Luminance of an `ACEScg` color — the Y row of the AP1 RGB→XYZ matrix
/// (ACES TB S-2014-004). The scalar "how bright" every power-proportional
/// sampling decision (light selection, environment CDFs) weighs by.
#[must_use]
pub fn luminance(acescg: Vec3) -> f32 {
    acescg.dot(Vec3::new(0.272_228_7, 0.674_081_8, 0.053_689_5))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// White is the same white in both spaces — chromatic adaptation maps
    /// the `Rec.709` white point onto ACES D60 by construction.
    #[test]
    fn white_maps_to_white() {
        let white = acescg_from_rec709(Vec3::ONE);
        assert!(white.abs_diff_eq(Vec3::ONE, 1e-6), "{white}");
    }

    /// The primaries land on their published `ACEScg` coordinates (ACES TB
    /// S-2014-004 derivation, Bradford CAT) — an independent check that the
    /// matrix is the right transform and the right way around.
    #[test]
    fn primaries_match_published_values() {
        let cases = [
            (Vec3::X, Vec3::new(0.613_097, 0.070_194, 0.020_616)),
            (Vec3::Y, Vec3::new(0.339_523, 0.916_354, 0.109_570)),
            (Vec3::Z, Vec3::new(0.047_379, 0.013_452, 0.869_815)),
        ];
        for (input, expected) in cases {
            let got = acescg_from_rec709(input);
            assert!(got.abs_diff_eq(expected, 1e-5), "{input} -> {got}");
        }
    }
}
