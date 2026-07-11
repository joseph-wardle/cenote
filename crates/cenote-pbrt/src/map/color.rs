//! Color science the mapper needs to lower pbrt spectra: conductor
//! complex-IOR reflectance, the named-metal F0 table, and a blackbody's
//! chromaticity. Pure functions over numbers — no graphics state — kept
//! apart from the mapping machinery.

/// Normal-incidence reflectance from a conductor's complex IOR — how
/// pbrt's `eta`/`k` spectra land in `base_color`'s F0 convention.
pub(super) fn conductor_f0(eta: [f32; 3], k: [f32; 3]) -> [f32; 3] {
    let mut f0 = [0.0; 3];
    for channel in 0..3 {
        let (n, k) = (eta[channel], k[channel]);
        f0[channel] = ((n - 1.0).powi(2) + k * k) / ((n + 1.0).powi(2) + k * k);
    }
    f0
}

/// Linear `Rec.709` F0 for the named conductor spectra the corpus uses
/// (pbrt's `metal-*-eta`/`-k` measurements, reduced to normal-incidence
/// RGB — the standard lookdev values).
pub(super) fn named_metal_f0(spectrum: &str) -> Option<[f32; 3]> {
    let metal = spectrum.strip_prefix("metal-")?;
    let metal = metal
        .strip_suffix("-eta")
        .or_else(|| metal.strip_suffix("-k"))?;
    Some(match metal {
        "Cu" => [0.955, 0.638, 0.538],
        "Au" => [1.000, 0.782, 0.344],
        "Ag" => [0.972, 0.960, 0.915],
        "Al" => [0.913, 0.921, 0.925],
        _ => return None,
    })
}

/// A blackbody's chromaticity as linear `Rec.709`, normalized to
/// luminance 1 — matching pbrt, which normalizes blackbody emitters to
/// 1 nit before its photometric scale (trap 1's blackbody half).
/// Krystek's Planckian-locus approximation in CIE 1960 UCS, accurate to
/// ~1e-3 in chromaticity over 1000–15000 K.
#[expect(
    clippy::many_single_char_names,
    reason = "the CIE variables are named what colorimetry names them"
)]
pub(super) fn blackbody_rec709(kelvin: f32) -> [f32; 3] {
    let t = f64::from(kelvin).clamp(1000.0, 15000.0);
    let u = (0.860_117_757 + 1.541_182_54e-4 * t + 1.286_412_12e-7 * t * t)
        / (1.0 + 8.424_202_35e-4 * t + 7.081_451_63e-7 * t * t);
    let v = (0.317_398_726 + 4.228_062_45e-5 * t + 4.204_816_91e-8 * t * t)
        / (1.0 - 2.897_418_16e-5 * t + 1.614_560_53e-7 * t * t);
    // CIE 1960 uv → xy → XYZ at Y = 1 → linear Rec.709.
    let x = 3.0 * u / (2.0 * u - 8.0 * v + 4.0);
    let y = 2.0 * v / (2.0 * u - 8.0 * v + 4.0);
    let xyz = [x / y, 1.0, (1.0 - x - y) / y];
    let rgb = [
        3.240_454_2 * xyz[0] - 1.537_138_5 * xyz[1] - 0.498_531_4 * xyz[2],
        -0.969_266_0 * xyz[0] + 1.876_010_8 * xyz[1] + 0.041_556_0 * xyz[2],
        0.055_643_4 * xyz[0] - 0.204_025_9 * xyz[1] + 1.057_225_2 * xyz[2],
    ];
    // Warm temperatures fall outside the Rec.709 gamut: clamp, then
    // restore unit luminance.
    let rgb = rgb.map(|channel| channel.max(0.0) as f32);
    let luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
    rgb.map(|channel| channel / luminance)
}
