//! The closure's baked lookup tables — GGX energy data and the LTC sheen
//! fit — embedded in the binary and uploaded once per scene, reached
//! through the scene table like everything else the kernels share.
//!
//! Two blobs with two provenances, concatenated into one GPU buffer whose
//! layout `shaders/openpbr.slang` mirrors constant for constant (the
//! `TABLE_*` offsets below):
//!
//! - `tables/ggx_energy.bin` is **baked here**, by [`bake`], against this
//!   kernel's exact integrand (GGX with `alpha = roughness²`, separable
//!   Smith `G1·G1`, spherical-caps VNDF sampling, exact dielectric
//!   Fresnel) — the same rule the M1 albedo fits followed. Regenerate
//!   with `cargo test -p cenote --release regenerate_closure_tables --
//!   --ignored` after changing any of that math; the spot-check tests
//!   below fail loudly if the blob and the integrand drift apart.
//! - `tables/ltc_sheen.bin` is **vendored, never regenerated**: the
//!   32×32 volumetric-fit table from Zeltner, Burley & Chiang,
//!   "Practical Multiple-Scattering Sheen Using Linearly Transformed
//!   Cosines" (SIGGRAPH 2022), taken at full precision from the authors'
//!   reference implementation, `github.com/tizian/ltc-sheen` @ 9262411
//!   (`fitting/python/data/ltc_table_sheen_volume.npy`), Apache-2.0,
//!   © Tizian Zeltner. The table *is* the sheen BSDF's definition — there
//!   is no integrand to rebake it from.
//!
//! Every table reads like Cycles' `lookup_table_read*`: grid point `k` of
//! an `n`-wide axis sits at coordinate `k/(n−1)` in [0, 1], sampled with
//! clamped multilinear interpolation. Axis order matches Cycles too —
//! the first-named axis varies fastest.

use crate::error::Result;
use crate::gpu::{Buffer, Context};

/// `tables/ggx_energy.bin`, baked by [`bake`].
static GGX_ENERGY: &[u8] = include_bytes!("tables/ggx_energy.bin");

/// `tables/ltc_sheen.bin`, vendored (see the module doc for provenance).
static LTC_SHEEN: &[u8] = include_bytes!("tables/ltc_sheen.bin");

/// Axis resolution of the 2D reflection-energy tables.
const GGX_SIZE: usize = 32;
/// Axis resolution of the 3D dielectric and glass tables.
const GLASS_SIZE: usize = 16;
/// Axis resolution of the LTC sheen table.
const SHEEN_SIZE: usize = 32;

// The buffer layout, in floats. `shaders/openpbr.slang` carries the same
// constants; the two lists must match entry for entry.
//
// E(rough, µ): directional albedo of the single-scatter GGX reflection
// lobe with Fresnel ≡ 1 — x = roughness (fastest), y = µ = cos θ.
const TABLE_GGX_E: usize = 0;
// E_avg(rough): its cosine-weighted average over the hemisphere.
const TABLE_GGX_E_AVG: usize = TABLE_GGX_E + GGX_SIZE * GGX_SIZE;
// E_glass(rough, µ, z): energy of the single-scatter dielectric
// reflection+refraction closure at IOR ≥ 1, z = √((η−1)/(η+1)).
const TABLE_GLASS_E: usize = TABLE_GGX_E_AVG + GGX_SIZE;
// Its cosine-weighted average, (rough, z).
const TABLE_GLASS_E_AVG: usize = TABLE_GLASS_E + GLASS_SIZE * GLASS_SIZE * GLASS_SIZE;
// The η < 1 branch (IOR inverted before the z remap).
const TABLE_GLASS_INV_E: usize = TABLE_GLASS_E_AVG + GLASS_SIZE * GLASS_SIZE;
const TABLE_GLASS_INV_E_AVG: usize = TABLE_GLASS_INV_E + GLASS_SIZE * GLASS_SIZE * GLASS_SIZE;
// E_spec(rough, µ, z): directional albedo of the *compensated* dielectric
// reflection lobe — exact Fresnel at η ≥ 1 times the multiple-scattering
// scale — the layering weight everything under a dielectric interface is
// scaled by.
const TABLE_DIELECTRIC_E: usize = TABLE_GLASS_INV_E_AVG + GLASS_SIZE * GLASS_SIZE;
/// Floats in `ggx_energy.bin`.
const GGX_ENERGY_LEN: usize = TABLE_DIELECTRIC_E + GLASS_SIZE * GLASS_SIZE * GLASS_SIZE;

// The sheen table appends after the baked blob: three 32×32 planes —
// the inverse-LTC-transform entries A and B, then the directional albedo
// R — each x = µ = cos θ (fastest), y = fuzz roughness.
#[cfg(test)]
const TABLE_SHEEN_A: usize = GGX_ENERGY_LEN;
#[cfg(test)]
const TABLE_SHEEN_B: usize = TABLE_SHEEN_A + SHEEN_SIZE * SHEEN_SIZE;
#[cfg(test)]
const TABLE_SHEEN_R: usize = TABLE_SHEEN_B + SHEEN_SIZE * SHEEN_SIZE;
/// Floats in `ltc_sheen.bin`.
const LTC_SHEEN_LEN: usize = 3 * SHEEN_SIZE * SHEEN_SIZE;

/// Upload the concatenated table buffer — called once per scene build
/// (the scene's resident buffers own it, and its address rides the scene
/// table).
///
/// # Errors
///
/// Any [`crate::Error`] from the upload.
pub(crate) fn upload(gpu: &Context) -> Result<Buffer> {
    assert_eq!(GGX_ENERGY.len(), GGX_ENERGY_LEN * 4, "ggx_energy.bin size");
    assert_eq!(LTC_SHEEN.len(), LTC_SHEEN_LEN * 4, "ltc_sheen.bin size");
    let mut bytes = Vec::with_capacity(GGX_ENERGY.len() + LTC_SHEEN.len());
    bytes.extend_from_slice(GGX_ENERGY);
    bytes.extend_from_slice(LTC_SHEEN);
    gpu.upload_buffer(
        "scene.bsdf_tables",
        &bytes,
        ash::vk::BufferUsageFlags::STORAGE_BUFFER
            | ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
    )
}

/// The baker: quasi–Monte Carlo tabulation of the kernel's own GGX
/// integrand. Pure host math, mirroring `shaders/openpbr.slang` function
/// for function — a mismatch here *is* a furnace leak, which is what the
/// spot-check tests pin.
#[cfg(test)]
mod bake {
    use glam::Vec3;

    use super::{GGX_SIZE, GLASS_SIZE};

    /// Samples per table entry. The G1 estimator below is low-variance
    /// (each VNDF sample contributes a value in [0, 1]), so this lands
    /// entries well inside 1e-3 of the integral.
    const SAMPLES: u32 = 1 << 16;

    /// Point `i` of a 3D Hammersley-style set: (i/n, φ₂(i), φ₃(i)) —
    /// deterministic, so regeneration is reproducible.
    fn quasi_random(i: u32, n: u32) -> [f32; 3] {
        [
            i as f32 / n as f32,
            radical_inverse(i, 2),
            radical_inverse(i, 3),
        ]
    }

    fn radical_inverse(mut i: u32, base: u32) -> f32 {
        let inv = 1.0 / f64::from(base);
        let (mut reversed, mut digit) = (0.0_f64, inv);
        while i > 0 {
            reversed += f64::from(i % base) * digit;
            digit *= inv;
            i /= base;
        }
        reversed as f32
    }

    // -- Mirrors of the kernel's GGX pieces (openpbr.slang) --------------

    fn smith_g1(alpha: f32, vz: f32) -> f32 {
        let a2 = alpha * alpha;
        2.0 * vz.abs() / (vz.abs() + (vz * vz * (1.0 - a2) + a2).sqrt())
    }

    /// Dupuy & Benyoub spherical-caps VNDF sample, as the kernel does it.
    fn sample_vndf(wo: Vec3, alpha: f32, u: [f32; 2]) -> Vec3 {
        let wo_std = Vec3::new(wo.x * alpha, wo.y * alpha, wo.z).normalize();
        let phi = 2.0 * std::f32::consts::PI * u[0];
        let z = (1.0 - u[1]).mul_add(1.0 + wo_std.z, -wo_std.z);
        let sin_theta = (1.0 - z * z).clamp(0.0, 1.0).sqrt();
        let c = Vec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), z);
        let h_std = c + wo_std;
        Vec3::new(h_std.x * alpha, h_std.y * alpha, h_std.z).normalize()
    }

    /// Exact unpolarized dielectric Fresnel, `eta` the relative IOR on
    /// the incident side.
    pub fn fresnel_dielectric(eta: f32, cosine: f32) -> f32 {
        let mu = cosine.abs();
        let sin2 = 1.0 - mu * mu;
        let cos2t = 1.0 - sin2 / (eta * eta);
        if cos2t < 0.0 {
            return 1.0; // total internal reflection
        }
        let t0 = cos2t.sqrt();
        let t1 = eta * t0;
        let t2 = eta * mu;
        let rs = (mu - t1) / (mu + t1);
        let rp = (t0 - t2) / (t0 + t2);
        f32::midpoint(rs * rs, rp * rp)
    }

    /// Kulla & Conty's fit of the cosine-weighted average dielectric
    /// Fresnel ("Revisiting Physically Based Shading at Imageworks"),
    /// both branches — the closed form the kernel's multiple-scattering
    /// term uses.
    pub fn fresnel_dielectric_average(eta: f32) -> f32 {
        if eta < 1.0 {
            0.130_607_f32
                .mul_add(-eta, -0.965_241)
                .mul_add(eta, 0.1014)
                .mul_add(eta, 0.997_118)
        } else {
            (eta - 1.0) / 1.000_71_f32.mul_add(eta, 4.085_67)
        }
    }

    /// A grid axis coordinate: point `k` of `n` sits at `k/(n−1)`.
    fn grid(k: usize, n: usize) -> f32 {
        k as f32 / (n - 1) as f32
    }

    /// The view direction at cos θ = µ, in the BSDF's local frame.
    fn view(mu: f32) -> Vec3 {
        let mu = mu.max(1e-4);
        Vec3::new((1.0 - mu * mu).max(0.0).sqrt(), 0.0, mu)
    }

    /// The IOR at glass-table coordinate `z = √((η−1)/(η+1))`, capped
    /// where z → 1 sends it to infinity.
    fn ior_from_remap(z: f32) -> f32 {
        let z2 = z * z;
        ((1.0 + z2) / (1.0 - z2).max(1e-4)).min(1e4)
    }

    /// Directional albedo of the single-scatter GGX reflection lobe with
    /// Fresnel ≡ 1: sampling the VNDF makes the estimator collapse to
    /// `E[G1(wi)]`, since `f·cosθᵢ/pdf = G1(wi)` exactly.
    pub fn ggx_albedo(rough: f32, mu: f32) -> f32 {
        let alpha = rough * rough;
        let wo = view(mu);
        let mut sum = 0.0_f64;
        for i in 0..SAMPLES {
            let u = quasi_random(i, SAMPLES);
            let h = sample_vndf(wo, alpha, [u[0], u[1]]);
            let wi = 2.0 * wo.dot(h) * h - wo;
            if wi.z > 0.0 {
                sum += f64::from(smith_g1(alpha, wi.z));
            }
        }
        (sum / f64::from(SAMPLES)) as f32
    }

    /// Its cosine-weighted hemisphere average: µ drawn with pdf 2µ.
    pub fn ggx_albedo_average(rough: f32) -> f32 {
        let alpha = rough * rough;
        let mut sum = 0.0_f64;
        for i in 0..SAMPLES {
            let u = quasi_random(i, SAMPLES);
            let wo = view(u[2].sqrt());
            let h = sample_vndf(wo, alpha, [u[0], u[1]]);
            let wi = 2.0 * wo.dot(h) * h - wo;
            if wi.z > 0.0 {
                sum += f64::from(smith_g1(alpha, wi.z));
            }
        }
        (sum / f64::from(SAMPLES)) as f32
    }

    /// Walter-style refraction of `wo` about half-vector `h` at relative
    /// IOR `eta`; `None` past the critical angle.
    fn refract(wo: Vec3, h: Vec3, eta: f32) -> Option<Vec3> {
        let c = wo.dot(h);
        let cos2t = 1.0 - (1.0 - c * c) / (eta * eta);
        if cos2t <= 0.0 {
            return None;
        }
        Some((c / eta - cos2t.sqrt()) * h - wo / eta)
    }

    /// Energy of the single-scatter glass closure (reflection +
    /// refraction, exact Fresnel choosing the side): `E[G1(wi)]` again,
    /// with samples that land on the wrong side of the surface counting
    /// zero — exactly the energy the kernel loses and divides back.
    pub fn glass_energy(rough: f32, mu: f32, eta: f32) -> f32 {
        glass_energy_n(rough, mu, eta, SAMPLES)
    }

    /// Its cosine-weighted hemisphere average — a sixteenth of the samples
    /// per view angle, since integrating over 256 of them averages the
    /// per-angle noise back out.
    pub fn glass_energy_average(rough: f32, eta: f32) -> f32 {
        const CHUNK: u32 = 256;
        let mut sum = 0.0_f64;
        for k in 0..CHUNK {
            let mu = (f64::from(k) + 0.5) / f64::from(CHUNK);
            let mu = (mu.sqrt()) as f32; // pdf 2µ via inverse CDF
            sum += f64::from(glass_energy_n(rough, mu, eta, SAMPLES / 16));
        }
        (sum / f64::from(CHUNK)) as f32
    }

    /// [`glass_energy`] at an explicit sample budget — the directional value
    /// spends the full count, its average spends a fraction per angle.
    fn glass_energy_n(rough: f32, mu: f32, eta: f32, samples: u32) -> f32 {
        let alpha = rough * rough;
        let wo = view(mu);
        let mut sum = 0.0_f64;
        for i in 0..samples {
            let u = quasi_random(i, samples);
            let h = sample_vndf(wo, alpha, [u[0], u[1]]);
            let f = fresnel_dielectric(eta, wo.dot(h));
            let wi = if u[2] < f {
                2.0 * wo.dot(h) * h - wo
            } else {
                match refract(wo, h, eta) {
                    Some(wt) => wt,
                    None => continue, // unreachable: F = 1 past the critical angle
                }
            };
            // A reflection must leave above the surface, a refraction
            // below; anything else is rejected by the kernel and its
            // energy lost.
            if (u[2] < f && wi.z > 0.0) || (u[2] >= f && wi.z < 0.0) {
                sum += f64::from(smith_g1(alpha, wi.z));
            }
        }
        (sum / f64::from(samples)) as f32
    }

    /// Directional albedo of the *compensated* dielectric reflection
    /// lobe: the Fresnel-weighted single-scatter albedo `E[F·G1]`, times
    /// the same multiple-scattering scale the kernel applies
    /// (`1 + Fms·(1−E)/E`, Fresnel-free `E`/`E_avg` with the analytic
    /// average Fresnel). This is the layering weight's ground truth — bake
    /// it
    /// from the finished lobe so `1 − E_spec` closes the furnace.
    pub fn dielectric_albedo(rough: f32, mu: f32, eta: f32) -> f32 {
        let alpha = rough * rough;
        let wo = view(mu);
        let mut sum = 0.0_f64;
        for i in 0..SAMPLES {
            let u = quasi_random(i, SAMPLES);
            let h = sample_vndf(wo, alpha, [u[0], u[1]]);
            let wi = 2.0 * wo.dot(h) * h - wo;
            if wi.z > 0.0 {
                sum += f64::from(fresnel_dielectric(eta, wo.dot(h)) * smith_g1(alpha, wi.z));
            }
        }
        let single = (sum / f64::from(SAMPLES)) as f32;

        let e = ggx_albedo(rough, mu).max(1e-4);
        let e_avg = ggx_albedo_average(rough);
        let fss = fresnel_dielectric_average(eta);
        let fms = fss * e_avg / (1.0 - fss * (1.0 - e_avg));
        single * (1.0 + fms * (1.0 - e) / e)
    }

    /// Bake `tables/ggx_energy.bin` in the module-doc layout.
    pub fn ggx_energy_blob() -> Vec<f32> {
        let mut blob = Vec::with_capacity(super::GGX_ENERGY_LEN);
        for mu in 0..GGX_SIZE {
            for rough in 0..GGX_SIZE {
                blob.push(ggx_albedo(grid(rough, GGX_SIZE), grid(mu, GGX_SIZE)));
            }
        }
        for rough in 0..GGX_SIZE {
            blob.push(ggx_albedo_average(grid(rough, GGX_SIZE)));
        }
        for inverted in [false, true] {
            for z in 0..GLASS_SIZE {
                let mut eta = ior_from_remap(grid(z, GLASS_SIZE));
                if inverted {
                    eta = 1.0 / eta;
                }
                for mu in 0..GLASS_SIZE {
                    for rough in 0..GLASS_SIZE {
                        blob.push(glass_energy(
                            grid(rough, GLASS_SIZE),
                            grid(mu, GLASS_SIZE),
                            eta,
                        ));
                    }
                }
            }
            for z in 0..GLASS_SIZE {
                let mut eta = ior_from_remap(grid(z, GLASS_SIZE));
                if inverted {
                    eta = 1.0 / eta;
                }
                for rough in 0..GLASS_SIZE {
                    blob.push(glass_energy_average(grid(rough, GLASS_SIZE), eta));
                }
            }
        }
        for z in 0..GLASS_SIZE {
            let eta = ior_from_remap(grid(z, GLASS_SIZE));
            for mu in 0..GLASS_SIZE {
                for rough in 0..GLASS_SIZE {
                    blob.push(dielectric_albedo(
                        grid(rough, GLASS_SIZE),
                        grid(mu, GLASS_SIZE),
                        eta,
                    ));
                }
            }
        }
        assert_eq!(blob.len(), super::GGX_ENERGY_LEN);
        blob
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn embedded(offset: usize, index: usize) -> f32 {
        let bytes = &GGX_ENERGY[(offset + index) * 4..(offset + index) * 4 + 4];
        f32::from_le_bytes(bytes.try_into().expect("four bytes"))
    }

    fn sheen(offset: usize, index: usize) -> f32 {
        let local = offset + index - TABLE_SHEEN_A;
        let bytes = &LTC_SHEEN[local * 4..local * 4 + 4];
        f32::from_le_bytes(bytes.try_into().expect("four bytes"))
    }

    /// Rebake `tables/ggx_energy.bin` from the integrand. `--ignored`
    /// because it takes minutes in a debug build — run release:
    /// `cargo test -p cenote --release regenerate_closure_tables -- --ignored`
    #[test]
    #[ignore = "regenerates tables/ggx_energy.bin; run explicitly, in release"]
    fn regenerate_closure_tables() {
        let blob = bake::ggx_energy_blob();
        let bytes: Vec<u8> = blob.iter().flat_map(|v| v.to_le_bytes()).collect();
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/tables/ggx_energy.bin");
        std::fs::write(path, bytes).expect("write ggx_energy.bin");
    }

    /// A handful of embedded entries re-integrated from scratch: the blob
    /// must match this kernel's integrand, not just *an* integrand. Runs
    /// on every entry class (`E`, `E_avg`, glass, inverted glass,
    /// dielectric) at interior and edge grid points.
    #[test]
    fn embedded_tables_match_the_integrand() {
        let ggx = GGX_SIZE - 1;
        let glass = GLASS_SIZE - 1;
        let grid2 = |k: usize| k as f32 / ggx as f32;
        let grid3 = |k: usize| k as f32 / glass as f32;
        let eta = |z: f32| {
            let z2 = z * z;
            ((1.0 + z2) / (1.0 - z2).max(1e-4)).min(1e4)
        };

        // (mu index, rough index) probes for the 2D table.
        for (mu, rough) in [(31, 8), (16, 16), (4, 31), (0, 2)] {
            let expected = bake::ggx_albedo(grid2(rough), grid2(mu));
            let stored = embedded(TABLE_GGX_E, mu * GGX_SIZE + rough);
            assert!(
                (stored - expected).abs() < 2e-3,
                "ggx_E[mu {mu}][rough {rough}]: {stored} vs {expected}"
            );
        }
        for rough in [0, 12, 31] {
            let expected = bake::ggx_albedo_average(grid2(rough));
            let stored = embedded(TABLE_GGX_E_AVG, rough);
            assert!(
                (stored - expected).abs() < 2e-3,
                "ggx_E_avg[rough {rough}]: {stored} vs {expected}"
            );
        }
        for (z, mu, rough) in [(4, 12, 8), (15, 3, 15), (0, 8, 4)] {
            let expected = bake::glass_energy(grid3(rough), grid3(mu), eta(grid3(z)));
            let stored = embedded(TABLE_GLASS_E, (z * GLASS_SIZE + mu) * GLASS_SIZE + rough);
            assert!(
                (stored - expected).abs() < 2e-3,
                "glass_E[z {z}][mu {mu}][rough {rough}]: {stored} vs {expected}"
            );
            let expected = bake::glass_energy(grid3(rough), grid3(mu), 1.0 / eta(grid3(z)));
            let stored = embedded(
                TABLE_GLASS_INV_E,
                (z * GLASS_SIZE + mu) * GLASS_SIZE + rough,
            );
            assert!(
                (stored - expected).abs() < 2e-3,
                "glass_inv_E[z {z}][mu {mu}][rough {rough}]: {stored} vs {expected}"
            );
            let expected = bake::dielectric_albedo(grid3(rough), grid3(mu), eta(grid3(z)));
            let stored = embedded(
                TABLE_DIELECTRIC_E,
                (z * GLASS_SIZE + mu) * GLASS_SIZE + rough,
            );
            assert!(
                (stored - expected).abs() < 3e-3,
                "dielectric_E[z {z}][mu {mu}][rough {rough}]: {stored} vs {expected}"
            );
        }
        for (z, rough) in [(6, 10), (12, 2)] {
            let expected = bake::glass_energy_average(grid3(rough), eta(grid3(z)));
            let stored = embedded(TABLE_GLASS_E_AVG, z * GLASS_SIZE + rough);
            assert!(
                (stored - expected).abs() < 3e-3,
                "glass_E_avg[z {z}][rough {rough}]: {stored} vs {expected}"
            );
        }
    }

    /// The energy data behaves like energy: everything in [0, 1], the
    /// reflection albedo 1 at roughness 0 (a mirror with F ≡ 1 loses
    /// nothing), decreasing in roughness at fixed µ.
    #[test]
    fn energy_tables_are_physical() {
        for i in 0..GGX_ENERGY_LEN {
            let value = embedded(0, i);
            assert!((0.0..=1.0 + 1e-3).contains(&value), "entry {i}: {value}");
        }
        for mu in 0..GGX_SIZE {
            let mirror = embedded(TABLE_GGX_E, mu * GGX_SIZE);
            assert!(
                (mirror - 1.0).abs() < 2e-3,
                "a smooth F≡1 mirror must reflect 1, got {mirror} at mu {mu}"
            );
        }
        for mu in [8, 31] {
            let smooth = embedded(TABLE_GGX_E, mu * GGX_SIZE + 4);
            let rough = embedded(TABLE_GGX_E, mu * GGX_SIZE + 31);
            assert!(
                smooth > rough,
                "single-scatter energy must fall with roughness: {smooth} vs {rough}"
            );
        }
    }

    /// The vendored sheen table is the one documented in the module doc:
    /// its corner entries match the published fit (transposed to this
    /// layout), and its degenerate entries — the fit skips (roughness, µ)
    /// cells whose reflectance is ~0, leaving A = 0 — always carry a
    /// near-zero albedo, which is what lets the kernel treat A ≈ 0 as
    /// "no fuzz lobe here" without losing real energy.
    #[test]
    fn sheen_table_matches_its_provenance() {
        // (rough index, mu index, A, B, R) from the published npy.
        let corners = [
            (0, 0, 0.014_150_003, 0.000_604_314_8, 1.4e-5),
            (1, 0, 0.019_405_415, -0.002_323_515, 0.058_39),
            (30, 31, 0.866_774_4, 1.869_452_5e-5, 0.315_084),
            (31, 31, 0.879_580_4, 2.791_145_8e-5, 0.341_875),
        ];
        for (rough, mu, a, b, r) in corners {
            let index = rough * SHEEN_SIZE + mu;
            assert!((sheen(TABLE_SHEEN_A, index) - a).abs() < 1e-6);
            assert!((sheen(TABLE_SHEEN_B, index) - b).abs() < 1e-6);
            assert!((sheen(TABLE_SHEEN_R, index) - r).abs() < 1e-6);
        }
        for index in 0..SHEEN_SIZE * SHEEN_SIZE {
            if sheen(TABLE_SHEEN_A, index).abs() < 1e-5 {
                assert!(
                    sheen(TABLE_SHEEN_R, index) < 1e-3,
                    "a degenerate LTC entry carries real energy at {index}"
                );
            }
        }
    }
}
