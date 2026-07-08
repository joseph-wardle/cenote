//! The light list: quad lights extracted from a scene's emissive objects,
//! plus the power-proportional alias table next-event estimation selects
//! from. Everything here runs once at scene prep; the GPU sees only the
//! finished [`LightRecord`]s, which `shaders/lights.slang` mirrors field
//! for field.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use crate::color::luminance;

/// The geometry record's light index for instances that are not lights.
/// Matches `LIGHT_NONE` in `shaders/lights.slang`.
pub(crate) const LIGHT_NONE: u32 = u32::MAX;

/// One quad light as the scene describes it: a world-space parallelogram
/// spanned from a corner, radiating `emission` from both faces (surfaces
/// are two-sided throughout the renderer).
pub struct QuadLight {
    /// One vertex of the parallelogram.
    pub corner: Vec3,
    /// The sides meeting at `corner`: the quad is
    /// `corner + u·edge1 + v·edge2`, u, v ∈ [0, 1].
    pub edge1: Vec3,
    /// See `edge1`.
    pub edge2: Vec3,
    /// Radiance, `ACEScg`, from both faces.
    pub emission: Vec3,
    /// TLAS custom index of the light's mesh instance — shadow rays test
    /// visibility by identity.
    pub instance: u32,
}

/// One light as the kernels read it: the quad, its selection-weighted
/// area pdf, and its slot of the alias table. Mirrors `LightRecord` in
/// `shaders/lights.slang` field for field.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct LightRecord {
    corner: Vec3,
    /// p(the alias table selects this light) / its area — next-event
    /// estimation's pdf for a uniform point on this quad.
    pdf_area: f32,
    edge1: Vec3,
    alias_threshold: f32,
    edge2: Vec3,
    alias_index: u32,
    emission: Vec3,
    instance: u32,
}

/// Build the GPU light list: each light's selection probability is
/// proportional to its power (emitted energy — luminance × area, with the
/// two-sided π factor common to every quad dropped), materialized as an
/// alias table so the kernel selects in O(1). One slot per light, carried
/// on the light's own record.
///
/// # Panics
///
/// On a degenerate (zero-area) quad or non-positive emission — programmer
/// bugs in scene construction.
pub(crate) fn build(lights: &[QuadLight]) -> Vec<LightRecord> {
    let powers: Vec<f64> = lights
        .iter()
        .map(|light| {
            let area = f64::from(light.edge1.cross(light.edge2).length());
            assert!(area > 0.0, "degenerate light quad");
            let luma = luminance(light.emission);
            assert!(luma > 0.0, "a light must emit: {:?}", light.emission);
            f64::from(luma) * area
        })
        .collect();
    let total: f64 = powers.iter().sum();

    let mut records: Vec<LightRecord> = lights
        .iter()
        .zip(&powers)
        .map(|(light, power)| LightRecord {
            corner: light.corner,
            pdf_area: ((power / total) / f64::from(light.edge1.cross(light.edge2).length())) as f32,
            edge1: light.edge1,
            alias_threshold: 1.0,
            edge2: light.edge2,
            alias_index: 0,
            emission: light.emission,
            instance: light.instance,
        })
        .collect();

    // Walker/Vose: rescale so the average power is 1 slot; every light
    // poorer than a full slot tops its slot up from a richer light (the
    // alias), which keeps exactly its own share across the slots it fills.
    let n = powers.len() as f64;
    let mut share: Vec<f64> = powers.iter().map(|power| power * n / total).collect();
    let (mut small, mut large): (Vec<usize>, Vec<usize>) =
        (0..share.len()).partition(|&index| share[index] < 1.0);
    while let (Some(poor), Some(rich)) = (small.pop(), large.pop()) {
        records[poor].alias_threshold = share[poor] as f32;
        records[poor].alias_index = rich as u32;
        share[rich] -= 1.0 - share[poor];
        if share[rich] < 1.0 {
            small.push(rich);
        } else {
            large.push(rich);
        }
    }
    // Leftovers (including the last rich light, whose share is 1 up to
    // rounding) keep their initialized threshold 1: never alias out.
    records
}

/// The list's total selection weight: Σ luminance × area over the quads —
/// the same power measure the alias table is built from, exposed so scene
/// prep can weigh the quads collectively against the environment.
pub(crate) fn total_power(lights: &[QuadLight]) -> f64 {
    lights
        .iter()
        .map(|light| {
            f64::from(luminance(light.emission))
                * f64::from(light.edge1.cross(light.edge2).length())
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad(edge1: Vec3, edge2: Vec3, emission: Vec3, instance: u32) -> QuadLight {
        QuadLight {
            corner: Vec3::ZERO,
            edge1,
            edge2,
            emission,
            instance,
        }
    }

    /// The exact probability the kernel's alias walk assigns light
    /// `index`: each slot contributes `threshold` to its own light and the
    /// remainder to its alias, and slots are chosen uniformly.
    fn selection_probability(records: &[LightRecord], index: u32) -> f64 {
        let mass: f64 = records
            .iter()
            .enumerate()
            .map(|(slot, record)| {
                let threshold = f64::from(record.alias_threshold);
                let own = if slot as u32 == index { threshold } else { 0.0 };
                let aliased = if record.alias_index == index {
                    1.0 - threshold
                } else {
                    0.0
                };
                own + aliased
            })
            .sum();
        mass / records.len() as f64
    }

    /// The alias table is exact, not approximate: summing each light's
    /// slot shares must reproduce its power fraction. Powers here differ
    /// through both emission (luminance 4:1) and area (2:1), and the quads
    /// are deliberately skewed parallelograms.
    #[test]
    fn selection_matches_power_share() {
        let lights = [
            quad(Vec3::X, Vec3::Z, Vec3::splat(4.0), 0),
            quad(Vec3::new(1.0, 0.5, 0.0), Vec3::Z * 2.0, Vec3::splat(1.0), 1),
            quad(
                Vec3::X,
                Vec3::new(0.3, 0.0, 1.0),
                Vec3::new(2.0, 0.5, 3.0),
                2,
            ),
        ];
        let powers: Vec<f64> = lights
            .iter()
            .map(|light| {
                f64::from(luminance(light.emission))
                    * f64::from(light.edge1.cross(light.edge2).length())
            })
            .collect();
        let total: f64 = powers.iter().sum();

        let records = build(&lights);
        for (index, power) in powers.iter().enumerate() {
            let got = selection_probability(&records, index as u32);
            let want = power / total;
            assert!(
                (got - want).abs() < 1e-6,
                "light {index}: alias mass {got} vs power share {want}"
            );
        }
    }

    /// `pdf_area` is the finished quantity the kernel divides by:
    /// selection probability over world-space area.
    #[test]
    fn pdf_area_folds_selection_and_area() {
        let lights = [
            quad(Vec3::X, Vec3::Z, Vec3::splat(1.0), 0),
            quad(Vec3::X * 3.0, Vec3::Z, Vec3::splat(1.0), 7),
        ];
        let records = build(&lights);
        // Equal emission, areas 1 and 3 → selection 1/4 and 3/4, so both
        // quads land at the same pdf: (1/4)/1 == (3/4)/3.
        assert!((records[0].pdf_area - 0.25).abs() < 1e-6);
        assert!((records[1].pdf_area - 0.25).abs() < 1e-6);
        assert_eq!(records[1].instance, 7);
    }

    /// A single light: threshold 1 (its slot never aliases), pdf = 1/area.
    #[test]
    fn single_light_never_aliases() {
        let records = build(&[quad(Vec3::X * 2.0, Vec3::Z * 2.0, Vec3::ONE, 0)]);
        assert_eq!(records.len(), 1);
        assert!((records[0].alias_threshold - 1.0).abs() < 1e-7);
        assert!((records[0].pdf_area - 0.25).abs() < 1e-6);
    }
}
