//! The light list: every emissive triangle in the scene plus the delta
//! lights (distant, point), flattened into one power-proportional alias
//! table next-event estimation selects from. Everything here runs once at
//! scene prep; the GPU sees only the finished [`LightRecord`]s, which
//! `shaders/lights.slang` mirrors field for field.
//!
//! Emissive geometry is per-triangle: any mesh whose material emits
//! becomes one record per triangle, in primitive order, so a hit on a
//! light finds its own record at `GeometryRecord.light + primitive` — the
//! lookup the BSDF strategy's MIS weight needs. Delta lights have no
//! geometry to hit (a BSDF sample reaches them with probability zero), so
//! next-event estimation is their only strategy and their MIS weight is 1.
//!
//! Selection weighs each light by [`power`], a luminance-scaled flux.
//! The measures are frankly approximate across kinds — a triangle counts
//! one face of its two-sided exitance, a point light its whole sphere of
//! flux with no receiver in sight — but selection probabilities only
//! steer noise: the estimator divides by whatever probability was used,
//! so the image converges to the same answer under any positive weights.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use crate::color::luminance;

/// The geometry record's light index for instances that are not lights.
/// Matches `LIGHT_NONE` in `shaders/lights.slang`.
pub(crate) const LIGHT_NONE: u32 = u32::MAX;

/// [`LightRecord::kind`] values. Match the `LIGHT_KIND_*` constants in
/// `shaders/lights.slang`.
const KIND_TRIANGLE: u32 = 0;
const KIND_DISTANT: u32 = 1;
const KIND_POINT: u32 = 2;

/// One emissive triangle as the scene describes it: world-space corners,
/// radiating `emission` from both faces (surfaces are two-sided throughout
/// the renderer).
pub struct TriangleLight {
    /// The corners, world space. A degenerate (zero-area) triangle is
    /// legal — it keeps its record slot so `light + primitive` indexing
    /// holds, but its selection probability is zero.
    pub corners: [Vec3; 3],
    /// Radiance, `ACEScg`, from both faces.
    pub emission: Vec3,
    /// TLAS custom index of the light's mesh instance — shadow rays test
    /// visibility by identity.
    pub instance: u32,
    /// The triangle's index within its mesh, so a BSDF-sampled hit on the
    /// light finds this record at `GeometryRecord.light + primitive`.
    pub primitive: u32,
}

/// A delta light: zero area, so next-event estimation is its only
/// sampling strategy. Colors are `ACEScg` — the description's `Rec.709`
/// values convert at prep, like every other color.
pub enum DeltaLight {
    /// Parallel light from infinitely far away — the sun.
    Distant {
        /// Unit direction the light *travels* (from the light toward the
        /// scene).
        direction: Vec3,
        /// Irradiance delivered on a surface facing the light.
        irradiance: Vec3,
    },
    /// An isotropic point.
    Point {
        /// Position, meters, world space.
        position: Vec3,
        /// Radiant intensity (flux per solid angle).
        intensity: Vec3,
    },
}

/// One light as the kernels read it. Mirrors `LightRecord` in
/// `shaders/lights.slang` field for field; the geometric fields are
/// per-kind (triangle: corner and edges; distant: travel direction in
/// `a`; point: position in `a`), as is `pdf`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct LightRecord {
    /// Triangle: one corner. Distant: unit travel direction. Point:
    /// position.
    a: Vec3,
    /// Triangle: p(the alias table selects this light) / its area — the
    /// area pdf of next-event estimation's uniform point. Delta lights:
    /// the selection probability itself (there is no area).
    pdf: f32,
    /// Triangle only: the sides meeting at `a` — the triangle spans
    /// `a + b1·edge1 + b2·edge2`, `b1 + b2 ≤ 1`.
    edge1: Vec3,
    alias_threshold: f32,
    edge2: Vec3,
    alias_index: u32,
    /// Triangle: radiance. Distant: irradiance. Point: intensity.
    emission: Vec3,
    /// Triangle: TLAS custom index for the shadow ray's identity test.
    /// Delta lights: [`LIGHT_NONE`] (nothing to hit — the connection is
    /// unshadowed exactly when the shadow ray hits nothing).
    instance: u32,
    /// A `KIND_*` value.
    kind: u32,
    /// Triangle only: index within its mesh — the shadow-ray identity is
    /// (instance, primitive), which stays exact on closed meshes where a
    /// ray toward a far-side point hits the near side of the same
    /// instance first.
    primitive: u32,
    _pad0: [u32; 2],
}

/// A light's selection weight: luminance-scaled flux, per kind. A
/// triangle weighs one face's exitance flux (π · luminance · area); a
/// distant light the flux it lands on a conventional ~1 m² facing
/// receiver (matching how the environment's power is measured); a point
/// light its total emitted flux (4π · luminance) — no receiver distance
/// is knowable at prep. See the module doc: these only steer noise.
fn power(record: &LightRecord) -> f64 {
    let luma = f64::from(luminance(record.emission));
    match record.kind {
        KIND_TRIANGLE => {
            let area = f64::from(record.edge1.cross(record.edge2).length()) / 2.0;
            std::f64::consts::PI * luma * area
        }
        KIND_DISTANT => luma,
        _ => 4.0 * std::f64::consts::PI * luma,
    }
}

/// A raw record before the alias pass: geometry and identity filled in,
/// sampling fields at their never-alias defaults.
fn record(
    kind: u32,
    a: Vec3,
    edge1: Vec3,
    edge2: Vec3,
    emission: Vec3,
    ids: (u32, u32),
) -> LightRecord {
    LightRecord {
        a,
        pdf: 0.0,
        edge1,
        alias_threshold: 1.0,
        edge2,
        alias_index: 0,
        emission,
        instance: ids.0,
        kind,
        primitive: ids.1,
        _pad0: [0; 2],
    }
}

/// Build the GPU light list: triangle records first (grouped by instance,
/// in primitive order — `GeometryRecord.light + primitive` indexing
/// depends on it), then the delta lights, each selected in proportion to
/// its [`power`] through a Walker/Vose alias table — one slot per light,
/// carried on the light's own record, selected in O(1).
///
/// Degenerate triangles and an all-dark list are legal: their selection
/// probability (and `pdf`) is zero, so sampling skips whatever it draws
/// there. Scene prep pins the environment-selection probability to 1
/// when the whole list is powerless, so such records are never drawn at
/// all.
pub(crate) fn build(triangles: &[TriangleLight], deltas: &[DeltaLight]) -> Vec<LightRecord> {
    let mut records = raw_records(triangles, deltas);
    let powers: Vec<f64> = records.iter().map(power).collect();
    let total: f64 = powers.iter().sum();
    if total <= 0.0 {
        // Nothing here can be sampled (see the doc above); the defaults —
        // pdf 0, threshold 1 — already say so.
        return records;
    }
    for (record, light_power) in records.iter_mut().zip(&powers) {
        let selection = light_power / total;
        record.pdf = match record.kind {
            KIND_TRIANGLE => {
                let area = f64::from(record.edge1.cross(record.edge2).length()) / 2.0;
                // A degenerate triangle has selection 0 too: keep the pdf
                // 0 rather than 0/0.
                if area > 0.0 {
                    (selection / area) as f32
                } else {
                    0.0
                }
            }
            _ => selection as f32,
        };
    }

    // Walker/Vose: rescale so the average power is 1 slot; every light
    // poorer than a full slot tops its slot up from a richer light (the
    // alias), which keeps exactly its own share across the slots it fills.
    let n = powers.len() as f64;
    let mut share: Vec<f64> = powers
        .iter()
        .map(|light_power| light_power * n / total)
        .collect();
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

/// The list's total selection weight — the same [`power`] measure the
/// alias table is built from, exposed so scene prep can weigh the whole
/// list against the environment.
pub(crate) fn total_power(triangles: &[TriangleLight], deltas: &[DeltaLight]) -> f64 {
    raw_records(triangles, deltas).iter().map(power).sum()
}

/// Every light as a record with geometry, emission, and identity filled
/// in, sampling fields at their never-alias defaults — triangles first,
/// in input order (the indexing contract), then the delta lights.
fn raw_records(triangles: &[TriangleLight], deltas: &[DeltaLight]) -> Vec<LightRecord> {
    let mut records: Vec<LightRecord> = triangles
        .iter()
        .map(|light| {
            record(
                KIND_TRIANGLE,
                light.corners[0],
                light.corners[1] - light.corners[0],
                light.corners[2] - light.corners[0],
                light.emission,
                (light.instance, light.primitive),
            )
        })
        .collect();
    records.extend(deltas.iter().map(|light| match *light {
        DeltaLight::Distant {
            direction,
            irradiance,
        } => record(
            KIND_DISTANT,
            direction,
            Vec3::ZERO,
            Vec3::ZERO,
            irradiance,
            (LIGHT_NONE, 0),
        ),
        DeltaLight::Point {
            position,
            intensity,
        } => record(
            KIND_POINT,
            position,
            Vec3::ZERO,
            Vec3::ZERO,
            intensity,
            (LIGHT_NONE, 0),
        ),
    }));
    records
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle(
        corners: [Vec3; 3],
        emission: Vec3,
        instance: u32,
        primitive: u32,
    ) -> TriangleLight {
        TriangleLight {
            corners,
            emission,
            instance,
            primitive,
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
    /// slot shares must reproduce its power fraction — across a mix of
    /// kinds, with powers differing through emission, area, and the
    /// per-kind flux conventions.
    #[test]
    fn selection_matches_power_share() {
        let triangles = [
            triangle([Vec3::ZERO, Vec3::X, Vec3::Z], Vec3::splat(4.0), 0, 0),
            triangle(
                [Vec3::ZERO, Vec3::new(1.0, 0.5, 0.0), Vec3::Z * 2.0],
                Vec3::splat(1.0),
                0,
                1,
            ),
            triangle(
                [Vec3::Y, Vec3::Y + Vec3::X * 3.0, Vec3::Y + Vec3::Z],
                Vec3::new(2.0, 0.5, 3.0),
                3,
                0,
            ),
        ];
        let deltas = [
            DeltaLight::Distant {
                direction: -Vec3::Y,
                irradiance: Vec3::splat(2.0),
            },
            DeltaLight::Point {
                position: Vec3::Y * 5.0,
                intensity: Vec3::splat(0.25),
            },
        ];
        let records = build(&triangles, &deltas);
        assert_eq!(records.len(), 5);

        let total = total_power(&triangles, &deltas);
        let expected: Vec<f64> = records.iter().map(|record| power(record) / total).collect();
        for (index, want) in expected.iter().enumerate() {
            let got = selection_probability(&records, index as u32);
            assert!(
                (got - want).abs() < 1e-6,
                "light {index}: alias mass {got} vs power share {want}"
            );
        }
    }

    /// `pdf` is the finished quantity the kernel divides by: for a
    /// triangle, selection probability over world-space area; for a delta
    /// light, the selection probability itself.
    #[test]
    fn pdf_folds_selection_per_kind() {
        let triangles = [
            triangle([Vec3::ZERO, Vec3::X * 2.0, Vec3::Z], Vec3::splat(1.0), 0, 0),
            triangle([Vec3::ZERO, Vec3::X * 6.0, Vec3::Z], Vec3::splat(1.0), 7, 0),
        ];
        let records = build(&triangles, &[]);
        // Equal emission, areas 1 and 3 → selection 1/4 and 3/4, so both
        // triangles land at the same area pdf: (1/4)/1 == (3/4)/3.
        assert!((records[0].pdf - 0.25).abs() < 1e-6);
        assert!((records[1].pdf - 0.25).abs() < 1e-6);
        assert_eq!(records[1].instance, 7);

        // One triangle against one point light chosen to match its power:
        // π·luma·area = 4π·luma_point → luma_point = luma·area/4.
        let deltas = [DeltaLight::Point {
            position: Vec3::Y,
            intensity: Vec3::splat(0.25),
        }];
        let records = build(&triangles[..1], &deltas);
        assert!((records[1].pdf - 0.5).abs() < 1e-6, "{}", records[1].pdf);
        assert_eq!(records[1].instance, LIGHT_NONE);
    }

    /// Triangle records precede delta records and sit in (instance,
    /// primitive) input order — the contract `GeometryRecord.light +
    /// primitive` indexing rests on.
    #[test]
    fn triangle_records_keep_input_order() {
        let triangles = [
            triangle([Vec3::ZERO, Vec3::X, Vec3::Z], Vec3::ONE, 2, 0),
            triangle([Vec3::ZERO, Vec3::X, Vec3::Z], Vec3::ONE, 2, 1),
            triangle([Vec3::ZERO, Vec3::X, Vec3::Z], Vec3::ONE, 5, 0),
        ];
        let deltas = [DeltaLight::Distant {
            direction: -Vec3::Y,
            irradiance: Vec3::ONE,
        }];
        let records = build(&triangles, &deltas);
        let ids: Vec<(u32, u32, u32)> = records
            .iter()
            .map(|record| (record.kind, record.instance, record.primitive))
            .collect();
        assert_eq!(
            ids,
            [
                (KIND_TRIANGLE, 2, 0),
                (KIND_TRIANGLE, 2, 1),
                (KIND_TRIANGLE, 5, 0),
                (KIND_DISTANT, LIGHT_NONE, 0),
            ]
        );
    }

    /// A single light: threshold 1 (its slot never aliases), pdf =
    /// 1/area over the triangle's real area.
    #[test]
    fn single_light_never_aliases() {
        let records = build(
            &[triangle(
                [Vec3::ZERO, Vec3::X * 2.0, Vec3::Z * 2.0],
                Vec3::ONE,
                0,
                0,
            )],
            &[],
        );
        assert_eq!(records.len(), 1);
        assert!((records[0].alias_threshold - 1.0).abs() < 1e-7);
        assert!((records[0].pdf - 0.5).abs() < 1e-6);
    }

    /// Degenerate triangles keep their record slot (hit-side indexing
    /// needs one record per triangle) but can never be selected: zero
    /// power, zero pdf, and their slot mass aliases to a real light.
    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "an unsampleable record's pdf is exactly zero"
    )]
    fn degenerate_triangles_hold_their_slot_unsampled() {
        let triangles = [
            triangle([Vec3::ZERO, Vec3::X, Vec3::X * 2.0], Vec3::ONE, 0, 0), // collinear
            triangle([Vec3::ZERO, Vec3::X, Vec3::Z], Vec3::ONE, 0, 1),
        ];
        let records = build(&triangles, &[]);
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].pdf, 0.0);
        assert!(selection_probability(&records, 0).abs() < 1e-12);
        assert!((selection_probability(&records, 1) - 1.0).abs() < 1e-9);
    }

    /// An all-dark list builds without dividing by zero; every record is
    /// unsampleable (prep pins selection to the environment in this case,
    /// so the table is never drawn from at all).
    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "an unsampleable record's pdf is exactly zero"
    )]
    fn a_powerless_list_builds_finite() {
        let triangles = [triangle(
            [Vec3::ZERO, Vec3::X, Vec3::X * 2.0], // zero area, so zero power
            Vec3::ONE,
            0,
            0,
        )];
        let records = build(&triangles, &[]);
        assert_eq!(records[0].pdf, 0.0);
        assert!((records[0].alias_threshold - 1.0).abs() < 1e-7);
        assert!(total_power(&triangles, &[]).abs() < 1e-12);
    }
}
