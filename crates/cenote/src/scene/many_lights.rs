//! The many-light validation scene — hundreds of small emitters raining
//! warm-to-cool light onto a cluster of matte occluders under a black sky.
//! [`ChangeSet::many_lights`] is the scene as data and [`Scene::many_lights`]
//! is that change-set prepped, the same production path the demo takes.
//!
//! This is the scene the validation harness renders for the hardest case
//! next-event estimation faces: the estimator draws one of the
//! [`LIGHT_GRID`]² emitters per sample and shadows it, and with this many
//! lights and this much occlusion that single draw is mostly wasted, so the
//! image crawls toward convergence. It is where the light list's
//! power-proportional selection has to earn its keep — the emitters carry
//! distinct power (colour sweeps the panel, luminance ripples across it) so
//! the weighting has something real to sort, and the occluders overlap their
//! penumbrae so the shadow field is genuinely hard.

use super::changeset::{
    CameraPatch, ChangeSet, InstancePatch, MaterialPatch, MeshPatch, Op, SettingsPatch,
};
use super::demo::inline;
use super::description::{SceneDescription, Texturable, Transform};
use super::{Scene, ground_plane, icosphere};
use crate::error::Result;
use crate::gpu::Context;

/// Emitters per side of the overhead panel; [`LIGHT_GRID`]² in all — the
/// many-light count the validation scene is built to stress.
const LIGHT_GRID: usize = 16;
/// Matte occluder spheres per side of the floor cluster; [`OCCLUDER_GRID`]²
/// in all, their overlapping soft shadows the scene's occlusion test.
const OCCLUDER_GRID: usize = 3;

impl Scene {
    /// The many-light validation scene: an overhead panel of
    /// [`LIGHT_GRID`]² small warm-to-cool emitters over a 3×3 cluster of
    /// matte spheres on a Lambert floor, under a black sky (no
    /// environment) so the emitters are the scene's only light — the
    /// convergence harness's many-light subject.
    ///
    /// # Errors
    ///
    /// Any [`crate::Error`] from prep — upload, decode, or
    /// acceleration-structure builds.
    #[expect(
        clippy::missing_panics_doc,
        reason = "the expect guards the built-in change-set applying to an empty \
                  description, which a unit test pins — not a reachable panic"
    )]
    pub fn many_lights(gpu: &Context) -> Result<Self> {
        let mut description = SceneDescription::new();
        description
            .apply(&ChangeSet::many_lights())
            .expect("the many-light change-set is valid");
        Self::prep(gpu, &mut description)
    }
}

impl ChangeSet {
    /// The many-light validation scene as the change-set that creates it
    /// from nothing — what [`Scene::many_lights`] preps. One unit plane is
    /// shared between floor and every emitter and one icosphere across the
    /// occluders, scaled per instance; colors are authored linear `Rec.709`
    /// (prep converts). No environment op: the sky is black, so the
    /// emitters carry the whole image.
    #[must_use]
    pub fn many_lights() -> Self {
        let mut ops = vec![
            Op::Settings(SettingsPatch::new("main")),
            Op::Camera(CameraPatch {
                position: Some([0.0, 5.0, 13.0]),
                look_at: Some([0.0, 0.6, 0.0]),
                vfov_degrees: Some(40.0),
                ..CameraPatch::new("main")
            }),
            Op::Mesh(MeshPatch {
                source: Some(inline(&ground_plane(1.0))),
                ..MeshPatch::new("plane")
            }),
            Op::Mesh(MeshPatch {
                source: Some(inline(&icosphere(3))),
                ..MeshPatch::new("sphere")
            }),
            // A matte Lambert floor (no specular layer), so the many soft
            // shadows read cleanly and the brute-force reference converges
            // without a glossy floor mirroring 256 emitters into fireflies.
            Op::Material(Box::new(MaterialPatch {
                base_color: Some(Texturable::Constant([0.55; 3])),
                specular_weight: Some(0.0),
                ..MaterialPatch::new("floor")
            })),
            Op::Instance(InstancePatch {
                mesh: Some("plane".into()),
                material: Some("floor".into()),
                transforms: Some(vec![Transform::Trs {
                    translate: [0.0; 3],
                    rotate_degrees: [0.0; 3],
                    scale: [12.0; 3],
                }]),
                ..InstancePatch::new("floor")
            }),
        ];
        // The occluders: a 3×3 grid of matte spheres resting on the floor,
        // radius alternating so the shadow field isn't perfectly regular.
        // The icosphere is unit-radius, so scale is the radius and the
        // sphere rests when lifted by it.
        for row in 0..OCCLUDER_GRID {
            for column in 0..OCCLUDER_GRID {
                let name = format!("occluder_r{row}c{column}");
                let radius = if (row + column) % 2 == 0 { 0.9 } else { 0.6 };
                ops.push(Op::Material(Box::new(MaterialPatch {
                    base_color: Some(Texturable::Constant([0.6; 3])),
                    specular_weight: Some(0.0),
                    ..MaterialPatch::new(name.clone())
                })));
                ops.push(Op::Instance(InstancePatch {
                    mesh: Some("sphere".into()),
                    material: Some(name.clone()),
                    transforms: Some(vec![Transform::Trs {
                        translate: [3.0 * (column as f32 - 1.0), radius, 3.0 * (row as f32 - 1.0)],
                        rotate_degrees: [0.0; 3],
                        scale: [radius; 3],
                    }]),
                    ..InstancePatch::new(name)
                }));
            }
        }
        // The panel: LIGHT_GRID² small quads on a ceiling 6 m up, each
        // rolled 180° so its one emitting face looks down (emitters are
        // one-sided, and the plane winds normal-up). Color sweeps warm to
        // cool along the panel diagonal and luminance ripples off it, so
        // no two emitters carry quite the same power.
        let span = 9.0;
        let height = 6.0;
        for row in 0..LIGHT_GRID {
            for column in 0..LIGHT_GRID {
                let name = format!("light_r{row}c{column}");
                let u = column as f32 / (LIGHT_GRID - 1) as f32;
                let v = row as f32 / (LIGHT_GRID - 1) as f32;
                let t = 0.5 * (u + v);
                let color = [1.0 - 0.6 * t, 0.6 + 0.05 * t, 0.3 + 0.7 * t];
                ops.push(Op::Material(Box::new(MaterialPatch {
                    base_color: Some(Texturable::Constant([0.0; 3])),
                    specular_weight: Some(0.0),
                    emission_color: Some(Texturable::Constant(color)),
                    emission_luminance: Some(6.0 + 4.0 * (u - v).abs()),
                    ..MaterialPatch::new(name.clone())
                })));
                ops.push(Op::Instance(InstancePatch {
                    mesh: Some("plane".into()),
                    material: Some(name.clone()),
                    transforms: Some(vec![Transform::Trs {
                        translate: [span * (u - 0.5), height, span * (v - 0.5)],
                        rotate_degrees: [180.0, 0.0, 0.0],
                        scale: [0.18; 3],
                    }]),
                    ..InstancePatch::new(name)
                }));
            }
        }
        Self { ops }
    }
}

#[cfg(test)]
mod tests {
    use super::super::description::SceneDescription;
    use super::*;

    /// The panel and the cluster come out at the sizes the harness counts
    /// on: 256 emitters, 9 occluders, one floor — all instances, no delta
    /// lights (the emitters are emissive instances) — and no environment,
    /// the black sky that leaves the emitters as the only light.
    #[test]
    fn many_lights_applies_to_an_empty_description() {
        let mut description = SceneDescription::new();
        description
            .apply(&ChangeSet::many_lights())
            .expect("many_lights is valid");
        assert_eq!(description.meshes().len(), 2);
        // 256 emitters + 9 occluders + 1 floor.
        assert_eq!(description.instances().len(), 266);
        assert_eq!(description.materials().len(), 266);
        assert_eq!(description.cameras().len(), 1);
        assert_eq!(description.settings().len(), 1);
        // The emitters light the scene, so there are no delta lights and no
        // environment.
        assert!(description.lights().is_empty());
        assert!(description.environments().is_empty());
        // Every object a fresh apply creates is dirty: 266 instances + 266
        // materials + 2 meshes + 1 camera + 1 settings.
        assert_eq!(description.take_dirty().changed.len(), 536);
    }

    /// The panel sweeps warm to cool: its near corner emits warm, its far
    /// corner cool, and every emitter carries real power — while the
    /// occluders emit nothing, so only the panel lights the scene.
    #[test]
    fn the_panel_sweeps_warm_to_cool() {
        let mut description = SceneDescription::new();
        description
            .apply(&ChangeSet::many_lights())
            .expect("many_lights is valid");
        let emission = |name: &str| {
            let material = &description.materials()[name];
            let Texturable::Constant(color) = material.emission_color else {
                panic!("emitter {name} should have a constant emission color");
            };
            (color, material.emission_luminance)
        };
        let (warm, warm_lum) = emission("light_r0c0");
        assert!(warm[0] > warm[2], "the near corner should be warm: {warm:?}");
        assert!(warm_lum > 0.0, "an emitter must emit: {warm_lum}");
        let (cool, _) = emission("light_r15c15");
        assert!(cool[2] > cool[0], "the far corner should be cool: {cool:?}");
        // The occluders are matte, not emissive: the panel is the only light.
        assert!(
            description.materials()["occluder_r0c0"].emission_luminance.abs() < 1e-6,
            "occluders must not emit"
        );
    }

    #[test]
    fn many_lights_round_trips_through_ron() {
        let set = ChangeSet::many_lights();
        let text = crate::format::to_ron(&set).expect("serializes");
        let parsed = crate::format::from_ron(&text).expect("parses back");
        assert_eq!(parsed, set);
    }

    /// The committed `scenes/many-lights.ron` — the flagship scene file the
    /// validation figures render, and the one a stranger opens to see the
    /// many-light case — stays in lockstep with the builder it's
    /// generated from. A change to [`ChangeSet::many_lights`] that isn't
    /// regenerated fails here (CPU-only, so it runs everywhere) rather than
    /// silently shipping a stale scene the figures no longer describe.
    /// Regenerate and eyeball with:
    /// `UPDATE_SCENES=1 cargo test -p cenote committed_many_lights_ron`.
    #[test]
    fn committed_many_lights_ron_matches_the_builder() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../scenes/many-lights.ron");
        let expected = crate::format::to_ron(&ChangeSet::many_lights()).expect("serializes");

        if std::env::var_os("UPDATE_SCENES").is_some() {
            std::fs::write(&path, &expected).expect("write scene");
            eprintln!("wrote {} — eyeball it before committing", path.display());
            return;
        }

        let committed = std::fs::read_to_string(&path).unwrap_or_else(|err| {
            panic!(
                "can't read {}: {err}\n\
                 generate it with: UPDATE_SCENES=1 cargo test -p cenote committed_many_lights_ron",
                path.display()
            )
        });
        assert_eq!(
            committed, expected,
            "scenes/many-lights.ron drifted from ChangeSet::many_lights() — regenerate with \
             UPDATE_SCENES=1 cargo test -p cenote committed_many_lights_ron"
        );
    }
}
