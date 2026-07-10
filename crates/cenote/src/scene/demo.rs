//! The demo scene as data: [`ChangeSet::demo`] describes exactly the scene
//! [`super::Scene::demo`] builds procedurally — the terracotta material
//! chart under the warm key quad and the Kloofendal sky. Two spellings of
//! one scene is a step-2 scaffold: the prep rewire (M2 step 3) makes the
//! GPU build consume this change-set, and the goldens hold it to sameness.

use std::path::Path;

use super::changeset::{
    CameraPatch, ChangeSet, EnvironmentPatch, InstancePatch, MaterialPatch, MeshPatch, Op,
    SettingsPatch,
};
use super::description::{MeshSource, Texturable, Transform};
use super::{Mesh, Scene, ground_plane, icosphere};

/// A host mesh flattened into an inline geometry payload.
fn inline(mesh: &Mesh) -> MeshSource {
    MeshSource::Inline {
        positions: mesh.positions.iter().map(glam::Vec3::to_array).collect(),
        normals: Some(mesh.normals.iter().map(glam::Vec3::to_array).collect()),
        uvs: None,
        triangles: mesh.triangles.clone(),
    }
}

impl ChangeSet {
    /// The demo scene — [`Scene::demo`]'s material chart, expressed as the
    /// change-set that creates it from nothing. Colors are the authored
    /// `Rec.709` values (the format's convention; prep converts), where the
    /// procedural builder converts in code. Unlike the builder's 27 meshes,
    /// the data form shares one icosphere across the chart and one unit
    /// plane between floor and key light, scaled per instance — the
    /// mesh/instance split doing its job.
    #[must_use]
    pub fn demo() -> Self {
        let mut ops = vec![
            Op::Settings(SettingsPatch::new("main")),
            Op::Camera(CameraPatch {
                position: Some([0.0, 5.5, 11.0]),
                look_at: Some([0.0; 3]),
                vfov_degrees: Some(40.0),
                ..CameraPatch::new("main")
            }),
            Op::Environment(EnvironmentPatch {
                path: Some(Path::new(env!("CARGO_MANIFEST_DIR")).join(Scene::DEMO_ENVIRONMENT)),
                ..EnvironmentPatch::new("sky")
            }),
            Op::Mesh(MeshPatch {
                source: Some(inline(&icosphere(4))),
                ..MeshPatch::new("sphere")
            }),
            Op::Mesh(MeshPatch {
                source: Some(inline(&ground_plane(1.0))),
                ..MeshPatch::new("plane")
            }),
            Op::Material(Box::new(MaterialPatch {
                base_color: Some(Texturable::Constant([0.65; 3])),
                base_diffuse_roughness: Some(0.1),
                specular_roughness: Some(Texturable::Constant(0.15)),
                ..MaterialPatch::new("floor")
            })),
            Op::Instance(InstancePatch {
                mesh: Some("plane".into()),
                material: Some("floor".into()),
                transform: Some(Transform::Trs {
                    translate: [0.0; 3],
                    rotate_degrees: [0.0; 3],
                    scale: [12.0; 3],
                }),
                ..InstancePatch::new("floor")
            }),
            // The key light: a pure emitter (black base, no specular
            // layer), up and off to the left, opposite the HDRI's sun.
            Op::Material(Box::new(MaterialPatch {
                base_color: Some(Texturable::Constant([0.0; 3])),
                specular_weight: Some(0.0),
                emission_color: Some(Texturable::Constant([1.0, 0.85, 0.6])),
                emission_luminance: Some(18.0),
                ..MaterialPatch::new("key")
            })),
            Op::Instance(InstancePatch {
                mesh: Some("plane".into()),
                material: Some("key".into()),
                transform: Some(Transform::Trs {
                    translate: [-3.5, 5.4, 1.0],
                    rotate_degrees: [0.0; 3],
                    scale: [0.75; 3],
                }),
                ..InstancePatch::new("key")
            }),
        ];
        // The chart: specular_roughness sweeps 0 → 1 left to right,
        // base_metalness 0 → 1 back to front, one material and one
        // instance per sphere.
        let sweep = |step: usize, steps: usize| step as f32 / (steps - 1) as f32;
        for row in 0..Scene::GRID_ROWS {
            for column in 0..Scene::GRID_COLUMNS {
                let name = format!("chart_r{row}c{column}");
                ops.push(Op::Material(Box::new(MaterialPatch {
                    base_color: Some(Texturable::Constant([0.7, 0.22, 0.08])),
                    base_diffuse_roughness: Some(0.4),
                    specular_roughness: Some(Texturable::Constant(sweep(
                        column,
                        Scene::GRID_COLUMNS,
                    ))),
                    base_metalness: Some(Texturable::Constant(sweep(row, Scene::GRID_ROWS))),
                    ..MaterialPatch::new(name.clone())
                })));
                ops.push(Op::Instance(InstancePatch {
                    mesh: Some("sphere".into()),
                    material: Some(name.clone()),
                    transform: Some(Transform::Trs {
                        translate: [1.2 * (column as f32 - 2.0), 0.5, 1.2 * (row as f32 - 2.0)],
                        rotate_degrees: [0.0; 3],
                        scale: [0.5; 3],
                    }),
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

    #[test]
    fn demo_applies_to_an_empty_description() {
        let mut description = SceneDescription::new();
        description
            .apply(&ChangeSet::demo())
            .expect("demo is valid");
        assert_eq!(description.meshes().len(), 2);
        assert_eq!(description.instances().len(), 27);
        assert_eq!(description.materials().len(), 27);
        assert_eq!(description.cameras().len(), 1);
        assert_eq!(description.environments().len(), 1);
        assert_eq!(description.settings().len(), 1);
        assert!(description.lights().is_empty());
        // Every object a fresh apply creates is dirty.
        assert_eq!(description.take_dirty().changed.len(), 59);
    }

    #[test]
    fn the_chart_sweeps_its_corners() {
        let mut description = SceneDescription::new();
        description
            .apply(&ChangeSet::demo())
            .expect("demo is valid");
        let corner = |name: &str| &description.materials()[name];
        assert_eq!(
            corner("chart_r0c0").specular_roughness,
            Texturable::Constant(0.0)
        );
        assert_eq!(
            corner("chart_r0c4").specular_roughness,
            Texturable::Constant(1.0)
        );
        assert_eq!(
            corner("chart_r0c0").base_metalness,
            Texturable::Constant(0.0)
        );
        assert_eq!(
            corner("chart_r4c0").base_metalness,
            Texturable::Constant(1.0)
        );
    }

    #[test]
    fn demo_round_trips_through_ron() {
        let set = ChangeSet::demo();
        let text = crate::format::to_ron(&set).expect("serializes");
        let parsed = crate::format::from_ron(&text).expect("parses back");
        assert_eq!(parsed, set);
    }
}
