//! Prep: the one path from a [`SceneDescription`] to GPU residency.
//! [`Scene::prep`] builds a description fresh; [`Scene::update`] follows
//! its accumulated [`Dirty`] state. Meshes, textures, the environment, and
//! the TLAS each rebuild only when an edit touched them — a changed mesh
//! re-uploads its own BLAS, an environment swap reloads the image and its
//! tables, an untouched one is left resident. The instance tables
//! (geometry, materials, lights) and the scene table rebuild wholesale on
//! any edit instead: cheap over a scene's handful of instances, and the
//! first thing to make granular when that stops holding. Iteration order
//! everywhere is name order, so an incremental update lands the exact scene
//! a fresh build would — the determinism invariant extends through editing.
//!
//! The fallible host-side lowering both paths run first — and the
//! [`Error::Scene`](crate::Error) contract that keeps a live session's last
//! good scene when an edit can't render — lives in [`super::lower`]:
//! everything that can fail on user data happens there, before the first
//! GPU call, so the GPU phase here can only fault on the device, which ends
//! the render anyway.

use std::collections::BTreeMap;

use super::changeset::Dirty;
use super::description::SceneDescription;
use super::lower::{InstanceSpec, all_dirty, host_phase};
use super::{
    GpuEnvironment, GpuMesh, Placement, ResidentBuffers, ResidentTexture, Scene, build_scene_tlas,
    select_probability, upload_environment, upload_instance_tables, upload_mesh,
    upload_scene_table,
};
use crate::error::Result;
use crate::gpu::Context;
use crate::texture;

impl Scene {
    /// Build `description` into a fresh, traceable scene, consuming its
    /// accumulated dirty state (a full build covers everything).
    ///
    /// # Errors
    ///
    /// [`Error::Scene`](crate::Error) when this build can't render the
    /// description — not exactly one camera and settings, more than one
    /// environment, no instances, or a referenced file (PLY, texture,
    /// environment) that doesn't read or decode. Any other error is a GPU
    /// fault from upload or acceleration-structure builds.
    #[expect(
        clippy::missing_panics_doc,
        reason = "the expects state all-dirty invariants — a fresh build always carries \
                  its environment and camera — not reachable panics"
    )]
    pub fn prep(gpu: &Context, description: &mut SceneDescription) -> Result<Self> {
        let host = host_phase(description, &all_dirty(description), true, &BTreeMap::new())?;
        let mut meshes = BTreeMap::new();
        for (name, mesh) in &host.meshes {
            meshes.insert(
                name.clone(),
                upload_mesh(gpu, &format!("scene.mesh.{name}"), mesh)?,
            );
        }
        let textures = upload_textures(gpu, BTreeMap::new(), &host.textures)?;
        let descriptors = textures
            .values()
            .map(|texture| texture.image.descriptor())
            .collect();
        let environment = host
            .environment
            .as_ref()
            .expect("a fresh build always carries its environment");
        let GpuEnvironment {
            image,
            marginal,
            conditional,
            pdfs,
            power,
        } = upload_environment(gpu, environment)?;
        let placements = placements(&meshes, &host.instances);
        let tlas = build_scene_tlas(gpu, &placements)?;
        let (geometry, materials, lights) =
            upload_instance_tables(gpu, &placements, &host.triangle_lights, &host.delta_lights)?;
        drop(placements);
        let resident = ResidentBuffers::assemble(
            gpu,
            geometry,
            materials,
            lights,
            marginal,
            conditional,
            pdfs,
        )?;
        let env_size = (environment.width(), environment.height());
        let table = upload_scene_table(
            gpu,
            &resident,
            env_size,
            select_probability(power, host.light_power()),
            host.light_count(),
        )?;
        description.take_dirty();
        Ok(Self {
            tlas,
            environment: image,
            table,
            resident,
            meshes,
            textures,
            descriptors,
            camera: host.camera.expect("a fresh build always adopts its camera"),
            env_size,
            env_power: power,
        })
    }

    /// Rebuild exactly what `dirty` names, leaving the rest of the
    /// residency in place — the wave-boundary half of the edit channel.
    ///
    /// # Errors
    ///
    /// [`Error::Scene`](crate::Error) means this build can't render the
    /// edited description (see [`Scene::prep`]); the scene is **untouched**,
    /// so the caller keeps rendering the previous residency and may retry
    /// after the next edit. Any other error is a device fault: the scene
    /// may be partially rebuilt, and callers treat it as fatal.
    pub(crate) fn update(
        &mut self,
        gpu: &Context,
        description: &SceneDescription,
        dirty: &Dirty,
    ) -> Result<()> {
        let resident_hashes = self
            .textures
            .iter()
            .map(|(key, texture)| (key.clone(), texture.hash))
            .collect();
        let host = host_phase(description, dirty, false, &resident_hashes)?;
        // Only device faults from here on — the untouched-on-Scene-error
        // contract holds because everything fallible already ran.
        for name in &host.removed_meshes {
            self.meshes.remove(name);
        }
        for (name, mesh) in &host.meshes {
            self.meshes.insert(
                name.clone(),
                upload_mesh(gpu, &format!("scene.mesh.{name}"), mesh)?,
            );
        }
        self.textures = upload_textures(gpu, std::mem::take(&mut self.textures), &host.textures)?;
        self.rebuild_texture_descriptors();
        if let Some(environment) = &host.environment {
            let env = upload_environment(gpu, environment)?;
            self.environment = env.image;
            self.resident.env_marginal = env.marginal;
            self.resident.env_conditional = env.conditional;
            self.resident.env_pdfs = env.pdfs;
            self.env_size = (environment.width(), environment.height());
            self.env_power = env.power;
        }
        let placements = placements(&self.meshes, &host.instances);
        if host.tlas_dirty {
            self.tlas = build_scene_tlas(gpu, &placements)?;
        }
        let (geometry, materials, lights) =
            upload_instance_tables(gpu, &placements, &host.triangle_lights, &host.delta_lights)?;
        drop(placements);
        self.resident.geometry = geometry;
        self.resident.materials = materials;
        self.resident.lights = lights;
        self.table = upload_scene_table(
            gpu,
            &self.resident,
            self.env_size,
            select_probability(self.env_power, host.light_power()),
            host.light_count(),
        )?;
        if let Some(camera) = host.camera {
            self.camera = camera;
        }
        Ok(())
    }
}

/// The GPU half of texture residency: keep the resident images the host
/// phase kept, upload the ones it prepped (new or content-changed), and
/// drop whatever nothing references anymore. Returns the new resident
/// map — iteration order is the bindless index order the lowered
/// materials already encode. A `None` entry is a texture the host phase
/// deliberately kept resident (its content hash matched), so its removal
/// from `resident` cannot miss.
fn upload_textures(
    gpu: &Context,
    mut resident: BTreeMap<texture::Key, ResidentTexture>,
    prepared: &BTreeMap<texture::Key, Option<texture::Prepared>>,
) -> Result<BTreeMap<texture::Key, ResidentTexture>> {
    let mut textures = BTreeMap::new();
    for (key, entry) in prepared {
        let texture = match entry {
            Some(prepared) => ResidentTexture {
                image: gpu.upload_texture(
                    &format!("scene.texture.{}", key.0.display()),
                    prepared.width,
                    prepared.height,
                    prepared.format,
                    &prepared.data,
                )?,
                hash: prepared.hash,
            },
            None => resident
                .remove(key)
                .expect("the host phase marks a texture None only when it is already resident"),
        };
        textures.insert(key.clone(), texture);
    }
    Ok(textures)
}

/// Resolve instance specs against the resident mesh map. The lookup can't
/// miss: apply validated every reference, and prep processes every dirty
/// mesh, so residency tracks the description name for name.
fn placements<'a>(
    meshes: &'a BTreeMap<String, GpuMesh>,
    instances: &[InstanceSpec],
) -> Vec<Placement<'a>> {
    instances
        .iter()
        .map(|spec| Placement {
            mesh: meshes
                .get(&spec.mesh)
                .expect("mesh residency tracks the description"),
            transform: spec.transform,
            material: spec.material,
            camera_visible: spec.camera_visible,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::changeset::{
        CameraPatch, ChangeSet, EnvironmentPatch, InstancePatch, Kind, MaterialPatch, MeshPatch,
        Op, SettingsPatch,
    };
    use super::super::description::{MeshSource, Texturable, TextureRef};
    use super::*;
    use crate::error::Error;
    use crate::render::Renderer;

    /// The demo, applied — the standing prep test subject.
    fn demo_description() -> SceneDescription {
        let mut description = SceneDescription::new();
        description
            .apply(&ChangeSet::demo())
            .expect("the demo change-set is valid");
        description
    }

    fn render(gpu: &Context, scene: &Scene) -> Vec<f32> {
        Renderer::new(gpu)
            .expect("renderer")
            .render(gpu, scene, 64, 64)
            .expect("render")
    }

    /// Rebuild a description from the change-set history — what a fresh
    /// process loading the current file state would hold.
    fn replay(sets: &[ChangeSet]) -> SceneDescription {
        let mut description = SceneDescription::new();
        for set in sets {
            description.apply(set).expect("replayed sets are valid");
        }
        description
    }

    /// Every re-prep path, one edit each — the walk
    /// [`incremental_updates_match_a_fresh_build`] takes: material-only
    /// (buffer upload), emission (light tables), transform (TLAS),
    /// topology (BLAS), removal (retired residency), environment swap, a
    /// camera move, a delta light, a camera-visibility flip (TLAS masks),
    /// a closure edit with fractional opacity (materials plus the TLAS
    /// opacity flags), a texture reference (the bindless table gains a
    /// slot mid-session), and its removal (the slot retires).
    #[expect(
        clippy::too_many_lines,
        reason = "a flat list of labeled edits, one per re-prep path — splitting it \
                  would hide the walk's shape"
    )]
    fn edit_walk(sky: &std::path::Path, wood: &std::path::Path) -> Vec<(&'static str, ChangeSet)> {
        vec![
            (
                "material",
                ChangeSet {
                    ops: vec![Op::Material(Box::new(MaterialPatch {
                        base_color: Some(Texturable::Constant([0.1, 0.6, 0.2])),
                        ..MaterialPatch::new("floor")
                    }))],
                },
            ),
            (
                "emission",
                ChangeSet {
                    ops: vec![Op::Material(Box::new(MaterialPatch {
                        emission_luminance: Some(30.0),
                        ..MaterialPatch::new("key")
                    }))],
                },
            ),
            (
                "transform",
                ChangeSet {
                    ops: vec![Op::Instance(InstancePatch {
                        transform: Some(super::super::description::Transform::Trs {
                            translate: [0.0, 1.5, 0.0],
                            rotate_degrees: [0.0; 3],
                            scale: [0.75; 3],
                        }),
                        ..InstancePatch::new("chart_r2c2")
                    })],
                },
            ),
            (
                "topology",
                ChangeSet {
                    ops: vec![Op::Mesh(MeshPatch {
                        source: Some(super::super::demo::inline(&super::super::icosphere(2))),
                        ..MeshPatch::new("sphere")
                    })],
                },
            ),
            (
                "removal",
                ChangeSet {
                    ops: vec![
                        Op::Remove(Kind::Instance, "chart_r4c4".into()),
                        Op::Remove(Kind::Material, "chart_r4c4".into()),
                    ],
                },
            ),
            (
                "environment",
                ChangeSet {
                    ops: vec![Op::Environment(EnvironmentPatch {
                        path: Some(sky.to_owned()),
                        ..EnvironmentPatch::new("sky")
                    })],
                },
            ),
            (
                "camera",
                ChangeSet {
                    ops: vec![Op::Camera(CameraPatch {
                        position: Some([2.0, 4.0, 9.0]),
                        ..CameraPatch::new("main")
                    })],
                },
            ),
            (
                "delta light",
                ChangeSet {
                    ops: vec![Op::Light(super::super::changeset::LightPatch {
                        light: Some(super::super::description::Light::Distant {
                            direction: [0.2, -1.0, 0.1],
                            irradiance: [1.5, 1.4, 1.2],
                        }),
                        ..super::super::changeset::LightPatch::new("sun")
                    })],
                },
            ),
            (
                "camera visibility",
                ChangeSet {
                    ops: vec![Op::Instance(InstancePatch {
                        camera_visible: Some(false),
                        ..InstancePatch::new("key")
                    })],
                },
            ),
            (
                "closure and opacity",
                ChangeSet {
                    ops: vec![Op::Material(Box::new(MaterialPatch {
                        coat_weight: Some(1.0),
                        coat_roughness: Some(0.2),
                        // Fractional opacity flips the instance's TLAS
                        // opacity flag — a material edit that must
                        // rebuild the TLAS on both prep paths.
                        geometry_opacity: Some(Texturable::Constant(0.5)),
                        ..MaterialPatch::new("floor")
                    }))],
                },
            ),
            (
                "texture",
                ChangeSet {
                    ops: vec![Op::Material(Box::new(MaterialPatch {
                        base_color: Some(Texturable::Texture(TextureRef {
                            path: wood.to_owned(),
                            color_space: None,
                        })),
                        ..MaterialPatch::new("floor")
                    }))],
                },
            ),
            (
                "texture removal",
                ChangeSet {
                    ops: vec![Op::Material(Box::new(MaterialPatch {
                        base_color: Some(Texturable::Constant([0.4, 0.35, 0.3])),
                        ..MaterialPatch::new("floor")
                    }))],
                },
            ),
        ]
    }

    /// The prep rewrite's core property: after any edit, the
    /// incrementally updated scene renders bit-identically to a fresh
    /// prep of the same description, across the whole [`edit_walk`].
    #[test]
    fn incremental_updates_match_a_fresh_build() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        // A fixture directory of their own: the walk's texture edit
        // writes a DDS cache next to its source, so cleanup is the
        // directory, not a file list.
        let dir = std::env::temp_dir().join(format!("cenote-prep-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("fixture dir");
        let sky = dir.join("sky.exr");
        crate::output::write_exr(&sky, 2, 2, &[0.3_f32; 16]).expect("test sky");
        let wood = dir.join("wood.png");
        // A visible two-tone map: the renders diverge if the incremental
        // path misindexes or fails to upload it.
        let texels: Vec<u8> = (0..64)
            .flat_map(|index| {
                if index % 2 == 0 {
                    [200u8, 120, 60, 255]
                } else {
                    [40u8, 90, 130, 255]
                }
            })
            .collect();
        crate::texture::write_png(&wood, 8, 8, &texels);

        let mut history = vec![ChangeSet::demo()];
        let mut description = replay(&history);
        let mut scene = Scene::prep(&gpu, &mut description).expect("prep");

        for (label, set) in edit_walk(&sky, &wood) {
            description.apply(&set).expect(label);
            let dirty = description.take_dirty();
            scene
                .update(&gpu, &description, &dirty)
                .unwrap_or_else(|error| panic!("{label}: {error}"));
            history.push(set);
            let fresh = Scene::prep(&gpu, &mut replay(&history)).expect("fresh prep");
            assert_eq!(
                render(&gpu, &scene),
                render(&gpu, &fresh),
                "{label}: the incremental update diverged from a fresh build"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The classic invisible-emitter trick, wired through the TLAS camera
    /// mask: a lamp with `camera_visible: false` must vanish from the
    /// frame — camera rays traverse straight past it — while still
    /// lighting the floor through next-event connections and bounces.
    #[test]
    fn an_invisible_emitter_lights_without_appearing() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        // A floor, and a lamp panel dead ahead of the camera with open
        // (black) sky behind it: the lamp's pixels fall to ~0 when it
        // goes camera-invisible, and the floor stays lamp-lit.
        let scene_with = |visible: bool| {
            let mut description = SceneDescription::new();
            description
                .apply(&ChangeSet {
                    ops: vec![
                        Op::Settings(SettingsPatch::new("main")),
                        Op::Camera(CameraPatch {
                            position: Some([0.0, 1.0, 4.0]),
                            look_at: Some([0.0, 1.0, 0.0]),
                            ..CameraPatch::new("main")
                        }),
                        Op::Mesh(MeshPatch {
                            source: Some(MeshSource::Inline {
                                positions: vec![
                                    [-1.0, 0.0, -1.0],
                                    [-1.0, 0.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 0.0, -1.0],
                                ],
                                normals: Some(vec![[0.0, 1.0, 0.0]; 4]),
                                uvs: None,
                                triangles: vec![[0, 1, 2], [0, 2, 3]],
                            }),
                            ..MeshPatch::new("plane")
                        }),
                        Op::Material(Box::new(MaterialPatch::new("gray"))),
                        Op::Instance(InstancePatch {
                            mesh: Some("plane".into()),
                            material: Some("gray".into()),
                            transform: Some(super::super::description::Transform::Trs {
                                translate: [0.0; 3],
                                rotate_degrees: [0.0; 3],
                                scale: [8.0, 1.0, 8.0],
                            }),
                            ..InstancePatch::new("floor")
                        }),
                        Op::Material(Box::new(MaterialPatch {
                            base_color: Some(Texturable::Constant([0.0; 3])),
                            specular_weight: Some(0.0),
                            emission_luminance: Some(10.0),
                            ..MaterialPatch::new("lamp")
                        })),
                        Op::Instance(InstancePatch {
                            mesh: Some("plane".into()),
                            material: Some("lamp".into()),
                            // Stood upright, facing the camera.
                            transform: Some(super::super::description::Transform::Trs {
                                translate: [0.0, 1.0, -1.0],
                                rotate_degrees: [90.0, 0.0, 0.0],
                                scale: [0.5, 1.0, 0.5],
                            }),
                            camera_visible: Some(visible),
                            ..InstancePatch::new("lamp")
                        }),
                    ],
                })
                .expect("valid data");
            Scene::prep(&gpu, &mut description).expect("prep")
        };

        let size = 64; // the shared render() helper's target size
        let probe = |pixels: &[f32], x: u32, y: u32| pixels[((y * size + x) * 4) as usize];
        let center = (size / 2, size / 2);
        let floor = (size / 2, size - 2);

        let seen = render(&gpu, &scene_with(true));
        let hidden = render(&gpu, &scene_with(false));
        assert!(
            probe(&seen, center.0, center.1) > 5.0,
            "the visible lamp should fill the frame center"
        );
        assert!(
            probe(&hidden, center.0, center.1) < 0.5,
            "the invisible lamp should leave only the sky behind it"
        );
        for (label, pixels) in [("visible", &seen), ("invisible", &hidden)] {
            assert!(
                probe(pixels, floor.0, floor.1) > 0.01,
                "{label}: the lamp should light the floor either way"
            );
        }
    }

    /// The untouched-on-error contract: an update rejected in the host
    /// phase leaves the previous residency rendering exactly as before.
    #[test]
    fn a_rejected_update_keeps_the_previous_scene() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let mut description = demo_description();
        let mut scene = Scene::prep(&gpu, &mut description).expect("prep");
        let before = render(&gpu, &scene);

        // A second camera is valid *data* but violates the prep-time
        // singleton rule.
        description
            .apply(&ChangeSet {
                ops: vec![Op::Camera(CameraPatch {
                    position: Some([9.0; 3]),
                    ..CameraPatch::new("second")
                })],
            })
            .expect("valid data");
        let dirty = description.take_dirty();
        let error = scene.update(&gpu, &description, &dirty).unwrap_err();
        assert!(matches!(error, Error::Scene(_)), "{error}");
        assert_eq!(render(&gpu, &scene), before);
    }
}
