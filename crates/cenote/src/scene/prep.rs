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
//! Both paths report where their time went, in the four [`Phases`] a build
//! spends it in — the measurement that turns "cheap over a scene's handful
//! of instances" from a claim into a number. Host `Instant`s are
//! exact here rather than approximate: every GPU call below blocks on a
//! fence before returning, so nothing this module starts is still running
//! when the clock is read.
//!
//! The fallible host-side lowering both paths run first — and the
//! [`Error::Scene`](crate::Error) contract that keeps a live session's last
//! good scene when an edit can't render — lives in [`super::lower`]:
//! everything that can fail on user data happens there, before the first
//! GPU call, so the GPU phase here can only fault on the device, which ends
//! the render anyway.

use std::collections::BTreeMap;
use std::time::Instant;

use super::changeset::Dirty;
use super::description::{Geometry, SceneDescription};
use super::lower::{InstanceSpec, all_dirty, host_phase};
use super::{
    GpuEnvironment, GpuMesh, Placement, ResidentBuffers, ResidentTexture, Scene,
    build_scene_tlas, select_probability, upload_environment, upload_instance_tables, upload_mesh,
    upload_scene_table, upload_texture_params,
};
use crate::error::Result;
use crate::gpu::Context;
use crate::stats::Phases;
use crate::texture;

impl Scene {
    /// Build `description` into a fresh, traceable scene, consuming its
    /// accumulated dirty state (a full build covers everything).
    ///
    /// # Errors
    ///
    /// [`Error::Scene`](crate::Error) when this build can't render the
    /// description — not exactly one camera and settings, more than one
    /// environment, or a referenced file (PLY, texture, environment) that
    /// doesn't read or decode. A description with no instances is *not* an
    /// error: it renders black (the render server starts on one, and a
    /// live edit may delete the last instance). Any other error is a GPU
    /// fault from upload or acceleration-structure builds.
    pub fn prep(gpu: &Context, description: &mut SceneDescription) -> Result<Self> {
        Self::prep_timed(gpu, description).map(|(scene, _)| scene)
    }

    /// [`Scene::prep`], reporting where a load's time went — ready to hand
    /// to [`Recorder::attribute_load`](crate::stats::Recorder::attribute_load)
    /// as it stands.
    ///
    /// [`Phases::before`] is process start to this call: the device
    /// creation, the shader compile, and the scene file read that a load
    /// waits through before any of the work below can begin. It is origined
    /// where the [`Recorder`](crate::stats::Recorder) origins
    /// time-to-first-ray, which is what makes the phases add up to that
    /// mark. A second build in the same process is not a startup and its
    /// `before` says nothing; that caller wants [`Scene::prep`].
    ///
    /// The two are the same code path, so a timed build and an untimed one
    /// cannot diverge in what they produce. Every stamp is a host `Instant`,
    /// which is exact here rather than approximate: every GPU call below
    /// blocks on a fence before returning, so the host clock and the device
    /// agree about when each phase ended.
    ///
    /// # Errors
    ///
    /// As [`Scene::prep`].
    #[expect(
        clippy::missing_panics_doc,
        reason = "the expects state all-dirty invariants — a fresh build always carries \
                  its environment and camera — not reachable panics"
    )]
    pub fn prep_timed(gpu: &Context, description: &mut SceneDescription) -> Result<(Self, Phases)> {
        let before = crate::stats::startup().elapsed();
        let lowering = Instant::now();
        let host = host_phase(description, &all_dirty(description), true, &BTreeMap::new(), None)?;
        let uploading = Instant::now();
        let mut upload = gpu.upload()?;
        let mut meshes = BTreeMap::new();
        for (name, mesh) in &host.meshes {
            meshes.insert(name.clone(), upload_mesh(&mut upload, name, mesh)?);
        }
        upload.finish()?;
        let textures = upload_textures(gpu, BTreeMap::new(), &host.textures)?;
        let descriptors = textures
            .values()
            .map(|texture| texture.image.descriptor())
            .collect();
        let texture_params = upload_texture_params(gpu, textures.keys())?;
        let spec = host
            .environment
            .as_ref()
            .expect("a fresh build always carries its environment");
        let environment = spec
            .image
            .as_ref()
            .expect("a fresh build has nothing resident to keep");
        let GpuEnvironment {
            image,
            marginal,
            conditional,
            pdfs,
            power,
        } = upload_environment(gpu, environment)?;
        let building = Instant::now();
        let placements = placements(&meshes, &host.instances);
        let tlas = build_scene_tlas(gpu, &placements)?;
        let tabling = Instant::now();
        let media = super::placement_media(&placements);
        let mut grids = crate::vdb::GridPool::new();
        let instances = upload_instance_tables(
            gpu,
            &mut grids,
            &placements,
            &host.triangle_lights,
            &host.delta_lights,
            host.global_medium.as_ref(),
        )?;
        drop(placements);
        let resident =
            ResidentBuffers::assemble(gpu, instances, texture_params, marginal, conditional, pdfs)?;
        let env_size = (environment.width(), environment.height());
        let tinted_power = power * f64::from(crate::color::luminance(spec.tint));
        let table = upload_scene_table(
            gpu,
            &resident,
            env_size,
            (spec.to_world, spec.from_world),
            spec.tint,
            select_probability(tinted_power, host.light_power()),
            host.light_count(),
        )?;
        description.take_dirty();
        let scene = Self {
            tlas,
            environment: image,
            table,
            resident,
            meshes,
            textures,
            descriptors,
            camera: host.camera.expect("a fresh build always adopts its camera"),
            media,
            grids,
            env_size,
            env_power: power,
            env_source: spec.source.clone(),
            env_tint: spec.tint,
            env_to_world: spec.to_world,
            env_from_world: spec.from_world,
        };
        let phases = Phases {
            before,
            ..phases(lowering, uploading, building, tabling)
        };
        Ok((scene, phases))
    }

    /// Rebuild exactly what `dirty` names, leaving the rest of the
    /// residency in place — the wave-boundary half of the edit channel.
    ///
    /// Returns where the rebuild's time went, in the four [`Phases`] it can
    /// spend it in; the rest of the struct is the caller's to fill. A phase
    /// an edit did not touch reads as very nearly zero rather than as
    /// absent, which is what makes two edit kinds comparable — and what
    /// makes "the tables cost this much on an edit that touched no
    /// instance" a number rather than an argument.
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
    ) -> Result<Phases> {
        let lowering = Instant::now();
        let resident_hashes = self
            .textures
            .iter()
            .map(|(key, texture)| (key.clone(), texture.hash))
            .collect();
        let host = host_phase(
            description,
            dirty,
            false,
            &resident_hashes,
            self.env_source.as_deref(),
        )?;
        let uploading = Instant::now();
        // Only device faults from here on — the untouched-on-Scene-error
        // contract holds because everything fallible already ran.
        for name in &host.removed_meshes {
            self.meshes.remove(name);
        }
        let mut upload = gpu.upload()?;
        for (name, mesh) in &host.meshes {
            self.meshes
                .insert(name.clone(), upload_mesh(&mut upload, name, mesh)?);
        }
        upload.finish()?;
        self.textures = upload_textures(gpu, std::mem::take(&mut self.textures), &host.textures)?;
        self.rebuild_texture_descriptors();
        self.resident.texture_params = upload_texture_params(gpu, self.textures.keys())?;
        if let Some(spec) = &host.environment {
            // The image re-uploads only when the source changed; the tint
            // and placement always re-land — they live in the scene table,
            // which rebuilds below regardless.
            if let Some(environment) = &spec.image {
                let env = upload_environment(gpu, environment)?;
                self.environment = env.image;
                self.resident.env_marginal = env.marginal;
                self.resident.env_conditional = env.conditional;
                self.resident.env_pdfs = env.pdfs;
                self.env_size = (environment.width(), environment.height());
                self.env_power = env.power;
            }
            self.env_source.clone_from(&spec.source);
            self.env_tint = spec.tint;
            self.env_to_world = spec.to_world;
            self.env_from_world = spec.from_world;
        }
        let building = Instant::now();
        let placements = placements(&self.meshes, &host.instances);
        if host.tlas_dirty {
            self.tlas = build_scene_tlas(gpu, &placements)?;
        }
        let tabling = Instant::now();
        self.media = super::placement_media(&placements);
        // Grid payloads stream from disk here rather than in the host phase
        // — multi-GiB files can't be held in memory across the boundary.
        // The host phase already validated each header, so a scene error
        // out of this call means the file changed under us mid-update.
        self.resident.instances = upload_instance_tables(
            gpu,
            &mut self.grids,
            &placements,
            &host.triangle_lights,
            &host.delta_lights,
            host.global_medium.as_ref(),
        )?;
        drop(placements);
        let env_power = self.tinted_env_power();
        self.table = upload_scene_table(
            gpu,
            &self.resident,
            self.env_size,
            (self.env_to_world, self.env_from_world),
            self.env_tint,
            select_probability(env_power, host.light_power()),
            host.light_count(),
        )?;
        if let Some(camera) = host.camera {
            self.camera = camera;
        }
        Ok(phases(lowering, uploading, building, tabling))
    }
}

/// The four phases a build or a rebuild spends its time in, from the stamps
/// at their shared boundaries.
///
/// Shared boundaries are the whole trick, and the one [`PassTimings`] uses:
/// each phase ends where the next begins, so the four telescope to the
/// interval from `lowering` to now with nothing between them to lose. The
/// closing stamp is taken here rather than passed in for the same reason —
/// one place decides where the last phase ends.
///
/// [`PassTimings`]: crate::stats::PassTimings
fn phases(lowering: Instant, uploading: Instant, building: Instant, tabling: Instant) -> Phases {
    Phases {
        lower: uploading - lowering,
        upload: building - uploading,
        // The placement list is built here too: it is the TLAS's input, and
        // at a scene's instance count it is not free.
        tlas: tabling - building,
        tables: tabling.elapsed(),
        ..Phases::default()
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

/// Resolve instance specs against the resident geometry map. The lookup
/// can't miss: apply validated every reference, and prep processes every
/// dirty mesh and curve batch, so residency tracks the description
/// reference for reference.
fn placements<'a>(
    meshes: &'a BTreeMap<Geometry, GpuMesh>,
    instances: &[InstanceSpec],
) -> Vec<Placement<'a>> {
    instances
        .iter()
        .map(|spec| Placement {
            mesh: meshes
                .get(&spec.mesh)
                .expect("geometry residency tracks the description"),
            transform: spec.transform,
            material: spec.material,
            camera_visible: spec.camera_visible,
            medium: spec.medium.clone(),
            priority: spec.priority,
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
    /// topology (BLAS), removal (retired residency), environment swap —
    /// then its tint and placement, which ride the keep-resident path — a
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
                        transforms: Some(vec![super::super::description::Transform::Trs {
                            translate: [0.0, 1.5, 0.0],
                            rotate_degrees: [0.0; 3],
                            scale: [0.75; 3],
                        }]),
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
                        path: Some(Some(sky.to_owned())),
                        ..EnvironmentPatch::new("sky")
                    })],
                },
            ),
            (
                // A tint edit rides the keep-resident path: same source,
                // new scene-table constants — the incremental build must
                // still land the exact fresh-build frame.
                "environment tint",
                ChangeSet {
                    ops: vec![Op::Environment(EnvironmentPatch {
                        tint: Some([0.9, 0.6, 0.3]),
                        ..EnvironmentPatch::new("sky")
                    })],
                },
            ),
            (
                "environment placement",
                ChangeSet {
                    ops: vec![Op::Environment(EnvironmentPatch {
                        transform: Some(super::super::description::Transform::Trs {
                            translate: [0.0; 3],
                            rotate_degrees: [0.0, 120.0, 0.0],
                            scale: [1.0; 3],
                        }),
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
                            channel: None,
                            scale: None,
                            uv: None,
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
                            transforms: Some(vec![super::super::description::Transform::Trs {
                                translate: [0.0; 3],
                                rotate_degrees: [0.0; 3],
                                scale: [8.0, 1.0, 8.0],
                            }]),
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
                            transforms: Some(vec![super::super::description::Transform::Trs {
                                translate: [0.0, 1.0, -1.0],
                                rotate_degrees: [90.0, 0.0, 0.0],
                                scale: [0.5, 1.0, 0.5],
                            }]),
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

    /// The render server stands up its `Session` on an empty scene —
    /// camera and settings only — so zero instances must prep, trace
    /// (every ray misses the empty TLAS), and render black under the
    /// default black sky rather than reject.
    #[test]
    fn an_empty_scene_preps_and_renders_black() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let mut description = SceneDescription::new();
        description
            .apply(&ChangeSet {
                ops: vec![
                    Op::Settings(SettingsPatch::new("main")),
                    Op::Camera(CameraPatch::new("main")),
                ],
            })
            .expect("valid data");
        let scene = Scene::prep(&gpu, &mut description).expect("an empty scene preps");
        let pixels = render(&gpu, &scene);
        assert!(
            pixels.chunks_exact(4).all(|texel| texel[..3] == [0.0; 3]),
            "an empty scene under a black sky should render black"
        );
    }

    /// …and a live edit may delete the last instance: the update lands and
    /// the scene renders empty instead of wedging the session on a
    /// rejection it can never edit its way out of.
    #[test]
    fn deleting_the_last_instance_updates_cleanly() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        // One emissive panel dead ahead of the default camera, so the
        // populated frame is provably non-black before the deletion.
        let mut description = SceneDescription::new();
        description
            .apply(&ChangeSet {
                ops: vec![
                    Op::Settings(SettingsPatch::new("main")),
                    Op::Camera(CameraPatch::new("main")),
                    Op::Mesh(MeshPatch {
                        source: Some(MeshSource::Inline {
                            positions: vec![
                                [-1.0, -1.0, 0.0],
                                [1.0, -1.0, 0.0],
                                [1.0, 1.0, 0.0],
                                [-1.0, 1.0, 0.0],
                            ],
                            normals: Some(vec![[0.0, 0.0, 1.0]; 4]),
                            uvs: None,
                            triangles: vec![[0, 1, 2], [0, 2, 3]],
                        }),
                        ..MeshPatch::new("panel")
                    }),
                    Op::Material(Box::new(MaterialPatch {
                        emission_luminance: Some(10.0),
                        ..MaterialPatch::new("lamp")
                    })),
                    Op::Instance(InstancePatch {
                        mesh: Some("panel".into()),
                        material: Some("lamp".into()),
                        ..InstancePatch::new("thing")
                    }),
                ],
            })
            .expect("valid data");
        let mut scene = Scene::prep(&gpu, &mut description).expect("prep");
        let lit = render(&gpu, &scene);
        assert!(
            lit.chunks_exact(4).any(|texel| texel[0] > 1.0),
            "the emissive panel should be visible before the deletion"
        );

        description
            .apply(&ChangeSet {
                ops: vec![
                    Op::Remove(Kind::Instance, "thing".into()),
                    Op::Remove(Kind::Mesh, "panel".into()),
                    Op::Remove(Kind::Material, "lamp".into()),
                ],
            })
            .expect("valid removal");
        let dirty = description.take_dirty();
        scene
            .update(&gpu, &description, &dirty)
            .expect("updating to an empty scene");
        let empty = render(&gpu, &scene);
        assert!(
            empty.chunks_exact(4).all(|texel| texel[..3] == [0.0; 3]),
            "the emptied scene should render black"
        );
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
