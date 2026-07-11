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
//! Prep is where the description meets what the renderer can currently
//! express, under two rules. Anything not wired up yet — and anything
//! legal but almost certainly a scene bug (a textured material over a
//! UV-less mesh) — is *warned by name* and rendered anyway, so a skipped
//! feature is never silent. Anything with no honest render at all is an
//! error: a PLY or texture file that doesn't read or decode, or a
//! description without its one camera and settings.
//! [`Error::Scene`] from here means *the description is fine but this
//! build can't render it* — residency is untouched, so a live session
//! keeps its last good scene and reports the edit instead of dying.
//!
//! Fallibility is split to keep that promise cheap: everything that can
//! fail on user data (file reads, decodes, capability checks) happens
//! host-side in [`host_phase`], before the first GPU call; the GPU phase
//! can only fail on device faults, which end the render anyway.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

use glam::{Mat4, Vec2, Vec3};

use super::changeset::{Dirty, Kind};
use super::description::{self, ColorSpace, MeshSource, SceneDescription, Texturable, TextureRef};
use super::{
    Camera, GpuEnvironment, GpuMesh, Lens, Mesh, Placement, ResidentBuffers, ResidentTexture,
    Scene, build_scene_tlas, emissive_triangles, scene_error, select_probability,
    upload_environment, upload_instance_tables, upload_mesh, upload_scene_table,
};
use crate::color::{acescg_from_rec709, luminance};
use crate::environment::Environment;
use crate::error::{Error, Result};
use crate::gpu::Context;
use crate::lights::{DeltaLight, TriangleLight};
use crate::material::{Material, TEXTURE_NONE};
use crate::texture;

impl Scene {
    /// Build `description` into a fresh, traceable scene, consuming its
    /// accumulated dirty state (a full build covers everything).
    ///
    /// # Errors
    ///
    /// [`Error::Scene`] when this build can't render the description —
    /// not exactly one camera and settings, more than one environment, no
    /// instances, or a referenced file (PLY, texture, environment) that
    /// doesn't read or decode. Any other error is a GPU fault from upload
    /// or acceleration-structure builds.
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
        let textures = upload_textures(gpu, BTreeMap::new(), &host)?;
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
    /// [`Error::Scene`] means this build can't render the edited
    /// description (see [`Scene::prep`]); the scene is **untouched**, so
    /// the caller keeps rendering the previous residency and may retry
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
        self.textures = upload_textures(gpu, std::mem::take(&mut self.textures), &host)?;
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

/// Everything prep derives host-side before touching the GPU — the
/// fallible half, so a rejected description leaves residency untouched.
struct HostScene {
    /// Meshes to (re)upload: the dirty subset, geometry resolved and
    /// normals derived where absent.
    meshes: BTreeMap<String, Mesh>,
    /// Mesh residency to retire. Processed before `meshes`, so a
    /// remove-then-recreate lands the new build.
    removed_meshes: Vec<String>,
    /// Every instance, in name order — custom index is position.
    instances: Vec<InstanceSpec>,
    /// The emissive geometry, one entry per triangle of every emissive
    /// instance.
    triangle_lights: Vec<TriangleLight>,
    /// The delta lights, lowered from the description's light objects.
    delta_lights: Vec<DeltaLight>,
    /// Every texture the description references, in bindless-index order
    /// (`BTreeMap` iteration *is* the index assignment). Values are the
    /// prepped data to (re)upload — `None` keeps the resident image, whose
    /// content hash matched.
    textures: BTreeMap<texture::Key, Option<texture::Prepared>>,
    /// Loaded when the environment changed (always, on a fresh build);
    /// `None` keeps the resident image and tables.
    environment: Option<Arc<Environment>>,
    /// The camera, when it changed — a material edit must not snap the
    /// view back to the authored pose.
    camera: Option<Camera>,
    /// The TLAS must rebuild. Set by a mesh, instance, *or* material edit —
    /// material because fractional opacity bakes into each instance's
    /// non-opaque flag (see where it's assigned).
    tlas_dirty: bool,
}

impl HostScene {
    /// Total records the light table will hold.
    fn light_count(&self) -> u32 {
        (self.triangle_lights.len() + self.delta_lights.len()) as u32
    }

    /// The list's selection weight against the environment.
    fn light_power(&self) -> f64 {
        crate::lights::total_power(&self.triangle_lights, &self.delta_lights)
    }
}

/// One instance lowered from the description, resolved against the
/// resident mesh map at assembly time.
struct InstanceSpec {
    mesh: String,
    transform: Mat4,
    material: Material,
    camera_visible: bool,
}

/// Derive everything the GPU phase consumes, validating as it goes. Warns
/// only about objects `dirty` names, so a long edit session doesn't
/// repeat itself about parameters it already reported. `fresh` marks a
/// full build, which loads its environment even when no dirt names one —
/// a description without an environment object leaves nothing to mark.
/// `resident_textures` maps already-uploaded textures to their content
/// hashes, so an edit re-preps only textures a dirty material references —
/// and re-uploads only those whose source content actually changed.
fn host_phase(
    description: &SceneDescription,
    dirty: &Dirty,
    fresh: bool,
    resident_textures: &BTreeMap<texture::Key, u64>,
) -> Result<HostScene> {
    let (_, camera_source) = singleton(description.cameras(), "camera")?;
    singleton(description.settings(), "settings")?;
    if description.environments().len() > 1 {
        return Err(scene_error(format!(
            "a scene renders at most one environment, this one has {}",
            description.environments().len()
        )));
    }
    if description.instances().is_empty() {
        return Err(scene_error("a scene needs at least one instance".into()));
    }

    let changed_meshes = names(&dirty.changed, Kind::Mesh);
    let mut meshes = BTreeMap::new();
    for (name, mesh) in description.meshes() {
        if changed_meshes.contains(name.as_str()) {
            meshes.insert(name.clone(), resolve_mesh(name, mesh)?);
        }
    }
    let removed_meshes: Vec<String> = names(&dirty.removed, Kind::Mesh)
        .into_iter()
        .map(str::to_owned)
        .collect();

    let changed_materials = names(&dirty.changed, Kind::Material);

    // Texture references, collected description-wide first so shared
    // images prep once and index assignment (key order) is deterministic.
    // A key preps when it isn't resident yet or a dirty material names it
    // — the latter re-hashes the source so a repainted image reloads on
    // the next material touch — and uploads only when the content hash
    // says the resident image is actually stale.
    let mut referenced: BTreeMap<texture::Key, bool> = BTreeMap::new();
    for (name, material) in description.materials() {
        let noisy = changed_materials.contains(name.as_str());
        for key in texture_keys(material) {
            *referenced.entry(key).or_insert(false) |= noisy;
        }
    }
    if referenced.len() > crate::gpu::MAX_SCENE_TEXTURES as usize {
        return Err(scene_error(format!(
            "the scene references {} textures; the bindless table holds {}",
            referenced.len(),
            crate::gpu::MAX_SCENE_TEXTURES
        )));
    }
    let mut textures: BTreeMap<texture::Key, Option<texture::Prepared>> = BTreeMap::new();
    for (key, touched) in referenced {
        let resident = resident_textures.get(&key).copied();
        let prepared = if resident.is_none() || touched {
            let prepared = texture::prepare(&key.0, key.1, key.2)?;
            (resident != Some(prepared.hash)).then_some(prepared)
        } else {
            None
        };
        textures.insert(key, prepared);
    }
    let texture_indices: BTreeMap<&texture::Key, u32> = textures
        .keys()
        .enumerate()
        .map(|(index, key)| (key, index as u32))
        .collect();

    let mut materials: BTreeMap<&str, Material> = BTreeMap::new();
    for (name, source) in description.materials() {
        materials.insert(
            name,
            lower_material(
                name,
                source,
                changed_materials.contains(name.as_str()),
                &texture_indices,
            ),
        );
    }
    warn_textured_without_uvs(description, dirty);

    let delta_lights = lower_delta_lights(description);
    let (instances, triangle_lights) = lower_instances(description, &materials, &meshes)?;

    let touched = |kind: Kind| {
        dirty
            .changed
            .iter()
            .chain(&dirty.removed)
            .any(|(entry, _)| *entry == kind)
    };
    let camera = touched(Kind::Camera).then(|| lower_camera(camera_source));
    let environment = if fresh || touched(Kind::Environment) {
        Some(match description.environments().iter().next() {
            Some((name, environment)) => load_environment(name, &environment.path)?,
            // No environment is a black sky: zero power, so next-event
            // estimation puts all its draws on the quads.
            None => Arc::new(Environment::constant(Vec3::ZERO)),
        })
    } else {
        None
    };

    Ok(HostScene {
        meshes,
        removed_meshes,
        instances,
        triangle_lights,
        delta_lights,
        textures,
        environment,
        camera,
        // Material dirt rebuilds the TLAS too: fractional opacity is baked
        // into each instance's non-opaque flag, and the TLAS over a scene's
        // handful of instances is the cheap structure (every BLAS stays).
        tlas_dirty: touched(Kind::Mesh) || touched(Kind::Instance) || touched(Kind::Material),
    })
}

/// Lower the description's camera, resolving the thin lens: a positive
/// aperture makes a [`Lens`], focused at `focus_distance` or — when the
/// author left it unset — at `look_at`.
fn lower_camera(source: &description::Camera) -> Camera {
    let position = Vec3::from(source.position);
    let look_at = Vec3::from(source.look_at);
    Camera {
        position,
        look_at,
        up: source.up.into(),
        vfov_degrees: source.vfov_degrees,
        lens: (source.aperture_radius > 0.0).then(|| Lens {
            aperture_radius: source.aperture_radius,
            focus_distance: source
                .focus_distance
                .unwrap_or_else(|| position.distance(look_at)),
        }),
    }
}

/// Lower the description's delta lights, in name order, converting their
/// `Rec.709` colors to `ACEScg` (prep owns that conversion, as with
/// materials). A powerless light is skipped outright — the get-or-create
/// placeholder is a black point light, and a record that can never be
/// selected would only pad the table.
fn lower_delta_lights(description: &SceneDescription) -> Vec<DeltaLight> {
    description
        .lights()
        .values()
        .filter_map(|light| match light {
            description::Light::Distant {
                direction,
                irradiance,
            } => {
                let irradiance = acescg_from_rec709(Vec3::from(*irradiance));
                (luminance(irradiance) > 0.0).then(|| DeltaLight::Distant {
                    // Validated nonzero at apply.
                    direction: Vec3::from(*direction).normalize(),
                    irradiance,
                })
            }
            description::Light::Point {
                position,
                intensity,
            } => {
                let intensity = acescg_from_rec709(Vec3::from(*intensity));
                (luminance(intensity) > 0.0).then(|| DeltaLight::Point {
                    position: Vec3::from(*position),
                    intensity,
                })
            }
        })
        .collect()
}

/// Lower every instance into its placement spec, unpacking each emissive
/// one into its per-triangle lights.
fn lower_instances(
    description: &SceneDescription,
    materials: &BTreeMap<&str, Material>,
    resolved: &BTreeMap<String, Mesh>,
) -> Result<(Vec<InstanceSpec>, Vec<TriangleLight>)> {
    let mut instances = Vec::with_capacity(description.instances().len());
    let mut triangle_lights = Vec::new();
    for (index, instance) in description.instances().values().enumerate() {
        // Apply validated the references and the transform, so lookups
        // can't miss and the inverse the records need exists.
        let material = materials[instance.material.as_str()];
        let transform = instance.transform.to_mat4();
        if luminance(material.emission) > 0.0 {
            let (positions, triangles) = emissive_geometry(
                resolved.get(&instance.mesh),
                &description.meshes()[&instance.mesh],
            )?;
            triangle_lights.extend(emissive_triangles(
                &positions,
                &triangles,
                transform,
                material.emission,
                index as u32,
            ));
        }
        instances.push(InstanceSpec {
            mesh: instance.mesh.clone(),
            transform,
            material,
            camera_visible: instance.camera_visible,
        });
    }
    Ok((instances, triangle_lights))
}

/// Lower an authoring-side material onto the GPU record, converting color
/// constants from the format's linear `Rec.709` into `ACEScg` — prep owns
/// that conversion (textures make the same trip in-shader, after the
/// hardware's sRGB decode) — and clamping weights into the ranges the
/// kernel's lerps assume. The coat's tint on emission folds in here: it
/// is a view-independent constant in this closure, and folding it keeps
/// the light table and the shading kernel reading the same emitted
/// radiance. Textured slots resolve to bindless indices through
/// `indices`; their constants lower to stand-ins — replaced slots get the
/// schema default, multiplied slots (emission, opacity) the identity.
fn lower_material(
    name: &str,
    source: &description::Material,
    warn: bool,
    indices: &BTreeMap<&texture::Key, u32>,
) -> Material {
    // The collection pass walked every reference through the same
    // `texture_key`, so these lookups cannot miss.
    let slot = |reference: Option<&TextureRef>, usage: texture::Usage| -> u32 {
        reference.map_or(TEXTURE_NONE, |reference| {
            indices[&texture_key(reference, usage)]
        })
    };
    let base_color = constant_or(&source.base_color, [0.8; 3]);
    let metalness = constant_or(&source.base_metalness, 0.0);
    let specular_roughness = constant_or(&source.specular_roughness, 0.3);
    let emission_color = constant_or(&source.emission_color, [1.0; 3]);
    let opacity = constant_or(&source.geometry_opacity, 1.0);
    if warn
        && source
            .geometry_normal
            .as_ref()
            .is_some_and(|reference| reference.color_space == Some(ColorSpace::Srgb))
    {
        log::warn!(
            "material \"{name}\": geometry_normal ignores its sRGB color-space \
             override — normal maps are always linear"
        );
    }

    let coat_weight = source.coat_weight.clamp(0.0, 1.0);
    let coat_color = acescg_from_rec709(Vec3::from(source.coat_color)).max(Vec3::ZERO);
    let mut material = Material::matte(
        acescg_from_rec709(Vec3::from(base_color)),
        source.base_diffuse_roughness.clamp(0.0, 1.0),
    );
    material.metalness = metalness.clamp(0.0, 1.0);
    material.specular_weight = source.specular_weight.max(0.0);
    material.specular_roughness = specular_roughness;
    material.specular_ior = source.specular_ior.max(1e-4);
    material.transmission_weight = source.transmission_weight.clamp(0.0, 1.0);
    // Transmittance above 1 would make Beer–Lambert *amplify*; the kernel
    // guards the lower end (a hard 0 means an infinite extinction).
    material.transmission_color =
        acescg_from_rec709(Vec3::from(source.transmission_color)).clamp(Vec3::ZERO, Vec3::ONE);
    material.transmission_depth = source.transmission_depth.max(0.0);
    material.coat_color = coat_color;
    material.coat_weight = coat_weight;
    material.coat_roughness = source.coat_roughness.clamp(0.0, 1.0);
    material.coat_ior = source.coat_ior.max(1.0);
    material.coat_darkening = source.coat_darkening.clamp(0.0, 1.0);
    material.fuzz_weight = source.fuzz_weight.clamp(0.0, 1.0);
    material.fuzz_color = acescg_from_rec709(Vec3::from(source.fuzz_color)).max(Vec3::ZERO);
    material.fuzz_roughness = source.fuzz_roughness.clamp(0.0, 1.0);
    material.opacity = opacity.clamp(0.0, 1.0);
    material.thin_walled = u32::from(source.geometry_thin_walled);
    // Emission leaves through the coat: L_e = lerp(1, coat_color, C)·E,
    // OpenPBR's reduction with its view-independent coat transmittance.
    // With an emission map, this is the map's scale (the light table
    // weighs selection by it too — the map's spatial variation only
    // steers noise, never the estimate).
    material.emission = acescg_from_rec709(Vec3::from(emission_color))
        * source.emission_luminance
        * Vec3::ONE.lerp(coat_color, coat_weight);
    material.base_color_texture = slot(source.base_color.texture(), texture::Usage::Color);
    material.specular_roughness_texture =
        slot(source.specular_roughness.texture(), texture::Usage::Scalar);
    material.metalness_texture = slot(source.base_metalness.texture(), texture::Usage::Scalar);
    material.emission_texture = slot(source.emission_color.texture(), texture::Usage::Color);
    material.opacity_texture = slot(source.geometry_opacity.texture(), texture::Usage::Scalar);
    material.normal_texture = slot(source.geometry_normal.as_ref(), texture::Usage::Normal);
    material
}

/// A texturable slot's constant, or `stand_in` when it is textured — the
/// schema default for slots the kernel replaces per hit, the identity for
/// slots it multiplies (emission, opacity).
fn constant_or<T: Copy>(value: &Texturable<T>, stand_in: T) -> T {
    match value {
        Texturable::Constant(constant) => *constant,
        Texturable::Texture(_) => stand_in,
    }
}

/// The prep request a texture reference makes when feeding `usage` — the
/// identity textures are collected, prepped, and indexed under.
fn texture_key(reference: &TextureRef, usage: texture::Usage) -> texture::Key {
    let srgb = match usage {
        // Normal maps are always linear; a stray override must not fork
        // the cache (its lowering warns instead).
        texture::Usage::Normal => None,
        texture::Usage::Color | texture::Usage::Scalar => {
            reference.color_space.map(|space| space == ColorSpace::Srgb)
        }
    };
    (reference.path.clone(), usage, srgb)
}

/// Every prep request a material makes, one per textured slot.
fn texture_keys(material: &description::Material) -> impl Iterator<Item = texture::Key> {
    [
        (material.base_color.texture(), texture::Usage::Color),
        (material.base_metalness.texture(), texture::Usage::Scalar),
        (
            material.specular_roughness.texture(),
            texture::Usage::Scalar,
        ),
        (material.emission_color.texture(), texture::Usage::Color),
        (material.geometry_opacity.texture(), texture::Usage::Scalar),
        (material.geometry_normal.as_ref(), texture::Usage::Normal),
    ]
    .into_iter()
    .filter_map(|(reference, usage)| reference.map(|reference| texture_key(reference, usage)))
}

/// A textured material over a mesh with no authored UVs samples texel
/// (0, 0) everywhere — legal, but almost certainly a scene bug, so it
/// warns once per touched (instance, material, mesh) combination.
fn warn_textured_without_uvs(description: &SceneDescription, dirty: &Dirty) {
    for (name, instance) in description.instances() {
        let material = &description.materials()[&instance.material];
        if material.textures().next().is_none() {
            continue;
        }
        let has_uvs = match &description.meshes()[&instance.mesh].source {
            MeshSource::Inline { uvs, .. } => uvs.is_some(),
            // This warning reads the description only; whether a PLY file
            // carries UVs is known after resolution, so a UV-less one gets
            // the benefit of the doubt (its lookups still read texel (0, 0)).
            MeshSource::Ply { .. } => true,
        };
        let touched = |kind: Kind, target: &str| dirty.changed.contains(&(kind, target.to_owned()));
        if !has_uvs
            && (touched(Kind::Instance, name)
                || touched(Kind::Material, &instance.material)
                || touched(Kind::Mesh, &instance.mesh))
        {
            log::warn!(
                "instance \"{name}\": material \"{}\" is textured but mesh \"{}\" has \
                 no UVs — every lookup reads texel (0, 0)",
                instance.material,
                instance.mesh
            );
        }
    }
}

/// The GPU half of texture residency: keep the resident images the host
/// phase kept, upload the ones it prepped (new or content-changed), and
/// drop whatever nothing references anymore. Returns the new resident
/// map — iteration order is the bindless index order the lowered
/// materials already encode.
fn upload_textures(
    gpu: &Context,
    mut resident: BTreeMap<texture::Key, ResidentTexture>,
    host: &HostScene,
) -> Result<BTreeMap<texture::Key, ResidentTexture>> {
    let mut textures = BTreeMap::new();
    for (key, prepared) in &host.textures {
        let texture = match prepared {
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
                .expect("the host phase keeps only resident textures"),
        };
        textures.insert(key.clone(), texture);
    }
    Ok(textures)
}

/// Resolve a mesh's geometry payload onto the host, deriving normals
/// where the payload carries none.
fn resolve_mesh(name: &str, mesh: &description::Mesh) -> Result<Mesh> {
    match &mesh.source {
        MeshSource::Inline {
            positions,
            normals,
            uvs,
            triangles,
        } => {
            let positions: Vec<Vec3> = positions.iter().copied().map(Vec3::from).collect();
            let normals = match normals {
                Some(normals) => normals.iter().copied().map(Vec3::from).collect(),
                None => smooth_normals(&positions, triangles),
            };
            // An unauthored stream carries zeros: textured lookups on it
            // read texel (0, 0) — constant, never out of bounds.
            let uvs = match uvs {
                Some(uvs) => uvs.iter().copied().map(Vec2::from).collect(),
                None => vec![Vec2::ZERO; positions.len()],
            };
            Ok(Mesh {
                positions,
                normals,
                uvs,
                triangles: triangles.clone(),
            })
        }
        MeshSource::Ply { path } => {
            let ply = crate::ply::read(path).map_err(|error| match error {
                Error::Scene(message) => scene_error(format!("mesh \"{name}\": {message}")),
                other => other,
            })?;
            let normals = ply
                .normals
                .unwrap_or_else(|| smooth_normals(&ply.positions, &ply.triangles));
            let uvs = ply
                .uvs
                .unwrap_or_else(|| vec![Vec2::ZERO; ply.positions.len()]);
            Ok(Mesh {
                positions: ply.positions,
                normals,
                uvs,
                triangles: ply.triangles,
            })
        }
    }
}

/// Area-weighted smooth vertex normals — the fallback when an inline mesh
/// carries none (imported meshes often don't). Each face's unnormalized
/// cross product accumulates onto its corners (its length is twice the
/// face's area, so larger faces weigh more), then everything normalizes.
/// A vertex no face touches, or whose faces cancel exactly, falls back to
/// +Y: it can't be hit, but its normal must still be finite.
fn smooth_normals(positions: &[Vec3], triangles: &[[u32; 3]]) -> Vec<Vec3> {
    let mut sums = vec![Vec3::ZERO; positions.len()];
    for &[a, b, c] in triangles {
        let (a, b, c) = (a as usize, b as usize, c as usize);
        let face = (positions[b] - positions[a]).cross(positions[c] - positions[a]);
        sums[a] += face;
        sums[b] += face;
        sums[c] += face;
    }
    sums.into_iter()
        .map(|sum| sum.try_normalize().unwrap_or(Vec3::Y))
        .collect()
}

/// A mesh's positions and triangles for the light table. The resolved
/// copy serves when this round already loaded the mesh; otherwise inline
/// geometry converts from the description and a PLY reference re-reads
/// its file — an emissive PLY mesh pays a re-read when a *non-mesh* edit
/// rebuilds the lights, which is rare enough (emitters are almost always
/// simple quads) to not be worth a host-side geometry cache.
fn emissive_geometry(
    resolved: Option<&Mesh>,
    mesh: &description::Mesh,
) -> Result<(Vec<Vec3>, Vec<[u32; 3]>)> {
    if let Some(mesh) = resolved {
        return Ok((mesh.positions.clone(), mesh.triangles.clone()));
    }
    match &mesh.source {
        MeshSource::Inline {
            positions,
            triangles,
            ..
        } => Ok((
            positions.iter().copied().map(Vec3::from).collect(),
            triangles.clone(),
        )),
        MeshSource::Ply { path } => {
            let ply = crate::ply::read(path)?;
            Ok((ply.positions, ply.triangles))
        }
    }
}

/// Read and decode an environment EXR. Failures are [`Error::Scene`] —
/// a bad image is scene data, not a device fault, and a live edit to one
/// must not end the render.
fn load_environment(name: &str, path: &Path) -> Result<Arc<Environment>> {
    // The lib test suite preps the demo scene dozens of times per
    // process, and its 4k decode is seconds of debug-profile CPU each —
    // tests share decoded environments by path. Outside tests a process
    // preps a scene once and shouldn't pin ~200 MB of host copies for its
    // lifetime.
    #[cfg(test)]
    {
        use std::path::PathBuf;
        use std::sync::Mutex;
        static CACHE: Mutex<BTreeMap<PathBuf, Arc<Environment>>> = Mutex::new(BTreeMap::new());
        let mut cache = CACHE.lock().expect("environment cache poisoned");
        if let Some(environment) = cache.get(path) {
            return Ok(Arc::clone(environment));
        }
        let environment = decode_environment(name, path)?;
        cache.insert(path.to_owned(), Arc::clone(&environment));
        Ok(environment)
    }
    #[cfg(not(test))]
    decode_environment(name, path)
}

fn decode_environment(name: &str, path: &Path) -> Result<Arc<Environment>> {
    let bytes = std::fs::read(path).map_err(|error| {
        scene_error(format!(
            "environment \"{name}\": can't read \"{}\": {error}",
            path.display()
        ))
    })?;
    let environment = Environment::from_equirect_exr(&bytes).map_err(|error| {
        scene_error(format!(
            "environment \"{name}\": \"{}\" doesn't decode as an EXR: {error}",
            path.display()
        ))
    })?;
    Ok(Arc::new(environment))
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

/// Exactly one object of a kind — the prep-time singleton rule the
/// description model deliberately doesn't enforce.
fn singleton<'a, T>(map: &'a BTreeMap<String, T>, kind: &str) -> Result<(&'a str, &'a T)> {
    let mut objects = map.iter();
    match (objects.next(), objects.next()) {
        (Some((name, value)), None) => Ok((name.as_str(), value)),
        (None, _) => Err(scene_error(format!(
            "a scene needs exactly one {kind}, this one has none"
        ))),
        (Some(_), Some(_)) => Err(scene_error(format!(
            "a scene renders exactly one {kind}, this one has {}",
            map.len()
        ))),
    }
}

/// The names of one kind within a dirty set, borrowed for cheap lookups.
fn names(set: &BTreeSet<(Kind, String)>, kind: Kind) -> BTreeSet<&str> {
    set.iter()
        .filter(|(entry, _)| *entry == kind)
        .map(|(_, name)| name.as_str())
        .collect()
}

/// Every object in the description marked changed — what a fresh build
/// hands the shared host phase.
fn all_dirty(description: &SceneDescription) -> Dirty {
    fn mark<T>(dirty: &mut Dirty, kind: Kind, map: &BTreeMap<String, T>) {
        for name in map.keys() {
            dirty.changed.insert((kind, name.clone()));
        }
    }
    let mut dirty = Dirty::default();
    mark(&mut dirty, Kind::Mesh, description.meshes());
    mark(&mut dirty, Kind::Instance, description.instances());
    mark(&mut dirty, Kind::Material, description.materials());
    mark(&mut dirty, Kind::Light, description.lights());
    mark(&mut dirty, Kind::Camera, description.cameras());
    mark(&mut dirty, Kind::Environment, description.environments());
    mark(&mut dirty, Kind::Settings, description.settings());
    dirty
}

#[cfg(test)]
mod tests {
    use super::super::changeset::{
        CameraPatch, ChangeSet, EnvironmentPatch, InstancePatch, MaterialPatch, MeshPatch, Op,
        SettingsPatch,
    };
    use super::super::description::TextureRef;
    use super::*;
    use crate::render::Renderer;

    /// The demo, applied — the standing prep test subject.
    fn demo_description() -> SceneDescription {
        let mut description = SceneDescription::new();
        description
            .apply(&ChangeSet::demo())
            .expect("the demo change-set is valid");
        description
    }

    /// A minimal valid description: one triangle instance under a camera
    /// and settings, no environment.
    fn triangle_description() -> SceneDescription {
        let mut description = SceneDescription::new();
        description
            .apply(&triangle_set())
            .expect("the triangle set is valid");
        description
    }

    fn triangle_set() -> ChangeSet {
        ChangeSet {
            ops: vec![
                Op::Settings(SettingsPatch::new("main")),
                Op::Camera(CameraPatch::new("main")),
                Op::Mesh(MeshPatch {
                    source: Some(MeshSource::Inline {
                        positions: vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                        normals: None,
                        uvs: None,
                        triangles: vec![[0, 1, 2]],
                    }),
                    ..MeshPatch::new("tri")
                }),
                Op::Material(Box::new(MaterialPatch::new("gray"))),
                Op::Instance(InstancePatch {
                    mesh: Some("tri".into()),
                    material: Some("gray".into()),
                    ..InstancePatch::new("thing")
                }),
            ],
        }
    }

    fn host(description: &SceneDescription) -> Result<HostScene> {
        host_phase(description, &all_dirty(description), true, &BTreeMap::new())
    }

    /// `unwrap_err` without demanding `Debug` of the GPU-adjacent
    /// [`HostScene`].
    fn host_error(description: &SceneDescription) -> Error {
        match host(description) {
            Err(error) => error,
            Ok(_) => panic!("the host phase accepted a description it must reject"),
        }
    }

    #[test]
    fn the_singleton_rules_hold() {
        let mut description = triangle_description();
        let error = host_error(&SceneDescription::new());
        assert!(error.to_string().contains("camera"), "{error}");

        description
            .apply(&ChangeSet {
                ops: vec![Op::Camera(CameraPatch {
                    position: Some([5.0; 3]),
                    ..CameraPatch::new("second")
                })],
            })
            .expect("a second camera is valid data");
        let error = host_error(&description);
        assert!(error.to_string().contains("exactly one camera"), "{error}");
    }

    #[test]
    fn a_scene_without_instances_is_rejected() {
        let mut description = SceneDescription::new();
        description
            .apply(&ChangeSet {
                ops: vec![
                    Op::Settings(SettingsPatch::new("main")),
                    Op::Camera(CameraPatch::new("main")),
                ],
            })
            .expect("valid data");
        let error = host_error(&description);
        assert!(error.to_string().contains("instance"), "{error}");
    }

    #[test]
    fn ply_geometry_is_rejected_by_name() {
        let mut description = triangle_description();
        description
            .apply(&ChangeSet {
                ops: vec![Op::Mesh(MeshPatch {
                    source: Some(MeshSource::Ply {
                        // Apply checks existence, not contents.
                        path: concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml").into(),
                    }),
                    ..MeshPatch::new("tri")
                })],
            })
            .expect("valid data");
        let error = host_error(&description);
        assert!(error.to_string().contains("PLY"), "{error}");
        assert!(error.to_string().contains("tri"), "{error}");
    }

    /// Any emissive mesh is a light — one record per triangle, in
    /// primitive order (a single bare triangle was a hard error while the
    /// light sampler only spoke parallelogram quads).
    #[test]
    fn any_emissive_mesh_is_a_light() {
        let mut description = triangle_description();
        description
            .apply(&ChangeSet {
                ops: vec![Op::Material(Box::new(MaterialPatch {
                    emission_luminance: Some(5.0),
                    ..MaterialPatch::new("gray")
                }))],
            })
            .expect("valid data");
        let host = host(&description).expect("a triangle emitter renders");
        assert_eq!(host.triangle_lights.len(), 1);
        assert_eq!(host.triangle_lights[0].primitive, 0);
        assert!(crate::color::luminance(host.triangle_lights[0].emission) > 0.0);
    }

    /// A texture that exists but doesn't decode is caught in the host
    /// phase — [`Error::Scene`], so a live session keeps its previous
    /// residency rather than dying on a bad image. (A *missing* file is
    /// already an apply-time error, like every dangling path.)
    #[test]
    fn an_undecodable_texture_is_rejected_at_prep() {
        let mut description = triangle_description();
        description
            .apply(&ChangeSet {
                ops: vec![Op::Material(Box::new(MaterialPatch {
                    base_color: Some(Texturable::Texture(TextureRef {
                        // Exists (so apply accepts it) but is no image.
                        path: concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml").into(),
                        color_space: None,
                    })),
                    ..MaterialPatch::new("gray")
                }))],
            })
            .expect("valid data");
        let error = host_error(&description);
        assert!(error.to_string().contains("texture"), "{error}");
        assert!(error.to_string().contains("decode"), "{error}");
    }

    /// A PLY mesh resolves through the host phase like inline geometry:
    /// its streams load, missing normals derive, and — because the
    /// resolved copy is on hand — its triangles feed the light table when
    /// the material emits.
    #[test]
    fn a_ply_mesh_resolves_and_can_emit() {
        let dir = std::env::temp_dir().join(format!("cenote-prep-ply-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("fixture dir");
        let path = dir.join("quad.ply");
        std::fs::write(
            &path,
            "ply\nformat ascii 1.0\nelement vertex 4\n\
             property float x\nproperty float y\nproperty float z\n\
             property float u\nproperty float v\n\
             element face 1\nproperty list uchar int vertex_indices\nend_header\n\
             0 0 0 0 0\n1 0 0 1 0\n1 1 0 1 1\n0 1 0 0 1\n4 0 1 2 3\n",
        )
        .expect("write fixture");

        let mut description = triangle_description();
        description
            .apply(&ChangeSet {
                ops: vec![
                    Op::Mesh(MeshPatch {
                        source: Some(MeshSource::Ply { path: path.clone() }),
                        ..MeshPatch::new("tri")
                    }),
                    Op::Material(Box::new(MaterialPatch {
                        emission_luminance: Some(3.0),
                        ..MaterialPatch::new("gray")
                    })),
                ],
            })
            .expect("valid data");
        let host = host(&description).expect("a PLY mesh preps");
        let mesh = &host.meshes["tri"];
        assert_eq!(mesh.positions.len(), 4);
        assert_eq!(mesh.triangles.len(), 2);
        // No authored normals: derived, and this quad's winding faces +Z.
        assert!(mesh.normals.iter().all(|n| n.abs_diff_eq(Vec3::Z, 1e-6)));
        assert_eq!(mesh.uvs[2], Vec2::new(1.0, 1.0));
        assert_eq!(host.triangle_lights.len(), 2);

        // A file that exists but isn't PLY is a host-phase rejection that
        // names the mesh, not a crash or a silent skip.
        description
            .apply(&ChangeSet {
                ops: vec![Op::Mesh(MeshPatch {
                    source: Some(MeshSource::Ply {
                        path: concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml").into(),
                    }),
                    ..MeshPatch::new("tri")
                })],
            })
            .expect("valid data");
        let error = host_error(&description);
        assert!(error.to_string().contains("mesh \"tri\""), "{error}");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_unreadable_environment_is_rejected_at_prep() {
        let mut description = triangle_description();
        description
            .apply(&ChangeSet {
                ops: vec![Op::Environment(EnvironmentPatch {
                    // Exists (so apply accepts it) but is no EXR.
                    path: Some(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml").into()),
                    ..EnvironmentPatch::new("sky")
                })],
            })
            .expect("valid data");
        let error = host_error(&description);
        assert!(error.to_string().contains("decode"), "{error}");
    }

    /// Delta lights lower into the light list — direction normalized,
    /// colors converted — while the get-or-create placeholder (a black
    /// point light) is skipped as powerless.
    #[test]
    fn delta_lights_lower_into_the_light_list() {
        use super::super::changeset::LightPatch;
        use super::super::description::Light;

        let mut description = triangle_description();
        description
            .apply(&ChangeSet {
                ops: vec![
                    Op::Light(LightPatch {
                        light: Some(Light::Distant {
                            direction: [0.0, -2.0, 0.0],
                            irradiance: [3.0; 3],
                        }),
                        ..LightPatch::new("sun")
                    }),
                    Op::Light(LightPatch::new("placeholder")),
                ],
            })
            .expect("valid data");
        let host = host(&description).expect("delta lights render");
        let [DeltaLight::Distant { direction, .. }] = host.delta_lights.as_slice() else {
            panic!("expected exactly the one distant light to survive lowering");
        };
        assert!((*direction - Vec3::NEG_Y).length() < 1e-6, "{direction}");
    }

    /// A positive aperture lowers into a thin lens focused at `look_at`
    /// when the author left `focus_distance` unset; aperture zero stays a
    /// pinhole no matter the focus value.
    #[test]
    fn the_camera_lens_lowers_with_focus_at_look_at() {
        let source = description::Camera {
            position: [0.0, 0.0, 5.0],
            look_at: [0.0, 0.0, 1.0],
            aperture_radius: 0.25,
            ..description::Camera::default()
        };
        let camera = lower_camera(&source);
        let lens = camera.lens.expect("a positive aperture is a lens");
        assert!((lens.aperture_radius - 0.25).abs() < 1e-6);
        assert!((lens.focus_distance - 4.0).abs() < 1e-6);

        let explicit = lower_camera(&description::Camera {
            focus_distance: Some(2.5),
            ..source.clone()
        });
        assert!((explicit.lens.expect("lens").focus_distance - 2.5).abs() < 1e-6);

        let pinhole = lower_camera(&description::Camera {
            aperture_radius: 0.0,
            focus_distance: Some(2.5),
            ..source
        });
        assert!(pinhole.lens.is_none());
    }

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "lowering passes authored closure constants through untouched"
    )]
    fn textured_slots_lower_to_indices_and_closure_params_carry() {
        use crate::material::TEXTURE_NONE;

        let source = description::Material {
            base_color: Texturable::Texture(TextureRef {
                path: "/wood.png".into(),
                color_space: None,
            }),
            geometry_normal: Some(TextureRef {
                path: "/weave.png".into(),
                color_space: None,
            }),
            coat_weight: 0.5,
            transmission_weight: 0.25,
            specular_ior: 1.8,
            geometry_thin_walled: true,
            geometry_opacity: Texturable::Constant(0.5),
            ..description::Material::default()
        };
        // The index map the collection pass would build for this material.
        let keys: Vec<texture::Key> = texture_keys(&source).collect();
        assert_eq!(keys.len(), 2);
        let indices: BTreeMap<&texture::Key, u32> = keys
            .iter()
            .enumerate()
            .map(|(index, key)| (key, index as u32))
            .collect();
        let lowered = lower_material("m", &source, false, &indices);
        // Textured slots resolve to their table index; the base-color
        // stand-in is the schema default (dead — the kernel replaces it);
        // every closure parameter reaches the GPU record as authored.
        assert_eq!(lowered.base_color_texture, indices[&keys[0]]);
        assert_eq!(lowered.normal_texture, indices[&keys[1]]);
        assert_eq!(lowered.emission_texture, TEXTURE_NONE);
        assert_eq!(lowered.opacity_texture, TEXTURE_NONE);
        assert_eq!(lowered.base_color, acescg_from_rec709(Vec3::splat(0.8)));
        assert_eq!(lowered.coat_weight, 0.5);
        assert_eq!(lowered.transmission_weight, 0.25);
        assert_eq!(lowered.specular_ior, 1.8);
        assert_eq!(lowered.thin_walled, 1);
        assert_eq!(lowered.opacity, 0.5);
    }

    /// The coat's tint on emission folds in at lowering — the one place
    /// both the light table and the shading kernel read from, so the two
    /// can't disagree about an emitter's radiance.
    #[test]
    fn emission_lowers_through_its_coat() {
        let source = description::Material {
            emission_luminance: 10.0,
            coat_weight: 1.0,
            coat_color: [0.5, 1.0, 1.0],
            ..description::Material::default()
        };
        let lowered = lower_material("m", &source, false, &BTreeMap::new());
        let expected =
            acescg_from_rec709(Vec3::ONE) * 10.0 * acescg_from_rec709(Vec3::new(0.5, 1.0, 1.0));
        assert!((lowered.emission - expected).length() < 1e-5);
    }

    #[test]
    fn missing_normals_derive_smooth_and_area_weighted() {
        let positions = [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::new(5.0, 5.0, 5.0)];
        let normals = smooth_normals(&positions, &[[0, 1, 2]]);
        // Every vertex of a single CCW triangle in the XY plane gets its
        // face normal +Z; the unreferenced vertex falls back finite.
        for normal in &normals[..3] {
            assert!((*normal - Vec3::Z).length() < 1e-6, "{normal}");
        }
        assert_eq!(normals[3], Vec3::Y);
    }

    #[test]
    #[expect(clippy::float_cmp, reason = "a black sky's power is exactly zero")]
    fn a_missing_environment_means_a_black_sky() {
        let host = host(&triangle_description()).expect("no environment is legal");
        let environment = host.environment.expect("fresh builds load one");
        assert_eq!(environment.tables().power, 0.0);
    }

    // — GPU-side: the incremental path must be indistinguishable from a
    // fresh build. —

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
