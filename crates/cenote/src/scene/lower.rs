//! Host lowering: the fallible half of prep. [`host_phase`] turns a
//! [`SceneDescription`] and its [`Dirty`] set into a [`HostScene`] — meshes
//! resolved and normals derived, textures collected and prepped in
//! bindless-index order, closure constants lowered to GPU records, emissive
//! geometry unpacked into per-triangle lights — validating against what the
//! renderer can currently express as it goes.
//!
//! This is where fallibility is concentrated so the untouched-on-error
//! contract stays cheap: everything that can fail on user data (file reads,
//! decodes, capability checks) happens here, before [`Scene::prep`] and
//! [`Scene::update`] make their first GPU call, so an [`Error::Scene`]
//! leaves residency untouched and a live session keeps its last good scene.
//! Anything not wired up yet — and anything legal but almost certainly a
//! scene bug (a textured material over a UV-less mesh) — is warned by name
//! and lowered anyway; only what has no honest render at all is an error.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

use glam::{Mat3, Mat4, Vec2, Vec3};

use super::changeset::{Dirty, Kind};
use super::description::{
    self, ColorSpace, Geometry, MeshSource, SceneDescription, Texturable, TextureRef,
};
use super::{Camera, Lens, Mesh, emissive_triangles, scene_error};
use crate::color::{acescg_from_rec709, luminance};
use crate::scene::environment::Environment;
use crate::error::{Error, Result};
use crate::scene::lights::{DeltaLight, TriangleLight};
use crate::scene::material::{Material, TEXTURE_NONE};
use crate::scene::source::texture;

/// Everything prep derives host-side before touching the GPU — the
/// fallible half, so a rejected description leaves residency untouched.
pub(super) struct HostScene {
    /// Geometry to (re)upload: the dirty subset, meshes resolved with
    /// their normals derived where absent and curve batches tessellated
    /// into tubes. Keyed by the reference an instance holds, so the two
    /// kinds share one residency map without sharing a namespace.
    pub(super) geometry: BTreeMap<Geometry, Mesh>,
    /// Geometry residency to retire. Processed before `meshes`, so a
    /// remove-then-recreate lands the new build.
    pub(super) removed_geometry: Vec<Geometry>,
    /// Every instance, in name order — custom index is position.
    pub(super) instances: Vec<InstanceSpec>,
    /// The emissive geometry, one entry per triangle of every emissive
    /// instance.
    pub(super) triangle_lights: Vec<TriangleLight>,
    /// The delta lights, lowered from the description's light objects.
    pub(super) delta_lights: Vec<DeltaLight>,
    /// Every texture the description references, in bindless-index order
    /// (`BTreeMap` iteration *is* the index assignment). Values are the
    /// prepped data to (re)upload — `None` keeps the resident image, whose
    /// content hash matched.
    pub(super) textures: BTreeMap<texture::Key, Option<texture::Prepared>>,
    /// Lowered when the environment changed (always, on a fresh build);
    /// `None` keeps the resident image and the scene-table constants.
    pub(super) environment: Option<EnvironmentSpec>,
    /// The camera, when it changed — a material edit must not snap the
    /// view back to the authored pose.
    pub(super) camera: Option<Camera>,
    /// What fills the open space between instances, or `None` for vacuum.
    /// Lowered on every pass: it is seven numbers, and it lives in the
    /// medium table, which rebuilds wholesale on any edit anyway.
    pub(super) global_medium: Option<super::Medium>,
    /// The TLAS must rebuild. Set by a mesh, instance, *or* material edit —
    /// material because fractional opacity bakes into each instance's
    /// non-opaque flag (see where it's assigned).
    pub(super) tlas_dirty: bool,
}

impl HostScene {
    /// Total records the light table will hold.
    pub(super) fn light_count(&self) -> u32 {
        (self.triangle_lights.len() + self.delta_lights.len()) as u32
    }

}

/// One *placement* lowered from the description — element `i` of an
/// instance's transforms array, so one description instance with N
/// placements lowers to N specs, and the flattened position is the TLAS
/// custom index everywhere downstream.
pub(super) struct InstanceSpec {
    pub(super) geometry: Geometry,
    pub(super) transform: Mat4,
    pub(super) material: Material,
    pub(super) camera_visible: bool,
    /// The medium this mesh bounds, already lowered — `None` for an
    /// ordinary surface. Some, and the surface is a null boundary: the
    /// medium fills it, and the material above is inert.
    pub(super) medium: Option<super::Medium>,
    /// Which solid wins where refractive interiors overlap — see
    /// [`description::Instance::interior_priority`].
    pub(super) priority: u32,
}

/// The environment lowered from the description: the image when it must be
/// (re)made resident, and the scene-table constants that always re-land —
/// so a tint or placement edit costs a table rebuild, never a decode.
pub(super) struct EnvironmentSpec {
    /// The decoded image; `None` keeps the resident one, whose source
    /// path matched.
    pub(super) image: Option<Arc<Environment>>,
    /// The image file the environment decodes from — the identity the
    /// keep-resident check compares. `None` for a constant sky.
    pub(super) source: Option<std::path::PathBuf>,
    /// `ACEScg` multiplier over the image's radiance, sampled-in by the
    /// kernel and folded into the selection power host-side.
    pub(super) tint: Vec3,
    /// The linear part of the environment-to-world placement — the sky is
    /// all directions, so the translation is dropped here.
    pub(super) to_world: Mat4,
    /// Its inverse: world directions into environment space.
    pub(super) from_world: Mat4,
}

/// Derive everything the GPU phase consumes, validating as it goes. Warns
/// only about objects `dirty` names, so a long edit session doesn't
/// repeat itself about parameters it already reported. `fresh` marks a
/// full build, which loads its environment even when no dirt names one —
/// a description without an environment object leaves nothing to mark.
/// `resident_textures` maps already-uploaded textures to their content
/// hashes, so an edit re-preps only textures a dirty material references —
/// and re-uploads only those whose source content actually changed.
/// `resident_environment` is the path the resident environment decoded
/// from, the same idea one image wide: an edit that leaves it alone (a
/// tint or placement change) keeps the decode and the upload.
pub(super) fn host_phase(
    description: &SceneDescription,
    dirty: &Dirty,
    fresh: bool,
    resident_textures: &BTreeMap<texture::Key, u64>,
    resident_environment: Option<&Path>,
) -> Result<HostScene> {
    let (_, camera_source) = singleton(description.cameras(), "camera")?;
    let (_, settings) = singleton(description.settings(), "settings")?;
    if description.environments().len() > 1 {
        return Err(scene_error(format!(
            "a scene renders at most one environment, this one has {}",
            description.environments().len()
        )));
    }

    let (geometry, removed_geometry) = resolve_geometry(description, dirty)?;

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
        // key.4 (the sample-time params) never reaches prep — the baked
        // image is transform-independent by design.
        let prepared = match resident_textures.get(&key).copied() {
            // Nothing resident: prep reads the file either way, so asking
            // first for the hash it would stamp would read it twice.
            None => Some(texture::prepare(&key.0, key.1, key.2, key.3)?),
            Some(_) if !touched => None,
            // A dirty material named it, so the source may have been
            // repainted: the expected hash costs a stat, and only a source
            // that actually moved gets read.
            Some(hash) => {
                if hash == texture::expected_hash(&key.0, key.1, key.2, key.3)? {
                    None
                } else {
                    let prepared = texture::prepare(&key.0, key.1, key.2, key.3)?;
                    (hash != prepared.hash).then_some(prepared)
                }
            }
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

    // Media lower once by name, however many instances share one — and a
    // heterogeneous medium's grid preps and parses here, in the fallible
    // half, so a bad file rejects the description before any GPU work.
    let mut media: BTreeMap<&str, super::Medium> = BTreeMap::new();
    for (name, medium) in description.media() {
        media.insert(name, lower_medium(name, medium)?);
    }

    let delta_lights = lower_delta_lights(description);
    let (instances, triangle_lights) =
        lower_instances(description, &materials, &media, &geometry)?;

    let touched = |kind: Kind| {
        dirty
            .changed
            .iter()
            .chain(&dirty.removed)
            .any(|(entry, _)| *entry == kind)
    };
    let camera = touched(Kind::Camera).then(|| lower_camera(camera_source));
    let environment = if fresh || touched(Kind::Environment) {
        Some(environment_spec(description, resident_environment)?)
    } else {
        None
    };

    let global_medium = global_medium(settings.global_medium.as_deref(), &media)?;

    Ok(HostScene {
        geometry,
        removed_geometry,
        instances,
        triangle_lights,
        delta_lights,
        textures,
        environment,
        camera,
        global_medium,
        // Material dirt rebuilds the TLAS too: fractional opacity is baked
        // into each instance's non-opaque flag, and the TLAS over a scene's
        // handful of instances is the cheap structure (every BLAS stays).
        tlas_dirty: touched(Kind::Mesh)
            || touched(Kind::Curves)
            || touched(Kind::Instance)
            || touched(Kind::Material),
    })
}

/// Resolve the geometry this round must (re)build, and name what it must
/// retire. Meshes and curve batches land in one map keyed by the
/// reference an instance holds: a tessellated groom is a mesh in every
/// way residency, the BLAS, and the light table can see, and keying by
/// the reference is what lets the two kinds share that map without
/// sharing a namespace.
fn resolve_geometry(
    description: &SceneDescription,
    dirty: &Dirty,
) -> Result<(BTreeMap<Geometry, Mesh>, Vec<Geometry>)> {
    let mut resolved = BTreeMap::new();
    let changed_meshes = names(&dirty.changed, Kind::Mesh);
    for (name, mesh) in description.meshes() {
        if changed_meshes.contains(name.as_str()) {
            resolved.insert(Geometry::Mesh(name.clone()), resolve_mesh(name, mesh)?);
        }
    }
    let changed_curves = names(&dirty.changed, Kind::Curves);
    for (name, curves) in description.curves() {
        if changed_curves.contains(name.as_str()) {
            resolved.insert(
                Geometry::Curves(name.clone()),
                super::curves::resolve(name, curves)?,
            );
        }
    }
    let removed = names(&dirty.removed, Kind::Mesh)
        .into_iter()
        .map(|name| Geometry::Mesh(name.to_owned()))
        .chain(
            names(&dirty.removed, Kind::Curves)
                .into_iter()
                .map(|name| Geometry::Curves(name.to_owned())),
        )
        .collect();
    Ok((resolved, removed))
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

/// Lower every instance into its placement specs — one per element of its
/// transforms array, flattened in name order so the enumerate position is
/// the TLAS custom index — unpacking each emissive element into its
/// per-triangle lights.
fn lower_instances(
    description: &SceneDescription,
    materials: &BTreeMap<&str, Material>,
    media: &BTreeMap<&str, super::Medium>,
    resolved: &BTreeMap<Geometry, Mesh>,
) -> Result<(Vec<InstanceSpec>, Vec<TriangleLight>)> {
    let mut instances = Vec::with_capacity(description.instances().len());
    let mut triangle_lights = Vec::new();
    for (name, instance) in description.instances() {
        // Apply validated the references and every transform, so lookups
        // can't miss and the inverses the records need exist.
        let authored = materials[instance.material.as_str()];
        let medium = instance
            .medium
            .as_ref()
            .map(|named| media[named.as_str()].clone());
        // A volume's boundary is crossed, never shaded, so nothing its
        // material describes can be honoured; `boundary_material` says which
        // parts have to be dropped rather than merely ignored, and this is
        // the only place an author hears that they were.
        let material = super::boundary_material(authored, medium.is_some());
        if medium.is_some() && !super::is_boundary_inert(&authored) {
            log::warn!(
                "instance \"{name}\" bounds a medium, so its material is inert — dropping its \
                 emission and opacity"
            );
        }
        warn_if_coarse_for_subsurface(name, instance, &material, resolved.get(&instance.geometry));
        // The geometry fetch is per instance, not per element: N emissive
        // placements share one resolve and pay only the N×T light records.
        let geometry = if luminance(material.emission) > 0.0 && !instance.transforms.is_empty() {
            Some(emissive_geometry(
                resolved.get(&instance.geometry),
                description,
                &instance.geometry,
            )?)
        } else {
            None
        };
        for transform in &instance.transforms {
            let index = instances.len() as u32;
            let mut transform = transform.to_mat4();
            // A heterogeneous medium's shell is the canonical unit cube:
            // the grid's dilated active bounds bake into each placement's
            // transform, so every volume shares one mesh (and one BLAS)
            // while the author's transform still places the asset.
            if let Some(volume) = medium.as_ref().and_then(|m| m.volume.as_ref()) {
                transform *= volume.bounds_to_asset;
            }
            if let Some((positions, triangles)) = &geometry {
                triangle_lights.extend(emissive_triangles(
                    positions,
                    triangles,
                    transform,
                    material.emission,
                    index,
                ));
            }
            instances.push(InstanceSpec {
                geometry: instance.geometry.clone(),
                transform,
                material,
                camera_visible: instance.camera_visible,
                medium: medium.clone(),
                priority: instance.interior_priority,
            });
        }
    }
    Ok((instances, triangle_lights))
}

/// Lower an authoring-side material onto the GPU record: color constants
/// convert from the format's linear `Rec.709` into `ACEScg` — prep owns
/// that conversion, and textures make the same trip in-shader after the
/// hardware's sRGB decode — and weights clamp into the ranges the kernel's
/// lerps assume. Textured slots resolve to bindless indices through
/// `indices`; their constants lower to stand-ins (the schema default for
/// slots the kernel replaces per hit, the identity for those it
/// multiplies). The point-of-use comments below carry the per-field why.
fn lower_material(
    name: &str,
    source: &description::Material,
    warn: bool,
    indices: &BTreeMap<&texture::Key, u32>,
) -> Material {
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
    // Clamped at zero on the far side of the gamut like a medium's
    // coefficients (`lower_medium`): a negative σ_s would be scattering
    // that removes light. The anisotropy clamps at the medium table.
    material.transmission_scatter =
        acescg_from_rec709(Vec3::from(source.transmission_scatter)).max(Vec3::ZERO);
    material.transmission_scatter_anisotropy = source.transmission_scatter_anisotropy;
    // Stand-ins for the textured five: the schema default everywhere but
    // the weight, which must stand in as *present*. Prep reads it to decide
    // whether to intern the interior at all, so a 0 there would gate the
    // medium away before any texel could ask for it.
    material.subsurface_weight = constant_or(&source.subsurface_weight, 1.0).clamp(0.0, 1.0);
    // An albedo: the inversion's fit is only defined on [0, 1], like the
    // transmittance above.
    material.subsurface_color =
        acescg_from_rec709(Vec3::from(constant_or(&source.subsurface_color, [0.8; 3])))
            .clamp(Vec3::ZERO, Vec3::ONE);
    material.subsurface_radius = constant_or(&source.subsurface_radius, 1.0).max(0.0);
    // Lengths, not colors — no working-space conversion.
    material.subsurface_radius_scale =
        Vec3::from(constant_or(&source.subsurface_radius_scale, [1.0, 0.5, 0.25])).max(Vec3::ZERO);
    material.subsurface_scatter_anisotropy =
        constant_or(&source.subsurface_scatter_anisotropy, 0.0);
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
    // Scatter without a filled interior derives no medium at all — milk
    // authored at transmission_depth 0 renders as clear glass — and every
    // downstream consumer reads the derived record, so this is the only
    // place an author can hear that it vanished.
    if warn
        && material.transmission_scatter.cmpgt(Vec3::ZERO).any()
        && super::interior(&material).is_none()
    {
        log::warn!(
            "material \"{name}\": transmission_scatter is ignored — a scattering interior \
             needs transmission_weight > 0, a positive transmission_depth, and no thin walls"
        );
    }
    // Every textured slot resolves to its bindless index through the shared
    // `textured_slots` list — the same list, in the same order, the
    // collection pass keyed on, so a slot can't be collected under one
    // usage and looked up under another (the lookup would panic). The
    // collection pass walked every reference already, so these can't miss.
    // The array pattern pins the count: a new slot won't compile until it
    // is assigned a field here.
    let [
        base_color_texture,
        metalness_texture,
        specular_roughness_texture,
        emission_texture,
        opacity_texture,
        normal_texture,
        subsurface_weight_texture,
        subsurface_color_texture,
        subsurface_radius_texture,
        subsurface_radius_scale_texture,
        subsurface_scatter_anisotropy_texture,
    ] = textured_slots(source).map(|(reference, usage)| {
        reference.map_or(TEXTURE_NONE, |reference| {
            indices[&texture_key(reference, usage)]
        })
    });
    material.base_color_texture = base_color_texture;
    material.metalness_texture = metalness_texture;
    material.specular_roughness_texture = specular_roughness_texture;
    material.emission_texture = emission_texture;
    material.opacity_texture = opacity_texture;
    material.normal_texture = normal_texture;
    material.subsurface_weight_texture = subsurface_weight_texture;
    material.subsurface_color_texture = subsurface_color_texture;
    material.subsurface_radius_texture = subsurface_radius_texture;
    material.subsurface_radius_scale_texture = subsurface_radius_scale_texture;
    material.subsurface_scatter_anisotropy_texture = subsurface_scatter_anisotropy_texture;
    material
}

/// A texturable slot's constant, or `stand_in` when it is textured — the
/// schema default for slots the kernel replaces per hit, the identity for
/// slots it multiplies (emission, opacity), and once neither (see
/// `subsurface_weight` above).
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
        texture::Usage::Color | texture::Usage::Scalar | texture::Usage::Vector => {
            reference.color_space.map(|space| space == ColorSpace::Srgb)
        }
    };
    let channel = match usage {
        // Only the scalar bake reads a chosen channel; a stray selector
        // on a color, vector or normal slot must not fork the cache.
        texture::Usage::Color | texture::Usage::Vector | texture::Usage::Normal => {
            texture::Channel::R
        }
        texture::Usage::Scalar => match reference.channel {
            None | Some(description::Channel::R) => texture::Channel::R,
            Some(description::Channel::G) => texture::Channel::G,
            Some(description::Channel::B) => texture::Channel::B,
            Some(description::Channel::A) => texture::Channel::A,
        },
    };
    let scale = match usage {
        // A normal is a direction, not a quantity — the multiplier must
        // not fork the cache (the schema documents the slot ignores it).
        texture::Usage::Normal => None,
        texture::Usage::Color | texture::Usage::Scalar | texture::Usage::Vector => reference.scale,
    };
    let params = texture::Params::new(
        reference
            .uv
            .map(|transform| (transform.scale, transform.offset)),
        scale,
    );
    (reference.path.clone(), usage, srgb, channel, params)
}

/// A material's textured slots paired with the texture usage each feeds,
/// in a fixed order — the one list [`texture_keys`] (collection) and
/// [`lower_material`] (lowering) share, so a slot can never be collected
/// under one usage and lowered under another. `description`'s own
/// `Material::textures` walks the same eleven slots for validation.
fn textured_slots(material: &description::Material) -> [(Option<&TextureRef>, texture::Usage); 11] {
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
        (material.subsurface_weight.texture(), texture::Usage::Scalar),
        (material.subsurface_color.texture(), texture::Usage::Color),
        (material.subsurface_radius.texture(), texture::Usage::Scalar),
        // A per-channel length, not a color — `Usage::Vector`, which bakes
        // like a color and samples like data.
        (
            material.subsurface_radius_scale.texture(),
            texture::Usage::Vector,
        ),
        (
            material.subsurface_scatter_anisotropy.texture(),
            texture::Usage::Scalar,
        ),
    ]
}

/// Every prep request a material makes, one per textured slot.
fn texture_keys(material: &description::Material) -> impl Iterator<Item = texture::Key> {
    textured_slots(material)
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
        let has_uvs = match &instance.geometry {
            // A tessellated strand always carries its own coordinates:
            // root-to-tip along `u`, one random value per strand on `v`.
            Geometry::Curves(_) => true,
            Geometry::Mesh(mesh) => match &description.meshes()[mesh].source {
                MeshSource::Inline { uvs, .. } => uvs.is_some(),
                // This warning reads the description only; whether a PLY
                // file carries UVs is known after resolution, so a UV-less
                // one gets the benefit of the doubt (its lookups still
                // read texel (0, 0)). A medium shell never shades, so its
                // zero UVs warn nothing.
                MeshSource::Ply { .. } | MeshSource::MediumBounds => true,
            },
        };
        let touched = |kind: Kind, target: &str| dirty.changed.contains(&(kind, target.to_owned()));
        if !has_uvs
            && (touched(Kind::Instance, name)
                || touched(Kind::Material, &instance.material)
                || touched(instance.geometry.kind(), instance.geometry.name()))
        {
            log::warn!(
                "instance \"{name}\": material \"{}\" is textured but {} has \
                 no UVs — every lookup reads texel (0, 0)",
                instance.material,
                instance.geometry
            );
        }
    }
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
        // The heterogeneous shell: the canonical unit cube, outward-wound.
        // The grid's bounds bake into each placement's transform (see
        // `lower_instances`), so one mesh — one BLAS — serves every volume.
        // A null boundary never shades, so only the winding matters; the
        // shared-corner normals below are never read.
        MeshSource::MediumBounds => {
            let positions: Vec<Vec3> = (0..8)
                .map(|corner| {
                    Vec3::new(
                        (corner & 1) as f32,
                        ((corner >> 1) & 1) as f32,
                        ((corner >> 2) & 1) as f32,
                    )
                })
                .collect();
            let triangles = vec![
                [0, 2, 3],
                [0, 3, 1], // −Z
                [4, 5, 7],
                [4, 7, 6], // +Z
                [0, 1, 5],
                [0, 5, 4], // −Y
                [2, 6, 7],
                [2, 7, 3], // +Y
                [0, 4, 6],
                [0, 6, 2], // −X
                [1, 3, 7],
                [1, 7, 5], // +X
            ];
            let normals = smooth_normals(&positions, &triangles);
            let uvs = vec![Vec2::ZERO; positions.len()];
            Ok(Mesh {
                positions,
                normals,
                uvs,
                triangles,
            })
        }
        MeshSource::Ply { path } => {
            let ply = crate::scene::source::ply::read(path).map_err(|error| match error {
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

/// Warn when an instance's geometry is too coarse for the medium behind
/// it: at free paths near the facet size a subsurface walk resolves the
/// polyhedron instead of the shape it stands for, and convex creases read
/// as bright seams.
///
/// Silent on a mesh this round did not resolve, rather than re-reading the
/// PLY the way [`emissive_geometry`] does. The subsurface instance is
/// usually the scene's subject, so a dragged `subsurface_radius` would
/// re-read and re-normalise it per frame to produce a log line; a full
/// build resolves everything, so a scene still hears this when it loads.
///
/// The angle is reported, not a share of area above the bar, because only
/// the angle measures progress. The corpus head reads 61% of its area
/// coarse; subdividing it twice — which removes the sawtooth and brings
/// the frame within 0.7% of pbrt in every channel — reads *70%*, because
/// the flat caps Catmull-Clark rounds cross into the count as fast as the
/// lean falls. The same two levels take the median lean 5.2° → 1.9°.
fn warn_if_coarse_for_subsurface(
    name: &str,
    instance: &description::Instance,
    material: &Material,
    mesh: Option<&Mesh>,
) {
    let Some(mesh) = mesh else { return };
    // A mapped free path is not the constant beside it, and is not
    // unpacked here to find out. Colour and anisotropy maps move neither.
    if material.subsurface_radius_texture != TEXTURE_NONE
        || material.subsurface_radius_scale_texture != TEXTURE_NONE
    {
        return;
    }
    // Asked of the interior the walk will march rather than of the slots
    // behind it, so the two cannot disagree about whether the lobe is on.
    let Some(medium) = super::subsurface(material) else {
        return;
    };
    let mfp = 1.0 / medium.sigma_t.into_iter().fold(0.0f32, f32::max);
    // Object-space edges against a world-space free path. The largest
    // scale the instance stands at makes its facets widest, so it is the
    // worst case among them; an instance placed nowhere divides by zero
    // and stays silent, which is all it deserves.
    let scale = instance
        .transforms
        .iter()
        .flat_map(|transform| {
            let placement = transform.to_mat4();
            [placement.x_axis, placement.y_axis, placement.z_axis]
                .map(|axis| axis.truncate().length())
        })
        .fold(0.0f32, f32::max);
    let lean = facet_lean_degrees(mesh, mfp / scale);
    if lean <= COARSE_FACET_DEGREES {
        return;
    }
    log::warn!(
        "instance \"{name}\" is coarse for its subsurface medium: half its area leans {lean:.1}° \
         off the surface its shading normals describe, over the {COARSE_FACET_DEGREES}° that its \
         {mfp:.3} mm free path resolves — the walk will read the facets as a sawtooth, so \
         subdivide until this angle falls under the bar",
        mfp = mfp * 1000.0,
    );
}

/// How far a facet may lean off the surface its shading normals describe
/// before a subsurface walk resolves it. Measured, not chosen: on a
/// subdivision ladder over an icosphere this is where crease excess falls
/// to 0.2% over the noise floor, and nothing is visible below it.
const COARSE_FACET_DEGREES: f32 = 1.25;

/// Lean angles are accumulated into fixed bins of this width, in degrees,
/// rather than sorted: this runs per instance per build, over meshes with
/// millions of facets. The reported median is the bin's lower edge, so it
/// never overstates.
const LEAN_BIN_DEGREES: f32 = 0.05;

/// The area-weighted median angle a facet of `mesh` leans off the surface
/// its shading normals describe, counting anything narrower than the
/// object-space free path `mfp` as flat: the walk steps over such a facet,
/// so its lean is never resolved. Zeroing them rather than dropping them
/// is what keeps the median a statement about the whole surface.
///
/// Lean is measured against the corner shading normals rather than across
/// an edge to the neighbouring face. It is the same angle wherever a
/// mesh's normals describe its shape, it needs no adjacency, and it reads
/// zero on geometry that means to be flat — which a dihedral does not.
///
/// A median, so a steep minority cannot carry it: the teapot's lid, whose
/// normals were averaged into the wall beside it, stands at 77° over 30.8%
/// of its area and says nothing about how well the shape is resolved.
fn facet_lean_degrees(mesh: &Mesh, mfp: f32) -> f32 {
    const LEAN_BINS: usize = (90.0 / LEAN_BIN_DEGREES) as usize;

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the quotient is clamped into the bin range before it is cast"
    )]
    fn bin(degrees: f32) -> usize {
        (degrees / LEAN_BIN_DEGREES).clamp(0.0, (LEAN_BINS - 1) as f32) as usize
    }

    let mut area_by_lean = [0.0f32; LEAN_BINS];
    let mut total = 0.0;
    for &[a, b, c] in &mesh.triangles {
        let (pa, pb, pc) = (
            mesh.positions[a as usize],
            mesh.positions[b as usize],
            mesh.positions[c as usize],
        );
        let (ab, bc, ca) = (pb - pa, pc - pb, pa - pc);
        let cross = ab.cross(bc);
        let plane = cross.length_squared();
        let normals = [a, b, c].map(|corner| mesh.normals[corner as usize]);
        // A shading normal with no direction is no evidence either way —
        // the head carries three. A degenerate triangle needs no such
        // guard: its area is zero, so it weighs nothing.
        if normals.iter().any(|n| !n.is_finite() || n.length_squared() <= 0.0) {
            continue;
        }
        let widest = ab.length_squared().max(bc.length_squared()).max(ca.length_squared());
        let lean = if widest > mfp * mfp {
            // The steepest corner, as cos² — one arc cosine per facet
            // rather than three, and the sign the square drops would only
            // matter for a normal pointing into its own face.
            let steepest = normals.iter().fold(1.0f32, |steepest, normal| {
                let aligned = cross.dot(*normal);
                steepest.min(aligned * aligned / (plane * normal.length_squared()))
            });
            steepest.clamp(0.0, 1.0).sqrt().acos().to_degrees()
        } else {
            0.0
        };
        let area = 0.5 * plane.sqrt();
        area_by_lean[bin(lean)] += area;
        total += area;
    }
    if total <= 0.0 {
        return 0.0;
    }
    let mut below = 0.0;
    for (index, area) in area_by_lean.iter().enumerate() {
        below += area;
        if below >= 0.5 * total {
            #[expect(
                clippy::cast_precision_loss,
                reason = "the bin index is far below f32's integer range"
            )]
            return index as f32 * LEAN_BIN_DEGREES;
        }
    }
    90.0
}

/// Some geometry's positions and triangles for the light table. The
/// resolved copy serves when this round already built it; otherwise
/// inline geometry converts from the description, a PLY reference
/// re-reads its file, and a curve batch re-tessellates — the cost falls
/// on an *emissive* object when a non-geometry edit rebuilds the lights,
/// which is rare enough (emitters are almost always simple quads, and
/// glowing hair rarer still) to not be worth a host-side geometry cache.
fn emissive_geometry(
    resolved: Option<&Mesh>,
    description: &SceneDescription,
    geometry: &Geometry,
) -> Result<(Vec<Vec3>, Vec<[u32; 3]>)> {
    if let Some(mesh) = resolved {
        return Ok((mesh.positions.clone(), mesh.triangles.clone()));
    }
    match geometry {
        Geometry::Curves(name) => {
            let tessellated = super::curves::resolve(name, &description.curves()[name])?;
            Ok((tessellated.positions, tessellated.triangles))
        }
        Geometry::Mesh(name) => match &description.meshes()[name].source {
            MeshSource::Inline {
                positions,
                triangles,
                ..
            } => Ok((
                positions.iter().copied().map(Vec3::from).collect(),
                triangles.clone(),
            )),
            MeshSource::Ply { path } => {
                let ply = crate::scene::source::ply::read(path)?;
                Ok((ply.positions, ply.triangles))
            }
            // A medium shell's material is forced inert, so it never emits
            // and this is never reached with one — but the resolver is total.
            MeshSource::MediumBounds => Ok((Vec::new(), Vec::new())),
        },
    }
}

/// The description's one environment, lowered — or the black sky a
/// description without one renders under: zero power, so next-event
/// estimation puts all its draws on the light list.
fn environment_spec(
    description: &SceneDescription,
    resident_environment: Option<&Path>,
) -> Result<EnvironmentSpec> {
    match description.environments().iter().next() {
        Some((name, environment)) => lower_environment(name, environment, resident_environment),
        None => Ok(EnvironmentSpec {
            image: Some(Arc::new(Environment::constant(Vec3::ZERO))),
            source: None,
            tint: Vec3::ONE,
            to_world: Mat4::IDENTITY,
            from_world: Mat4::IDENTITY,
        }),
    }
}

/// Resolve and lower the medium a description's settings name as the
/// atmosphere. The reference is validated at apply, but `replace` swaps a
/// whole description in without one — so it is checked here too, where every
/// other failure on user data is caught.
fn global_medium(
    name: Option<&str>,
    media: &BTreeMap<&str, super::Medium>,
) -> Result<Option<super::Medium>> {
    match name {
        Some(name) => {
            let lowered = media
                .get(name)
                .ok_or_else(|| scene_error(format!("the global medium \"{name}\" does not exist")))?
                .clone();
            // Only an unbounded medium is affected: it extinguishes over
            // an infinite path, so with nothing scattering light back
            // there is no route to the sky at all. A bounded volume that
            // only absorbs is an ordinary dark solid.
            if lowered.scattering == Vec3::ZERO && lowered.absorption != Vec3::ZERO {
                log::warn!(
                    "the global medium \"{name}\" only absorbs, which leaves the environment \
                     and the distant lights unreachable"
                );
            }
            Ok(Some(lowered))
        }
        None => Ok(None),
    }
}

/// Lower one description medium into the renderer's coefficients — and,
/// for a heterogeneous one, resolve its grid. The multi-GiB payload is
/// deliberately not read here: the GPU phase streams it into the pool.
///
/// The coefficients convert to the working space like every other authored
/// color, and clamp at zero on the far side: the `Rec.709` → `ACEScg` matrix
/// has negative entries, so a saturated authored coefficient can land below
/// zero — which would be a medium that *amplifies*, and one path's runaway
/// throughput reaches the whole image through next-event estimation.
fn lower_medium(name: &str, medium: &description::Medium) -> Result<super::Medium> {
    let convert = |what: &str, authored: [f32; 3]| {
        let converted = crate::color::acescg_from_rec709(Vec3::from(authored));
        if converted.min_element() < 0.0 {
            log::warn!(
                "medium \"{name}\": {what} {authored:?} is outside the working space's gamut; \
                 clamping the negative channels to zero"
            );
        }
        converted.max(Vec3::ZERO)
    };
    if !(-super::MAX_ANISOTROPY..=super::MAX_ANISOTROPY).contains(&medium.anisotropy) {
        log::warn!(
            "medium \"{name}\": anisotropy {} will clamp to ±{}",
            medium.anisotropy,
            super::MAX_ANISOTROPY
        );
    }
    let volume = medium
        .volume
        .as_ref()
        .map(|volume| lower_volume(volume, convert("emission", volume.emission)))
        .transpose()?;
    Ok(super::Medium {
        absorption: convert("absorption", medium.absorption),
        scattering: convert("scattering", medium.scattering),
        anisotropy: medium.anisotropy,
        volume,
    })
}

/// Resolve one grid reference into a [`super::GridVolume`]: prep the
/// `.vdb` if needed (content-cached), parse the header, and derive the
/// shell transform from the active bounds.
///
/// A temperature field is resolved here too, and refused unless it maps
/// index space the way the density grid does. Everything downstream — the
/// shell, the majorant lattice, the one world→index affine a record
/// carries — is derived from the density grid alone, so a temperature
/// field on a different lattice would be sampled at the wrong place; a
/// solver's fields share a transform, and the ones that do not are worth
/// stopping for.
/// Whether two grids' world→index affines agree closely enough to be read
/// through one of them. Compared to a tolerance rather than bit for bit:
/// the rows arrive as `f32` from the file, and two grids that a solver
/// wrote from the same transform can still differ in the last place after
/// a resample or a round trip through another DCC. A relative disagreement
/// this small is worth far less than a voxel; a real mismatch is orders
/// above it and still refused.
fn same_index_map(left: &[[f32; 4]; 3], right: &[[f32; 4]; 3]) -> bool {
    // Row-relative, so the translation column is judged against the row's
    // own magnitude rather than against a scale it has no relation to.
    left.iter().zip(right).all(|(a, b)| {
        let scale = a.iter().chain(b).fold(0.0_f32, |m, v| m.max(v.abs()));
        a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1e-5 * scale)
    })
}

fn lower_volume(source: &description::VolumeSource, emission: Vec3) -> Result<super::GridVolume> {
    let nvdb = crate::scene::source::vdb::prepared(&source.path)?;
    let header = crate::scene::source::vdb::grid_header(&nvdb, &source.grid)?;
    let mut temperature_grid = source.temperature_grid.clone();
    if let Some(name) = &temperature_grid {
        let temperature = crate::scene::source::vdb::grid_header(&nvdb, name)?;
        if !same_index_map(&temperature.asset_to_index, &header.asset_to_index) {
            return Err(super::scene_error(format!(
                "volume \"{}\": temperature grid \"{name}\" maps index space differently from \
                 density grid \"{}\" ({:?} vs {:?}); one simulation's fields share a transform",
                nvdb.display(),
                source.grid,
                temperature.asset_to_index,
                header.asset_to_index
            )));
        }
        // Named but multiplied by nothing: the field is dropped here rather
        // than carried, so a fire turned down to black stops paying for a
        // second grid — its upload, its residency, and the emissive volume
        // stage all key off this being `Some`. Checked *after* the pair is
        // validated, so a broken file is refused whatever the scale says.
        if emission.cmple(Vec3::ZERO).all() {
            temperature_grid = None;
        }
    }
    let (lo, hi) = crate::scene::source::vdb::shell_box(&header.meta.index_bbox);
    let (lo, hi) = (Vec3::from(lo), Vec3::from(hi));
    // The unit cube maps onto the shell box, and so does the majorant
    // lattice — `res` cells across it, which is what turns an index-space
    // position into the cell coordinate the tracker's walk steps through.
    let bounds_to_index = Mat4::from_translation(lo) * Mat4::from_scale(hi - lo);
    let bounds_to_asset = super::rows_to_mat4(&header.index_to_asset) * bounds_to_index;
    let res = crate::scene::source::vdb::majorant_resolution(&header.meta.index_bbox);
    let majorant_scale = Vec3::new(res[0] as f32, res[1] as f32, res[2] as f32) / (hi - lo);
    Ok(super::GridVolume {
        nvdb,
        grid: source.grid.clone(),
        asset_to_index: header.asset_to_index,
        bounds_to_asset,
        majorant_res: res,
        majorant_scale: majorant_scale.to_array(),
        majorant_bias: (-lo * majorant_scale).to_array(),
        temperature: temperature_grid,
        kelvin_scale: source.temperature_scale,
        kelvin_offset: source.temperature_offset,
        emission,
    })
}

/// Lower one description environment: the tint converts to `ACEScg` and
/// sanitizes the way material colors do, the placement drops to its linear
/// part (apply validated invertibility, so the inverse exists), and the
/// image loads only when the source path actually changed — a pathless
/// environment is the constant white sky, colored by the tint alone.
fn lower_environment(
    name: &str,
    environment: &description::Environment,
    resident: Option<&Path>,
) -> Result<EnvironmentSpec> {
    let image = match &environment.path {
        Some(path) if Some(path.as_path()) == resident => None,
        Some(path) => Some(load_environment(name, path)?),
        None => Some(Arc::new(Environment::constant(Vec3::ONE))),
    };
    let to_world = Mat4::from_mat3(Mat3::from_mat4(environment.transform.to_mat4()));
    Ok(EnvironmentSpec {
        image,
        source: environment.path.clone(),
        tint: acescg_from_rec709(Vec3::from(environment.tint)).max(Vec3::ZERO),
        to_world,
        from_world: to_world.inverse(),
    })
}

/// Read and decode an environment image. Failures are [`Error::Scene`] —
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
    // The format comes from the bytes, not the extension: EXR opens with
    // its magic number, Radiance HDR with "#?" (usually "#?RADIANCE").
    let environment = if bytes.starts_with(&[0x76, 0x2f, 0x31, 0x01]) {
        Environment::from_equirect_exr(&bytes)
    } else if bytes.starts_with(b"#?") {
        Environment::from_equirect_hdr(&bytes)
    } else {
        return Err(scene_error(format!(
            "environment \"{name}\": \"{}\" is neither an EXR nor a Radiance HDR",
            path.display()
        )));
    }
    .map_err(|error| {
        scene_error(format!(
            "environment \"{name}\": \"{}\" doesn't decode: {error}",
            path.display()
        ))
    })?;
    Ok(Arc::new(environment))
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
pub(super) fn all_dirty(description: &SceneDescription) -> Dirty {
    fn mark<T>(dirty: &mut Dirty, kind: Kind, map: &BTreeMap<String, T>) {
        for name in map.keys() {
            dirty.changed.insert((kind, name.clone()));
        }
    }
    let mut dirty = Dirty::default();
    mark(&mut dirty, Kind::Mesh, description.meshes());
    mark(&mut dirty, Kind::Curves, description.curves());
    mark(&mut dirty, Kind::Instance, description.instances());
    mark(&mut dirty, Kind::Material, description.materials());
    mark(&mut dirty, Kind::Medium, description.media());
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
        host_phase(
            description,
            &all_dirty(description),
            true,
            &BTreeMap::new(),
            None,
        )
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

    /// The empty scene is renderable, not rejected: the render server
    /// stands up its session on camera + settings alone, and a live edit
    /// may delete the last instance. The host phase lowers it to zero
    /// instances, zero lights, and the black default sky.
    #[test]
    fn a_scene_without_instances_lowers_empty() {
        let mut description = SceneDescription::new();
        description
            .apply(&ChangeSet {
                ops: vec![
                    Op::Settings(SettingsPatch::new("main")),
                    Op::Camera(CameraPatch::new("main")),
                ],
            })
            .expect("valid data");
        let host = host(&description).expect("an empty scene lowers");
        assert!(host.instances.is_empty());
        assert_eq!(host.light_count(), 0);
    }

    /// One description instance with N placements lowers to N specs — the
    /// flattened position is the TLAS custom index — and an emissive one
    /// unpacks into per-element lights under per-element identity names.
    #[test]
    fn transforms_arrays_flatten_into_per_element_specs_and_lights() {
        let mut description = triangle_description();
        description
            .apply(&ChangeSet {
                ops: vec![
                    Op::Material(Box::new(MaterialPatch {
                        emission_luminance: Some(5.0),
                        ..MaterialPatch::new("gray")
                    })),
                    Op::Instance(InstancePatch {
                        transforms: Some(vec![
                            description::Transform::Trs {
                                translate: [1.0, 0.0, 0.0],
                                rotate_degrees: [0.0; 3],
                                scale: [1.0; 3],
                            },
                            description::Transform::Trs {
                                translate: [-1.0, 0.0, 0.0],
                                rotate_degrees: [0.0; 3],
                                scale: [1.0; 3],
                            },
                        ]),
                        ..InstancePatch::new("thing")
                    }),
                ],
            })
            .expect("valid data");
        let host = host(&description).expect("an instanced emitter lowers");
        assert_eq!(host.instances.len(), 2);
        // One triangle × two placements: a light record each, tied to its
        // own flattened index, its corners under its own placement.
        assert_eq!(host.triangle_lights.len(), 2);
        assert_eq!(host.triangle_lights[0].instance, 0);
        assert_eq!(host.triangle_lights[1].instance, 1);
        assert!((host.triangle_lights[0].corners[0].x - 1.0).abs() < 1e-6);
        assert!((host.triangle_lights[1].corners[0].x + 1.0).abs() < 1e-6);
    }

    /// The empty placements array is resident but places nothing: no
    /// specs, no lights, no error.
    #[test]
    fn an_empty_transforms_array_lowers_to_nothing() {
        let mut description = triangle_description();
        description
            .apply(&ChangeSet {
                ops: vec![Op::Instance(InstancePatch {
                    transforms: Some(vec![]),
                    ..InstancePatch::new("thing")
                })],
            })
            .expect("the empty array is legal");
        let host = host(&description).expect("a placement-less instance lowers");
        assert!(host.instances.is_empty());
        assert_eq!(host.light_count(), 0);
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
                        channel: None,
                        scale: None,
                        uv: None,
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
        let mesh = &host.geometry[&Geometry::Mesh("tri".to_owned())];
        assert_eq!(mesh.positions.len(), 4);
        assert_eq!(mesh.triangles.len(), 2);
        // No authored normals: derived, and this quad's winding faces +Z.
        assert!(mesh.normals.iter().all(|n| n.abs_diff_eq(Vec3::Z, 1e-6)));
        // Authored (1, 1); the reader flips v into sampler storage order.
        assert_eq!(mesh.uvs[2], Vec2::new(1.0, 0.0));
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
        assert!(error.to_string().contains("PLY"), "{error}");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_unreadable_environment_is_rejected_at_prep() {
        let mut description = triangle_description();
        description
            .apply(&ChangeSet {
                ops: vec![Op::Environment(EnvironmentPatch {
                    // Exists (so apply accepts it) but is no radiance image.
                    path: Some(Some(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml").into())),
                    ..EnvironmentPatch::new("sky")
                })],
            })
            .expect("valid data");
        let error = host_error(&description);
        assert!(
            error.to_string().contains("neither an EXR nor a Radiance HDR"),
            "{error}"
        );
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

    /// The channel forks a scalar key — two masks packed into one image
    /// prep separately — and never a color or normal key, where a stray
    /// selector is inert.
    #[test]
    fn texture_keys_fork_on_scalar_channels_only() {
        let with_channel = |channel| TextureRef {
            path: "/orm.png".into(),
            color_space: None,
            channel,
            scale: None,
            uv: None,
        };
        let scalar = |channel| texture_key(&with_channel(channel), texture::Usage::Scalar);
        assert_ne!(
            scalar(Some(description::Channel::G)),
            scalar(Some(description::Channel::B))
        );
        // Absent and explicit red are the same prep.
        assert_eq!(scalar(None), scalar(Some(description::Channel::R)));
        for usage in [texture::Usage::Color, texture::Usage::Normal] {
            assert_eq!(
                texture_key(&with_channel(Some(description::Channel::A)), usage),
                texture_key(&with_channel(None), usage),
            );
        }
    }

    /// Sample-time parameters fork the key — two transforms of one image
    /// are two bindless slots, since the params record rides the index —
    /// while a normal slot's stray value scale is inert (the schema says
    /// the slot ignores it) and the identity spellings collapse.
    #[test]
    fn texture_keys_fork_on_sample_time_params() {
        let with = |scale, uv| TextureRef {
            path: "/wood.png".into(),
            color_space: None,
            channel: None,
            scale,
            uv,
        };
        let color = |scale, uv| texture_key(&with(scale, uv), texture::Usage::Color);
        let remap = description::UvTransform {
            scale: [2.0, 2.0],
            offset: [0.0; 2],
        };
        assert_ne!(color(None, None), color(Some(3.0), None));
        assert_ne!(color(None, None), color(None, Some(remap)));
        // Explicit identity and absent are one key.
        assert_eq!(color(None, None), color(Some(1.0), None));
        assert_eq!(
            color(None, None),
            color(None, Some(description::UvTransform::default()))
        );
        assert_eq!(
            texture_key(&with(Some(3.0), None), texture::Usage::Normal),
            texture_key(&with(None, None), texture::Usage::Normal),
        );
    }

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "lowering passes authored closure constants through untouched"
    )]
    fn textured_slots_lower_to_indices_and_closure_params_carry() {
        use crate::scene::material::TEXTURE_NONE;

        let source = description::Material {
            base_color: Texturable::Texture(TextureRef {
                path: "/wood.png".into(),
                color_space: None,
                channel: None,
                scale: None,
                uv: None,
            }),
            geometry_normal: Some(TextureRef {
                path: "/weave.png".into(),
                color_space: None,
                channel: None,
                scale: None,
                uv: None,
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

    /// All five subsurface slots reach the record as live indices *and* the
    /// interior survives the lowering — the way this feature fails
    /// silently. A textured weight standing in at the schema default would
    /// gate `scene::subsurface` off, and the walk would have no interior to
    /// enter however bright the texel turned out to be.
    #[test]
    fn every_subsurface_slot_textures_without_gating_its_interior() {
        use crate::scene::material::TEXTURE_NONE;

        fn map<T>(path: &str) -> Texturable<T> {
            Texturable::Texture(TextureRef {
                path: path.into(),
                color_space: None,
                channel: None,
                scale: None,
                uv: None,
            })
        }
        let source = description::Material {
            subsurface_weight: map("/mask.png"),
            subsurface_color: map("/albedo.png"),
            subsurface_radius: map("/depth.png"),
            subsurface_radius_scale: map("/shape.png"),
            subsurface_scatter_anisotropy: map("/aniso.png"),
            ..description::Material::default()
        };
        let keys: Vec<texture::Key> = texture_keys(&source).collect();
        assert_eq!(keys.len(), 5);
        let indices: BTreeMap<&texture::Key, u32> = keys
            .iter()
            .enumerate()
            .map(|(index, key)| (key, index as u32))
            .collect();
        let lowered = lower_material("skin", &source, false, &indices);
        for slot in [
            lowered.subsurface_weight_texture,
            lowered.subsurface_color_texture,
            lowered.subsurface_radius_texture,
            lowered.subsurface_radius_scale_texture,
            lowered.subsurface_scatter_anisotropy_texture,
        ] {
            assert_ne!(slot, TEXTURE_NONE);
        }
        // Five distinct images, five distinct slots — a slot collected
        // under one usage and looked up under another would collide here
        // rather than at a hit.
        let resolved: BTreeSet<u32> = [
            lowered.subsurface_weight_texture,
            lowered.subsurface_color_texture,
            lowered.subsurface_radius_texture,
            lowered.subsurface_radius_scale_texture,
            lowered.subsurface_scatter_anisotropy_texture,
        ]
        .into_iter()
        .collect();
        assert_eq!(resolved.len(), 5);
        assert!(
            crate::scene::subsurface(&lowered).is_some(),
            "a fully textured lobe must still intern an interior"
        );
    }

    /// A per-channel mean free path is baked apart from an albedo, even
    /// from the same file: one is data and one is color, they decode
    /// differently by default, and only one takes the working-space
    /// conversion at sample time.
    #[test]
    fn a_vector_slot_preps_apart_from_a_color_one() {
        let reference = TextureRef {
            path: "/skin.png".into(),
            color_space: None,
            channel: None,
            scale: None,
            uv: None,
        };
        assert_ne!(
            texture_key(&reference, texture::Usage::Vector),
            texture_key(&reference, texture::Usage::Color),
        );
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
    fn missing_normals_derive_from_the_face() {
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
        let image = environment.image.expect("fresh builds carry the image");
        assert_eq!(image.tables().power, 0.0);
    }

    /// The host phase over a triangle scene whose environment is authored
    /// imageless — `path` never set — with the given tint and placement.
    fn pathless_sky_host(tint: [f32; 3], transform: description::Transform) -> HostScene {
        let mut description = triangle_description();
        description
            .apply(&ChangeSet {
                ops: vec![Op::Environment(EnvironmentPatch {
                    tint: Some(tint),
                    transform: Some(transform),
                    ..EnvironmentPatch::new("sky")
                })],
            })
            .expect("valid data");
        host(&description).expect("a pathless sky is legal")
    }

    /// No image means the constant *white* sky — 1×1, unit radiance,
    /// through the one environment code path — with the authored tint
    /// converted to `ACEScg` (the way every authored color enters) and
    /// the placement reduced to its linear part, inverse in hand.
    #[test]
    fn a_pathless_environment_is_a_tinted_white_sky() {
        let host = pathless_sky_host(
            [0.5, 0.25, 0.125],
            description::Transform::Trs {
                translate: [7.0, 8.0, 9.0],
                rotate_degrees: [0.0, 90.0, 0.0],
                scale: [1.0; 3],
            },
        );
        let environment = host.environment.expect("fresh builds load one");
        let image = environment.image.expect("fresh builds carry the image");
        assert_eq!((image.width(), image.height()), (1, 1));
        assert_eq!(image.texels(), &[1.0, 1.0, 1.0, 1.0]);
        assert!(environment.source.is_none());
        let want = acescg_from_rec709(Vec3::new(0.5, 0.25, 0.125));
        assert!(environment.tint.abs_diff_eq(want, 1e-6));
        // The linear part only: the translation is gone, the turn stays,
        // and the inverse undoes it.
        assert_eq!(environment.to_world.w_axis, glam::Vec4::W);
        let turned = environment.to_world.transform_vector3(Vec3::X);
        assert!(turned.abs_diff_eq(-Vec3::Z, 1e-6), "{turned}");
        let back = environment.from_world.transform_vector3(turned);
        assert!(back.abs_diff_eq(Vec3::X, 1e-6), "{back}");
    }

    /// A Radiance `.hdr` file loads through the same path as an EXR —
    /// told apart by magic bytes, not extension.
    #[test]
    fn a_radiance_hdr_environment_loads() {
        let dir = std::env::temp_dir().join(format!("cenote-lower-hdr-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("fixture dir");
        let path = dir.join("sky.hdr");
        let mut bytes = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 2\n".to_vec();
        bytes.extend_from_slice(&[128, 64, 32, 129, 0, 0, 0, 0]);
        std::fs::write(&path, bytes).expect("write sky");

        let mut description = triangle_description();
        description
            .apply(&ChangeSet {
                ops: vec![Op::Environment(EnvironmentPatch {
                    path: Some(Some(path.clone())),
                    ..EnvironmentPatch::new("sky")
                })],
            })
            .expect("valid data");
        let host = host(&description).expect("the .hdr sky loads");
        let environment = host.environment.expect("fresh builds load one");
        let image = environment.image.expect("fresh builds carry the image");
        assert_eq!((image.width(), image.height()), (2, 1));
        assert_eq!(environment.source.as_deref(), Some(path.as_path()));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The keep-resident check: an edit that leaves the source path alone
    /// (a tint drag) reuses the resident image — no decode, no upload —
    /// while a changed path loads. The environment's analogue of the
    /// texture content-hash gate.
    #[test]
    fn an_unchanged_environment_source_keeps_the_resident_image() {
        let dir = std::env::temp_dir().join(format!("cenote-lower-keep-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("fixture dir");
        let sky = dir.join("sky.exr");
        crate::output::write_exr(&sky, 2, 2, &[0.3_f32; 16]).expect("test sky");
        let mut description = triangle_description();
        description
            .apply(&ChangeSet {
                ops: vec![Op::Environment(EnvironmentPatch {
                    path: Some(Some(sky.clone())),
                    tint: Some([0.5; 3]),
                    ..EnvironmentPatch::new("sky")
                })],
            })
            .expect("valid data");
        let with_resident = |resident: Option<&Path>| {
            host_phase(
                &description,
                &all_dirty(&description),
                false,
                &BTreeMap::new(),
                resident,
            )
            .expect("host phase")
            .environment
            .expect("environment dirt lowers a spec")
        };
        let kept = with_resident(Some(&sky));
        assert!(kept.image.is_none(), "same source must keep the image");
        assert_eq!(kept.source.as_deref(), Some(sky.as_path()));
        assert!(kept.tint.abs_diff_eq(acescg_from_rec709(Vec3::splat(0.5)), 1e-6));
        let loaded = with_resident(Some(Path::new("/somewhere/else.exr")));
        assert!(
            loaded.image.is_some(),
            "a changed source must load the new image"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The wall of a unit-radius prism with `segments` sides and unit
    /// height, smooth-normalled as the cylinder it approximates. Each
    /// facet's own normal bisects the turn between its corners, so every
    /// triangle leans exactly 180/`segments` degrees — the whole wall is
    /// coarse or none of it is, at an angle known in closed form.
    fn prism_wall(segments: usize) -> Mesh {
        let mut mesh = Mesh {
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            triangles: Vec::new(),
        };
        for i in 0..segments {
            let turn = std::f32::consts::TAU * i as f32 / segments as f32;
            let radial = Vec3::new(turn.cos(), 0.0, turn.sin());
            mesh.positions.push(radial);
            mesh.positions.push(radial + Vec3::Y);
            mesh.normals.push(radial);
            mesh.normals.push(radial);
            mesh.uvs.push(Vec2::ZERO);
            mesh.uvs.push(Vec2::ZERO);
        }
        for i in 0..segments {
            let (a, b) = (2 * i as u32, 2 * ((i + 1) % segments) as u32);
            mesh.triangles.push([a, a + 1, b + 1]);
            mesh.triangles.push([a, b + 1, b]);
        }
        mesh
    }

    /// Two ways a mesh can be no business of the walk's: geometry that
    /// means to be flat carries its face normal at every corner, and
    /// facets narrower than the free path are stepped over whatever their
    /// lean. The prism is coarse at a free path that resolves it and
    /// clean at one that does not, from the same geometry.
    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "a mesh with nothing to resolve lands in bin 0, whose lower edge is an \
                  exact zero"
    )]
    fn flat_and_sub_free_path_geometry_is_never_coarse() {
        let quad = Mesh {
            positions: vec![
                Vec3::ZERO,
                Vec3::X,
                Vec3::new(1.0, 0.0, 1.0),
                Vec3::new(0.0, 0.0, 1.0),
            ],
            normals: vec![Vec3::Y; 4],
            uvs: vec![Vec2::ZERO; 4],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
        };
        for mfp in [1e-6, 1.0, 1e3] {
            assert_eq!(facet_lean_degrees(&quad, mfp), 0.0);
        }
        // Every facet of an octagonal prism leans half its 45° turn.
        assert!((facet_lean_degrees(&prism_wall(8), 1e-6) - 22.5).abs() <= LEAN_BIN_DEGREES);
        // The widest thing in an octagonal facet is its diagonal, 1.26
        // across; a two-metre free path steps over the whole shape.
        assert_eq!(facet_lean_degrees(&prism_wall(8), 2.0), 0.0);
    }

    /// The reported angle *is* the geometry's own, so subdividing halves
    /// it — the property the share of area above a fixed bar does not
    /// have, and the reason a user can tell whether refining helped. The
    /// warning fires where 180/segments crosses [`COARSE_FACET_DEGREES`],
    /// between 128 and 256 sides.
    #[test]
    fn the_reported_lean_is_the_geometry_s_own_angle() {
        for segments in [8, 16, 32, 64, 128, 256, 512] {
            let expected = 180.0 / segments as f32;
            let lean = facet_lean_degrees(&prism_wall(segments), 1e-6);
            assert!(
                (lean - expected).abs() <= LEAN_BIN_DEGREES,
                "{segments} segments lean {expected:.3}°, reported {lean:.3}°"
            );
            assert_eq!(
                lean > COARSE_FACET_DEGREES,
                expected > COARSE_FACET_DEGREES,
                "{segments} segments warned against its own {expected:.3}° lean"
            );
        }
    }

    /// A steep feature over a large *minority* cannot carry the median —
    /// a lid whose normals were averaged into the wall beside it says
    /// nothing about how well the shape is resolved, and the teapot does
    /// exactly this with 30.8% of its area at 77°. Degenerate data is
    /// skipped rather than counted either way: the head carries three
    /// zero-length shading normals.
    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "degenerate facets must not move the median by so much as a bin edge"
    )]
    fn steep_minorities_and_degenerate_data_do_not_move_the_median() {
        let mut mesh = prism_wall(512); // fine enough to read zero alone
        let base = mesh.positions.len() as u32;
        for corner in [
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(-1.0, 1.0, 1.0),
        ] {
            mesh.positions.push(corner);
            mesh.normals.push(Vec3::new(0.97, 0.24, 0.0).normalize()); // ~77° off +Y
            mesh.uvs.push(Vec2::ZERO);
        }
        mesh.triangles.push([base, base + 1, base + 2]);
        mesh.triangles.push([base, base + 2, base + 3]);
        let lidded = facet_lean_degrees(&mesh, 1e-6);
        assert!(
            lidded <= COARSE_FACET_DEGREES,
            "a mis-normalled lid must not carry the median, read {lidded}°"
        );

        let base = mesh.positions.len() as u32;
        mesh.positions.extend([Vec3::ZERO, Vec3::X, Vec3::X]);
        mesh.normals.extend([Vec3::Y, Vec3::ZERO, Vec3::Y]);
        mesh.uvs.extend([Vec2::ZERO; 3]);
        mesh.triangles.push([base, base + 1, base + 2]); // zero area
        mesh.triangles.push([0, 1, base + 1]); // zero-length normal
        assert_eq!(facet_lean_degrees(&mesh, 1e-6), lidded);
    }

    /// One affine reaches both of a medium's fields, so two lattices
    /// cannot: lowering must refuse the pair rather than sample the
    /// temperature somewhere the density is not. Checked against a
    /// fixture whose two grids differ only in voxel size, and against its
    /// twin that does not — the tolerance the comparison carries is for
    /// last-place drift, not for a real mismatch.
    #[test]
    fn a_temperature_grid_on_its_own_lattice_is_refused() {
        let Some(tool) = crate::scene::source::vdb::test_prep_tool() else {
            return;
        };
        let fire = |voxel: &str, tag: &str| {
            let path = std::env::temp_dir()
                .join(format!("cenote-lower-fire-{tag}-{}.nvdb", std::process::id()));
            let run = std::process::Command::new(&tool)
                .args(["--fire", "8", "1.0", "1000", "0.5"])
                .arg(&path)
                .args(voxel.split_whitespace())
                .output()
                .expect("run cenote-vdb-prep");
            assert!(run.status.success(), "--fire failed (rebuild vdb-prep)");
            path
        };
        let source = |path: &std::path::Path| description::VolumeSource {
            path: path.to_owned(),
            grid: "density".to_owned(),
            temperature_grid: Some("temperature".to_owned()),
            temperature_scale: 1.0,
            temperature_offset: 0.0,
            emission: [1.0; 3],
        };

        let matched = fire("", "matched");
        lower_volume(&source(&matched), Vec3::ONE).expect("one transform, both fields");
        let _ = std::fs::remove_file(&matched);

        let skewed = fire("0.25", "skewed");
        let error = lower_volume(&source(&skewed), Vec3::ONE)
            .expect_err("two lattices under one affine")
            .to_string();
        let _ = std::fs::remove_file(&skewed);
        assert!(error.contains("maps index space differently"), "{error}");
    }
}
