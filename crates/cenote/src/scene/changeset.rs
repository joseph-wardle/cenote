//! Change-sets: ordered, typed edits to a [`SceneDescription`] — the scene
//! model's one verb. A scene file *is* a change-set against the empty
//! description, the pbrt importer emits one, the lookdev panel emits tiny
//! ones, and a network client could carry serialized ones — file, wire,
//! and edit are the same value by construction.
//!
//! The apply contract:
//!
//! - **Get-or-create**: a patch targets an object by name; first mention
//!   creates it with its kind's defaults, and only the patch's `Some`
//!   fields overwrite. Ops apply in order, later fields winning.
//! - **References resolve after the whole set** — an instance may be
//!   patched before the mesh it names exists, as long as the set as a
//!   whole leaves every reference resolvable.
//! - **Validate-then-apply**: every check (references, geometry
//!   consistency, referenced files existing on disk) runs against the
//!   post-set state before any of it becomes visible. A rejected set
//!   leaves the description — and its dirty state — exactly as it was.
//! - **Dirty accumulation**: every applied op records what prep must
//!   rebuild, in [`Dirty`], until [`SceneDescription::take_dirty`] hands
//!   it over. Equality gates the record: an op whose values are already
//!   in place dirties nothing, so re-applying a scene file forces no
//!   re-prep.

use std::collections::BTreeSet;
use std::fmt;
use std::path::{Path, PathBuf};

use glam::Vec3;
use serde::{Deserialize, Serialize};

use super::description::{
    Camera, CurveWrap, Curves, CurvesSource, Geometry, Instance, Light, Medium, Mesh, MeshSource,
    Objects, SceneDescription, Settings, Texturable, TextureRef, Transform, VolumeSource,
    WidthInterpolation,
};
use super::scene_error;
use crate::error::Result;

/// The nine object kinds a description holds — the closed schema.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Kind {
    /// Triangle geometry ([`Mesh`]).
    Mesh,
    /// Curve geometry ([`Curves`]) — hair, fur, grass, fiber.
    Curves,
    /// Placed geometry with a material ([`Instance`]).
    Instance,
    /// An `OpenPBR` surface ([`Material`](super::description::Material)).
    Material,
    /// A participating medium ([`Medium`]).
    Medium,
    /// A delta light ([`Light`]).
    Light,
    /// A viewpoint ([`Camera`]).
    Camera,
    /// The surrounding light image
    /// ([`Environment`](super::description::Environment)).
    Environment,
    /// Render settings ([`Settings`]).
    Settings,
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Mesh => "mesh",
            Self::Curves => "curves",
            Self::Instance => "instance",
            Self::Material => "material",
            Self::Medium => "medium",
            Self::Light => "light",
            Self::Camera => "camera",
            Self::Environment => "environment",
            Self::Settings => "settings",
        })
    }
}

/// One edit: a patch upserting an object of some kind, or a removal.
/// Every variant names its target; [`Op::target`] extracts that uniformly.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Op {
    /// Upsert a mesh.
    Mesh(MeshPatch),
    /// Upsert a curve batch.
    Curves(CurvesPatch),
    /// Upsert an instance.
    Instance(InstancePatch),
    /// Upsert a material. Boxed: the patch is an order of magnitude wider
    /// than any other, and importers build long op lists.
    Material(Box<MaterialPatch>),
    /// Upsert a medium.
    Medium(MediumPatch),
    /// Upsert a delta light.
    Light(LightPatch),
    /// Upsert a camera.
    Camera(CameraPatch),
    /// Upsert an environment.
    Environment(EnvironmentPatch),
    /// Upsert render settings.
    Settings(SettingsPatch),
    /// Delete an object outright. Errors if the target does not exist or
    /// if removing it would strand a reference. Deletion is real —
    /// residency retires with the object — because a scene-graph delegate
    /// (Hydra-style) requires it (renames arrive as remove + re-insert).
    Remove(Kind, String),
}

impl Op {
    /// The kind and name this op targets.
    #[must_use]
    pub fn target(&self) -> (Kind, &str) {
        match self {
            Self::Mesh(patch) => (Kind::Mesh, &patch.name),
            Self::Curves(patch) => (Kind::Curves, &patch.name),
            Self::Instance(patch) => (Kind::Instance, &patch.name),
            Self::Material(patch) => (Kind::Material, &patch.name),
            Self::Medium(patch) => (Kind::Medium, &patch.name),
            Self::Light(patch) => (Kind::Light, &patch.name),
            Self::Camera(patch) => (Kind::Camera, &patch.name),
            Self::Environment(patch) => (Kind::Environment, &patch.name),
            Self::Settings(patch) => (Kind::Settings, &patch.name),
            Self::Remove(kind, name) => (*kind, name),
        }
    }
}

/// Patch for a [`Mesh`]: the payload replaces wholesale.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MeshPatch {
    /// Target name.
    pub name: String,
    /// New geometry payload.
    pub source: Option<MeshSource>,
}

/// Patch for a [`Curves`] batch: the payload replaces wholesale, like a
/// mesh's.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CurvesPatch {
    /// Target name.
    pub name: String,
    /// New curve payload.
    pub source: Option<CurvesSource>,
}

/// Patch for an [`Instance`].
///
/// The two geometry references are one field on the target: naming a mesh
/// clears any curves reference and the other way round, because an
/// instance places one thing. A patch that names both is refused rather
/// than resolved by field order — there is no reading of "place this mesh
/// and these curves" that the scene model can honour.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct InstancePatch {
    /// Target name.
    pub name: String,
    /// Mesh reference, by name.
    pub mesh: Option<String>,
    /// Curves reference, by name — the other spelling of the same field.
    pub curves: Option<String>,
    /// Material reference, by name.
    pub material: Option<String>,
    /// Object-to-world placements, one per copy; the whole array
    /// replaces (`[]` legal — resident, places nothing).
    pub transforms: Option<Vec<Transform>>,
    /// Whether camera rays see it.
    pub camera_visible: Option<bool>,
    /// The medium this mesh bounds, by name. Doubly optional: `None` leaves
    /// the reference alone, `Some(None)` clears it.
    pub medium: Option<Option<String>>,
    /// Which solid wins where refractive interiors overlap.
    pub interior_priority: Option<u32>,
}

/// Patch for a [`Material`](super::description::Material). Fields mirror
/// the target one for one; see there for meanings and defaults.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[expect(missing_docs, reason = "fields document themselves on `Material`")]
pub struct MaterialPatch {
    /// Target name.
    pub name: String,
    pub base_color: Option<Texturable<[f32; 3]>>,
    pub base_diffuse_roughness: Option<f32>,
    pub base_metalness: Option<Texturable<f32>>,
    pub specular_weight: Option<f32>,
    pub specular_roughness: Option<Texturable<f32>>,
    pub specular_ior: Option<f32>,
    pub transmission_weight: Option<f32>,
    pub transmission_color: Option<[f32; 3]>,
    pub transmission_depth: Option<f32>,
    pub transmission_scatter: Option<[f32; 3]>,
    pub transmission_scatter_anisotropy: Option<f32>,
    pub subsurface_weight: Option<Texturable<f32>>,
    pub subsurface_color: Option<Texturable<[f32; 3]>>,
    pub subsurface_radius: Option<Texturable<f32>>,
    pub subsurface_radius_scale: Option<Texturable<[f32; 3]>>,
    pub subsurface_scatter_anisotropy: Option<Texturable<f32>>,
    pub coat_weight: Option<f32>,
    pub coat_color: Option<[f32; 3]>,
    pub coat_roughness: Option<f32>,
    pub coat_ior: Option<f32>,
    pub coat_darkening: Option<f32>,
    pub fuzz_weight: Option<f32>,
    pub fuzz_color: Option<[f32; 3]>,
    pub fuzz_roughness: Option<f32>,
    pub emission_luminance: Option<f32>,
    pub emission_color: Option<Texturable<[f32; 3]>>,
    pub geometry_opacity: Option<Texturable<f32>>,
    pub geometry_thin_walled: Option<bool>,
    /// Doubly optional: `None` leaves the normal map alone, `Some(None)`
    /// clears it.
    pub geometry_normal: Option<Option<TextureRef>>,
}

impl MaterialPatch {
    /// Every texture reference this patch mentions — path rebasing walks
    /// these.
    fn textures_mut(&mut self) -> impl Iterator<Item = &mut TextureRef> {
        [
            self.base_color.as_mut().and_then(Texturable::texture_mut),
            self.base_metalness
                .as_mut()
                .and_then(Texturable::texture_mut),
            self.specular_roughness
                .as_mut()
                .and_then(Texturable::texture_mut),
            self.emission_color
                .as_mut()
                .and_then(Texturable::texture_mut),
            self.geometry_opacity
                .as_mut()
                .and_then(Texturable::texture_mut),
            self.geometry_normal.as_mut().and_then(Option::as_mut),
            self.subsurface_weight
                .as_mut()
                .and_then(Texturable::texture_mut),
            self.subsurface_color
                .as_mut()
                .and_then(Texturable::texture_mut),
            self.subsurface_radius
                .as_mut()
                .and_then(Texturable::texture_mut),
            self.subsurface_radius_scale
                .as_mut()
                .and_then(Texturable::texture_mut),
            self.subsurface_scatter_anisotropy
                .as_mut()
                .and_then(Texturable::texture_mut),
        ]
        .into_iter()
        .flatten()
    }
}

/// Patch for a [`Medium`]. Fields mirror the target one for one; see there
/// for meanings and defaults.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MediumPatch {
    /// Target name.
    pub name: String,
    /// Absorption coefficient per meter, linear `Rec.709`.
    pub absorption: Option<[f32; 3]>,
    /// Scattering coefficient per meter, linear `Rec.709`.
    pub scattering: Option<[f32; 3]>,
    /// Henyey–Greenstein anisotropy.
    pub anisotropy: Option<f32>,
    /// Doubly optional, like the environment's path: `None` leaves the
    /// grid alone, `Some(None)` makes the medium homogeneous again.
    pub volume: Option<Option<VolumeSource>>,
}

/// Patch for a [`Light`]: the definition replaces wholesale — a delta
/// light is a handful of numbers and its variant is its identity.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LightPatch {
    /// Target name.
    pub name: String,
    /// New definition.
    pub light: Option<Light>,
}

/// Patch for a [`Camera`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CameraPatch {
    /// Target name.
    pub name: String,
    /// Eye position, meters.
    pub position: Option<[f32; 3]>,
    /// The point the view axis passes through.
    pub look_at: Option<[f32; 3]>,
    /// Screen-up direction (carries roll).
    pub up: Option<[f32; 3]>,
    /// Vertical field of view, degrees.
    pub vfov_degrees: Option<f32>,
    /// Doubly optional: `None` leaves focus alone, `Some(None)` restores
    /// focus-at-`look_at`.
    pub focus_distance: Option<Option<f32>>,
    /// Lens radius, meters; 0 is a pinhole.
    pub aperture_radius: Option<f32>,
}

/// Patch for an [`Environment`](super::description::Environment).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EnvironmentPatch {
    /// Target name.
    pub name: String,
    /// The equirect radiance image (`.exr` or `.hdr`). Doubly optional:
    /// `None` leaves the image alone, `Some(None)` clears it back to the
    /// constant white sky.
    pub path: Option<Option<PathBuf>>,
    /// Linear `Rec.709` multiplier over the sky's radiance.
    pub tint: Option<[f32; 3]>,
    /// Environment-to-world placement (the linear part turns the sky).
    pub transform: Option<Transform>,
}

/// Patch for [`Settings`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SettingsPatch {
    /// Target name.
    pub name: String,
    /// Output width × height, pixels.
    pub resolution: Option<[u32; 2]>,
    /// The sample budget, in samples per pixel.
    pub spp: Option<u32>,
    /// The convergence early-out threshold. Doubly optional: `None` leaves
    /// it alone, `Some(None)` turns the early-out off and spends the whole
    /// budget.
    pub noise_threshold: Option<Option<f32>>,
    /// Maximum path length in bounces.
    pub max_bounces: Option<u32>,
    /// Denoise the published image — a view of the film, so it never
    /// disturbs the accumulation.
    pub denoise: Option<bool>,
    /// Sampler seed.
    pub seed: Option<u32>,
    /// The medium filling open space, by name. Doubly optional: `None`
    /// leaves it alone, `Some(None)` empties the scene back to vacuum.
    pub global_medium: Option<Option<String>>,
}

/// Generate the shared constructor: a patch that names its target and
/// changes nothing — get-or-create with defaults on its own, or the base
/// for struct-update syntax (`..MaterialPatch::new("floor")`).
macro_rules! named_patches {
    ($($patch:ident),+ $(,)?) => {
        $(impl $patch {
            /// A patch of `name` that changes nothing — get-or-create
            /// with defaults, or the base for struct-update syntax.
            #[must_use]
            pub fn new(name: impl Into<String>) -> Self {
                Self { name: name.into(), ..Self::default() }
            }
        })+
    };
}

named_patches!(
    MeshPatch,
    CurvesPatch,
    InstancePatch,
    MaterialPatch,
    MediumPatch,
    LightPatch,
    CameraPatch,
    EnvironmentPatch,
    SettingsPatch,
);

/// An ordered list of edits, applied atomically — the format's one
/// first-class value. [`ChangeSet::demo`] builds the standing demo scene
/// as one; `crate::format` moves them through `.ron` files.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChangeSet {
    /// The edits, in application order.
    pub ops: Vec<Op>,
}

impl ChangeSet {
    /// Rebase every relative path in the set onto `base` — called by
    /// `crate::format::load` with the scene file's directory, so that
    /// paths mean file-relative and never working-directory-relative.
    pub fn rebase_paths(&mut self, base: &Path) {
        self.for_each_path(|path| {
            if path.is_relative() {
                *path = base.join(&path);
            }
        });
    }

    /// The inverse of [`ChangeSet::rebase_paths`]: strip `base` from
    /// every path under it, leaving the rest absolute — how an importer's
    /// apply-ready (all-absolute) set becomes a portable scene file whose
    /// references travel with it.
    pub fn relativize_paths(&mut self, base: &Path) {
        self.for_each_path(|path| {
            if let Ok(relative) = path.strip_prefix(base) {
                *path = relative.to_owned();
            }
        });
    }

    /// Every filesystem path the set references, one visit each.
    fn for_each_path(&mut self, mut visit: impl FnMut(&mut PathBuf)) {
        for op in &mut self.ops {
            match op {
                Op::Mesh(patch) => {
                    if let Some(MeshSource::Ply { path }) = &mut patch.source {
                        visit(path);
                    }
                }
                Op::Curves(patch) => {
                    if let Some(CurvesSource::Hair { path }) = &mut patch.source {
                        visit(path);
                    }
                }
                Op::Material(patch) => {
                    for texture in patch.textures_mut() {
                        visit(&mut texture.path);
                    }
                }
                Op::Environment(patch) => {
                    if let Some(Some(path)) = &mut patch.path {
                        visit(path);
                    }
                }
                Op::Medium(patch) => {
                    if let Some(Some(volume)) = &mut patch.volume {
                        visit(&mut volume.path);
                    }
                }
                _ => {}
            }
        }
    }
}

/// What prep must rebuild, accumulated across applies until
/// [`SceneDescription::take_dirty`] hands it over.
///
/// `changed` says "(re)build this object's residency"; `removed` says
/// "retire whatever residency it had" (idempotent — the name may never
/// have been prepped). A remove-then-recreate legitimately appears in
/// both. An object patched and then removed appears only in `removed`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Dirty {
    /// Objects created or modified since prep last looked.
    pub changed: BTreeSet<(Kind, String)>,
    /// Objects deleted since prep last looked.
    pub removed: BTreeSet<(Kind, String)>,
}

impl Dirty {
    /// True when there is nothing to rebuild.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.changed.is_empty() && self.removed.is_empty()
    }

    /// Fold a newer round of dirt into this one, keeping the semantics
    /// above (a newer removal supersedes an older change).
    pub fn merge(&mut self, newer: Self) {
        self.changed.retain(|entry| !newer.removed.contains(entry));
        self.changed.extend(newer.changed);
        self.removed.extend(newer.removed);
    }
}

/// Copy every `Some` field of an owned patch onto its target — the merge
/// half of get-or-create-then-patch, one field name per line instead of
/// twenty `if let`s.
macro_rules! merge {
    ($target:expr, $patch:expr; $($field:ident),+ $(,)?) => {
        $(if let Some(value) = $patch.$field {
            $target.$field = value;
        })+
    };
}

impl SceneDescription {
    /// Apply a change-set: the description's only mutation path.
    ///
    /// The whole set lands or none of it does — ops merge into a copy of
    /// the object maps, the result is validated as a whole (so forward
    /// references within the set are legal), and only a fully valid
    /// outcome replaces the originals and records its [`Dirty`] state.
    /// The copy makes atomicity trivially correct; sharing payloads is
    /// the known optimization if edit-rate profiling ever asks for it.
    ///
    /// # Errors
    ///
    /// [`Error::Scene`](crate::Error) when the set is invalid: a removal
    /// that targets nothing or strands a reference, an instance left without a
    /// resolvable mesh or material, inconsistent inline geometry, a
    /// degenerate camera or transform, zero-valued settings, or a
    /// referenced file that is relative or missing on disk.
    pub fn apply(&mut self, set: &ChangeSet) -> Result<()> {
        let mut next = self.objects.clone();
        let mut dirty = Dirty::default();
        for op in &set.ops {
            apply_op(&mut next, &mut dirty, op)?;
        }
        validate(&next)?;
        self.objects = next;
        self.dirty.merge(dirty);
        Ok(())
    }
}

/// Merge one op into the working copy, recording what it dirtied.
fn apply_op(objects: &mut Objects, dirty: &mut Dirty, op: &Op) -> Result<()> {
    let (kind, name) = op.target();
    if name.is_empty() {
        return Err(scene_error(format!("a {kind} op has an empty name")));
    }
    if let Op::Instance(patch) = op
        && patch.mesh.is_some()
        && patch.curves.is_some()
    {
        return Err(scene_error(format!(
            "instance \"{name}\" names both a mesh and a curves batch — an instance places one"
        )));
    }
    if let Op::Remove(kind, name) = op {
        if !objects.remove(*kind, name) {
            return Err(scene_error(format!(
                "Remove targets a {kind} named \"{name}\" that does not exist"
            )));
        }
        dirty.changed.remove(&(*kind, name.clone()));
        dirty.removed.insert((*kind, name.clone()));
        return Ok(());
    }
    let name = name.to_owned();
    let changed = match op.clone() {
        Op::Mesh(patch) => upsert(&mut objects.meshes, &name, |mesh| {
            merge!(mesh, patch; source);
        }),
        Op::Curves(patch) => upsert(&mut objects.curves, &name, |curves| {
            merge!(curves, patch; source);
        }),
        Op::Instance(patch) => upsert(&mut objects.instances, &name, |instance| {
            // The two references are one field: whichever the patch names
            // becomes what this instance places.
            if let Some(mesh) = patch.mesh {
                instance.geometry = Geometry::Mesh(mesh);
            }
            if let Some(curves) = patch.curves {
                instance.geometry = Geometry::Curves(curves);
            }
            merge!(instance, patch;
                material, transforms, camera_visible, medium, interior_priority);
        }),
        Op::Material(patch) => upsert(&mut objects.materials, &name, |material| {
            merge!(material, patch;
                base_color, base_diffuse_roughness, base_metalness,
                specular_weight, specular_roughness, specular_ior,
                transmission_weight, transmission_color, transmission_depth,
                transmission_scatter, transmission_scatter_anisotropy,
                subsurface_weight, subsurface_color, subsurface_radius,
                subsurface_radius_scale, subsurface_scatter_anisotropy,
                coat_weight, coat_color, coat_roughness, coat_ior, coat_darkening,
                fuzz_weight, fuzz_color, fuzz_roughness,
                emission_luminance, emission_color,
                geometry_opacity, geometry_thin_walled, geometry_normal,
            );
        }),
        Op::Medium(patch) => upsert(&mut objects.media, &name, |medium| {
            merge!(medium, patch; absorption, scattering, anisotropy, volume);
        }),
        Op::Light(patch) => upsert(&mut objects.lights, &name, |light| {
            if let Some(value) = patch.light {
                *light = value;
            }
        }),
        Op::Camera(patch) => upsert(&mut objects.cameras, &name, |camera| {
            merge!(camera, patch;
                position, look_at, up, vfov_degrees, focus_distance, aperture_radius,
            );
        }),
        Op::Environment(patch) => upsert(&mut objects.environments, &name, |environment| {
            merge!(environment, patch; path, tint, transform);
        }),
        Op::Settings(patch) => upsert(&mut objects.settings, &name, |settings| {
            merge!(settings, patch;
                resolution, spp, noise_threshold, max_bounces, denoise, seed, global_medium,
            );
        }),
        Op::Remove(..) => unreachable!("handled above"),
    };
    if changed {
        dirty.changed.insert((kind, name));
    }
    Ok(())
}

/// Get-or-create `name` and run the patch merge over it; true when the
/// object is new or the merge changed its value. This equality gate is
/// what keeps a re-applied scene file from dirtying anything: a patch
/// that lands values already in place forces no re-prep and no restart.
fn upsert<T: Clone + Default + PartialEq>(
    map: &mut std::collections::BTreeMap<String, T>,
    name: &str,
    merge: impl FnOnce(&mut T),
) -> bool {
    if let Some(existing) = map.get_mut(name) {
        let before = existing.clone();
        merge(existing);
        *existing != before
    } else {
        let mut fresh = T::default();
        merge(&mut fresh);
        map.insert(name.to_owned(), fresh);
        true
    }
}

impl Objects {
    /// Drop the named object; false if it was never there.
    fn remove(&mut self, kind: Kind, name: &str) -> bool {
        match kind {
            Kind::Mesh => self.meshes.remove(name).is_some(),
            Kind::Curves => self.curves.remove(name).is_some(),
            Kind::Instance => self.instances.remove(name).is_some(),
            Kind::Material => self.materials.remove(name).is_some(),
            Kind::Medium => self.media.remove(name).is_some(),
            Kind::Light => self.lights.remove(name).is_some(),
            Kind::Camera => self.cameras.remove(name).is_some(),
            Kind::Environment => self.environments.remove(name).is_some(),
            Kind::Settings => self.settings.remove(name).is_some(),
        }
    }
}

/// Check the post-set state as a whole. Runs before any of it becomes
/// visible; the first problem aborts the apply.
fn validate(objects: &Objects) -> Result<()> {
    for (name, mesh) in &objects.meshes {
        validate_mesh(name, mesh)?;
    }
    for (name, curves) in &objects.curves {
        validate_curves(name, curves)?;
    }
    for (name, instance) in &objects.instances {
        validate_instance(objects, name, instance)?;
    }
    for (name, material) in &objects.materials {
        for texture in material.textures() {
            validate_path(&format!("a texture of material \"{name}\""), &texture.path)?;
        }
    }
    for (name, light) in &objects.lights {
        if let Light::Distant { direction, .. } = light
            && Vec3::from(*direction) == Vec3::ZERO
        {
            return Err(scene_error(format!(
                "distant light \"{name}\" has a zero direction"
            )));
        }
    }
    for (name, camera) in &objects.cameras {
        validate_camera(name, camera)?;
    }
    for (name, environment) in &objects.environments {
        if let Some(path) = &environment.path {
            validate_path(&format!("environment \"{name}\""), path)?;
        }
        // Sampling maps world directions through the inverse — the same
        // invertibility instances need for their records.
        let matrix = environment.transform.to_mat4();
        if !(matrix.is_finite() && matrix.inverse().is_finite()) {
            return Err(scene_error(format!(
                "environment \"{name}\" has a non-invertible transform"
            )));
        }
    }
    for (name, medium) in &objects.media {
        validate_medium(name, medium)?;
    }
    for (name, settings) in &objects.settings {
        validate_settings(objects, name, settings)?;
    }
    Ok(())
}

fn validate_mesh(name: &str, mesh: &Mesh) -> Result<()> {
    match &mesh.source {
        MeshSource::Inline {
            positions,
            normals,
            uvs,
            triangles,
        } => {
            if positions.is_empty() || triangles.is_empty() {
                return Err(scene_error(format!("mesh \"{name}\" has no geometry")));
            }
            if let Some(normals) = normals
                && normals.len() != positions.len()
            {
                return Err(scene_error(format!(
                    "mesh \"{name}\" has {} normals for {} positions",
                    normals.len(),
                    positions.len()
                )));
            }
            if let Some(uvs) = uvs
                && uvs.len() != positions.len()
            {
                return Err(scene_error(format!(
                    "mesh \"{name}\" has {} uvs for {} positions",
                    uvs.len(),
                    positions.len()
                )));
            }
            let count = positions.len() as u32;
            if triangles.iter().flatten().any(|&index| index >= count) {
                return Err(scene_error(format!(
                    "mesh \"{name}\" has a triangle index out of bounds"
                )));
            }
            Ok(())
        }
        MeshSource::Ply { path } => validate_path(&format!("mesh \"{name}\""), path),
        // The canonical unit cube — nothing of the author's to check; the
        // instance-side pairing rule is validate_instance's.
        MeshSource::MediumBounds => Ok(()),
    }
}

/// Check a curve batch's cells against `UsdGeomBasisCurves`' own rules:
/// the vertex counts have to partition the points array, each count has
/// to be one a segment rule accepts, and the width stream has to hold
/// exactly as many values as its interpolation says. Periodic curves are
/// refused here rather than at prep, so the scene is told before anything
/// is built.
fn validate_curves(name: &str, curves: &Curves) -> Result<()> {
    let named = |message: String| scene_error(format!("curves \"{name}\" {message}"));
    let CurvesSource::Inline {
        points,
        curve_vertex_counts,
        widths,
        curve_type,
        basis,
        wrap,
    } = &curves.source
    else {
        let CurvesSource::Hair { path } = &curves.source else {
            unreachable!("the source is one of two variants")
        };
        return validate_path(&format!("curves \"{name}\""), path);
    };
    if curve_vertex_counts.is_empty() {
        return Err(named("has no curves".to_owned()));
    }
    if *wrap == CurveWrap::Periodic {
        return Err(named(
            "are periodic; the renderer sweeps a strand from its root, and a closed loop \
             has none"
                .to_owned(),
        ));
    }
    let mut vertices = 0usize;
    let mut varying = 0usize;
    for (index, &count) in curve_vertex_counts.iter().enumerate() {
        let count = count as usize;
        let segments = super::curves::segment_count(count, *curve_type, *basis, *wrap)
            .map_err(|error| match error {
                crate::error::Error::Scene(message) => named(format!("hold curve {index}: {message}")),
                other => other,
            })?;
        vertices += count;
        varying += segments + 1;
    }
    if vertices != points.len() {
        return Err(named(format!(
            "name {vertices} vertices across {} curves, but carry {} points",
            curve_vertex_counts.len(),
            points.len()
        )));
    }
    if let Some(widths) = widths {
        let expected = match widths.interpolation {
            WidthInterpolation::Constant => 1,
            WidthInterpolation::Uniform => curve_vertex_counts.len(),
            WidthInterpolation::Varying => varying,
            WidthInterpolation::Vertex => vertices,
        };
        if widths.values.len() != expected {
            return Err(named(format!(
                "carry {} {} widths, but their topology asks for {expected}",
                widths.values.len(),
                match widths.interpolation {
                    WidthInterpolation::Constant => "constant",
                    WidthInterpolation::Uniform => "uniform",
                    WidthInterpolation::Varying => "varying",
                    WidthInterpolation::Vertex => "vertex",
                }
            )));
        }
    }
    Ok(())
}

fn validate_instance(objects: &Objects, name: &str, instance: &Instance) -> Result<()> {
    match &instance.geometry {
        Geometry::Mesh(mesh) => validate_reference(&objects.meshes, "mesh", name, mesh)?,
        Geometry::Curves(curves) => {
            validate_reference(&objects.curves, "curves", name, curves)?;
        }
    }
    validate_reference(&objects.materials, "material", name, &instance.material)?;
    if let Some(medium) = &instance.medium
        && !objects.media.contains_key(medium)
    {
        return Err(scene_error(format!(
            "instance \"{name}\" bounds a medium \"{medium}\" that does not exist"
        )));
    }
    // A heterogeneous medium's extent is its grid's active bounds and
    // nothing else, so it pairs with the auto-generated shell exactly:
    // a user mesh under one — or the shell under anything else — would be
    // a mesh-clipped volume, which is a non-goal (clip in the DCC).
    let heterogeneous = instance
        .medium
        .as_ref()
        .and_then(|medium| objects.media.get(medium))
        .is_some_and(|medium| medium.volume.is_some());
    let bounds_mesh = match &instance.geometry {
        Geometry::Mesh(mesh) => objects
            .meshes
            .get(mesh)
            .is_some_and(|mesh| matches!(mesh.source, MeshSource::MediumBounds)),
        // A groom is not a shell: the auto-generated cube is the only
        // geometry a heterogeneous medium accepts.
        Geometry::Curves(_) => false,
    };
    if heterogeneous && !bounds_mesh {
        return Err(scene_error(format!(
            "instance \"{name}\" bounds a heterogeneous medium on a user mesh — the grid \
             defines its own extent, so the instance must place a MediumBounds mesh"
        )));
    }
    if bounds_mesh && !heterogeneous {
        return Err(scene_error(format!(
            "instance \"{name}\" places a MediumBounds mesh without a heterogeneous medium — \
             the shell's size comes from the medium's grid, so there is nothing to derive it from"
        )));
    }
    // Element by element — an empty array is a valid instance that places
    // nothing, so there is nothing to check.
    for (element, transform) in instance.transforms.iter().enumerate() {
        let matrix = transform.to_mat4();
        if !(matrix.is_finite() && matrix.inverse().is_finite()) {
            return Err(scene_error(format!(
                "instance \"{name}\" has a non-invertible transform (element {element})"
            )));
        }
    }
    Ok(())
}

fn validate_reference<T>(
    map: &std::collections::BTreeMap<String, T>,
    kind: &str,
    instance: &str,
    reference: &str,
) -> Result<()> {
    if reference.is_empty() {
        return Err(scene_error(format!(
            "instance \"{instance}\" was never given a {kind}"
        )));
    }
    if !map.contains_key(reference) {
        return Err(scene_error(format!(
            "instance \"{instance}\" references a {kind} \"{reference}\" that does not exist"
        )));
    }
    Ok(())
}

fn validate_camera(name: &str, camera: &Camera) -> Result<()> {
    let forward = Vec3::from(camera.look_at) - Vec3::from(camera.position);
    if forward == Vec3::ZERO {
        return Err(scene_error(format!(
            "camera \"{name}\": position and look_at coincide"
        )));
    }
    if forward.cross(Vec3::from(camera.up)) == Vec3::ZERO {
        return Err(scene_error(format!(
            "camera \"{name}\": up is parallel to the view axis"
        )));
    }
    if !camera.vfov_degrees.is_finite()
        || camera.vfov_degrees <= 0.0
        || camera.vfov_degrees >= 180.0
    {
        return Err(scene_error(format!(
            "camera \"{name}\": vertical fov must be inside (0, 180) degrees, got {}",
            camera.vfov_degrees
        )));
    }
    if let Some(distance) = camera.focus_distance
        && (distance.is_nan() || distance <= 0.0)
    {
        return Err(scene_error(format!(
            "camera \"{name}\": focus distance must be positive, got {distance}"
        )));
    }
    if camera.aperture_radius.is_nan() || camera.aperture_radius < 0.0 {
        return Err(scene_error(format!(
            "camera \"{name}\": aperture radius must not be negative, got {}",
            camera.aperture_radius
        )));
    }
    Ok(())
}

/// A medium's coefficients must be finite and non-negative: a negative
/// extinction amplifies instead of attenuating, and one path's runaway
/// throughput reaches the whole image through next-event estimation.
fn validate_medium(name: &str, medium: &Medium) -> Result<()> {
    for (what, coefficients) in [
        ("absorption", medium.absorption),
        ("scattering", medium.scattering),
    ] {
        if coefficients.iter().any(|c| !c.is_finite() || *c < 0.0) {
            return Err(scene_error(format!(
                "medium \"{name}\": {what} must be finite and non-negative, got {coefficients:?}"
            )));
        }
    }
    if !medium.anisotropy.is_finite() {
        return Err(scene_error(format!(
            "medium \"{name}\": anisotropy must be finite, got {}",
            medium.anisotropy
        )));
    }
    if let Some(volume) = &medium.volume {
        validate_path(&format!("medium \"{name}\"'s volume"), &volume.path)?;
        if volume.grid.is_empty() {
            return Err(scene_error(format!(
                "medium \"{name}\": the volume's grid name is empty"
            )));
        }
    }
    Ok(())
}

fn validate_settings(objects: &Objects, name: &str, settings: &Settings) -> Result<()> {
    if let Some(medium) = &settings.global_medium {
        let Some(global) = objects.media.get(medium) else {
            return Err(scene_error(format!(
                "settings \"{name}\" name a global medium \"{medium}\" that does not exist"
            )));
        };
        if global.volume.is_some() {
            return Err(scene_error(format!(
                "settings \"{name}\": the global medium \"{medium}\" is heterogeneous — a grid \
                 has bounds and the open space has none; place it with a MediumBounds instance"
            )));
        }
    }
    if settings.resolution.contains(&0) {
        return Err(scene_error(format!(
            "settings \"{name}\": resolution has a zero dimension"
        )));
    }
    if settings.spp == 0 || settings.max_bounces == 0 {
        return Err(scene_error(format!(
            "settings \"{name}\": spp and max_bounces must be at least 1"
        )));
    }
    // The wavefront packs the cap into a byte, and the assert that catches
    // an over-wide one lives in a constructor the session cannot fail into
    // — so the description is where a too-deep render is refused.
    if settings.max_bounces > crate::wavefront::Wavefront::MAX_BOUNCES_LIMIT {
        return Err(scene_error(format!(
            "settings \"{name}\": max_bounces must be at most {}, got {}",
            crate::wavefront::Wavefront::MAX_BOUNCES_LIMIT,
            settings.max_bounces
        )));
    }
    // A relative standard error is a fraction: at 1 the stop fires on the
    // first sample it is trusted at, and above it the number means nothing.
    if let Some(threshold) = settings.noise_threshold
        && !(threshold.is_finite() && threshold > 0.0 && threshold <= 1.0)
    {
        return Err(scene_error(format!(
            "settings \"{name}\": noise_threshold must be in (0, 1], got {threshold}"
        )));
    }
    Ok(())
}

/// Referenced files must exist by apply time, and must already be
/// absolute — relative paths mean scene-file-relative and are rebased at
/// load, so a relative path reaching apply is a caller who skipped that
/// (and whose paths would silently depend on the working directory).
fn validate_path(what: &str, path: &Path) -> Result<()> {
    if path.as_os_str().is_empty() {
        return Err(scene_error(format!("{what} has no file path")));
    }
    if path.is_relative() {
        return Err(scene_error(format!(
            "{what} references the relative path \"{}\" — rebase against the scene directory first \
             (crate::format::load does)",
            path.display()
        )));
    }
    if !path.is_file() {
        return Err(scene_error(format!(
            "{what} references \"{}\", which does not exist",
            path.display()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::description::{CurveBasis, CurveType, Widths};
    use super::*;

    /// An absolute path that certainly exists — apply only checks
    /// existence, never contents, so the crate manifest stands in for any
    /// referenced file.
    fn existing_file() -> PathBuf {
        PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"))
    }

    /// A minimal valid set: one triangle, one material, one instance.
    fn triangle_scene() -> ChangeSet {
        ChangeSet {
            ops: vec![
                Op::Instance(InstancePatch {
                    mesh: Some("tri".into()),
                    material: Some("gray".into()),
                    ..InstancePatch::new("thing")
                }),
                Op::Material(Box::new(MaterialPatch::new("gray"))),
                Op::Mesh(MeshPatch {
                    source: Some(MeshSource::Inline {
                        positions: vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                        normals: None,
                        uvs: None,
                        triangles: vec![[0, 1, 2]],
                    }),
                    ..MeshPatch::new("tri")
                }),
            ],
        }
    }

    /// The instance op above precedes the mesh and material it names —
    /// legal, because references resolve after the whole set.
    #[test]
    fn forward_references_within_a_set_resolve() {
        let mut description = SceneDescription::new();
        description.apply(&triangle_scene()).expect("valid set");
        assert_eq!(description.instances().len(), 1);
        assert_eq!(description.meshes().len(), 1);
    }

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "patched values must copy through bit-exact — no arithmetic is involved"
    )]
    fn later_ops_win_field_by_field() {
        let mut description = SceneDescription::new();
        let set = ChangeSet {
            ops: vec![
                Op::Material(Box::new(MaterialPatch {
                    coat_weight: Some(0.25),
                    specular_ior: Some(1.8),
                    ..MaterialPatch::new("m")
                })),
                Op::Material(Box::new(MaterialPatch {
                    coat_weight: Some(1.0),
                    ..MaterialPatch::new("m")
                })),
            ],
        };
        description.apply(&set).expect("valid set");
        let material = &description.materials()["m"];
        // The later op's field wins; the earlier op's other field survives.
        assert_eq!(material.coat_weight, 1.0);
        assert_eq!(material.specular_ior, 1.8);
        // Untouched fields keep OpenPBR defaults.
        assert_eq!(material.coat_ior, 1.6);
    }

    /// The equality gate: patches that land values already in place must
    /// not dirty — a re-applied scene file would otherwise rebuild (and
    /// restart) the world on every save.
    #[test]
    fn reapplying_a_set_dirties_nothing() {
        let mut description = SceneDescription::new();
        description.apply(&triangle_scene()).expect("valid set");
        assert!(!description.take_dirty().is_empty());
        description.apply(&triangle_scene()).expect("valid set");
        assert!(description.take_dirty().is_empty());
    }

    /// …but creation always dirties, even when the created object holds
    /// nothing beyond its defaults: prep must learn it exists.
    #[test]
    fn creation_dirties_even_at_defaults() {
        let mut description = SceneDescription::new();
        let set = ChangeSet {
            ops: vec![Op::Settings(SettingsPatch::new("main"))],
        };
        description.apply(&set).expect("valid set");
        let dirty = description.take_dirty();
        assert!(dirty.changed.contains(&(Kind::Settings, "main".into())));
        // The same all-default patch against the existing object: no-op.
        description.apply(&set).expect("valid set");
        assert!(description.take_dirty().is_empty());
    }

    #[test]
    fn a_rejected_set_changes_nothing() {
        let mut description = SceneDescription::new();
        let mut set = triangle_scene();
        set.ops.push(Op::Instance(InstancePatch {
            mesh: Some("no-such-mesh".into()),
            material: Some("gray".into()),
            ..InstancePatch::new("broken")
        }));
        let error = description.apply(&set).unwrap_err();
        assert!(error.to_string().contains("no-such-mesh"), "{error}");
        // Atomicity: the valid leading ops did not land either.
        assert!(description.meshes().is_empty());
        assert!(description.instances().is_empty());
        assert!(description.take_dirty().is_empty());
    }

    #[test]
    fn remove_deletes_and_supersedes_earlier_dirt() {
        let mut description = SceneDescription::new();
        description.apply(&triangle_scene()).expect("valid set");
        let removal = ChangeSet {
            ops: vec![
                Op::Remove(Kind::Instance, "thing".into()),
                Op::Remove(Kind::Mesh, "tri".into()),
            ],
        };
        description.apply(&removal).expect("valid removal");
        assert!(description.instances().is_empty());
        assert!(description.meshes().is_empty());
        let dirty = description.take_dirty();
        // The create-then-remove nets out of `changed`; `removed` tells
        // prep to retire whatever residency the names had (none yet).
        assert!(!dirty.changed.contains(&(Kind::Mesh, "tri".into())));
        assert!(dirty.removed.contains(&(Kind::Mesh, "tri".into())));
        assert!(dirty.changed.contains(&(Kind::Material, "gray".into())));
    }

    #[test]
    fn remove_of_a_missing_object_is_an_error() {
        let mut description = SceneDescription::new();
        let set = ChangeSet {
            ops: vec![Op::Remove(Kind::Camera, "ghost".into())],
        };
        let error = description.apply(&set).unwrap_err();
        assert!(error.to_string().contains("ghost"), "{error}");
    }

    #[test]
    fn remove_that_strands_a_reference_is_an_error() {
        let mut description = SceneDescription::new();
        description.apply(&triangle_scene()).expect("valid set");
        let removal = ChangeSet {
            ops: vec![Op::Remove(Kind::Material, "gray".into())],
        };
        let error = description.apply(&removal).unwrap_err();
        assert!(error.to_string().contains("\"gray\""), "{error}");
        // The strandable reference kept its material.
        assert_eq!(description.materials().len(), 1);
    }

    #[test]
    fn remove_then_recreate_is_legal_and_dirties_both_ways() {
        let mut description = SceneDescription::new();
        description.apply(&triangle_scene()).expect("valid set");
        description.take_dirty();
        let mut set = triangle_scene();
        set.ops.insert(0, Op::Remove(Kind::Mesh, "tri".into()));
        description.apply(&set).expect("remove then recreate");
        let dirty = description.take_dirty();
        assert!(dirty.removed.contains(&(Kind::Mesh, "tri".into())));
        assert!(dirty.changed.contains(&(Kind::Mesh, "tri".into())));
    }

    #[test]
    fn an_instance_created_bare_is_rejected() {
        let mut description = SceneDescription::new();
        let set = ChangeSet {
            ops: vec![Op::Instance(InstancePatch::new("bare"))],
        };
        let error = description.apply(&set).unwrap_err();
        assert!(error.to_string().contains("never given a mesh"), "{error}");
    }

    #[test]
    fn empty_names_are_rejected() {
        let mut description = SceneDescription::new();
        let set = ChangeSet {
            ops: vec![Op::Material(Box::new(MaterialPatch::new("")))],
        };
        let error = description.apply(&set).unwrap_err();
        assert!(error.to_string().contains("empty name"), "{error}");
    }

    #[test]
    fn inconsistent_inline_geometry_is_rejected() {
        let mut description = SceneDescription::new();
        let mut set = triangle_scene();
        let Op::Mesh(mesh) = &mut set.ops[2] else {
            panic!("triangle_scene changed shape");
        };
        mesh.source = Some(MeshSource::Inline {
            positions: vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: Some(vec![[0.0, 0.0, 1.0]]),
            uvs: None,
            triangles: vec![[0, 1, 2]],
        });
        let error = description.apply(&set).unwrap_err();
        assert!(error.to_string().contains("1 normals"), "{error}");
    }

    #[test]
    fn out_of_bounds_indices_are_rejected() {
        let mut description = SceneDescription::new();
        let mut set = triangle_scene();
        let Op::Mesh(mesh) = &mut set.ops[2] else {
            panic!("triangle_scene changed shape");
        };
        mesh.source = Some(MeshSource::Inline {
            positions: vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: None,
            uvs: None,
            triangles: vec![[0, 1, 3]],
        });
        let error = description.apply(&set).unwrap_err();
        assert!(error.to_string().contains("out of bounds"), "{error}");
    }

    /// The environment's own validation: a placement whose linear part
    /// can't invert is rejected (sampling maps directions back through the
    /// inverse), and the doubly-optional path clears back to the constant
    /// sky rather than sticking.
    #[test]
    fn environment_transforms_must_invert_and_paths_clear() {
        let mut description = SceneDescription::new();
        let squashed = ChangeSet {
            ops: vec![Op::Environment(EnvironmentPatch {
                transform: Some(Transform::Trs {
                    translate: [0.0; 3],
                    rotate_degrees: [0.0; 3],
                    scale: [1.0, 0.0, 1.0],
                }),
                ..EnvironmentPatch::new("sky")
            })],
        };
        let error = description.apply(&squashed).unwrap_err();
        assert!(error.to_string().contains("non-invertible"), "{error}");

        description
            .apply(&ChangeSet {
                ops: vec![Op::Environment(EnvironmentPatch {
                    path: Some(Some(existing_file())),
                    ..EnvironmentPatch::new("sky")
                })],
            })
            .expect("a set image applies");
        assert_eq!(
            description.environments()["sky"].path,
            Some(existing_file())
        );
        description
            .apply(&ChangeSet {
                ops: vec![Op::Environment(EnvironmentPatch {
                    path: Some(None),
                    ..EnvironmentPatch::new("sky")
                })],
            })
            .expect("the clear applies");
        assert_eq!(description.environments()["sky"].path, None);
    }

    #[test]
    fn referenced_files_must_exist_and_be_absolute() {
        let mut description = SceneDescription::new();
        let relative = ChangeSet {
            ops: vec![Op::Environment(EnvironmentPatch {
                path: Some(Some("sky.exr".into())),
                ..EnvironmentPatch::new("sky")
            })],
        };
        let error = description.apply(&relative).unwrap_err();
        assert!(error.to_string().contains("relative"), "{error}");

        let missing = ChangeSet {
            ops: vec![Op::Environment(EnvironmentPatch {
                path: Some(Some("/no/such/sky.exr".into())),
                ..EnvironmentPatch::new("sky")
            })],
        };
        let error = description.apply(&missing).unwrap_err();
        assert!(error.to_string().contains("does not exist"), "{error}");

        let present = ChangeSet {
            ops: vec![Op::Mesh(MeshPatch {
                source: Some(MeshSource::Ply {
                    path: existing_file(),
                }),
                ..MeshPatch::new("ply-mesh")
            })],
        };
        description.apply(&present).expect("existing absolute path");
    }

    #[test]
    fn degenerate_cameras_are_rejected() {
        let mut description = SceneDescription::new();
        let coincident = ChangeSet {
            ops: vec![Op::Camera(CameraPatch {
                position: Some([0.0; 3]),
                look_at: Some([0.0; 3]),
                ..CameraPatch::new("main")
            })],
        };
        let error = description.apply(&coincident).unwrap_err();
        assert!(error.to_string().contains("coincide"), "{error}");

        let vertical = ChangeSet {
            ops: vec![Op::Camera(CameraPatch {
                position: Some([0.0; 3]),
                look_at: Some([0.0, 1.0, 0.0]),
                ..CameraPatch::new("main")
            })],
        };
        let error = description.apply(&vertical).unwrap_err();
        assert!(error.to_string().contains("parallel"), "{error}");
    }

    /// A global medium is a reference like any other: it must resolve
    /// after the whole set, and removing the object it names strands it.
    /// Coefficients are checked here rather than clamped at prep because a
    /// negative extinction amplifies, and one runaway path reaches the whole
    /// image through next-event estimation.
    #[test]
    fn a_global_medium_must_resolve_and_be_physical() {
        let settings = |medium: &str| {
            Op::Settings(SettingsPatch {
                global_medium: Some(Some(medium.to_owned())),
                ..SettingsPatch::new("main")
            })
        };
        let mut description = SceneDescription::new();
        let error = description
            .apply(&ChangeSet {
                ops: vec![settings("fog")],
            })
            .unwrap_err();
        assert!(error.to_string().contains("does not exist"), "{error}");

        // Forward references within one set are legal, as everywhere else.
        description
            .apply(&ChangeSet {
                ops: vec![
                    settings("fog"),
                    Op::Medium(MediumPatch {
                        scattering: Some([0.1; 3]),
                        ..MediumPatch::new("fog")
                    }),
                ],
            })
            .expect("valid");
        assert_eq!(
            description.media()["fog"],
            Medium {
                scattering: [0.1; 3],
                ..Medium::default()
            }
        );

        let error = description
            .apply(&ChangeSet {
                ops: vec![Op::Remove(Kind::Medium, "fog".into())],
            })
            .unwrap_err();
        assert!(error.to_string().contains("does not exist"), "{error}");

        let error = description
            .apply(&ChangeSet {
                ops: vec![Op::Medium(MediumPatch {
                    absorption: Some([0.1, -0.2, 0.1]),
                    ..MediumPatch::new("fog")
                })],
            })
            .unwrap_err();
        assert!(error.to_string().contains("non-negative"), "{error}");

        // Emptying the reference releases the object again.
        description
            .apply(&ChangeSet {
                ops: vec![
                    Op::Settings(SettingsPatch {
                        global_medium: Some(None),
                        ..SettingsPatch::new("main")
                    }),
                    Op::Remove(Kind::Medium, "fog".into()),
                ],
            })
            .expect("valid");
        assert!(description.media().is_empty());
    }

    /// An instance bounds a medium by name, which makes it a reference like
    /// the mesh and material beside it: dangling on the way in, and holding
    /// the object against removal on the way out.
    #[test]
    fn a_bounded_volume_holds_its_medium() {
        let mut description = SceneDescription::new();
        let instance = |medium: Option<&str>| {
            Op::Instance(InstancePatch {
                mesh: Some("tri".into()),
                material: Some("gray".into()),
                medium: Some(medium.map(str::to_owned)),
                ..InstancePatch::new("fog-box")
            })
        };
        let geometry = || {
            [
                Op::Material(Box::new(MaterialPatch::new("gray"))),
                Op::Mesh(MeshPatch {
                    source: Some(MeshSource::Inline {
                        positions: vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                        normals: None,
                        uvs: None,
                        triangles: vec![[0, 1, 2]],
                    }),
                    ..MeshPatch::new("tri")
                }),
            ]
        };
        let error = description
            .apply(&ChangeSet {
                ops: [instance(Some("haze"))]
                    .into_iter()
                    .chain(geometry())
                    .collect(),
            })
            .unwrap_err();
        assert!(error.to_string().contains("bounds a medium"), "{error}");

        description
            .apply(&ChangeSet {
                ops: [
                    instance(Some("haze")),
                    Op::Medium(MediumPatch {
                        scattering: Some([0.2; 3]),
                        ..MediumPatch::new("haze")
                    }),
                ]
                .into_iter()
                .chain(geometry())
                .collect(),
            })
            .expect("valid");
        assert_eq!(
            description.instances()["fog-box"].medium.as_deref(),
            Some("haze")
        );

        let error = description
            .apply(&ChangeSet {
                ops: vec![Op::Remove(Kind::Medium, "haze".into())],
            })
            .unwrap_err();
        assert!(error.to_string().contains("does not exist"), "{error}");

        // Clearing the reference releases it, and leaves an ordinary surface.
        description
            .apply(&ChangeSet {
                ops: vec![instance(None), Op::Remove(Kind::Medium, "haze".into())],
            })
            .expect("valid");
        assert!(description.instances()["fog-box"].medium.is_none());
    }

    #[test]
    fn zero_settings_are_rejected() {
        let mut description = SceneDescription::new();
        let set = ChangeSet {
            ops: vec![Op::Settings(SettingsPatch {
                spp: Some(0),
                ..SettingsPatch::new("main")
            })],
        };
        let error = description.apply(&set).unwrap_err();
        assert!(error.to_string().contains("at least 1"), "{error}");
    }

    /// The path-length cap travels to the kernels in one byte, and the
    /// assert that catches an over-wide one lives in a constructor the
    /// running session never calls again. So the description is where a
    /// too-deep render has to be refused: a host that hands 512 straight
    /// through would take the whole change-set — every mesh in the same
    /// flush — down with it on a panic instead of a rejection.
    #[test]
    fn a_path_depth_past_the_packed_byte_is_rejected() {
        let mut description = SceneDescription::new();
        let set = ChangeSet {
            ops: vec![Op::Settings(SettingsPatch {
                max_bounces: Some(512),
                ..SettingsPatch::new("main")
            })],
        };
        let error = description.apply(&set).unwrap_err();
        assert!(error.to_string().contains("at most 255"), "{error}");
        // And the boundary itself is legal.
        description
            .apply(&ChangeSet {
                ops: vec![Op::Settings(SettingsPatch {
                    max_bounces: Some(crate::wavefront::Wavefront::MAX_BOUNCES_LIMIT),
                    ..SettingsPatch::new("main")
                })],
            })
            .expect("the cap itself renders");
    }

    /// The convergence threshold is a *relative standard error*, so it is a
    /// fraction: at 1 the stop fires as soon as the metric is trusted, and
    /// past that the number describes nothing. Zero and negatives would stop
    /// the render never and immediately respectively — both silently.
    #[test]
    fn a_noise_threshold_outside_the_unit_interval_is_rejected() {
        let threshold = |value: f32| ChangeSet {
            ops: vec![Op::Settings(SettingsPatch {
                noise_threshold: Some(Some(value)),
                ..SettingsPatch::new("main")
            })],
        };
        for bad in [0.0, -0.01, 1.5, f32::NAN] {
            let mut description = SceneDescription::new();
            let error = description.apply(&threshold(bad)).unwrap_err();
            assert!(error.to_string().contains("in (0, 1]"), "{bad}: {error}");
        }
        let mut description = SceneDescription::new();
        description.apply(&threshold(1.0)).expect("1 is a fraction");
        assert_eq!(description.settings()["main"].noise_threshold, Some(1.0));
        // And the early stop is switchable back off without touching the rest.
        description
            .apply(&ChangeSet {
                ops: vec![Op::Settings(SettingsPatch {
                    noise_threshold: Some(None),
                    ..SettingsPatch::new("main")
                })],
            })
            .expect("no threshold is valid");
        assert_eq!(description.settings()["main"].noise_threshold, None);
    }

    /// Validation is per element: a valid placement doesn't shield a
    /// singular one, and the error names the offender by index.
    #[test]
    fn singular_transforms_are_rejected() {
        let mut description = SceneDescription::new();
        let mut set = triangle_scene();
        let Op::Instance(instance) = &mut set.ops[0] else {
            panic!("triangle_scene changed shape");
        };
        instance.transforms = Some(vec![
            Transform::default(),
            Transform::Trs {
                translate: [0.0; 3],
                rotate_degrees: [0.0; 3],
                scale: [1.0, 0.0, 1.0],
            },
        ]);
        let error = description.apply(&set).unwrap_err();
        assert!(error.to_string().contains("non-invertible"), "{error}");
        assert!(error.to_string().contains("element 1"), "{error}");
    }

    /// The placements array replaces wholesale — a later patch's array
    /// wins entirely — and the empty array is legal: the instance stays
    /// resident (references still validate) while placing nothing.
    #[test]
    fn transforms_replace_wholesale_and_empty_is_legal() {
        let mut description = SceneDescription::new();
        let mut set = triangle_scene();
        let Op::Instance(instance) = &mut set.ops[0] else {
            panic!("triangle_scene changed shape");
        };
        instance.transforms = Some(vec![
            Transform::default(),
            Transform::Trs {
                translate: [2.0, 0.0, 0.0],
                rotate_degrees: [0.0; 3],
                scale: [1.0; 3],
            },
        ]);
        description.apply(&set).expect("two placements are valid");
        assert_eq!(description.instances()["thing"].transforms.len(), 2);

        description
            .apply(&ChangeSet {
                ops: vec![Op::Instance(InstancePatch {
                    transforms: Some(vec![]),
                    ..InstancePatch::new("thing")
                })],
            })
            .expect("the empty array is legal");
        assert!(description.instances()["thing"].transforms.is_empty());
        // …and a dangling reference is still an error on the resident,
        // placement-less instance.
        let error = description
            .apply(&ChangeSet {
                ops: vec![Op::Remove(Kind::Material, "gray".into())],
            })
            .unwrap_err();
        assert!(error.to_string().contains("\"gray\""), "{error}");
    }

    #[test]
    fn rebase_touches_only_relative_paths() {
        let mut set = ChangeSet {
            ops: vec![
                Op::Mesh(MeshPatch {
                    source: Some(MeshSource::Ply {
                        path: "geo/mesh.ply".into(),
                    }),
                    ..MeshPatch::new("m")
                }),
                Op::Environment(EnvironmentPatch {
                    path: Some(Some("/already/absolute.exr".into())),
                    ..EnvironmentPatch::new("sky")
                }),
                Op::Material(Box::new(MaterialPatch {
                    base_color: Some(Texturable::Texture(TextureRef {
                        path: "wood.png".into(),
                        color_space: None,
                        channel: None,
                        scale: None,
                        uv: None,
                    })),
                    ..MaterialPatch::new("wood")
                })),
            ],
        };
        set.rebase_paths(Path::new("/scenes"));
        let Op::Mesh(mesh) = &set.ops[0] else {
            unreachable!()
        };
        assert_eq!(
            mesh.source,
            Some(MeshSource::Ply {
                path: "/scenes/geo/mesh.ply".into()
            })
        );
        let Op::Environment(environment) = &set.ops[1] else {
            unreachable!()
        };
        assert_eq!(environment.path, Some(Some("/already/absolute.exr".into())));
        let Op::Material(material) = &set.ops[2] else {
            unreachable!()
        };
        assert_eq!(
            material.base_color.as_ref().and_then(Texturable::texture),
            Some(&TextureRef {
                path: "/scenes/wood.png".into(),
                color_space: None,
                channel: None,
                scale: None,
                uv: None,
            })
        );
    }

    #[test]
    fn relativize_strips_exactly_the_base() {
        let mut set = ChangeSet {
            ops: vec![
                Op::Mesh(MeshPatch {
                    source: Some(MeshSource::Ply {
                        path: "/scenes/geo/mesh.ply".into(),
                    }),
                    ..MeshPatch::new("m")
                }),
                Op::Environment(EnvironmentPatch {
                    path: Some(Some("/elsewhere/sky.exr".into())),
                    ..EnvironmentPatch::new("sky")
                }),
            ],
        };
        set.relativize_paths(Path::new("/scenes"));
        let Op::Mesh(mesh) = &set.ops[0] else {
            unreachable!()
        };
        assert_eq!(
            mesh.source,
            Some(MeshSource::Ply {
                path: "geo/mesh.ply".into()
            })
        );
        // A path outside the base stays absolute — still correct, just
        // not portable.
        let Op::Environment(environment) = &set.ops[1] else {
            unreachable!()
        };
        assert_eq!(environment.path, Some(Some("/elsewhere/sky.exr".into())));

        // Round trip: rebasing against the same directory restores it.
        set.rebase_paths(Path::new("/scenes"));
        let Op::Mesh(mesh) = &set.ops[0] else {
            unreachable!()
        };
        assert_eq!(
            mesh.source,
            Some(MeshSource::Ply {
                path: "/scenes/geo/mesh.ply".into()
            })
        );
    }

    #[test]
    fn dirty_merge_keeps_retire_then_rebuild() {
        let mut older = Dirty::default();
        older.changed.insert((Kind::Mesh, "a".into()));
        older.removed.insert((Kind::Mesh, "b".into()));
        let mut newer = Dirty::default();
        newer.removed.insert((Kind::Mesh, "a".into()));
        newer.changed.insert((Kind::Mesh, "b".into()));
        older.merge(newer);
        // "a" was changed then removed: retire only. "b" was removed then
        // recreated: retire the old residency, build the new.
        assert!(!older.changed.contains(&(Kind::Mesh, "a".into())));
        assert!(older.removed.contains(&(Kind::Mesh, "a".into())));
        assert!(older.changed.contains(&(Kind::Mesh, "b".into())));
        assert!(older.removed.contains(&(Kind::Mesh, "b".into())));
    }

    /// A minimal valid curve batch: one four-vertex bezier strand under a
    /// material and an instance that places it.
    fn groom_scene() -> ChangeSet {
        ChangeSet {
            ops: vec![
                Op::Curves(CurvesPatch {
                    source: Some(cells(&[4], 4, None)),
                    ..CurvesPatch::new("groom")
                }),
                Op::Material(Box::new(MaterialPatch::new("gray"))),
                Op::Instance(InstancePatch {
                    curves: Some("groom".into()),
                    material: Some("gray".into()),
                    ..InstancePatch::new("hair")
                }),
            ],
        }
    }

    /// `points` bezier control vertices split by `counts`, optionally with
    /// a width stream.
    fn cells(counts: &[u32], points: u32, widths: Option<Widths>) -> CurvesSource {
        CurvesSource::Inline {
            points: (0..points)
                .map(|index| [0.0, index as f32, 0.0])
                .collect(),
            curve_vertex_counts: counts.to_vec(),
            widths,
            curve_type: CurveType::Cubic,
            basis: CurveBasis::Bezier,
            wrap: CurveWrap::Nonperiodic,
        }
    }

    /// Curves are a kind of their own: an instance places one or the
    /// other, and the reference resolves in that kind's map.
    #[test]
    fn an_instance_places_curves_the_way_it_places_a_mesh() {
        let mut description = SceneDescription::new();
        description.apply(&groom_scene()).expect("valid set");
        assert_eq!(
            description.instances()["hair"].geometry,
            Geometry::Curves("groom".into())
        );
        let dirty = description.take_dirty();
        assert!(dirty.changed.contains(&(Kind::Curves, "groom".into())));

        // A later patch naming a mesh moves the same instance across
        // kinds — the two spellings are one field.
        description
            .apply(&ChangeSet {
                ops: vec![
                    Op::Mesh(MeshPatch {
                        source: Some(MeshSource::Inline {
                            positions: vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                            normals: None,
                            uvs: None,
                            triangles: vec![[0, 1, 2]],
                        }),
                        ..MeshPatch::new("tri")
                    }),
                    Op::Instance(InstancePatch {
                        mesh: Some("tri".into()),
                        ..InstancePatch::new("hair")
                    }),
                ],
            })
            .expect("valid set");
        assert_eq!(
            description.instances()["hair"].geometry,
            Geometry::Mesh("tri".into())
        );
    }

    #[test]
    fn an_instance_naming_both_kinds_is_refused() {
        let mut description = SceneDescription::new();
        let mut set = groom_scene();
        set.ops.push(Op::Instance(InstancePatch {
            mesh: Some("tri".into()),
            curves: Some("groom".into()),
            ..InstancePatch::new("hair")
        }));
        let error = description.apply(&set).unwrap_err();
        assert!(error.to_string().contains("places one"), "{error}");
    }

    #[test]
    fn a_curves_reference_that_names_a_mesh_does_not_resolve() {
        let mut description = SceneDescription::new();
        let mut set = triangle_scene();
        set.ops.push(Op::Instance(InstancePatch {
            curves: Some("tri".into()),
            material: Some("gray".into()),
            ..InstancePatch::new("confused")
        }));
        let error = description.apply(&set).unwrap_err();
        assert!(error.to_string().contains("curves \"tri\""), "{error}");
    }

    /// Removing a curve batch retires it exactly as a mesh removal does —
    /// and strands its instance, which is what makes the order matter.
    #[test]
    fn removing_curves_retires_them() {
        let mut description = SceneDescription::new();
        description.apply(&groom_scene()).expect("valid set");
        description.take_dirty();
        description
            .apply(&ChangeSet {
                ops: vec![Op::Remove(Kind::Curves, "groom".into())],
            })
            .expect_err("the instance still names it");
        description
            .apply(&ChangeSet {
                ops: vec![
                    Op::Remove(Kind::Instance, "hair".into()),
                    Op::Remove(Kind::Curves, "groom".into()),
                ],
            })
            .expect("valid set");
        assert!(description.curves().is_empty());
        let dirty = description.take_dirty();
        assert!(dirty.removed.contains(&(Kind::Curves, "groom".into())));
    }

    /// The topology rules are `UsdGeomBasisCurves`', applied before
    /// anything is built: counts that do not partition the points, a
    /// vertex count no segment rule accepts, a periodic wrap, and a width
    /// stream of the wrong length are each refused by name.
    #[test]
    fn curve_topology_is_validated_against_the_usd_rules() {
        let refuse = |source: CurvesSource, expected: &str| {
            let mut description = SceneDescription::new();
            let error = description
                .apply(&ChangeSet {
                    ops: vec![Op::Curves(CurvesPatch {
                        source: Some(source),
                        ..CurvesPatch::new("groom")
                    })],
                })
                .unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
        };
        refuse(cells(&[4, 4], 7, None), "carry 7 points");
        refuse(cells(&[6], 6, None), "cannot have 6 vertices");
        let CurvesSource::Inline {
            points,
            curve_vertex_counts,
            curve_type,
            basis,
            ..
        } = cells(&[4], 4, None)
        else {
            unreachable!("the fixture is inline")
        };
        refuse(
            CurvesSource::Inline {
                points,
                curve_vertex_counts,
                widths: None,
                curve_type,
                basis,
                wrap: CurveWrap::Periodic,
            },
            "periodic",
        );
        for (interpolation, wrong) in [
            (WidthInterpolation::Constant, 2),
            (WidthInterpolation::Uniform, 2),
            (WidthInterpolation::Varying, 3),
            (WidthInterpolation::Vertex, 3),
        ] {
            refuse(
                cells(
                    &[4],
                    4,
                    Some(Widths {
                        values: vec![0.1; wrong],
                        interpolation,
                    }),
                ),
                "asks for",
            );
        }
        // …and the right lengths pass: one for the batch, one per curve,
        // one per segment end, one per vertex.
        let mut description = SceneDescription::new();
        for (interpolation, right) in [
            (WidthInterpolation::Constant, 1),
            (WidthInterpolation::Uniform, 1),
            (WidthInterpolation::Varying, 2),
            (WidthInterpolation::Vertex, 4),
        ] {
            description
                .apply(&ChangeSet {
                    ops: vec![Op::Curves(CurvesPatch {
                        source: Some(cells(
                            &[4],
                            4,
                            Some(Widths {
                                values: vec![0.1; right],
                                interpolation,
                            }),
                        )),
                        ..CurvesPatch::new("groom")
                    })],
                })
                .unwrap_or_else(|error| panic!("{interpolation:?}: {error}"));
        }
    }

    /// A groom by reference is a path like any other: rebased at load,
    /// absolute by the time it applies, and checked to exist.
    #[test]
    fn a_hair_reference_is_a_path_like_any_other() {
        let mut description = SceneDescription::new();
        let error = description
            .apply(&ChangeSet {
                ops: vec![Op::Curves(CurvesPatch {
                    source: Some(CurvesSource::Hair {
                        path: "/no/such/groom.hair".into(),
                    }),
                    ..CurvesPatch::new("groom")
                })],
            })
            .unwrap_err();
        assert!(error.to_string().contains("groom.hair"), "{error}");

        let mut set = ChangeSet {
            ops: vec![Op::Curves(CurvesPatch {
                source: Some(CurvesSource::Hair {
                    path: "groom.hair".into(),
                }),
                ..CurvesPatch::new("groom")
            })],
        };
        set.rebase_paths(Path::new("/scenes"));
        assert!(matches!(
            &set.ops[0],
            Op::Curves(CurvesPatch { source: Some(CurvesSource::Hair { path }), .. })
                if path == Path::new("/scenes/groom.hair")
        ));
    }
}
