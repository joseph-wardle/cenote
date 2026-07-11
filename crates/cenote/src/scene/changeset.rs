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
    Camera, Instance, Light, Mesh, MeshSource, Objects, SceneDescription, Settings, Texturable,
    TextureRef, Transform,
};
use super::scene_error;
use crate::error::Result;

/// The seven object kinds a description holds — the closed schema.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Kind {
    /// Triangle geometry ([`Mesh`]).
    Mesh,
    /// A placed mesh with a material ([`Instance`]).
    Instance,
    /// An `OpenPBR` surface ([`Material`](super::description::Material)).
    Material,
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
            Self::Instance => "instance",
            Self::Material => "material",
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
    /// Upsert an instance.
    Instance(InstancePatch),
    /// Upsert a material. Boxed: the patch is an order of magnitude wider
    /// than any other, and importers build long op lists.
    Material(Box<MaterialPatch>),
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
            Self::Instance(patch) => (Kind::Instance, &patch.name),
            Self::Material(patch) => (Kind::Material, &patch.name),
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

/// Patch for an [`Instance`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct InstancePatch {
    /// Target name.
    pub name: String,
    /// Mesh reference, by name.
    pub mesh: Option<String>,
    /// Material reference, by name.
    pub material: Option<String>,
    /// Object-to-world placement.
    pub transform: Option<Transform>,
    /// Whether camera rays see it.
    pub camera_visible: Option<bool>,
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
        ]
        .into_iter()
        .flatten()
    }
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
    /// The equirect radiance EXR.
    pub path: Option<PathBuf>,
}

/// Patch for [`Settings`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SettingsPatch {
    /// Target name.
    pub name: String,
    /// Output width × height, pixels.
    pub resolution: Option<[u32; 2]>,
    /// Samples per pixel for batch renders.
    pub spp: Option<u32>,
    /// Maximum path length in bounces.
    pub max_bounces: Option<u32>,
    /// Sampler seed.
    pub seed: Option<u32>,
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
    InstancePatch,
    MaterialPatch,
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
                Op::Material(patch) => {
                    for texture in patch.textures_mut() {
                        visit(&mut texture.path);
                    }
                }
                Op::Environment(patch) => {
                    if let Some(path) = &mut patch.path {
                        visit(path);
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
        Op::Instance(patch) => upsert(&mut objects.instances, &name, |instance| {
            merge!(instance, patch; mesh, material, transform, camera_visible);
        }),
        Op::Material(patch) => upsert(&mut objects.materials, &name, |material| {
            merge!(material, patch;
                base_color, base_diffuse_roughness, base_metalness,
                specular_weight, specular_roughness, specular_ior,
                transmission_weight, transmission_color, transmission_depth,
                coat_weight, coat_color, coat_roughness, coat_ior, coat_darkening,
                fuzz_weight, fuzz_color, fuzz_roughness,
                emission_luminance, emission_color,
                geometry_opacity, geometry_thin_walled, geometry_normal,
            );
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
            merge!(environment, patch; path);
        }),
        Op::Settings(patch) => upsert(&mut objects.settings, &name, |settings| {
            merge!(settings, patch; resolution, spp, max_bounces, seed);
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
            Kind::Instance => self.instances.remove(name).is_some(),
            Kind::Material => self.materials.remove(name).is_some(),
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
        validate_path(&format!("environment \"{name}\""), &environment.path)?;
    }
    for (name, settings) in &objects.settings {
        validate_settings(name, settings)?;
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
    }
}

fn validate_instance(objects: &Objects, name: &str, instance: &Instance) -> Result<()> {
    validate_reference(&objects.meshes, "mesh", name, &instance.mesh)?;
    validate_reference(&objects.materials, "material", name, &instance.material)?;
    let matrix = instance.transform.to_mat4();
    if !(matrix.is_finite() && matrix.inverse().is_finite()) {
        return Err(scene_error(format!(
            "instance \"{name}\" has a non-invertible transform"
        )));
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

fn validate_settings(name: &str, settings: &Settings) -> Result<()> {
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

    #[test]
    fn referenced_files_must_exist_and_be_absolute() {
        let mut description = SceneDescription::new();
        let relative = ChangeSet {
            ops: vec![Op::Environment(EnvironmentPatch {
                path: Some("sky.exr".into()),
                ..EnvironmentPatch::new("sky")
            })],
        };
        let error = description.apply(&relative).unwrap_err();
        assert!(error.to_string().contains("relative"), "{error}");

        let missing = ChangeSet {
            ops: vec![Op::Environment(EnvironmentPatch {
                path: Some("/no/such/sky.exr".into()),
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

    #[test]
    fn singular_transforms_are_rejected() {
        let mut description = SceneDescription::new();
        let mut set = triangle_scene();
        let Op::Instance(instance) = &mut set.ops[0] else {
            panic!("triangle_scene changed shape");
        };
        instance.transform = Some(Transform::Trs {
            translate: [0.0; 3],
            rotate_degrees: [0.0; 3],
            scale: [1.0, 0.0, 1.0],
        });
        let error = description.apply(&set).unwrap_err();
        assert!(error.to_string().contains("non-invertible"), "{error}");
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
                    path: Some("/already/absolute.exr".into()),
                    ..EnvironmentPatch::new("sky")
                }),
                Op::Material(Box::new(MaterialPatch {
                    base_color: Some(Texturable::Texture(TextureRef {
                        path: "wood.png".into(),
                        color_space: None,
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
        assert_eq!(environment.path, Some("/already/absolute.exr".into()));
        let Op::Material(material) = &set.ops[2] else {
            unreachable!()
        };
        assert_eq!(
            material.base_color.as_ref().and_then(Texturable::texture),
            Some(&TextureRef {
                path: "/scenes/wood.png".into(),
                color_space: None
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
                    path: Some("/elsewhere/sky.exr".into()),
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
        assert_eq!(environment.path, Some("/elsewhere/sky.exr".into()));

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
}
