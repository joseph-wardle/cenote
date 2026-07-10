//! The scene description: the typed, named object model that scene files,
//! the pbrt importer, and the lookdev panel all speak. A description is
//! plain data — no GPU state — and [`SceneDescription::apply`] (in
//! [`super::changeset`]) is its *only* mutation path, so every consumer of
//! an edit sees the same dirty accounting.
//!
//! The model is a closed set of seven object kinds — mesh, instance,
//! material, light, camera, environment, settings — each a map of objects
//! addressed by name. Names are stable identities: patches target objects
//! by name (creating them on first mention), references between objects
//! (instance → mesh, instance → material) are names, and a rename is a
//! remove plus a create. A description may hold any number of objects of
//! any kind; it is prep, not the description, that requires exactly one
//! camera and settings (and at most one environment) to render.
//!
//! Conventions the format commits to: right-handed, Y-up, meters, vertical
//! field of view in degrees. Color constants are **linear `Rec.709`**;
//! conversion to the `ACEScg` working space happens at prep — the same
//! ownership rule as textures, which store source-space values and convert
//! on the way into the renderer. Material parameters mirror `OpenPBR`'s
//! slugs (`base_color`, `coat_weight`, …) and defaults exactly.
//!
//! Every filesystem path in an applied description is absolute:
//! [`crate::format::load`] rebases a scene file's relative paths against
//! the file's own directory, and `apply` rejects any path still relative —
//! so the working directory can never leak into path resolution.

use std::collections::BTreeMap;
use std::path::PathBuf;

use glam::Mat4;
use serde::{Deserialize, Serialize};

use super::changeset::Dirty;

/// A material parameter that is either one value everywhere or a per-hit
/// texture lookup.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Texturable<T> {
    /// The same value across the whole surface.
    Constant(T),
    /// Sampled from an image at the hit's UV.
    Texture(TextureRef),
}

impl<T> Texturable<T> {
    /// The texture reference, if this parameter is textured.
    #[must_use]
    pub fn texture(&self) -> Option<&TextureRef> {
        match self {
            Self::Constant(_) => None,
            Self::Texture(reference) => Some(reference),
        }
    }

    /// Mutable access to the texture reference, if this parameter is
    /// textured — how path rebasing reaches into a patch.
    pub fn texture_mut(&mut self) -> Option<&mut TextureRef> {
        match self {
            Self::Constant(_) => None,
            Self::Texture(reference) => Some(reference),
        }
    }
}

/// A reference to an image file feeding a material parameter.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextureRef {
    /// The image file. Relative paths in a scene file mean
    /// file's-directory-relative and are rebased at load; by the time a
    /// change-set applies, the path must be absolute.
    pub path: PathBuf,
    /// Color-space override. `None` derives it from the slot: color slots
    /// read 8-bit images as sRGB and float images as linear; data and
    /// normal slots are always linear (pbrt's 8-bit-defaults-sRGB rule
    /// maps straight onto this).
    #[serde(default)]
    pub color_space: Option<ColorSpace>,
}

/// How an image's stored values map to linear light.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorSpace {
    /// sRGB-encoded values, linearized on sampling.
    Srgb,
    /// Values are already linear.
    Linear,
}

/// An object-to-world placement. Two spellings of the same thing: `Trs`
/// for hand-authored scenes, `Matrix` for imported ones — both must be
/// invertible (normals and ray offsets transform through the inverse).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Transform {
    /// Translate · rotate · scale, applied to a point in that
    /// reverse-reading order: scale first, then rotation about X, then Y,
    /// then Z (world axes, degrees), then translation.
    Trs {
        /// Translation, meters.
        #[serde(default = "zero3")]
        translate: [f32; 3],
        /// Rotation angles in degrees about the world X, Y, Z axes,
        /// applied in that order.
        #[serde(default = "zero3")]
        rotate_degrees: [f32; 3],
        /// Per-axis scale factors.
        #[serde(default = "one3")]
        scale: [f32; 3],
    },
    /// The top three rows of an affine matrix (translation in the last
    /// column; the implied bottom row is `0 0 0 1`).
    Matrix([[f32; 4]; 3]),
}

fn zero3() -> [f32; 3] {
    [0.0; 3]
}

fn one3() -> [f32; 3] {
    [1.0; 3]
}

impl Default for Transform {
    fn default() -> Self {
        Self::Trs {
            translate: zero3(),
            rotate_degrees: zero3(),
            scale: one3(),
        }
    }
}

impl Transform {
    /// The transform as a matrix, ready for prep.
    #[must_use]
    pub fn to_mat4(&self) -> Mat4 {
        match self {
            Self::Trs {
                translate,
                rotate_degrees,
                scale,
            } => {
                let [rx, ry, rz] = rotate_degrees.map(f32::to_radians);
                Mat4::from_translation((*translate).into())
                    * Mat4::from_rotation_z(rz)
                    * Mat4::from_rotation_y(ry)
                    * Mat4::from_rotation_x(rx)
                    * Mat4::from_scale((*scale).into())
            }
            Self::Matrix(rows) => {
                // The rows shape is row-major; glam is column-major, so
                // assemble the transpose.
                Mat4::from_cols_array_2d(&[rows[0], rows[1], rows[2], [0.0, 0.0, 0.0, 1.0]])
                    .transpose()
            }
        }
    }
}

/// A named triangle mesh: its geometry payload, inline or by reference.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mesh {
    /// Where the triangles come from.
    pub source: MeshSource,
}

/// A mesh's geometry payload. Small meshes stay inline — self-contained
/// and diffable in the scene file; big geometry lives in PLY, the format
/// everyone already has (D-056's bulk-data line).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum MeshSource {
    /// Geometry spelled out in the scene file.
    Inline {
        /// Vertex positions, meters, object space.
        positions: Vec<[f32; 3]>,
        /// Unit shading normals, one per position. Absent means prep
        /// derives them (smooth, area-weighted) — pbrt meshes often
        /// carry none.
        #[serde(default)]
        normals: Option<Vec<[f32; 3]>>,
        /// Texture coordinates, one per position. Absent means the mesh
        /// has no textured lookups.
        #[serde(default)]
        uvs: Option<Vec<[f32; 2]>>,
        /// Counter-clockwise-outward index triples into `positions`.
        triangles: Vec<[u32; 3]>,
    },
    /// Geometry loaded from a PLY file at prep. Apply only checks the
    /// file exists; parsing it is prep's job.
    Ply {
        /// The `.ply` file (absolute once applied, like every path).
        path: PathBuf,
    },
}

impl Default for MeshSource {
    /// An empty inline payload — the get-or-create placeholder. It never
    /// survives an apply: validation rejects meshes with no geometry.
    fn default() -> Self {
        Self::Inline {
            positions: Vec::new(),
            normals: None,
            uvs: None,
            triangles: Vec::new(),
        }
    }
}

/// One thing standing in the scene: a mesh placed by a transform, wearing
/// a material — both referenced by name.
#[derive(Clone, Debug, PartialEq)]
pub struct Instance {
    /// Name of the [`Mesh`] this instance places. Every instance must
    /// name one; the empty default only exists so get-or-create has a
    /// value, and validation rejects it.
    pub mesh: String,
    /// Name of the [`Material`] on its surface (same rule as `mesh`).
    pub material: String,
    /// Object-to-world placement.
    pub transform: Transform,
    /// Whether camera rays see this instance. `false` is the classic
    /// invisible-emitter trick: the light still illuminates, but never
    /// appears in frame. Bounce rays always see everything — the full
    /// per-ray-type visibility set is a deferral.
    pub camera_visible: bool,
}

impl Default for Instance {
    fn default() -> Self {
        Self {
            mesh: String::new(),
            material: String::new(),
            transform: Transform::default(),
            camera_visible: true,
        }
    }
}

/// An `OpenPBR` surface. Field names and defaults mirror the `OpenPBR`
/// v1.1.1 slugs exactly — interop alignment as a commitment. This is the
/// authoring-side schema; `crate::material::Material` is its GPU-resident
/// counterpart, and prep maps one onto the other (the M2 closure work
/// widens that mapping lobe by lobe).
///
/// All colors are linear `Rec.709` (module doc); weights live in [0, 1].
/// Out-of-range values are not validation errors — prep and the kernels
/// clamp where physics demands, matching how `OpenPBR` itself specifies
/// soft parameter ranges.
#[derive(Clone, Debug, PartialEq)]
pub struct Material {
    /// Diffuse albedo — and the conductor's F0 as `base_metalness` rises.
    /// Default 0.8 gray.
    pub base_color: Texturable<[f32; 3]>,
    /// Diffuse (Oren-Nayar) roughness; 0 is Lambert.
    pub base_diffuse_roughness: f32,
    /// Conductor blend: 0 dielectric base, 1 pure metal.
    pub base_metalness: Texturable<f32>,
    /// Weight of the dielectric specular layer; 0 removes it. Default 1.
    pub specular_weight: f32,
    /// GGX roughness of both specular lobes (conductor and dielectric).
    /// Default 0.3.
    pub specular_roughness: Texturable<f32>,
    /// Index of refraction of the dielectric specular layer and of
    /// transmission. Default 1.5.
    pub specular_ior: f32,
    /// Weight of the transmissive (glass) lobe; 0 is opaque. Default 0.
    pub transmission_weight: f32,
    /// The color transmitted light has picked up after traveling
    /// `transmission_depth` through the interior (Beer–Lambert). White
    /// transmits everything.
    pub transmission_color: [f32; 3],
    /// Distance in meters at which `transmission_color` is reached; 0
    /// applies the tint at the interface itself.
    pub transmission_depth: f32,
    /// Weight of the clear coat layer. Default 0.
    pub coat_weight: f32,
    /// Tint the coat multiplies onto the base. White is untinted.
    pub coat_color: [f32; 3],
    /// GGX roughness of the coat lobe. Default 0.
    pub coat_roughness: f32,
    /// Index of refraction of the coat. Default 1.6.
    pub coat_ior: f32,
    /// How strongly the coat's internal reflections darken and saturate
    /// the base, 0 (off) to 1 (physical). Default 1.
    pub coat_darkening: f32,
    /// Weight of the fuzz (sheen) lobe. Default 0.
    pub fuzz_weight: f32,
    /// Fuzz color. White is neutral fiber scatter.
    pub fuzz_color: [f32; 3],
    /// Fuzz fiber roughness. Default 0.5.
    pub fuzz_roughness: f32,
    /// Emitted luminance scale in the working space's units — the linear
    /// radiance the tonemap's ~nit convention reads; `OpenPBR`'s literal
    /// photometric reading applies once physical camera exposure exists.
    /// Nonzero marks the instance as a light.
    pub emission_luminance: f32,
    /// Emission tint (or map — the LDR-map × `emission_luminance` scale
    /// convention). Default white.
    pub emission_color: Texturable<[f32; 3]>,
    /// Coverage: 1 opaque, 0 invisible. Fractional or textured opacity is
    /// resolved stochastically on camera and bounce rays, multiplicatively
    /// on shadow rays.
    pub geometry_opacity: Texturable<f32>,
    /// Thin-walled surfaces (leaves, soap bubbles, paper) have no
    /// interior: transmission passes straight through without refraction
    /// or Beer–Lambert.
    pub geometry_thin_walled: bool,
    /// Tangent-space normal map, if any.
    pub geometry_normal: Option<TextureRef>,
}

impl Default for Material {
    /// `OpenPBR`'s own defaults, field for field.
    fn default() -> Self {
        Self {
            base_color: Texturable::Constant([0.8; 3]),
            base_diffuse_roughness: 0.0,
            base_metalness: Texturable::Constant(0.0),
            specular_weight: 1.0,
            specular_roughness: Texturable::Constant(0.3),
            specular_ior: 1.5,
            transmission_weight: 0.0,
            transmission_color: [1.0; 3],
            transmission_depth: 0.0,
            coat_weight: 0.0,
            coat_color: [1.0; 3],
            coat_roughness: 0.0,
            coat_ior: 1.6,
            coat_darkening: 1.0,
            fuzz_weight: 0.0,
            fuzz_color: [1.0; 3],
            fuzz_roughness: 0.5,
            emission_luminance: 0.0,
            emission_color: Texturable::Constant([1.0; 3]),
            geometry_opacity: Texturable::Constant(1.0),
            geometry_thin_walled: false,
            geometry_normal: None,
        }
    }
}

impl Material {
    /// Every texture reference this material holds — validation walks
    /// these, and prep will collect them for upload.
    pub(crate) fn textures(&self) -> impl Iterator<Item = &TextureRef> {
        [
            self.base_color.texture(),
            self.base_metalness.texture(),
            self.specular_roughness.texture(),
            self.emission_color.texture(),
            self.geometry_opacity.texture(),
            self.geometry_normal.as_ref(),
        ]
        .into_iter()
        .flatten()
    }
}

/// A delta light — zero area, so next-event estimation is its only
/// sampling strategy (MIS weight 1). Area lighting is not a light object:
/// any instance whose material emits is an emitter.
///
/// Patched wholesale rather than per-field: a light is a handful of
/// numbers, and its variant *is* its identity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Light {
    /// Parallel light from infinitely far away — the sun.
    Distant {
        /// Unit-defining direction the light *travels* (from the light
        /// toward the scene); normalized at prep.
        direction: [f32; 3],
        /// Irradiance delivered on a surface facing the light, W/m² as
        /// linear `Rec.709`.
        irradiance: [f32; 3],
    },
    /// An isotropic point.
    Point {
        /// Position, meters, world space.
        position: [f32; 3],
        /// Radiant intensity, W/sr as linear `Rec.709`.
        intensity: [f32; 3],
    },
}

impl Default for Light {
    /// A black point light at the origin — the get-or-create placeholder;
    /// harmless if it survives, visible in any inspector.
    fn default() -> Self {
        Self::Point {
            position: [0.0; 3],
            intensity: [0.0; 3],
        }
    }
}

/// A thin-lens camera (pinhole at zero aperture), described by where it
/// sits and what it looks at; `up` carries roll.
#[derive(Clone, Debug, PartialEq)]
pub struct Camera {
    /// Eye position, meters.
    pub position: [f32; 3],
    /// The point the view axis passes through.
    pub look_at: [f32; 3],
    /// Which way is up on screen — need not be exactly perpendicular to
    /// the view axis, just not parallel to it. Default +Y.
    pub up: [f32; 3],
    /// Vertical field of view, degrees, in (0, 180).
    pub vfov_degrees: f32,
    /// Distance to the focal plane, meters. `None` focuses at `look_at`.
    pub focus_distance: Option<f32>,
    /// Lens radius, meters; 0 is a pinhole (everything sharp). Default 0.
    pub aperture_radius: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 5.0],
            look_at: [0.0; 3],
            up: [0.0, 1.0, 0.0],
            vfov_degrees: 40.0,
            focus_distance: None,
            aperture_radius: 0.0,
        }
    }
}

/// The environment light: an equirect EXR surrounding the scene.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Environment {
    /// The radiance image — linear, equirect, `.exr`. Never empty in a
    /// valid description.
    pub path: PathBuf,
}

/// Render settings — the minimal set, so the format doesn't churn while
/// the render loop learns to read it.
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    /// Output width × height, pixels.
    pub resolution: [u32; 2],
    /// Samples per pixel a batch render accumulates (the viewer
    /// accumulates forever regardless).
    pub spp: u32,
    /// Maximum path length in bounces.
    pub max_bounces: u32,
    /// Sampler seed, decorrelating repeat renders; wired to the sampler
    /// when scene loading reaches the render loop.
    pub seed: u32,
}

impl Default for Settings {
    /// The CLI's defaults, so a scene file that says nothing renders the
    /// same as `cenote-cli` with no flags.
    fn default() -> Self {
        Self {
            resolution: [1280, 720],
            spp: 64,
            max_bounces: crate::wavefront::Wavefront::DEFAULT_MAX_BOUNCES,
            seed: 0,
        }
    }
}

/// The seven object maps — the description's entire contents, split out
/// so apply can clone, mutate, validate, and swap them atomically.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Objects {
    pub(crate) meshes: BTreeMap<String, Mesh>,
    pub(crate) instances: BTreeMap<String, Instance>,
    pub(crate) materials: BTreeMap<String, Material>,
    pub(crate) lights: BTreeMap<String, Light>,
    pub(crate) cameras: BTreeMap<String, Camera>,
    pub(crate) environments: BTreeMap<String, Environment>,
    pub(crate) settings: BTreeMap<String, Settings>,
}

/// A whole scene as data, plus the dirty state its edits have accumulated
/// since prep last looked. Starts empty; every mutation goes through
/// [`SceneDescription::apply`].
///
/// Iteration order everywhere is name order (`BTreeMap`), so the same
/// description always preps into the same GPU layout — the determinism
/// invariant extends to scene loading.
#[derive(Debug, Default)]
pub struct SceneDescription {
    pub(crate) objects: Objects,
    pub(crate) dirty: Dirty,
}

impl SceneDescription {
    /// An empty scene.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The meshes, by name.
    #[must_use]
    pub fn meshes(&self) -> &BTreeMap<String, Mesh> {
        &self.objects.meshes
    }

    /// The instances, by name.
    #[must_use]
    pub fn instances(&self) -> &BTreeMap<String, Instance> {
        &self.objects.instances
    }

    /// The materials, by name.
    #[must_use]
    pub fn materials(&self) -> &BTreeMap<String, Material> {
        &self.objects.materials
    }

    /// The delta lights, by name.
    #[must_use]
    pub fn lights(&self) -> &BTreeMap<String, Light> {
        &self.objects.lights
    }

    /// The cameras, by name.
    #[must_use]
    pub fn cameras(&self) -> &BTreeMap<String, Camera> {
        &self.objects.cameras
    }

    /// The environments, by name.
    #[must_use]
    pub fn environments(&self) -> &BTreeMap<String, Environment> {
        &self.objects.environments
    }

    /// The render settings, by name.
    #[must_use]
    pub fn settings(&self) -> &BTreeMap<String, Settings> {
        &self.objects.settings
    }

    /// Hand over the accumulated dirty state, leaving none — prep calls
    /// this to learn what to rebuild.
    pub fn take_dirty(&mut self) -> Dirty {
        std::mem::take(&mut self.dirty)
    }
}

#[cfg(test)]
mod tests {
    use glam::{Mat4, Vec3};

    use super::*;

    #[test]
    fn trs_composes_scale_then_rotation_then_translation() {
        let transform = Transform::Trs {
            translate: [1.0, 2.0, 3.0],
            rotate_degrees: [0.0, 90.0, 0.0],
            scale: [2.0; 3],
        };
        // +X scaled to length 2, rotated 90° about Y onto −Z, then moved.
        let p = transform.to_mat4().transform_point3(Vec3::X);
        assert!(p.abs_diff_eq(Vec3::new(1.0, 2.0, 1.0), 1e-5), "{p}");
    }

    #[test]
    fn matrix_rows_round_trip_through_glam() {
        let reference = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0))
            * Mat4::from_rotation_x(0.5)
            * Mat4::from_scale(Vec3::new(2.0, 1.0, 0.5));
        let rows = reference.transpose().to_cols_array_2d();
        let transform = Transform::Matrix([rows[0], rows[1], rows[2]]);
        assert!(transform.to_mat4().abs_diff_eq(reference, 1e-6));
    }

    #[test]
    fn default_transform_is_identity() {
        assert_eq!(Transform::default().to_mat4(), Mat4::IDENTITY);
    }

    #[test]
    fn textures_walks_every_textured_slot() {
        let mut material = Material {
            base_color: Texturable::Texture(TextureRef {
                path: "color.png".into(),
                color_space: None,
            }),
            geometry_normal: Some(TextureRef {
                path: "normal.png".into(),
                color_space: None,
            }),
            ..Material::default()
        };
        assert_eq!(material.textures().count(), 2);
        material.geometry_opacity = Texturable::Texture(TextureRef {
            path: "mask.png".into(),
            color_space: None,
        });
        assert_eq!(material.textures().count(), 3);
        assert_eq!(Material::default().textures().count(), 0);
    }
}
