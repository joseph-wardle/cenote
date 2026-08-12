//! The scene half of the wire: a full 1:1 mirror of the renderer's
//! change-set schema (`cenote::scene::changeset`), field for field.
//!
//! Two deliberate spelling differences from the originals, both forced by
//! the wire itself:
//!
//! - **Paths are `String`**, not `PathBuf`: `MessagePack` has strings, C++
//!   has `std::string`, and a path that isn't valid UTF-8 has no portable
//!   wire form. The server turns them into `PathBuf`s on arrival.
//! - **Doubly-optional fields use [`Reset`]** instead of `Option<Option<T>>`:
//!   serde encodes `Some(None)` and `None` to the same `MessagePack` nil, so
//!   the "clear this back to its default" edit would silently decode as
//!   "leave it alone". `Reset` is an enum, whose variants stay distinct.
//!
//! Everything else — names, field order, variant order — matches the
//! renderer exactly, and the semantics ride along: a patch targets an
//! object by name (get-or-create), only `Some` fields overwrite, removal
//! errors if the target is missing or a reference would strand. See the
//! originals for per-field meaning; duplicating those docs here would
//! only let the two drift.

use serde::{Deserialize, Serialize};

/// The seven object kinds — mirror of `changeset::Kind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    /// Triangle geometry.
    Mesh,
    /// A placed mesh with a material.
    Instance,
    /// An `OpenPBR` surface.
    Material,
    /// A delta light.
    Light,
    /// A viewpoint.
    Camera,
    /// The surrounding light image.
    Environment,
    /// Render settings.
    Settings,
}

/// One edit — mirror of `changeset::Op`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Op {
    /// Upsert a mesh.
    Mesh(MeshPatch),
    /// Upsert an instance.
    Instance(InstancePatch),
    /// Upsert a material. Boxed like the original: the patch is an order
    /// of magnitude wider than any other, and hosts build long op lists.
    Material(Box<MaterialPatch>),
    /// Upsert a delta light.
    Light(LightPatch),
    /// Upsert a camera.
    Camera(CameraPatch),
    /// Upsert an environment.
    Environment(EnvironmentPatch),
    /// Upsert render settings.
    Settings(SettingsPatch),
    /// Delete an object outright (renames arrive as remove + re-insert).
    Remove(Kind, String),
}

/// An ordered list of edits, applied atomically — mirror of
/// `changeset::ChangeSet`, the payload of `Replace` and `Apply`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ChangeSet {
    /// The edits, in application order.
    pub ops: Vec<Op>,
}

/// The wire spelling of the renderer's doubly-optional patch fields
/// (`Option<Option<T>>`). An enum because `MessagePack` cannot tell
/// `Some(None)` from `None` — serde flattens both to nil — and the
/// difference is an edit: `None` (absent) leaves the field alone, where
/// `Some(Reset::Clear)` restores its default.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Reset<T> {
    /// Clear back to the default — the renderer's `Some(None)`.
    Clear,
    /// Set the value — the renderer's `Some(Some(value))`.
    Set(T),
}

/// A constant-or-textured material parameter — mirror of
/// `description::Texturable`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Texturable<T> {
    /// The same value across the whole surface.
    Constant(T),
    /// Sampled from an image at the hit's UV.
    Texture(TextureRef),
}

/// A reference to an image file — mirror of `description::TextureRef`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TextureRef {
    /// The image file. Relative paths are resolved server-side against
    /// nothing — send absolute paths.
    pub path: String,
    /// Color-space override; `None` derives it from the slot.
    pub color_space: Option<ColorSpace>,
    /// The source channel a scalar slot reads; `None` means red. Color
    /// and normal slots ignore it.
    pub channel: Option<Channel>,
}

/// One channel of a source image — mirror of `description::Channel`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Channel {
    /// The red component (the default when unstated).
    R,
    /// The green component.
    G,
    /// The blue component.
    B,
    /// The alpha component.
    A,
}

/// How an image's stored values map to linear light — mirror of
/// `description::ColorSpace`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorSpace {
    /// sRGB-encoded values, linearized on sampling.
    Srgb,
    /// Values are already linear.
    Linear,
}

/// An object-to-world placement — mirror of `description::Transform`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Transform {
    /// Translate · rotate · scale (scale first; rotations about world
    /// X, then Y, then Z, in degrees; translation last).
    Trs {
        /// Translation, meters.
        translate: [f32; 3],
        /// Rotation angles in degrees about the world X, Y, Z axes.
        rotate_degrees: [f32; 3],
        /// Per-axis scale factors.
        scale: [f32; 3],
    },
    /// The top three rows of an affine matrix (translation in the last
    /// column; the implied bottom row is `0 0 0 1`).
    Matrix([[f32; 4]; 3]),
}

/// A mesh's geometry payload — mirror of `description::MeshSource`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum MeshSource {
    /// Geometry spelled out in the message — the shape a scene-index
    /// observer's topology arrays translate into directly.
    Inline {
        /// Vertex positions, meters, object space.
        positions: Vec<[f32; 3]>,
        /// Unit shading normals, one per position; absent means the
        /// server derives them (smooth, area-weighted).
        normals: Option<Vec<[f32; 3]>>,
        /// Texture coordinates, one per position.
        uvs: Option<Vec<[f32; 2]>>,
        /// Counter-clockwise-outward index triples into `positions`.
        triangles: Vec<[u32; 3]>,
    },
    /// Geometry loaded server-side from a PLY file.
    Ply {
        /// The `.ply` file, absolute.
        path: String,
    },
}

/// A delta light's definition — mirror of `description::Light`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Light {
    /// Parallel light from infinitely far away.
    Distant {
        /// Direction the light travels (light toward scene).
        direction: [f32; 3],
        /// Irradiance on a facing surface, W/m² linear `Rec.709`.
        irradiance: [f32; 3],
    },
    /// An isotropic point.
    Point {
        /// Position, meters, world space.
        position: [f32; 3],
        /// Radiant intensity, W/sr linear `Rec.709`.
        intensity: [f32; 3],
    },
}

/// Mirror of `changeset::MeshPatch`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MeshPatch {
    /// Target name.
    pub name: String,
    /// New geometry payload.
    pub source: Option<MeshSource>,
}

/// Mirror of `changeset::InstancePatch`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct InstancePatch {
    /// Target name.
    pub name: String,
    /// Mesh reference, by name.
    pub mesh: Option<String>,
    /// Material reference, by name.
    pub material: Option<String>,
    /// Object-to-world placements, one per copy; the whole array
    /// replaces (`[]` legal — resident, places nothing).
    pub transforms: Option<Vec<Transform>>,
    /// Whether camera rays see it.
    pub camera_visible: Option<bool>,
}

/// Mirror of `changeset::MaterialPatch`, field for field — the surface
/// the drift guard exists for. Field meanings and defaults are
/// `OpenPBR`'s; see the renderer's `description::Material`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[expect(missing_docs, reason = "fields document themselves on the original")]
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
    /// Doubly optional in the renderer: absent leaves the normal map
    /// alone, [`Reset::Clear`] removes it.
    pub geometry_normal: Option<Reset<TextureRef>>,
}

/// Mirror of `changeset::LightPatch`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LightPatch {
    /// Target name.
    pub name: String,
    /// New definition (a delta light replaces wholesale).
    pub light: Option<Light>,
}

/// Mirror of `changeset::CameraPatch`. Note the *active* camera never
/// travels as scene data — the session's inputs-lane camera overwrites the
/// scene camera at every wave, so a `CameraPatch` cannot move the view.
/// Use `Request::SetCamera`; this patch exists because the mirror is total.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
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
    /// Doubly optional in the renderer: absent leaves focus alone,
    /// [`Reset::Clear`] restores focus-at-`look_at`.
    pub focus_distance: Option<Reset<f32>>,
    /// Lens radius, meters; 0 is a pinhole.
    pub aperture_radius: Option<f32>,
}

/// Mirror of `changeset::EnvironmentPatch`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EnvironmentPatch {
    /// Target name.
    pub name: String,
    /// The equirect radiance image (`.exr` or `.hdr`), absolute. Doubly
    /// optional in the renderer: absent leaves the image alone,
    /// [`Reset::Clear`] restores the constant white sky.
    pub path: Option<Reset<String>>,
    /// Linear `Rec.709` multiplier over the sky's radiance.
    pub tint: Option<[f32; 3]>,
    /// Environment-to-world placement (the linear part turns the sky).
    pub transform: Option<Transform>,
}

/// Mirror of `changeset::SettingsPatch`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SettingsPatch {
    /// Target name.
    pub name: String,
    /// Output width × height, pixels.
    pub resolution: Option<[u32; 2]>,
    /// The sample budget: samples per pixel before the render is done. A
    /// hard cap — `noise_threshold` only ever stops it earlier.
    pub spp: Option<u32>,
    /// Relative estimator standard error to stop early at, in (0, 1].
    /// Doubly optional like the camera's focus: absent leaves it alone,
    /// [`Reset::Clear`] turns the early-out off and spends the whole
    /// budget.
    pub noise_threshold: Option<Reset<f32>>,
    /// Maximum path length in bounces.
    pub max_bounces: Option<u32>,
    /// Sampler seed.
    pub seed: Option<u32>,
}
