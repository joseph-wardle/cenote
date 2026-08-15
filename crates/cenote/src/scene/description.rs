//! The scene description: the typed, named object model that scene files,
//! the pbrt importer, and the lookdev panel all speak. A description is
//! plain data — no GPU state — and [`SceneDescription::apply`] (in
//! [`super::changeset`]) is its *only* mutation path, so every consumer of
//! an edit sees the same dirty accounting.
//!
//! The model is a closed set of nine object kinds — mesh, curves,
//! instance, material, medium, light, camera, environment, settings —
//! each a map of
//! objects addressed by name. Names are stable identities: patches target objects
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
use std::fmt;
use std::path::PathBuf;

use glam::Mat4;
use serde::{Deserialize, Serialize};

use super::changeset::{Dirty, Kind};

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
    /// The source channel a scalar slot reads; `None` means red. This is
    /// how one packed image serves several masks (roughness from green,
    /// metalness from blue) and an RGBA color's alpha drives opacity.
    /// Color and normal slots ignore it.
    #[serde(default)]
    pub channel: Option<Channel>,
    /// A uniform multiplier folded over every sampled value (pbrt's
    /// imagemap `scale`) — applied after color-space decode, before the
    /// working-space conversion, so it scales linear light. `None` means 1.
    /// Normal slots ignore it: a normal is a direction, not a quantity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<f32>,
    /// An affine remap of the mesh UV before sampling, in this schema's
    /// storage convention (`v = 0` is the image's top row): the sampled
    /// coordinate is `uv * scale + offset`, wrapping. `None` is the
    /// identity. Importers converting from a v-up convention (pbrt's
    /// uscale/vscale/udelta/vdelta) flip the v leg at the boundary, like
    /// the UVs themselves.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uv: Option<UvTransform>,
}

/// The affine UV remap a [`TextureRef`] may carry: scale about the origin,
/// then offset.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UvTransform {
    /// Per-axis multiplier over the mesh UV (tiling above 1).
    #[serde(default = "one2")]
    pub scale: [f32; 2],
    /// Added after the scale.
    #[serde(default = "zero2")]
    pub offset: [f32; 2],
}

impl Default for UvTransform {
    fn default() -> Self {
        Self {
            scale: [1.0; 2],
            offset: [0.0; 2],
        }
    }
}

/// One channel of a source image — which component a scalar slot samples.
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

fn one() -> f32 {
    1.0
}

fn zero2() -> [f32; 2] {
    [0.0; 2]
}

fn one2() -> [f32; 2] {
    [1.0; 2]
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
/// everyone already has.
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
    /// The auto-generated shell of a heterogeneous medium: a canonical unit
    /// cube whose per-placement transform prep derives from the medium's
    /// grid bounds — the author never sizes it. An instance placing this
    /// mesh must bound a [`Medium`] with a [`volume`](Medium::volume), and
    /// vice versa; either half alone is a validation error, because a
    /// grid's extent is its own active bounds and nothing else
    /// (mesh-clipped volumes are a non-goal — clip in the DCC).
    MediumBounds,
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

/// A named batch of curves — the `BasisCurves` prim, which is how every
/// DCC in the pipeline spells hair, fur, grass, and fiber.
///
/// Curves are a *geometry kind*, not a mesh source: what travels is the
/// cells USD authored (points, vertex counts, widths, and the tokens that
/// say how to read them), and the renderer owns every piece of curve
/// mathematics from there — evaluation, flattening, and the tube sweep
/// alike (see [`super::curves`]). That is what keeps one groom's worth of
/// control points on the wire instead of a hundred times as many
/// triangles, and what leaves room for a native curve primitive to become
/// a second backend without the scene model noticing.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Curves {
    /// Where the strands come from.
    pub source: CurvesSource,
}

/// A curve batch's payload: cells in the scene file, or a groom read from
/// a `.hair` file at prep — the same split meshes make between inline
/// geometry and PLY, for the same reason.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum CurvesSource {
    /// `BasisCurves` cells, spelled out — the shape a scene-index
    /// observer's arrays translate into directly.
    Inline {
        /// Control vertices of every curve, end to end, meters, object
        /// space. `curve_vertex_counts` says where each curve's run
        /// starts and stops.
        points: Vec<[f32; 3]>,
        /// How many control vertices each curve holds. Its length is the
        /// number of curves; what counts are *valid* depends on the type,
        /// basis, and wrap (see `UsdGeomBasisCurves`' segment table,
        /// which validation applies verbatim).
        curve_vertex_counts: Vec<u32>,
        /// The strands' widths — diameters, not radii, like USD's own.
        /// Absent means one meter everywhere, which is `UsdGeomCurves`'
        /// fallback: unmistakable rather than invisible.
        #[serde(default)]
        widths: Option<Widths>,
        /// Whether the vertices are a polyline or a cubic control
        /// polygon. Default cubic, like USD.
        #[serde(default)]
        curve_type: CurveType,
        /// Which cubic basis the control polygon is read through; ignored
        /// for linear curves. Default bezier, like USD.
        #[serde(default)]
        basis: CurveBasis,
        /// How the ends behave. Default nonperiodic, like USD.
        #[serde(default)]
        wrap: CurveWrap,
    },
    /// A groom loaded from a `.hair` file at prep — polylines with a
    /// width at every point, so no basis applies. Apply only checks the
    /// file exists; reading it is prep's job.
    Hair {
        /// The `.hair` file (absolute once applied, like every path).
        path: PathBuf,
    },
}

impl Default for CurvesSource {
    /// An empty inline payload — the get-or-create placeholder. It never
    /// survives an apply: validation rejects curves with no vertices.
    fn default() -> Self {
        Self::Inline {
            points: Vec::new(),
            curve_vertex_counts: Vec::new(),
            widths: None,
            curve_type: CurveType::default(),
            basis: CurveBasis::default(),
            wrap: CurveWrap::default(),
        }
    }
}

/// The width stream of a curve batch, with the interpolation that says
/// how many values it holds and how they spread along a strand.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Widths {
    /// The widths themselves; how many there must be follows from
    /// `interpolation` and the topology.
    pub values: Vec<f32>,
    /// How they map onto the strands. Default `vertex`, which is
    /// `UsdGeomCurves`' own fallback when a file leaves it unstated.
    #[serde(default)]
    pub interpolation: WidthInterpolation,
}

/// How a width stream maps onto the curves it belongs to — the four
/// `UsdGeomPrimvar` interpolations that apply to curves.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum WidthInterpolation {
    /// One value for the whole batch.
    Constant,
    /// One value per curve.
    Uniform,
    /// One value per segment end, linear across each segment — neither
    /// per point nor per curve, the shape USD warns implementations
    /// routinely get wrong.
    Varying,
    /// One value per control vertex, interpolated through the curve's own
    /// basis. The default.
    #[default]
    Vertex,
}

/// Polyline or cubic — `UsdGeomBasisCurves`' `type`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CurveType {
    /// Vertices are the strand: a segment per consecutive pair.
    Linear,
    /// Vertices are a control polygon read through [`CurveBasis`]. The
    /// default, as in USD.
    #[default]
    Cubic,
}

/// Which cubic basis a control polygon is read through —
/// `UsdGeomBasisCurves`' `basis`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CurveBasis {
    /// Interpolating, four vertices per span, window sliding by three.
    /// The default, as in USD.
    #[default]
    Bezier,
    /// Approximating, window sliding by one — the smooth basis grooms are
    /// usually authored in.
    BSpline,
    /// Interpolating, window sliding by one.
    CatmullRom,
}

/// How a curve's ends behave — `UsdGeomBasisCurves`' `wrap`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CurveWrap {
    /// Open, and the basis is read from the vertices as authored. The
    /// default, as in USD.
    #[default]
    Nonperiodic,
    /// Open, with a phantom vertex mirrored off each end so the curve
    /// passes through its first and last vertex — how Houdini writes hair
    /// under an approximating basis. Meaningless for bezier, which
    /// already interpolates its ends, and treated as nonperiodic there.
    Pinned,
    /// Closed. **Refused at validation**: the renderer sweeps a strand
    /// from a root, and a loop has none. It is carried by the schema so a
    /// scene that authors one is told what it authored, rather than
    /// silently rendered as an open curve.
    Periodic,
}

impl fmt::Display for CurveType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Linear => "linear",
            Self::Cubic => "cubic",
        })
    }
}

impl fmt::Display for CurveBasis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Bezier => "bezier",
            Self::BSpline => "bspline",
            Self::CatmullRom => "catmullRom",
        })
    }
}

impl fmt::Display for CurveWrap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Nonperiodic => "nonperiodic",
            Self::Pinned => "pinned",
            Self::Periodic => "periodic",
        })
    }
}

/// What an instance places: geometry of one kind or the other, by name.
///
/// The two kinds keep separate maps — their schemas have nothing in
/// common — so the reference has to say which map it means. Everything
/// downstream of lowering treats them alike: curves resolve to triangles,
/// and a resolved batch is a mesh in every way that residency, the BLAS,
/// and the light table can see.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Geometry {
    /// A [`Mesh`], by name.
    Mesh(String),
    /// A [`Curves`] batch, by name.
    Curves(String),
}

impl Geometry {
    /// The name it references.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Mesh(name) | Self::Curves(name) => name,
        }
    }

    /// The `kind.name` stem the GPU debug labels are built from — and,
    /// through them, the memory ledger's buckets.
    pub(crate) fn label(&self) -> String {
        format!("{}.{}", self.kind(), self.name())
    }

    /// The kind it references — how dirty accounting and validation reach
    /// the right map.
    #[must_use]
    pub fn kind(&self) -> Kind {
        match self {
            Self::Mesh(_) => Kind::Mesh,
            Self::Curves(_) => Kind::Curves,
        }
    }
}

impl Default for Geometry {
    /// An unnamed mesh — the get-or-create placeholder, which validation
    /// rejects like any other unset reference.
    fn default() -> Self {
        Self::Mesh(String::new())
    }
}

impl fmt::Display for Geometry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} \"{}\"", self.kind(), self.name())
    }
}

/// One thing standing in the scene: geometry placed zero or more times,
/// wearing a material — both referenced by name.
#[derive(Clone, Debug, PartialEq)]
pub struct Instance {
    /// The [`Mesh`] or [`Curves`] this instance places. Every instance
    /// must name one; the empty default only exists so get-or-create has
    /// a value, and validation rejects it.
    pub geometry: Geometry,
    /// Name of the [`Material`] on its surface (same rule as `mesh`).
    pub material: String,
    /// Object-to-world placements, one per copy — the array-instancer
    /// shape: every element places the same mesh in the same
    /// material. Empty is legal and places nothing; the instance stays
    /// resident, ready for its next placements (a host's fully-masked
    /// instancer needs no remove/re-create round trip).
    pub transforms: Vec<Transform>,
    /// Whether camera rays see this instance. `false` is the classic
    /// invisible-emitter trick: the light still illuminates, but never
    /// appears in frame. Bounce rays always see everything — the full
    /// per-ray-type visibility set is a deferral.
    pub camera_visible: bool,
    /// The [`Medium`] this mesh *bounds*, by name — fog, smoke, a light
    /// shaft. Naming one makes the surface a null boundary: a pure medium
    /// extent with no surface response at all, so the material is unused
    /// (it still has to name one, like every instance) and the mesh should
    /// be closed and outward-wound. Overlapping volumes add, which is what
    /// makes their order irrelevant.
    ///
    /// Two limits, each degrading toward *less* fog rather than more: a
    /// path may be inside at most four media at once, volumes and
    /// refractive interiors sharing the four; and a refractive interior
    /// displaces every volume while the path is inside it, rather than
    /// nesting with them. A camera that starts inside one just works: the
    /// initial set is resolved by tracing at every restart (see
    /// `resolve_camera.slang`).
    pub medium: Option<String>,
    /// Which solid wins where two refractive interiors overlap: the higher
    /// number is the one that is really there, and the interface of the
    /// lower one *inside* it is cut away — crossed without being shaded,
    /// like a boolean subtraction. The classic case is a drink: model the
    /// liquid so it interpenetrates the glass wall, give the glass the
    /// higher priority, and the overlap renders as glass rather than as
    /// two interfaces and twice the absorption.
    ///
    /// 0..=63; the default 0 is the weakest and everything participates,
    /// which is why a scene that authors none renders exactly as it always
    /// has (all-equal is never *strictly* less, so no interface is ever
    /// false). Equal priorities both shade, which is the double refraction
    /// priority exists to avoid — set them, or accept it.
    ///
    /// It lives on the instance, not the material, because two things
    /// wearing one glass routinely need different priorities (the ice
    /// cubes and the tumbler), and because `OpenPBR` has no priority
    /// parameter for it to be a slug of. Overlap *generously* — surfaces
    /// that merely touch fall foul of the same coincidence limit volumes
    /// do.
    ///
    /// Only refractive surfaces have it; a volume's boundary or an opaque
    /// one carries it harmlessly and nothing reads it.
    pub interior_priority: u32,
}

impl Default for Instance {
    /// One identity placement — an instance never told where to stand
    /// stands once at the origin, as it always has.
    fn default() -> Self {
        Self {
            geometry: Geometry::default(),
            material: String::new(),
            transforms: vec![Transform::default()],
            camera_visible: true,
            medium: None,
            interior_priority: 0,
        }
    }
}

/// An `OpenPBR` surface. Field names and defaults mirror the `OpenPBR`
/// v1.1.1 slugs exactly — interop alignment as a commitment. This is the
/// authoring-side schema; `crate::scene::material::Material` is its GPU-resident
/// counterpart, and prep maps one onto the other (the closure work
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
    /// The part of that extinction which scatters rather than absorbs
    /// (juice, milk): `σ_s` = `transmission_scatter` / `transmission_depth`.
    /// Zero — the default — keeps the interior purely absorbing.
    pub transmission_scatter: [f32; 3],
    /// Henyey–Greenstein anisotropy of the interior's scattering; 0 is
    /// isotropic, positive leans forward.
    pub transmission_scatter_anisotropy: f32,
    /// Weight of the subsurface lobe; 0 — the default — removes it.
    pub subsurface_weight: Texturable<f32>,
    /// The color light has picked up after scattering many times through
    /// the interior — the multiple-scatter albedo prep inverts (van de
    /// Hulst) into a single-scatter `σ_s`. Default 0.8 gray.
    pub subsurface_color: Texturable<[f32; 3]>,
    /// Scattering mean free path in meters, before the per-channel scale.
    /// Default 1.
    pub subsurface_radius: Texturable<f32>,
    /// Per-channel multipliers on `subsurface_radius` — the chromatic
    /// shape of the mean free path. Default (1, 0.5, 0.25), skin-like:
    /// red travels farthest.
    pub subsurface_radius_scale: Texturable<[f32; 3]>,
    /// Henyey–Greenstein anisotropy of the subsurface interior's
    /// scattering; 0 is isotropic, positive leans forward.
    pub subsurface_scatter_anisotropy: Texturable<f32>,
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
            transmission_scatter: [0.0; 3],
            transmission_scatter_anisotropy: 0.0,
            subsurface_weight: Texturable::Constant(0.0),
            subsurface_color: Texturable::Constant([0.8; 3]),
            subsurface_radius: Texturable::Constant(1.0),
            subsurface_radius_scale: Texturable::Constant([1.0, 0.5, 0.25]),
            subsurface_scatter_anisotropy: Texturable::Constant(0.0),
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
    /// these, and prep collects them for upload. The same eleven slots,
    /// paired with their texture usage, drive prep's collection and
    /// lowering; a new textured parameter joins here, there, and
    /// `MaterialPatch`.
    pub(crate) fn textures(&self) -> impl Iterator<Item = &TextureRef> {
        [
            self.base_color.texture(),
            self.base_metalness.texture(),
            self.specular_roughness.texture(),
            self.emission_color.texture(),
            self.geometry_opacity.texture(),
            self.geometry_normal.as_ref(),
            self.subsurface_weight.texture(),
            self.subsurface_color.texture(),
            self.subsurface_radius.texture(),
            self.subsurface_radius_scale.texture(),
            self.subsurface_scatter_anisotropy.texture(),
        ]
        .into_iter()
        .flatten()
    }
}

/// A participating medium: what fills a volume, rather than what covers a
/// surface. Named and referenced like every other object: by
/// [`Settings::global_medium`], the atmosphere everything stands in, and by
/// [`Instance::medium`], a mesh that bounds it.
///
/// Coefficients are per meter, in linear `Rec.709` like every other color
/// constant, and convert to the working space at prep — where a saturated
/// authored value can land slightly negative, and clamps to zero. The
/// default is vacuum, which is also what a get-or-create placeholder wants:
/// a medium with no coefficients extinguishes nothing.
///
/// With a [`volume`](Self::volume) block the medium is *heterogeneous*: the
/// grid's unitless density multiplies the authored coefficients point by
/// point — σ(x) = density(x) · σ, the Arnold `standard_volume` / Cycles
/// Principled Volume convention, so a lookdev edit of the coefficients or
/// anisotropy touches no grid data. It is placed by an instance whose mesh
/// is [`MeshSource::MediumBounds`], and cannot be the global medium (a grid
/// has bounds; the open space has none).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Medium {
    /// Absorption coefficient `σ_a`: how much light the medium removes per
    /// meter. Default zero.
    #[serde(default = "zero3")]
    pub absorption: [f32; 3],
    /// Scattering coefficient `σ_s`: how much it redirects per meter. Zero
    /// (the default) is a purely absorbing medium — Beer–Lambert and
    /// nothing more.
    #[serde(default = "zero3")]
    pub scattering: [f32; 3],
    /// Henyey–Greenstein anisotropy: 0 scatters isotropically, positive
    /// leans forward (haze, clouds), negative back. Clamped inside (−1, 1)
    /// at prep, where the phase function stays finite.
    #[serde(default)]
    pub anisotropy: f32,
    /// The density grid, or `None` (the default) for a homogeneous medium.
    #[serde(default)]
    pub volume: Option<VolumeSource>,
}

/// A density grid reference: what makes a [`Medium`] heterogeneous.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VolumeSource {
    /// The grid file — `.vdb` (converted through the beside-the-source
    /// prep cache) or `.nvdb` (read directly). Absolute once applied,
    /// like every path.
    pub path: PathBuf,
    /// The density grid's name inside the file. Default `"density"`, the
    /// convention every DCC writes.
    #[serde(default = "default_density_grid")]
    pub grid: String,
    /// The temperature field's name in the same file, or `None` (the
    /// default) for a medium that does not emit. `"temperature"` is what
    /// every pyro solver writes beside its density.
    ///
    /// It must share the density grid's index transform — one simulation's
    /// fields do, and prep refuses the pair rather than sampling the
    /// temperature somewhere the density is not.
    #[serde(default)]
    pub temperature_grid: Option<String>,
    /// Kelvin per unit of that field: `K = value · scale + offset`. The
    /// default reads the field as Kelvin outright; a solver writing a
    /// normalized `[0, 1]` heat wants the scale set to the temperature its
    /// 1 stands for.
    #[serde(default = "one")]
    pub temperature_scale: f32,
    /// The other half of that map — a floor a normalized field never
    /// reaches below. Default zero.
    #[serde(default)]
    pub temperature_offset: f32,
    /// What multiplies the blackbody radiance at that temperature, linear
    /// `Rec.709` like every other color.
    ///
    /// The source term is `density · absorption · emission · B(K)`, so
    /// this is what the medium radiates per unit of *absorption* optical
    /// depth, in units of a 6500 K body — which is why fire wants a large
    /// number twice over. A flame is a thousand times dimmer than 6500 K
    /// in the visible, and smoke that reads well usually absorbs at a
    /// hundredth. Scene load prints the peak the mapping reaches, so the
    /// number can be read rather than guessed.
    ///
    /// Emission riding on absorption is what the transport equation says
    /// and what keeps the estimator bounded (`tracking.slang`); its cost
    /// is that making the medium more scattering makes it dimmer.
    #[serde(default = "one3")]
    pub emission: [f32; 3],
}

fn default_density_grid() -> String {
    "density".to_owned()
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

/// The environment light: an equirect radiance image surrounding the
/// scene — or, with no image, a constant white sky — tinted and turned.
#[derive(Clone, Debug, PartialEq)]
pub struct Environment {
    /// The radiance image — linear, equirect, `.exr` or Radiance `.hdr`
    /// (prep tells them apart by content, not extension). `None` is a
    /// constant white sky, colored by `tint`.
    pub path: Option<PathBuf>,
    /// Linear `Rec.709` multiplier over the image's radiance — the whole
    /// sky dims, brightens, or colors through it without touching the
    /// image. Default white (no change).
    pub tint: [f32; 3],
    /// Environment-to-world placement. Only the linear part acts — the sky
    /// is all directions, so translation has nothing to move — and it must
    /// be invertible (sampling maps world directions back through the
    /// inverse). Default identity.
    pub transform: Transform,
}

impl Default for Environment {
    /// A constant white sky: no image, no tint, no turn.
    fn default() -> Self {
        Self {
            path: None,
            tint: one3(),
            transform: Transform::default(),
        }
    }
}

/// Render settings — the minimal set, so the format doesn't churn while
/// the render loop learns to read it.
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    /// Output width × height, pixels.
    pub resolution: [u32; 2],
    /// The sample budget: how many samples per pixel a render accumulates
    /// before it is done. A hard cap — `noise_threshold` can only stop a
    /// render earlier, never carry it past this.
    pub spp: u32,
    /// Stop early once
    /// [`CONVERGENCE_TARGET`](crate::render::Renderer::CONVERGENCE_TARGET)
    /// of the pixels have reached this relative estimator standard error,
    /// a fraction in (0, 1]. `None` — the default — spends the whole
    /// budget.
    pub noise_threshold: Option<f32>,
    /// Maximum path length in bounces.
    pub max_bounces: u32,
    /// Run the published image through the denoiser. A view of the film,
    /// never part of the estimator: it changes what a consumer is handed,
    /// not what is accumulated, so turning it on or off mid-render keeps
    /// every sample already taken. Off by default — a host that wants the
    /// estimator's own pixels gets them by saying nothing.
    pub denoise: bool,
    /// Sampler seed, decorrelating repeat renders. Carried by the format
    /// but not yet wired into the sampler — honest decorrelation needs a
    /// seed input in the RNG, not a sample-index offset.
    pub seed: u32,
    /// The [`Medium`] filling the open space between instances, by name;
    /// `None` is vacuum. Unbounded: with one, a ray that reaches no surface
    /// crosses infinite optical depth, so the environment and the distant
    /// lights are unreachable — an atmosphere with a sky in it wants a
    /// bounded volume, not this.
    pub global_medium: Option<String>,
}

impl Default for Settings {
    /// The CLI's defaults, so a scene file that says nothing renders the
    /// same as `cenote-cli` with no flags.
    fn default() -> Self {
        Self {
            resolution: [1280, 720],
            spp: 64,
            noise_threshold: None,
            max_bounces: crate::wavefront::Wavefront::DEFAULT_MAX_BOUNCES,
            denoise: false,
            seed: 0,
            global_medium: None,
        }
    }
}

/// The nine object maps — the description's entire contents, split out
/// so apply can clone, mutate, validate, and swap them atomically.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Objects {
    pub(crate) meshes: BTreeMap<String, Mesh>,
    pub(crate) curves: BTreeMap<String, Curves>,
    pub(crate) instances: BTreeMap<String, Instance>,
    pub(crate) materials: BTreeMap<String, Material>,
    pub(crate) media: BTreeMap<String, Medium>,
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

    /// The curve batches, by name.
    #[must_use]
    pub fn curves(&self) -> &BTreeMap<String, Curves> {
        &self.objects.curves
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

    /// The media, by name.
    #[must_use]
    pub fn media(&self) -> &BTreeMap<String, Medium> {
        &self.objects.media
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

    /// Become `new`, recording the *difference* as dirty state — the
    /// file-reload semantic. Where [`SceneDescription::apply`] overlays
    /// edits, `replace` says the incoming description **is** the scene:
    /// objects it lacks are removed (their residency retires), objects
    /// whose values differ (or are new) are changed, and identical
    /// objects contribute nothing — so re-saving an untouched file
    /// rebuilds nothing. `new`'s own dirty state is discarded; only the
    /// diff against `self` matters.
    pub fn replace(&mut self, new: SceneDescription) {
        let mut dirty = Dirty::default();
        diff(
            Kind::Mesh,
            &self.objects.meshes,
            &new.objects.meshes,
            &mut dirty,
        );
        diff(
            Kind::Curves,
            &self.objects.curves,
            &new.objects.curves,
            &mut dirty,
        );
        diff(
            Kind::Instance,
            &self.objects.instances,
            &new.objects.instances,
            &mut dirty,
        );
        diff(
            Kind::Material,
            &self.objects.materials,
            &new.objects.materials,
            &mut dirty,
        );
        diff(
            Kind::Medium,
            &self.objects.media,
            &new.objects.media,
            &mut dirty,
        );
        diff(
            Kind::Light,
            &self.objects.lights,
            &new.objects.lights,
            &mut dirty,
        );
        diff(
            Kind::Camera,
            &self.objects.cameras,
            &new.objects.cameras,
            &mut dirty,
        );
        diff(
            Kind::Environment,
            &self.objects.environments,
            &new.objects.environments,
            &mut dirty,
        );
        diff(
            Kind::Settings,
            &self.objects.settings,
            &new.objects.settings,
            &mut dirty,
        );
        self.objects = new.objects;
        self.dirty.merge(dirty);
    }
}

/// One kind's contribution to a [`SceneDescription::replace`] diff:
/// new-or-different names are changed, names only the old side holds are
/// removed.
fn diff<T: PartialEq>(
    kind: Kind,
    old: &BTreeMap<String, T>,
    new: &BTreeMap<String, T>,
    dirty: &mut Dirty,
) {
    for (name, value) in new {
        if old.get(name) != Some(value) {
            dirty.changed.insert((kind, name.clone()));
        }
    }
    for name in old.keys() {
        if !new.contains_key(name) {
            dirty.removed.insert((kind, name.clone()));
        }
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

    /// `replace` dirt is the diff, not the file: unchanged objects are
    /// silent, changed values and new names mark changed, and names the
    /// incoming description lacks mark removed — how deleting an object
    /// from a scene file retires its residency on reload.
    #[test]
    fn replace_dirties_exactly_the_difference() {
        use crate::scene::changeset::{ChangeSet, MaterialPatch, Op, SettingsPatch};

        let base = || ChangeSet {
            ops: vec![
                Op::Settings(SettingsPatch::new("main")),
                Op::Material(Box::new(MaterialPatch::new("keep"))),
                Op::Material(Box::new(MaterialPatch::new("gone"))),
            ],
        };
        let mut description = SceneDescription::new();
        description.apply(&base()).expect("valid");
        description.take_dirty();

        // The "file" now drops one material and retunes another.
        let mut incoming = SceneDescription::new();
        incoming
            .apply(&ChangeSet {
                ops: vec![
                    Op::Settings(SettingsPatch::new("main")),
                    Op::Material(Box::new(MaterialPatch {
                        coat_weight: Some(1.0),
                        ..MaterialPatch::new("keep")
                    })),
                ],
            })
            .expect("valid");
        description.replace(incoming);

        let dirty = description.take_dirty();
        assert_eq!(
            dirty.changed,
            [(Kind::Material, "keep".to_owned())].into_iter().collect()
        );
        assert_eq!(
            dirty.removed,
            [(Kind::Material, "gone".to_owned())].into_iter().collect()
        );
        assert!(!description.materials().contains_key("gone"));

        // Replacing with an identical description dirties nothing.
        let mut identical = SceneDescription::new();
        identical
            .apply(&ChangeSet {
                ops: vec![
                    Op::Settings(SettingsPatch::new("main")),
                    Op::Material(Box::new(MaterialPatch {
                        coat_weight: Some(1.0),
                        ..MaterialPatch::new("keep")
                    })),
                ],
            })
            .expect("valid");
        description.replace(identical);
        assert!(description.take_dirty().is_empty());
    }

    #[test]
    fn textures_walks_every_textured_slot() {
        let mut material = Material {
            base_color: Texturable::Texture(TextureRef {
                path: "color.png".into(),
                color_space: None,
                channel: None,
                scale: None,
                uv: None,
            }),
            geometry_normal: Some(TextureRef {
                path: "normal.png".into(),
                color_space: None,
                channel: None,
                scale: None,
                uv: None,
            }),
            ..Material::default()
        };
        assert_eq!(material.textures().count(), 2);
        material.geometry_opacity = Texturable::Texture(TextureRef {
            path: "mask.png".into(),
            color_space: None,
            channel: Some(Channel::A),
            scale: None,
            uv: None,
        });
        assert_eq!(material.textures().count(), 3);
        assert_eq!(Material::default().textures().count(), 0);
    }
}
