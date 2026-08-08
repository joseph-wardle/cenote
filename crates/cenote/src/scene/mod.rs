//! The scene, in two halves. [`description`] is the typed, named object
//! schema and [`changeset`] its one edit path — what scene files, the pbrt
//! importer, and lookdev edits all speak. The private `prep` module joins
//! that model to the GPU residency below — meshes built into acceleration
//! structures, per-instance materials, emissive triangles and delta
//! lights with their sampling table, a thin-lens camera, and an equirect
//! environment. [`Scene::prep`]
//! builds a description fresh; `Scene::update` follows its accumulated
//! dirty state, reusing the residency an edit leaves untouched. [`Scene::new`]
//! remains as the procedural build the furnace and estimator tests speak
//! (they need materials and environments no scene file can express).
//!
//! [`Scene::demo`] is the standing test subject — a grid of smooth-shaded
//! spheres sweeping roughness × metalness across a glossy floor, where
//! winding, handedness, shading-normal, or energy mistakes are instantly
//! visible, under a warm quad light and the bundled Kloofendal sky. It is
//! [`changeset::ChangeSet::demo`] prepped: the demo scene is data first.
//!
//! [`Scene::many_lights`] is the validation subject — hundreds of small
//! emitters over matte occluders under a black sky, the case that stresses
//! next-event estimation hardest. It is
//! [`changeset::ChangeSet::many_lights`] prepped, the same scene-as-data
//! path as the demo.

pub mod changeset;
mod demo;
pub mod description;
mod lower;
mod many_lights;
mod prep;
mod shapes;

pub use shapes::{cube, ground_plane, icosphere};

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec2, Vec3};

use crate::color::luminance;
use crate::environment::Environment;
use crate::error::{Error, Result};
use crate::gpu::{AccelerationStructure, Buffer, Context, SampledImage, TlasInstance, Upload};
use crate::lights::{DeltaLight, LIGHT_NONE, TriangleLight};
use crate::material::{Material, TEXTURE_NONE};
use crate::tables::BsdfTables;
use crate::texture;

/// A rejected edit or malformed scene input, as an [`Error::Scene`] — the
/// one failure the change-set apply and prep paths raise, shared so its
/// spelling stays uniform across both.
pub(super) fn scene_error(message: String) -> Error {
    Error::Scene(message)
}

/// Ray-visibility mask bits, matched by the mask each TLAS instance
/// carries. Camera rays trace with [`ray_mask::CAMERA`]; every other ray
/// (bounce, shadow) traces with [`ray_mask::ALL`], so a camera-invisible
/// instance still bounces light, casts shadows, and — when emissive —
/// illuminates. The full per-ray-type set (diffuse/glossy/shadow) is not
/// yet wired up; today only the camera bit is real.
pub(crate) mod ray_mask {
    pub const CAMERA: u32 = 0x01;
    pub const ALL: u32 = 0xFF;
}

/// A triangle mesh on the host: tightly packed positions, matching shading
/// normals, plus index triples.
pub struct Mesh {
    /// Vertex positions, in meters, in object space.
    pub positions: Vec<Vec3>,
    /// Unit shading normals, one per position, in object space. Shading
    /// interpolates these across each triangle, which is what makes a
    /// coarse sphere render smooth; geometry that *should* look flat
    /// (planes, quads) carries its face normal at every vertex.
    pub normals: Vec<Vec3>,
    /// Texture coordinates, one per position. A mesh authored without any
    /// carries zeros — textured lookups then read texel (0, 0), constant
    /// but never out of bounds.
    pub uvs: Vec<Vec2>,
    /// Counter-clockwise-outward index triples into `positions`.
    pub triangles: Vec<[u32; 3]>,
}

/// One thing in a scene: a mesh, where it stands, and what its surface is.
pub struct Object {
    /// The geometry, in object space.
    pub mesh: Mesh,
    /// Object-to-world placement. Must be invertible — normals and ray
    /// offsets transform through the inverse.
    pub transform: Mat4,
    /// The surface, constant across the mesh (per-face materials are not modelled).
    pub material: Material,
    /// The medium this mesh bounds — the procedural spelling of a
    /// description's [`description::Instance::medium`]. `Some`, and the
    /// mesh is a null boundary: `material` is never shaded.
    pub medium: Option<Medium>,
    /// Which solid wins where refractive interiors overlap — the
    /// procedural spelling of
    /// [`description::Instance::interior_priority`], and 0 like it.
    pub interior_priority: u32,
}

/// One mesh resident on the GPU. The vertex, normal, and index buffers stay
/// alive past the BLAS build: the surface-shading kernel fetches triangle
/// corners from them to compute geometric normals and interpolate shading
/// normals.
struct GpuMesh {
    blas: AccelerationStructure,
    vertices: Buffer,
    normals: Buffer,
    uvs: Buffer,
    indices: Buffer,
}

/// One entry of the geometry lookup table, indexed by instance custom index:
/// where the instance's triangles live plus its transforms — everything a
/// kernel needs to re-evaluate shading at a hit. Mirrors
/// `struct GeometryRecord` in `shaders/scene.slang` field for field.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GeometryRecord {
    positions: vk::DeviceAddress,
    normals: vk::DeviceAddress,
    uvs: vk::DeviceAddress,
    indices: vk::DeviceAddress,
    /// Rows of the instance's 3×4 object-to-world transform — the same
    /// shape the TLAS instance itself carries.
    object_to_world: [[f32; 4]; 3],
    /// Rows of the inverse: normals transform through it, and the
    /// spawn-point error bounds need both directions.
    world_to_object: [[f32; 4]; 3],
    /// Index of the instance's *first* light record, or [`LIGHT_NONE`] —
    /// an emissive instance has one record per triangle, in primitive
    /// order, so a BSDF-sampled hit finds the pdf its MIS weight competes
    /// against at `light + primitive`.
    light: u32,
    /// What crossing this instance's surface means ([`boundary`]) and,
    /// above it in the medium-set entry's `STACK_PRIORITY` field, which
    /// solid wins where interiors overlap ([`interior_priority`]) — see
    /// [`boundary_word`]. One word because the shader wants both together;
    /// the Slang side reads the class back through `boundaryOf`.
    boundary: u32,
    /// Index into the medium table of what this instance's interior is
    /// filled with, or [`MEDIUM_NONE`] — see [`placement_medium`].
    medium: u32,
    _pad0: u32,
}

const _: () = assert!(size_of::<GeometryRecord>() == 144);

/// `GeometryRecord::boundary` values — the `BOUNDARY_*` constants in
/// `shaders/scene.slang`.
const BOUNDARY_OPAQUE: u32 = 0;
const BOUNDARY_REFRACTIVE: u32 = 1;
const BOUNDARY_NULL: u32 = 2;

/// A medium index that names vacuum — an index no table can reach. Matches
/// `MEDIUM_NONE` in `shaders/scene.slang`. Deliberately not the stack's
/// all-ones `STACK_EMPTY`: media and instances are separate index spaces,
/// and a shared value invites masking one with the other's bounds.
const MEDIUM_NONE: u32 = u32::MAX;

/// What crossing this placement's surface means. The single authority on
/// interiors: it classifies every instance's boundary word *and*, through
/// [`has_interiors`], decides whether the wavefront allocates the medium
/// stack — so the two can't disagree.
///
/// Bounding a medium wins over any surface the material describes: a
/// volume's boundary is crossed, never shaded. Otherwise transmission
/// decides, and transmission *depth* is deliberately not consulted — a
/// depth-0 interior absorbs nothing but still drives the closure's
/// `exiting` flag, which reads the stack.
fn boundary(material: &Material, bounds_medium: bool) -> u32 {
    if bounds_medium {
        BOUNDARY_NULL
    } else if material.transmission_weight > 0.0 && material.thin_walled == 0 {
        BOUNDARY_REFRACTIVE
    } else {
        BOUNDARY_OPAQUE
    }
}

/// [`boundary`] for a lowered placement — the one caller that has both
/// halves of the question.
fn placement_boundary(placement: &Placement) -> u32 {
    boundary(&placement.material, placement.medium.is_some())
}

/// The material a mesh that bounds a medium renders with. A null boundary
/// is crossed, never shaded, so no surface the author described may survive
/// into the tables:
///
/// * emission is *removed* rather than ignored, because it would otherwise
///   enter the light list and next-event estimation would reach a surface
///   path sampling can never hit — two strategies covering different paths
///   is bias, not a lost highlight;
/// * opacity is forced full, because the boundary traverses as non-opaque
///   so the shadow pass can see every crossing, and full coverage is what
///   makes the stochastic test commit the nearest one.
///
/// Both scene paths call this where a material first meets its medium, so
/// an emissive or fractional null boundary does not exist downstream —
/// [`upload_instance_tables`] asserts as much rather than repairing it.
fn boundary_material(material: Material, bounds_medium: bool) -> Material {
    if !bounds_medium {
        return material;
    }
    Material {
        emission: Vec3::ZERO,
        opacity: 1.0,
        opacity_texture: TEXTURE_NONE,
        ..material
    }
}

/// Whether any placement has a closed transmissive interior — see
/// [`Scene::has_interiors`].
fn has_interiors(placements: &[Placement]) -> bool {
    placements
        .iter()
        .any(|placement| placement_boundary(placement) == BOUNDARY_REFRACTIVE)
}

/// Whether any placement bounds a medium — see [`Scene::has_volumes`].
fn has_volumes(placements: &[Placement]) -> bool {
    placements
        .iter()
        .any(|placement| placement.medium.is_some())
}

/// Whether any placement closes a *scattering* interior — the one kind of
/// interior that owes the volume stage an event, so it counts toward
/// [`Scene::has_media`] where a purely absorbing one does not. Defined
/// through [`boundary_word`], the same bit the kernels route on, so the
/// volume stage runs exactly where a segment can reach it.
fn has_scattering_interiors(placements: &[Placement]) -> bool {
    placements
        .iter()
        .any(|placement| boundary_word(placement) & STACK_SCATTERING != 0)
}

/// Medium-set entry bits, mirroring `STACK_*` in `shaders/pathstate.slang`
/// — [`boundary_word`] bakes them into [`GeometryRecord::boundary`], so
/// the shader's `stackEntry` lifts them into an entry without shifting
/// anything.
const STACK_PRIORITY_SHIFT: u32 = 25;
const STACK_SCATTERING: u32 = 1 << 24;
const STACK_INTERIOR: u32 = 1 << 31;

/// How far a nesting priority may reach: the six bits the medium-set entry
/// has room for, directly under `STACK_INTERIOR` — which is what lets one
/// `max` rank the whole set by priority. Held here, at the table, so no
/// construction path can bypass it; the same place [`MAX_ANISOTROPY`] is
/// held.
const MAX_PRIORITY: u32 = 63;

/// A placement's interior priority as the tables carry it. Zero on
/// anything but a refractive boundary — only a refractive interface can be
/// cut away, so a number anywhere else could only make [`has_priority`]
/// disagree with what a shader can act on.
fn interior_priority(placement: &Placement) -> u32 {
    if placement_boundary(placement) != BOUNDARY_REFRACTIVE {
        return 0;
    }
    if placement.priority > MAX_PRIORITY {
        log::warn!(
            "interior priority {} is higher than the medium set can rank — clamping to \
             {MAX_PRIORITY}",
            placement.priority
        );
    }
    placement.priority.min(MAX_PRIORITY)
}

/// [`GeometryRecord::boundary`]: the `BOUNDARY_*` class in the low byte
/// and, on a refractive boundary, the medium-set entry's flag bits above
/// it — `STACK_INTERIOR` and the nesting priority — pre-baked so the
/// shader's `stackEntry` is one load and one OR. Packed into one word
/// because a second field would be a second scalar load per crossing.
fn boundary_word(placement: &Placement) -> u32 {
    let class = placement_boundary(placement);
    let flags = if class == BOUNDARY_REFRACTIVE {
        // The scattering bit is baked only where the derived medium truly
        // scatters — which guarantees any entry carrying it names a medium
        // record, asserted at [`upload_instance_tables`]. This bit is the
        // single authority the kernels route on, gate the surface stage's
        // closed-form absorption with, and resolve the segment's medium by:
        // one value, so no two of those can disagree.
        let scattering = interior(&placement.material)
            .is_some_and(|record| record.sigma_s.iter().any(|&sigma| sigma > 0.0));
        STACK_INTERIOR
            | interior_priority(placement) << STACK_PRIORITY_SHIFT
            | if scattering { STACK_SCATTERING } else { 0 }
    } else {
        0
    };
    class | flags
}

/// Whether any placement authors a priority the kernels can act on — see
/// [`Scene::has_priority`]. Defined through [`interior_priority`], so
/// `has_priority` implies [`has_interiors`] by construction and can never
/// gate a read of a buffer that was not allocated.
///
/// All-zero priorities are never *strictly* less than one another, so a
/// scene that authors none can never suppress an interface — which is why
/// it renders bit-identically rather than by a special case.
fn has_priority(placements: &[Placement]) -> bool {
    placements
        .iter()
        .any(|placement| interior_priority(placement) != 0)
}

// The class must survive the packing: `BOUNDARY_MASK` in `scene.slang` is
// a byte, and the priority has to clear it and stay under `STACK_INTERIOR`.
const _: () = assert!(BOUNDARY_NULL < 0xff && (1 << STACK_PRIORITY_SHIFT) > 0xff);
const _: () = assert!(MAX_PRIORITY << STACK_PRIORITY_SHIFT < STACK_INTERIOR);
const _: () = assert!(STACK_SCATTERING > 0xff && STACK_SCATTERING < 1 << STACK_PRIORITY_SHIFT);

/// A homogeneous medium as the renderer holds it: coefficients per `ACEScg`
/// channel in inverse meters, and the phase function's anisotropy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Medium {
    /// Absorption `σ_a`: the part of the extinction that removes light.
    pub absorption: Vec3,
    /// Scattering `σ_s`: the part that redirects it. Zero is pure
    /// absorption, which has no event to sample.
    pub scattering: Vec3,
    /// Henyey–Greenstein anisotropy; 0 is isotropic, positive leans
    /// forward. Clamped to [`MAX_ANISOTROPY`] on its way into the table —
    /// the one place it can reach a shader.
    pub anisotropy: f32,
}

/// One entry of the medium table. Mirrors `struct MediumRecord` in
/// `shaders/scene.slang`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MediumRecord {
    /// Extinction `σ_a` + `σ_s`, per `ACEScg` channel, in inverse meters —
    /// the σ of Beer–Lambert, and the density distance sampling draws against.
    sigma_t: [f32; 3],
    /// Henyey–Greenstein anisotropy, clamped inside (−1, 1).
    g: f32,
    /// Scattering `σ_s`. Zero routes no path to the volume stage.
    sigma_s: [f32; 3],
    _pad0: f32,
}

const _: () = assert!(size_of::<MediumRecord>() == 32);

/// How far `g` may reach: at ±1 the Henyey–Greenstein denominator vanishes
/// and both the phase value and its sampled direction go non-finite.
/// Enforced here, at the table, so no other construction path can bypass it;
/// lowering only warns that it happened.
pub(super) const MAX_ANISOTROPY: f32 = 0.99;

/// `g` as the table may carry it: bounded to ±[`MAX_ANISOTROPY`], with a
/// NaN falling to isotropic — `clamp` alone would pass one straight
/// through to the phase function.
fn bounded_anisotropy(g: f32) -> f32 {
    if g.is_nan() {
        0.0
    } else {
        g.clamp(-MAX_ANISOTROPY, MAX_ANISOTROPY)
    }
}

impl From<Medium> for MediumRecord {
    fn from(medium: Medium) -> Self {
        Self {
            sigma_t: (medium.absorption + medium.scattering).to_array(),
            g: bounded_anisotropy(medium.anisotropy),
            sigma_s: medium.scattering.to_array(),
            _pad0: 0.0,
        }
    }
}

/// What fills the interior this material closes, as an extinction — or
/// `None` for vacuum: an opaque or thin-walled surface bounds none, and a
/// depth of zero puts the tint on the interface, where the closure applies
/// it instead. The depth test is written positively so that a NaN falls to
/// vacuum, as it did on the device, rather than to a σ of NaN.
///
/// Not bit-identical to the per-hit form it replaces: Vulkan specifies
/// `log` only to 2^-21 *absolute* inside [0.5, 2], which is a large
/// *relative* error where the color is near 1 and the logarithm near 0 —
/// up to 9e-6 on the corpus's σ. The host's correctly rounded `ln` is the
/// accurate side of that.
///
/// The scattering split follows `OpenPBR`'s subtractive convention: the
/// color/depth logarithm stays the *extinction* `σ_t` exactly as authored,
/// `σ_s` = `transmission_scatter` / depth is the part of it that redirects,
/// and `σ_a` is their difference. A channel whose scatter outruns its
/// extinction would absorb negatively — amplify — so `σ_a` is repaired by
/// shifting it up uniformly (gray, keeping the hue the author picked)
/// until nothing does, which grows `σ_t` by the same shift. The scatter
/// tests are written positively like the depth's, so a NaN falls to
/// purely absorbing.
fn interior(material: &Material) -> Option<MediumRecord> {
    let depth = material.transmission_depth;
    (boundary(material, false) == BOUNDARY_REFRACTIVE && depth > 0.0).then(|| {
        let sigma_t = material
            .transmission_color
            .to_array()
            .map(|channel| -channel.clamp(1e-4, 1.0).ln() / depth);
        let scatter = material.transmission_scatter.max(Vec3::ZERO);
        if !scatter.cmpgt(Vec3::ZERO).any() {
            // Purely absorbing — the pre-scatter record, bit for bit: even
            // adding a zero shift would flip a white channel's −0.0.
            return MediumRecord {
                sigma_t,
                g: 0.0,
                sigma_s: [0.0; 3],
                _pad0: 0.0,
            };
        }
        let sigma_s = scatter / depth;
        let shift = (sigma_s - Vec3::from(sigma_t)).max_element().max(0.0);
        MediumRecord {
            sigma_t: (Vec3::from(sigma_t) + shift).to_array(),
            g: bounded_anisotropy(material.transmission_scatter_anisotropy),
            sigma_s: sigma_s.to_array(),
            _pad0: 0.0,
        }
    })
}

/// What fills a placement: the medium its mesh bounds, or — for an ordinary
/// surface — whatever its material's interior holds. A mesh that bounds a
/// medium is a null boundary, so the material's own interior can never also
/// apply.
fn placement_medium(placement: &Placement) -> Option<MediumRecord> {
    placement.medium.map_or_else(
        || interior(&placement.material),
        |medium| Some(medium.into()),
    )
}

/// The medium table for the placements' `filled` interiors and the scene's
/// `global` medium: one record per distinct medium, however many places it
/// fills, plus the index each placement takes in it and the index of the
/// global medium (both [`MEDIUM_NONE`] where there is none).
///
/// Records are keyed by the coefficients they compute to — by their bits, so
/// ±0 key apart, conservative in the safe direction — not by what produced
/// them: two interiors that extinguish identically *are* one medium, and so
/// is a global medium that matches one. Identity stays on the instance,
/// which is what the stack holds and what entering and exiting compare, so
/// this sharing can never make a path inside one instance read as leaving
/// another.
fn medium_table(
    filled: impl Iterator<Item = Option<MediumRecord>>,
    global: Option<Medium>,
) -> (Vec<u32>, u32, Vec<MediumRecord>) {
    let mut records: Vec<MediumRecord> = Vec::new();
    let mut interned: HashMap<[u32; 8], u32> = HashMap::new();
    let mut intern = |record: Option<MediumRecord>| {
        record.map_or(MEDIUM_NONE, |record| {
            *interned
                .entry(bytemuck::cast(record))
                .or_insert_with(|| {
                    records.push(record);
                    records.len() as u32 - 1
                })
        })
    };
    let indices = filled.map(&mut intern).collect();
    let global = intern(global.map(MediumRecord::from));
    (indices, global, records)
}

/// Every buffer the scene shares with the kernels, one address each, plus
/// the embedded environment tables — kernels carry a single pointer to
/// this table in their push constants. Mirrors `struct SceneTable` (with
/// its nested `struct Environment`) in `shaders/scene.slang` and
/// `shaders/environment.slang` field for field.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SceneTable {
    geometry: vk::DeviceAddress,
    materials: vk::DeviceAddress,
    lights: vk::DeviceAddress,
    /// Per-texture sample-time parameters ([`TextureParams`]), one record
    /// per bindless slot in the same index order.
    texture_params: vk::DeviceAddress,
    /// Rows of the environment's placement (linear part only — the sky is
    /// all directions) — `envToWorld` on the Slang side.
    env_to_world: [[f32; 4]; 3],
    /// Rows of its inverse: world directions into environment space.
    env_from_world: [[f32; 4]; 3],
    /// `ACEScg` multiplier the kernel folds over the image's radiance.
    env_tint: [f32; 3],
    /// p(next-event estimation samples the environment rather than the
    /// light list) — `selectProb` on the Slang side.
    env_select_prob: f32,
    env_marginal: vk::DeviceAddress,
    env_conditional: vk::DeviceAddress,
    env_pdfs: vk::DeviceAddress,
    env_width: u32,
    env_height: u32,
    /// The [`MediumRecord`] table.
    media: vk::DeviceAddress,
    light_count: u32,
    /// What fills the open space between instances, or [`MEDIUM_NONE`].
    global_medium: u32,
}

// The Slang side lays these out from its own rules, so the sizes are
// pinned here: a mirror that drifts reads garbage rather than failing.
const _: () = assert!(size_of::<SceneTable>() == 192);

/// Sample-time parameters of one bindless texture slot: the affine UV
/// remap and the value multiplier a [`description::TextureRef`] carries
/// (identity when it carries none). They fork the bindless index — the
/// texture [`Key`](texture::Key) includes them — so the record is
/// per-slot, indexed exactly like the descriptor table. Mirrors `struct
/// TextureParams` in `shaders/textures.slang` field for field.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TextureParams {
    uv_scale: [f32; 2],
    uv_offset: [f32; 2],
    scale: f32,
    _pad0: [u32; 3],
}

const _: () = assert!(size_of::<TextureParams>() == 32);

/// Upload the sample-time parameter table for `keys`, in their (bindless)
/// order. A texture-less scene uploads one identity record: the table
/// address in [`SceneTable`] must point at a real buffer even when no
/// material record will index it.
fn upload_texture_params<'k>(
    gpu: &Context,
    keys: impl Iterator<Item = &'k texture::Key>,
) -> Result<Buffer> {
    let mut records: Vec<TextureParams> = keys
        .map(|key| {
            let (uv_scale, uv_offset, scale) = key.4.unpack();
            TextureParams {
                uv_scale,
                uv_offset,
                scale,
                _pad0: [0; 3],
            }
        })
        .collect();
    if records.is_empty() {
        records.push(TextureParams {
            uv_scale: [1.0; 2],
            uv_offset: [0.0; 2],
            scale: 1.0,
            _pad0: [0; 3],
        });
    }
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
    gpu.upload_buffer(
        "scene.texture_params",
        bytemuck::cast_slice(&records),
        usage,
    )
}

/// The scene, resident on the GPU and ready to trace against.
#[expect(
    clippy::struct_excessive_bools,
    reason = "four independent scene predicates, each cached for a hot query"
)]
pub struct Scene {
    // Declared before `meshes`: the TLAS dies before the BLASes its
    // instances reference.
    tlas: AccelerationStructure,
    /// The environment radiance image — the binding model's one texture,
    /// bound next to the TLAS at every scene-resource dispatch.
    environment: SampledImage,
    /// The one [`SceneTable`] every kernel reaches scene data through.
    table: Buffer,
    /// The buffers `table` points into, replaced piecewise as edits dirty
    /// them.
    resident: ResidentBuffers,
    /// Mesh residency by name — prep rebuilds only the names an edit
    /// dirtied. The procedural [`Scene::new`] path keys them by object
    /// index and never updates.
    meshes: BTreeMap<String, GpuMesh>,
    /// Material-texture residency by prep request, with the content hash
    /// each image was built from — how an update tells a real image edit
    /// from a mere re-reference. Bindless indices are this map's iteration
    /// order, the order `descriptors` holds and material records index.
    /// The procedural [`Scene::new`] path has no textures.
    textures: BTreeMap<texture::Key, ResidentTexture>,
    /// The bindless table's write list, rebuilt whenever `textures`
    /// changes; every wave binds it next to the TLAS.
    descriptors: Vec<vk::DescriptorImageInfo>,
    camera: Camera,
    /// Whether any instance has a closed transmissive interior —
    /// recomputed on every build and update. See [`Scene::has_interiors`].
    has_interiors: bool,
    /// Likewise for the volume-bounding meshes. See [`Scene::has_volumes`].
    has_volumes: bool,
    /// Likewise for authored nesting priority. See [`Scene::has_priority`].
    has_priority: bool,
    /// Likewise for scattering interiors. See [`Scene::has_media`].
    has_scattering_interiors: bool,
    /// The environment's dimensions and emitted power (untinted), retained
    /// so a light edit can rebuild the scene table (its selection
    /// probability weighs the light list against the environment) without
    /// reloading the image.
    env_size: (u32, u32),
    env_power: f64,
    /// The image file the resident environment decoded from (`None` for a
    /// constant sky) — the identity that lets a tint or placement edit
    /// keep the decode and the upload.
    env_source: Option<PathBuf>,
    /// The environment's `ACEScg` tint and placement (linear part and
    /// inverse) — scene-table constants, retained beside `env_power` for
    /// the same table rebuilds.
    env_tint: Vec3,
    env_to_world: Mat4,
    env_from_world: Mat4,
}

/// One material texture resident on the GPU, with the content hash of the
/// source it was prepped from.
struct ResidentTexture {
    image: SampledImage,
    hash: u64,
}

/// Every buffer the [`SceneTable`] points into — geometry records,
/// materials, light records, and the environment's three sampling tables —
/// plus the closure's table images, which are built the same once-per-scene
/// way but reach the kernels as descriptors. Held to keep the residency
/// alive; replaced piecewise by prep as edits dirty them.
struct ResidentBuffers {
    /// Everything one pass over the placements produces, replaced as a set
    /// — a table that outlived its instances would read stale addresses.
    instances: InstanceTables,
    /// The closure's lookup tables — uploaded once at build and never
    /// dirtied (the data is embedded in the binary).
    bsdf_tables: BsdfTables,
    /// Per-texture sample-time parameters, in bindless-index order —
    /// rebuilt whenever the texture residency changes.
    texture_params: Buffer,
    env_marginal: Buffer,
    env_conditional: Buffer,
    env_pdfs: Buffer,
}

impl ResidentBuffers {
    /// Gather the freshly uploaded instance tables and environment sampling
    /// tables into one residency, uploading the embedded closure tables
    /// alongside. The one place both the procedural [`Scene::new`] and the
    /// description-driven [`Scene::prep`] build this set, so a new resident
    /// buffer is added here once rather than in two build paths that drift.
    fn assemble(
        gpu: &Context,
        instances: InstanceTables,
        texture_params: Buffer,
        env_marginal: Buffer,
        env_conditional: Buffer,
        env_pdfs: Buffer,
    ) -> Result<Self> {
        Ok(Self {
            instances,
            bsdf_tables: crate::tables::upload(gpu)?,
            texture_params,
            env_marginal,
            env_conditional,
            env_pdfs,
        })
    }
}

impl Scene {
    /// Upload `objects` and build them into a traceable scene, lit by its
    /// emissive objects and `environment` — the procedural build the
    /// estimator tests speak. Production scenes go through
    /// [`Scene::prep`], which builds the same residency from a
    /// [`description::SceneDescription`].
    pub fn new(
        gpu: &Context,
        objects: &[Object],
        camera: Camera,
        environment: &Environment,
    ) -> Result<Self> {
        Self::new_in_medium(gpu, objects, camera, environment, None)
    }

    /// [`Scene::new`] with the open space around the objects filled by
    /// `global` — the procedural spelling of a description's global medium,
    /// which is how the volumetric estimator tests reach it.
    ///
    /// # Errors
    ///
    /// As [`Scene::new`].
    pub fn new_in_medium(
        gpu: &Context,
        objects: &[Object],
        camera: Camera,
        environment: &Environment,
        global: Option<Medium>,
    ) -> Result<Self> {
        assert!(!objects.is_empty(), "a scene needs at least one object");
        let mut upload = gpu.upload()?;
        let meshes = objects
            .iter()
            .enumerate()
            .map(|(index, object)| {
                upload_mesh(&mut upload, &format!("object{index}"), &object.mesh)
            })
            .collect::<Result<Vec<GpuMesh>>>()?;
        // Before the TLAS below, which is the first thing to read them.
        upload.finish()?;
        // Resolved once, so the light list and the tables read the same
        // surface — a mesh that bounds a medium has none.
        let materials: Vec<Material> = objects
            .iter()
            .map(|object| boundary_material(object.material, object.medium.is_some()))
            .collect();
        // The light list: every triangle of every emissive object, in
        // world space. The procedural path has no delta lights — those
        // are description objects, exercised through prep.
        let triangle_lights: Vec<TriangleLight> = objects
            .iter()
            .zip(&materials)
            .enumerate()
            .filter(|(_, (_, material))| material.emission != Vec3::ZERO)
            .flat_map(|(index, (object, material))| {
                emissive_triangles(
                    &object.mesh.positions,
                    &object.mesh.triangles,
                    object.transform,
                    material.emission,
                    index as u32,
                )
            })
            .collect();
        let placements: Vec<Placement> = meshes
            .iter()
            .zip(objects)
            .zip(&materials)
            .map(|((mesh, object), material)| Placement {
                mesh,
                transform: object.transform,
                material: *material,
                camera_visible: true,
                medium: object.medium,
                priority: object.interior_priority,
            })
            .collect();
        let tlas = build_scene_tlas(gpu, &placements)?;
        let has_interiors = has_interiors(&placements);
        let has_volumes = has_volumes(&placements);
        let has_priority = has_priority(&placements);
        let has_scattering_interiors = has_scattering_interiors(&placements);
        let instances = upload_instance_tables(gpu, &placements, &triangle_lights, &[], global)?;
        let GpuEnvironment {
            image,
            marginal,
            conditional,
            pdfs,
            power,
        } = upload_environment(gpu, environment)?;
        let resident = ResidentBuffers::assemble(
            gpu,
            instances,
            // The procedural path has no textures — the identity record
            // keeps the table address valid.
            upload_texture_params(gpu, std::iter::empty())?,
            marginal,
            conditional,
            pdfs,
        )?;
        let env_size = (environment.width(), environment.height());
        let light_count = triangle_lights.len() as u32;
        // The procedural path takes its environment as-is: white tint,
        // identity placement.
        let table = upload_scene_table(
            gpu,
            &resident,
            env_size,
            (Mat4::IDENTITY, Mat4::IDENTITY),
            Vec3::ONE,
            select_probability(power, crate::lights::total_power(&triangle_lights, &[])),
            light_count,
        )?;
        Ok(Self {
            tlas,
            environment: image,
            table,
            resident,
            meshes: meshes
                .into_iter()
                .enumerate()
                .map(|(index, mesh)| (index.to_string(), mesh))
                .collect(),
            textures: BTreeMap::new(),
            descriptors: Vec::new(),
            camera,
            has_interiors,
            has_volumes,
            has_priority,
            has_scattering_interiors,
            env_size,
            env_power: power,
            env_source: None,
            env_tint: Vec3::ONE,
            env_to_world: Mat4::IDENTITY,
            env_from_world: Mat4::IDENTITY,
        })
    }

    /// The scene's TLAS, ready to bind for ray queries.
    #[must_use]
    pub fn tlas(&self) -> &AccelerationStructure {
        &self.tlas
    }

    /// Whether any instance has a closed transmissive interior (see
    /// [`boundary`]). True, the wavefront allocates the medium stack and
    /// its kernels track which interior each path travels; false, the
    /// stack is never allocated and no kernel addresses it.
    #[must_use]
    pub fn has_interiors(&self) -> bool {
        self.has_interiors
    }

    /// Whether the scene has media — the things that route a path segment
    /// through the volume stage, and so record it at all. A pure-absorbing
    /// *interior* is not one: it has no event to sample, and shades through
    /// the surface stage exactly as it did before media existed. A
    /// *scattering* interior is: its extinction and scatter are the volume
    /// stage's to sample.
    #[must_use]
    pub fn has_media(&self) -> bool {
        self.has_global_medium() || self.has_volumes() || self.has_scattering_interiors
    }

    /// Whether the open space between instances is filled — the flag
    /// intersect routes every segment outside an interior on.
    #[must_use]
    pub fn has_global_medium(&self) -> bool {
        self.resident.instances.global_medium != MEDIUM_NONE
    }

    /// Whether any mesh bounds a medium. True, the wavefront allocates the
    /// medium stack for the set of volumes each path is inside, and the
    /// volume stage marches across their boundaries.
    #[must_use]
    pub fn has_volumes(&self) -> bool {
        self.has_volumes
    }

    /// Whether any interior is authored with a nesting priority the
    /// kernels can act on — see [`has_priority`]. True, a hit can be an
    /// interface a higher-priority interior cuts away, which is resolved
    /// in the volume stage: so this is what makes that stage run in a
    /// scene with no media at all.
    #[must_use]
    pub fn has_priority(&self) -> bool {
        self.has_priority
    }

    /// The environment's emitted power as the selection heuristic weighs
    /// it: the image's luminance integral scaled by the tint's luminance.
    /// Exact for a neutral tint; a chromatic one lands within the
    /// heuristic's tolerance (the MIS weights stay exact regardless).
    fn tinted_env_power(&self) -> f64 {
        self.env_power * f64::from(luminance(self.env_tint))
    }

    /// The scene table: the one buffer of addresses kernels reach all
    /// shared scene data through (geometry records, materials, lights,
    /// environment tables).
    #[must_use]
    pub fn table(&self) -> &Buffer {
        &self.table
    }

    /// The environment radiance image, ready to bind next to the TLAS.
    #[must_use]
    pub fn environment(&self) -> &SampledImage {
        &self.environment
    }

    /// The bindless texture table's write list, in the index order
    /// material records use — what every wave binds at binding 2.
    pub fn texture_descriptors(&self) -> &[vk::DescriptorImageInfo] {
        &self.descriptors
    }

    /// The closure's lookup-table descriptors, 2D then 3D — bindings 3
    /// and 4. Not [`Self::table`], which is the scene's address table.
    pub fn bsdf_table_descriptors(
        &self,
    ) -> (&[vk::DescriptorImageInfo], &[vk::DescriptorImageInfo]) {
        (
            self.resident.bsdf_tables.planes(),
            self.resident.bsdf_tables.volumes(),
        )
    }

    /// Rebuild the bindless write list from the resident map — called
    /// after any prep that changed it. Iteration order is key order, the
    /// same order prep assigned the material records' indices in.
    fn rebuild_texture_descriptors(&mut self) {
        self.descriptors = self
            .textures
            .values()
            .map(|texture| texture.image.descriptor())
            .collect();
    }

    /// The scene's camera.
    #[must_use]
    pub fn camera(&self) -> &Camera {
        &self.camera
    }

    /// Mutable camera access — the viewer's orbit control writes here
    /// between frames.
    pub fn camera_mut(&mut self) -> &mut Camera {
        &mut self.camera
    }
}

/// A camera, described by where it sits, what it looks at, and which way
/// is up on screen — a pinhole unless it carries a [`Lens`].
#[derive(Clone, Copy)]
pub struct Camera {
    /// Eye position, meters.
    pub position: Vec3,
    /// The point the view axis passes through.
    pub look_at: Vec3,
    /// Which way is up on screen — the roll control. Usually world up
    /// ([`Vec3::Y`]); need not be perpendicular to the view axis, just
    /// not parallel to it.
    pub up: Vec3,
    /// Vertical field of view, degrees.
    pub vfov_degrees: f32,
    /// The thin lens, when depth of field is wanted; `None` is a pinhole
    /// (everything sharp).
    pub lens: Option<Lens>,
}

/// A thin lens: rays leave a disk instead of a point, and only the focal
/// plane images sharply. Raygen consumes it by scaling the [`RayBasis`]
/// to the focal plane — `position + forward + x·right + y·up` is then a
/// pixel's focal *point* — and re-aiming each ray from a sampled point on
/// the lens disk.
#[derive(Clone, Copy)]
pub struct Lens {
    /// Lens radius, meters; larger blurs out-of-focus geometry more.
    /// Zero is exactly a pinhole.
    pub aperture_radius: f32,
    /// Distance from the camera to the focal plane along the view axis,
    /// meters. Must be positive.
    pub focus_distance: f32,
}

/// A camera's ray-generation basis: the kernel builds each pixel's ray as
/// `normalize(forward + ndc.x · right + ndc.y · up)` with NDC in [-1, 1],
/// +y up. `forward` is unit length; `right` and `up` are scaled by the
/// field of view and aspect ratio.
pub struct RayBasis {
    /// Screen-right, scaled by `tan(vfov/2) · aspect`.
    pub right: Vec3,
    /// Screen-up, scaled by `tan(vfov/2)`.
    pub up: Vec3,
    /// Unit view direction.
    pub forward: Vec3,
}

impl Camera {
    /// The ray-generation basis for a target with the given aspect ratio
    /// (width / height).
    #[must_use]
    pub fn basis(&self, aspect: f32) -> RayBasis {
        let forward = (self.look_at - self.position).normalize();
        assert!(forward.is_finite(), "camera position and look_at coincide");
        let right = forward.cross(self.up).normalize();
        assert!(right.is_finite(), "camera up is parallel to the view axis");
        let up = right.cross(forward);
        let half_height = (self.vfov_degrees.to_radians() / 2.0).tan();
        RayBasis {
            right: right * half_height * aspect,
            up: up * half_height,
            forward,
        }
    }
}

/// One instance as the GPU assembly reads it: the resident mesh it
/// places, where it stands, and its finished GPU material — what
/// [`Scene::new`] lowers objects into and prep lowers a description into,
/// so both build the same residency through the same helpers.
struct Placement<'a> {
    mesh: &'a GpuMesh,
    transform: Mat4,
    material: Material,
    /// Whether camera rays see it — lowered into the instance's TLAS
    /// visibility mask.
    camera_visible: bool,
    /// The medium this mesh bounds, or `None` for an ordinary surface —
    /// see [`boundary`].
    medium: Option<Medium>,
    /// Which solid wins where refractive interiors overlap — see
    /// [`description::Instance::interior_priority`] and [`interior_priority`].
    priority: u32,
}

/// Build the TLAS: one instance per placement, with `custom_index` =
/// position, so a hit leads back to the right geometry record and
/// material. A camera-invisible placement drops [`ray_mask::CAMERA`]
/// from its mask, so camera rays traverse past it while every other ray
/// still sees it.
fn build_scene_tlas(gpu: &Context, placements: &[Placement]) -> Result<AccelerationStructure> {
    let instances: Vec<TlasInstance> = placements
        .iter()
        .enumerate()
        .map(|(index, placement)| TlasInstance {
            blas: &placement.mesh.blas,
            transform: placement.transform,
            custom_index: index as u32,
            mask: if placement.camera_visible {
                ray_mask::ALL
            } else {
                ray_mask::ALL & !ray_mask::CAMERA
            } as u8,
            // An opacity *map* forces the non-opaque path no matter the
            // constant: the traversal loop must get its per-texel look. A
            // volume boundary forces it too, for the opposite reason — the
            // shadow traversal has to *see* each crossing to measure the
            // extent it bounds, and an opaque one would commit and read as
            // a solid occluder.
            opaque: placement.medium.is_none()
                && placement.material.opacity >= 1.0
                && placement.material.opacity_texture == TEXTURE_NONE,
        })
        .collect();
    gpu.build_tlas("scene.tlas", &instances)
}

/// Per placement, the index of its first [`TriangleLight`], or
/// [`LIGHT_NONE`] — what `GeometryRecord.light` carries. A light's
/// `instance` is its TLAS custom index, which [`build_scene_tlas`] sets to
/// the placement's own position, so it indexes the result directly. One
/// pass, not a scan per placement: a dark instance must not cost the length
/// of the light list to learn that it is dark.
fn first_light_indices(placements: usize, triangles: &[TriangleLight]) -> Vec<u32> {
    let mut first = vec![LIGHT_NONE; placements];
    for (index, light) in triangles.iter().enumerate() {
        let slot = &mut first[light.instance as usize];
        if *slot == LIGHT_NONE {
            *slot = index as u32;
        }
    }
    first
}

/// The tables one pass over the placements produces, each its own buffer.
struct InstanceTables {
    geometry: Buffer,
    materials: Buffer,
    /// The deduplicated media — a handful of records, indexed by
    /// `GeometryRecord::medium` or by `global_medium`, not by instance.
    media: Buffer,
    lights: Buffer,
    /// What fills the open space between instances, indexing `media`, or
    /// [`MEDIUM_NONE`]. Interned alongside the interiors, so it is derived
    /// here rather than carried separately.
    global_medium: u32,
}

/// Upload one scene table as a shader-addressable buffer. Vulkan forbids
/// empty buffers, so an empty table uploads one zeroed record instead: it
/// keeps the address valid, and nothing indexes it — whatever would have
/// pointed here carries a sentinel or a count of zero.
fn upload_table<T: Pod>(gpu: &Context, name: &str, records: &[T]) -> Result<Buffer> {
    let padding = [T::zeroed()];
    gpu.upload_buffer(
        name,
        bytemuck::cast_slice(if records.is_empty() {
            &padding
        } else {
            records
        }),
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
    )
}

/// Upload the per-instance tables: geometry records (each carrying the
/// index of its instance's *first* light record, or [`LIGHT_NONE`], and of
/// its interior's medium), the material array, the deduplicated medium
/// table, and the light records — laid out in the contiguous primitive
/// order `GeometryRecord.light` depends on.
fn upload_instance_tables(
    gpu: &Context,
    placements: &[Placement],
    triangle_lights: &[TriangleLight],
    delta_lights: &[DeltaLight],
    global: Option<Medium>,
) -> Result<InstanceTables> {
    let light_records = crate::lights::build(triangle_lights, delta_lights);
    assert!(
        placements.iter().all(|placement| placement.medium.is_none()
            || placement.material == boundary_material(placement.material, true)),
        "a mesh that bounds a medium must reach the tables inert — see `boundary_material`"
    );
    let materials: Vec<Material> = placements
        .iter()
        .map(|placement| placement.material)
        .collect();
    let (medium_indices, global_medium, media) =
        medium_table(placements.iter().map(placement_medium), global);
    let records: Vec<GeometryRecord> = placements
        .iter()
        .zip(first_light_indices(placements.len(), triangle_lights))
        .zip(medium_indices)
        .map(|((placement, light), medium)| {
            let inverse = placement.transform.inverse();
            assert!(
                inverse.is_finite(),
                "instance transform must be invertible, got {:?}",
                placement.transform
            );
            let boundary = boundary_word(placement);
            assert!(
                boundary & STACK_SCATTERING == 0 || medium != MEDIUM_NONE,
                "an entry carrying the scattering bit must name a medium — see `boundary_word`"
            );
            GeometryRecord {
                positions: placement.mesh.vertices.device_address(),
                normals: placement.mesh.normals.device_address(),
                uvs: placement.mesh.uvs.device_address(),
                indices: placement.mesh.indices.device_address(),
                object_to_world: transform_rows(placement.transform),
                world_to_object: transform_rows(inverse),
                light,
                boundary,
                medium,
                _pad0: 0,
            }
        })
        .collect();
    Ok(InstanceTables {
        geometry: upload_table(gpu, "scene.geometry", &records)?,
        materials: upload_table(gpu, "scene.materials", &materials)?,
        media: upload_table(gpu, "scene.media", &media)?,
        lights: upload_table(gpu, "scene.lights", &light_records)?,
        global_medium,
    })
}

/// Upload the [`SceneTable`] — the one buffer of addresses every kernel
/// reaches scene data through, rebuilt whenever anything it points at
/// moved. `env_placement` is the environment-to-world linear part and its
/// inverse; `env_tint` the `ACEScg` multiplier over the image's radiance.
fn upload_scene_table(
    gpu: &Context,
    resident: &ResidentBuffers,
    env_size: (u32, u32),
    env_placement: (Mat4, Mat4),
    env_tint: Vec3,
    env_select_prob: f32,
    light_count: u32,
) -> Result<Buffer> {
    let table = SceneTable {
        geometry: resident.instances.geometry.device_address(),
        materials: resident.instances.materials.device_address(),
        lights: resident.instances.lights.device_address(),
        texture_params: resident.texture_params.device_address(),
        env_to_world: transform_rows(env_placement.0),
        env_from_world: transform_rows(env_placement.1),
        env_tint: env_tint.to_array(),
        env_select_prob,
        env_marginal: resident.env_marginal.device_address(),
        env_conditional: resident.env_conditional.device_address(),
        env_pdfs: resident.env_pdfs.device_address(),
        env_width: env_size.0,
        env_height: env_size.1,
        media: resident.instances.media.device_address(),
        light_count,
        global_medium: resident.instances.global_medium,
    };
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
    gpu.upload_buffer("scene.table", bytemuck::bytes_of(&table), usage)
}

/// Unpack an emissive instance into per-triangle lights: one record per
/// triangle of the mesh (degenerate ones included), in primitive order,
/// transformed to world space — the contiguity `GeometryRecord.light`
/// depends on.
///
/// Emission is one-sided off the *object-space* winding front; a
/// mirroring transform (negative determinant) reverses the baked corners'
/// world winding relative to that front, so the sign rides along for
/// `connectTriangle` to restore the side — keeping next-event
/// connections on the same face BSDF-sampled hits already emit from
/// (their inverse-transpose normal is mirror-corrected by construction).
fn emissive_triangles(
    positions: &[Vec3],
    triangles: &[[u32; 3]],
    transform: Mat4,
    emission: Vec3,
    instance: u32,
) -> Vec<TriangleLight> {
    let winding = if Mat3::from_mat4(transform).determinant() < 0.0 {
        -1.0
    } else {
        1.0
    };
    triangles
        .iter()
        .enumerate()
        .map(|(primitive, corners)| TriangleLight {
            corners: corners.map(|vertex| transform.transform_point3(positions[vertex as usize])),
            emission,
            instance,
            primitive: primitive as u32,
            winding,
        })
        .collect()
}

/// The environment's GPU half: the radiance image, the three sampling
/// tables, and the emitted power the selection probability weighs.
struct GpuEnvironment {
    image: SampledImage,
    marginal: Buffer,
    conditional: Buffer,
    pdfs: Buffer,
    power: f64,
}

/// Upload the environment's image and sampling tables.
fn upload_environment(gpu: &Context, environment: &Environment) -> Result<GpuEnvironment> {
    let tables = environment.tables();
    let image = gpu.upload_sampled_image(
        "scene.environment",
        environment.width(),
        environment.height(),
        environment.texels(),
    )?;
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
    Ok(GpuEnvironment {
        image,
        marginal: gpu.upload_buffer(
            "scene.env.marginal",
            bytemuck::cast_slice(&tables.marginal),
            usage,
        )?,
        conditional: gpu.upload_buffer(
            "scene.env.conditional",
            bytemuck::cast_slice(&tables.conditional),
            usage,
        )?,
        pdfs: gpu.upload_buffer("scene.env.pdfs", bytemuck::cast_slice(&tables.pdfs), usage)?,
        power: tables.power,
    })
}

/// Weigh the environment against the light list: the power-proportional
/// probability that next-event estimation samples the environment rather
/// than the list. The environment weighs its luminance integral over the
/// sphere — a flux per unit receiver area, so the comparison implicitly
/// stands in a ~1 m² receiver — against [`crate::lights::total_power`]'s
/// per-kind flux measures. The approximations only steer noise: the MIS
/// weights stay exact whatever this probability is. The exact-0/exact-1
/// endpoints *are* load-bearing: the shader walks the light list
/// whenever its draw lands above `select_prob`, so a scene whose list is
/// powerless must pin it to 1, and a black environment (with no other
/// lights either) disables next-event estimation entirely.
fn select_probability(env_power: f64, light_power: f64) -> f32 {
    if light_power == 0.0 {
        f32::from(u8::from(env_power > 0.0))
    } else {
        (env_power / (env_power + light_power)) as f32
    }
}

/// The top three rows of an affine transform, in the kernels' `float4[3]`
/// row-major shape (glam matrices are column-major, hence the transpose).
fn transform_rows(transform: Mat4) -> [[f32; 4]; 3] {
    let rows = transform.transpose();
    [
        rows.x_axis.to_array(),
        rows.y_axis.to_array(),
        rows.z_axis.to_array(),
    ]
}

/// Queue one mesh's buffers and its BLAS build on `upload`. `name` is the
/// mesh's bare name — the `scene.mesh.` prefix that puts the bytes in the
/// right memory bucket ([`crate::gpu`]'s ledger) is added here, once, so no
/// caller has to remember it and no caller can add it twice.
///
/// The returned [`GpuMesh`] is resident only once the caller's
/// [`Upload::finish`] returns; every caller finishes before the TLAS build
/// that first reads these BLASes.
fn upload_mesh(upload: &mut Upload, name: &str, mesh: &Mesh) -> Result<GpuMesh> {
    assert_eq!(
        mesh.normals.len(),
        mesh.positions.len(),
        "a mesh needs one shading normal per vertex"
    );
    assert_eq!(
        mesh.uvs.len(),
        mesh.positions.len(),
        "a mesh needs one uv per vertex (zeros when unauthored)"
    );
    // BUILD_INPUT for the BLAS build; STORAGE + device address so the
    // shading kernel can fetch triangle corners afterwards.
    let usage = vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
        | vk::BufferUsageFlags::STORAGE_BUFFER;
    let vertices = upload.buffer(
        &format!("scene.mesh.{name}.vertices"),
        bytemuck::cast_slice(&mesh.positions),
        usage,
    )?;
    let normals = upload.buffer(
        &format!("scene.mesh.{name}.normals"),
        bytemuck::cast_slice(&mesh.normals),
        usage,
    )?;
    let uvs = upload.buffer(
        &format!("scene.mesh.{name}.uvs"),
        bytemuck::cast_slice(&mesh.uvs),
        usage,
    )?;
    let indices = upload.buffer(
        &format!("scene.mesh.{name}.indices"),
        bytemuck::cast_slice(&mesh.triangles),
        usage,
    )?;
    let blas = upload.blas(
        &format!("scene.mesh.{name}.blas"),
        &vertices,
        mesh.positions.len() as u32,
        &indices,
        mesh.triangles.len() as u32,
    )?;
    Ok(GpuMesh {
        blas,
        vertices,
        normals,
        uvs,
        indices,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One predicate decides two things — an instance's boundary word and
    /// whether the wavefront allocates the medium stack at all — so what
    /// counts as an interior is worth pinning. Closed glass qualifies at
    /// transmission depth 0: it absorbs nothing, but the closure's
    /// `exiting` flag still reads the stack to know it is leaving. A
    /// thin-walled sheet has no interior to be inside of.
    #[test]
    fn closed_transmissive_surfaces_bound_interiors() {
        let glass = Material {
            transmission_depth: 0.0,
            ..Material::glass(0.1, 1.5)
        };
        assert_eq!(boundary(&glass, false), BOUNDARY_REFRACTIVE);
        assert_eq!(boundary(&glass.thin_walled(), false), BOUNDARY_OPAQUE);
        assert_eq!(
            boundary(&Material::matte(Vec3::ONE, 0.5), false),
            BOUNDARY_OPAQUE,
            "an opaque surface has no interior"
        );
        assert_eq!(
            boundary(&glass, true),
            BOUNDARY_NULL,
            "bounding a medium wins over any surface the material describes"
        );
    }

    /// What a null boundary keeps of its material, and what it must not.
    /// Emission is the load-bearing one: it would reach the light list, and
    /// next-event estimation would then aim at a surface phase and BSDF
    /// sampling can never hit. Opacity follows because the boundary has to
    /// traverse as non-opaque and still commit.
    #[test]
    fn a_volume_boundary_describes_no_surface() {
        let authored = Material {
            emission: Vec3::splat(4.0),
            opacity: 0.25,
            opacity_texture: 7,
            ..Material::matte(Vec3::ONE, 0.5)
        };
        assert_eq!(
            boundary_material(authored, false),
            authored,
            "an ordinary surface keeps everything it was authored with"
        );
        let inert = boundary_material(authored, true);
        assert_eq!(inert.emission, Vec3::ZERO);
        assert!(inert.opacity >= 1.0);
        assert_eq!(inert.opacity_texture, TEXTURE_NONE);
        assert_eq!(
            inert.base_color, authored.base_color,
            "only what would be mistaken for a surface is dropped"
        );
    }

    /// Bounding an interior and filling one are separate questions: the
    /// depth-0 glass above is still a refractive boundary, but its tint
    /// sits on the interface, so it encloses vacuum and takes no record.
    /// Where there is a medium, the record is the σ the kernel used to
    /// recompute at every hit.
    #[test]
    fn only_absorbing_interiors_take_a_medium_record() {
        let clear = Material {
            transmission_depth: 0.0,
            ..Material::glass(0.1, 1.5)
        };
        let tinted = Material {
            transmission_color: Vec3::new(0.25, 0.5, 1.0),
            transmission_depth: 2.0,
            ..clear
        };
        let fog = Medium {
            absorption: Vec3::splat(0.2),
            scattering: Vec3::splat(0.3),
            anisotropy: 0.0,
        };
        let (indices, global, records) = medium_table(
            [
                tinted,
                clear,
                tinted,
                Material::matte(Vec3::ONE, 0.5),
                tinted.thin_walled(),
            ]
            .iter()
            .map(interior),
            Some(fog),
        );
        assert_eq!(
            indices,
            [0, MEDIUM_NONE, 0, MEDIUM_NONE, MEDIUM_NONE],
            "instances sharing an interior share its record"
        );
        assert_eq!(
            (global, records.len()),
            (1, 2),
            "the global medium interns beside the interiors, in its own record"
        );
        assert_eq!(
            records[0].sigma_t.map(f32::to_bits),
            [0.25f32, 0.5, 1.0].map(|channel| (-channel.ln() / 2.0).to_bits()),
            "σ_t = -ln(transmission color) / transmission depth, to the bit"
        );
        assert_eq!(
            (records[1].sigma_t, records[1].sigma_s),
            ([0.5; 3], [0.3; 3]),
            "an authored medium's extinction is σ_a + σ_s"
        );
        // Value keying, not identity: a global medium that matches an
        // interior *is* that interior's medium, one record for both.
        let deep = Material {
            transmission_color: Vec3::new(0.25, 0.5, 0.75),
            ..tinted
        };
        let matching = Medium {
            absorption: Vec3::from(interior(&deep).expect("an interior").sigma_t),
            scattering: Vec3::ZERO,
            anisotropy: 0.0,
        };
        let shared = medium_table(std::iter::once(interior(&deep)), Some(matching));
        assert_eq!(
            (shared.0, shared.1, shared.2.len()),
            (vec![0], 0, 1),
            "a global medium that matches an interior is that interior's medium"
        );
        assert!(
            medium_table(std::iter::empty(), None).2.is_empty()
                && medium_table(std::iter::once(interior(&clear)), None)
                    .2
                    .is_empty(),
            "a scene with nothing to fill uploads no records"
        );
    }

    /// σ is per-instance data that outlives the prep pass, so a malformed
    /// material must resolve to vacuum here rather than to a coefficient
    /// that poisons every hit on the instance. These are the inputs the
    /// kernel's old `transmissionDepth > 0.0` test used to turn away.
    #[test]
    fn a_malformed_interior_is_vacuum() {
        let glass = Material::glass(0.1, 1.5);
        for depth in [f32::NAN, -1.0, -0.0] {
            let material = Material {
                transmission_color: Vec3::splat(0.5),
                transmission_depth: depth,
                ..glass
            };
            assert!(
                interior(&material).is_none(),
                "depth {depth} is not an interior"
            );
        }
        // Out-of-range channels clamp exactly as the kernel's did: a hard
        // zero would take σ to infinity, and anything above 1 would make
        // Beer–Lambert amplify.
        let extremes = interior(&Material {
            transmission_color: Vec3::new(0.0, -3.0, 8.0),
            transmission_depth: 1.0,
            ..glass
        })
        .expect("a positive depth fills the interior");
        assert_eq!(
            extremes.sigma_t.map(f32::to_bits),
            [1e-4f32, 1e-4, 1.0].map(|channel| (-channel.ln()).to_bits())
        );
        assert!(
            extremes.sigma_s.iter().all(|channel| *channel <= 0.0),
            "an interior with no authored scatter must derive none: σ_s is what routes its \
             paths to the volume stage, and there would be nothing there to sample"
        );
    }

    /// The scattering split is `OpenPBR`'s subtractive read: the
    /// color/depth logarithm stays the extinction bit for bit, the scatter
    /// carves `σ_s` out of it, and a channel whose scatter outruns the
    /// extinction it sits inside is repaired by a gray shift — a medium
    /// may fall short of the authored color, never amplify.
    #[test]
    fn transmission_scatter_splits_the_extinction_subtractively() {
        let absorbing = Material {
            transmission_color: Vec3::new(0.25, 0.5, 0.8),
            transmission_depth: 2.0,
            ..Material::glass(0.1, 1.5)
        };
        let base = interior(&absorbing).expect("an interior");
        let split = interior(&Material {
            transmission_scatter: Vec3::new(0.1, 0.2, 0.0),
            transmission_scatter_anisotropy: 0.7,
            ..absorbing
        })
        .expect("an interior");
        assert_eq!(
            split.sigma_t.map(f32::to_bits),
            base.sigma_t.map(f32::to_bits),
            "scatter within the extinction leaves σ_t exactly the color's"
        );
        assert_eq!(
            (split.sigma_s, split.g),
            ([0.05, 0.1, 0.0], 0.7),
            "σ_s = transmission_scatter / transmission_depth"
        );
        // Milk: a white color implies zero extinction, so every scattered
        // channel is negative absorption until the shift repairs it — and
        // the shift is gray, the same in every channel, so the repaired
        // σ_a = σ_t − σ_s keeps the authored hue.
        let milk = interior(&Material {
            transmission_color: Vec3::ONE,
            transmission_depth: 1.0,
            transmission_scatter: Vec3::new(0.4, 0.5, 0.6),
            ..Material::glass(0.1, 1.5)
        })
        .expect("an interior");
        assert_eq!(
            (milk.sigma_t, milk.sigma_s),
            ([0.6; 3], [0.4, 0.5, 0.6]),
            "the gray shift grows σ_t just enough that no channel amplifies"
        );
        // A NaN anisotropy falls to isotropic rather than riding `clamp`
        // through to the phase function.
        let spun = interior(&Material {
            transmission_scatter: Vec3::ONE,
            transmission_scatter_anisotropy: f32::NAN,
            ..absorbing
        })
        .expect("an interior");
        assert_eq!(
            spun.g.to_bits(),
            0.0f32.to_bits(),
            "a NaN anisotropy is no anisotropy at all"
        );
        // Malformed scatter falls to purely absorbing — the same record,
        // to the bit, as if it were never authored — mirroring how a
        // malformed depth falls to vacuum.
        for scatter in [Vec3::splat(f32::NAN), Vec3::splat(-1.0), Vec3::ZERO] {
            let fallen = interior(&Material {
                transmission_scatter: scatter,
                transmission_scatter_anisotropy: 0.7,
                ..absorbing
            })
            .expect("an interior");
            assert_eq!(
                bytemuck::cast::<_, [u32; 8]>(fallen),
                bytemuck::cast::<_, [u32; 8]>(base),
                "scatter {scatter} is no scatter at all"
            );
        }
    }

    /// The ray basis must be orthogonal, oriented (up skyward, right = +X
    /// when looking down −Z), and scaled by fov and aspect — the kernel
    /// trusts it blindly.
    #[test]
    fn camera_basis_is_orthogonal_and_fov_scaled() {
        let camera = Camera {
            position: Vec3::new(0.0, 2.0, 5.0),
            look_at: Vec3::new(0.0, 1.0, 0.0),
            up: Vec3::Y,
            vfov_degrees: 90.0,
            lens: None,
        };
        let basis = camera.basis(2.0);
        assert!((basis.forward.length() - 1.0).abs() < 1e-6);
        assert!(basis.forward.dot(basis.right).abs() < 1e-6);
        assert!(basis.forward.dot(basis.up).abs() < 1e-6);
        assert!(basis.right.dot(basis.up).abs() < 1e-6);
        // tan(90° / 2) = 1, so |up| = 1 and |right| = aspect.
        assert!((basis.up.length() - 1.0).abs() < 1e-6);
        assert!((basis.right.length() - 2.0).abs() < 1e-6);
        assert!(basis.up.y > 0.0);
        assert!(basis.right.x > 0.0);
    }

    /// `up` carries roll: flipping it upside down while looking down −Z
    /// must flip the screen — up and right both negate. This is the
    /// orientation the format's camera op commits to (`up` is the roll
    /// control), so the basis has to honor it, not just world +Y.
    #[test]
    fn camera_up_carries_roll() {
        let level = Camera {
            position: Vec3::new(0.0, 0.0, 5.0),
            look_at: Vec3::ZERO,
            up: Vec3::Y,
            vfov_degrees: 60.0,
            lens: None,
        };
        let inverted = Camera {
            up: -Vec3::Y,
            ..level
        };
        let (a, b) = (level.basis(1.0), inverted.basis(1.0));
        assert!((a.up + b.up).length() < 1e-6, "{} vs {}", a.up, b.up);
        assert!((a.right + b.right).length() < 1e-6);
        assert!((a.forward - b.forward).length() < 1e-6);
    }

    /// Every placement points at the first of its own light records, and a
    /// placement with no emission points nowhere.
    #[test]
    fn light_indices_point_at_each_instance_first_record() {
        let triangle = |instance, primitive| TriangleLight {
            corners: [Vec3::ZERO, Vec3::X, Vec3::Y],
            emission: Vec3::ONE,
            instance,
            primitive,
            winding: 1.0,
        };
        // Grouped, the shape lowering emits: instance 0 over two triangles,
        // instance 1 dark, instance 2 over one.
        let grouped = [triangle(0, 0), triangle(0, 1), triangle(2, 0)];
        assert_eq!(first_light_indices(3, &grouped), vec![0, LIGHT_NONE, 2]);
        // Interleaved: still the lowest index per instance. Contiguity is
        // lowering's contract; this fill does not depend on it, so a
        // restructured lowering cannot silently change the answer.
        let mixed = [
            triangle(2, 0),
            triangle(0, 0),
            triangle(2, 1),
            triangle(0, 1),
        ];
        assert_eq!(first_light_indices(3, &mixed), vec![1, LIGHT_NONE, 0]);
        assert_eq!(first_light_indices(3, &[]), vec![LIGHT_NONE; 3]);
        assert!(first_light_indices(0, &[]).is_empty());
    }

    /// Two BLASes and the TLAS build without errors on real hardware
    /// (validation complaints appear via the debug messenger in the log).
    #[test]
    fn demo_scene_builds() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        Scene::demo(&gpu).expect("demo scene should build");
    }
}
