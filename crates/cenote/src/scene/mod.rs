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
pub(crate) mod curves;
mod demo;
pub mod description;
mod lower;
mod many_lights;
mod prep;
mod shapes;

pub use shapes::{cube, ground_plane, icosphere};

use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec2, Vec3};

use self::description::Geometry;
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
    /// Index into the same table of the subsurface interior behind this
    /// instance's material, or [`MEDIUM_NONE`] — see [`subsurface`],
    /// whose gate the closure mirrors so the walk kernel only ever reads
    /// a real index.
    subsurface: u32,
}

const _: () = assert!(size_of::<GeometryRecord>() == 144);

/// `GeometryRecord::boundary` values — the `BOUNDARY_*` constants in
/// `shaders/scene.slang`.
const BOUNDARY_OPAQUE: u32 = 0;
const BOUNDARY_REFRACTIVE: u32 = 1;
const BOUNDARY_NULL: u32 = 2;
/// The class's byte of the boundary word — `BOUNDARY_MASK` in
/// `shaders/scene.slang`.
const BOUNDARY_MASK: u32 = 0xff;

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

/// Whether a material already is what [`boundary_material`] would leave of
/// it. Tested on these three fields rather than by `==` on the whole
/// struct, which a NaN in any *unrelated* float — authored NaNs pass
/// through lowering — would fail: the warn in `lower.rs` would misreport
/// an untouched material, and the assert at [`upload_instance_tables`]
/// would panic over an invariant that holds.
fn is_boundary_inert(material: &Material) -> bool {
    material.emission == Vec3::ZERO
        && material.opacity >= 1.0
        && material.opacity_texture == TEXTURE_NONE
}

/// The placement-derived media predicates the `Scene::has_*` accessors
/// cache — read in one pass through [`boundary_word`], the same
/// classification the tables carry, so what the wavefront allocates and
/// routes on can never disagree with what a kernel can act on.
/// [`Scene::has_global_medium`] stays outside: settings carry it, not
/// placements.
#[derive(Clone, Copy)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "five independent placement predicates, cached together for the wavefront's \
              per-wave read"
)]
struct PlacementMedia {
    /// Some placement closes a refractive interior, so the medium set can
    /// hold one — see [`Scene::has_interiors`].
    interiors: bool,
    /// Some placement bounds a medium: a null boundary, crossed never
    /// shaded — see [`Scene::has_volumes`].
    volumes: bool,
    /// Some interior authors a nonzero nesting priority. Implies
    /// `interiors` — only a refractive boundary carries priority bits — so
    /// this can never gate a read of a buffer that was not allocated. And
    /// all-zero priorities are never *strictly* less than one another, so
    /// a scene that authors none can never suppress an interface — which
    /// is why it renders bit-identically rather than by a special case.
    priority: bool,
    /// Some interior *scatters* — the one kind that owes the volume stage
    /// an event, so it counts toward [`Scene::has_media`] where a purely
    /// absorbing one does not.
    scattering_interiors: bool,
    /// Some material carries an active subsurface lobe — the exact
    /// condition [`subsurface`] interns a medium record under, so what the
    /// wavefront records the walk stage for and what a closure can draw
    /// can never disagree.
    subsurface: bool,
    /// Some bounded medium is heterogeneous — a density grid fills it, so
    /// the volume stage dispatches its tracking entry point and binds the
    /// grid pool. Implies `volumes`.
    heterogeneous: bool,
    /// Some heterogeneous medium carries a temperature field, so the
    /// tracker collects blackbody emission at its collisions. Implies
    /// `heterogeneous`.
    emissive: bool,
}

fn placement_media(placements: &[Placement]) -> PlacementMedia {
    let mut media = PlacementMedia {
        interiors: false,
        volumes: false,
        priority: false,
        scattering_interiors: false,
        subsurface: false,
        heterogeneous: false,
        emissive: false,
    };
    for placement in placements {
        let word = boundary_word(placement);
        media.interiors |= word & BOUNDARY_MASK == BOUNDARY_REFRACTIVE;
        media.volumes |= word & BOUNDARY_MASK == BOUNDARY_NULL;
        media.priority |= word & (MAX_PRIORITY << STACK_PRIORITY_SHIFT) != 0;
        media.scattering_interiors |= word & STACK_SCATTERING != 0;
        media.subsurface |= subsurface(&placement.material).is_some();
        let volume = placement.medium.as_ref().and_then(|medium| medium.volume.as_ref());
        media.heterogeneous |= volume.is_some();
        media.emissive |= volume.is_some_and(|volume| volume.temperature.is_some());
    }
    media
}

/// Medium-set entry bits, mirroring `STACK_*` in `shaders/pathstate.slang`
/// — [`boundary_word`] bakes them into [`GeometryRecord::boundary`], so
/// the shader's `stackEntry` lifts them into an entry without shifting
/// anything.
pub(crate) const STACK_PRIORITY_SHIFT: u32 = 25;
pub(crate) const STACK_SCATTERING: u32 = 1 << 24;
pub(crate) const STACK_INTERIOR: u32 = 1 << 31;

/// The empty medium-set slot — `STACK_EMPTY` in `shaders/pathstate.slang`:
/// the instance-index field at all ones, no flag bits. What
/// [`SceneTable::camera_media`] uploads [`MEDIUM_SLOTS`] of.
pub(crate) const STACK_EMPTY: u32 = 0x00ff_ffff;

/// How many media a path can be inside at once — `MEDIUM_SLOTS` in
/// `shaders/pathstate.slang`, where the policy lives. The host knows it
/// only as a width: [`SceneTable::camera_media`]'s, and the stack buffer's
/// per-path stride in `wavefront.rs`.
pub(crate) const MEDIUM_SLOTS: usize = 4;

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

// The class must survive the packing: the priority has to clear the
// class's byte, fill exactly the bits directly under `STACK_INTERIOR` (one
// `max` ranks the whole set only because nothing sits between them), and
// leave the scattering bit its own gap.
const _: () = assert!(BOUNDARY_NULL < BOUNDARY_MASK && (1 << STACK_PRIORITY_SHIFT) > BOUNDARY_MASK);
const _: () = assert!((MAX_PRIORITY + 1) << STACK_PRIORITY_SHIFT == STACK_INTERIOR);
const _: () =
    assert!(STACK_SCATTERING > BOUNDARY_MASK && STACK_SCATTERING < 1 << STACK_PRIORITY_SHIFT);

/// A medium as the renderer holds it: coefficients per `ACEScg` channel in
/// inverse meters, the phase function's anisotropy, and — for a
/// heterogeneous medium — the density grid that multiplies the
/// coefficients point by point.
#[derive(Clone, Debug, PartialEq)]
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
    /// The density grid, resolved at lowering; `None` is homogeneous.
    pub volume: Option<GridVolume>,
}

/// A heterogeneous medium's grid, resolved at lowering from the `.nvdb`
/// header: everything prep derives the shell and the two sampling
/// transforms from, with the multi-GiB payload — and the majorant lattice
/// built out of it — left to the [`crate::vdb::GridPool`] upload.
///
/// "Asset space" is the grid's own world space — the space its `.vdb` was
/// authored in, which the instance transform then places in the scene.
#[derive(Clone, Debug, PartialEq)]
pub struct GridVolume {
    /// The prepared `.nvdb` (the cache output, or the authored file
    /// itself) — what the pool uploads.
    pub nvdb: PathBuf,
    /// The grid's name inside the container.
    pub grid: String,
    /// Rows of the affine asset→index map, from the grid's own transform.
    pub asset_to_index: [[f32; 4]; 3],
    /// Unit cube [0, 1]³ → the grid's active bounds in asset space,
    /// dilated by one voxel so the trilinear stencil's whole support sits
    /// inside the shell — the transform the `MediumBounds` placement bakes.
    pub bounds_to_asset: Mat4,
    /// Cells of the majorant lattice per axis, from
    /// [`crate::vdb::majorant_resolution`].
    pub majorant_res: [u32; 3],
    /// Index space → lattice cell coordinates: `cell = index · scale +
    /// bias`. The lattice holds unitless density ceilings, so coefficients
    /// multiply in-shader and a lookdev edit of σ touches no grid data.
    pub majorant_scale: [f32; 3],
    /// The translation half of the same map.
    pub majorant_bias: [f32; 3],
    /// The temperature field beside the density, or `None` for a medium
    /// that does not emit — including one whose emission is black, which
    /// lowering resolves to the same thing. It shares `asset_to_index`
    /// (lowering refuses the pair otherwise), so one affine reaches both.
    pub temperature: Option<String>,
    /// Kelvin from a field value: `K = value · scale + offset`.
    pub kelvin_scale: f32,
    /// The offset half of that map.
    pub kelvin_offset: f32,
    /// `ACEScg` multiplier on the blackbody radiance at that temperature.
    pub emission: Vec3,
}

/// One entry of the medium table. Mirrors `struct MediumRecord` in
/// `shaders/scene.slang`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MediumRecord {
    /// Extinction `σ_a` + `σ_s`, per `ACEScg` channel, in inverse meters —
    /// the σ of Beer–Lambert, and the density distance sampling draws
    /// against. For a heterogeneous medium these are the *coefficients*
    /// the grid's density multiplies: σ(x) = density(x) · σ.
    sigma_t: [f32; 3],
    /// Henyey–Greenstein anisotropy, clamped inside (−1, 1).
    g: f32,
    /// Scattering `σ_s`. Zero routes no path to the volume stage.
    sigma_s: [f32; 3],
    /// The grid's byte offset in the pool, or [`GRID_NONE`] for a
    /// homogeneous medium. It took `pad0`'s word, so every field a
    /// homogeneous record uses kept its offset.
    grid: u32,
    /// Rows of the world→index transform: the grid's asset→index map
    /// composed with the placement's world→asset inverse. Zero (harmless)
    /// on homogeneous records.
    world_to_index: [[f32; 4]; 3],
    /// Index space → majorant-cell coordinates (`cell = index · scale +
    /// bias`), the lattice's byte offset in the pool, and its cell counts
    /// packed ten bits per axis, x lowest.
    majorant_scale: [f32; 3],
    majorant: u32,
    majorant_bias: [f32; 3],
    majorant_res: u32,
    /// `ACEScg` multiplier on the blackbody radiance at that temperature.
    emission: [f32; 3],
    /// The temperature grid's byte offset in the pool, or [`GRID_NONE`] for
    /// a medium that does not emit. It is read through `world_to_index`,
    /// the density grid's — lowering refuses a pair that would not be.
    temperature_grid: u32,
    /// Kelvin from a field value: `K = value · scale + offset`.
    kelvin_scale: f32,
    kelvin_offset: f32,
    /// `world_to_index`'s `float4` rows force a 16-byte stride, so the
    /// record rounds to 144 bytes with or without this — the tail is only
    /// named here so both sides spell the same layout.
    _pad0: [f32; 2],
}

const _: () = assert!(size_of::<MediumRecord>() == 144);

/// A [`MediumRecord`] as its own identity: the bits it computed to, which
/// is what interning keys on. Sized off the record so a field added to one
/// cannot leave the other behind.
type MediumKey = [u32; size_of::<MediumRecord>() / 4];

/// [`MediumRecord::grid`] for a homogeneous medium — mirrors `GRID_NONE`
/// in `shaders/scene.slang`.
const GRID_NONE: u32 = u32::MAX;

impl MediumRecord {
    /// A homogeneous record: coefficients only, no grid, zeroed transform.
    /// The one construction path for interiors, subsurface interiors, and
    /// gridless authored media, so none of them can drift from the
    /// sentinel.
    fn homogeneous(sigma_t: [f32; 3], g: f32, sigma_s: [f32; 3]) -> Self {
        Self {
            sigma_t,
            g,
            sigma_s,
            grid: GRID_NONE,
            world_to_index: [[0.0; 4]; 3],
            majorant_scale: [0.0; 3],
            majorant: 0,
            majorant_bias: [0.0; 3],
            majorant_res: 0,
            emission: [0.0; 3],
            temperature_grid: GRID_NONE,
            kelvin_scale: 0.0,
            kelvin_offset: 0.0,
            _pad0: [0.0; 2],
        }
    }
}

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

impl From<&Medium> for MediumRecord {
    /// The homogeneous half of a medium — coefficients and phase. A
    /// heterogeneous medium's grid fields land in
    /// [`upload_instance_tables`], where the pool offset and the
    /// placement transform exist.
    fn from(medium: &Medium) -> Self {
        Self::homogeneous(
            (medium.absorption + medium.scattering).to_array(),
            bounded_anisotropy(medium.anisotropy),
            medium.scattering.to_array(),
        )
    }
}

/// What fills the interior this material closes, as an extinction — or
/// `None` for vacuum: an opaque or thin-walled surface bounds none, and a
/// depth of zero puts the tint on the interface, where the closure applies
/// it instead. The depth test is written positively so that a NaN falls to
/// vacuum rather than to a σ of NaN.
///
/// σ derived here on the host and σ derived from the same color on the
/// device differ: Vulkan specifies `log` only to 2^-21 *absolute* inside
/// [0.5, 2], a large *relative* error where the color is near 1 and the
/// logarithm near 0 — up to 9e-6 on the corpus's σ. The host's correctly
/// rounded `ln` is the accurate side of that.
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
        let sigma_t = material.transmission_color.to_array().map(|channel| {
            // `clamp` would pass a NaN channel straight into σ, poisoning
            // every hit on the instance; it falls to transparent instead,
            // in the spirit of the depth's vacuum.
            if channel.is_nan() {
                0.0
            } else {
                -channel.clamp(1e-4, 1.0).ln() / depth
            }
        });
        let scatter = material.transmission_scatter.max(Vec3::ZERO);
        if !scatter.cmpgt(Vec3::ZERO).any() {
            // Purely absorbing: return σ_t untouched — even adding a zero
            // shift would flip a white channel's −0.0, and an absorbing
            // interior's record must be exactly its authored logarithm.
            return MediumRecord::homogeneous(sigma_t, 0.0, [0.0; 3]);
        }
        let sigma_s = scatter / depth;
        let shift = (sigma_s - Vec3::from(sigma_t)).max_element().max(0.0);
        MediumRecord::homogeneous(
            (Vec3::from(sigma_t) + shift).to_array(),
            bounded_anisotropy(material.transmission_scatter_anisotropy),
            sigma_s.to_array(),
        )
    })
}

/// Van de Hulst's fit, as `OpenPBR` pins it — named so the two directions
/// cannot drift apart. `F` is `B`² to the five digits the specification
/// prints, which is what leaves [`multiple_scatter_color`] *linear*; the
/// printed value is what the forward direction evaluates, so prep's
/// arithmetic is the specification's to the bit.
const FIT_A: f32 = 4.09712;
const FIT_B: f32 = 4.20863;
const FIT_D: f32 = 9.59217;
const FIT_E: f32 = 41.6808;
const FIT_F: f32 = 17.7126;

/// The *single*-scatter albedo `α` a walk needs for its interior to arrive
/// at the authored *multiple*-scatter albedo `color` (`C`), through van de
/// Hulst's inversion: `s` = A + B·C − √(D + E·C + F·C²), then
/// `α` = (1 − s²) / (1 − g·s²). With `s²` in [0, 1] and `g` bounded inside
/// (−1, 1), `α` lands in [0, 1) on its own — `σ_a` = `σ_t` − `σ_s` can
/// never go negative, so unlike [`interior`] no repair shift is needed.
#[must_use]
pub fn single_scatter_albedo(color: f32, g: f32) -> f32 {
    // A NaN channel falls to black — and black is exact: the fit leaves a
    // ~6e-6 residual at its black endpoint, snapped out so an unscattering
    // channel is *purely* absorbing, not almost.
    if color.is_nan() || color <= 0.0 {
        return 0.0;
    }
    let g = bounded_anisotropy(g);
    let c = color.min(1.0);
    let s = FIT_A + FIT_B * c - (FIT_D + FIT_E * c + FIT_F * c * c).sqrt();
    (1.0 - s * s) / (1.0 - g * s * s)
}

/// The way back: the `subsurface_color` that authors a given
/// *single*-scatter albedo — what an importer needs when its source
/// describes an interior by its coefficients rather than by the color they
/// walk to.
///
/// Closed-form, not searched. Inverting the albedo gives `s²` =
/// (1 − α) / (1 − g·α); squaring the fit then cancels its `F·C²` against
/// the radicand's `B²·C²` — the two agree by construction — and what
/// remains is linear:
/// `C` = ((A − s)² − D) / (E − 2·B·(A − s)), whose denominator stays near
/// 7 to 16 across the whole domain.
///
/// That numerator is written factored, `(A − √D − s)(A + √D − s)`, because
/// unfactored it is a difference of two numbers near 9.6 that vanishes as
/// `C` → 0 — the whole black end would be cancellation. Factored, `A − √D`
/// is 0.999997 and the subtraction is between numbers of the size of the
/// answer. The endpoints still clamp: the fit's own residual would
/// otherwise put `C` a part in 10⁵ outside [0, 1].
#[must_use]
pub fn multiple_scatter_color(alpha: f32, g: f32) -> f32 {
    let root = FIT_D.sqrt();
    let g = bounded_anisotropy(g);
    let alpha = if alpha.is_nan() {
        0.0
    } else {
        alpha.clamp(0.0, 1.0)
    };
    let s = ((1.0 - alpha) / (1.0 - g * alpha)).max(0.0).sqrt();
    ((FIT_A - root - s) * (FIT_A + root - s) / (FIT_E - 2.0 * FIT_B * (FIT_A - s))).clamp(0.0, 1.0)
}

/// What fills the interior behind a subsurface material — or `None` where
/// the lobe is off (weight or radius nonpositive, thin-walled has no
/// interior). Gates are written positively so a NaN falls to `None`.
///
/// `σ_t` per channel is the reciprocal mean free path,
/// 1 / (`subsurface_radius` · `subsurface_radius_scale`), floored at a
/// micron so an authored zero stays a finite (just enormous) extinction.
/// `σ_s` is [`single_scatter_albedo`] of the authored color.
fn subsurface(material: &Material) -> Option<MediumRecord> {
    (material.subsurface_weight > 0.0
        && material.subsurface_radius > 0.0
        && material.thin_walled == 0)
        .then(|| {
            let g = bounded_anisotropy(material.subsurface_scatter_anisotropy);
            let sigma_t = material.subsurface_radius_scale.to_array().map(|scale| {
                1.0 / (material.subsurface_radius * scale.max(0.0)).max(1e-6)
            });
            let alpha = material
                .subsurface_color
                .to_array()
                .map(|channel| single_scatter_albedo(channel, g));
            MediumRecord::homogeneous(
                sigma_t,
                g,
                [
                    alpha[0] * sigma_t[0],
                    alpha[1] * sigma_t[1],
                    alpha[2] * sigma_t[2],
                ],
            )
        })
}

/// Set above a [`GeometryRecord::subsurface`] index where a map rather than
/// a constant decides what the interior is made of — `SUBSURFACE_TEXTURED`
/// in `shaders/scene.slang`, whose comment carries the reasoning.
const SUBSURFACE_TEXTURED: u32 = 1 << 31;

/// Whether the interior behind this material varies across its surface —
/// exactly the slots [`subsurface`] computes the record *from*. The weight
/// is deliberately absent: it decides whether the lobe is drawn at all,
/// which the closure settles from the resolved material, and it never
/// reaches the record's coefficients.
fn subsurface_textured(material: &Material) -> bool {
    [
        material.subsurface_color_texture,
        material.subsurface_radius_texture,
        material.subsurface_radius_scale_texture,
        material.subsurface_scatter_anisotropy_texture,
    ]
    .iter()
    .any(|&slot| slot != TEXTURE_NONE)
}

/// What the temperature mapping actually reached, at scene load.
///
/// A medium's emission scale reads as absurd — `explosion.ron` wants 400 —
/// because it multiplies two small physical numbers: how dim a flame is
/// beside a 6500 K body, and how weakly smoke absorbs. Both are knowable
/// and neither is guessable, so the hottest place the mapping reaches is
/// printed with the radiance an optically thick core there would settle
/// at: `emission · B(K) · σ_a/σ_t`, which is what the film will show and
/// is free of the density. The peak itself costs nothing — the majorant
/// scan every uploaded grid gets has already reduced the field to its
/// largest stored value, which for a temperature grid is exactly this.
fn report_emission(field: &str, peak: f32, volume: &GridVolume, record: &MediumRecord) {
    let kelvin = f64::from(peak * volume.kelvin_scale + volume.kelvin_offset);
    let sigma_t = Vec3::from(record.sigma_t);
    // Per channel and then to luminance, so a colored medium is not
    // reported through its red. A medium that extinguishes nothing never
    // collides, so it settles at nothing — which the division would
    // otherwise say as a NaN.
    let share = (sigma_t - Vec3::from(record.sigma_s)) / sigma_t.max(Vec3::splat(f32::MIN));
    let core = volume.emission * crate::blackbody::radiance(kelvin) * share;
    log::info!(
        "  emission: \"{field}\" peaks at {peak:.4} → {kelvin:.0} K; \
         a thick core there settles at {:.3}",
        crate::color::luminance(core),
    );
}

/// What fills a placement: the medium its mesh bounds, or — for an ordinary
/// surface — whatever its material's interior holds. A mesh that bounds a
/// medium is a null boundary, so the material's own interior can never also
/// apply.
///
/// `grids` maps every heterogeneous medium's (nvdb, grid) to where its
/// payload and majorant lattice landed — uploaded by
/// [`upload_instance_tables`] before any record is built, so the lookup
/// cannot miss.
fn placement_medium(
    placement: &Placement,
    grids: &HashMap<(PathBuf, String), crate::vdb::ResidentGrid>,
) -> Option<MediumRecord> {
    let Some(medium) = placement.medium.as_ref() else {
        return interior(&placement.material);
    };
    let mut record = MediumRecord::from(medium);
    if let Some(volume) = &medium.volume {
        let resident = grids[&(volume.nvdb.clone(), volume.grid.clone())];
        record.grid = resident.grid;
        record.majorant = resident.majorant;
        record.majorant_scale = volume.majorant_scale;
        record.majorant_bias = volume.majorant_bias;
        let res = volume.majorant_res;
        assert!(
            res.iter().all(|&axis| axis <= MAJORANT_RES_MAX),
            "majorant resolution {res:?} exceeds the ten bits the record packs it into"
        );
        record.majorant_res = res[0] | res[1] << 10 | res[2] << 20;
        if let Some(temperature) = &volume.temperature {
            let field = grids[&(volume.nvdb.clone(), temperature.clone())];
            record.temperature_grid = field.grid;
            record.kelvin_scale = volume.kelvin_scale;
            record.kelvin_offset = volume.kelvin_offset;
            record.emission = volume.emission.to_array();
            report_emission(temperature, field.field_max, volume, &record);
        }
        // world → asset is the *author's* transform inverted, which is the
        // placement's (element · bounds) with the bounds folded back out:
        // B · (E·B)⁻¹ = E⁻¹. Composed with the grid's own asset→index map,
        // one affine takes a world sample point to index space.
        record.world_to_index = transform_rows(
            rows_to_mat4(&volume.asset_to_index)
                * volume.bounds_to_asset
                * placement.transform.inverse(),
        );
    }
    Some(record)
}

/// The largest cell count per axis the record's packed `majorant_res` word
/// can hold — mirrors the shader's ten-bit unpack.
const MAJORANT_RES_MAX: u32 = 1023;

/// Rows-of-affine (the [`GridVolume`] form) back into a `Mat4`.
fn rows_to_mat4(rows: &[[f32; 4]; 3]) -> Mat4 {
    Mat4::from_cols(
        glam::Vec4::new(rows[0][0], rows[1][0], rows[2][0], 0.0),
        glam::Vec4::new(rows[0][1], rows[1][1], rows[2][1], 0.0),
        glam::Vec4::new(rows[0][2], rows[1][2], rows[2][2], 0.0),
        glam::Vec4::new(rows[0][3], rows[1][3], rows[2][3], 1.0),
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
    global: Option<&Medium>,
) -> (Vec<u32>, u32, Vec<MediumRecord>) {
    let mut records: Vec<MediumRecord> = Vec::new();
    let mut interned: HashMap<MediumKey, u32> = HashMap::new();
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
    /// [`crate::blackbody::table`], read only where a medium emits.
    blackbody: vk::DeviceAddress,
    /// Padding, so `camera_media` keeps the alignment below.
    _pad0: [u32; 2],
    /// The camera's medium set — the bounce-0 seed. Uploaded
    /// all-[`STACK_EMPTY`] and overwritten in place by `resolve_camera`
    /// at every accumulation restart; sits last so the `uint4` lands on
    /// its 16-byte alignment without padding.
    camera_media: [u32; MEDIUM_SLOTS],
}

// The Slang side lays these out from its own rules, so the sizes are
// pinned here: a mirror that drifts reads garbage rather than failing.
const _: () = assert!(size_of::<SceneTable>() == 224);

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
pub struct Scene {
    // Declared before `geometry`: the TLAS dies before the BLASes its
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
    /// Geometry residency, keyed by the reference an instance holds —
    /// prep rebuilds only what an edit dirtied. The procedural
    /// [`Scene::new`] path keys them by object index and never updates.
    geometry: BTreeMap<Geometry, GpuMesh>,
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
    /// The placement-derived media predicates, recomputed on every build
    /// and update — see [`PlacementMedia`].
    media: PlacementMedia,
    /// The `NanoVDB` grid pool: every heterogeneous medium's density grid,
    /// resident for the scene's lifetime. Grow-only and content-keyed, so
    /// edits re-upload nothing; a scene swap drops it.
    grids: crate::vdb::GridPool,
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
    /// Temperature → emitted color, on the same terms: baked from physics
    /// rather than from the scene, so no edit can dirty it.
    blackbody: Buffer,
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
            blackbody: crate::blackbody::upload(gpu)?,
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
    /// which is how the volumetric estimator tests reach it. Crate-only:
    /// every caller is a test, and the description path is the real API.
    ///
    /// # Errors
    ///
    /// As [`Scene::new`].
    pub(crate) fn new_in_medium(
        gpu: &Context,
        objects: &[Object],
        camera: Camera,
        environment: &Environment,
        global: Option<&Medium>,
    ) -> Result<Self> {
        assert!(!objects.is_empty(), "a scene needs at least one object");
        let mut upload = gpu.upload()?;
        let meshes = objects
            .iter()
            .enumerate()
            .map(|(index, object)| {
                let name = Geometry::Mesh(format!("object{index}"));
                upload_mesh(&mut upload, &name, &object.mesh)
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
                medium: object.medium.clone(),
                priority: object.interior_priority,
            })
            .collect();
        let tlas = build_scene_tlas(gpu, &placements)?;
        let media = placement_media(&placements);
        let mut grids = crate::vdb::GridPool::new();
        let instances =
            upload_instance_tables(gpu, &mut grids, &placements, &triangle_lights, &[], global)?;
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
            geometry: meshes
                .into_iter()
                .enumerate()
                .map(|(index, mesh)| (Geometry::Mesh(format!("object{index}")), mesh))
                .collect(),
            textures: BTreeMap::new(),
            descriptors: Vec::new(),
            camera,
            media,
            grids,
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
        self.media.interiors
    }

    /// Whether the scene has media — the things that route a path segment
    /// through the volume stage, and so record it at all. A pure-absorbing
    /// *interior* is not one: it has no event to sample, and its
    /// Beer–Lambert factor is the surface stage's closed form. A
    /// *scattering* interior is: its extinction and scatter are the volume
    /// stage's to sample.
    #[must_use]
    pub fn has_media(&self) -> bool {
        self.has_global_medium() || self.media.volumes || self.media.scattering_interiors
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
        self.media.volumes
    }

    /// Whether any bounded medium is heterogeneous — a density grid fills
    /// it. True, the wavefront dispatches the volume stage's tracking
    /// entry point and binds the grid pool beside the TLAS; false, the
    /// homogeneous entry runs and no dispatch ever reads the pool binding.
    #[must_use]
    pub fn has_heterogeneous(&self) -> bool {
        self.media.heterogeneous
    }

    /// Whether any heterogeneous medium carries a temperature field. True,
    /// the volume stage dispatches the entry point that collects blackbody
    /// emission at the tracker's collisions.
    #[must_use]
    pub fn has_emissive(&self) -> bool {
        self.media.emissive
    }

    /// The grid pool's buffer for descriptor binding — `None` until some
    /// heterogeneous medium made a grid resident. [`Self::has_heterogeneous`]
    /// implies `Some`: the same prep pass that set the flag uploaded the
    /// grid.
    #[must_use]
    pub fn grid_pool(&self) -> Option<&Buffer> {
        self.grids.buffer()
    }

    /// Whether any interior is authored with a nesting priority the
    /// kernels can act on — see [`PlacementMedia`]. True, a hit can be an
    /// interface a higher-priority interior cuts away, which is resolved
    /// in the volume stage: so this is what makes that stage run in a
    /// scene with no media at all.
    #[must_use]
    pub fn has_priority(&self) -> bool {
        self.media.priority
    }

    /// Whether any material carries an active subsurface lobe (see
    /// [`subsurface`]). True, the wavefront allocates the walk queue and
    /// records the subsurface walk stage after every surface pass.
    #[must_use]
    pub fn has_subsurface(&self) -> bool {
        self.media.subsurface
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
            opaque: placement_boundary(placement) != BOUNDARY_NULL
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
/// its interior's and subsurface media), the material array, the
/// deduplicated medium table, and the light records — laid out in the
/// contiguous primitive order `GeometryRecord.light` depends on.
fn upload_instance_tables(
    gpu: &Context,
    pool: &mut crate::vdb::GridPool,
    placements: &[Placement],
    triangle_lights: &[TriangleLight],
    delta_lights: &[DeltaLight],
    global: Option<&Medium>,
) -> Result<InstanceTables> {
    let light_records = crate::lights::build(triangle_lights, delta_lights);
    assert!(
        placements
            .iter()
            .all(|placement| placement.medium.is_none()
                || is_boundary_inert(&placement.material)),
        "a mesh that bounds a medium must reach the tables inert — see `boundary_material`"
    );
    let materials: Vec<Material> = placements
        .iter()
        .map(|placement| placement.material)
        .collect();
    // Grid payloads first: every field a heterogeneous medium reads —
    // its density, and the temperature it may emit by — becomes pool-
    // resident (the pool dedups, so an edit that keeps a grid re-uploads
    // nothing), and the offsets feed the records below. A temperature
    // field goes through the same call, so it arrives with a majorant
    // lattice the tracker never walks — emission is collected at the
    // density's collisions, never sampled against a bound of its own —
    // but its reduction is the peak `report_emission` reports.
    let mut grids: HashMap<(PathBuf, String), crate::vdb::ResidentGrid> = HashMap::new();
    for placement in placements {
        let Some(volume) = placement.medium.as_ref().and_then(|m| m.volume.as_ref()) else {
            continue;
        };
        for field in [Some(&volume.grid), volume.temperature.as_ref()]
            .into_iter()
            .flatten()
        {
            if let Entry::Vacant(slot) = grids.entry((volume.nvdb.clone(), field.clone())) {
                slot.insert(pool.upload(gpu, &volume.nvdb, field)?);
            }
        }
    }
    // The tracking entry point reads binding 5 unconditionally, and the
    // binding is only written when the pool has a buffer: a scene that
    // dispatches it without one would read an unwritten descriptor.
    assert!(
        grids.is_empty() || pool.buffer().is_some(),
        "a scene with grid media must leave the pool resident"
    );
    // One intern pass over both kinds of interior — a subsurface interior
    // that extinguishes like a transmissive one *is* the same medium.
    let (medium_indices, global_medium, media) = medium_table(
        placements
            .iter()
            .map(|p| placement_medium(p, &grids))
            .chain(placements.iter().map(|p| subsurface(&p.material))),
        global,
    );
    let (interior_indices, subsurface_indices) = medium_indices.split_at(placements.len());
    let records: Vec<GeometryRecord> = placements
        .iter()
        .zip(first_light_indices(placements.len(), triangle_lights))
        .zip(interior_indices.iter().zip(subsurface_indices))
        .map(|((placement, light), (&medium, &subsurface))| {
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
                // The flag rides only over a live index: MEDIUM_NONE
                // already carries every bit, and marking a lobe that
                // does not exist would say nothing.
                subsurface: if subsurface != MEDIUM_NONE
                    && subsurface_textured(&placement.material)
                {
                    subsurface | SUBSURFACE_TEXTURED
                } else {
                    subsurface
                },
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
        blackbody: resident.blackbody.device_address(),
        _pad0: [0; 2],
        camera_media: [STACK_EMPTY; MEDIUM_SLOTS],
    };
    // TRANSFER_SRC so tests can read back the one field a kernel writes
    // (`camera_media`).
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
        | vk::BufferUsageFlags::TRANSFER_SRC;
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

/// Queue one mesh's buffers and its BLAS build on `upload`. `geometry` is
/// the reference it was resolved from — the `scene.` prefix and the kind
/// that put the bytes in the right memory bucket ([`crate::gpu`]'s ledger)
/// are added here, once, so no caller has to remember them and no caller
/// can add them twice.
///
/// The returned [`GpuMesh`] is resident only once the caller's
/// [`Upload::finish`] returns; every caller finishes before the TLAS build
/// that first reads these BLASes.
fn upload_mesh(upload: &mut Upload, geometry: &Geometry, mesh: &Mesh) -> Result<GpuMesh> {
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
    let label = geometry.label();
    // BUILD_INPUT for the BLAS build; STORAGE + device address so the
    // shading kernel can fetch triangle corners afterwards.
    let usage = vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
        | vk::BufferUsageFlags::STORAGE_BUFFER;
    let vertices = upload.buffer(
        &format!("scene.{label}.vertices"),
        bytemuck::cast_slice(&mesh.positions),
        usage,
    )?;
    let normals = upload.buffer(
        &format!("scene.{label}.normals"),
        bytemuck::cast_slice(&mesh.normals),
        usage,
    )?;
    let uvs = upload.buffer(
        &format!("scene.{label}.uvs"),
        bytemuck::cast_slice(&mesh.uvs),
        usage,
    )?;
    let indices = upload.buffer(
        &format!("scene.{label}.indices"),
        bytemuck::cast_slice(&mesh.triangles),
        usage,
    )?;
    let blas = upload.blas(
        &format!("scene.{label}.blas"),
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
    /// Everything that does fill something — interiors, authored media,
    /// the global — interns into one table, keyed by value.
    #[test]
    fn the_medium_table_interns_every_medium_by_value() {
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
            volume: None,
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
            Some(&fog),
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
            volume: None,
        };
        let shared = medium_table(std::iter::once(interior(&deep)), Some(&matching));
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
    /// that poisons every hit on the instance.
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
        // Out-of-range channels clamp: a hard zero would take σ to
        // infinity, and anything above 1 would make Beer–Lambert amplify.
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
        // A NaN channel would ride `clamp` into the table; it falls to a
        // transparent channel instead, and the others keep their σ.
        let poisoned = interior(&Material {
            transmission_color: Vec3::new(f32::NAN, 0.5, 0.5),
            transmission_depth: 1.0,
            ..glass
        })
        .expect("a NaN channel does not unmake the interior");
        assert_eq!(
            poisoned.sigma_t.map(f32::to_bits),
            [0.0f32, -0.5f32.ln(), -0.5f32.ln()].map(f32::to_bits)
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
                bytemuck::cast::<_, MediumKey>(fallen),
                bytemuck::cast::<_, MediumKey>(base),
                "scatter {scatter} is no scatter at all"
            );
        }
    }

    /// The van de Hulst inversion: the authored color is a *multiple*-
    /// scatter albedo, so its single-scatter `α` must pin the endpoints —
    /// black scatters nothing, white scatters everything — and land `σ_s`
    /// inside `σ_t`, so the walk can never amplify.
    #[test]
    fn subsurface_inverts_the_multiple_scatter_albedo() {
        let marble = Material {
            subsurface_weight: 1.0,
            subsurface_color: Vec3::new(0.8, 0.5, 0.2),
            subsurface_radius: 2.0,
            subsurface_radius_scale: Vec3::new(1.0, 0.5, 0.25),
            ..Material::matte(Vec3::splat(0.8), 0.0)
        };
        let record = subsurface(&marble).expect("a weighted lobe fills an interior");
        assert_eq!(
            record.sigma_t.map(f32::to_bits),
            [0.5f32, 1.0, 2.0].map(f32::to_bits),
            "σ_t is the reciprocal mean free path, 1 / (radius · scale)"
        );
        let alpha = |color: f32, g: f32| {
            let record = subsurface(&Material {
                subsurface_color: Vec3::splat(color),
                subsurface_scatter_anisotropy: g,
                ..marble
            })
            .expect("an interior");
            record.sigma_s[0] / record.sigma_t[0]
        };
        assert!(
            alpha(0.0, 0.0).to_bits() == 0 && alpha(1.0, 0.0) > 0.999,
            "black is snapped to exactly no scatter, white walks to (nearly) pure"
        );
        let (low, mid, high) = (alpha(0.3, 0.0), alpha(0.6, 0.0), alpha(0.9, 0.0));
        assert!(
            (0.0..=1.0).contains(&low) && low < mid && mid < high && high <= 1.0,
            "α grows with the authored color and never leaves [0, 1]: \
             σ_a = σ_t − σ_s cannot go negative, no repair shift needed"
        );
        assert!(
            alpha(0.6, 0.7) > alpha(0.6, 0.0) && alpha(0.6, -0.7) < alpha(0.6, 0.0),
            "forward scattering needs a higher α to walk to the same color"
        );
        // The two colors the walk cap's gate is authored at — the top of
        // the band it claims exact, and the stress regime that proves the
        // counter fires — pinned where the fit that produces them lives.
        // Isotropic, so α = 1 − s²; the fit's C² term cancels against the
        // radicand's, leaving the inverse *linear* in C, so both are
        // solved constants rather than searched ones. See
        // `render::subsurface_cap_never_fires_at_production_albedo`.
        for (color, target) in [(0.630_271_f32, 0.96_f32), (0.849_846, 0.995)] {
            assert!(
                (alpha(color, 0.0) - target).abs() < 1e-5,
                "the cap gate's α = {target} constant drifted: {}",
                alpha(color, 0.0)
            );
        }
        // A NaN channel falls to purely absorbing; the others keep theirs.
        let poisoned = subsurface(&Material {
            subsurface_color: Vec3::new(f32::NAN, 0.5, 0.2),
            ..marble
        })
        .expect("a NaN channel does not unmake the interior");
        assert_eq!(
            (poisoned.sigma_s[0].to_bits(), poisoned.sigma_s[1] > 0.0),
            (0.0f32.to_bits(), true),
            "a NaN color channel scatters nothing"
        );
        // A zero scale channel floors the mean free path at a micron —
        // enormous extinction, never infinite.
        let dense = subsurface(&Material {
            subsurface_radius_scale: Vec3::new(0.0, 0.5, 0.25),
            ..marble
        })
        .expect("an interior");
        assert!(
            dense.sigma_t[0].to_bits() == 1e6f32.to_bits()
                && dense.sigma_t.iter().all(|sigma| sigma.is_finite()),
            "σ_t stays finite where a channel's radius reaches zero"
        );
    }

    /// The inverse is a real inverse, tested in the direction that carries
    /// the work: an importer holding `σ_s`/`σ_t` authors a color, and prep must
    /// walk that color back to the α it started from.
    ///
    /// The other direction is deliberately *not* asserted tightly, and the
    /// reason is worth keeping. The fit is stationary at white — dα/dC → 0
    /// as C → 1 — so α saturates against 32-bit resolution there and a
    /// color of 0.999 comes back 2e-4 away. That is the map's conditioning,
    /// not a defect, and it costs nothing: every C in that tail describes
    /// the same barely-absorbing interior. Read the other way the flatness
    /// works *for* us, contracting the error instead of amplifying it.
    #[test]
    fn the_albedo_inversion_round_trips() {
        for g in [-0.99f32, -0.5, 0.0, 0.5, 0.99] {
            let mut worst: (f32, f32) = (0.0, 0.0);
            for step in 0..=1000u32 {
                let alpha = f32::from(u16::try_from(step).expect("in range")) / 1000.0;
                let back = single_scatter_albedo(multiple_scatter_color(alpha, g), g);
                if (back - alpha).abs() > worst.0 {
                    worst = ((back - alpha).abs(), alpha);
                }
            }
            // What sets the floor at the extremes is the *forward* path's
            // own 32-bit arithmetic, not the inverse: at |g| near the
            // table's bound, α = (1 − s²)/(1 − g·s²) divides two
            // cancelling differences, and prep evaluates it in f32 like
            // everything else. The inverse is the accurate side here.
            let bound = if g.abs() > 0.9 { 2e-4 } else { 1e-5 };
            assert!(
                worst.0 < bound,
                "at g = {g}, α = {} round-tripped {} away — over the {bound} bound",
                worst.1,
                worst.0
            );
        }
        // And the colors themselves come back, over the range where the fit
        // is not yet flat — which covers every material anyone authors.
        for g in [-0.9f32, 0.0, 0.9] {
            for step in 0..=95u32 {
                let color = f32::from(u16::try_from(step).expect("in range")) / 100.0;
                let back = multiple_scatter_color(single_scatter_albedo(color, g), g);
                assert!(
                    (back - color).abs() < 5e-5,
                    "C = {color} at g = {g} came back as {back}"
                );
            }
        }
        // Black is exact — an interior that scatters nothing must author a
        // color that walks back to *purely* absorbing, not almost, which is
        // the same snap the forward direction makes. White is not, and
        // cannot be: the fit itself lands 1.6e-6 short of α = 1 at C = 1,
        // so the honest inverse of a non-absorbing interior is the color
        // just below white rather than a snap that would not reach it
        // anyway. What matters is that it stays inside the schema's range.
        let (black, white) = (
            multiple_scatter_color(0.0, 0.0),
            multiple_scatter_color(1.0, 0.0),
        );
        assert_eq!(black.to_bits(), 0, "α = 0 authors exactly black");
        assert!(
            (0.9999..=1.0).contains(&white),
            "α = 1 authors white to the fit's residual, and never past it: {white}"
        );
        // Out-of-domain input is clamped, never propagated: an importer
        // handing over a σ_s that outran its σ_t, or a NaN channel, must
        // still author a color the schema accepts.
        assert_eq!(
            [
                multiple_scatter_color(-1.0, 0.0),
                multiple_scatter_color(2.0, 0.0),
                multiple_scatter_color(f32::NAN, 0.0),
            ]
            .map(f32::to_bits),
            [black, white, black].map(f32::to_bits),
            "α outside [0, 1] clamps to the endpoints, a NaN to black"
        );
    }

    /// The lobe's gates, written positively so a NaN falls to `None`: no
    /// weight, no travel, or no interior at all means no medium record.
    #[test]
    fn a_gated_subsurface_is_no_interior() {
        let marble = Material {
            subsurface_weight: 1.0,
            ..Material::matte(Vec3::splat(0.8), 0.0)
        };
        for (name, material) in [
            ("zero weight", Material::matte(Vec3::splat(0.8), 0.0)),
            (
                "negative weight",
                Material {
                    subsurface_weight: -1.0,
                    ..marble
                },
            ),
            (
                "NaN weight",
                Material {
                    subsurface_weight: f32::NAN,
                    ..marble
                },
            ),
            (
                "zero radius",
                Material {
                    subsurface_radius: 0.0,
                    ..marble
                },
            ),
            (
                "negative radius",
                Material {
                    subsurface_radius: -1.0,
                    ..marble
                },
            ),
            (
                "NaN radius",
                Material {
                    subsurface_radius: f32::NAN,
                    ..marble
                },
            ),
            (
                "thin-walled",
                Material {
                    thin_walled: 1,
                    ..marble
                },
            ),
        ] {
            assert!(subsurface(&material).is_none(), "{name} is not an interior");
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

    /// The GPU's copy of the inversion against the host's, on the hardware
    /// it ships on — no render can compare them, since a material is
    /// textured or it is not and only one route ever runs. The domain is a
    /// *texel*'s rather than a scene file's: negative, above-one and NaN
    /// channels all reach the fit, and each has a defined answer here.
    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "the anisotropy is a clamp on both sides, so it must land exactly"
    )]
    #[expect(
        clippy::too_many_lines,
        reason = "the cases, the dispatch and the comparison read as one argument — \
                  the table of inputs is only meaningful beside the bound it feeds"
    )]
    fn the_gpu_inverts_the_subsurface_albedo_as_the_host_does() {
        /// Mirrors `struct Params` in `shaders/subsurface_test.slang`.
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct Params {
            materials: vk::DeviceAddress,
            extinction: vk::DeviceAddress,
            scattering: vk::DeviceAddress,
            count: u32,
            _pad: [u32; 3],
        }

        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let sample = |color: Vec3, radius: f32, scale: Vec3, g: f32| Material {
            subsurface_weight: 1.0,
            subsurface_color: color,
            subsurface_radius: radius,
            subsurface_radius_scale: scale,
            subsurface_scatter_anisotropy: g,
            ..Material::matte(Vec3::ZERO, 0.0)
        };
        let skin = Vec3::new(0.6251, 0.3835, 0.2394);
        let materials = vec![
            // The three authored tiers and the head, at their own shapes.
            sample(Vec3::splat(0.596), 0.23, Vec3::ONE, 0.0),
            sample(Vec3::splat(0.722), 0.069, Vec3::ONE, 0.0),
            sample(Vec3::splat(0.850), 0.023, Vec3::ONE, 0.0),
            sample(skin, 0.001_295_3, Vec3::new(1.0, 0.735_258_3, 0.518_134_8), 0.0),
            // Anisotropy, including past the bound and NaN.
            sample(skin, 0.01, Vec3::ONE, 0.6),
            sample(skin, 0.01, Vec3::ONE, -0.9995),
            sample(skin, 0.01, Vec3::ONE, f32::NAN),
            // The fit's endpoints, and the domain only a texel reaches.
            sample(Vec3::new(0.0, 1.0, 0.5), 0.01, Vec3::ONE, 0.0),
            sample(Vec3::new(-0.25, 1.75, f32::NAN), 0.01, Vec3::ONE, 0.2),
            // The micron floor, and a zero channel of the shape hitting it
            // too. A radius of *zero* is deliberately absent: it is the one
            // value both gates refuse, so the walk never sees it — a texel
            // that reads zero turns the lobe off at the entry vertex, where
            // the closure resolves the same map.
            sample(skin, 1e-9, Vec3::ONE, 0.0),
            sample(skin, 0.01, Vec3::new(1.0, 0.0, -1.0), 0.0),
        ];

        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        let count = materials.len();
        let table = upload_table(&gpu, "test.subsurface.materials", &materials)
            .expect("material table");
        let extinction = gpu
            .create_buffer(
                "test.subsurface.extinction",
                (count * 16) as u64,
                usage,
                gpu_allocator::MemoryLocation::GpuOnly,
            )
            .expect("extinction buffer");
        let scattering = gpu
            .create_buffer(
                "test.subsurface.scattering",
                (count * 16) as u64,
                usage,
                gpu_allocator::MemoryLocation::GpuOnly,
            )
            .expect("scattering buffer");

        let spirv =
            crate::shaders::compile_fixture("subsurface_test").expect("compile subsurface_test");
        let pipeline = gpu
            .create_compute_pipeline(
                &spirv,
                c"subsurface_test",
                size_of::<Params>() as u32,
                crate::gpu::Bindings::None,
            )
            .expect("pipeline");
        let params = Params {
            materials: table.device_address(),
            extinction: extinction.device_address(),
            scattering: scattering.device_address(),
            count: count as u32,
            _pad: [0; 3],
        };
        gpu.dispatch(&pipeline, None, bytemuck::bytes_of(&params), [1, 1, 1])
            .expect("dispatch");
        let extinction: Vec<f32> =
            bytemuck::pod_collect_to_vec(&gpu.download_buffer(&extinction).expect("download"));
        let scattering: Vec<f32> =
            bytemuck::pod_collect_to_vec(&gpu.download_buffer(&scattering).expect("download"));

        // A relative bound rather than equality: the GPU may fuse a
        // multiply and an add where the host does not, which moves the last
        // place or two. Anything larger is a transcription error.
        let mut worst = 0.0f32;
        for (index, material) in materials.iter().enumerate() {
            let host = subsurface(material).expect("every case here has a live lobe");
            let gpu_sigma_t = &extinction[index * 4..index * 4 + 3];
            let gpu_sigma_s = &scattering[index * 4..index * 4 + 3];
            assert_eq!(
                extinction[index * 4 + 3],
                host.g,
                "case {index}: anisotropy is a clamp, not a computation, and must land exactly"
            );
            for (channel, (&theirs, &ours)) in gpu_sigma_t
                .iter()
                .zip(host.sigma_t.iter())
                .chain(gpu_sigma_s.iter().zip(host.sigma_s.iter()))
                .enumerate()
            {
                let scale = ours.abs().max(theirs.abs());
                let relative = if scale > 0.0 {
                    (theirs - ours).abs() / scale
                } else {
                    0.0
                };
                assert!(
                    relative < 1e-5,
                    "case {index} channel {channel}: GPU {theirs} vs host {ours}"
                );
                worst = worst.max(relative);
            }
        }
        // Loud on purpose: a run that notices drift should say how much.
        println!("worst relative disagreement {worst:e}");
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
