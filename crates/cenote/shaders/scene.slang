// The GPU scene, reached through one `SceneTable` of buffer addresses that
// mirrors `struct SceneTable` in `scene.rs` (not the resource-owning
// `struct Scene` beside it): kernels carry a single scene pointer in their
// push constants, and new scene-wide tables land here without growing any
// kernel's parameter block.

import environment;
import lights;
import openpbr;
import pathstate; // STACK_EMPTY, MEDIUM_SLOTS, crossMedium
import textures;

// What crossing an instance's surface means, classified at prep: an
// ordinary surface, a refractive interface into a closed interior, or a
// null boundary — pure medium extent, no surface response at all. Read by
// intersect's route, through `boundaryOf`.
public static const uint BOUNDARY_OPAQUE = 0;
public static const uint BOUNDARY_REFRACTIVE = 1;
public static const uint BOUNDARY_NULL = 2;
public static const uint BOUNDARY_MASK = 0xff;

// A medium index that names vacuum — an index no table can reach.
// Deliberately not pathstate's all-ones `STACK_EMPTY`: media and instances
// are separate index spaces.
public static const uint MEDIUM_NONE = 0xffffffff;

// `MediumRecord.grid` for a homogeneous medium — mirrors `scene/mod.rs`.
public static const uint GRID_NONE = 0xffffffff;

// What fills an instance's interior, or the open space around everything —
// one record per distinct medium, however many places it fills. Mirrors
// `struct MediumRecord` in `scene.rs`.
public struct MediumRecord {
    // Extinction σ_t = σ_a + σ_s, per ACEScg channel, in inverse meters:
    // the σ of Beer–Lambert, and the density distance sampling draws
    // against. Derived at prep — storing σ_t rather than σ_a is what keeps
    // a purely absorbing interior's value bit-for-bit what it always was.
    // For a heterogeneous medium these are the *coefficients* the grid's
    // density multiplies: σ(x) = density(x) · σ.
    public float3 sigmaT;
    // Henyey–Greenstein anisotropy, in (−1, 1); 0 is isotropic.
    public float g;
    // Scattering σ_s, the part of the extinction that redirects rather than
    // removes. Zero is a purely absorbing medium: no event to sample, so
    // intersect never routes a path to the volume stage.
    public float3 sigmaS;
    // The grid's byte offset in the pool (binding 5), or GRID_NONE for a
    // homogeneous medium. Only the volume stage's tracking entry reads past
    // this word.
    public uint grid;
    // Rows of the world→index transform: the grid's own map composed with
    // the placement's inverse at prep, so a sample point is one affine from
    // world space to the trilinear lattice.
    public float4 worldToIndex[3];
    // Index space → majorant-cell coordinates, `cell = index · scale + bias`:
    // the second half of the affine that takes a world point to the lattice
    // the tracker's free flights are bounded by.
    public float3 majorantScale;
    // The lattice's byte offset in the pool — `res.x·res.y·res.z` unitless
    // density ceilings, x fastest.
    public uint majorant;
    public float3 majorantBias;
    // Its cell counts, ten bits per axis with x lowest.
    public uint majorantRes;
    // ACEScg multiplier on the blackbody radiance at that temperature — so
    // the source term is density·σ_a·emission·blackbody(K), and `emission`
    // is what the medium radiates per unit of *absorption* optical depth,
    // in units of a 6500 K body.
    public float3 emission;
    // The temperature grid's byte offset in the pool, or GRID_NONE for a
    // medium that does not emit. Read through `worldToIndex`, the density
    // grid's: prep refuses a pair whose index maps differ. Pool offsets are
    // 32-byte aligned, so the low bits are spare if a field ever needs to
    // be read as radiance outright rather than as a temperature.
    public uint temperatureGrid;
    // Kelvin from a field value: value·kelvinScale + kelvinOffset.
    public float kelvinScale;
    public float kelvinOffset;
    // worldToIndex's float4 rows already round the record to 144 bytes; this
    // only names the tail so scene.rs can spell the same layout.
    float2 pad0;
};

// Where an instance's triangles live, plus its transforms — everything
// needed to re-evaluate shading at a Hit. Mirrors `GeometryRecord` in
// `scene.rs` field for field.
public struct GeometryRecord {
    public float* positions;        // x, y, z per vertex, object space
    public float* normals;          // unit shading normal per vertex, object space
    public float2* uvs;             // texture coordinates per vertex (zeros when unauthored)
    public uint* indices;           // three per triangle
    public float4 objectToWorld[3]; // rows of the instance's 3×4 transform
    public float4 worldToObject[3]; // rows of its inverse — normals, error bounds
    public uint light;              // index into the light list, or LIGHT_NONE
    // What crossing this instance's surface means: the `BOUNDARY_*` class
    // in the low byte (read through `boundaryOf`, never compared raw) and,
    // above it, the medium-set entry's flag bits — `STACK_INTERIOR`, the
    // nesting priority, and `STACK_SCATTERING` — pre-baked by prep so
    // `stackEntry` is one load and one OR. One word and not two because
    // the record's fields reach a kernel as separate scalar loads: a
    // second field measured `shade_surface` +2…8% on the glass drivers.
    uint boundary;
    public uint medium;             // this interior's record, or MEDIUM_NONE
    // The subsurface interior's record, or MEDIUM_NONE — interned by prep,
    // walked by shade_subsurface. A positive `subsurfaceBlend` in the
    // closure implies this is a real index: both mirror the same gate.
    // Above the index rides `SUBSURFACE_TEXTURED`, so mask before use.
    public uint subsurface;
};

// Set in `GeometryRecord::subsurface` where a map, not a constant, decides
// what the interior is made of: the record below it is then a stand-in and
// the walk must invert the resolved material itself (`subsurfaceMedium`,
// medium.slang). Read the index through `SUBSURFACE_INDEX`, never raw.
//
// Sharp edge: MEDIUM_NONE is all ones and so carries the bit too. Harmless
// here — the walk only reaches records whose lobe is live — but a reader
// that tests the bit before the sentinel would call every instance
// textured.
public static const uint SUBSURFACE_TEXTURED = 1u << 31;
public static const uint SUBSURFACE_INDEX = ~SUBSURFACE_TEXTURED;

// The boundary class alone — what `BOUNDARY_*` names.
public uint boundaryOf(GeometryRecord mesh) {
    return mesh.boundary & BOUNDARY_MASK;
}

// The medium-set entry naming an instance the path has just entered: its
// TLAS custom index, and above it the entry's flag bits — lifted verbatim
// from the boundary word, where prep baked them, so every insertion site
// builds identical entries from the load it already made. Crossing *out*
// matches on the index alone (`crossMedium` in pathstate.slang), so
// insertion is the only place the flags can be set — and an entry built
// without them ranks in `interiorOf` as something it is not.
public uint stackEntry(GeometryRecord mesh, uint instance) {
    return instance | (mesh.boundary & (STACK_INTERIOR | STACK_PRIORITY | STACK_SCATTERING));
}

// Is a hit on this instance crossed rather than shaded? A null boundary
// always is — pure medium extent, no surface response. A refractive
// interface is when its priority sits *strictly below* the interior the
// path is already inside: the nested-dielectric rule (Schmidt & Budge
// 2002), cutting the weaker solid away inside the stronger one as though
// the two had been booleaned. `interior` is `interiorOf(set)`, whose
// `STACK_EMPTY` carries no priority bits, so a path inside nothing
// suppresses nothing without a test of its own. Both sides of the compare
// are masked to the priority field alone — the bits around it name things
// priority must not rank.
//
// The crossing's direction is deliberately not asked: entering a false
// boundary and leaving one both still toggle the set, else a path leaving
// the dominant glass into the water it overlaps would find an empty set
// and refract into air.
public bool crossable(GeometryRecord mesh, uint interior) {
    if (boundaryOf(mesh) == BOUNDARY_NULL) {
        return true;
    }
    return boundaryOf(mesh) == BOUNDARY_REFRACTIVE
        && (mesh.boundary & STACK_PRIORITY) < (interior & STACK_PRIORITY);
}

// The UV at barycentrics (b1, b2) of a triangle — weights on the second
// and third corner, the convention ray-query barycentrics and the light
// sampler's triangle parameterization share.
public float2 interpolateUv(GeometryRecord mesh, uint primitive, float2 bary) {
    uint base = 3 * primitive;
    let uv0 = mesh.uvs[mesh.indices[base]];
    let uv1 = mesh.uvs[mesh.indices[base + 1]];
    let uv2 = mesh.uvs[mesh.indices[base + 2]];
    return uv0 + bary.x * (uv1 - uv0) + bary.y * (uv2 - uv0);
}

// Everything the scene shares with the kernels: buffer addresses (indexed
// by TLAS instance custom index, except the light list), plus the embedded
// environment light. The closure's baked tables travel as descriptors
// instead — see `openpbr.slang`. Field order is the contract with
// `struct SceneTable` in `scene.rs`, which only asserts the total size:
// reordering here moves offsets the Rust mirror will not notice.
public struct SceneTable {
    public GeometryRecord* geometry;
    public Material* materials;
    public LightRecord* lights;
    public TextureParams* textureParams; // per bindless slot, same index order
    public Environment env;
    public MediumRecord* media;
    public uint lightCount;
    // What fills the open space between instances, or MEDIUM_NONE for
    // vacuum. Unbounded, so a segment that reaches no surface crosses
    // infinite optical depth in every channel that extinguishes at all: the
    // environment and the distant lights are unreachable through one.
    public uint globalMedium;
    // `blackbody::table` — log2 ACEScg radiance sampled uniformly in
    // temperature, read only where a medium emits. `pad0` keeps the uint4
    // below on its alignment.
    public float4* blackbody;
    uint2 pad0;
    // The camera's medium set — what the eye itself is inside of, and the
    // seed intersect gives every path at bounce 0. It lives here because
    // raygen's and intersect's push constants are both full to Vulkan's
    // 128-byte guarantee: the one place a new scene-wide word can land.
    // Uploaded all-`STACK_EMPTY` and overwritten in place by
    // `resolve_camera` at the head of every accumulation restart — the one
    // field of this table a kernel writes.
    public uint4 cameraMedia;
};

// The blackbody table's grid, mirroring `blackbody.rs`: this many samples
// spread over [0, BLACKBODY_MAX_K], sample k standing at k·MAX/(SIZE − 1).
static const uint BLACKBODY_SIZE = 2048;
static const float BLACKBODY_MAX_K = 20000.0;

// What a blackbody at `kelvin` emits, in ACEScg, scaled so that 6500 K has
// luminance 1. The samples are interpolated where they are *stored* — in
// log2 — because the radiance they stand for climbs by orders of magnitude
// between neighbours, and only its logarithm is smooth enough for a lerp to
// land on. Hotter than the table reads as its top sample.
//
// Nothing here is physics: `blackbody.rs` integrates Planck's law against
// the CIE observer and bakes the result, and this only reads it back, so
// there is no second spelling of the emission spectrum to drift from that
// one.
//
// The bound is written as a comparison rather than `clamp`: SPIR-V's FClamp
// is undefined on a NaN operand, and a NaN Kelvin would then index wherever
// the driver put it.
public float3 blackbody(SceneTable* scene, float kelvin) {
    let bounded = kelvin > 0.0 ? min(kelvin, BLACKBODY_MAX_K) : 0.0;
    let x = bounded * (float(BLACKBODY_SIZE - 1) / BLACKBODY_MAX_K);
    let lo = uint(x);
    let hi = min(lo + 1, BLACKBODY_SIZE - 1);
    return exp2(lerp(scene.blackbody[lo].xyz, scene.blackbody[hi].xyz, x - float(lo)));
}

// Crossings after this many commit as opaque, in whichever traversal is
// resolving them: a bound on pathological transparent stacks, and on the
// loop that walks them. Separate from the path depth cap.
public static const uint TRANSPARENT_CAP = 8;

// The coverage of one traversal crossing: the instance's opacity constant
// times its opacity map at the crossing point — the one value both halves
// of the transparency policy test, stochastically in intersect and
// multiplicatively in trace_shadow.
public float opacityAt(SceneTable* scene, uint instance, uint primitive, float2 bary) {
    let material = scene.materials[instance];
    var opacity = material.opacity;
    if (material.opacityTexture != TEXTURE_NONE) {
        opacity *= sampleScalar(scene.textureParams, material.opacityTexture,
                                interpolateUv(scene.geometry[instance], primitive, bary));
    }
    return opacity;
}

// exp(-σ·d) over one path segment through the interior of `instance` — the
// instance the path is inside, which is what the medium stack names. A
// vacuum interior (MEDIUM_NONE: the material tinted its interface rather
// than its volume) transmits fully.
public float3 interiorTransmittance(SceneTable* scene, uint instance,
                                    float3 from, float3 to) {
    let medium = scene.geometry[instance].medium;
    if (medium == MEDIUM_NONE) {
        return float3(1.0);
    }
    return exp(-scene.media[medium].sigmaT * distance(from, to));
}

// The extinction of whatever fills `instance`, or zero where nothing does.
// A heterogeneous medium reads as zero: its σ varies point by point, which
// no closed-form extent can express, so the shadow pass ratio-tracks it
// separately (`gridTransmittance`) and this sum must not count it twice.
public float3 extinctionOf(SceneTable* scene, uint instance) {
    let medium = scene.geometry[instance].medium;
    if (medium == MEDIUM_NONE) {
        return float3(0.0);
    }
    let record = scene.media[medium];
    return record.grid == GRID_NONE ? record.sigmaT : float3(0.0);
}

