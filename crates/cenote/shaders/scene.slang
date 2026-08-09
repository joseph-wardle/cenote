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

// What fills an instance's interior, or the open space around everything —
// one record per distinct medium, however many places it fills. Mirrors
// `struct MediumRecord` in `scene.rs`.
public struct MediumRecord {
    // Extinction σ_t = σ_a + σ_s, per ACEScg channel, in inverse meters:
    // the σ of Beer–Lambert, and the density distance sampling draws
    // against. Derived at prep — storing σ_t rather than σ_a is what keeps
    // a purely absorbing interior's value bit-for-bit what it always was.
    public float3 sigmaT;
    // Henyey–Greenstein anisotropy, in (−1, 1); 0 is isotropic.
    public float g;
    // Scattering σ_s, the part of the extinction that redirects rather than
    // removes. Zero is a purely absorbing medium: no event to sample, so
    // intersect never routes a path to the volume stage.
    public float3 sigmaS;
    float pad0;
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
    // unread until the walk kernel (rung 9b) marches it.
    uint subsurface;
};

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
    // The camera's medium set — what the eye itself is inside of, and the
    // seed intersect gives every path at bounce 0. It lives here because
    // raygen's and intersect's push constants are both full to Vulkan's
    // 128-byte guarantee: the one place a new scene-wide word can land.
    // Uploaded all-`STACK_EMPTY` and overwritten in place by
    // `resolve_camera` at the head of every accumulation restart — the one
    // field of this table a kernel writes.
    public uint4 cameraMedia;
};

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
public float3 extinctionOf(SceneTable* scene, uint instance) {
    let medium = scene.geometry[instance].medium;
    return medium == MEDIUM_NONE ? float3(0.0) : scene.media[medium].sigmaT;
}

// Trace one next-event connection and return what reaches the light: 0 when a
// solid occluder blocks it, otherwise the product of (1 − opacity) over every
// fractional-opacity crossing — Cycles' deterministic transparency split, so
// this stays RNG-free — times the transmittance of every volume the
// connection passes through. `light`/`primitive` name the connection's aimed
// target (or `LIGHT_NONE`, which no instance carries, for the environment and
// delta lights): the connection is visible iff the nearest solid thing along
// the ray is that target or nothing. The identity test needs no distance
// epsilon — the ray's `tmax` is bounded to the target (see `ShadowRay` in
// pathstate.slang). Used by `trace_shadow`, for the queued next-event rays.
//
// `VOLUMES` is a specialization, not a flag — the reason is in
// trace_shadow.slang, where the two entry points that instantiate it live.
//
// `inside` is the launching vertex's medium set; the volumes in it are what
// this pass measures, and the interior — whose extent one instance index
// cannot express — is the approximation the TODO in trace_shadow.slang
// names. Volume extents are
// measured, not marched: for one volume the optical depth over [0, tmax] is
// σ_t times the extent inside it, and that extent is
//
//     Σ over crossings of (∓ t)  +  tmax if the far end is still inside,
//
// signed − where the ray enters and + where it leaves. Every term is a plain
// sum, so the candidates may arrive in any order — which is the whole reason
// overlapping volumes were defined to *add*, since an ordered walk is the one
// thing `Proceed()` cannot give. Membership at the far end is the same
// order-free question, answered by parity.
//
// The price of unordered is cancellation: a thickness reaches this sum as
// the difference of two distances measured from the connection's origin, so
// a shell thin against its own range loses that difference to float32, and
// at 1e-7 of it there is nothing left. Marching would subtract adjacent
// crossings instead, which is the one thing this pass cannot do.
public float3 shadowVisibility<let VOLUMES : int>(
    RaytracingAccelerationStructure tlas, SceneTable* scene, RayDesc ray,
    uint light, uint primitive, uint4 inside) {
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(tlas, RAY_FLAG_NONE, 0xFF, ray);
    var attenuation = 1.0;
    var extent = float3(0.0);
    var atEnd = inside;
    while (query.Proceed()) {
        let instance = query.CandidateInstanceID();
        if (instance == light && query.CandidatePrimitiveIndex() == primitive) {
            query.CommitNonOpaqueTriangleHit();
        } else if (VOLUMES != 0 && boundaryOf(scene.geometry[instance]) == BOUNDARY_NULL) {
            // A volume's boundary is not a surface: it never occludes, and
            // committing it — which its full coverage would otherwise do —
            // would read a fog bank as a solid wall.
            let t = query.CandidateTriangleRayT();
            extent += extinctionOf(scene, instance)
                      * (query.CandidateTriangleFrontFace() ? -t : t);
            atEnd = crossMedium(atEnd, instance).set;
        } else {
            attenuation *= 1.0 - opacityAt(scene, instance,
                                           query.CandidatePrimitiveIndex(),
                                           query.CandidateTriangleBarycentrics());
            if (attenuation < 1e-4) {
                return float3(0.0); // solidly occluded, to any precision that matters
            }
        }
    }
    if (query.CommittedStatus() != COMMITTED_NOTHING
        && (query.CommittedInstanceID() != light
            || query.CommittedPrimitiveIndex() != primitive)) {
        return float3(0.0);
    }
    if (VOLUMES == 0) {
        return float3(attenuation);
    }
    for (uint slot = 0; slot < MEDIUM_SLOTS; slot++) {
        // Volumes only: an interior shares the set but not this sum, and
        // the crossings above could only ever have added a null boundary.
        if (atEnd[slot] != STACK_EMPTY && (atEnd[slot] & STACK_INTERIOR) == 0) {
            extent += extinctionOf(scene, atEnd[slot] & STACK_INDEX_MASK) * ray.TMax;
        }
    }
    // A negative extent means a volume was entered and never left, or wound
    // inside-out; clamping is what keeps that a missing shadow rather than a
    // medium that amplifies.
    return attenuation * exp(-max(extent, float3(0.0)));
}
