// The GPU scene, reached through one `SceneTable` of buffer addresses that
// mirrors `struct SceneTable` in `scene.rs` (not the resource-owning
// `struct Scene` beside it): kernels carry a single scene pointer in their
// push constants, and new scene-wide tables land here without growing any
// kernel's parameter block.

import environment;
import lights;
import openpbr;
import textures;

// What crossing an instance's surface means, classified at prep: an
// ordinary surface, a refractive interface into a closed interior, or a
// null boundary — pure medium extent, no surface response at all. Read by
// intersect's route.
public static const uint BOUNDARY_OPAQUE = 0;
public static const uint BOUNDARY_REFRACTIVE = 1;
public static const uint BOUNDARY_NULL = 2;

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
    // against. Derived at prep.
    public float3 sigmaT;
    // Henyey–Greenstein anisotropy, in (−1, 1); 0 is isotropic.
    public float g;
    // Scattering σ_s, the part of the extinction that redirects rather
    // than removes. Zero is a purely absorbing medium, which has no event
    // to sample and so never routes a path to the volume stage.
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
    public uint boundary;           // BOUNDARY_* — what crossing this means
    public uint medium;             // this interior's record, or MEDIUM_NONE
    uint pad0;
};

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
    // vacuum. Unbounded, so a segment that reaches no surface has infinite
    // optical depth through it: with a global medium the environment and
    // the distant lights are unreachable by construction.
    public uint globalMedium;
};

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

// Trace one next-event connection and return its visibility in [0, 1]: 0 when a
// solid occluder blocks it, otherwise the product of (1 − opacity) over every
// fractional-opacity crossing — Cycles' deterministic transparency split, so
// this stays RNG-free. `light`/`primitive` name the connection's aimed target
// (or `LIGHT_NONE`, which no instance carries, for the environment and delta
// lights): the connection is visible iff the nearest solid thing along the ray
// is that target or nothing. The identity test needs no distance epsilon — the
// ray's `tmax` is bounded to the target (see `ShadowRay` in pathstate.slang).
// Used by `trace_shadow`, for the queued next-event rays.
public float shadowVisibility(RaytracingAccelerationStructure tlas, SceneTable* scene,
                              RayDesc ray, uint light, uint primitive) {
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(tlas, RAY_FLAG_NONE, 0xFF, ray);
    var attenuation = 1.0;
    while (query.Proceed()) {
        if (query.CandidateInstanceID() == light
            && query.CandidatePrimitiveIndex() == primitive) {
            query.CommitNonOpaqueTriangleHit();
        } else {
            attenuation *= 1.0 - opacityAt(scene, query.CandidateInstanceID(),
                                           query.CandidatePrimitiveIndex(),
                                           query.CandidateTriangleBarycentrics());
            if (attenuation < 1e-4) {
                return 0.0; // solidly occluded, to any precision that matters
            }
        }
    }
    if (query.CommittedStatus() == COMMITTED_NOTHING
        || (query.CommittedInstanceID() == light
            && query.CommittedPrimitiveIndex() == primitive)) {
        return attenuation;
    }
    return 0.0;
}
