// The GPU scene, reached through one `SceneTable` of buffer addresses that
// mirrors `struct SceneTable` in `scene.rs` (not the resource-owning
// `struct Scene` beside it): kernels carry a single scene pointer in their
// push constants, and new scene-wide tables land here without growing any
// kernel's parameter block.

import environment;
import lights;
import openpbr;
import textures;

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
    uint pad0;
    uint pad1;
    uint pad2;
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
// by TLAS instance custom index, except the light list), the closure's
// baked lookup tables, plus the embedded environment light.
public struct SceneTable {
    public GeometryRecord* geometry;
    public Material* materials;
    public LightRecord* lights;
    public float* bsdfTables; // layout in `tables.rs` ↔ openpbr.slang TABLE_*
    public TextureParams* textureParams; // per bindless slot, same index order
    public Environment env;
    public uint lightCount;
    uint pad0;
    uint pad1;
    uint pad2;
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

// Beer–Lambert transmittance over one path segment through an instance's
// interior: absorption toward the medium's target color, scaled by its
// transmission depth. A zero depth means no interior — the tint sat on the
// interface instead — and transmits fully. Callers pass a real interior
// (`medium != MEDIUM_NONE`); shared by `shade_surface`'s per-bounce loop and
// the queued shadow rays.
public float3 interiorTransmittance(SceneTable* scene, uint medium,
                                    float3 from, float3 to) {
    Material interior = scene.materials[medium];
    if (interior.transmissionDepth > 0.0) {
        let sigma = -log(clamp(interior.transmissionColor, 1e-4, 1.0))
                  / interior.transmissionDepth;
        return exp(-sigma * distance(from, to));
    }
    return float3(1.0);
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
