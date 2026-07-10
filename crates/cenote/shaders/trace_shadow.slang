// The shadow-ray stage: trace each queued next-event connection and, when
// the light is reached, add its already-weighted contribution to the
// pixel. Fractional-opacity occluders attenuate the connection
// *deterministically* — each crossing multiplies 1 − opacity — so alpha
// cards cast correct soft shadows and this kernel stays RNG-free
// (Cycles' shipped split of the transparency policy). Runs once per
// bounce round, so a pixel sees at most one shadow ray per dispatch — the
// plain += below stays race-free, and the barriers between rounds order
// the adds deterministically.

import pathstate;
import scene;

// The scene TLAS
[vk::binding(0, 0)]
RaytracingAccelerationStructure sceneTlas;

struct Params {
    Queue<ShadowRay> shadows;
    float4* radiance; // the film's per-pixel accumulator for this wave
    SceneTable* scene; // materials — the opacity each crossing attenuates by
};

[vk::push_constant]
Params params;

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void trace_shadow(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.shadows.count()) {
        return;
    }
    ShadowRay shadow = params.shadows.entries[tid.x];

    RayDesc ray;
    ray.Origin = shadow.origin;
    ray.Direction = shadow.direction;
    ray.TMin = 0.0;
    ray.TMax = shadow.tmax;

    // Closest hit, not accept-first: the ray is aimed at a point on one
    // triangle of the light, so the connection is visible iff the nearest
    // solid thing along the ray *is* that (instance, primitive) — or
    // nothing, when the ray numerically grazes past its silhouette, aims
    // at the environment, or aims at a delta light within `tmax`. The
    // identity test needs no distance epsilon (see `ShadowRay` in
    // pathstate.slang). Fractional-opacity candidates that aren't the
    // target pass through, each multiplying its transparency into the
    // connection; the target itself commits, opaque or not.
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(sceneTlas, RAY_FLAG_NONE, 0xFF, ray);
    var attenuation = 1.0;
    while (query.Proceed()) {
        if (query.CandidateInstanceID() == shadow.light
            && query.CandidatePrimitiveIndex() == shadow.primitive) {
            query.CommitNonOpaqueTriangleHit();
        } else {
            attenuation *= 1.0 - opacityAt(params.scene, query.CandidateInstanceID(),
                                           query.CandidatePrimitiveIndex(),
                                           query.CandidateTriangleBarycentrics());
            if (attenuation < 1e-4) {
                return; // solidly occluded, to any precision that matters
            }
        }
    }

    if (query.CommittedStatus() == COMMITTED_NOTHING
        || (query.CommittedInstanceID() == shadow.light
            && query.CommittedPrimitiveIndex() == shadow.primitive)) {
        params.radiance[shadow.pixel] += float4(shadow.contribution * attenuation, 0.0);
    }
}
