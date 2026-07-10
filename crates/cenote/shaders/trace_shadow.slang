// The shadow-ray stage: trace each queued next-event connection and, when
// the light is reached, add its already-weighted contribution to the
// pixel. Runs once per bounce round, so a pixel sees at most one shadow
// ray per dispatch — the plain += below stays race-free, and the barriers
// between rounds order the adds deterministically.

import pathstate;

// The scene TLAS
[vk::binding(0, 0)]
RaytracingAccelerationStructure scene;

struct Params {
    Queue<ShadowRay> shadows;
    float4* radiance; // the film's per-pixel accumulator for this wave
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
    // thing along the ray *is* that (instance, primitive) — or nothing,
    // when the ray numerically grazes past its silhouette, aims at the
    // environment, or aims at a delta light within `tmax`. The identity
    // test needs no distance epsilon (see `ShadowRay` in pathstate.slang).
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(scene, RAY_FLAG_NONE, 0xFF, ray);
    while (query.Proceed()) {
        // Nothing to consider: every BLAS is opaque, so traversal commits
        // triangle hits on its own.
    }

    if (query.CommittedStatus() == COMMITTED_NOTHING
        || (query.CommittedInstanceID() == shadow.light
            && query.CommittedPrimitiveIndex() == shadow.primitive)) {
        params.radiance[shadow.pixel] += float4(shadow.contribution, 0.0);
    }
}
