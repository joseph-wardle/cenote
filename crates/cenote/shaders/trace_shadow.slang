// The shadow-ray stage: trace each queued record for occlusion and, when
// the light is visible, add its unshadowed contribution to the pixel.
// Degenerate only in that nothing pushes shadow rays yet — the stage runs
// every wave as an indirect dispatch of zero workgroups until NEE starts
// feeding it (M1 step 8).

import pathstate;

// The scene TLAS
[vk::binding(0, 0)]
RaytracingAccelerationStructure scene;

struct Params {
    Queue<ShadowRay> shadows;
    float4* radiance; // the film's per-pixel sample for this wave
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

    // Occlusion only: any hit kills the ray, so accept the first and stop.
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
    query.TraceRayInline(scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, ray);
    while (query.Proceed()) {}

    if (query.CommittedStatus() == COMMITTED_NOTHING) {
        // Plain read-modify-write, not atomic: a pixel sees at most one
        // shadow ray per wave, and determinism forbids atomic accumulation.
        params.radiance[shadow.pixel] += float4(shadow.contribution, 0.0);
    }
}
