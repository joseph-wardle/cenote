// Pure traversal: consume the ray queue, trace inline against the TLAS,
// record where each ray stopped, and route the path to shade_surface (hit
// queue) or shade_miss (miss queue). No shading here — traversal is the
// layer that later learns procedural primitives, and the queue boundaries
// on both sides are where GRIS candidate/visibility passes insert.

import pathstate;

// The scene TLAS
[vk::binding(0, 0)]
RaytracingAccelerationStructure scene;

struct Params {
    Paths paths;
    Queue<uint> rays;
    Queue<uint> hits;
    Queue<uint> misses;
    uint rayMask; // which instances these rays see: camera rays carry the
                  // camera bit, so camera-invisible instances skip them;
                  // every later bounce traces with all bits set
    uint pad;
};

[vk::push_constant]
Params params;

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void intersect(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.rays.count()) {
        return;
    }
    uint path = params.rays.entries[tid.x];

    RayDesc ray;
    ray.Origin = params.paths.origin[path].xyz;
    ray.Direction = params.paths.direction[path].xyz;
    ray.TMin = 0.0;
    ray.TMax = 1e30;

    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(scene, RAY_FLAG_NONE, params.rayMask, ray);
    while (query.Proceed()) {
        // Nothing to consider: every BLAS is opaque, so traversal commits
        // triangle hits on its own.
    }

    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        Hit hit;
        hit.instance = query.CommittedInstanceID();
        hit.primitive = query.CommittedPrimitiveIndex();
        hit.barycentrics = query.CommittedTriangleBarycentrics();
        params.paths.hit[path] = hit;
        params.hits.push(path);
    } else {
        params.misses.push(path);
    }
}
