// Pure traversal: consume the ray queue, trace inline against the TLAS,
// record where each ray stopped, and route the path to shade_surface (hit
// queue) or shade_miss (miss queue). The one shading concern here is
// stochastic opacity: instances with fractional coverage carry the
// force-non-opaque TLAS flag, and each candidate crossing passes through
// with probability 1 − opacity — resolving transparency during traversal,
// so a pass-through never costs a path bounce. No other shading here —
// traversal is the layer that later learns procedural primitives, and the
// queue boundaries on both sides are where GRIS candidate/visibility
// passes insert.

import pathstate;
import rng;
import scene;

// The scene TLAS
[vk::binding(0, 0)]
RaytracingAccelerationStructure sceneTlas;

// Crossings after this many commit as opaque: a bound on pathological
// transparent stacks (and on the traversal loop), separate from the path
// depth cap.
static const uint TRANSPARENT_CAP = 8;

struct Params {
    Paths paths;
    Queue<uint> rays;
    Queue<uint> hits;
    Queue<uint> misses;
    SceneTable* scene; // materials — the opacity each crossing tests
    uint rayMask; // which instances these rays see: camera rays carry the
                  // camera bit, so camera-invisible instances skip them;
                  // every later bounce traces with all bits set
    uint sampleIndex; // which sample of every pixel's sequence this wave is
    uint bounce;      // which bounce these rays leave from — the
                      // transparency stream must differ per bounce, or one
                      // draw would correlate a path's crossings and bias
                      // their product
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
    uint pixel = params.paths.pixel[path];

    RayDesc ray;
    ray.Origin = params.paths.origin[path].xyz;
    ray.Direction = params.paths.direction[path].xyz;
    ray.TMin = 0.0;
    ray.TMax = 1e30;

    // Opaque BLASes commit on their own; only fractional-opacity
    // instances surface as candidates, each an independent Bernoulli
    // trial. Unordered traversal is fine: the nearest crossing that
    // commits wins, exactly the distribution a front-to-back walk gives.
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(sceneTlas, RAY_FLAG_NONE, params.rayMask, ray);
    uint crossings = 0;
    while (query.Proceed()) {
        let opacity = params.scene.materials[query.CandidateInstanceID()].opacity;
        if (crossings >= TRANSPARENT_CAP
            || hashUniform(pixel, params.sampleIndex, params.bounce, crossings) < opacity) {
            query.CommitNonOpaqueTriangleHit();
        }
        crossings++;
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
