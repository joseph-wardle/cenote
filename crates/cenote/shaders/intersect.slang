// Pure traversal: consume the ray queue, trace inline against the TLAS,
// record where each ray stopped, and route the path to shade_surface (hit
// queue) or shade_miss (miss queue). The one shading concern here is
// stochastic opacity: instances with fractional coverage carry the
// force-non-opaque TLAS flag, and each candidate crossing passes through
// with probability 1 − opacity — resolving transparency during traversal,
// so a pass-through never costs a path bounce. No other shading here —
// traversal is the layer that later learns procedural primitives.

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

// Bit 16 of `Params.packed`: whether the scene has media. Clear, the
// route below is one uniform never-taken branch and this kernel reads
// neither the medium stack (which was never allocated) nor the boundary
// word.
static const uint FLAG_MEDIA = 1 << 16;

struct Params {
    Paths paths;
    Queue<uint> rays;
    Queue<uint> hits;
    Queue<uint> misses;
    Queue<uint> volumes; // hits owed a medium event first — shade_volume's input
    SceneTable* scene; // materials — the opacity each crossing tests
    uint sampleIndex; // which sample of every pixel's sequence this wave is
    // rayMask | bounce << 8 | FLAG_MEDIA. Packed because this block sits
    // exactly at Vulkan's guaranteed 128 push-constant bytes: the volume
    // queue's 16 bytes come out of these small scalars.
    uint packed;
};

[vk::push_constant]
Params params;

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void intersect(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.rays.count()) {
        return;
    }
    // rayMask: which instances these rays see — camera rays carry the
    // camera bit, so camera-invisible instances skip them; every later
    // bounce traces with all bits set. bounce: which bounce these rays
    // leave from — the transparency stream must differ per bounce, or one
    // draw would correlate a path's crossings and bias their product.
    let rayMask = params.packed & 0xff;
    let bounce = (params.packed >> 8) & 0xff;
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
    query.TraceRayInline(sceneTlas, RAY_FLAG_NONE, rayMask, ray);
    uint crossings = 0;
    while (query.Proceed()) {
        let opacity = opacityAt(params.scene, query.CandidateInstanceID(),
                                query.CandidatePrimitiveIndex(),
                                query.CandidateTriangleBarycentrics());
        if (crossings >= TRANSPARENT_CAP
            || hashUniform(pixel, params.sampleIndex, bounce, crossings) < opacity) {
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
        // A hit that stopped at a null boundary (pure medium extent, no
        // surface response), or one whose path is inside a scattering
        // medium, owes a medium event before — or instead of — shading:
        // shade_volume's job. Every other hit shades as ever. Bounce 0
        // starts in vacuum, so its stack entry is unwritten.
        var event = false;
        if ((params.packed & FLAG_MEDIA) != 0) {
            event = params.scene.geometry[hit.instance].boundary == BOUNDARY_NULL
                    || (bounce != 0 && (params.paths.top(path) & STACK_SCATTERING) != 0);
        }
        if (event) {
            params.volumes.push(path);
        } else {
            params.hits.push(path);
        }
    } else {
        params.misses.push(path);
    }
}
