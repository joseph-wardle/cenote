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

// Bits 16+ of `Params.packed`: whether the scene has media at all. Set,
// the route below reads real path and boundary state; clear (every scene
// today without a scattering medium or null boundary), the route is one
// uniform never-taken branch and this kernel touches nothing new.
static const uint FLAG_MEDIA = 1 << 16;

struct Params {
    Paths paths;
    Queue<uint> rays;
    Queue<uint> hits;
    Queue<uint> misses;
    Queue<uint> volumes; // hits owed a medium event first — shade_volume's input
    SceneTable* scene; // materials — the opacity each crossing tests
    uint sampleIndex; // which sample of every pixel's sequence this wave is
    // rayMask | bounce << 8 | media flags at bit 16 — see the unpack below.
    // Packed because this block sits exactly at Vulkan's guaranteed 128
    // push-constant bytes: the volume queue's 16 bytes come out of these
    // small scalars.
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
        // The three-way route: a segment through a scattering medium, or
        // one stopped by a null boundary, has a medium event to sample
        // before (or instead of) any surface — shade_volume's job. Every
        // other hit shades as ever. Media-free scenes take the uniform
        // early-out and read neither the stack nor the boundary word.
        var volume = false;
        if ((params.packed & FLAG_MEDIA) != 0) {
            volume = params.scene.geometry[hit.instance].boundary == BOUNDARY_NULL;
            if (!volume && bounce != 0) {
                volume = (params.paths.stack[path].x & STACK_SCATTERING) != 0;
            }
        }
        if (volume) {
            params.volumes.push(path);
        } else {
            params.hits.push(path);
        }
    } else {
        params.misses.push(path);
    }
}
