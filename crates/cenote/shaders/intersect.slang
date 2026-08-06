// Pure traversal: consume the ray queue, trace inline against the TLAS,
// record where each ray stopped, and route the path to shade_surface (hit
// queue), shade_miss (miss queue), or — when a medium fills the segment —
// shade_volume (volume queue). The one shading concern here is
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

// Flags in `Params.packed`. MEDIA clear, the routing below is one uniform
// never-taken branch reading neither the medium stack nor the boundary
// word. INTERIORS says the stack's x lane is live; VOLUMES, that its other
// three are; GLOBAL, that the open space between instances is filled.
static const uint FLAG_MEDIA = 1 << 16;
static const uint FLAG_INTERIORS = 1 << 17;
static const uint FLAG_GLOBAL = 1 << 18;
static const uint FLAG_VOLUMES = 1 << 19;

struct Params {
    Paths paths;
    Queue<uint> rays;
    Queue<uint> hits;
    Queue<uint> misses;
    Queue<uint> volumes; // hits owed a medium event first — shade_volume's input
    SceneTable* scene; // materials — the opacity each crossing tests
    uint sampleIndex; // which sample of every pixel's sequence this wave is
    // rayMask | bounce << 8 | the FLAG_* bits. Packed because this block sits
    // exactly at Vulkan's guaranteed 128 push-constant bytes: the volume
    // queue's 16 bytes come out of these small scalars.
    uint packed;
};

[vk::push_constant]
Params params;

// Does the volume stage owe the arriving segment anything? A refractive
// interior displaces whatever surrounds it and none of them scatters yet,
// so a path inside one owes nothing: its Beer–Lambert factor is
// closed-form, and shade_surface folds it as it always has. Otherwise any
// volume the path is inside owes one — even a purely absorbing volume,
// since nothing else can apply its extinction — and so does a filled open
// space. Callers guard on FLAG_MEDIA, without which neither the stack nor
// the flags mean anything.
bool travelledMedium(Paths paths, uint path, uint bounce, uint packed, uint3 inside) {
    let entry = paths.arriving(path, bounce, (packed & FLAG_INTERIORS) != 0);
    if (entry != STACK_EMPTY) {
        return (entry & STACK_SCATTERING) != 0;
    }
    return any(inside != STACK_EMPTY) || (packed & FLAG_GLOBAL) != 0;
}

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

    let media = (params.packed & FLAG_MEDIA) != 0;
    // The volumes this path is inside. Bounce 0 seeds them rather than
    // reading them: raygen's push constants are full to Vulkan's guarantee
    // and cannot reach the stack, so this is the first stage that can give
    // the lanes a value — and every later one then reads without asking
    // what bounce it is. The seed is the empty set; a camera that starts
    // inside a volume is not yet resolved (see `Instance::medium`).
    var inside = uint3(STACK_EMPTY);
    if ((params.packed & FLAG_VOLUMES) != 0) {
        if (bounce == 0) {
            params.paths.setInside(path, inside);
        } else {
            inside = params.paths.inside(path, true);
        }
    }
    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        Hit hit;
        hit.instance = query.CommittedInstanceID();
        hit.primitive = query.CommittedPrimitiveIndex();
        hit.barycentrics = query.CommittedTriangleBarycentrics();
        params.paths.hit[path] = hit;
        // A hit reached through a medium, or one that stopped at a null
        // boundary (pure medium extent, no surface response), owes a
        // medium event before — or instead of — shading: shade_volume's
        // job. Every other hit shades as ever.
        var event = false;
        if (media) {
            event = params.scene.geometry[hit.instance].boundary == BOUNDARY_NULL
                    || travelledMedium(params.paths, path, bounce, params.packed, inside);
        }
        if (event) {
            params.paths.setSegment(path, query.CommittedRayT());
            params.volumes.push(path);
        } else {
            params.hits.push(path);
        }
    } else if (media && travelledMedium(params.paths, path, bounce, params.packed, inside)) {
        // Nothing stopped the segment, but a medium still fills it. The
        // sentinel is what tells the volume stage the rest of this record
        // holds nothing to read — and that the segment has no far end.
        Hit hit;
        hit.instance = INSTANCE_NONE;
        hit.primitive = 0;
        hit.barycentrics = float2(0.0);
        params.paths.hit[path] = hit;
        params.volumes.push(path);
    } else {
        params.misses.push(path);
    }
}
