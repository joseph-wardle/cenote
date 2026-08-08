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

// Flags in `Params.packed`. MEDIA and PRIORITY both clear, the routing
// below is one uniform never-taken branch reading neither the medium set
// nor the boundary word. SET says the path pool carries a medium set — a
// scene whose only medium is the global one has none, there being nothing
// to be inside of. GLOBAL says the open space between instances is filled.
// PRIORITY says some interior is authored with a nesting priority, so a
// hit can be an interface the volume stage has to cross rather than shade.
static const uint FLAG_MEDIA = 1 << 16;
static const uint FLAG_SET = 1 << 17;
static const uint FLAG_GLOBAL = 1 << 18;
static const uint FLAG_PRIORITY = 1 << 19;

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
// interior displaces whatever surrounds it, so a path inside one owes
// exactly what that interior's medium does: an event where it scatters,
// nothing where it only absorbs (the Beer–Lambert factor stays
// shade_surface's closed form). The bit is register-held state the caller
// already has — no load answers this. Outside any interior, any volume
// the path is inside owes an event, even a purely absorbing one (nothing
// else can apply its extinction), and so does a filled open space.
// Callers guard on FLAG_MEDIA and hand in the dominant interior they
// already need for the routing test below.
bool travelledMedium(uint4 set, uint interior, uint packed) {
    if (interior != STACK_EMPTY) {
        return (interior & STACK_SCATTERING) != 0;
    }
    return any(set != STACK_EMPTY) || (packed & FLAG_GLOBAL) != 0;
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

    // Bounce 0 seeds the medium set for every stage that will read it —
    // raygen's push constants are full to Vulkan's guarantee and cannot
    // reach the pool, so this is the only place it can happen. The seed is
    // the camera's own set, from the scene table. A scene with interiors
    // but nothing to sample seeds and never reads, which is why this is
    // not under `media`.
    let live = (params.packed & FLAG_SET) != 0;
    if (live && bounce == 0) {
        params.paths.setMedia(path, params.scene.cameraMedia);
    }
    // What the set has to say about this segment: whether the volume stage
    // owes it an event, and which interior — if any — the path is inside.
    // Asked once, and only where an answer can be anything but no: reading
    // the set under `live` alone — a scene can carry a set with nothing to
    // ask of it — measured +4…6% of this kernel on glass-of-water and
    // spaceship. Whatever else comes to ask the set a question widens this
    // guard rather than removing it.
    let media = (params.packed & FLAG_MEDIA) != 0;
    var owed = false;
    var interior = STACK_EMPTY;
    if ((params.packed & (FLAG_MEDIA | FLAG_PRIORITY)) != 0) {
        let set = !live ? uint4(STACK_EMPTY)
                  : bounce != 0 ? params.paths.media(path, true)
                                : params.scene.cameraMedia;
        interior = interiorOf(set);
        owed = media && travelledMedium(set, interior, params.packed);
    }
    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        Hit hit;
        hit.instance = query.CommittedInstanceID();
        hit.primitive = query.CommittedPrimitiveIndex();
        hit.barycentrics = query.CommittedTriangleBarycentrics();
        params.paths.hit[path] = hit;
        // A hit reached through a medium, or one the path crosses rather
        // than shades — a volume's boundary, or an interface the dominant
        // interior cuts away — owes the volume stage a visit before, or
        // instead of, shading. The `interior` clause keeps a priority-only
        // scene from loading a geometry record for every hit: no interior,
        // nothing to be lower-priority than.
        let event = (media || interior != STACK_EMPTY)
                    && (owed
                        || crossable(params.scene.geometry[hit.instance], interior));
        if (event) {
            params.paths.setSegment(path, query.CommittedRayT());
            params.volumes.push(path);
        } else {
            params.hits.push(path);
        }
    } else if (owed) {
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
