// What is the camera itself inside of? One thread, dispatched once at the
// head of every accumulation restart, answers by the same rule every other
// stage uses: membership by crossing parity. A ray from the camera position
// walks to infinity, and every boundary surface it crosses an odd number of
// times is a medium the camera is in — an inside-out mesh still bounds
// exactly its own interior, and opaque geometry in the way hides nothing,
// because membership is geometric, not visibility. The result lands in
// `SceneTable.cameraMedia`, where intersect's bounce 0 seeds every path
// from it.
//
// A kernel rather than host code because the TLAS already *is* the scene's
// one definition of where surfaces are: a CPU-side point-in-mesh test would
// be a second definition of containment, agreeing with this one only until
// it doesn't.

import pathstate;
import scene;

// The scene TLAS — the walk traces against it.
[vk::binding(0, 0)]
RaytracingAccelerationStructure sceneTlas;

// Boundary crossings the walk resolves before giving up and keeping what it
// has: a bound on pathological nesting, far above anything the four-slot
// set can express. Truncation past it is the same class of accepted error
// as the march's `MARCH_BOUNDARY_CAP`.
static const uint RESOLVE_CROSSING_CAP = 32;

// How far past a crossing the next leg starts — `MARCH_NUDGE`'s value and
// reasoning (shade_volume.slang): relative so it holds at any scene scale,
// wide enough that re-testing the same triangle cannot return the crossing
// just made. Two boundaries closer together than this read as one.
static const float RESOLVE_NUDGE = 1e-6;

// The one direction the walk looks: components distinct and irrational-ish
// ((1, 2, 3)/√14), so a scene full of axis-aligned boxes — most of them —
// never sees the ray running inside a face's own plane. A camera sitting
// within the nudge of a boundary can still land on either side of it; one
// fixed direction, no retry.
static const float3 RESOLVE_DIRECTION = float3(0.26726124, 0.53452248, 0.80178373);

struct Params {
    float3 position; // the camera, world space — the pinhole; lens origins share its answer
    uint pad0;
    SceneTable* scene;
};

[vk::push_constant]
Params params;

[shader("compute")]
[numthreads(1, 1, 1)]
void resolve_camera() {
    var set = uint4(STACK_EMPTY);
    var start = 0.0;
    for (uint crossed = 0; crossed < RESOLVE_CROSSING_CAP; crossed++) {
        RayDesc ray;
        ray.Origin = params.position;
        ray.Direction = RESOLVE_DIRECTION;
        // The first leg starts at the camera itself; each later one just
        // past the crossing that ended the last, in the ray's own parameter
        // — shade_volume's march, minus the medium sampling.
        ray.TMin = start == 0.0 ? 0.0 : start * (1.0 + RESOLVE_NUDGE) + RESOLVE_NUDGE;
        ray.TMax = 1e30;
        RayQuery<RAY_FLAG_NONE> query;
        // Force everything non-opaque so every surface arrives as a
        // candidate, then commit only the boundaries: the nearest committed
        // hit is the nearest *boundary* crossing, and the walls, props, and
        // fractional-opacity leaves between here and it neither end a leg
        // nor spend the cap — they are not medium extents, however solid.
        // All mask bits, unconditionally: containment is physical, so a
        // camera-invisible glass shell still contains.
        query.TraceRayInline(sceneTlas, RAY_FLAG_FORCE_NON_OPAQUE, 0xff, ray);
        while (query.Proceed()) {
            let candidate = query.CandidateInstanceID();
            if (boundaryOf(params.scene.geometry[candidate]) != BOUNDARY_OPAQUE) {
                query.CommitNonOpaqueTriangleHit();
            }
        }
        if (query.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
            break;
        }
        // Toggle by membership, flags and all — the entry built exactly as
        // intersect and the march build it, so ranking, capacity, and the
        // drop policy cannot disagree with what a path would carry.
        let instance = query.CommittedInstanceID();
        set = crossMedium(set, stackEntry(params.scene.geometry[instance], instance)).set;
        start = query.CommittedRayT();
    }
    params.scene.cameraMedia = set;
}
