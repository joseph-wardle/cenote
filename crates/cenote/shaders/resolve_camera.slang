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

// Legs the walk makes before giving up and keeping what it has — a leg is
// one re-trace, and a coincident group spends one however many boundaries
// stand in it. A bound on pathological nesting, far above anything the
// four-slot set can express; truncation past it is the same class of
// accepted error as the march's `MARCH_LEG_CAP`.
static const uint RESOLVE_LEG_CAP = 32;

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
    // The instance the last leg crossed, so this one can tell its coincident
    // partners from itself. No instance until the first leg commits.
    var crossedInstance = ~0u;
    for (uint leg = 0; leg < RESOLVE_LEG_CAP; leg++) {
        // What counts as the same distance — the march's `window`, for the
        // march's reasons (shade_volume.slang).
        let window = coincidenceWindow(start);
        RayDesc ray;
        ray.Origin = params.position;
        ray.Direction = RESOLVE_DIRECTION;
        // The first leg starts at the camera itself; each later one a
        // window's width *behind* the crossing that ended the last, so the
        // boundaries coincident with it arrive as candidates rather than
        // being stepped over — the march's leg, minus the medium sampling.
        ray.TMin = start == 0.0 ? 0.0 : max(0.0, start - window);
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
            let mesh = params.scene.geometry[candidate];
            if (boundaryOf(mesh) == BOUNDARY_OPAQUE) {
                continue;
            }
            // Coincident with the crossing that ended the last leg: cross it
            // here and let the leg run on to the next distinct one. Nothing
            // in the window may commit — the leg would end where it began.
            // Every boundary class groups here, unlike the march: this walk
            // forces non-opaque, so a dielectric interface arrives as a
            // candidate too.
            if (start > 0.0 && query.CandidateTriangleRayT() <= start + window) {
                if (candidate != crossedInstance) {
                    set = crossMedium(set, stackEntry(mesh, candidate)).set;
                }
                continue;
            }
            query.CommitNonOpaqueTriangleHit();
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
        crossedInstance = instance;
    }
    params.scene.cameraMedia = set;
}
