// The shadow-ray stage: trace each queued next-event connection and, when
// the light is reached, add its already-weighted contribution to the
// pixel. Fractional-opacity occluders attenuate the connection
// *deterministically* — each crossing multiplies 1 − opacity — so alpha
// cards cast correct soft shadows and this kernel draws no random numbers at
// all unless a grid medium is in the way (Cycles' shipped split of the
// transparency policy). A pixel sees at most one shadow ray per dispatch —
// one vertex connection per round in the main queue, one walk-exit
// connection in its own queue and dispatch — so the plain += below stays
// race-free, and the barriers between dispatches order the adds
// deterministically.
//
// Three entry points from one body, `MEASURES` apart, because register
// allocation is per *kernel*: a scene with no volumes was measured to pay
// for the loop-carried extent and membership anyway — a uniformly-false
// branch, a hoisted tail, and a generic instantiated twice in one kernel
// all cost the same quarter of this stage, and only the split recovers it.
// Which entry point a wave records is `has_volumes`/`has_heterogeneous`,
// host-side.

import pathstate;
import rng;
import scene;
import tracking;

// The scene TLAS
[vk::binding(0, 0)]
RaytracingAccelerationStructure sceneTlas;

// The NanoVDB grid pool. Partially bound: only the tracking entry point
// reads it, and it is dispatched only for scenes whose prep wrote it.
[[vk::binding(5, 0)]]
StructuredBuffer<uint> gridPool;

// What this stage's three shapes measure, each adding to the one below.
static const int MEASURE_OPACITY = 0; // fractional-opacity crossings alone
static const int MEASURE_VOLUMES = 1; // + the extent of the volumes crossed
static const int MEASURE_GRIDS = 2;   // + ratio tracking through their grids

struct Params {
    Queue<ShadowRay> shadows;
    float4* radiance; // the film's per-pixel accumulator for this wave
    SceneTable* scene; // materials — the opacity each crossing attenuates by
    // The medium set, for the volumes each connection *starts* inside.
    // Read only by the volume-measuring entry points, and null for the
    // plain one.
    uint4* stack;
    // The wave's sample index, which with the connection's own key seeds
    // the grid transmittance's draws. Read only by the grid entry point.
    uint sampleIndex;
    uint pad0;
};

[vk::push_constant]
Params params;

// Remember `instance` if a grid fills it and we have not got it already.
// One entry per *placement*, matching `splitMedia`, which gives a medium
// filling two coincident placements twice the density; deduplicated because
// one connection can cross the same shell twice — in and out — and
// `latticeSpan` already spans the whole crossing. Past the cap the rest are
// dropped, which is one grid's worth of missing shadow: too little fog,
// never a wall that is not there.
void rememberGrid(SceneTable* scene, uint instance, inout uint gridInstances[MEDIUM_SLOTS],
                  inout uint count) {
    let medium = scene.geometry[instance].medium;
    if (medium == MEDIUM_NONE || scene.media[medium].grid == GRID_NONE) {
        return;
    }
    for (uint slot = 0; slot < count; slot++) {
        if (gridInstances[slot] == instance) {
            return;
        }
    }
    if (count < MEDIUM_SLOTS) {
        gridInstances[count++] = instance;
    }
}

// Trace one next-event connection and return what reaches the light: 0 when a
// solid occluder blocks it, otherwise the product of (1 − opacity) over every
// fractional-opacity crossing — Cycles' deterministic transparency split —
// times the transmittance of every volume the connection passes through.
// `light`/`primitive` name the connection's aimed target (or `LIGHT_NONE`,
// which no instance carries, for the environment and delta lights): the
// connection is visible iff the nearest solid thing along the ray is that
// target or nothing. The identity test needs no distance epsilon — the ray's
// `tmax` is bounded to the target (see `ShadowRay` in pathstate.slang).
//
// `inside` is the launching vertex's medium set; the volumes in it are what
// this pass measures, and the interior — whose extent one instance index
// cannot express — is the approximation the TODO below names. Homogeneous
// volume extents are measured, not marched: for one volume the optical depth
// over [0, tmax] is σ_t times the extent inside it, and that extent is
//
//     Σ over crossings of (∓ t)  +  tmax if the far end is still inside,
//
// signed − where the ray enters and + where it leaves. Every term is a plain
// sum, so the candidates may arrive in any order — which is the whole reason
// overlapping volumes were defined to *add*, since an ordered walk is the one
// thing `Proceed()` cannot give. Membership at the far end is the same
// order-free question, answered by parity.
//
// A grid has no such closed form, and takes none of this: the traversal only
// *names* the placements a grid fills, and each is ratio-tracked afterwards
// over the span its own lattice box cuts out of [0, tmax]. That span is a
// property of the ray and the grid alone, so the estimate stays independent
// of the order the candidates arrived in, exactly as the sum is.
//
// The price of unordered is cancellation: a homogeneous thickness reaches
// the sum as the difference of two distances measured from the connection's
// origin, so a shell thin against its own range loses that difference to
// float32, and at 1e-7 of it there is nothing left. Marching would subtract
// adjacent crossings instead, which is the one thing this pass cannot do.
float3 connectionTransmittance<let MEASURES : int>(RayDesc ray, uint light, uint primitive,
                                                   uint4 inside, uint pixel, uint key) {
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(sceneTlas, RAY_FLAG_NONE, 0xFF, ray);
    var attenuation = 1.0;
    var extent = float3(0.0);
    var atEnd = inside;
    uint gridInstances[MEDIUM_SLOTS];
    var gridCount = 0u;
    if (MEASURES == MEASURE_GRIDS) {
        // A connection that starts inside a shell crosses no face of it, so
        // membership is where those placements are named.
        for (uint slot = 0; slot < MEDIUM_SLOTS; slot++) {
            if (inside[slot] != STACK_EMPTY && (inside[slot] & STACK_INTERIOR) == 0) {
                rememberGrid(params.scene, inside[slot] & STACK_INDEX_MASK, gridInstances,
                             gridCount);
            }
        }
    }
    while (query.Proceed()) {
        let instance = query.CandidateInstanceID();
        if (instance == light && query.CandidatePrimitiveIndex() == primitive) {
            query.CommitNonOpaqueTriangleHit();
        } else if (MEASURES != MEASURE_OPACITY
                   && boundaryOf(params.scene.geometry[instance]) == BOUNDARY_NULL) {
            // A volume's boundary is not a surface: it never occludes, and
            // committing it — which its full coverage would otherwise do —
            // would read a fog bank as a solid wall.
            let t = query.CandidateTriangleRayT();
            extent += extinctionOf(params.scene, instance)
                      * (query.CandidateTriangleFrontFace() ? -t : t);
            atEnd = crossMedium(atEnd, instance).set;
            if (MEASURES == MEASURE_GRIDS) {
                rememberGrid(params.scene, instance, gridInstances, gridCount);
            }
        } else {
            attenuation *= 1.0 - opacityAt(params.scene, instance,
                                           query.CandidatePrimitiveIndex(),
                                           query.CandidateTriangleBarycentrics());
            if (attenuation < 1e-4) {
                return float3(0.0); // solidly occluded, to any precision that matters
            }
        }
    }
    if (query.CommittedStatus() != COMMITTED_NOTHING
        && (query.CommittedInstanceID() != light
            || query.CommittedPrimitiveIndex() != primitive)) {
        return float3(0.0);
    }
    if (MEASURES == MEASURE_OPACITY) {
        return float3(attenuation);
    }
    for (uint slot = 0; slot < MEDIUM_SLOTS; slot++) {
        // Volumes only: an interior shares the set but not this sum, and
        // the crossings above could only ever have added a null boundary.
        if (atEnd[slot] != STACK_EMPTY && (atEnd[slot] & STACK_INTERIOR) == 0) {
            extent += extinctionOf(params.scene, atEnd[slot] & STACK_INDEX_MASK) * ray.TMax;
        }
    }
    // A negative extent means a volume was entered and never left, or wound
    // inside-out; clamping is what keeps that a missing shadow rather than a
    // medium that amplifies.
    var transmittance = attenuation * exp(-max(extent, float3(0.0)));
    if (MEASURES == MEASURE_GRIDS) {
        for (uint slot = 0; slot < gridCount; slot++) {
            transmittance *= gridTransmittance(
                params.scene, gridPool, gridInstances[slot], ray.Origin, ray.Direction,
                ray.TMax, shadowSeed(pixel, params.sampleIndex, key, gridInstances[slot]));
            if (!any(transmittance > 0.0)) {
                // The roulette took the connection. Every grid left would be
                // a walk through a cloud to multiply zero by.
                break;
            }
        }
    }
    return transmittance;
}

void traceShadow<let MEASURES : int>(uint3 tid) {
    if (tid.x >= params.shadows.count()) {
        return;
    }
    ShadowRay shadow = params.shadows.entries[tid.x];

    // Every connection crosses the open space a global medium fills, and a
    // homogeneous extinction over a known length is closed-form — so media
    // attenuate connections here, once for both the surface and the volume
    // stage. A connection aimed at the environment or a distant light runs
    // to `tmax` 1e30: infinite optical depth, and nothing arrives. Bounded
    // volumes are the traversal's to measure, below.
    //
    // TODO: a connection launched from *inside* a refractive interior takes
    // the global extinction over its whole length, including the part still
    // inside the glass. The renderer applied no extinction at all there
    // before a global medium existed, so this is not a regression — but it
    // is wrong, and only a per-crossing extent — which this pass measures
    // for volumes and cannot for an interior, a connection being unable to
    // refract out of one — fixes it. Until then, a scene with both a global
    // medium and refractive interiors is approximate.
    var contribution = shadow.contribution;
    let global = params.scene.globalMedium;
    if (global != MEDIUM_NONE) {
        contribution *= exp(-params.scene.media[global].sigmaT * shadow.tmax);
        if (!any(contribution > 0.0)) {
            return;
        }
    }

    RayDesc ray;
    ray.Origin = shadow.origin;
    ray.Direction = shadow.direction;
    ray.TMin = 0.0;
    ray.TMax = shadow.tmax;

    // Closest hit, not accept-first: the ray is aimed at a point on one
    // triangle of the light, so the connection is visible iff the nearest
    // solid thing along the ray *is* that (instance, primitive) — or
    // nothing, when the ray numerically grazes past its silhouette, aims
    // at the environment, or aims at a delta light within `tmax`.
    // An `if`, not a select: `params.stack` is null in the plain
    // specialization, and a select would have to evaluate the load.
    var inside = uint4(STACK_EMPTY);
    if (MEASURES != MEASURE_OPACITY) {
        inside = params.stack[shadow.path];
    }
    let attenuation = connectionTransmittance<MEASURES>(
        ray, shadow.light, shadow.primitive, inside, shadow.pixel, shadow.key);
    if (any(attenuation > 0.0)) {
        params.radiance[shadow.pixel] += float4(contribution * attenuation, 0.0);
    }
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void trace_shadow(uint3 tid: SV_DispatchThreadID) {
    traceShadow<MEASURE_OPACITY>(tid);
}

// The same stage for a scene with volume-bounding meshes: `params.stack`
// then addresses the pool's medium stack, and every connection's way through
// the volumes it crosses is measured and folded in.
[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void trace_shadow_volumes(uint3 tid: SV_DispatchThreadID) {
    traceShadow<MEASURE_VOLUMES>(tid);
}

// And for a scene where one of those volumes is a density grid, whose
// transmittance is tracked rather than integrated.
[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void trace_shadow_grids(uint3 tid: SV_DispatchThreadID) {
    traceShadow<MEASURE_GRIDS>(tid);
}
