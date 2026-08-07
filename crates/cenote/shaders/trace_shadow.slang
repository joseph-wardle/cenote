// The shadow-ray stage: trace each queued next-event connection and, when
// the light is reached, add its already-weighted contribution to the
// pixel. Fractional-opacity occluders attenuate the connection
// *deterministically* — each crossing multiplies 1 − opacity — so alpha
// cards cast correct soft shadows and this kernel stays RNG-free
// (Cycles' shipped split of the transparency policy). Runs once per
// bounce round, so a pixel sees at most one shadow ray per dispatch — the
// plain += below stays race-free, and the barriers between rounds order
// the adds deterministically.
//
// Two entry points from one body, `VOLUMES` apart, because register
// allocation is per *kernel*: measuring a connection's way through bounded
// volumes costs a loop-carried extent and membership, and a scene with no
// volumes was measured to pay for them anyway — a uniformly-false branch,
// a hoisted tail, and a generic instantiated twice in one kernel all cost
// the same quarter of this stage. Only the split recovers it. Which entry
// point a wave records is `has_volumes`, host-side.

import pathstate;
import scene;

// The scene TLAS
[vk::binding(0, 0)]
RaytracingAccelerationStructure sceneTlas;

struct Params {
    Queue<ShadowRay> shadows;
    float4* radiance; // the film's per-pixel accumulator for this wave
    SceneTable* scene; // materials — the opacity each crossing attenuates by
    // The medium stack, for the volumes each connection *starts* inside.
    // Read only by the `trace_shadow_volumes` entry point, and null for the
    // other — a scene with interiors alone allocates a stack whose volume
    // lanes were never seeded, so having one is not the same as having them.
    uint4* stack;
};

[vk::push_constant]
Params params;

void traceShadow<let VOLUMES : int>(uint3 tid) {
    if (tid.x >= params.shadows.count()) {
        return;
    }
    ShadowRay shadow = params.shadows.entries[tid.x];

    // Every connection crosses the open space a global medium fills, and a
    // homogeneous extinction over a known length is closed-form — so media
    // attenuate connections here, once for both the surface and the volume
    // stage, and this kernel stays RNG-free. A connection aimed at the
    // environment or a distant light runs to `tmax` 1e30: infinite optical
    // depth, and nothing arrives. Bounded volumes are the traversal's to
    // measure, below.
    //
    // TODO: a connection launched from *inside* a refractive interior takes
    // the global extinction over its whole length, including the part still
    // inside the glass. The renderer applied no extinction at all there
    // before a global medium existed, so this is not a regression — but it
    // is wrong, and only a per-crossing extent (which the interior lane, one
    // instance wide, cannot express) fixes it. Until then, a scene with both
    // a global medium and refractive interiors is approximate.
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
    // Fractional-opacity candidates that aren't the target pass through, each
    // multiplying its transparency into the connection; the target itself
    // commits, opaque or not. The identity, opacity, and epsilon-free reasoning
    // all live in `shadowVisibility` (scene.slang).
    // An `if`, not a select: `params.stack` is null in the other
    // specialization, and a select would have to evaluate the load.
    var inside = uint3(STACK_EMPTY);
    if (VOLUMES != 0) {
        inside = insideOf(params.stack, shadow.path);
    }
    let attenuation = shadowVisibility<VOLUMES>(sceneTlas, params.scene, ray,
                                                shadow.light, shadow.primitive, inside);
    if (any(attenuation > 0.0)) {
        params.radiance[shadow.pixel] += float4(contribution * attenuation, 0.0);
    }
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void trace_shadow(uint3 tid: SV_DispatchThreadID) {
    traceShadow<0>(tid);
}

// The same stage for a scene with volume-bounding meshes: `params.stack`
// then addresses the pool's medium stack, and every connection's way through
// the volumes it crosses is measured and folded in.
[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void trace_shadow_volumes(uint3 tid: SV_DispatchThreadID) {
    traceShadow<1>(tid);
}
