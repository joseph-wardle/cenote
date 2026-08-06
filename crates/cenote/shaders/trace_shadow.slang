// The shadow-ray stage: trace each queued next-event connection and, when
// the light is reached, add its already-weighted contribution to the
// pixel. Fractional-opacity occluders attenuate the connection
// *deterministically* — each crossing multiplies 1 − opacity — so alpha
// cards cast correct soft shadows and this kernel stays RNG-free
// (Cycles' shipped split of the transparency policy). Runs once per
// bounce round, so a pixel sees at most one shadow ray per dispatch — the
// plain += below stays race-free, and the barriers between rounds order
// the adds deterministically.

import pathstate;
import scene;

// The scene TLAS
[vk::binding(0, 0)]
RaytracingAccelerationStructure sceneTlas;

struct Params {
    Queue<ShadowRay> shadows;
    float4* radiance; // the film's per-pixel accumulator for this wave
    SceneTable* scene; // materials — the opacity each crossing attenuates by
};

[vk::push_constant]
Params params;

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void trace_shadow(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.shadows.count()) {
        return;
    }
    ShadowRay shadow = params.shadows.entries[tid.x];

    // Every connection crosses the open space a global medium fills, and a
    // homogeneous extinction over a known length is closed-form — so media
    // attenuate connections here, once for both the surface and the volume
    // stage, and this kernel stays RNG-free. A connection aimed at the
    // environment or a distant light runs to `tmax` 1e30: infinite optical
    // depth, and nothing arrives.
    //
    // TODO: a connection launched from *inside* a refractive interior takes
    // the global extinction over its whole length, including the part still
    // inside the glass. The renderer applied no extinction at all there
    // before a global medium existed, so this is not a regression — but it
    // is wrong, and only a per-crossing extent (which bounded media carry
    // and the depth-1 stack cannot express) fixes it. Until then, a scene
    // with both a global medium and refractive interiors is approximate.
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
    let attenuation = shadowVisibility(sceneTlas, params.scene, ray,
                                       shadow.light, shadow.primitive);
    if (attenuation > 0.0) {
        params.radiance[shadow.pixel] += float4(contribution * attenuation, 0.0);
    }
}
