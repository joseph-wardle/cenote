// Resolve the primary hit's reservoir: shade the survivor that came out of
// initial RIS (and, from step 4 on, reuse) and hand its visibility to
// `trace_shadow`. The last of M3's four discrete reservoir stages (D-091).
//
// The reservoir sorted its survivor on an *unshadowed* scalar target; this
// stage re-evaluates it for real — the full RGB L·f·cos and the path
// throughput — scales it by the unbiased contribution weight W the reservoir
// carries, and queues one shadow ray to fold in visibility. No MIS weight is
// applied here: the reservoir already combined the light and BSDF techniques
// under the count-weighted balance heuristic (restir_candidates), so that
// weighting is baked into W. The reservoir has replaced *how* the whole direct
// term is estimated, not what the connection is worth, so ReSTIR-on and
// ReSTIR-off converge to the same image (the unbiasedness gate).
//
// The estimator identity E[H(Y)·W_Y] = ∫H holds for any H, so the visibility
// and throughput left out of the target reappear here at no cost to
// unbiasedness. A directly visible light's emission stays in `shade_surface`
// (bounce 0); the BSDF continuation is indirect-only, its first-hit emission
// suppressed — that emitter is already a reservoir candidate (D-088).
//
// The delta lights (distant, point) are the reservoir's deliberate exclusion:
// zero solid angle, so no BSDF sample reaches them and no reservoir candidate
// represents them — a reservoir would only add resampling variance to a term
// that is already exact. So this stage also adds their primary-hit direct
// lighting, additively and outside the reservoir: one delta drawn ∝ power from
// the delta-only table, connected exactly (MIS weight 1), its visibility traced
// inline. `shade_surface` skips its own NEE at the primary hit (the reservoir
// owns it), so without this add the primary hit would see no delta light at all.
//
// Runs at bounce 0 only: the medium is vacuum and the throughput white.

import lights;
import openpbr;
import pathstate;
import reservoir;
import reservoir_di;
import restir_scene;
import rng;
import scene;
import surface;
import textures;

// The scene TLAS, shared with `intersect`/`trace_shadow`/`restir_candidates`:
// the additive delta-light term traces its own visibility ray inline here (§the
// delta block below), rather than through the shadow queue the survivor claims.
[vk::binding(0, 0)]
RaytracingAccelerationStructure sceneTlas;

struct Params {
    Paths paths;
    Queue<uint> hits;            // primary hits, dispatched indirectly; not consumed here
    Queue<ShadowRay> shadows;    // the survivor's visibility ray, for trace_shadow
    SceneTable* scene;
    RestirScene* restir;
    StoredReservoir* reservoirs; // this frame's per-pixel reservoirs (curr), read here
    float4* radiance;            // the film accumulator — the inline delta term adds here
    // The D-092 debug surface's single enum-selected target (§below), non-null
    // exactly when debugView != DEBUG_NONE. One RGBA float per pixel, zero-
    // filled before this wave, so pixels this stage never reaches read black.
    float4* debug;
    // Which debug view to false-colour into `debug`, or DEBUG_NONE to write
    // none. Mirrors src/wavefront.rs's DebugView.
    uint debugView;
    uint sampleIndex;            // this wave's sample — the delta term's one NEE draw
};

[vk::push_constant]
Params params;

// -- The D-092 debug surface -------------------------------------------------
// A first-class observability workstream, landing with the first estimator so
// steps 4-5 are never debugged blind: ReSTIR bias shifts a converged mean a few
// percent and looks plausible, so the survivor's *selection* is made visible
// here, at the one stage that owns it. The enum picks one view; the colour is
// display-ready (the viewer presents it without the tone curve). Mirrors
// src/wavefront.rs's DebugView.
static const uint DEBUG_NONE = 0;
static const uint DEBUG_SELECTED_LIGHT = 1; // false-colour by stable light id
static const uint DEBUG_CONFIDENCE = 2;     // M heatmap
static const uint DEBUG_UNBIASED_WEIGHT = 3; // W heatmap

// A saturated colour from a hue in [0, 1): HSV(h, 1, 1) → RGB. Distinct hues
// make neighbouring light ids read apart at a glance.
float3 hueColor(float h) {
    let p = abs(frac(h + float3(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
    return saturate(p - 1.0);
}

// A perceptual-ish blue→cyan→green→yellow→red ramp for scalar heatmaps.
float3 heat(float t) {
    const float3 stops[5] = {
        float3(0.05, 0.05, 0.35), // deep blue — low
        float3(0.10, 0.55, 0.85), // cyan
        float3(0.15, 0.75, 0.30), // green
        float3(0.95, 0.85, 0.20), // yellow
        float3(0.90, 0.20, 0.15), // red — high
    };
    let x = saturate(t) * 4.0;
    let i = clamp(int(floor(x)), 0, 3);
    return lerp(stops[i], stops[i + 1], saturate(x - float(i)));
}

// The false-colour for the survivor under the selected view. `reservoir` is the
// held reservoir; its confidence is M and its unbiasedWeight is W.
float3 debugColor(uint view, Reservoir<DirectLightSample> reservoir) {
    switch (view) {
    case DEBUG_SELECTED_LIGHT:
        // The environment reads as one steady near-white; instances scatter
        // across the hue wheel by stable id (a golden-ratio step keeps
        // consecutive ids far apart).
        if (isEnvironmentSample(reservoir.sample)) {
            return float3(0.75, 0.75, 0.78);
        }
        return hueColor(frac(float(reservoir.sample.light) * 0.6180339887));
    case DEBUG_CONFIDENCE:
        // M — the generalized sample count. Flat at the candidate count for
        // single-frame RIS; the slot that fills in once reuse compounds it.
        return heat(1.0 - exp(-reservoir.confidence / 16.0));
    case DEBUG_UNBIASED_WEIGHT:
        // W — the unbiased contribution weight, Reinhard-mapped so any
        // magnitude lands in [0, 1).
        return heat(reservoir.unbiasedWeight / (1.0 + reservoir.unbiasedWeight));
    default:
        return float3(0.0);
    }
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void restir_resolve(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.hits.count()) {
        return;
    }
    uint path = params.hits.entries[tid.x];
    uint pixel = params.paths.pixel[path];

    Hit hit = params.paths.hit[path];
    let direction = params.paths.direction[path].xyz;
    let throughput = params.paths.throughput[path].xyz;

    // The same surface and closure `restir_candidates` weighed the sample on —
    // reconstructed unconditionally, because both the additive delta term and
    // the reservoir survivor shade on it (and the delta term runs even where the
    // reservoir came back empty).
    GeometryRecord mesh = params.scene.geometry[hit.instance];
    let uv = interpolateUv(mesh, hit.primitive, hit.barycentrics);
    let material = resolveMaterial(params.scene.materials[hit.instance], uv);
    let spawn = spawnPoint(mesh, hit, direction);
    var shading = spawn.shading;
    if (material.normalTexture != TEXTURE_NONE) {
        shading = mappedShadingNormal(mesh, hit, uv, material.normalTexture, spawn);
    }
    let frame = Frame(shading);
    let wo = frame.toLocal(-direction);
    let closure = resolveClosure(params.scene.bsdfTables, material, wo, false);

    // -- Delta lights: exact additive NEE, outside the reservoir (D-088) -----
    // Draw one delta ∝ power and add its exact connection straight to the film.
    // The selection pdf is the delta's share of the delta-only table; there is
    // no env/table split and no MIS weight — a delta has no BSDF partner. The
    // write is pixel-owned (one resolve thread per primary hit), so it stays
    // bitwise-deterministic without a shadow queue — whose one-ray-per-pixel
    // rule the survivor already claims this dispatch. Independent of the
    // reservoir, so it runs before the empty-reservoir return below.
    if (params.restir.deltaCount > 0) {
        let uSelect = sample_1d(pixel, params.sampleIndex, restirDeltaDimension());
        // The delta table holds no triangles, so `sampleLights`' point-on-light
        // draw is never consumed; a zero placeholder keeps its shared signature.
        let c = sampleLights(params.restir.deltaLights, params.restir.deltaCount,
                             uSelect, float2(0.0), spawn.position);
        let wiLight = frame.toLocal(c.direction);
        let throughSurface = dot(c.direction, spawn.normal) < 0.0;
        if (c.pdf > 0.0 && wiLight.z != 0.0 && (wiLight.z < 0.0) == throughSurface) {
            // L·f·cos / p_select, weight 1. A point light's 1/r² falloff is
            // already folded into c.radiance by `sampleLights`.
            let deltaTerm = throughput * c.radiance
                * evalBsdf(closure, wiLight, wo) * abs(wiLight.z) / c.pdf;
            if (any(deltaTerm > 0.0)) {
                RayDesc ray;
                ray.Origin = throughSurface ? spawn.below : spawn.position;
                ray.Direction = c.direction;
                ray.TMin = 0.0;
                ray.TMax = c.tmax;
                let vis = shadowVisibility(sceneTlas, params.scene, ray,
                                           c.instance, c.primitive);
                if (vis > 0.0) {
                    params.radiance[pixel] += float4(deltaTerm * vis, 0.0);
                }
            }
        }
    }

    // -- The reservoir survivor ----------------------------------------------
    // The survivor and its unbiased contribution weight. An empty reservoir
    // (nothing selected, or the target rejected it) carries W 0 and adds
    // nothing — no shadow ray, no continuation (that is shade_surface's). The
    // delta term above already landed; this pixel's debug surface stays at its
    // black zero-fill (nothing selected).
    Reservoir<DirectLightSample> reservoir = loadReservoir(params.reservoirs[pixel]);
    if (!(reservoir.unbiasedWeight > 0.0)) {
        return;
    }
    DirectLightSample sample = reservoir.sample;

    // The debug surface shows what the reservoir *selected*, so it is written
    // here — before the connection is reconstructed — independent of whether
    // the shadow ray or the shade below fires.
    if (params.debugView != DEBUG_NONE) {
        params.debug[pixel] = float4(debugColor(params.debugView, reservoir), 1.0);
    }

    // Reconstruct the survivor's connection — the exact direction, radiance,
    // and next-event pdf its target was built from (reconnection is exact: the
    // environment re-samples its stored coordinate, a triangle re-derives from
    // its stored point).
    float3 toLight;
    float3 radiance;
    float connectPdf;      // the reconnection's solid-angle pdf — a validity
                           //   gate only; W already carries the MIS weighting
    uint shadowLight;      // shadow-ray identity target, or LIGHT_NONE
    uint shadowPrimitive;
    float shadowTmax;
    if (isEnvironmentSample(sample)) {
        let e = params.scene.env.connect(sample.barycentrics);
        toLight = e.direction;
        radiance = e.radiance;
        connectPdf = e.pdf;
        shadowLight = LIGHT_NONE;
        shadowPrimitive = 0;
        shadowTmax = 1e30;
    } else {
        // Translate the stable id back to the current instance; a light that
        // was deleted since the sample was stored resolves to nothing (drop).
        let instance = params.restir.idToInstance[sample.light];
        if (instance == LIGHT_NONE) {
            return;
        }
        let lightMesh = params.scene.geometry[instance];
        // The candidate table's triangle records share the combined table's
        // indices, so the stored (instance, primitive) addresses this row.
        LightRecord record = params.restir.candidateLights[lightMesh.light + sample.primitive];
        var c = connectTriangle(record, sample.barycentrics, spawn.position);
        let emitter = params.scene.materials[instance];
        if (emitter.emissionTexture != TEXTURE_NONE) {
            c.radiance *= sampleColor(
                emitter.emissionTexture,
                interpolateUv(lightMesh, sample.primitive, sample.barycentrics));
        }
        toLight = c.direction;
        radiance = c.radiance;
        connectPdf = c.pdf;
        shadowLight = instance;
        shadowPrimitive = sample.primitive;
        shadowTmax = c.tmax;
    }
    // A survivor whose connection degenerated (behind the emitter, a pole) adds
    // nothing; W built on a positive target keeps this rare.
    if (!(connectPdf > 0.0)) {
        return;
    }

    let wiLight = frame.toLocal(toLight);
    let throughSurface = dot(toLight, spawn.normal) < 0.0;
    if (wiLight.z == 0.0 || (wiLight.z < 0.0) != throughSurface) {
        return;
    }

    // The full contribution, unshadowed: L·f·cos over the real closure, scaled
    // by the throughput and the reservoir's UCW. No MIS weight — the light-vs-
    // BSDF balance heuristic was already folded into W at candidate time, so
    // applying one here would double-weight the survivor.
    let f = throughput * radiance * evalBsdf(closure, wiLight, wo) * abs(wiLight.z);
    let contribution = f * reservoir.unbiasedWeight;
    if (any(contribution > 0.0)) {
        ShadowRay shadow;
        shadow.origin = throughSurface ? spawn.below : spawn.position;
        shadow.light = shadowLight;
        shadow.direction = toLight;
        shadow.primitive = shadowPrimitive;
        shadow.contribution = contribution;
        shadow.tmax = shadowTmax;
        shadow.pixel = pixel;
        shadow.pad0 = 0;
        shadow.pad1 = 0;
        shadow.pad2 = 0;
        params.shadows.push(shadow);
    }
}
