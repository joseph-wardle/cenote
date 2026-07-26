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
// (bounce 0); the indirect tail excludes its first vertex's emission — that
// emitter is the BSDF draw's own light candidate (D-088, D-134).
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
import reservoir_path;
import restir_scene;
import restir_target;
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
    StoredPathReservoir* reservoirs; // this frame's per-pixel reservoirs (curr), read here
    float4* radiance;            // the film accumulator — the inline delta term adds here
    // The D-092 debug surface's single enum-selected target (§below), non-null
    // exactly when debugView != DEBUG_NONE. One RGBA float per pixel, zero-
    // filled before this wave, so pixels this stage never reaches read black.
    float4* debug;
    // Packed control word: the DebugView enum in the low byte (which view to
    // false-colour into `debug`, DEBUG_NONE etc.), mirroring src/wavefront.rs.
    // Bit 8: visibility-in-weight — the spatial stage ran, so the survivor's W
    // already carries its own visibility (the spatial pass spent the ray). Bit 9:
    // temporal-ran — scales the confidence heatmap. Both flags ride here rather
    // than in fields of their own: the block sits at Vulkan's guaranteed 128
    // push-constant bytes.
    uint flags;
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

// The packed `flags` word: the enum above in the low byte, then two
// which-stages-ran flags that set the confidence heatmap's scale — the
// spatial visibility-in-weight flag in bit 8, the temporal-ran flag in bit 9.
static const uint DEBUG_VIEW_MASK = 0xff;
static const uint DEBUG_VISIBILITY_IN_WEIGHT = 0x100;
static const uint DEBUG_TEMPORAL_IN_WEIGHT = 0x200;

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
// `confidenceScale` is the M value the heatmap saturates near — the candidate
// count for single-frame RIS, its multiple after temporal or spatial reuse
// compounds M (the caller sets it from which stages ran).
float3 debugColor(uint view, Reservoir<PathSample> reservoir, float confidenceScale) {
    switch (view) {
    case DEBUG_SELECTED_LIGHT:
        // The environment reads as one steady near-white; instances scatter
        // across the hue wheel by stable id (a golden-ratio step keeps
        // consecutive ids far apart).
        if (isEnvironmentSample(reservoir.sample)) {
            return float3(0.75, 0.75, 0.78);
        }
        return hueColor(frac(float(reservoir.sample.rcVertex.instance) * 0.6180339887));
    case DEBUG_CONFIDENCE:
        // M — the generalized sample count. Flat at the candidate count for
        // single-frame RIS; temporal history ramps it toward the M-cap and
        // spatial reuse compounds it toward c_tot, so the heatmap's scale grows
        // with the stages that ran (D-093) rather than saturating at once. On a
        // held camera the decay ramp (D-094) hands temporal off, so M visibly
        // relaxes back toward the candidate count as the frame converges.
        return heat(1.0 - exp(-reservoir.confidence / confidenceScale));
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
    Reservoir<PathSample> reservoir = loadPathReservoir(params.reservoirs[pixel]);
    if (!(reservoir.unbiasedWeight > 0.0)) {
        return;
    }
    PathSample sample = reservoir.sample;

    // Unpack the debug word: the view enum, whether the spatial stage already
    // folded the survivor's visibility into W (so no shadow ray is owed here),
    // and whether temporal ran (a scale-only signal for the confidence heatmap).
    let debugView = params.flags & DEBUG_VIEW_MASK;
    let visibilityInWeight = (params.flags & DEBUG_VISIBILITY_IN_WEIGHT) != 0;
    let temporalInWeight = (params.flags & DEBUG_TEMPORAL_IN_WEIGHT) != 0;

    // The debug surface shows what the reservoir *selected*, so it is written
    // here — before the connection is reconstructed — independent of whether
    // the shadow ray or the shade below fires. The confidence heatmap's scale
    // tracks the M actually present so it never clips: the single-frame
    // candidate count on its own; temporal compounds M toward its M-cap ceiling
    // (≈ RESTIR_TEMPORAL_M_CAP × candidates ≈ 320, mirrored from wavefront.rs);
    // spatial reuse a further rough 6× for the default k ≈ 5 neighbours (D-093).
    if (debugView != DEBUG_NONE) {
        var confidenceScale = temporalInWeight ? 320.0 : 16.0;
        if (visibilityInWeight) { confidenceScale *= 6.0; }
        params.debug[pixel] = float4(debugColor(debugView, reservoir, confidenceScale), 1.0);
    }

    // -- The replay-carried survivor (k > 2, D-138; replay kind, D-139) ------
    // Its integrand was formed in *this* pixel's domain by whichever stage
    // last held it — generation at this pixel, or the spatial shift's replay —
    // and re-forming it here would re-trace that replay (m6-plan §4a), so
    // resolve trusts the stored F. No visibility is owed either: the own-pixel
    // walk traced every segment concretely, and a foreign survivor only
    // arrives through spatial, whose target spent the connection ray and
    // whose replay rays are their own geometry tests. Pixel-owned add, like
    // the delta term above.
    if (isReplaySample(sample)
        || (isReconnectionSample(sample) && reconnectionK(sample) > 2)) {
        let deep = throughput * sample.F * reservoir.unbiasedWeight;
        if (any(deep > 0.0)) {
            params.radiance[pixel] += float4(deep, 0.0);
        }
        return;
    }

    // Reconstruct the survivor's connection through the shared `connectSampleAt`
    // (restir_target.slang — the same reconstruction every reuse target ran, so
    // resolve can never drift from what was resampled), whichever kind survived:
    // a light point re-derived exactly (the environment re-samples its stored
    // coordinate, a triangle re-connects its stored point), or a reconnection
    // vertex x₂ re-shaded against its cached suffix (D-134, D-137) — the
    // reconnection shift, but to the resolving pixel itself; re-forming in
    // *this* domain (rather than reading a stored F) is what keeps a reused
    // survivor's colour local.
    // A dropped light or a degenerate connection (behind the emitter, a pole)
    // is invalid and adds nothing; W already carries the Jacobian and MIS
    // weighting, so validity is the only gate here.
    let connection = connectSampleAt(params.scene, params.restir, sample, spawn.position);
    if (!connection.valid) {
        return;
    }
    let toLight = connection.toLight;

    let wiLight = frame.toLocal(toLight);
    let throughSurface = dot(toLight, spawn.normal) < 0.0;
    if (wiLight.z == 0.0 || (wiLight.z < 0.0) != throughSurface) {
        return;
    }

    // The full contribution, unshadowed: L·f·cos over the real closure, scaled
    // by the throughput and the reservoir's UCW. No MIS weight — the light-vs-
    // BSDF balance heuristic was already folded into W at candidate time, so
    // applying one here would double-weight the survivor.
    let f = throughput * connection.radiance * evalBsdf(closure, wiLight, wo) * abs(wiLight.z);
    let contribution = f * reservoir.unbiasedWeight;
    if (!any(contribution > 0.0)) {
        return;
    }

    if (visibilityInWeight || isReconnectionSample(sample)) {
        // No shadow ray owed: the spatial stage already traced this survivor's
        // visibility and folded it into W (W = weightSum/U, the unshadowed
        // target — see restir_spatial), or the sample is a reconnection whose
        // x₁→x₂ segment was found by the own pixel's real trace (a foreign
        // reconnection survivor can only arrive through spatial, whose target
        // spent the ray). Shade unshadowed and add straight to the film —
        // pixel-owned, exactly like the delta term above, so no shadow queue
        // and determinism holds.
        params.radiance[pixel] += float4(contribution, 0.0);
        return;
    }

    // No reuse ran (single-frame RIS): the survivor still owes its one visibility
    // ray. Queue it — trace_shadow applies visibility and lands the contribution.
    ShadowRay shadow;
    shadow.origin = throughSurface ? spawn.below : spawn.position;
    shadow.light = connection.shadowLight;
    shadow.direction = toLight;
    shadow.primitive = connection.shadowPrimitive;
    shadow.contribution = contribution;
    shadow.tmax = connection.shadowTmax;
    shadow.pixel = pixel;
    shadow.pad0 = 0;
    shadow.pad1 = 0;
    shadow.pad2 = 0;
    params.shadows.push(shadow);
}
