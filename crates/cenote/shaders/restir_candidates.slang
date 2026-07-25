// Initial RIS over the primary hit: stream M light candidates plus one
// BSDF-sampled candidate through a per-pixel reservoir and keep one, weighted by
// an unshadowed target. The first of M3's four discrete reservoir stages
// (D-091) — it writes the reservoir the reuse stages will later fold history
// into, and that `restir_resolve` shades.
//
// Two candidate techniques feed the stream, so the reservoir owns *all* of the
// primary hit's non-delta direct lighting (D-088):
//
//   - M **light** candidates from the triangle-only alias table plus the
//     environment coin flip (the light-sampling technique, count M).
//   - one internalized **BSDF** candidate: sample the closure, trace the
//     scattered ray, and reconnect to whatever light it reaches — an emitter
//     triangle, the environment, or nothing (the BSDF technique, count 1).
//
// Combining them is the count-weighted balance heuristic. A candidate at point
// x, whichever technique drew it, gets resampling weight
//
//     w = p̂(x) / (M·p_L(x) + p_B(x)),
//
// the balance-heuristic MIS weight m(x) = q_s(x)/Σ c_t·q_t(x) times p̂(x)/q_s(x)
// with the drawing technique's own density q_s cancelled — so it needs no
// branch on which technique drew x, and reduces to the single-technique
// (1/M)·p̂/p_L exactly when the BSDF cannot sample the direction (p_B = 0). Both
// source densities live in the solid-angle domain (the env and triangle pdfs
// lifted, the BSDF pdf native), so they are commensurable. The target
//
//     p̂(x) = luminance(L(x) · f(x) · |cosθ|),   unshadowed,
//
// is the scalar the reservoir sorts on; visibility, the path throughput, and
// the RGB are deferred to `restir_resolve`, which re-evaluates the survivor for
// real. Because p̂ is unshadowed, the estimator is unbiased only if some
// candidate covers the support of f — the power-alias table (every emitter)
// plus the environment do, by construction (the support-coverage guard in
// wavefront.rs pins M ≥ 1). Delta lights stay out of the reservoir (D-088).
//
// Runs at bounce 0 only, so the medium is vacuum and the throughput is white —
// neither is read here. The BSDF candidate's trace is closest-hit inline
// traversal against the same TLAS the wavefront's other stages bind.

import bluenoise;
import color;
import lights;
import openpbr;
import pathstate;
import reservoir;
import reservoir_path;
import restir_scene;
import rng;
import scene;
import surface;
import textures;

// The scene TLAS, shared with `intersect`/`trace_shadow`: the BSDF candidate
// traces the scattered ray through it to name the light that direction reaches.
[vk::binding(0, 0)]
RaytracingAccelerationStructure sceneTlas;

struct Params {
    Paths paths;
    Queue<uint> hits;      // primary hits, dispatched indirectly; not consumed here
    SceneTable* scene;
    RestirScene* restir;
    StoredPathReservoir* reservoirs; // this frame's per-pixel reservoirs (curr), written here
    uint sampleIndex;      // which sample of every pixel's sequence this wave is
    uint candidates;       // M — the initial-RIS light-candidate count
    uint width;            // frame width — decodes the pixel's (x,y) for the blue-noise key
    uint pad0;             // matches the host's tail padding to 8-byte alignment
};

[vk::push_constant]
Params params;

// Fold one reconnected candidate into the reservoir under the count-weighted
// balance heuristic. `toLight`/`radiance` reconstruct the connection; `lightPdf`
// is p_L (the next-event solid-angle density of this point, selection folded in,
// 0 where light sampling can't reach it); `M` is the light-candidate count. The
// BSDF technique contributes count 1, its density scored here from the closure.
// `reachable` is false for a draw that found no light (a degenerate light draw,
// or a BSDF ray that hit a non-emitter) — it still streams, worth zero weight
// but one unit of confidence, because the technique did draw a sample and the
// balance heuristic's counts must match the draws. On selection the caller's
// running p̂(Y) is updated so finalize needs no re-evaluation.
void streamCandidate(inout Reservoir<PathSample> reservoir,
                     inout float survivorTarget, Closure closure, Frame frame,
                     float3 spawnNormal, float3 wo, PathSample sample,
                     float3 toLight, float3 radiance, float lightPdf, bool reachable,
                     uint M, float uAccept) {
    float target = 0.0;
    float weight = 0.0;
    if (reachable) {
        // The unshadowed target and the BSDF source density share the same
        // sidedness guard the path tracer applies: a candidate below the
        // horizon is legal only where the base transmits, and a direction the
        // closure can't carry scores zero on both — the BSDF technique cannot
        // have produced it either, keeping the balance heuristic consistent.
        let wiLight = frame.toLocal(toLight);
        let throughSurface = dot(toLight, spawnNormal) < 0.0;
        if (wiLight.z != 0.0 && (wiLight.z < 0.0) == throughSurface) {
            target = luminance(radiance * evalBsdf(closure, wiLight, wo) * abs(wiLight.z));
            let bsdfDensity = bsdfPdf(closure, wiLight, wo);
            let denom = float(M) * lightPdf + bsdfDensity;
            weight = (target > 0.0 && denom > 0.0) ? target / denom : 0.0;
        }
    }
    if (reservoirUpdate(reservoir, sample, weight, uAccept)) {
        survivorTarget = target;
    }
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void restir_candidates(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.hits.count()) {
        return;
    }
    uint slot = params.hits.entries[tid.x];
    uint pixel = params.paths.pixel[slot];
    // The blue-noise-ranked sample index, shared across every draw this stage
    // makes for the pixel, so its RIS error lands as screen-space blue noise at
    // low sample counts (D-095). One rank per (pixel, sample), keyed by the mask.
    let ranked = blueRank(uint2(pixel % params.width, pixel / params.width),
                          params.sampleIndex);
    Hit hit = params.paths.hit[slot];
    let direction = params.paths.direction[slot].xyz;

    // Reconstruct the same surface and closure `shade_surface` would — the
    // reservoir must weigh candidates on the exact BSDF the path tracer shades.
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
    // Bounce 0 is a camera ray in vacuum — never inside the surface it hits.
    let closure = resolveClosure(params.scene.bsdfTables, material, wo, false);

    let restir = params.restir;
    let envProb = restir.envSelectProb;
    let M = params.candidates;

    Reservoir<PathSample> reservoir = reservoirInit<PathSample>();
    // p̂ at the held survivor, tracked as the stream runs so finalize needs no
    // re-evaluation (the survivor's target is whatever won the last acceptance).
    float survivorTarget = 0.0;

    // -- M light candidates: the alias table + the environment coin flip -----
    for (uint i = 0; i < M; i++) {
        let uLight = sample_ranked_1d(pixel, ranked,
                                      restirCandidateDimension(i, RESTIR_LIGHT));
        let uPoint = sample_ranked_2d(pixel, ranked,
                                      restirCandidateDimension(i, RESTIR_POINT));
        let uAccept = sample_ranked_1d(pixel, ranked,
                                       restirCandidateDimension(i, RESTIR_ACCEPT));

        // The unified reservoir's NEE (length-2) sample: the reconnection vertex
        // is the light point, the multi-vertex path fields stay zero until step 2.
        // Zero-init so those unused fields are deterministic in the stored buffer.
        PathSample sample = PathSample();
        float3 toLight;
        float3 radiance;
        float lightPdf; // p_L: the draw's solid-angle pdf, selection folded in
        if (uLight < envProb) {
            // The environment: store the image point the draw landed on, which
            // reproduces the direction exactly at resolve (see restir_scene /
            // environment.connect). Storing the *point* — not the primary
            // sample — is what lets the BSDF-escape candidate below share it.
            let e = params.scene.env.sample(uPoint);
            sample.rcVertex.instance = LIGHT_ID_ENVIRONMENT;
            sample.rcVertex.primitive = 0;
            sample.rcVertex.barycentrics = e.uv;
            toLight = e.direction;
            radiance = e.radiance;
            lightPdf = envProb * e.pdf;
        } else {
            // A triangle from the candidate table. The list branch is only
            // reached when there is one (envProb < 1), so the remap is safe.
            let uList = (uLight - envProb) / (1.0 - envProb);
            var c = sampleLights(restir.candidateLights, restir.lightCount,
                                 uList, uPoint, spawn.position);
            // A textured emitter's record carries the map's scale; evaluate the
            // map at the sampled point so p̂ sees the radiance a BSDF-sampled
            // path hitting that point would (the estimators must agree
            // pointwise, exactly as in shade_surface).
            let emitter = params.scene.materials[c.instance];
            if (emitter.emissionTexture != TEXTURE_NONE) {
                c.radiance *= sampleColor(
                    emitter.emissionTexture,
                    interpolateUv(params.scene.geometry[c.instance], c.primitive, c.bary));
            }
            // Stamp the STABLE id, not the volatile instance: the sample must
            // survive a light being added or removed before it is reused.
            sample.rcVertex.instance = restir.instanceToId[c.instance];
            sample.rcVertex.primitive = c.primitive;
            sample.rcVertex.barycentrics = c.bary;
            toLight = c.direction;
            radiance = c.radiance;
            lightPdf = (1.0 - envProb) * c.pdf;
        }

        streamCandidate(reservoir, survivorTarget, closure, frame, spawn.normal, wo,
                        sample, toLight, radiance, lightPdf, lightPdf > 0.0, M, uAccept);
    }

    // -- one internalized BSDF candidate (D-088) -----------------------------
    // Sample the closure, trace the scattered ray, and reconnect to the light
    // that direction reaches. This is the technique light sampling is worst at —
    // a sharp lobe onto a small or distant emitter — so folding it into the same
    // reservoir (rather than a separate MIS'd term) is what makes the reservoir
    // the single owner of primary direct lighting.
    {
        let uLobe = sample_ranked_1d(pixel, ranked,
                                     restirBsdfDimension(RESTIR_BSDF_LOBE));
        let uDir = sample_ranked_2d(pixel, ranked,
                                    restirBsdfDimension(RESTIR_BSDF_DIR));
        let uAccept = sample_ranked_1d(pixel, ranked,
                                       restirBsdfDimension(RESTIR_BSDF_ACCEPT));

        let scatter = sampleBsdf(closure, wo, uLobe, uDir);
        let wiWorld = frame.toWorld(scatter.wi);

        PathSample sample = PathSample();
        float3 toLight = float3(0.0);
        float3 radiance = float3(0.0);
        float lightPdf = 0.0; // p_L at the point the BSDF ray reached
        bool found = false;

        // Trace only a real, correctly-sided scatter — the same guard the
        // continuation ray uses, so a direction through the geometric horizon
        // never spawns a candidate.
        if (scatter.pdf > 0.0 && scatter.wi.z != 0.0
            && (scatter.wi.z > 0.0) == (dot(wiWorld, spawn.normal) > 0.0)) {
            RayDesc ray;
            ray.Origin = scatter.wi.z < 0.0 ? spawn.below : spawn.position;
            ray.Direction = wiWorld;
            ray.TMin = 0.0;
            ray.TMax = 1e30;
            // Closest hit, all instances. This trace only *names* the
            // reconnection target — the survivor's real visibility (opacity
            // included) is retested at resolve — so every crossing commits as
            // opaque and the nearest wins, exactly as `intersect` walks.
            RayQuery<RAY_FLAG_NONE> query;
            query.TraceRayInline(sceneTlas, RAY_FLAG_NONE, 0xFF, ray);
            while (query.Proceed()) {
                query.CommitNonOpaqueTriangleHit();
            }

            if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
                let instance = query.CommittedInstanceID();
                GeometryRecord lightMesh = params.scene.geometry[instance];
                if (lightMesh.light != LIGHT_NONE) {
                    // An emitter: reconnect to the exact point hit. The TLAS
                    // barycentrics index the light record the same way resolve
                    // will (`geometry.light + primitive`), and `connectTriangle`
                    // re-derives the connection — one-sided, so a back-face hit
                    // reports pdf 0 and the candidate is null.
                    let primitive = query.CommittedPrimitiveIndex();
                    let bary = query.CommittedTriangleBarycentrics();
                    LightRecord record = restir.candidateLights[lightMesh.light + primitive];
                    var c = connectTriangle(record, bary, spawn.position);
                    if (c.pdf > 0.0) {
                        let emitter = params.scene.materials[instance];
                        if (emitter.emissionTexture != TEXTURE_NONE) {
                            c.radiance *= sampleColor(emitter.emissionTexture,
                                interpolateUv(lightMesh, primitive, bary));
                        }
                        sample.rcVertex.instance = restir.instanceToId[instance];
                        sample.rcVertex.primitive = primitive;
                        sample.rcVertex.barycentrics = bary;
                        toLight = c.direction;
                        radiance = c.radiance;
                        lightPdf = (1.0 - envProb) * c.pdf;
                        found = true;
                    }
                }
            } else {
                // Escaped to the sky: an environment candidate, stored as the
                // image point its direction maps to, so resolve reconnects it
                // through the same `environment.connect` a light candidate uses.
                let hitUv = params.scene.env.locate(wiWorld);
                let e = params.scene.env.connect(hitUv);
                sample.rcVertex.instance = LIGHT_ID_ENVIRONMENT;
                sample.rcVertex.primitive = 0;
                sample.rcVertex.barycentrics = hitUv;
                toLight = e.direction;
                radiance = e.radiance;
                lightPdf = envProb * e.pdf;
                found = true;
            }
        }

        streamCandidate(reservoir, survivorTarget, closure, frame, spawn.normal, wo,
                        sample, toLight, radiance, lightPdf, found, M, uAccept);
    }

    reservoirFinalize(reservoir, survivorTarget);
    params.reservoirs[pixel] = storePathReservoir(reservoir);
}
