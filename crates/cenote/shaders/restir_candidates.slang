// Initial RIS over the primary hit: stream M light candidates plus one
// BSDF-sampled draw through a per-pixel reservoir and keep one, weighted by
// an unshadowed target. The first of M3's four discrete reservoir stages
// (D-091) — it writes the reservoir the reuse stages will later fold history
// into, and that `restir_resolve` shades.
//
// Two candidate techniques feed the stream, so the reservoir owns *all* of the
// primary hit's non-delta path integral — direct and indirect (D-088, D-134):
//
//   - M **light** candidates from the triangle-only alias table plus the
//     environment coin flip (the light-sampling technique, count M).
//   - one internalized **BSDF** draw (the BSDF technique, count 1): sample the
//     closure and trace the scattered ray. It yields up to two candidates with
//     disjoint path supports — the *light* that direction reaches (an emitter
//     triangle or the environment, a length-2 sample), and the *continuation*
//     past a surface hit (the indirect tail from the reconnection vertex x₂, a
//     length-≥3 sample only this technique can produce).
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
// Runs at bounce 0 only, so the arriving segment is vacuum and the throughput
// white — the path-state buffers are never read here. (The indirect tail tracks
// its own medium from the BSDF draw's lobe onward.) The BSDF candidate's trace
// is closest-hit inline traversal against the same TLAS the wavefront's other
// stages bind.

import bluenoise;
import color;
import lights;
import nee;
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
    uint maxBounces;       // path depth cap — the indirect tail's own bound (D-134)
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

// Trace the indirect tail from the reconnection vertex x₂ and return Lo(x₂→x₁):
// the outgoing *reflected* radiance leaving x₂ toward x₁, estimated by one
// path-traced sub-path — the very per-vertex estimator `shade_surface` runs
// across bounces (`sampleNee` + BSDF continuation under per-vertex MIS, the
// shared pieces in nee.slang), accumulated into a return value instead of the
// film. This is the length-≥3 candidate's tail (D-134): the reservoir owns the
// whole path integral at x₁, so the BSDF candidate continues past its hit
// rather than dying. The value is baked with the actual wo₂ = x₂→x₁; for a
// diffuse x₂ it equals the direction-independent exit radiance the reconnection
// shift reuses (step 3 un-bakes f₂ for glossy vertices).
//
// x₂'s own emission is deliberately excluded (reflected radiance only): for an
// emitter x₂ it is the BSDF draw's *light* candidate — a length-2 sample
// competing with next-event sampling — and folding it into this tail, whose
// weight sees no light-sampling density, would double-count it.
//
// Light sampling is MIS here (ReSTIR's reservoir owns the primary; the tail
// matching the default estimator is what converges it to brute force).
// Traversal treats every crossing as opaque (nearest hit wins) — stochastic
// opacity in the reused tail is deferred with volumes (deferrals.md); the
// checkpoint scenes carry none.
//
// `hit`/`dir` are x₂ and the unit x₁→x₂ direction; `origin` is the traced ray's
// start (x₁'s spawn point, below-surface for a transmission) and `medium` the
// interior that segment crossed — x₁'s own instance when the draw refracted
// into a closed surface, `MEDIUM_NONE` otherwise — so the tail absorbs and
// resolves x₂'s closure exactly as `shade_surface`'s continuation would have.
float3 traceTail(uint pixel, uint sampleIndex, SceneTable* scene, uint maxBounces,
                 Hit hit, float3 dir, float3 origin, uint medium) {
    float3 outgoing = float3(0.0);   // Lo(x₂→x₁), accumulated
    float3 throughput = float3(1.0); // the tail's own path weight (x₁'s f·cos is
                                     //   applied by the caller's streamCandidate)
    float prevScatterPdf = 0.0;      // the last scatter's pdf, for emission MIS;
                                     //   unread at bounce 1 (emission excluded)

    // x₂ is the path's bounce 1 (x₁ the primary hit, bounce 0), so the tail's
    // draws take the per-bounce dimensions from bounce 1 on — decorrelated from
    // the reservoir's own candidate blocks, and unused by bounce 0 in ReSTIR
    // mode (shade_surface's continuation is off there).
    for (uint bounce = 1; bounce < maxBounces; bounce++) {
        GeometryRecord mesh = scene.geometry[hit.instance];
        let uv = interpolateUv(mesh, hit.primitive, hit.barycentrics);
        let material = resolveMaterial(scene.materials[hit.instance], uv);
        let spawn = spawnPoint(mesh, hit, dir);

        // Beer–Lambert over the arriving segment, if it crossed an interior.
        if (medium != MEDIUM_NONE) {
            throughput *= interiorTransmittance(scene, medium, origin, spawn.position);
        }

        // Emission: the BSDF strategy reaching a light, MIS-weighted against
        // the previous vertex's next-event draw. Excluded at bounce 1 — see the
        // header — where it is either absent (a non-emitter x₂) or the BSDF
        // draw's own light candidate (an emitter x₂).
        if (bounce > 1 && mesh.light != LIGHT_NONE && spawn.frontFace) {
            let neePdf = emitterHitNeePdf(scene, mesh, hit.primitive, spawn, origin, dir);
            outgoing += throughput * material.emission
                      * powerHeuristic(prevScatterPdf, neePdf);
        }

        var shading = spawn.shading;
        if (material.normalTexture != TEXTURE_NONE) {
            shading = mappedShadingNormal(mesh, hit, uv, material.normalTexture, spawn);
        }
        let frame = Frame(shading);
        let wo = frame.toLocal(-dir);
        let closure = resolveClosure(scene.bsdfTables, material, wo, medium == hit.instance);

        // The depth cap: the path ends here (NEE from this vertex would outrun
        // any BSDF-traced path length, so the two strategies stay matched).
        if (bounce + 1 >= maxBounces) {
            return outgoing;
        }

        // Next-event estimation through the shared `sampleNee`, its visibility
        // traced inline here (the queued shadow path is the primary's; the tail
        // spends its own).
        if (scene.lightCount > 0 || scene.env.selectProb > 0.0) {
            let uLight = sample_1d(pixel, sampleIndex, bounceDimension(bounce, NEE_LIGHT));
            let uPoint = sample_2d(pixel, sampleIndex, bounceDimension(bounce, NEE_POINT));
            let nee = sampleNee(scene, spawn, frame, closure, wo, throughput,
                                LIGHT_SAMPLING_MIS, uLight, uPoint);
            if (any(nee.unshadowed > 0.0)) {
                RayDesc ray;
                ray.Origin = nee.origin;
                ray.Direction = nee.direction;
                ray.TMin = 0.0;
                ray.TMax = nee.tmax;
                let vis = shadowVisibility(sceneTlas, scene, ray, nee.light, nee.primitive);
                outgoing += nee.unshadowed * vis;
            }
        }

        // Scatter and continue.
        let uLobe = sample_1d(pixel, sampleIndex, bounceDimension(bounce, BSDF_LOBE));
        let uDir = sample_2d(pixel, sampleIndex, bounceDimension(bounce, BSDF_DIR));
        let scatter = sampleBsdf(closure, wo, uLobe, uDir);
        let wiWorld = frame.toWorld(scatter.wi);
        var scattered = float3(0.0);
        if (scatterExitsSurface(scatter, wiWorld, spawn.normal)) {
            scattered = throughput * evalBsdf(closure, scatter.wi, wo)
                      * (abs(scatter.wi.z) / scatter.pdf);
        }
        var survival = max(scattered.r, max(scattered.g, scattered.b));
        if (survival <= 0.0) {
            return outgoing;
        }
        if (bounce >= ROULETTE_START) {
            survival = min(1.0, survival);
            if (sample_1d(pixel, sampleIndex, bounceDimension(bounce, RR)) >= survival) {
                return outgoing;
            }
            scattered /= survival;
        }

        // A transmission through a closed surface swaps the interior medium.
        if (scatter.lobe == LOBE_TRANSMISSION && material.thinWalled == 0) {
            medium = medium == hit.instance ? MEDIUM_NONE : hit.instance;
        }
        let transmitted = scatter.wi.z < 0.0;
        let nextOrigin = transmitted ? spawn.below : spawn.position;

        // Trace the continuation to the next vertex, opaque-nearest like the
        // BSDF candidate's own trace below.
        RayDesc ray;
        ray.Origin = nextOrigin;
        ray.Direction = wiWorld;
        ray.TMin = 0.0;
        ray.TMax = 1e30;
        RayQuery<RAY_FLAG_NONE> query;
        query.TraceRayInline(sceneTlas, RAY_FLAG_NONE, 0xFF, ray);
        while (query.Proceed()) {
            query.CommitNonOpaqueTriangleHit();
        }
        if (query.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
            // Escaped to the sky: the environment's emission, MIS-weighted.
            var weight = 1.0;
            if (scene.env.selectProb > 0.0) {
                let neePdf = scene.env.selectProb * scene.env.pdf(wiWorld);
                weight = powerHeuristic(scatter.pdf, neePdf);
            }
            outgoing += scattered * scene.env.radiance(wiWorld) * weight;
            return outgoing;
        }

        hit.instance = query.CommittedInstanceID();
        hit.primitive = query.CommittedPrimitiveIndex();
        hit.barycentrics = query.CommittedTriangleBarycentrics();
        dir = wiWorld;
        origin = nextOrigin;
        throughput = scattered;
        prevScatterPdf = scatter.pdf;
    }
    return outgoing;
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

    // -- one internalized BSDF draw (D-088), two path candidates (D-134) -----
    // Sample the closure once and trace the scattered ray. The direction yields
    // up to two candidates with *disjoint* path supports, each streamed under
    // its own acceptance draw:
    //
    //   - the **light** that direction reaches — an emitter triangle's emission
    //     or the environment — a length-2 sample competing with the M light
    //     candidates (denominator M·p_L + p_B);
    //   - the **continuation** past a surface hit — the indirect tail from x₂,
    //     a length-≥3 reconnection sample only BSDF sampling can produce, so
    //     its light pdf is 0 and the balance heuristic leaves only the BSDF
    //     density in the denominator (MIS weight 1). Emitters reflect too:
    //     their emission is the light candidate, their reflection this tail.
    //
    // This is the technique light sampling is worst at — a sharp lobe onto a
    // small or distant emitter, and every indirect path — so folding it into
    // the same reservoir (rather than a separate MIS'd term) is what makes the
    // reservoir the single owner of the primary hit's path integral.
    {
        let uLobe = sample_ranked_1d(pixel, ranked,
                                     restirBsdfDimension(RESTIR_BSDF_LOBE));
        let uDir = sample_ranked_2d(pixel, ranked,
                                    restirBsdfDimension(RESTIR_BSDF_DIR));
        let uAccept = sample_ranked_1d(pixel, ranked,
                                       restirBsdfDimension(RESTIR_BSDF_ACCEPT));
        let uAcceptTail = sample_ranked_1d(pixel, ranked,
                                           restirBsdfDimension(RESTIR_BSDF_TAIL_ACCEPT));

        let scatter = sampleBsdf(closure, wo, uLobe, uDir);
        let wiWorld = frame.toWorld(scatter.wi);

        // The light candidate.
        PathSample sample = PathSample();
        float3 toLight = float3(0.0);
        float3 radiance = float3(0.0);
        float lightPdf = 0.0; // p_L at the point the BSDF ray reached
        bool found = false;

        // The continuation candidate.
        PathSample tail = PathSample();
        float3 toX2 = float3(0.0);
        float3 tailRadiance = float3(0.0);
        bool foundTail = false;

        // Trace only a real, correctly-sided scatter — the shared guard every
        // scattering estimator applies (openpbr.slang), so a direction through
        // the geometric horizon never spawns a candidate.
        if (scatterExitsSurface(scatter, wiWorld, spawn.normal)) {
            let scatterOrigin = scatter.wi.z < 0.0 ? spawn.below : spawn.position;
            // A refraction into a closed surface puts the scattered segment —
            // and the tail it seeds — inside x₁'s own interior, exactly the
            // medium swap `shade_surface` would record into path state.
            let scatterMedium = scatter.lobe == LOBE_TRANSMISSION && material.thinWalled == 0
                ? hit.instance
                : MEDIUM_NONE;
            RayDesc ray;
            ray.Origin = scatterOrigin;
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
                let primitive = query.CommittedPrimitiveIndex();
                let bary = query.CommittedTriangleBarycentrics();
                GeometryRecord hitMesh = params.scene.geometry[instance];

                if (hitMesh.light != LIGHT_NONE) {
                    // An emitter: reconnect to the exact point hit. The TLAS
                    // barycentrics index the light record the same way resolve
                    // will (`geometry.light + primitive`), and `connectTriangle`
                    // re-derives the connection — one-sided, so a back-face hit
                    // reports pdf 0 and the candidate is null.
                    LightRecord record = restir.candidateLights[hitMesh.light + primitive];
                    var c = connectTriangle(record, bary, spawn.position);
                    if (c.pdf > 0.0) {
                        let emitter = params.scene.materials[instance];
                        if (emitter.emissionTexture != TEXTURE_NONE) {
                            c.radiance *= sampleColor(emitter.emissionTexture,
                                interpolateUv(hitMesh, primitive, bary));
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

                // The continuation tail from x₂, whatever the surface — the
                // hit is the reconnection vertex and the tail estimates
                // Lo(x₂→x₁), reflected only (an emitter x₂'s emission is the
                // light candidate above). F = f(x₁)·cos·Lo is this candidate's
                // own-domain integrand; `F` stays unset — resolve (and the
                // shift) re-form the RGB integrand in *their* domain from the
                // cached Lo, so no stored per-domain integrand can go stale
                // under reuse.
                Hit rc;
                rc.instance = instance;
                rc.primitive = primitive;
                rc.barycentrics = bary;
                // Reconstruct x₂ the direction-free way, so this direction is
                // bit-identical to the one resolve (and, at reuse, a
                // neighbour) recomputes from the stored vertex.
                let x2 = reconnectionPoint(hitMesh, rc);
                let toRc = normalize(x2 - spawn.position);
                let wiX2 = frame.toLocal(toRc);
                // A correctly-sided reconnection only (the tail's first
                // segment leaves through the face the closure sits on).
                if (wiX2.z != 0.0
                    && (wiX2.z > 0.0) == (dot(toRc, spawn.normal) > 0.0)) {
                    let lo = traceTail(pixel, params.sampleIndex, params.scene,
                                       params.maxBounces, rc, toRc, scatterOrigin,
                                       scatterMedium);
                    if (any(lo > 0.0)) {
                        tail.reserved = PATH_KIND_RECONNECTION;
                        // A live TLAS instance, not a light id — the shift
                        // re-shades x₂ from it (D-131).
                        tail.rcVertex = rc;
                        tail.rcVertexRadiance = lo; // Lo(x₂) — the shift reuses it
                        toX2 = toRc;
                        tailRadiance = lo;
                        foundTail = true;
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
        streamCandidate(reservoir, survivorTarget, closure, frame, spawn.normal, wo,
                        tail, toX2, tailRadiance, 0.0, foundTail, M, uAcceptTail);
    }

    reservoirFinalize(reservoir, survivorTarget);
    params.reservoirs[pixel] = storePathReservoir(reservoir);
}
