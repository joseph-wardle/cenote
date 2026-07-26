// Initial RIS over the primary hit: stream M light candidates plus the paths
// one BSDF-sampled walk produces through a per-pixel reservoir and keep one,
// weighted by an unshadowed target. The first of M3's four discrete reservoir
// stages (D-091) — it writes the reservoir the reuse stages will later fold
// history into, and that `restir_resolve` shades.
//
// Two candidate techniques feed the stream, so the reservoir owns *all* of the
// primary hit's non-delta path integral — direct and indirect (D-088, D-134):
//
//   - M **light** candidates from the triangle-only alias table plus the
//     environment coin flip (the light-sampling technique, count M).
//   - one internalized **BSDF** draw (the BSDF technique, count 1): sample the
//     closure and trace the scattered ray. The *light* that direction reaches
//     (an emitter triangle or the environment) is a length-2 candidate
//     competing with the M above; past a surface hit, the walk streams
//     **every lighting event it meets as its own path candidate** (D-137, the
//     Enhanced Alg 1 shape) — the NEE connection at each vertex, each
//     emission hit, the sky — each with its own suffix, MIS fold, and
//     acceptance draw. The first pair along the walk that passes the
//     variance criteria locks the reconnection vertex there (D-138): the
//     earlier bounces become a seed-replayable prefix and the samples stream
//     in the un-baked reconnection form the hybrid shift re-shades
//     (`reconnectionOutgoing`, restir_target.slang). Events before any lock
//     stream as **pure-replay** samples instead (D-139) — seed, shape, and
//     own-domain integrand — whose shift replays the whole path. The stored
//     winner is one concrete path either way.
//
// Combining the two techniques is the count-weighted balance heuristic. A
// length-2 candidate at light point x, whichever technique drew it, gets
// resampling weight
//
//     w = p̂(x) / (M·p_L(x) + p_B(x)),
//
// the balance-heuristic MIS weight m(x) = q_s(x)/Σ c_t·q_t(x) times p̂(x)/q_s(x)
// with the drawing technique's own density q_s cancelled — so it needs no
// branch on which technique drew x, and reduces to the single-technique
// (1/M)·p̂/p_L exactly when the BSDF cannot sample the direction (p_B = 0). Both
// source densities live in the solid-angle domain (the env and triangle pdfs
// lifted, the BSDF pdf native), so they are commensurable. A walked path
// candidate is reachable only through the BSDF draw (light sampling cannot
// produce a length-≥3 path), so its denominator is p_B alone — the suffix
// densities live inside its cached suffix, exactly where the shift needs them
// to stay. The target
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
// white — the path-state buffers are never read here. (The walk tracks its
// own medium from the BSDF draw's lobe onward.) Every trace is closest-hit
// inline traversal against the same TLAS the wavefront's other stages bind.

import bluenoise;
import color;
import lights;
import nee;
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
    uint maxBounces;       // path depth cap — the candidate walk's own bound (D-134)
};

[vk::push_constant]
Params params;

// -- The candidate-mean CV init (M6 step 6b-i, D-144) ------------------------
// Beside the scalar stream, every candidate adds its MIS-weighted contribution
// m·F/q into the CV lane — the balance-heuristic estimate of the whole
// reservoir-owned integral from the *same* draws, every candidate counted
// where the reservoir keeps one. No ÷M finalize is owed: the reference's mean
// divides M uniform weights, and here the m's normalize across techniques by
// construction. The lane's semantics are the stream's own: an NEE candidate
// enters unshadowed (its visibility ray is the one the pipeline defers —
// spatial repays the deficit at the sample whose ray it traces), an indirect
// candidate enters as its concretely-traced F — visibility lived. A non-finite
// term is dropped, never poisoning the lane (the reference's isfinite policy).
void cvAccumulate(inout float3 cvLane, float3 contribution) {
    if (all(isfinite(contribution))) {
        cvLane += contribution;
    }
}

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
void streamCandidate(inout Reservoir<PathSample> reservoir, inout float3 cvLane,
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
            let rgb = radiance * evalBsdf(closure, wiLight, wo) * abs(wiLight.z);
            target = luminance(rgb);
            let bsdfDensity = bsdfPdf(closure, wiLight, wo);
            let denom = float(M) * lightPdf + bsdfDensity;
            weight = (target > 0.0 && denom > 0.0) ? target / denom : 0.0;
            if (weight > 0.0) {
                // m·F/q with the technique density cancelled — the same
                // denominator the scalar weight divides (see the header MIS).
                cvAccumulate(cvLane, rgb / denom);
            }
        }
    }
    if (reservoirUpdate(reservoir, sample, weight, uAccept)) {
        survivorTarget = target;
    }
}

// The reconnection context: the lens the walk's streamed samples are stored
// through once a pair qualifies — which vertex re-shades, and the
// walk-constant halves of every streamed weight. Unlocked on entry
// (`shiftable` false, the other fields dead); locked exactly once, at the
// first pair along the walk — (x₁, x₂) included — that passes the variance
// criteria (D-138/D-139). Before the lock, events stream as pure-replay
// samples that never read it.
struct RcContext {
    Hit rc;               // x_k — the vertex every re-shading domain re-forms
    uint k;               // its index along the path (x₂ = 2)
    bool inside;          // x_k shades from inside its own instance
    bool shiftable;       // the lock: true once a pair passed the criteria
    uint seed;            // the per-path replay seed (D-138)
    Closure closure;      // x_k through the re-derived segment direction
    Frame frame;
    float3 wo;
    float3 prefix;        // Π f·cos/pdf · transmittance over the prefix, roulette-free
    float survival;       // Π roulette survivals up to the lock — the weight fold (D-138)
    float3 fkm1cos;       // f·|cosθ| of the reconnection segment at x_{k−1}
    float reconPdf;       // its BSDF density — the balance-heuristic denominator
    float cachedJacobian; // the source Jacobian half |cosθₖ|/d² (D-138)
};

// Stream one walked lighting event as a path candidate (D-137). `wiWorld` is
// the suffix direction leaving x_k — the NEE direction for the x_k connection,
// the x_k scatter for everything deeper — packed into the sample and *decoded
// back* before the target evaluates, so this pixel weighs exactly the sample
// every re-shading domain will see (the stored direction, not the drawn one).
// `suffix` is the cached incident-side value (`rcVertexRadiance` docs). The
// candidate weight is the target over the technique's density: the
// reconnection draw's BSDF pdf times the prefix's folded roulette survivals —
// only the BSDF technique reaches a length-≥3 path, and the roulette fold
// (D-138) is what lets the shift's replay never roll. A dead event (target 0)
// streams nothing: the walk is one technique draw whatever it meets, so
// unlike the counted light mixture above there is no draw-count to keep
// honest, and confidence tracks the events actually offered.
void streamTailEvent(inout Reservoir<PathSample> reservoir, inout float3 cvLane,
                     inout float survivorTarget, RcContext ctx, float3 wiWorld,
                     float3 suffix, float neeLightPdf, uint terminal, float uAccept) {
    PathSample sample = PathSample();
    sample.rcVertex = ctx.rc;
    sample.rcVertexWi = packUnitDirection(wiWorld);
    sample.rcVertexRadiance = suffix;
    sample.neeLightPdf = neeLightPdf;
    sample.initRandomSeed = ctx.seed;
    sample.cachedJacobian = ctx.cachedJacobian;
    sample.reserved = reconnectionFlags(terminal, ctx.inside, ctx.shiftable, ctx.k);
    let wi = ctx.frame.toLocal(unpackUnitDirection(sample.rcVertexWi));
    let lo = reconnectionOutgoing(ctx.closure, wi, ctx.wo, suffix, neeLightPdf, terminal);
    // The own-domain integrand, stored on the sample (D-138). The association
    // mirrors `shiftReconnection`'s exactly, so a same-pixel shift reproduces
    // it bit for bit — the step-3b gate.
    sample.F = ctx.prefix * ctx.fkm1cos * lo;
    let target = luminance(sample.F);
    if (!(target > 0.0)) {
        return;
    }
    // F/q — the walk is one technique draw, so m = 1 and the density is the
    // candidate weight's own denominator (reconnection pdf × roulette fold).
    cvAccumulate(cvLane, sample.F / (ctx.reconPdf * ctx.survival));
    if (reservoirUpdate(reservoir, sample, target / (ctx.reconPdf * ctx.survival),
                        uAccept)) {
        survivorTarget = target;
    }
}

// Stream one pre-lock lighting event as a pure-replay candidate (D-139): the
// sample stores only its seed, its shape (the terminal kind and the bounce it
// fired at), and the own-domain integrand F — a shift replays every bounce at
// the destination and re-draws the terminal from the same dims
// (`shiftReplay`, restir_target.slang). F carries the roulette-free chain
// with every pdf divided in, so the candidate weight divides only the
// survivals the walk actually rolled before the event — the D-138 fold, which
// is what lets the replay never roll.
void streamReplayEvent(inout Reservoir<PathSample> reservoir, inout float3 cvLane,
                       inout float survivorTarget, uint seed, float3 F,
                       uint terminal, uint bounce, float survival, float uAccept) {
    let target = luminance(F);
    if (!(target > 0.0)) {
        return;
    }
    // F/q, as in the tail event: F carries every pdf divided in, so only the
    // roulette fold remains to divide.
    cvAccumulate(cvLane, F / survival);
    PathSample sample = PathSample();
    sample.F = F;
    sample.initRandomSeed = seed;
    sample.reserved = replayFlags(terminal, bounce);
    if (reservoirUpdate(reservoir, sample, target / survival, uAccept)) {
        survivorTarget = target;
    }
}

// The walk's previous-vertex shading, held for the pair search: a lock
// re-derives its reconnection segment off this surface through the shared
// `reconnectionEnd`, exactly as the shift's replay will. x₁'s at entry, then
// each walked vertex's in turn.
struct WalkVertex {
    float3 position;
    float3 normal;  // geometric
    Closure closure;
    Frame frame;
    float3 wo;
    bool rough;     // its half of the pair criteria (`reconnectionRough`)
};

// Walk the path onward from x₂ and stream every lighting event it meets — the
// per-vertex estimator `shade_surface` runs (`sampleNee` + BSDF continuation
// under per-vertex MIS, the shared pieces in nee.slang), each event landing in
// the reservoir as its own candidate (D-137). The walk itself runs on its
// *drawn* directions and seeded draws — exactly the chain a shift replays
// (D-138) — while the reconnection context is a separate lens, re-derived
// direction-free at its vertex (`reconnectionEnd`), so what the samples store
// is what every re-shading domain re-forms.
//
// The context starts UNLOCKED and locks exactly once, at the first pair along
// the walk — (x₁, x₂) included — that passes the variance criteria: from the
// lock on, events stream as reconnection samples whose prefix (the bounces
// before x_{k−1}) replays from the seed. Before any lock, events stream as
// pure-replay samples (D-139): seed, shape, and own-domain integrand only —
// their shift replays the whole path and re-draws the terminal from the same
// dims. Every streamed event therefore shifts; nothing waits on a later rung.
//
// Roulette: the walk rolls exactly as the path tracer (survival read off the
// true running throughput), but the division lands by where the roll sits
// relative to the lock. After it, the division goes into the baked suffix — a
// value, never replayed; before it, into the pending survival that a future
// lock — or a replay event's candidate weight — folds into its denominator.
// Neither replayed chain carries a 1/s, so no shift's replay ever rolls —
// §6's RR-replay trap, closed by construction (D-138).
//
// x₂'s own emission is deliberately excluded: it is the BSDF draw's *light*
// candidate — a length-2 sample competing with next-event sampling — and
// streaming it here, where the weight sees no light-sampling density, would
// double-count it.
//
// Traversal treats every crossing as opaque (nearest hit wins) — stochastic
// opacity in the reused tail is deferred with volumes (deferrals.md).
//
// `hit` is x₂ — already traced by the kernel, so the loop's first iteration
// (bounce 1) skips its own trace; `drawnDir` is the *drawn* x₁→x₂ direction
// the walk (and replay) shades by; `origin` the traced ray's start and
// `medium` the interior that segment crossed. `rhs` is the footprint
// criterion's per-pixel threshold — the (c/100)-scaled primary spread every
// pair's dual footprints measure against (restir_target); `x1FOverPdf` the
// drawn scatter's f·cos/pdf (the prefix chain's first factor, in replay's
// exact form); `x1` the previous-vertex shading the bounce-1 pair search
// reconnects off.
void walkTail(inout Reservoir<PathSample> reservoir, inout float3 cvLane,
              inout float survivorTarget,
              uint pixel, uint ranked, uint seed, SceneTable* scene,
              uint maxBounces, Hit hit, float3 drawnDir, float3 origin,
              uint medium, float rhs, float3 x1FOverPdf, WalkVertex x1) {
    // The context, unlocked: `shiftable` is the lock flag, and the re-shading
    // fields stay dead until a pair search assigns them (seeded from x₁ only
    // so every field is defined).
    RcContext ctx;
    ctx.rc = hit;
    ctx.k = 0;
    ctx.inside = false;
    ctx.shiftable = false;
    ctx.seed = seed;
    ctx.closure = x1.closure;
    ctx.frame = x1.frame;
    ctx.wo = x1.wo;
    ctx.prefix = float3(1.0);
    ctx.survival = 1.0;
    ctx.fkm1cos = float3(0.0);
    ctx.reconPdf = 0.0;
    ctx.cachedJacobian = 0.0;

    // -- The accumulators ----------------------------------------------------
    // `prefixHere`: the roulette-free f·cos/pdf · transmittance chain up to
    // the walk's current vertex, in exactly replay's multiplication order —
    // the pre-lock events' integrand half, and the source of a lock's prefix;
    // `prefixBeforeScatter` trails it by one scatter, since a lock's prefix
    // excludes the reconnection draw itself. `suffix` is the locked context's
    // beyond-f_k value, live from the lock on. `walkThroughput` is the true
    // running throughput — roulette divisions included — that survival and
    // termination read, as `shade_surface`'s would. `pendingSurvival` is the
    // roulette fold the pre-lock events' weights divide and a lock inherits.
    var suffix = float3(1.0);
    var prefixHere = x1FOverPdf;
    var prefixBeforeScatter = float3(1.0);
    var walkThroughput = x1FOverPdf;
    var pendingSurvival = 1.0;

    var walkMedium = medium;
    var walkOrigin = origin;
    var walkDir = drawnDir;
    var prevScatterPdf = 0.0;    // set at every scatter; unread at bounce 1
    var scatterWasRcDraw = false; // the last scatter was the lock's own draw
    var rcWiWorld = drawnDir;    // ωₖ — re-pointed at the lock's own scatter
    var wiEval = drawnDir;       // its decoded form, for the adjacent-to-lock MIS evals
    var prev = x1;               // the previous vertex, for the pair search
    Hit vHit = hit;              // bounce 1's vertex is the kernel's trace

    for (uint bounce = 1; bounce < maxBounces; bounce++) {
        if (bounce > 1) {
            // Trace the continuation to the next vertex, opaque-nearest like
            // the BSDF candidate's own trace in the kernel below.
            RayDesc ray;
            ray.Origin = walkOrigin;
            ray.Direction = walkDir;
            ray.TMin = 0.0;
            ray.TMax = 1e30;
            RayQuery<RAY_FLAG_NONE> query;
            query.TraceRayInline(sceneTlas, RAY_FLAG_NONE, 0xFF, ray);
            while (query.Proceed()) {
                query.CommitNonOpaqueTriangleHit();
            }
            if (query.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
                // Escaped to the sky. Unlocked, the event is a whole-path
                // replay whose terminal segment re-finds its own light.
                // Locked and off the lock's own scatter, the MIS involves the
                // re-shaded pdf_k (SCATTER, light pdf stored); deeper, both
                // densities are fixed and the weight bakes.
                let uAccept = sample_ranked_1d(pixel, ranked,
                    restirTailDimension(bounce, RESTIR_TAIL_SCATTER_ACCEPT));
                if (!ctx.shiftable) {
                    streamReplayEvent(reservoir, cvLane, survivorTarget, seed,
                                      prefixHere * scene.env.radiance(walkDir)
                                          * envEscapeMisWeight(scene, walkDir,
                                                               prevScatterPdf),
                                      RC_TERMINAL_SCATTER, bounce, pendingSurvival,
                                      uAccept);
                } else if (scatterWasRcDraw) {
                    let neePdf = scene.env.selectProb > 0.0
                        ? scene.env.selectProb * scene.env.pdf(wiEval)
                        : 0.0;
                    streamTailEvent(reservoir, cvLane, survivorTarget, ctx, rcWiWorld,
                                    suffix * scene.env.radiance(wiEval), neePdf,
                                    RC_TERMINAL_SCATTER, uAccept);
                } else {
                    streamTailEvent(reservoir, cvLane, survivorTarget, ctx, rcWiWorld,
                                    suffix * scene.env.radiance(walkDir)
                                        * envEscapeMisWeight(scene, walkDir,
                                                             prevScatterPdf),
                                    0.0, RC_TERMINAL_BAKED, uAccept);
                }
                return;
            }
            vHit.instance = query.CommittedInstanceID();
            vHit.primitive = query.CommittedPrimitiveIndex();
            vHit.barycentrics = query.CommittedTriangleBarycentrics();
        }

        GeometryRecord vMesh = scene.geometry[vHit.instance];
        let vUv = interpolateUv(vMesh, vHit.primitive, vHit.barycentrics);
        let vMaterial = resolveMaterial(scene.materials[vHit.instance], vUv);
        let vSpawn = spawnPoint(vMesh, vHit, walkDir);

        // Beer–Lambert over the arriving segment, if it crossed an interior —
        // into the prefix chain always (replay recomputes it), and into the
        // suffix once a lock consumes it.
        if (walkMedium != MEDIUM_NONE) {
            let t = interiorTransmittance(scene, walkMedium, walkOrigin, vSpawn.position);
            if (ctx.shiftable) {
                suffix *= t;
            }
            prefixHere *= t;
            walkThroughput *= t;
        }

        // Emission: the BSDF strategy reaching a light — except at x₂, whose
        // emission is the BSDF draw's own light candidate (see the header).
        // Unlocked, the event is a whole-path replay (its own reconnection
        // vertex would be the emitter itself — exactly the shape D-139
        // covers). Locked and adjacent to the lock's scatter, the MIS
        // involves the re-shaded pdf_k (SCATTER, light pdf stored); deeper it
        // bakes against the previous vertex's fixed scatter pdf. Streamed
        // before any lock at this vertex.
        if (bounce > 1 && vMesh.light != LIGHT_NONE && vSpawn.frontFace) {
            let neePdf = emitterHitNeePdf(scene, vMesh, vHit.primitive, vSpawn,
                                          walkOrigin, walkDir);
            let uAccept = sample_ranked_1d(pixel, ranked,
                restirTailDimension(bounce, RESTIR_TAIL_SCATTER_ACCEPT));
            if (!ctx.shiftable) {
                streamReplayEvent(reservoir, cvLane, survivorTarget, seed,
                                  prefixHere * vMaterial.emission
                                      * powerHeuristic(prevScatterPdf, neePdf),
                                  RC_TERMINAL_SCATTER, bounce, pendingSurvival,
                                  uAccept);
            } else if (scatterWasRcDraw) {
                streamTailEvent(reservoir, cvLane, survivorTarget, ctx, rcWiWorld,
                                suffix * vMaterial.emission, neePdf,
                                RC_TERMINAL_SCATTER, uAccept);
            } else {
                streamTailEvent(reservoir, cvLane, survivorTarget, ctx, rcWiWorld,
                                suffix * vMaterial.emission
                                    * powerHeuristic(prevScatterPdf, neePdf),
                                0.0, RC_TERMINAL_BAKED, uAccept);
            }
        }

        // The depth cap: the path ends here (NEE from this vertex would outrun
        // any BSDF-traced path length, so the two strategies stay matched).
        // Before the pair search and the vertex's shading resolve, which only
        // the events past this return could use — a lock at the terminal
        // vertex could never stream a sample.
        if (bounce + 1 >= maxBounces) {
            return;
        }

        // -- The pair search (D-138/D-140): still unlocked, so test this ------
        // arrival against the previous vertex — x₁ itself on the first
        // iteration, so the k = 2 shape is just the general case. The x_{k−1}
        // roughness guard, an exterior segment, and the *forward* footprint —
        // Eq 5's first term, whose two factors the re-derived end already
        // computed (`reconPdf` the lobe's density, `partial` the segment's
        // G), so the test costs nothing new. x_k itself needs no roughness:
        // a lock here demotes below if its own scatter fails the inverse
        // term. Evaluated on the re-derived end so a lock is
        // shiftable-by-construction.
        var justLocked = false;
        var inverseG = 0.0; // G(x_k→x_{k−1}) = |⟨n_{k−1}, x̂⟩|/d², stashed at lock
        if (!ctx.shiftable && prev.rough && walkMedium == MEDIUM_NONE) {
            let e = reconnectionEnd(scene, vHit, walkMedium == vHit.instance,
                                    prev.position, prev.closure, prev.frame,
                                    prev.wo, prev.normal);
            // Paper form: 1/(p·G) ≥ rhs. Multiply-form, so a grazing
            // primary's rhs = ∞ refuses cleanly instead of dividing.
            if (e.valid && rhs * (e.reconPdf * e.partial) <= 1.0) {
                ctx.rc = vHit;
                ctx.k = bounce + 1;
                ctx.inside = walkMedium == vHit.instance;
                ctx.shiftable = true;
                ctx.closure = e.closure;
                ctx.frame = e.frame;
                ctx.wo = e.wo;
                ctx.prefix = prefixBeforeScatter;
                ctx.survival = pendingSurvival;
                ctx.fkm1cos = e.fkm1cos;
                ctx.reconPdf = e.reconPdf;
                ctx.cachedJacobian = e.partial;
                suffix = float3(1.0);
                inverseG = abs(dot(prev.normal, e.toRc)) / e.dist2;
                justLocked = true;
            }
        }

        var vShading = vSpawn.shading;
        if (vMaterial.normalTexture != TEXTURE_NONE) {
            vShading = mappedShadingNormal(vMesh, vHit, vUv, vMaterial.normalTexture, vSpawn);
        }
        let vFrame = Frame(vShading);
        let vWo = vFrame.toLocal(-walkDir);
        let vClosure = resolveClosure(scene.bsdfTables, vMaterial, vWo,
                                      walkMedium == vHit.instance);

        // Next-event estimation through the shared `sampleNee`. Unlocked, the
        // event is a whole-path replay: the destination re-draws this exact
        // connection from the same dims, so the streamed value is simply the
        // chain-scaled `unshadowed` — MIS, delta-ness and all — times the
        // traced visibility. At the lock's own vertex it is the
        // RC_TERMINAL_NEE event — raw halves stored, MIS re-evaluated per
        // domain; deeper its MIS is local to this vertex and bakes.
        // Visibility traced inline (the queued shadow path is the primary's).
        if (scene.lightCount > 0 || scene.env.selectProb > 0.0) {
            let uLight = replay_1d(seed, bounceDimension(bounce, NEE_LIGHT));
            let uPoint = replay_2d(seed, bounceDimension(bounce, NEE_POINT));
            let nee = sampleNee(scene, vSpawn, vFrame, vClosure, vWo,
                                ctx.shiftable ? suffix : prefixHere,
                                LIGHT_SAMPLING_MIS, uLight, uPoint);
            if (any(nee.unshadowed > 0.0)) {
                RayDesc shadowRay;
                shadowRay.Origin = nee.origin;
                shadowRay.Direction = nee.direction;
                shadowRay.TMin = 0.0;
                shadowRay.TMax = nee.tmax;
                let vis = shadowVisibility(sceneTlas, scene, shadowRay, nee.light,
                                           nee.primitive);
                if (vis > 0.0) {
                    let uAccept = sample_ranked_1d(pixel, ranked,
                        restirTailDimension(bounce, RESTIR_TAIL_NEE_ACCEPT));
                    if (!ctx.shiftable) {
                        streamReplayEvent(reservoir, cvLane, survivorTarget, seed,
                                          nee.unshadowed * vis, RC_TERMINAL_NEE,
                                          bounce, pendingSurvival, uAccept);
                    } else if (justLocked) {
                        streamTailEvent(reservoir, cvLane, survivorTarget, ctx, nee.direction,
                                        suffix * nee.radiance * (vis / nee.pdf), nee.pdf,
                                        nee.delta ? RC_TERMINAL_BAKED : RC_TERMINAL_NEE,
                                        uAccept);
                    } else {
                        streamTailEvent(reservoir, cvLane, survivorTarget, ctx, rcWiWorld,
                                        nee.unshadowed * vis, 0.0, RC_TERMINAL_BAKED,
                                        uAccept);
                    }
                }
            }
        }

        // Scatter and continue.
        let vLobe = replay_1d(seed, bounceDimension(bounce, BSDF_LOBE));
        let vDir = replay_2d(seed, bounceDimension(bounce, BSDF_DIR));
        let vScatter = sampleBsdf(vClosure, vWo, vLobe, vDir);
        let vWiWorld = vFrame.toWorld(vScatter.wi);
        if (!scatterExitsSurface(vScatter, vWiWorld, vSpawn.normal)) {
            return;
        }
        let fOverPdf = evalBsdf(vClosure, vScatter.wi, vWo)
                     * (abs(vScatter.wi.z) / vScatter.pdf);
        prefixBeforeScatter = prefixHere;
        prefixHere *= fOverPdf;
        var tentative = walkThroughput * fOverPdf;
        var survival = max(tentative.r, max(tentative.g, tentative.b));
        if (!(survival > 0.0)) {
            return;
        }
        // The inverse footprint (Eq 5's second term, D-140), testable only
        // now that the scatter out of x_k exists. Skipped for a diffuse-
        // dominated x_k, whose continuation pdf ignores the arrival — exact
        // for the at-lock NEE event above, whose light draw does the same
        // (footnote 6); it streamed with the lock and keeps its verdict. On
        // failure the fresh lock demotes and the search re-arms at later
        // pairs: the ctx fields are dead while unlocked, and the roulette
        // division below already routes to `pendingSurvival` once
        // `ctx.shiftable` drops, so the bookkeeping self-heals.
        if (justLocked && sharpCapable(vMaterial)
            && rhs * (vScatter.pdf * inverseG) > 1.0) {
            ctx.shiftable = false;
            justLocked = false;
        }
        if (justLocked) {
            // A fresh lock's own scatter is its reconnection draw: the suffix
            // starts at 1/pdf beyond f_k, and ωₖ re-points along it.
            suffix = float3(1.0 / vScatter.pdf);
            rcWiWorld = vWiWorld;
            wiEval = unpackUnitDirection(packUnitDirection(vWiWorld));
            scatterWasRcDraw = true;
        } else if (ctx.shiftable) {
            suffix *= fOverPdf;
            scatterWasRcDraw = false;
        }
        if (bounce >= ROULETTE_START) {
            survival = min(1.0, survival);
            if (replay_1d(seed, bounceDimension(bounce, RR)) >= survival) {
                return;
            }
            tentative /= survival;
            // The division lands by where the roll sits relative to the lock
            // (see the header): into the baked suffix after it, into the
            // pending fold — future locks' and replay events' weights — before.
            if (ctx.shiftable) {
                suffix /= survival;
            } else {
                pendingSurvival *= survival;
            }
        }
        walkThroughput = tentative;

        // A transmission through a closed surface swaps the interior medium.
        if (vScatter.lobe == LOBE_TRANSMISSION && vMaterial.thinWalled == 0) {
            walkMedium = walkMedium == vHit.instance ? MEDIUM_NONE : vHit.instance;
        }
        walkOrigin = vScatter.wi.z < 0.0 ? vSpawn.below : vSpawn.position;
        walkDir = vWiWorld;
        prevScatterPdf = vScatter.pdf;
        prev.position = vSpawn.position;
        prev.normal = vSpawn.normal;
        prev.closure = vClosure;
        prev.frame = vFrame;
        prev.wo = vWo;
        prev.rough = reconnectionRough(vMaterial);
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
    // The footprint criterion's per-pixel threshold (D-140, Enhanced Eq 5):
    // a (c/100) fraction of the primary ray's spread ‖x₀−x₁‖²·4π/|⟨n₁,x̂₁x₀⟩|,
    // the area its full sphere of directions covers at the hit — against
    // which every candidate pair's dual footprints measure in `walkTail`.
    // The *geometric* normal, matching the G-term cosines the footprints
    // themselves use. A grazing primary pushes it to ∞ and simply never
    // locks — the criterion's own behavior, not a special case.
    let toCamera = params.paths.origin[slot].xyz - spawn.position;
    let rhs = FOOTPRINT_C / 100.0 * dot(toCamera, toCamera) * 4.0 * PI
            / abs(dot(spawn.normal, normalize(toCamera)));

    let restir = params.restir;
    let envProb = restir.envSelectProb;
    let M = params.candidates;

    Reservoir<PathSample> reservoir = reservoirInit<PathSample>();
    // p̂ at the held survivor, tracked as the stream runs so finalize needs no
    // re-evaluation (the survivor's target is whatever won the last acceptance).
    float survivorTarget = 0.0;
    // The candidate-mean CV init (6b-i): Σ m·F/q over the whole stream — see
    // `cvAccumulate`. Every stream site below feeds it beside the reservoir.
    float3 cvLane = float3(0.0);

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

        streamCandidate(reservoir, cvLane, survivorTarget, closure, frame, spawn.normal, wo,
                        sample, toLight, radiance, lightPdf, lightPdf > 0.0, M, uAccept);
    }

    // -- one internalized BSDF draw (D-088): the light it reaches, then the
    //    walk (D-134, D-137) ---------------------------------------------------
    // Sample the closure once and trace the scattered ray. The *light* that
    // direction reaches — an emitter triangle's emission or the environment —
    // is a length-2 sample competing with the M light candidates (denominator
    // M·p_L + p_B). Past a surface hit, `walkTail` streams every deeper
    // lighting event as its own reconnection candidate (denominator p_B alone
    // — only BSDF sampling produces a length-≥3 path). Emitters reflect too:
    // their emission is the light candidate, their reflection the walk's.
    //
    // This is the technique light sampling is worst at — a sharp lobe onto a
    // small or distant emitter, and every indirect path — so folding it into
    // the same reservoir (rather than a separate MIS'd term) is what makes the
    // reservoir the single owner of the primary hit's path integral.
    {
        // The scatter's uniforms come from the per-path replay stream (D-138):
        // this draw is the first bounce a k > 2 shift replays at a neighbour,
        // so generation and replay must be the same pure function of the
        // stored seed. The acceptance is reservoir-local and stays ranked.
        let seed = pathReplaySeed(pixel, params.sampleIndex);
        let uLobe = replay_1d(seed, bounceDimension(0, BSDF_LOBE));
        let uDir = replay_2d(seed, bounceDimension(0, BSDF_DIR));
        let uAccept = sample_ranked_1d(pixel, ranked,
                                       restirBsdfDimension(RESTIR_BSDF_ACCEPT));

        let scatter = sampleBsdf(closure, wo, uLobe, uDir);
        let wiWorld = frame.toWorld(scatter.wi);

        // The light candidate.
        PathSample sample = PathSample();
        float3 toLight = float3(0.0);
        float3 radiance = float3(0.0);
        float lightPdf = 0.0; // p_L at the point the BSDF ray reached
        bool found = false;

        // Trace only a real, correctly-sided scatter — the shared guard every
        // scattering estimator applies (openpbr.slang), so a direction through
        // the geometric horizon never spawns a candidate.
        if (scatterExitsSurface(scatter, wiWorld, spawn.normal)) {
            let scatterOrigin = scatter.wi.z < 0.0 ? spawn.below : spawn.position;
            // A refraction into a closed surface puts the scattered segment —
            // and the walk it seeds — inside x₁'s own interior, exactly the
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

                // The walk from x₂, whatever the surface — every lighting
                // event past it streams on its own (an emitter x₂'s emission
                // is the light candidate above; its reflection rides the
                // walk), as a reconnection sample once a pair locks and as a
                // whole-path replay before (D-138/D-139). The walk shades by
                // the *drawn* direction — the chain replay reproduces — and
                // re-derives its reconnection ends direction-free inside
                // (`reconnectionEnd`); x₁'s shading and the prefix chain's
                // first factor ride along for the bounce-1 pair search.
                Hit rc;
                rc.instance = instance;
                rc.primitive = primitive;
                rc.barycentrics = bary;
                let fOverPdf = evalBsdf(closure, scatter.wi, wo)
                             * (abs(scatter.wi.z) / scatter.pdf);
                WalkVertex x1;
                x1.position = spawn.position;
                x1.normal = spawn.normal;
                x1.closure = closure;
                x1.frame = frame;
                x1.wo = wo;
                x1.rough = reconnectionRough(material);
                walkTail(reservoir, cvLane, survivorTarget, pixel, ranked, seed,
                         params.scene, params.maxBounces, rc, wiWorld,
                         scatterOrigin, scatterMedium, rhs, fOverPdf, x1);
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

        streamCandidate(reservoir, cvLane, survivorTarget, closure, frame, spawn.normal, wo,
                        sample, toLight, radiance, lightPdf, found, M, uAccept);
    }

    reservoirFinalize(reservoir, survivorTarget);
    params.reservoirs[pixel] = storePathReservoir(reservoir, cvLane);
}
