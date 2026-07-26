// Spatial reuse: fold each of k screen-space neighbours' reservoirs into the
// pixel's own under defensive pairwise MIS, so one pixel's hard-won light
// sample lightens its neighbourhood. The third of M3's four discrete reservoir
// stages (D-091), and the first that reads *other* pixels — every earlier stage
// touched only its own pixel, which is why this one carries the step's whole
// correctness weight (D-093).
//
// -- The estimator -----------------------------------------------------------
// The pixel's own reservoir (from restir_candidates) is the *canonical*
// candidate; the k neighbours are non-canonical. All k+1 are combined into a
// fresh reservoir at the canonical (this pixel's) surface — the DI shift is the
// identity because the reservoir stores a light *surface point*, so the GRIS
// Jacobian is 1 (D-087) and the shift is just "re-evaluate the target here".
//
// MIS is the *defensive* confidence-weighted pairwise heuristic (Bitterli 2022
// thesis §9.1 / the 2023 course §7.1.3, Eq 7.8). Each neighbour is compared only
// to the canonical — O(k), not the O(k²) generalized balance heuristic. With
// c_c the canonical confidence, c_i a neighbour's, c_tot their sum, V the count
// of *accepted* neighbours, and t_c = c_c/c_tot the defensive fraction:
//
//   b_i = c_i·p̂_i(Y_i) / (c_i·p̂_i(Y_i) + c_c·p̂_c(Y_i))       neighbour i at Y_i
//   m_i = (1 − t_c)·(1/V)·b_i
//   m_c = t_c + (1 − t_c)·(1/V)·Σ_i c_c·p̂_c(Y_c)/(c_i·p̂_i(Y_c) + c_c·p̂_c(Y_c))
//
// which provably sum to 1 (the unbiasedness condition) for any positive targets:
// (1−t_c)(1/V)[Σb + Σ(1−b)] + t_c = (1−t_c) + t_c = 1. The *defensive* t_c floor
// — RTXDI ships the plain form (Eq 7.7) without it — matters here because the
// canonical is the one technique that covers the full support of f (D-088), and
// t_c > 0 keeps it from ever being zero-weighted by a neighbour whose target
// poorly approximates this surface's. That is what makes the converge-to-
// reference checkpoint (D-090) reachable, so it is required, not gilding.
//
// -- Visibility-aware target, folded into W ----------------------------------
// p̂_c — the canonical (receiving) target — is **visibility-aware**: p̂_c(Y) =
// U_c(Y)·V(Y), the unshadowed luminance times a shadow ray traced from *this*
// surface to Y's light point (D-093, the course's recommended Tip 4.3 variant).
// p̂_i — a neighbour's own-domain target, used only inside the MIS ratios — is
// left unshadowed, so exactly one ray is spent per candidate: k neighbours + the
// canonical = k+1 inline rays through the shared `shadowVisibility` helper.
// Re-tracing at *this* pixel (never trusting the neighbour's stored occlusion)
// is what keeps reuse unbiased across a shadow boundary, the classic ReSTIR bias
// source (D-087).
//
// The survivor's contribution weight is finalized by dividing weightSum by the
// *unshadowed* U_c(survivor), not p̂_c. Since the survivor was resampled ∝ p̂_c =
// U_c·V, dividing by U_c folds its own visibility into W (W = weightSum/U_c is
// exactly weightSum/p̂_c · V). restir_resolve then shades the survivor with the
// plain unshadowed L·f·cos and no shadow ray of its own — the ray was already
// spent here. Equivalent to the textbook "shade with a resolve-time shadow ray"
// but one ray cheaper (D-093).
//
// -- Neighbour geometry, and the range guard ---------------------------------
// A neighbour's own target p̂_i needs its surface (position, frame, closure),
// reconstructed from the path pool at slot `neighbourPixel − base`, where the
// range's base is recovered per-thread as `myPixel − mySlot` (raygen wrote
// pixel = base + slot). A target larger than the pool is walked in ranges, and
// this stage runs per range; a neighbour in a *different* range has no live path
// state, so it is dropped — verified by re-reading paths.pixel at the slot and
// requiring it to match. Dropped neighbours simply leave the accepted count V
// (the canonical absorbs their share); the only cost is slightly less reuse in a
// one-pixel band at a range seam, never bias. A neighbour that is a miss, or was
// never reached, reads confidence 0 from the wave-cleared reservoir buffer and
// is likewise dropped.
//
// Runs at bounce 0 only, dispatched from the hit queue it reads but never
// consumes. Reads the committed candidate reservoirs (`reservoirsIn`), writes
// the merged survivors to a separate buffer (`reservoirsOut`) — the ping-pong a
// barrier orders, so no thread reads a half-written neighbour and determinism
// holds (M3 plan §2).

import bluenoise;
import color;
import openpbr;
import pathstate;
import reservoir;
import reservoir_path;
import restir_mis;
import restir_scene;
import restir_target;
import rng;
import scene;
import surface;

// The scene TLAS, shared with `intersect`/`trace_shadow`: the k+1 visibility
// rays trace it through the shared `shadowVisibility` helper.
[vk::binding(0, 0)]
RaytracingAccelerationStructure sceneTlas;

// The most neighbours a spatial pass will gather — the fixed bound on the
// per-thread accepted-neighbour store below. The host caps k at this
// (wavefront.rs); k default is ~5 (D-088).
static const uint MAX_SPATIAL_NEIGHBOURS = 8;

struct Params {
    Paths paths;
    Queue<uint> hits;            // primary hits, dispatched indirectly; not consumed
    SceneTable* scene;
    RestirScene* restir;
    StoredPathReservoir* reservoirsIn;  // candidate stage's output — self + neighbours, read
    StoredPathReservoir* reservoirsOut; // this stage's output — the merged survivor, written
    uint sampleIndex;
    uint width;
    uint height;
    uint capacity;               // path-pool capacity — a neighbour slot must be < this
    uint neighbours;             // k
    float radius;                // gather radius, pixels
    float normalThreshold;       // reuse gate: dot(n_c, n_i) must exceed this
    float depthThreshold;        // reuse gate: relative camera-depth difference cap
};

[vk::push_constant]
Params params;

// The reconstructed `Surface`, the light-sample `connectSampleAt`, the
// unshadowed target `targetUnshadowed`, and the cross-domain shift-and-target
// block (`shiftIntoDomain`/`targetInDomain`) this stage shares with the
// temporal combine live in `restir_target.slang` (imported above) — one copy,
// so the two reuse stages can never drift into a silent bias. Only the
// visibility-aware NEE target below stays here: shadowing the NEE arm is this
// stage's visibility policy (temporal's is unshadowed, D-094), not shared
// machinery.

// The chroma direction F/luminance(F) of an unshadowed RGB contribution — the
// §6.3 vector-weight factor (M6 step 6a, m6-plan §4d decision 3): multiplied
// by a scalar target already in hand (visibility, Jacobian folded), it
// re-forms that target's RGB — chroma·p̂ᶜᵛⁱˢ = F·V·J — which is how the CV's
// F-estimates below shade every candidate in colour without a second
// evaluation. luminance(F) = 0 ⇔ F = 0, so the zero guard loses nothing.
float3 chromaDirection(float3 f, float lum) {
    return lum > 0.0 ? f / lum : float3(0.0);
}

// -- The spatial control variate (M6 step 6b-i, D-144) -----------------------
// ReSTCV's per-neighbour update (Shi 2026 Eq 8), riding the pass's own
// samples, targets, and pairwise ratios — resampling is untouched. For each
// accepted neighbour j the lane earns one estimate of this pixel's integral,
//
//   ⟨F_c⟩_{←j} = estF_j + α_j ⊙ (⟨G_j⟩ − estG_j),
//
// estF_j the two-sample pairwise-MIS estimate of F_c (visibility-aware — the
// rays this pass traces), ⟨G_j⟩ the neighbour's transported lane, and estG_j
// the matching estimate of the *same* G_j from the same two samples, so the
// bracket is zero-mean and the whole term unbiased for F_c whatever α is.
//
// G — the lane's transported semantics — is the pre-visibility integral the
// candidate stream itself estimates: NEE terms unshadowed, indirect terms
// concrete (their walks traced every ray). The reference's integrands all
// embed visibility because its shifts always trace; cenote's one-ray-per-
// candidate policy (D-093/D-094) prices own-domain and reverse targets
// unshadowed, so the CV divergences land exactly where the missing rays are
// (the named D-144 divergence, both halves below):
//
//   - estG_j's canonical-sample term exists only for an NEE canonical (its
//     backshift re-forms unshadowed for free); an indirect canonical's
//     backshift is the one evaluation cenote never prices, so the pair
//     partition for G_j sets the canonical technique's weight to zero on
//     indirect points — the neighbour's own sample covers them alone (its
//     weight there is 1), the Kettunen coverage move the dead-shift
//     fallbacks already make. Partition of unity holds pointwise either way,
//     so unbiasedness survives; only the residual's pairing weakens on
//     indirect samples.
//
//   - the centre estimate repays the canonical lane's visibility deficit at
//     the one sample whose shadow ray this pass traces:
//     centre = lane_c + W_c·chroma_c·(p̂ᵛⁱˢ − p̂ᵘⁿˢʰ) — the correction's mean
//     is exactly F_c − G_c (zero for an indirect canonical, where the two
//     targets are the same number), so the centre is unbiased for F_c.
//
// The combination is the reference's: (Σ_j ⟨F_c⟩_{←j} + c·centre)/(V + c),
// uniform neighbour weights (M-weighting is temporal-only), c the fixed
// central importance below. Signed per channel by design — no positivity
// clamp, non-finite terms dropped (the reference's isfinite policy) — and
// resolve shades the lane; accumulation averages the transients out.

// The central importance c — how many neighbours' worth of weight the centre
// estimate carries. The supplemental's closed form c = ½(√(1+6N+N²)−N+1)
// gives ≈1.6 at the reference's N≈2.4; fixed here (decision 5), the
// reference's live per-frame variance model (~5%) deferred behind a
// measurement (D-043).
static const float CV_CENTRAL_IMPORTANCE = 1.6;

// α — the per-channel primary-hit albedo ratio (ReSTCV Eq 9): the
// neighbour's colour estimate is demodulated by its own albedo and
// remodulated by this pixel's, the denoiser-demodulation intuition — and the
// same tabulated `aovAlbedo` the denoiser guides read supplies both sides.
// α is deterministic (geometry and material only), so any value is unbiased;
// the guards — non-finite ratio → 1, clamp ≤ 2 — are the reference's own
// variance hygiene against near-zero albedos.
float alphaChannel(float c, float n) {
    let a = c / n;
    return isfinite(a) ? min(a, 2.0) : 1.0;
}

float3 albedoRatio(float3 canonical, float3 neighbour) {
    return float3(alphaChannel(canonical.x, neighbour.x),
                  alphaChannel(canonical.y, neighbour.y),
                  alphaChannel(canonical.z, neighbour.z));
}

// The visibility-aware canonical target p̂_c = U_c·V of a light sample at *this*
// surface, tracing one shadow ray. Returns p̂_c and reports the unshadowed U_c
// through `unshadowed` — the survivor's finalize divides by U_c, folding its own
// V into W (see the header) — and the unshadowed RGB the luminance was taken of
// through `contribution`, the §6.3 chroma numerator. One inline ray, or none
// when the unshadowed target is already zero (nothing to shade, so nothing to
// occlude).
float targetVisAware(Surface s, PathSample sample, out float unshadowed,
                     out float3 contribution) {
    let t = evalTarget(params.scene, params.restir, s, sample);
    unshadowed = t.unshadowed;
    contribution = t.contribution;
    if (!(t.unshadowed > 0.0)) {
        return 0.0; // nothing to shade, so nothing to occlude — skip the ray
    }
    let c = t.connection;
    RayDesc ray;
    ray.Origin = t.throughSurface ? s.below : s.position;
    ray.Direction = c.toLight;
    ray.TMin = 0.0;
    ray.TMax = c.shadowTmax;
    let v = shadowVisibility(sceneTlas, params.scene, ray, c.shadowLight, c.shadowPrimitive);
    return t.unshadowed * v;
}

// One accepted neighbour, held between the gather pass (which spends its rays)
// and the stream pass (which needs t_c and V, known only once every neighbour
// is gathered). `IDefaultInitializable` so the fixed-size array below can be
// default-constructed — a struct holding a `PathSample` needs the
// conformance spelled out for that.
struct AcceptedNeighbour : IDefaultInitializable {
    PathSample sample;
    float confidence;  // c_i
    float weightNumer; // b_i · p̂_c(Y_i) · W_i — the resampling weight before (1−t_c)/V
    float unshadowed;  // U_c(Y_i) — the finalize divisor if this neighbour wins
    float accept;      // WRS acceptance draw
};

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void restir_spatial(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.hits.count()) {
        return;
    }
    uint slot = params.hits.entries[tid.x];
    uint pixel = params.paths.pixel[slot];

    // The range's base pixel: raygen wrote pixel = base + slot, so base recovers
    // per-thread with no push constant. A neighbour is in this range iff its slot
    // (neighbourPixel − base) addresses the same pixel back.
    uint base = pixel - slot;

    Surface canonical = reconstructSurface(params.paths, params.scene, slot);
    StoredPathReservoir canonicalStored = params.reservoirsIn[pixel];
    Reservoir<PathSample> canonicalReservoir = loadPathReservoir(canonicalStored);
    let cCanon = canonicalReservoir.confidence; // c_c > 0 for any hit pixel
    let canonNee = pathKind(canonicalReservoir.sample) == PATH_KIND_NEE;

    // p̂_c(Y_c) and U_c(Y_c): the canonical sample re-targeted here, with its one
    // visibility ray — the "+1" of the k+1. An empty/degenerate canonical (rare
    // at a real hit) yields 0 and simply loses to any neighbour. An indirect
    // sample's — reconnection or replay — own-domain target is its stored
    // integrand (D-138/D-139): F was formed in this very domain this frame —
    // by generation, or by temporal's cross-frame shift, which re-roots a
    // winning prev sample at this pixel (prefix replayed, visibility
    // re-traced) before storing it (step 5b) — so it is trusted, and no ray
    // is spent.
    float ucYc = 0.0;
    float pcYc = 0.0;
    float3 canonChroma = float3(0.0);
    if (canonicalReservoir.unbiasedWeight > 0.0) {
        if (!canonNee) {
            ucYc = luminance(canonicalReservoir.sample.F);
            pcYc = ucYc;
            canonChroma = chromaDirection(canonicalReservoir.sample.F, ucYc);
        } else {
            float3 rgb;
            pcYc = targetVisAware(canonical, canonicalReservoir.sample, ucYc, rgb);
            canonChroma = chromaDirection(rgb, ucYc);
        }
    }
    // The canonical sample's visibility-aware RGB contribution W_c·F̂(Y_c) =
    // W_c·p̂ᵛⁱˢ·chroma — rgb·V·W_c for an NEE canonical, the concrete F·W_c for
    // an indirect one. Every pair's F-estimate below reuses it as its
    // canonical-sample term.
    let canonVisF = canonicalReservoir.unbiasedWeight * (pcYc * canonChroma);

    let px = int(pixel % params.width);
    let py = int(pixel / params.width);

    // The blue-noise-ranked sample index, shared across this stage's draws for
    // the pixel so its reuse error stays screen-space blue noise (D-095) —
    // coherent with candidates, which ranked the same (pixel, sample) key.
    let ranked = blueRank(uint2(uint(px), uint(py)), params.sampleIndex);

    // -- Gather: gate each neighbour, spend its ray, stash it --------------------
    AcceptedNeighbour accepted[MAX_SPATIAL_NEIGHBOURS];
    uint validCount = 0;             // V — accepted neighbours
    float cTot = cCanon;             // c_tot — running Σ confidence
    float canonShareSum = 0.0;       // Σ_i c_c·p̂_c(Y_c)/(c_i·p̂_i(Y_c) + c_c·p̂_c(Y_c))
    float3 cvPairSum = float3(0.0);  // Σ_j ⟨F_c⟩_{←j} — the per-neighbour CV terms

    let k = min(params.neighbours, MAX_SPATIAL_NEIGHBOURS);
    for (uint n = 0; n < k; n++) {
        let uOffset = sample_ranked_2d(pixel, ranked,
                                       restirSpatialDimension(n, RESTIR_SPATIAL_OFFSET));
        // A uniform point on the gather disk (√u for area-uniform radius), rounded
        // to a pixel offset. The self pixel (zero offset) is the canonical — never
        // reused as its own neighbour, or it would double-count.
        let r = params.radius * sqrt(uOffset.x);
        let theta = 6.28318530718 * uOffset.y;
        let dx = int(round(r * cos(theta)));
        let dy = int(round(r * sin(theta)));
        if (dx == 0 && dy == 0) {
            continue;
        }
        let nx = px + dx;
        let ny = py + dy;
        if (nx < 0 || ny < 0 || nx >= int(params.width) || ny >= int(params.height)) {
            continue;
        }
        let neighbourPixel = uint(ny) * params.width + uint(nx);

        // The neighbour's path slot in *this* range. Out of range (a different
        // pool walk, or the arithmetic underflowed) → no live path state → drop.
        let neighbourSlot = neighbourPixel - base;
        if (neighbourSlot >= params.capacity
            || params.paths.pixel[neighbourSlot] != neighbourPixel) {
            continue;
        }

        // A miss or unreached neighbour reads confidence 0 from the wave-cleared
        // buffer — nothing to reuse, drop before its surface is rebuilt.
        StoredPathReservoir neighbourStored = params.reservoirsIn[neighbourPixel];
        Reservoir<PathSample> neighbour = loadPathReservoir(neighbourStored);
        if (!(neighbour.confidence > 0.0)) {
            continue;
        }

        Surface neighbourSurface = reconstructSurface(params.paths, params.scene, neighbourSlot);
        // The reuse gate: neighbours must share this surface's orientation and
        // camera depth, or their light samples describe a different shading
        // context (D-093, stricter than RTXDI's 0.5/0.1 on the normal — the
        // conservative choice, and the first knob to loosen if convergence lags).
        if (dot(canonical.normal, neighbourSurface.normal) <= params.normalThreshold) {
            continue;
        }
        if (abs(canonical.depth - neighbourSurface.depth)
            > params.depthThreshold * canonical.depth) {
            continue;
        }

        // The shifted targets (D-134/D-135/D-138/D-139). An indirect sample —
        // reconnection or replay — crosses through the shared forward block
        // (`shiftIntoDomain`, restir_target.slang): the shift map, its one
        // visibility ray, the partial-shift zero for an undefined shift
        // (D-136), and the re-rooted sample to store on a win. An NEE sample
        // stores a domain-independent light point — identity shift, Jacobian
        // 1 — through this stage's own arms: the visibility-aware p̂_c and
        // the unshadowed own-domain p̂_i.
        let neeNeighbour = pathKind(neighbour.sample) == PATH_KIND_NEE;
        float ucYi = 0.0;
        float pcYiShifted = 0.0; // p̂_c(shift(Y_i))·J_i, 1 ray
        float piYi = 0.0;        // p̂_i(Y_i)
        float3 chroma = float3(0.0);
        float3 fOwn = float3(0.0); // f_j(Y_j) at G-semantics — the residual's own term
        var shifted = neighbour.sample;
        if (!neeNeighbour) {
            let sh = shiftIntoDomain(sceneTlas, params.scene,
                                     neighbour.sample, canonical);
            pcYiShifted = sh.target;
            piYi = sh.ownTarget;
            ucYi = sh.unshadowed;
            shifted = sh.sample;
            // The shifted integrand is this domain's F (the shift rewrote it),
            // and `unshadowed` is its luminance — the §6.3 chroma pair.
            chroma = chromaDirection(sh.sample.F, sh.unshadowed);
            // The own-domain integrand, concretely traced at generation —
            // G's indirect terms carry their visibility already lived.
            fOwn = neighbour.sample.F;
        } else {
            float3 rgb;
            pcYiShifted = targetVisAware(canonical, neighbour.sample, ucYi, rgb);
            chroma = chromaDirection(rgb, ucYi);
            let tOwn = evalTarget(params.scene, params.restir, neighbourSurface,
                                  neighbour.sample);
            piYi = tOwn.unshadowed;
            // G's NEE terms are unshadowed — the same RGB the own-domain
            // target is the luminance of.
            fOwn = tOwn.contribution;
        }
        // Symmetrically, the canonical's sample stated in the neighbour's
        // domain: p̂_i(Y_c)·J through the shared ray-free reverse — own-domain
        // targets spend no ray, any kind. An NEE canonical re-forms through
        // `evalTarget` (what `targetInDomain` routes to for its kind), and its
        // RGB doubles as the residual's backshift term; an indirect canonical
        // prices through `targetInDomain`, and the residual deliberately never
        // reads its backshift (the D-144 partition — see the module header).
        // Skipped when the canonical's own-domain target is already zero (its
        // share is zero regardless), which also spares the replay rays.
        float piYcShifted = 0.0;
        float3 fBack = float3(0.0); // f_j(T(Y_c)) at G-semantics — NEE canonical only
        if (canonicalReservoir.unbiasedWeight > 0.0 && pcYc > 0.0) {
            if (canonNee) {
                let tBack = evalTarget(params.scene, params.restir,
                                       neighbourSurface, canonicalReservoir.sample);
                piYcShifted = tBack.unshadowed;
                fBack = tBack.contribution;
            } else {
                piYcShifted = targetInDomain(sceneTlas, params.scene, params.restir,
                                             canonicalReservoir.sample,
                                             neighbourSurface);
            }
        }

        let ci = neighbour.confidence;
        // b_i at Y_i, and the canonical's share of this pair at Y_c.
        let mis = pairwiseMis(ci, piYi, piYcShifted, cCanon, pcYiShifted, pcYc);

        // -- This pair's control-variate term (6b-i, D-144; module header) ---
        // estF: the two-sample pairwise estimate of this pixel's integral,
        // visibility-aware — each scalar target re-formed in RGB through its
        // chroma. estG: the matching estimate of the neighbour's transported
        // G_j — its own sample under the same b_i (weight 1 on indirect
        // points, where the canonical technique is declared uncovered), plus
        // the canonical's unshadowed backshift for an NEE canonical. The
        // bracket α⊙(lane_j − estG) is zero-mean; a non-finite term drops to
        // zero while the neighbour stays counted in V (the reference's
        // isfinite policy, and V must match the resampling stream's).
        let alpha = albedoRatio(canonical.closure.aovAlbedo,
                                neighbourSurface.closure.aovAlbedo);
        let weightNumer = mis.bi * pcYiShifted * neighbour.unbiasedWeight;
        let estF = weightNumer * chroma + mis.canonShare * canonVisF;
        var estG = (neeNeighbour ? mis.bi : 1.0) * neighbour.unbiasedWeight * fOwn;
        if (canonNee) {
            estG += mis.canonShare * canonicalReservoir.unbiasedWeight * fBack;
        }
        let cvPair = estF + alpha * (neighbourStored.cvAccumulator - estG);
        if (all(isfinite(cvPair))) {
            cvPairSum += cvPair;
        }

        AcceptedNeighbour a;
        a.sample = shifted;
        a.confidence = ci;
        // The GRIS weight of the shifted neighbour: m_i·p̂_c(shift(Y_i))·W_i·J_i.
        a.weightNumer = weightNumer;
        a.unshadowed = ucYi;
        a.accept = sample_ranked_1d(pixel, ranked,
                                    restirSpatialDimension(n, RESTIR_SPATIAL_ACCEPT));
        accepted[validCount] = a;
        validCount++;
        cTot += ci;
        canonShareSum += mis.canonShare;
    }

    // -- Stream: now that t_c and V are known, combine under the MIS weights -----
    let tc = cTot > 0.0 ? cCanon / cTot : 1.0; // c_c/c_tot; =1 when V=0 (canonical only)
    let invV = validCount > 0 ? 1.0 / float(validCount) : 0.0;

    Reservoir<PathSample> merged = reservoirInit<PathSample>();
    float survivorTarget = 0.0;

    // The canonical: m_c = t_c + (1−t_c)·(1/V)·Σ shares. With V=0 the sum term
    // vanishes (t_c=1) and it passes through as its sole candidate.
    let mCanon = tc + (1.0 - tc) * invV * canonShareSum;
    let wCanon = mCanon * pcYc * canonicalReservoir.unbiasedWeight;
    let acceptCanon = sample_ranked_1d(pixel, ranked, restirSpatialCanonicalDimension());
    if (reservoirMerge(merged, canonicalReservoir, wCanon, acceptCanon)) {
        survivorTarget = ucYc;
    }

    // Each accepted neighbour: m_i = (1−t_c)·(1/V)·b_i, its resampling weight the
    // stashed b_i·p̂_c(Y_i)·W_i scaled by that. reservoirMerge inherits the
    // neighbour's confidence, so the merged M is c_tot — the compounded history.
    for (uint i = 0; i < validCount; i++) {
        let a = accepted[i];
        Reservoir<PathSample> nb = reservoirInit<PathSample>();
        nb.sample = a.sample;
        nb.confidence = a.confidence;
        let weight = (1.0 - tc) * invV * a.weightNumer;
        if (reservoirMerge(merged, nb, weight, a.accept)) {
            survivorTarget = a.unshadowed;
        }
    }

    // Finalize by the *unshadowed* survivor target: W = weightSum/U_c folds the
    // survivor's own visibility into W (it was resampled ∝ p̂_c = U_c·V), so
    // restir_resolve shades it unshadowed with no further ray.
    reservoirFinalize(merged, survivorTarget);

    // -- The CV combination (6b-i, D-144; module header) ---------------------
    // The centre: the canonical's transported candidate mean, its NEE
    // visibility deficit repaid at the one sample whose ray this pass traced —
    // chroma·(p̂ᵛⁱˢ − p̂ᵘⁿˢʰ)·W_c has mean F_c − G_c, and is exactly zero for an
    // indirect canonical (the two targets are the same number). Then the
    // reference's convex combination over the V pair estimates, the centre at
    // its fixed importance c. With V = 0 the lane is the corrected centre
    // alone — spatial-on with no accepted neighbours still shades the CV.
    // The lane rides this pass's output, pass-local as ever (the scratch
    // never persists a frame); resolve shades it when the host's CV toggle is
    // on, and toggle-off never reads it (the D-130 survivor-shading
    // degenerate). Signed per channel; the only sanitization is non-finite → 0.
    let centre = canonicalStored.cvAccumulator
               + canonicalReservoir.unbiasedWeight * canonChroma * (pcYc - ucYc);
    var cvOut = (cvPairSum + CV_CENTRAL_IMPORTANCE * centre)
              / (float(validCount) + CV_CENTRAL_IMPORTANCE);
    if (!all(isfinite(cvOut))) {
        cvOut = float3(0.0);
    }
    params.reservoirsOut[pixel] = storePathReservoir(merged, cvOut);
}
