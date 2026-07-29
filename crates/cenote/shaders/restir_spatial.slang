// Spatial reuse: fold each of k screen-space partners' reservoirs into the
// pixel's own under defensive pairwise MIS, so one pixel's hard-won light
// sample lightens its neighbourhood. The third of M3's four discrete reservoir
// stages (D-091), and the first that reads *other* pixels — every earlier stage
// touched only its own pixel, which is why this one carries the step's whole
// correctness weight (D-093).
//
// Since step 7b the stage is split in two: the gather pre-pass
// (`restir_spatial_gather.slang`) runs every pixel's *forward* evaluations —
// gates, shifts, all of the stage's visibility rays — and records the
// shift-dependent values (`restir_pair.slang`); this kernel is the ray-free
// combine. Selection is reciprocal (7a), so a pair's two shift evaluations are
// each some pixel's forward gather: the combine reads its own records for the
// forward terms and its partner's slot-n record for the backshift — the
// evaluation the pre-split stage re-ran from scratch (`targetInDomain`,
// 59–65% of the pass by D-146's measurement). Same functions, same inputs,
// same evaluation order: nothing about the estimator changes, and the split's
// checkpoint is bit-identity with 7a.
//
// -- The estimator -----------------------------------------------------------
// The pixel's own reservoir (from restir_candidates) is the *canonical*
// candidate; the k partners are non-canonical. All k+1 are combined into a
// fresh reservoir at the canonical (this pixel's) surface — the DI shift is the
// identity because the reservoir stores a light *surface point*, so the GRIS
// Jacobian is 1 (D-087) and the shift is just "re-evaluate the target here".
//
// MIS is the *defensive* confidence-weighted pairwise heuristic (Bitterli 2022
// thesis §9.1 / the 2023 course §7.1.3, Eq 7.8). Each partner is compared only
// to the canonical — O(k), not the O(k²) generalized balance heuristic. With
// c_c the canonical confidence, c_i a partner's, c_tot their sum, V the count
// of *accepted* partners, and t_c = c_c/c_tot the defensive fraction:
//
//   b_i = c_i·p̂_i(Y_i) / (c_i·p̂_i(Y_i) + c_c·p̂_c(Y_i))       partner i at Y_i
//   m_i = (1 − t_c)·(1/V)·b_i
//   m_c = t_c + (1 − t_c)·(1/V)·Σ_i c_c·p̂_c(Y_c)/(c_i·p̂_i(Y_c) + c_c·p̂_c(Y_c))
//
// which provably sum to 1 (the unbiasedness condition) for any positive targets:
// (1−t_c)(1/V)[Σb + Σ(1−b)] + t_c = (1−t_c) + t_c = 1. The *defensive* t_c floor
// — RTXDI ships the plain form (Eq 7.7) without it — matters here because the
// canonical is the one technique that covers the full support of f (D-088), and
// t_c > 0 keeps it from ever being zero-weighted by a partner whose target
// poorly approximates this surface's. That is what makes the converge-to-
// reference checkpoint (D-090) reachable, so it is required, not gilding.
//
// -- Visibility-aware target, folded into W ----------------------------------
// p̂_c — the canonical (receiving) target — is **visibility-aware**: p̂_c(Y) =
// U_c(Y)·V(Y), the unshadowed luminance times a shadow ray traced from *this*
// surface to Y's light point (D-093, the course's recommended Tip 4.3 variant)
// — the rays now all spent by the pre-pass, read back here from the records.
// p̂_i — a partner's own-domain target, used only inside the MIS ratios — is
// left unshadowed, so exactly one ray is spent per candidate: k partners + the
// canonical = k+1 rays, all the pre-pass's. Re-tracing at *this* pixel (never
// trusting the partner's stored occlusion) is what keeps reuse unbiased across
// a shadow boundary, the classic ReSTIR bias source (D-087).
//
// The survivor's contribution weight is finalized by dividing weightSum by the
// *unshadowed* U_c(survivor), not p̂_c. Since the survivor was resampled ∝ p̂_c =
// U_c·V, dividing by U_c folds its own visibility into W (W = weightSum/U_c is
// exactly weightSum/p̂_c · V). restir_resolve then shades the survivor with the
// plain unshadowed L·f·cos and no shadow ray of its own — the ray was already
// spent by the pre-pass. Equivalent to the textbook "shade with a resolve-time
// shadow ray" but one ray cheaper (D-093).
//
// -- Neighbour selection: reciprocal Gaussian pairing (M6 step 7a) -----------
// Which k pixels are the partners comes from the self-inverting pairing
// textures (pairing.slang, Enhanced §3): slot n's texture names each pixel one
// partner — Gaussian around it (σ 16 px, the mean partner distance of the
// R = 30 uniform-disk draw this replaced), never itself, and *reciprocal*:
// A gathers B on slot n exactly when B gathers A. Any sample-independent
// selection is unbiased, so no estimator math changes; the Gaussian
// concentration picks more shift-compatible partners (the paper's quality
// claim), and reciprocity is what pays for this file's shape — each pair's two
// shift evaluations computed once, in the pre-pass, and shared. The reuse
// gates (all in the pre-pass now) are symmetric for the same reason: both
// sides of a pair must agree on acceptance, so a partner this pixel accepted
// is guaranteed to have run, accepted it back, and written the records this
// kernel reads (restir_pair.slang).
//
// Runs at bounce 0 only, dispatched from the hit queue it reads but never
// consumes. Reads the committed candidate reservoirs (`reservoirsIn`) and the
// pre-pass records, writes the merged survivors to a separate buffer
// (`reservoirsOut`) — the ping-pong a barrier orders, so no thread reads a
// half-written record or reservoir and determinism holds (M3 plan §2).

import bluenoise;
import pairing;
import pathstate;
import reservoir;
import reservoir_path;
import restir_mis;
import restir_pair;
import restir_scene;
import rng;

// The most partners a spatial pass will gather — the fixed bound on the
// per-thread accepted-partner store below. The host caps k at this
// (wavefront.rs); k default is ~5 (D-088).
static const uint MAX_SPATIAL_NEIGHBOURS = 8;

struct Params {
    Paths paths;
    Queue<uint> hits;            // primary hits, dispatched indirectly; not consumed
    StoredPathReservoir* reservoirsIn;  // candidate stage's output — self + partners, read
    StoredPathReservoir* reservoirsOut; // this stage's output — the merged survivor, written
    PairRecord* pairs;           // the pre-pass's forward records, k per slot — read
    SelfRecord* selfs;           // its self records, one per slot — read
    uint sampleIndex;
    uint width;
    uint neighbours;             // k
    uint pad0;
};

[vk::push_constant]
Params params;

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
// accepted partner j the lane earns one estimate of this pixel's integral,
//
//   ⟨F_c⟩_{←j} = estF_j + α_j ⊙ (⟨G_j⟩ − estG_j),
//
// estF_j the two-sample pairwise-MIS estimate of F_c (visibility-aware — the
// rays the pre-pass traces), ⟨G_j⟩ the partner's transported lane, and estG_j
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
//     indirect points — the partner's own sample covers them alone (its
//     weight there is 1), the Kettunen coverage move the dead-shift
//     fallbacks already make. Partition of unity holds pointwise either way,
//     so unbiasedness survives; only the residual's pairing weakens on
//     indirect samples.
//
//   - the centre estimate repays the canonical lane's visibility deficit at
//     the one sample whose shadow ray the pre-pass traces:
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
// partner's colour estimate is demodulated by its own albedo and
// remodulated by this pixel's, the denoiser-demodulation intuition — and the
// same tabulated `aovAlbedo` the denoiser guides read supplies both sides,
// via the pre-pass's self records. α is deterministic (geometry and material
// only), so any value is unbiased; the guards — non-finite ratio → 1, clamp
// ≤ 2 — are the reference's own variance hygiene against near-zero albedos.
float alphaChannel(float c, float n) {
    let a = c / n;
    return isfinite(a) ? min(a, 2.0) : 1.0;
}

float3 albedoRatio(float3 canonical, float3 neighbour) {
    return float3(alphaChannel(canonical.x, neighbour.x),
                  alphaChannel(canonical.y, neighbour.y),
                  alphaChannel(canonical.z, neighbour.z));
}

// One accepted partner, held between the gather loop (which reads the records)
// and the stream pass (which needs t_c and V, known only once every partner
// is counted). `IDefaultInitializable` so the fixed-size array below can be
// default-constructed — a struct holding a `PathSample` needs the
// conformance spelled out for that.
struct AcceptedNeighbour : IDefaultInitializable {
    PathSample sample;
    float confidence;  // c_i
    float weightNumer; // b_i · p̂_c(Y_i) · W_i — the resampling weight before (1−t_c)/V
    float unshadowed;  // U_c(Y_i) — the finalize divisor if this partner wins
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

    // The range's base pixel: raygen wrote pixel = base + slot, so an accepted
    // partner's slot recovers as partnerPixel − base with no re-check — the
    // pre-pass's range guard already vouched for it.
    uint base = pixel - slot;

    StoredPathReservoir canonicalStored = params.reservoirsIn[pixel];
    Reservoir<PathSample> canonicalReservoir = loadPathReservoir(canonicalStored);
    let cCanon = canonicalReservoir.confidence; // c_c > 0 for any hit pixel
    let canonNee = pathKind(canonicalReservoir.sample) == PATH_KIND_NEE;

    // p̂_c(Y_c) and U_c(Y_c): the canonical sample re-targeted here, from the
    // pre-pass's self record. `pcYc` was stored under the W guard already; the
    // raw unshadowed pieces re-apply it here, reproducing the pre-split
    // stage's exact values (an empty/degenerate canonical yields 0 and simply
    // loses to any partner). An indirect sample's own-domain target is its
    // stored integrand (D-138/D-139) — pcYc = ucYc = luminance(F), which is
    // exactly what the record holds for it.
    SelfRecord self = params.selfs[slot];
    let hasW = canonicalReservoir.unbiasedWeight > 0.0;
    let ucYc = hasW ? self.ucYc : 0.0;
    let pcYc = self.pcYc;
    let canonChroma = hasW ? chromaDirection(self.rgb, self.ucYc) : float3(0.0);
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

    // -- Gather: read each accepted pair's records, form its terms ------------
    AcceptedNeighbour accepted[MAX_SPATIAL_NEIGHBOURS];
    uint validCount = 0;             // V — accepted partners
    float cTot = cCanon;             // c_tot — running Σ confidence
    float canonShareSum = 0.0;       // Σ_i c_c·p̂_c(Y_c)/(c_i·p̂_i(Y_c) + c_c·p̂_c(Y_c))
    float3 cvPairSum = float3(0.0);  // Σ_j ⟨F_c⟩_{←j} — the per-neighbour CV terms

    // k is capped by the pairing textures — one per slot, and fewer than the
    // accepted store's MAX_SPATIAL_NEIGHBOURS bound.
    let k = min(params.neighbours, PAIRING_TEXTURES);
    for (uint n = 0; n < k; n++) {
        // The pre-pass's accept/reject decision for slot n — bounds, range,
        // confidence, and the symmetric geometry gates, all already applied.
        if ((self.accepted & (1u << n)) == 0) {
            continue;
        }
        // Slot n's reciprocal partner this frame — the same deterministic
        // lookup the pre-pass drew, so the pair re-forms without storing it.
        let d = pairingOffset(n, int2(px, py), params.sampleIndex);
        let partnerPixel = uint(py + d.y) * params.width + uint(px + d.x);
        let partnerSlot = partnerPixel - base;

        StoredPathReservoir neighbourStored = params.reservoirsIn[partnerPixel];
        Reservoir<PathSample> neighbour = loadPathReservoir(neighbourStored);
        let neeNeighbour = pathKind(neighbour.sample) == PATH_KIND_NEE;

        // This pixel's forward record of the pair: the partner's sample
        // shifted here and priced here (the visibility ray already spent),
        // p̂_c(shift(Y_i))·J, U_c, the shift-guarded own-domain target, and
        // the F/cachedJacobian a win stores.
        let rec = params.pairs[slot * k + n];
        // The partner's half: its self record (own-domain target and fOwn for
        // an NEE sample, its albedo for α) and its slot-n forward record —
        // its evaluation of *this* pixel's sample in *its* domain, the
        // backshift the pre-split stage re-computed per neighbour.
        let partnerSelf = params.selfs[partnerSlot];
        let partnerRec = params.pairs[partnerSlot * k + n];

        let pcYiShifted = rec.visTarget; // p̂_c(shift(Y_i))·J_i, the pre-pass's ray
        let ucYi = rec.unshadowed;
        // The shifted integrand is this domain's F (the shift rewrote it),
        // and `unshadowed` is its luminance — the §6.3 chroma pair.
        let chroma = chromaDirection(rec.F, rec.unshadowed);
        // p̂_i(Y_i): an indirect sample's rides this pixel's record
        // (shift-guarded, D-136); an NEE sample's is its partner's raw
        // self-evaluation — the identical `evalTarget` the pre-split stage
        // re-ran at this pixel.
        let piYi = neeNeighbour ? partnerSelf.ucYc : rec.ownTarget;
        // f_j(Y_j) at G-semantics — the CV residual's own term: the concrete
        // stored integrand for an indirect partner (its visibility already
        // lived), the partner's raw unshadowed self RGB for an NEE one.
        let fOwn = neeNeighbour ? partnerSelf.rgb : neighbour.sample.F;
        // Reconstitute the shifted sample from a fresh reservoir read: it
        // differs from the stored one only in F and cachedJacobian, which the
        // record carries (patched by a defined shift, original otherwise).
        // An NEE sample shifts by identity — nothing to patch, and its
        // record's F is the rgb contribution, not the (absent) integrand.
        var shifted = neighbour.sample;
        if (!neeNeighbour) {
            shifted.F = rec.F;
            shifted.cachedJacobian = rec.partial;
        }
        // Symmetrically, the canonical's sample stated in the partner's
        // domain: p̂_i(Y_c)·J — the partner's forward evaluation of it, read
        // from its record instead of re-run (the sharing). Its record's F is
        // an NEE canonical's backshift RGB, the residual's fBack; an indirect
        // canonical's backshift the residual deliberately never reads (the
        // D-144 partition — see the module header). Skipped when the
        // canonical's own target is already zero (its share is zero
        // regardless) — the pre-split stage's exact guard.
        float piYcShifted = 0.0;
        float3 fBack = float3(0.0);
        if (hasW && pcYc > 0.0) {
            piYcShifted = partnerRec.backTarget;
            if (canonNee) {
                fBack = partnerRec.F;
            }
        }

        let ci = neighbour.confidence;
        // b_i at Y_i, and the canonical's share of this pair at Y_c.
        let mis = pairwiseMis(ci, piYi, piYcShifted, cCanon, pcYiShifted, pcYc);

        // -- This pair's control-variate term (6b-i, D-144; module header) ---
        // estF: the two-sample pairwise estimate of this pixel's integral,
        // visibility-aware — each scalar target re-formed in RGB through its
        // chroma. estG: the matching estimate of the partner's transported
        // G_j — its own sample under the same b_i (weight 1 on indirect
        // points, where the canonical technique is declared uncovered), plus
        // the canonical's unshadowed backshift for an NEE canonical. The
        // bracket α⊙(lane_j − estG) is zero-mean; a non-finite term drops to
        // zero while the partner stays counted in V (the reference's
        // isfinite policy, and V must match the resampling stream's).
        let alpha = albedoRatio(self.albedo, partnerSelf.albedo);
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
        // The GRIS weight of the shifted partner: m_i·p̂_c(shift(Y_i))·W_i·J_i.
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

    // Each accepted partner: m_i = (1−t_c)·(1/V)·b_i, its resampling weight the
    // stashed b_i·p̂_c(Y_i)·W_i scaled by that. reservoirMerge inherits the
    // partner's confidence, so the merged M is c_tot — the compounded history.
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
    // visibility deficit repaid at the one sample whose ray the pre-pass
    // traced — chroma·(p̂ᵛⁱˢ − p̂ᵘⁿˢʰ)·W_c has mean F_c − G_c, and is exactly
    // zero for an indirect canonical (the two targets are the same number).
    // Then the reference's convex combination over the V pair estimates, the
    // centre at its fixed importance c. With V = 0 the lane is the corrected
    // centre alone — spatial-on with no accepted partners still shades the CV.
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
