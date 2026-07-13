// Test-only kernel: the temporal combine — the defensive pairwise-MIS fold of
// one M-capped history reservoir into the canonical — on a 1D analytic domain,
// so `restir_temporal.slang`'s estimator math is audited directly. Like the
// spatial combine, temporal bias is silent: a wrong MIS normalizer, or an M-cap
// applied to one of {the MIS weights, the merged confidence} but not the other,
// shifts the converged mean or the compounded history a few percent and renders
// plausibly. Compiled at test time (never embedded).
//
// Each thread runs one independent estimate of ∫₀¹ x dx = ½ by: building a
// canonical reservoir and one "prev" reservoir by RIS (each with its own
// target), M-capping the prev's confidence, then combining them under the *same*
// defensive pairwise-MIS weights and finalize-by-target the real stage uses.
// Temporal is **unshadowed** — no synthetic visibility, unlike the spatial test
// — so the closed form is the plain ∫₀¹ x dx = ½, which a broken combine misses
// by far more than this thread count's noise. The merged confidence is dumped
// too, and it is what pins the M-cap: it must be c_c + min(c_prev, M_CAP·c_c),
// so a cap applied inconsistently (or not at all) lands a value the host did not
// predict, even though the *integral* alone — unbiased for any partition-of-
// unity weights — would not catch it.
//
// Distinct targets for the canonical (p̂_c = x + 0.1) and the neighbour
// (p̂_i = 1.1 − x) make the pairwise ratios non-trivial, exercising the general
// MIS the real stage runs (there p̂_i ≡ p̂_c until 5c reprojects, a special case
// this subsumes) rather than letting equal targets collapse the weights to bare
// confidence ratios.

import rng;
import reservoir;

// The microtest's sample: a coordinate on [0, 1].
struct ScalarSample : IDefaultInitializable {
    float x;
};

struct Params {
    float* estimate; // per thread: integrand(Y)·W from the combined reservoir
    float* mergedM;  // per thread: the combined reservoir's confidence
    uint count;      // independent estimates (one per thread)
    uint canonM;     // M_c — canonical RIS candidate count
    uint prevM;      // M_prev — the history reservoir's RIS candidate count
    float mCap;      // history-length multiplier: c_prev ≤ mCap·M_c
    uint pad0;
    uint pad1;
};

[vk::push_constant]
Params params;

float integrand(float x) { return x; }
// Unshadowed targets — temporal traces no ray, so p̂_c carries no visibility and
// the finalize divides by it directly (W folds nothing extra).
float canonTarget(float x) { return x + 0.1; }
// The neighbour's own-domain target p̂_i — distinct, so the MIS is exercised.
float neighbourTarget(float x) { return 1.1 - x; }

// Build one reservoir by RIS. Candidates are i.i.d. uniform from the plain hash
// stream (Sobol would correlate across the candidate index and bias the estimate
// a fraction of a percent — the exact band this test guards); the target is the
// canonical's or the neighbour's.
Reservoir<ScalarSample> build(uint pixel, uint m, uint stream, bool canonical) {
    Reservoir<ScalarSample> r = reservoirInit<ScalarSample>();
    for (uint i = 0; i < m; i++) {
        float u = hashUniform(pixel, i, stream, 0);
        ScalarSample s;
        s.x = u;
        let target = canonical ? canonTarget(u) : neighbourTarget(u);
        reservoirUpdate(r, s, (1.0 / m) * target, hashUniform(pixel, i, stream, 1));
    }
    let ty = canonical ? canonTarget(r.sample.x) : neighbourTarget(r.sample.x);
    reservoirFinalize(r, ty);
    return r;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void restir_temporal_test(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.count) {
        return;
    }
    let pixel = tid.x;
    let cCanon = float(params.canonM);

    // The canonical reservoir and its target at its own sample.
    Reservoir<ScalarSample> canon = build(pixel, params.canonM, 0, true);
    let ycx = canon.sample.x;
    let pcYc = canonTarget(ycx);

    // The history ("prev") reservoir, then the M-cap: clamp its confidence to
    // mCap·c_c and let the merge inherit the *capped* value — exactly
    // restir_temporal.slang, applied once so the MIS weights and the merged M
    // stay consistent (the trap this test guards).
    Reservoir<ScalarSample> prev = build(pixel, params.prevM, 1, false);
    prev.confidence = min(prev.confidence, params.mCap * cCanon);
    let yi = prev.sample.x;
    let ci = prev.confidence;

    let pcYi = canonTarget(yi);      // p̂_c(Y_i)
    let piYi = neighbourTarget(yi);  // p̂_i(Y_i)
    let piYc = neighbourTarget(ycx); // p̂_i(Y_c)

    let cTot = cCanon + ci;
    let tc = cTot > 0.0 ? cCanon / cTot : 1.0;
    let invV = ci > 0.0 ? 1.0 : 0.0; // one neighbour when prev carries history

    let bDenom = ci * piYi + cCanon * pcYi;
    let bi = bDenom > 0.0 ? (ci * piYi) / bDenom : 0.0;
    let shareDenom = ci * piYc + cCanon * pcYc;
    let canonShare = shareDenom > 0.0 ? (cCanon * pcYc) / shareDenom : 0.0;

    Reservoir<ScalarSample> merged = reservoirInit<ScalarSample>();
    float survivorTarget = 0.0;

    let mCanon = tc + (1.0 - tc) * invV * canonShare;
    if (reservoirMerge(merged, canon, mCanon * pcYc * canon.unbiasedWeight,
                       hashUniform(pixel, 0, 99, 0))) {
        survivorTarget = pcYc;
    }
    let weight = (1.0 - tc) * invV * bi * pcYi * prev.unbiasedWeight;
    if (reservoirMerge(merged, prev, weight, hashUniform(pixel, 1, 99, 0))) {
        survivorTarget = pcYi;
    }
    reservoirFinalize(merged, survivorTarget);

    // Unshadowed, so the estimate is integrand(Y)·W with no visibility factor:
    // E = ∫₀¹ x dx = ½.
    params.estimate[tid.x] = integrand(merged.sample.x) * merged.unbiasedWeight;
    params.mergedM[tid.x] = merged.confidence;
}
