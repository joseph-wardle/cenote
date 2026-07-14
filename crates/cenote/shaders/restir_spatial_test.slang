// Test-only kernel: the defensive pairwise-MIS spatial combine, on a 1D
// analytic domain, so `restir_spatial.slang`'s estimator math is audited
// directly. ReSTIR bias is silent — a wrong MIS normalizer, a mishandled
// accepted count, or a botched visibility fold shifts the converged mean a few
// percent and renders plausibly — so, exactly as with `reservoir_test`, no
// image-level test would catch it. Compiled at test time (never embedded).
//
// Each thread runs one independent estimate of ∫₀¹ x·V(x) dx by: building a
// canonical reservoir and K neighbour reservoirs by RIS (each with its own
// target), then combining them under the *same* defensive pairwise-MIS weights,
// visibility-aware canonical target, and finalize-by-unshadowed the real stage
// uses. With a synthetic binary visibility V(x) = [x ≥ ½], the answer is
// ∫_½¹ x dx = 3/8 exactly — a value a broken combine misses by far more than
// the estimator's own (tiny, at this thread count) noise. The merged confidence
// is dumped too, to pin the accepted-count bookkeeping (it must be c_tot).
//
// Distinct targets for the canonical (U_c = x + 0.1) and neighbours
// (p̂_i = 1.1 − x) make the pairwise ratios non-trivial, so the MIS weights are
// genuinely exercised rather than cancelling to a constant. The binary V drives
// p̂_c to zero on half the domain, exercising the "the canonical can't see this
// sample" path the visibility-aware target must handle.

import rng;
import reservoir;
import restir_mis;

// The most neighbours a test thread combines — the fixed local-array bound,
// mirroring the host's `K`.
static const uint MAX_TEST_NEIGHBOURS = 8;

// The microtest's sample: a coordinate on [0, 1].
struct ScalarSample : IDefaultInitializable {
    float x;
};

struct Params {
    float* estimate; // per thread: integrand(Y)·W from the combined reservoir
    float* mergedM;  // per thread: the combined reservoir's confidence
    uint count;      // independent estimates (one per thread)
    uint canonM;     // M_c — canonical RIS candidate count
    uint neighbourM; // M_i — each neighbour's RIS candidate count
    uint neighbours; // K
    uint pad0;
    uint pad1;
};

[vk::push_constant]
Params params;

float integrand(float x) { return x; }
// Synthetic visibility: the light sample x is visible from the canonical
// surface iff x ≥ ½. Binary, so p̂_c vanishes on half the domain.
float visibility(float x) { return x >= 0.5 ? 1.0 : 0.0; }
// The canonical's *unshadowed* target U_c (finalize divides by this), and its
// visibility-aware target p̂_c = U_c·V (the resampling weights use this).
float canonUnshadowed(float x) { return x + 0.1; }
float canonTarget(float x) { return canonUnshadowed(x) * visibility(x); }
// A neighbour's own-domain target p̂_i — unshadowed, as in the real stage.
float neighbourTarget(float x) { return 1.1 - x; }

// Build one reservoir by RIS. Candidates are i.i.d. uniform from the plain hash
// stream (Sobol would correlate across the candidate index and bias the
// estimate a fraction of a percent — the exact band this test guards); the
// target is the canonical's or a neighbour's.
Reservoir<ScalarSample> build(uint pixel, uint m, uint stream, bool canonical) {
    Reservoir<ScalarSample> r = reservoirInit<ScalarSample>();
    for (uint i = 0; i < m; i++) {
        float u = hashUniform(pixel, i, stream, 0);
        ScalarSample s;
        s.x = u;
        let target = canonical ? canonUnshadowed(u) : neighbourTarget(u);
        reservoirUpdate(r, s, (1.0 / m) * target, hashUniform(pixel, i, stream, 1));
    }
    let ty = canonical ? canonUnshadowed(r.sample.x) : neighbourTarget(r.sample.x);
    reservoirFinalize(r, ty);
    return r;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void restir_spatial_test(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.count) {
        return;
    }
    let pixel = tid.x;
    let k = min(params.neighbours, MAX_TEST_NEIGHBOURS);
    let cCanon = float(params.canonM);

    // The canonical reservoir, and its target at its own sample — p̂_c (with the
    // synthetic visibility) for the resampling weight, U_c for the finalize.
    Reservoir<ScalarSample> canon = build(pixel, params.canonM, 0, true);
    let ycx = canon.sample.x;
    let pcYc = canonTarget(ycx);
    let ucYc = canonUnshadowed(ycx);

    // Gather K neighbours: build each, and stash its pairwise numbers exactly as
    // restir_spatial does (b_i, its resampling weight, its U_c, and the
    // canonical's per-pair share at Y_c).
    Reservoir<ScalarSample> nbRes[MAX_TEST_NEIGHBOURS];
    float nbWeight[MAX_TEST_NEIGHBOURS];
    float nbUc[MAX_TEST_NEIGHBOURS];
    float cTot = cCanon;
    float canonShareSum = 0.0;
    for (uint i = 0; i < k; i++) {
        Reservoir<ScalarSample> nb = build(pixel, params.neighbourM, 1 + i, false);
        let yi = nb.sample.x;
        let ci = float(params.neighbourM);
        let pcYi = canonTarget(yi);       // p̂_c(Y_i), visibility-aware
        let piYi = neighbourTarget(yi);   // p̂_i(Y_i), unshadowed
        let piYc = neighbourTarget(ycx);  // p̂_i(Y_c), unshadowed

        let mis = pairwiseMis(ci, piYi, piYc, cCanon, pcYi, pcYc);

        nbRes[i] = nb;
        nbWeight[i] = mis.bi * pcYi * nb.unbiasedWeight;
        nbUc[i] = canonUnshadowed(yi);
        cTot += ci;
        canonShareSum += mis.canonShare;
    }

    let tc = cTot > 0.0 ? cCanon / cTot : 1.0;
    let invV = k > 0 ? 1.0 / float(k) : 0.0;

    Reservoir<ScalarSample> merged = reservoirInit<ScalarSample>();
    float survivorTarget = 0.0;

    let mCanon = tc + (1.0 - tc) * invV * canonShareSum;
    if (reservoirMerge(merged, canon, mCanon * pcYc * canon.unbiasedWeight,
                       hashUniform(pixel, 0, 99, 0))) {
        survivorTarget = ucYc;
    }
    for (uint i = 0; i < k; i++) {
        let weight = (1.0 - tc) * invV * nbWeight[i];
        if (reservoirMerge(merged, nbRes[i], weight, hashUniform(pixel, 1 + i, 99, 0))) {
            survivorTarget = nbUc[i];
        }
    }
    reservoirFinalize(merged, survivorTarget);

    // The estimate is integrand(Y)·W with the *unshadowed* integrand — the
    // survivor's own visibility is already folded into W (finalize by U_c, not
    // p̂_c), so E = ∫ x·V(x) dx = 3/8.
    params.estimate[tid.x] = integrand(merged.sample.x) * merged.unbiasedWeight;
    params.mergedM[tid.x] = merged.confidence;
}
