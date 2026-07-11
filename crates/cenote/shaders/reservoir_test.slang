// Test-only kernel: exercises the reservoir primitive on the GPU it ships
// on, so the WRS/GRIS bookkeeping is audited directly. ReSTIR bias does not
// crash or NaN — it shifts a converged mean by a few percent and renders
// plausibly — so, exactly as with the sampler, no image-level test would
// catch a broken reservoir. Compiled at test time through the hot-reload
// compiler; never embedded, never shipped (`slangc.rs` lists what ships).
//
// Each thread runs an independent Monte Carlo estimate of ∫₀¹ x dx = 1/2 by
// resampled importance sampling, and the host averages over hundreds of
// thousands of threads. The analytic answer is known, so a missing 1/M, a
// wrong selection probability, or a wrong UCW formula moves the mean off 1/2
// by far more than the estimator's own (tiny, at this thread count) noise.
// Two estimators are dumped: one from reservoirUpdate + reservoirFinalize
// (initial RIS), one from reservoirMerge (the GRIS reuse combine that carries
// the real bias risk). The merged confidence is dumped too, to pin the
// sample-count bookkeeping exactly.

import rng;
import reservoir;

// The microtest's concrete sample: just the sampled coordinate on [0, 1].
struct ScalarSample : IDefaultInitializable {
    float x;
};

struct Params {
    float* estimateUpdate; // per thread: f(Y)·W from single-reservoir RIS
    float* estimateMerge;  // per thread: f(Y)·W from the merged reservoir
    float* mergedM;        // per thread: the merged reservoir's confidence
    uint count;            // independent reservoirs (one per thread)
    uint candidates;       // M for the update estimator
    uint candidatesA;      // M_A for the merge estimator's first reservoir
    uint candidatesB;      // M_B for the merge estimator's second reservoir
    uint pad;
};

[vk::push_constant]
Params params;

// The integrand being estimated, ∫₀¹ = 1/2, and a strictly-positive target
// that correlates with it (so RIS actually resamples). Uniform source, so the
// source pdf is 1 and a candidate's resampling weight is just (1/M)·p̂.
float integrand(float x) { return x; }
float target(float x) { return x + 0.1; }

// Candidates must be i.i.d. from the source pdf for RIS to be unbiased — the
// stratified Sobol sampler (sample_1d) is negatively correlated across the
// candidate index and would bias the estimate a fraction of a percent, which
// is exactly the error band this test exists to catch. So candidates draw
// from the plain decorrelated hash stream instead, keyed by (pixel, index)
// within a per-reservoir `stream` id: two independent uniforms per candidate,
// the position (axis 0) and the WRS decision (axis 1).
Reservoir<ScalarSample> buildReservoir(uint pixel, uint m, uint stream) {
    Reservoir<ScalarSample> r = reservoirInit<ScalarSample>();
    for (uint i = 0; i < m; i++) {
        float u = hashUniform(pixel, i, stream, 0);
        ScalarSample s;
        s.x = u;
        reservoirUpdate(r, s, (1.0 / m) * target(u), hashUniform(pixel, i, stream, 1));
    }
    reservoirFinalize(r, target(r.sample.x));
    return r;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void reservoir_test(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.count) {
        return;
    }
    let pixel = tid.x;

    // Initial RIS: one reservoir, f(Y)·W_Y is an unbiased estimate of ∫.
    let single = buildReservoir(pixel, params.candidates, 0);
    params.estimateUpdate[tid.x] = integrand(single.sample.x) * single.unbiasedWeight;

    // GRIS reuse: two independent reservoirs combined under the confidence-
    // weighted (balance-heuristic) MIS mᵢ = cᵢ / Σc. Identity shift (same
    // domain), so the combined estimate must still be unbiased for the same
    // integral — the property temporal/spatial reuse rests on.
    let a = buildReservoir(pixel, params.candidatesA, 1);
    let b = buildReservoir(pixel, params.candidatesB, 2);
    Reservoir<ScalarSample> c = reservoirInit<ScalarSample>();
    let mSum = a.confidence + b.confidence;
    reservoirMerge(c, a, (a.confidence / mSum) * target(a.sample.x) * a.unbiasedWeight,
                   hashUniform(pixel, 0, 3, 0));
    reservoirMerge(c, b, (b.confidence / mSum) * target(b.sample.x) * b.unbiasedWeight,
                   hashUniform(pixel, 1, 3, 0));
    reservoirFinalize(c, target(c.sample.x));
    params.estimateMerge[tid.x] = integrand(c.sample.x) * c.unbiasedWeight;
    params.mergedM[tid.x] = c.confidence;
}
