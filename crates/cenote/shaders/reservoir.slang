// The reservoir primitive: weighted reservoir sampling (WRS) and the
// generalized-RIS combine, as an index-agnostic building block. This module
// is deliberately small and deliberately dumb — it holds a running selection
// and knows how to fold one more candidate (or one more reservoir) into it.
// It knows nothing about lights, BSDFs, visibility, or MIS. Everything that
// gives a candidate its weight — the target function p̂, the MIS weight, the
// random number — is computed by the caller and handed in. That wall is the
// whole point: M6 path reuse and a future ReGIR grid instantiate this same
// type additively, and none of them can turn it into a direct-lighting
// framework (D-086, D-091).
//
// -- Rosetta stone ------------------------------------------------------
// The literature's vocabulary, mapped to the readable identifiers here. The
// papers: Bitterli et al. 2020 (ReSTIR-DI), Lin et al. 2022 ("Foundations
// of ReSTIR"/GRIS), Bitterli 2022 (thesis, pairwise MIS).
//
//   sample            Y   — the held sample (the survivor of the stream)
//   weightSum         wSum, Σⱼ wⱼ — the running sum of resampling weights
//   confidence        M / cᵢ — the generalized sample count (a float, so a
//                     temporal reservoir can carry fractional, capped, and
//                     decayed history — the M-cap and confidence decay live
//                     in the temporal stage, not here)
//   unbiasedWeight    W_Y — the unbiased contribution weight (UCW)
//
//   reservoirUpdate   Algorithm 2's stream insert: fold one fresh candidate
//   reservoirMerge    the GRIS combine: fold one *reservoir* in as a single
//                     candidate (temporal/spatial reuse)
//   reservoirFinalize compute W_Y once the stream is complete
//   targetFunction    p̂(·) — supplied by the caller, never defined here
//
// -- The one invariant that must not be broken --------------------------
// weightSum is PASS-LOCAL. A reservoir streams its candidates within a
// single pass, and weightSum is live only for that pass. It is *never*
// serialized to the persistent per-pixel buffer — that buffer stores only
// {sample, unbiasedWeight, confidence}. When a persistent reservoir is read
// back to be reused, its weightSum starts at zero and is rebuilt from
// scratch by the pass doing the reusing (see reservoirMerge). Storing and
// restoring weightSum across passes double-counts and biases the estimator
// silently — no crash, just a converged mean a few percent off. This is the
// single subtlest correctness trap in the whole reservoir path.
//
// -- The UCW is the combined form ---------------------------------------
// reservoirFinalize computes W_Y = weightSum / p̂(Y). This is the combined
// GRIS unbiased contribution weight, NOT the split 1/M · (1/p̂) · wSum of
// classic ReSTIR: the confidence weights are folded into each candidate's
// resampling weight wⱼ by the caller (via MIS), so no separate 1/M factor
// appears here. Get this wrong in either direction and reuse biases.

// A reservoir over an arbitrary sample type. `S` must be default-
// constructible so an empty reservoir has a well-defined (unused) sample;
// confidence == 0 is the real "nothing selected yet" signal.
public struct Reservoir<S : IDefaultInitializable> {
    public S sample;           // Y: the current survivor
    public float weightSum;    // wSum: pass-local running Σ wⱼ — never stored
    public float confidence;   // M: the generalized sample count
    public float unbiasedWeight; // W_Y: the UCW, set by reservoirFinalize
};

// An empty reservoir: no sample, no weight, no confidence.
public Reservoir<S> reservoirInit<S : IDefaultInitializable>() {
    Reservoir<S> r;
    r.sample = S();
    r.weightSum = 0;
    r.confidence = 0;
    r.unbiasedWeight = 0;
    return r;
}

// The shared WRS core: add one candidate of resampling weight `weight`,
// crediting `confidenceDelta` to the running count, and let it become the
// survivor with probability weight / weightSum. Returns whether it did.
//
// The test u·weightSum < weight is u < weight/weightSum with the division
// folded out; because weight is already in weightSum, the ratio is in [0, 1]
// and the first positive-weight candidate always wins (u < 1). A zero or
// negative weight can never be selected, but its confidence still counts —
// an occluded or zero-target candidate is still a sample the MIS balance saw.
bool absorb<S : IDefaultInitializable>(
    inout Reservoir<S> r, S sample, float weight, float confidenceDelta, float rand
) {
    r.weightSum += weight;
    r.confidence += confidenceDelta;
    if (r.weightSum > 0 && rand * r.weightSum < weight) {
        r.sample = sample;
        return true;
    }
    return false;
}

// Stream one fresh candidate into the reservoir (initial RIS). Each fresh
// candidate is worth one unit of confidence. `weight` is the caller's
// resampling weight wᵢ = mᵢ · p̂(xᵢ) · (1/pᵢ) — target over source pdf,
// MIS-weighted; this primitive neither knows nor checks how it was formed.
public bool reservoirUpdate<S : IDefaultInitializable>(
    inout Reservoir<S> r, S sample, float weight, float rand
) {
    return absorb(r, sample, weight, 1.0, rand);
}

// Fold another reservoir in as a single candidate (temporal/spatial reuse).
// `resamplingWeight` is the caller's MIS-weighted contribution for `other`'s
// survivor re-evaluated in *this* domain — mᵢ · p̂ᵢ(other.sample) ·
// other.unbiasedWeight. The combined reservoir inherits `other`'s confidence,
// so the merged M is the sum of the two histories. reservoirMerge is
// MIS-agnostic: which MIS weight mᵢ went into `resamplingWeight` (defensive
// pairwise, generalized balance, …) is entirely the caller's business.
public bool reservoirMerge<S : IDefaultInitializable>(
    inout Reservoir<S> r, Reservoir<S> other, float resamplingWeight, float rand
) {
    return absorb(r, other.sample, resamplingWeight, other.confidence, rand);
}

// Close the stream: compute the unbiased contribution weight of the survivor.
// `targetAtSample` is p̂(Y) — the target function evaluated at the *held*
// sample, supplied by the caller. A non-positive target (a survivor the
// target function rejects, or an empty reservoir) yields W = 0, which zeroes
// the survivor's contribution at resolve rather than dividing by zero.
public void reservoirFinalize<S : IDefaultInitializable>(
    inout Reservoir<S> r, float targetAtSample
) {
    r.unbiasedWeight = (targetAtSample > 0) ? r.weightSum / targetAtSample : 0;
}
