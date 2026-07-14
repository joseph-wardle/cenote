// Accumulation kernel: add one rendered sample — beauty and the three
// AOVs — into the film's running f32 sums. `reset` makes the sample
// overwrite instead, so starting a fresh accumulation needs no separate
// clear pass — the first sample is the clear. The host tracks the sample
// count (uniform across pixels by construction); the resolve kernel and
// the host readback divide.
//
// Alongside the sums it folds the second moment of beauty luminance — a
// single f32 per pixel, Σ luminance² — the variance substrate (M3 step 6a).
// The first moment is already free in the beauty sum (luminance is a linear
// functional of RGB, so mean luminance = luminance of the mean RGB), so this
// one extra buffer is all the per-pixel sample variance needs:
// Var = mean(L²) − mean(L)². It is taken over the *same* guarded beauty
// sample the beauty sum sees, so the two moments always agree on which
// samples were accepted and Var can't go negative from a mismatch.
//
// From that substrate it also computes the global auto-stop metric (M3 step
// 6b): each pixel's relative estimator standard error e = sqrt(Var/N) /
// max(mean L, floor), and a single global tally of how many pixels fell below
// the noise threshold. The host reads the tally back (one u32) and stops the
// render once the converged fraction crosses its target — cheaper by far than
// downloading the whole variance field every frame. The tally is zero-filled
// before this kernel each sample, so it is a fresh snapshot, not a running
// count; and no pixel counts until N passes `minSamples`, below which a
// handful of samples make Var meaningless (and ReSTIR's temporal history has
// not yet handed off to independent frames — D-094).
//
// Every contribution is finite-guarded before it lands: a NaN or Inf in
// any channel drops that buffer's whole sample for the pixel — one bad
// sample must not poison the film forever (the m1-plan's unconditional
// robustness rule). Each of the four buffers is guarded on its own, so a
// glitch in one AOV never censors the others — and depth is the deliberate
// exception: +inf *is* its miss value, so only NaN is dropped there.

import color;

// Everything the kernel reads, visible in one struct. The samples are the
// wave's per-pixel accumulators, row-major width×height; each sum shares
// its sample's layout.
struct Params {
    float4* sample;       // beauty: one sample's radiance
    float4* sum;
    float4* albedoSample; // denoiser albedo guide
    float4* albedoSum;
    float4* normalSample; // denoiser normal guide
    float4* normalSum;
    float* depthSample;   // camera-plane z at the first hit; +inf on miss
    float* depthSum;
    float* moment2Sum;    // Σ luminance² of the guarded beauty sample
    Atomic<uint>* convergedCount; // pixels below the noise threshold this sample
    uint width;
    uint height;
    uint reset;           // bool: overwrite instead of add
    uint samples;         // sample count after this contribution (N)
    uint minSamples;      // below this N the noise metric is untrusted
    float noiseThreshold; // relative std-error below which a pixel is converged
    float noiseFloor;     // luminance floor in the relative-error denominator
    uint pad0;
};

[vk::push_constant]
Params params;

// One guarded sample: NaN/Inf in any channel drops the whole sample for this
// buffer (0 adds as a no-op and overwrites as the clear a reset wants).
float4 guarded(float4 contribution) {
    return (any(isnan(contribution)) || any(isinf(contribution)))
               ? float4(0.0)
               : contribution;
}

// One guarded accumulation, folding the reset (overwrite) vs add choice in.
float4 accumulated(float4* sample, float4* sum, uint pixel) {
    let contribution = guarded(sample[pixel]);
    return params.reset != 0 ? contribution : sum[pixel] + contribution;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void accumulate(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) {
        return;
    }
    uint pixel = tid.y * params.width + tid.x;

    // Beauty: guard the sample once, then fold it into both the running sum
    // (the first moment, free) and the luminance second moment — same accepted
    // sample for both, so Var = mean(L²) − mean(L)² stays consistent.
    let beauty = guarded(params.sample[pixel]);
    let beautySum = params.reset != 0 ? beauty : params.sum[pixel] + beauty;
    params.sum[pixel] = beautySum;
    let l = luminance(beauty.rgb);
    let l2 = l * l;
    let moment2 = params.reset != 0 ? l2 : params.moment2Sum[pixel] + l2;
    params.moment2Sum[pixel] = moment2;

    // Auto-stop metric: count this pixel if its relative estimator standard
    // error has fallen below the threshold. Skipped below minSamples, where
    // Var is too noisy to trust (see the header). Var = mean(L²) − mean(L)²,
    // clamped against the rounding that can drive it a hair negative; the
    // relative error divides the standard error by the mean luminance, floored
    // so a near-black pixel doesn't read as infinitely noisy.
    if (params.samples >= params.minSamples) {
        let n = float(params.samples);
        let meanL = luminance(beautySum.rgb) / n;
        let variance = max(moment2 / n - meanL * meanL, 0.0);
        let relativeError = sqrt(variance / n) / max(meanL, params.noiseFloor);
        if (relativeError < params.noiseThreshold) {
            params.convergedCount[0].add(1u);
        }
    }

    params.albedoSum[pixel] = accumulated(params.albedoSample, params.albedoSum, pixel);
    params.normalSum[pixel] = accumulated(params.normalSample, params.normalSum, pixel);

    // Depth keeps +inf — a sky pixel's depth is infinite by definition,
    // and an average touched by any miss stays so, which is what keeps
    // defocused silhouettes honest rather than inventing a finite blend.
    var depth = params.depthSample[pixel];
    if (isnan(depth)) {
        depth = 0.0;
    }
    params.depthSum[pixel] = params.reset != 0 ? depth
                                               : params.depthSum[pixel] + depth;
}
