// Accumulation kernel: add one rendered sample — beauty and the three
// AOVs — into the film's running f32 sums, plus the second moment of beauty
// luminance that the variance and auto-stop metrics are built on. The host
// tracks the sample count (uniform across pixels by construction); the
// resolve kernel and the host readback divide.
//
// Three things here are load-bearing and not local:
//
// - `reset` overwrites instead of adding, so a fresh accumulation needs no
//   clear pass — the first sample *is* the clear.
// - Every contribution is finite-guarded, per buffer, so one NaN sample
//   can't poison the film forever and a glitch in one AOV never censors the
//   others. Depth is the deliberate exception: +inf *is* its miss value.
// - The converged tally is zero-filled by the host before each dispatch, so
//   it is a fresh snapshot rather than a running count. The host reads back
//   that one u32 instead of the whole variance field.

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

    // Guard once, then fold into both the running sum and the luminance second
    // moment. The first moment is free in the beauty sum (luminance is linear
    // in RGB), and both moments see the *same* accepted sample, so
    // Var = mean(L²) − mean(L)² can't go negative from a mismatch.
    let beauty = guarded(params.sample[pixel]);
    let beautySum = params.reset != 0 ? beauty : params.sum[pixel] + beauty;
    params.sum[pixel] = beautySum;
    let l = luminance(beauty.rgb);
    let l2 = l * l;
    let moment2 = params.reset != 0 ? l2 : params.moment2Sum[pixel] + l2;
    params.moment2Sum[pixel] = moment2;

    // Auto-stop metric: count this pixel if its relative estimator standard
    // error sqrt(Var/N)/mean(L) has fallen below the threshold. Skipped below
    // minSamples, where a handful of samples make Var meaningless. Var is
    // clamped against the rounding that can drive it a hair negative, and the
    // denominator floored so a near-black pixel doesn't read as infinitely
    // noisy.
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
