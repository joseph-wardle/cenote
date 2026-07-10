// Accumulation kernel: add one rendered sample — beauty and the three
// AOVs — into the film's running f32 sums. `reset` makes the sample
// overwrite instead, so starting a fresh accumulation needs no separate
// clear pass — the first sample is the clear. The host tracks the sample
// count (uniform across pixels by construction); the resolve kernel and
// the host readback divide.
//
// Every contribution is finite-guarded before it lands: a NaN or Inf in
// any channel drops that buffer's whole sample for the pixel — one bad
// sample must not poison the film forever (the m1-plan's unconditional
// robustness rule). Each of the four buffers is guarded on its own, so a
// glitch in one AOV never censors the others — and depth is the deliberate
// exception: +inf *is* its miss value, so only NaN is dropped there.

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
    uint width;
    uint height;
    uint reset;           // bool: overwrite instead of add
    uint pad0;
};

[vk::push_constant]
Params params;

// One guarded accumulation: NaN/Inf drops the sample (0 accumulates as a
// no-op and overwrites as the clear a reset wants).
float4 accumulated(float4* sample, float4* sum, uint pixel) {
    var contribution = sample[pixel];
    if (any(isnan(contribution)) || any(isinf(contribution))) {
        contribution = float4(0.0);
    }
    return params.reset != 0 ? contribution : sum[pixel] + contribution;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void accumulate(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) {
        return;
    }
    uint pixel = tid.y * params.width + tid.x;

    params.sum[pixel] = accumulated(params.sample, params.sum, pixel);
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
