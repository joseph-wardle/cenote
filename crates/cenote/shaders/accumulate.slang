// Accumulation kernel: add one rendered sample into the film's running
// RGBA f32 sums. `reset` makes the sample overwrite instead, so starting a
// fresh accumulation needs no separate clear pass — the first sample is the
// clear. The host tracks the sample count (uniform across pixels by
// construction); the tonemap kernel divides.
//
// Every contribution is finite-guarded before it lands: a NaN or Inf in any
// channel drops the whole sample for that pixel — one bad sample must not
// poison the film forever (the m1-plan's unconditional robustness rule).

// Everything the kernel reads, visible in one struct.
struct Params {
    float4* frame; // the new sample, row-major width×height
    float4* sum;   // the film's running sums, same layout
    uint width;
    uint height;
    uint reset;    // bool: overwrite instead of add
    uint pad0;
};

[vk::push_constant]
Params params;

[shader("compute")]
[numthreads(8, 8, 1)]
void accumulate(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) {
        return;
    }
    uint pixel = tid.y * params.width + tid.x;

    var contribution = params.frame[pixel];
    if (any(isnan(contribution)) || any(isinf(contribution))) {
        contribution = float4(0.0);
    }
    params.sum[pixel] = params.reset != 0 ? contribution
                                          : params.sum[pixel] + contribution;
}
