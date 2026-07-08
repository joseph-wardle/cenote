// Test-only kernel: dumps the sampler's raw output so the stratification
// test in `wavefront.rs` can audit the GPU implementation directly — a
// subtly broken sampler still renders plausible images, so no image-level
// test would catch it. Compiled at test time through the hot-reload
// compiler; never embedded, never shipped (`slangc.rs` lists what ships).

import rng;

struct Params {
    float2* points; // one sample_2d point per sample index
    float* values;  // one sample_1d value per sample index
    uint pixel;     // sampler keys, fixed across the dispatch
    uint dimension;
    uint count;
    uint pad;
};

[vk::push_constant]
Params params;

[shader("compute")]
[numthreads(64, 1, 1)]
void rng_test(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.count) {
        return;
    }
    params.points[tid.x] = sample_2d(params.pixel, tid.x, params.dimension);
    params.values[tid.x] = sample_1d(params.pixel, tid.x, params.dimension);
}
