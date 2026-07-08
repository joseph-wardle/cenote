// The escaped-ray stage: a constant sky, until the importance-sampled
// equirect HDRI (M1 step 10) lands at exactly this queue boundary. No
// next-event strategy samples the sky yet, so the escape needs no MIS
// weight. Escape is terminal: this add carries the pixel's alpha 1.

import pathstate;

struct Params {
    Paths paths;
    Queue<uint> misses;
    float4* radiance; // the film's per-pixel accumulator for this wave
    float3 sky;       // constant environment radiance, ACEScg
    uint pad;
};

[vk::push_constant]
Params params;

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void shade_miss(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.misses.count()) {
        return;
    }
    uint path = params.misses.entries[tid.x];
    let contribution = params.paths.throughput[path].xyz * params.sky;
    // Pixel-owned add — each pixel has exactly one path per wave, and a
    // path escapes at most once — so the nondeterministic queue order
    // never reaches the image.
    params.radiance[params.paths.pixel[path]] += float4(contribution, 1.0);
}
