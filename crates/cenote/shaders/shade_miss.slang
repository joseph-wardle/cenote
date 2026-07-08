// The escaped-ray stage: a constant sky, M1's only light until the emissive
// quad (step 8) and the importance-sampled equirect HDRI (step 10), which
// lands at exactly this queue boundary. Escape is terminal, so the sky's
// radiance — weighted by everything the path's bounces absorbed on the way
// — is the pixel's one radiance write for the wave.

import pathstate;

struct Params {
    Paths paths;
    Queue<uint> misses;
    float4* radiance; // the film's per-pixel sample for this wave
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
    // Pixel-owned write — each pixel has exactly one path per wave, and a
    // path terminates exactly once — so the nondeterministic queue order
    // never reaches the image.
    params.radiance[params.paths.pixel[path]] = float4(contribution, 1.0);
}
