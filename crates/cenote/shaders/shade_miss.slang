// The escaped-ray stage, degenerate for now: a constant black sky, matching
// the M0 look. The equirect HDRI environment — importance-sampled, MIS
// against BSDF rays — lands at exactly this queue boundary (M1 step 10).

import pathstate;

struct Params {
    Paths paths;
    Queue<uint> misses;
    float4* radiance; // the film's per-pixel sample for this wave
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
    // Pixel-owned write — each pixel has exactly one path per wave — so the
    // nondeterministic queue order never reaches the image.
    params.radiance[params.paths.pixel[path]] = float4(0.0, 0.0, 0.0, 1.0);
}
