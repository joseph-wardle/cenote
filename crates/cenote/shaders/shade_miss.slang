// The escaped-ray stage: look up the environment's radiance along the ray
// and add it, MIS-weighted — an escaped BSDF-sampled ray is the BSDF
// strategy's way of reaching the sky, competing with the next-event
// environment samples shade_surface sends. Camera rays keep weight 1 (no
// other strategy competed). Escape is terminal: this add carries the
// pixel's alpha 1.

import lights;
import pathstate;
import scene;

struct Params {
    Paths paths;
    Queue<uint> misses;
    SceneTable* scene;
    float4* radiance;   // the film's per-pixel accumulator for this wave
    uint lightSampling; // LIGHT_SAMPLING_*: which strategies reach the lights
    uint pad;
};

[vk::push_constant]
Params params;

// Veach's power heuristic (β = 2) — see shade_surface.slang, whose
// next-event weights these complete.
float powerHeuristic(float p, float q) {
    return (p * p) / (p * p + q * q);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void shade_miss(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.misses.count()) {
        return;
    }
    uint path = params.misses.entries[tid.x];
    let direction = params.paths.direction[path].xyz;
    // xyz = the path's weight; w = the previous scatter's pdf (0: camera ray).
    let throughput = params.paths.throughput[path];

    var weight = 1.0;
    if (throughput.w > 0.0) {
        if (params.lightSampling == LIGHT_SAMPLING_NEE) {
            weight = 0.0;
        } else if (params.lightSampling == LIGHT_SAMPLING_MIS) {
            // The probability next-event estimation at the previous vertex
            // had of picking this direction: choose the environment, then
            // this direction from its tables. At a pole the env pdf reads
            // 0 — no next-event sample lands there, so the weight is 1.
            let neePdf = params.scene.env.selectProb * params.scene.env.pdf(direction);
            weight = powerHeuristic(throughput.w, neePdf);
        }
    }

    let contribution = throughput.xyz * params.scene.env.radiance(direction) * weight;
    // Pixel-owned add — each pixel has exactly one path per wave, and a
    // path escapes at most once — so the nondeterministic queue order
    // never reaches the image.
    params.radiance[params.paths.pixel[path]] += float4(contribution, 1.0);
}
