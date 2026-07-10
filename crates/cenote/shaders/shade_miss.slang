// The escaped-ray stage: look up the environment's radiance along the ray
// and add it, MIS-weighted — an escaped BSDF-sampled ray is the BSDF
// strategy's way of reaching the sky, competing with the next-event
// environment samples shade_surface sends. Camera rays keep weight 1 (no
// other strategy competed). Escape is terminal: this add carries the
// pixel's alpha 1 — and, when the wave carries AOV targets, closes the
// pixel's denoiser guides (white albedo, no normal) and stamps a
// first-hit miss's depth as +inf.

import lights;
import pathstate;
import scene;

struct Params {
    Paths paths;
    Queue<uint> misses;
    SceneTable* scene;
    float4* radiance;   // the film's per-pixel accumulator for this wave
    AovTable* aov;      // the wave's AOV targets; enabled = 0 skips them
    uint lightSampling; // LIGHT_SAMPLING_*: which strategies reach the lights
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

    uint pixel = params.paths.pixel[path];
    if (params.aov.enabled != 0) {
        // throughput.w = 0 marks a camera ray — the same test the MIS
        // weight above leans on.
        let cameraRay = throughput.w == 0.0;
        if (cameraRay) {
            // Nothing was hit, so the pixel has no depth: +inf, the
            // value the film's guard deliberately lets through.
            params.aov.depth[pixel] += asfloat(0x7f800000);
        }
        // Whatever feature throughput the specular chain left records the
        // environment: albedo white (OIDN's own guidance for the
        // background — the sky's radiance is signal, not reflectance),
        // and no surface means no normal.
        let guide = cameraRay ? float3(1.0) : params.aov.guide[pixel].xyz;
        params.aov.albedo[pixel] += float4(guide, 0.0);
    }

    let contribution = throughput.xyz * params.scene.env.radiance(direction) * weight;
    // Pixel-owned add — each pixel has exactly one path per wave, and a
    // path escapes at most once — so the nondeterministic queue order
    // never reaches the image.
    params.radiance[pixel] += float4(contribution, 1.0);
}
