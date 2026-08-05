// The volume stage: paths whose segment has a medium event to sample —
// a scattering medium on their stack, or a null boundary where their ray
// stopped — arrive here instead of shade_surface. This is deliberately a
// separate kernel: media code in intersect or shade_surface would cost
// every scene register pressure, while a stage of its own is recorded
// only when the scene has media at all, so media-free scenes keep a
// byte-identical pass list.
//
// Today the body is the skeleton alone: nothing routes here until a
// scattering medium or null boundary can be authored, and a path that
// somehow arrives shades its held hit as ever. The real body — distance
// sampling against the stack's σ_t, phase scatter + NEE, and the
// null-crossing march (re-trace, capped) — lands with the global-medium
// and null-prim rungs.

import pathstate;

struct Params {
    Paths paths;
    Queue<uint> volumes; // this stage's input, pushed by intersect
    Queue<uint> hits;    // where surviving hits continue to shade_surface
};

[vk::push_constant]
Params params;

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void shade_volume(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.volumes.count()) {
        return;
    }
    uint path = params.volumes.entries[tid.x];
    params.hits.push(path);
}
