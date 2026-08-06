// The volume stage: paths whose segment has a medium event to sample —
// a scattering medium on their stack, or a null boundary where their ray
// stopped — arrive here from intersect instead of shade_surface, and the
// ones that survive continue to shade_surface through the hit queue.
//
// A stage of its own rather than a branch in shade_surface: it is
// recorded only when the scene has media, so media-free scenes keep a
// byte-identical pass list and no scene pays register pressure for
// medium code it never runs.
//
// Skeleton only. Nothing routes here until a scattering medium or null
// boundary can be authored; distance sampling, phase scatter with NEE,
// and the capped null-crossing march land with those rungs.

import pathstate;

struct Params {
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
    params.hits.push(params.volumes.entries[tid.x]);
}
