// Test-only kernel: evaluates the GPU's copy of the subsurface inversion
// (`subsurfaceMedium`, medium.slang) over a table of materials, so the test
// in `scene/mod.rs` can hold it against the host's `subsurface` directly.
//
// No render can make that comparison: a material is textured or it is not,
// so one route runs or the other, never both. Compiled at test time through
// the hot-reload compiler; never embedded, never shipped (`slangc.rs` lists
// what ships).

import medium;
import openpbr;

struct Params {
    Material* materials;
    float4* extinction; // xyz = σ_t, w = g
    float4* scattering; // xyz = σ_s
    uint count;
    uint pad0;
    uint pad1;
    uint pad2;
};

[vk::push_constant]
Params params;

[shader("compute")]
[numthreads(64, 1, 1)]
void subsurface_test(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.count) {
        return;
    }
    let record = subsurfaceMedium(params.materials[tid.x]);
    params.extinction[tid.x] = float4(record.sigmaT, record.g);
    params.scattering[tid.x] = float4(record.sigmaS, 0.0);
}
