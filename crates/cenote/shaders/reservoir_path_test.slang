// Test-only kernel: proves the persistent path-reservoir record round-trips
// between the GPU's `StoredPathReservoir` and the host's byte-for-byte mirror in
// src/restir.rs, and that load/store are inverse — the reserved CV lane zeroed,
// the pass-local weightSum dropped. Never embedded, never shipped. The same
// reasoning the reservoir primitive's own test rests on applies: a pure
// load→store copy would survive a symmetric layout mismatch unseen, so each
// field gets a *distinct* transform the host predicts per field, and any swapped
// or mis-aligned field lands a value the host did not expect at an offset it sees.
// The 96 B path record has three `float3` fields on 16-byte lanes — exactly the
// packing the fixture exists to pin.

import reservoir;
import reservoir_path;
import pathstate;

struct Params {
    StoredPathReservoir* input;   // host-written records, distinctive per field
    StoredPathReservoir* output;  // load → per-field transform → store
    float* loadedWeightSum;       // the live reservoir's weightSum after load
    uint count;
    uint pad0;
    uint pad1;
    uint pad2;
};

[vk::push_constant]
Params params;

[shader("compute")]
[numthreads(64, 1, 1)]
void reservoir_path_test(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.count) {
        return;
    }

    // Round-trip through the live form, exercising load (fields populated) and
    // store (fields serialized, reserved lanes zeroed) on the way through.
    Reservoir<PathSample> r = loadPathReservoir(params.input[tid.x]);
    params.loadedWeightSum[tid.x] = r.weightSum; // must be 0, always

    // A distinct transform per scalar field — the host reproduces each. A
    // per-component offset on the float3 fields catches a rolled vector too.
    r.sample.rcVertex.instance += 1;
    r.sample.rcVertex.primitive += 2;
    r.sample.rcVertex.barycentrics += float2(0.25, 0.5);
    r.sample.rcVertexRadiance += float3(1.0, 2.0, 3.0);
    r.sample.rcVertexWi += 4;
    r.sample.F += float3(0.1, 0.2, 0.3);
    r.sample.neeLightPdf *= 2.0;
    r.sample.initRandomSeed += 5;
    r.sample.rcVertexRandomSeed += 6;
    r.sample.cachedJacobian *= 3.0;
    r.sample.reserved += 7;
    r.unbiasedWeight *= 2.0;
    r.confidence += 100.0;

    // store zeroes flags/pad/CV regardless of input — the host asserts they
    // come back zero, which pins both that those lanes exist and that the step-6
    // slots start clean.
    params.output[tid.x] = storePathReservoir(r);
}
