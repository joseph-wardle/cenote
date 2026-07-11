// Test-only kernel: proves the persistent reservoir record round-trips
// between the GPU's `StoredReservoir` and the host's byte-for-byte mirror in
// src/restir.rs, and that load/store are inverse (minus the deliberately
// dropped weightSum). Never embedded, never shipped.
//
// A pure load→store copy would NOT catch a layout mismatch: if the GPU read a
// field from the wrong offset and wrote it back to that same wrong offset, the
// bytes would survive unchanged and the test would pass while the layouts
// disagreed. So each field is given a *distinct* transform the host predicts
// per field — a swapped or mis-aligned field then lands a value the host did
// not expect, at an offset it can see. The loaded weightSum is dumped
// separately: load must zero it regardless of what was stored.

import reservoir;
import reservoir_di;

struct Params {
    StoredReservoir* input;   // host-written records, distinctive per field
    StoredReservoir* output;  // load → per-field transform → store
    float* loadedWeightSum;   // the live reservoir's weightSum after load
    uint count;
    uint pad0;
    uint pad1;
    uint pad2;
};

[vk::push_constant]
Params params;

[shader("compute")]
[numthreads(64, 1, 1)]
void reservoir_di_test(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.count) {
        return;
    }

    // Round-trip through the live form, so both load (fields populated) and
    // store (fields serialized) are exercised on the way through.
    Reservoir<DirectLightSample> r = loadReservoir(params.input[tid.x]);
    params.loadedWeightSum[tid.x] = r.weightSum; // must be 0, always

    // A distinct transform per scalar field — the host reproduces each.
    r.sample.light += 1;
    r.sample.primitive += 2;
    r.sample.barycentrics += float2(0.25, 0.5);
    r.unbiasedWeight *= 2.0;
    r.confidence += 100.0;

    params.output[tid.x] = storeReservoir(r);
}
