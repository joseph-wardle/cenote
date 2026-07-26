// Test-only kernel: proves the persistent path-reservoir record round-trips
// between the GPU's `StoredPathReservoir` and the host's byte-for-byte mirror in
// src/restir.rs, and that load/store are inverse — the CV lane carried through
// store (live from 6b-i), the pass-local weightSum dropped. Never embedded,
// never shipped. The same
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
    float* octAgreement;          // dot(v, unpack(pack(v))) per thread's direction
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

    // The CV lane rides beside the live reservoir (it is a store argument, not
    // a `Reservoir` field), so its transform pins the 16-byte-aligned float3
    // at bytes 80..92 by value — a per-component offset catches a rolled or
    // shifted lane. store still zeroes flags/pad/cvNormalization regardless of
    // input; the host asserts those come back zero.
    let cvLane = params.input[tid.x].cvAccumulator + float3(0.4, 0.5, 0.6);
    params.output[tid.x] = storePathReservoir(r, cvLane);

    // The ωₖ packing round-trips: a hashed unit direction per thread — both
    // hemispheres, the octahedron's folds included — must decode back to
    // within the 16-bit quantization cone. A broken fold or a swapped
    // component lands a direction degrees away, far below the bound.
    let h = tid.x * 2654435761u;
    let azimuth = float(h & 0xffff) * (6.28318530718 / 65536.0);
    let z = float(h >> 16) * (2.0 / 65536.0) - 1.0;
    let s = sqrt(max(0.0, 1.0 - z * z));
    let v = float3(s * cos(azimuth), s * sin(azimuth), z);
    params.octAgreement[tid.x] = dot(v, unpackUnitDirection(packUnitDirection(v)));
}
