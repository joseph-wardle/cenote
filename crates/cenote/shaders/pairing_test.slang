// Test-only kernel: the pairing-texture lookup (`pairing.slang`), driven over
// a whole texture so the host mirror in `src/pairing.rs` is pinned against the
// shader texel for texel — transform hash, forward coordinate map, i8 unpack,
// and inverse transform. A sign-extension or base-offset slip would still
// render plausibly (selection stays in-bounds, merely non-reciprocal and
// non-Gaussian), so no image-level test would catch it. Compiled at test time
// (never embedded); the texture data arrives by device address, so the pure
// lookup helpers are exercised without the binding-4 descriptor.

import pairing;

struct Params {
    uint* words;    // the host-built textures, `pairing::textures()` verbatim
    int2* out;      // one partner offset per texel of the texture under test
    uint tex;       // texture index
    uint sampleIndex;
    uint size;      // its edge — threads walk pixels (i % size, i / size)
    uint count;     // size²
};

[vk::push_constant]
Params params;

[shader("compute")]
[numthreads(64, 1, 1)]
void pairing_test(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.count) {
        return;
    }
    let l = pairingLookup(params.tex, params.sampleIndex);
    let p = int2(int(tid.x % params.size), int(tid.x / params.size));
    params.out[tid.x] = pairingDelta(l, params.words[pairingTexel(l, p)]);
}
