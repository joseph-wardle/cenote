// Test fixture: the G-buffer round trip is surface-lossless (the step-5b
// gate, §4c decision 4). The G-buffer entry is the one input temporal sources
// differently from spatial — spatial reconstructs a neighbour's surface from
// the live path pool, temporal reconstructs last frame's from the persisted
// entry — so the same-pixel invariant the shift harness pins only transfers
// to the frame boundary if the two reconstructions agree exactly. This kernel
// proves it where it matters: in *one* invocation (per-kernel fma contraction
// can't differ between the two calls), it rebuilds the surface from the path
// pool and from the G-buffer entry `restir_temporal` actually wrote this
// frame — layout, store, and load included — and compares every field of the
// two Surfaces bit for bit, plus the raw entry words against the pool's. The
// closure is deliberately not compared field-by-field: it is a pure function
// of the material (from the hit words, compared raw) and `wo` (compared
// bitwise), so its equality is implied by what is. Not embedded — the host
// (`wavefront.rs`) compiles it on demand (`shaders::compile_fixture`).

import pathstate;
import restir_reproject;
import restir_target;
import scene;

// The fixture ran at this pixel; the low bits then carry any mismatch. A
// pixel left at 0 was a miss (or never reached) — not evaluated, not a pass.
static const uint GBUF_EVALUATED = 0x80000000;
// Raw round-trip mismatches: the entry read back differs from the pool words
// temporal stored it from — a layout drift or a corrupted store.
static const uint GBUF_RAW_HIT = 0x1;
static const uint GBUF_RAW_ORIGIN = 0x2;
static const uint GBUF_RAW_DIRECTION = 0x4;
// Reconstructed-Surface mismatches, one bit per field.
static const uint GBUF_POSITION = 0x8;
static const uint GBUF_BELOW = 0x10;
static const uint GBUF_NORMAL = 0x20;
static const uint GBUF_ORIGIN = 0x40;
static const uint GBUF_DEPTH = 0x80;
static const uint GBUF_WO = 0x100;
static const uint GBUF_IDENTITY = 0x200; // instance / thinWalled / valid

struct Params {
    Paths paths;
    Queue<uint> hits;      // primary hits, dispatched over; not consumed
    SceneTable* scene;
    GBufEntry* gbuffer;    // the entries `restir_temporal` wrote this frame
    uint* mismatch;        // out: GBUF_EVALUATED | mismatch bits (0 = miss)
};

[vk::push_constant]
Params params;

bool differ3(float3 a, float3 b) {
    return any(asuint(a) != asuint(b));
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void restir_gbuffer_test(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.hits.count()) {
        return;
    }
    uint slot = params.hits.entries[tid.x];
    uint pixel = params.paths.pixel[slot];

    let entry = params.gbuffer[pixel];
    let hit = params.paths.hit[slot];
    let origin = params.paths.origin[slot];
    let direction = params.paths.direction[slot];

    var mask = GBUF_EVALUATED;
    if (entry.hit.instance != hit.instance || entry.hit.primitive != hit.primitive
        || any(asuint(entry.hit.barycentrics) != asuint(hit.barycentrics))) {
        mask |= GBUF_RAW_HIT;
    }
    // Only xyz round-trips meaningfully: the origin's w is unused padding.
    if (differ3(entry.origin.xyz, origin.xyz)) {
        mask |= GBUF_RAW_ORIGIN;
    }
    if (any(asuint(entry.direction) != asuint(direction))) {
        mask |= GBUF_RAW_DIRECTION;
    }

    // Both reconstructions, side by side in this one invocation: the pool's
    // (spatial's view of this pixel) and the persisted entry's (temporal's
    // view of it next frame).
    Surface a = reconstructSurface(params.paths, params.scene, slot);
    Surface b = reconstructSurfaceFrom(params.scene, entry.hit, entry.direction,
                                       entry.origin.xyz);
    if (differ3(a.position, b.position)) {
        mask |= GBUF_POSITION;
    }
    if (differ3(a.below, b.below)) {
        mask |= GBUF_BELOW;
    }
    if (differ3(a.normal, b.normal)) {
        mask |= GBUF_NORMAL;
    }
    if (differ3(a.origin, b.origin)) {
        mask |= GBUF_ORIGIN;
    }
    if (asuint(a.depth) != asuint(b.depth)) {
        mask |= GBUF_DEPTH;
    }
    if (differ3(a.wo, b.wo)) {
        mask |= GBUF_WO;
    }
    if (a.instance != b.instance || a.thinWalled != b.thinWalled
        || a.valid != b.valid) {
        mask |= GBUF_IDENTITY;
    }
    params.mismatch[pixel] = mask;
}
