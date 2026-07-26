// The full-path instantiation of the generic reservoir: the concrete Sample
// type M6's reuse path holds, and the persistent per-pixel record its buffers
// store. `reservoir.slang` stays index-agnostic; this is where the abstract
// `Reservoir<Sample>` becomes a reservoir of whole *light paths*. It is the
// former DI reservoir generalized from a length-2 direct connection to a path
// of any length — the generalization D-086 built the DI reservoir as the
// dormant base case of. As of step 1 (D-124) the DI path routes through this
// reservoir restricted to length-2 (NEE) samples and the separate DI reservoir
// is retired: a direct-lighting sample is now a length-2 path whose
// reconnection vertex `rcVertex` is the light point, and the NEE-specific
// interpretation of that vertex (the light-id sentinels, the environment
// predicate) lives in `restir_scene.slang`. The multi-vertex fields below light
// up from step 2 onward. The host mirror (src/restir.rs) reproduces
// `StoredPathReservoir` field for field — the AoS buffer is an array of it.
//
// -- Rosetta stone ------------------------------------------------------
// The layout follows Enhanced's Algorithm 1 (Lin, Kettunen, Wyman, I3D 2026),
// mapped to readable identifiers, with three deliberate cenote divergences
// noted where they occur. The path sample is held in the *re-evaluable
// reconnection form* — the reconnection vertex plus the seeds a random-replay
// shift needs — never a serialized path: constant per-pixel storage, and the
// shift re-resolves the `Closure` at the reconnection `Hit` on demand (D-131).
//
//   rcVertex          xₖ  — the reconnection vertex, the existing `Hit`
//                     (instance+primitive+barycentrics). Reused verbatim
//                     (D-128): the shift re-shades from it exactly as primary
//                     shading recomputes from a `Hit`. [cenote divergence: the
//                     paper packs barycentrics to 2×16-bit unorm; cenote keeps
//                     `Hit`'s float2 for one shared vertex form.]
//   rcVertexWi        ωₖ  — outgoing direction at the reconnection vertex,
//                     oct-encoded into one word (packUnitDirection below). Live
//                     from step 3a: the shift re-shades fₖ against it (D-131).
//   rcVertexRadiance  Lₖ  — the *incident-side* suffix cached at the
//                     reconnection vertex: everything the path carries beyond
//                     fₖ — the suffix throughputs, pdfs, and shift-invariant
//                     MIS folds — with fₖ·|cosθₖ| deliberately excluded, so
//                     the shift recomputes F' = fₖ(shifted wo)·|cosθₖ|·Lₖ
//                     without re-tracing the tail (`reconnectionOutgoing`,
//                     restir_target.slang).
//   F                 F   — the RGB path integrand in *this* pixel's domain.
//                     The target is scalar p̂ = luminance(F); F carries colour so
//                     resolve shades in colour and the §6.3 colour-noise fix
//                     (vector resampling weights) is a stored field away (D-133).
//                     Live from step 3b on, for every indirect kind: generation
//                     stores the own-domain integrand, a spatial shift rewrites
//                     it with the destination's, and the own-domain target of a
//                     stored sample is luminance(F) — trusted rather than
//                     re-formed wherever re-forming would re-trace the replay
//                     (D-138; the same-pixel bit-identity gate proves the two
//                     agree bit for bit).
//   initRandomSeed    — the per-path replay seed (live from step 3b, D-138):
//                     the walk's sampling draws are a pure function of
//                     (seed, dimension), so a shift replays a k>2 sample's
//                     prefix bounces — or a replay-kind sample's whole path,
//                     terminal redraw included (step 3c, D-139) — at any
//                     pixel with exactly the source's draws. Doubles as the
//                     duplication-map hash key (dup maps deferred,
//                     preview-only; deferrals.md).
//   rcVertexRandomSeed — reserved. Designed for a per-segment reseed, but the
//                     per-path seed above covers the terminal redraw too
//                     (dimensions are per-bounce), so it stays zero — D-128
//                     headroom it remains.
//   cachedJacobian    — the *source half* of the shift Jacobian, |cosθₖ|/d² of
//                     the reconnection segment where the sample was drawn
//                     (live from step 3b): a k>2 sample's x_{k−1} is a replayed
//                     point no reusing domain can recompute, so the source term
//                     rides the sample and the shift divides its own half by it
//                     (`reconnectionEnd`, restir_target.slang). Same-pixel
//                     reconnection is Jacobian 1 to the last-place bit (the
//                     step-2 validation invariant, D-125).
//   neeLightPdf       — the NEE light pdf, kept UNPACKED (Enhanced Alg 1): it
//                     feeds the path MIS at resolve and must not be folded into
//                     `cachedJacobian`.
//   unbiasedWeight    W_Y — the UCW (from `reservoir.slang`).
//   confidence        M   — the generalized sample count. [cenote divergence:
//                     the paper packs M to 8 bits in pathFlags; cenote keeps the
//                     float `confidence` its reservoir primitive already carries,
//                     so temporal decay stays fractional (reservoir.slang).]
//   cvAccumulator/cvNormalization — the ReSTCV control-variate slot (D-127): a
//                     per-reservoir accumulated-colour estimate, placed LAST so
//                     step 6 could pin its update math without moving any field
//                     before it (D-128). Live from step 6a: the spatial pass
//                     accumulates its merge stream's vector weights
//                     w⃗ = w_scalar·F/luminance(F) here (§6.3 colour-noise fix)
//                     and resolve shades from the lane — pass-local, since the
//                     spatial scratch never persists a frame. Zero-CV shading
//                     (the host toggle off) is plain ReSTIR PT — the
//                     degenerate, D-130. `cvNormalization` stays reserved-zero
//                     until 6b.
//
// -- The pass-local weightSum invariant carries over --------------------
// `StoredPathReservoir` stores {sample, W, M} plus the reserved CV slot, and
// deliberately NOT weightSum — it is pass-local and rebuilt from scratch every
// pass (reservoir.slang documents why at length). Persisting it would
// double-count and silently bias.

import reservoir;
import pathstate;

// One resampled light path, held in the re-evaluable reconnection form. The
// fields are laid out in std430 16-byte lanes (each `float3` on a 16-byte
// boundary, a trailing scalar filling its lane) so the host mirror packs
// identically without alignment surprises — the round-trip fixture proves the
// offsets agree, which the size assert alone cannot.
public struct PathSample : IDefaultInitializable {
    public Hit rcVertex;             // 0..16  xₖ — reconnection vertex
    public float3 rcVertexRadiance;  // 16..28 Lₖ — incident suffix beyond fₖ
    public uint rcVertexWi;          // 28..32 ωₖ — oct-encoded outgoing dir
    public float3 F;                 // 32..44 F  — RGB path integrand, this domain
    public float neeLightPdf;        // 44..48 the NEE light pdf, kept unpacked
    public uint initRandomSeed;      // 48..52 replay/dup-map key for early segments
    public uint rcVertexRandomSeed;  // 52..56 replay seed from xₖ onward
    public float cachedJacobian;     // 56..60 shift Jacobian terms, pre-multiplied
    public uint reserved;            // 60..64 the path-kind/terminal/shiftable
                                     //   bits (restir_scene.slang); rest reserved
};

// -- Unit-direction packing (ωₖ) ----------------------------------------
// The standard octahedral mapping, 2×16-bit unorm in one word: project onto
// the |x|+|y|+|z| = 1 octahedron, fold the lower hemisphere over the diagonal.
// ~1e-4 rad worst-case error, negligible against the ≥0.2-roughness lobes the
// pair criteria admit to reuse; the candidate stage evaluates its own targets
// through the *decoded* direction, so every domain re-shades the exact stored
// ωₖ and generation can never drift from resolve.

float2 octSignNotZero(float2 v) {
    return float2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
}

public uint packUnitDirection(float3 v) {
    let p = v.xy / (abs(v.x) + abs(v.y) + abs(v.z));
    let e = v.z >= 0.0 ? p : (1.0 - abs(p.yx)) * octSignNotZero(p);
    let q = uint2(round(saturate(e * 0.5 + 0.5) * 65535.0));
    return q.x | (q.y << 16);
}

public float3 unpackUnitDirection(uint packed) {
    let e = float2(uint2(packed & 0xffff, packed >> 16)) * (2.0 / 65535.0) - 1.0;
    var v = float3(e, 1.0 - abs(e.x) - abs(e.y));
    if (v.z < 0.0) {
        v.xy = (1.0 - abs(v.yx)) * octSignNotZero(v.xy);
    }
    return normalize(v);
}

// The persistent per-pixel path reservoir record: {sample, W, M} + the reserved
// ReSTCV lane = 96 B. weightSum is deliberately absent (pass-local; see above).
// This is the byte layout src/restir.rs mirrors and the path reservoir buffers
// are arrays of. Larger than Enhanced's 64 B by three named choices — `Hit`'s
// float2 barycentrics, the float `confidence`, and the reserved CV lane — each
// a clarity/reuse win; packing back toward 64 B (2×16u barycentrics, 8-bit M) is
// the deferred size optimization if the per-view × N-viewport budget bites.
public struct StoredPathReservoir {
    public PathSample sample;      // 0..64  — Y
    public float unbiasedWeight;   // 64..68 — W_Y
    public float confidence;       // 68..72 — M
    public uint flags;             // 72..76 — reserved (path/reconnection flags)
    public uint pad0;              // 76..80 — pad: 16-byte-aligns the CV lane
    public float3 cvAccumulator;   // 80..92 — ReSTCV accumulated colour (step 6)
    public float cvNormalization;  // 92..96 — ReSTCV running weight (step 6)
};

// Persist a live path reservoir, dropping the pass-local weightSum and zeroing
// the reserved slots until the steps that own them fill them in.
public StoredPathReservoir storePathReservoir(Reservoir<PathSample> r) {
    StoredPathReservoir s;
    s.sample = r.sample;
    s.unbiasedWeight = r.unbiasedWeight;
    s.confidence = r.confidence;
    s.flags = 0;
    s.pad0 = 0;
    s.cvAccumulator = float3(0);
    s.cvNormalization = 0;
    return s;
}

// Read a persisted path reservoir back for reuse. weightSum starts at zero, to
// be rebuilt by the reusing pass; restoring a stored weightSum would double-
// count and bias (see reservoir.slang). The reserved slots are ignored here
// until their owning steps read them.
public Reservoir<PathSample> loadPathReservoir(StoredPathReservoir s) {
    Reservoir<PathSample> r = reservoirInit<PathSample>();
    r.sample = s.sample;
    r.unbiasedWeight = s.unbiasedWeight;
    r.confidence = s.confidence;
    return r;
}
