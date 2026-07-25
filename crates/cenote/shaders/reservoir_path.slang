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
//                     oct-encoded into one word (encode/decode land in step 2).
//   rcVertexRadiance  Lₖ  — incident radiance cached at the reconnection vertex
//                     (the sub-path beyond xₖ), so the reconnection shift
//                     recomputes the shifted contribution without re-tracing
//                     the tail: F' = f_local(shifted)·Lₖ.
//   F                 F   — the RGB path integrand in *this* pixel's domain.
//                     The target is scalar p̂ = luminance(F); F carries colour so
//                     resolve shades in colour and the §6.3 colour-noise fix
//                     (vector resampling weights) is a stored field away (D-133).
//   initRandomSeed    — the initial-sample RNG seed: the random-replay key for
//                     the early (near-specular) segments, and the duplication-
//                     map hash key (dup maps deferred, preview-only — the seed
//                     is kept so reviving them needs no re-layout; deferrals.md).
//   rcVertexRandomSeed — the RNG seed replayed from the reconnection vertex on.
//   cachedJacobian    — the shift Jacobian terms pre-multiplied into one float.
//                     Same-pixel reconnection has Jacobian 1 (the step-2
//                     validation invariant, D-125).
//   neeLightPdf       — the NEE light pdf, kept UNPACKED (Enhanced Alg 1): it
//                     feeds the path MIS at resolve and must not be folded into
//                     `cachedJacobian`.
//   unbiasedWeight    W_Y — the UCW (from `reservoir.slang`).
//   confidence        M   — the generalized sample count. [cenote divergence:
//                     the paper packs M to 8 bits in pathFlags; cenote keeps the
//                     float `confidence` its reservoir primitive already carries,
//                     so temporal decay stays fractional (reservoir.slang).]
//   cvAccumulator/cvNormalization — the ReSTCV control-variate slot (D-127): a
//                     per-reservoir accumulated-colour estimate. Reserved and
//                     placed LAST so the step-6 deep-read can pin its exact
//                     update math without moving any field before it — the whole
//                     point of designing it in now (D-128). Zero until step 6,
//                     where zero-CV is plain ReSTIR PT (the degenerate, D-130).
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
    public float3 rcVertexRadiance;  // 16..28 Lₖ — radiance cached beyond xₖ
    public uint rcVertexWi;          // 28..32 ωₖ — oct-encoded outgoing dir
    public float3 F;                 // 32..44 F  — RGB path integrand, this domain
    public float neeLightPdf;        // 44..48 the NEE light pdf, kept unpacked
    public uint initRandomSeed;      // 48..52 replay/dup-map key for early segments
    public uint rcVertexRandomSeed;  // 52..56 replay seed from xₖ onward
    public float cachedJacobian;     // 56..60 shift Jacobian terms, pre-multiplied
    public uint reserved;            // 60..64 reserved (reconnection classification)
};

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
