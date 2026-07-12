// Sobol-Burley sampling — hash-based Owen-scrambled Sobol points
// ("Practical Hash-Based Owen Scrambling", Burley, JCGT 2020) — the
// renderer's one source of randomness.
//
// Stateless by design: every value is a pure function of (pixel, sample
// index, dimension), so path state carries no RNG state and any decision
// can be replayed by re-asking with the same keys — the property bitwise
// determinism and, later, GRIS shift mappings rest on.
//
// Rather than assigning each named dimension its own high-order Sobol
// dimension, every dimension gets an independently shuffled and scrambled
// copy of the same 2D Sobol sequence ("padding" — the construction Cycles
// and pbrt-v4 ship as their defaults).

// -- The dimension registry --------------------------------------------
// Every random decision draws from a named dimension — never allocated by
// call order, where reordering two sample_* calls would silently change
// every decision downstream.
//
// The numbering carries reserved headroom: a decision added later claims a
// spare slot instead of renumbering the ones after it, since every renumber
// changes every image ever rendered (and forces a golden regeneration).
// Headroom is free here — each dimension is an independently shuffled copy
// of the same sequence (see below), keyed by a hash, so unused slots cost
// nothing and spacing decisions apart never correlates them.

// Pre-path dimensions, drawn once per camera ray. The block reserves room
// ahead of the bounces for shutter time (motion blur).
public static const uint CAMERA_JITTER = 0; // sub-pixel ray position
public static const uint CAMERA_LENS = 1;   // point on the thin-lens disk
static const uint PATH_BASE = 8;            // first per-bounce dimension

// Per-bounce decisions get one slot each in fixed-size blocks: bounce b's
// slot s is dimension `bounceDimension(b, s)`. The block is sized past its
// named slots so a later lobe (transmission's Fresnel choice, a light BVH's
// RIS candidates, a GRIS shift decision) can take a reserved slot without
// shifting the next bounce's block.
public static const uint BSDF_LOBE = 0; // which sampling technique scatters
public static const uint BSDF_DIR = 1;  // scattered direction
public static const uint NEE_LIGHT = 2; // environment vs the list, and which light
public static const uint NEE_POINT = 3; // point on the light, or env texel
public static const uint RR = 4;        // Russian-roulette survival
static const uint BOUNCE_SLOTS = 8;     // 5 named + 3 reserved

public uint bounceDimension(uint bounce, uint slot) {
    return PATH_BASE + bounce * BOUNCE_SLOTS + slot;
}

// ReSTIR initial-RIS candidate dimensions. The reservoir streams M light
// candidates at the primary hit, each needing its own light-selection,
// point-on-light, and reservoir-acceptance draws. They sit in a dedicated high
// block, clear of the per-bounce path dimensions above — the bounce-0 scatter
// shares this same vertex, and the M candidates must not correlate with it.
// Candidate c's slot s is `restirCandidateDimension(c, s)`.
public static const uint RESTIR_BASE = 256;
static const uint RESTIR_SLOTS = 4;   // 3 named + 1 reserved
public static const uint RESTIR_LIGHT = 0; // environment vs the list, and which light
public static const uint RESTIR_POINT = 1; // point on the light, or env texel (2D)
public static const uint RESTIR_ACCEPT = 2; // reservoir stream acceptance

public uint restirCandidateDimension(uint candidate, uint slot) {
    return RESTIR_BASE + candidate * RESTIR_SLOTS + slot;
}

// The one internalized BSDF-sampled candidate (D-088). It draws a scattered
// direction from the closure and reconnects to whatever light that direction
// reaches, so it needs a lobe pick, a 2D direction, and its own reservoir
// acceptance — its own dedicated slots, distinct from both the bounce-0
// continuation scatter (or it would just re-trace the same ray) and every light
// candidate above. Fixed base, clear of the M-candidate block for M up to 64,
// so growing M never shifts it.
public static const uint RESTIR_BSDF_BASE = 512;
public static const uint RESTIR_BSDF_LOBE = 0;   // which technique scatters
public static const uint RESTIR_BSDF_DIR = 1;    // scattered direction (2D: slots 1,2)
public static const uint RESTIR_BSDF_ACCEPT = 3; // reservoir stream acceptance

public uint restirBsdfDimension(uint slot) {
    return RESTIR_BSDF_BASE + slot;
}

// The one delta-light next-event draw `restir_resolve` adds outside the
// reservoir (D-088): a single selection of a delta light ∝ power. A delta light
// is a point or a direction, so its connection has no positional sampling — one
// 1D draw picks the light and that is the whole sample. Fixed, immediately past
// the BSDF block (which owns slots 0..3), so neither shifts the other.
public static const uint RESTIR_DELTA = RESTIR_BSDF_BASE + 4;

public uint restirDeltaDimension() {
    return RESTIR_DELTA;
}

// -- Implementation ----------------------------------------------------

// Generator-matrix columns of the Sobol sequence's second dimension, one
// column per index bit — Pascal's triangle mod 2, from the recurrence
// m_k = 2·m_{k−1} ⊕ m_{k−1} of primitive polynomial x + 1. The first
// dimension needs no table: its matrix is the identity.
static const uint SOBOL_DIM1[32] = {
    0x80000000u, 0xc0000000u, 0xa0000000u, 0xf0000000u,
    0x88000000u, 0xcc000000u, 0xaa000000u, 0xff000000u,
    0x80800000u, 0xc0c00000u, 0xa0a00000u, 0xf0f00000u,
    0x88880000u, 0xcccc0000u, 0xaaaa0000u, 0xffff0000u,
    0x80008000u, 0xc000c000u, 0xa000a000u, 0xf000f000u,
    0x88008800u, 0xcc00cc00u, 0xaa00aa00u, 0xff00ff00u,
    0x80808080u, 0xc0c0c0c0u, 0xa0a0a0a0u, 0xf0f0f0f0u,
    0x88888888u, 0xccccccccu, 0xaaaaaaaau, 0xffffffffu,
};

// lowbias32 (the Hash Prospector search): a fast, well-mixed 32-bit hash.
// Derives the mutually independent seeds each (pixel, dimension) key needs.
uint lowbias32(uint x) {
    x ^= x >> 16;
    x *= 0x21f0aaadu;
    x ^= x >> 15;
    x *= 0xd35a2d97u;
    x ^= x >> 15;
    return x;
}

// Burley's improved Laine-Karras permutation: a seeded random permutation
// of the integers in which each bit is affected only by the bits *below*
// it — which, applied in reversed bit order, is exactly the tree structure
// of an Owen scramble.
uint laineKarras(uint x, uint seed) {
    x += seed;
    x ^= x * 0x6c50b47cu;
    x ^= x * 0xb82f1e52u;
    x ^= x * 0xc7afe638u;
    x ^= x * 0x8d22f6e6u;
    return x;
}

// Owen-scramble x: permute each bit conditioned on the bits above it.
// Scrambling per-axis values decorrelates them without breaking their
// stratification; scrambling a sample index shuffles the sequence order.
uint nestedUniformScramble(uint x, uint seed) {
    return reversebits(laineKarras(reversebits(x), seed));
}

// The index-th point of the base 2D Sobol sequence, one axis at a time.
// Axis 0's generator matrix is the identity, so its value is the index
// bit-reversed (the van der Corput sequence).
uint sobol(uint index, uint axis) {
    if (axis == 0) {
        return reversebits(index);
    }
    uint value = 0;
    for (uint bit = 0; index != 0; bit++, index >>= 1) {
        if ((index & 1) != 0) {
            value ^= SOBOL_DIM1[bit];
        }
    }
    return value;
}

// Top 24 bits → [0, 1): every step is exact in f32 and 1.0 never occurs.
float toUnitFloat(uint x) {
    return float(x >> 8) * (1.0 / 16777216.0);
}

// One axis of the Owen-scrambled point at a (shuffled) index.
float scrambledSobol(uint index, uint axis, uint seed) {
    return toUnitFloat(nestedUniformScramble(sobol(index, axis), seed));
}

// A stateless uniform for the transparency chain: stochastic pass-through
// is pure visibility — stratification buys nothing across an unbounded,
// traversal-ordered sequence of crossings — so it draws from a plain hash
// stream rather than a Sobol dimension. Keyed by (pixel, sample, bounce,
// crossing): every decision independent, every decision replayable.
public float hashUniform(uint pixel, uint sampleIndex, uint bounce, uint crossing) {
    var h = lowbias32(pixel ^ lowbias32(sampleIndex ^ 0x9e3779b9u));
    h = lowbias32(h ^ (bounce << 8 | crossing));
    return toUnitFloat(h);
}

// A dimension's 2D point for one pixel's sampleIndex-th sample. The index
// shuffle gives every pixel its own ordering of an Owen-scrambled sequence:
// each pixel's samples are individually well-stratified (so its average
// converges fast) yet independent of every other pixel's (so error stays
// white noise across the frame). The hex salts only separate the shuffle
// stream from each axis's scramble stream.
public float2 sample_2d(uint pixel, uint sampleIndex, uint dimension) {
    let seed = lowbias32(pixel ^ lowbias32(dimension));
    let index = nestedUniformScramble(sampleIndex, lowbias32(seed ^ 0xf8ade99au));
    return float2(scrambledSobol(index, 0, lowbias32(seed ^ 0xe0aaaf76u)),
                  scrambledSobol(index, 1, lowbias32(seed ^ 0x94964d4eu)));
}

// The 1D counterpart, for scalar decisions (lobe choice, Russian roulette).
public float sample_1d(uint pixel, uint sampleIndex, uint dimension) {
    let seed = lowbias32(pixel ^ lowbias32(dimension));
    let index = nestedUniformScramble(sampleIndex, lowbias32(seed ^ 0xbff95bfeu));
    return scrambledSobol(index, 0, lowbias32(seed ^ 0x635fc02fu));
}
