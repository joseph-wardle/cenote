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

// Sub-pixel position of the pixel's camera ray.
public static const uint CAMERA_JITTER = 0;

// Per-bounce decisions get one slot each in fixed blocks after the camera
// dimensions: bounce b's slot s is dimension `bounceDimension(b, s)`. Every
// M1 slot is named up front even though they go live across steps 7–9,
// because inserting one later would renumber the rest — silently changing
// every image ever rendered.
public static const uint BSDF_LOBE = 0; // which lobe scatters (step 9)
public static const uint BSDF_DIR = 1; // scattered direction
public static const uint NEE_LIGHT = 2; // which light to sample
public static const uint NEE_POINT = 3; // point on that light
public static const uint RR = 4; // Russian-roulette survival
static const uint BOUNCE_SLOTS = 5;

public uint bounceDimension(uint bounce, uint slot) {
    return 1 + bounce * BOUNCE_SLOTS + slot;
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
