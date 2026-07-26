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

// The one internalized BSDF-sampled candidate's reservoir acceptance (D-088).
// The draw itself — the lobe pick and 2D direction that used to occupy slots
// 0..2 — moved to the per-path replay stream at step 3b (D-138): the scatter
// at x₁ is the first bounce a k>2 shift replays, so its uniforms must be a
// function of the stored seed, not of this pixel's ranked stream. Only the
// acceptance (a reservoir-local decision, never replayed) stays here, at its
// original slot so the delta term after it never shifts.
public static const uint RESTIR_BSDF_BASE = 512;
public static const uint RESTIR_BSDF_ACCEPT = 3; // reservoir stream acceptance
// Slots 0..2 were the T2/T3a lobe + direction draws (now seeded, D-138);
// slot 4 is the delta term below; slot 5 was the T2 aggregate tail's
// acceptance, retired when step 3a streamed the tail per event (the walk's
// acceptances live in the RESTIR_TAIL block below).

public uint restirBsdfDimension(uint slot) {
    return RESTIR_BSDF_BASE + slot;
}

// The one delta-light next-event draw `restir_resolve` adds outside the
// reservoir (D-088): a single selection of a delta light ∝ power. A delta light
// is a point or a direction, so its connection has no positional sampling — one
// 1D draw picks the light and that is the whole sample. Fixed, immediately past
// the BSDF acceptance (slots 0..3; the tail acceptance sits at 5), so none of
// the three shift each other.
public static const uint RESTIR_DELTA = RESTIR_BSDF_BASE + 4;

public uint restirDeltaDimension() {
    return RESTIR_DELTA;
}

// ReSTIR spatial-reuse dimensions (step 4). The spatial stage gathers k
// neighbours around the primary hit and streams each — plus the pixel's own
// canonical reservoir — through a fresh reservoir under defensive pairwise MIS.
// Each neighbour needs a 2D disk offset (which pixel) and a 1D reservoir-merge
// acceptance; the canonical needs one acceptance of its own. A dedicated high
// block, clear of the initial-RIS candidate (256) and BSDF (512) blocks, so the
// spatial draws never correlate with the candidates that built the reservoirs
// being reused. Neighbour n's slot s is `restirSpatialDimension(n, s)`.
public static const uint RESTIR_SPATIAL_BASE = 768;
static const uint RESTIR_SPATIAL_SLOTS = 4;      // 3 named + 1 reserved
public static const uint RESTIR_SPATIAL_OFFSET = 0; // disk offset on the neighbour (2D: slots 0,1)
public static const uint RESTIR_SPATIAL_ACCEPT = 2; // reservoir-merge acceptance

public uint restirSpatialDimension(uint neighbor, uint slot) {
    return RESTIR_SPATIAL_BASE + neighbor * RESTIR_SPATIAL_SLOTS + slot;
}

// The canonical reservoir's own merge acceptance — one 1D draw, fixed past the
// neighbour block for k up to 64 so growing k never shifts it.
public static const uint RESTIR_SPATIAL_CANONICAL = RESTIR_SPATIAL_BASE + 64 * RESTIR_SPATIAL_SLOTS;

public uint restirSpatialCanonicalDimension() {
    return RESTIR_SPATIAL_CANONICAL;
}

// ReSTIR temporal-reuse dimensions (step 5). The temporal stage folds last
// frame's `prev` reservoir into this frame's `cand` at the same pixel, so it
// needs just two reservoir-merge acceptance draws: one for the canonical
// (`cand`), one for the single neighbour (`prev`). A dedicated block past the
// spatial canonical (which sits at `RESTIR_SPATIAL_BASE + 64·4` = 1024), so the
// combine's acceptance never correlates with the candidate or spatial streams
// that ran at the same vertex (D-091, D-094).
public static const uint RESTIR_TEMPORAL_BASE = 1280;
public static const uint RESTIR_TEMPORAL_CANONICAL = 0; // `cand`'s merge acceptance
public static const uint RESTIR_TEMPORAL_PREV = 1;      // `prev`'s merge acceptance

public uint restirTemporalDimension(uint slot) {
    return RESTIR_TEMPORAL_BASE + slot;
}

// ReSTIR tail-event acceptances (step 3a, D-137). The candidate walk streams
// every lighting event past the primary scatter as its own reservoir
// candidate, so each walk bounce needs two acceptance draws: the vertex's NEE
// connection, and the scatter-found event (an emission hit, or the sky)
// discovered arriving at it — mutually exclusive per bounce, so they share a
// slot. The walk's *sampling* draws stay in the per-bounce path block
// (`bounceDimension`), untouched, exactly as the T2 tail drew them — this
// block adds only the reservoir acceptances, clear of every other family.
public static const uint RESTIR_TAIL_BASE = 1536;
static const uint RESTIR_TAIL_SLOTS = 4; // 2 named + 2 reserved
public static const uint RESTIR_TAIL_NEE_ACCEPT = 0;     // the vertex's NEE event
public static const uint RESTIR_TAIL_SCATTER_ACCEPT = 1; // emission hit / sky escape

public uint restirTailDimension(uint bounce, uint slot) {
    return RESTIR_TAIL_BASE + bounce * RESTIR_TAIL_SLOTS + slot;
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

// A dimension's Owen-scrambled point at an *explicit* sample index — the
// scramble half of the sampler, split from the choice of *which* index so that
// choice is pluggable: the per-dimension padding shuffle of sample_2d/1d, or
// the shared blue-noise ranking of sample_ranked_2d/1d (D-095). The scramble is
// keyed per (pixel, dimension) either way — the padding that decorrelates the
// axes and neighbouring pixels' point *values*. The hex salts only separate
// each axis's scramble stream.
float2 scrambledPoint2d(uint pixel, uint index, uint dimension) {
    let seed = lowbias32(pixel ^ lowbias32(dimension));
    return float2(scrambledSobol(index, 0, lowbias32(seed ^ 0xe0aaaf76u)),
                  scrambledSobol(index, 1, lowbias32(seed ^ 0x94964d4eu)));
}

float scrambledPoint1d(uint pixel, uint index, uint dimension) {
    let seed = lowbias32(pixel ^ lowbias32(dimension));
    return scrambledSobol(index, 0, lowbias32(seed ^ 0x635fc02fu));
}

// A dimension's 2D point for one pixel's sampleIndex-th sample. The index
// shuffle gives every pixel its own ordering of an Owen-scrambled sequence:
// each pixel's samples are individually well-stratified (so its average
// converges fast) yet independent of every other pixel's (so error stays
// white noise across the frame). The salt separates the shuffle stream from
// the scramble streams. The ReSTIR light streams take the blue-noise-ranked
// counterpart below; everything else (camera jitter, the path tracer) keeps
// this per-dimension padding.
public float2 sample_2d(uint pixel, uint sampleIndex, uint dimension) {
    let seed = lowbias32(pixel ^ lowbias32(dimension));
    let index = nestedUniformScramble(sampleIndex, lowbias32(seed ^ 0xf8ade99au));
    return scrambledPoint2d(pixel, index, dimension);
}

// The 1D counterpart, for scalar decisions (lobe choice, Russian roulette).
public float sample_1d(uint pixel, uint sampleIndex, uint dimension) {
    let seed = lowbias32(pixel ^ lowbias32(dimension));
    let index = nestedUniformScramble(sampleIndex, lowbias32(seed ^ 0xbff95bfeu));
    return scrambledPoint1d(pixel, index, dimension);
}

// -- The per-path replay re-key (step 3b, D-138) -----------------------
// The candidate walk's *sampling* draws — the scatter at x₁ and every tail
// vertex's NEE/lobe/direction/roulette decision — are keyed by a per-path
// seed instead of (pixel, sampleIndex), because a k>2 shift replays the
// prefix bounces at a *different* pixel and must reproduce the source's
// draws bit for bit: generation and replay call the same pure function of
// (seed, dimension), so they cannot diverge. The seed rides the sample
// (`PathSample.initRandomSeed`). Dimension numbering keeps the per-bounce
// path registry (`bounceDimension`) — only the keying changes. The walk's
// reservoir *acceptances* are pixel-local decisions, never replayed, and
// stay on the ranked stream. The price is per-sample stratification on the
// walk (each path's draws are hash-independent, not a progressive Sobol
// point) — the resampling machine, not tail stratification, carries the
// convergence, exactly as the reference ReSTIR PT implementations accept.

public uint pathReplaySeed(uint pixel, uint sampleIndex) {
    return lowbias32(pixel ^ lowbias32(sampleIndex ^ 0x7f4a7c15u));
}

// A seeded draw: the Owen-scrambled point of the seed's dimension stream at
// index 0 — a pure hash of (seed, dimension), replayable from the stored
// seed alone.
public float2 replay_2d(uint seed, uint dimension) {
    return scrambledPoint2d(seed, 0, dimension);
}

public float replay_1d(uint seed, uint dimension) {
    return scrambledPoint1d(seed, 0, dimension);
}

// -- Blue-noise sample-index ranking (D-095) ---------------------------
// A fixed per-pixel permutation of the sample index, keyed by a screen-space
// blue-noise value (`key`, from bluenoise.slang's mask). Unlike sample_*'s
// per-dimension padding shuffle, one ranking is *shared* across all of a
// sample's dimensions, so the whole pixel estimate's Monte-Carlo error
// correlates with the key and — the key being blue across the screen — lands as
// blue noise rather than white. Fixed across sampleIndex (the seed depends only
// on the key, not the sample): each pixel still walks a progressive
// low-discrepancy set, so this renderer's per-pixel convergence is unharmed and
// the converged image is identical — only the low-sample error *distribution*
// changes. (Heitz et al. 2019.)
public uint blueRankedIndex(uint key, uint sampleIndex) {
    return nestedUniformScramble(sampleIndex, lowbias32(key ^ 0x9e3779b9u));
}

// A dimension's 2D / 1D point at a pre-ranked sample index — the blue-noise
// counterpart of sample_2d/1d. The caller ranks once per (pixel, sample) with
// blueRankedIndex and passes that index for every dimension of the sample.
public float2 sample_ranked_2d(uint pixel, uint rankedIndex, uint dimension) {
    return scrambledPoint2d(pixel, rankedIndex, dimension);
}

public float sample_ranked_1d(uint pixel, uint rankedIndex, uint dimension) {
    return scrambledPoint1d(pixel, rankedIndex, dimension);
}
