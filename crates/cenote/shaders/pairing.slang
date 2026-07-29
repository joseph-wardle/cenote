// Self-inverting Gaussian pairing textures (M6 step 7a; ReSTIR PT Enhanced
// §3) — the spatial stage's reciprocal neighbour selection. Each neighbour
// slot has its own toroidal delta texture in which texel A stores the offset
// to its partner B and B stores exactly the negation, so the tiled screen
// lookup `partner(p) = p + delta[p]` is a fixed-point-free involution: A picks
// B on slot n exactly when B picks A. The deltas are Gaussian (σ 16 px,
// matched to the mean partner distance of the R = 30 disk draw this replaces),
// and each frame every texture is re-randomized by a hash-derived D4 symmetry
// + toroidal translation — conjugation by an isometry, which preserves both
// invariants (proved by the host mirror's unit tests, pinned against this
// very code by the `pairing_test` fixture).
//
// Host: `src/pairing.rs` builds the textures (fixed seed, deterministic) and
// uploads them as one storage buffer at set 0 binding 4 — the blue-noise
// mask's binding model: a renderer-global read-only resource, not scene data.
// The math below is split pure (`pairingLookup`/`pairingTexel`/`pairingDelta`)
// so the fixture kernel can drive it through a raw pointer; `pairingOffset`
// composes them over the bound buffer for the spatial stage.

import rng;

// One texture per spatial neighbour slot — the shader-side cap on k.
public static const uint PAIRING_TEXTURES = 5;

// Texture edges and their first words in the buffer. Mirrors
// `pairing::SIZES`; the bases are its prefix sums, and the fixture test fails
// loudly if either list drifts from the host layout.
static const uint PAIRING_SIZES[PAIRING_TEXTURES] = { 254, 230, 210, 190, 178 };
static const uint PAIRING_BASES[PAIRING_TEXTURES] = { 0, 64516, 117416, 161516, 197616 };

// One texture's per-frame state: where it starts, and the transform that
// re-randomizes it this frame.
public struct PairingLookup {
    public uint base;      // first word of the texture in the buffer
    public int size;       // texture edge, texels
    public uint flags;     // D4 bits: 1 = negate x, 2 = negate y, 4 = transpose
    public int2 translate; // toroidal translation, texels
};

// The texture geometry plus this frame's hash-derived transform. lowbias32 of
// the (sampleIndex, texture) key — injective for tex < PAIRING_TEXTURES — then
// chained for the two translation words.
public PairingLookup pairingLookup(uint tex, uint sampleIndex) {
    let size = PAIRING_SIZES[tex];
    let h = lowbias32(sampleIndex * PAIRING_TEXTURES + tex);
    let h2 = lowbias32(h);
    let h3 = lowbias32(h2);
    PairingLookup l;
    l.base = PAIRING_BASES[tex];
    l.size = int(size);
    l.flags = h & 7;
    l.translate = int2(int(h2 % size), int(h3 % size));
    return l;
}

// Floored modulo — screen coordinates go negative under the transform's
// negations (and, for the involution's partner, past the frame edge). The
// operand is shifted positive before the `%`: Slang lowers a negative-operand
// `%` inconsistently (observed compiling to *unsigned* modulo when the divisor
// loads from a struct field), and on non-negative operands every semantics
// agrees. Valid for x > −2¹⁶·s — comfortably every screen coordinate.
int wrapCoord(int x, int s) {
    return (x + (1 << 16) * s) % s;
}

// The absolute buffer word holding pixel `p`'s delta: the forward transform
// g(p) + t, wrapped onto the texture torus. g applies negate-x, negate-y,
// then transpose; `pairingDelta` undoes it in the reverse order.
public uint pairingTexel(PairingLookup l, int2 pixel) {
    var u = pixel;
    if ((l.flags & 1) != 0) {
        u.x = -u.x;
    }
    if ((l.flags & 2) != 0) {
        u.y = -u.y;
    }
    if ((l.flags & 4) != 0) {
        u = u.yx;
    }
    let ux = wrapCoord(u.x + l.translate.x, l.size);
    let uy = wrapCoord(u.y + l.translate.y, l.size);
    return l.base + uint(uy * l.size + ux);
}

// Unpack the texel's two i8 deltas (x low byte, y next) and take them back
// through the transform's inverse — the screen-space offset to the partner.
public int2 pairingDelta(PairingLookup l, uint word) {
    var d = int2(int(word << 24) >> 24, int(word << 16) >> 24);
    if ((l.flags & 4) != 0) {
        d = d.yx;
    }
    if ((l.flags & 1) != 0) {
        d.x = -d.x;
    }
    if ((l.flags & 2) != 0) {
        d.y = -d.y;
    }
    return d;
}

// The pairing textures (set 0 binding 4): all PAIRING_TEXTURES delta images
// back to back, one u32 word per texel, uploaded once at renderer init.
[vk::binding(4, 0)]
StructuredBuffer<uint> pairingTextures;

// The one call the spatial stage makes: this pixel's slot-`tex` partner
// offset, this frame. Never zero (the textures have no fixed points), and
// reciprocal by construction: partner(partner(p)) == p at every pixel.
public int2 pairingOffset(uint tex, int2 pixel, uint sampleIndex) {
    let l = pairingLookup(tex, sampleIndex);
    return pairingDelta(l, pairingTextures[pairingTexel(l, pixel)]);
}
