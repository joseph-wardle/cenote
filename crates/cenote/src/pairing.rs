//! Self-inverting Gaussian pairing textures — the reciprocal neighbour
//! selection for `ReSTIR`'s spatial reuse (M6 step 7a; `ReSTIR` PT Enhanced §3).
//!
//! Pairwise MIS makes pixel A's reuse of pixel B need the same two shift
//! evaluations B's reuse of A needs — so if selection is *reciprocal* (A picks
//! B ⟺ B picks A), a later pass can compute each once and share it (step 7b).
//! Reciprocity comes from a *pairing texture*: a toroidal two-channel image of
//! coordinate deltas in which texel A stores the delta to its partner B and B
//! stores exactly the negation. Tiled over the frame, `partner(p) = p +
//! delta[p mod s]` is then an involution at every pixel — and the deltas come
//! out Gaussian, which is itself the quality half of the step (distance-
//! concentrated neighbours are more shift-compatible than a uniform disk).
//!
//! Construction (Enhanced §3.1): fill the image with consecutive link indices,
//! shuffle with `n_σ` tiled 2×2 shuffles — a uniformly random permutation of
//! each block's four texels, every other pass offset diagonally by one, looping
//! over the edge — then pair the texels holding indices 2k and 2k+1. The two
//! endpoints random-walk apart, so their delta lands Gaussian; Eq 3 maps the
//! target σ to the shuffle count (σ = 16 → 128 passes). Involution and
//! fixed-point-freeness hold *exactly* by construction: distinct indices sit at
//! distinct texels, and both endpoints store mutually negated deltas.
//!
//! One texture per neighbour slot, at distinct even sizes chosen against
//! near-period moiré within a frame (test below); per frame each texture is
//! re-randomized by a hash-derived D4 symmetry + toroidal translation, which
//! conjugates the involution by an isometry and so preserves both invariants —
//! asserted for *every* transform by the tests here, which mirror the shader
//! lookup (`shaders/pairing.slang`) line for line, and pinned against the GPU
//! by the `pairing_test` fixture kernel.
//!
//! The textures are deterministic (fixed seed, fixed iteration order), built
//! once per process, and uploaded as one storage buffer at set 0 binding 4 —
//! the blue-noise mask's binding model (D-095): a renderer-global read-only
//! resource, not scene data.

use std::sync::OnceLock;

/// Number of pairing textures — one per spatial neighbour slot, so the
/// shader's `PAIRING_TEXTURES` and the host's spatial k are both capped here.
pub const COUNT: usize = 5;

/// Texture edges, texels. Square (the transpose transform needs it), even (2×2
/// blocks tile the torus at either offset), ≤ 254 (wrapped deltas fit i8),
/// distinct and mutually near-period-free within a 512 px frame (test below).
/// Mirrors `PAIRING_SIZES` in `shaders/pairing.slang`; the byte layout the two
/// lists imply is cross-checked by the `pairing_test` GPU fixture.
pub const SIZES: [u32; COUNT] = [254, 230, 210, 190, 178];

/// Per-axis standard deviation of the coordinate deltas, pixels:
/// √(8/(9π))·30 ≈ 16 — the mean partner distance of the uniform-disk draw this
/// selection replaces (R = 30, `RESTIR_SPATIAL_RADIUS` before 7a), so the A/B
/// against the pre-7a commit varies the *shape* of the distribution alone.
pub const SIGMA: f32 = 16.0;

/// The packed pairing textures, one `u32` word per texel — x delta in the low
/// byte, y delta in the next, both i8 — texture-major in [`SIZES`] order, each
/// texture row-major. Built once per process (fixed seed, deterministic);
/// uploaded verbatim as the binding-4 storage buffer.
#[must_use]
pub fn textures() -> &'static [u32] {
    static TEXTURES: OnceLock<Vec<u32>> = OnceLock::new();
    TEXTURES.get_or_init(build)
}

fn build() -> Vec<u32> {
    let mut words = Vec::with_capacity(SIZES.iter().map(|&s| (s * s) as usize).sum());
    for (index, &size) in SIZES.iter().enumerate() {
        // Any fixed per-texture seed works; SplitMix mixes sequential seeds
        // into independent streams.
        texture(&mut words, size, 0x7a00 + index as u64);
    }
    words
}

/// Eq 3 of Enhanced §3.1: the tiled-shuffle count that reaches a target σ.
/// The σ²/2 term is the random walk (each shuffle adds ~1 px² of pair-delta
/// variance per axis); the negative powers are the paper's fit correction for
/// the early, still-correlated passes. σ = 16 → 128.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the fit is small and positive; truncation is Eq 3's own floor"
)]
fn shuffle_count(sigma: f64) -> u32 {
    (sigma * sigma / 2.0 + 1.46 / sigma + 1.76 / (sigma * sigma)
        + 0.656 / (sigma * sigma * sigma)
        + 0.5) as u32
}

/// Build one texture (Enhanced §3.1) and append its packed words.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "texel counts and coordinates are far below every cast's bound"
)]
fn texture(words: &mut Vec<u32>, size: u32, mut seed: u64) {
    let s = size as usize;
    let n = s * s;

    // Consecutive link indices, then n_σ tiled 2×2 shuffles: each pass visits
    // the blocks row-major and applies an independent uniformly random
    // permutation (Fisher–Yates) to the four texels' contents; odd passes
    // offset the block grid by (1, 1), wrapping over the edge.
    let mut img: Vec<u32> = (0..n as u32).collect();
    for pass in 0..shuffle_count(f64::from(SIGMA)) {
        let off = (pass & 1) as usize;
        for by in 0..s / 2 {
            for bx in 0..s / 2 {
                let x0 = (2 * bx + off) % s;
                let x1 = (2 * bx + off + 1) % s;
                let y0 = (2 * by + off) % s;
                let y1 = (2 * by + off + 1) % s;
                let cells = [y0 * s + x0, y0 * s + x1, y1 * s + x0, y1 * s + x1];
                for i in (1..4).rev() {
                    let j = (split_mix64(&mut seed) % (i as u64 + 1)) as usize;
                    img.swap(cells[i], cells[j]);
                }
            }
        }
    }

    // Pair the texels holding indices 2k and 2k+1: each stores the shortest
    // toroidal delta to the other — negations of one another, which is the
    // involution. A delta reaching ±s/2 would break the negation symmetry
    // (both wraps land on −s/2); at ≥ 5σ out it never occurs with this fixed
    // seed, and the assert keeps the property loud, not lucky.
    let mut pos = vec![0u32; n];
    for (p, &link) in img.iter().enumerate() {
        pos[link as usize] = p as u32;
    }
    let base = words.len();
    words.resize(base + n, 0);
    let wrap = |d: i32| {
        let w = d.rem_euclid(size as i32);
        if w >= size as i32 / 2 { w - size as i32 } else { w }
    };
    for k in 0..n / 2 {
        let a = pos[2 * k] as usize;
        let b = pos[2 * k + 1] as usize;
        let dx = wrap((b % s) as i32 - (a % s) as i32);
        let dy = wrap((b / s) as i32 - (a / s) as i32);
        assert!(
            dx.abs() < size as i32 / 2 && dy.abs() < size as i32 / 2,
            "pairing delta hit ±{size}/2 — reseed or resize the texture"
        );
        words[base + a] = pack(dx, dy);
        words[base + b] = pack(-dx, -dy);
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the build asserted both deltas fit i8"
)]
fn pack(dx: i32, dy: i32) -> u32 {
    u32::from(dx as i8 as u8) | (u32::from(dy as i8 as u8) << 8)
}

/// `SplitMix64` — a fixed, well-mixed PRNG for the deterministic shuffles
/// (`bluenoise.rs`'s generator uses the same).
fn split_mix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The CPU mirror of the shader lookup (`shaders/pairing.slang`), line for
/// line — what the unit tests below prove the invariants on, and what the
/// `pairing_test` GPU fixture pins the shader against. Test-only: the renderer
/// reads the textures through the shader alone.
#[cfg(test)]
mod mirror {
    use super::{COUNT, SIZES};

    /// One texture's per-frame state: where it starts in the buffer, and the
    /// hash-derived transform (D4 flags + toroidal translation) that
    /// re-randomizes it. Mirrors `PairingLookup`.
    pub struct Lookup {
        pub base: u32,
        pub size: i32,
        /// D4 bits: 1 = negate x, 2 = negate y, 4 = transpose (applied to the
        /// coordinate in that order; inverted on the delta in the reverse).
        pub flags: u32,
        pub tx: i32,
        pub ty: i32,
    }

    /// lowbias32 (`rng.slang`) — the hash the transform derives from.
    fn lowbias32(mut x: u32) -> u32 {
        x ^= x >> 16;
        x = x.wrapping_mul(0x21f0_aaad);
        x ^= x >> 15;
        x = x.wrapping_mul(0xd35a_2d97);
        x ^= x >> 15;
        x
    }

    /// Mirrors `pairingLookup`: texture geometry plus this frame's transform.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "sizes and hash residues are far below every cast's bound"
    )]
    pub fn lookup(tex: usize, sample_index: u32) -> Lookup {
        let base = SIZES[..tex].iter().map(|&s| s * s).sum();
        let size = SIZES[tex] as i32;
        let h = lowbias32(sample_index * COUNT as u32 + tex as u32);
        let h2 = lowbias32(h);
        let h3 = lowbias32(h2);
        Lookup {
            base,
            size,
            flags: h & 7,
            tx: (h2 % size as u32) as i32,
            ty: (h3 % size as u32) as i32,
        }
    }

    /// Mirrors `pairingTexel`: the absolute buffer word holding pixel `p`'s
    /// delta — forward transform g(p) + t, wrapped onto the texture torus.
    pub fn texel(l: &Lookup, px: i32, py: i32) -> usize {
        let (mut ux, mut uy) = (px, py);
        if l.flags & 1 != 0 {
            ux = -ux;
        }
        if l.flags & 2 != 0 {
            uy = -uy;
        }
        if l.flags & 4 != 0 {
            std::mem::swap(&mut ux, &mut uy);
        }
        // The shader's shifted-positive wrap, verbatim (its comment explains
        // why it avoids a negative-operand `%`); equal to rem_euclid here.
        let ux = (ux + l.tx + (1 << 16) * l.size) % l.size;
        let uy = (uy + l.ty + (1 << 16) * l.size) % l.size;
        l.base as usize + (uy * l.size + ux) as usize
    }

    /// Mirrors `pairingDelta`: unpack the two i8 deltas and take them back
    /// through the transform's inverse.
    #[expect(
        clippy::cast_possible_wrap,
        reason = "the cast reinterprets bits; the shifts sign-extend the i8s"
    )]
    pub fn delta(l: &Lookup, word: u32) -> (i32, i32) {
        let (mut dx, mut dy) = (((word as i32) << 24) >> 24, ((word as i32) << 16) >> 24);
        if l.flags & 4 != 0 {
            std::mem::swap(&mut dx, &mut dy);
        }
        if l.flags & 1 != 0 {
            dx = -dx;
        }
        if l.flags & 2 != 0 {
            dy = -dy;
        }
        (dx, dy)
    }

    /// Mirrors `pairingOffset` — the full lookup the spatial stage makes.
    pub fn offset(words: &[u32], tex: usize, px: i32, py: i32, sample_index: u32) -> (i32, i32) {
        let l = lookup(tex, sample_index);
        delta(&l, words[texel(&l, px, py)])
    }
}

#[cfg(test)]
mod tests {
    use super::mirror::{self, Lookup};
    use super::{textures, COUNT, SIGMA, SIZES};

    /// Walk (texture index, its first word, its edge).
    fn each_texture() -> impl Iterator<Item = (usize, usize, i32)> {
        SIZES.iter().enumerate().scan(0usize, |base, (t, &s)| {
            let here = *base;
            *base += (s * s) as usize;
            #[expect(clippy::cast_possible_wrap, reason = "edges are ≤ 254")]
            Some((t, here, s as i32))
        })
    }

    #[expect(
        clippy::cast_possible_wrap,
        reason = "the cast reinterprets bits; the shifts sign-extend the i8s"
    )]
    fn unpack(word: u32) -> (i32, i32) {
        (((word as i32) << 24) >> 24, ((word as i32) << 16) >> 24)
    }

    /// The raw textures are fixed-point-free involutions on their tori: no
    /// texel pairs with itself, and A's delta is exactly the negation of its
    /// partner's — the property every reciprocity claim downstream stands on.
    #[test]
    fn every_texture_is_a_fixed_point_free_involution() {
        let words = textures();
        for (_, base, s) in each_texture() {
            for y in 0..s {
                for x in 0..s {
                    let (dx, dy) = unpack(words[base + (y * s + x) as usize]);
                    assert!((dx, dy) != (0, 0), "fixed point at ({x}, {y})");
                    let qx = (x + dx).rem_euclid(s);
                    let qy = (y + dy).rem_euclid(s);
                    assert_eq!(
                        unpack(words[base + (qy * s + qx) as usize]),
                        (-dx, -dy),
                        "involution broken at ({x}, {y})"
                    );
                }
            }
        }
    }

    /// The deltas are the promised distribution: zero mean exactly (pairs
    /// contribute d and −d), per-axis σ within 5% of the target — the knob
    /// that matched the old disk draw's mean partner distance.
    #[test]
    fn deltas_match_the_target_sigma() {
        let words = textures();
        for (t, base, s) in each_texture() {
            let n = (s * s) as usize;
            let deltas: Vec<(i32, i32)> = words[base..base + n].iter().map(|&w| unpack(w)).collect();
            let sum = deltas.iter().fold((0i64, 0i64), |acc, &(dx, dy)| {
                (acc.0 + i64::from(dx), acc.1 + i64::from(dy))
            });
            assert_eq!(sum, (0, 0), "texture {t}: deltas do not pair to zero mean");
            #[expect(clippy::cast_precision_loss, reason = "counts are ≤ 254²")]
            let sigma = |axis: fn(&(i32, i32)) -> i32| {
                (deltas.iter().map(|d| f64::from(axis(d)).powi(2)).sum::<f64>() / n as f64).sqrt()
            };
            let (sx, sy) = (sigma(|d| d.0), sigma(|d| d.1));
            let target = f64::from(SIGMA);
            assert!(
                (sx - target).abs() < 0.05 * target && (sy - target).abs() < 0.05 * target,
                "texture {t}: per-axis σ ({sx:.2}, {sy:.2}) misses {target}"
            );
        }
    }

    /// The screen-space lookup stays a fixed-point-free involution under
    /// *every* per-frame transform — all eight D4 elements, translations, and
    /// negative screen coordinates (the wrap the transform's negations
    /// exercise). This is the conjugation-by-isometry argument, tested rather
    /// than trusted.
    #[test]
    fn lookup_is_an_involution_under_every_transform() {
        let words = textures();
        for (t, _, s) in each_texture() {
            for flags in 0..8 {
                for &(tx, ty) in &[(0, 0), (37, 101), (s - 1, 13)] {
                    let l = Lookup { base: SIZES[..t].iter().map(|&z| z * z).sum(), size: s, flags, tx, ty };
                    for py in -3..s - 3 {
                        for px in -3..s - 3 {
                            let d = mirror::delta(&l, words[mirror::texel(&l, px, py)]);
                            assert!(d != (0, 0), "fixed point under flags {flags}");
                            let back =
                                mirror::delta(&l, words[mirror::texel(&l, px + d.0, py + d.1)]);
                            assert_eq!(
                                back,
                                (-d.0, -d.1),
                                "involution broken at ({px}, {py}), flags {flags}, t ({tx}, {ty})"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The per-frame hash genuinely re-randomizes each texture (the paper's
    /// fix for the self-inverting texture's frame-to-frame correlation), and
    /// the full hash-derived lookup keeps the involution on real screen
    /// coordinates.
    #[test]
    fn frame_hash_decorrelates_and_preserves_involution() {
        let words = textures();
        for tex in 0..COUNT {
            let mut transforms = std::collections::HashSet::new();
            for sample in 0..64 {
                let l = mirror::lookup(tex, sample);
                transforms.insert((l.flags, l.tx, l.ty));
            }
            assert!(
                transforms.len() > 32,
                "texture {tex}: only {} distinct transforms in 64 frames",
                transforms.len()
            );
            for sample in 0..8 {
                for py in (0..512).step_by(7) {
                    for px in (0..512).step_by(7) {
                        let d = mirror::offset(words, tex, px, py, sample);
                        assert_eq!(
                            mirror::offset(words, tex, px + d.0, py + d.1, sample),
                            (-d.0, -d.1),
                            "involution broken at ({px}, {py}), sample {sample}"
                        );
                    }
                }
            }
        }
    }

    /// The size set keeps every texture's tiling period clear of every
    /// other's within a frame: small multiples of two sizes never land within
    /// 8 px of each other below 512, so no two textures' repeats align into
    /// moiré. Plus the standing shape constraints: even (2×2 blocks tile the
    /// torus), ≤ 254 (deltas fit i8), distinct.
    #[test]
    fn sizes_avoid_near_periods() {
        for (i, &a) in SIZES.iter().enumerate() {
            assert!(a % 2 == 0 && a <= 254, "size {a} breaks the shape constraints");
            for &b in &SIZES[i + 1..] {
                assert_ne!(a, b, "duplicate size {a}");
                for ka in 1..=512 / a {
                    for kb in 1..=512 / b {
                        let (pa, pb) = (ka * a, kb * b);
                        assert!(
                            pa.abs_diff(pb) >= 8,
                            "near-period: {ka}·{a} = {pa} vs {kb}·{b} = {pb}"
                        );
                    }
                }
            }
        }
    }

    /// The shader lookup (`shaders/pairing.slang`) agrees with the CPU mirror
    /// texel for texel on the GPU it ships on — transform hash, forward
    /// coordinate map, unpack, and inverse all pinned exactly, plus the
    /// buffer layout the shader's hardcoded bases imply. A sign-extension or
    /// base-offset slip here would still render plausibly (selection stays
    /// in-bounds, merely non-reciprocal), so no image test would catch it.
    #[test]
    fn shader_lookup_matches_the_mirror() {
        use crate::gpu::{Bindings, MemoryLocation};
        use ash::vk;
        use bytemuck::{Pod, Zeroable};

        /// Mirrors `struct Params` in `shaders/pairing_test.slang`.
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct FixtureParams {
            words: vk::DeviceAddress,
            out: vk::DeviceAddress,
            tex: u32,
            sample_index: u32,
            size: u32,
            count: u32,
        }

        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let spirv = crate::shaders::compile_fixture("pairing_test").expect("compile pairing_test");
        let pipeline = gpu
            .create_compute_pipeline(
                &spirv,
                c"pairing_test",
                size_of::<FixtureParams>() as u32,
                Bindings::None,
            )
            .expect("pipeline");

        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        let words = textures();
        let words_buffer = gpu
            .upload_buffer("test.pairing.words", bytemuck::cast_slice(words), usage)
            .expect("upload");

        for (tex, _, s) in each_texture() {
            let count = (s * s) as u32;
            let out = gpu
                .create_buffer(
                    "test.pairing.out",
                    u64::from(count) * 8,
                    usage | vk::BufferUsageFlags::TRANSFER_SRC,
                    MemoryLocation::GpuOnly,
                )
                .expect("buffer");
            for sample_index in [0, 1, 977] {
                let params = FixtureParams {
                    words: words_buffer.device_address(),
                    out: out.device_address(),
                    tex: tex as u32,
                    sample_index,
                    size: s as u32,
                    count,
                };
                gpu.dispatch(
                    &pipeline,
                    None,
                    bytemuck::bytes_of(&params),
                    [count.div_ceil(64), 1, 1],
                )
                .expect("dispatch");
                let got: Vec<i32> =
                    bytemuck::pod_collect_to_vec(&gpu.download_buffer(&out).expect("download"));
                for i in 0..count as usize {
                    #[expect(clippy::cast_possible_wrap, reason = "texel counts fit i32")]
                    let (px, py) = ((i as i32) % s, (i as i32) / s);
                    let want = mirror::offset(words, tex, px, py, sample_index);
                    assert_eq!(
                        (got[2 * i], got[2 * i + 1]),
                        want,
                        "texture {tex}, sample {sample_index}, pixel ({px}, {py})"
                    );
                }
            }
        }
    }
}
