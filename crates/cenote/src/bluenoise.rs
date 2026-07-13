//! The screen-space blue-noise mask that keys `ReSTIR`'s Sobol sample-index
//! ranking (D-095).
//!
//! cenote accumulates: every kernel takes a `sampleIndex`, and the film
//! averages sample 0, 1, 2, … into the running mean the viewer displays. The
//! Sobol-Burley sampler ([`rng.slang`](../shaders/rng.slang)) is *progressive* —
//! a pixel walking indices 0,1,2,… covers the sequence as a low-discrepancy
//! set, so its average converges fast. Blue-noise ranking keeps that progressive
//! convergence while changing how the *residual* error at low sample counts is
//! laid out across the screen: a per-pixel permutation of the sample index,
//! keyed by this mask and held **fixed across samples**, so neighbouring pixels
//! draw complementary strata and the Monte-Carlo error lands as blue noise
//! rather than white — perceptually far cleaner at equal spp, converging to the
//! identical image (Heitz et al. 2019, "A Low-Discrepancy Sampler that
//! Distributes Monte Carlo Errors as a Blue Noise in Screen Space").
//!
//! The *temporal* axis of spatiotemporal blue noise (`STBN`) is deliberately
//! absent: it varies the permutation per sample to decorrelate consecutive
//! frames, which is what a one-sample-per-frame real-time pipeline wants — but
//! here it would make each pixel draw a pseudo-random Sobol index every sample
//! instead of covering the sequence, reverting convergence toward white-noise
//! √N. A fixed 2-D mask is the right construction for a renderer that
//! integrates over the sample axis.
//!
//! The mask is a 64×64 toroidal tile, **precomputed** by the deterministic
//! void-and-cluster generator (in the test module below, Ulichney 1993) and
//! committed as `assets/bluenoise_64.bin` — startup just loads the bytes. Its
//! provenance is the in-tree generator, not an opaque downloaded blob: the
//! `committed_tile_matches_generator` test regenerates it and asserts the bytes
//! match, so the asset is reproducible from source and any tweak to the
//! generator fails loudly until the tile is regenerated.

/// Tile edge, in pixels. The mask is `TILE`² ranks, tiled toroidally over the
/// frame — void-and-cluster produces a seamlessly wrapping pattern, so the
/// repeat is invisible in the noise.
pub const TILE: u32 = 64;

const SIZE: usize = (TILE * TILE) as usize;

/// The committed blue-noise mask: `TILE`² ranks in `[0, TILE²)`, row-major
/// (`y·TILE + x`), one `u32` each. Loaded from the precomputed asset; the
/// bytes are exactly what the generator produces (guarded by test).
#[must_use]
pub fn mask() -> Vec<u32> {
    const BYTES: &[u8] = include_bytes!("../assets/bluenoise_64.bin");
    debug_assert_eq!(BYTES.len(), SIZE * 4, "blue-noise asset has stale dimensions");
    BYTES
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// The void-and-cluster generator — the single source of truth [`mask`]'s
/// committed bytes are checked against. Test-only: it runs to verify or (with
/// `UPDATE_BLUE_NOISE`) regenerate the asset; the renderer loads bytes.
#[cfg(test)]
mod generator {
    use super::{SIZE, TILE};

    /// Regenerate the blue-noise tile from scratch with the void-and-cluster
    /// method. Deterministic (a fixed-seed scatter, no wall-clock or
    /// thread-order input), so two runs on any machine agree bit for bit.
    ///
    /// Void-and-cluster (Ulichney, "The void-and-cluster method for dither
    /// array generation", 1993): grow a uniform *prototype* pattern, then rank
    /// every pixel by how it breaks up clusters (low ranks, the isolated points
    /// that appear first) or fills voids (high ranks). The ranks, laid out in
    /// screen space, are the blue-noise mask.
    pub fn generate() -> Vec<u32> {
        let gauss = gaussian_table();
        let prototype = uniform_prototype(&gauss);

        let mut rank = vec![0u32; SIZE];

        // Phase 1 — the prototype's ones, ranked M0−1 down to 0 by peeling the
        // tightest cluster each step: the most clustered point leaves first (the
        // highest rank), the last isolated point keeps rank 0.
        let mut pattern = prototype.clone();
        let mut energy = filter(&pattern, &gauss);
        let ones = prototype.iter().filter(|&&on| on).count();
        for r in (0..ones).rev() {
            let c = tightest_cluster(&pattern, &energy);
            pattern[c] = false;
            splat(&mut energy, c, &gauss, -1.0);
            rank[c] = r as u32;
        }

        // Phase 2 — the remaining pixels, ranked M0 up to N−1 by filling the
        // largest void (the zero with the fewest neighbouring ones) each step.
        let mut pattern = prototype;
        let mut energy = filter(&pattern, &gauss);
        for r in ones..SIZE {
            let v = largest_void(&pattern, &energy);
            pattern[v] = true;
            splat(&mut energy, v, &gauss, 1.0);
            rank[v] = r as u32;
        }

        rank
    }

    /// A uniform initial pattern of `SIZE/10` ones: scatter that many at a
    /// fixed-seed PRNG's positions, then relocate the tightest-clustered one to
    /// the largest void until the two coincide (as uniform as it gets).
    fn uniform_prototype(gauss: &[f32]) -> Vec<bool> {
        let target = SIZE / 10;
        let mut pattern = vec![false; SIZE];
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut placed = 0;
        while placed < target {
            let p = (split_mix64(&mut seed) as usize) % SIZE;
            if !pattern[p] {
                pattern[p] = true;
                placed += 1;
            }
        }

        let mut energy = filter(&pattern, gauss);
        loop {
            let c = tightest_cluster(&pattern, &energy);
            pattern[c] = false;
            splat(&mut energy, c, gauss, -1.0);
            let v = largest_void(&pattern, &energy);
            pattern[v] = true;
            splat(&mut energy, v, gauss, 1.0);
            if v == c {
                break;
            }
        }
        pattern
    }

    /// The full energy field of a pattern: each pixel's sum of the Gaussian from
    /// every set pixel. Rebuilt once per phase; the incremental [`splat`] keeps
    /// it current thereafter.
    fn filter(pattern: &[bool], gauss: &[f32]) -> Vec<f32> {
        let mut energy = vec![0.0f32; SIZE];
        for (p, &on) in pattern.iter().enumerate() {
            if on {
                splat(&mut energy, p, gauss, 1.0);
            }
        }
        energy
    }

    /// Add (`sign = 1`) or remove (`sign = −1`) one set pixel's Gaussian to the
    /// energy field, wrapping toroidally — the O(tile²) update that keeps
    /// [`filter`]'s field current as the pattern changes.
    fn splat(energy: &mut [f32], center: usize, gauss: &[f32], sign: f32) {
        let tile = TILE as usize;
        let (cx, cy) = (center % tile, center / tile);
        for qy in 0..tile {
            let dy = (qy + tile - cy) % tile;
            let row = qy * tile;
            let grow = dy * tile;
            for qx in 0..tile {
                let dx = (qx + tile - cx) % tile;
                energy[row + qx] += sign * gauss[grow + dx];
            }
        }
    }

    /// The set pixel with the most energy around it — the centre of the tightest
    /// cluster, the next to peel in phase 1.
    fn tightest_cluster(pattern: &[bool], energy: &[f32]) -> usize {
        let mut best = usize::MAX;
        let mut best_energy = f32::NEG_INFINITY;
        for (p, &on) in pattern.iter().enumerate() {
            if on && energy[p] > best_energy {
                best_energy = energy[p];
                best = p;
            }
        }
        best
    }

    /// The unset pixel with the least energy around it — the centre of the
    /// largest void, the next to fill in phase 2.
    fn largest_void(pattern: &[bool], energy: &[f32]) -> usize {
        let mut best = usize::MAX;
        let mut best_energy = f32::INFINITY;
        for (p, &on) in pattern.iter().enumerate() {
            if !on && energy[p] < best_energy {
                best_energy = energy[p];
                best = p;
            }
        }
        best
    }

    /// The Gaussian energy kernel over toroidal offsets, `exp(−(dx²+dy²)/2σ²)`
    /// with σ = 1.5 (Ulichney's value): indexed `[dy·TILE + dx]`, each axis
    /// folded to its shorter wrap so the tile stays seamless.
    fn gaussian_table() -> Vec<f32> {
        let tile = TILE as usize;
        let sigma = 1.5f32;
        let denom = 2.0 * sigma * sigma;
        let mut table = vec![0.0f32; SIZE];
        for dy in 0..tile {
            let ry = dy.min(tile - dy) as f32;
            for dx in 0..tile {
                let rx = dx.min(tile - dx) as f32;
                table[dy * tile + dx] = (-(rx * rx + ry * ry) / denom).exp();
            }
        }
        // The self-offset (0,0) would count a pixel's own Gaussian; zero it so
        // the energy measures only *neighbours'* contribution.
        table[0] = 0.0;
        table
    }

    /// `SplitMix64` — a fixed, well-mixed PRNG for the deterministic scatter.
    fn split_mix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::generator::generate;
    use super::{mask, SIZE};

    /// The committed asset is exactly what the in-tree generator produces —
    /// the provenance guarantee.
    #[test]
    fn committed_tile_matches_generator() {
        assert_eq!(
            mask(),
            generate(),
            "assets/bluenoise_64.bin is stale — regenerate it (see below)"
        );
    }

    /// (Re)write the committed asset from the generator — the counterpart to
    /// the goldens' `UPDATE_GOLDENS`. Run when the generator changes on purpose:
    /// `UPDATE_BLUE_NOISE=1 cargo test -p cenote bluenoise -- --test-threads=1`.
    #[test]
    fn regenerate_committed_tile() {
        if std::env::var_os("UPDATE_BLUE_NOISE").is_none() {
            return;
        }
        let bytes: Vec<u8> = generate().iter().flat_map(|r| r.to_le_bytes()).collect();
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/bluenoise_64.bin");
        std::fs::create_dir_all(path.parent().expect("asset path has a parent"))
            .expect("create assets dir");
        std::fs::write(&path, &bytes).expect("write blue-noise asset");
        eprintln!("wrote {}", path.display());
    }

    /// The mask is a permutation of `[0, TILE²)` — every rank once, so it is a
    /// valid dither array (and the fixed per-pixel index permutation is a
    /// bijection, which is what keeps convergence progressive).
    #[test]
    fn ranks_are_a_permutation() {
        let mut seen = vec![false; SIZE];
        for &r in &generate() {
            assert!((r as usize) < SIZE, "rank {r} out of range");
            assert!(!seen[r as usize], "rank {r} repeats");
            seen[r as usize] = true;
        }
    }

    /// The mask is genuinely *blue*, not merely a valid permutation: threshold
    /// it at the median and its power spectrum concentrates at high spatial
    /// frequencies — little energy near DC, lots at the edge. This is the
    /// property that turns per-pixel error into screen-space blue noise; a
    /// botched generator (white or clustered) would fail here where the
    /// permutation test would not.
    #[test]
    fn the_mask_has_a_blue_spectrum() {
        const TILE: usize = super::TILE as usize;
        // A zero-mean binary pattern: +1 for the half of pixels with the higher
        // ranks, −1 for the lower half. Its DFT has no DC term by construction,
        // so the spectrum is purely the noise's spatial structure.
        let ranks = generate();
        let signal: Vec<f32> = ranks
            .iter()
            .map(|&r| if (r as usize) >= SIZE / 2 { 1.0 } else { -1.0 })
            .collect();

        // Radial power in a low band (near DC) versus a high band (near the
        // Nyquist edge), each frequency folded to its shorter toroidal wrap.
        let (mut low, mut high) = (0.0f64, 0.0f64);
        let (mut low_n, mut high_n) = (0usize, 0usize);
        let nyquist = (TILE / 2) as f32;
        for fy in 0..TILE {
            for fx in 0..TILE {
                if fx == 0 && fy == 0 {
                    continue; // DC — zero anyway, and not a noise frequency
                }
                let (mut re, mut im) = (0.0f64, 0.0f64);
                for y in 0..TILE {
                    for x in 0..TILE {
                        let phase = -2.0 * std::f64::consts::PI
                            * ((fx * x % TILE) as f64 / TILE as f64
                                + (fy * y % TILE) as f64 / TILE as f64);
                        let s = f64::from(signal[y * TILE + x]);
                        re += s * phase.cos();
                        im += s * phase.sin();
                    }
                }
                let power = re * re + im * im;
                let rx = (fx.min(TILE - fx)) as f32;
                let ry = (fy.min(TILE - fy)) as f32;
                let radius = (rx * rx + ry * ry).sqrt();
                if radius < nyquist * 0.4 {
                    low += power;
                    low_n += 1;
                } else if radius > nyquist * 0.8 {
                    high += power;
                    high_n += 1;
                }
            }
        }
        let low_mean = low / low_n as f64;
        let high_mean = high / high_n as f64;
        assert!(
            high_mean > 3.0 * low_mean,
            "spectrum is not blue: high-band mean power {high_mean:.1} should dominate \
             the low band {low_mean:.1} (a clustered or white mask fails here)"
        );
    }
}
