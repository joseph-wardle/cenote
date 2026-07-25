// Temporal reuse: fold last frame's committed reservoir (`prev`) into this
// frame's fresh candidate reservoir (`cand`), so a pixel's light sample compounds
// history across frames — the warm-start that makes a held camera converge in a
// handful of frames instead of hundreds. The stage that sits between
// `restir_candidates` and `restir_spatial` when temporal reuse is on (D-094):
// candidates writes `cand`, this writes `curr`, spatial reads `curr`.
//
// -- The estimator -----------------------------------------------------------
// Exactly the defensive pairwise-MIS combine `restir_spatial` runs, with one
// neighbour — the reprojected `prev` — and two differences:
//
//   1. It is **unshadowed**. Temporal adds no shadow rays; both the canonical
//      and the neighbour are weighed by the plain unshadowed target p̂ =
//      luminance(L·f·|cosθ|), and `curr` carries the *unshadowed* W. The one
//      committed visibility test still happens exactly once downstream —
//      spatial's V-fold, or resolve's traced shadow ray — so the feed
//      convention (unshadowed W into `prev`/spatial, visibility-folded W into
//      the dead-end `scratch`) is never mixed (D-094).
//
//   2. The neighbour's confidence is **M-capped** before it is combined:
//      c_prev ← min(c_prev, M_CAP · c_cand). A candidates frame already credits
//      ~M confidence, so the cap is a *history-length multiplier* — an absolute
//      cap near c_cand would throttle history to nothing, an unbounded one would
//      let a stale reservoir dominate forever. This clamp is the one
//      bias-critical numeric of the temporal stage, audited by the analytic
//      microtest (`restir_temporal_test.slang`). The merged reservoir inherits
//      the capped confidence, so history saturates at (1 + M_CAP)·c_cand rather
//      than growing without bound.
//
// -- Reprojection and the disocclusion gate (step 5c) ------------------------
// The neighbour is not blindly the same pixel: this frame's shading point is
// reprojected through last frame's camera to the pixel that resolved it, and
// `prev` is read there (`restir_reproject.slang`). A persistent G-buffer holds
// last frame's hit, so the previous surface is reconstructed in full — closure
// included — and the neighbour's own-domain target p̂_i is evaluated on *that*
// surface, genuinely distinct from the canonical p̂_c here (unlike 5b, where the
// same-pixel identity made p̂_i ≡ p̂_c). A disocclusion gate drops the reuse when
// the reprojected surface no longer matches (orientation, camera depth) — the
// same gate spatial runs on its neighbours. For a held camera reprojection is
// the identity pixel and the gate passes, so this reduces to same-pixel reuse;
// only a moved camera exercises the displacement. All of it is variance and
// quality — a wrong reuse is a bad candidate the pairwise MIS down-weights,
// never a bias (D-094) — so the estimator below is unchanged; only *which* prev
// reservoir and *which* surface feed it move.
//
// -- The decay ramp (step 5d) ------------------------------------------------
// The history confidence is scaled by a short linear ramp on samples-since-reset
// (`sampleIndex`), `decay = saturate(1 − sampleIndex / decayFrames)`, before the
// M-cap and the combine. On a moving camera the film resets every frame, so
// `sampleIndex` stays low and `decay ≈ 1` — temporal is fully live, the
// warm-start. On a held camera `sampleIndex` climbs and `decay` reaches *exactly*
// zero at `decayFrames`; from there `prev.confidence` is zero, so `ci = 0`, the
// neighbour's MIS weight and resampling weight both vanish, and the combine
// reduces to the candidate passing straight through — `curr ≡ cand`. The
// converged still is therefore temporal-*free*: spatial-only, fresh per-frame
// RNG, independent unbiased frames averaging at 1/N to the brute-force reference
// (D-085/D-094). The exact zero — not an asymptote — is what makes that provable;
// an exponential decay would leave the converged image forever correlated. Like
// reprojection and the gate, the ramp is variance and efficiency, never bias: it
// only reweights how much history the (already unbiased) pairwise MIS folds in.
// `decayFrames == 0` disables the ramp (`decay ≡ 1`) — the pinned-temporal CI
// gate forces temporal live to convergence, where the ordinary converge-to-
// reference test would otherwise go blind to a temporal bias (D-094).
//
// Runs at bounce 0 only, dispatched from the hit queue it reads but never
// consumes. Writes `curr` and this frame's G-buffer at its own pixel; reads
// `cand` at its own pixel and `prev` (+ its G-buffer) at the reprojected pixel.

import bluenoise;
import pathstate;
import reservoir;
import reservoir_path;
import restir_mis;
import restir_reproject;
import restir_scene;
import restir_target;
import rng;
import scene;

struct Params {
    Paths paths;
    Queue<uint> hits;              // primary hits, dispatched indirectly; not consumed
    SceneTable* scene;
    RestirScene* restir;
    StoredPathReservoir* reservoirsIn;   // this frame's candidate reservoirs (`cand`) — read
    StoredPathReservoir* reservoirsPrev; // last frame's committed reservoirs (`prev`) — read
    StoredPathReservoir* reservoirsOut;  // the temporal survivors (`curr`) — written
    Reproject* reproject;            // prev camera + G-buffers + gate thresholds (5c)
    uint sampleIndex;
    // History-length multiplier: c_prev is clamped to `mCap · c_cand` before
    // the combine (the bias-critical numeric — see the header).
    float mCap;
    // Decay window: history confidence is scaled by `saturate(1 −
    // sampleIndex/decayFrames)` so a held camera hands temporal off to
    // spatial-only accumulation (step 5d). `0` disables the ramp (the pinned CI
    // gate); see the header.
    uint decayFrames;
    uint pad0; // matches the host's tail padding to 8-byte alignment
};

[vk::push_constant]
Params params;

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void restir_temporal(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.hits.count()) {
        return;
    }
    uint slot = params.hits.entries[tid.x];
    uint pixel = params.paths.pixel[slot];
    Reproject reproject = params.reproject[0];

    // The blue-noise-ranked sample index, coherent with candidates/spatial for
    // this (pixel, sample) so the reuse error stays screen-space blue (D-095).
    // Width rides the reprojection block — temporal carries no resolution of its
    // own.
    let ranked = blueRank(uint2(pixel % reproject.width, pixel / reproject.width),
                          params.sampleIndex);

    Surface surface = reconstructSurface(params.paths, params.scene, slot);

    // Persist this frame's hit as the G-buffer next frame reprojects into. The
    // raw hit + ray, not a resolved surface, so the reconstruction can never
    // drift from the path tracer's. Written at every hit pixel; miss pixels keep
    // a stale entry that is never read (see the module header).
    GBufEntry entry;
    entry.hit = params.paths.hit[slot];
    entry.origin = params.paths.origin[slot];
    entry.direction = params.paths.direction[slot];
    reproject.currGbuffer[pixel] = entry;

    Reservoir<PathSample> cand = loadPathReservoir(params.reservoirsIn[pixel]);
    let cCanon = cand.confidence; // c_c ≥ M+1 > 0 for any hit pixel

    // Reproject this surface into last frame's screen and read `prev` there. The
    // neighbour is valid only if reprojection landed on-screen, that pixel
    // carries history (confidence > 0 — a hit last frame, so its G-buffer is
    // live), and the reprojected surface still matches this one (the gate).
    bool prevValid = false;
    Reservoir<PathSample> prev = reservoirInit<PathSample>();
    // Defaulted to this surface so it is definitely assigned; only ever read once
    // reprojection has overwritten it with the real previous surface (guarded by
    // `prevValid`).
    Surface prevSurface = surface;
    uint prevPixel;
    if (reprojectPixel(reproject, surface.position, prevPixel)) {
        prev = loadPathReservoir(params.reservoirsPrev[prevPixel]);
        if (prev.confidence > 0.0) {
            GBufEntry prevEntry = reproject.prevGbuffer[prevPixel];
            prevSurface = reconstructSurfaceFrom(params.scene, prevEntry.hit,
                                                 prevEntry.direction, prevEntry.origin.xyz);
            prevValid = surfacesAgree(surface.normal, surface.depth, prevSurface.normal,
                                      prevSurface.depth, reproject.normalThreshold,
                                      reproject.depthThreshold);
        }
    }

    // The M-cap, then the decay ramp: clamp last frame's compounded confidence to
    // a fixed multiple of one candidates frame, then scale by the samples-since-
    // reset ramp, and let the merged reservoir inherit that value (reservoirMerge
    // adds `prev.confidence`). The cap bounds history *length*; the decay anneals
    // the whole temporal contribution to zero as a held camera converges (see the
    // header). Both fold into the single confidence the MIS weights and the merge
    // read, before either — the consistency the microtest guards. `decay` hits
    // exactly zero past the window, so `prev.confidence` is zero and the neighbour
    // drops out entirely, leaving `curr ≡ cand`.
    let decay = params.decayFrames > 0
        ? saturate(1.0 - float(params.sampleIndex) / float(params.decayFrames))
        : 1.0;
    if (prevValid) {
        prev.confidence = min(prev.confidence, params.mCap * cCanon) * decay;
    }

    // p̂_c(Y_c): the canonical (this frame's candidate) sample's unshadowed target
    // on this surface. Re-evaluated rather than trusting the stored W, exactly as
    // spatial re-targets its canonical. Zero for an empty/degenerate candidate.
    float pcYc = 0.0;
    if (cand.unbiasedWeight > 0.0) {
        pcYc = targetUnshadowed(params.scene, params.restir, surface, cand.sample);
    }

    let ci = prevValid ? prev.confidence : 0.0;
    let cTot = cCanon + ci;
    let tc = cTot > 0.0 ? cCanon / cTot : 1.0; // c_c/c_tot; =1 when prev is empty
    let invV = prevValid ? 1.0 : 0.0;          // V ∈ {0, 1}

    // The pairwise ratios (spatial's, now with p̂_i on the reprojected *prev*
    // surface — distinct from p̂_c once the camera has moved):
    //   b_i        = c_i·p̂_i(Y_i) / (c_i·p̂_i(Y_i) + c_c·p̂_c(Y_i))   at prev's sample
    //   canonShare = c_c·p̂_c(Y_c) / (c_i·p̂_i(Y_c) + c_c·p̂_c(Y_c))   at cand's sample
    float pcYi = 0.0;      // p̂_c(Y_i) — cand's target at prev's sample, this surface
    float bi = 0.0;
    float canonShare = 0.0;
    if (prevValid) {
        pcYi = targetUnshadowed(params.scene, params.restir, surface, prev.sample);
        // p̂_i — prev's own-domain target — on the reconstructed previous surface.
        let piYi = targetUnshadowed(params.scene, params.restir, prevSurface, prev.sample);
        let piYc = cand.unbiasedWeight > 0.0
            ? targetUnshadowed(params.scene, params.restir, prevSurface, cand.sample)
            : 0.0;
        let mis = pairwiseMis(ci, piYi, piYc, cCanon, pcYi, pcYc);
        bi = mis.bi;
        canonShare = mis.canonShare;
    }

    Reservoir<PathSample> merged = reservoirInit<PathSample>();
    float survivorTarget = 0.0;

    // The canonical: m_c = t_c + (1−t_c)·(1/V)·canonShare. With prev empty the
    // share term vanishes (t_c = 1) and cand passes through as its sole candidate.
    let mCanon = tc + (1.0 - tc) * invV * canonShare;
    let acceptCanon = sample_ranked_1d(pixel, ranked,
                                       restirTemporalDimension(RESTIR_TEMPORAL_CANONICAL));
    if (reservoirMerge(merged, cand, mCanon * pcYc * cand.unbiasedWeight, acceptCanon)) {
        survivorTarget = pcYc;
    }

    // The neighbour: m_i = (1−t_c)·(1/V)·b_i, its resampling weight
    // b_i·p̂_c(Y_i)·W_prev scaled by that. reservoirMerge inherits prev's (capped)
    // confidence, so the merged M is c_c + min(c_prev, M_CAP·c_c) — the bounded
    // history.
    if (prevValid) {
        let acceptPrev = sample_ranked_1d(pixel, ranked,
                                          restirTemporalDimension(RESTIR_TEMPORAL_PREV));
        let weight = (1.0 - tc) * invV * bi * pcYi * prev.unbiasedWeight;
        if (reservoirMerge(merged, prev, weight, acceptPrev)) {
            survivorTarget = pcYi;
        }
    }

    // Finalize by the *unshadowed* canonical target at the survivor: W =
    // weightSum/p̂. Temporal folds no visibility, so — unlike spatial — this is
    // the plain target, and `curr` carries the unshadowed W the next frame's prev
    // (and this frame's spatial) expects.
    reservoirFinalize(merged, survivorTarget);
    params.reservoirsOut[pixel] = storePathReservoir(merged);
}
