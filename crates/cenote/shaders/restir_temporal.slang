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
//   1. Its NEE arms are **unshadowed**. Temporal spends no shadow ray on a
//      light sample: both sides of an NEE pair are weighed by the plain
//      unshadowed target p̂ = luminance(L·f·|cosθ|), and an NEE survivor's
//      `curr` carries the *unshadowed* W. The one committed visibility test
//      still happens exactly once downstream — spatial's V-fold, or resolve's
//      traced shadow ray — so the feed convention (unshadowed W into
//      `prev`/spatial, visibility-folded W into the dead-end `scratch`) is
//      never mixed (D-094). An *indirect* prev sample — reconnection or
//      replay — crosses the frame boundary through the shared shift block
//      instead (`shiftIntoDomain`, restir_target.slang; step 5b): its F is
//      re-formed at this pixel through the current TLAS — prefix replayed,
//      reconnection visibility re-traced, Jacobian priced — exactly as
//      spatial shifts a screen-space neighbour. That ray belongs to the
//      shift (the integrand is undefined without it), not to an NEE resolve;
//      on a win the *re-rooted* sample is stored, its W finalized by its
//      complete integrand — precisely what generation at this pixel would
//      have produced, so downstream's canonical trust (D-138/D-139) holds.
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
// -- The frame boundary and the epoch gate (step 5b) -------------------------
// History may have been rendered against an *older scene build*: an edit
// re-uploads the instance tables and may renumber TLAS custom indices, and an
// indirect sample at rest holds one raw (`rcVertex.instance`). The host
// reports whether prev's build is still current (`prevSameScene`); when it
// isn't, indirect history is dropped before any dereference — the neighbour
// simply doesn't exist that frame — while NEE history keeps surviving edits
// through the light-id registry (M3). Camera-only motion rebuilds nothing, so
// the common temporal case (orbiting) never trips the gate.
//
// -- The decay ramp (step 5d) ------------------------------------------------
// The history confidence is scaled by a short linear ramp on samples-since-reset
// (`sampleIndex`), `decay = saturate(1 − sampleIndex / decayFrames)`, before the
// M-cap and the combine. On a moving camera the film resets every frame, so
// `sampleIndex` stays low and `decay ≈ 1` — temporal is fully live, the
// warm-start. On a held camera `sampleIndex` climbs and `decay` reaches *exactly*
// zero at `decayFrames`; from there the zeroed confidence folds into
// `prevValid` before any shift work (§4c decision 3) — the neighbour is
// dropped outright, no shift or visibility ray spent — and the combine
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
import color;
import pathstate;
import reservoir;
import reservoir_path;
import restir_mis;
import restir_reproject;
import restir_scene;
import restir_target;
import rng;
import scene;

// The scene TLAS, as every reuse stage binds it: the cross-frame shift of an
// indirect prev sample traces it — replay segments and the shift's one
// visibility ray. The NEE arms trace nothing (D-094).
[vk::binding(0, 0)]
RaytracingAccelerationStructure sceneTlas;

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
    // 1 when `prev` was rendered against the current scene build, 0 across an
    // edit — the epoch gate (see the header). At 0 indirect history is
    // dropped before its stored TLAS index is ever dereferenced.
    uint prevSameScene;
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

    StoredPathReservoir candStored = params.reservoirsIn[pixel];
    Reservoir<PathSample> cand = loadPathReservoir(candStored);
    let cCanon = cand.confidence; // c_c > 0 for any hit pixel

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
        // The epoch gate (see the header): across a scene edit an indirect
        // prev sample's `rcVertex.instance` is a raw TLAS custom index the
        // rebuild may have renumbered — nothing may dereference it, so the
        // neighbour simply doesn't exist this frame. The kind bits are plain
        // data, safe to read; NEE history flows on through the light-id
        // registry. One frame of indirect warm-start is the whole cost, on
        // an edit that reset the film anyway.
        let staleIndirect = params.prevSameScene == 0
                          && pathKind(prev.sample) != PATH_KIND_NEE;
        if (!staleIndirect && prev.confidence > 0.0) {
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
    // read, before either — the consistency the microtest guards.
    let decay = params.decayFrames > 0
        ? saturate(1.0 - float(params.sampleIndex) / float(params.decayFrames))
        : 1.0;
    if (prevValid) {
        prev.confidence = min(prev.confidence, params.mCap * cCanon) * decay;
        // A decayed-to-zero history is no neighbour at all (§4c decision 3).
        // The estimate is untouched — a zero-confidence neighbour's MIS and
        // resampling weights already vanish (both carry c_i), and with the
        // pair gone t_c = 1 collapses the combine to the same pass-through —
        // but folding the zero into `prevValid` here, *before* any shift
        // work, is what lets the converged still (decay 0) provably spend
        // zero temporal rays: `curr ≡ cand` with nothing traced.
        prevValid = prev.confidence > 0.0;
    }

    // p̂_c(Y_c): the canonical (this frame's candidate) sample's unshadowed target
    // on this surface. Re-evaluated rather than trusting the stored W, exactly as
    // spatial re-targets its canonical. Zero for an empty/degenerate candidate.
    // An indirect candidate's — reconnection or replay — own-domain target is
    // its stored integrand (D-138/D-139): candidates formed F at this very
    // pixel this frame, and re-forming it here would re-trace its replay.
    float pcYc = 0.0;
    if (cand.unbiasedWeight > 0.0) {
        pcYc = pathKind(cand.sample) != PATH_KIND_NEE
            ? luminance(cand.sample.F)
            : targetUnshadowed(params.scene, params.restir, surface, cand.sample);
    }

    // The cross-frame pair (step 5b), spatial's per-neighbour block with the
    // one reprojected neighbour. An indirect sample — reconnection or replay
    // — crosses through the shared shift-and-target block
    // (restir_target.slang): forward, `shiftIntoDomain` re-forms prev's F at
    // this pixel through the current TLAS — the shift map, its one
    // visibility ray, the partial-shift zero for an undefined shift (D-136)
    // — and hands back the re-rooted sample to store on a win; reverse,
    // `targetInDomain` states cand's sample in prev's domain ray-free, as
    // own-domain targets always are. The NEE arms are this stage's own and
    // *unshadowed* — no shadow ray, the committed visibility test still
    // downstream (D-094).
    //
    // The pairwise ratios (spatial's, p̂_i on the reconstructed *prev*
    // surface):
    //   b_i        = c_i·p̂_i(Y_i) / (c_i·p̂_i(Y_i) + c_c·p̂_c(Y_i))   at prev's sample
    //   canonShare = c_c·p̂_c(Y_c) / (c_i·p̂_i(Y_c) + c_c·p̂_c(Y_c))   at cand's sample
    let ci = prevValid ? prev.confidence : 0.0;
    let cTot = cCanon + ci;
    let tc = cTot > 0.0 ? cCanon / cTot : 1.0; // c_c/c_tot; =1 when prev is empty
    let invV = prevValid ? 1.0 : 0.0;          // V ∈ {0, 1}

    float pcYi = 0.0;              // p̂_c(shift(Y_i))·J — prev's sample priced here
    float ucYi = 0.0;              // its unshadowed target — the finalize divisor
    float bi = 0.0;
    float canonShare = 0.0;
    var prevShifted = prev.sample; // the re-rooted sample, stored on a prev win
    if (prevValid) {
        float piYi = 0.0; // p̂_i(Y_i) — prev's own-domain target
        if (pathKind(prev.sample) != PATH_KIND_NEE) {
            let sh = shiftIntoDomain(sceneTlas, params.scene, prev.sample, surface);
            pcYi = sh.target;
            piYi = sh.ownTarget;
            ucYi = sh.unshadowed;
            prevShifted = sh.sample;
        } else {
            // An NEE sample stores a domain-independent light point —
            // identity shift, Jacobian 1 — priced unshadowed on both
            // surfaces (temporal's NEE policy; spatial's is vis-aware).
            pcYi = targetUnshadowed(params.scene, params.restir, surface, prev.sample);
            piYi = targetUnshadowed(params.scene, params.restir, prevSurface, prev.sample);
            ucYi = pcYi;
        }
        // The canonical's sample stated in prev's domain. Skipped when its
        // own-domain target is already zero (its share is zero regardless),
        // which also spares the replay rays — spatial's same guard.
        float piYc = 0.0;
        if (cand.unbiasedWeight > 0.0 && pcYc > 0.0) {
            piYc = targetInDomain(sceneTlas, params.scene, params.restir,
                                  cand.sample, prevSurface);
        }
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
    // b_i·p̂_c(shift(Y_i))·J·W_prev scaled by that. The merged reservoir takes
    // the *shifted* sample — the one whose F is this pixel's — and inherits
    // prev's (capped) confidence, so the merged M is c_c + min(c_prev,
    // M_CAP·c_c) — the bounded history.
    if (prevValid) {
        let acceptPrev = sample_ranked_1d(pixel, ranked,
                                          restirTemporalDimension(RESTIR_TEMPORAL_PREV));
        Reservoir<PathSample> nb = reservoirInit<PathSample>();
        nb.sample = prevShifted;
        nb.confidence = prev.confidence;
        let weight = (1.0 - tc) * invV * bi * pcYi * prev.unbiasedWeight;
        if (reservoirMerge(merged, nb, weight, acceptPrev)) {
            survivorTarget = ucYi;
        }
    }

    // Finalize by the survivor's *unshadowed* target in this domain: W =
    // weightSum/p̂. An NEE survivor folded no visibility — temporal's NEE arms
    // are rayless — so `curr` carries the unshadowed W the next frame's prev
    // (and this frame's spatial) expects (D-094). An indirect survivor's
    // divisor is the luminance of its re-formed, complete integrand (a
    // winner's shift visibility was 1), so its stored F/W pair reads
    // downstream exactly as generation's would (D-138/D-139).
    reservoirFinalize(merged, survivorTarget);
    // The CV lane passes through untouched (6b-i): the candidate mean rides to
    // the spatial combine exactly as the candidate stage formed it. The 6b-ii
    // temporal recurrence — the decayed-M blend against prev's lane — lands
    // here, beside the merge it mirrors, when that rung opens.
    params.reservoirsOut[pixel] = storePathReservoir(merged, candStored.cvAccumulator);
}
