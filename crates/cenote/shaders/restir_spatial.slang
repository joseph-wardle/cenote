// Spatial reuse: fold each of k screen-space neighbours' reservoirs into the
// pixel's own under defensive pairwise MIS, so one pixel's hard-won light
// sample lightens its neighbourhood. The third of M3's four discrete reservoir
// stages (D-091), and the first that reads *other* pixels — every earlier stage
// touched only its own pixel, which is why this one carries the step's whole
// correctness weight (D-093).
//
// -- The estimator -----------------------------------------------------------
// The pixel's own reservoir (from restir_candidates) is the *canonical*
// candidate; the k neighbours are non-canonical. All k+1 are combined into a
// fresh reservoir at the canonical (this pixel's) surface — the DI shift is the
// identity because the reservoir stores a light *surface point*, so the GRIS
// Jacobian is 1 (D-087) and the shift is just "re-evaluate the target here".
//
// MIS is the *defensive* confidence-weighted pairwise heuristic (Bitterli 2022
// thesis §9.1 / the 2023 course §7.1.3, Eq 7.8). Each neighbour is compared only
// to the canonical — O(k), not the O(k²) generalized balance heuristic. With
// c_c the canonical confidence, c_i a neighbour's, c_tot their sum, V the count
// of *accepted* neighbours, and t_c = c_c/c_tot the defensive fraction:
//
//   b_i = c_i·p̂_i(Y_i) / (c_i·p̂_i(Y_i) + c_c·p̂_c(Y_i))       neighbour i at Y_i
//   m_i = (1 − t_c)·(1/V)·b_i
//   m_c = t_c + (1 − t_c)·(1/V)·Σ_i c_c·p̂_c(Y_c)/(c_i·p̂_i(Y_c) + c_c·p̂_c(Y_c))
//
// which provably sum to 1 (the unbiasedness condition) for any positive targets:
// (1−t_c)(1/V)[Σb + Σ(1−b)] + t_c = (1−t_c) + t_c = 1. The *defensive* t_c floor
// — RTXDI ships the plain form (Eq 7.7) without it — matters here because the
// canonical is the one technique that covers the full support of f (D-088), and
// t_c > 0 keeps it from ever being zero-weighted by a neighbour whose target
// poorly approximates this surface's. That is what makes the converge-to-
// reference checkpoint (D-090) reachable, so it is required, not gilding.
//
// -- Visibility-aware target, folded into W ----------------------------------
// p̂_c — the canonical (receiving) target — is **visibility-aware**: p̂_c(Y) =
// U_c(Y)·V(Y), the unshadowed luminance times a shadow ray traced from *this*
// surface to Y's light point (D-093, the course's recommended Tip 4.3 variant).
// p̂_i — a neighbour's own-domain target, used only inside the MIS ratios — is
// left unshadowed, so exactly one ray is spent per candidate: k neighbours + the
// canonical = k+1 inline rays through the shared `shadowVisibility` helper.
// Re-tracing at *this* pixel (never trusting the neighbour's stored occlusion)
// is what keeps reuse unbiased across a shadow boundary, the classic ReSTIR bias
// source (D-087).
//
// The survivor's contribution weight is finalized by dividing weightSum by the
// *unshadowed* U_c(survivor), not p̂_c. Since the survivor was resampled ∝ p̂_c =
// U_c·V, dividing by U_c folds its own visibility into W (W = weightSum/U_c is
// exactly weightSum/p̂_c · V). restir_resolve then shades the survivor with the
// plain unshadowed L·f·cos and no shadow ray of its own — the ray was already
// spent here. Equivalent to the textbook "shade with a resolve-time shadow ray"
// but one ray cheaper (D-093).
//
// -- Neighbour geometry, and the range guard ---------------------------------
// A neighbour's own target p̂_i needs its surface (position, frame, closure),
// reconstructed from the path pool at slot `neighbourPixel − base`, where the
// range's base is recovered per-thread as `myPixel − mySlot` (raygen wrote
// pixel = base + slot). A target larger than the pool is walked in ranges, and
// this stage runs per range; a neighbour in a *different* range has no live path
// state, so it is dropped — verified by re-reading paths.pixel at the slot and
// requiring it to match. Dropped neighbours simply leave the accepted count V
// (the canonical absorbs their share); the only cost is slightly less reuse in a
// one-pixel band at a range seam, never bias. A neighbour that is a miss, or was
// never reached, reads confidence 0 from the wave-cleared reservoir buffer and
// is likewise dropped.
//
// Runs at bounce 0 only, dispatched from the hit queue it reads but never
// consumes. Reads the committed candidate reservoirs (`reservoirsIn`), writes
// the merged survivors to a separate buffer (`reservoirsOut`) — the ping-pong a
// barrier orders, so no thread reads a half-written neighbour and determinism
// holds (M3 plan §2).

import bluenoise;
import color;
import openpbr;
import pathstate;
import reservoir;
import reservoir_path;
import restir_mis;
import restir_scene;
import restir_target;
import rng;
import scene;
import surface;

// The scene TLAS, shared with `intersect`/`trace_shadow`: the k+1 visibility
// rays trace it through the shared `shadowVisibility` helper.
[vk::binding(0, 0)]
RaytracingAccelerationStructure sceneTlas;

// The most neighbours a spatial pass will gather — the fixed bound on the
// per-thread accepted-neighbour store below. The host caps k at this
// (wavefront.rs); k default is ~5 (D-088).
static const uint MAX_SPATIAL_NEIGHBOURS = 8;

struct Params {
    Paths paths;
    Queue<uint> hits;            // primary hits, dispatched indirectly; not consumed
    SceneTable* scene;
    RestirScene* restir;
    StoredPathReservoir* reservoirsIn;  // candidate stage's output — self + neighbours, read
    StoredPathReservoir* reservoirsOut; // this stage's output — the merged survivor, written
    uint sampleIndex;
    uint width;
    uint height;
    uint capacity;               // path-pool capacity — a neighbour slot must be < this
    uint neighbours;             // k
    float radius;                // gather radius, pixels
    float normalThreshold;       // reuse gate: dot(n_c, n_i) must exceed this
    float depthThreshold;        // reuse gate: relative camera-depth difference cap
};

[vk::push_constant]
Params params;

// The reconstructed `Surface`, the light-sample `connectSampleAt`, the
// unshadowed target `targetUnshadowed`, and the cross-domain shift-and-target
// block (`shiftIntoDomain`/`targetInDomain`) this stage shares with the
// temporal combine live in `restir_target.slang` (imported above) — one copy,
// so the two reuse stages can never drift into a silent bias. Only the
// visibility-aware NEE target below stays here: shadowing the NEE arm is this
// stage's visibility policy (temporal's is unshadowed, D-094), not shared
// machinery.

// The visibility-aware canonical target p̂_c = U_c·V of a light sample at *this*
// surface, tracing one shadow ray. Returns p̂_c and reports the unshadowed U_c
// through `unshadowed` — the survivor's finalize divides by U_c, folding its own
// V into W (see the header). One inline ray, or none when the unshadowed target
// is already zero (nothing to shade, so nothing to occlude).
float targetVisAware(Surface s, PathSample sample, out float unshadowed) {
    let t = evalTarget(params.scene, params.restir, s, sample);
    unshadowed = t.unshadowed;
    if (!(t.unshadowed > 0.0)) {
        return 0.0; // nothing to shade, so nothing to occlude — skip the ray
    }
    let c = t.connection;
    RayDesc ray;
    ray.Origin = t.throughSurface ? s.below : s.position;
    ray.Direction = c.toLight;
    ray.TMin = 0.0;
    ray.TMax = c.shadowTmax;
    let v = shadowVisibility(sceneTlas, params.scene, ray, c.shadowLight, c.shadowPrimitive);
    return t.unshadowed * v;
}

// One accepted neighbour, held between the gather pass (which spends its rays)
// and the stream pass (which needs t_c and V, known only once every neighbour
// is gathered). `IDefaultInitializable` so the fixed-size array below can be
// default-constructed — a struct holding a `PathSample` needs the
// conformance spelled out for that.
struct AcceptedNeighbour : IDefaultInitializable {
    PathSample sample;
    float confidence;  // c_i
    float weightNumer; // b_i · p̂_c(Y_i) · W_i — the resampling weight before (1−t_c)/V
    float unshadowed;  // U_c(Y_i) — the finalize divisor if this neighbour wins
    float accept;      // WRS acceptance draw
};

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void restir_spatial(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.hits.count()) {
        return;
    }
    uint slot = params.hits.entries[tid.x];
    uint pixel = params.paths.pixel[slot];

    // The range's base pixel: raygen wrote pixel = base + slot, so base recovers
    // per-thread with no push constant. A neighbour is in this range iff its slot
    // (neighbourPixel − base) addresses the same pixel back.
    uint base = pixel - slot;

    Surface canonical = reconstructSurface(params.paths, params.scene, slot);
    Reservoir<PathSample> canonicalReservoir = loadPathReservoir(params.reservoirsIn[pixel]);
    let cCanon = canonicalReservoir.confidence; // c_c > 0 for any hit pixel

    // p̂_c(Y_c) and U_c(Y_c): the canonical sample re-targeted here, with its one
    // visibility ray — the "+1" of the k+1. An empty/degenerate canonical (rare
    // at a real hit) yields 0 and simply loses to any neighbour. An indirect
    // sample's — reconnection or replay — own-domain target is its stored
    // integrand (D-138/D-139): generation formed F in this very domain this
    // frame (temporal never passes a foreign indirect sample through), its
    // path traced concretely — visibility included — so it is trusted, and no
    // ray is spent.
    float ucYc = 0.0;
    float pcYc = 0.0;
    if (canonicalReservoir.unbiasedWeight > 0.0) {
        if (pathKind(canonicalReservoir.sample) != PATH_KIND_NEE) {
            ucYc = luminance(canonicalReservoir.sample.F);
            pcYc = ucYc;
        } else {
            pcYc = targetVisAware(canonical, canonicalReservoir.sample, ucYc);
        }
    }

    let px = int(pixel % params.width);
    let py = int(pixel / params.width);

    // The blue-noise-ranked sample index, shared across this stage's draws for
    // the pixel so its reuse error stays screen-space blue noise (D-095) —
    // coherent with candidates, which ranked the same (pixel, sample) key.
    let ranked = blueRank(uint2(uint(px), uint(py)), params.sampleIndex);

    // -- Gather: gate each neighbour, spend its ray, stash it --------------------
    AcceptedNeighbour accepted[MAX_SPATIAL_NEIGHBOURS];
    uint validCount = 0;             // V — accepted neighbours
    float cTot = cCanon;             // c_tot — running Σ confidence
    float canonShareSum = 0.0;       // Σ_i c_c·p̂_c(Y_c)/(c_i·p̂_i(Y_c) + c_c·p̂_c(Y_c))

    let k = min(params.neighbours, MAX_SPATIAL_NEIGHBOURS);
    for (uint n = 0; n < k; n++) {
        let uOffset = sample_ranked_2d(pixel, ranked,
                                       restirSpatialDimension(n, RESTIR_SPATIAL_OFFSET));
        // A uniform point on the gather disk (√u for area-uniform radius), rounded
        // to a pixel offset. The self pixel (zero offset) is the canonical — never
        // reused as its own neighbour, or it would double-count.
        let r = params.radius * sqrt(uOffset.x);
        let theta = 6.28318530718 * uOffset.y;
        let dx = int(round(r * cos(theta)));
        let dy = int(round(r * sin(theta)));
        if (dx == 0 && dy == 0) {
            continue;
        }
        let nx = px + dx;
        let ny = py + dy;
        if (nx < 0 || ny < 0 || nx >= int(params.width) || ny >= int(params.height)) {
            continue;
        }
        let neighbourPixel = uint(ny) * params.width + uint(nx);

        // The neighbour's path slot in *this* range. Out of range (a different
        // pool walk, or the arithmetic underflowed) → no live path state → drop.
        let neighbourSlot = neighbourPixel - base;
        if (neighbourSlot >= params.capacity
            || params.paths.pixel[neighbourSlot] != neighbourPixel) {
            continue;
        }

        // A miss or unreached neighbour reads confidence 0 from the wave-cleared
        // buffer — nothing to reuse, drop before its surface is rebuilt.
        Reservoir<PathSample> neighbour = loadPathReservoir(params.reservoirsIn[neighbourPixel]);
        if (!(neighbour.confidence > 0.0)) {
            continue;
        }

        Surface neighbourSurface = reconstructSurface(params.paths, params.scene, neighbourSlot);
        // The reuse gate: neighbours must share this surface's orientation and
        // camera depth, or their light samples describe a different shading
        // context (D-093, stricter than RTXDI's 0.5/0.1 on the normal — the
        // conservative choice, and the first knob to loosen if convergence lags).
        if (dot(canonical.normal, neighbourSurface.normal) <= params.normalThreshold) {
            continue;
        }
        if (abs(canonical.depth - neighbourSurface.depth)
            > params.depthThreshold * canonical.depth) {
            continue;
        }

        // The shifted targets (D-134/D-135/D-138/D-139). An indirect sample —
        // reconnection or replay — crosses through the shared forward block
        // (`shiftIntoDomain`, restir_target.slang): the shift map, its one
        // visibility ray, the partial-shift zero for an undefined shift
        // (D-136), and the re-rooted sample to store on a win. An NEE sample
        // stores a domain-independent light point — identity shift, Jacobian
        // 1 — through this stage's own arms: the visibility-aware p̂_c and
        // the unshadowed own-domain p̂_i.
        float ucYi = 0.0;
        float pcYiShifted = 0.0; // p̂_c(shift(Y_i))·J_i, 1 ray
        float piYi = 0.0;        // p̂_i(Y_i)
        var shifted = neighbour.sample;
        if (pathKind(neighbour.sample) != PATH_KIND_NEE) {
            let sh = shiftIntoDomain(sceneTlas, params.scene,
                                     neighbour.sample, canonical);
            pcYiShifted = sh.target;
            piYi = sh.ownTarget;
            ucYi = sh.unshadowed;
            shifted = sh.sample;
        } else {
            pcYiShifted = targetVisAware(canonical, neighbour.sample, ucYi);
            piYi = targetUnshadowed(params.scene, params.restir,
                                    neighbourSurface, neighbour.sample);
        }
        // Symmetrically, the canonical's sample stated in the neighbour's
        // domain: p̂_i(Y_c)·J through the shared ray-free reverse
        // (`targetInDomain` — own-domain targets spend no ray, any kind).
        // Skipped when its own-domain target is already zero (the canonical's
        // share is zero regardless), which also spares the per-neighbour
        // replay rays.
        float piYcShifted = 0.0;
        if (canonicalReservoir.unbiasedWeight > 0.0 && pcYc > 0.0) {
            piYcShifted = targetInDomain(sceneTlas, params.scene, params.restir,
                                         canonicalReservoir.sample,
                                         neighbourSurface);
        }

        let ci = neighbour.confidence;
        // b_i at Y_i, and the canonical's share of this pair at Y_c.
        let mis = pairwiseMis(ci, piYi, piYcShifted, cCanon, pcYiShifted, pcYc);

        AcceptedNeighbour a;
        a.sample = shifted;
        a.confidence = ci;
        // The GRIS weight of the shifted neighbour: m_i·p̂_c(shift(Y_i))·W_i·J_i.
        a.weightNumer = mis.bi * pcYiShifted * neighbour.unbiasedWeight;
        a.unshadowed = ucYi;
        a.accept = sample_ranked_1d(pixel, ranked,
                                    restirSpatialDimension(n, RESTIR_SPATIAL_ACCEPT));
        accepted[validCount] = a;
        validCount++;
        cTot += ci;
        canonShareSum += mis.canonShare;
    }

    // -- Stream: now that t_c and V are known, combine under the MIS weights -----
    let tc = cTot > 0.0 ? cCanon / cTot : 1.0; // c_c/c_tot; =1 when V=0 (canonical only)
    let invV = validCount > 0 ? 1.0 / float(validCount) : 0.0;

    Reservoir<PathSample> merged = reservoirInit<PathSample>();
    float survivorTarget = 0.0;

    // The canonical: m_c = t_c + (1−t_c)·(1/V)·Σ shares. With V=0 the sum term
    // vanishes (t_c=1) and it passes through as its sole candidate.
    let mCanon = tc + (1.0 - tc) * invV * canonShareSum;
    let acceptCanon = sample_ranked_1d(pixel, ranked, restirSpatialCanonicalDimension());
    if (reservoirMerge(merged, canonicalReservoir, mCanon * pcYc * canonicalReservoir.unbiasedWeight,
                       acceptCanon)) {
        survivorTarget = ucYc;
    }

    // Each accepted neighbour: m_i = (1−t_c)·(1/V)·b_i, its resampling weight the
    // stashed b_i·p̂_c(Y_i)·W_i scaled by that. reservoirMerge inherits the
    // neighbour's confidence, so the merged M is c_tot — the compounded history.
    for (uint i = 0; i < validCount; i++) {
        let a = accepted[i];
        Reservoir<PathSample> nb = reservoirInit<PathSample>();
        nb.sample = a.sample;
        nb.confidence = a.confidence;
        let weight = (1.0 - tc) * invV * a.weightNumer;
        if (reservoirMerge(merged, nb, weight, a.accept)) {
            survivorTarget = a.unshadowed;
        }
    }

    // Finalize by the *unshadowed* survivor target: W = weightSum/U_c folds the
    // survivor's own visibility into W (it was resampled ∝ p̂_c = U_c·V), so
    // restir_resolve shades it unshadowed with no further ray.
    reservoirFinalize(merged, survivorTarget);
    params.reservoirsOut[pixel] = storePathReservoir(merged);
}
