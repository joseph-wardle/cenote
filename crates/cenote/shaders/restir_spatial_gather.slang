// Spatial reuse, first half (M6 step 7b): every pixel's *forward* shift
// evaluations, run once and recorded, so the ray-free combine
// (`restir_spatial.slang`) can read each pair's two evaluations — its own
// forward, its reciprocal partner's forward — instead of re-running the
// backshift per neighbour. The records and the reciprocity argument that makes
// a partner's record always fresh when read live in `restir_pair.slang`; the
// estimator itself is untouched (the checkpoint is bit-identity with 7a), and
// everything this kernel computes is exactly what the pre-split stage's gather
// loop computed, in the same order with the same guards.
//
// Per hit pixel: reconstruct the canonical surface, evaluate the pixel's own
// sample there (the self record: the raw unshadowed pieces unguarded — the
// partner's own-domain read — and the visibility-aware target under the
// pre-split stage's exact W guard), then walk the k pairing-texture partners
// through the reuse gates and, for each accepted one, run the forward shift
// with its one visibility ray at *this* surface and record the
// shift-dependent values. All of the stage's rays are spent here; the combine
// spends none.
//
// Runs at bounce 0 only, dispatched from the hit queue it reads but never
// consumes, between the candidate/temporal reservoir commit and the combine —
// the full inter-pass barrier orders the records for the combine's
// cross-pixel reads.

import color;
import openpbr;
import pairing;
import pathstate;
import reservoir;
import reservoir_path;
import restir_pair;
import restir_scene;
import restir_target;
import rng;
import scene;
import surface;

// The scene TLAS, shared with `intersect`/`trace_shadow`: the k+1 visibility
// rays trace it through the shared `shadowVisibility` helper.
[vk::binding(0, 0)]
RaytracingAccelerationStructure sceneTlas;

struct Params {
    Paths paths;
    Queue<uint> hits;            // primary hits, dispatched indirectly; not consumed
    SceneTable* scene;
    RestirScene* restir;
    StoredPathReservoir* reservoirsIn;  // candidate stage's output — self + neighbours, read
    PairRecord* pairs;           // forward records, k per slot — written
    SelfRecord* selfs;           // self records, one per slot — written
    uint sampleIndex;
    uint dims;                   // width | height << 16 — packed to keep 128 B
    uint capacity;               // path-pool capacity — a partner slot must be < this
    uint neighbours;             // k
    float normalThreshold;       // reuse gate: dot(n_c, n_i) must exceed this
    float depthThreshold;        // reuse gate: relative camera-depth difference cap
};

[vk::push_constant]
Params params;

// The visibility ray on top of an already-evaluated unshadowed target: p̂ =
// U·V with one inline ray, or zero rays when U is already zero (nothing to
// shade, so nothing to occlude). The pre-split stage's `targetVisAware`, split
// from its `evalTarget` so the self record can keep the raw pieces too.
float visAware(TargetEval t, Surface s) {
    if (!(t.unshadowed > 0.0)) {
        return 0.0;
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

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void restir_spatial_gather(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.hits.count()) {
        return;
    }
    uint slot = params.hits.entries[tid.x];
    uint pixel = params.paths.pixel[slot];
    let width = params.dims & 0xffff;
    let height = params.dims >> 16;

    // The range's base pixel: raygen wrote pixel = base + slot, so base recovers
    // per-thread with no push constant. A partner is in this range iff its slot
    // (partnerPixel − base) addresses the same pixel back.
    uint base = pixel - slot;

    Surface canonical = reconstructSurface(params.paths, params.scene, slot);
    Reservoir<PathSample> canonicalReservoir =
        loadPathReservoir(params.reservoirsIn[pixel]);
    let canonNee = pathKind(canonicalReservoir.sample) == PATH_KIND_NEE;

    // -- The self record: this pixel's own sample at its own surface ---------
    // The raw unshadowed pieces are recorded *unguarded* — the reciprocal
    // partner reads them as an NEE sample's own-domain target and fOwn, which
    // the pre-split stage evaluated unconditionally — while the
    // visibility-aware p̂_c(Y_c) keeps that stage's exact guard (no ray, and a
    // zero target, for a zero-W canonical). An indirect sample's own-domain
    // target is its stored integrand (D-138/D-139), no ray any way.
    SelfRecord self;
    self.albedo = canonical.closure.aovAlbedo;
    self.accepted = 0;
    self.pad0 = 0;
    self.pad1 = 0;
    self.pad2 = 0;
    if (!canonNee) {
        self.rgb = canonicalReservoir.sample.F;
        self.ucYc = luminance(canonicalReservoir.sample.F);
        self.pcYc = canonicalReservoir.unbiasedWeight > 0.0 ? self.ucYc : 0.0;
    } else {
        let t = evalTarget(params.scene, params.restir, canonical,
                           canonicalReservoir.sample);
        self.rgb = t.contribution;
        self.ucYc = t.unshadowed;
        self.pcYc = canonicalReservoir.unbiasedWeight > 0.0 ? visAware(t, canonical)
                                                            : 0.0;
    }

    let px = int(pixel % width);
    let py = int(pixel / width);

    // -- The forward gather: gate each partner, shift, spend the ray, record --
    let k = min(params.neighbours, PAIRING_TEXTURES);
    for (uint n = 0; n < k; n++) {
        // Slot n's reciprocal partner this frame (pairing.slang). Never the
        // self pixel — the textures have no fixed points.
        let d = pairingOffset(n, int2(px, py), params.sampleIndex);
        let nx = px + d.x;
        let ny = py + d.y;
        if (nx < 0 || ny < 0 || nx >= int(width) || ny >= int(height)) {
            continue;
        }
        let partnerPixel = uint(ny) * width + uint(nx);

        // The partner's path slot in *this* range. Out of range (a different
        // pool walk, or the arithmetic underflowed) → no live path state → drop.
        let partnerSlot = partnerPixel - base;
        if (partnerSlot >= params.capacity
            || params.paths.pixel[partnerSlot] != partnerPixel) {
            continue;
        }

        // A miss or unreached partner reads confidence 0 from the wave-cleared
        // buffer — nothing to reuse, drop before its surface is rebuilt.
        Reservoir<PathSample> neighbour =
            loadPathReservoir(params.reservoirsIn[partnerPixel]);
        if (!(neighbour.confidence > 0.0)) {
            continue;
        }

        Surface neighbourSurface = reconstructSurface(params.paths, params.scene,
                                                      partnerSlot);
        // The reuse gate: partners must share this surface's orientation and
        // camera depth, or their light samples describe a different shading
        // context (D-093, stricter than RTXDI's 0.5/0.1 on the normal — the
        // conservative choice, and the first knob to loosen if convergence
        // lags). Both tests are symmetric in the pair (7a): the depth
        // difference is measured against the smaller depth — which is also
        // what lets the combine trust that an accepted partner accepted us
        // back and wrote its records (restir_pair.slang).
        if (dot(canonical.normal, neighbourSurface.normal) <= params.normalThreshold) {
            continue;
        }
        if (abs(canonical.depth - neighbourSurface.depth)
            > params.depthThreshold * min(canonical.depth, neighbourSurface.depth)) {
            continue;
        }

        // The forward shift and targets (D-134/D-135/D-138/D-139), exactly the
        // pre-split gather's: an indirect sample — reconnection or replay —
        // crosses through the shared forward block (`shiftIntoDomain`,
        // restir_target.slang), which now also reports the `targetInDomain`
        // value the partner's backshift needs, from the same shift call. An
        // NEE sample stores a domain-independent light point — identity
        // shift, Jacobian 1 — through the visibility-aware arm above; its
        // backshift value is the same unshadowed target, and its own-domain
        // target rides the partner's self record instead.
        PairRecord rec;
        if (pathKind(neighbour.sample) != PATH_KIND_NEE) {
            let sh = shiftIntoDomain(sceneTlas, params.scene,
                                     neighbour.sample, canonical);
            rec.F = sh.sample.F;
            rec.partial = sh.sample.cachedJacobian;
            rec.visTarget = sh.target;
            rec.unshadowed = sh.unshadowed;
            rec.ownTarget = sh.ownTarget;
            rec.backTarget = sh.backTarget;
        } else {
            let t = evalTarget(params.scene, params.restir, canonical,
                               neighbour.sample);
            rec.F = t.contribution;
            rec.partial = 0.0;
            rec.visTarget = visAware(t, canonical);
            rec.unshadowed = t.unshadowed;
            rec.ownTarget = 0.0; // in the partner's self record (restir_pair)
            rec.backTarget = t.unshadowed;
        }
        params.pairs[slot * k + n] = rec;
        self.accepted |= 1u << n;
    }
    params.selfs[slot] = self;
}
