// Heterogeneous-medium transport: null-collision (delta) tracking against a
// piecewise-constant majorant. Only shade_volume's tracking entry point
// imports this module, so gridless scenes — and every other kernel — pay
// none of PNanoVDB's code or the tracker's registers.
//
// The estimator is weighted delta tracking: one uniform decides each
// collision's fate from the channel-average probabilities, and the weight
// carries the spectral ratio. Absorption is never an analog termination —
// it rides the null weight (m − σ_t < m − σ_s), so a purely absorbing grid
// degenerates to ratio tracking exactly and survival stays Russian
// roulette's job.
//
// The majorant is the leading grid's lattice of per-cell density ceilings
// (prep's `Majorants`), walked cell by cell. A flight is drawn in optical
// depth rather than in distance: the walk spends that depth cell by cell,
// so empty space costs one lookup per cell instead of a collision, and one
// uniform still buys exactly one collision however many cells the flight
// crosses. A 1³ lattice is the scalar majorant this replaced, and takes the
// identical sequence of draws.

#define PNANOVDB_HLSL
#include "PNanoVDB.h"

import medium;    // MediaSplit
import pathstate; // MEDIUM_SLOTS
import rng;       // trackerUniform2
import scene;     // SceneTable, MediumRecord

// Density at a continuous index-space position, nanovdb convention: values
// on the integer lattice, trilinear over the 2×2×2 neighborhood.
//
// The result is bounded into [0, ceiling], and that bound is load-bearing
// rather than defensive: it is what makes σ_t ≤ m hold *by construction*, so
// no lattice error — a stale bound, an off-by-one cell at a boundary, a grid
// whose stored voxels outran the majorant built from them — can produce a
// negative null weight or a scatter probability above one. An underestimated
// majorant costs density, never soundness.
//
// Written as a comparison rather than `clamp`: SPIR-V's FClamp is
// FMin(FMax(…)), and both are *undefined* on a NaN operand, so an authored
// NaN voxel would land wherever the driver put it. `> 0.0` is false for NaN.
float sampleDensity(pnanovdb_buf_t buf, inout pnanovdb_readaccessor_t acc,
                    float3 index, float ceiling) {
    float3 base = floor(index);
    float3 t = index - base;
    int3 ijk0 = int3(base);
    float value = 0.0;
    [unroll]
    for (uint corner = 0; corner < 8; corner++) {
        int3 ijk = ijk0 + int3(corner & 1, (corner >> 1) & 1, corner >> 2);
        float weight = ((corner & 1) != 0 ? t.x : 1.0 - t.x)
            * (((corner >> 1) & 1) != 0 ? t.y : 1.0 - t.y)
            * ((corner >> 2) != 0 ? t.z : 1.0 - t.z);
        pnanovdb_address_t addr = pnanovdb_readaccessor_get_value_address(
            PNANOVDB_GRID_TYPE_FLOAT, buf, PNANOVDB_REF(acc), PNANOVDB_REF(ijk));
        value += weight * pnanovdb_read_float(buf, addr);
    }
    return value > 0.0 ? min(value, ceiling) : 0.0;
}

// The record's world→index affine applied to a world position.
float3 toIndex(MediumRecord record, float3 world) {
    let p = float4(world, 1.0);
    return float3(dot(record.worldToIndex[0], p),
                  dot(record.worldToIndex[1], p),
                  dot(record.worldToIndex[2], p));
}

// Where the tracker ended a sub-segment, and what carrying the path there
// costs — `MediumEvent` plus the phase context the closed form reads off
// its one record but a collision must evaluate in place.
public struct TrackEvent {
    public bool scattered; // did the path scatter before reaching `tmax`?
    public float distance; // where it scattered, or `tmax`
    public float3 weight;  // the collision weights' running product
    public float g;        // σ_s-weighted anisotropy at the scatter point
    public float3 albedo;  // σ_s/σ_t at the scatter point, for the guides
#ifdef PROBES
    public uint collisions;
    public uint nulls;
    public uint steps;
#endif
};

// Collisions per sub-segment before the tracker gives up and transmits what
// it has — the key space `trackerUniform2` is fed (12 bits), far above any
// real optical depth per sub-segment. A truncation reads as too little fog,
// the march's own degradation direction.
static const uint TRACKER_CAP = 1 << 12;

// And the same bound on lattice cells crossed, across every flight of one
// sub-segment. Three times the lattice's longest axis bounds a diagonal
// crossing, and this leaves room for any resolution a record can carry.
static const uint MAJORANT_STEP_CAP = 1 << 12;

// The lattice walk, in the leading grid's cell coordinates.
//
// NUMERICAL CONSTRAINT: boundary times are *stepped* (`tNext += tDelta`),
// never recomputed from a reprojected `origin + t·direction`. The round trip
// through world space lands epsilon behind the boundary just crossed, and
// the walk re-crosses it forever. Only `tNext += tDelta` is monotone.
//
// Grids beyond the first — an overlap, which prep does not forbid — get no
// second walk. Their whole-lattice maxima ride along in `constant`, which is
// a valid majorant and exactly the bound the scalar tracker used: the looser
// estimator is paid for only where volumes actually overlap. Which grid
// leads is `split.grids[0]`, the lowest occupied medium slot — a function of
// the order the path crossed the shells. Delta tracking is unbiased under
// *any* valid majorant, so that ordering costs variance, never correctness.
struct Walk {
    int3 cell;      // the cell the ray is in; outside the lattice at the ends
    int3 advance;   // ±1 per axis, 0 where the ray does not move along it
    float3 tNext;   // where each axis next crosses a cell boundary
    float3 tDelta;  // t per whole cell on each axis
    int3 res;       // the lattice's cell counts
    uint base;      // its byte offset in the pool
    float3 sigmaT;  // the leading grid's coefficients
    float3 constant; // the majorant everything but the walked cell contributes
    float ceiling;  // the current cell's density ceiling
};

// Where the current cell stops being the majorant's authority.
float cellExit(Walk walk) {
    return min(walk.tNext.x, min(walk.tNext.y, walk.tNext.z));
}

// The current cell's ceiling, cached in the walk. A cell outside the lattice
// clamps to the border — and so did every splat that reached past it, so the
// border bounds whatever the tree reads out there too.
void readCell(pnanovdb_buf_t buf, inout Walk walk) {
    let at = clamp(walk.cell, int3(0), walk.res - 1);
    pnanovdb_address_t addr = {
        walk.base + 4u * uint((at.z * walk.res.y + at.y) * walk.res.x + at.x)
    };
    walk.ceiling = pnanovdb_read_float(buf, addr);
}

// Open the walk at `t = 0` of the sub-segment.
Walk openWalk(SceneTable* scene, pnanovdb_buf_t buf, MediaSplit split,
              float3 origin, float3 direction, float start) {
    Walk walk;
    let leading = scene.media[split.grids[0]];
    walk.res = int3(leading.majorantRes & 0x3ff, (leading.majorantRes >> 10) & 0x3ff,
                    (leading.majorantRes >> 20) & 0x3ff);
    walk.base = leading.majorant;
    walk.sigmaT = leading.sigmaT;
    let at = toIndex(leading, origin + start * direction) * leading.majorantScale
             + leading.majorantBias;
    let perT = float3(dot(leading.worldToIndex[0].xyz, direction),
                      dot(leading.worldToIndex[1].xyz, direction),
                      dot(leading.worldToIndex[2].xyz, direction))
               * leading.majorantScale;
    [unroll]
    for (uint axis = 0; axis < 3; axis++) {
        // A ray sitting exactly on a boundary is in the cell it is heading
        // *into*: `floor` names that cell running forwards, `ceil(at) − 1`
        // running backwards. Taking `floor` for both would put the whole
        // backward walk one cell ahead of the ray — every boundary is
        // reached on schedule, but against the wrong cell's ceiling.
        if (perT[axis] > 0.0) {
            let ahead = floor(at[axis]);
            walk.advance[axis] = 1;
            walk.cell[axis] = int(ahead);
            walk.tNext[axis] = (ahead + 1.0 - at[axis]) / perT[axis];
            walk.tDelta[axis] = 1.0 / perT[axis];
        } else if (perT[axis] < 0.0) {
            let behind = ceil(at[axis] - 1.0);
            walk.advance[axis] = -1;
            walk.cell[axis] = int(behind);
            walk.tNext[axis] = (behind - at[axis]) / perT[axis];
            walk.tDelta[axis] = -1.0 / perT[axis];
        } else {
            walk.advance[axis] = 0;
            walk.cell[axis] = int(floor(at[axis]));
            walk.tNext[axis] = 1e30;
            walk.tDelta[axis] = 1e30;
        }
    }
    // Everything the cell lookup does not move: the gridless media summed,
    // plus any overlapping grid at its own whole-lattice maximum.
    walk.constant = split.homogeneous.sigmaT;
    [unroll]
    for (uint slot = 1; slot < MEDIUM_SLOTS; slot++) {
        if (slot < split.gridCount) {
            let record = scene.media[split.grids[slot]];
            walk.constant += record.densityMax * record.sigmaT;
        }
    }
    readCell(buf, walk);
    return walk;
}

// Cross into the next cell along whichever axis the ray reaches first — two
// at once where it passes exactly through an edge.
void stepWalk(pnanovdb_buf_t buf, inout Walk walk) {
    let least = cellExit(walk);
    [unroll]
    for (uint axis = 0; axis < 3; axis++) {
        if (walk.tNext[axis] <= least) {
            walk.cell[axis] += walk.advance[axis];
            walk.tNext[axis] += walk.tDelta[axis];
        }
    }
    readCell(buf, walk);
}

// Track a distance through `split`'s grids over `[start, start + tmax)` of
// the ray `origin + t·direction` (world units). `uFirst` is the bounce
// block's reserved Sobol pair — the first collision is the one draw whose
// stratification matters — and the rest ride the hash stream keyed by
// (crossed, collision).
public TrackEvent trackMedia(SceneTable* scene, pnanovdb_buf_t buf, MediaSplit split,
                             float3 origin, float3 direction, float start, float tmax,
                             uint pixel, uint sampleIndex, uint bounce, uint crossed,
                             float2 uFirst) {
    TrackEvent event;
    event.scattered = false;
    event.distance = tmax;
    event.weight = float3(1.0);
    event.g = split.homogeneous.g;
    event.albedo = float3(0.0);
#ifdef PROBES
    event.collisions = 0;
    event.nulls = 0;
    event.steps = 0;
#endif

    var walk = openWalk(scene, buf, split, origin, direction, start);
    var t = 0.0;
    var u = uFirst;
    var steps = 0u;
    for (uint collision = 0; collision < TRACKER_CAP; collision++) {
        // The optical depth this flight must accumulate, in majorant units.
        // Spending it across cells rather than redrawing per cell is what
        // keeps the sequence of uniforms — and so a 1³ lattice — exactly the
        // scalar-majorant tracker's.
        var tau = -log(1.0 - u.y);
        var m = 0.0;
        var transmitted = false;
        for (;;) {
            let majorant = walk.constant + walk.ceiling * walk.sigmaT;
            m = max(majorant.x, max(majorant.y, majorant.z));
            let end = min(cellExit(walk), tmax);
            // `m > 0` and not just the depth test: `toUnitFloat` can return
            // exactly 0, making tau exactly 0, and an empty cell (m = 0) is
            // ordinary in a grid — the test alone would then divide 0 by 0.
            // Skipping the cell is also what a zero depth means: the
            // collision belongs at the first point that has a majorant.
            if (m > 0.0 && m * (end - t) >= tau) {
                t += tau / m; // the flight ends inside this cell
                break;
            }
            // Nothing collides in this cell, and an empty one costs the walk
            // nothing but the lookup that found it empty.
            tau -= m * (end - t);
            t = end;
            if (t >= tmax || ++steps >= MAJORANT_STEP_CAP) {
                transmitted = true; // the null weights are the estimate
                break;
            }
            stepWalk(buf, walk);
        }
#ifdef PROBES
        // Every exit below is in this iteration, so one assignment carries
        // the count out of all of them.
        event.steps = steps;
#endif
        if (transmitted) {
            return event;
        }
        let x = origin + (start + t) * direction;
        var sigmaT = split.homogeneous.sigmaT;
        var sigmaS = split.homogeneous.sigmaS;
        // The homogeneous g is already σ_s-normalized; multiplying it back
        // out recovers Σ σ_s·g, which the grids then add their share to.
        var gSum = split.homogeneous.g * (sigmaS.x + sigmaS.y + sigmaS.z);
        [unroll]
        for (uint slot = 0; slot < MEDIUM_SLOTS; slot++) {
            if (slot >= split.gridCount) {
                continue;
            }
            let record = scene.media[split.grids[slot]];
            pnanovdb_grid_handle_t grid = { pnanovdb_address_t(record.grid) };
            pnanovdb_tree_handle_t tree = pnanovdb_grid_get_tree(buf, grid);
            pnanovdb_root_handle_t root = pnanovdb_tree_get_root(buf, tree);
            pnanovdb_readaccessor_t acc;
            pnanovdb_readaccessor_init(PNANOVDB_REF(acc), root);
            // Each grid clamps to the ceiling that entered the majorant
            // above: the walked cell's for the leading grid, the whole
            // lattice's maximum for any other.
            let density = sampleDensity(buf, acc, toIndex(record, x),
                                        slot == 0 ? walk.ceiling : record.densityMax);
            sigmaT += density * record.sigmaT;
            let scatter = density * record.sigmaS;
            sigmaS += scatter;
            gSum += record.g * (scatter.x + scatter.y + scatter.z);
        }
#ifdef PROBES
        event.collisions++;
#endif
        // The channel-average scatter probability; its complement is the
        // null's. Absorption is folded into the null weight — never a
        // termination — so an absorbing channel attenuates smoothly and a
        // zero-σ_s medium runs pure ratio tracking.
        let scatterSum = sigmaS.x + sigmaS.y + sigmaS.z;
        let ps = scatterSum / (3.0 * m);
        if (u.x < ps) {
            event.scattered = true;
            event.distance = t;
            event.weight *= sigmaS / (m * ps);
            event.g = gSum / scatterSum;
            event.albedo = sigmaS / max(sigmaT, float3(1e-9));
            return event;
        }
        // σ_t ≤ m holds per channel by construction (density clamps to the
        // ceiling the majorant was read from); max(…, 0) is for float
        // rounding at exact equality, where a −0 scale must stay zero rather
        // than flip sign. 1 − ps cannot be zero here: ps = 1 needs every
        // channel at the majorant with σ_s = σ_t, and u.x < 1 always
        // (`toUnitFloat`), so that collision took the scatter arm above.
#ifdef PROBES
        event.nulls++;
#endif
        event.weight *= max(m - sigmaT, float3(0.0)) / (m * (1.0 - ps));
        u = trackerUniform2(pixel, sampleIndex, bounce,
                            (crossed << 12) | ((collision + 1) & 0xfff));
    }
    return event;
}
