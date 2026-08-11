// Homogeneous-medium transport: what fills a path's current segment, how far
// it travels before it scatters, and which way it leaves. The last two are
// closed-form — one extinction holds everywhere in the medium, so there is
// nothing to track along the segment and no majorant to bound.

import openpbr; // Frame, PI
import pathstate; // STACK_EMPTY, MEDIUM_SLOTS, interiorOf
import scene;

// The medium filling the segment a path is travelling: the volumes it is
// inside, added coefficient by coefficient to whatever fills the open space
// around them. Addition is what makes overlap order-free, and order-free is
// what lets the shadow pass measure the same extinction from unordered
// traversal candidates.
//
// A refractive interior displaces all of it — a path inside glass is not
// also in the room's haze. A *scattering* interior (the `STACK_SCATTERING`
// bit, prep's single authority) is then the whole segment's medium: its
// record, alone. An absorbing one returns vacuum here — its Beer–Lambert
// absorption is the surface stage's closed form, applied once over the
// whole segment, so it stays bit-identical to a renderer with no volume
// stage at all. Both resolve by the
// register-held bit, never by re-reading σ_s: a second spelling of the
// predicate could drift from the one intersect routed by, and a segment
// would be weighted twice or not at all.
public MediumRecord combineMedia(SceneTable* scene, uint4 set) {
    MediumRecord total;
    total.sigmaT = float3(0.0);
    total.sigmaS = float3(0.0);
    total.g = 0.0;
    let interior = interiorOf(set);
    if (interior != STACK_EMPTY) {
        if ((interior & STACK_SCATTERING) != 0) {
            return scene.media[scene.geometry[interior & STACK_INDEX_MASK].medium];
        }
        return total;
    }
    // A mixture's true phase function is the σ_s-weighted average of its
    // components', which one Henyey–Greenstein lobe cannot be; its
    // anisotropy is that average instead — `total.g` carries the weighted
    // sum until the division below. Exact wherever a single component
    // scatters, which is every case but two scattering volumes overlapping.
    var weight = 0.0;
    for (uint slot = 0; slot <= MEDIUM_SLOTS; slot++) {
        // Slots first, then the open space they stand in, which adds the
        // same way they do.
        let index = slot < MEDIUM_SLOTS
                        ? (set[slot] == STACK_EMPTY
                               ? MEDIUM_NONE
                               : scene.geometry[set[slot] & STACK_INDEX_MASK].medium)
                        : scene.globalMedium;
        if (index == MEDIUM_NONE) {
            continue;
        }
        let record = scene.media[index];
        total.sigmaT += record.sigmaT;
        total.sigmaS += record.sigmaS;
        let scattering = record.sigmaS.x + record.sigmaS.y + record.sigmaS.z;
        total.g += scattering * record.g;
        weight += scattering;
    }
    total.g = weight > 0.0 ? total.g / weight : 0.0;
    return total;
}

// -- The subsurface interior, resolved on the GPU -------------------------
// The mirror of `scene::subsurface` and `scene::single_scatter_albedo`, for
// the one case prep cannot intern a record for: an interior a map decides
// per texel. The host is the authority and runs everywhere else; the two
// are pinned against each other by a readback test in scene/mod.rs, so a
// change here without one there fails.
static const float FIT_A = 4.09712;
static const float FIT_B = 4.20863;
static const float FIT_D = 9.59217;
static const float FIT_E = 41.6808;
static const float FIT_F = 17.7126;
// `MAX_ANISOTROPY` in scene/mod.rs.
static const float MAX_ANISOTROPY = 0.99;

// The single-scatter albedo whose interior walks to the authored
// multiple-scatter one, van de Hulst's inversion. Black snaps exact: the
// fit leaves a residual there, and a channel authored black must absorb
// purely rather than almost.
float singleScatterAlbedo(float color, float g) {
    if (isnan(color) || color <= 0.0) {
        return 0.0;
    }
    let c = min(color, 1.0);
    let s = FIT_A + FIT_B * c - sqrt(FIT_D + FIT_E * c + FIT_F * c * c);
    return (1.0 - s * s) / (1.0 - g * s * s);
}

// What fills the interior behind a resolved subsurface material. The lobe
// is already known live — the closure's `subsurfaceBlend` gate and the
// host's intern gate are one predicate — so this only clamps, where
// `scene::subsurface` clamps: the mean free path floored at a micron so an
// authored zero stays a finite extinction, and `g` bounded and NaN-swept
// before the fit divides by it.
public MediumRecord subsurfaceMedium(Material m) {
    let g = isnan(m.subsurfaceScatterAnisotropy)
                ? 0.0
                : clamp(m.subsurfaceScatterAnisotropy, -MAX_ANISOTROPY, MAX_ANISOTROPY);
    let radius = max(m.subsurfaceRadius, 0.0);
    MediumRecord record;
    record.sigmaT = 1.0 / max(radius * max(m.subsurfaceRadiusScale, 0.0), 1e-6);
    record.g = g;
    record.sigmaS = float3(singleScatterAlbedo(m.subsurfaceColor.r, g),
                           singleScatterAlbedo(m.subsurfaceColor.g, g),
                           singleScatterAlbedo(m.subsurfaceColor.b, g))
                    * record.sigmaT;
    return record;
}

// Where a path's segment ended in a medium, and what carrying it there
// costs. `weight` already divides by the density the distance was drawn
// with, and on a scatter it carries σ_s too: the caller multiplies it into
// the throughput and asks nothing further.
public struct MediumEvent {
    public bool scattered; // did the path scatter before reaching `tmax`?
    public float distance; // where it scattered, or `tmax`
    public float3 weight;  // transmittance / pdf, times σ_s on a scatter
};

// Draw a distance against `medium`'s extinction over a segment of length
// `tmax`. Chromatic: one channel picked uniformly decides *where*, and the
// density is the balance heuristic over all three — pbrt Eq. 15.11 and
// 15.12. Sampling one luminance-averaged extinction would be unbiased too,
// and visibly noisier in exactly the saturated media a per-channel σ exists
// to describe.
public MediumEvent sampleMedium(MediumRecord medium, float tmax, float2 u) {
    if (!any(medium.sigmaS > 0.0)) {
        // Nothing to scatter, so there is no event to draw and the segment
        // is Beer–Lambert alone: taking it in closed form is exact and
        // noiseless where drawing a distance would kill the path outright
        // whenever it landed short. Vacuum takes this path too, at σ_t = 0.
        MediumEvent event;
        event.scattered = false;
        event.distance = tmax;
        event.weight = exp(-medium.sigmaT * tmax);
        return event;
    }
    let channel = min(uint(u.x * 3.0), 2u);
    let sigma = medium.sigmaT[channel];
    // A channel that does not extinguish never scatters: its draws run past
    // every surface and come back as pure transmittance.
    let t = sigma > 0.0 ? -log(1.0 - u.y) / sigma : 1e30;

    MediumEvent event;
    event.scattered = t < tmax;
    event.distance = min(t, tmax);
    let transmittance = exp(-medium.sigmaT * event.distance);
    let densities = event.scattered ? medium.sigmaT * transmittance : transmittance;
    let pdf = (densities.x + densities.y + densities.z) / 3.0;
    let numerator = event.scattered ? transmittance * medium.sigmaS : transmittance;
    // A pdf of zero is a segment of infinite optical depth in every channel —
    // the far end is unreachable, and nothing arrives from it.
    event.weight = pdf > 0.0 ? numerator / pdf : float3(0.0);
    return event;
}

// The Henyey–Greenstein phase function at `cosTheta` — the cosine between
// the two directions *away* from the scattering point, so g > 0 leans
// forward (a path continues in the direction it arrived). No isotropic fast
// path: HG reduces to 1/4π exactly at g = 0, and a branch would cost
// divergence. The denominator bottoms out at (1 − |g|)², which the host's
// anisotropy clamp keeps at 1e-4; the guard covers a σ table that reached
// the device by some other route.
public float hgPhase(float cosTheta, float g) {
    let g2 = g * g;
    let denominator = 1.0 + g2 + 2.0 * g * cosTheta;
    return (1.0 - g2) / (4.0 * PI * denominator * sqrt(max(denominator, 1e-9)));
}

// A phase-sampled direction. HG samples its own density exactly, so the
// estimator's f/pdf is 1 and the pdf here exists only for the MIS weight
// against next-event estimation.
public struct PhaseSample {
    public float3 wi;
    public float pdf;
};

// Sample a direction from HG about `wo`, which points away from the
// scattering point back along the arriving ray — the convention `hgPhase`
// reads its cosine in. The near-isotropic branch is not an optimization: the
// closed form divides by g, and near zero its numerator cancels away.
public PhaseSample samplePhase(float g, float3 wo, float2 u) {
    float cosTheta;
    if (abs(g) < 1e-3) {
        cosTheta = 1.0 - 2.0 * u.x;
    } else {
        let s = (1.0 - g * g) / (1.0 + g - 2.0 * g * u.x);
        cosTheta = -(1.0 + g * g - s * s) / (2.0 * g);
    }
    let sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    let phi = 2.0 * PI * u.y;
    PhaseSample sample;
    sample.wi = Frame(wo).toWorld(
        float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta));
    sample.pdf = hgPhase(cosTheta, g);
    return sample;
}
