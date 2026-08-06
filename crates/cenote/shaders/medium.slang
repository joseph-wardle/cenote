// Homogeneous-medium transport: how far a path travels before it scatters,
// and which way it leaves. Both halves are closed-form — one extinction
// holds everywhere in the medium, so there is nothing to track along the
// segment and no majorant to bound.

import openpbr; // Frame, PI
import scene;

// Where a path's segment ended in a medium, and what carrying it there
// costs. `weight` already divides by the density the distance was drawn
// with, and on a scatter it carries the scattering coefficient too: the
// caller multiplies it into the throughput and asks nothing further.
public struct MediumEvent {
    public bool scattered; // did the path scatter before reaching `tmax`?
    public float distance; // where it scattered, or `tmax`
    public float3 weight;  // transmittance / pdf, times σ_s on a scatter
};

// Draw a distance against `medium`'s extinction over a segment of length
// `tmax`. Chromatic: one channel picked uniformly decides *where*, and the
// density is the balance heuristic over all three — pbrt Eq. 15.11 and
// 15.12. Sampling from a luminance-averaged extinction would be unbiased
// too, and visibly noisier in exactly the saturated media a per-channel σ
// exists to describe.
public MediumEvent sampleMedium(MediumRecord medium, float tmax, float2 u) {
    let channel = min(uint(u.x * 3.0), 2u);
    let sigma = medium.sigmaT[channel];
    // A channel that does not extinguish never scatters: its draws run
    // past every surface and come back as pure transmittance.
    let t = sigma > 0.0 ? -log(1.0 - u.y) / sigma : 1e30;

    MediumEvent event;
    event.scattered = t < tmax;
    event.distance = min(t, tmax);
    let transmittance = exp(-medium.sigmaT * event.distance);
    // What each of the three draws would have had to do to land here: a
    // scatter competes against every channel's density at this distance,
    // reaching the far end against every channel's transmittance.
    let densities = event.scattered ? medium.sigmaT * transmittance : transmittance;
    let pdf = (densities.x + densities.y + densities.z) / 3.0;
    let numerator = event.scattered ? transmittance * medium.sigmaS : transmittance;
    // A pdf of zero is a segment of infinite optical depth in every
    // channel — the far end is unreachable, and nothing arrives from it.
    event.weight = pdf > 0.0 ? numerator / pdf : float3(0.0);
    return event;
}

// The Henyey–Greenstein phase function at `cosTheta` — the cosine between
// the two directions *away* from the scattering point, so g > 0 leans
// forward (a path continues in the direction it arrived). One lobe, no
// isotropic fast path: HG reduces to 1/4π exactly at g = 0, and a branch
// would buy flops and cost divergence.
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
// closed form divides by g.
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
