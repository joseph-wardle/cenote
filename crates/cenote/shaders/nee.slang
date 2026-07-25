// The two halves of the per-vertex NEE↔BSDF MIS pair, shared by the path
// tracer's shading loop (`shade_surface`) and the `ReSTIR` candidate stage's
// inline indirect tail (`restir_candidates`, D-134). Both loops run the same
// per-vertex estimator — one next-event connection plus a BSDF continuation
// under the power heuristic — and the unbiasedness gate (`ReSTIR` ≡ brute
// force) holds only while the two evaluate identical densities and weights.
// One copy here, so they can never drift into a silent bias — the same
// rationale `restir_target.slang` gives for the shared surface reconstruction.

import environment;
import lights;
import openpbr;
import scene;
import surface;
import textures;

// One next-event candidate at a path vertex: the fully weighted unshadowed
// contribution and the shadow-ray geometry that gates it. `unshadowed` is zero
// when the draw degenerated (pdf 0, a sideways direction, a horizon the base
// doesn't transmit through) — the caller adds `unshadowed · visibility` after
// its own shadow test, queued (`shade_surface`) or inline (the tail).
public struct NeeCandidate {
    public float3 unshadowed; // throughput · L·f·|cosθ| · weight/pdf, or zero
    public float3 origin;     // the spawn point, below-surface for transmission
    public float3 direction;
    public float tmax;
    public uint light;        // shadow identity target (TLAS index), or LIGHT_NONE
    public uint primitive;
};

// Draw one next-event candidate: a direction toward the environment or a light
// from the list, chosen in proportion to their power (the choice rides the low
// end of `uLight`; the list branch remaps the leftover back to a full-range
// uniform for the alias walk). A candidate below the horizon is legal when the
// base transmits — the connection then leaves through the surface. The
// contribution is fully weighted here (MIS against BSDF sampling under
// `LIGHT_SAMPLING_MIS`; delta lights always weight 1). Callers guard that a
// light exists (`lightCount > 0 || env.selectProb > 0`) before drawing.
public NeeCandidate sampleNee(SceneTable* scene, SpawnPoint spawn, Frame frame,
                              Closure closure, float3 wo, float3 throughput,
                              uint lightSampling, float uLight, float2 uPoint) {
    NeeCandidate nee;
    nee.unshadowed = float3(0.0);
    nee.origin = spawn.position;
    nee.direction = float3(0.0);
    nee.tmax = 0.0;
    nee.light = LIGHT_NONE;
    nee.primitive = 0;

    let envProb = scene.env.selectProb;
    if (uLight < envProb) {
        // Environment: a direction, unoccluded iff the shadow ray hits
        // nothing (LIGHT_NONE is no instance's ID). A pole sample
        // reports pdf 0 and is skipped — it carries no usable density.
        let candidate = scene.env.sample(uPoint);
        let wiLight = frame.toLocal(candidate.direction);
        let throughSurface = dot(candidate.direction, spawn.normal) < 0.0;
        if (candidate.pdf > 0.0 && wiLight.z != 0.0
            && (wiLight.z < 0.0) == throughSurface) {
            let neePdf = envProb * candidate.pdf;
            let scatterPdf = bsdfPdf(closure, wiLight, wo);
            let weight = lightSampling == LIGHT_SAMPLING_MIS
                       ? powerHeuristic(neePdf, scatterPdf)
                       : 1.0;
            nee.unshadowed = throughput * candidate.radiance
                * evalBsdf(closure, wiLight, wo)
                * abs(wiLight.z) * (weight / neePdf);
            nee.origin = throughSurface ? spawn.below : spawn.position;
            nee.direction = candidate.direction;
            nee.tmax = 1e30;
        }
    } else {
        // A light from the list. Scene prep pins envProb to exactly 1
        // when the list is powerless, so this branch always has
        // something to walk.
        let uList = (uLight - envProb) / (1.0 - envProb);
        var candidate = sampleLights(scene.lights, scene.lightCount,
                                     uList, uPoint, spawn.position);
        // A textured emitter's record carries the map's scale; the
        // map itself evaluates at the sampled point, so this strategy
        // sees exactly the radiance a BSDF path hitting that point
        // would (the estimators must agree pointwise, not on average).
        if (!candidate.delta) {
            let emitter = scene.materials[candidate.instance];
            if (emitter.emissionTexture != TEXTURE_NONE) {
                candidate.radiance *= sampleColor(
                    emitter.emissionTexture,
                    interpolateUv(scene.geometry[candidate.instance],
                                  candidate.primitive, candidate.bary));
            }
        }
        let wiLight = frame.toLocal(candidate.direction);
        let throughSurface = dot(candidate.direction, spawn.normal) < 0.0;
        if (candidate.pdf > 0.0 && wiLight.z != 0.0
            && (wiLight.z < 0.0) == throughSurface) {
            // The candidate's pdf at this vertex against the pdf BSDF
            // sampling would give the direction — except at a delta
            // light, which BSDF sampling reaches with probability
            // zero: its weight is 1 under every mode.
            let neePdf = (1.0 - envProb) * candidate.pdf;
            let weight = !candidate.delta
                      && lightSampling == LIGHT_SAMPLING_MIS
                       ? powerHeuristic(neePdf, bsdfPdf(closure, wiLight, wo))
                       : 1.0;
            nee.unshadowed = throughput * candidate.radiance
                * evalBsdf(closure, wiLight, wo)
                * abs(wiLight.z) * (weight / neePdf);
            nee.origin = throughSurface ? spawn.below : spawn.position;
            nee.direction = candidate.direction;
            nee.tmax = candidate.tmax;
            nee.light = candidate.instance;
            nee.primitive = candidate.primitive;
        }
    }
    return nee;
}

// The full next-event pdf of the emitter point a BSDF-sampled path landed on —
// the other half of the pair: choose the light list over the environment, then
// this triangle (records sit in primitive order from the instance's first) and
// this point on it, lifted to solid angle at the previous vertex. This is the
// density the emission's power-heuristic weight competes against, and it must
// match `sampleLights`' composition exactly or MIS silently biases.
public float emitterHitNeePdf(SceneTable* scene, GeometryRecord mesh, uint primitive,
                              SpawnPoint spawn, float3 origin, float3 direction) {
    let delta = spawn.position - origin;
    let cosLight = abs(dot(spawn.normal, direction));
    return (1.0 - scene.env.selectProb)
         * scene.lights[mesh.light + primitive].pdf
         * dot(delta, delta) / cosLight;
}
