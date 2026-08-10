// The two halves of the per-vertex NEE↔scatter MIS pair used by the path
// tracer's shading loop: one next-event connection plus a scattered
// continuation, combined under the power heuristic. One copy here, so the
// density a connection is drawn with and the weight it is combined with can
// never drift apart into a silent bias.
//
// A surface vertex draws through its BSDF and a medium vertex through its
// phase function, and the two spell the category choice out separately:
// routing the surface through `drawLight` joins two specialized branches
// into one and measured 4–7% slower on the shading kernel. The duplication
// is held level by test, not by care — the MIS-agreement tests in
// wavefront.rs render a medium vertex against an environment, two emitters
// of different power, and a textured emitter, so a drift in either
// spelling's category split, alias remap, pdf scale, or emitter radiance
// shows up as a disagreement rather than a silent bias.

import lights;
import medium;
import openpbr;
import pathstate;
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

// One light-sampling draw at a point, before anything about the vertex
// weighs in: where the light is, what it emits, and — the part that must not
// drift — the full next-event density, category choice included.
public struct LightDraw {
    public float3 radiance;
    public float3 direction;
    public float pdf;       // solid-angle density at the drawing point, 0 if degenerate
    public float tmax;      // shadow-ray extent: bounded for lights with a position
    public uint light;      // shadow identity target (TLAS index), or LIGHT_NONE
    public uint primitive;
    public bool delta;      // no scatter strategy competes: MIS weight is 1
};

// Draw one light at `position`: the environment or a light from the list,
// chosen in proportion to their power (the choice rides the low end of
// `uLight`; the list branch remaps the leftover back to a full-range uniform
// for the alias walk). Callers guard that a light exists
// (`lightCount > 0 || env.selectProb > 0`) before drawing.
//
// The reference spelling of the composition — `sampleNee`'s inline copy is
// the one that must be kept level with this.
public LightDraw drawLight(SceneTable* scene, float3 position,
                           float uLight, float2 uPoint) {
    LightDraw draw;
    let envProb = scene.env.selectProb;
    if (uLight < envProb) {
        // Environment: a direction, unoccluded iff the shadow ray hits
        // nothing (LIGHT_NONE is no instance's ID). A pole sample
        // reports pdf 0 and is skipped — it carries no usable density.
        let candidate = scene.env.sample(uPoint);
        draw.radiance = candidate.radiance;
        draw.direction = candidate.direction;
        draw.pdf = envProb * candidate.pdf;
        draw.tmax = 1e30;
        draw.light = LIGHT_NONE;
        draw.primitive = 0;
        draw.delta = false;
    } else {
        // A light from the list. Scene prep pins envProb to exactly 1
        // when the list is powerless, so this branch always has
        // something to walk.
        let uList = (uLight - envProb) / (1.0 - envProb);
        var candidate = sampleLights(scene.lights, scene.lightCount,
                                     uList, uPoint, position);
        // A textured emitter's record carries the map's scale; the
        // map itself evaluates at the sampled point, so this strategy
        // sees exactly the radiance a BSDF path hitting that point
        // would (the estimators must agree pointwise, not on average).
        if (!candidate.delta) {
            let emitter = scene.materials[candidate.instance];
            if (emitter.emissionTexture != TEXTURE_NONE) {
                candidate.radiance *= sampleColor(
                    scene.textureParams, emitter.emissionTexture,
                    interpolateUv(scene.geometry[candidate.instance],
                                  candidate.primitive, candidate.bary));
            }
        }
        draw.radiance = candidate.radiance;
        draw.direction = candidate.direction;
        draw.pdf = (1.0 - envProb) * candidate.pdf;
        draw.tmax = candidate.tmax;
        draw.light = candidate.instance;
        draw.primitive = candidate.primitive;
        draw.delta = candidate.delta;
    }
    return draw;
}

// Draw one next-event candidate at a surface. A candidate below the horizon
// is legal when the base transmits — the connection then leaves through the
// surface. The contribution is fully weighted here: the draw's density
// against the density BSDF sampling would give the same direction, under
// `LIGHT_SAMPLING_MIS`. A delta light weights 1 under every mode, since BSDF
// sampling reaches it with probability zero. The category choice and its
// density are `drawLight`'s, spelled out again — see the module header.
//
// TODO: an emitter with opacity 0 belongs in that same exception. Its any-hit
// rejects every ray, so BSDF sampling can never reach it either, yet the
// heuristic still discounts this sample against a strategy that cannot fire —
// a fully invisible emitter lands at roughly half strength. Kroken's alpha-0
// sun is the live case (scenes/corpus/curate-kroken.py). The fix needs the
// emitter's opacity here, which the light candidate does not carry today.
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
        let uList = (uLight - envProb) / (1.0 - envProb);
        var candidate = sampleLights(scene.lights, scene.lightCount,
                                     uList, uPoint, spawn.position);
        if (!candidate.delta) {
            let emitter = scene.materials[candidate.instance];
            if (emitter.emissionTexture != TEXTURE_NONE) {
                candidate.radiance *= sampleColor(
                    scene.textureParams, emitter.emissionTexture,
                    interpolateUv(scene.geometry[candidate.instance],
                                  candidate.primitive, candidate.bary));
            }
        }
        let wiLight = frame.toLocal(candidate.direction);
        let throughSurface = dot(candidate.direction, spawn.normal) < 0.0;
        if (candidate.pdf > 0.0 && wiLight.z != 0.0
            && (wiLight.z < 0.0) == throughSurface) {
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

// The same connection from a subsurface walk's exit point: the Lambertian
// transmission lobe (f = 1/π about `outward`, the negated shading normal)
// stands where the BSDF stood, weighted against the walk's cosine-sampled
// continuation. The connection must leave through the real triangle's
// outside — the same shading-normal trade every scatter makes.
public NeeCandidate sampleNeeExit(SceneTable* scene, SpawnPoint spawn,
                                  float3 outward, float3 throughput,
                                  uint lightSampling, float uLight, float2 uPoint) {
    NeeCandidate nee;
    nee.unshadowed = float3(0.0);
    nee.origin = spawn.below;
    nee.direction = float3(0.0);
    nee.tmax = 0.0;
    nee.light = LIGHT_NONE;
    nee.primitive = 0;

    let draw = drawLight(scene, spawn.below, uLight, uPoint);
    let cosine = dot(draw.direction, outward);
    if (draw.pdf > 0.0 && cosine > 0.0
        && dot(draw.direction, spawn.normal) < 0.0) {
        let weight = !draw.delta && lightSampling == LIGHT_SAMPLING_MIS
                   ? powerHeuristic(draw.pdf, cosine / PI)
                   : 1.0;
        nee.unshadowed = throughput * draw.radiance
                       * (cosine / PI) * (weight / draw.pdf);
        nee.direction = draw.direction;
        nee.tmax = draw.tmax;
        nee.light = draw.light;
        nee.primitive = draw.primitive;
    }
    return nee;
}

// A candidate as the shadow queue carries it — trace_shadow adds
// `contribution` if the connection is unoccluded. The pads are wire
// layout only.
public ShadowRay shadowRay(NeeCandidate nee, uint pixel, uint path) {
    ShadowRay shadow;
    shadow.origin = nee.origin;
    shadow.light = nee.light;
    shadow.direction = nee.direction;
    shadow.primitive = nee.primitive;
    shadow.contribution = nee.unshadowed;
    shadow.tmax = nee.tmax;
    shadow.pixel = pixel;
    shadow.path = path;
    shadow.pad0 = 0;
    shadow.pad1 = 0;
    return shadow;
}

// The same connection from a scattering point inside a medium: no surface,
// so no horizon to test and no offset to spawn from, and the phase function
// stands where the BSDF and its cosine stood. `wo` points back along the
// arriving ray, the convention `hgPhase` reads its cosine in.
public NeeCandidate sampleNeeVolume(SceneTable* scene, float3 position, float g,
                                    float3 wo, float3 throughput,
                                    uint lightSampling, float uLight, float2 uPoint) {
    NeeCandidate nee;
    nee.unshadowed = float3(0.0);
    nee.origin = position;
    nee.direction = float3(0.0);
    nee.tmax = 0.0;
    nee.light = LIGHT_NONE;
    nee.primitive = 0;

    let draw = drawLight(scene, position, uLight, uPoint);
    if (draw.pdf > 0.0) {
        let phase = hgPhase(dot(wo, draw.direction), g);
        let weight = !draw.delta && lightSampling == LIGHT_SAMPLING_MIS
                   ? powerHeuristic(draw.pdf, phase)
                   : 1.0;
        nee.unshadowed = throughput * draw.radiance * phase * (weight / draw.pdf);
        nee.direction = draw.direction;
        nee.tmax = draw.tmax;
        nee.light = draw.light;
        nee.primitive = draw.primitive;
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
