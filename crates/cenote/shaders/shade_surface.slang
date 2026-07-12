// The surface-shading stage: reconstruct the hit, settle the interior
// medium's Beer–Lambert absorption over the arriving segment, collect the
// surface's emission (MIS-weighted — a BSDF-sampled path landing on a
// light competes with next-event estimation for the same connection),
// send one shadow ray toward the lights (the light list or the
// environment, chosen by power), and scatter off the OpenPBR mixture —
// reflection or, through the dielectric base, refraction — until the path
// escapes to the sky, hits the depth cap, or loses Russian roulette.
//
// Radiance bookkeeping: the wave's per-pixel accumulator starts at zero
// and every write below is a plain +=. A pixel has one path and a path
// visits this kernel once per bounce, so each dispatch performs at most
// one read-modify-write per pixel — which keeps writes pixel-owned and
// renders bitwise deterministic. The one *terminal* add per path carries
// alpha 1 (here or in shade_miss): "every pixel finished exactly once"
// stays mechanically checkable.
//
// When the wave carries AOV targets, this kernel also feeds the denoiser
// guides: the first hit writes the depth AOV, and the albedo/normal
// guides record here — unless the sampled lobe is near-specular, in
// which case they ride the path's feature throughput to whatever the
// mirror or glass shows. Same pixel-owned adds, same determinism.

import lights;
import openpbr;
import pathstate;
import rng;
import scene;
import surface;
import textures;

struct Params {
    Paths paths;
    Queue<uint> hits;
    Queue<uint> rays;         // the next bounce's input: survivors push here
    Queue<ShadowRay> shadows; // next-event connections, for trace_shadow
    SceneTable* scene;
    float4* radiance;         // the film's per-pixel accumulator for this wave
    AovTable* aov;            // the wave's AOV targets; enabled = 0 skips them
    uint sampleIndex; // which sample of every pixel's sequence this wave is
    // bounce | maxBounces << 8 | lightSampling << 16. Packed because this
    // block sits exactly at Vulkan's guaranteed 128 push-constant bytes:
    // the AOV pointer's 8 bytes come out of these three small scalars.
    uint packed;
};

[vk::push_constant]
Params params;

// Bounces before this always survive — they carry most of the image's
// energy, and killing them early would just trade traced rays for noise.
// From here on, paths gamble survival against their throughput and the
// survivors are reweighted to keep the estimate unbiased.
static const uint ROULETTE_START = 3;

// Record the denoiser guides for this pixel: `weight` of the surface's
// albedo estimate and shading normal, added to the wave's zero-filled
// per-pixel accumulators. Pixel-owned like every radiance write — one
// path per pixel — so guide order never reaches the image.
void recordGuides(AovTable* aov, uint pixel, float3 weight, float3 albedo,
                  float3 normal) {
    aov.albedo[pixel] += float4(weight * albedo, 0.0);
    aov.normal[pixel] += float4(weight * normal, 0.0);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void shade_surface(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.hits.count()) {
        return;
    }
    let bounce = params.packed & 0xff;
    let maxBounces = (params.packed >> 8) & 0xff;
    let lightSampling = (params.packed >> 16) & 0xff;
    // Bit 24 marks ReSTIR mode, set on every bounce. At the primary hit
    // (bounce 0) the reservoir stages own next-event estimation, so this kernel
    // skips its own NEE there. At the primary continuation's first hit
    // (bounce 1) the reservoir's internalized BSDF candidate already owns
    // whatever emitter that ray lands on (D-088), so this kernel suppresses
    // that emission — the two halves of "the reservoir owns primary direct
    // lighting," which together keep ReSTIR-on and ReSTIR-off unbiased.
    let restirEnabled = ((params.packed >> 24) & 1) != 0;
    let restirPrimary = restirEnabled && bounce == 0;
    let restirSuppressEmission = restirEnabled && bounce == 1;
    uint path = params.hits.entries[tid.x];
    uint pixel = params.paths.pixel[path];
    Hit hit = params.paths.hit[path];
    let direction4 = params.paths.direction[path];
    let direction = direction4.xyz;
    let origin = params.paths.origin[path].xyz;
    GeometryRecord mesh = params.scene.geometry[hit.instance];
    // Textured parameters resolve at the hit's UV once; everything below
    // reads one plain material, whatever mix of maps and constants fed it.
    let uv = interpolateUv(mesh, hit.primitive, hit.barycentrics);
    let material = resolveMaterial(params.scene.materials[hit.instance], uv);
    let spawn = spawnPoint(mesh, hit, direction);
    // xyz = the path's weight; w = the previous scatter's pdf (0: camera ray).
    var throughput = params.paths.throughput[path];

    // The interior the segment crossed: Beer–Lambert absorption toward
    // this medium's target color, folded into the throughput before
    // anything at this vertex spends it. A zero transmission depth means
    // no interior — the tint sat on the interface instead. Camera rays
    // start in vacuum, so bounce 0 never reads the (unwritten) state.
    var medium = bounce == 0 ? MEDIUM_NONE
               : pathStateMedium(params.paths.state[path]);
    if (medium != MEDIUM_NONE) {
        Material interior = params.scene.materials[medium];
        if (interior.transmissionDepth > 0.0) {
            let sigma = -log(clamp(interior.transmissionColor, 1e-4, 1.0))
                      / interior.transmissionDepth;
            throughput.xyz *= exp(-sigma * distance(origin, spawn.position));
        }
    }

    // Emission. This hit is the BSDF strategy's way of reaching a light,
    // so it is weighted against the probability that next-event sampling
    // at the previous vertex would have picked the same point — except on
    // camera rays, where no other strategy competed and the weight is 1.
    // Emitters are one-sided (winding front), so a back-face hit adds
    // nothing — and NEE could not have sampled it either, keeping the two
    // strategies consistent for MIS.
    var contribution = float3(0.0);
    if (mesh.light != LIGHT_NONE && spawn.frontFace) {
        var weight = 1.0;
        if (restirSuppressEmission) {
            // The primary continuation's first-hit emission is the BSDF
            // strategy's reach to this light — already a reservoir candidate at
            // bounce 0, so counting it here would double it.
            weight = 0.0;
        } else if (throughput.w > 0.0) {
            if (lightSampling == LIGHT_SAMPLING_NEE) {
                weight = 0.0;
            } else if (lightSampling == LIGHT_SAMPLING_MIS) {
                let delta = spawn.position - origin;
                let cosLight = abs(dot(spawn.normal, direction));
                // The full next-event pdf: choose the light list over the
                // environment, then this triangle (records sit in
                // primitive order from the instance's first) and this
                // point on it.
                let neePdf = (1.0 - params.scene.env.selectProb)
                           * params.scene.lights[mesh.light + hit.primitive].pdf
                           * dot(delta, delta) / cosLight;
                weight = powerHeuristic(throughput.w, neePdf);
            }
        }
        contribution = throughput.xyz * material.emission * weight;
    }

    // The BSDF lives in the shading frame — the interpolated normal,
    // perturbed by the normal map where the material carries one —
    // resolved once per vertex against the view direction. A hit on the
    // far wall of the interior the path is inside sees the bare glass
    // interface at the inverted IOR. Directions that are valid in the
    // shading frame but pass through the real triangle (the wrong side of
    // the geometric horizon) are rejected at each use below — the
    // standard shading-normal trade: a sliver of energy lost near
    // silhouettes, never light through walls, and the same treatment for
    // every strategy so MIS stays consistent. Resolved before the depth
    // cap because a path ending there still records its open guides.
    var shading = spawn.shading;
    if (material.normalTexture != TEXTURE_NONE) {
        shading = mappedShadingNormal(mesh, hit, uv, material.normalTexture, spawn);
    }
    let frame = Frame(shading);
    let wo = frame.toLocal(-direction);
    let closure = resolveClosure(params.scene.bsdfTables, material, wo,
                                 medium == hit.instance);

    // The denoiser guides' remaining feature throughput: fully open on a
    // camera ray (which never reads the buffer), whatever the specular
    // pass-throughs so far left otherwise. A path that ends at this
    // vertex records it all here; one that continues splits it below by
    // the sampled lobe's ramp. The albedo estimate is clamped: a guide
    // wants a reflectance, and the estimate can drift just past 1.
    let aovOn = params.aov.enabled != 0;
    var guide = float3(0.0);
    if (aovOn) {
        guide = bounce == 0 ? float3(1.0) : params.aov.guide[pixel].xyz;
        if (bounce == 0) {
            // First hit: camera-plane z, via the cosine raygen stashed in
            // direction.w. First hits only — depth never passes through a
            // mirror, unlike the guides.
            params.aov.depth[pixel] += distance(origin, spawn.position)
                                     * direction4.w;
        }
    }
    let aovAlbedo = saturate(closure.aovAlbedo);

    // The depth cap: the path ends at this vertex, so this is its pixel's
    // terminal add. No next-event estimation here — an NEE connection from
    // this vertex would be one segment longer than any path BSDF sampling
    // can trace, and the two strategies must cover the same path lengths
    // to converge to the same image.
    if (bounce + 1 >= maxBounces) {
        if (aovOn && any(guide > 0.0)) {
            recordGuides(params.aov, pixel, guide, aovAlbedo, shading);
        }
        params.radiance[pixel] += float4(contribution, 1.0);
        return;
    }

    // Next-event estimation: connect this vertex to one candidate — a
    // direction toward the environment or a light from the list, chosen
    // in proportion to their power (the choice rides the low end of the
    // NEE_LIGHT draw; the list branch remaps the leftover back to a full-
    // range uniform for the alias walk). A candidate below the horizon is
    // legal when the base transmits — the connection then leaves through
    // the surface. The contribution is fully weighted here; trace_shadow
    // just adds it if the connection is unoccluded.
    let envProb = params.scene.env.selectProb;
    if (!restirPrimary && lightSampling != LIGHT_SAMPLING_BSDF
        && (params.scene.lightCount > 0 || envProb > 0.0)) {
        let uLight = sample_1d(pixel, params.sampleIndex,
                               bounceDimension(bounce, NEE_LIGHT));
        let uPoint = sample_2d(pixel, params.sampleIndex,
                               bounceDimension(bounce, NEE_POINT));
        ShadowRay shadow;
        shadow.pixel = pixel;
        shadow.pad0 = 0;
        shadow.pad1 = 0;
        shadow.pad2 = 0;
        var unshadowed = float3(0.0);
        if (uLight < envProb) {
            // Environment: a direction, unoccluded iff the shadow ray hits
            // nothing (LIGHT_NONE is no instance's ID). A pole sample
            // reports pdf 0 and is skipped — it carries no usable density.
            let candidate = params.scene.env.sample(uPoint);
            let wiLight = frame.toLocal(candidate.direction);
            let throughSurface = dot(candidate.direction, spawn.normal) < 0.0;
            if (candidate.pdf > 0.0 && wiLight.z != 0.0
                && (wiLight.z < 0.0) == throughSurface) {
                let neePdf = envProb * candidate.pdf;
                let scatterPdf = bsdfPdf(closure, wiLight, wo);
                let weight = lightSampling == LIGHT_SAMPLING_MIS
                           ? powerHeuristic(neePdf, scatterPdf)
                           : 1.0;
                unshadowed = throughput.xyz * candidate.radiance
                    * evalBsdf(closure, wiLight, wo)
                    * abs(wiLight.z) * (weight / neePdf);
                shadow.origin = throughSurface ? spawn.below : spawn.position;
                shadow.light = LIGHT_NONE;
                shadow.primitive = 0;
                shadow.direction = candidate.direction;
                shadow.tmax = 1e30;
            }
        } else {
            // A light from the list. Scene prep pins envProb to exactly 1
            // when the list is powerless, so this branch always has
            // something to walk.
            let uList = (uLight - envProb) / (1.0 - envProb);
            var candidate = sampleLights(params.scene.lights, params.scene.lightCount,
                                         uList, uPoint, spawn.position);
            // A textured emitter's record carries the map's scale; the
            // map itself evaluates at the sampled point, so this strategy
            // sees exactly the radiance a BSDF path hitting that point
            // would (the estimators must agree pointwise, not on average).
            if (!candidate.delta) {
                let emitter = params.scene.materials[candidate.instance];
                if (emitter.emissionTexture != TEXTURE_NONE) {
                    candidate.radiance *= sampleColor(
                        emitter.emissionTexture,
                        interpolateUv(params.scene.geometry[candidate.instance],
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
                unshadowed = throughput.xyz * candidate.radiance
                    * evalBsdf(closure, wiLight, wo)
                    * abs(wiLight.z) * (weight / neePdf);
                shadow.origin = throughSurface ? spawn.below : spawn.position;
                shadow.light = candidate.instance;
                shadow.primitive = candidate.primitive;
                shadow.direction = candidate.direction;
                shadow.tmax = candidate.tmax;
            }
        }
        if (any(unshadowed > 0.0)) {
            shadow.contribution = unshadowed;
            params.shadows.push(shadow);
        }
    }

    // Scatter off the mixture: one technique drawn by BSDF_LOBE (the
    // pick's leftover, rescaled, drives the glass technique's
    // reflect-or-refract branch), the estimator divided by the full
    // mixture pdf. The sidedness guard requires the scattered ray to
    // leave through the same geometric face the shading frame put it on.
    let uLobe = sample_1d(pixel, params.sampleIndex,
                          bounceDimension(bounce, BSDF_LOBE));
    let u = sample_2d(pixel, params.sampleIndex,
                      bounceDimension(bounce, BSDF_DIR));
    let scatter = sampleBsdf(closure, wo, uLobe, u);
    let wiWorld = frame.toWorld(scatter.wi);
    var scattered = float3(0.0);
    if (scatter.pdf > 0.0 && scatter.wi.z != 0.0
        && (scatter.wi.z > 0.0) == (dot(wiWorld, spawn.normal) > 0.0)) {
        scattered = throughput.xyz * evalBsdf(closure, scatter.wi, wo)
                  * (abs(scatter.wi.z) / scatter.pdf);
    }

    // A path that can carry nothing forward is done — lights themselves
    // scatter nothing (their base color is black), and continuing would
    // just trace dead rays until roulette noticed.
    var survival = max(scattered.r, max(scattered.g, scattered.b));
    if (survival <= 0.0) {
        if (aovOn && any(guide > 0.0)) {
            recordGuides(params.aov, pixel, guide, aovAlbedo, shading);
        }
        params.radiance[pixel] += float4(contribution, 1.0);
        return;
    }

    if (bounce >= ROULETTE_START) {
        survival = min(1.0, survival);
        if (sample_1d(pixel, params.sampleIndex,
                      bounceDimension(bounce, RR)) >= survival) {
            if (aovOn && any(guide > 0.0)) {
                recordGuides(params.aov, pixel, guide, aovAlbedo, shading);
            }
            params.radiance[pixel] += float4(contribution, 1.0);
            return;
        }
        scattered /= survival;
    }

    // The path continues: the sampled lobe decides how much of the open
    // guides record this surface and how much rides along the scattered
    // ray, tinted by the lobe it passed through. Never reweighted by
    // roulette survival — a guide wants low noise, not unbiasedness.
    if (aovOn) {
        let record = aovRecordFraction(closure, scatter.lobe);
        if (record > 0.0 && any(guide > 0.0)) {
            recordGuides(params.aov, pixel, guide * record, aovAlbedo, shading);
        }
        params.aov.guide[pixel] =
            float4(guide * (1.0 - record) * aovPassTint(closure, scatter.lobe), 0.0);
    }

    // A transmission through a closed surface crosses into (or out of) its
    // interior; thin walls have none. Reflections keep whatever medium the
    // path was already in.
    if (scatter.lobe == LOBE_TRANSMISSION && material.thinWalled == 0) {
        medium = medium == hit.instance ? MEDIUM_NONE : hit.instance;
    }
    let transmitted = scatter.wi.z < 0.0;
    params.paths.origin[path] = float4(transmitted ? spawn.below : spawn.position, 0.0);
    params.paths.direction[path] = float4(wiWorld, 0.0);
    params.paths.throughput[path] = float4(scattered, scatter.pdf);
    params.paths.state[path] = packPathState(scatter.lobe, medium);
    params.rays.push(path);

    // The path continues, but its pixel keeps the emission found here.
    if (any(contribution > 0.0)) {
        params.radiance[pixel] += float4(contribution, 0.0);
    }
}
