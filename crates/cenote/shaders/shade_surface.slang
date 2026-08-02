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
import nee;
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
    let material = resolveMaterial(params.scene.textureParams, params.scene.materials[hit.instance], uv);
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
        throughput.xyz *= interiorTransmittance(params.scene, medium,
                                                origin, spawn.position);
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
        if (throughput.w > 0.0) {
            if (lightSampling == LIGHT_SAMPLING_NEE) {
                weight = 0.0;
            } else if (lightSampling == LIGHT_SAMPLING_MIS) {
                let neePdf = emitterHitNeePdf(params.scene, mesh, hit.primitive,
                                              spawn, origin, direction);
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
        shading = mappedShadingNormal(params.scene.textureParams, mesh, hit, uv, material.normalTexture, spawn);
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

    // Next-event estimation: one candidate connection, drawn and weighted by
    // the shared `sampleNee` (nee.slang). The contribution is fully weighted
    // there; trace_shadow just adds it if the connection is unoccluded.
    if (lightSampling != LIGHT_SAMPLING_BSDF
        && (params.scene.lightCount > 0 || params.scene.env.selectProb > 0.0)) {
        let uLight = sample_1d(pixel, params.sampleIndex,
                               bounceDimension(bounce, NEE_LIGHT));
        let uPoint = sample_2d(pixel, params.sampleIndex,
                               bounceDimension(bounce, NEE_POINT));
        let nee = sampleNee(params.scene, spawn, frame, closure, wo,
                            throughput.xyz, lightSampling, uLight, uPoint);
        if (any(nee.unshadowed > 0.0)) {
            ShadowRay shadow;
            shadow.origin = nee.origin;
            shadow.light = nee.light;
            shadow.direction = nee.direction;
            shadow.primitive = nee.primitive;
            shadow.contribution = nee.unshadowed;
            shadow.tmax = nee.tmax;
            shadow.pixel = pixel;
            shadow.pad0 = 0;
            shadow.pad1 = 0;
            shadow.pad2 = 0;
            params.shadows.push(shadow);
        }
    }


    // Scatter off the mixture: one technique drawn by BSDF_LOBE (the
    // pick's leftover, rescaled, drives the glass technique's
    // reflect-or-refract branch), the estimator divided by the full
    // mixture pdf, gated by the shared sidedness test (openpbr.slang).
    let uLobe = sample_1d(pixel, params.sampleIndex,
                          bounceDimension(bounce, BSDF_LOBE));
    let u = sample_2d(pixel, params.sampleIndex,
                      bounceDimension(bounce, BSDF_DIR));
    let scatter = sampleBsdf(closure, wo, uLobe, u);
    let wiWorld = frame.toWorld(scatter.wi);
    var scattered = float3(0.0);
    if (scatterExitsSurface(scatter, wiWorld, spawn.normal)) {
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
