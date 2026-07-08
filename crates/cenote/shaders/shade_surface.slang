// The surface-shading stage: reconstruct the hit, collect the surface's
// emission (MIS-weighted — a BSDF-sampled path landing on a light competes
// with next-event estimation for the same connection), send one shadow ray
// toward the lights (a quad or the environment, chosen by power), and
// scatter off the OpenPBR mixture — until the path escapes to the sky,
// hits the depth cap, or loses Russian roulette.
//
// Radiance bookkeeping: the wave's per-pixel accumulator starts at zero
// and every write below is a plain +=. A pixel has one path and a path
// visits this kernel once per bounce, so each dispatch performs at most
// one read-modify-write per pixel — which keeps writes pixel-owned and
// renders bitwise deterministic. The one *terminal* add per path carries
// alpha 1 (here or in shade_miss): "every pixel finished exactly once"
// stays mechanically checkable.

import lights;
import openpbr;
import pathstate;
import rng;
import scene;

struct Params {
    Paths paths;
    Queue<uint> hits;
    Queue<uint> rays;         // the next bounce's input: survivors push here
    Queue<ShadowRay> shadows; // next-event connections, for trace_shadow
    SceneTable* scene;
    float4* radiance;         // the film's per-pixel accumulator for this wave
    uint sampleIndex; // which sample of every pixel's sequence this wave is
    uint bounce;      // which bounce of the wave this dispatch shades
    uint maxBounces;  // path-length cap: the last bounce never continues
    uint lightSampling; // LIGHT_SAMPLING_*: which strategies reach the lights
};

[vk::push_constant]
Params params;

// Bounces before this always survive — they carry most of the image's
// energy, and killing them early would just trade traced rays for noise.
// From here on, paths gamble survival against their throughput and the
// survivors are reweighted to keep the estimate unbiased.
static const uint ROULETTE_START = 3;

// Veach's power heuristic (β = 2): the weight of a sample drawn from the
// strategy with pdf `p` when the strategy with pdf `q` could have produced
// it too. The weights of the two draws sum to 1, so combining strategies
// stays unbiased; squaring favors whichever pdf is sharper.
float powerHeuristic(float p, float q) {
    return (p * p) / (p * p + q * q);
}

float3 position(float* positions, uint vertex) {
    return float3(positions[3 * vertex],
                  positions[3 * vertex + 1],
                  positions[3 * vertex + 2]);
}

// A reconstructed hit: where the next ray segment starts, already nudged
// off the surface, and the unit geometric normal it shades with.
struct SpawnPoint {
    float3 position;
    float3 normal;
};

// Reconstruct the world-space hit point and geometric normal, both facing
// the arriving ray (surfaces are two-sided), and offset the point along
// the normal by a rigorous bound on its own floating-point reconstruction
// error — so the next segment can trace with TMin 0 and still never hit
// the triangle it left. Ported from the reference GLSL of "Solving
// Self-Intersection Artifacts in DirectX Raytracing" (van Antwerpen,
// NVIDIA, 2023). Never a magic epsilon: every term scales with the
// coordinates it bounds, so this holds at any scene scale.
SpawnPoint spawnPoint(GeometryRecord mesh, Hit hit, float3 rayDirection) {
    uint base = 3 * hit.primitive;
    let v0 = position(mesh.positions, mesh.indices[base]);
    let v1 = position(mesh.positions, mesh.indices[base + 1]);
    let v2 = position(mesh.positions, mesh.indices[base + 2]);
    let o2w = mesh.objectToWorld;
    let w2o = mesh.worldToObject;

    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let objPosition = v0 + fma(float3(hit.barycentrics.x), edge1,
                               hit.barycentrics.y * edge2);
    let objNormal = cross(edge1, edge2);

    let wldPosition = float3(
        o2w[0].w + fma(o2w[0].x, objPosition.x,
                       fma(o2w[0].y, objPosition.y, o2w[0].z * objPosition.z)),
        o2w[1].w + fma(o2w[1].x, objPosition.x,
                       fma(o2w[1].y, objPosition.y, o2w[1].z * objPosition.z)),
        o2w[2].w + fma(o2w[2].x, objPosition.x,
                       fma(o2w[2].y, objPosition.y, o2w[2].z * objPosition.z)));
    // Normals transform by the inverse transpose: sum the world-to-object
    // rows weighted by the object normal's components.
    var wldNormal = w2o[0].xyz * objNormal.x
                  + w2o[1].xyz * objNormal.y
                  + w2o[2].xyz * objNormal.z;
    let wldScale = rsqrt(dot(wldNormal, wldNormal));
    wldNormal *= wldScale;

    // The reference's error constants: c0/c1 bound the barycentric lerp's
    // round-off, c1/c2 the transforms'.
    const float c0 = 5.9604644775390625e-8;
    const float c1 = 1.788139769587360206060111522674560546875e-7;
    const float c2 = 1.19209317972490680404007434844970703125e-7;

    let extent3 = abs(edge1) + abs(edge2) + abs(edge1 - edge2);
    let extent = max(max(extent3.x, extent3.y), extent3.z);
    var objErr = fma(float3(c0), abs(v0), float3(c1 * extent));

    let wldErr = fma(float3(c1),
                     float3(dot(abs(o2w[0].xyz), abs(objPosition)),
                            dot(abs(o2w[1].xyz), abs(objPosition)),
                            dot(abs(o2w[2].xyz), abs(objPosition))),
                     c2 * abs(float3(o2w[0].w, o2w[1].w, o2w[2].w)));
    objErr = fma(float3(c2),
                 float3(dot(abs(w2o[0].xyz), abs(wldPosition)) + abs(w2o[0].w),
                        dot(abs(w2o[1].xyz), abs(wldPosition)) + abs(w2o[1].w),
                        dot(abs(w2o[2].xyz), abs(wldPosition)) + abs(w2o[2].w)),
                 objErr);

    var wldOffset = dot(wldErr, abs(wldNormal));
    let objOffset = dot(objErr, abs(objNormal));
    wldOffset = fma(wldScale, objOffset, wldOffset);

    // Two-sided: face the arriving ray (instances disable facing cull), so
    // the offset pushes off the side the scattered ray leaves from.
    if (dot(wldNormal, rayDirection) > 0.0) {
        wldNormal = -wldNormal;
    }
    SpawnPoint spawn;
    spawn.position = fma(float3(wldOffset), wldNormal, wldPosition);
    spawn.normal = wldNormal;
    return spawn;
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void shade_surface(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.hits.count()) {
        return;
    }
    uint path = params.hits.entries[tid.x];
    uint pixel = params.paths.pixel[path];
    Hit hit = params.paths.hit[path];
    let direction = params.paths.direction[path].xyz;
    GeometryRecord mesh = params.scene.geometry[hit.instance];
    Material material = params.scene.materials[hit.instance];
    let spawn = spawnPoint(mesh, hit, direction);
    // xyz = the path's weight; w = the previous scatter's pdf (0: camera ray).
    let throughput = params.paths.throughput[path];

    // Emission. This hit is the BSDF strategy's way of reaching a light,
    // so it is weighted against the probability that next-event sampling
    // at the previous vertex would have picked the same point — except on
    // camera rays, where no other strategy competed and the weight is 1.
    var contribution = float3(0.0);
    if (mesh.light != LIGHT_NONE) {
        var weight = 1.0;
        if (throughput.w > 0.0) {
            if (params.lightSampling == LIGHT_SAMPLING_NEE) {
                weight = 0.0;
            } else if (params.lightSampling == LIGHT_SAMPLING_MIS) {
                let delta = spawn.position - params.paths.origin[path].xyz;
                let cosLight = abs(dot(spawn.normal, direction));
                // The full next-event pdf: choose the quads over the
                // environment, then this quad and this point.
                let neePdf = (1.0 - params.scene.env.selectProb)
                           * params.scene.lights[mesh.light].pdfArea
                           * dot(delta, delta) / cosLight;
                weight = powerHeuristic(throughput.w, neePdf);
            }
        }
        contribution = throughput.xyz * material.emission * weight;
    }

    // The depth cap: the path ends at this vertex, so this is its pixel's
    // terminal add. No next-event estimation here — an NEE connection from
    // this vertex would be one segment longer than any path BSDF sampling
    // can trace, and the two strategies must cover the same path lengths
    // to converge to the same image.
    if (params.bounce + 1 >= params.maxBounces) {
        params.radiance[pixel] += float4(contribution, 1.0);
        return;
    }

    let frame = Frame(spawn.normal);
    let wo = frame.toLocal(-direction);

    // Next-event estimation: connect this vertex to one candidate — a
    // direction toward the environment or a point on a quad light, chosen
    // in proportion to their power (the choice rides the low end of the
    // NEE_LIGHT draw; the quad branch remaps the leftover back to a full-
    // range uniform for the alias walk). The contribution is fully
    // weighted here; trace_shadow just adds it if the connection is
    // unoccluded.
    let envProb = params.scene.env.selectProb;
    if (params.lightSampling != LIGHT_SAMPLING_BSDF
        && (params.scene.lightCount > 0 || envProb > 0.0)) {
        let uLight = sample_1d(pixel, params.sampleIndex,
                               bounceDimension(params.bounce, NEE_LIGHT));
        let uPoint = sample_2d(pixel, params.sampleIndex,
                               bounceDimension(params.bounce, NEE_POINT));
        ShadowRay shadow;
        shadow.origin = spawn.position;
        shadow.pixel = pixel;
        shadow.pad = 0.0;
        var unshadowed = float3(0.0);
        if (uLight < envProb) {
            // Environment: a direction, unoccluded iff the shadow ray hits
            // nothing (LIGHT_NONE is no instance's ID). A pole sample
            // reports pdf 0 and is skipped — it carries no usable density.
            let candidate = params.scene.env.sample(uPoint);
            let wiLight = frame.toLocal(candidate.direction);
            if (candidate.pdf > 0.0 && wiLight.z > 0.0) {
                let neePdf = envProb * candidate.pdf;
                let scatterPdf = bsdfPdf(material, wiLight, wo);
                let weight = params.lightSampling == LIGHT_SAMPLING_MIS
                           ? powerHeuristic(neePdf, scatterPdf)
                           : 1.0;
                unshadowed = throughput.xyz * candidate.radiance
                    * evalBsdf(material, wiLight, wo)
                    * wiLight.z * (weight / neePdf);
                shadow.light = LIGHT_NONE;
                shadow.direction = candidate.direction;
            }
        } else {
            // A quad light. Scene prep pins envProb to exactly 1 when no
            // quads exist, so this branch always has a light list to walk.
            let uQuad = (uLight - envProb) / (1.0 - envProb);
            let candidate = sampleLights(params.scene.lights,
                                         params.scene.lightCount, uQuad, uPoint);
            let delta = candidate.position - spawn.position;
            let r2 = dot(delta, delta);
            let wiWorld = delta * rsqrt(r2);
            let wiLight = frame.toLocal(wiWorld);
            let cosLight = abs(dot(candidate.normal, wiWorld));
            if (r2 > 0.0 && wiLight.z > 0.0 && cosLight > 0.0) {
                // The candidate's pdf in solid angle at this vertex,
                // against the pdf BSDF sampling would give the direction.
                let neePdf = (1.0 - envProb) * candidate.pdfArea * r2 / cosLight;
                let scatterPdf = bsdfPdf(material, wiLight, wo);
                let weight = params.lightSampling == LIGHT_SAMPLING_MIS
                           ? powerHeuristic(neePdf, scatterPdf)
                           : 1.0;
                unshadowed = throughput.xyz * candidate.emission
                    * evalBsdf(material, wiLight, wo)
                    * wiLight.z * (weight / neePdf);
                shadow.light = candidate.instance;
                shadow.direction = wiWorld;
            }
        }
        if (any(unshadowed > 0.0)) {
            shadow.contribution = unshadowed;
            params.shadows.push(shadow);
        }
    }

    // Scatter off the mixture: one technique (VNDF or cosine) drawn by
    // BSDF_LOBE, the estimator divided by the full mixture pdf.
    let uLobe = sample_1d(pixel, params.sampleIndex,
                          bounceDimension(params.bounce, BSDF_LOBE));
    let u = sample_2d(pixel, params.sampleIndex,
                      bounceDimension(params.bounce, BSDF_DIR));
    let wi = sampleBsdf(material, wo, uLobe, u);
    var scattered = float3(0.0);
    if (wi.z > 0.0 && wi.w > 0.0) {
        scattered = throughput.xyz * evalBsdf(material, wi.xyz, wo)
                  * (wi.z / wi.w);
    }

    // A path that can carry nothing forward is done — lights themselves
    // scatter nothing (their base color is black), and continuing would
    // just trace dead rays until roulette noticed.
    var survival = max(scattered.r, max(scattered.g, scattered.b));
    if (survival <= 0.0) {
        params.radiance[pixel] += float4(contribution, 1.0);
        return;
    }

    if (params.bounce >= ROULETTE_START) {
        survival = min(1.0, survival);
        if (sample_1d(pixel, params.sampleIndex,
                      bounceDimension(params.bounce, RR)) >= survival) {
            params.radiance[pixel] += float4(contribution, 1.0);
            return;
        }
        scattered /= survival;
    }

    params.paths.origin[path] = float4(spawn.position, 0.0);
    params.paths.direction[path] = float4(frame.toWorld(wi.xyz), 0.0);
    params.paths.throughput[path] = float4(scattered, wi.w);
    params.rays.push(path);

    // The path continues, but its pixel keeps the emission found here.
    if (any(contribution > 0.0)) {
        params.radiance[pixel] += float4(contribution, 0.0);
    }
}
