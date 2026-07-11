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

// One float3 out of a tightly packed vertex-attribute buffer.
float3 fetch3(float* attributes, uint vertex) {
    return float3(attributes[3 * vertex],
                  attributes[3 * vertex + 1],
                  attributes[3 * vertex + 2]);
}

// A reconstructed hit: where the next ray segment starts — nudged off
// either face, since refraction continues on the far side — plus both
// unit normals. The interpolated shading normal carries the BSDF frame —
// it is what makes a coarse sphere render smooth — while the geometric
// normal keeps the jobs that must match the actual triangle: the spawn
// offsets, and the sidedness guards that reject directions passing
// through the surface the shading frame pretends away.
struct SpawnPoint {
    float3 position; // nudged toward the arriving ray — reflections
    float3 below;    // nudged past the surface — transmissions
    float3 normal;  // geometric, facing the arriving ray
    float3 shading; // interpolated, flipped to the geometric side
    bool frontFace; // did the ray strike the winding-front side? (emission is one-sided)
};

// Reconstruct the world-space hit point and geometric normal, both facing
// the arriving ray (surfaces are two-sided), and offset the point along
// the normal by a rigorous bound on its own floating-point reconstruction
// error — so the next segment can trace with TMin 0 and still never hit
// the triangle it left, whichever side it leaves from. Ported from the
// reference GLSL of "Solving Self-Intersection Artifacts in DirectX
// Raytracing" (van Antwerpen, NVIDIA, 2023). Never a magic epsilon: every
// term scales with the coordinates it bounds, so this holds at any scene
// scale.
SpawnPoint spawnPoint(GeometryRecord mesh, Hit hit, float3 rayDirection) {
    uint base = 3 * hit.primitive;
    let v0 = fetch3(mesh.positions, mesh.indices[base]);
    let v1 = fetch3(mesh.positions, mesh.indices[base + 1]);
    let v2 = fetch3(mesh.positions, mesh.indices[base + 2]);
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

    // Emission is one-sided: record which side the ray struck by the raw
    // winding normal, before it is flushed to face the ray below.
    let frontFace = dot(wldNormal, rayDirection) < 0.0;
    // Face the arriving ray (instances disable facing cull), so the offset
    // pushes off the side the scattered ray leaves from.
    if (dot(wldNormal, rayDirection) > 0.0) {
        wldNormal = -wldNormal;
    }

    // The shading normal: vertex normals lerped by the hit's barycentrics,
    // through the same inverse transpose. Flipped to the geometric side —
    // shading must agree with the face the ray actually struck.
    let n0 = fetch3(mesh.normals, mesh.indices[base]);
    let n1 = fetch3(mesh.normals, mesh.indices[base + 1]);
    let n2 = fetch3(mesh.normals, mesh.indices[base + 2]);
    let objShading = n0 + hit.barycentrics.x * (n1 - n0)
                        + hit.barycentrics.y * (n2 - n0);
    var shading = normalize(w2o[0].xyz * objShading.x
                          + w2o[1].xyz * objShading.y
                          + w2o[2].xyz * objShading.z);
    if (dot(shading, wldNormal) < 0.0) {
        shading = -shading;
    }

    SpawnPoint spawn;
    spawn.position = fma(float3(wldOffset), wldNormal, wldPosition);
    spawn.below = fma(float3(-wldOffset), wldNormal, wldPosition);
    spawn.normal = wldNormal;
    spawn.shading = shading;
    spawn.frontFace = frontFace;
    return spawn;
}

// The shading normal under a tangent-space normal map. The tangent frame
// derives per hit from the triangle's positions and UVs (no authored
// tangents until anisotropy demands them): solve the triangle's 2×2 UV
// system for the world direction of +u, orthonormalize it against the
// interpolated shading normal, and take the bitangent's sign from the
// +v solution so mirrored UV islands map correctly. A degenerate
// parameterization (zero UV area — including the all-zero stream of an
// unauthored mesh) keeps the unmapped normal. Like the interpolated
// normal it replaces, the result is flipped to the geometric side.
float3 mappedShadingNormal(GeometryRecord mesh, Hit hit, float2 uv, uint map,
                           SpawnPoint spawn) {
    uint base = 3 * hit.primitive;
    let i0 = mesh.indices[base];
    let i1 = mesh.indices[base + 1];
    let i2 = mesh.indices[base + 2];
    let duv1 = mesh.uvs[i1] - mesh.uvs[i0];
    let duv2 = mesh.uvs[i2] - mesh.uvs[i0];
    let det = duv1.x * duv2.y - duv1.y * duv2.x;
    if (abs(det) < 1e-12) {
        return spawn.shading;
    }
    let e1 = fetch3(mesh.positions, i1) - fetch3(mesh.positions, i0);
    let e2 = fetch3(mesh.positions, i2) - fetch3(mesh.positions, i0);
    let o2w = mesh.objectToWorld;
    // World-space edges — the transform's linear part only.
    let we1 = float3(dot(o2w[0].xyz, e1), dot(o2w[1].xyz, e1), dot(o2w[2].xyz, e1));
    let we2 = float3(dot(o2w[0].xyz, e2), dot(o2w[1].xyz, e2), dot(o2w[2].xyz, e2));
    let dpdu = (duv2.y * we1 - duv1.y * we2) / det;
    let dpdv = (duv1.x * we2 - duv2.x * we1) / det;
    var tangent = dpdu - spawn.shading * dot(spawn.shading, dpdu);
    let lenSqr = dot(tangent, tangent);
    if (!(lenSqr > 1e-20)) {
        return spawn.shading;
    }
    tangent *= rsqrt(lenSqr);
    var bitangent = cross(spawn.shading, tangent);
    if (dot(bitangent, dpdv) < 0.0) {
        bitangent = -bitangent;
    }
    let texel = sampleNormal(map, uv);
    var mapped = normalize(texel.x * tangent + texel.y * bitangent
                         + texel.z * spawn.shading);
    // Shading must agree with the face the ray actually struck, exactly
    // as the interpolated normal it perturbs.
    if (dot(mapped, spawn.normal) < 0.0) {
        mapped = -mapped;
    }
    return mapped;
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void shade_surface(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.hits.count()) {
        return;
    }
    let bounce = params.packed & 0xff;
    let maxBounces = (params.packed >> 8) & 0xff;
    let lightSampling = params.packed >> 16;
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
        if (throughput.w > 0.0) {
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
    if (lightSampling != LIGHT_SAMPLING_BSDF
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
