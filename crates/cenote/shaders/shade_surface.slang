// The surface-shading stage: reconstruct the hit, scatter off the EON
// diffuse base, and push the path back onto the ray queue for the next
// bounce — until it escapes to the sky, hits the depth cap, or loses
// Russian roulette. No NEE or emissive surfaces yet (M1 step 8), so a
// surface adds no radiance of its own: a path that terminates here
// contributes black, written now — every pixel gets exactly one radiance
// write per wave (here or in shade_miss), which keeps writes pixel-owned
// and renders bitwise deterministic.

import openpbr;
import pathstate;
import rng;

// Where an instance's triangles live, plus its transforms — everything
// needed to re-evaluate shading at a Hit. Mirrors `GeometryRecord` in
// `scene.rs` field for field.
struct GeometryRecord {
    float* positions;        // x, y, z per vertex, object space
    uint* indices;           // three per triangle
    float4 objectToWorld[3]; // rows of the instance's 3×4 transform
    float4 worldToObject[3]; // rows of its inverse — normals, error bounds
};

struct Params {
    Paths paths;
    Queue<uint> hits;
    Queue<uint> rays;         // the next bounce's input: survivors push here
    GeometryRecord* geometry; // indexed by instance custom index
    Material* materials;      // indexed by instance custom index
    float4* radiance;         // the film's per-pixel sample for this wave
    uint sampleIndex; // which sample of every pixel's sequence this wave is
    uint bounce;      // which bounce of the wave this dispatch shades
    uint maxBounces;  // path-length cap: the last bounce never continues
    uint pad;
};

[vk::push_constant]
Params params;

// Bounces before this always survive — they carry most of the image's
// energy, and killing them early would just trade traced rays for noise.
// From here on, paths gamble survival against their throughput and the
// survivors are reweighted to keep the estimate unbiased.
static const uint ROULETTE_START = 3;

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

    // The depth cap. A capped path carries no radiance (only the sky emits,
    // and this path never reached it), so its pixel's one write is black.
    if (params.bounce + 1 >= params.maxBounces) {
        params.radiance[pixel] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    Hit hit = params.paths.hit[path];
    let direction = params.paths.direction[path].xyz;
    let spawn = spawnPoint(params.geometry[hit.instance], hit, direction);
    Material material = params.materials[hit.instance];

    // Scatter off the diffuse lobe. Cosine-sampled, so the path-throughput
    // update f·cos/pdf collapses to f·π.
    let frame = Frame(spawn.normal);
    let wo = frame.toLocal(-direction);
    let u = sample_2d(pixel, params.sampleIndex,
                      bounceDimension(params.bounce, BSDF_DIR));
    let wi = sampleCosineHemisphere(u);
    var throughput = params.paths.throughput[path].xyz * PI
        * eonBrdf(material.baseColor, material.baseRoughness, wi.xyz, wo);

    if (params.bounce >= ROULETTE_START) {
        let survival = min(1.0, max(throughput.r, max(throughput.g, throughput.b)));
        if (sample_1d(pixel, params.sampleIndex,
                      bounceDimension(params.bounce, RR)) >= survival) {
            params.radiance[pixel] = float4(0.0, 0.0, 0.0, 1.0);
            return;
        }
        throughput /= survival;
    }

    params.paths.origin[path] = float4(spawn.position, 0.0);
    params.paths.direction[path] = float4(frame.toWorld(wi.xyz), 0.0);
    params.paths.throughput[path] = float4(throughput, 0.0);
    params.rays.push(path);
}
