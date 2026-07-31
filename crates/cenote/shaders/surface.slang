// Hit reconstruction shared by every stage that shades a surface: the
// wavefront's `shade_surface`, and M3's `restir_candidates`/`restir_resolve`,
// which re-derive the same spawn point and shading frame to weigh and shade a
// reservoir's light sample. Splitting it here keeps that geometry in one place
// — a drift between the three would put the reservoir's target function and
// its resolve on a different surface than the path tracer, a silent bias.

import pathstate;
import scene;
import textures;

// A reconstructed hit: where the next ray segment starts — nudged off either
// face, since refraction continues on the far side — plus both unit normals.
// The interpolated shading normal carries the BSDF frame — it is what makes a
// coarse sphere render smooth — while the geometric normal keeps the jobs that
// must match the actual triangle: the spawn offsets, and the sidedness guards
// that reject directions passing through the surface the shading frame pretends
// away.
public struct SpawnPoint {
    public float3 position; // nudged toward the arriving ray — reflections
    public float3 below;    // nudged past the surface — transmissions
    public float3 normal;  // geometric, facing the arriving ray
    public float3 shading; // interpolated, flipped to the geometric side
    public bool frontFace; // did the ray strike the winding-front side? (emission is one-sided)
};

// One float3 out of a tightly packed vertex-attribute buffer.
public float3 fetch3(float* attributes, uint vertex) {
    return float3(attributes[3 * vertex],
                  attributes[3 * vertex + 1],
                  attributes[3 * vertex + 2]);
}

// Reconstruct the world-space hit point and geometric normal, both facing the
// arriving ray (surfaces are two-sided), and offset the point along the normal
// by a rigorous bound on its own floating-point reconstruction error — so the
// next segment can trace with TMin 0 and still never hit the triangle it left,
// whichever side it leaves from. Ported from the reference GLSL of "Solving
// Self-Intersection Artifacts in DirectX Raytracing" (van Antwerpen, NVIDIA,
// 2023). Never a magic epsilon: every term scales with the coordinates it
// bounds, so this holds at any scene scale.
public SpawnPoint spawnPoint(GeometryRecord mesh, Hit hit, float3 rayDirection) {
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
    // winding normal, before it is flushed to face the ray below. No
    // mirror correction is needed here — dot(M·n, M⁻ᵀ·n) = |n|² > 0 for
    // any invertible M, so the inverse-transpose normal above always lands
    // on the mapped image of the object-space front, even under a
    // negative-determinant instance transform. This is the convention the
    // light records' `winding` sign keeps `connectTriangle` in agreement
    // with (their baked world edges do flip under a mirror).
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

// The world-space point of a reconnection vertex, independent of the ray that
// found it: `spawnPoint`'s two offset points straddle the raw reconstructed
// position, so their midpoint recovers it without the direction-dependent
// nudge. Candidate, resolve, and every reusing neighbour recover the exact same
// x₂ from its `Hit` this way, keeping the reconnection direction — and so the
// shift's target and Jacobian — bit-identical across the stages (D-134).
public float3 reconnectionPoint(GeometryRecord mesh, Hit hit) {
    let spawn = spawnPoint(mesh, hit, float3(0.0, 0.0, 1.0));
    return 0.5 * (spawn.position + spawn.below);
}

// The shading normal under a tangent-space normal map. The tangent frame
// derives per hit from the triangle's positions and UVs (no authored tangents
// until anisotropy demands them): solve the triangle's 2×2 UV system for the
// world direction of +u, orthonormalize it against the interpolated shading
// normal, and take the bitangent's sign from the +v solution so mirrored UV
// islands map correctly. A degenerate parameterization (zero UV area —
// including the all-zero stream of an unauthored mesh) keeps the unmapped
// normal. Like the interpolated normal it replaces, the result is flipped to
// the geometric side.
public float3 mappedShadingNormal(TextureParams *params, GeometryRecord mesh, Hit hit,
                                  float2 uv, uint map, SpawnPoint spawn) {
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
    let texel = sampleNormal(params, map, uv);
    var mapped = normalize(texel.x * tangent + texel.y * bitangent
                         + texel.z * spawn.shading);
    // Shading must agree with the face the ray actually struck, exactly as the
    // interpolated normal it perturbs.
    if (dot(mapped, spawn.normal) < 0.0) {
        mapped = -mapped;
    }
    return mapped;
}
