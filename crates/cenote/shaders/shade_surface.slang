// The surface-shading stage, degenerate for now: shade the hit as its
// world-space geometric normal mapped to color (0.5·n + 0.5) — the M0 demo
// look — and terminate the path by pushing nothing. BSDF sampling, bounce
// continuation, and NEE shadow-ray pushes arrive in M1 steps 7–9; this is
// the kernel they grow inside.

import pathstate;

// Where an instance's triangles live, plus its transform — everything
// needed to re-evaluate shading at a Hit. Mirrors `GeometryRecord` in
// `scene.rs` field for field.
struct GeometryRecord {
    float* positions;        // x, y, z per vertex, object space
    uint* indices;           // three per triangle
    float4 objectToWorld[3]; // rows of the instance's 3×4 transform
};

struct Params {
    Paths paths;
    Queue<uint> hits;
    GeometryRecord* geometry; // indexed by instance custom index
    float4* radiance;         // the film's per-pixel sample for this wave
};

[vk::push_constant]
Params params;

float3 position(float* positions, uint vertex) {
    return float3(positions[3 * vertex],
                  positions[3 * vertex + 1],
                  positions[3 * vertex + 2]);
}

float3 toWorld(GeometryRecord mesh, float3 p) {
    let h = float4(p, 1.0);
    return float3(dot(mesh.objectToWorld[0], h),
                  dot(mesh.objectToWorld[1], h),
                  dot(mesh.objectToWorld[2], h));
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void shade_surface(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.hits.count()) {
        return;
    }
    uint path = params.hits.entries[tid.x];
    Hit hit = params.paths.hit[path];
    GeometryRecord mesh = params.geometry[hit.instance];

    // Fetch the hit triangle's corners and take the cross product in world
    // space — correct for any affine instance transform, since the world
    // triangle's edges *are* the transformed edges.
    uint base = 3 * hit.primitive;
    float3 a = toWorld(mesh, position(mesh.positions, mesh.indices[base]));
    float3 b = toWorld(mesh, position(mesh.positions, mesh.indices[base + 1]));
    float3 c = toWorld(mesh, position(mesh.positions, mesh.indices[base + 2]));
    float3 normal = normalize(cross(b - a, c - a));
    // Two-sided: face the ray (instances disable facing cull).
    if (dot(normal, params.paths.direction[path].xyz) > 0.0) {
        normal = -normal;
    }
    // Pixel-owned write, like shade_miss — determinism needs no atomics.
    params.radiance[params.paths.pixel[path]] = float4(0.5 * normal + 0.5, 1.0);
}
