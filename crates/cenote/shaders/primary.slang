// Primary-visibility kernel: one pinhole-camera ray per pixel, traced inline
// against the scene TLAS. A hit shades as the world-space geometric normal
// mapped to color (0.5·n + 0.5); a miss shades black. This is the M0 demo
// image — winding or handedness mistakes show up as the wrong rainbow.
//
// Conventions (must match the crate doc in src/lib.rs): right-handed, Y-up,
// camera looks -Z. Pixel (0,0) is top-left; the pixel buffer is row-major
// RGBA f32.

// The scene TLAS — the one resource that can't travel as a device address
// (decision D-006), so it owns the binding model's single descriptor set.
[vk::binding(0, 0)]
RaytracingAccelerationStructure scene;

// Where an instance's triangles live (D-017): the tightly packed position and
// index buffers its BLAS was built from, keyed by the instance custom index.
// Scalar pointers, because std430 would stride a float3 array by 16 bytes.
struct GeometryRecord {
    float* positions; // x, y, z per vertex
    uint* indices;    // three per triangle
};

// Everything else the kernel reads, visible in one struct (D-006: BDA-first —
// buffers arrive as raw device addresses in push constants). The camera comes
// pre-baked into ray-basis vectors by scene::Camera::basis on the host:
// direction = normalize(forward + ndc.x · right + ndc.y · up).
struct Params {
    float4* pixels;
    GeometryRecord* geometry;
    float3 cameraPosition; uint width;   // scalars pack into the float3s'
    float3 cameraRight;    uint height;  // std430 padding lanes
    float3 cameraUp;       float pad0;
    float3 cameraForward;  float pad1;
};

[vk::push_constant]
Params params;

float3 position(float* positions, uint vertex) {
    return float3(positions[3 * vertex],
                  positions[3 * vertex + 1],
                  positions[3 * vertex + 2]);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void primary(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) {
        return;
    }

    // Pixel center → NDC with +y up (pixel rows grow downward).
    let uv = (float2(tid.xy) + 0.5) / float2(params.width, params.height);
    let ndc = float2(2.0 * uv.x - 1.0, 1.0 - 2.0 * uv.y);

    RayDesc ray;
    ray.Origin = params.cameraPosition;
    ray.Direction = normalize(params.cameraForward
                              + ndc.x * params.cameraRight
                              + ndc.y * params.cameraUp);
    ray.TMin = 0.0;
    ray.TMax = 1e30;

    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(scene, RAY_FLAG_NONE, 0xFF, ray);
    while (query.Proceed()) {
        // Nothing to consider: every BLAS is opaque, so traversal commits
        // triangle hits on its own.
    }

    var color = float3(0.0);
    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        // Fetch the hit triangle's corners (D-017) and take the cross
        // product in world space — correct for any affine instance
        // transform, since translation cancels in the edge differences.
        GeometryRecord mesh = params.geometry[query.CommittedInstanceID()];
        uint base = 3 * query.CommittedPrimitiveIndex();
        float3x4 objectToWorld = query.CommittedObjectToWorld3x4();
        float3 a = mul(objectToWorld, float4(position(mesh.positions, mesh.indices[base]), 1.0));
        float3 b = mul(objectToWorld, float4(position(mesh.positions, mesh.indices[base + 1]), 1.0));
        float3 c = mul(objectToWorld, float4(position(mesh.positions, mesh.indices[base + 2]), 1.0));
        float3 normal = normalize(cross(b - a, c - a));
        // Two-sided: face the camera (instances disable facing cull).
        if (dot(normal, ray.Direction) > 0.0) {
            normal = -normal;
        }
        color = 0.5 * normal + 0.5;
    }
    params.pixels[tid.y * params.width + tid.x] = float4(color, 1.0);
}
