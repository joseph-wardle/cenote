// Wave entry: one thread per path in the pixel range, generating the
// pinhole-camera ray and pushing the path onto the ray queue — the first
// link of the stage chain, and (with `base`/`count`) the seam where a
// target larger than the path pool is walked range by range.
//
// Right-handed, Y-up, camera looks -Z. Pixel (0,0) is top-left.

import pathstate;

// The camera comes pre-baked into ray-basis vectors by scene::Camera::basis
// on the host: direction = normalize(forward + ndc.x·right + ndc.y·up).
struct Params {
    Paths paths;
    Queue<uint> rays;
    float3 cameraPosition; uint width;  // scalars pack into the float3s'
    float3 cameraRight;    uint height; // std430 padding lanes
    float3 cameraUp;       uint base;   // first pixel of this range
    float3 cameraForward;  uint count;  // paths in this range
};

[vk::push_constant]
Params params;

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void raygen(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.count) {
        return;
    }
    uint pixel = params.base + tid.x;

    // Pixel center → NDC with +y up (pixel rows grow downward).
    let xy = float2(uint2(pixel % params.width, pixel / params.width));
    let uv = (xy + 0.5) / float2(params.width, params.height);
    let ndc = float2(2.0 * uv.x - 1.0, 1.0 - 2.0 * uv.y);
    let direction = normalize(params.cameraForward
                              + ndc.x * params.cameraRight
                              + ndc.y * params.cameraUp);

    params.paths.origin[tid.x] = float4(params.cameraPosition, 0.0);
    params.paths.direction[tid.x] = float4(direction, 0.0);
    params.paths.pixel[tid.x] = pixel;
    params.rays.push(tid.x);
}
