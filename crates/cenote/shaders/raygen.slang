// Wave entry: one thread per path in the pixel range, generating the
// jittered camera ray — pinhole, or thin lens when the aperture is open —
// and pushing the path onto the ray queue: the first link of the stage
// chain, and (with `base`/`count`) the seam where a target larger than
// the path pool is walked range by range.
//
// Right-handed, Y-up, camera looks -Z. Pixel (0,0) is top-left.

import pathstate;
import rng;

static const float PI = 3.14159265358979323846;

// The camera comes pre-baked into ray-basis vectors by scene::Camera::basis
// on the host: direction = normalize(forward + ndc.x·right + ndc.y·up).
// With an open aperture the host scales the whole basis by the focus
// distance, which turns that same sum into the pixel's focal *point*
// offset from the camera — the point every ray of the pixel converges on,
// wherever on the lens disk it starts.
struct Params {
    Paths paths;
    Queue<uint> rays;
    uint sampleIndex; // which sample of every pixel's sequence this wave is
    float apertureRadius; // lens radius, meters; 0 is the pinhole path
    float3 cameraPosition; uint width;  // scalars pack into the float3s'
    float3 cameraRight;    uint height; // std430 padding lanes
    float3 cameraUp;       uint base;   // first pixel of this range
    float3 cameraForward;  uint count;  // paths in this range
};

[vk::push_constant]
Params params;

// Shirley–Chiu concentric mapping: the unit square onto the unit disk,
// area-preserving and low-distortion, so the lens stays uniformly and
// well-stratifiedly covered. Wedges are folded so the divisor is always
// the larger coordinate — no pole, no division by ~0.
float2 concentricDisk(float2 u) {
    let offset = 2.0 * u - 1.0;
    if (offset.x == 0.0 && offset.y == 0.0) {
        return float2(0.0);
    }
    float r;
    float theta;
    if (abs(offset.x) > abs(offset.y)) {
        r = offset.x;
        theta = (PI / 4.0) * (offset.y / offset.x);
    } else {
        r = offset.y;
        theta = (PI / 2.0) - (PI / 4.0) * (offset.x / offset.y);
    }
    return r * float2(cos(theta), sin(theta));
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void raygen(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.count) {
        return;
    }
    uint pixel = params.base + tid.x;

    // A jittered point of the pixel footprint → NDC with +y up (pixel rows
    // grow downward). The jitter is box-filter anti-aliasing and the
    // sampler's first consumer: each accumulated sample looks through a
    // different sub-pixel point, so edges converge as the film averages.
    let jitter = sample_2d(pixel, params.sampleIndex, CAMERA_JITTER);
    let xy = float2(uint2(pixel % params.width, pixel / params.width)) + jitter;
    let uv = xy / float2(params.width, params.height);
    let ndc = float2(2.0 * uv.x - 1.0, 1.0 - 2.0 * uv.y);
    let target = params.cameraForward
               + ndc.x * params.cameraRight
               + ndc.y * params.cameraUp;
    var origin = params.cameraPosition;
    var direction = normalize(target);
    if (params.apertureRadius > 0.0) {
        // Thin lens: start the ray from a uniform point on the lens disk
        // and aim it at the pixel's focal point. Geometry on the focal
        // plane images identically from everywhere on the disk (sharp);
        // everything else smears across the accumulating samples (bokeh).
        let disk = params.apertureRadius
                 * concentricDisk(sample_2d(pixel, params.sampleIndex, CAMERA_LENS));
        let offset = disk.x * normalize(params.cameraRight)
                   + disk.y * normalize(params.cameraUp);
        origin += offset;
        direction = normalize(target - offset);
    }

    params.paths.origin[tid.x] = float4(origin, 0.0);
    params.paths.direction[tid.x] = float4(direction, 0.0);
    params.paths.pixel[tid.x] = pixel;
    // w = 0: a camera ray has no scatter pdf, which tells the first
    // surface's emission that no light-sampling strategy competed for it.
    params.paths.throughput[tid.x] = float4(1.0, 1.0, 1.0, 0.0);
    params.rays.push(tid.x);
}
