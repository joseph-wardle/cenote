// Primary-visibility kernel.
//
// M0 step 2: fills a UV gradient so the first dispatch (m0-plan §4 step 5) is
// verifiable end-to-end by eye; the ray query against the scene TLAS replaces
// the gradient in step 7.
//
// Conventions (must match the crate doc in src/lib.rs): right-handed, Y-up,
// camera looks -Z. Pixel (0,0) is top-left; the pixel buffer is row-major
// RGBA f32.

// Everything the kernel reads, visible in one struct (decision D-006:
// BDA-first — buffers arrive as raw device addresses in push constants).
struct Params {
    float4* pixels;
    uint2 resolution;
};

[vk::push_constant]
Params params;

[shader("compute")]
[numthreads(8, 8, 1)]
void primary(uint3 tid: SV_DispatchThreadID) {
    if (any(tid.xy >= params.resolution)) {
        return;
    }
    let idx = tid.y * params.resolution.x + tid.x;
    let uv = float2(tid.xy) / float2(params.resolution);
    params.pixels[idx] = float4(uv, 0.25, 1.0);
}
