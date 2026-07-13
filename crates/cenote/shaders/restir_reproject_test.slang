// Test-only kernel: `restir_reproject.slang`'s `reprojectPixel` is the exact
// inverse of raygen's pixel→ray construction, and the whole held-camera argument
// rests on that — a point seen at pixel p, reprojected through the *same* camera,
// must land back on p. This runs the real shader function on a grid of world
// points synthesized as the hits of known pixels, so a sign flip, a dropped
// aspect factor, or a swapped ndc→uv axis is caught directly (never mind that a
// wrong reprojection only costs reuse quality, not correctness — this pins the
// math so the identity reduction is real). Compiled at test time, never embedded.
//
// The host builds a `Reproject` for a pinhole camera and, per thread, the world
// point `origin + t·dir(pixel)` for that thread's pixel (exactly raygen's ray),
// then checks reprojectPixel returns the same pixel. It also drives one moved
// camera (a shifted eye, points must still map somewhere on-screen or off) to
// exercise the non-identity path, and points behind / off-screen to pin the
// rejects. The kernel just reports, per input point, the recovered pixel and
// whether reprojection accepted it; the host asserts the geometry.

import restir_reproject;

struct Params {
    Reproject* reproject;  // the camera under test (its G-buffer pointers unused here)
    float4* points;        // per thread: xyz = a world point to reproject
    uint* pixelOut;        // per thread: the recovered pixel, or 0xffffffff on reject
    uint count;
    uint pad0;
    uint pad1;
    uint pad2;
};

[vk::push_constant]
Params params;

[shader("compute")]
[numthreads(64, 1, 1)]
void restir_reproject_test(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.count) {
        return;
    }
    Reproject r = params.reproject[0];
    let world = params.points[tid.x].xyz;
    uint pixel;
    params.pixelOut[tid.x] = reprojectPixel(r, world, pixel) ? pixel : 0xffffffff;
}
