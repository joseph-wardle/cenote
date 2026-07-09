// Resolve kernel: divide the film's running RGBA f32 sums by the sample
// count, producing the linear ACEScg average — the estimator's current best
// image. This is the render's published output: a standalone linear frame a
// consumer tonemaps (or writes to EXR) without needing the sample count. The
// division is per channel; it agrees with the host `Film::average` readback
// to within a few ULP — GPU division is only approximately rounded, unlike
// the host's — so the viewer's image and the batch EXR are the same picture
// down to display noise.

// Everything the kernel reads, visible in one struct.
struct Params {
    float4* sum;     // the film's running sums, row-major width×height
    float4* average; // sum ÷ samples, same layout
    uint width;
    uint height;
    float samples;   // the sample count, as a float to divide by
    float pad0;
};

[vk::push_constant]
Params params;

[shader("compute")]
[numthreads(8, 8, 1)]
void resolve(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) {
        return;
    }
    uint pixel = tid.y * params.width + tid.x;
    params.average[pixel] = params.sum[pixel] / params.samples;
}
