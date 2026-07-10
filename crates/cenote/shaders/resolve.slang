// Resolve kernel: divide the film's running f32 sums — beauty and the
// three AOVs — by the sample count, producing the linear averages that are
// the render's published output: a standalone frame a consumer tonemaps,
// denoises, or writes to EXR without needing the sample count. The
// division is per channel; it agrees with the host `Film::averages`
// readback to within a few ULP — GPU division is only approximately
// rounded, unlike the host's — so the viewer's image and the batch EXR
// are the same picture down to display noise.

// Everything the kernel reads, visible in one struct. Sums are the film's
// accumulators, row-major width×height; each average target shares its
// sum's layout.
struct Params {
    float4* sum;        // beauty: linear ACEScg radiance
    float4* average;
    float4* albedoSum;  // denoiser albedo guide
    float4* albedoAverage;
    float4* normalSum;  // denoiser normal guide
    float4* normalAverage;
    float* depthSum;    // camera-plane z; +inf divides to +inf, as it should
    float* depthAverage;
    uint width;
    uint height;
    float samples;      // the sample count, as a float to divide by
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
    params.albedoAverage[pixel] = params.albedoSum[pixel] / params.samples;
    params.normalAverage[pixel] = params.normalSum[pixel] / params.samples;
    params.depthAverage[pixel] = params.depthSum[pixel] / params.samples;
}
