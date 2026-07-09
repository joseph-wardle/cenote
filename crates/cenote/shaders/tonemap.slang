// Tonemap kernel: turn the film's resolved linear ACEScg average into the
// displayable frame. Per pixel: expose (× 2^stops), then the ACES display
// transform, the sRGB transfer curve, and a pack to the RGBA8 bytes the
// presenter blits to the swapchain. The average-over-samples already
// happened in the resolve kernel — this is the pure view transform, the
// consumer's half of the estimator/view split.
//
// The ACES transform is Stephen Hill's analytic RRT+ODT fit (BakingLab,
// github.com/TheRealMJP/BakingLab), the shader-friendly stand-in the ACES
// community recommends until a baked-LUT stage replaces it. This kernel is
// that swappable display stage, not a baked-in look: AgX or ACES 2.0 slot
// in here without touching the film. Hill's fit takes linear Rec.709, so
// ACEScg values pass through SRGB_FROM_ACESCG first. EXR output never comes
// through here — files stay linear ACEScg.

// Everything the kernel reads, visible in one struct.
struct Params {
    float4* average;    // the film's resolved linear average, row-major width×height
    uint* display;      // packed RGBA8 output, sRGB-encoded, same layout
    uint width;
    uint height;
    float exposureScale; // exp2(exposure stops)
    float pad0;
};

[vk::push_constant]
Params params;

// ACEScg (AP1, D60 white) → linear Rec.709 (D65, Bradford-adapted), from
// the ACES reference implementation's primaries. Rows sum to 1: AP1 white
// maps to Rec.709 white.
static const float3x3 SRGB_FROM_ACESCG = float3x3(
     1.705050993, -0.621792121, -0.083258872,
    -0.130256418,  1.140804737, -0.010548319,
    -0.024003357, -0.128968976,  1.152972333);

// Hill's fit, verbatim: sRGB→AP1 with the RRT's saturation adjustment
// baked in, the rational curve approximating RRT+ODT, and the way back to
// linear Rec.709.
static const float3x3 ACES_INPUT = float3x3(
    0.59719, 0.35458, 0.04823,
    0.07600, 0.90834, 0.01566,
    0.02840, 0.13383, 0.83777);

static const float3x3 ACES_OUTPUT = float3x3(
     1.60475, -0.53108, -0.07367,
    -0.10208,  1.10813, -0.00605,
    -0.00327, -0.07276,  1.07602);

float3 rrtAndOdtFit(float3 v) {
    let a = v * (v + 0.0245786) - 0.000090537;
    let b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}

// The transform composed: display-linear Rec.709 in [0, 1] from linear
// ACEScg radiance.
float3 acesDisplay(float3 acescg) {
    let color = mul(ACES_INPUT, mul(SRGB_FROM_ACESCG, acescg));
    return saturate(mul(ACES_OUTPUT, rrtAndOdtFit(color)));
}

// The sRGB transfer function (IEC 61966-2-1), applied here in the shader so
// the display buffer holds final bytes — the presenter's blit into an sRGB
// image then converts losslessly instead of quantizing linear values.
float srgbEncode(float channel) {
    return channel <= 0.0031308 ? 12.92 * channel
                                : 1.055 * pow(channel, 1.0 / 2.4) - 0.055;
}

uint quantize(float channel) {
    return uint(round(saturate(channel) * 255.0));
}

// One texel of R8G8B8A8: R in the low byte, matching the format's byte
// order through the little-endian buffer.
uint packRgba8(float3 rgb, float alpha) {
    return quantize(rgb.r)
         | quantize(rgb.g) << 8
         | quantize(rgb.b) << 16
         | quantize(alpha) << 24;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void tonemap(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) {
        return;
    }
    uint pixel = tid.y * params.width + tid.x;

    let color = acesDisplay(params.average[pixel].rgb * params.exposureScale);
    let encoded = float3(srgbEncode(color.r), srgbEncode(color.g), srgbEncode(color.b));
    params.display[pixel] = packRgba8(encoded, 1.0);
}
