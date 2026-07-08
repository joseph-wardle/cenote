// The environment light: an equirect radiance image, importance-sampled
// through prep-built CDF tables. Radiance lives in the binding model's one
// sampled texture (bilinear; wrapping in azimuth, clamped at the poles);
// the tables are plain buffers built by `environment.rs`, which documents
// the two prep-time choices the kernel leans on — sampling weights dilated
// to cover the bilinear footprint, and per-texel pdfs stored outright
// rather than recovered as f32 CDF differences.
//
// `sample()` and `pdf()` are deliberately separate entry points: sampling
// serves next-event estimation now, and the standalone pdf query is what
// BSDF-strategy MIS weights need — and what every ReSTIR target-function
// evaluation will ask for later.
//
// Directions map to the image with +Y up: v = θ/π from the zenith, u wraps
// the azimuth with the image's horizontal center facing −Z (the way the
// crate's cameras conventionally look). Poles are the equirect singularity:
// a direction with sin θ = 0 has no well-defined pdf, so both entry points
// report zero there and callers treat the sample as carrying nothing —
// next-event skips it, and an escaped ray's MIS weight becomes 1.

// The environment texture, next to the TLAS in the one descriptor set.
[vk::binding(1, 0)]
Sampler2D<float4> environmentTexture;

static const float PI = 3.14159265358979323846;

// A next-event candidate direction toward the environment.
public struct EnvironmentSample {
    public float3 direction; // unit, world space
    public float3 radiance;  // ACEScg, read from the texture
    public float pdf;        // solid-angle pdf of `direction`; 0 = invalid
};

// One row or column draw from a piecewise-constant CDF: the largest index
// with cdf[index] <= u, plus the continuous position within [0, 1).
struct CdfSlot {
    uint index;
    float position;
};

CdfSlot sampleCdf(float* cdf, uint count, float u) {
    uint lo = 0;
    uint hi = count;
    while (lo + 1 < hi) {
        let mid = (lo + hi) / 2;
        if (cdf[mid] <= u) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let width = cdf[lo + 1] - cdf[lo];
    let frac = width > 0.0 ? (u - cdf[lo]) / width : 0.0;
    CdfSlot slot;
    slot.index = lo;
    slot.position = (float(lo) + frac) / float(count);
    return slot;
}

// The environment as the scene table carries it. Mirrors the table layout
// `environment.rs` builds; `selectProb` is scene prep's power-proportional
// probability that next-event estimation samples the environment rather
// than a quad light.
public struct Environment {
    float* marginalCdf;    // height + 1 entries: CDF over rows
    float* conditionalCdf; // height runs of width + 1: per-row CDF
    float* pdfs;           // height × width: joint density over (u, v)
    uint width;
    uint height;
    public float selectProb;
    uint pad;

    float2 uvFromDirection(float3 d) {
        let u = 0.5 + atan2(d.x, -d.z) / (2.0 * PI);
        let v = acos(clamp(d.y, -1.0, 1.0)) / PI;
        return float2(u, v);
    }

    float3 directionFromUv(float2 uv) {
        let phi = 2.0 * PI * (uv.x - 0.5);
        let theta = PI * uv.y;
        let sinTheta = sin(theta);
        return float3(sinTheta * sin(phi), cos(theta), -sinTheta * cos(phi));
    }

    // The radiance arriving from direction `d` — what an escaped ray sees.
    public float3 radiance(float3 d) {
        return environmentTexture.SampleLevel(uvFromDirection(d), 0.0).xyz;
    }

    // Draw a direction ∝ the sampling tables. The returned pdf is this
    // light's own; callers fold in `selectProb` where selection competed.
    public EnvironmentSample sample(float2 u) {
        let row = sampleCdf(marginalCdf, height, u.x);
        let col = sampleCdf(conditionalCdf + row.index * (width + 1), width, u.y);
        let uv = float2(col.position, row.position);
        let sinTheta = sin(PI * uv.y);
        EnvironmentSample s;
        s.direction = directionFromUv(uv);
        s.radiance = environmentTexture.SampleLevel(uv, 0.0).xyz;
        s.pdf = sinTheta > 0.0
              ? pdfs[row.index * width + col.index] / (2.0 * PI * PI * sinTheta)
              : 0.0;
        return s;
    }

    // The solid-angle pdf `sample` would report for direction `d` — the
    // MIS competitor for a BSDF-sampled ray that escaped to the sky.
    public float pdf(float3 d) {
        let sinTheta = sqrt(max(0.0, 1.0 - d.y * d.y));
        if (sinTheta <= 0.0) {
            return 0.0;
        }
        let uv = uvFromDirection(d);
        let col = min(uint(uv.x * float(width)), width - 1);
        let row = min(uint(uv.y * float(height)), height - 1);
        return pdfs[row * width + col] / (2.0 * PI * PI * sinTheta);
    }
};
