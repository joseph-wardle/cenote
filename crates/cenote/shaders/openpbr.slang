// The OpenPBR surface model, M1 subset: an EON diffuse base under a
// dielectric GGX specular layer, blended with a conductor GGX by metalness,
// plus emission. Lobes are plain eval/sample/pdf functions combined by the
// one-sample mixture evaluator at the bottom — no interfaces, no dynamic
// dispatch.
//
// The specular lobes are single-scatter GGX (alpha = roughness², separable
// Smith) with Turquin-style multiple-scattering compensation ("Practical
// multiple scattering compensation for microfacet models", Turquin 2019):
// the lobe is rescaled by the energy its single-scatter form loses, read
// from analytic rational fits of the directional albedo in the manner of
// Sforza & Pellacini ("Enforcing Energy Preservation in Microfacet
// Models", STAG 2022). Both fits were regenerated for this kernel's exact
// integrand (QMC-tabulated, then fit with their rational form), so an
// albedo-1 surface reflects 1 to within the fit residual (~0.6%) — which
// the furnace matrix in render.rs pins.
//
// Layering follows OpenPBR's albedo-scaling approximation: the diffuse
// base is scaled by the energy the specular layer above it lets through,
// 1 - specular_weight·E_glossy(roughness, μo).
//
// BSDF convention: wi (scattered) and wo (toward the viewer) are unit
// directions pointing away from the surface, in a local frame with the
// normal along +z, so wo.z > 0.

public static const float PI = 3.14159265358979;

// OpenPBR's default specular IOR. M1 fixes it — the glossy albedo fit
// below is specialized to its 0.04 normal reflectivity.
static const float SPECULAR_IOR = 1.5;

// The albedo fits were tabulated with grazing angles and roughness clamped
// away from their degenerate limits (a delta lobe has no finite D); the
// kernel clamps identically so it never evaluates a fit outside its
// trained domain.
static const float MIN_SPECULAR_ROUGHNESS = 0.035;
static const float MIN_MU = 0.02;

// One instance's surface parameters, named after OpenPBR attributes.
// Mirrors `Material` in `material.rs` field for field.
public struct Material {
    public float3 baseColor;   // diffuse albedo / conductor F0, ACEScg
    public float baseRoughness; // Oren-Nayar roughness in [0, 1]; 0 = Lambert
    public float3 emission;    // radiance, ACEScg — nonzero marks a light
    public float metalness;    // conductor blend: 0 dielectric, 1 metal
    public float specularWeight; // dielectric specular layer: 0 none, 1 full
    public float specularRoughness; // GGX roughness of both specular lobes
    float pad0;
    float pad1;
};

// -- EON diffuse ---------------------------------------------------------

// The Fujii Oren-Nayar model's A and B coefficients depend on roughness r
// as A = 1/(1 + FON_A·r), B = r·A; FON_B is the constant in its exact
// cosine-weighted average albedo, avg = A·(1 + FON_B·r).
static const float FON_A = 0.5 - 2.0 / (3.0 * PI);
static const float FON_B = 2.0 / 3.0 - 28.0 / (15.0 * PI);

// Directional albedo of the single-scatter lobe: the paper's rational fit,
// within 3.5e-4 of the exact form — which has a removable singularity at
// grazing that the fit sidesteps.
float fonAlbedo(float mu, float r) {
    let g = 1.0 - mu;
    let gOverPi = g * (0.0571085289 + g * (0.491881867
                + g * (-0.332181442 + g * 0.0714429953)));
    return (1.0 + r * gOverPi) / (1.0 + FON_A * r);
}

// EON: the Fujii Oren-Nayar single-scatter lobe plus an analytic
// multiple-scattering lobe that restores exactly the energy single
// scattering loses — an albedo-1 surface reflects 1 at every angle
// ("EON: A Practical Energy-Preserving Rough Diffuse BRDF", Portsmouth,
// Kutz & Hill, JCGT 2025).
public float3 eonBrdf(float3 rho, float r, float3 wi, float3 wo) {
    let s = dot(wi, wo) - wi.z * wo.z;
    let sOverT = s > 0.0 ? s / max(wi.z, wo.z) : s;
    let A = 1.0 / (1.0 + FON_A * r);
    let singleScatter = (rho / PI) * A * (1.0 + r * sOverT);

    let albedoI = fonAlbedo(wi.z, r);
    let albedoO = fonAlbedo(wo.z, r);
    let albedoAvg = A * (1.0 + FON_B * r);
    let rhoMulti = (rho * rho) * albedoAvg / (float3(1.0) - rho * (1.0 - albedoAvg));
    let eps = 1e-7;
    let multiScatter = (rhoMulti / PI) * max(eps, 1.0 - albedoO)
                     * max(eps, 1.0 - albedoI) / max(eps, 1.0 - albedoAvg);
    return singleScatter + multiScatter;
}

// Cosine-weighted hemisphere sample of u in [0,1)²: xyz the direction,
// w its pdf cos(θ)/π — never zero, since u.x < 1.
public float4 sampleCosineHemisphere(float2 u) {
    let r = sqrt(u.x);
    let phi = 2.0 * PI * u.y;
    let z = sqrt(1.0 - u.x);
    return float4(r * cos(phi), r * sin(phi), z, z / PI);
}

// -- GGX (alpha = roughness², separable Smith) ---------------------------

float ggxD(float alpha, float hz) {
    let a2 = alpha * alpha;
    let denom = hz * hz * (a2 - 1.0) + 1.0;
    return hz > 0.0 ? a2 / (PI * denom * denom) : 0.0;
}

// Smith's G1 for GGX. The full masking-shadowing term is the separable
// product G1(wo)·G1(wi) — the form the albedo tables behind the fits below
// were integrated with; swapping in height-correlated Smith would need
// those fits regenerated together with it.
float smithG1(float alpha, float vz) {
    let a2 = alpha * alpha;
    return 2.0 * abs(vz) / (abs(vz) + sqrt(vz * vz * (1.0 - a2) + a2));
}

// Sample the GGX distribution of visible normals: a half vector given the
// view direction. Dupuy & Benyoub's spherical-caps form ("Sampling
// Visible GGX Normals with Spherical Caps", HPG 2023), ported from the
// paper's listings 1 and 3.
float3 sampleVndf(float3 wo, float alpha, float2 u) {
    // warp to the hemisphere configuration
    let woStd = normalize(float3(wo.xy * alpha, wo.z));
    // sample a spherical cap in (-woStd.z, 1]
    let phi = 2.0 * PI * u.x;
    let z = fma(1.0 - u.y, 1.0 + woStd.z, -woStd.z);
    let sinTheta = sqrt(clamp(1.0 - z * z, 0.0, 1.0));
    let c = float3(sinTheta * cos(phi), sinTheta * sin(phi), z);
    // the halfway direction is the sampled visible normal
    let hStd = c + woStd;
    // warp back to the ellipsoid configuration
    return normalize(float3(hStd.xy * alpha, hStd.z));
}

// The solid-angle pdf of the direction reflected from a VNDF-sampled half
// vector: Dvis(h)/(4 h·wo) with the h·wo factors cancelled.
float vndfReflectPdf(float alpha, float3 wo, float3 h) {
    return smithG1(alpha, wo.z) * ggxD(alpha, h.z) / (4.0 * wo.z);
}

// -- Fresnel --------------------------------------------------------------

// Schlick's approximation — the conductor Fresnel, with F0 = base color.
float3 fresnelSchlick(float3 f0, float cosine) {
    let m = 1.0 - abs(cosine);
    let m2 = m * m;
    return f0 + (1.0 - f0) * m2 * m2 * m;
}

// Exact unpolarized dielectric Fresnel — what the glossy albedo fit was
// integrated with (Schlick undershoots near grazing for IOR 1.5).
float fresnelDielectric(float eta, float cosine) {
    let mu = abs(cosine);
    let sin2 = 1.0 - mu * mu;
    let cos2t = 1.0 - sin2 / (eta * eta);
    if (cos2t < 0.0) {
        return 1.0; // total internal reflection
    }
    let t0 = sqrt(cos2t);
    let t1 = eta * t0;
    let t2 = eta * mu;
    let rs = (mu - t1) / (mu + t1);
    let rp = (t0 - t2) / (t0 + t2);
    return (rs * rs + rp * rp) / 2.0;
}

// -- Directional-albedo fits ----------------------------------------------
// Degree-3/4 rationals over PolynomialFeatures-ordered monomials, the
// Sforza-Pellacini form, regenerated for this kernel's exact integrand
// (QMC over the sampler above; coefficients in f32-safe ranges). The
// numerator and 1 + denominator share the same feature order minus the
// constant term.

// E_ss(roughness, μ): single-scatter GGX albedo with F = 1 — what Turquin
// compensation divides by. Degree-4 rational in (roughness, μ), fit with
// relative-error weighting because the furnace's error is this fit's
// *relative* error; max 1.3%, well under 1% away from grazing.
float conductorAlbedo(float rough, float mu) {
    let r = rough;
    let m = clamp(mu, MIN_MU, 1.0);
    float f[15];
    f[0] = 1.0;
    f[1] = r; f[2] = m;
    f[3] = r * r; f[4] = r * m; f[5] = m * m;
    f[6] = f[3] * r; f[7] = f[3] * m; f[8] = r * f[5]; f[9] = f[5] * m;
    f[10] = f[3] * f[3]; f[11] = f[6] * m; f[12] = f[3] * f[5];
    f[13] = r * f[9]; f[14] = f[5] * f[5];
    const float num[15] = {
        0.991224051, -16.4138546, 13.8225307, 95.7334213, -130.445663,
        91.3484192, -124.505646, 195.404709, -133.961868, -21.9732971,
        51.1392593, -94.6726608, 73.0841827, -3.473351, 8.35645866,
    };
    const float den[14] = {
        -16.6203041, 13.7559481, 99.0378036, -131.199844, 91.3512268,
        -128.797775, 199.281601, -135.754532, -21.2633991, 56.8038139,
        -98.9809723, 82.9306183, -5.07848787, 7.92696524,
    };
    float top = 0.0;
    float bottom = 1.0;
    for (uint i = 0; i < 15; i++) {
        top += num[i] * f[i];
    }
    for (uint i = 0; i < 14; i++) {
        bottom += den[i] * f[i + 1];
    }
    return clamp(top / bottom, 1e-4, 1.0);
}

// E_glossy(roughness, μ): albedo of the *compensated* dielectric specular
// lobe at IOR 1.5 — both the layering scale on the diffuse base and the
// specular half of the lobe-selection weight. Degree-3 rational in
// (roughness, √μ); √μ tames the Fresnel rise at grazing. Max error 1.4%.
float glossyAlbedo(float rough, float mu) {
    let r = rough;
    let m = sqrt(clamp(mu, MIN_MU, 1.0));
    float f[10];
    f[0] = 1.0;
    f[1] = r; f[2] = m;
    f[3] = r * r; f[4] = r * m; f[5] = m * m;
    f[6] = f[3] * r; f[7] = f[3] * m; f[8] = r * f[5]; f[9] = f[5] * m;
    const float num[10] = {
        0.897914171, 0.471493632, -1.04169869, 0.580212533, -1.15793955,
        -0.476999015, 1.00325811, -1.24675167, 0.806432366, 0.795318246,
    };
    const float den[9] = {
        0.456833839, -1.65103114, 9.70270634, -2.89981961, 4.59609556,
        25.6256561, -28.4835682, 6.16417933, 0.348338723,
    };
    float top = 0.0;
    float bottom = 1.0;
    for (uint i = 0; i < 10; i++) {
        top += num[i] * f[i];
    }
    for (uint i = 0; i < 9; i++) {
        bottom += den[i] * f[i + 1];
    }
    return clamp(top / bottom, 0.0, 1.0);
}

// -- The mixture evaluator --------------------------------------------------
// One BSDF, three lobes, two sampling techniques (the specular lobes share
// the VNDF; the diffuse base is cosine-sampled). Selection probability of
// the specular technique — metalness plus the dielectric layer's own
// albedo — makes the technique weights sum to 1 by construction.

float clampedSpecularRoughness(Material material) {
    return clamp(material.specularRoughness, MIN_SPECULAR_ROUGHNESS, 1.0);
}

float specularSelectionWeight(Material material, float3 wo) {
    let rough = clampedSpecularRoughness(material);
    let dielectric = material.specularWeight * glossyAlbedo(rough, wo.z);
    return lerp(dielectric, 1.0, material.metalness);
}

// f(wi, wo) — the full BSDF, no cosine factor.
public float3 evalBsdf(Material material, float3 wi, float3 wo) {
    if (wi.z <= 0.0 || wo.z <= 0.0) {
        return float3(0.0);
    }
    let rough = clampedSpecularRoughness(material);
    let alpha = rough * rough;
    let h = normalize(wi + wo);
    let core = ggxD(alpha, h.z) * smithG1(alpha, wo.z) * smithG1(alpha, wi.z)
             / (4.0 * wo.z * wi.z);
    let cosVH = dot(h, wi);
    // Turquin: add back the multiple-scattering energy 1/E_ss recovers,
    // tinted F0 per extra bounce for the conductor.
    let E = conductorAlbedo(rough, wo.z);
    let conductor = fresnelSchlick(material.baseColor, cosVH) * core
                  * (1.0 + material.baseColor * (1.0 - E) / E);
    let dielectricSpec = material.specularWeight / E
                       * fresnelDielectric(SPECULAR_IOR, cosVH) * core;
    // Albedo-scaling layering: the base sees what the layer lets through.
    let layered = 1.0 - material.specularWeight * glossyAlbedo(rough, wo.z);
    let diffuse = layered * eonBrdf(material.baseColor, material.baseRoughness, wi, wo);
    return lerp(float3(dielectricSpec) + diffuse, conductor, material.metalness);
}

// The mixture's solid-angle pdf of wi — the divisor for sampled scatters,
// and MIS's competitor for light samples.
public float bsdfPdf(Material material, float3 wi, float3 wo) {
    if (wi.z <= 0.0 || wo.z <= 0.0) {
        return 0.0;
    }
    let rough = clampedSpecularRoughness(material);
    let wSpec = specularSelectionWeight(material, wo);
    let h = normalize(wi + wo);
    return wSpec * vndfReflectPdf(rough * rough, wo, h)
         + (1.0 - wSpec) * wi.z / PI;
}

// Draw a scattered direction from the mixture: uLobe picks the technique,
// u places the sample. xyz = wi (below-horizon results carry f = 0 and
// terminate the path); w = the mixture pdf of that wi.
public float4 sampleBsdf(Material material, float3 wo, float uLobe, float2 u) {
    let rough = clampedSpecularRoughness(material);
    let wSpec = specularSelectionWeight(material, wo);
    float3 wi;
    if (uLobe < wSpec) {
        let h = sampleVndf(wo, rough * rough, u);
        wi = 2.0 * dot(wo, h) * h - wo;
    } else {
        wi = sampleCosineHemisphere(u).xyz;
    }
    return float4(wi, bsdfPdf(material, wi, wo));
}

// An orthonormal frame around a unit normal, branchless (Duff et al.,
// "Building an Orthonormal Basis, Revisited", JCGT 2017).
public struct Frame {
    float3 t;
    float3 b;
    float3 n;

    public __init(float3 n) {
        this.n = n;
        let s = n.z >= 0.0 ? 1.0 : -1.0;
        let a = -1.0 / (s + n.z);
        let m = n.x * n.y * a;
        this.t = float3(1.0 + s * n.x * n.x * a, s * m, -s * n.x);
        this.b = float3(m, s + n.y * n.y * a, -n.y);
    }

    public float3 toLocal(float3 v) {
        return float3(dot(v, t), dot(v, b), dot(v, n));
    }

    public float3 toWorld(float3 v) {
        return v.x * t + v.y * b + v.z * n;
    }
};
