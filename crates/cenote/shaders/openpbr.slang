// The OpenPBR surface model, M2 closure cut: an EON diffuse base under a
// dielectric GGX interface at a variable IOR — reflection only when opaque,
// coupled reflection + refraction as `transmissionWeight` rises — blended
// toward a conductor by `metalness`, under an optional clear coat and a
// fuzz (sheen) layer. Lobes are plain eval/sample/pdf functions over a
// per-hit `Closure` — parameters resolved once per vertex against the
// view direction — combined by one-sample MIS at the bottom: pick one
// sampling technique by its albedo-estimate weight (rescaling the used
// random number to keep its stratification), evaluate every lobe, and
// weight by the balance-heuristic mixture pdf Σ(pdfᵢ·wᵢ)/Σwᵢ.
//
// Energy preservation follows Cycles' shipped mechanism: each specular
// lobe is single-scatter GGX (alpha = roughness², separable Smith) times
// a multiplicative multiple-scattering scale `1 + Fms·(1−E)/E`, with E and
// E_avg read from Fresnel-free tables baked in `tables.rs` against this
// exact integrand, and Fresnel entering analytically through the average
// single-scattering Fresnel (Kulla & Conty, "Revisiting Physically Based
// Shading at Imageworks"): Fms = Fss·E_avg/(1 − Fss·(1 − E_avg)). The
// coupled glass lobe reads its E from 3D tables (roughness × cosθ × an
// IOR remap, a separate branch for η < 1); the sampling pdf is never
// scaled, so MIS stays consistent. Layering is OpenPBR's albedo-scaling
// reduction: what sits under an interface is scaled by one minus the
// interface's *tabulated* directional albedo, so a white stack reflects
// exactly 1 — the property the furnace matrix pins.
//
// Fuzz is the Zeltner–Burley–Chiang LTC sheen ("Practical
// Multiple-Scattering Sheen Using Linearly Transformed Cosines", 2022):
// the vendored 32×32 fit *is* the BSDF, sampled perfectly by pushing a
// cosine sample through the fitted transform.
//
// BSDF convention: wi (scattered) and wo (toward the viewer) are unit
// directions in a local frame with the normal along +z; wo.z > 0 always
// (surfaces shade two-sided), while wi.z < 0 is transmission. `evalBsdf`
// returns f without the |wi.z| cosine. Refraction uses the adjoint
// (camera-path) form of Walter's BTDF — no solid-angle-compression
// factor — so a VNDF sample's weight is exactly (1−F)·G1(wi), the
// quantity the glass energy tables integrate; bidirectional transport
// would revisit this.

public static const float PI = 3.14159265358979;

// The kernel keeps a roughness floor: a delta lobe has no finite D, and
// the energy tables were baked over the same clamped domain. True mirrors
// are a later, delta-lobe feature.
static const float MIN_SPECULAR_ROUGHNESS = 0.035;

// One instance's surface parameters, named after OpenPBR attributes.
// Mirrors `Material` in `material.rs` field for field.
public struct Material {
    public float3 baseColor;   // diffuse albedo / conductor F0, ACEScg
    public float baseRoughness; // Oren-Nayar roughness in [0, 1]; 0 = Lambert
    public float3 emission;    // radiance, ACEScg — nonzero marks a light
    public float metalness;    // conductor blend: 0 dielectric, 1 metal
    public float specularWeight; // dielectric reflectivity: 0 none, 1 full Fresnel
    public float specularRoughness; // GGX roughness of the base's specular lobes
    public float specularIor;  // dielectric IOR, reflection and refraction
    public float transmissionWeight; // dielectric base: 0 opaque, 1 glass
    public float3 transmissionColor; // Beer–Lambert target color, ACEScg
    public float transmissionDepth;  // meters to reach it; 0 tints the interface
    public float3 coatColor;   // tint the coat multiplies onto its base
    public float coatWeight;   // clear coat: 0 none, 1 full
    public float coatRoughness; // GGX roughness of the coat lobe
    public float coatIor;      // coat IOR
    public float coatDarkening; // 0 disables the physical base darkening
    public float fuzzWeight;   // sheen layer: 0 none, 1 full
    public float3 fuzzColor;   // sheen tint, ACEScg
    public float fuzzRoughness; // sheen fiber roughness
    public float opacity;      // resolved in the intersect and shadow stages
    public uint thinWalled;    // bool: no interior — transmission passes straight
    // Bindless-table indices, TEXTURE_NONE (textures.slang) = constant
    // everywhere. resolveMaterial there replaces the constants per hit.
    public uint baseColorTexture;
    public uint specularRoughnessTexture;
    public uint metalnessTexture;
    public uint emissionTexture; // multiplies onto `emission`
    public uint opacityTexture;  // multiplies onto `opacity`, in traversal
    public uint normalTexture;   // tangent-space, BC5 x/y
};

// -- Lookup tables ---------------------------------------------------------
// One float buffer, uploaded by the engine from `tables.rs` — offsets and
// axis conventions must match its `TABLE_*` constants entry for entry.
// Grid point k of an n-wide axis sits at k/(n−1); reads interpolate with
// clamped multilinear filtering, the first-named axis fastest.

static const uint TABLE_GGX_E = 0;               // 32×32 (rough, µ)
static const uint TABLE_GGX_E_AVG = 1024;        // 32 (rough)
static const uint TABLE_GLASS_E = 1056;          // 16³ (rough, µ, z)
static const uint TABLE_GLASS_E_AVG = 5152;      // 16² (rough, z)
static const uint TABLE_GLASS_INV_E = 5408;      // 16³, η < 1
static const uint TABLE_GLASS_INV_E_AVG = 9504;  // 16², η < 1
static const uint TABLE_DIELECTRIC_E = 9760;     // 16³ (rough, µ, z)
static const uint TABLE_SHEEN_A = 13856;         // 32×32 (µ, rough)
static const uint TABLE_SHEEN_B = 14880;
static const uint TABLE_SHEEN_R = 15904;

float tableRead1D(float* tables, uint offset, uint size, float x) {
    let fx = saturate(x) * float(size - 1);
    let i = min(uint(fx), size - 1);
    let j = min(i + 1, size - 1);
    return lerp(tables[offset + i], tables[offset + j], fx - float(i));
}

float tableRead2D(float* tables, uint offset, uint2 size, float x, float y) {
    let fy = saturate(y) * float(size.y - 1);
    let i = min(uint(fy), size.y - 1);
    let j = min(i + 1, size.y - 1);
    return lerp(tableRead1D(tables, offset + size.x * i, size.x, x),
                tableRead1D(tables, offset + size.x * j, size.x, x),
                fy - float(i));
}

float tableRead3D(float* tables, uint offset, uint3 size, float x, float y, float z) {
    let fz = saturate(z) * float(size.z - 1);
    let i = min(uint(fz), size.z - 1);
    let j = min(i + 1, size.z - 1);
    let plane = size.x * size.y;
    return lerp(tableRead2D(tables, offset + plane * i, size.xy, x, y),
                tableRead2D(tables, offset + plane * j, size.xy, x, y),
                fz - float(i));
}

// The glass tables' IOR axis: z = √((η−1)/(η+1)) for η ≥ 1 — callers
// invert an η < 1 first and switch to the inverted-branch tables.
float iorRemap(float eta) {
    return sqrt(abs(eta - 1.0) / (eta + 1.0));
}

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
// product G1(wo)·G1(wi) — the form the energy tables in `tables.rs` were
// integrated with; swapping in height-correlated Smith would need those
// tables rebaked together with it.
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

// Its cosine-weighted hemisphere average, exactly F0 + (1−F0)/21.
float3 fresnelSchlickAverage(float3 f0) {
    return lerp(f0, float3(1.0), 1.0 / 21.0);
}

// Exact unpolarized dielectric Fresnel; eta is the relative IOR of the
// far side, so total internal reflection (from the dense side, eta < 1)
// reports exactly 1.
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

// The cosine-weighted average dielectric Fresnel — Kulla & Conty's
// rational fit, both branches. Feeds the multiple-scattering term and the
// coat's internal-reflection coefficient.
float fresnelDielectricAverage(float eta) {
    if (eta < 1.0) {
        return 0.997118 + eta * (0.1014 - eta * (0.965241 + eta * 0.130607));
    }
    return (eta - 1.0) / (4.08567 + 1.00071 * eta);
}

// Kulla & Conty's multiple-scattering Fresnel: the fraction of the energy
// re-emerging from inter-microfacet bounces, per average-Fresnel Fss —
// one Fss cancelled because the scale below multiplies a single-scatter
// lobe already carrying one bounce of Fresnel.
float3 fresnelMultiScatter(float3 fss, float eAvg) {
    return fss * eAvg / (float3(1.0) - fss * (1.0 - eAvg));
}

// Refract wo about half-vector h at relative IOR eta; w = 1 when valid,
// 0 past the critical angle.
float4 refractDirection(float3 wo, float3 h, float eta) {
    let c = dot(wo, h);
    let cos2t = 1.0 - (1.0 - c * c) / (eta * eta);
    if (cos2t <= 0.0) {
        return float4(0.0);
    }
    return float4((c / eta - sqrt(cos2t)) * h - wo / eta, 1.0);
}

// -- The resolved closure ---------------------------------------------------
// Everything the mixture needs, computed once per hit from the material,
// the view direction, and the tables: remapped roughness and IOR, each
// layer's attenuation of what sits below it (view-side albedo scaling,
// OpenPBR's reduction), the multiple-scattering scales, and the
// one-sample-MIS technique weights.

// Which technique scattered — the path-state lobe tag. Feeds the AOV
// specular pass-through ramp and M3's GRIS replay; until those land it
// is a free debugging channel.
public static const uint LOBE_DIFFUSE = 0;
public static const uint LOBE_SPECULAR = 1;     // any base reflection
public static const uint LOBE_TRANSMISSION = 2; // through the surface
public static const uint LOBE_COAT = 3;
public static const uint LOBE_FUZZ = 4;

public struct Closure {
    Material m;
    float* tables;
    float alpha;      // base specular alpha, coat-roughened, floor-clamped
    float coatAlpha;
    float eta;        // relative IOR at this hit (< 1 leaving the interior)
    bool exiting;     // the hit leaves the current medium: bare interface
    // Whether the base carries a specular interface at all. A material
    // with no metalness, no transmission, and specular_weight exactly 0
    // drops the interface lobes *entirely* — matching their zero energy
    // with a zero technique weight, so a pure-diffuse surface keeps the
    // single-technique per-sample exactness the furnace tests pin.
    bool hasSpecular;
    // Fuzz, resolved at wo: LTC transform entries, tabulated albedo R
    // (zero when absent or the fit is degenerate there), and 1 − F·R —
    // what the fuzz layer lets through.
    float fuzzA;
    float fuzzB;
    float fuzzR;
    float fuzzAtten;
    float coatE;       // E[f_coat](µo), from the dielectric albedo table
    float coatScale;   // the coat lobe's multiple-scattering scale
    float3 baseWeight; // fuzz and coat attenuation over the base substrate
    float specE;       // E[f_R_specular](µo, η) — the dielectric layering albedo
    float3 conductorScale;  // multiple-scattering scales per lobe family
    float dielectricScale;
    float3 glassScale;
    float3 glassTint;  // interface tint when transmissionDepth = 0
    // Thin-walled dielectric: pre-integrated two-interface reflectance and
    // transmittance (geometric series over internal bounces), the
    // transmission lobe's own roughened alpha, and their scales.
    float3 thinR;
    float3 thinT;
    float thinAlpha;
    float thinScaleR;
    float thinScaleT;
    float thinTShare;  // probability the spec technique picks transmission
    // One-sample MIS technique weights (unnormalized) and their sum.
    float wFuzz;
    float wCoat;
    float wSpec;
    float wDiffuse;
    float wTotal;
};

float average(float3 v) {
    return (v.x + v.y + v.z) / 3.0;
}

// The multiple-scattering scale on a single-scatter reflection lobe:
// 1 + Fms·(1−E)/E, with E/E_avg from the Fresnel-free tables.
float3 reflectionScale(float* tables, float rough, float mu, float3 fss) {
    let e = max(1e-4, tableRead2D(tables, TABLE_GGX_E, uint2(32, 32), rough, mu));
    let eAvg = tableRead1D(tables, TABLE_GGX_E_AVG, 32, rough);
    return 1.0 + fresnelMultiScatter(fss, eAvg) * ((1.0 - e) / e);
}

// Directional albedo of the compensated dielectric reflection lobe at
// η ≥ 1 from its table; below 1 (rare: authored IOR under 1) the smooth-
// surface Fresnel stands in, Cycles' own fallback.
float dielectricAlbedo(float* tables, float rough, float mu, float eta) {
    if (eta < 1.0) {
        return fresnelDielectric(eta, mu);
    }
    return tableRead3D(tables, TABLE_DIELECTRIC_E, uint3(16, 16, 16),
                       rough, mu, iorRemap(eta));
}

// OpenPBR's coat darkening: internal reflection off the coat's underside
// returns K of the base's exitance for another pass, darkening and
// saturating it by Δ = (1−K)/(1−E_b·K) — exactly 1 for a white base, so
// the furnace never sees it. K blends the smooth-base coefficient (the
// coat Fresnel at the view angle) toward the rough-base one (its
// hemispheric average) by an estimate of how rough the base looks.
float3 coatDarkening(Material m, float rough, float mu, float baseFresnel) {
    let fssCoat = fresnelDielectricAverage(m.coatIor);
    let kRough = 1.0 - (1.0 - fssCoat) / (m.coatIor * m.coatIor);
    let kSmooth = fresnelDielectric(m.coatIor, mu);
    let roughDiffuse = lerp(1.0, rough, saturate(m.specularWeight * baseFresnel));
    let baseRough = lerp(roughDiffuse, rough, m.metalness);
    let k = lerp(kSmooth, kRough, baseRough);
    // The base's normal-incidence albedo: its color for diffuse and
    // conductor, ~1 for glass (nothing absorbed at the interface).
    let baseAlbedo = lerp(m.baseColor, float3(1.0),
                          (1.0 - m.metalness) * m.transmissionWeight);
    let darkened = (1.0 - k) / max(float3(1e-4), 1.0 - baseAlbedo * k);
    return lerp(float3(1.0), darkened, m.coatWeight * m.coatDarkening);
}

// Resolve a material against the view direction. `exiting` marks a hit
// that leaves the interior the path is traveling through (the shading
// kernel knows from its medium slot): the closure degrades to the bare
// dielectric interface at the inverted IOR — coat, fuzz, and base live
// on the outside.
public Closure resolveClosure(float* tables, Material m, float3 wo, bool exiting) {
    Closure c;
    c.m = m;
    c.tables = tables;
    c.exiting = exiting;
    c.hasSpecular = exiting || m.metalness > 0.0 || m.transmissionWeight > 0.0
                 || m.specularWeight > 0.0;
    let mu = wo.z;

    // Coat roughening: light reaching the base crossed the coat's rough
    // boundary twice, adding twice its slope variance (the spec's
    // min-clamped fourth-power remap).
    var rough = clamp(m.specularRoughness, MIN_SPECULAR_ROUGHNESS, 1.0);
    let coatRough = clamp(m.coatRoughness, MIN_SPECULAR_ROUGHNESS, 1.0);
    if (m.coatWeight > 0.0 && !exiting) {
        let variance = min(1.0, rough * rough * rough * rough
                              + 2.0 * coatRough * coatRough * coatRough * coatRough);
        rough = lerp(rough, sqrt(sqrt(variance)), m.coatWeight);
    }
    c.alpha = rough * rough;
    c.coatAlpha = coatRough * coatRough;

    // The base IOR: refracted by the coat above it (with the spec's
    // total-internal-reflection fix — the boundary ratio inverted to stay
    // above 1), then remapped by specular_weight, which scales F0 and
    // keeps refraction consistent with the modulated ratio.
    var etaS = max(m.specularIor, 1e-4);
    if (m.coatWeight > 0.0 && !exiting) {
        let etaBc = m.coatIor > etaS ? m.coatIor / etaS : etaS / m.coatIor;
        etaS = lerp(etaS, etaBc, m.coatWeight);
    }
    let f0 = (etaS - 1.0) * (etaS - 1.0) / ((etaS + 1.0) * (etaS + 1.0));
    let weight = clamp(m.specularWeight, 0.0, f0 > 1e-6 ? 1.0 / f0 : 0.0);
    let epsilon = sign(etaS - 1.0) * sqrt(weight * f0);
    var eta = (1.0 + epsilon) / max(1e-3, 1.0 - epsilon);
    // η = 1 is a degenerate interface (no reflection, an undefined
    // refraction half-vector); hold it just off unity instead.
    if (abs(eta - 1.0) < 1e-3) {
        eta = eta < 1.0 ? 1.0 - 1e-3 : 1.0 + 1e-3;
    }
    c.eta = exiting ? 1.0 / eta : eta;

    // Fuzz at wo. The published fit skips cells whose reflectance is
    // ~zero (A = 0 there), so a degenerate read means "no fuzz lobe
    // here" — its energy is genuinely negligible.
    c.fuzzA = 0.0;
    c.fuzzB = 0.0;
    c.fuzzR = 0.0;
    c.fuzzAtten = 1.0;
    if (m.fuzzWeight > 0.0 && !exiting) {
        let fuzzRough = clamp(m.fuzzRoughness, 1e-3, 1.0);
        let a = tableRead2D(tables, TABLE_SHEEN_A, uint2(32, 32), mu, fuzzRough);
        let r = tableRead2D(tables, TABLE_SHEEN_R, uint2(32, 32), mu, fuzzRough);
        if (abs(a) >= 1e-5 && r >= 1e-5) {
            c.fuzzA = a;
            c.fuzzB = tableRead2D(tables, TABLE_SHEEN_B, uint2(32, 32), mu, fuzzRough);
            c.fuzzR = r;
            c.fuzzAtten = 1.0 - m.fuzzWeight * r;
        }
    }

    // Coat at wo: its tabulated albedo is both its own lobe's share and
    // (with the coat tint and darkening) the base's attenuation.
    c.coatE = 0.0;
    c.coatScale = 1.0;
    var coatAtten = float3(1.0);
    if (m.coatWeight > 0.0 && !exiting) {
        c.coatE = dielectricAlbedo(tables, coatRough, mu, m.coatIor);
        let fssCoat = fresnelDielectricAverage(m.coatIor);
        c.coatScale = reflectionScale(tables, coatRough, mu, float3(fssCoat)).x;
        coatAtten = lerp(float3(1.0), m.coatColor * (1.0 - c.coatE), m.coatWeight)
                  * coatDarkening(m, rough, mu, f0);
    }
    c.baseWeight = c.fuzzAtten * coatAtten;

    // The base interface: layering albedo and multiple-scattering scales.
    c.specE = c.hasSpecular ? dielectricAlbedo(tables, rough, mu, c.eta) : 0.0;
    c.conductorScale = reflectionScale(tables, rough, mu,
                                       fresnelSchlickAverage(m.baseColor));
    c.dielectricScale = reflectionScale(tables, rough, mu,
                                        float3(fresnelDielectricAverage(c.eta))).x;

    // The coupled glass lobe's energy, from the 3D tables (η < 1 inverts
    // into its own branch), with the interface tint standing in for the
    // multi-bounce Fresnel color.
    c.glassTint = m.transmissionDepth == 0.0 ? m.transmissionColor : float3(1.0);
    var glassEta = c.eta;
    var glassOffset = TABLE_GLASS_E;
    var glassAvgOffset = TABLE_GLASS_E_AVG;
    if (glassEta < 1.0) {
        glassEta = 1.0 / glassEta;
        glassOffset = TABLE_GLASS_INV_E;
        glassAvgOffset = TABLE_GLASS_INV_E_AVG;
    }
    let glassZ = iorRemap(glassEta);
    let glassE = max(1e-4, tableRead3D(tables, glassOffset, uint3(16, 16, 16),
                                       rough, mu, glassZ));
    let glassEAvg = tableRead2D(tables, glassAvgOffset, uint2(16, 16), rough, glassZ);
    c.glassScale = 1.0 + fresnelMultiScatter(c.glassTint, glassEAvg)
                       * ((1.0 - glassE) / glassE);

    // Thin-walled dielectric: sum the internal-bounce series into one
    // reflectance/transmittance pair at the view angle (Cycles' shape),
    // model transmission as a mirrored reflection with the Kulla-Conty
    // double-refraction roughening, and compensate each lobe at its own
    // roughness.
    c.thinR = float3(0.0);
    c.thinT = float3(0.0);
    c.thinAlpha = c.alpha;
    c.thinScaleR = 1.0;
    c.thinScaleT = 1.0;
    c.thinTShare = 0.0;
    let thin = m.thinWalled != 0 && m.transmissionWeight > 0.0;
    if (thin) {
        let thinEta = max(eta, 1.0 / eta); // a symmetric sheet: one branch
        let r1 = fresnelDielectric(thinEta, mu);
        let t1 = 1.0 - r1;
        let cos2t = max(0.0, 1.0 - (1.0 - mu * mu) / (thinEta * thinEta));
        // The sheet's normal-incidence transmittance, at this angle's
        // longer internal path (Beer–Lambert through 1/cosθt thicknesses).
        let absorb = cos2t > 0.0
                   ? pow(clamp(m.transmissionColor, 1e-4, 1.0),
                         float3(1.0 / sqrt(cos2t)))
                   : float3(0.0);
        // t' = c·t1·t2 / (1 − (r2·c)²), r' = r1 + t'·r2·c — the geometric
        // series over internal bounces (back-side coefficients equal the
        // front's by Fresnel reciprocity).
        let denom = max(float3(1e-4), 1.0 - absorb * absorb * (r1 * r1));
        c.thinT = absorb * (t1 * t1) / denom;
        c.thinR = r1 + c.thinT * (r1 * absorb);
        let thinRough = clamp(
            rough * sqrt(3.4 * (thinEta - 1.0)
                         * (thinEta - 0.5) * (thinEta - 0.5)
                         / (thinEta * thinEta * thinEta)),
            MIN_SPECULAR_ROUGHNESS, 1.0);
        // Both lobes recover their full multiple-scattering energy
        // (Fss = 1): the sheet's Fresnel already sits pre-integrated in
        // thinR/thinT, and what inter-microfacet bounces don't reflect,
        // a lossless sheet transmits — so a white sheet must sum to
        // exactly 1, which the thin furnace pins.
        c.thinAlpha = thinRough * thinRough;
        c.thinScaleR = reflectionScale(tables, rough, mu, float3(1.0)).x;
        c.thinScaleT = reflectionScale(tables, thinRough, mu, float3(1.0)).x;
        let total = average(c.thinR) + average(c.thinT);
        c.thinTShare = total > 0.0 ? average(c.thinT) / total : 0.0;
    }

    // Technique weights: each technique in proportion to the energy its
    // lobes are expected to carry. These only steer which technique gets
    // sampled — every lobe is still evaluated and pdf-weighted — but a
    // lobe with energy must keep a nonzero weight, so each estimate leans
    // on the same tabulated albedos the lobes themselves use.
    // The conductor's albedo estimate borrows the dielectric table's
    // shape: specE sits between F0 and 1 at some interpolant s, and the
    // conductor's own F0 rides the same s — Cycles' Schlick-interpolant
    // trick, run in reverse.
    let schlickS = saturate((c.specE - f0) / max(1e-4, 1.0 - f0));
    let metalE = lerp(m.baseColor, float3(1.0), schlickS);
    let glassTotal = thin ? average(c.thinR + c.thinT)
                          : c.specE + (1.0 - c.specE) * average(c.glassTint);
    let dielectricE = lerp(c.specE, glassTotal, m.transmissionWeight);
    let baseAvg = average(c.baseWeight);
    c.wFuzz = exiting ? 0.0 : m.fuzzWeight * c.fuzzR * average(m.fuzzColor);
    c.wCoat = exiting ? 0.0 : c.fuzzAtten * m.coatWeight * c.coatE;
    c.wSpec = c.hasSpecular
            ? baseAvg * lerp(dielectricE, average(metalE), m.metalness)
            : 0.0;
    c.wDiffuse = exiting ? 0.0
               : baseAvg * (1.0 - m.metalness) * (1.0 - m.transmissionWeight)
                         * (1.0 - c.specE) * average(m.baseColor);
    if (exiting) {
        c.wSpec = 1.0;
    }
    c.wTotal = c.wFuzz + c.wCoat + c.wSpec + c.wDiffuse;
    return c;
}

// -- The lobes --------------------------------------------------------------

// The LTC sheen density at wi, in the frame where wo lies at φ = 0 — a
// transformed cosine lobe, so this is simultaneously the (cosine-included)
// lobe shape and its perfect sampling pdf.
float sheenDensity(Closure c, float3 wiStd) {
    let a = c.fuzzA;
    let b = c.fuzzB;
    let lenSqr = (a * wiStd.x + b * wiStd.z) * (a * wiStd.x + b * wiStd.z)
               + (a * wiStd.y) * (a * wiStd.y) + wiStd.z * wiStd.z;
    let ratio = a / lenSqr;
    return max(wiStd.z, 0.0) * ratio * ratio / PI;
}

// Rotate v about +z by the angle that brings wo to φ = 0 (negate = back).
float3 alignToView(float3 v, float3 wo, bool inverse) {
    let len = length(wo.xy);
    if (len < 1e-6) {
        return v;
    }
    let cosPhi = wo.x / len;
    let sinPhi = (inverse ? -wo.y : wo.y) / len;
    return float3(cosPhi * v.x + sinPhi * v.y, cosPhi * v.y - sinPhi * v.x, v.z);
}

// The single-scatter GGX reflection core: D·G1·G1/(4 µo µi), no Fresnel.
float reflectionCore(float alpha, float3 wi, float3 wo, float3 h) {
    return ggxD(alpha, h.z) * smithG1(alpha, wo.z) * smithG1(alpha, wi.z)
         / (4.0 * wo.z * wi.z);
}

// The base interface's reflection: conductor and dielectric single-scatter
// share the core; each carries its own Fresnel and multiple-scattering
// scale, and the dielectric side blends its opaque and glass compensations
// by the transmission weight. Thin-walled surfaces swap the per-half-vector
// glass Fresnel for the pre-integrated sheet reflectance.
float3 baseReflection(Closure c, float3 wi, float3 wo, float3 h) {
    let core = reflectionCore(c.alpha, wi, wo, h);
    let cosVH = dot(h, wo);
    let fd = fresnelDielectric(c.eta, cosVH);
    if (c.exiting) {
        // From inside, the interface is the bare coupled glass: total
        // internal reflection and all.
        return core * fd * c.glassScale;
    }
    let conductor = fresnelSchlick(c.m.baseColor, cosVH) * c.conductorScale;
    var dielectric = lerp(float3(fd * c.dielectricScale),
                          fd * c.glassScale, c.m.transmissionWeight);
    if (c.m.thinWalled != 0) {
        dielectric = lerp(float3(fd * c.dielectricScale),
                          c.thinR * c.thinScaleR, c.m.transmissionWeight);
    }
    return core * lerp(dielectric, conductor, c.m.metalness);
}

// f(wi, wo) — the full mixture, no cosine factor; wi.z < 0 evaluates the
// transmission side.
public float3 evalBsdf(Closure c, float3 wi, float3 wo) {
    if (wo.z <= 0.0 || wi.z == 0.0) {
        return float3(0.0);
    }
    let m = c.m;
    if (wi.z < 0.0) {
        // Below the horizon only transmission lives, and only the
        // dielectric base transmits. From inside, the interface is all
        // glass.
        let glassShare = c.exiting ? 1.0 : (1.0 - m.metalness) * m.transmissionWeight;
        if (glassShare <= 0.0 || (c.exiting && m.thinWalled != 0)) {
            return float3(0.0);
        }
        if (m.thinWalled != 0) {
            // The mirrored-reflection transmission lobe: evaluate the
            // reflection shape at wi flipped to the upper hemisphere.
            let flipped = float3(wi.xy, -wi.z);
            let h = normalize(flipped + wo);
            let core = reflectionCore(c.thinAlpha, flipped, wo, h);
            return c.baseWeight * glassShare * c.thinT * c.thinScaleT * core;
        }
        // Rough refraction, Walter's form in the camera-path (adjoint)
        // convention — see the module doc.
        let hUnnormalized = wo + c.eta * wi;
        let len2 = dot(hUnnormalized, hUnnormalized);
        if (len2 < 1e-9) {
            return float3(0.0);
        }
        var h = hUnnormalized / sqrt(len2);
        h = h.z >= 0.0 ? h : -h;
        let cosVH = dot(h, wo);
        let cosLH = dot(h, wi);
        if (cosVH <= 0.0 || cosLH >= 0.0) {
            return float3(0.0);
        }
        let f = fresnelDielectric(c.eta, cosVH);
        let core = ggxD(c.alpha, h.z) * smithG1(c.alpha, wo.z) * smithG1(c.alpha, wi.z)
                 * cosVH * (c.eta * c.eta) * abs(cosLH)
                 / (wo.z * abs(wi.z) * len2);
        return c.baseWeight * glassShare * (1.0 - f) * c.glassTint * c.glassScale * core;
    }

    var result = float3(0.0);
    if (c.fuzzR > 0.0) {
        let density = sheenDensity(c, alignToView(wi, wo, false));
        result += m.fuzzWeight * m.fuzzColor * c.fuzzR * density / wi.z;
    }
    let h = normalize(wi + wo);
    if (m.coatWeight > 0.0 && !c.exiting) {
        let coatF = fresnelDielectric(m.coatIor, dot(h, wo));
        result += c.fuzzAtten * m.coatWeight * coatF * c.coatScale
                * reflectionCore(c.coatAlpha, wi, wo, h);
    }
    var base = float3(0.0);
    if (c.hasSpecular) {
        base = baseReflection(c, wi, wo, h);
    }
    if (!c.exiting) {
        base += (1.0 - m.metalness) * (1.0 - m.transmissionWeight)
              * (1.0 - c.specE)
              * eonBrdf(m.baseColor, m.baseRoughness, wi, wo);
    }
    return result + c.baseWeight * base;
}

// The spec technique's per-half-vector probability of reflecting rather
// than refracting: only the glass share of the dielectric base transmits.
float reflectProbability(Closure c, float fresnel) {
    let glassShare = c.exiting ? 1.0 : (1.0 - c.m.metalness) * c.m.transmissionWeight;
    let transmit = c.m.thinWalled != 0 ? c.thinTShare : 1.0 - fresnel;
    return 1.0 - glassShare * transmit;
}

// The mixture's solid-angle pdf of wi — the balance heuristic over the
// technique weights: Σ wᵢ·pdfᵢ / Σ wᵢ.
public float bsdfPdf(Closure c, float3 wi, float3 wo) {
    if (wo.z <= 0.0 || wi.z == 0.0 || c.wTotal <= 0.0) {
        return 0.0;
    }
    if (wi.z < 0.0) {
        if (c.wSpec <= 0.0) {
            return 0.0;
        }
        var pdf = 0.0;
        if (c.m.thinWalled != 0) {
            let flipped = float3(wi.xy, -wi.z);
            let h = normalize(flipped + wo);
            pdf = vndfReflectPdf(c.thinAlpha, wo, h)
                * (1.0 - reflectProbability(c, 0.0));
        } else {
            let hUnnormalized = wo + c.eta * wi;
            let len2 = dot(hUnnormalized, hUnnormalized);
            if (len2 < 1e-9) {
                return 0.0;
            }
            var h = hUnnormalized / sqrt(len2);
            h = h.z >= 0.0 ? h : -h;
            let cosVH = dot(h, wo);
            let cosLH = dot(h, wi);
            if (cosVH <= 0.0 || cosLH >= 0.0) {
                return 0.0;
            }
            let f = fresnelDielectric(c.eta, cosVH);
            // Dvis(h) through the refraction Jacobian dωh/dωi.
            let vndf = smithG1(c.alpha, wo.z) * ggxD(c.alpha, h.z) * cosVH / wo.z;
            let jacobian = (c.eta * c.eta) * abs(cosLH) / len2;
            pdf = vndf * jacobian * (1.0 - reflectProbability(c, f));
        }
        return c.wSpec * pdf / c.wTotal;
    }

    let h = normalize(wi + wo);
    var pdf = 0.0;
    if (c.wFuzz > 0.0) {
        pdf += c.wFuzz * sheenDensity(c, alignToView(wi, wo, false));
    }
    if (c.wCoat > 0.0) {
        pdf += c.wCoat * vndfReflectPdf(c.coatAlpha, wo, h);
    }
    if (c.wSpec > 0.0) {
        let f = fresnelDielectric(c.eta, dot(h, wo));
        pdf += c.wSpec * vndfReflectPdf(c.alpha, wo, h) * reflectProbability(c, f);
    }
    if (c.wDiffuse > 0.0) {
        pdf += c.wDiffuse * wi.z / PI;
    }
    return pdf / c.wTotal;
}

// A sampled scatter: direction, mixture pdf, and the technique tag.
public struct BsdfSample {
    public float3 wi;
    public float pdf; // 0 marks a rejected sample — the path terminates
    public uint lobe; // LOBE_*, recorded into path state
};

// Draw a scattered direction: pick one technique by the closure's weights
// — rescaling uLobe to a full-range uniform, which the glass technique
// then spends on its reflect/refract branch — and return the mixture pdf
// of the result.
public BsdfSample sampleBsdf(Closure c, float3 wo, float uLobe, float2 u) {
    BsdfSample result;
    result.wi = float3(0.0, 0.0, 1.0);
    result.pdf = 0.0;
    result.lobe = LOBE_DIFFUSE;
    if (c.wTotal <= 0.0) {
        return result;
    }

    let pick = uLobe * c.wTotal;
    var wi = float3(0.0);
    if (pick < c.wFuzz) {
        result.lobe = LOBE_FUZZ;
        // A cosine-hemisphere sample pushed through the LTC transform.
        let r = sqrt(u.x);
        let phi = 2.0 * PI * u.y;
        let disk = float2(r * cos(phi), r * sin(phi));
        let diskZ = sqrt(max(0.0, 1.0 - dot(disk, disk)));
        let wiStd = normalize(float3(disk.x - diskZ * c.fuzzB, disk.y,
                                     diskZ * c.fuzzA));
        wi = alignToView(wiStd, wo, true);
    } else if (pick < c.wFuzz + c.wCoat) {
        result.lobe = LOBE_COAT;
        let h = sampleVndf(wo, c.coatAlpha, u);
        wi = 2.0 * dot(wo, h) * h - wo;
    } else if (pick < c.wFuzz + c.wCoat + c.wSpec) {
        // The leftover of the technique draw, remapped to a full-range
        // uniform for the reflect/refract branch.
        let uBranch = (pick - c.wFuzz - c.wCoat) / max(1e-6, c.wSpec);
        if (c.m.thinWalled != 0) {
            if (uBranch < reflectProbability(c, 0.0)) {
                result.lobe = LOBE_SPECULAR;
                let h = sampleVndf(wo, c.alpha, u);
                wi = 2.0 * dot(wo, h) * h - wo;
            } else {
                result.lobe = LOBE_TRANSMISSION;
                let h = sampleVndf(wo, c.thinAlpha, u);
                let reflected = 2.0 * dot(wo, h) * h - wo;
                wi = float3(reflected.xy, -reflected.z);
            }
        } else {
            let h = sampleVndf(wo, c.alpha, u);
            let f = fresnelDielectric(c.eta, dot(h, wo));
            if (uBranch < reflectProbability(c, f)) {
                result.lobe = LOBE_SPECULAR;
                wi = 2.0 * dot(wo, h) * h - wo;
            } else {
                result.lobe = LOBE_TRANSMISSION;
                let refracted = refractDirection(wo, h, c.eta);
                if (refracted.w == 0.0) {
                    return result; // unreachable: F = 1 past the critical angle
                }
                wi = refracted.xyz;
            }
        }
    } else {
        result.lobe = LOBE_DIFFUSE;
        wi = sampleCosineHemisphere(u).xyz;
    }

    result.wi = wi;
    result.pdf = bsdfPdf(c, wi, wo);
    return result;
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
