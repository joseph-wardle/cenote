// The OpenPBR surface model, M1 subset: the EON diffuse base lobe —
// energy-preserving Oren-Nayar ("EON: A Practical Energy-Preserving Rough
// Diffuse BRDF", Portsmouth, Kutz & Hill, JCGT 2025) — which is the diffuse
// lobe OpenPBR actually specifies (plain Lambert is not) — plus emission.
// The conductor and dielectric GGX lobes join in M1 step 9.
//
// BSDF convention: wi (scattered) and wo (toward the viewer) are unit
// directions pointing away from the surface, in a local frame with the
// normal along +z, so wo.z > 0. Lobes are plain eval/sample functions — no
// interfaces, no dynamic dispatch.

public static const float PI = 3.14159265358979;

// One instance's surface parameters. Mirrors `Material` in `material.rs`
// field for field.
public struct Material {
    public float3 baseColor;    // diffuse single-scattering albedo, ACEScg
    public float baseRoughness; // Oren-Nayar roughness in [0, 1]; 0 = Lambert
    public float3 emission;     // radiance, ACEScg — nonzero marks a light
    float pad;
};

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
// scattering loses — an albedo-1 surface reflects 1 at every angle, which
// is the white-furnace property the test suite pins.
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
// w its pdf cos(θ)/π — never zero, since u.x < 1. Sampling the estimator's
// cosine factor; the EON paper's CLTC sampling (matching the lobe's own
// shape) is a documented later variance upgrade.
public float4 sampleCosineHemisphere(float2 u) {
    let r = sqrt(u.x);
    let phi = 2.0 * PI * u.y;
    let z = sqrt(1.0 - u.x);
    return float4(r * cos(phi), r * sin(phi), z, z / PI);
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
