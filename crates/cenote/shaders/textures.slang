// The bindless texture table: every material texture the scene holds, in
// one runtime-sized descriptor array that material records index
// (TEXTURE_NONE = the parameter is its constant everywhere). The images
// are what `texture.rs` prepped: BC7 color — sRGB-decoded by the sampler
// hardware, converted to the ACEScg working space here, the in-shader
// IDT — BC6H HDR color, BC4 scalars in the red channel, and BC5
// tangent-space normals with z reconstructed from unit length. Every
// read is explicit level 0: one mip exists (the mip-cap policy — jittered
// accumulation integrates the pixel footprint), and compute kernels have
// no implicit derivatives anyway.
//
// Beside the images rides a parameter table, indexed identically: the
// affine UV remap and value multiplier a texture reference carries
// (identity for most). Sampling applies the remap before the lookup —
// the sampler wraps, so tiling scales tile — and the multiplier after
// decode, on linear values. Callers pass the table's address; it lives on
// `SceneTable` (scene.slang), which imports this module and so can't be
// named here.

import openpbr;

// "No texture" in a material's index slots — matches `TEXTURE_NONE` in
// `material.rs`.
public static const uint TEXTURE_NONE = 0xffffffff;

// One bindless slot's sample-time parameters. Mirrors `struct
// TextureParams` in `scene/mod.rs` field for field.
public struct TextureParams {
    public float2 uvScale;  // multiplier over the mesh UV (tiling above 1)
    public float2 uvOffset; // added after the scale
    public float scale;     // multiplier over decoded values (not normals)
    uint pad0;
    uint pad1;
    uint pad2;
};

// The table itself, next to the TLAS and the environment in the one
// descriptor set. Partially bound: a scene writes only as many slots as
// it holds, and no kernel indexes past what its material records name.
[vk::binding(2, 0)]
Sampler2D<float4> sceneTextures[];

// One texel at the reference's remapped UV. The index is per-hit data —
// materials vary across a wave — so the descriptor access must be
// declared non-uniform; without it the compiler may assume the whole
// subgroup reads one texture.
float4 sampleTexture(TextureParams *params, uint index, float2 uv) {
    let p = params[index];
    let mapped = uv * p.uvScale + p.uvOffset;
    return sceneTextures[NonUniformResourceIndex(index)].SampleLevel(mapped, 0.0);
}

// Linear Rec.709 — what the sampler's sRGB decode (and a linear color
// source) yields — into ACEScg. The same matrix `color.rs` applies to
// color constants at prep (it documents the derivation), so constants
// and texels land in one working space; rows here are its columns, since
// this multiplies as rows-dot-vector.
static const float3x3 ACESCG_FROM_REC709 = float3x3(
    0.613097400, 0.339523140, 0.047379450,
    0.070193730, 0.916353900, 0.013452399,
    0.020615594, 0.109569780, 0.869814660);

// A color slot's texel in the working space. The value multiplier lands
// on the linear Rec.709 side, where the reference's convention puts it —
// a uniform scale commutes with the matrix anyway.
public float3 sampleColor(TextureParams *params, uint index, float2 uv) {
    let texel = sampleTexture(params, index, uv).rgb * params[index].scale;
    return mul(ACESCG_FROM_REC709, texel);
}

// A scalar slot's texel — BC4 carries it in the red channel.
public float sampleScalar(TextureParams *params, uint index, float2 uv) {
    return sampleTexture(params, index, uv).r * params[index].scale;
}

// Three channels that are data rather than color — a per-channel mean free
// path. Baked like a color (`Usage::Vector` in texture.rs, the same BC7 or
// BC6H) and sampled like a scalar: the working-space matrix is what makes a
// color a color, and a length has no space to be converted from. Without
// this the constant and the map would disagree, since prep converts one
// and not the other.
public float3 sampleVector(TextureParams *params, uint index, float2 uv) {
    return sampleTexture(params, index, uv).rgb * params[index].scale;
}

// A tangent-space normal-map texel as a unit vector: BC5 stores x and y,
// z rebuilds from unit length (normal maps point out of the surface, so
// the root's positive branch is the whole upper hemisphere). The UV remap
// applies; the value multiplier does not — a normal is a direction, not a
// quantity (lowering keeps the slot's params at identity scale).
public float3 sampleNormal(TextureParams *params, uint index, float2 uv) {
    let xy = sampleTexture(params, index, uv).rg * 2.0 - 1.0;
    return float3(xy, sqrt(max(1.0 - dot(xy, xy), 0.0)));
}

// The material at a hit: textured parameters resolved at its UV, so
// everything downstream — closure resolution, lobe weights, emission —
// reads one plain Material and never learns which parameters were maps.
// Replaced slots take the texel outright; emission multiplies it (the
// LDR-map × luminance-scale convention, with the coat tint prep already
// folded in). Two slots resolve elsewhere by design: opacity belongs to
// the traversal kernels (`opacityAt` in scene.slang, per crossing), and
// the normal map perturbs the shading frame this material is then
// evaluated against.
public Material resolveMaterial(TextureParams *params, Material material, float2 uv) {
    if (material.baseColorTexture != TEXTURE_NONE) {
        material.baseColor = sampleColor(params, material.baseColorTexture, uv);
    }
    if (material.specularRoughnessTexture != TEXTURE_NONE) {
        material.specularRoughness = sampleScalar(params, material.specularRoughnessTexture, uv);
    }
    if (material.metalnessTexture != TEXTURE_NONE) {
        material.metalness = sampleScalar(params, material.metalnessTexture, uv);
    }
    if (material.emissionTexture != TEXTURE_NONE) {
        material.emission *= sampleColor(params, material.emissionTexture, uv);
    }
    return resolveSubsurface(params, material, uv);
}

// The subsurface five, split out because they have two readers on two
// schedules: the surface stage resolves the whole material at the entry
// vertex, where the weight and the color decide the lobe's share of the
// closure, and the walk kernel resolves *only* these at that same vertex,
// to invert into the interior it is about to walk. Splitting keeps the
// walk from sampling a base-color or emission map it will never look at,
// and keeps one list of the five rather than two that can drift.
public Material resolveSubsurface(TextureParams *params, Material material, float2 uv) {
    if (material.subsurfaceWeightTexture != TEXTURE_NONE) {
        material.subsurfaceWeight = sampleScalar(params, material.subsurfaceWeightTexture, uv);
    }
    if (material.subsurfaceColorTexture != TEXTURE_NONE) {
        material.subsurfaceColor = sampleColor(params, material.subsurfaceColorTexture, uv);
    }
    if (material.subsurfaceRadiusTexture != TEXTURE_NONE) {
        material.subsurfaceRadius = sampleScalar(params, material.subsurfaceRadiusTexture, uv);
    }
    if (material.subsurfaceRadiusScaleTexture != TEXTURE_NONE) {
        material.subsurfaceRadiusScale =
            sampleVector(params, material.subsurfaceRadiusScaleTexture, uv);
    }
    if (material.subsurfaceScatterAnisotropyTexture != TEXTURE_NONE) {
        material.subsurfaceScatterAnisotropy =
            sampleScalar(params, material.subsurfaceScatterAnisotropyTexture, uv);
    }
    return material;
}
