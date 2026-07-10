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

import openpbr;

// "No texture" in a material's index slots — matches `TEXTURE_NONE` in
// `material.rs`.
public static const uint TEXTURE_NONE = 0xffffffff;

// The table itself, next to the TLAS and the environment in the one
// descriptor set. Partially bound: a scene writes only as many slots as
// it holds, and no kernel indexes past what its material records name.
[vk::binding(2, 0)]
Sampler2D<float4> sceneTextures[];

// One texel. The index is per-hit data — materials vary across a wave —
// so the descriptor access must be declared non-uniform; without it the
// compiler may assume the whole subgroup reads one texture.
float4 sampleTexture(uint index, float2 uv) {
    return sceneTextures[NonUniformResourceIndex(index)].SampleLevel(uv, 0.0);
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

// A color slot's texel in the working space.
public float3 sampleColor(uint index, float2 uv) {
    return mul(ACESCG_FROM_REC709, sampleTexture(index, uv).rgb);
}

// A scalar slot's texel — BC4 carries it in the red channel.
public float sampleScalar(uint index, float2 uv) {
    return sampleTexture(index, uv).r;
}

// A tangent-space normal-map texel as a unit vector: BC5 stores x and y,
// z rebuilds from unit length (normal maps point out of the surface, so
// the root's positive branch is the whole upper hemisphere).
public float3 sampleNormal(uint index, float2 uv) {
    let xy = sampleTexture(index, uv).rg * 2.0 - 1.0;
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
public Material resolveMaterial(Material material, float2 uv) {
    if (material.baseColorTexture != TEXTURE_NONE) {
        material.baseColor = sampleColor(material.baseColorTexture, uv);
    }
    if (material.specularRoughnessTexture != TEXTURE_NONE) {
        material.specularRoughness = sampleScalar(material.specularRoughnessTexture, uv);
    }
    if (material.metalnessTexture != TEXTURE_NONE) {
        material.metalness = sampleScalar(material.metalnessTexture, uv);
    }
    if (material.emissionTexture != TEXTURE_NONE) {
        material.emission *= sampleColor(material.emissionTexture, uv);
    }
    return material;
}
