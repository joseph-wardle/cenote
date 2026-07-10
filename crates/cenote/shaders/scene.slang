// The GPU scene, reached through one `SceneTable` of buffer addresses that
// mirrors `struct SceneTable` in `scene.rs` (not the resource-owning
// `struct Scene` beside it): kernels carry a single scene pointer in their
// push constants, and new scene-wide tables land here without growing any
// kernel's parameter block.

import environment;
import lights;
import openpbr;
import textures;

// Where an instance's triangles live, plus its transforms — everything
// needed to re-evaluate shading at a Hit. Mirrors `GeometryRecord` in
// `scene.rs` field for field.
public struct GeometryRecord {
    public float* positions;        // x, y, z per vertex, object space
    public float* normals;          // unit shading normal per vertex, object space
    public float2* uvs;             // texture coordinates per vertex (zeros when unauthored)
    public uint* indices;           // three per triangle
    public float4 objectToWorld[3]; // rows of the instance's 3×4 transform
    public float4 worldToObject[3]; // rows of its inverse — normals, error bounds
    public uint light;              // index into the light list, or LIGHT_NONE
    uint pad0;
    uint pad1;
    uint pad2;
};

// The UV at barycentrics (b1, b2) of a triangle — weights on the second
// and third corner, the convention ray-query barycentrics and the light
// sampler's triangle parameterization share.
public float2 interpolateUv(GeometryRecord mesh, uint primitive, float2 bary) {
    uint base = 3 * primitive;
    let uv0 = mesh.uvs[mesh.indices[base]];
    let uv1 = mesh.uvs[mesh.indices[base + 1]];
    let uv2 = mesh.uvs[mesh.indices[base + 2]];
    return uv0 + bary.x * (uv1 - uv0) + bary.y * (uv2 - uv0);
}

// Everything the scene shares with the kernels: buffer addresses (indexed
// by TLAS instance custom index, except the light list), the closure's
// baked lookup tables, plus the embedded environment light.
public struct SceneTable {
    public GeometryRecord* geometry;
    public Material* materials;
    public LightRecord* lights;
    public float* bsdfTables; // layout in `tables.rs` ↔ openpbr.slang TABLE_*
    public Environment env;
    public uint lightCount;
    uint pad;
};

// The coverage of one traversal crossing: the instance's opacity constant
// times its opacity map at the crossing point — the one value both halves
// of the transparency policy test, stochastically in intersect and
// multiplicatively in trace_shadow.
public float opacityAt(SceneTable* scene, uint instance, uint primitive, float2 bary) {
    let material = scene.materials[instance];
    var opacity = material.opacity;
    if (material.opacityTexture != TEXTURE_NONE) {
        opacity *= sampleScalar(material.opacityTexture,
                                interpolateUv(scene.geometry[instance], primitive, bary));
    }
    return opacity;
}
