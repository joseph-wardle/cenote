// The GPU scene, reached through one `SceneTable` of buffer addresses that
// mirrors `Scene` in `scene.rs`: kernels carry a single scene pointer in
// their push constants, and new scene-wide tables (the HDRI's CDFs next)
// land here without growing any kernel's parameter block.

import lights;
import openpbr;

// Where an instance's triangles live, plus its transforms — everything
// needed to re-evaluate shading at a Hit. Mirrors `GeometryRecord` in
// `scene.rs` field for field.
public struct GeometryRecord {
    public float* positions;        // x, y, z per vertex, object space
    public uint* indices;           // three per triangle
    public float4 objectToWorld[3]; // rows of the instance's 3×4 transform
    public float4 worldToObject[3]; // rows of its inverse — normals, error bounds
    public uint light;              // index into the light list, or LIGHT_NONE
    uint pad0;
    uint pad1;
    uint pad2;
};

// Everything the scene shares with the kernels, one address each, all
// indexed by TLAS instance custom index except the light list.
public struct SceneTable {
    public GeometryRecord* geometry;
    public Material* materials;
    public LightRecord* lights;
    public uint lightCount;
    uint pad;
};
