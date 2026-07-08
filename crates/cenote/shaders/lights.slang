// Quad lights and their next-event sampling. A light is an emissive
// parallelogram mesh instance in the TLAS — BSDF-sampled paths reach it
// through ordinary traversal, so MIS never needs a second intersection
// code path. This module holds the record the light list stores per quad
// and the sampling that picks one and a point on it; the records are built
// at prep by `lights.rs`, which mirrors them struct for struct.
//
// Lights are two-sided, matching the renderer's two-sided surfaces: both
// faces emit, and every light-side cosine is an absolute value.

// `GeometryRecord.light` for instances that are not lights — and, on a
// queued shadow ray, the target "instance" of an environment connection
// (no TLAS instance can carry this ID, so such a ray is unshadowed exactly
// when it hits nothing).
public static const uint LIGHT_NONE = 0xffffffff;

// Which sampling strategies reach the lights (quads and environment
// alike). MIS is the renderer; the single-strategy modes exist because the
// strongest test of the weights is that either strategy alone converges to
// the same image (the MIS-agreement test in `wavefront.rs`). Values match
// `enum LightSampling` there.
public static const uint LIGHT_SAMPLING_MIS = 0;
public static const uint LIGHT_SAMPLING_BSDF = 1;
public static const uint LIGHT_SAMPLING_NEE = 2;

// One quad light. Selection is a power-proportional alias table — one slot
// per light — so each record carries its own slot (`aliasThreshold`,
// `aliasIndex`) in what std430 would otherwise spend on padding: one
// buffer, and the common case reads one record.
public struct LightRecord {
    public float3 corner;        // one vertex of the parallelogram, world space
    public float pdfArea;        // p(selecting this light) / its area
    public float3 edge1;         // the sides meeting at corner: the quad is
    public float aliasThreshold; //   corner + u·edge1 + v·edge2, u,v ∈ [0,1]
    public float3 edge2;
    public uint aliasIndex;
    public float3 emission;      // radiance, ACEScg, from both faces
    public uint instance;        // TLAS custom index — visibility tests need it
};

// A next-event candidate: a point on some light, with everything shading
// needs to weigh the connection and hand it to trace_shadow.
public struct LightSample {
    public float3 position;
    public float pdfArea; // p(this light) / area; ×r²/cosθ for solid angle
    public float3 normal; // unit — remember both faces emit
    public uint instance;
    public float3 emission;
};

// Draw one light ∝ power, then a uniform point on it. The alias-table
// lookup: `u` scaled by the light count picks a slot, and the leftover
// fraction picks between the slot's own light and its alias — every light
// lands at exactly its power share in two coin flips flattened into one.
public LightSample sampleLights(LightRecord* lights, uint count, float u,
                                float2 uPoint) {
    let scaled = u * float(count);
    let slot = min(uint(scaled), count - 1);
    var light = lights[slot];
    if (scaled - float(slot) >= light.aliasThreshold) {
        light = lights[light.aliasIndex];
    }
    LightSample s;
    s.position = light.corner + uPoint.x * light.edge1 + uPoint.y * light.edge2;
    s.pdfArea = light.pdfArea;
    s.normal = normalize(cross(light.edge1, light.edge2));
    s.instance = light.instance;
    s.emission = light.emission;
    return s;
}
