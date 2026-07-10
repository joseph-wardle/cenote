// The light list and its next-event sampling: emissive triangles plus
// delta lights (distant, point), selected from one power-proportional
// alias table. Emissive geometry lives in the TLAS — BSDF-sampled paths
// reach it through ordinary traversal, so MIS never needs a second
// intersection code path. Delta lights have zero area: a BSDF sample hits
// them with probability zero, so next-event estimation is their only
// strategy and their MIS weight is 1. The records are built at prep by
// `lights.rs`, which mirrors them struct for struct.
//
// Emissive surfaces are two-sided, matching the renderer's two-sided
// surfaces: both faces emit, and every light-side cosine is an absolute
// value.

// `GeometryRecord.light` for instances that are not lights — and, on a
// queued shadow ray, the target "instance" of a connection with nothing
// to hit (environment, delta lights): no TLAS instance can carry this ID,
// so such a ray is unshadowed exactly when it hits nothing.
public static const uint LIGHT_NONE = 0xffffffff;

// `LightRecord.kind` values. Match the `KIND_*` constants in `lights.rs`.
public static const uint LIGHT_KIND_TRIANGLE = 0;
public static const uint LIGHT_KIND_DISTANT = 1;
public static const uint LIGHT_KIND_POINT = 2;

// Which sampling strategies reach the lights. MIS is the renderer; the
// single-strategy modes exist because the strongest test of the weights
// is that either strategy alone converges to the same image (the
// MIS-agreement test in `wavefront.rs`) — with the caveat that delta
// lights exist only through next-event connections, so BSDF-only mode
// cannot see them at all. Values match `enum LightSampling` there.
public static const uint LIGHT_SAMPLING_MIS = 0;
public static const uint LIGHT_SAMPLING_BSDF = 1;
public static const uint LIGHT_SAMPLING_NEE = 2;

// Veach's power heuristic (β = 2): the weight of a sample drawn from the
// strategy with pdf `p` when the strategy with pdf `q` could have produced
// it too. The weights of the two draws sum to 1, so combining strategies
// stays unbiased; squaring favors whichever pdf is sharper. Shared by the
// two kernels that weigh next-event against BSDF sampling (shade_surface,
// shade_miss).
public float powerHeuristic(float p, float q) {
    return (p * p) / (p * p + q * q);
}

// One light. Selection is a power-proportional alias table — one slot per
// light — so each record carries its own slot (`aliasThreshold`,
// `aliasIndex`). The geometric fields are per-kind, as documented.
public struct LightRecord {
    public float3 a;             // triangle: one corner, world space;
                                 //   distant: unit travel direction; point: position
    public float pdf;            // triangle: p(selecting this light) / its area;
                                 //   delta: the selection probability itself. 0 = never drawn
    public float3 edge1;         // triangle only: the sides meeting at `a` — it
    public float aliasThreshold; //   spans a + b1·edge1 + b2·edge2, b1 + b2 ≤ 1
    public float3 edge2;
    public uint aliasIndex;
    public float3 emission;      // triangle: radiance; distant: irradiance;
                                 //   point: intensity — all ACEScg
    public uint instance;        // triangle: TLAS custom index; delta: LIGHT_NONE
    public uint kind;            // LIGHT_KIND_*
    public uint primitive;       // triangle only: index within its mesh
    uint pad0;
    uint pad1;
};

// A next-event connection from one shading point to one light: everything
// shading needs to weigh it and hand it to trace_shadow. `pdf` 0 means the
// draw carries nothing (degenerate record, zero-distance point) — skip it.
public struct LightConnection {
    public float3 direction; // unit, from the shading point toward the light
    public float tmax;       // shadow-ray extent: bounded for point lights
                             //   (nothing sits at the light itself), open otherwise
    public float3 radiance;  // what arrives if unoccluded (point falloff folded in)
    public float pdf;        // pdf of this draw within the table: per solid
                             //   angle (triangle) or the discrete selection
                             //   probability (delta) — callers fold in the
                             //   table-vs-environment selection
    public bool delta;       // no BSDF strategy competes: MIS weight is 1
    public uint instance;    // shadow-ray identity target, or LIGHT_NONE
    public uint primitive;
};

// Draw one light ∝ power and connect `from` to it. The alias-table
// lookup: `u` scaled by the light count picks a slot, and the leftover
// fraction picks between the slot's own light and its alias — every light
// lands at exactly its power share in two coin flips flattened into one.
public LightConnection sampleLights(LightRecord* lights, uint count, float u,
                                    float2 uPoint, float3 from) {
    let scaled = u * float(count);
    let slot = min(uint(scaled), count - 1);
    var light = lights[slot];
    if (scaled - float(slot) >= light.aliasThreshold) {
        light = lights[light.aliasIndex];
    }

    LightConnection c;
    c.pdf = 0.0;
    c.direction = float3(0.0);
    c.tmax = 1e30;
    c.radiance = light.emission;
    c.delta = light.kind != LIGHT_KIND_TRIANGLE;
    c.instance = light.instance;
    c.primitive = light.primitive;

    switch (light.kind) {
    case LIGHT_KIND_TRIANGLE: {
        // A uniform point on the triangle: √u1 picks the slice so area is
        // preserved, u2 the position along it.
        let su = sqrt(uPoint.x);
        let position = light.a + (1.0 - su) * light.edge1 + (uPoint.y * su) * light.edge2;
        let delta = position - from;
        let r2 = dot(delta, delta);
        let normal = normalize(cross(light.edge1, light.edge2));
        let direction = delta * rsqrt(r2);
        let cosLight = abs(dot(normal, direction)); // both faces emit
        if (r2 > 0.0 && cosLight > 0.0 && light.pdf > 0.0) {
            c.direction = direction;
            // The area pdf lifted to solid angle at the shading point —
            // the measure BSDF sampling competes in.
            c.pdf = light.pdf * r2 / cosLight;
        }
        break;
    }
    case LIGHT_KIND_DISTANT:
        c.direction = -light.a; // stored as the light's travel direction
        c.pdf = light.pdf;
        break;
    default: { // LIGHT_KIND_POINT
        let delta = light.a - from;
        let r2 = dot(delta, delta);
        if (r2 > 0.0 && light.pdf > 0.0) {
            let r = sqrt(r2);
            c.direction = delta / r;
            c.radiance = light.emission / r2; // intensity → arriving radiance
            c.pdf = light.pdf;
            // The light itself is not geometry: anything committed before
            // it is a real occluder, and nothing sits at the light, so the
            // exact distance needs no epsilon.
            c.tmax = r;
        }
        break;
    }
    }
    return c;
}
