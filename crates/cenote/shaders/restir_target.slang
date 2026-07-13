// Shared reservoir-reuse machinery: reconstruct a primary hit's shading surface
// from the path pool, reconnect a stored light sample to it, and evaluate the
// *unshadowed* target p̂ = luminance(L·f·|cosθ|) there. The spatial and temporal
// reuse stages both rebuild a surface and re-target a neighbour's sample on it,
// and the reconstruction must reproduce `restir_candidates`' (and the path
// tracer's) surface *exactly* — a drift is a silent bias (surface.slang). One
// copy, imported by both stages, so the two can never diverge.
//
// What stays with its caller, deliberately: the *visibility-aware* target
// `targetVisAware` lives in `restir_spatial.slang`, because it traces a shadow
// ray against that stage's TLAS binding — a resource a module cannot own. The
// temporal combine is unshadowed (D-094), so it needs only the pieces here.

import lights;
import openpbr;
import pathstate;
import reservoir_di;
import restir_scene;
import scene;
import surface;
import textures;

// Relative luminance of an ACEScg (AP1) colour — the CIE Y row of the AP1→XYZ
// matrix, the reservoir's scalar target. Every stage that weighs a candidate by
// brightness sorts on this one function, so they agree pointwise.
public float luminance(float3 acescg) {
    return dot(acescg, float3(0.2722287168, 0.6740817658, 0.0536895174));
}

// A reconstructed shading point: everything the target function needs to weigh a
// light sample here, rebuilt from a `Hit` exactly as `restir_candidates` does,
// so the reservoir's target and its reuse land on the same surface as the path
// tracer (a drift would be a silent bias — see surface.slang).
public struct Surface {
    public float3 position;
    public float3 below;
    public float3 normal;   // geometric, faced toward the arriving ray
    public float3 origin;   // the ray's origin — camera-plane depth is |pos − origin|·cosCam
    public float depth;     // camera-plane z, for the reuse gate
    public Frame frame;
    public float3 wo;
    public Closure closure;
    public bool valid;
};

// Rebuild the surface at path slot `slot`'s primary hit. `slot` indexes the path
// pool directly (its Hit, its ray). An out-of-range or non-hit slot is the
// caller's responsibility to exclude; this assumes a live primary hit.
public Surface reconstructSurface(Paths paths, SceneTable* scene, uint slot) {
    return reconstructSurfaceFrom(scene, paths.hit[slot], paths.direction[slot],
                                  paths.origin[slot].xyz);
}

// Rebuild a surface from an explicit hit and ray, rather than from the live path
// pool. The temporal reuse stage keeps this to reconstruct *last* frame's surface
// from a persisted hit (`restir_reproject.slang`'s G-buffer) — the path pool is
// this frame's and no longer holds it — so `p̂_i` weighs the reprojected
// neighbour on its own real closure. `ray.xyz` is the unit direction; `ray.w` is
// the camera-forward cosine (raygen's depth trick); `origin` is the ray origin.
// The scene geometry is assumed static across the frame boundary (no animation on
// the roadmap), so this frame's `scene` reproduces last frame's surface exactly.
public Surface reconstructSurfaceFrom(SceneTable* scene, Hit hit, float4 ray,
                                      float3 origin) {
    let direction = ray.xyz;

    GeometryRecord mesh = scene.geometry[hit.instance];
    let uv = interpolateUv(mesh, hit.primitive, hit.barycentrics);
    let material = resolveMaterial(scene.materials[hit.instance], uv);
    let spawn = spawnPoint(mesh, hit, direction);
    var shading = spawn.shading;
    if (material.normalTexture != TEXTURE_NONE) {
        shading = mappedShadingNormal(mesh, hit, uv, material.normalTexture, spawn);
    }

    Surface s;
    s.position = spawn.position;
    s.below = spawn.below;
    s.normal = spawn.normal;
    s.origin = origin;
    // Camera-plane z: hit distance projected onto the camera forward axis, whose
    // cosine raygen stashed in direction.w (the depth-AOV trick, decisions §D-062).
    s.depth = length(spawn.position - origin) * ray.w;
    s.frame = Frame(shading);
    s.wo = s.frame.toLocal(-direction);
    s.closure = resolveClosure(scene.bsdfTables, material, s.wo, false);
    s.valid = true;
    return s;
}

// A light sample reconnected to a shading point: the direction, radiance, and
// shadow-ray target a stored `DirectLightSample` makes at `from`. The one place
// the light-type branch lives (D-093): an environment sample is a *direction*,
// position-independent, so the same `connect` serves every surface; a triangle
// sample re-derives its cosθ/d² geometry against `from`. A dropped light (id no
// longer present) or a degenerate/back-face connection returns `valid = false`.
public struct Connection {
    public float3 toLight;   // unit, from `from` toward the light point
    public float3 radiance;  // arriving radiance, emission map folded in
    public bool valid;
    public uint shadowLight;     // shadow-ray identity target (TLAS index), or LIGHT_NONE
    public uint shadowPrimitive;
    public float shadowTmax;
};

public Connection connectSampleAt(SceneTable* scene, RestirScene* restir,
                                  DirectLightSample sample, float3 from) {
    Connection c;
    c.toLight = float3(0.0);
    c.radiance = float3(0.0);
    c.valid = false;
    c.shadowLight = LIGHT_NONE;
    c.shadowPrimitive = 0;
    c.shadowTmax = 1e30;

    if (isEnvironmentSample(sample)) {
        let e = scene.env.connect(sample.barycentrics);
        if (e.pdf > 0.0) {
            c.toLight = e.direction;
            c.radiance = e.radiance;
            c.valid = true;
        }
        return c; // env: nothing to hit — LIGHT_NONE, open tmax already set
    }

    // Translate the stable id back to the current instance; a light deleted
    // since the neighbour stored this sample resolves to nothing (drop).
    let instance = restir.idToInstance[sample.light];
    if (instance == LIGHT_NONE) {
        return c;
    }
    let lightMesh = scene.geometry[instance];
    LightRecord record = restir.candidateLights[lightMesh.light + sample.primitive];
    var tri = connectTriangle(record, sample.barycentrics, from);
    if (tri.pdf > 0.0) {
        let emitter = scene.materials[instance];
        if (emitter.emissionTexture != TEXTURE_NONE) {
            tri.radiance *= sampleColor(
                emitter.emissionTexture,
                interpolateUv(lightMesh, sample.primitive, sample.barycentrics));
        }
        c.toLight = tri.direction;
        c.radiance = tri.radiance;
        c.valid = true;
        c.shadowLight = instance;
        c.shadowPrimitive = sample.primitive;
        c.shadowTmax = tri.tmax;
    }
    return c;
}

// The unshadowed target p̂ = luminance(L·f·|cosθ|) of a light sample at a
// surface, over the real closure. Zero for a sample the surface can't carry
// (below a horizon the base doesn't transmit through, or a degenerate
// connection). This is the own-domain target p̂ every reuse stage evaluates
// without a shadow ray; the visibility-aware canonical target adds one ray on
// top (`targetVisAware`, in restir_spatial.slang).
public float targetUnshadowed(SceneTable* scene, RestirScene* restir,
                              Surface s, DirectLightSample sample) {
    let c = connectSampleAt(scene, restir, sample, s.position);
    if (!c.valid) {
        return 0.0;
    }
    let wiLight = s.frame.toLocal(c.toLight);
    let throughSurface = dot(c.toLight, s.normal) < 0.0;
    if (wiLight.z == 0.0 || (wiLight.z < 0.0) != throughSurface) {
        return 0.0;
    }
    return luminance(c.radiance * evalBsdf(s.closure, wiLight, s.wo) * abs(wiLight.z));
}
