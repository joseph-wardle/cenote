// The reservoir path's own slice of the scene, carried beside the shared
// `SceneTable` in the two initial-RIS stages' push constants: the
// reservoir-candidate light table and the stable-identity remap samples are
// stored through. It is deliberately separate from `SceneTable` — the rest of
// the wavefront neither has nor needs it, and folding it in would grow every
// kernel's push-constant block. Mirrors `struct RestirScene` in src/restir.rs.
//
// The candidate table is the triangle-only alias table (`lights::candidate_table`),
// normalized over the emitters alone — the environment is its coin-flip partner,
// not a row in it (D-088). Its triangle records occupy the same indices as the
// combined `SceneTable.lights`, so a stored sample resolves through
// `geometry[instance].light + primitive` in either.

import lights;
import reservoir_path;

// -- The unified reservoir's two path kinds ----------------------------------
// M6's unified reservoir (D-124) holds two shapes of sample, discriminated by
// the low bit of `PathSample.reserved` — zero by default, so a zero-initialized
// sample reads as the NEE base case:
//
//   PATH_KIND_NEE (step 1, D-124) — a length-2 direct-lighting path whose
//     reconnection vertex `rcVertex` *is* the light surface point. The light's
//     stable identity lives in `rcVertex.instance` (never the volatile TLAS
//     custom index), `primitive`/`barycentrics` the emitter point (or the env
//     sampling coordinate). The light-id sentinels and the environment
//     predicate below interpret it.
//
//   PATH_KIND_RECONNECTION (step 2, D-134) — a length-≥3 indirect path whose
//     reconnection vertex is a *geometry* vertex x₂ (the first indirect bounce,
//     a real surface hit): `rcVertex.instance` is a live TLAS instance, not a
//     light id, and `rcVertexRadiance` caches Lo(x₂) — the outgoing reflected
//     radiance there — so the reconnection shift re-forms the contribution
//     without re-tracing the tail. The NEE interpretation below does *not*
//     apply to it, so every predicate here guards on the kind first.
//
// The kind lives beside the NEE interpretation (rather than in the
// light-agnostic `reservoir_path.slang`) because it is exactly the discriminator
// that says whether that interpretation applies.
public static const uint PATH_KIND_NEE = 0;          // length-2 direct: rcVertex = light point
public static const uint PATH_KIND_RECONNECTION = 1; // length-≥3 indirect: rcVertex = x₂

public uint pathKind(PathSample sample) {
    return sample.reserved & 0x1;
}

// A length-≥3 indirect sample: `rcVertex` is a geometry vertex, `rcVertexRadiance`
// its cached outgoing radiance. Resolve and the shift both re-form the RGB
// integrand *in their own domain* from `rcVertexRadiance` — no per-domain `F`
// is ever stored, so none can go stale under reuse (step 2, D-134).
public bool isReconnectionSample(PathSample sample) {
    return pathKind(sample) == PATH_KIND_RECONNECTION;
}

// Reserved stable ids, mirroring `src/restir/identity.rs`. The environment
// (HDRI) is a reservoir candidate but not an instance, so it carries a fixed
// sentinel; instance ids begin above the reserved block. `LIGHT_ID_NONE` marks
// an empty or dropped reservoir.
public static const uint LIGHT_ID_ENVIRONMENT = 0;
public static const uint LIGHT_ID_NONE = 0xffffffff;

// An HDRI draw held in the unified reservoir: the reconnection vertex names the
// environment, and its `barycentrics` hold the 2D sampling coordinate
// `Environment::sample` inverts — not a point on a triangle. Storing the primary
// sample (rather than the direction) makes the reconnection exact: `restir_resolve`
// re-samples the same coordinate to reproduce the direction, radiance, and pdf
// bit for bit.
public bool isEnvironmentSample(PathSample sample) {
    // Guard on the kind first: a reconnection sample's `rcVertex.instance` is a
    // live TLAS instance id, which can legitimately be 0 — only an NEE sample's
    // instance carries the environment sentinel.
    return pathKind(sample) == PATH_KIND_NEE
        && sample.rcVertex.instance == LIGHT_ID_ENVIRONMENT;
}

public struct RestirScene {
    // Triangle-only power-alias table — the reservoir candidates that are
    // emitters. `lightCount` rows; the environment is drawn by the coin flip
    // below, not from here.
    public LightRecord* candidateLights;
    public uint lightCount;
    // p(a candidate is the environment) = env / (env + triangle power). The
    // triangle-only companion to `SceneTable.env.selectProb`, which is diluted
    // by the delta lights the reservoir excludes.
    public float envSelectProb;
    // TLAS custom index → stable light id: `restir_candidates` stamps a fresh
    // sample's light with this so it survives a later light add/delete.
    public uint* instanceToId;
    // Stable id → current TLAS custom index, or `LIGHT_NONE` if the light is
    // gone: `restir_resolve` translates a stored sample back, dropping it when
    // the light it named no longer exists.
    public uint* idToInstance;
    // The delta-only power-alias table (`lights::delta_table`) — the distant and
    // point lights the reservoir excludes (D-088). `restir_resolve` draws one
    // ∝ power and adds its exact connection to the film, outside the reservoir:
    // a delta carries no BSDF partner and no Monte-Carlo noise, so a reservoir
    // over it would only inject resampling variance into an already-exact term.
    public LightRecord* deltaLights;
    public uint deltaCount;
    uint pad0;
};
