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
