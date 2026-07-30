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

// -- The unified reservoir's three path kinds --------------------------------
// M6's unified reservoir (D-124) holds three shapes of sample, discriminated
// by the low two bits of `PathSample.pathFlags` — zero by default, so a
// zero-initialized sample reads as the NEE base case:
//
//   PATH_KIND_NEE (step 1, D-124) — a length-2 direct-lighting path whose
//     reconnection vertex `rcVertex` *is* the light surface point. The light's
//     stable identity lives in `rcVertex.instance` (never the volatile TLAS
//     custom index), `primitive`/`barycentrics` the emitter point (or the env
//     sampling coordinate). The light-id sentinels and the environment
//     predicate below interpret it.
//
//   PATH_KIND_RECONNECTION (step 2, D-134; un-baked at step 3a, D-137) — a
//     length-≥3 indirect path whose reconnection vertex is a *geometry* vertex
//     x₂ (the first indirect bounce, a real surface hit): `rcVertex.instance`
//     is a live TLAS instance, not a light id, and `rcVertexRadiance` caches
//     the *incident-side* suffix beyond f₂ — every domain re-shades f₂ at the
//     stored vertex to re-form the contribution (`reconnectionOutgoing`,
//     restir_target.slang). The NEE interpretation below does *not* apply to
//     it, so every predicate here guards on the kind first.
//
//   PATH_KIND_REPLAY (step 3c, D-139) — an indirect path with *no* qualifying
//     reconnection pair anywhere: it stores only its seed, its shape (the
//     terminal kind and bounce below), and its own-domain integrand `F`. A
//     shift replays every bounce at the destination from the seed and
//     re-draws the terminal event from the same dims (`shiftReplay`,
//     restir_target.slang) — a primary-sample-space identity, Jacobian 1.
//     The reconnection fields (`rcVertex`, `rcVertexWi`, `rcVertexRadiance`,
//     `neeLightPdf`, `cachedJacobian`) stay zero and are never read.
//
// The kinds live beside the NEE interpretation (rather than in the
// light-agnostic `reservoir_path.slang`) because it is exactly the discriminator
// that says whether that interpretation applies.
public static const uint PATH_KIND_NEE = 0;          // length-2 direct: rcVertex = light point
public static const uint PATH_KIND_RECONNECTION = 1; // length-≥3 indirect: rcVertex = x_k
public static const uint PATH_KIND_REPLAY = 2;       // length-≥3 indirect: seed replay only

public uint pathKind(PathSample sample) {
    return sample.pathFlags & 0x3;
}

// A length-≥3 indirect sample: `rcVertex` is a geometry vertex, `rcVertexRadiance`
// its cached incident-side suffix. Resolve and the shift both re-form the RGB
// integrand *in their own domain* by re-shading f₂ against it — no per-domain
// `F` is ever stored, so none can go stale under reuse (D-134, D-137).
public bool isReconnectionSample(PathSample sample) {
    return pathKind(sample) == PATH_KIND_RECONNECTION;
}

// A no-pair indirect path, reused by whole-path seed replay (D-139). Its
// own-domain target is `luminance(F)` and resolve trusts the stored `F`
// outright — re-forming it anywhere would re-trace the replay.
public bool isReplaySample(PathSample sample) {
    return pathKind(sample) == PATH_KIND_REPLAY;
}

// -- The indirect samples' packed flags (steps 3a-3c, D-137/D-138/D-139) -----
// Beyond the kind bits, an indirect sample packs its walk's facts into
// `pathFlags` — no re-layout, exactly the headroom D-128 designed in:
//
//   bits 2..3 — the TERMINAL kind: which suffix event ended the path *at* the
//     reconnection vertex, and so which MIS weight must re-evaluate against
//     the re-shaded closure at shift time (the reason `neeLightPdf` is kept
//     unpacked). RC_TERMINAL_NEE: the suffix is an NEE connection drawn at x₂
//     — MIS = pH(neeLightPdf, bsdfPdf₂). RC_TERMINAL_SCATTER: the suffix is
//     x₂'s own scatter reaching an emitter or the sky — MIS =
//     pH(bsdfPdf₂, neeLightPdf). RC_TERMINAL_BAKED: the terminal event lies
//     deeper (or is a delta connection, MIS-free), so its weight involves no
//     x₂ density and is already folded into `rcVertexRadiance` — MIS = 1.
//     A replay sample uses the same field for its terminal *redraw* shape:
//     RC_TERMINAL_NEE re-draws the vertex's NEE connection from the stored
//     dims, RC_TERMINAL_SCATTER re-traces the last scatter into whatever
//     light it finds (an emitter's front face, or the sky). Never BAKED —
//     the replay re-forms everything locally.
//
//   bit 4 — x₂ shades from *inside* its own instance: the candidate walk
//     reached it through a refraction into a closed surface, so the re-shade
//     resolves the closure at the inverted IOR, exactly as the walk did.
//     Reconnection only; a replay tracks its media from the destination
//     surface itself.
//
//   bit 5 — the pair criteria verdict (the variance guard, D-137): whether
//     the (x_{k−1}, x_k) pair was reconnection-worthy where the sample was
//     *drawn* — both endpoints rough enough, the segment long enough, and no
//     interior medium on it. Evaluated once, at generation, so the shift
//     needs no material lookups and the MIS weights stay a fixed function of
//     each sample (the D-136 partial-shift argument). Since step 3c a
//     reconnection sample only exists where a pair qualified, so the walk
//     always stamps it — kept as an explicit predicate the shift guards on.
//
//   bits 6..13 — k, the reconnection vertex's index (step 3b, D-138): 2 when
//     x₂ reconnects directly off the primary surface (the T3a shape), deeper
//     when the first qualifying pair sat further along the walk. A shift
//     replays the k−2 prefix bounces from the stored seed before it
//     reconnects; resolve re-forms k = 2 in its own domain but trusts the
//     stored `F` for k > 2 (re-forming would re-trace the replay). A replay
//     sample stores its TERMINAL BOUNCE here instead: the walk iteration its
//     terminal event fired at, so the shift replays exactly that many
//     scatters (bounces 0..bounce−1) before re-forming the terminal.
public static const uint RC_TERMINAL_NEE = 0;     // NEE drawn at x_k
public static const uint RC_TERMINAL_SCATTER = 1; // x_k's scatter found the light
public static const uint RC_TERMINAL_BAKED = 2;   // deeper (or delta): MIS folded

public uint reconnectionTerminal(PathSample sample) {
    return (sample.pathFlags >> 2) & 0x3;
}

public bool reconnectionInside(PathSample sample) {
    return (sample.pathFlags & 0x10) != 0;
}

public bool reconnectionShiftable(PathSample sample) {
    return (sample.pathFlags & 0x20) != 0;
}

// k — the reconnection vertex's index along the path (x₂ is k = 2).
public uint reconnectionK(PathSample sample) {
    return (sample.pathFlags >> 6) & 0xff;
}

// The packed `pathFlags` word of a reconnection sample. k caps at the 8-bit
// field — beyond any real path depth (max_bounces itself is byte-bounded).
public uint reconnectionFlags(uint terminal, bool inside, bool shiftable, uint k) {
    return PATH_KIND_RECONNECTION | (terminal << 2)
         | (inside ? 0x10 : 0) | (shiftable ? 0x20 : 0) | (min(k, 255) << 6);
}

// A replay sample's readings of the shared fields (D-139): the terminal
// redraw kind, and the walk bounce it fired at.
public uint replayTerminal(PathSample sample) {
    return (sample.pathFlags >> 2) & 0x3;
}

public uint replayTerminalBounce(PathSample sample) {
    return (sample.pathFlags >> 6) & 0xff;
}

// The packed `pathFlags` word of a pure-replay sample.
public uint replayFlags(uint terminal, uint bounce) {
    return PATH_KIND_REPLAY | (terminal << 2) | (min(bounce, 255) << 6);
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
