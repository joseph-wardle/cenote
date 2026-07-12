// The direct-lighting instantiation of the generic reservoir: the concrete
// Sample type the DI reuse path holds, and the persistent per-pixel record
// its buffers store. `reservoir.slang` stays index-agnostic; this is where the
// abstract `Reservoir<Sample>` becomes a reservoir of *light samples*, and the
// four `restir_*` stages import it for that one concrete type. The host mirror
// (src/restir.rs) reproduces `StoredReservoir` field for field — the AoS
// buffer is an array of it.

import reservoir;

// One resampled direct-light candidate: a point on an emitter, held in the
// re-evaluable reconnection form (the `Hit` shape — instance-analogue +
// primitive + barycentrics). Storing the light *surface point*, not a
// solid-angle direction, is what makes the DI shift map the identity and the
// GRIS Jacobian exactly 1 (D-087) — the single most important estimator
// invariant, guarded here by construction.
//
// The light is named by its STABLE identity (the light-identity registry in
// src/restir.rs), never the volatile power-filtered, name-ordered TLAS custom
// index: a reservoir must survive a light being added or removed elsewhere in
// the scene, so reuse translates stable id → current dense slot at read time.
public struct DirectLightSample : IDefaultInitializable {
    public uint light;          // stable light id (see LIGHT_ID sentinels)
    public uint primitive;      // triangle index within the light's mesh
    public float2 barycentrics; // the sampled point on that triangle
};

// Reserved stable ids, mirroring `src/restir/identity.rs` (and `LIGHT_NONE` in
// lights.slang, the same 0xffffffff). The environment (HDRI) is a reservoir
// candidate too but is not an instance, so it carries a fixed sentinel;
// instance ids begin above the reserved block. `LIGHT_ID_NONE` marks an empty
// or dropped reservoir.
public static const uint LIGHT_ID_ENVIRONMENT = 0;
public static const uint LIGHT_ID_NONE = 0xffffffff;

// An HDRI draw held in the reservoir: `light` names the environment, and
// `barycentrics` hold the 2D sampling coordinate `Environment::sample` inverts
// — not a point on a triangle. Storing the primary sample (rather than the
// direction) makes the reconnection exact: `restir_resolve` re-samples the same
// coordinate to reproduce the direction, radiance, and pdf bit for bit.
public bool isEnvironmentSample(DirectLightSample sample) {
    return sample.light == LIGHT_ID_ENVIRONMENT;
}

// The persistent per-pixel reservoir record: {sample, W, c} = 24 B exactly.
// weightSum is deliberately absent — it is pass-local and rebuilt from scratch
// every pass (the one invariant `reservoir.slang` documents at length), so
// persisting it would double-count and silently bias. This struct is the byte
// layout src/restir.rs mirrors and the reservoir buffers are arrays of.
public struct StoredReservoir {
    public DirectLightSample sample; // 16 B — Y
    public float unbiasedWeight;     // 4 B — W_Y
    public float confidence;         // 4 B — M
};

// Persist a live reservoir, dropping the pass-local weightSum.
public StoredReservoir storeReservoir(Reservoir<DirectLightSample> r) {
    StoredReservoir s;
    s.sample = r.sample;
    s.unbiasedWeight = r.unbiasedWeight;
    s.confidence = r.confidence;
    return s;
}

// Read a persisted reservoir back for reuse. weightSum starts at zero, to be
// rebuilt by the reusing pass; restoring a stored weightSum instead would
// double-count and bias (see reservoir.slang).
public Reservoir<DirectLightSample> loadReservoir(StoredReservoir s) {
    Reservoir<DirectLightSample> r = reservoirInit<DirectLightSample>();
    r.sample = s.sample;
    r.unbiasedWeight = s.unbiasedWeight;
    r.confidence = s.confidence;
    return r;
}
