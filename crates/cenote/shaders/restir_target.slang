// Shared reservoir-reuse machinery: everything two or more reuse domains must
// evaluate *identically*, kept in one module so they can never drift into a
// silent bias (surface.slang). Four layers, top to bottom:
//
//   - the reconstructed shading `Surface` at a primary hit, rebuilt from the
//     path pool exactly as `restir_candidates` (and the path tracer) built it;
//   - the hybrid shift (steps 3a-3c, D-137/D-138/D-139): the pair criteria,
//     the shared reconnection end, the seed-replay walker, and the two shift
//     maps that state a stored indirect sample in another pixel's domain —
//     `shiftReconnection` (replay the prefix, reconnect at the stored x_k,
//     re-shade f_k) and `shiftReplay` (replay the whole path, re-draw the
//     terminal) — plus the un-baked outgoing radiance every re-shading
//     domain re-forms (`reconnectionOutgoing`);
//   - a stored sample's light connection and unshadowed target
//     p̂ = luminance(L·f·|cosθ|) at a surface (`connectSampleAt`,
//     `evalTarget`), the re-form the reuse stages and resolve share;
//   - the cross-domain shift-and-target block (step 5a, §4c): the per-sample
//     evaluations a reuse stage runs on a foreign *indirect* sample —
//     `shiftIntoDomain` (shift it here, price it here, visibility ray and
//     re-rooted sample included) and `targetInDomain` (a sample's unshadowed
//     target stated in a foreign domain, no ray).
//
// What stays with its caller, deliberately: the NEE arms of the reuse
// combines. Their visibility policy is per-stage (D-094) — spatial's
// canonical NEE target is visibility-aware (`targetVisAware`, in
// restir_spatial.slang), temporal's is unshadowed — a stage decision, not
// shared machinery. Nothing here owns a TLAS binding; the shifts and the
// block above them take the caller's as a parameter.

import color;
import lights;
import nee;
import openpbr;
import pathstate;
import reservoir_path;
import restir_scene;
import rng;
import scene;
import surface;
import textures;

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
    public uint instance;   // the hit's TLAS instance — the interior a refraction enters
    public uint thinWalled; // its material's thin-walled flag — a thin wall swaps no medium
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
    s.instance = hit.instance;
    s.thinWalled = material.thinWalled;
    s.valid = true;
    return s;
}

// A light sample reconnected to a shading point: the direction, radiance, and
// shadow-ray target a stored NEE `PathSample` (its reconnection vertex the light
// point) makes at `from`. The one place
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

// -- The pair criteria: a VARIANCE guard, not a bias guard (steps 3a/4) ------
// With f₂ un-baked (the shift re-shades the closure at x₂, D-131), reusing
// *any* reconnection vertex is unbiased — a sharp one is merely firefly-prone:
// its re-shaded lobe can be enormous where the source pdf was tiny. So the
// pair criteria gate quality, not correctness — since step 4 (D-140) as the
// dual ray footprint test (Enhanced Eq 5), replacing the classic roughness +
// distance floors: a pair (x_{k−1}, x_k) reconnects when the area each
// endpoint's lobe spreads over at the other end stays above a fixed fraction
// of the primary ray's own spread,
//
//   1 / (p_{k−1}(ω_{k−1}) · G(x_{k−1}→x_k)) ≥ rhs   forward — at the lock
//   1 / (p_k(ω_k)         · G(x_k→x_{k−1})) ≥ rhs   inverse — after the
//                                                   scatter out of x_k draws
//   rhs = (FOOTPRINT_C/100) · ‖x₀−x₁‖² · 4π / |⟨n₁, x̂₁x₀⟩|
//
// A sharp lobe over a long segment still covers area — reuse it; the same
// lobe over a short one concentrates — fireflies. Both tests run in
// multiply-form (rhs·p·G against 1) so a grazing primary's rhs = ∞ simply
// never locks; both use the marginal all-lobe pdfs `bsdfPdf`/`sampleBsdf`
// already return (supplemental §3). The inverse test lives ~60 lines past
// the lock (restir_candidates) as a *demote*, and is skipped for a
// diffuse-dominated x_k, whose continuation pdf ignores the arrival — exact
// for the at-lock NEE event, whose light draw does the same (footnote 6).
// A path with no accepted pair anywhere streams as the pure-replay kind
// (D-139), so every stored indirect sample still shifts.

// Enhanced §4's c: the fraction (c/100) of the primary footprint a
// reconnection footprint must reach. The /100 and the 4π stay explicit where
// the rhs forms (restir_candidates), so the constant reads as the paper's.
public static const float FOOTPRINT_C = 0.02;

// The lobe weight above which a sharp-capable group (metal / dielectric
// specular / transmission, or coat) counts as carrying the vertex — shared
// by the surviving roughness guard and the inverse test's diffuse skip.
public static const float SHARP_LOBE_WEIGHT = 0.25;

// The roughness floor of the one *surviving* single-vertex guard, at x_{k−1}
// only (Enhanced §4.2): the footprint test cannot see environment-segment
// lobes, parallax, or curvature, so the vertex the segment *leaves* still
// needs honest width. x_k needs none — reconnecting through a sharp vertex
// far away is the criterion's headline win.
public static const float RECONNECT_MIN_ROUGHNESS = 0.2;

// The x_{k−1} guard: diffuse-dominated is always rough enough; a vertex
// whose sharp-capable lobes carry weight qualifies only at their roughness
// floor.
public bool reconnectionRough(Material m) {
    let sharp = max(max(m.metalness, m.specularWeight), m.transmissionWeight);
    return (sharp < SHARP_LOBE_WEIGHT || m.specularRoughness >= RECONNECT_MIN_ROUGHNESS)
        && (m.coatWeight < SHARP_LOBE_WEIGHT || m.coatRoughness >= RECONNECT_MIN_ROUGHNESS);
}

// The inverse test's subject: only a vertex with sharp-capable lobe weight
// can concentrate its scatter pdf enough to fail Eq 5's second term; a
// diffuse-dominated x_k skips the test (footnote 6's exact cases and their
// near-diffuse neighbours alike — `reconnectionRough`'s weight shape).
public bool sharpCapable(Material m) {
    let sharp = max(max(m.metalness, m.specularWeight), m.transmissionWeight);
    return sharp >= SHARP_LOBE_WEIGHT || m.coatWeight >= SHARP_LOBE_WEIGHT;
}

// -- The reconnection end (steps 3a/3b, D-137/D-138) -------------------------
// The seam every domain evaluates identically: re-derive the reconnection
// segment's direction from the stored vertex (the direction-free discipline,
// D-134), guard it on the connecting surface, evaluate the segment's f·|cos|
// and BSDF density there, compute this domain's Jacobian half, and re-shade
// x_k through the re-derived direction (D-131). One function, shared by
// generation's context lock (restir_candidates) and the shift's replay end
// below — the bit-identity the same-pixel gate asserts, made structural.
// `prev*` is the surface the segment leaves — x₁ for k = 2, the last replayed
// (or walked) prefix vertex for k > 2 — in its *walk* shading (the drawn
// arrival direction), which replay reproduces bit for bit from the seed.
// Invalid when the re-derived direction dies on that surface or the technique
// could not have drawn it (density 0) — the support guard every domain
// shares, so a locked context is shiftable-by-construction at its own pixel.
public struct ReconnectionEnd {
    public float3 toRc;     // unit direction along the reconnection segment
    public float dist2;     // its squared length
    public float3 fkm1cos;  // f·|cosθ| of the segment at x_{k−1}
    public float reconPdf;  // its BSDF density — the balance-heuristic denominator
    public float partial;   // this domain's Jacobian half |cosθₖ|/d² (D-138)
    public Closure closure; // x_k, re-shaded through the re-derived direction
    public Frame frame;
    public float3 wo;
    public bool valid;
};

public ReconnectionEnd reconnectionEnd(SceneTable* scene, Hit rc, bool inside,
                                       float3 prevPosition, Closure prevClosure,
                                       Frame prevFrame, float3 prevWo,
                                       float3 prevNormal) {
    ReconnectionEnd e;
    e.toRc = float3(0.0);
    e.dist2 = 0.0;
    e.fkm1cos = float3(0.0);
    e.reconPdf = 0.0;
    e.partial = 0.0;
    e.wo = float3(0.0);
    e.valid = false;

    GeometryRecord mesh = scene.geometry[rc.instance];
    let xk = reconnectionPoint(mesh, rc);
    let delta = xk - prevPosition;
    e.dist2 = dot(delta, delta);
    if (!(e.dist2 > 0.0)) {
        return e;
    }
    // normalize(), not a hand-rolled rsqrt: every domain derives the segment
    // with the same instruction on the same vector, so they stay bit-identical.
    e.toRc = normalize(delta);
    let wiLocal = prevFrame.toLocal(e.toRc);
    let throughSurface = dot(e.toRc, prevNormal) < 0.0;
    if (wiLocal.z == 0.0 || (wiLocal.z < 0.0) != throughSurface) {
        return e;
    }
    e.fkm1cos = evalBsdf(prevClosure, wiLocal, prevWo) * abs(wiLocal.z);
    e.reconPdf = bsdfPdf(prevClosure, wiLocal, prevWo);
    if (!(any(e.fkm1cos > 0.0) && e.reconPdf > 0.0)) {
        return e;
    }
    // Re-shade x_k (D-131): rebuild its shading exactly as every other domain
    // does — same spawn, same normal map, same inside-the-instance IOR flip —
    // with wo this domain's own arriving direction.
    let uv = interpolateUv(mesh, rc.primitive, rc.barycentrics);
    let material = resolveMaterial(scene.materials[rc.instance], uv);
    let spawn = spawnPoint(mesh, rc, e.toRc);
    // This domain's Jacobian half, |cosθₖ|/d² of the segment (D-138): the
    // full shift Jacobian is the destination half over the sample's cached
    // source half — coinciding domains agree to the last-place bit, so the
    // same-pixel ratio is 1 within one ulp (D-125; the step-3b gate pins it
    // there — the reconstruction chain's fma contraction is the compiler's
    // per-kernel choice, so exact cross-kernel equality of d² is not
    // promisable, unlike the integrand's own algebra, which the gate holds
    // strictly bit-for-bit). Formed from the segment pieces above and the
    // re-shade's own geometric normal (the |·| makes its facing irrelevant),
    // deliberately *not* from a fresh reconstruction.
    e.partial = abs(dot(spawn.normal, e.toRc)) / e.dist2;
    if (!(e.partial > 0.0)) {
        return e;
    }
    var shading = spawn.shading;
    if (material.normalTexture != TEXTURE_NONE) {
        shading = mappedShadingNormal(mesh, rc, uv, material.normalTexture, spawn);
    }
    e.frame = Frame(shading);
    e.wo = e.frame.toLocal(-e.toRc);
    e.closure = resolveClosure(scene.bsdfTables, material, e.wo, inside);
    e.valid = true;
    return e;
}

// -- The replay walker (steps 3b/3c, D-138/D-139) ----------------------------
// The seed-replay half of the hybrid shift, shared by the reconnection
// shift's prefix and the pure-replay shift: draw the source's scatter at this
// domain's current vertex (the same pure function of (seed, dimension)
// generation used), trace it, and resolve the next vertex's walk shading.
// The state and the multiplication order — f·cos/pdf at the scatter, then the
// arriving segment's transmittance — mirror the generation walk exactly, so a
// same-pixel replay reproduces the stored chain bit for bit (the step-3b/3c
// gate). The walker never rolls roulette: the source's survivals are folded
// into its candidate weight at generation (D-138).
struct ReplayWalker {
    SpawnPoint spawn;     // the current vertex, as `spawnPoint` reconstructs it
    Frame frame;
    float3 wo;
    Closure closure;
    uint instance;
    bool thinWalled;
    uint medium;
    float3 prefix;        // Π f·cos/pdf · transmittance, roulette-free
    float3 segmentOrigin; // the last traced segment's origin…
    float3 direction;     // …and direction
    float scatterPdf;     // the last scatter's pdf — the SCATTER terminal's MIS half
    Hit hit;              // the last vertex reached (terminal emission re-derives from it)
};

ReplayWalker replayStart(Surface s) {
    ReplayWalker w;
    // Only the fields the first step reads are live here; `spawn.shading` and
    // `frontFace` are placeholders a step overwrites before any use (the
    // primary surface's frame already rides `s.frame`).
    w.spawn.position = s.position;
    w.spawn.below = s.below;
    w.spawn.normal = s.normal;
    w.spawn.shading = s.normal;
    w.spawn.frontFace = true;
    w.frame = s.frame;
    w.wo = s.wo;
    w.closure = s.closure;
    w.instance = s.instance;
    w.thinWalled = s.thinWalled != 0;
    w.medium = MEDIUM_NONE; // bounce 0 is a camera ray in vacuum
    w.prefix = float3(1.0);
    w.segmentOrigin = s.position;
    w.direction = float3(0.0);
    w.scatterPdf = 0.0;
    w.hit.instance = 0;
    w.hit.primitive = 0;
    w.hit.barycentrics = float2(0.0);
    return w;
}

static const uint REPLAY_DEAD = 0; // the replayed lobe died on this surface
static const uint REPLAY_HIT = 1;  // advanced to the next vertex
static const uint REPLAY_MISS = 2; // the segment escaped to the sky

uint replayStep(RaytracingAccelerationStructure tlas, SceneTable* scene,
                uint seed, uint bounce, inout ReplayWalker w) {
    let uLobe = replay_1d(seed, bounceDimension(bounce, BSDF_LOBE));
    let uDir = replay_2d(seed, bounceDimension(bounce, BSDF_DIR));
    let scatter = sampleBsdf(w.closure, w.wo, uLobe, uDir);
    let wiWorld = w.frame.toWorld(scatter.wi);
    if (!scatterExitsSurface(scatter, wiWorld, w.spawn.normal)) {
        return REPLAY_DEAD;
    }
    let fOverPdf = evalBsdf(w.closure, scatter.wi, w.wo)
                 * (abs(scatter.wi.z) / scatter.pdf);
    w.prefix *= fOverPdf;
    if (scatter.lobe == LOBE_TRANSMISSION && !w.thinWalled) {
        w.medium = w.medium == w.instance ? MEDIUM_NONE : w.instance;
    }
    w.segmentOrigin = scatter.wi.z < 0.0 ? w.spawn.below : w.spawn.position;
    w.direction = wiWorld;
    w.scatterPdf = scatter.pdf;

    RayDesc ray;
    ray.Origin = w.segmentOrigin;
    ray.Direction = wiWorld;
    ray.TMin = 0.0;
    ray.TMax = 1e30;
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(tlas, RAY_FLAG_NONE, 0xFF, ray);
    while (query.Proceed()) {
        query.CommitNonOpaqueTriangleHit();
    }
    if (query.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
        return REPLAY_MISS;
    }
    w.hit.instance = query.CommittedInstanceID();
    w.hit.primitive = query.CommittedPrimitiveIndex();
    w.hit.barycentrics = query.CommittedTriangleBarycentrics();

    GeometryRecord mesh = scene.geometry[w.hit.instance];
    let uv = interpolateUv(mesh, w.hit.primitive, w.hit.barycentrics);
    let material = resolveMaterial(scene.materials[w.hit.instance], uv);
    let spawn = spawnPoint(mesh, w.hit, wiWorld);
    if (w.medium != MEDIUM_NONE) {
        w.prefix *= interiorTransmittance(scene, w.medium, w.segmentOrigin,
                                          spawn.position);
    }
    var shading = spawn.shading;
    if (material.normalTexture != TEXTURE_NONE) {
        shading = mappedShadingNormal(mesh, w.hit, uv, material.normalTexture, spawn);
    }
    w.spawn = spawn;
    w.frame = Frame(shading);
    w.wo = w.frame.toLocal(-wiWorld);
    w.closure = resolveClosure(scene.bsdfTables, material, w.wo,
                               w.medium == w.hit.instance);
    w.instance = w.hit.instance;
    w.thinWalled = material.thinWalled != 0;
    return REPLAY_HIT;
}

// -- The hybrid shift of a reconnection sample (step 3b, D-138) --------------
// Map a stored reconnection sample into the domain rooted at surface `s`:
// replay the k−2 prefix bounces from the stored seed — the same draws
// generation made, at this domain's surfaces — then reconnect the last
// replayed vertex to the stored x_k and re-shade through the shared
// `reconnectionEnd`. k = 2 replays nothing and degenerates to the plain
// geometric reconnection off the primary surface. The Jacobian is this
// domain's half over the sample's cached source half: measure change on the
// reconnection segment only — the replayed prefix is an identity map in
// primary-sample space (its uniforms are copied), so it contributes Jacobian
// 1. At the same pixel F reproduces the stored integrand bit for bit and the
// Jacobian is 1 to the ulp — the D-125 invariant, pinned by the step-3b gate.
//
// The replay walker never rolls roulette: the source's survival probabilities
// are folded into its candidate weight — hence into W — at generation
// (D-138), so the replayed path exists exactly as far as the stored k says
// and a survived base path can never become a killed shifted path (§6's
// RR-replay trap, closed by construction). An `invalid` result — the pair
// failed the variance criteria, a replayed lobe died, the prefix escaped to
// the sky, or the reconnection end is unreachable — is the partial-shift
// convention (D-136): the shifted target is zero while the sample still
// counts in the pairwise MIS.
public struct ReconnectionShift {
    public float3 F;         // the destination-domain integrand, unshadowed
    public float jacobian;   // |cosθₖ/d²| here over the cached source half
    public float partial;    // this domain's half — the cachedJacobian to store on a win
    public float3 rayOrigin; // the reconnection segment's visibility-ray origin
    public float3 toRc;      // its unit direction
    public float tmax;       // its length ×1.001 — the identity shadow-test bound
    public bool valid;
};

public ReconnectionShift shiftReconnection(RaytracingAccelerationStructure tlas,
                                           SceneTable* scene, PathSample sample,
                                           Surface s) {
    ReconnectionShift r;
    r.F = float3(0.0);
    r.jacobian = 0.0;
    r.partial = 0.0;
    r.rayOrigin = float3(0.0);
    r.toRc = float3(0.0);
    r.tmax = 0.0;
    r.valid = false;
    if (!reconnectionShiftable(sample) || !(sample.cachedJacobian > 0.0)) {
        return r;
    }

    // -- Replay the prefix: the source's draws, this domain's surfaces --------
    // The shared walker mirrors the generation walk exactly, so a same-pixel
    // replay is bit-identical to the walk that stored the sample. A dead lobe
    // or an escaped segment leaves no vertex to reconnect from: shift
    // undefined.
    var w = replayStart(s);
    let k = reconnectionK(sample);
    for (uint bounce = 0; bounce + 2 < k; bounce++) {
        if (replayStep(tlas, scene, sample.initRandomSeed, bounce, w) != REPLAY_HIT) {
            return r;
        }
    }

    // -- Reconnect and re-shade through the shared end ------------------------
    let end = reconnectionEnd(scene, sample.rcVertex, reconnectionInside(sample),
                              w.spawn.position, w.closure, w.frame, w.wo,
                              w.spawn.normal);
    if (!end.valid) {
        return r;
    }
    let wi = end.frame.toLocal(unpackUnitDirection(sample.rcVertexWi));
    let lo = reconnectionOutgoing(end.closure, wi, end.wo, sample.rcVertexRadiance,
                                  sample.neeLightPdf, reconnectionTerminal(sample));
    r.F = w.prefix * end.fkm1cos * lo;
    r.partial = end.partial;
    r.jacobian = end.partial / sample.cachedJacobian;
    r.rayOrigin = dot(end.toRc, w.spawn.normal) < 0.0 ? w.spawn.below
                                                      : w.spawn.position;
    r.toRc = end.toRc;
    r.tmax = sqrt(end.dist2) * 1.001;
    r.valid = true;
    return r;
}

// -- The pure-replay shift (step 3c, D-139) ----------------------------------
// Map a stored no-pair path into the domain rooted at surface `s`: replay
// every scatter from the seed (bounces 0..terminalBounce−1 — the same draws
// generation made, at this domain's surfaces), then re-form the terminal
// event in its stored shape. An NEE terminal re-draws the vertex's connection
// from the same dims and hands the caller its shadow ray (`tmax > 0`); a
// SCATTER terminal's last segment finds its own light here — an emitter's
// front face, or the sky — so its value is complete and no ray is owed. The
// shift is a primary-sample-space identity (every uniform copied), so its
// Jacobian is exactly 1 and the caller applies none; the source's roulette
// survivals live in its candidate weight, never here (D-138), and the walk
// exists exactly as far as the stored shape says. A structural mismatch — a
// dead lobe, a mid-path escape, a non-emitter where the emitter stood — is
// the partial-shift convention (D-136): invalid, target zero, the sample
// still counts in the pairwise MIS. At the same pixel the replayed F
// reproduces the stored integrand bit for bit (the step-3c gate).
public struct ReplayShift {
    public float3 F;          // destination integrand; vis-free for an NEE terminal
    public float3 rayOrigin;  // the NEE terminal's shadow ray (tmax 0: none owed)
    public float3 direction;
    public float tmax;
    public uint shadowLight;  // its identity target, as `shadowVisibility` expects
    public uint shadowPrimitive;
    public bool valid;
};

public ReplayShift shiftReplay(RaytracingAccelerationStructure tlas,
                               SceneTable* scene, PathSample sample, Surface s) {
    ReplayShift r;
    r.F = float3(0.0);
    r.rayOrigin = float3(0.0);
    r.direction = float3(0.0);
    r.tmax = 0.0;
    r.shadowLight = LIGHT_NONE;
    r.shadowPrimitive = 0;
    r.valid = false;

    let terminal = replayTerminal(sample);
    let terminalBounce = replayTerminalBounce(sample);
    var w = replayStart(s);
    for (uint bounce = 0; bounce < terminalBounce; bounce++) {
        let step = replayStep(tlas, scene, sample.initRandomSeed, bounce, w);
        if (step == REPLAY_DEAD) {
            return r;
        }
        if (step == REPLAY_MISS) {
            // Only a SCATTER terminal's last segment may escape — the sky is
            // a light this path shape can end on. Anywhere else the replayed
            // path has no vertex to continue from.
            if (bounce + 1 < terminalBounce || terminal != RC_TERMINAL_SCATTER) {
                return r;
            }
            r.F = w.prefix * scene.env.radiance(w.direction)
                * envEscapeMisWeight(scene, w.direction, w.scatterPdf);
            r.valid = true;
            return r;
        }
    }

    if (terminal == RC_TERMINAL_SCATTER) {
        // The last replayed segment must land on an emitter's front face here.
        GeometryRecord mesh = scene.geometry[w.hit.instance];
        if (mesh.light == LIGHT_NONE || !w.spawn.frontFace) {
            return r;
        }
        let uv = interpolateUv(mesh, w.hit.primitive, w.hit.barycentrics);
        let material = resolveMaterial(scene.materials[w.hit.instance], uv);
        let neePdf = emitterHitNeePdf(scene, mesh, w.hit.primitive, w.spawn,
                                      w.segmentOrigin, w.direction);
        r.F = w.prefix * material.emission * powerHeuristic(w.scatterPdf, neePdf);
        r.valid = true;
        return r;
    }

    // The NEE terminal: re-draw the stored dims at the replayed vertex — the
    // whole connection (light choice, point, MIS, delta-ness) re-forms
    // locally, exactly as generation drew it.
    if (!(scene.lightCount > 0 || scene.env.selectProb > 0.0)) {
        return r;
    }
    let uLight = replay_1d(sample.initRandomSeed,
                           bounceDimension(terminalBounce, NEE_LIGHT));
    let uPoint = replay_2d(sample.initRandomSeed,
                           bounceDimension(terminalBounce, NEE_POINT));
    let nee = sampleNee(scene, w.spawn, w.frame, w.closure, w.wo, w.prefix,
                        LIGHT_SAMPLING_MIS, uLight, uPoint);
    if (!any(nee.unshadowed > 0.0)) {
        return r;
    }
    r.F = nee.unshadowed;
    r.rayOrigin = nee.origin;
    r.direction = nee.direction;
    r.tmax = nee.tmax;
    r.shadowLight = nee.light;
    r.shadowPrimitive = nee.primitive;
    r.valid = true;
    return r;
}

// -- The un-baked reconnection radiance (step 3a, D-137) ---------------------
// Lo(x₂ → the connecting surface) re-formed from the incident-side suffix: the
// re-shaded f₂ times the stored |cosθ| times the cached suffix, times the one
// MIS weight that involves x₂'s closure and so must re-evaluate per domain
// (`RC_TERMINAL_*`, restir_scene.slang). All in x₂'s local frame. Shared by
// the candidate walk (which streams its targets through the same formula on
// the same decoded ωₖ) and every re-shading domain — one copy, so generation
// and reuse can never drift.
public float3 reconnectionOutgoing(Closure closure, float3 wi, float3 wo,
                                   float3 suffix, float neeLightPdf, uint terminal) {
    var mis = 1.0;
    if (terminal == RC_TERMINAL_NEE) {
        mis = powerHeuristic(neeLightPdf, bsdfPdf(closure, wi, wo));
    } else if (terminal == RC_TERMINAL_SCATTER) {
        mis = powerHeuristic(bsdfPdf(closure, wi, wo), neeLightPdf);
    }
    return evalBsdf(closure, wi, wo) * abs(wi.z) * suffix * mis;
}

public Connection connectSampleAt(SceneTable* scene, RestirScene* restir,
                                  PathSample sample, float3 from) {
    Connection c;
    c.toLight = float3(0.0);
    c.radiance = float3(0.0);
    c.valid = false;
    c.shadowLight = LIGHT_NONE;
    c.shadowPrimitive = 0;
    c.shadowTmax = 1e30;

    let rc = sample.rcVertex;

    // A replay sample's reconnection fields are all zero — nothing here can
    // re-form it (that would re-trace the replay), and falling through would
    // misread it as an NEE sample. The stages that hold one trust its stored
    // F instead; reaching here is a caller bug, answered invalid (D-139).
    if (isReplaySample(sample)) {
        return c;
    }

    // A length-≥3 indirect sample: the reconnection segment connects `from` to
    // the stored geometry vertex x₂ (D-134). The direction re-forms from x₂'s
    // reconstructed point (bit-identical to candidate and every reusing domain),
    // the radiance re-shades f₂ against the cached incident suffix (D-137) in
    // *this* domain — the un-baked hybrid-shift form — and the visibility
    // target is x₂'s own triangle: an identity shadow test along the
    // reconnection segment, exactly as a light point's (connectTriangle). The
    // reconnection Jacobian is the caller's (restir_spatial): it needs both the
    // source and target surfaces, which a single-surface evaluation does not.
    // Only k = 2 re-forms off the primary surface this way: a deeper vertex
    // reconnects off a *replayed* prefix (`shiftReconnection`, D-138), and the
    // stages that hold such a sample trust its stored F instead — a k > 2
    // sample reaching here is a caller bug, answered with an invalid connection.
    if (isReconnectionSample(sample)) {
        if (reconnectionK(sample) != 2) {
            return c;
        }
        let mesh = scene.geometry[rc.instance];
        let x2 = reconnectionPoint(mesh, rc);
        let delta = x2 - from;
        let dist2 = dot(delta, delta);
        if (dist2 > 0.0) {
            // normalize(), not a hand-rolled rsqrt: the candidate stage derives
            // its tail direction with the same instruction on the same vector,
            // so the two stay bit-identical.
            c.toLight = normalize(delta);
            // Re-shade x₂ (D-131): rebuild its shading exactly as the walk did
            // — same spawn, same normal map, same inside-the-instance IOR flip
            // (the stored bit) — with wo₂ this domain's own arriving direction.
            let uv = interpolateUv(mesh, rc.primitive, rc.barycentrics);
            let material = resolveMaterial(scene.materials[rc.instance], uv);
            let spawn = spawnPoint(mesh, rc, c.toLight);
            var shading = spawn.shading;
            if (material.normalTexture != TEXTURE_NONE) {
                shading = mappedShadingNormal(mesh, rc, uv, material.normalTexture, spawn);
            }
            let frame = Frame(shading);
            let wo = frame.toLocal(-c.toLight);
            let closure = resolveClosure(scene.bsdfTables, material, wo,
                                         reconnectionInside(sample));
            let wi = frame.toLocal(unpackUnitDirection(sample.rcVertexWi));
            c.radiance = reconnectionOutgoing(closure, wi, wo, sample.rcVertexRadiance,
                                              sample.neeLightPdf,
                                              reconnectionTerminal(sample));
            c.valid = true;
            c.shadowLight = rc.instance;
            c.shadowPrimitive = rc.primitive;
            c.shadowTmax = sqrt(dist2) * 1.001;
        }
        return c;
    }

    // Otherwise the reconnection vertex is the light point (length-2 NEE base
    // case): its `instance` field carries the stable light id,
    // `primitive`/`barycentrics` the point on the emitter (or the env
    // sampling coordinate).
    if (isEnvironmentSample(sample)) {
        let e = scene.env.connect(rc.barycentrics);
        if (e.pdf > 0.0) {
            c.toLight = e.direction;
            c.radiance = e.radiance;
            c.valid = true;
        }
        return c; // env: nothing to hit — LIGHT_NONE, open tmax already set
    }

    // Translate the stable id back to the current instance; a light deleted
    // since the neighbour stored this sample resolves to nothing (drop).
    let instance = restir.idToInstance[rc.instance];
    if (instance == LIGHT_NONE) {
        return c;
    }
    let lightMesh = scene.geometry[instance];
    LightRecord record = restir.candidateLights[lightMesh.light + rc.primitive];
    var tri = connectTriangle(record, rc.barycentrics, from);
    if (tri.pdf > 0.0) {
        let emitter = scene.materials[instance];
        if (emitter.emissionTexture != TEXTURE_NONE) {
            tri.radiance *= sampleColor(
                emitter.emissionTexture,
                interpolateUv(lightMesh, rc.primitive, rc.barycentrics));
        }
        c.toLight = tri.direction;
        c.radiance = tri.radiance;
        c.valid = true;
        c.shadowLight = instance;
        c.shadowPrimitive = rc.primitive;
        c.shadowTmax = tri.tmax;
    }
    return c;
}

// The unshadowed target and the light connection at a surface, the shared core
// of every reuse target. `unshadowed` is p̂ = luminance(L·f·|cosθ|), zero for a
// sample the surface can't carry (below a horizon the base doesn't transmit
// through, or a degenerate connection); `connection` and `throughSurface` carry
// the shadow-ray geometry the visibility-aware target needs on top of it.
// `contribution` is the RGB L·f·|cosθ| the luminance was taken of — the §6.3
// chroma numerator (M6 step 6a): an NEE sample stores no own-domain F, so the
// vector-weight accumulation re-forms its colour here, from the exact product
// the scalar target already computes.
public struct TargetEval {
    public float unshadowed;
    public float3 contribution;  // unshadowed RGB L·f·|cosθ|; luminance(·) = unshadowed
    public Connection connection;
    public bool throughSurface;  // the light is on the far side of the surface
};

public TargetEval evalTarget(SceneTable* scene, RestirScene* restir,
                             Surface s, PathSample sample) {
    TargetEval t;
    t.connection = connectSampleAt(scene, restir, sample, s.position);
    t.throughSurface = false;
    t.unshadowed = 0.0;
    t.contribution = float3(0.0);
    if (!t.connection.valid) {
        return t;
    }
    let wiLight = s.frame.toLocal(t.connection.toLight);
    t.throughSurface = dot(t.connection.toLight, s.normal) < 0.0;
    if (wiLight.z == 0.0 || (wiLight.z < 0.0) != t.throughSurface) {
        return t;
    }
    t.contribution = t.connection.radiance
                   * evalBsdf(s.closure, wiLight, s.wo) * abs(wiLight.z);
    t.unshadowed = luminance(t.contribution);
    return t;
}

// The unshadowed target p̂ = luminance(L·f·|cosθ|) of a light sample at a
// surface: the own-domain target p̂ every reuse stage evaluates without a shadow
// ray. The visibility-aware canonical target adds one ray on top of the same
// `evalTarget` core (`targetVisAware`, in restir_spatial.slang).
public float targetUnshadowed(SceneTable* scene, RestirScene* restir,
                              Surface s, PathSample sample) {
    return evalTarget(scene, restir, s, sample).unshadowed;
}

// -- The cross-domain shift-and-target block (step 5a, §4c decision 1) -------
// The per-sample evaluations a reuse stage runs on an *indirect* sample —
// reconnection or replay — crossing into another pixel's (or, at step 5b,
// another frame's) domain: the shift maps above composed with their targets,
// visibility, and re-rooting, in the one place both stages read them. The NEE
// arms stay with their stages (see the module header).

// An indirect sample stated and priced in the destination domain: the shifted
// target the receiving reservoir resamples by, the sample's own-domain target
// for the pairwise ratios, the unshadowed destination target the finalize
// divides by on a win, and the re-rooted sample to store on that win.
// `IDefaultInitializable` so the local below can be default-constructed — a
// struct holding a `PathSample` needs the conformance spelled out.
public struct DomainShift : IDefaultInitializable {
    public float target;      // p̂_dest(shift(Y))·J, visibility folded in — 1 ray
    public float ownTarget;   // p̂_src(Y): the stored integrand (D-138/D-139)
    public float unshadowed;  // U_dest(shift(Y)) — the finalize divisor on a win
    public float backTarget;  // exactly what `targetInDomain(sample, s)` returns —
                              //   the same shift's unshadowed p̂·J, exposed so the
                              //   paired spatial split (step 7b) hands it to the
                              //   reciprocal partner instead of re-running the shift
    public PathSample sample; // carries this domain's F (and Jacobian half) onward
};

// The forward direction (D-134/D-135/D-138/D-139): map an indirect sample
// through its shift into the domain rooted at `s` and evaluate the receiving
// target there. A reconnection sample maps through `shiftReconnection` —
// prefix replay from the stored seed (k > 2), the re-derived reconnection off
// its end, the re-shaded f_k, the geometry-term Jacobian against the cached
// source half — with one visibility ray along the reconnection segment. A
// replay sample maps through `shiftReplay` — the whole path replayed from the
// seed, the terminal re-drawn from the same dims, a primary-sample-space
// identity with Jacobian 1 — with one ray along the re-drawn NEE connection
// (a SCATTER terminal's segments were traced concretely, so no ray is owed).
// The own-domain target is the stored integrand (D-138): generation formed F
// concretely in the source domain — visibility included — so it is trusted
// and spends nothing.
//
// An undefined shift — a replayed lobe died, the prefix escaped, the terminal
// shape mismatched — is the partial-shift convention (D-136): every output
// zero while the pair still counts in the caller's V/c_tot, so the MIS
// weights stay a fixed function of each sample. Folded into the cross-domain
// targets so the shipped defensive pairwise MIS (restir_mis) needs no
// kind-aware variant: it already compares c·p̂ across the pair. On a win the
// returned sample is the one to store onward: resolve trusts F for the
// indirect kinds, and a reconnection sample's cached half re-roots it in this
// domain (D-138/D-139). An NEE sample is the caller's own arm (D-094); one
// reaching here is a caller bug, answered with zero targets.
public DomainShift shiftIntoDomain(RaytracingAccelerationStructure tlas,
                                   SceneTable* scene, PathSample sample,
                                   Surface s) {
    DomainShift d;
    d.target = 0.0;
    d.ownTarget = 0.0;
    d.unshadowed = 0.0;
    d.backTarget = 0.0;
    d.sample = sample;
    if (isReconnectionSample(sample)) {
        let sh = shiftReconnection(tlas, scene, sample, s);
        if (sh.valid && sh.jacobian > 0.0) {
            d.unshadowed = luminance(sh.F);
            var vis = 0.0;
            if (d.unshadowed > 0.0) {
                RayDesc ray;
                ray.Origin = sh.rayOrigin;
                ray.Direction = sh.toRc;
                ray.TMin = 0.0;
                ray.TMax = sh.tmax;
                vis = shadowVisibility(tlas, scene, ray,
                                       sample.rcVertex.instance,
                                       sample.rcVertex.primitive);
            }
            d.target = d.unshadowed * vis * sh.jacobian;
            d.ownTarget = luminance(sample.F);
            d.sample.F = sh.F;
            d.sample.cachedJacobian = sh.partial;
        }
        // `targetInDomain`'s value from the shift already run: it guards only
        // on `valid` (no jacobian floor), so mirror that exactly.
        if (sh.valid) {
            d.backTarget = luminance(sh.F) * sh.jacobian;
        }
    } else if (isReplaySample(sample)) {
        let sh = shiftReplay(tlas, scene, sample, s);
        if (sh.valid) {
            d.unshadowed = luminance(sh.F);
            var vis = 1.0; // a SCATTER terminal's value is complete
            if (d.unshadowed > 0.0 && sh.tmax > 0.0) {
                RayDesc ray;
                ray.Origin = sh.rayOrigin;
                ray.Direction = sh.direction;
                ray.TMin = 0.0;
                ray.TMax = sh.tmax;
                vis = shadowVisibility(tlas, scene, ray,
                                       sh.shadowLight, sh.shadowPrimitive);
            }
            d.target = d.unshadowed * vis; // J = 1: the replay is a PSS identity
            d.ownTarget = luminance(sample.F);
            d.sample.F = sh.F;
        }
        // `targetInDomain`'s value: with J = 1 it is the unshadowed target
        // itself (zero when the shift is undefined, same as the field's init).
        d.backTarget = d.unshadowed;
    }
    return d;
}

// The reverse direction: a sample's *unshadowed* target stated in the foreign
// domain rooted at `s` — p̂(Y)·J through the same shift maps for the indirect
// kinds, the plain re-form for NEE. The direction the pairwise MIS needs of
// the canonical sample (what would this pixel's survivor be worth over
// there?), so no visibility ray is spent: own-domain targets never are. An
// undefined shift is the same partial-shift zero as above.
public float targetInDomain(RaytracingAccelerationStructure tlas,
                            SceneTable* scene, RestirScene* restir,
                            PathSample sample, Surface s) {
    if (isReconnectionSample(sample)) {
        let sh = shiftReconnection(tlas, scene, sample, s);
        return sh.valid ? luminance(sh.F) * sh.jacobian : 0.0;
    }
    if (isReplaySample(sample)) {
        let sh = shiftReplay(tlas, scene, sample, s);
        return sh.valid ? luminance(sh.F) : 0.0;
    }
    return targetUnshadowed(scene, restir, s, sample);
}
