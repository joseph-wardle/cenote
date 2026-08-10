// The subsurface walk stage: paths whose surface scatter drew the
// subsurface technique arrive here from shade_surface with their origin
// already below the entry surface and their direction cosine-distributed
// into it. Each walks its instance's interior — free flights against the
// interned subsurface medium, Henyey–Greenstein turns — until a flight
// reaches the boundary, where the path leaves through a Lambertian
// transmission: one next-event connection from the exit point and a
// cosine-sampled continuation. The whole walk is one path bounce, so the
// depth cap never sees its interior events; the walk carries its own cap
// instead.
//
// The boundary hit *is* the exit, always: a walk that never scatters
// (σ_t = 0, or a thin chord) leaves on its first flight as pure diffuse
// translucency through the same code, and one that escapes an open shell
// dies as a leak rather than looping. Interior legs trace against the
// entry instance alone — embedded geometry is passed through, outside the
// model. The exit lobe is white and Fresnel-free: the walk's throughput
// already carries the medium's color, and the entry carried the
// interface's (1 − E) once, in `subsurfaceEntry`.
//
// Radiance bookkeeping matches shade_surface's: a path that dies here
// (leak, cap, spent throughput, a rejected exit) carries its pixel's one
// terminal alpha; one that continues pushes the ray queue and adds
// nothing. The guides recorded at the entry vertex, so this stage never
// touches the AOVs.

import lights;
import medium;
import nee;
import openpbr;
import pathstate;
import rng;
import scene;
import surface;

// The scene TLAS — the walk's legs trace against it.
[vk::binding(0, 0)]
RaytracingAccelerationStructure sceneTlas;

// Bit 25 of `Params.packed`, as in shade_volume: the path pool carries a
// medium set. The walk asks it one question — whether the entry vertex
// sits inside a scattering interior — which skips the exit's next-event
// connection exactly as it skipped the entry vertex's.
static const uint FLAG_SET = 1 << 25;

// Interior scatter events one walk may take before it is killed.
// Production materials self-terminate far below this, so a kill is
// expected never — the probes histogram is the check — while the
// α ≥ 0.995 stress regime trades a ~1–2% loss for boundedness.
static const uint WALK_CAP = 256;

#ifdef PROBES
// The walk's histogram bins sit above the volume stage's 256 — walk
// lengths at exit, cap kills in the last bin. See `PROBE_BINS` in
// `wavefront.rs`, whose layout these mirror.
static const uint PROBE_WALK_BASE = 256;
static const uint PROBE_WALK_CAP_BIN = 511;
#endif

struct Params {
    Paths paths;
    Queue<uint> walks; // this stage's input, pushed by shade_surface
    Queue<uint> rays;  // the next bounce's input: exits push here
    // The exit's next-event connection — a queue of its own, traced by the
    // round's second trace_shadow dispatch: the entry vertex's connection
    // for this same pixel is already in the main shadow queue, and one
    // dispatch may carry at most one ray per pixel (trace_shadow.slang).
    Queue<ShadowRay> shadows;
    SceneTable* scene;
    float4* radiance;         // the film's per-pixel accumulator for this wave
    uint sampleIndex;
    // bounce | maxBounces << 8 | lightSampling << 16 | the FLAG_* bits —
    // shade_surface's packing exactly.
    uint packed;
#ifdef PROBES
    Atomic<uint>* probes;
#endif
};

[vk::push_constant]
Params params;

// One leg: the next boundary of the entry instance along the ray, at any
// distance. Every other instance is passed through, and traversal is
// forced non-opaque because a BLAS baked with the opaque flag would
// otherwise auto-commit past the filter. Opacity itself is deliberately
// not consulted: the interior is bounded by geometry, not coverage.
bool traceLeg(float3 origin, float3 direction, uint instance, out Hit hit,
              out float distance) {
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.0;
    ray.TMax = 1e30;
    RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> query;
    query.TraceRayInline(sceneTlas, RAY_FLAG_FORCE_NON_OPAQUE, 0xFF, ray);
    while (query.Proceed()) {
        if (query.CandidateInstanceID() == instance) {
            query.CommitNonOpaqueTriangleHit();
        }
    }
    hit.instance = instance;
    hit.primitive = query.CommittedPrimitiveIndex();
    hit.barycentrics = query.CommittedTriangleBarycentrics();
    distance = query.CommittedRayT();
    return query.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void shade_subsurface(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.walks.count()) {
        return;
    }
    let bounce = params.packed & 0xff;
    let lightSampling = (params.packed >> 16) & 0xff;
    uint path = params.walks.entries[tid.x];
    uint pixel = params.paths.pixel[path];
    // The entry hit still names the instance — intersect only overwrites
    // it next bounce — and the entry gate guarantees `subsurface` is a
    // real medium index wherever the technique could be drawn.
    let instance = params.paths.hit[path].instance;
    var origin = params.paths.origin[path].xyz;
    var direction = params.paths.direction[path].xyz;
    var throughput = params.paths.throughput[path].xyz;
    GeometryRecord mesh = params.scene.geometry[instance];
    let interior = params.scene.media[mesh.subsurface];

    // The walk. Every draw is keyed by (event, draw slot) on the walk's
    // own hash stream, so every decision is replayable; the exit's draws
    // reuse the count it ends at, in slots the interior never spends.
    // `stake` is the walk's own attenuation — what the roulette gambles:
    // the arriving throughput was the entry roulette's business, and
    // gambling it again would over-kill paths that arrived dim.
    Hit exit;
    var events = 0u;
    var distance = 0.0;
    var stake = float3(1.0);
    for (;;) {
        if (!traceLeg(origin, direction, instance, exit, distance)) {
            // An open shell: the energy leaks — a death, never a loop.
            params.radiance[pixel] += float4(0.0, 0.0, 0.0, 1.0);
            return;
        }
        let event = sampleMedium(
            interior, distance,
            walkUniform2(pixel, params.sampleIndex, bounce, events, 0));
        throughput *= event.weight;
        stake *= event.weight;
        if (!event.scattered) {
            break; // the flight reached the boundary: this hit is the exit
        }
        events++;
        if (events >= WALK_CAP) {
#ifdef PROBES
            params.probes[PROBE_WALK_CAP_BIN].add(1);
#endif
            params.radiance[pixel] += float4(0.0, 0.0, 0.0, 1.0);
            return;
        }
        // Roulette — not the cap — is what ends a deep walk: absorption
        // decays the stake but never zeroes it. The outer loop's grace
        // and reweighting.
        if (events >= ROULETTE_START) {
            let survival = min(1.0, max(stake.r, max(stake.g, stake.b)));
            if (walkUniform2(pixel, params.sampleIndex, bounce, events, 5).x
                >= survival) {
                params.radiance[pixel] += float4(0.0, 0.0, 0.0, 1.0);
                return;
            }
            throughput /= survival;
            stake /= survival;
        }
        origin += event.distance * direction;
        direction = samplePhase(
            interior.g, -direction,
            walkUniform2(pixel, params.sampleIndex, bounce, events, 1)).wi;
    }

#ifdef PROBES
    params.probes[min(PROBE_WALK_BASE + events, PROBE_WALK_CAP_BIN - 1)].add(1);
#endif
    if (!any(throughput > 0.0)) {
        params.radiance[pixel] += float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // The exit: reconstruct the boundary hit. The spawn's normals face the
    // arriving — interior — ray, so the outward frame is their negation,
    // and `below` is the outside, where both the connection and the
    // continuation start.
    let spawn = spawnPoint(mesh, exit, direction);
    let outward = -spawn.shading;
    let frame = Frame(outward);

    let set = params.paths.media(path, (params.packed & FLAG_SET) != 0);
    let scatterInterior = (interiorOf(set) & STACK_SCATTERING) != 0;

    // The exit's next-event connection, drawn and weighted by the shared
    // `sampleNeeExit` (nee.slang). Skipped inside a scattering interior
    // exactly as at the entry vertex — the boundary occludes every
    // straight connection — conveyed onward as throughput.w = 0.
    if (!scatterInterior && lightSampling != LIGHT_SAMPLING_BSDF
        && (params.scene.lightCount > 0 || params.scene.env.selectProb > 0.0)) {
        let uLight = walkUniform2(pixel, params.sampleIndex, bounce, events, 3).x;
        let uPoint = walkUniform2(pixel, params.sampleIndex, bounce, events, 4);
        let nee = sampleNeeExit(params.scene, spawn, outward, throughput,
                                lightSampling, uLight, uPoint);
        if (any(nee.unshadowed > 0.0)) {
            params.shadows.push(shadowRay(nee, pixel, path));
        }
    }

    // The exit scatter: cosine about the outward shading normal — the
    // lobe samples its own density exactly, so the throughput carries
    // only what the walk cost it. A draw that fails to leave the real
    // triangle is the standard trade's sliver, and ends the path.
    let local = sampleCosineHemisphere(
        walkUniform2(pixel, params.sampleIndex, bounce, events, 2)).xyz;
    let wiWorld = frame.toWorld(local);
    if (local.z <= 0.0 || dot(wiWorld, spawn.normal) >= 0.0) {
        params.radiance[pixel] += float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // One bounce spent, however many events the walk took. w = the exit
    // cosine's solid-angle pdf, for the next vertex's emission MIS — or 0
    // where no connection competed.
    params.paths.origin[path] = float4(spawn.below, 0.0);
    params.paths.direction[path] = float4(wiWorld, 0.0);
    params.paths.throughput[path] =
        float4(throughput, scatterInterior ? 0.0 : local.z / PI);
    params.rays.push(path);
}
