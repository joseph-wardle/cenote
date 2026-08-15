// The volume stage: paths whose segment has a medium event to sample
// arrive here from intersect instead of shade_surface or shade_miss. Each
// draws a distance against its medium's extinction; a path that scatters
// before the segment ends turns through the phase function and sends its
// own next-event connection, and one that does not carries the segment's
// transmittance on to whatever intersect found.
//
// A segment that ends at a crossable boundary — a volume's surface, or an
// interface the dominant interior cuts away — is not over: the path
// crosses into different media, and the stage re-traces and draws again.
// That march is why this stage binds the acceleration structure, and why
// the routing cannot live in intersect's candidate loop: the crossings
// must be resolved in order, and `Proceed()` delivers candidates in none.
//
// A stage of its own rather than a branch in shade_surface, so that
// media-free scenes keep a byte-identical pass list and no scene pays
// register pressure for medium code it never runs.
//
// Radiance bookkeeping matches shade_surface's: every write is a
// pixel-owned add, and a path that ends here carries its pixel's one
// terminal alpha.

import lights;
import medium;
import nee;
import pathstate;
import rng;
import scene;
import tracking;

// The scene TLAS — the march re-traces against it.
[vk::binding(0, 0)]
RaytracingAccelerationStructure sceneTlas;

// The NanoVDB grid pool. Partially bound: only the tracking entry point
// reads it, and it is dispatched only for scenes whose prep wrote it.
[[vk::binding(5, 0)]]
StructuredBuffer<uint> gridPool;

// Bit 25 of `Params.packed`, as in shade_surface: the path pool carries a
// medium set. A scene whose only medium is the global one has none — there
// is nothing to be inside of — and this stage still runs.
static const uint FLAG_SET = 1 << 25;

// Legs one segment's march may make — a leg is one re-trace, and crossing a
// coincident group spends one leg however many boundaries stand in it; null
// boundaries and the false interfaces a dominant interior cuts away spend
// alike. Past it the march stops tracking media and runs straight through
// the rest: the bound on a pathological stack of shells, mirroring
// `TRANSPARENT_CAP` for the other unbounded traversal chain. Too-thin media
// beat both a black pixel and a wall that is not there.
static const uint MARCH_LEG_CAP = 8;

// The march's own budget of stochastic-opacity crossings, across every
// re-trace it makes: what one traversal is allowed, once per leg. It starts
// where intersect's chain left off, so no two of a path's crossings this
// bounce ever draw the same key.
static const uint MARCH_OPACITY_CAP = TRANSPARENT_CAP * (MARCH_LEG_CAP + 1);

// The march opens `coincidenceWindow` (pathstate.slang) around crossings:
// a leg leaving a *null* boundary starts a window's width behind it and
// crosses the whole group at once. Duplicated shells, volumes sharing a
// face plane, and a wall standing exactly on a boundary are all ordinary
// authoring.
//
// Refractive interfaces keep the plain forward nudge: an opaque one commits
// in hardware and never reaches a candidate loop to be grouped, so
// coincident *dielectric* surfaces remain outside the model.

// The five queues this stage touches, behind one pointer: consuming one and
// feeding four is more push-constant bytes than Vulkan guarantees. Their
// addresses are fixed for the wavefront's lifetime, so the block is written
// once — unlike the per-bounce scalars below.
struct VolumeQueues {
    Queue<uint> volumes;      // this stage's input, pushed by intersect
    Queue<uint> hits;         // segments that reached a surface
    Queue<uint> misses;       // segments that reached nothing
    Queue<uint> rays;         // the next bounce's input: scatters push here
    Queue<ShadowRay> shadows; // next-event connections, for trace_shadow
#ifdef PROBES
    // Scatter events binned by bounce. The bounce is uniform across a
    // dispatch, so the whole dispatch contends on one word — fine for a
    // measurement build.
    Atomic<uint>* probes;
#endif
};

struct Params {
    Paths paths;
    VolumeQueues* queues;
    SceneTable* scene;
    float4* radiance; // the film's per-pixel accumulator for this wave
    AovTable* aov;    // the wave's AOV targets; enabled = 0 skips them
    uint sampleIndex; // which sample of every pixel's sequence this wave is
    // bounce | maxBounces << 8 | lightSampling << 16 | the FLAG_* bits —
    // shade_surface's packing exactly.
    uint packed;
    // Which instances the march's re-traces see — the same mask intersect
    // traced this bounce with, since crossing a boundary does not make a
    // camera ray something else.
    uint rayMask;
    uint pad0;
};

[vk::push_constant]
Params params;

// The stage body, specialized like `connectionTransmittance`: HETERO = 0 is the
// homogeneous engine exactly, HETERO = 1 adds the null-collision tracker
// for sub-segments whose set holds a grid medium, and EMISSIVE = 1 has that
// tracker collect what the medium radiates on its way. A specialization and
// not a flag so a scene keeps the register allocation of the transport it
// actually runs; which entry point a wave records is `has_heterogeneous`
// and `has_emissive_media`, host-side.
void shadeVolume<let HETERO : int, let EMISSIVE : int>(uint3 tid) {
    if (tid.x >= params.queues.volumes.count()) {
        return;
    }
    let bounce = params.packed & 0xff;
    let maxBounces = (params.packed >> 8) & 0xff;
    let lightSampling = (params.packed >> 16) & 0xff;
    uint path = params.queues.volumes.entries[tid.x];
    uint pixel = params.paths.pixel[path];
    var hit = params.paths.hit[path];
    let direction4 = params.paths.direction[path];
    let direction = direction4.xyz;
    let origin4 = params.paths.origin[path];
    let origin = origin4.xyz;
    var throughput = params.paths.throughput[path];

    // Everything below measures along the ray intersect traced, from the
    // vertex it left. The march moves the ray's near end and never its
    // origin, so distances need no reconstruction and a refractive
    // interior's absorption is still the surface stage's one exp() over the
    // whole segment.
    var start = 0.0;
    // How far the current sub-segment runs: intersect parked the first
    // one's length in origin's w lane, and the sentinel says there is no
    // far end.
    var end = hit.instance == INSTANCE_NONE ? 1e30 : origin4.w;
    // The media the path is travelling. The dominant interior never moves
    // across the march — everything it crosses ranks strictly below it —
    // and a crossing at all implies the set exists.
    var set = params.paths.media(path, (params.packed & FLAG_SET) != 0);
    // Continues intersect's transparency chain past its own cap, so the
    // march's stochastic-opacity draws can never repeat one it already made.
    var crossing = TRANSPARENT_CAP;
    var leg = 0u;
    MediumEvent event;
    MediumRecord record;
    // Whether the sub-segment that ended the loop had a grid medium in its
    // set — which is when `record.g` and this albedo came from the tracker's
    // collision point rather than the record. Constant false under
    // HETERO = 0.
    var hadGrid = false;
    var trackedAlbedo = float3(0.0);
    for (;;) {
        // The first sub-segment keeps the Sobol dimension the bounce block
        // reserved; the rest draw from the hash stream, having no reserved
        // dimensions of their own. A segment no boundary interrupts
        // therefore draws exactly the samples an un-marched one would.
        let uDistance = leg == 0
                            ? sample_2d(pixel, params.sampleIndex,
                                        bounceDimension(bounce, VOLUME_DISTANCE))
                            : hashUniform2(pixel, params.sampleIndex, bounce, leg);
        let split = splitMedia<HETERO>(params.scene, set);
        record = split.homogeneous;
        hadGrid = HETERO != 0 && split.gridCount != 0;
        if (!hadGrid) {
            event = sampleMedium(record, end - start, uDistance);
        } else {
            let tracked = trackMedia<EMISSIVE>(params.scene, gridPool, split, origin,
                                               direction, start, end - start, pixel,
                                               params.sampleIndex, bounce, leg,
                                               uDistance);
            if (EMISSIVE != 0) {
                // Emitted light joins the pixel where it was emitted,
                // carrying the throughput that reached the sub-segment —
                // the estimator already carries everything inside it. No
                // next-event strategy competes for it, so no MIS weight
                // does either.
                params.radiance[pixel] += float4(throughput.xyz * tracked.emission, 0.0);
            }
            event.scattered = tracked.scattered;
            event.distance = tracked.distance;
            event.weight = tracked.weight;
            // The phase context is the collision point's, not the record's:
            // g is the σ_s-weighted mix of everything scattering *there*,
            // and the guide albedo likewise.
            record.g = tracked.g;
            trackedAlbedo = tracked.albedo;
#ifdef PROBES
            // Per sub-segment rather than per path: the totals are the same
            // and nothing has to be carried to the exits.
            params.queues.probes[PROBE_TRACKER_COLLISION_BIN].add(tracked.collisions);
            params.queues.probes[PROBE_TRACKER_NULL_BIN].add(tracked.nulls);
            params.queues.probes[PROBE_TRACKER_STEP_BIN].add(tracked.steps);
#endif
        }
        throughput.xyz *= event.weight;
        if (event.scattered) {
            break;
        }
        // The segment reached its far end. Whatever intersect (or the last
        // re-trace) found shades as ever, carrying every sub-segment's
        // transmittance. Origin and the previous scatter's pdf are left
        // alone — they describe the vertex before all this, which is still
        // where the next stage's MIS weight is measured from.
        //
        // A path whose throughput has gone to zero leaves here too, even at
        // a boundary it would have crossed: it contributes nothing wherever
        // it lands, and the surface stage is what records its pixel's
        // terminal alpha.
        //
        // What the march may cross is `crossable`: a volume's boundary, or
        // an interface the dominant interior cuts away. One exception, for
        // the false kind only: a crossing with no free slot is shaded, not
        // crossed — dropping it would leave the set denying a solid the
        // path is physically inside, and the far face would read as an
        // entry that outlives the real dominant. Shading walks an
        // overflowing scene back toward no-priority instead. A dropped
        // *null* crossing keeps marching, as it always has: too little fog
        // beats a wall that is not there.
        var toggled = Crossing(set, MEDIUM_SLOTS);
        var cross = false;
        // Whether the crossing about to be made is a null boundary, which is
        // what decides how the next leg opens — see the window below.
        var grouped = false;
        if (hit.instance != INSTANCE_NONE) {
            let mesh = params.scene.geometry[hit.instance];
            if (crossable(mesh, interiorOf(set))) {
                toggled = crossMedium(set, stackEntry(mesh, hit.instance));
                grouped = boundaryOf(mesh) == BOUNDARY_NULL;
                cross = grouped || toggled.slot < MEDIUM_SLOTS;
            }
        }
        if (!cross || !any(throughput.xyz > 0.0)) {
            params.paths.throughput[path] = throughput;
            // Only a march has anything to write back: without one the hit
            // is intersect's own and the membership is unchanged.
            if (leg != 0) {
                params.paths.hit[path] = hit;
                params.paths.setMedia(path, set);
            }
            if (hit.instance == INSTANCE_NONE) {
                params.queues.misses.push(path);
            } else {
                params.queues.hits.push(path);
            }
            return;
        }

        // Cross it, and trace on from just past it. Moving the near end
        // rather than respawning from the surface keeps every distance in
        // the original ray's parameter and needs no error bounds.
        leg++;
        set = toggled.set;
        start = end;

        // Past the cap the march stops *tracking* media but keeps the
        // geometry honest: the re-trace runs through the boundaries it would
        // have crossed instead of stopping at them, so a pathological stack
        // of shells reads as too little fog rather than as a wall that is
        // not there. It also ends the loop — the hit it commits cannot be
        // one of them.
        let exhausted = leg >= MARCH_LEG_CAP;
        // The dominant interior is invariant across every crossing the
        // march makes — a false one is strictly below it, a null one is no
        // interior at all — so reading it here is reading it at entry.
        // Recomputed per use rather than hoisted above the loop: the hoist
        // measured shade_volume +2.4% on an L2-resident fog scene, the
        // march being register-bound while three maxes are free.
        let interior = interiorOf(set);
        // Half the width of the nudge, and the leg's whole story about
        // coincidence: everything within it of the crossing just made is at
        // the *same* distance as far as this march can tell, because a
        // forward-nudged leg would step over all of it. Where the crossing
        // was a null boundary the leg opens a window's width *behind* it
        // instead and toggles that group in the candidate loop below — so
        // the group is exactly what the nudge would have lost, no wider and
        // no narrower, and one constant defines both.
        //
        // Behind, not from the crossing itself: a coincident face's `t` can
        // compute marginally nearer than the one that committed, and a
        // one-sided open would cut it.
        let window = coincidenceWindow(start);
        RayDesc ray;
        ray.Origin = origin;
        ray.Direction = direction;
        ray.TMin = grouped ? max(0.0, start - window) : start + window;
        ray.TMax = 1e30;
        RayQuery<RAY_FLAG_NONE> query;
        query.TraceRayInline(sceneTlas, RAY_FLAG_NONE, params.rayMask, ray);
        while (query.Proceed()) {
            let candidate = query.CandidateInstanceID();
            if (exhausted && crossable(params.scene.geometry[candidate], interior)) {
                continue;
            }
            // Inside the window: this candidate is coincident with the
            // crossing just made, and the leg opened behind it to find
            // exactly this.
            //
            // The surface that committed is dropped outright — it has been
            // crossed, and committing it again would end the leg where it
            // began and cross straight back. (A mesh contributing two faces
            // to one group loses the second this way, exactly as it does
            // without any of this.) A crossable partner is crossed here and
            // the leg runs on. Anything else — a wall standing on the
            // boundary — falls through and shades rather than being
            // skipped over by the forward nudge.
            //
            // Partners are never committed, so `TMax` still closes on the
            // first thing that ends the leg, and they are skipped before the
            // opacity draw below, so a scene with no coincident boundary
            // spends the same keys in the same order as it always has.
            //
            // Toggles commute — `crossMedium` matches on membership, never
            // on facing, and the dominant interior is invariant across the
            // march — so a group needs no order, only completeness.
            if (grouped && query.CandidateTriangleRayT() <= start + window) {
                if (candidate == hit.instance) {
                    continue;
                }
                let mesh = params.scene.geometry[candidate];
                if (crossable(mesh, interior)) {
                    set = crossMedium(set, stackEntry(mesh, candidate)).set;
                    continue;
                }
            }
            let opacity = opacityAt(params.scene, candidate,
                                    query.CandidatePrimitiveIndex(),
                                    query.CandidateTriangleBarycentrics());
            if (crossing >= MARCH_OPACITY_CAP
                || hashUniform(pixel, params.sampleIndex, bounce, crossing) < opacity) {
                query.CommitNonOpaqueTriangleHit();
            }
            crossing++;
        }
        if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            hit.instance = query.CommittedInstanceID();
            hit.primitive = query.CommittedPrimitiveIndex();
            hit.barycentrics = query.CommittedTriangleBarycentrics();
            end = query.CommittedRayT();
        } else {
            hit.instance = INSTANCE_NONE;
            hit.primitive = 0;
            hit.barycentrics = float2(0.0);
            end = 1e30;
        }
    }

#ifdef PROBES
    params.queues.probes[bounce].add(1);
#endif
    let distance = start + event.distance;
    let position = origin + distance * direction;
    // The scatter ends the segment, so the stale hit above is never read —
    // the next bounce's intersect overwrites it. Membership is not stale:
    // the connection below and the next bounce both measure from here.
    if (leg != 0) {
        params.paths.setMedia(path, set);
    }
    let aovOn = params.aov.enabled != 0;
    if (aovOn) {
        if (bounce == 0) {
            // First vertex: camera-plane z, via the cosine raygen stashed
            // in direction.w. A pixel that sees only fog has its depth here
            // or nowhere.
            params.aov.depth[pixel] += distance * direction4.w;
        }
        // The guides close on the medium: its single-scattering albedo is
        // what a denoiser should divide out here, and a scattering point
        // has no surface, so — as with the environment — no normal.
        let guide = bounce == 0 ? float3(1.0) : params.aov.guide[pixel].xyz;
        if (any(guide > 0.0)) {
            let albedo = hadGrid ? trackedAlbedo
                                 : record.sigmaS / max(record.sigmaT, 1e-9);
            params.aov.albedo[pixel] += float4(guide * albedo, 0.0);
        }
        params.aov.guide[pixel] = float4(0.0);
    }

    // The depth cap: the path ends at this vertex, so this is its pixel's
    // terminal add. No next-event estimation here — a connection from this
    // vertex would be one segment longer than any path phase sampling can
    // trace, and the two strategies must cover the same path lengths.
    if (bounce + 1 >= maxBounces) {
        params.radiance[pixel] += float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // Next-event estimation from inside the medium, drawn and weighted by
    // `sampleNeeVolume` (nee.slang); the connection's own transmittance is
    // trace_shadow's to apply — ratio-tracked where it crosses a grid, so a
    // scatter inside a cloud connects to the lights like any other. Skipped
    // wherever the dominant interior scatters: a connection from inside one
    // cannot refract through its boundary, and the boundary occludes every
    // straight one — so directional sampling is the only strategy, conveyed
    // to the next vertex as throughput.w = 0. The dominant is invariant
    // across the march (see the loop), so reading it here is reading it at
    // entry.
    let skipNee = (interiorOf(set) & STACK_SCATTERING) != 0;
    let wo = -direction;
    if (!skipNee && lightSampling != LIGHT_SAMPLING_BSDF
        && (params.scene.lightCount > 0 || params.scene.env.selectProb > 0.0)) {
        let uLight = sample_1d(pixel, params.sampleIndex,
                               bounceDimension(bounce, NEE_LIGHT));
        let uPoint = sample_2d(pixel, params.sampleIndex,
                               bounceDimension(bounce, NEE_POINT));
        let nee = sampleNeeVolume(params.scene, position, record.g, wo,
                                  throughput.xyz, lightSampling, uLight, uPoint);
        if (any(nee.unshadowed > 0.0)) {
            params.queues.shadows.push(shadowRay(nee, pixel, path, bounce));
        }
    }

    // Turn through the phase function. HG samples its own density exactly,
    // so f/pdf is 1 and the throughput carries only what the distance
    // sample cost it.
    let uPhase = sample_2d(pixel, params.sampleIndex,
                           bounceDimension(bounce, BSDF_DIR));
    let scatter = samplePhase(record.g, wo, uPhase);
    var scattered = throughput.xyz;

    var survival = max(scattered.r, max(scattered.g, scattered.b));
    if (survival <= 0.0) {
        params.radiance[pixel] += float4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    if (bounce >= ROULETTE_START) {
        survival = min(1.0, survival);
        if (sample_1d(pixel, params.sampleIndex,
                      bounceDimension(bounce, RR)) >= survival) {
            params.radiance[pixel] += float4(0.0, 0.0, 0.0, 1.0);
            return;
        }
        scattered /= survival;
    }

    // The path continues from inside the same media — a scatter crosses no
    // boundary, so neither half of the stack moves. Direction.w is zeroed:
    // only camera rays have a camera to measure depth against. w = 0 where
    // next-event estimation was skipped above: no strategy competed, so
    // the next vertex's emission keeps weight 1.
    params.paths.origin[path] = float4(position, 0.0);
    params.paths.direction[path] = float4(scatter.wi, 0.0);
    params.paths.throughput[path] =
        float4(scattered, skipNee ? 0.0 : scatter.pdf);
    params.queues.rays.push(path);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void shade_volume(uint3 tid: SV_DispatchThreadID) {
    shadeVolume<0, 0>(tid);
}

// Dispatched instead of (never beside) `shade_volume` when the scene holds
// a grid medium.
[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void shade_volume_hetero(uint3 tid: SV_DispatchThreadID) {
    shadeVolume<1, 0>(tid);
}

// And instead of that one when a grid medium carries a temperature field.
[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void shade_volume_emissive(uint3 tid: SV_DispatchThreadID) {
    shadeVolume<1, 1>(tid);
}
