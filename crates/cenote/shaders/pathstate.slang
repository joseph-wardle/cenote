// The path-state schema and stage-queue machinery shared by every wavefront
// kernel. This module and `wavefront.rs` are the two halves of a single
// definition: `Paths` mirrors the Rust `PathPool` field for field, and
// `QueueState` mirrors the queue-header layout. Adding a path-state field
// touches those two files and no kernel signature.
//
// Queues drive the engine. A kernel pushes a path index (or a self-contained
// shadow record) into the next stage's queue; that push is stream compaction
// and stage routing in one. The consuming stage is dispatched indirectly
// from the queue's own header, so no count ever crosses back to the host
// mid-wave.
//
// Determinism rule: queue push order varies run to run, so radiance is
// written only to path-owned or pixel-owned memory — never atomically
// accumulated. Each pixel has one path, so any one dispatch performs at
// most one read-modify-write per pixel, and the barriers between stages
// order the adds across dispatches into a fixed sequence. That is what
// makes renders bitwise reproducible, which the replay guarantee depends
// on.

// Threads per workgroup of every 1D path-stage kernel — must match
// `WORKGROUP_SIZE` in `wavefront.rs`.
public static const uint WORKGROUP_SIZE = 64;

// How far past a boundary crossing the next leg of a media walk starts.
// Relative, so it holds at any scene scale, and at least eight float32
// ulps wide, so re-testing the same triangle cannot return the crossing
// just made. Two boundaries closer together than this read as one — both
// walkers (the volume march and the camera resolve) must group everything
// inside the window into one event, because losing one crossing of such a
// pair leaves a medium the path is not in, or omits one it is.
public static const float BOUNDARY_NUDGE = 1e-6;

// The window a leg starting at `start` opens around a crossing — the one
// spelling both walkers share, so the epsilon logic cannot drift apart.
public float coincidenceWindow(float start)
{
    return start * BOUNDARY_NUDGE + BOUNDARY_NUDGE;
}

// The probe histogram's layout (measurement builds, `-DPROBES`) — the one
// Slang authority, mirrored by the `PROBE_*` constants in wavefront.rs
// that the readback indexes by. The volume stage bins scatter events by
// bounce in [0, PROBE_VOLUME_BINS); the walk bins exit lengths above that,
// its last two bins taken by the exitless deaths; the rejection, roulette,
// and tracker counters append above both halves.
public static const uint PROBE_VOLUME_BINS = 256;
public static const uint PROBE_WALK_BASE = PROBE_VOLUME_BINS;
public static const uint PROBE_WALK_LEAK_BIN = 2 * PROBE_VOLUME_BINS - 2;
public static const uint PROBE_WALK_CAP_BIN = 2 * PROBE_VOLUME_BINS - 1;
public static const uint PROBE_ENTRY_REJECT_BIN = 2 * PROBE_VOLUME_BINS;
public static const uint PROBE_EXIT_REJECT_BIN = PROBE_ENTRY_REJECT_BIN + 1;
public static const uint PROBE_ROULETTE_BIN = PROBE_EXIT_REJECT_BIN + 1;
public static const uint PROBE_TRACKER_COLLISION_BIN = PROBE_ROULETTE_BIN + 1;
public static const uint PROBE_TRACKER_NULL_BIN = PROBE_TRACKER_COLLISION_BIN + 1;
public static const uint PROBE_TRACKER_STEP_BIN = PROBE_TRACKER_NULL_BIN + 1;

// Where a path's ray stopped, in the re-evaluable form (instance +
// primitive + barycentrics) that shading recomputes from.
public struct Hit {
    public uint instance; // TLAS instance custom index, or INSTANCE_NONE
    public uint primitive; // triangle index within the instance's mesh
    public float2 barycentrics;
};

// `Hit.instance` for a ray that reached nothing. Only paths owed a medium
// event carry it — a segment that runs to infinity still has a medium to
// sample, so the volume stage needs to tell "ends at a surface" from
// "never ends", and reads the rest of the record only in the first case.
// Above the 24-bit custom-index range every real instance lives in.
public static const uint INSTANCE_NONE = 0xffffffff;

// A medium-set entry: an instance's TLAS custom index in the low 24 bits,
// flags above. The entry names the *instance* whose medium the path
// travels, not what fills it — identity is what entering and exiting
// compare, and two instances can share one medium. `STACK_EMPTY` is the
// all-ones index — a slot no instance can claim (custom indices are 24
// bits, and `raw_instance` in `accel.rs` asserts they stay below this
// value). `STACK_INTERIOR` marks a refractive interior, which *displaces*
// what surrounds it, as against a null-bounded volume, which *adds* to it
// (see `combineMedia`). `STACK_PRIORITY` is an interior's nesting
// priority, directly under `STACK_INTERIOR` so the one `max` in
// `interiorOf` finds the dominant interior and ranks interiors against
// each other at once. `STACK_SCATTERING` marks an interior whose medium
// scatters — baked by prep exactly where the derived σ_s is nonzero, and
// the single authority for everything that follows from that: intersect
// routes the segment to the volume stage by it, `combineMedia` resolves
// the segment's medium by it, and shade_surface withholds its closed-form
// absorption by it, so no segment can be weighted twice or zero times.
// Below `STACK_PRIORITY`, so among equal-priority overlapping interiors
// the max tie-breaks toward a scattering one — ties were already
// documented arbitrary, so this orders the undefined and nothing else.
//
// Built by `stackEntry` in scene.slang, which is where an instance's
// boundary word lives — the flags are insertion's to set, and this module
// only ever matches on the index.
public static const uint STACK_INDEX_MASK = 0xffffff;
public static const uint STACK_EMPTY = STACK_INDEX_MASK;
public static const uint STACK_SCATTERING = 1 << 24;
public static const uint STACK_PRIORITY_SHIFT = 25;
public static const uint STACK_PRIORITY = 63 << STACK_PRIORITY_SHIFT;
public static const uint STACK_INTERIOR = 1 << 31;

// How many media a path can be inside at once — one per lane of its stack
// entry, refractive interiors and null-bounded volumes sharing the four.
// Sharing is what lets an interior nest, which a lane of its own could
// never do, and the two kinds do not compete in practice: the scenes that
// overlap volumes are not the scenes that nest glass. Deeper overlap is not
// modelled — the fifth crossing is ignored, which leaves that medium's
// extent empty (and, since the set is what exits compare against, fills its
// far side instead).
public static const uint MEDIUM_SLOTS = 4;

// Bounces before this always survive — they carry most of the image's energy,
// and killing them early would just trade traced rays for noise. From here on,
// paths gamble survival against their throughput and the survivors are
// reweighted to keep the estimate unbiased. Integrator policy, not schema, but
// it lives with the other every-kernel constants.
public static const uint ROULETTE_START = 3;

// The SoA path pool: one pointer per logical field, indexed by path slot.
public struct Paths {
    // xyz = ray origin; w = how far the arriving segment ran, written by
    // intersect for the paths it routes to shade_volume and read there —
    // the tracer already has the distance, and recomputing the surface
    // point to recover it costs a geometry record and three vertex fetches
    // per volume path. Worth 16.8% of shade_volume and 2.4% of the frame on
    // a hazy sanmiguel; on a fog-filled cornell box, whose geometry never
    // leaves L2, the same trade measured slightly negative. It also drops
    // the spawn-point epsilon out of the transport, since this is the true
    // ray t. Dead in every other stage: raygen and both scatters write 0.
    public float4* origin;
    // xyz = unit ray direction; w = the ray's cosine against the camera
    // forward axis, written by raygen and read once at the first hit —
    // hit distance × w is the camera-plane z the depth AOV records, with
    // no camera data in any shading kernel's push constants (they have no
    // room). Scatters write 0: only camera rays have a camera to measure
    // against.
    public float4* direction;
    public uint* pixel; // the film pixel this path contributes to
    public Hit* hit; // written by intersect, read by both shading stages
    // xyz = the path's accumulated weight; w = the solid-angle pdf of the
    // scatter that produced this ray, which the next vertex's emission
    // needs for MIS — 0 where no light-sampling strategy competed at the
    // vertex that scattered: camera rays, and any vertex whose dominant
    // interior scatters, since a connection from inside one could not
    // refract through its boundary and next-event estimation is skipped
    // there. Both MIS consumers then weight the emission 1, which is
    // exactly the estimator that remains.
    public float4* throughput;
    // The media the path is inside: one [`MEDIUM_SLOTS`]-wide set,
    // allocated only for a scene that has interiors or volumes — so this
    // address is null in a media-free scene, and every access sits behind
    // the kernel flag for the kind it cares about. Slots are
    // interchangeable and unordered; which kind an entry is, is its
    // `STACK_INTERIOR` bit.
    //
    // Intersect seeds the whole set at bounce 0, so every later stage may
    // read it without asking what bounce it is — raygen cannot, its push
    // constants being full at Vulkan's guaranteed 128 bytes.
    //
    // Reach it through the accessors below rather than indexing this
    // directly: the sentinels and the store widths are theirs to keep.
    public uint4* stack;

    // The set as the stages read it. `live` is the kernel's flag: a scene
    // with no medium to be inside of never allocated the buffer. The load
    // is full width because the pool moves whole sectors regardless — it is
    // *stores* that cost bytes here.
    public uint4 media(uint path, bool live) {
        return live ? stack[path] : uint4(STACK_EMPTY);
    }

    // One crossing changes one slot, so that is what reaches memory: a
    // full-width store of this vector measured +2.9% frame on
    // glass-of-water, the pool being residency-bound.
    public void setSlot(uint path, uint slot, uint entry) {
        ((uint*)stack)[path * 4 + slot] = entry;
    }

    // The whole set at once, for the two writers that change every slot:
    // intersect's bounce-0 seed and the volume march's write-back.
    public void setMedia(uint path, uint4 set) { stack[path] = set; }

    // The segment length lane of `origin` — see the field above. A one-lane
    // store, like `setSlot`, for the same reason.
    public void setSegment(uint path, float length) {
        ((float*)origin)[path * 4 + 3] = length;
    }
};

// The refractive interior the path is inside, or `STACK_EMPTY` for none —
// the entry, flags and all. Interiors displace what surrounds them, so this
// alone decides what fills a segment when it is set.
//
// Three `max`es, not a search, which is what puts `STACK_INTERIOR` at the
// *top* bit: an interior entry outranks every volume entry and every empty
// slot, so the widest word in the set is an interior whenever the set holds
// one. The bits below it rank interiors against each other.
public uint interiorOf(uint4 set) {
    let widest = max(max(set.x, set.y), max(set.z, set.w));
    return (widest & STACK_INTERIOR) != 0 ? widest : STACK_EMPTY;
}

// Is the path inside this instance? Membership, not dominance: a hit on a
// surface the path is already inside *leaves* it, whichever of the media in
// the set happens to fill the segment.
public bool insideMedium(uint4 set, uint instance) {
    return any((set & uint4(STACK_INDEX_MASK)) == uint4(instance));
}

// The set after a crossing, and the one slot it changed — `MEDIUM_SLOTS`
// where the set was full and the crossing had to be dropped. The slot is
// returned because one crossing is one store, and the writer needs to know
// which.
public struct Crossing {
    public uint4 set;
    public uint slot;
};

// Cross `entry`'s boundary: it is in the set, so this crossing leaves it;
// it is absent, so this one enters, flags and all.
//
// Membership, not which way the triangle faces. A volume mesh authored
// inside-out — the winding mistake every USD pipeline makes at least once —
// still bounds exactly its own interior this way, where a front-face rule
// would fog the entire world outside it instead. The two agree on any
// closed, outward-wound mesh, which is what the shadow pass's signed
// distances need anyway.
//
// Both failures of that agreement are one-sided by construction. A crossing
// with no slot to go to is dropped, so its far face reads as an entry and
// the medium's *outside* fills instead (see `MEDIUM_SLOTS`); a crossing the
// march could not separate from the one before it does the same. Either way
// a path leaves carrying a medium it is no longer in, which the shadow pass
// clamps to no shadow rather than to one that amplifies.
public Crossing crossMedium(uint4 set, uint entry) {
    let instance = entry & STACK_INDEX_MASK;
    Crossing crossing;
    crossing.set = set;
    crossing.slot = MEDIUM_SLOTS;
    var free = MEDIUM_SLOTS;
    for (uint slot = 0; slot < MEDIUM_SLOTS; slot++) {
        if ((set[slot] & STACK_INDEX_MASK) == instance) {
            crossing.set[slot] = STACK_EMPTY;
            crossing.slot = slot;
            return crossing;
        }
        if (set[slot] == STACK_EMPTY && free == MEDIUM_SLOTS) {
            free = slot;
        }
    }
    if (free < MEDIUM_SLOTS) {
        crossing.set[free] = entry;
        crossing.slot = free;
    }
    return crossing;
}

// The wave's AOV targets, one indirection behind the shading kernels'
// push constants (shade_surface sits exactly at Vulkan's guaranteed 128
// bytes, so the block lives in memory and costs one pointer). The first
// three are per-pixel accumulators for the wave, zero-filled at wave
// start like radiance; every write is a pixel-owned add, keeping renders
// bitwise deterministic. `enabled` = 0 is the one-shot render path: no
// AOV buffers exist and the kernels skip every guide read and write.
public struct AovTable {
    public float4* albedo; // xyz = denoiser albedo guide
    public float4* normal; // xyz = world-space shading normal, post-map
    public float* depth;   // camera-plane z at the first hit; +inf on miss
    // The guides' feature throughput: the weight with which the
    // albedo/normal AOVs will record whatever surface the pixel's path
    // reaches next, tinted by every near-specular interface passed
    // through — mirrors record what they show. Weight zero means the
    // guides already recorded (the "written" flag folded into the value).
    // Logically path state, indexed by pixel — legal because a wave
    // carries exactly one path per pixel, and what freed the pool pointer
    // the push-constant budget couldn't fit. Written only by scatters; a
    // camera ray's guides are fully open, weight 1, which bounce 0 knows
    // without reading — so this never needs clearing between waves.
    public float4* guide;
    public uint enabled;
    public uint pad;
};

// A queue header. groupsX/Y/Z sit contiguous so the host can aim an
// indirect dispatch straight at them (byte offset 4) — the header *is* the
// stage's VkDispatchIndirectCommand. groupsY/Z hold constant 1.
public struct QueueState {
    public Atomic<uint> count;
    public Atomic<uint> groupsX;
    public uint groupsY;
    public uint groupsZ;
};

public struct Queue<Entry> {
    public QueueState* state;
    public Entry* entries;

    // Append one entry. The thread that claims the first slot of each
    // would-be workgroup also grows the indirect dispatch, so groupsX ends
    // at exactly ceil(count / WORKGROUP_SIZE) with no fix-up pass.
    public void push(Entry entry) {
        uint slot = state.count.add(1);
        if (slot % WORKGROUP_SIZE == 0) {
            state.groupsX.add(1);
        }
        entries[slot] = entry;
    }

    // How many entries the producing stage pushed. The tail workgroup of
    // an indirect dispatch overshoots; consumers bounds-check against this.
    public uint count() {
        return state.count.load();
    }
};

// A queued shadow ray: self-contained (not fields of the launching path),
// which keeps shade_surface fire-and-forget and is already the shape of a
// separate shadow-path pool once multi-candidate NEE wants one.
//
// The visibility test is by identity, never by a distance epsilon: a ray
// aimed at a point on triangle (light, primitive) hits that triangle only
// at that point (a segment crosses a plane once), so the connection is
// unoccluded iff the committed hit *is* the target triangle — exact even
// on closed emitters, where a ray toward a far-side sample hits the near
// side of the same instance first and must count as occluded. Connections
// with nothing to hit (environment, delta lights) carry LIGHT_NONE, which
// no instance's ID matches: they pass iff nothing commits within `tmax`
// (open for the environment and distant lights; the distance itself for
// point lights, where anything nearer is a real occluder).
// The bit of `ShadowRay.key` that separates the walk-exit dispatch's
// connections from the vertex connections a pixel makes in the same bounce.
public static const uint SHADOW_KEY_WALK = 1 << 8;

public struct ShadowRay {
    public float3 origin;
    public uint light; // target light's TLAS custom index, or LIGHT_NONE
    public float3 direction;
    public uint primitive; // target triangle within the light's mesh
    public float3 contribution; // unshadowed radiance, fully weighted
    public float tmax;
    public uint pixel;
    // The path that launched it — not to make this record depend on the
    // pool, but so the shadow stage can find the volumes the connection
    // starts inside. The launching stage may go on to change the set before
    // this one reads it, but only ever an *interior* slot, and the shadow
    // pass measures volumes alone: the membership it reads is the one at the
    // connection's own vertex.
    public uint path;
    // Which of the pixel's connections this is within the wave — its bounce,
    // plus which queue carried it. Nothing but the grid transmittance's seed
    // reads it, and it lives here rather than in the dispatch's push
    // constants because the two queues' dispatches share one.
    public uint key;
    public uint pad0;
};
