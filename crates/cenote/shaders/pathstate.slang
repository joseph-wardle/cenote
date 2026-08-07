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

// A medium-stack entry: an instance's TLAS custom index in the low 24
// bits, flags above. The entry names the *instance* whose medium the
// path travels, not what fills it — identity is what entering and exiting
// compare, and two instances can share one medium. `STACK_EMPTY` is the
// all-ones index — a path outside everything, and an entry no instance can
// claim (custom indices are 24 bits, and `raw_instance` in `accel.rs`
// asserts they stay below this value). `STACK_SCATTERING` marks an entry
// whose medium scatters (σ_s > 0), which routes a segment to shade_volume.
public static const uint STACK_INDEX_MASK = 0xffffff;
public static const uint STACK_EMPTY = STACK_INDEX_MASK;
public static const uint STACK_SCATTERING = 1 << 24;

// How many null-bounded volumes a path can be inside at once — the y, z,
// and w lanes of its stack entry. Overlap deeper than this is not modelled:
// the fourth volume's boundary is ignored, which leaves its extent unfogged
// (and, since the set is what exits compare against, leaves its far side
// fogged instead). Three is what the layout affords; the volumes production
// scenes overlap are one or two.
public static const uint VOLUME_SLOTS = 3;

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
    // needs for MIS (0 on camera rays: no light-sampling strategy competed).
    public float4* throughput;
    // The path's medium stack, allocated only for a scene that has
    // interiors or volumes — so this address is null in a media-free
    // scene, and every access sits behind the kernel flag for the half it
    // reads. Two independent halves, because the two kinds nest
    // differently and the surface stage only ever wants the first:
    //
    //   x       the innermost refractive interior — `top`/`setTop`
    //   y, z, w the null-bounded volumes the path is inside — `inside`/
    //           `setInside`, an unordered set
    //
    // Reach either through its accessors rather than indexing this
    // directly: the sentinels and the store widths are theirs to keep.
    public uint4* stack;

    // The innermost entry: the interior the ray is currently travelling,
    // set when a refraction enters a closed surface and cleared when one
    // leaves. Read at the next vertex for Beer–Lambert absorption, and by
    // intersect for STACK_SCATTERING.
    //
    // Both touch one lane, not the vector: `stack[path].x` would load a
    // whole uint4, and a full-width store measured +2.9% frame on
    // glass-of-water. The pool is residency-bound — bytes moved are the
    // cost. Written only by scatters, so bounce 0 must not read it: a
    // camera ray starts in vacuum and raygen leaves the entry unwritten.
    public uint top(uint path) { return ((uint*)stack)[path * 4]; }
    public void setTop(uint path, uint entry) { stack[path].x = entry; }

    // The interior the arriving segment travelled, or `STACK_EMPTY` for
    // open space — the question intersect and shade_volume both ask (and
    // shade_surface spells out again, for a reason recorded there).
    // `interiors` is the kernel's flag: a scene without them never
    // allocated the buffer. Bounce 0 must not read it either — a camera ray
    // starts in vacuum and raygen leaves the entry unwritten.
    public uint arriving(uint path, uint bounce, bool interiors) {
        return interiors && bounce != 0 ? top(path) : STACK_EMPTY;
    }

    // The segment length lane of `origin` — see the field above. A one-lane
    // store, like `setTop`: the pool is residency-bound, and a full float4
    // store measured +2.9% frame on glass-of-water.
    public void setSegment(uint path, float length) {
        ((float*)origin)[path * 4 + 3] = length;
    }

    // The null-bounded volumes the path is inside: an unordered set, since
    // overlapping volumes add and addition has no order. `volumes` is the
    // kernel's flag — a scene with no volume-bounding mesh never seeds
    // these lanes, so nothing may read them there. Intersect seeds them at
    // bounce 0, which is why every later stage can read without asking
    // what bounce it is.
    public uint3 inside(uint path, bool volumes) {
        return volumes ? stack[path].yzw : uint3(STACK_EMPTY);
    }

    // Three one-lane stores, for `setTop`'s reason: the pool is
    // residency-bound and the x lane belongs to the surface stage.
    public void setInside(uint path, uint3 set) {
        stack[path].y = set.x;
        stack[path].z = set.y;
        stack[path].w = set.z;
    }
};

// The volumes lanes of one stack entry — what `Paths::inside` reads, for a
// stage that carries the stack alone rather than the whole pool.
public uint3 insideOf(uint4* stack, uint path) {
    return stack[path].yzw;
}

// Crossing a null boundary flips the instance's membership in `set`: it is
// there, so this crossing leaves it; it is absent, so this one enters.
//
// Membership, not which way the triangle faces. A volume mesh authored
// inside-out — the winding mistake every USD pipeline makes at least once —
// still bounds exactly its own interior this way, where a front-face rule
// would fog the entire world outside it instead. The two agree on any
// closed, outward-wound mesh, which is what the shadow pass's signed
// distances need anyway.
//
// Both failures of that agreement are one-sided by construction. A fourth
// simultaneous volume has nowhere to go and is dropped, so its far face
// reads as an entry and its *outside* fogs (see `VOLUME_SLOTS`); a crossing
// the march could not separate from the one before it does the same. Either
// way a path leaves carrying a volume it is no longer in, which the shadow
// pass clamps to no shadow rather than to a medium that amplifies.
public uint3 crossVolume(uint3 set, uint instance) {
    var free = VOLUME_SLOTS;
    for (uint slot = 0; slot < VOLUME_SLOTS; slot++) {
        if (set[slot] == instance) {
            set[slot] = STACK_EMPTY;
            return set;
        }
        if (set[slot] == STACK_EMPTY && free == VOLUME_SLOTS) {
            free = slot;
        }
    }
    if (free < VOLUME_SLOTS) {
        set[free] = instance;
    }
    return set;
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
    // starts inside. Nothing writes that set between the launching stage
    // and this one (a surface scatter moves the interior lane, never these),
    // so the entry it reads is the one at the connection's own vertex.
    public uint path;
    public uint pad0;
    public uint pad1;
};
