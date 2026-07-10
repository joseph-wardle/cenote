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
// makes renders bitwise reproducible, which the replay guarantee (and
// later GRIS shift mappings) depend on.

// Threads per workgroup of every 1D path-stage kernel — must match
// `WORKGROUP_SIZE` in `wavefront.rs`.
public static const uint WORKGROUP_SIZE = 64;

// Where a path's ray stopped, in the re-evaluable form (instance +
// primitive + barycentrics) that shading recomputes from — and the form
// later reservoir-based passes must hold.
public struct Hit {
    public uint instance; // TLAS instance custom index
    public uint primitive; // triangle index within the instance's mesh
    public float2 barycentrics;
};

// No instance's TLAS custom index: a path traveling vacuum. 24 bits —
// the width of a custom index — so the medium packs beside the lobe tag.
public static const uint MEDIUM_NONE = 0xffffff;

// The scatter state packed into one word: the sampled-lobe tag (`LOBE_*`
// in openpbr.slang) in the top byte, the interior medium's instance in
// the low 24 bits. Packing keeps every kernel's push constants inside
// Vulkan's guaranteed 128 bytes.
public uint packPathState(uint lobe, uint medium) {
    return (lobe << 24) | (medium & 0xffffff);
}

public uint pathStateMedium(uint state) {
    return state & 0xffffff;
}

public uint pathStateLobe(uint state) {
    return state >> 24;
}

// The SoA path pool: one pointer per logical field, indexed by path slot.
public struct Paths {
    public float4* origin; // xyz = ray origin
    // xyz = unit ray direction; w = the ray's cosine against the camera
    // forward axis, written by raygen and read once at the first hit —
    // hit distance × w is the camera-plane z the depth AOV records, with
    // no camera data in any shading kernel's push constants (they have no
    // room). Scatters write 0: only camera rays have a camera to measure
    // against.
    public float4* direction;
    public uint* pixel; // the film pixel this path contributes to
    public Hit* hit; // written by intersect, read by shade_surface
    // xyz = the path's accumulated weight; w = the solid-angle pdf of the
    // scatter that produced this ray, which the next vertex's emission
    // needs for MIS (0 on camera rays: no light-sampling strategy competed).
    public float4* throughput;
    // The scatter's packed state (see packPathState): which technique
    // made this ray — the per-bounce record the AOV specular pass-through
    // ramp and M3's GRIS replay consume — and the instance whose interior
    // it travels ([`MEDIUM_NONE`] in vacuum), set when a refraction enters
    // a closed surface, cleared when one leaves, and read at the next
    // vertex for Beer–Lambert absorption over the segment. One medium
    // slot — nested media are a later, wider seam. Written only by
    // scatters: bounce 0 knows a camera ray starts in vacuum with no lobe,
    // so raygen never touches it.
    public uint* state;
};

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
    public uint pad0;
    public uint pad1;
    public uint pad2;
};
