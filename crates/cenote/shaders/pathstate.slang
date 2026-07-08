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

// The SoA path pool: one pointer per logical field, indexed by path slot.
public struct Paths {
    public float4* origin; // xyz = ray origin
    public float4* direction; // xyz = unit ray direction
    public uint* pixel; // the film pixel this path contributes to
    public Hit* hit; // written by intersect, read by shade_surface
    // xyz = the path's accumulated weight; w = the solid-angle pdf of the
    // scatter that produced this ray, which the next vertex's emission
    // needs for MIS (0 on camera rays: no light-sampling strategy competed).
    public float4* throughput;
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
public struct ShadowRay {
    public float3 origin;
    public uint light; // TLAS custom index of the target light instance
    public float3 direction;
    public uint pixel;
    public float3 contribution; // unshadowed radiance, fully weighted
    public float pad;
};
