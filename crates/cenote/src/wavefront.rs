//! The wavefront engine: `SoA` path state, GPU stage queues, and the
//! indirect-dispatch stage chain — the renderer's core.
//!
//! One wave traces one sample for every pixel of a target. The host records
//! the fixed stage sequence — raygen, then per bounce intersect →
//! (`shade_miss` | `shade_surface`) → `trace_shadow` — into a single
//! submission; `shade_surface` pushes scattered paths back onto the ray
//! queue and next-event connections onto the shadow queue, so the recorded
//! per-bounce round is the path tracer's bounce loop. Stages talk through
//! GPU queues: a kernel pushes surviving paths into the next stage's
//! queue, and every stage after raygen is dispatched indirectly from its
//! queue's own header, so no path count ever crosses back to the host
//! mid-wave. Termination is implicit — a path that pushes nothing is done
//! — and a wave whose paths all die early just dispatches empty rounds
//! until the recording runs out.
//!
//! In `ReSTIR` mode a reservoir chain is recorded into bounce 0 between
//! intersect and `shade_surface` — candidates, temporal, the spatial
//! gather/combine pair, resolve — and owns the primary hit's whole
//! non-delta path integral (D-134). Its reservoirs carry whole light
//! paths (`reservoir_path.slang`): candidates streams M light samples
//! plus every event of one BSDF walk, the reuse stages fold in last
//! frame's and the neighbours' path reservoirs through the hybrid shift,
//! and resolve shades the survivor. `shade_surface` keeps only the
//! directly visible emission there and pushes no continuation, so a
//! `ReSTIR` wave records bounce 0 alone (`record_wave` is the sequence,
//! pass by pass).
//!
//! Radiance starts the wave zero-filled and every kernel write is a plain
//! add: emission and shadow-ray contributions land per bounce, and each
//! path's terminal add carries alpha 1, so "every pixel finished exactly
//! once" stays checkable. Any one dispatch touches a pixel at most once
//! (one path per pixel), and the barriers between passes order the adds —
//! which is what keeps renders bitwise deterministic.
//!
//! The path pool is fixed capacity; a target with more pixels is walked in
//! pool-sized pixel ranges within the same submission. Path state is `SoA` —
//! one buffer per logical field — defined once, here and in
//! `shaders/pathstate.slang` ([`PathPool`] ↔ `struct Paths`): adding a
//! field touches those two files and no kernel signature.

use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use crate::error::Result;
use crate::gpu::{
    Bindings, Buffer, ComputePipeline, Context, MemoryLocation, Pass, PassTimer, SceneBindings,
};
use crate::scene::{Scene, ray_mask};
use crate::shaders::{Kernel, Kernels};
use crate::stats::PassTimings;

/// Threads per workgroup of every 1D path-stage kernel — must match
/// `WORKGROUP_SIZE` in `shaders/pathstate.slang`.
const WORKGROUP_SIZE: u32 = 64;

/// The stage queues, in header order within [`Queues::headers`].
mod queue {
    pub const RAY: u64 = 0;
    pub const HIT: u64 = 1;
    pub const MISS: u64 = 2;
    pub const SHADOW: u64 = 3;
    pub const COUNT: u64 = 4;
}

/// Byte size of one queue header — `struct QueueState` in
/// `shaders/pathstate.slang`: `{count, groupsX, groupsY, groupsZ}`, the
/// last three doubling as the stage's `VkDispatchIndirectCommand`.
const QUEUE_HEADER_SIZE: u64 = 16;

/// Byte offset of that indirect command within a header.
const INDIRECT_OFFSET: u64 = 4;

/// Byte size of one `ShadowRay` record (`shaders/pathstate.slang`).
const SHADOW_RAY_SIZE: u64 = 64;

/// The path pool's field-buffer addresses — `struct Paths` in
/// `shaders/pathstate.slang`, embedded in every stage's push constants.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PathsAddrs {
    origin: vk::DeviceAddress,
    direction: vk::DeviceAddress,
    pixel: vk::DeviceAddress,
    hit: vk::DeviceAddress,
    throughput: vk::DeviceAddress,
    state: vk::DeviceAddress,
}

/// A queue as kernels see it — `struct Queue<T>` in
/// `shaders/pathstate.slang`: header address + entry-array address.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct QueueAddrs {
    state: vk::DeviceAddress,
    entries: vk::DeviceAddress,
}

/// Push constants for the raygen kernel; mirrors `struct Params` in
/// `shaders/raygen.slang`. As in every kernel, the scalars after each
/// `Vec3` sit in what std430 would otherwise spend on padding — field
/// order is layout. Raygen names the four path fields it writes rather
/// than embedding [`PathsAddrs`]: camera rays own the defaults for the
/// rest, and the trimmed block stays inside Vulkan's guaranteed 128
/// push-constant bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RaygenParams {
    origin: vk::DeviceAddress,
    direction: vk::DeviceAddress,
    pixel: vk::DeviceAddress,
    throughput: vk::DeviceAddress,
    rays: QueueAddrs,
    /// Which sample of every pixel's sequence this wave traces.
    sample_index: u32,
    /// Thin-lens radius, meters; 0 takes the pinhole path. When open,
    /// the basis below arrives pre-scaled to the focal plane.
    aperture_radius: f32,
    /// With the two scalars above, these square the block off to 16
    /// bytes, so the `Vec3`s land on their required alignment.
    width: u32,
    height: u32,
    camera_position: Vec3,
    /// First pixel of this range.
    base: u32,
    camera_right: Vec3,
    /// Paths in this range.
    count: u32,
    camera_up: Vec3,
    _pad0: u32,
    camera_forward: Vec3,
    _pad1: u32,
}

/// Push constants for the intersect kernel (`shaders/intersect.slang`).
/// One instance per bounce: camera rays (bounce 0) trace with the camera
/// visibility bit, every later bounce with all bits — and the stochastic
/// transparency stream is keyed by the bounce, so a path's crossings stay
/// independent from round to round.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct IntersectParams {
    paths: PathsAddrs,
    rays: QueueAddrs,
    hits: QueueAddrs,
    misses: QueueAddrs,
    /// Device address of the scene table — opacity lives in the materials.
    scene: vk::DeviceAddress,
    /// Which instances these rays see — a [`ray_mask`] value.
    ray_mask: u32,
    /// Which sample of every pixel's sequence this wave traces.
    sample_index: u32,
    /// Which bounce these rays leave from.
    bounce: u32,
    _pad0: u32,
}

/// Push constants for the miss-shading kernel (`shaders/shade_miss.slang`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShadeMissParams {
    paths: PathsAddrs,
    misses: QueueAddrs,
    /// Device address of the scene table — escapes read the environment.
    scene: vk::DeviceAddress,
    /// Device address of the wave's per-pixel radiance target (`float4*`).
    radiance: vk::DeviceAddress,
    /// Device address of the wave's [`AovTable`] — escapes close the
    /// denoiser guides and stamp first-hit misses' depth.
    aov: vk::DeviceAddress,
    /// Which strategies reach the lights — a [`LightSampling`] as `u32`.
    light_sampling: u32,
    /// Explicit tail padding to the struct's 8-byte alignment (`Pod` forbids the
    /// implicit padding a lone trailing `u32` would leave).
    _pad0: u32,
}

/// Push constants for the surface-shading kernel
/// (`shaders/shade_surface.slang`). One instance per bounce — the bounce
/// inside `packed` is the only field that varies.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShadeSurfaceParams {
    paths: PathsAddrs,
    hits: QueueAddrs,
    /// The next bounce's input: scattered paths push themselves back here.
    rays: QueueAddrs,
    /// Next-event connections, consumed by this round's `trace_shadow`.
    shadows: QueueAddrs,
    /// Device address of the scene table (geometry, materials, lights,
    /// the closure's lookup tables).
    scene: vk::DeviceAddress,
    radiance: vk::DeviceAddress,
    /// Device address of the wave's [`AovTable`].
    aov: vk::DeviceAddress,
    /// Which sample of every pixel's sequence this wave traces.
    sample_index: u32,
    /// `bounce | max_bounces << 8 | light_sampling << 16` — see
    /// [`pack_shade_surface`]. Packed because this block sits exactly at
    /// Vulkan's guaranteed 128 push-constant bytes: the AOV pointer's
    /// 8 bytes come out of these three small scalars.
    packed: u32,
}

/// Pack `ShadeSurfaceParams::packed`, mirrored by the unpack at the top of
/// `shade_surface.slang`. Both byte-wide fields are asserted in range by
/// [`Wavefront::new`]. `restir` sets bit 24 — `ReSTIR` mode, where the unified
/// reservoir owns the whole primary-hit path integral (D-134): the kernel keeps
/// only the directly visible emission and the guides at bounce 0, pushing no
/// continuation, so `ReSTIR` waves record bounce 0 alone.
fn pack_shade_surface(
    bounce: u32,
    max_bounces: u32,
    light_sampling: LightSampling,
    restir: bool,
) -> u32 {
    bounce | max_bounces << 8 | (light_sampling as u32) << 16 | (u32::from(restir) << 24)
}

/// Push constants for the `ReSTIR` initial-RIS stage
/// (`shaders/restir_candidates.slang`). Runs at bounce 0 only; dispatched
/// indirectly from the hit queue it reads but does not consume.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RestirCandidatesParams {
    paths: PathsAddrs,
    hits: QueueAddrs,
    scene: vk::DeviceAddress,
    /// The `RestirScene` slice — candidate table, env coin flip, identity remap.
    restir: vk::DeviceAddress,
    /// This wave's per-pixel reservoirs (curr), written here.
    reservoirs: vk::DeviceAddress,
    sample_index: u32,
    /// M — the initial-RIS candidate count.
    candidates: u32,
    /// Frame width — the kernel decodes the pixel's `(x, y)` from its linear
    /// index for the blue-noise sample-index key (D-095). Spatial reads its own
    /// `width`; temporal reads the reprojection block's; candidates carried
    /// neither, so it gains this scalar (its block has the room).
    width: u32,
    /// Path depth cap — the indirect tail the candidate stage traces inline is
    /// bounded to the same depth as the path tracer (D-134), so reuse and brute
    /// force cover the same path lengths and converge to the same image.
    max_bounces: u32,
}

/// Push constants for the `ReSTIR` temporal-reuse stage
/// (`shaders/restir_temporal.slang`). Runs at bounce 0 only, between candidates
/// and spatial when temporal reuse is on; folds this frame's `cand` and last
/// frame's `prev` into `curr` at the same pixel, unshadowed and M-capped. Reads
/// only its own pixel, so — unlike spatial — it carries no pool capacity or
/// resolution.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RestirTemporalParams {
    paths: PathsAddrs,
    hits: QueueAddrs,
    scene: vk::DeviceAddress,
    restir: vk::DeviceAddress,
    /// This frame's candidate reservoirs (`cand`) — the canonical, read.
    reservoirs_in: vk::DeviceAddress,
    /// Last frame's committed reservoirs (`prev`) — the history, read.
    reservoirs_prev: vk::DeviceAddress,
    /// The temporal survivors (`curr`) — written; spatial, or resolve, reads it.
    reservoirs_out: vk::DeviceAddress,
    /// The per-frame reprojection block (`Reproject`): the previous camera basis,
    /// the ping-ponged G-buffer addresses, and the reuse-gate thresholds. One
    /// push pointer; the block itself is host-written before the wave.
    reproject: vk::DeviceAddress,
    sample_index: u32,
    /// History-length multiplier: `c_prev` is clamped to `m_cap · c_cand` before
    /// the combine — the stage's one bias-critical numeric (D-094).
    m_cap: f32,
    /// Decay window: history confidence is scaled by `saturate(1 −
    /// sample_index/decay_frames)`, so a held camera hands temporal off to
    /// spatial-only accumulation (step 5d). `0` disables the ramp — the
    /// pinned-temporal gate forces temporal live. Sourced from
    /// [`TemporalReuse::decay_frames`].
    decay_frames: u32,
    /// 1 when `prev` was rendered against the current scene build, 0 across
    /// an edit — the epoch gate ([`TemporalReuse::prev_same_scene`]): at 0
    /// the kernel drops indirect history before dereferencing its stored
    /// TLAS index. Also the slot that pads the struct to its 8-byte
    /// alignment (implicit padding would break `Pod`).
    prev_same_scene: u32,
}

/// The GPU mirror of `struct Reproject` in `shaders/restir_reproject.slang`: the
/// previous frame's camera basis (to reproject this frame's shading points into
/// last frame's screen), the ping-ponged G-buffer addresses, the frame
/// dimensions, and the reuse-gate thresholds. Host-written into a `CpuToGpu`
/// buffer each frame ([`Buffer::write`]) and reached by `restir_temporal` through
/// one push pointer. `std430`: each `[f32; 3]` packs a trailing scalar into its
/// 16-byte slot, exactly as the shader's `float3 … ; float … ;` pairs do.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Reproject {
    /// Previous camera eye; the trailing float is the normal reuse-gate threshold.
    position: [f32; 3],
    normal_threshold: f32,
    /// Previous basis right (scaled by tan(vfov/2)·aspect); trailing depth gate.
    right: [f32; 3],
    depth_threshold: f32,
    /// Previous basis up (scaled by tan(vfov/2)).
    up: [f32; 3],
    _pad0: f32,
    /// Previous basis forward (unit).
    forward: [f32; 3],
    _pad1: f32,
    /// Last frame's per-pixel hits — reprojection reads these.
    prev_gbuffer: vk::DeviceAddress,
    /// This frame's per-pixel hits — `restir_temporal` writes these.
    curr_gbuffer: vk::DeviceAddress,
    width: u32,
    height: u32,
    /// 1 once a previous camera exists (from the second frame after a reset or
    /// resize); 0 disables reprojection, so the empty same-pixel history reads
    /// as nothing.
    valid: u32,
    _pad2: u32,
}

// The host mirror and the shader struct — and the `REPROJECT_STRIDE` the view
// sizes the buffer to — must agree byte for byte, or the previous camera decodes
// to garbage and reprojection silently reads the wrong pixel.
const _: () = assert!(size_of::<Reproject>() == 96);

/// The previous frame's pinhole camera, captured after each accumulate so the
/// next frame can reproject its shading points into last frame's screen. Position
/// plus the orthogonal [`RayBasis`](crate::scene::RayBasis) (right/up scaled by
/// tan(vfov/2)·aspect, forward unit) — exactly what inverts raygen's ray
/// construction (`restir_reproject.slang`). The *pinhole* basis even for a
/// thin-lens camera: reprojection places the real hit, not a focal point.
#[derive(Clone, Copy)]
pub struct ReprojectCamera {
    /// The camera eye.
    pub position: Vec3,
    /// Basis right, scaled by tan(vfov/2)·aspect (a pixel's NDC-x extent).
    pub right: Vec3,
    /// Basis up, scaled by tan(vfov/2).
    pub up: Vec3,
    /// Unit forward (the view axis).
    pub forward: Vec3,
}

impl Reproject {
    /// Assemble the per-frame reprojection block. `prev` is last frame's camera,
    /// or `None` on the first frame after a reset or resize — which sets `valid`
    /// to 0, so `restir_temporal` skips reprojection and the (empty) same-pixel
    /// history reads as nothing. The G-buffer addresses are this frame's, after
    /// the swap. Thresholds are the shared reuse-gate constants, so temporal and
    /// spatial gate identically.
    #[must_use]
    pub fn new(
        prev: Option<ReprojectCamera>,
        prev_gbuffer: vk::DeviceAddress,
        curr_gbuffer: vk::DeviceAddress,
        width: u32,
        height: u32,
    ) -> Self {
        let valid = u32::from(prev.is_some());
        // A dummy basis when there is no previous camera; `valid == 0` makes it
        // unread, but it must still be finite (a NaN basis could trip a checker).
        let cam = prev.unwrap_or(ReprojectCamera {
            position: Vec3::ZERO,
            right: Vec3::X,
            up: Vec3::Y,
            forward: Vec3::NEG_Z,
        });
        Self {
            position: cam.position.to_array(),
            normal_threshold: Self::NORMAL_THRESHOLD,
            right: cam.right.to_array(),
            depth_threshold: Self::DEPTH_THRESHOLD,
            up: cam.up.to_array(),
            _pad0: 0.0,
            forward: cam.forward.to_array(),
            _pad1: 0.0,
            prev_gbuffer,
            curr_gbuffer,
            width,
            height,
            valid,
            _pad2: 0,
        }
    }

    /// The temporal disocclusion gate reuses spatial's normal threshold
    /// ([`Wavefront::RESTIR_NORMAL_THRESHOLD`]) — one gate semantics across both
    /// reuse stages.
    const NORMAL_THRESHOLD: f32 = Wavefront::RESTIR_NORMAL_THRESHOLD;
    /// The temporal disocclusion gate reuses spatial's depth threshold
    /// ([`Wavefront::RESTIR_DEPTH_THRESHOLD`]).
    const DEPTH_THRESHOLD: f32 = Wavefront::RESTIR_DEPTH_THRESHOLD;
}

/// Push constants for the `ReSTIR` resolve stage
/// (`shaders/restir_resolve.slang`). Runs at bounce 0 only; reads the reservoir
/// and queues the survivor's shadow ray.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RestirResolveParams {
    paths: PathsAddrs,
    hits: QueueAddrs,
    shadows: QueueAddrs,
    scene: vk::DeviceAddress,
    restir: vk::DeviceAddress,
    reservoirs: vk::DeviceAddress,
    /// The film accumulator: the additive delta-light term adds straight to it
    /// (a pixel-owned write — one resolve thread per primary hit).
    radiance: vk::DeviceAddress,
    /// The D-092 debug false-colour target, or 0 when no [`DebugView`] is
    /// selected (the low byte of `flags` is then [`DebugView::Off`]).
    debug: vk::DeviceAddress,
    /// Packed control word: the [`DebugView`] enum in the low byte (which view to
    /// write into `debug`), plus [`DebugView::VISIBILITY_IN_WEIGHT`] and
    /// [`DebugView::TEMPORAL_IN_WEIGHT`] in bits 8–9. Mirrors `flags` in
    /// `shaders/restir_resolve.slang`.
    flags: u32,
    /// This wave's sample index — the delta term's one next-event draw.
    sample_index: u32,
}

/// Push constants for the `ReSTIR` spatial gather pre-pass (M6 step 7b,
/// `shaders/restir_spatial_gather.slang`). Runs at bounce 0 only, right before
/// the combine; reads the candidate reservoirs, runs every pixel's forward
/// shift evaluations (gates and all of the stage's visibility rays), and
/// writes the pair/self records the combine reads. Sits at Vulkan's
/// guaranteed 128 push-constant bytes exactly — which is why the target
/// dimensions share one packed word.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RestirSpatialGatherParams {
    paths: PathsAddrs,
    hits: QueueAddrs,
    scene: vk::DeviceAddress,
    restir: vk::DeviceAddress,
    /// The candidate stage's output — read for the pixel and its partners.
    reservoirs_in: vk::DeviceAddress,
    /// The forward records, k per path slot — written.
    pairs: vk::DeviceAddress,
    /// The self records, one per path slot — written.
    selfs: vk::DeviceAddress,
    sample_index: u32,
    /// `width | height << 16` — packed so the struct holds 128 B exactly
    /// (each dimension is bounded far below 2¹⁶; asserted where this is
    /// built).
    dims: u32,
    /// Path-pool capacity: a partner's path slot must be below it (the range
    /// guard that keeps partner geometry reads in the live pool).
    capacity: u32,
    /// k — spatial partners gathered.
    neighbours: u32,
    /// Reuse gate: `dot(n_center, n_neighbour)` must exceed this.
    normal_threshold: f32,
    /// Reuse gate: relative camera-depth difference cap.
    depth_threshold: f32,
}

/// Push constants for the `ReSTIR` spatial-reuse combine
/// (`shaders/restir_spatial.slang`). Runs at bounce 0 only, between the gather
/// pre-pass and resolve; ray-free since 7b — it reads the candidate
/// reservoirs and the pre-pass's records (its own for the forward terms, its
/// reciprocal partner's for the backshift) and writes the merged survivors to
/// the scratch buffer.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RestirSpatialParams {
    paths: PathsAddrs,
    hits: QueueAddrs,
    /// The candidate stage's output — read for the pixel and its partners.
    reservoirs_in: vk::DeviceAddress,
    /// The merged survivor's destination — the buffer resolve then reads.
    reservoirs_out: vk::DeviceAddress,
    /// The gather pre-pass's forward records — read.
    pairs: vk::DeviceAddress,
    /// Its self records — read.
    selfs: vk::DeviceAddress,
    sample_index: u32,
    width: u32,
    /// k — spatial partners gathered.
    neighbours: u32,
    /// Tail padding, explicit — the device addresses align the struct to 8,
    /// and `Pod` forbids the implicit kind.
    _pad: u32,
}

/// Push constants for the shadow-ray kernel (`shaders/trace_shadow.slang`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TraceShadowParams {
    shadows: QueueAddrs,
    radiance: vk::DeviceAddress,
    /// Device address of the scene table — opacity attenuates connections.
    scene: vk::DeviceAddress,
}

/// The `SoA` path state: one GPU buffer per logical field, `capacity` slots
/// each. The Rust half of the path-state schema — `struct Paths` in
/// `shaders/pathstate.slang` mirrors it field for field. Adding a field
/// (flags, reservoirs, …) is a buffer here, an address in [`PathsAddrs`],
/// and a pointer in the Slang struct — no kernel signature changes.
struct PathPool {
    /// xyz = ray origin; 16 B/path.
    origin: Buffer,
    /// xyz = unit ray direction; 16 B/path.
    direction: Buffer,
    /// The film pixel each path contributes to; 4 B/path.
    pixel: Buffer,
    /// Hit record — instance + primitive + barycentrics; 16 B/path.
    hit: Buffer,
    /// xyz = the path's accumulated weight; w = the solid-angle pdf of the
    /// scatter that produced this ray (0 on camera rays), kept for the next
    /// vertex's MIS weight; 16 B/path.
    throughput: Buffer,
    /// The scatter's packed state (`packPathState` in
    /// `shaders/pathstate.slang`): the sampled-lobe tag — the record the
    /// AOV specular pass-through ramp and M3's GRIS replay consume — and
    /// the interior medium's instance, which refraction sets and the next
    /// vertex's Beer–Lambert absorption reads; 4 B/path.
    state: Buffer,
}

impl PathPool {
    fn new(gpu: &Context, capacity: u32) -> Result<Self> {
        let paths = u64::from(capacity);
        let storage =
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        Ok(Self {
            origin: gpu.create_buffer(
                "wavefront.origin",
                paths * 16,
                storage,
                MemoryLocation::GpuOnly,
            )?,
            direction: gpu.create_buffer(
                "wavefront.direction",
                paths * 16,
                storage,
                MemoryLocation::GpuOnly,
            )?,
            pixel: gpu.create_buffer(
                "wavefront.pixel",
                paths * 4,
                storage,
                MemoryLocation::GpuOnly,
            )?,
            hit: gpu.create_buffer(
                "wavefront.hit",
                paths * 16,
                storage,
                MemoryLocation::GpuOnly,
            )?,
            throughput: gpu.create_buffer(
                "wavefront.throughput",
                paths * 16,
                storage,
                MemoryLocation::GpuOnly,
            )?,
            state: gpu.create_buffer(
                "wavefront.state",
                paths * 4,
                storage,
                MemoryLocation::GpuOnly,
            )?,
        })
    }

    fn addresses(&self) -> PathsAddrs {
        PathsAddrs {
            origin: self.origin.device_address(),
            direction: self.direction.device_address(),
            pixel: self.pixel.device_address(),
            hit: self.hit.device_address(),
            throughput: self.throughput.device_address(),
            state: self.state.device_address(),
        }
    }
}

/// The stage queues: one shared header buffer plus an entry buffer per
/// queue, each sized for every path at once (a stage can never push more
/// than the pool holds).
struct Queues {
    /// [`queue::COUNT`] × [`QUEUE_HEADER_SIZE`]. `count` and `groupsX` are
    /// zeroed by fill passes just before each queue's producer runs;
    /// `groupsY`/`groupsZ` are uploaded as 1 and never change.
    /// `TRANSFER_SRC` so tests can audit the routing.
    headers: Buffer,
    /// Path indices awaiting intersect.
    ray: Buffer,
    /// Path indices whose rays hit — awaiting `shade_surface`.
    hit: Buffer,
    /// Path indices whose rays escaped — awaiting `shade_miss`.
    miss: Buffer,
    /// Self-contained [`SHADOW_RAY_SIZE`]-byte records awaiting
    /// `trace_shadow`.
    shadow: Buffer,
}

impl Queues {
    fn new(gpu: &Context, capacity: u32) -> Result<Self> {
        let paths = u64::from(capacity);
        let storage =
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        let headers = [[0u32, 0, 1, 1]; queue::COUNT as usize];
        Ok(Self {
            headers: gpu.upload_buffer(
                "wavefront.queue.headers",
                bytemuck::cast_slice(&headers),
                storage
                    | vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC,
            )?,
            ray: gpu.create_buffer(
                "wavefront.queue.ray",
                paths * 4,
                storage,
                MemoryLocation::GpuOnly,
            )?,
            hit: gpu.create_buffer(
                "wavefront.queue.hit",
                paths * 4,
                storage,
                MemoryLocation::GpuOnly,
            )?,
            miss: gpu.create_buffer(
                "wavefront.queue.miss",
                paths * 4,
                storage,
                MemoryLocation::GpuOnly,
            )?,
            shadow: gpu.create_buffer(
                "wavefront.queue.shadow",
                paths * SHADOW_RAY_SIZE,
                storage,
                MemoryLocation::GpuOnly,
            )?,
        })
    }

    fn addresses(&self, index: u64, entries: &Buffer) -> QueueAddrs {
        QueueAddrs {
            state: self.headers.device_address() + index * QUEUE_HEADER_SIZE,
            entries: entries.device_address(),
        }
    }
}

/// The GPU-side AOV table — `struct AovTable` in `shaders/pathstate.slang`,
/// field for field: the wave's per-pixel AOV accumulators and the guides'
/// feature-throughput scratch, behind one pointer because the
/// surface-shading kernel's push constants sit at Vulkan's guaranteed
/// 128-byte limit.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AovTableData {
    albedo: vk::DeviceAddress,
    normal: vk::DeviceAddress,
    depth: vk::DeviceAddress,
    guide: vk::DeviceAddress,
    enabled: u32,
    _pad0: u32,
}

/// The per-pixel AOV buffers a wave writes, owned by the film and handed
/// to [`Wavefront::trace_then`]: the three accumulators (zero-filled at
/// wave start, exactly like radiance; sized `width × height` at 16, 16,
/// and 4 bytes per pixel), and the uploaded table the kernels read them
/// through. The guides' feature-throughput scratch rides inside the table
/// only — it needs no fill (see `AovTable` in `shaders/pathstate.slang`).
pub struct AovTargets<'a> {
    /// The wave's albedo-guide accumulator, RGBA f32 per pixel.
    pub albedo: &'a Buffer,
    /// The wave's normal-guide accumulator, RGBA f32 per pixel.
    pub normal: &'a Buffer,
    /// The wave's first-hit depth, one f32 per pixel.
    pub depth: &'a Buffer,
    /// The uploaded table ([`upload_aov_table`]) naming all of the above.
    pub table: &'a Buffer,
}

/// Upload an [`AovTable`](AovTableData) pointing at the film's per-pixel
/// AOV buffers (`albedo`/`normal` RGBA f32, `depth` f32, `guide` RGBA f32
/// scratch), for [`AovTargets::table`].
///
/// # Errors
///
/// Any [`crate::Error`] from buffer creation.
pub(crate) fn upload_aov_table(
    gpu: &Context,
    albedo: &Buffer,
    normal: &Buffer,
    depth: &Buffer,
    guide: &Buffer,
) -> Result<Buffer> {
    let table = AovTableData {
        albedo: albedo.device_address(),
        normal: normal.device_address(),
        depth: depth.device_address(),
        guide: guide.device_address(),
        enabled: 1,
        _pad0: 0,
    };
    gpu.upload_buffer(
        "film.aov.table",
        bytemuck::bytes_of(&table),
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
    )
}

/// Which sampling strategies reach the lights. [`LightSampling::Mis`] is
/// the renderer; the single-strategy modes exist because the strongest
/// test of the MIS weights is that either strategy alone converges to the
/// same image (the MIS-agreement test below). Delta lights exist only
/// through next-event connections — a BSDF sample hits zero area with
/// probability zero — so [`LightSampling::BsdfOnly`] cannot see them, and
/// agreement scenes stick to area lights and the environment. Values
/// match the `LIGHT_SAMPLING_*` constants in `shaders/lights.slang`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum LightSampling {
    /// Next-event estimation and BSDF sampling, combined by Veach's power
    /// heuristic.
    Mis = 0,
    /// Lights count only when a scattered ray happens to hit them.
    BsdfOnly = 1,
    /// Lights count only through next-event shadow rays (plus directly
    /// visible lights, which no shadow ray can reach).
    NeeOnly = 2,
}

/// How the primary hit is estimated. Orthogonal to [`LightSampling`]: it
/// chooses *which estimator* owns the primary hit's light transport, not
/// which sampling strategies exist.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RenderMode {
    /// The M2 path tracer: `shade_surface` does per-bounce NEE itself.
    PathTracer,
    /// `ReSTIR` — M3's DI reservoir, unified to whole paths in M6
    /// (D-124/D-134): at bounce 0 the reservoir chain (`restir_candidates`,
    /// the reuse stages, `restir_resolve`) owns the primary hit's whole
    /// non-delta path integral, resolved into a per-view reservoir buffer;
    /// `shade_surface` keeps only the directly visible emission there and
    /// pushes no continuation, so bounce 0 is a wave's one recorded round.
    /// At convergence this matches [`PathTracer`](Self::PathTracer) exactly
    /// — the unbiasedness gate.
    Restir,
}

/// The D-092 debug surface's enum-selected view: what `restir_resolve`
/// false-colours into the debug buffer at the primary hit. [`Off`](Self::Off)
/// writes nothing (and needs no debug buffer). The `u32` values mirror the
/// `DEBUG_*` constants in `shaders/restir_resolve.slang`.
///
/// A first-class observability workstream (D-092): `ReSTIR` bias shifts a
/// converged mean a few percent and looks plausible, so the survivor's
/// selection is made inspectable from the first estimator on, not just at the
/// end-of-pipeline reference gate.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum DebugView {
    /// No debug view — the common path; the debug buffer stays untouched.
    #[default]
    Off = 0,
    /// False-colour by stable light id — the checkpoint's "false-colour the
    /// selected light". The environment reads as a steady near-white.
    SelectedLight = 1,
    /// The confidence M as a heatmap (flat at the candidate count until reuse
    /// compounds it, from step 4 on).
    Confidence = 2,
    /// The unbiased contribution weight W as a heatmap — the bias detector.
    UnbiasedWeight = 3,
}

impl DebugView {
    /// Bit 8 of the resolve stage's packed `flags` word: the spatial stage ran,
    /// so the survivor's visibility is already folded into W and `restir_resolve`
    /// shades it unshadowed. Mirrors `DEBUG_VISIBILITY_IN_WEIGHT` in
    /// `shaders/restir_resolve.slang`.
    const VISIBILITY_IN_WEIGHT: u32 = 0x100;
    /// Bit 9 of that word: the temporal stage ran, so the confidence heatmap
    /// scales to temporal's compounded M-cap ceiling rather than the candidate
    /// count. Mirrors `DEBUG_TEMPORAL_IN_WEIGHT` in `shaders/restir_resolve.slang`.
    const TEMPORAL_IN_WEIGHT: u32 = 0x200;
    /// Bit 10 of that word: shade the spatial pass's control-variate lane
    /// instead of the survivor — `ReSTCV`, M6 steps 6a/6b-i. Set only when the
    /// spatial stage ran *and* [`RestirInputs::cv_shading`] is on; clear is
    /// survivor-only shading, the zero-CV degenerate (D-130). Mirrors
    /// `DEBUG_CV_SHADING` in `shaders/restir_resolve.slang`.
    const CV_SHADING: u32 = 0x400;
}

/// The bounce-0 `ReSTIR` targets a wave writes into, in [`RenderMode::Restir`]:
/// the per-pixel reservoir the candidate/resolve stages stream through, and —
/// only when a [`DebugView`] is selected — the debug buffer `restir_resolve`
/// false-colours the survivor into. `debug` is `Some` exactly when `debug_view`
/// is not [`DebugView::Off`]; the wave zero-fills it so unreached pixels read
/// black.
pub struct RestirInputs<'a> {
    /// This wave's committed per-pixel reservoirs (`curr`), sized to the target.
    /// With temporal reuse off, the candidate stage writes it directly; with
    /// temporal on, `restir_temporal` writes it from [`TemporalReuse::cand`] +
    /// [`TemporalReuse::prev`]. With spatial reuse off, `restir_resolve` reads it.
    pub reservoir: &'a Buffer,
    /// The two extra reservoirs temporal reuse threads through, or `None` for the
    /// single-frame path. `Some` turns temporal reuse **on**: candidates write
    /// `cand` (not `reservoir`), and `restir_temporal` folds `cand` + `prev` into
    /// `reservoir` (`curr`) before spatial — the warm-start (D-094).
    pub temporal: Option<TemporalReuse<'a>>,
    /// The spatial stage's output reservoir (the "committed prior-pass" ping-pong
    /// buffer), sized to the target. `Some` turns spatial reuse **on**: the wave
    /// inserts `restir_spatial` after the candidate/temporal reservoir (`curr`)
    /// and before resolve, resolve reads *this* buffer instead of `reservoir`,
    /// and the survivor arrives with its visibility already folded into W (so
    /// resolve shades it without a shadow ray). `None` is the step-3
    /// single-frame-RIS path.
    pub scratch: Option<&'a Buffer>,
    /// How many light candidates initial RIS streams this frame (M). The caller
    /// owns the policy, the same split `TemporalReuse::decay_frames` already
    /// has: [`Renderer::restir_candidates`] is the shipping one, and a test
    /// wanting a fixed cost per sample passes [`Wavefront::RESTIR_CANDIDATES`].
    ///
    /// [`Renderer::restir_candidates`]: crate::render::Renderer
    pub candidates: u32,
    /// Whether resolve shades the spatial pass's control-variate lane rather
    /// than the survivor (`ReSTCV`, M6 steps 6a/6b-i). Acted on only when
    /// `scratch` is `Some` — spatial is the stage that combines the lane;
    /// without it resolve shades the survivor regardless. `false` is the
    /// zero-CV degenerate (D-130) the A/B gates flip to.
    pub cv_shading: bool,
    /// The debug false-colour target (one RGBA f32 per pixel), or `None` when
    /// no view is selected.
    pub debug: Option<&'a Buffer>,
    /// Which view `restir_resolve` writes into `debug`.
    pub debug_view: DebugView,
}

/// The two extra reservoirs temporal reuse reads, present in [`RestirInputs`]
/// exactly when temporal reuse is on. `cand` is candidates' write target (so it
/// no longer aliases `curr`, keeping the candidate lineage that feeds `prev`
/// distinct from next frame's history — the feed convention, D-094); `prev` is
/// last frame's committed `curr`, delivered by the film's frame-end swap.
pub struct TemporalReuse<'a> {
    /// This frame's candidate reservoirs — candidates' write target with temporal
    /// on, the temporal combine's canonical input.
    pub cand: &'a Buffer,
    /// Last frame's committed reservoirs — the temporal combine's history input,
    /// read at the *reprojected* pixel (step 5c).
    pub prev: &'a Buffer,
    /// The per-frame reprojection block (`Reproject`), already host-written this
    /// frame: the previous camera basis, the two G-buffer addresses, and the
    /// reuse-gate thresholds. `restir_temporal` reaches it by address. Building
    /// its contents (which needs the previous camera) is the caller's — the
    /// wavefront only forwards the pointer.
    pub reproject: &'a Buffer,
    /// The temporal decay window (step 5d): history confidence is scaled by
    /// `saturate(1 − samples_since_reset / decay_frames)` so a held camera hands
    /// off to spatial-only accumulation. The renderer passes
    /// [`Wavefront::RESTIR_TEMPORAL_DECAY_FRAMES`]; `0` disables the ramp, which is
    /// how the pinned-temporal gate forces temporal live to convergence (D-094).
    /// Carried here rather than fixed like the M-cap because that gate is exactly
    /// the caller that must override it.
    pub decay_frames: u32,
    /// Whether `prev` was rendered against the *current* scene build — the
    /// epoch gate (§4c decision 2). The renderer compares the build recorded
    /// at the last frame-end swap ([`ViewState::prev_epoch`](crate::restir::ViewState::prev_epoch))
    /// with the live [`Scene::epoch`](crate::scene::Scene::epoch). `false`
    /// means an edit landed since: an indirect prev sample's
    /// `rcVertex.instance` is a raw TLAS custom index the rebuild may have
    /// renumbered, so `restir_temporal` drops such history before any
    /// dereference this frame; NEE history survives through the light-id
    /// registry. Camera-only motion never bumps the build, so the common
    /// temporal case (orbiting) always passes `true`.
    pub prev_same_scene: bool,
}

/// The engine: five stage pipelines over one path pool and its queues.
/// Created once and reused across waves — nothing in it depends on the
/// target size or the scene.
pub struct Wavefront {
    raygen: ComputePipeline,
    intersect: ComputePipeline,
    shade_miss: ComputePipeline,
    shade_surface: ComputePipeline,
    trace_shadow: ComputePipeline,
    /// M3 initial RIS: streams the primary hit's light candidates into the
    /// reservoir. Built alongside the rest; used only in [`RenderMode::Restir`].
    restir_candidates: ComputePipeline,
    /// M3 temporal reuse: folds last frame's `prev` reservoir into this frame's
    /// `cand`. Built alongside the rest; used only when temporal reuse is on.
    restir_temporal: ComputePipeline,
    /// M6 step-7b gather pre-pass: every pixel's forward shift evaluations and
    /// visibility rays, recorded for the combine. Built alongside the rest;
    /// used only when spatial reuse is on.
    restir_spatial_gather: ComputePipeline,
    /// M3 spatial reuse: folds k neighbours' reservoirs into each pixel's —
    /// the ray-free combine since 7b. Built alongside the rest; used only when
    /// spatial reuse is on.
    restir_spatial: ComputePipeline,
    /// M3 resolve: shades the surviving light sample and queues its shadow ray.
    restir_resolve: ComputePipeline,
    paths: PathPool,
    queues: Queues,
    /// The all-zero [`AovTableData`] a wave binds when the caller brings
    /// no AOV targets — `enabled` 0, so the kernels skip every guide read
    /// and write.
    aov_disabled: Buffer,
    /// The blue-noise mask (D-095): a renderer-global void-and-cluster tile the
    /// reservoir stages key their sample-index ranking on, uploaded once and
    /// bound at set 0 binding 3. Owned here — scene-independent, renderer
    /// lifetime — and handed to every [`SceneBindings`].
    blue_noise: Buffer,
    /// The pairing textures (M6 step 7a): the self-inverting Gaussian delta
    /// images the spatial stage draws its reciprocal neighbours from
    /// (`src/pairing.rs`), uploaded once and bound at set 0 binding 4 — the
    /// blue-noise mask's model, for the same renderer-global reason.
    pairing: Buffer,
    /// The spatial gather pre-pass's forward records (M6 step 7b): k per path
    /// slot (`PairRecord` in `shaders/restir_pair.slang`), written by
    /// `restir_spatial_gather` and read back — own and reciprocal partner's —
    /// by the combine. Slot-indexed, so capacity-sized here rather than
    /// target-sized by the caller, and never cleared: the combine reads only
    /// records its own pre-pass thread or an accepted (hence fresh) partner
    /// wrote this dispatch.
    spatial_pairs: Buffer,
    /// Its self records, one per path slot (`SelfRecord`, same module): the
    /// pixel's own-sample evaluation, its albedo, and the accept mask.
    spatial_selfs: Buffer,
    capacity: u32,
    max_bounces: u32,
    light_sampling: LightSampling,
}

impl Wavefront {
    /// Default path-pool capacity: 2²⁰ paths (≈ 64 MB of state at today's
    /// schema). Bounds VRAM at any resolution — larger targets walk ranges
    /// — and comfortably covers a viewer-sized window in one.
    pub const DEFAULT_CAPACITY: u32 = 1 << 20;

    /// Default path-length cap. Deep bounces matter only to near-specular
    /// chains — Russian roulette settles everything else well before the
    /// cap — and eight covers the deepest transport the demo makes visible
    /// (mirror spheres reflecting each other's reflections) with margin.
    pub const DEFAULT_MAX_BOUNCES: u32 = 8;

    /// Initial-RIS candidate count M at the primary hit (D-088: ~16 emitter/env
    /// candidates, plus one internalized BSDF candidate the stage adds). Tuned
    /// in validation; the estimator is unbiased at any M ≥ 1, so this trades
    /// variance for cost, never correctness. Every frame is drawn with it except
    /// the one [`Renderer::restir_candidates`] cheapens.
    ///
    /// [`Renderer::restir_candidates`]: crate::render::Renderer
    pub const RESTIR_CANDIDATES: u32 = 16;

    /// M on a frame that restarts accumulation onto a live temporal history —
    /// which is every frame a moving camera renders (M7 step 7a). The history
    /// already supplies the confidence a fresh sweep would buy; paying for
    /// sixteen more is paying twice. Whether a given frame qualifies is
    /// [`Renderer::restir_candidates`]' call.
    ///
    /// One, not zero: light sampling is the only technique covering the support
    /// of *f* (see `_SUPPORT_COVERAGE`). If 1 proves too thin on a disocclusion,
    /// 2 is one character away.
    ///
    /// [`Renderer::restir_candidates`]: crate::render::Renderer
    pub const RESTIR_RESTART_CANDIDATES: u32 = 1;

    /// Spatial neighbours gathered per pixel (D-088: ~5). The estimator is
    /// unbiased at any k ≥ 0 — this trades variance (more reuse) for cost (a
    /// visibility ray each) — so it is tuned in validation, not load-bearing.
    /// Since 7a each slot draws from its own pairing texture, so raising k
    /// past `pairing::COUNT` means adding textures.
    pub const RESTIR_SPATIAL_NEIGHBOURS: u32 = 5;

    /// Compile-time cap on k, mirrored by `MAX_SPATIAL_NEIGHBOURS` in
    /// `shaders/restir_spatial.slang` (the shader's per-thread accepted-neighbour
    /// store is a fixed array of this size). Raising it means raising both.
    pub const RESTIR_MAX_SPATIAL_NEIGHBOURS: u32 = 8;

    /// Reuse gate — the neighbour's geometric normal must satisfy
    /// `dot(n_center, n_neighbour) > this`. 0.9 (≈26°) is deliberately stricter
    /// than RTXDI's 0.5 (≈60°): the conservative, correctness-first choice, and
    /// the first knob to loosen before adding a second pass if convergence lags
    /// (D-093).
    pub const RESTIR_NORMAL_THRESHOLD: f32 = 0.9;

    /// Reuse gate — relative camera-depth difference cap, matching RTXDI's 0.1.
    pub const RESTIR_DEPTH_THRESHOLD: f32 = 0.1;

    /// Temporal history-length multiplier (D-094): last frame's reservoir
    /// confidence is clamped to `M_CAP · c_cand` before the combine, so history
    /// saturates at ~20 candidate-frames rather than growing without bound. A
    /// *multiplier*, not an absolute confidence — a candidates frame already
    /// credits ~`M + 1` confidence, so an absolute cap near that
    /// would throttle history to nothing. Unbiased at any positive value (it only
    /// reweights the MIS); tuned for lag-vs-noise, not correctness — the shipping
    /// ~20 of the RTXDI lineage.
    ///
    /// Relative is what lets M vary per frame (M7 step 7a) with no compensating
    /// retune: `c_cand` falls with M, so the capped prev:cand ratio is the same
    /// ~20:1 at M = 1 as at M = 16.
    pub const RESTIR_TEMPORAL_M_CAP: f32 = 20.0;

    /// Temporal decay window (D-094): on a held camera the history confidence is
    /// scaled by `saturate(1 − samples_since_reset / DECAY_FRAMES)`, reaching
    /// *exactly* zero after this many frames — from there the estimator is
    /// temporal-free (spatial-only, fresh per-frame RNG), so the converged still
    /// is a mean of independent unbiased frames converging at 1/N to the
    /// brute-force reference. A moving camera resets the film each frame, holding
    /// the ramp near one (temporal live, the warm-start). Short by design so the
    /// motion→hold handoff has no visible pop; unbiased at any window (it only
    /// reweights the MIS), tuned for lag-vs-decorrelation. `0` disables the ramp —
    /// the pinned-temporal CI gate uses it to force temporal live to convergence.
    ///
    /// This window is where M must *not* be cut (M7 step 7a): annealing history
    /// away is what decorrelates a settling still, and propping its confidence
    /// back up with thin candidate frames works against exactly that. The
    /// measurement is on [`Renderer::restir_candidates`], which is why only
    /// sample 0 is ever cheapened.
    ///
    /// [`Renderer::restir_candidates`]: crate::render::Renderer
    pub const RESTIR_TEMPORAL_DECAY_FRAMES: u32 = 16;

    // k must fit the shader's fixed accepted-neighbour array — a build error,
    // not a runtime clamp, if a tuning edit outgrows it.
    const _SPATIAL_FITS: () = assert!(
        Self::RESTIR_SPATIAL_NEIGHBOURS <= Self::RESTIR_MAX_SPATIAL_NEIGHBOURS,
        "RESTIR_SPATIAL_NEIGHBOURS exceeds the shader's MAX_SPATIAL_NEIGHBOURS store"
    );

    // ... and every slot must have a pairing texture (7a) — the shader also
    // caps k at PAIRING_TEXTURES, so outgrowing this would silently gather
    // fewer neighbours than asked.
    const _SPATIAL_PAIRED: () = assert!(
        Self::RESTIR_SPATIAL_NEIGHBOURS as usize <= crate::pairing::COUNT,
        "RESTIR_SPATIAL_NEIGHBOURS exceeds the pairing textures"
    );

    /// Byte size of one gather pre-pass pair record — `PairRecord` in
    /// `shaders/restir_pair.slang` (std430: one float3-led 16-byte lane plus
    /// four scalars). The host only allocates; both the writer and the reader
    /// compile from the one shader-side definition.
    const RESTIR_PAIR_RECORD_BYTES: u64 = 32;

    /// Byte size of one self record — `SelfRecord` in the same module (three
    /// 16-byte lanes).
    const RESTIR_SELF_RECORD_BYTES: u64 = 48;

    // Support-coverage guard (M3 plan §2, §6): the reservoir's *unshadowed*
    // target makes RIS unbiased only if some candidate covers the whole support
    // of f — and the light-sampling technique (the power-alias table over every
    // emitter, plus the environment) does, on its own, for any M ≥ 1. The
    // internalized BSDF candidate only sharpens variance; it cannot be the sole
    // cover. So a future candidate-budget edit that drops M to 0 would silently
    // bias the mean — this pins it as a build error instead, one of the two
    // subtlest bias traps the reference course flags to bake in early. Both
    // counts, since nothing orders them and either can be the one a frame is
    // drawn with.
    const _SUPPORT_COVERAGE: () = assert!(
        Self::RESTIR_CANDIDATES >= 1 && Self::RESTIR_RESTART_CANDIDATES >= 1,
        "ReSTIR needs at least one light candidate: it is the only technique \
         guaranteed to cover the support of the unshadowed target (M3 plan §2)"
    );

    /// Build the five stage pipelines and allocate the pool and queues.
    /// Each wave shades at most `max_bounces` bounces per path and reaches
    /// lights via `light_sampling` (always [`LightSampling::Mis`] outside
    /// the MIS-agreement test).
    ///
    /// # Errors
    ///
    /// Any [`crate::Error`] from pipeline or buffer creation.
    ///
    /// # Panics
    ///
    /// On zero capacity or a bounce cap outside 1..=255 (the cap shares a
    /// packed push-constant byte) — programmer bugs.
    pub fn new(
        gpu: &Context,
        kernels: &Kernels,
        capacity: u32,
        max_bounces: u32,
        light_sampling: LightSampling,
    ) -> Result<Self> {
        assert!(capacity > 0, "zero-capacity path pool");
        assert!(max_bounces > 0, "zero-bounce wavefront");
        assert!(
            max_bounces <= 255,
            "a bounce cap above 255 doesn't fit its packed push-constant byte"
        );
        let pipeline = |kernel: &Kernel, push_constant_size: usize, bindings| {
            gpu.create_compute_pipeline(
                &kernel.spirv,
                kernel.entry,
                push_constant_size as u32,
                bindings,
            )
        };
        Ok(Self {
            raygen: pipeline(&kernels.raygen, size_of::<RaygenParams>(), Bindings::None)?,
            intersect: pipeline(
                &kernels.intersect,
                size_of::<IntersectParams>(),
                Bindings::Scene,
            )?,
            shade_miss: pipeline(
                &kernels.shade_miss,
                size_of::<ShadeMissParams>(),
                Bindings::Scene,
            )?,
            shade_surface: pipeline(
                &kernels.shade_surface,
                size_of::<ShadeSurfaceParams>(),
                Bindings::Scene,
            )?,
            trace_shadow: pipeline(
                &kernels.trace_shadow,
                size_of::<TraceShadowParams>(),
                Bindings::Scene,
            )?,
            // The reservoir stages read the environment map and bindless
            // textures (target evaluation, textured emitters), so they bind the
            // same scene set the shading kernels do.
            restir_candidates: pipeline(
                &kernels.restir_candidates,
                size_of::<RestirCandidatesParams>(),
                Bindings::Scene,
            )?,
            // Also binds the scene set: the cross-frame shift of an indirect
            // prev sample traces the TLAS (replay segments and its one
            // visibility ray), and the NEE target eval reads the environment
            // and textured emitters. The NEE arms stay rayless (D-094).
            restir_temporal: pipeline(
                &kernels.restir_temporal,
                size_of::<RestirTemporalParams>(),
                Bindings::Scene,
            )?,
            // Also binds the scene set: the k+1 visibility rays trace the TLAS,
            // and the target eval reads the environment and textured emitters.
            restir_spatial_gather: pipeline(
                &kernels.restir_spatial_gather,
                size_of::<RestirSpatialGatherParams>(),
                Bindings::Scene,
            )?,
            // Ray-free since 7b, but still on the scene set: the blue-noise
            // mask (binding 3) and the pairing textures (binding 4) ride it.
            restir_spatial: pipeline(
                &kernels.restir_spatial,
                size_of::<RestirSpatialParams>(),
                Bindings::Scene,
            )?,
            restir_resolve: pipeline(
                &kernels.restir_resolve,
                size_of::<RestirResolveParams>(),
                Bindings::Scene,
            )?,
            paths: PathPool::new(gpu, capacity)?,
            queues: Queues::new(gpu, capacity)?,
            aov_disabled: gpu.upload_buffer(
                "wavefront.aov.disabled",
                bytemuck::bytes_of(&AovTableData::zeroed()),
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            )?,
            // The blue-noise mask: read through the descriptor (binding 3), so
            // no device address — a plain storage buffer (D-095).
            blue_noise: gpu.upload_buffer(
                "wavefront.bluenoise",
                bytemuck::cast_slice(crate::bluenoise::mask().as_slice()),
                vk::BufferUsageFlags::STORAGE_BUFFER,
            )?,
            // The pairing textures: read through the descriptor (binding 4),
            // like the blue-noise mask above.
            pairing: gpu.upload_buffer(
                "wavefront.pairing",
                bytemuck::cast_slice(crate::pairing::textures()),
                vk::BufferUsageFlags::STORAGE_BUFFER,
            )?,
            spatial_pairs: gpu.create_buffer(
                "wavefront.spatial.pairs",
                u64::from(capacity)
                    * u64::from(Self::RESTIR_SPATIAL_NEIGHBOURS)
                    * Self::RESTIR_PAIR_RECORD_BYTES,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                MemoryLocation::GpuOnly,
            )?,
            spatial_selfs: gpu.create_buffer(
                "wavefront.spatial.selfs",
                u64::from(capacity) * Self::RESTIR_SELF_RECORD_BYTES,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                MemoryLocation::GpuOnly,
            )?,
            capacity,
            max_bounces,
            light_sampling,
        })
    }

    /// Trace one sample: one full path per pixel of a `width`×`height`
    /// target — camera ray, then per bounce an MIS-weighted direct-light
    /// sample and an `OpenPBR` scatter, Russian roulette from bounce 3 —
    /// with the path's radiance
    /// accumulated into `radiance` (zero-filled first; needs
    /// `TRANSFER_DST`) as row-major RGBA `f32`, pixel (0, 0) top-left,
    /// alpha 1 exactly once per pixel. One blocking submission; targets
    /// larger than the pool are walked in pool-sized pixel ranges within
    /// it.
    ///
    /// `sample` indexes every pixel's sample sequence: it selects the
    /// camera jitter and every scattering decision along the path, so
    /// accumulating consecutive indices is progressive refinement.
    ///
    /// Bitwise deterministic: the same `sample` re-traces the same wave bit
    /// for bit. Queue push order varies run to run, but radiance writes are
    /// pixel-owned, so the image never sees it.
    ///
    /// # Errors
    ///
    /// Any [`crate::Error`] from submission.
    ///
    /// # Panics
    ///
    /// On a zero-sized target or a `radiance` buffer smaller than it —
    /// programmer bugs.
    pub fn trace(
        &self,
        gpu: &Context,
        scene: &Scene,
        radiance: &Buffer,
        width: u32,
        height: u32,
        sample: u32,
    ) -> Result<()> {
        self.trace_then(gpu, scene, radiance, width, height, sample, None, None, &[], None)
            .map(|_| ())
    }

    /// [`Wavefront::trace`], then `trailing` — extra passes appended to the
    /// same submission, sharing its single fence. The full memory barrier
    /// [`Context::submit_passes`] places between passes flushes the wave's
    /// radiance writes before the first trailing pass reads them, so folding
    /// the film's accumulate in here spends one GPU round-trip per sample
    /// instead of two — bit-for-bit as if they ran as separate submissions,
    /// since a barrier orders the same writes a fence does.
    ///
    /// With `timer`, the submission is bracketed and the returned
    /// [`PassTimings`] say what the device spent — and, on the frames the
    /// timer resolves, what each kernel spent. Without, the timings are
    /// empty and the submission is byte-for-byte the one this always
    /// recorded. Stamping is not free, which is why the fine-grained half
    /// of it is rationed; see [`crate::gpu::PassTimer`] for the measurement
    /// and what it cost the design.
    ///
    /// With `aovs`, the wave also feeds the film's AOV accumulators —
    /// zero-filled at wave start like radiance, written by the shading
    /// kernels (first-hit depth, and the albedo/normal denoiser guides
    /// with their specular pass-through). Without, the kernels skip every
    /// AOV read and write.
    ///
    /// # Errors
    ///
    /// Any [`crate::Error`] from submission.
    ///
    /// # Panics
    ///
    /// As [`Wavefront::trace`]: on a zero-sized target or a `radiance`
    /// buffer smaller than it.
    // The target (radiance, width, height), which sample, and the AOV and
    // trailing extensions: exactly `trace`'s parameters plus two. A struct
    // would only scatter the call — every caller already hands these same
    // values to `trace`.
    #[allow(clippy::too_many_arguments)]
    pub fn trace_then(
        &self,
        gpu: &Context,
        scene: &Scene,
        radiance: &Buffer,
        width: u32,
        height: u32,
        sample: u32,
        aovs: Option<&AovTargets>,
        restir: Option<&RestirInputs>,
        trailing: &[Pass],
        timer: Option<&mut PassTimer>,
    ) -> Result<PassTimings> {
        assert!(width > 0 && height > 0, "zero-sized trace target");
        let pixels = u64::from(width) * u64::from(height);
        assert!(
            radiance.size() >= pixels * 16,
            "radiance buffer smaller than the target"
        );
        if let Some(restir) = restir {
            assert!(
                restir.reservoir.size()
                    >= pixels * size_of::<crate::restir::StoredPathReservoir>() as u64,
                "reservoir buffer smaller than the target"
            );
            assert_eq!(
                restir.debug.is_some(),
                restir.debug_view != DebugView::Off,
                "a debug buffer is required exactly when a DebugView is selected"
            );
            if let Some(debug) = restir.debug {
                assert!(
                    debug.size() >= pixels * 16,
                    "debug buffer smaller than the target"
                );
            }
        }
        let aov_table = aovs.map_or(&self.aov_disabled, |aov| aov.table);
        let params = self.wave_params(scene, radiance, aov_table, width, height, sample, restir);
        let mut passes = self.record_wave(scene, radiance, aovs, restir, pixels, &params);
        passes.extend_from_slice(trailing);
        gpu.submit_passes_timed(&passes, timer)
    }

    /// Every stage's push constants for one wave, built up front so the
    /// recorded passes can borrow them.
    // The wave's target, sizing, sample index, and optional reservoir — one
    // more than the linter's cutoff, and every one already threaded through
    // `trace_then`; a struct would only scatter the single call site.
    #[allow(clippy::too_many_arguments)]
    fn wave_params(
        &self,
        scene: &Scene,
        radiance: &Buffer,
        aov_table: &Buffer,
        width: u32,
        height: u32,
        sample: u32,
        restir: Option<&RestirInputs>,
    ) -> WaveParams {
        let pixels = u64::from(width) * u64::from(height);
        let mut basis = scene.camera().basis(width as f32 / height as f32);
        // An open aperture scales the basis to the focal plane, making
        // `forward + x·right + y·up` each pixel's focal point — the form
        // the thin-lens raygen path re-aims lens rays at. A pinhole keeps
        // the unit basis and the exact ray construction it always had.
        let aperture_radius = scene.camera().lens.map_or(0.0, |lens| {
            basis.right *= lens.focus_distance;
            basis.up *= lens.focus_distance;
            basis.forward *= lens.focus_distance;
            lens.aperture_radius
        });
        let ranges = (0..pixels)
            .step_by(self.capacity as usize)
            .map(|base| RaygenParams {
                origin: self.paths.origin.device_address(),
                direction: self.paths.direction.device_address(),
                pixel: self.paths.pixel.device_address(),
                throughput: self.paths.throughput.device_address(),
                rays: self.queues.addresses(queue::RAY, &self.queues.ray),
                sample_index: sample,
                aperture_radius,
                width,
                height,
                camera_position: scene.camera().position,
                base: base as u32,
                camera_right: basis.right,
                count: (pixels - base).min(u64::from(self.capacity)) as u32,
                camera_up: basis.up,
                _pad0: 0,
                camera_forward: basis.forward,
                _pad1: 0,
            })
            .collect();
        let intersect = |bounce: u32| IntersectParams {
            paths: self.paths.addresses(),
            rays: self.queues.addresses(queue::RAY, &self.queues.ray),
            hits: self.queues.addresses(queue::HIT, &self.queues.hit),
            misses: self.queues.addresses(queue::MISS, &self.queues.miss),
            scene: scene.table().device_address(),
            ray_mask: if bounce == 0 {
                ray_mask::CAMERA
            } else {
                ray_mask::ALL
            },
            sample_index: sample,
            bounce,
            _pad0: 0,
        };
        // ReSTIR mode records bounce 0 alone: the reservoir owns the whole
        // primary-hit path integral (D-134), shade_surface pushes no
        // continuation there, and every later round would dispatch nothing —
        // so the wave doesn't record them.
        let bounces = if restir.is_some() { 1 } else { self.max_bounces };
        WaveParams {
            ranges,
            intersect: (0..bounces).map(intersect).collect(),
            shade_miss: ShadeMissParams {
                paths: self.paths.addresses(),
                misses: self.queues.addresses(queue::MISS, &self.queues.miss),
                scene: scene.table().device_address(),
                radiance: radiance.device_address(),
                aov: aov_table.device_address(),
                light_sampling: self.light_sampling as u32,
                _pad0: 0,
            },
            shade_surface: (0..bounces)
                .map(|bounce| ShadeSurfaceParams {
                    paths: self.paths.addresses(),
                    hits: self.queues.addresses(queue::HIT, &self.queues.hit),
                    rays: self.queues.addresses(queue::RAY, &self.queues.ray),
                    shadows: self.queues.addresses(queue::SHADOW, &self.queues.shadow),
                    scene: scene.table().device_address(),
                    radiance: radiance.device_address(),
                    aov: aov_table.device_address(),
                    sample_index: sample,
                    packed: pack_shade_surface(
                        bounce,
                        self.max_bounces,
                        self.light_sampling,
                        restir.is_some(),
                    ),
                })
                .collect(),
            trace_shadow: TraceShadowParams {
                shadows: self.queues.addresses(queue::SHADOW, &self.queues.shadow),
                radiance: radiance.device_address(),
                scene: scene.table().device_address(),
            },
            // The reservoir stages run at bounce 0 only, when the caller brought
            // a reservoir target (ReSTIR mode).
            restir: restir
                .map(|restir| self.restir_wave_params(scene, restir, radiance, width, height, sample)),
        }
    }

    /// The reservoir stages' push constants for one wave — built only in
    /// [`RenderMode::Restir`], when the caller supplies [`RestirInputs`].
    ///
    /// The buffer routing is the four-role wiring of D-094. Candidates writes
    /// `cand` when temporal reuse is on (`restir.temporal` is `Some`) else
    /// `reservoir` (`curr`) directly; with temporal on, `restir_temporal` folds
    /// `cand` + `prev` into `curr`. With spatial reuse on (`restir.scratch` is
    /// `Some`), `restir_spatial` reads `curr` (self + neighbours) and writes the
    /// scratch, and resolve reads the scratch — the committed-prior-pass ping-pong
    /// (M3 plan §2), the survivor's visibility already folded into W. With spatial
    /// off, resolve reads `curr` directly and owes the survivor its own shadow
    /// ray. Resolve also carries the debug target and view, so it can
    /// false-colour the survivor (D-092).
    #[allow(clippy::too_many_lines, reason = "one struct literal per stage — the wiring is the map")]
    fn restir_wave_params(
        &self,
        scene: &Scene,
        restir: &RestirInputs,
        radiance: &Buffer,
        width: u32,
        height: u32,
        sample: u32,
    ) -> RestirWaveParams {
        let paths = self.paths.addresses();
        let hits = self.queues.addresses(queue::HIT, &self.queues.hit);
        let scene_table = scene.table().device_address();
        let restir_table = scene.restir_table().device_address();
        let reservoirs = restir.reservoir.device_address();

        // Temporal reuse on: candidates write `cand` (not `curr`), and
        // `restir_temporal` folds `cand` + `prev` into `curr` (`reservoirs`).
        // Off: candidates write `curr` directly, the step-3/4 path. Either way
        // `curr` is what spatial or resolve reads next.
        let candidate_reservoirs = restir
            .temporal
            .as_ref()
            .map_or(reservoirs, |t| t.cand.device_address());
        let temporal = restir.temporal.as_ref().map(|t| RestirTemporalParams {
            paths,
            hits,
            scene: scene_table,
            restir: restir_table,
            reservoirs_in: t.cand.device_address(),
            reservoirs_prev: t.prev.device_address(),
            reservoirs_out: reservoirs,
            reproject: t.reproject.device_address(),
            sample_index: sample,
            m_cap: Self::RESTIR_TEMPORAL_M_CAP,
            decay_frames: t.decay_frames,
            prev_same_scene: u32::from(t.prev_same_scene),
        });

        // Spatial reuse on: build the pre-pass + combine pair (7b), and route
        // resolve at the scratch the survivor lands in. The
        // visibility-in-weight bit (DebugView::VISIBILITY_IN_WEIGHT, mirrored
        // in restir_resolve.slang) rides in the packed `flags` word — it tells
        // resolve the survivor's visibility is already in W, so shade
        // unshadowed.
        let spatial = restir.scratch.map(|scratch| {
            // The gather params pack both target dimensions into one word to
            // hold the 128 B push-constant budget.
            assert!(
                width < (1 << 16) && height < (1 << 16),
                "target dimension overflows the gather pre-pass's packed dims word"
            );
            (
                RestirSpatialGatherParams {
                    paths,
                    hits,
                    scene: scene_table,
                    restir: restir_table,
                    reservoirs_in: reservoirs,
                    pairs: self.spatial_pairs.device_address(),
                    selfs: self.spatial_selfs.device_address(),
                    sample_index: sample,
                    dims: width | (height << 16),
                    capacity: self.capacity,
                    neighbours: Self::RESTIR_SPATIAL_NEIGHBOURS,
                    normal_threshold: Self::RESTIR_NORMAL_THRESHOLD,
                    depth_threshold: Self::RESTIR_DEPTH_THRESHOLD,
                },
                RestirSpatialParams {
                    paths,
                    hits,
                    reservoirs_in: reservoirs,
                    reservoirs_out: scratch.device_address(),
                    pairs: self.spatial_pairs.device_address(),
                    selfs: self.spatial_selfs.device_address(),
                    sample_index: sample,
                    width,
                    neighbours: Self::RESTIR_SPATIAL_NEIGHBOURS,
                    _pad: 0,
                },
            )
        });
        let (resolve_reservoirs, visibility_in_weight) = match restir.scratch {
            Some(scratch) => (scratch.device_address(), DebugView::VISIBILITY_IN_WEIGHT),
            None => (reservoirs, 0),
        };
        // The temporal-ran flag (DebugView::TEMPORAL_IN_WEIGHT, mirrored in
        // restir_resolve.slang) rides the same word — it only sets the confidence
        // heatmap's scale, since temporal compounds M toward its M-cap ceiling far
        // past the candidate count, and an unscaled heatmap would clip a warmed
        // pixel to solid red.
        let temporal_in_weight = if restir.temporal.is_some() {
            DebugView::TEMPORAL_IN_WEIGHT
        } else {
            0
        };
        // The CV-shading flag (DebugView::CV_SHADING, mirrored in
        // restir_resolve.slang): resolve shades the control-variate lane the
        // spatial stage combined (steps 6a/6b-i) — so it is gated on spatial
        // actually running, not just the toggle.
        let cv_shading = if restir.cv_shading && restir.scratch.is_some() {
            DebugView::CV_SHADING
        } else {
            0
        };

        RestirWaveParams {
            candidates: RestirCandidatesParams {
                paths,
                hits,
                scene: scene_table,
                restir: restir_table,
                reservoirs: candidate_reservoirs,
                sample_index: sample,
                candidates: restir.candidates,
                width,
                max_bounces: self.max_bounces,
            },
            temporal,
            spatial,
            resolve: RestirResolveParams {
                paths,
                hits,
                shadows: self.queues.addresses(queue::SHADOW, &self.queues.shadow),
                scene: scene_table,
                restir: restir_table,
                reservoirs: resolve_reservoirs,
                radiance: radiance.device_address(),
                // 0 when no view is selected; restir_resolve then writes nothing.
                debug: restir.debug.map_or(0, Buffer::device_address),
                flags: restir.debug_view as u32
                    | visibility_in_weight
                    | temporal_in_weight
                    | cv_shading,
                sample_index: sample,
            },
        }
    }

    /// Record one wave's pass sequence: zero the radiance target (and the
    /// AOV accumulators, when the wave carries them), then per pixel
    /// range, raygen and the bounce loop.
    // The recorded sequence *is* the map of a frame (see the module header), so
    // it reads top-to-bottom as one function rather than being split into
    // helpers that would scatter the order a reader traces.
    #[allow(clippy::too_many_lines, reason = "the linear pass sequence is the map")]
    fn record_wave<'a>(
        &'a self,
        scene: &'a Scene,
        radiance: &'a Buffer,
        aovs: Option<&AovTargets<'a>>,
        restir: Option<&RestirInputs<'a>>,
        pixels: u64,
        params: &'a WaveParams,
    ) -> Vec<Pass<'a>> {
        // Every post-raygen stage touches a scene resource — the TLAS, the
        // sampled images, or both — and they share one descriptor layout,
        // so each binds the same set.
        let bindings = SceneBindings {
            tlas: scene.tlas(),
            environment: scene.environment(),
            textures: scene.texture_descriptors(),
            blue_noise: &self.blue_noise,
            pairing: &self.pairing,
        };
        // An indirect stage: workgroup counts read from its queue's header,
        // which the producing stage maintained.
        let indirect = |pipeline, push_constants, index: u64| Pass::DispatchIndirect {
            pipeline,
            scene: Some(bindings),
            push_constants,
            args: &self.queues.headers,
            offset: index * QUEUE_HEADER_SIZE + INDIRECT_OFFSET,
        };
        // Reset a queue to empty, just before its producer runs (groupsY/Z
        // stay 1 from the upload — only count and groupsX reset).
        let fill = |index: u64| Pass::Fill {
            buffer: &self.queues.headers,
            offset: index * QUEUE_HEADER_SIZE,
            size: 8,
            value: 0,
        };

        // Radiance accumulates across the wave's bounce rounds, so the
        // wave starts from zero rather than each pixel being written once.
        let mut passes = vec![Pass::Fill {
            buffer: radiance,
            offset: 0,
            size: pixels * 16,
            value: 0,
        }];
        // The AOV accumulators likewise: a pixel's guides can land at any
        // bounce (the specular pass-through), so they too are plain adds
        // onto zero. The guide scratch inside the table needs no fill —
        // bounce 0 never reads it.
        if let Some(aov) = aovs {
            for (buffer, texel) in [(aov.albedo, 16), (aov.normal, 16), (aov.depth, 4)] {
                passes.push(Pass::Fill {
                    buffer,
                    offset: 0,
                    size: pixels * texel,
                    value: 0,
                });
            }
        }
        // The debug surface is single-shot, not accumulated: `restir_resolve`
        // writes only the pixels whose reservoir selected a light, so the wave
        // clears it first — unreached and empty-reservoir pixels then read as
        // black (nothing selected).
        if let Some(debug) = restir.and_then(|restir| restir.debug) {
            passes.push(Pass::Fill {
                buffer: debug,
                offset: 0,
                size: pixels * 16,
                value: 0,
            });
        }
        // Clear `curr` once, before any range writes it, so its miss and
        // unreached pixels read as the empty reservoir (confidence 0) rather than
        // stale VRAM. Two readers need that: spatial reads *neighbour* reservoirs
        // of `curr`, and — with temporal on — next frame's temporal reads this
        // frame's `curr` as `prev` (the swap), so a missed pixel must not persist
        // as bogus high-confidence history. (Single-frame RIS with neither on
        // needs no clear: resolve reads only the pixels candidates just wrote, at
        // the same pixel it dispatched from. `cand` likewise needs none — temporal
        // reads it only at its own hit pixel.)
        if let Some(restir) =
            restir.filter(|restir| restir.scratch.is_some() || restir.temporal.is_some())
        {
            passes.push(Pass::Fill {
                buffer: restir.reservoir,
                offset: 0,
                size: pixels * size_of::<crate::restir::StoredPathReservoir>() as u64,
                value: 0,
            });
        }
        for raygen in &params.ranges {
            passes.push(fill(queue::RAY));
            passes.push(Pass::Dispatch {
                pipeline: &self.raygen,
                scene: None,
                push_constants: bytemuck::bytes_of(raygen),
                group_counts: [raygen.count.div_ceil(WORKGROUP_SIZE), 1, 1],
            });
            // The bounce loop, recorded ahead of time: each round consumes
            // the ray queue, refills it with the paths that scattered, and
            // ends by tracing the round's next-event shadow rays. Rounds
            // after every path has died dispatch nothing — and ReSTIR mode,
            // whose primary hit pushes no continuation, records bounce 0 alone
            // (the params vectors are sized to the recorded rounds).
            let bounces = params.shade_surface.len() as u32;
            for bounce in 0..bounces {
                passes.push(fill(queue::HIT));
                passes.push(fill(queue::MISS));
                passes.push(fill(queue::SHADOW));
                passes.push(indirect(
                    &self.intersect,
                    bytemuck::bytes_of(&params.intersect[bounce as usize]),
                    queue::RAY,
                ));
                // The ray queue was just consumed; empty it for this
                // round's shade_surface — except on the last recorded bounce,
                // where the kernel terminates every path instead of pushing.
                if bounce + 1 < bounces {
                    passes.push(fill(queue::RAY));
                }
                passes.push(indirect(
                    &self.shade_miss,
                    bytemuck::bytes_of(&params.shade_miss),
                    queue::MISS,
                ));
                // ReSTIR mode, primary hit: stream the reservoir candidates,
                // optionally fold in last frame's history (temporal) then the
                // spatial neighbours, and resolve the survivor — all before
                // shade_surface (whose bounce-0 NEE is off). Each dispatches from
                // the hit queue without consuming it, and the full barrier between
                // passes orders them: candidates writes `cand` (or `curr` with
                // temporal off), temporal folds `cand` + `prev` into `curr`,
                // spatial reads `curr` (self + neighbours) and writes the scratch,
                // resolve reads whichever holds the survivor. Later bounces are the
                // ordinary path.
                if let Some(restir) = params.restir.as_ref().filter(|_| bounce == 0) {
                    passes.push(indirect(
                        &self.restir_candidates,
                        bytemuck::bytes_of(&restir.candidates),
                        queue::HIT,
                    ));
                    if let Some(temporal) = restir.temporal.as_ref() {
                        passes.push(indirect(
                            &self.restir_temporal,
                            bytemuck::bytes_of(temporal),
                            queue::HIT,
                        ));
                    }
                    if let Some((gather, spatial)) = restir.spatial.as_ref() {
                        // The pre-pass writes the pair/self records; the full
                        // inter-pass barrier orders them for the combine's
                        // cross-pixel reads.
                        passes.push(indirect(
                            &self.restir_spatial_gather,
                            bytemuck::bytes_of(gather),
                            queue::HIT,
                        ));
                        passes.push(indirect(
                            &self.restir_spatial,
                            bytemuck::bytes_of(spatial),
                            queue::HIT,
                        ));
                    }
                    passes.push(indirect(
                        &self.restir_resolve,
                        bytemuck::bytes_of(&restir.resolve),
                        queue::HIT,
                    ));
                }
                passes.push(indirect(
                    &self.shade_surface,
                    bytemuck::bytes_of(&params.shade_surface[bounce as usize]),
                    queue::HIT,
                ));
                passes.push(indirect(
                    &self.trace_shadow,
                    bytemuck::bytes_of(&params.trace_shadow),
                    queue::SHADOW,
                ));
            }
        }
        passes
    }
}

/// One wave's push constants — see [`Wavefront::wave_params`].
struct WaveParams {
    /// One raygen instance per pool-sized pixel range.
    ranges: Vec<RaygenParams>,
    /// One instance per recorded bounce (bounce 0 alone in `ReSTIR` mode):
    /// bounce 0 traces with the camera visibility bit, later bounces with all
    /// bits, and each keys its own transparency stream.
    intersect: Vec<IntersectParams>,
    /// One instance for every bounce — nothing in it varies per round.
    shade_miss: ShadeMissParams,
    /// One instance per recorded bounce; its length is the wave's bounce count.
    shade_surface: Vec<ShadeSurfaceParams>,
    trace_shadow: TraceShadowParams,
    /// The bounce-0 reservoir stages' push constants, present only in
    /// [`RenderMode::Restir`].
    restir: Option<RestirWaveParams>,
}

/// The reservoir stages' push constants for one wave — bounce 0 only.
struct RestirWaveParams {
    candidates: RestirCandidatesParams,
    /// The temporal stage, present only when temporal reuse is on
    /// (`RestirInputs::temporal` was `Some`).
    temporal: Option<RestirTemporalParams>,
    /// The spatial stage — gather pre-pass then combine, dispatched
    /// back-to-back — present only when spatial reuse is on
    /// (`RestirInputs::scratch` was `Some`).
    spatial: Option<(RestirSpatialGatherParams, RestirSpatialParams)>,
    resolve: RestirResolveParams,
}

#[cfg(test)]
mod tests {
    use glam::{Mat4, Vec3};

    use super::*;
    use crate::environment::Environment;
    use crate::material::Material;
    use crate::scene::{Camera, Object, ground_plane, icosphere};

    fn radiance_buffer(gpu: &Context, width: u32, height: u32) -> Buffer {
        gpu.create_buffer(
            "test.radiance",
            u64::from(width) * u64::from(height) * 16,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
        )
        .expect("radiance buffer")
    }

    /// Audit the queue machinery after one wave over a ragged 33×17 target,
    /// on a single-bounce engine so the post-wave headers still hold the
    /// whole wave's routing (multi-bounce rounds reset them mid-wave):
    /// raygen pushed every path exactly once, intersect routed each to hit
    /// *or* miss (both non-empty in the demo scene), nothing fed the shadow
    /// queue, and every incrementally-maintained `groupsX` is exactly
    /// `ceil(count / WORKGROUP_SIZE)`.
    #[test]
    fn queues_route_every_path_exactly_once() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let scene = Scene::demo(&gpu).expect("demo scene");
        let wavefront = Wavefront::new(&gpu, &Kernels::embedded(), 4096, 1, LightSampling::Mis)
            .expect("wavefront");
        let (width, height) = (33, 17);
        let radiance = radiance_buffer(&gpu, width, height);
        wavefront
            .trace(&gpu, &scene, &radiance, width, height, 0)
            .expect("trace");

        let headers: Vec<u32> = bytemuck::pod_collect_to_vec(
            &gpu.download_buffer(&wavefront.queues.headers)
                .expect("download"),
        );
        let header = |index: u64| &headers[(index * 4) as usize..(index * 4 + 4) as usize];
        let (ray, hit, miss, shadow) = (
            header(queue::RAY),
            header(queue::HIT),
            header(queue::MISS),
            header(queue::SHADOW),
        );

        let paths = width * height;
        assert_eq!(ray[0], paths);
        assert_eq!(hit[0] + miss[0], paths, "every ray routed exactly once");
        assert!(hit[0] > 0, "the demo scene fills most of the frame");
        assert!(miss[0] > 0, "the demo scene has open sky");
        assert_eq!(
            shadow[0], 0,
            "the depth-cap bounce (here the only one) sends no shadow rays"
        );
        for state in [ray, hit, miss, shadow] {
            assert_eq!(state[1], state[0].div_ceil(WORKGROUP_SIZE));
            assert_eq!(&state[2..], &[1, 1], "groupsY/Z hold constant 1");
        }
    }

    /// A pool smaller than the target walks pixel ranges inside one
    /// submission; the image must be bitwise identical to a pool that
    /// covers the target in one range.
    #[test]
    fn pool_sized_ranges_cover_larger_targets() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let scene = Scene::demo(&gpu).expect("demo scene");
        let kernels = Kernels::embedded();
        let (width, height) = (33, 17); // 561 pixels → 9 ranges of ≤ 64
        let render = |capacity: u32| {
            let wavefront = Wavefront::new(
                &gpu,
                &kernels,
                capacity,
                Wavefront::DEFAULT_MAX_BOUNCES,
                LightSampling::Mis,
            )
            .expect("wavefront");
            let radiance = radiance_buffer(&gpu, width, height);
            wavefront
                .trace(&gpu, &scene, &radiance, width, height, 0)
                .expect("trace");
            gpu.download_buffer(&radiance).expect("download")
        };
        assert_eq!(render(64), render(4096));
    }

    /// Progressive refinement is real. A camera ray that misses adds the
    /// environment radiance to
    /// a zero-filled pixel exactly (throughput is still 1, and a constant
    /// environment reads back its one texel exactly), and no surface path
    /// plausibly lands on that exact value, so "this sample saw the sky"
    /// is an exact test. Across the first 16 samples of a small render,
    /// some silhouette pixel must see both a surface and the sky — its
    /// average is then a partial-coverage value no single sample can
    /// produce, which is edges converging — while a pixel fully inside the
    /// ground plane must never see sky: its jitter stays within the pixel
    /// footprint. (A dedicated constant-sky scene: the demo wears an HDRI
    /// now, whose background varies per direction.)
    #[test]
    fn camera_jitter_mixes_edge_pixels() {
        const SKY: [f32; 4] = [0.4, 0.4, 0.4, 1.0]; // the scene's constant sky
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let objects = [
            Object {
                mesh: icosphere(2),
                transform: Mat4::from_translation(Vec3::Y),
                material: Material::matte(Vec3::splat(0.5), 0.3),
            },
            Object {
                // Large enough that the frame's bottom edge lands on it.
                mesh: ground_plane(12.0),
                transform: Mat4::IDENTITY,
                material: Material::matte(Vec3::splat(0.5), 0.1),
            },
        ];
        let camera = Camera {
            position: Vec3::new(0.0, 2.0, 8.5),
            look_at: Vec3::new(0.0, 0.5, 0.0),
            up: Vec3::Y,
            vfov_degrees: 40.0,
            lens: None,
        };
        let scene = Scene::new(
            &gpu,
            &objects,
            camera,
            &Environment::constant(Vec3::splat(0.4)),
        )
        .expect("scene");
        let wavefront = Wavefront::new(
            &gpu,
            &Kernels::embedded(),
            4096,
            Wavefront::DEFAULT_MAX_BOUNCES,
            LightSampling::Mis,
        )
        .expect("wavefront");
        let (width, height) = (32, 32);
        let radiance = radiance_buffer(&gpu, width, height);

        let bottom_center = ((height - 1) * width + width / 2) as usize * 4;
        let mut saw_sky = vec![false; (width * height) as usize];
        let mut saw_surface = vec![false; (width * height) as usize];
        for sample in 0..16 {
            wavefront
                .trace(&gpu, &scene, &radiance, width, height, sample)
                .expect("trace");
            let pixels: Vec<f32> =
                bytemuck::pod_collect_to_vec(&gpu.download_buffer(&radiance).expect("download"));
            assert_ne!(
                &pixels[bottom_center..bottom_center + 4],
                &SKY,
                "plane-interior pixel saw the sky at sample {sample}"
            );
            for (index, pixel) in pixels.chunks_exact(4).enumerate() {
                if pixel == SKY {
                    saw_sky[index] = true;
                } else {
                    saw_surface[index] = true;
                }
            }
        }
        let mixed = saw_sky
            .iter()
            .zip(&saw_surface)
            .filter(|(sky, surface)| **sky && **surface)
            .count();
        assert!(
            mixed > 0,
            "no silhouette pixel saw both surface and sky across 16 samples"
        );
    }

    /// Audit the sampler on the GPU it ships on, through the test-only dump
    /// kernel `shaders/rng_test.slang` (compiled here via the hot-reload
    /// compiler). Owen scrambling must preserve the Sobol (0,2)-sequence
    /// guarantee: among the first 64 samples of any (pixel, dimension) key,
    /// every cell of an 8×8 grid and every width-1/64 bin per axis holds
    /// exactly one point. White noise fails immediately, and so does any
    /// bit-order, matrix, or hash bug — while image-level tests would
    /// render plausibly through all of them.
    #[test]
    fn sampler_is_stratified_and_decorrelated() {
        const COUNT: u32 = 64;

        /// Mirrors `struct Params` in `shaders/rng_test.slang`.
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct DumpParams {
            points: vk::DeviceAddress,
            values: vk::DeviceAddress,
            pixel: u32,
            dimension: u32,
            count: u32,
            _pad0: u32,
        }

        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let spirv = crate::shaders::compile_fixture("rng_test").expect("compile rng_test");
        let pipeline = gpu
            .create_compute_pipeline(
                &spirv,
                c"rng_test",
                size_of::<DumpParams>() as u32,
                Bindings::None,
            )
            .expect("pipeline");

        // One dispatch per key: the first COUNT (2D point, 1D value) pairs.
        let dump = |pixel: u32, dimension: u32| -> (Vec<f32>, Vec<f32>) {
            let usage = vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_SRC;
            let points = gpu
                .create_buffer(
                    "test.rng.points",
                    u64::from(COUNT) * 8,
                    usage,
                    MemoryLocation::GpuOnly,
                )
                .expect("points buffer");
            let values = gpu
                .create_buffer(
                    "test.rng.values",
                    u64::from(COUNT) * 4,
                    usage,
                    MemoryLocation::GpuOnly,
                )
                .expect("values buffer");
            let params = DumpParams {
                points: points.device_address(),
                values: values.device_address(),
                pixel,
                dimension,
                count: COUNT,
                _pad0: 0,
            };
            gpu.dispatch(&pipeline, None, bytemuck::bytes_of(&params), [1, 1, 1])
                .expect("dispatch");
            (
                bytemuck::pod_collect_to_vec(&gpu.download_buffer(&points).expect("download")),
                bytemuck::pod_collect_to_vec(&gpu.download_buffer(&values).expect("download")),
            )
        };

        let bin = |value: f32, bins: u32| {
            assert!((0.0..1.0).contains(&value), "sample {value} outside [0, 1)");
            (value * bins as f32) as usize
        };
        for (pixel, dimension) in [(0, 0), (7, 0), (123_456, 3)] {
            let (points, values) = dump(pixel, dimension);
            let mut cells = [0u32; 64]; // 8×8 grid over the 2D points
            let mut x_bins = [0u32; 64];
            let mut y_bins = [0u32; 64];
            for point in points.chunks_exact(2) {
                cells[bin(point[1], 8) * 8 + bin(point[0], 8)] += 1;
                x_bins[bin(point[0], 64)] += 1;
                y_bins[bin(point[1], 64)] += 1;
            }
            let mut value_bins = [0u32; 64];
            for &value in &values {
                value_bins[bin(value, 64)] += 1;
            }
            for bins in [cells, x_bins, y_bins, value_bins] {
                assert!(
                    bins.iter().all(|&count| count == 1),
                    "key ({pixel}, {dimension}): a stratum holds ≠ 1 points: {bins:?}"
                );
            }
        }

        // Different keys must give different sequences.
        assert_ne!(dump(0, 0), dump(1, 0), "pixels must decorrelate");
        assert_ne!(dump(0, 0), dump(0, 1), "dimensions must decorrelate");
    }

    /// Depth of field, observed through the same exact-sky trick as the
    /// jitter test: a pixel whose 16 samples ever saw both the constant
    /// sky and a surface is a "mixed" pixel, and mixing happens exactly
    /// where rays of one pixel disagree about what they hit. A pinhole
    /// mixes only the one-jitter-wide silhouette ring; a wide aperture
    /// focused far in front of the geometry swings each sample's ray
    /// across the lens disk, so silhouettes smear over many more pixels.
    /// The energy side of the lens is pinned exactly by the thin-lens
    /// furnace in `render/mod.rs`; this is the geometry side.
    #[test]
    fn an_open_aperture_blurs_out_of_focus_silhouettes() {
        const SKY: [f32; 4] = [0.4, 0.4, 0.4, 1.0];
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let objects = [
            Object {
                mesh: icosphere(2),
                transform: Mat4::from_translation(Vec3::Y),
                material: Material::matte(Vec3::splat(0.5), 0.3),
            },
            Object {
                mesh: ground_plane(12.0),
                transform: Mat4::IDENTITY,
                material: Material::matte(Vec3::splat(0.5), 0.1),
            },
        ];
        let camera = |lens| Camera {
            position: Vec3::new(0.0, 2.0, 8.5),
            look_at: Vec3::new(0.0, 0.5, 0.0),
            up: Vec3::Y,
            vfov_degrees: 40.0,
            lens,
        };
        let kernels = Kernels::embedded();
        let (width, height) = (32, 32);
        let mixed_pixels = |lens: Option<crate::scene::Lens>| -> usize {
            let scene = Scene::new(
                &gpu,
                &objects,
                camera(lens),
                &Environment::constant(Vec3::splat(0.4)),
            )
            .expect("scene");
            let wavefront = Wavefront::new(
                &gpu,
                &kernels,
                4096,
                Wavefront::DEFAULT_MAX_BOUNCES,
                LightSampling::Mis,
            )
            .expect("wavefront");
            let radiance = radiance_buffer(&gpu, width, height);
            let mut saw_sky = vec![false; (width * height) as usize];
            let mut saw_surface = vec![false; (width * height) as usize];
            for sample in 0..16 {
                wavefront
                    .trace(&gpu, &scene, &radiance, width, height, sample)
                    .expect("trace");
                let pixels: Vec<f32> = bytemuck::pod_collect_to_vec(
                    &gpu.download_buffer(&radiance).expect("download"),
                );
                for (index, pixel) in pixels.chunks_exact(4).enumerate() {
                    if pixel == SKY {
                        saw_sky[index] = true;
                    } else {
                        saw_surface[index] = true;
                    }
                }
            }
            saw_sky
                .iter()
                .zip(&saw_surface)
                .filter(|(sky, surface)| **sky && **surface)
                .count()
        };

        let pinhole = mixed_pixels(None);
        let blurred = mixed_pixels(Some(crate::scene::Lens {
            aperture_radius: 0.4,
            focus_distance: 2.0, // the sphere sits ~7.7 m out: far out of focus
        }));
        assert!(pinhole > 0, "the silhouette ring itself should mix");
        assert!(
            blurred > 2 * pinhole,
            "an out-of-focus silhouette should smear across far more pixels: \
             {blurred} blurred vs {pinhole} pinhole"
        );
    }

    /// The test that catches wrong-but-plausible MIS: next-event-only,
    /// BSDF-only, and MIS renders of one scene must
    /// converge to the same mean. A pdf mismatch or a weight pair that
    /// doesn't sum to 1 biases the strategies apart (double-counting shows
    /// up as 2×); goldens can't see this — they'd normalize the bias into
    /// the reference. The sky is black, so every photon comes from the
    /// emitter, and the shaded sphere really occludes it — broken
    /// shadow-ray visibility shifts the next-event modes but not
    /// BSDF-only. The emitter is a *sphere* deliberately: hundreds of
    /// triangle records through the alias table, curved-emitter cosines,
    /// and — the sharp edge — next-event samples on its far side, which
    /// must count as occluded by its own near side. An identity test that
    /// stops at the instance (instead of the exact triangle) double-counts
    /// those and biases NEE-only high, right here. The glass ball beside
    /// the sphere extends the agreement through refraction: its exit-face
    /// vertices connect to the emitter *through* the interface, and the
    /// transmission pdf competes in every weight — a wrong refraction
    /// Jacobian splits the strategies here.
    #[test]
    fn light_sampling_strategies_agree() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let objects = [
            // Half-metal sphere and glossy floor: sharp specular lobes
            // are where wrong-but-plausible MIS weights actually live.
            Object {
                mesh: icosphere(2),
                transform: Mat4::from_translation(Vec3::Y),
                material: Material::glossy(Vec3::splat(0.6), 0.4, 0.3).with_metalness(0.5),
            },
            Object {
                mesh: ground_plane(4.0),
                transform: Mat4::IDENTITY,
                material: Material::glossy(Vec3::splat(0.7), 0.0, 0.2),
            },
            Object {
                mesh: icosphere(2),
                transform: Mat4::from_translation(Vec3::new(1.6, 0.6, 1.2))
                    * Mat4::from_scale(Vec3::splat(0.6)),
                material: Material::glass(0.4, 1.5),
            },
            Object {
                // An emissive ball right above the sphere: big enough that
                // BSDF sampling finds it often (variance stays testable),
                // low enough that its shadow occludes real floor.
                mesh: icosphere(2),
                transform: Mat4::from_translation(Vec3::Y * 3.0)
                    * Mat4::from_scale(Vec3::splat(0.7)),
                material: Material::emitter(Vec3::splat(4.0)),
            },
        ];
        let camera = Camera {
            position: Vec3::new(0.0, 2.5, 6.0),
            look_at: Vec3::new(0.0, 1.0, 0.0),
            up: Vec3::Y,
            vfov_degrees: 45.0,
            lens: None,
        };
        let scene =
            Scene::new(&gpu, &objects, camera, &Environment::constant(Vec3::ZERO)).expect("scene");

        assert_strategies_agree(&gpu, &scene);
    }

    /// The same agreement, with the *environment* as the only light. The
    /// synthetic sky is the CDF tables' worst
    /// case: one bright texel flanked by hard zeros over a dim base, so
    /// next-event sampling must importance-sample the sun through the
    /// marginal/conditional tables *and* reach the zero texels its
    /// bilinear footprint bleeds into (the dilated sampling support — an
    /// undilated build biases NEE-only low right here), while BSDF-only
    /// must be weighted consistently by `pdf(dir)` in `shade_miss`.
    #[test]
    fn light_sampling_strategies_agree_on_the_environment() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let objects = [
            Object {
                mesh: icosphere(2),
                transform: Mat4::from_translation(Vec3::Y),
                material: Material::glossy(Vec3::splat(0.6), 0.4, 0.3).with_metalness(0.5),
            },
            Object {
                mesh: ground_plane(4.0),
                transform: Mat4::IDENTITY,
                material: Material::glossy(Vec3::splat(0.7), 0.0, 0.2),
            },
        ];
        let camera = Camera {
            position: Vec3::new(0.0, 2.5, 6.0),
            look_at: Vec3::new(0.0, 1.0, 0.0),
            up: Vec3::Y,
            vfov_degrees: 45.0,
            lens: None,
        };
        let (width, height) = (8, 4);
        let mut texels = vec![0.2_f32; (width * height * 4) as usize];
        for col in 0..width as usize {
            // A hard-zero band in the sky's upper row...
            texels[(width as usize + col) * 4..(width as usize + col) * 4 + 3].fill(0.0);
        }
        // ...with the sun in the middle of it.
        texels[(width as usize + 4) * 4..(width as usize + 4) * 4 + 3].fill(8.0);
        let sky = Environment::equirect(width, height, texels);
        let scene = Scene::new(&gpu, &objects, camera, &sky).expect("scene");
        assert_strategies_agree(&gpu, &scene);
    }

    /// Estimator consistency must survive textures in both places they
    /// touch the light transport. The
    /// emitter's radiance is a *map* — next-event estimation evaluates it
    /// at its own sampled point (through the connection's barycentrics)
    /// while BSDF paths evaluate it where they land, and the two only
    /// converge together if both read the same function pointwise (a
    /// map's *scale* in the light record with the texel applied twice, or
    /// not at all, splits them). And a fractionally-transparent *textured*
    /// card hangs between the emitter and the floor: path rays resolve
    /// its per-texel coverage stochastically in traversal while shadow
    /// rays attenuate deterministically — same map, two policies, and any
    /// disagreement between them biases the NEE modes away from
    /// BSDF-only. Black sky, so the textured emitter is the only light.
    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "one flat change-set literal is the whole scene — splitting it \
                  would hide its shape"
    )]
    fn light_sampling_strategies_agree_on_textured_lights_and_opacity() {
        use crate::scene::changeset::{
            CameraPatch, ChangeSet, InstancePatch, MaterialPatch, MeshPatch, Op, SettingsPatch,
        };
        use crate::scene::description::{
            MeshSource, SceneDescription, Texturable, TextureRef, Transform,
        };

        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let dir = std::env::temp_dir().join(format!("cenote-mis-textured-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("fixture dir");
        // The emitter's map: three brightness levels and a hard zero.
        let glow = dir.join("glow.png");
        #[rustfmt::skip]
        crate::texture::write_png(&glow, 2, 2, &[
            255, 255, 255, 255,   100, 100, 100, 255,
            180, 180, 180, 255,   0, 0, 0, 255,
        ]);
        // The card's coverage: opaque, half, quarter, and open quadrants.
        let holes = dir.join("holes.png");
        let coverage: Vec<u8> = (0..64)
            .flat_map(|index| {
                let (x, y) = (index % 8, index / 8);
                let value = match (x < 4, y < 4) {
                    (true, true) => 255u8,
                    (false, true) => 128,
                    (true, false) => 64,
                    (false, false) => 0,
                };
                [value, 0, 0, 255]
            })
            .collect();
        crate::texture::write_png(&holes, 8, 8, &coverage);

        let sphere = icosphere(2);
        let plane = |scale: [f32; 3], translate: [f32; 3]| Transform::Trs {
            translate,
            rotate_degrees: [0.0; 3],
            scale,
        };
        let mut description = SceneDescription::new();
        description
            .apply(&ChangeSet {
                ops: vec![
                    Op::Settings(SettingsPatch::new("main")),
                    Op::Camera(CameraPatch {
                        position: Some([0.0, 2.5, 6.0]),
                        look_at: Some([0.0, 1.0, 0.0]),
                        vfov_degrees: Some(45.0),
                        ..CameraPatch::new("main")
                    }),
                    Op::Mesh(MeshPatch {
                        source: Some(MeshSource::Inline {
                            positions: vec![
                                [-1.0, 0.0, -1.0],
                                [-1.0, 0.0, 1.0],
                                [1.0, 0.0, 1.0],
                                [1.0, 0.0, -1.0],
                            ],
                            normals: Some(vec![[0.0, 1.0, 0.0]; 4]),
                            uvs: Some(vec![[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]),
                            triangles: vec![[0, 1, 2], [0, 2, 3]],
                        }),
                        ..MeshPatch::new("plane")
                    }),
                    Op::Mesh(MeshPatch {
                        source: Some(MeshSource::Inline {
                            positions: sphere.positions.iter().map(glam::Vec3::to_array).collect(),
                            normals: Some(
                                sphere.normals.iter().map(glam::Vec3::to_array).collect(),
                            ),
                            uvs: None,
                            triangles: sphere.triangles.clone(),
                        }),
                        ..MeshPatch::new("sphere")
                    }),
                    // Glossy floor and a half-metal sphere: sharp lobes are
                    // where wrong-but-plausible weights live.
                    Op::Material(Box::new(MaterialPatch {
                        base_color: Some(Texturable::Constant([0.7; 3])),
                        specular_roughness: Some(Texturable::Constant(0.2)),
                        ..MaterialPatch::new("floor")
                    })),
                    Op::Instance(InstancePatch {
                        mesh: Some("plane".into()),
                        material: Some("floor".into()),
                        transforms: Some(vec![plane([4.0, 1.0, 4.0], [0.0; 3])]),
                        ..InstancePatch::new("floor")
                    }),
                    Op::Material(Box::new(MaterialPatch {
                        base_color: Some(Texturable::Constant([0.6; 3])),
                        specular_roughness: Some(Texturable::Constant(0.3)),
                        base_metalness: Some(Texturable::Constant(0.5)),
                        ..MaterialPatch::new("shell")
                    })),
                    Op::Instance(InstancePatch {
                        mesh: Some("sphere".into()),
                        material: Some("shell".into()),
                        transforms: Some(vec![plane([1.0; 3], [0.0, 1.0, 0.0])]),
                        ..InstancePatch::new("shell")
                    }),
                    // The textured emitter overhead...
                    Op::Material(Box::new(MaterialPatch {
                        base_color: Some(Texturable::Constant([0.0; 3])),
                        specular_weight: Some(0.0),
                        emission_luminance: Some(4.0),
                        emission_color: Some(Texturable::Texture(TextureRef {
                            path: glow,
                            color_space: None,
                            channel: None,
                            scale: None,
                            uv: None,
                        })),
                        ..MaterialPatch::new("lamp")
                    })),
                    Op::Instance(InstancePatch {
                        mesh: Some("plane".into()),
                        material: Some("lamp".into()),
                        // Rolled 180° so its one face looks down at the floor:
                        // the plane winds normal-up, and emission is one-sided.
                        transforms: Some(vec![Transform::Trs {
                            translate: [0.0, 3.0, 0.0],
                            rotate_degrees: [180.0, 0.0, 0.0],
                            scale: [0.7; 3],
                        }]),
                        ..InstancePatch::new("lamp")
                    }),
                    // ...and the perforated card between it and the floor.
                    Op::Material(Box::new(MaterialPatch {
                        base_color: Some(Texturable::Constant([0.5; 3])),
                        specular_weight: Some(0.0),
                        geometry_opacity: Some(Texturable::Texture(TextureRef {
                            path: holes,
                            color_space: None,
                            channel: None,
                            scale: None,
                            uv: None,
                        })),
                        ..MaterialPatch::new("card")
                    })),
                    Op::Instance(InstancePatch {
                        mesh: Some("plane".into()),
                        material: Some("card".into()),
                        transforms: Some(vec![plane([1.2, 1.0, 1.2], [0.0, 2.0, 0.0])]),
                        ..InstancePatch::new("card")
                    }),
                ],
            })
            .expect("valid scene");
        let scene = crate::scene::Scene::prep(&gpu, &mut description).expect("prep");
        assert_strategies_agree(&gpu, &scene);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Render `scene` under all three light-sampling modes and require the
    /// means to agree within 3% — the shared teeth of the MIS-agreement
    /// tests above.
    fn assert_strategies_agree(gpu: &Context, scene: &Scene) {
        let kernels = Kernels::embedded();
        let (width, height) = (32, 32);
        // Enough that the frame-average converges under the environment
        // test's worst case — a lone bright sun texel is high-variance for
        // NEE, and at 64 spp the mean still swings several percent between
        // sampler realizations, more than the 3% agreement bound below. 256
        // brings that swing under ~1%, so the bound keeps its teeth against a
        // real bias rather than tripping on noise.
        let samples: u32 = 256;
        let mean = |light_sampling: LightSampling| -> f64 {
            let wavefront = Wavefront::new(
                gpu,
                &kernels,
                4096,
                Wavefront::DEFAULT_MAX_BOUNCES,
                light_sampling,
            )
            .expect("wavefront");
            let radiance = radiance_buffer(gpu, width, height);
            let mut total = 0.0;
            for sample in 0..samples {
                wavefront
                    .trace(gpu, scene, &radiance, width, height, sample)
                    .expect("trace");
                let pixels: Vec<f32> = bytemuck::pod_collect_to_vec(
                    &gpu.download_buffer(&radiance).expect("download"),
                );
                total += pixels
                    .chunks_exact(4)
                    .map(|pixel| f64::from(pixel[0]) + f64::from(pixel[1]) + f64::from(pixel[2]))
                    .sum::<f64>();
            }
            total / f64::from(samples * width * height)
        };

        let mis = mean(LightSampling::Mis);
        let bsdf = mean(LightSampling::BsdfOnly);
        let nee = mean(LightSampling::NeeOnly);
        assert!(mis > 0.01, "the scene should be lit, got mean {mis}");
        for (name, value) in [("BSDF-only", bsdf), ("NEE-only", nee)] {
            let deviation = (value - mis).abs() / mis;
            assert!(
                deviation < 0.03,
                "{name} disagrees with MIS: {value} vs {mis} ({deviation:.4} relative)"
            );
        }
    }

    /// The M3 unbiasedness gate (D-090), run for both reservoir estimators: the
    /// single-frame RIS of step 3 *and* the spatial reuse of step 4 must each
    /// converge to the same image as the M2 path tracer. Since M6 step 2 the
    /// reservoir owns the *whole* path integral at the primary hit (D-134) — M
    /// light candidates plus the internalized BSDF draw's light and continuation
    /// candidates, combined under the count-weighted balance heuristic; only the
    /// directly visible emission and the delta term stay outside it. It is the
    /// same integral estimated three ways: a biased reservoir (a dropped
    /// Jacobian, a wrong count in a balance or pairwise-MIS weight, a stale
    /// target, a mishandled visibility fold) would shift the converged mean a
    /// few percent — exactly what this catches. Spatial adds the neighbour gather, the defensive
    /// pairwise MIS, and the k+1 visibility rays on top, so its agreement gates
    /// the whole step-4 estimator end to end. The scene is a lit environment
    /// *and* an area emitter, so both light-candidate branches — the env coin
    /// flip and the triangle table — and the BSDF candidate's reconnection to
    /// each are all exercised, including the spatial shift's env-vs-area branch
    /// (D-093). No delta lights: they stay on exact additive NEE outside the
    /// reservoir (D-088), a separate term this gate deliberately leaves out.
    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "the scene literal plus the three-way path/RIS/spatial comparison \
                  is one gate — splitting it would scatter what it checks"
    )]
    fn restir_matches_the_path_tracer() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let objects = [
            Object {
                mesh: icosphere(2),
                transform: Mat4::from_translation(Vec3::Y),
                material: Material::glossy(Vec3::splat(0.6), 0.4, 0.3).with_metalness(0.5),
            },
            Object {
                mesh: ground_plane(4.0),
                transform: Mat4::IDENTITY,
                material: Material::glossy(Vec3::splat(0.7), 0.0, 0.2),
            },
            Object {
                // An area emitter above the sphere — the triangle candidates.
                mesh: icosphere(2),
                transform: Mat4::from_translation(Vec3::Y * 3.0)
                    * Mat4::from_scale(Vec3::splat(0.7)),
                material: Material::emitter(Vec3::splat(4.0)),
            },
        ];
        let camera = Camera {
            position: Vec3::new(0.0, 2.5, 6.0),
            look_at: Vec3::new(0.0, 1.0, 0.0),
            up: Vec3::Y,
            vfov_degrees: 45.0,
            lens: None,
        };
        // A lit environment as well, so the env coin flip fires alongside the
        // triangle table.
        let scene = Scene::new(&gpu, &objects, camera, &Environment::constant(Vec3::splat(0.3)))
            .expect("scene");

        let kernels = Kernels::embedded();
        let (width, height) = (32, 32);
        // As with the strategy-agreement gate: 256 spp brings the sampler
        // realization swing under ~1%, so the 3% bound answers to bias, not
        // noise.
        let samples: u32 = 256;
        let reservoir = gpu
            .create_buffer(
                "test.reservoir",
                u64::from(width) * u64::from(height) * size_of::<crate::restir::StoredPathReservoir>() as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                MemoryLocation::GpuOnly,
            )
            .expect("reservoir buffer");
        let wavefront = Wavefront::new(
            &gpu,
            &kernels,
            4096,
            Wavefront::DEFAULT_MAX_BOUNCES,
            LightSampling::Mis,
        )
        .expect("wavefront");
        // Spatial reuse (step 4) reads a second reservoir buffer — the scratch
        // the merged survivor lands in — so it too must converge to the same
        // image (D-090's gate runs from step 3 onward). The wave clears `reservoir`
        // itself when spatial is on, so no host-side clear is needed here.
        let scratch = gpu
            .create_buffer(
                "test.reservoir.scratch",
                u64::from(width) * u64::from(height) * size_of::<crate::restir::StoredPathReservoir>() as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuOnly,
            )
            .expect("scratch buffer");
        let reservoir_initial = RestirInputs {
            reservoir: &reservoir,
            temporal: None,
            scratch: None,
            candidates: Wavefront::RESTIR_CANDIDATES,
            cv_shading: true,
            debug: None,
            debug_view: DebugView::Off,
        };
        let reservoir_spatial = RestirInputs {
            reservoir: &reservoir,
            temporal: None,
            scratch: Some(&scratch),
            candidates: Wavefront::RESTIR_CANDIDATES,
            cv_shading: true,
            debug: None,
            debug_view: DebugView::Off,
        };
        let mean = |inputs: Option<&RestirInputs>| -> f64 {
            let radiance = radiance_buffer(&gpu, width, height);
            let mut total = 0.0;
            for sample in 0..samples {
                wavefront
                    .trace_then(
                        &gpu, &scene, &radiance, width, height, sample, None, inputs, &[], None,
                    )
                    .expect("trace");
                let pixels: Vec<f32> = bytemuck::pod_collect_to_vec(
                    &gpu.download_buffer(&radiance).expect("download"),
                );
                total += pixels
                    .chunks_exact(4)
                    .map(|pixel| f64::from(pixel[0]) + f64::from(pixel[1]) + f64::from(pixel[2]))
                    .sum::<f64>();
            }
            total / f64::from(samples * width * height)
        };

        let path = mean(None);
        assert!(path > 0.01, "the scene should be lit, got mean {path}");
        // Both the single-frame RIS estimator (step 3) and the spatial-reuse
        // estimator (step 4) are the same integral as the path tracer.
        for (label, inputs) in [
            ("initial RIS", &reservoir_initial),
            ("spatial reuse", &reservoir_spatial),
        ] {
            let restir = mean(Some(inputs));
            let deviation = (restir - path).abs() / path;
            assert!(
                deviation < 0.03,
                "{label} disagrees with the path tracer: {restir} vs {path} ({deviation:.4} relative)"
            );
        }
    }

    /// The step-2 checkpoint (D-134): first true path reuse. The unified
    /// reservoir now owns the *whole* path integral at the primary hit — direct
    /// *and* indirect — so the M3 unbiasedness gate lifts onto indirect paths.
    /// On an all-matte GI scene every reconnection vertex is reuse-eligible, so
    /// the reconnection shift and its geometry-term Jacobian carry the full
    /// weight (a glossy scene would drop most reuse, D-125): a dropped Jacobian,
    /// a wrong reconnection target, or a mis-baked tail radiance would shift the
    /// converged mean, which this catches against the brute-force path tracer.
    /// Both the initial RIS (own-pixel, identity shift) and the spatial reuse
    /// (cross-pixel reconnection shift) must land on the path tracer's image; the
    /// sphere-and-ground interreflection under a bright emitter makes the indirect
    /// term the reservoir carries a real fraction of the picture, not incidental.
    #[test]
    fn restir_path_reuse_matches_the_path_tracer_on_diffuse_gi() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        // All matte: every bounce vertex is a valid reconnection point, so
        // reconnection reuse is exercised, not gated away. A bright environment
        // plus an area emitter light the diffuse interreflection between the
        // sphere and the ground.
        let objects = [
            Object {
                mesh: icosphere(2),
                transform: Mat4::from_translation(Vec3::Y),
                material: Material::matte(Vec3::new(0.75, 0.4, 0.35), 0.5),
            },
            Object {
                mesh: ground_plane(4.0),
                transform: Mat4::IDENTITY,
                material: Material::matte(Vec3::splat(0.8), 0.5),
            },
            Object {
                mesh: icosphere(2),
                transform: Mat4::from_translation(Vec3::Y * 3.0)
                    * Mat4::from_scale(Vec3::splat(0.7)),
                material: Material::emitter(Vec3::splat(4.0)),
            },
        ];
        let camera = Camera {
            position: Vec3::new(0.0, 2.5, 6.0),
            look_at: Vec3::new(0.0, 1.0, 0.0),
            up: Vec3::Y,
            vfov_degrees: 45.0,
            lens: None,
        };
        let scene = Scene::new(&gpu, &objects, camera, &Environment::constant(Vec3::splat(0.5)))
            .expect("scene");
        assert_spatial_reuse_matches_brute_force(&gpu, &scene, "diffuse GI");
    }

    /// The step-3a checkpoint (D-137): the un-baked reconnection shift is
    /// exact on a *direction-dependent* x₂. The bounce surface is a glossy
    /// metal (roughness 0.3) — a vertex the T2 baked-radiance guard provably
    /// refused outright (any metalness), because its cached Lo changed with
    /// the viewing direction — behind a diffuse primary, so the ground is lit
    /// substantially by the metal's reflection and the reconnection samples
    /// that carry it now shift across pixels. The pair criteria admit it
    /// (roughness past the 0.2 floor), so spatial reuse re-shades f₂ per
    /// neighbour: a stale baked radiance, a wrong terminal-MIS re-evaluation,
    /// or a mis-decoded ωₖ shifts the converged mean against the brute-force
    /// path tracer, which this catches.
    #[test]
    fn restir_reuses_a_glossy_metal_reconnection_vertex() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let objects = [
            Object {
                mesh: icosphere(2),
                transform: Mat4::from_translation(Vec3::Y),
                material: Material::glossy(Vec3::new(0.9, 0.7, 0.4), 0.0, 0.3)
                    .with_metalness(1.0),
            },
            Object {
                mesh: ground_plane(4.0),
                transform: Mat4::IDENTITY,
                material: Material::matte(Vec3::splat(0.8), 0.5),
            },
            Object {
                mesh: icosphere(2),
                transform: Mat4::from_translation(Vec3::Y * 3.0)
                    * Mat4::from_scale(Vec3::splat(0.7)),
                material: Material::emitter(Vec3::splat(4.0)),
            },
        ];
        let camera = Camera {
            position: Vec3::new(0.0, 2.5, 6.0),
            look_at: Vec3::new(0.0, 1.0, 0.0),
            up: Vec3::Y,
            vfov_degrees: 45.0,
            lens: None,
        };
        let scene = Scene::new(&gpu, &objects, camera, &Environment::constant(Vec3::splat(0.5)))
            .expect("scene");
        assert_spatial_reuse_matches_brute_force(&gpu, &scene, "a glossy metal bounce");
    }

    /// The step-3b prefix scene: a glossy metal *panel* at roughness 0.12 —
    /// below the 0.2 pair-criteria floor, so the (x₁, x₂) pair never
    /// qualifies on its pixels — filling the camera's view, with a matte
    /// sphere-and-ground interreflection (and an emitter) living in its
    /// reflection. The GI the panel pixels carry reuses only through k = 3
    /// samples: the glossy bounce replays from the stored seed and the
    /// reconnection lands one vertex deeper, on a diffuse-diffuse pair
    /// (D-138). The sphere sits in frame too, so the k = 2 shape lives
    /// beside it.
    fn glossy_primary_scene(gpu: &Context) -> Scene {
        let objects = [
            Object {
                mesh: ground_plane(3.0),
                transform: Mat4::from_translation(Vec3::new(0.0, 2.0, -1.5))
                    * Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2),
                material: Material::glossy(Vec3::new(0.9, 0.8, 0.6), 0.0, 0.12)
                    .with_metalness(1.0),
            },
            Object {
                mesh: ground_plane(4.0),
                transform: Mat4::IDENTITY,
                material: Material::matte(Vec3::splat(0.8), 0.5),
            },
            Object {
                mesh: icosphere(2),
                transform: Mat4::from_translation(Vec3::new(0.9, 1.0, 0.6)),
                material: Material::matte(Vec3::new(0.75, 0.4, 0.35), 0.5),
            },
            Object {
                mesh: icosphere(2),
                transform: Mat4::from_translation(Vec3::new(0.0, 3.6, 1.0))
                    * Mat4::from_scale(Vec3::splat(0.7)),
                material: Material::emitter(Vec3::splat(4.0)),
            },
        ];
        let camera = Camera {
            position: Vec3::new(0.0, 2.0, 4.5),
            look_at: Vec3::new(0.0, 1.8, -1.5),
            up: Vec3::Y,
            vfov_degrees: 45.0,
            lens: None,
        };
        Scene::new(gpu, &objects, camera, &Environment::constant(Vec3::splat(0.5)))
            .expect("scene")
    }

    /// The step-3b roulette scene: two facing near-mirror walls (roughness
    /// 0.06, far below the pair floor) around a matte floor and back wall,
    /// with the camera angled into the corridor so paths walk several sharp
    /// bounces before their first diffuse pair. Locks land at k = 4 and
    /// deeper; from k = 5 on, the reconnection draw sits at bounce 3 and the
    /// walk's roulette *rolls on prefix bounces* — exercising the D-138 fold
    /// (survival into the candidate weight, never into the replayed
    /// integrand) that lets the shift replay without rolling.
    fn mirror_chain_scene(gpu: &Context) -> Scene {
        let mirror = Material::glossy(Vec3::new(0.85, 0.85, 0.9), 0.0, 0.06)
            .with_metalness(1.0);
        let objects = [
            Object {
                mesh: ground_plane(2.0),
                transform: Mat4::from_translation(Vec3::new(-1.5, 1.5, 0.0))
                    * Mat4::from_rotation_z(-std::f32::consts::FRAC_PI_2),
                material: mirror,
            },
            Object {
                mesh: ground_plane(2.0),
                transform: Mat4::from_translation(Vec3::new(1.5, 1.5, 0.0))
                    * Mat4::from_rotation_z(std::f32::consts::FRAC_PI_2),
                material: mirror,
            },
            Object {
                mesh: ground_plane(4.0),
                transform: Mat4::IDENTITY,
                material: Material::matte(Vec3::splat(0.8), 0.5),
            },
            Object {
                mesh: ground_plane(2.0),
                transform: Mat4::from_translation(Vec3::new(0.0, 1.5, -2.0))
                    * Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2),
                material: Material::matte(Vec3::new(0.7, 0.75, 0.6), 0.5),
            },
            Object {
                mesh: icosphere(2),
                transform: Mat4::from_translation(Vec3::new(0.0, 3.0, 0.0))
                    * Mat4::from_scale(Vec3::splat(0.5)),
                material: Material::emitter(Vec3::splat(6.0)),
            },
        ];
        let camera = Camera {
            position: Vec3::new(1.2, 1.5, 0.8),
            look_at: Vec3::new(-1.5, 1.3, 0.3),
            up: Vec3::Y,
            vfov_degrees: 45.0,
            lens: None,
        };
        Scene::new(gpu, &objects, camera, &Environment::constant(Vec3::splat(0.3)))
            .expect("scene")
    }

    /// The step-3b unbiasedness checkpoint on the replayed prefix (D-138):
    /// a sub-floor glossy primary forces every reusable GI sample through the
    /// k = 3 shape — seed-replayed first bounce, reconnection one vertex
    /// deeper. A wrong replay draw, a prefix accumulated in the wrong
    /// measure, or a stale cached Jacobian half shifts the converged mean
    /// against the brute-force path tracer, which this catches.
    #[test]
    fn restir_replays_the_prefix_behind_a_glossy_primary() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let scene = glossy_primary_scene(&gpu);
        assert_spatial_reuse_matches_brute_force(&gpu, &scene, "a glossy primary's replayed prefix");
    }

    /// The step-3b roulette checkpoint (D-138): deep sharp chains lock at
    /// k ≥ 5, where generation's roulette rolls on prefix bounces. The fold —
    /// survival probabilities into the candidate weight, never into the
    /// stored integrand — is what keeps a survived base path from becoming a
    /// killed shifted path (§6's RR-replay trap). Folding on the wrong side,
    /// or replaying a roll, shifts the converged mean against brute force.
    #[test]
    fn restir_folds_roulette_out_of_the_replayed_prefix() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let scene = mirror_chain_scene(&gpu);
        assert_spatial_reuse_matches_brute_force(&gpu, &scene, "a roulette-deep mirror chain");
    }

    /// The step-3c scene: the mirror corridor with *every* surface — floor,
    /// back wall, and the emitter itself — a metal below the 0.2 roughness
    /// floor, so the pair criteria never pass anywhere and no reconnection
    /// lock ever forms. Everything the reservoir carries past the primary hit
    /// is a pure-replay sample (D-139): whole-path seed replay, the terminal
    /// re-drawn from the stored dims. The emitter is metal-based (emission
    /// rides any base) precisely so its surface fails the roughness half too
    /// — a matte emitter would hand the walk a lockable pair.
    fn sharp_chain_scene(gpu: &Context) -> Scene {
        let mirror = Material::glossy(Vec3::new(0.85, 0.85, 0.9), 0.0, 0.06)
            .with_metalness(1.0);
        let objects = [
            Object {
                mesh: ground_plane(2.0),
                transform: Mat4::from_translation(Vec3::new(-1.5, 1.5, 0.0))
                    * Mat4::from_rotation_z(-std::f32::consts::FRAC_PI_2),
                material: mirror,
            },
            Object {
                mesh: ground_plane(2.0),
                transform: Mat4::from_translation(Vec3::new(1.5, 1.5, 0.0))
                    * Mat4::from_rotation_z(std::f32::consts::FRAC_PI_2),
                material: mirror,
            },
            Object {
                mesh: ground_plane(4.0),
                transform: Mat4::IDENTITY,
                material: Material::glossy(Vec3::splat(0.8), 0.0, 0.12).with_metalness(1.0),
            },
            Object {
                mesh: ground_plane(2.0),
                transform: Mat4::from_translation(Vec3::new(0.0, 1.5, -2.0))
                    * Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2),
                material: Material::glossy(Vec3::new(0.7, 0.75, 0.6), 0.0, 0.15)
                    .with_metalness(1.0),
            },
            Object {
                mesh: icosphere(2),
                transform: Mat4::from_translation(Vec3::new(0.0, 3.0, 0.0))
                    * Mat4::from_scale(Vec3::splat(0.5)),
                material: Material {
                    emission: Vec3::splat(6.0),
                    ..Material::glossy(Vec3::splat(0.9), 0.0, 0.05).with_metalness(1.0)
                },
            },
        ];
        let camera = Camera {
            position: Vec3::new(1.2, 1.5, 0.8),
            look_at: Vec3::new(-1.5, 1.3, 0.3),
            up: Vec3::Y,
            vfov_degrees: 45.0,
            lens: None,
        };
        Scene::new(gpu, &objects, camera, &Environment::constant(Vec3::splat(0.3)))
            .expect("scene")
    }

    /// The step-3c unbiasedness checkpoint (D-139): with reconnection gated
    /// away everywhere, the whole reusable signal rides the pure-replay kind
    /// — whole-path seed replay across pixels, the terminal event re-drawn
    /// from the stored dims at each destination. A wrong replayed draw, a
    /// terminal re-formed in the wrong shape, a survival folded on the wrong
    /// side, or a Jacobian applied where the PSS identity owes none shifts
    /// the converged mean against the brute-force path tracer, which this
    /// catches. (That no lock ever forms here is pinned by the bit-identity
    /// gate's coverage assert, so this can't silently degenerate into a
    /// reconnection test.)
    #[test]
    fn restir_replays_whole_paths_on_a_sharp_chain() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let scene = sharp_chain_scene(&gpu);
        assert_spatial_reuse_matches_brute_force(&gpu, &scene, "an all-sharp mirror chain");
    }

    /// The step-4 evidence scene (§4b): a sharp metal panel — roughness 0.12,
    /// far below classic's 0.2 floor — standing *far* (16 m) from a diffuse
    /// floor lit exclusively through it. The emitter card floats just in
    /// front of the panel, facing it, so its one-sided front halfspace holds
    /// the panel alone: no surface anywhere sees it directly, and the floor's
    /// every path is floor → panel → card. That (floor, panel) pair is the
    /// discriminator: classic refused it on the panel's roughness and fell to
    /// whole-path replay; the footprint criterion accepts it on the long
    /// segment — reconnection at k = 2, Enhanced §4's distant-glossy win.
    /// The numbers sit where Eq 5 itself says yes: the roughness-0.12 lobe's
    /// footprint on the floor at 16 m clears (c/100)·the primary spread with
    /// ~2× margin, while classic's 0.2 floor refuses outright — sharper or
    /// nearer would fail the *inverse* test too and prove nothing. The panel
    /// fills the frame's top band, keeping the sub-floor-primary replay shape
    /// exercised beside the win.
    fn distant_glossy_scene(gpu: &Context) -> Scene {
        let objects = [
            Object {
                mesh: ground_plane(6.0),
                transform: Mat4::IDENTITY,
                material: Material::matte(Vec3::splat(0.8), 0.5),
            },
            Object {
                mesh: ground_plane(6.0),
                transform: Mat4::from_translation(Vec3::new(0.0, 4.0, -16.0))
                    * Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2),
                material: Material::glossy(Vec3::new(0.9, 0.85, 0.7), 0.0, 0.12)
                    .with_metalness(1.0),
            },
            // The one-sided card: high enough that the floor's sightlines to
            // its mirror image pass under its own silhouette, so the hot band
            // it paints on the panel reaches the whole camera-visible floor.
            Object {
                mesh: ground_plane(2.1),
                transform: Mat4::from_translation(Vec3::new(0.0, 8.0, -13.5))
                    * Mat4::from_rotation_x(-std::f32::consts::FRAC_PI_2),
                material: Material::emitter(Vec3::splat(40.0)),
            },
        ];
        let camera = Camera {
            position: Vec3::new(0.0, 3.0, 5.0),
            look_at: Vec3::new(0.0, 0.0, -3.0),
            up: Vec3::Y,
            vfov_degrees: 45.0,
            lens: None,
        };
        Scene::new(gpu, &objects, camera, &Environment::constant(Vec3::ZERO))
            .expect("scene")
    }

    /// The step-4 unbiasedness checkpoint: the footprint criterion's headline
    /// acceptance — reconnecting *through* a sharp vertex when the segment is
    /// long — must leave the converged mean on the brute-force line. A wrong
    /// forward test, a mis-stashed inverse factor, or demote bookkeeping
    /// leaking into the streamed weights shifts it, which this catches.
    #[test]
    fn restir_reconnects_through_a_distant_sharp_panel() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let scene = distant_glossy_scene(&gpu);
        assert_spatial_reuse_matches_brute_force(&gpu, &scene, "a distant sharp panel");
    }

    /// The step-3b/3c gate proper (D-138/D-139): **same-pixel replay
    /// bit-identity** — the strongest invariant the hybrid shift offers. Run
    /// the candidate stage once, then shift every pixel's own stored survivor
    /// into its own domain through the *shipping* shift — `shiftReconnection`
    /// or `shiftReplay` by the stored kind: the replayed integrand must equal
    /// the stored F **bit for bit**, and the Jacobian must be 1 to the ulp
    /// (a reconnection's squared-distance ratio's last bit belongs to
    /// per-kernel fma contraction; a replay's is 1 by construction). A single
    /// mis-associated multiply, a draw keyed off the wrong stream, a roulette
    /// division leaking into the replayed chain, or a re-shade context
    /// drifting from generation's breaks the equality outright — silent-bias
    /// classes a convergence gate would smear into plausible noise. Asserted
    /// on all three shift scenes so the k = 2 degenerate, the k = 3
    /// single-hop replay, the k ≥ 5 roulette-crossed replay, and the
    /// pure-replay kind's two terminal shapes are each pinned live.
    #[test]
    fn restir_same_pixel_replay_is_bit_identical() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let coverage = assert_same_pixel_replay_identity(
            &gpu,
            &glossy_primary_scene(&gpu),
            "the glossy-primary scene",
        );
        assert!(
            coverage.reconnection_ks.contains(&2),
            "the matte sphere should hold k = 2 survivors"
        );
        assert!(
            coverage.reconnection_ks.iter().any(|&k| k > 2),
            "the glossy ground should hold replay-prefixed (k > 2) survivors"
        );
        let coverage = assert_same_pixel_replay_identity(
            &gpu,
            &mirror_chain_scene(&gpu),
            "the mirror-chain scene",
        );
        assert!(
            coverage.reconnection_ks.iter().any(|&k| k >= 5),
            "the mirror corridor should hold roulette-deep (k ≥ 5) survivors, got {:?}",
            coverage.reconnection_ks
        );
        let coverage = assert_same_pixel_replay_identity(
            &gpu,
            &sharp_chain_scene(&gpu),
            "the sharp-chain scene",
        );
        assert!(
            coverage.reconnection_ks.is_empty(),
            "no pair on the all-sharp chain should ever lock, got k {:?}",
            coverage.reconnection_ks
        );
        assert!(
            coverage.replay_bounces.iter().any(|&b| b >= 2),
            "the sharp corridor should hold multi-bounce replay survivors, got {:?}",
            coverage.replay_bounces
        );
        for (terminal, name) in [(0, "NEE-redraw"), (1, "scatter-found")] {
            assert!(
                coverage.replay_terminals.contains(&terminal),
                "the sharp chain should hold {name} replay survivors, got {:?}",
                coverage.replay_terminals
            );
        }
        // The step-4 pin (§4b): the footprint criterion reconnects through
        // the sharp distant panel at k = 2 — the acceptance classic's
        // roughness floor structurally refused. Regressing toward a
        // roughness-gated x_k would empty this and trip it.
        let coverage = assert_same_pixel_replay_identity(
            &gpu,
            &distant_glossy_scene(&gpu),
            "the distant-glossy scene",
        );
        assert!(
            coverage.reconnection_ks.contains(&2),
            "the distant sharp panel should hold k = 2 reconnection survivors \
             under the footprint criterion, got {:?}",
            coverage.reconnection_ks
        );
    }

    /// The measurement seam of the step-4+ checkpoints (§4b, §4c) — ms/frame
    /// of the full `ReSTIR` wave on the perf-tracked scenes, spatial off
    /// (candidates alone — subtracting it from the next row prices the
    /// spatial stage, step 7's denominator, §4e), temporal off (candidates +
    /// spatial, the step-4 baseline), and temporal on. The on
    /// row pins temporal **live** (decay 0, held camera, identity
    /// reprojection): the steady-state cost of the stage while it works. The
    /// shipping ramp anneals a held camera back to the off row within
    /// [`Wavefront::RESTIR_TEMPORAL_DECAY_FRAMES`], and the decayed-zero fold
    /// then spends no shift work at all (§4c decision 3), so the live row is
    /// the honest worst case, not the converged still's. A report, not a
    /// gate: `#[ignore]` keeps it out of the suite; run it explicitly when a
    /// rung's checkpoint needs numbers:
    ///
    /// ```text
    /// cargo test -p cenote --release restir_frame_time_report -- \
    ///     --ignored --test-threads=1 --nocapture
    /// ```
    ///
    /// `trace_then` blocks on its submission fence, so wall-clock around it
    /// is the frame.
    #[expect(
        clippy::too_many_lines,
        reason = "the two timed configurations and the temporal four-buffer \
                  wiring are one report — splitting them would scatter what \
                  the numbers compare"
    )]
    #[test]
    #[ignore = "a measurement report, not a gate — run explicitly for checkpoint numbers"]
    fn restir_frame_time_report() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let (width, height) = (512u32, 512u32);
        let pixels = u64::from(width) * u64::from(height);
        let scenes = vec![
            ("demo", Scene::demo(&gpu).expect("demo scene")),
            ("glossy-primary", glossy_primary_scene(&gpu)),
            ("distant-glossy", distant_glossy_scene(&gpu)),
        ];
        let reservoir_bytes = pixels * size_of::<crate::restir::StoredPathReservoir>() as u64;
        let gbuffer_bytes = pixels * 48;
        let store_usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_DST;
        let buffer = |name: &str, bytes: u64| {
            gpu.create_buffer(name, bytes, store_usage, MemoryLocation::GpuOnly)
                .expect("buffer")
        };
        let scratch = buffer("test.time.scratch", reservoir_bytes);
        let cand = buffer("test.time.cand", reservoir_bytes);
        let mut prev = buffer("test.time.prev", reservoir_bytes);
        let mut curr = buffer("test.time.curr", reservoir_bytes);
        let mut gbuffer_prev = buffer("test.time.gbuffer.prev", gbuffer_bytes);
        let mut gbuffer_curr = buffer("test.time.gbuffer.curr", gbuffer_bytes);
        let mut reproject_buf = gpu
            .create_buffer(
                "test.time.reproject",
                96,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                MemoryLocation::CpuToGpu,
            )
            .expect("reproject");
        let wavefront = Wavefront::new(
            &gpu,
            &Kernels::embedded(),
            width * height,
            Wavefront::DEFAULT_MAX_BOUNCES,
            LightSampling::Mis,
        )
        .expect("wavefront");
        let radiance = radiance_buffer(&gpu, width, height);
        let (warmup, timed) = (8u32, 64u32);
        for (name, scene) in &scenes {
            // No temporal: candidates alone, then candidates + spatial (the
            // step-4 baseline's wave) — the subtraction prices the spatial
            // stage itself, the denominator of step 7's sharing claim (§4e).
            let wave_ms = |spatial: Option<&Buffer>| {
                let inputs = RestirInputs {
                    reservoir: &curr,
                    temporal: None,
                    scratch: spatial,
                    // Flat across the loop: this averages wall-clock over
                    // `sample` 0..warmup + timed, and a per-sample cost that
                    // varied with the index is not what the subtraction prices.
                    candidates: Wavefront::RESTIR_CANDIDATES,
                    cv_shading: true,
                    debug: None,
                    debug_view: DebugView::Off,
                };
                let mut started = std::time::Instant::now();
                for sample in 0..warmup + timed {
                    if sample == warmup {
                        started = std::time::Instant::now();
                    }
                    wavefront
                        .trace_then(
                            &gpu, scene, &radiance, width, height, sample, None, Some(&inputs), &[], None,
                        )
                        .expect("trace");
                }
                started.elapsed().as_secs_f64() * 1e3 / f64::from(timed)
            };
            let alone_ms = wave_ms(None);
            let off_ms = wave_ms(Some(&scratch));
            // Temporal on, pinned live: candidates + temporal + spatial, the
            // renderer's four-buffer routing on a held camera. `prev` and the
            // G-buffers clear per scene — history from another scene would
            // hold that scene's instance indices.
            let on_ms = {
                gpu.submit_passes(&[
                    Pass::Fill { buffer: &prev, offset: 0, size: reservoir_bytes, value: 0 },
                    Pass::Fill { buffer: &gbuffer_prev, offset: 0, size: gbuffer_bytes, value: 0 },
                    Pass::Fill { buffer: &gbuffer_curr, offset: 0, size: gbuffer_bytes, value: 0 },
                ])
                .expect("clear");
                let camera = *scene.camera();
                let basis = camera.basis(width as f32 / height as f32);
                let cam = ReprojectCamera {
                    position: camera.position,
                    right: basis.right,
                    up: basis.up,
                    forward: basis.forward,
                };
                let mut started = std::time::Instant::now();
                for sample in 0..warmup + timed {
                    if sample == warmup {
                        started = std::time::Instant::now();
                    }
                    let reproject = Reproject::new(
                        (sample > 0).then_some(cam),
                        gbuffer_prev.device_address(),
                        gbuffer_curr.device_address(),
                        width,
                        height,
                    );
                    reproject_buf.write(bytemuck::bytes_of(&reproject));
                    let inputs = RestirInputs {
                        reservoir: &curr,
                        temporal: Some(TemporalReuse {
                            cand: &cand,
                            prev: &prev,
                            reproject: &reproject_buf,
                            decay_frames: 0,
                            prev_same_scene: true,
                        }),
                        scratch: Some(&scratch),
                        // Flat, for the same reason `decay_frames` is 0 here.
                        candidates: Wavefront::RESTIR_CANDIDATES,
                        cv_shading: true,
                        debug: None,
                        debug_view: DebugView::Off,
                    };
                    wavefront
                        .trace_then(
                            &gpu, scene, &radiance, width, height, sample, None, Some(&inputs), &[], None,
                        )
                        .expect("trace");
                    std::mem::swap(&mut prev, &mut curr);
                    std::mem::swap(&mut gbuffer_prev, &mut gbuffer_curr);
                }
                started.elapsed().as_secs_f64() * 1e3 / f64::from(timed)
            };
            eprintln!(
                "frame time on {name}: spatial-off {alone_ms:.2} ms, \
                 temporal-off {off_ms:.2} ms, temporal-on {on_ms:.2} ms/frame \
                 at {width}x{height} ({timed} frames)"
            );
        }
    }

    /// What the bit-identity harness saw evaluated: the k of every shiftable
    /// reconnection survivor, and the terminal bounce and kind (0 = NEE
    /// redraw, 1 = scatter-found) of every replay survivor — for the callers'
    /// coverage asserts. `nee` counts the length-2 direct-lighting survivors,
    /// completing the kind-mix denominator the step-4 baseline reports (§4b).
    struct ShiftCoverage {
        reconnection_ks: Vec<u32>,
        replay_bounces: Vec<u32>,
        replay_terminals: Vec<u32>,
        nee: usize,
    }

    /// The bit-identity harness: one candidates frame into `reservoir`, then
    /// the `restir_shift_test` fixture shifts each pixel's stored survivor at
    /// its own pixel and the host compares bitwise.
    #[allow(clippy::too_many_lines, reason = "one linear frame-then-verify sequence")]
    fn assert_same_pixel_replay_identity(gpu: &Context, scene: &Scene, what: &str) -> ShiftCoverage {
        /// Mirrors `struct Params` in `shaders/restir_shift_test.slang`.
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct ShiftTestParams {
            paths: PathsAddrs,
            hits: QueueAddrs,
            scene: vk::DeviceAddress,
            reservoirs: vk::DeviceAddress,
            shifted: vk::DeviceAddress,
            kinds: vk::DeviceAddress,
        }
        const SHIFT_EVALUATED: u32 = 0x100; // mirrors the fixture's tags
        const SHIFT_BROKEN: u32 = 0x200;
        const SHIFT_REPLAY: u32 = 0x400;
        const SHIFT_NEE: u32 = 0x800;

        let (width, height) = (32u32, 32u32);
        let pixels = u64::from(width) * u64::from(height);
        let reservoir = gpu
            .create_buffer(
                "test.shift.reservoir",
                pixels * size_of::<crate::restir::StoredPathReservoir>() as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuOnly,
            )
            .expect("reservoir buffer");
        let wavefront = Wavefront::new(
            gpu,
            &Kernels::embedded(),
            4096,
            Wavefront::DEFAULT_MAX_BOUNCES,
            LightSampling::Mis,
        )
        .expect("wavefront");
        // Candidates only (no temporal, no spatial): `reservoir` holds the
        // candidate stage's own-pixel survivors, exactly what generation
        // stored.
        let inputs = RestirInputs {
            reservoir: &reservoir,
            temporal: None,
            scratch: None,
            candidates: Wavefront::RESTIR_CANDIDATES,
            cv_shading: true,
            debug: None,
            debug_view: DebugView::Off,
        };
        let radiance = radiance_buffer(gpu, width, height);
        let spirv =
            crate::shaders::compile_fixture("restir_shift_test").expect("compile restir_shift_test");
        let pipeline = gpu
            .create_compute_pipeline(
                &spirv,
                c"restir_shift_test",
                size_of::<ShiftTestParams>() as u32,
                Bindings::Scene,
            )
            .expect("pipeline");
        // Zero-uploaded outputs: pixels the fixture never reaches (misses)
        // must read as "not evaluated", not as stale VRAM.
        let out_usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_SRC;
        let shifted = gpu
            .upload_buffer("test.shift.f", &vec![0u8; (pixels * 16) as usize], out_usage)
            .expect("shifted buffer");
        let kinds = gpu
            .upload_buffer("test.shift.kinds", &vec![0u8; (pixels * 4) as usize], out_usage)
            .expect("kinds buffer");
        let params = ShiftTestParams {
            paths: wavefront.paths.addresses(),
            hits: wavefront.queues.addresses(queue::HIT, &wavefront.queues.hit),
            scene: scene.table().device_address(),
            reservoirs: reservoir.device_address(),
            shifted: shifted.device_address(),
            kinds: kinds.device_address(),
        };
        let bindings = SceneBindings {
            tlas: scene.tlas(),
            environment: scene.environment(),
            textures: scene.texture_descriptors(),
            blue_noise: &wavefront.blue_noise,
            pairing: &wavefront.pairing,
        };

        // A handful of candidate frames: each pixel keeps one survivor per
        // frame, so sweeping a few sample indices unions enough shiftable
        // winners to pin every shape the callers assert on. Bitwise
        // deterministic, like everything upstream — the coverage cannot flake.
        let mut coverage = ShiftCoverage {
            reconnection_ks: Vec::new(),
            replay_bounces: Vec::new(),
            replay_terminals: Vec::new(),
            nee: 0,
        };
        for sample in 0..8 {
            wavefront
                .trace_then(
                    gpu,
                    scene,
                    &radiance,
                    width,
                    height,
                    sample,
                    None,
                    Some(&inputs),
                    &[],
                    None,
                )
                .expect("trace");
            gpu.dispatch(
                &pipeline,
                Some(bindings),
                bytemuck::bytes_of(&params),
                [(width * height).div_ceil(WORKGROUP_SIZE), 1, 1],
            )
            .expect("dispatch");

            let stored: Vec<crate::restir::StoredPathReservoir> =
                bytemuck::pod_collect_to_vec(&gpu.download_buffer(&reservoir).expect("download"));
            let results: Vec<[f32; 4]> =
                bytemuck::pod_collect_to_vec(&gpu.download_buffer(&shifted).expect("download"));
            let tags: Vec<u32> =
                bytemuck::pod_collect_to_vec(&gpu.download_buffer(&kinds).expect("download"));

            for pixel in 0..(width * height) as usize {
                let tag = tags[pixel];
                assert_eq!(
                    tag & SHIFT_BROKEN,
                    0,
                    "pixel {pixel} sample {sample} on {what}: a shiftable survivor's \
                     own-pixel shift came back invalid — generation and shift share \
                     one validity predicate, so this is a lock/shift divergence"
                );
                if tag & SHIFT_NEE != 0 {
                    coverage.nee += 1;
                }
                if tag & SHIFT_EVALUATED == 0 {
                    continue;
                }
                let k = tag & 0xff; // reconnection k, or the replay terminal bounce
                let f = stored[pixel].sample.f;
                let got = results[pixel];
                assert!(
                    f[0].to_bits() == got[0].to_bits()
                        && f[1].to_bits() == got[1].to_bits()
                        && f[2].to_bits() == got[2].to_bits(),
                    "pixel {pixel} sample {sample} on {what} (k/bounce = {k}): the \
                     same-pixel replay must reproduce the stored integrand bit \
                     for bit: stored {f:?}, replayed {:?}",
                    &got[..3]
                );
                // The reconnection Jacobian ratio rides a squared segment
                // length whose last-place bit belongs to each kernel's fma
                // contraction, so "exactly 1" is pinned to within one ulp
                // (the replay kind reports a constructed 1.0); any real
                // divergence — a stale cached half, a wrong vertex or surface
                // — is orders of magnitude larger.
                assert!(
                    (got[3] - 1.0).abs() <= f32::EPSILON,
                    "pixel {pixel} sample {sample} on {what} (k/bounce = {k}): the \
                     same-pixel Jacobian must be 1 to the ulp, got {}",
                    got[3]
                );
                if tag & SHIFT_REPLAY != 0 {
                    coverage.replay_bounces.push(k);
                    coverage.replay_terminals.push((tag >> 12) & 0x3);
                } else {
                    coverage.reconnection_ks.push(k);
                }
            }
        }
        // The kind mix — the step-4 checkpoint metric (§4b): footprint
        // criteria should move survivors from replay toward reconnection.
        let (nee, rc, rp) = (
            coverage.nee,
            coverage.reconnection_ks.len(),
            coverage.replay_bounces.len(),
        );
        eprintln!(
            "kind mix on {what}: {nee} NEE / {rc} reconnection / {rp} replay \
             ({} live survivors over 8 frames)",
            nee + rc + rp
        );
        assert!(
            !(coverage.reconnection_ks.is_empty() && coverage.replay_bounces.is_empty()),
            "no shiftable survivor on {what} — the gate would be vacuous"
        );
        coverage
    }

    /// Emitters reflect too: the reflected component of an emissive *and*
    /// reflective surface — light bouncing off a glowing object onto its
    /// surroundings — rides the BSDF draw's continuation-tail candidate, a
    /// separate sample from the emission the light candidate carries (D-134).
    /// Dropping it (or double-counting the emission into the tail) would shift
    /// the converged mean against the brute-force path tracer, which this
    /// catches: the sphere glows dimly but reflects brightly, so the ground
    /// near it is lit substantially by the lost-term-if-lost.
    #[test]
    fn restir_carries_reflection_off_emissive_surfaces() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let objects = [
            Object {
                mesh: icosphere(2),
                transform: Mat4::from_translation(Vec3::Y),
                // Emissive *and* diffuse — the material Hydra can author
                // (emissiveColor beside diffuseColor) that a pure
                // `Material::emitter` (black base) never exercises.
                material: Material {
                    emission: Vec3::splat(0.5),
                    ..Material::matte(Vec3::splat(0.9), 0.5)
                },
            },
            Object {
                mesh: ground_plane(4.0),
                transform: Mat4::IDENTITY,
                material: Material::matte(Vec3::splat(0.8), 0.5),
            },
        ];
        let camera = Camera {
            position: Vec3::new(0.0, 2.5, 6.0),
            look_at: Vec3::new(0.0, 1.0, 0.0),
            up: Vec3::Y,
            vfov_degrees: 45.0,
            lens: None,
        };
        let scene = Scene::new(&gpu, &objects, camera, &Environment::constant(Vec3::ONE))
            .expect("scene");
        assert_spatial_reuse_matches_brute_force(&gpu, &scene, "an emissive-reflective surface");
    }

    /// The indirect tail must carry the path tracer's interior-medium state: a
    /// closed-surface refraction at the primary hit puts the scattered segment
    /// inside x₁'s own interior, so the tail's first vertex — the glass's far
    /// wall — shades at the inverted IOR, the segment absorbs by Beer–Lambert,
    /// and every later transmission toggles the medium from *inside*, not
    /// vacuum (D-134). `shade_surface`'s continuation carried all three in path
    /// state; the inline tail must seed them from the draw's lobe or everything
    /// seen *through* glass diverges from brute force — which this catches: the
    /// tinted sphere fills the frame's center, so its transmitted background
    /// and the light it passes onto the ground are a real fraction of the mean.
    #[test]
    fn restir_tracks_interior_media_through_the_indirect_tail() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let mut glass = Material::glass(0.4, 1.5);
        glass.transmission_color = Vec3::new(0.85, 0.35, 0.2);
        glass.transmission_depth = 0.5;
        let objects = [
            Object {
                mesh: icosphere(2),
                transform: Mat4::from_translation(Vec3::Y),
                material: glass,
            },
            Object {
                mesh: ground_plane(4.0),
                transform: Mat4::IDENTITY,
                material: Material::matte(Vec3::splat(0.8), 0.5),
            },
        ];
        let camera = Camera {
            position: Vec3::new(0.0, 2.5, 6.0),
            look_at: Vec3::new(0.0, 1.0, 0.0),
            up: Vec3::Y,
            vfov_degrees: 45.0,
            lens: None,
        };
        let scene = Scene::new(&gpu, &objects, camera, &Environment::constant(Vec3::ONE))
            .expect("scene");
        assert_spatial_reuse_matches_brute_force(&gpu, &scene, "a tinted glass interior");
    }

    /// The shared unbiasedness harness of the path-reuse gates above: render
    /// `scene` with the brute-force path tracer and with candidates + spatial
    /// reuse, accumulate both to the same budget, and require the converged
    /// means to agree — the M3 gate (`ReSTIR` ≡ the path tracer), on whatever
    /// estimator seam the calling test's scene isolates.
    fn assert_spatial_reuse_matches_brute_force(gpu: &Context, scene: &Scene, what: &str) {
        let kernels = Kernels::embedded();
        let (width, height) = (32, 32);
        let samples: u32 = 256;
        let buffer = |name: &str| {
            gpu.create_buffer(
                name,
                u64::from(width)
                    * u64::from(height)
                    * size_of::<crate::restir::StoredPathReservoir>() as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuOnly,
            )
            .expect("reservoir buffer")
        };
        let reservoir = buffer("test.gi.reservoir");
        let scratch = buffer("test.gi.scratch");
        let wavefront = Wavefront::new(
            gpu,
            &kernels,
            4096,
            Wavefront::DEFAULT_MAX_BOUNCES,
            LightSampling::Mis,
        )
        .expect("wavefront");
        let spatial = RestirInputs {
            reservoir: &reservoir,
            temporal: None,
            scratch: Some(&scratch),
            candidates: Wavefront::RESTIR_CANDIDATES,
            cv_shading: true,
            debug: None,
            debug_view: DebugView::Off,
        };
        let mean = |inputs: Option<&RestirInputs>| -> f64 {
            let radiance = radiance_buffer(gpu, width, height);
            let mut total = 0.0;
            for sample in 0..samples {
                wavefront
                    .trace_then(
                        gpu, scene, &radiance, width, height, sample, None, inputs, &[], None,
                    )
                    .expect("trace");
                let pixels: Vec<f32> = bytemuck::pod_collect_to_vec(
                    &gpu.download_buffer(&radiance).expect("download"),
                );
                total += pixels
                    .chunks_exact(4)
                    .map(|pixel| f64::from(pixel[0]) + f64::from(pixel[1]) + f64::from(pixel[2]))
                    .sum::<f64>();
            }
            total / f64::from(samples * width * height)
        };

        let path = mean(None);
        assert!(path > 0.01, "the scene should be lit, got mean {path}");
        let restir = mean(Some(&spatial));
        let deviation = (restir - path).abs() / path;
        assert!(
            deviation < 0.03,
            "spatial path reuse disagrees with the path tracer on {what}: \
             {restir} vs {path} ({deviation:.4} relative)"
        );
    }

    /// The scene the temporal gates share — a lit environment *and* an area
    /// emitter, so both candidate branches (environment and area light) feed the
    /// compounding history — and the held camera that views it.
    fn temporal_gate_scene(gpu: &crate::gpu::Context) -> (Scene, Camera) {
        let objects = [
            Object {
                mesh: icosphere(2),
                transform: Mat4::from_translation(Vec3::Y),
                material: Material::glossy(Vec3::splat(0.6), 0.4, 0.3).with_metalness(0.5),
            },
            Object {
                mesh: ground_plane(4.0),
                transform: Mat4::IDENTITY,
                material: Material::glossy(Vec3::splat(0.7), 0.0, 0.2),
            },
            Object {
                mesh: icosphere(2),
                transform: Mat4::from_translation(Vec3::Y * 3.0)
                    * Mat4::from_scale(Vec3::splat(0.7)),
                material: Material::emitter(Vec3::splat(4.0)),
            },
        ];
        let camera = Camera {
            position: Vec3::new(0.0, 2.5, 6.0),
            look_at: Vec3::new(0.0, 1.0, 0.0),
            up: Vec3::Y,
            vfov_degrees: 45.0,
            lens: None,
        };
        let scene = Scene::new(gpu, &objects, camera, &Environment::constant(Vec3::splat(0.3)))
            .expect("scene");
        (scene, camera)
    }

    /// Run the temporal pipeline (held camera) and the matching path-tracer
    /// reference on `scene`, both to `samples` spp, and return
    /// `(temporal_mean, path_mean)`. `decay_frames` is the ramp window
    /// forwarded to `restir_temporal` (`0` disables it —
    /// [`Wavefront::RESTIR_TEMPORAL_DECAY_FRAMES`]); the convergence gates
    /// below differ in that argument and in the scene — sample-kind coverage
    /// is the caller's scene choice (§4c decision 4). `spatial_cv` selects the
    /// wave past temporal: `None` is the step-5 shape (spatial off — the
    /// kind-covering gates isolate the temporal combine), `Some(cv)` appends
    /// spatial reuse and resolves with control-variate shading on or off —
    /// the 6b-ii gate's full pipeline, where the temporal lane blend feeds
    /// spatial's CV combination. Factored out so the wiring — the four-buffer
    /// routing, the per-frame reprojection block, and the frame-end swap of
    /// prev/curr and the G-buffers — is written and audited once. On the held
    /// camera reprojection is the identity pixel, the disocclusion gate
    /// passes, and the G-buffer is written, swapped, and read every frame
    /// while history compounds under the M-cap.
    #[expect(
        clippy::too_many_lines,
        reason = "the four-buffer temporal wiring and the per-frame swap are one \
                  harness — splitting them would scatter what the gates check"
    )]
    fn temporal_gate_means(
        gpu: &crate::gpu::Context,
        scene: &Scene,
        samples: u32,
        decay_frames: u32,
        spatial_cv: Option<bool>,
    ) -> (f64, f64) {
        let camera = *scene.camera();
        let kernels = Kernels::embedded();
        let (width, height) = (32u32, 32u32);
        let wavefront = Wavefront::new(
            gpu,
            &kernels,
            4096,
            Wavefront::DEFAULT_MAX_BOUNCES,
            LightSampling::Mis,
        )
        .expect("wavefront");

        let reservoir_bytes =
            u64::from(width) * u64::from(height) * size_of::<crate::restir::StoredPathReservoir>() as u64;
        let gbuffer_bytes = u64::from(width) * u64::from(height) * 48;
        let store_usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_DST;
        let reservoir = |name: &str| {
            gpu.create_buffer(name, reservoir_bytes, store_usage, MemoryLocation::GpuOnly)
                .expect("reservoir")
        };
        let gbuffer = |name: &str| {
            gpu.create_buffer(name, gbuffer_bytes, store_usage, MemoryLocation::GpuOnly)
                .expect("gbuffer")
        };

        // The path-tracer reference: the same integral, no reservoir at all.
        let path = {
            let radiance = radiance_buffer(gpu, width, height);
            let mut total = 0.0_f64;
            for sample in 0..samples {
                wavefront
                    .trace_then(gpu, scene, &radiance, width, height, sample, None, None, &[], None)
                    .expect("trace");
                let pixels: Vec<f32> = bytemuck::pod_collect_to_vec(
                    &gpu.download_buffer(&radiance).expect("download"),
                );
                total += pixels
                    .chunks_exact(4)
                    .map(|p| f64::from(p[0]) + f64::from(p[1]) + f64::from(p[2]))
                    .sum::<f64>();
            }
            total / f64::from(samples * width * height)
        };
        assert!(path > 0.01, "the scene should be lit, got mean {path}");

        // Temporal reuse, held camera; spatial and CV shading per `spatial_cv`.
        let temporal = {
            let cand = reservoir("test.temporal.cand");
            let mut prev = reservoir("test.temporal.prev");
            let mut curr = reservoir("test.temporal.curr");
            let scratch = spatial_cv.map(|_| reservoir("test.temporal.scratch"));
            let mut gbuffer_prev = gbuffer("test.temporal.gbuffer.prev");
            let mut gbuffer_curr = gbuffer("test.temporal.gbuffer.curr");
            // Empty reservoirs / G-buffers to start (frame 0 reads `prev` before
            // any candidate streamed; the wave clears `curr` itself each frame).
            gpu.submit_passes(&[
                Pass::Fill { buffer: &cand, offset: 0, size: reservoir_bytes, value: 0 },
                Pass::Fill { buffer: &prev, offset: 0, size: reservoir_bytes, value: 0 },
                Pass::Fill { buffer: &curr, offset: 0, size: reservoir_bytes, value: 0 },
                Pass::Fill { buffer: &gbuffer_prev, offset: 0, size: gbuffer_bytes, value: 0 },
                Pass::Fill { buffer: &gbuffer_curr, offset: 0, size: gbuffer_bytes, value: 0 },
            ])
            .expect("clear");
            let mut reproject_buf = gpu
                .create_buffer(
                    "test.temporal.reproject",
                    96,
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    MemoryLocation::CpuToGpu,
                )
                .expect("reproject");
            // Held camera: the previous camera equals the current one every frame.
            let basis = camera.basis(width as f32 / height as f32);
            let cam = ReprojectCamera {
                position: camera.position,
                right: basis.right,
                up: basis.up,
                forward: basis.forward,
            };

            let radiance = radiance_buffer(gpu, width, height);
            let mut total = 0.0_f64;
            for sample in 0..samples {
                // The renderer's per-frame reprojection block: `valid` from the
                // second frame on (a previous camera then exists), the current
                // G-buffer addresses.
                let reproject = Reproject::new(
                    (sample > 0).then_some(cam),
                    gbuffer_prev.device_address(),
                    gbuffer_curr.device_address(),
                    width,
                    height,
                );
                reproject_buf.write(bytemuck::bytes_of(&reproject));
                let inputs = RestirInputs {
                    reservoir: &curr,
                    temporal: Some(TemporalReuse {
                        cand: &cand,
                        prev: &prev,
                        reproject: &reproject_buf,
                        decay_frames,
                        // No edits land mid-harness: every frame renders the
                        // same scene build, as the renderer would report.
                        prev_same_scene: true,
                    }),
                    scratch: scratch.as_ref(),
                    candidates: Wavefront::RESTIR_CANDIDATES,
                    cv_shading: spatial_cv.unwrap_or(true),
                    debug: None,
                    debug_view: DebugView::Off,
                };
                wavefront
                    .trace_then(
                        gpu, scene, &radiance, width, height, sample, None, Some(&inputs), &[], None,
                    )
                    .expect("trace");
                let pixels: Vec<f32> = bytemuck::pod_collect_to_vec(
                    &gpu.download_buffer(&radiance).expect("download"),
                );
                total += pixels
                    .chunks_exact(4)
                    .map(|p| f64::from(p[0]) + f64::from(p[1]) + f64::from(p[2]))
                    .sum::<f64>();
                // Frame end: wind the ping-pong, reservoirs and G-buffers together.
                std::mem::swap(&mut prev, &mut curr);
                std::mem::swap(&mut gbuffer_prev, &mut gbuffer_curr);
            }
            total / f64::from(samples * width * height)
        };

        (temporal, path)
    }

    /// The pinned-temporal gate (D-094's CI spine): temporal reuse forced live —
    /// spatial off, decay *off* (window 0) — must still converge to the path
    /// tracer. The default estimator decays temporal off on a held camera (step
    /// 5d), which would leave the ordinary converge-to-reference gate blind to a
    /// temporal bias; so this holds temporal on to convergence instead. It runs
    /// the full step-5 pipeline the renderer does — the four-buffer routing, the
    /// frame-end prev/curr swap, and the per-frame reprojection block. History
    /// compounds under the M-cap across all 256 frames; a broken M-cap, a mixed
    /// feed convention, a stale target, or a gate that corrupted the surface it
    /// compares would shift the converged mean off the reference.
    #[test]
    fn temporal_reuse_matches_the_path_tracer() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        // Decay off: if the ramp were live it would anneal temporal away by frame
        // `RESTIR_TEMPORAL_DECAY_FRAMES`, silently turning this into a
        // candidate-only convergence check, blind to a temporal bias (D-094).
        let (scene, _) = temporal_gate_scene(&gpu);
        let (temporal, path) = temporal_gate_means(&gpu, &scene, 256, 0, None);
        let deviation = (temporal - path).abs() / path;
        assert!(
            deviation < 0.03,
            "temporal reuse disagrees with the path tracer: {temporal} vs {path} ({deviation:.4} relative)"
        );
    }

    /// The step-5b pinned-temporal gate on the replayed-prefix shape (§4c
    /// decision 4): the sub-floor glossy primary forces every reusable GI
    /// sample through k = 3 — seed-replayed first bounce, reconnection one
    /// vertex deeper — and those samples now cross the *frame* boundary
    /// through the shared shift block. Held live to convergence (decay off),
    /// the mean must still match the path tracer: a cross-frame Jacobian
    /// against the wrong cached half, a reconnection visibility trusted
    /// instead of re-traced, or a survivor stored un-re-rooted all shift it.
    #[test]
    fn temporal_reuse_matches_the_path_tracer_behind_a_glossy_primary() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let scene = glossy_primary_scene(&gpu);
        let (temporal, path) = temporal_gate_means(&gpu, &scene, 256, 0, None);
        let deviation = (temporal - path).abs() / path;
        assert!(
            deviation < 0.03,
            "temporal reuse disagrees with the path tracer behind a glossy \
             primary: {temporal} vs {path} ({deviation:.4} relative)"
        );
    }

    /// The step-5b pinned-temporal gate on the pure-replay shape (§4c
    /// decision 4): the all-sharp mirror corridor forms no reconnection lock
    /// anywhere, so every indirect sample crossing the frame boundary is a
    /// whole-path seed replay with its terminal — NEE or SCATTER — re-drawn
    /// from the stored dims. A replay keyed off the wrong stream across the
    /// boundary, a terminal-kind mismatch scored as a hit, or an NEE
    /// terminal's re-drawn connection left untested would all surface here as
    /// a converged mean off the reference.
    #[test]
    fn temporal_reuse_matches_the_path_tracer_on_a_sharp_chain() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let scene = sharp_chain_scene(&gpu);
        let (temporal, path) = temporal_gate_means(&gpu, &scene, 256, 0, None);
        let deviation = (temporal - path).abs() / path;
        assert!(
            deviation < 0.03,
            "temporal reuse disagrees with the path tracer on an all-sharp \
             chain: {temporal} vs {path} ({deviation:.4} relative)"
        );
    }

    /// [`glossy_primary_scene`] with its lighting turned indirect-only — the
    /// convergence suite's indirect-glossy regime (its builder there rides the
    /// public API; this is the same set through the internal one): the emitter
    /// becomes a one-sided plane facing the sub-floor metal panel, showing the
    /// world its dark back, and the environment goes black, so every
    /// camera-visible surface is lit exclusively through the panel's glossy
    /// reflection. Per pixel the good path is rare, so the reservoir carries
    /// most of the image and the CV lane carries most of its colour — the
    /// regime that makes the 6b-ii gate below bite.
    fn indirect_glossy_scene(gpu: &Context) -> Scene {
        let objects = [
            Object {
                mesh: ground_plane(3.0),
                transform: Mat4::from_translation(Vec3::new(0.0, 2.0, -1.5))
                    * Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2),
                material: Material::glossy(Vec3::new(0.9, 0.8, 0.6), 0.0, 0.12)
                    .with_metalness(1.0),
            },
            Object {
                mesh: ground_plane(4.0),
                transform: Mat4::IDENTITY,
                material: Material::matte(Vec3::splat(0.8), 0.5),
            },
            Object {
                mesh: icosphere(2),
                transform: Mat4::from_translation(Vec3::new(0.9, 1.0, 0.6)),
                material: Material::matte(Vec3::new(0.75, 0.4, 0.35), 0.5),
            },
            Object {
                mesh: ground_plane(1.0),
                transform: Mat4::from_translation(Vec3::new(0.0, 2.5, 2.5))
                    * Mat4::from_rotation_x(-std::f32::consts::FRAC_PI_2),
                material: Material::emitter(Vec3::splat(12.0)),
            },
        ];
        let camera = Camera {
            position: Vec3::new(0.0, 2.0, 4.5),
            look_at: Vec3::new(0.0, 1.8, -1.5),
            up: Vec3::Y,
            vfov_degrees: 45.0,
            lens: None,
        };
        Scene::new(gpu, &objects, camera, &Environment::constant(Vec3::ZERO))
            .expect("indirect glossy scene")
    }

    /// The 6b-ii pinned-live CV gate (M6 §4d): the **full** pipeline —
    /// temporal held live to convergence (decay off), spatial reuse, the
    /// control-variate resolve — on the indirect-glossy scene, against its
    /// own zero-CV degenerate and the path tracer. With the ramp disabled the
    /// lane blend compounds history across all 256 frames, so a broken
    /// recurrence — blend weights inconsistent with the capped confidence,
    /// F-semantics leaking into the G-lane across the frame boundary, a
    /// reset path that kept stale history — shifts the converged CV mean
    /// away from the survivor mean the same samples produce. The shipping
    /// ramp would anneal the blend off by frame 16 and dilute exactly the
    /// bias this pins (the same blindness D-094 named for resampling).
    #[test]
    fn temporal_cv_recurrence_matches_survivor_shading_pinned_live() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let scene = indirect_glossy_scene(&gpu);
        let (cv_on, path) = temporal_gate_means(&gpu, &scene, 256, 0, Some(true));
        let (cv_off, _) = temporal_gate_means(&gpu, &scene, 256, 0, Some(false));
        let on_vs_off = (cv_on - cv_off).abs() / cv_off;
        let on_vs_path = (cv_on - path).abs() / path;
        eprintln!(
            "pinned-live CV means: cv-on {cv_on:.5}, cv-off {cv_off:.5}, path {path:.5} \
             (on-vs-off {on_vs_off:.4}, on-vs-path {on_vs_path:.4} relative)"
        );
        assert!(
            on_vs_off < 0.03,
            "CV shading disagrees with survivor shading under live temporal \
             history: {cv_on} vs {cv_off} ({on_vs_off:.4} relative)"
        );
        assert!(
            on_vs_path < 0.03,
            "the temporal CV recurrence disagrees with the path tracer: \
             {cv_on} vs {path} ({on_vs_path:.4} relative)"
        );
    }

    /// The step-5b G-buffer-surface gate (§4c decision 4): the surface
    /// temporal reconstructs from a persisted `GBufEntry` must equal the
    /// path-pool reconstruction **bit for bit** — the entry is the one input
    /// temporal sources differently from spatial, so this is what transfers
    /// the same-pixel shift invariant (D-138/D-139) to the frame boundary.
    /// One temporal frame writes the real G-buffer; the fixture
    /// (`restir_gbuffer_test.slang`) then rebuilds both surfaces in a single
    /// invocation — layout, store, and load included — and reports a per-field
    /// mismatch mask this asserts empty at every hit pixel.
    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "the one-frame temporal wiring and the fixture dispatch are \
                  one flow — splitting them would scatter what the gate checks"
    )]
    fn temporal_gbuffer_surface_is_bit_identical() {
        /// Mirrors `struct Params` in `shaders/restir_gbuffer_test.slang`.
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct GbufferTestParams {
            paths: PathsAddrs,
            hits: QueueAddrs,
            scene: vk::DeviceAddress,
            gbuffer: vk::DeviceAddress,
            mismatch: vk::DeviceAddress,
        }
        const GBUF_EVALUATED: u32 = 0x8000_0000; // mirrors the fixture's tags

        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let (scene, _) = temporal_gate_scene(&gpu);
        let (width, height) = (32u32, 32u32);
        let pixels = u64::from(width) * u64::from(height);
        let wavefront = Wavefront::new(
            &gpu,
            &Kernels::embedded(),
            4096,
            Wavefront::DEFAULT_MAX_BOUNCES,
            LightSampling::Mis,
        )
        .expect("wavefront");

        let reservoir_bytes = pixels * size_of::<crate::restir::StoredPathReservoir>() as u64;
        let gbuffer_bytes = pixels * 48;
        let store_usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_DST;
        let buffer = |name: &str, bytes: u64| {
            gpu.create_buffer(name, bytes, store_usage, MemoryLocation::GpuOnly)
                .expect("buffer")
        };
        let cand = buffer("test.gbuf.cand", reservoir_bytes);
        let prev = buffer("test.gbuf.prev", reservoir_bytes);
        let curr = buffer("test.gbuf.curr", reservoir_bytes);
        let gbuffer_prev = buffer("test.gbuf.gbuffer.prev", gbuffer_bytes);
        let gbuffer_curr = buffer("test.gbuf.gbuffer.curr", gbuffer_bytes);
        gpu.submit_passes(&[
            Pass::Fill { buffer: &cand, offset: 0, size: reservoir_bytes, value: 0 },
            Pass::Fill { buffer: &prev, offset: 0, size: reservoir_bytes, value: 0 },
            Pass::Fill { buffer: &curr, offset: 0, size: reservoir_bytes, value: 0 },
            Pass::Fill { buffer: &gbuffer_prev, offset: 0, size: gbuffer_bytes, value: 0 },
            Pass::Fill { buffer: &gbuffer_curr, offset: 0, size: gbuffer_bytes, value: 0 },
        ])
        .expect("clear");
        let mut reproject_buf = gpu
            .create_buffer(
                "test.gbuf.reproject",
                96,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                MemoryLocation::CpuToGpu,
            )
            .expect("reproject");
        // Frame 0: no previous camera, so reprojection is off — but the
        // temporal stage still writes this frame's G-buffer entries, which is
        // all the fixture needs, and exactly the shipping write path.
        let reproject = Reproject::new(
            None,
            gbuffer_prev.device_address(),
            gbuffer_curr.device_address(),
            width,
            height,
        );
        reproject_buf.write(bytemuck::bytes_of(&reproject));
        let inputs = RestirInputs {
            reservoir: &curr,
            temporal: Some(TemporalReuse {
                cand: &cand,
                prev: &prev,
                reproject: &reproject_buf,
                decay_frames: 0,
                prev_same_scene: true,
            }),
            scratch: None,
            candidates: Wavefront::RESTIR_CANDIDATES,
            cv_shading: true,
            debug: None,
            debug_view: DebugView::Off,
        };
        let radiance = radiance_buffer(&gpu, width, height);
        wavefront
            .trace_then(&gpu, &scene, &radiance, width, height, 0, None, Some(&inputs), &[], None)
            .expect("trace");

        let spirv = crate::shaders::compile_fixture("restir_gbuffer_test")
            .expect("compile restir_gbuffer_test");
        let pipeline = gpu
            .create_compute_pipeline(
                &spirv,
                c"restir_gbuffer_test",
                size_of::<GbufferTestParams>() as u32,
                Bindings::Scene,
            )
            .expect("pipeline");
        // Zero-uploaded so miss pixels (never reached) read as "not evaluated".
        let mismatch = gpu
            .upload_buffer(
                "test.gbuf.mismatch",
                &vec![0u8; (pixels * 4) as usize],
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::TRANSFER_SRC,
            )
            .expect("mismatch buffer");
        let params = GbufferTestParams {
            paths: wavefront.paths.addresses(),
            hits: wavefront.queues.addresses(queue::HIT, &wavefront.queues.hit),
            scene: scene.table().device_address(),
            gbuffer: gbuffer_curr.device_address(),
            mismatch: mismatch.device_address(),
        };
        gpu.dispatch(
            &pipeline,
            Some(SceneBindings {
                tlas: scene.tlas(),
                environment: scene.environment(),
                textures: scene.texture_descriptors(),
                blue_noise: &wavefront.blue_noise,
                pairing: &wavefront.pairing,
            }),
            bytemuck::bytes_of(&params),
            [(width * height).div_ceil(WORKGROUP_SIZE), 1, 1],
        )
        .expect("dispatch");

        let masks: Vec<u32> =
            bytemuck::pod_collect_to_vec(&gpu.download_buffer(&mismatch).expect("download"));
        let mut evaluated = 0u32;
        for (pixel, mask) in masks.iter().enumerate() {
            if *mask == 0 {
                continue; // a miss pixel — the fixture never ran there
            }
            evaluated += 1;
            assert_eq!(
                *mask, GBUF_EVALUATED,
                "pixel {pixel}: the G-buffer surface diverged from the \
                 path-pool reconstruction (mismatch bits {:#x} — see \
                 restir_gbuffer_test.slang for the field map)",
                *mask & !GBUF_EVALUATED
            );
        }
        // The scene fills most of the frame; a near-empty evaluation would
        // mean the gate went vacuous, not that it passed.
        assert!(
            evaluated > (width * height) / 2,
            "only {evaluated} of {} pixels evaluated — the gate is vacuous",
            width * height
        );
    }

    /// Step 5d's default handoff gate (D-094): the *shipping* estimator — temporal
    /// on with the decay ramp live — must still converge to the path tracer on a
    /// held camera. This is the path a user renders: the first frames after the
    /// camera settled are temporal-warm, then the ramp anneals temporal off by
    /// `RESTIR_TEMPORAL_DECAY_FRAMES`, handing off to spatial-only fresh-RNG
    /// accumulation. Unlike the pinned gate (decay off), this exercises the ramp
    /// *interior* — the partial-decay frames — integrated to convergence, so a
    /// decay that scaled confidence inconsistently across the MIS weights and the
    /// merge, or overshot, would shift the mean off the reference.
    #[test]
    fn temporal_decay_hands_off_to_the_reference() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let (scene, _) = temporal_gate_scene(&gpu);
        let (temporal, path) =
            temporal_gate_means(&gpu, &scene, 256, Wavefront::RESTIR_TEMPORAL_DECAY_FRAMES, None);
        let deviation = (temporal - path).abs() / path;
        assert!(
            deviation < 0.03,
            "decay-on temporal disagrees with the path tracer: {temporal} vs {path} ({deviation:.4} relative)"
        );
    }

    /// Step 5d's exact-zero endpoint: once the decay ramp reaches zero (a held
    /// camera past the window), the temporal stage is a *no-op* — `curr ≡ cand` —
    /// so the converged still is temporal-free (spatial-only, fresh per-frame RNG),
    /// the property that makes it provably the brute-force reference (D-094). This
    /// distinguishes the linear ramp, which hits *exactly* zero, from any
    /// asymptotic decay that would leave the image forever correlated: it builds a
    /// real, valid history (frame 0, then swap), then runs one frame at
    /// `sampleIndex == decayFrames` (decay 0) and checks the temporal output equals
    /// the candidate reservoir — bit-for-bit in confidence, to a hair in W. The
    /// history is asserted non-empty first, so the zeroing is real, not vacuous. At
    /// decay 0 `prev.confidence` is scaled to zero *before* the combine, so the
    /// neighbour drops out whatever it held and whichever pixel reprojection read —
    /// the no-op is unconditional across every hit pixel; `cand`/`curr` are cleared
    /// so miss pixels (written by neither stage) compare equal as zero too.
    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "the two-frame build-then-handoff and its per-pixel no-op check \
                  are one flow — splitting them would scatter what it pins"
    )]
    #[expect(
        clippy::float_cmp,
        reason = "exact equality is the claim: at decay 0 the merge adds 0.0 to the \
                  candidate confidence, so `curr ≡ cand` bit-for-bit, not merely close"
    )]
    fn temporal_decay_is_a_noop_past_its_window() {
        use crate::restir::StoredPathReservoir;
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let (scene, camera) = temporal_gate_scene(&gpu);
        let kernels = Kernels::embedded();
        let (width, height) = (32u32, 32u32);
        let window = Wavefront::RESTIR_TEMPORAL_DECAY_FRAMES;
        let wavefront = Wavefront::new(
            &gpu,
            &kernels,
            4096,
            Wavefront::DEFAULT_MAX_BOUNCES,
            LightSampling::Mis,
        )
        .expect("wavefront");

        let reservoir_bytes =
            u64::from(width) * u64::from(height) * size_of::<StoredPathReservoir>() as u64;
        let gbuffer_bytes = u64::from(width) * u64::from(height) * 48;
        let store_usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_DST;
        let buffer = |name: &str, bytes: u64| {
            gpu.create_buffer(name, bytes, store_usage, MemoryLocation::GpuOnly)
                .expect("buffer")
        };
        let cand = buffer("test.decay.cand", reservoir_bytes);
        let mut prev = buffer("test.decay.prev", reservoir_bytes);
        let mut curr = buffer("test.decay.curr", reservoir_bytes);
        let mut gbuffer_prev = buffer("test.decay.gbuffer.prev", gbuffer_bytes);
        let mut gbuffer_curr = buffer("test.decay.gbuffer.curr", gbuffer_bytes);
        gpu.submit_passes(&[
            Pass::Fill { buffer: &cand, offset: 0, size: reservoir_bytes, value: 0 },
            Pass::Fill { buffer: &prev, offset: 0, size: reservoir_bytes, value: 0 },
            Pass::Fill { buffer: &curr, offset: 0, size: reservoir_bytes, value: 0 },
            Pass::Fill { buffer: &gbuffer_prev, offset: 0, size: gbuffer_bytes, value: 0 },
            Pass::Fill { buffer: &gbuffer_curr, offset: 0, size: gbuffer_bytes, value: 0 },
        ])
        .expect("clear");
        let mut reproject_buf = gpu
            .create_buffer(
                "test.decay.reproject",
                96,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                MemoryLocation::CpuToGpu,
            )
            .expect("reproject");
        let basis = camera.basis(width as f32 / height as f32);
        let cam = ReprojectCamera {
            position: camera.position,
            right: basis.right,
            up: basis.up,
            forward: basis.forward,
        };
        let radiance = radiance_buffer(&gpu, width, height);

        // One temporal frame at `sample` with reprojection validity `valid`.
        let mut frame = |sample: u32, valid: bool, gp: &Buffer, gc: &Buffer, pv: &Buffer, cu: &Buffer| {
            let reproject = Reproject::new(
                valid.then_some(cam),
                gp.device_address(),
                gc.device_address(),
                width,
                height,
            );
            reproject_buf.write(bytemuck::bytes_of(&reproject));
            let inputs = RestirInputs {
                reservoir: cu,
                temporal: Some(TemporalReuse {
                    cand: &cand,
                    prev: pv,
                    reproject: &reproject_buf,
                    decay_frames: window,
                    // No edits land mid-harness: the build never changes.
                    prev_same_scene: true,
                }),
                scratch: None,
                candidates: Wavefront::RESTIR_CANDIDATES,
                cv_shading: true,
                debug: None,
                debug_view: DebugView::Off,
            };
            wavefront
                .trace_then(&gpu, &scene, &radiance, width, height, sample, None, Some(&inputs), &[], None)
                .expect("trace");
        };

        // Frame 0 builds a history into `curr`; the swap makes it next frame's
        // `prev` (and its G-buffer the one reprojection reads).
        frame(0, false, &gbuffer_prev, &gbuffer_curr, &prev, &curr);
        std::mem::swap(&mut prev, &mut curr);
        std::mem::swap(&mut gbuffer_prev, &mut gbuffer_curr);

        let history: Vec<StoredPathReservoir> =
            bytemuck::pod_collect_to_vec(&gpu.download_buffer(&prev).expect("download prev"));
        let history_max = history.iter().map(|r| r.confidence).fold(0.0_f32, f32::max);
        assert!(history_max > 0.0, "frame 0 should leave a history for the ramp to decay");

        // Clear `cand`/`curr` so the miss pixels neither stage touches this frame
        // compare equal as zero; `prev` (the history) is kept.
        gpu.submit_passes(&[
            Pass::Fill { buffer: &cand, offset: 0, size: reservoir_bytes, value: 0 },
            Pass::Fill { buffer: &curr, offset: 0, size: reservoir_bytes, value: 0 },
        ])
        .expect("clear");

        // The handoff frame: sampleIndex == window ⇒ decay = saturate(1 − 1) = 0.
        frame(window, true, &gbuffer_prev, &gbuffer_curr, &prev, &curr);

        let cand_r: Vec<StoredPathReservoir> =
            bytemuck::pod_collect_to_vec(&gpu.download_buffer(&cand).expect("download cand"));
        let curr_r: Vec<StoredPathReservoir> =
            bytemuck::pod_collect_to_vec(&gpu.download_buffer(&curr).expect("download curr"));
        for (i, (c, u)) in cand_r.iter().zip(&curr_r).enumerate() {
            assert_eq!(
                u.confidence, c.confidence,
                "pixel {i}: decay-0 temporal changed confidence ({} vs {}) — history leaked in",
                u.confidence, c.confidence
            );
            let dw = (u.unbiased_weight - c.unbiased_weight).abs();
            assert!(
                dw <= 1e-4 * (1.0 + c.unbiased_weight.abs()),
                "pixel {i}: decay-0 temporal changed W: {} vs {}",
                u.unbiased_weight,
                c.unbiased_weight
            );
            // The CV lane obeys the same endpoint (6b-ii): a zeroed decay
            // folds history out before the blend, so the candidate mean
            // passes through bit-for-bit and converged-still frames stay
            // independent on the shading estimate too (D-145).
            assert_eq!(
                u.cv_accumulator, c.cv_accumulator,
                "pixel {i}: decay-0 temporal changed the CV lane — history \
                 leaked into the blend past the ramp"
            );
        }
    }

    /// The delta-light half of D-088: delta lights stay *outside* the reservoir,
    /// added by `restir_resolve` as an exact additive NEE term. A white Lambert
    /// plane under a black sky, lit by one distant light straight down with
    /// irradiance π, has the closed form (albedo/π)·π = 1 at every pixel — the
    /// same ground truth `a_distant_light_is_analytically_exact` pins for the
    /// path tracer, now driven through `ReSTIR`. Two things must hold at once:
    /// the primary hit sees the delta (the reservoir owns no delta candidate, and
    /// `shade_surface` skips its own NEE there), and the delta term lands even
    /// though the reservoir comes back *empty* — a black sky and no area lights
    /// leave it nothing to select, so this exercises the path that adds the delta
    /// before the empty-reservoir return. Drop the delta term, or gate it behind
    /// that return, and the plane goes black. Built through description → prep,
    /// the only route delta lights exist on.
    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "one flat change-set literal is the whole scene — splitting it \
                  would hide its shape"
    )]
    fn restir_adds_the_delta_lights_the_reservoir_excludes() {
        use crate::scene::changeset::{
            CameraPatch, ChangeSet, InstancePatch, LightPatch, MaterialPatch, MeshPatch, Op,
            SettingsPatch,
        };
        use crate::scene::description::{Light, MeshSource, SceneDescription, Texturable};

        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let mut description = SceneDescription::new();
        description
            .apply(&ChangeSet {
                ops: vec![
                    Op::Settings(SettingsPatch::new("main")),
                    Op::Camera(CameraPatch {
                        position: Some([0.0, 1.0, 0.0]),
                        look_at: Some([0.0, 0.0, -1.0]),
                        ..CameraPatch::new("main")
                    }),
                    Op::Mesh(MeshPatch {
                        source: Some(MeshSource::Inline {
                            positions: vec![
                                [-5.0, 0.0, -5.0],
                                [-5.0, 0.0, 5.0],
                                [5.0, 0.0, 5.0],
                                [5.0, 0.0, -5.0],
                            ],
                            normals: Some(vec![[0.0, 1.0, 0.0]; 4]),
                            uvs: None,
                            triangles: vec![[0, 1, 2], [0, 2, 3]],
                        }),
                        ..MeshPatch::new("plane")
                    }),
                    Op::Material(Box::new(MaterialPatch {
                        base_color: Some(Texturable::Constant([1.0; 3])),
                        specular_weight: Some(0.0),
                        ..MaterialPatch::new("lambert")
                    })),
                    Op::Instance(InstancePatch {
                        mesh: Some("plane".into()),
                        material: Some("lambert".into()),
                        ..InstancePatch::new("floor")
                    }),
                    Op::Light(LightPatch {
                        light: Some(Light::Distant {
                            direction: [0.0, -1.0, 0.0],
                            irradiance: [std::f32::consts::PI; 3],
                        }),
                        ..LightPatch::new("sun")
                    }),
                ],
            })
            .expect("valid scene data");
        let scene = Scene::prep(&gpu, &mut description).expect("prep");

        let kernels = Kernels::embedded();
        let (width, height) = (16, 16);
        // The delta term is variance-free here — one light, cosθ = 1, open sky —
        // so a handful of samples already sits on the closed form.
        let samples: u32 = 16;
        let reservoir = gpu
            .create_buffer(
                "test.reservoir",
                u64::from(width)
                    * u64::from(height)
                    * size_of::<crate::restir::StoredPathReservoir>() as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                MemoryLocation::GpuOnly,
            )
            .expect("reservoir buffer");
        let wavefront = Wavefront::new(
            &gpu,
            &kernels,
            4096,
            Wavefront::DEFAULT_MAX_BOUNCES,
            LightSampling::Mis,
        )
        .expect("wavefront");
        let inputs = RestirInputs {
            reservoir: &reservoir,
            temporal: None,
            scratch: None,
            candidates: Wavefront::RESTIR_CANDIDATES,
            cv_shading: true,
            debug: None,
            debug_view: DebugView::Off,
        };
        let radiance = radiance_buffer(&gpu, width, height);
        let mut total = 0.0f64;
        for sample in 0..samples {
            wavefront
                .trace_then(
                    &gpu,
                    &scene,
                    &radiance,
                    width,
                    height,
                    sample,
                    None,
                    Some(&inputs),
                    &[],
                    None,
                )
                .expect("trace");
            let pixels: Vec<f32> = bytemuck::pod_collect_to_vec(
                &gpu.download_buffer(&radiance).expect("download"),
            );
            total += pixels
                .chunks_exact(4)
                .map(|pixel| f64::from(pixel[0]) + f64::from(pixel[1]) + f64::from(pixel[2]))
                .sum::<f64>();
        }
        let mean = total / f64::from(samples * width * height * 3);
        assert!(
            (mean - 1.0).abs() < 5e-3,
            "ReSTIR delta light off the closed form: {mean} vs 1"
        );
    }

    /// The `ReSTIR` white furnace (M3 plan §6, §7): an albedo-1 Lambert plane
    /// under a uniform sky must reflect exactly that sky — energy neutral. It is
    /// the cheap, always-on bias tripwire that fails *fast* on the silent-bias
    /// class step 3.4 opens up. The internalized BSDF draw's candidates and the
    /// M light candidates must combine, under the count-weighted balance
    /// heuristic, into one unbiased estimate of the reservoir's whole term —
    /// count either side twice (or drop one) and the furnace leaks light or
    /// darkens, long before a 4096-spp FLIP run would notice. No area or delta
    /// lights, so envSelectProb is 1 and every candidate that doesn't escape is
    /// an env draw: the env BSDF candidate is squarely on the hot path.
    #[test]
    fn restir_white_furnace_stays_energy_neutral() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let sky = 0.5;
        // One big white Lambert plane, camera just above looking obliquely down
        // (the basis forbids straight down) — render::tests' furnace framing,
        // rebuilt here to drive the reservoir directly.
        let objects = [Object {
            mesh: ground_plane(5.0),
            transform: Mat4::IDENTITY,
            material: Material::matte(Vec3::ONE, 0.0),
        }];
        let camera = Camera {
            position: Vec3::new(0.0, 1.0, 0.0),
            look_at: Vec3::new(0.0, 0.0, -1.0),
            up: Vec3::Y,
            vfov_degrees: 40.0,
            lens: None,
        };
        let scene = Scene::new(&gpu, &objects, camera, &Environment::constant(Vec3::splat(sky)))
            .expect("scene");

        let kernels = Kernels::embedded();
        let (width, height) = (32, 32);
        // The mean over this many samples brings the estimator's swing well
        // under the 2% bound, so the bound answers to bias.
        let samples: u32 = 256;
        let reservoir = gpu
            .create_buffer(
                "test.reservoir.furnace",
                u64::from(width)
                    * u64::from(height)
                    * size_of::<crate::restir::StoredPathReservoir>() as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                MemoryLocation::GpuOnly,
            )
            .expect("reservoir buffer");
        let wavefront = Wavefront::new(
            &gpu,
            &kernels,
            4096,
            Wavefront::DEFAULT_MAX_BOUNCES,
            LightSampling::Mis,
        )
        .expect("wavefront");
        let inputs = RestirInputs {
            reservoir: &reservoir,
            temporal: None,
            scratch: None,
            candidates: Wavefront::RESTIR_CANDIDATES,
            cv_shading: true,
            debug: None,
            debug_view: DebugView::Off,
        };

        let radiance = radiance_buffer(&gpu, width, height);
        let mut total = 0.0_f64;
        for sample in 0..samples {
            wavefront
                .trace_then(
                    &gpu,
                    &scene,
                    &radiance,
                    width,
                    height,
                    sample,
                    None,
                    Some(&inputs),
                    &[],
                    None,
                )
                .expect("trace");
            let pixels: Vec<f32> =
                bytemuck::pod_collect_to_vec(&gpu.download_buffer(&radiance).expect("download"));
            total += pixels
                .chunks_exact(4)
                .map(|pixel| f64::from(pixel[0]))
                .sum::<f64>();
        }
        let mean = total / f64::from(samples * width * height);
        assert!(
            (mean - f64::from(sky)).abs() / f64::from(sky) < 0.02,
            "the ReSTIR furnace leaked: mean {mean} vs sky {sky}"
        );
    }

    /// The D-092 debug surface: with a [`DebugView`] selected, `restir_resolve`
    /// false-colours the survivor into the debug buffer. This is the step-3
    /// checkpoint — "you can false-colour the selected light". A lit surface
    /// pixel (which resampled a light) comes out non-black and opaque; a sky
    /// pixel, which the wave zero-fills and the resolve stage never reaches,
    /// stays black — so the view genuinely marks *where* a light was chosen,
    /// not a flat wash.
    #[test]
    fn restir_debug_surface_paints_the_selected_light() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let objects = [
            Object {
                mesh: icosphere(2),
                transform: Mat4::from_translation(Vec3::Y),
                material: Material::glossy(Vec3::splat(0.6), 0.4, 0.3),
            },
            Object {
                mesh: ground_plane(4.0),
                transform: Mat4::IDENTITY,
                material: Material::glossy(Vec3::splat(0.7), 0.0, 0.2),
            },
            Object {
                // An area emitter above the sphere — the triangle candidates.
                mesh: icosphere(2),
                transform: Mat4::from_translation(Vec3::Y * 3.0)
                    * Mat4::from_scale(Vec3::splat(0.7)),
                material: Material::emitter(Vec3::splat(4.0)),
            },
        ];
        // The camera looks slightly down from above, so the frame's upper rows
        // clear the ground plane and see the (lit) environment: guaranteed sky.
        let camera = Camera {
            position: Vec3::new(0.0, 2.5, 6.0),
            look_at: Vec3::new(0.0, 1.0, 0.0),
            up: Vec3::Y,
            vfov_degrees: 45.0,
            lens: None,
        };
        let scene = Scene::new(&gpu, &objects, camera, &Environment::constant(Vec3::splat(0.3)))
            .expect("scene");

        let (width, height) = (32, 32);
        let texels = u64::from(width) * u64::from(height);
        let storage =
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        let reservoir = gpu
            .create_buffer(
                "test.debug.reservoir",
                texels * size_of::<crate::restir::StoredPathReservoir>() as u64,
                storage,
                MemoryLocation::GpuOnly,
            )
            .expect("reservoir buffer");
        let debug = gpu
            .create_buffer(
                "test.debug.surface",
                texels * 16,
                storage | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuOnly,
            )
            .expect("debug buffer");
        let radiance = radiance_buffer(&gpu, width, height);
        let wavefront = Wavefront::new(
            &gpu,
            &Kernels::embedded(),
            4096,
            Wavefront::DEFAULT_MAX_BOUNCES,
            LightSampling::Mis,
        )
        .expect("wavefront");

        let inputs = RestirInputs {
            reservoir: &reservoir,
            temporal: None,
            scratch: None,
            candidates: Wavefront::RESTIR_CANDIDATES,
            cv_shading: true,
            debug: Some(&debug),
            debug_view: DebugView::SelectedLight,
        };
        wavefront
            .trace_then(
                &gpu, &scene, &radiance, width, height, 0, None, Some(&inputs), &[], None,
            )
            .expect("trace");
        let pixels: Vec<f32> =
            bytemuck::pod_collect_to_vec(&gpu.download_buffer(&debug).expect("download"));

        let lit = pixels
            .chunks_exact(4)
            .filter(|texel| texel[0] + texel[1] + texel[2] > 0.0)
            .count();
        let black = pixels
            .chunks_exact(4)
            .filter(|texel| texel[..3].iter().all(|c| *c == 0.0))
            .count();
        assert!(lit > 0, "no pixel was false-coloured — the surface is blank");
        assert!(
            black > 0,
            "every pixel was painted — sky pixels should stay at the zero-fill"
        );
        // A false-coloured pixel is opaque (alpha 1); the zero-fill leaves
        // unpainted pixels at alpha 0, so a 0.5 split cleanly tells them apart.
        assert!(
            pixels
                .chunks_exact(4)
                .filter(|texel| texel[0] + texel[1] + texel[2] > 0.0)
                .all(|texel| texel[3] > 0.5),
            "a painted debug pixel is not opaque"
        );
    }

    /// The M3 §2/§6 determinism invariant, carried through the reservoir. The
    /// plain [`rendering_is_bitwise_deterministic`] gate drives `trace`, which
    /// bypasses `ReSTIR` entirely, so it never touches the writes the reservoir
    /// path adds: `restir_candidates` streaming the WRS reservoir into its
    /// pixel-owned slot, `restir_resolve` folding the survivor's shade into the
    /// film, and the survivor's `trace_shadow` visibility ray. Every one of
    /// those is pixel-owned, never atomic — the same rule the whole wavefront
    /// keeps — so the queue push order, which varies run to run, can never
    /// reach the image. This re-runs the four-stage chain twice, each from a
    /// fresh reservoir, and demands the accumulated films agree bit for bit: a
    /// stray shared write or an atomic accumulation slipped into any reservoir
    /// stage would surface here as flickering low bits, exactly as it does for
    /// the path tracer. The scene lights through both candidate branches (a lit
    /// environment *and* an area emitter), so the reservoir is genuinely
    /// populated by weighted resampling and the survivor's shadow ray fires —
    /// the stochastic machinery whose determinism the invariant is about.
    #[test]
    fn restir_is_bitwise_deterministic() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let objects = [
            Object {
                mesh: icosphere(2),
                transform: Mat4::from_translation(Vec3::Y),
                material: Material::glossy(Vec3::splat(0.6), 0.4, 0.3).with_metalness(0.5),
            },
            Object {
                mesh: ground_plane(4.0),
                transform: Mat4::IDENTITY,
                material: Material::glossy(Vec3::splat(0.7), 0.0, 0.2),
            },
            Object {
                // An area emitter above the sphere feeds the triangle candidates.
                mesh: icosphere(2),
                transform: Mat4::from_translation(Vec3::Y * 3.0)
                    * Mat4::from_scale(Vec3::splat(0.7)),
                material: Material::emitter(Vec3::splat(4.0)),
            },
        ];
        let camera = Camera {
            position: Vec3::new(0.0, 2.5, 6.0),
            look_at: Vec3::new(0.0, 1.0, 0.0),
            up: Vec3::Y,
            vfov_degrees: 45.0,
            lens: None,
        };
        // A lit environment as well, so the env coin flip fires alongside the
        // triangle table and every reservoir write is on the hot path.
        let scene = Scene::new(&gpu, &objects, camera, &Environment::constant(Vec3::splat(0.3)))
            .expect("scene");

        let kernels = Kernels::embedded();
        let (width, height) = (32, 32);
        let wavefront = Wavefront::new(
            &gpu,
            &kernels,
            4096,
            Wavefront::DEFAULT_MAX_BOUNCES,
            LightSampling::Mis,
        )
        .expect("wavefront");

        // One full run of the ReSTIR chain from a fresh reservoir, concatenating
        // every sample's raw film bytes — so a difference in any single sample
        // is caught exactly, not just in the sum where two could cancel. Run for
        // both estimators: spatial reuse reads *neighbour* reservoirs, so its
        // determinism leans harder on the committed-prior-pass ping-pong (a
        // barrier between candidates and spatial, and a separate scratch buffer)
        // — a stray read of a half-written neighbour would surface here.
        let reservoir_usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_DST;
        let bytes = u64::from(width)
            * u64::from(height)
            * size_of::<crate::restir::StoredPathReservoir>() as u64;
        let run = |spatial: bool| -> Vec<u8> {
            let reservoir = gpu
                .create_buffer("test.determinism.reservoir", bytes, reservoir_usage, MemoryLocation::GpuOnly)
                .expect("reservoir buffer");
            let scratch = gpu
                .create_buffer("test.determinism.scratch", bytes, reservoir_usage, MemoryLocation::GpuOnly)
                .expect("scratch buffer");
            let inputs = RestirInputs {
                reservoir: &reservoir,
                temporal: None,
                scratch: spatial.then_some(&scratch),
                candidates: Wavefront::RESTIR_CANDIDATES,
                cv_shading: true,
                debug: None,
                debug_view: DebugView::Off,
            };
            let radiance = radiance_buffer(&gpu, width, height);
            let mut film = Vec::new();
            for sample in 0..8 {
                wavefront
                    .trace_then(
                        &gpu, &scene, &radiance, width, height, sample, None, Some(&inputs), &[], None,
                    )
                    .expect("trace");
                film.extend_from_slice(&gpu.download_buffer(&radiance).expect("download"));
            }
            film
        };

        assert_eq!(run(false), run(false), "the ReSTIR RIS path is not bitwise deterministic");
        assert_eq!(run(true), run(true), "the ReSTIR spatial path is not bitwise deterministic");
    }
}
