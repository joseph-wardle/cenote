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
use crate::gpu::{Bindings, Buffer, ComputePipeline, Context, MemoryLocation, Pass, SceneBindings};
use crate::scene::{Scene, ray_mask};
use crate::shaders::{Kernel, Kernels};

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
/// [`Wavefront::new`].
fn pack_shade_surface(bounce: u32, max_bounces: u32, light_sampling: LightSampling) -> u32 {
    bounce | max_bounces << 8 | (light_sampling as u32) << 16
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

/// The engine: five stage pipelines over one path pool and its queues.
/// Created once and reused across waves — nothing in it depends on the
/// target size or the scene.
pub struct Wavefront {
    raygen: ComputePipeline,
    intersect: ComputePipeline,
    shade_miss: ComputePipeline,
    shade_surface: ComputePipeline,
    trace_shadow: ComputePipeline,
    paths: PathPool,
    queues: Queues,
    /// The all-zero [`AovTableData`] a wave binds when the caller brings
    /// no AOV targets — `enabled` 0, so the kernels skip every guide read
    /// and write.
    aov_disabled: Buffer,
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
            paths: PathPool::new(gpu, capacity)?,
            queues: Queues::new(gpu, capacity)?,
            aov_disabled: gpu.upload_buffer(
                "wavefront.aov.disabled",
                bytemuck::bytes_of(&AovTableData::zeroed()),
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
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
        self.trace_then(gpu, scene, radiance, width, height, sample, None, &[])
    }

    /// [`Wavefront::trace`], then `trailing` — extra passes appended to the
    /// same submission, sharing its single fence. The full memory barrier
    /// [`Context::submit_passes`] places between passes flushes the wave's
    /// radiance writes before the first trailing pass reads them, so folding
    /// the film's accumulate in here spends one GPU round-trip per sample
    /// instead of two — bit-for-bit as if they ran as separate submissions,
    /// since a barrier orders the same writes a fence does.
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
        trailing: &[Pass],
    ) -> Result<()> {
        assert!(width > 0 && height > 0, "zero-sized trace target");
        let pixels = u64::from(width) * u64::from(height);
        assert!(
            radiance.size() >= pixels * 16,
            "radiance buffer smaller than the target"
        );
        let aov_table = aovs.map_or(&self.aov_disabled, |aov| aov.table);
        let params = self.wave_params(scene, radiance, aov_table, width, height, sample);
        let mut passes = self.record_wave(scene, radiance, aovs, pixels, &params);
        passes.extend_from_slice(trailing);
        gpu.submit_passes(&passes)
    }

    /// Every stage's push constants for one wave, built up front so the
    /// recorded passes can borrow them.
    fn wave_params(
        &self,
        scene: &Scene,
        radiance: &Buffer,
        aov_table: &Buffer,
        width: u32,
        height: u32,
        sample: u32,
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
        WaveParams {
            ranges,
            intersect: (0..self.max_bounces).map(intersect).collect(),
            shade_miss: ShadeMissParams {
                paths: self.paths.addresses(),
                misses: self.queues.addresses(queue::MISS, &self.queues.miss),
                scene: scene.table().device_address(),
                radiance: radiance.device_address(),
                aov: aov_table.device_address(),
                light_sampling: self.light_sampling as u32,
                _pad0: 0,
            },
            shade_surface: (0..self.max_bounces)
                .map(|bounce| ShadeSurfaceParams {
                    paths: self.paths.addresses(),
                    hits: self.queues.addresses(queue::HIT, &self.queues.hit),
                    rays: self.queues.addresses(queue::RAY, &self.queues.ray),
                    shadows: self.queues.addresses(queue::SHADOW, &self.queues.shadow),
                    scene: scene.table().device_address(),
                    radiance: radiance.device_address(),
                    aov: aov_table.device_address(),
                    sample_index: sample,
                    packed: pack_shade_surface(bounce, self.max_bounces, self.light_sampling),
                })
                .collect(),
            trace_shadow: TraceShadowParams {
                shadows: self.queues.addresses(queue::SHADOW, &self.queues.shadow),
                radiance: radiance.device_address(),
                scene: scene.table().device_address(),
            },
        }
    }

    /// Record one wave's pass sequence: zero the radiance target (and the
    /// AOV accumulators, when the wave carries them), then per pixel
    /// range, raygen and the bounce loop.
    fn record_wave<'a>(
        &'a self,
        scene: &'a Scene,
        radiance: &'a Buffer,
        aovs: Option<&AovTargets<'a>>,
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
            // after every path has died dispatch nothing.
            for bounce in 0..self.max_bounces {
                passes.push(fill(queue::HIT));
                passes.push(fill(queue::MISS));
                passes.push(fill(queue::SHADOW));
                passes.push(indirect(
                    &self.intersect,
                    bytemuck::bytes_of(&params.intersect[bounce as usize]),
                    queue::RAY,
                ));
                // The ray queue was just consumed; empty it for this
                // round's shade_surface — except on the last bounce, where
                // the kernel terminates every path instead of pushing.
                if bounce + 1 < self.max_bounces {
                    passes.push(fill(queue::RAY));
                }
                passes.push(indirect(
                    &self.shade_miss,
                    bytemuck::bytes_of(&params.shade_miss),
                    queue::MISS,
                ));
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
    /// One instance per bounce: bounce 0 traces with the camera visibility
    /// bit, later bounces with all bits, and each keys its own
    /// transparency stream.
    intersect: Vec<IntersectParams>,
    shade_miss: ShadeMissParams,
    /// One instance per bounce.
    shade_surface: Vec<ShadeSurfaceParams>,
    trace_shadow: TraceShadowParams,
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

    /// The step-6 checkpoint, kept honest since — progressive refinement
    /// is real. A camera ray that misses adds the environment radiance to
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

    /// The step-8 checkpoint, and the test that catches wrong-but-plausible
    /// MIS: next-event-only, BSDF-only, and MIS renders of one scene must
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

    /// The same agreement, with the *environment* as the only light — the
    /// step-10 counterpart. The synthetic sky is the CDF tables' worst
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

    /// The step-6 estimator-consistency test: agreement must survive
    /// textures in both places they touch the light transport. The
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
                        transform: Some(plane([4.0, 1.0, 4.0], [0.0; 3])),
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
                        transform: Some(plane([1.0; 3], [0.0, 1.0, 0.0])),
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
                        })),
                        ..MaterialPatch::new("lamp")
                    })),
                    Op::Instance(InstancePatch {
                        mesh: Some("plane".into()),
                        material: Some("lamp".into()),
                        transform: Some(plane([0.7; 3], [0.0, 3.0, 0.0])),
                        ..InstancePatch::new("lamp")
                    }),
                    // ...and the perforated card between it and the floor.
                    Op::Material(Box::new(MaterialPatch {
                        base_color: Some(Texturable::Constant([0.5; 3])),
                        specular_weight: Some(0.0),
                        geometry_opacity: Some(Texturable::Texture(TextureRef {
                            path: holes,
                            color_space: None,
                        })),
                        ..MaterialPatch::new("card")
                    })),
                    Op::Instance(InstancePatch {
                        mesh: Some("plane".into()),
                        material: Some("card".into()),
                        transform: Some(plane([1.2, 1.0, 1.2], [0.0, 2.0, 0.0])),
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
}
