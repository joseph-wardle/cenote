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
use crate::scene::Scene;
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
const SHADOW_RAY_SIZE: u64 = 48;

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
/// order is layout.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RaygenParams {
    paths: PathsAddrs,
    rays: QueueAddrs,
    /// Which sample of every pixel's sequence this wave traces.
    sample_index: u32,
    /// Aligns the camera block to the 16-byte stride std430 gives its
    /// `float3`s (`Pod` forbids implicit padding bytes).
    _pad0: u32,
    camera_position: Vec3,
    width: u32,
    camera_right: Vec3,
    height: u32,
    camera_up: Vec3,
    /// First pixel of this range.
    base: u32,
    camera_forward: Vec3,
    /// Paths in this range.
    count: u32,
}

/// Push constants for the intersect kernel (`shaders/intersect.slang`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct IntersectParams {
    paths: PathsAddrs,
    rays: QueueAddrs,
    hits: QueueAddrs,
    misses: QueueAddrs,
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
    /// Which strategies reach the lights — a [`LightSampling`] as `u32`.
    light_sampling: u32,
    _pad0: u32,
}

/// Push constants for the surface-shading kernel
/// (`shaders/shade_surface.slang`). One instance per bounce — `bounce` is
/// the only field that varies.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShadeSurfaceParams {
    paths: PathsAddrs,
    hits: QueueAddrs,
    /// The next bounce's input: scattered paths push themselves back here.
    rays: QueueAddrs,
    /// Next-event connections, consumed by this round's `trace_shadow`.
    shadows: QueueAddrs,
    /// Device address of the scene table (geometry, materials, lights).
    scene: vk::DeviceAddress,
    radiance: vk::DeviceAddress,
    /// Which sample of every pixel's sequence this wave traces.
    sample_index: u32,
    /// Which bounce of the wave this dispatch shades.
    bounce: u32,
    /// Path-length cap: the dispatch at `max_bounces - 1` never continues.
    max_bounces: u32,
    /// Which strategies reach the lights — a [`LightSampling`] as `u32`.
    light_sampling: u32,
}

/// Push constants for the shadow-ray kernel (`shaders/trace_shadow.slang`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TraceShadowParams {
    shadows: QueueAddrs,
    radiance: vk::DeviceAddress,
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
        })
    }

    fn addresses(&self) -> PathsAddrs {
        PathsAddrs {
            origin: self.origin.device_address(),
            direction: self.direction.device_address(),
            pixel: self.pixel.device_address(),
            hit: self.hit.device_address(),
            throughput: self.throughput.device_address(),
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

/// Which sampling strategies reach the lights. [`LightSampling::Mis`] is
/// the renderer; the single-strategy modes exist because the strongest
/// test of the MIS weights is that either strategy alone converges to the
/// same image (the MIS-agreement test below). Values match the
/// `LIGHT_SAMPLING_*` constants in `shaders/lights.slang`.
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
    /// On zero capacity or zero bounces — programmer bugs.
    pub fn new(
        gpu: &Context,
        kernels: &Kernels,
        capacity: u32,
        max_bounces: u32,
        light_sampling: LightSampling,
    ) -> Result<Self> {
        assert!(capacity > 0, "zero-capacity path pool");
        assert!(max_bounces > 0, "zero-bounce wavefront");
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
        assert!(width > 0 && height > 0, "zero-sized trace target");
        let pixels = u64::from(width) * u64::from(height);
        assert!(
            radiance.size() >= pixels * 16,
            "radiance buffer smaller than the target"
        );
        let params = self.wave_params(scene, radiance, width, height, sample);
        gpu.submit_passes(&self.record_wave(scene, radiance, pixels, &params))
    }

    /// Every stage's push constants for one wave, built up front so the
    /// recorded passes can borrow them.
    fn wave_params(
        &self,
        scene: &Scene,
        radiance: &Buffer,
        width: u32,
        height: u32,
        sample: u32,
    ) -> WaveParams {
        let pixels = u64::from(width) * u64::from(height);
        let basis = scene.camera().basis(width as f32 / height as f32);
        let ranges = (0..pixels)
            .step_by(self.capacity as usize)
            .map(|base| RaygenParams {
                paths: self.paths.addresses(),
                rays: self.queues.addresses(queue::RAY, &self.queues.ray),
                sample_index: sample,
                _pad0: 0,
                camera_position: scene.camera().position,
                width,
                camera_right: basis.right,
                height,
                camera_up: basis.up,
                base: base as u32,
                camera_forward: basis.forward,
                count: (pixels - base).min(u64::from(self.capacity)) as u32,
            })
            .collect();
        WaveParams {
            ranges,
            intersect: IntersectParams {
                paths: self.paths.addresses(),
                rays: self.queues.addresses(queue::RAY, &self.queues.ray),
                hits: self.queues.addresses(queue::HIT, &self.queues.hit),
                misses: self.queues.addresses(queue::MISS, &self.queues.miss),
            },
            shade_miss: ShadeMissParams {
                paths: self.paths.addresses(),
                misses: self.queues.addresses(queue::MISS, &self.queues.miss),
                scene: scene.table().device_address(),
                radiance: radiance.device_address(),
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
                    sample_index: sample,
                    bounce,
                    max_bounces: self.max_bounces,
                    light_sampling: self.light_sampling as u32,
                })
                .collect(),
            trace_shadow: TraceShadowParams {
                shadows: self.queues.addresses(queue::SHADOW, &self.queues.shadow),
                radiance: radiance.device_address(),
            },
        }
    }

    /// Record one wave's pass sequence: zero the radiance target, then per
    /// pixel range, raygen and the bounce loop.
    fn record_wave<'a>(
        &'a self,
        scene: &'a Scene,
        radiance: &'a Buffer,
        pixels: u64,
        params: &'a WaveParams,
    ) -> Vec<Pass<'a>> {
        // Every post-raygen stage touches a scene resource — the TLAS, the
        // environment texture, or both — and they share one descriptor
        // layout, so each binds the same pair.
        let bindings = SceneBindings {
            tlas: scene.tlas(),
            environment: scene.environment(),
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
                    bytemuck::bytes_of(&params.intersect),
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
    intersect: IntersectParams,
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
            vfov_degrees: 40.0,
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

    /// The step-8 checkpoint, and the test that catches wrong-but-plausible
    /// MIS: next-event-only, BSDF-only, and MIS renders of one scene must
    /// converge to the same mean. A pdf mismatch or a weight pair that
    /// doesn't sum to 1 biases the strategies apart (double-counting shows
    /// up as 2×); goldens can't see this — they'd normalize the bias into
    /// the reference. The sky is black, so every photon comes from the
    /// quad, and the sphere really occludes it — broken shadow-ray
    /// visibility shifts the next-event modes but not BSDF-only.
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
                // A 2 m × 2 m quad right above the sphere: big enough that
                // BSDF sampling finds it often (variance stays testable),
                // low enough that its shadow occludes real floor.
                mesh: ground_plane(1.0),
                transform: Mat4::from_translation(Vec3::Y * 3.0),
                material: Material::emitter(Vec3::splat(5.0)),
            },
        ];
        let camera = Camera {
            position: Vec3::new(0.0, 2.5, 6.0),
            look_at: Vec3::new(0.0, 1.0, 0.0),
            vfov_degrees: 45.0,
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
            vfov_degrees: 45.0,
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

    /// Render `scene` under all three light-sampling modes and require the
    /// means to agree within 3% — the shared teeth of the MIS-agreement
    /// tests above.
    fn assert_strategies_agree(gpu: &Context, scene: &Scene) {
        let kernels = Kernels::embedded();
        let (width, height) = (32, 32);
        let samples = 64;
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
