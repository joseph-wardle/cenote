//! The `ReSTIR` substrate: reservoir buffers, their per-view ownership, and the
//! stable light-identity remap the reservoirs store samples through.
//!
//! This module is the host half of the reservoir path. Its GPU half is
//! `shaders/reservoir.slang` (the index-agnostic `Reservoir<Sample>` primitive)
//! and the `restir_*.slang` stages that stream candidates through it. Nothing
//! here computes lighting; it owns the memory reuse runs on and keeps that
//! memory correct across the two events that make `ReSTIR` hard to retrofit —
//! a light being added or removed (the identity remap) and a viewport being
//! resized or replaced (per-view ownership). Both land now, in step 2, because
//! both are high-retrofit-cost seams (M3 plan §6, D-085).
//!
//! The reservoir buffer is deliberately `AoS` where the path pool is `SoA`:
//! every reuse stage touches a whole reservoir at once, so packing its fields
//! together is the cache-friendly layout here (see `PathSample` and the
//! packed-reservoir mirror below). The persistent per-pixel record is
//! {sample, `unbiasedWeight`, confidence} — `weightSum` is pass-local and never
//! stored, the one invariant `reservoir.slang` documents at length.

use ash::vk;
use bytemuck::{Pod, Zeroable};

use crate::error::Result;
use crate::gpu::{Buffer, Context};
use crate::lights::LightRecord;

pub(crate) use identity::{EmissiveLight, LightIdentityRegistry};
use identity::{LIGHT_ID_NONE, LightRemap};

/// The path reservoir's reconnection vertex — the host mirror of the shader
/// `Hit` (`pathstate.slang`) in its reservoir role: instance + primitive +
/// barycentrics, the re-evaluable form the shift re-shades from (D-131). Reused
/// verbatim (D-128) rather than a bespoke packed vertex, so the reconnection
/// point has one shared shape across primary shading and reuse.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct ReconnectionVertex {
    /// TLAS instance custom index.
    pub instance: u32,
    /// Triangle index within the instance's mesh.
    pub primitive: u32,
    /// The reconnection point on that triangle.
    pub barycentrics: [f32; 2],
}

/// The full-path reservoir's concrete sample — the host mirror of `PathSample`
/// in `shaders/reservoir_path.slang`. A whole light path held in the
/// re-evaluable reconnection form (reconnection vertex + the seeds a
/// random-replay shift replays from), never a serialized path: constant
/// per-pixel storage, the shift re-resolves the closure on demand (D-131).
/// Laid out in std430 16-byte lanes so this mirror packs identically to the
/// shader struct — the round-trip fixture pins the offsets agree.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct PathSample {
    /// xₖ — the reconnection vertex.
    pub rc_vertex: ReconnectionVertex,
    /// Lₖ — the incident-side suffix cached beyond the reconnection vertex
    /// (fₖ·|cosθₖ| excluded), so the shift re-shades the contribution without
    /// re-tracing the tail (D-137).
    pub rc_vertex_radiance: [f32; 3],
    /// ωₖ — oct-encoded outgoing direction at the reconnection vertex.
    pub rc_vertex_wi: u32,
    /// F — the RGB path integrand in this pixel's domain; p̂ = luminance(F).
    /// Live from step 3b on, for every indirect kind (D-138/D-139):
    /// generation stores the own-domain value, a winning spatial shift
    /// rewrites it with the destination's, and it is trusted wherever
    /// re-forming would re-trace a replay (own-domain targets; resolve for
    /// k > 2 and replay-kind samples).
    pub f: [f32; 3],
    /// The NEE light pdf, kept unpacked — it feeds the path MIS at resolve.
    pub nee_light_pdf: f32,
    /// The per-path replay seed (live from step 3b, D-138): the walk's
    /// sampling draws are a pure function of (seed, dimension), so a shift
    /// replays a k > 2 sample's prefix bounces — or a replay-kind sample's
    /// whole path, terminal redraw included (D-139) — with exactly the
    /// source's draws. Doubles as the deferred duplication-map hash key.
    pub init_random_seed: u32,
    /// Reserved (D-128 headroom): designed for a per-segment reseed, but the
    /// per-path seed above covers the terminal redraw too — dimensions are
    /// per-bounce — so it stays zero (D-139).
    pub rc_vertex_random_seed: u32,
    /// The source half of the shift Jacobian, |cosθₖ|/d² where the sample was
    /// drawn (live from step 3b, D-138): the shift divides its own half by
    /// it, so a same-pixel shift is 1 to the last-place bit.
    pub cached_jacobian: f32,
    /// Bits 0..1: the path kind — 0 a length-2 NEE sample, 1 a length-≥3
    /// reconnection sample, 2 a pure-replay sample. Bits 2..3: the
    /// terminal-event kind, bit 4: the inside-the-instance re-shade flag,
    /// bit 5: the pair criteria verdict, bits 6..13: k, the reconnection
    /// vertex's path index — or a replay sample's terminal bounce
    /// (`restir_scene.slang`, D-137/D-138/D-139); the rest reserved.
    pub reserved: u32,
}

/// The persistent per-pixel path reservoir record — the host mirror of
/// `StoredPathReservoir`. {sample, W, M} plus the reserved `ReSTCV` lane = 96 B;
/// `weightSum` is pass-local and never stored (see `shaders/reservoir.slang`).
/// The path reservoir buffers are arrays of this, one entry per pixel, row-major.
/// Larger than Enhanced's 64 B by three named choices (see the shader) — packing
/// back toward 64 B is the deferred size optimization if the per-view budget bites.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct StoredPathReservoir {
    /// Y — the held path sample.
    pub sample: PathSample,
    /// `W_Y` — the unbiased contribution weight.
    pub unbiased_weight: f32,
    /// M — the generalized sample count (kept float; see the shader).
    pub confidence: f32,
    /// Reserved (path/reconnection flags); zero until step 2.
    pub flags: u32,
    /// Pad: 16-byte-aligns the `ReSTCV` lane so this mirror matches std430.
    pub pad0: u32,
    /// `ReSTCV` accumulated colour — the per-reservoir control variate (D-127),
    /// live from 6b-i (D-144): candidate mean out of the candidate stage,
    /// passed through temporal, the CV combination out of spatial.
    pub cv_accumulator: [f32; 3],
    /// `ReSTCV` running weight; reserved-zero until 6b-ii claims or retires it.
    pub cv_normalization: f32,
}

// The layout mirrors `shaders/reservoir_path.slang`. As with the DI record, a
// drift on either side is a silent bias bug, so pin the sizes at compile time —
// the GPU round-trip test then proves the field *offsets* agree, which the
// three std430 `float3` lanes make load-bearing and sizes alone cannot.
const _: () = assert!(size_of::<ReconnectionVertex>() == 16);
const _: () = assert!(size_of::<PathSample>() == 64);
const _: () = assert!(size_of::<StoredPathReservoir>() == 96);

/// The GPU mirror of `struct RestirScene` in `shaders/restir_scene.slang`: the
/// reservoir path's own scene slice — the triangle-only candidate light table,
/// the environment coin-flip probability, and the stable-identity remap — behind
/// one address the two initial-RIS stages carry beside the shared `SceneTable`.
/// Kept off `SceneTable` so the rest of the wavefront, which neither has nor
/// needs it, keeps its push constants small.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RestirSceneTable {
    candidate_lights: vk::DeviceAddress,
    /// Candidate-table length; the environment is the coin-flip partner, not a
    /// row (D-088).
    light_count: u32,
    /// p(a candidate is the environment) = env / (env + triangle power) — the
    /// triangle-only companion to `SceneTable.env_select_prob`, undiluted by the
    /// delta lights the reservoir excludes.
    env_select_prob: f32,
    instance_to_id: vk::DeviceAddress,
    id_to_instance: vk::DeviceAddress,
    /// The delta-only power-alias table (`lights::delta_table`) — the distant
    /// and point lights the reservoir excludes (D-088), added on exact NEE by
    /// `restir_resolve`.
    delta_lights: vk::DeviceAddress,
    /// Delta-table length; 0 leaves the padded record below unread, exactly as
    /// `light_count` guards the candidate table.
    delta_count: u32,
    _pad0: u32,
}

// Two u32 scalars fill the gap between the leading pointer and the next, so the
// layout is tight — mirrored field for field in restir_scene.slang.
const _: () = assert!(size_of::<RestirSceneTable>() == 48);

/// The reservoir path's resident scene buffers, held to keep them alive while
/// the two restir stages reach them through [`RestirResources::table`]'s
/// addresses. Rebuilt whenever a light edit churns the candidate table or the
/// identity remap, exactly as the shared scene table is.
pub(crate) struct RestirResources {
    // Reached only by GPU address, through `table`; held for residency.
    #[expect(dead_code, reason = "resident, reached by address via the table")]
    candidate_lights: Buffer,
    #[expect(dead_code, reason = "resident, reached by address via the table")]
    instance_to_id: Buffer,
    #[expect(dead_code, reason = "resident, reached by address via the table")]
    id_to_instance: Buffer,
    #[expect(dead_code, reason = "resident, reached by address via the table")]
    delta_lights: Buffer,
    /// The uploaded [`RestirSceneTable`] the two restir stages point at.
    table: Buffer,
}

impl RestirResources {
    /// The `RestirScene` table address, for the restir stages' push constants.
    pub(crate) fn table(&self) -> &Buffer {
        &self.table
    }

    /// Build the reservoir path's resident scene slice: upload the candidate
    /// table, the delta-only table, the two remap tables, and the
    /// [`RestirSceneTable`] naming them. `env_select_prob` is the triangle-only
    /// environment coin-flip probability.
    ///
    /// # Errors
    ///
    /// Any [`crate::Error`] from buffer creation.
    pub(crate) fn upload(
        gpu: &Context,
        candidate_table: &[LightRecord],
        delta_table: &[LightRecord],
        remap: &LightRemap,
        env_select_prob: f32,
    ) -> Result<Self> {
        let usage =
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        // Vulkan forbids empty buffers; an environment-only scene has no
        // triangle candidates, so it uploads one zeroed record the kernels
        // never read (the table says count 0).
        let padded = [LightRecord::zeroed()];
        let candidate_lights = gpu.upload_buffer(
            "restir.candidates",
            bytemuck::cast_slice(if candidate_table.is_empty() {
                &padded
            } else {
                candidate_table
            }),
            usage,
        )?;
        // The delta lights, uploaded the same way: a scene with none uploads the
        // zeroed placeholder the kernel never reads (the table says count 0).
        let delta_lights = gpu.upload_buffer(
            "restir.deltas",
            bytemuck::cast_slice(if delta_table.is_empty() {
                &padded
            } else {
                delta_table
            }),
            usage,
        )?;
        // `instance_to_id` sizes to the instance count, so an empty scene pads
        // it with one unreachable slot the same way; `id_to_instance` is never
        // empty (the reserved id block holds at least the environment
        // sentinel's slot).
        let padded_id = [LIGHT_ID_NONE];
        let instance_to_id = gpu.upload_buffer(
            "restir.instance_to_id",
            bytemuck::cast_slice(if remap.instance_to_id.is_empty() {
                &padded_id
            } else {
                &remap.instance_to_id
            }),
            usage,
        )?;
        let id_to_instance = gpu.upload_buffer(
            "restir.id_to_instance",
            bytemuck::cast_slice(&remap.id_to_instance),
            usage,
        )?;
        let table = RestirSceneTable {
            candidate_lights: candidate_lights.device_address(),
            light_count: candidate_table.len() as u32,
            env_select_prob,
            instance_to_id: instance_to_id.device_address(),
            id_to_instance: id_to_instance.device_address(),
            delta_lights: delta_lights.device_address(),
            delta_count: delta_table.len() as u32,
            _pad0: 0,
        };
        let table = gpu.upload_buffer("restir.table", bytemuck::bytes_of(&table), usage)?;
        Ok(Self {
            candidate_lights,
            instance_to_id,
            id_to_instance,
            delta_lights,
            table,
        })
    }
}

// Stable light identity — the registry that keeps a reservoir's stored light
// reference meaningful across scene edits — lives in its own file, colocated
// with its unit tests (T2).
mod identity;

// Per-view reservoir ownership — the reservoir buffers a viewport's reuse
// reads and writes, keyed by a stable view identity — lives in its own file,
// colocated with its lifecycle test (T4). The render loop owns one `ViewState`
// per film for the temporal ping-pong (step 5); the `ViewId`-keyed `Views` map
// is the still-dormant multi-view seam (step 6).
mod view;
pub(crate) use view::{ViewId, ViewState};

#[cfg(test)]
mod tests {
    use ash::vk;
    use bytemuck::{Pod, Zeroable};

    use crate::gpu::{Bindings, MemoryLocation};

    /// Audit the reservoir primitive on the GPU it ships on, through the
    /// test-only kernel `shaders/reservoir_test.slang`. `ReSTIR` bias is
    /// silent — it shifts a converged mean a few percent and renders
    /// plausibly — so, exactly as with the sampler's stratification test, no
    /// image-level test would catch a broken WRS/GRIS bookkeeping. Each of
    /// ~260k threads runs an independent RIS estimate of ∫₀¹ x dx = 1/2; the
    /// mean over all threads must land on 1/2 far tighter than a missing
    /// 1/M, a wrong selection probability, or a wrong UCW formula would
    /// allow. Both the initial-RIS path (reservoirUpdate + reservoirFinalize)
    /// and the reuse path (reservoirMerge) are checked, and the merged
    /// confidence is pinned exactly.
    #[test]
    fn reservoir_wrs_is_unbiased() {
        const COUNT: u32 = 1 << 18; // ~260k independent reservoirs
        const CANDIDATES: u32 = 16; // M for the initial-RIS estimator
        const CANDIDATES_A: u32 = 8; // M_A for the merge estimator
        const CANDIDATES_B: u32 = 24; // M_B for the merge estimator

        /// Mirrors `struct Params` in `shaders/reservoir_test.slang`.
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct DumpParams {
            estimate_update: vk::DeviceAddress,
            estimate_merge: vk::DeviceAddress,
            merged_m: vk::DeviceAddress,
            count: u32,
            candidates: u32,
            candidates_a: u32,
            candidates_b: u32,
            _pad0: u32,
            _pad1: u32,
        }

        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let spirv =
            crate::shaders::compile_fixture("reservoir_test").expect("compile reservoir_test");
        let pipeline = gpu
            .create_compute_pipeline(
                &spirv,
                c"reservoir_test",
                size_of::<DumpParams>() as u32,
                Bindings::None,
            )
            .expect("pipeline");

        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_SRC;
        let dump = |name: &str| {
            gpu.create_buffer(name, u64::from(COUNT) * 4, usage, MemoryLocation::GpuOnly)
                .expect("buffer")
        };
        let estimate_update = dump("test.reservoir.update");
        let estimate_merge = dump("test.reservoir.merge");
        let merged_m = dump("test.reservoir.m");

        let params = DumpParams {
            estimate_update: estimate_update.device_address(),
            estimate_merge: estimate_merge.device_address(),
            merged_m: merged_m.device_address(),
            count: COUNT,
            candidates: CANDIDATES,
            candidates_a: CANDIDATES_A,
            candidates_b: CANDIDATES_B,
            _pad0: 0,
            _pad1: 0,
        };
        gpu.dispatch(
            &pipeline,
            None,
            bytemuck::bytes_of(&params),
            [COUNT.div_ceil(64), 1, 1],
        )
        .expect("dispatch");

        let read = |buffer| -> Vec<f32> {
            bytemuck::pod_collect_to_vec(&gpu.download_buffer(buffer).expect("download"))
        };
        let update = read(&estimate_update);
        let merge = read(&estimate_merge);
        let confidence = read(&merged_m);

        let mean = |values: &[f32]| f64::from(values.iter().sum::<f32>()) / values.len() as f64;
        // Per-thread variance is ≤ Var(U[0,1]) = 1/12, so the standard error
        // over 2^18 threads is ~6e-4 — this 5e-3 window is ~8σ of headroom
        // yet still shuts on a 1% bias, the regime ReSTIR errors live in.
        let update_mean = mean(&update);
        let merge_mean = mean(&merge);
        assert!(
            (update_mean - 0.5).abs() < 5e-3,
            "initial-RIS estimate of 1/2 is biased: {update_mean}"
        );
        assert!(
            (merge_mean - 0.5).abs() < 5e-3,
            "GRIS-merge estimate of 1/2 is biased: {merge_mean}"
        );

        // The merge folds two histories in, so the combined confidence is the
        // sum of the two sample counts, exactly — a merge that mislaid a
        // candidate's confidence would drift the MIS weights of every reuse.
        let expected = (CANDIDATES_A + CANDIDATES_B) as f32;
        assert!(
            confidence.iter().all(|&m| (m - expected).abs() < 1e-3),
            "merged confidence is not M_A + M_B = {expected}"
        );
    }

    /// The M3 step-4 correctness spine (D-093): the defensive pairwise-MIS spatial
    /// combine — its MIS weights, its accepted-count normalizer, and its
    /// visibility-aware target folded into W — is unbiased. Each of ~260k threads
    /// runs `shaders/restir_spatial_test.slang`, an independent estimate of
    /// ∫₀¹ x·V(x) dx with a synthetic binary visibility V = [x ≥ ½], whose closed
    /// form is ∫_½¹ x dx = 3/8. A wrong MIS normalizer, a mishandled accepted
    /// count, or dividing the finalize by `p̂_c` instead of the unshadowed target
    /// moves the mean off 3/8 by far more than this thread count's noise — the
    /// silent-bias class no image test would catch.
    ///
    /// Run at K = 5 (the shipping neighbour count) *and* K = 0: both must land on
    /// 3/8, which is the accepted-count normalizer working — a combine that
    /// divided by the nominal k rather than the accepted count would diverge as K
    /// changed. The merged confidence is pinned at `c_tot` = `M_c` + K·`M_i`.
    #[test]
    fn spatial_pairwise_mis_is_unbiased() {
        const COUNT: u32 = 1 << 18; // ~260k independent estimates
        const CANON_M: u32 = 12; // M_c — canonical RIS candidate count
        const NEIGHBOUR_M: u32 = 20; // M_i — each neighbour's candidate count

        /// Mirrors `struct Params` in `shaders/restir_spatial_test.slang`.
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct SpatialParams {
            estimate: vk::DeviceAddress,
            merged_m: vk::DeviceAddress,
            count: u32,
            canon_m: u32,
            neighbour_m: u32,
            neighbours: u32,
            _pad0: u32,
            _pad1: u32,
        }

        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let spirv = crate::shaders::compile_fixture("restir_spatial_test")
            .expect("compile restir_spatial_test");
        let pipeline = gpu
            .create_compute_pipeline(
                &spirv,
                c"restir_spatial_test",
                size_of::<SpatialParams>() as u32,
                Bindings::None,
            )
            .expect("pipeline");

        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_SRC;
        let dump = |name: &str| {
            gpu.create_buffer(name, u64::from(COUNT) * 4, usage, MemoryLocation::GpuOnly)
                .expect("buffer")
        };

        let run = |neighbours: u32| -> (f64, Vec<f32>) {
            let estimate = dump("test.spatial.estimate");
            let merged_m = dump("test.spatial.m");
            let params = SpatialParams {
                estimate: estimate.device_address(),
                merged_m: merged_m.device_address(),
                count: COUNT,
                canon_m: CANON_M,
                neighbour_m: NEIGHBOUR_M,
                neighbours,
                _pad0: 0,
                _pad1: 0,
            };
            gpu.dispatch(
                &pipeline,
                None,
                bytemuck::bytes_of(&params),
                [COUNT.div_ceil(64), 1, 1],
            )
            .expect("dispatch");
            let read = |buffer| -> Vec<f32> {
                bytemuck::pod_collect_to_vec(&gpu.download_buffer(buffer).expect("download"))
            };
            let estimates = read(&estimate);
            let mean = f64::from(estimates.iter().sum::<f32>()) / estimates.len() as f64;
            (mean, read(&merged_m))
        };

        // The binary-visibility integral: ∫_½¹ x dx = 3/8. The window is wide
        // enough to clear the estimator's own noise at this thread count, yet
        // shuts on the several-percent shift a broken combine produces.
        let truth = 0.375_f64;
        for neighbours in [5u32, 0] {
            let (mean, confidence) = run(neighbours);
            assert!(
                (mean - truth).abs() < 1e-2,
                "K={neighbours}: spatial combine is biased: {mean} vs {truth}"
            );
            // The combine sums every history: c_tot = M_c + K·M_i, exactly.
            let expected = (CANON_M + neighbours * NEIGHBOUR_M) as f32;
            assert!(
                confidence.iter().all(|&m| (m - expected).abs() < 1e-3),
                "K={neighbours}: merged confidence is not c_tot = {expected}"
            );
        }
    }

    /// The step-5b correctness spine (D-094): the defensive pairwise-MIS
    /// *temporal* combine — folding one M-capped history reservoir into the
    /// canonical — is unbiased, and the M-cap is applied consistently. Each of
    /// ~260k threads runs `shaders/restir_temporal_test.slang`, an independent
    /// estimate of ∫₀¹ x dx = ½ (unshadowed, so no synthetic visibility unlike
    /// the spatial test). A wrong MIS normalizer moves the mean off ½ by far more
    /// than this thread count's noise.
    ///
    /// The integral alone cannot catch a broken M-cap — the estimator is unbiased
    /// for *any* partition-of-unity weights, so clamping confidence never shifts
    /// the mean. The merged confidence does: it must be `c_c + min(c_prev,
    /// M_CAP·c_c)`, so the test runs one config where the cap does *not* bind
    /// (`c_prev` below the cap) and one where it does (well above), and pins the
    /// merged confidence in both — a cap applied to the MIS weights but not the
    /// merged count (or vice versa) lands a value the host did not predict.
    #[test]
    fn temporal_pairwise_mis_is_unbiased_and_capped() {
        const COUNT: u32 = 1 << 18; // ~260k independent estimates
        const CANON_M: u32 = 12; // M_c — canonical RIS candidate count
        const M_CAP: f32 = 20.0; // the shipping history-length multiplier

        /// Mirrors `struct Params` in `shaders/restir_temporal_test.slang`.
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct TemporalParams {
            estimate: vk::DeviceAddress,
            merged_m: vk::DeviceAddress,
            count: u32,
            canon_m: u32,
            prev_m: u32,
            m_cap: f32,
            _pad0: u32,
            _pad1: u32,
        }

        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let spirv = crate::shaders::compile_fixture("restir_temporal_test")
            .expect("compile restir_temporal_test");
        let pipeline = gpu
            .create_compute_pipeline(
                &spirv,
                c"restir_temporal_test",
                size_of::<TemporalParams>() as u32,
                Bindings::None,
            )
            .expect("pipeline");

        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_SRC;
        let dump = |name: &str| {
            gpu.create_buffer(name, u64::from(COUNT) * 4, usage, MemoryLocation::GpuOnly)
                .expect("buffer")
        };

        let run = |prev_m: u32| -> (f64, Vec<f32>) {
            let estimate = dump("test.temporal.estimate");
            let merged_m = dump("test.temporal.m");
            let params = TemporalParams {
                estimate: estimate.device_address(),
                merged_m: merged_m.device_address(),
                count: COUNT,
                canon_m: CANON_M,
                prev_m,
                m_cap: M_CAP,
                _pad0: 0,
                _pad1: 0,
            };
            gpu.dispatch(
                &pipeline,
                None,
                bytemuck::bytes_of(&params),
                [COUNT.div_ceil(64), 1, 1],
            )
            .expect("dispatch");
            let read = |buffer| -> Vec<f32> {
                bytemuck::pod_collect_to_vec(&gpu.download_buffer(buffer).expect("download"))
            };
            let estimates = read(&estimate);
            let mean = f64::from(estimates.iter().sum::<f32>()) / estimates.len() as f64;
            (mean, read(&merged_m))
        };

        // The unshadowed integral: ∫₀¹ x dx = ½.
        let truth = 0.5_f64;
        // `cap` = M_CAP·M_c = 240. Below it the cap is inert; above it, it binds.
        let cap = (M_CAP * CANON_M as f32) as u32;
        for prev_m in [CANON_M * 8, cap * 2] {
            let (mean, confidence) = run(prev_m);
            assert!(
                (mean - truth).abs() < 1e-2,
                "prev_M={prev_m}: temporal combine is biased: {mean} vs {truth}"
            );
            // The merged history is c_c + the *capped* prev confidence, exactly.
            let expected = (CANON_M + prev_m.min(cap)) as f32;
            assert!(
                confidence.iter().all(|&m| (m - expected).abs() < 1e-3),
                "prev_M={prev_m}: merged confidence is not c_c + min(c_prev, cap) = {expected}"
            );
        }
    }

    /// Step 5c: `reprojectPixel` (`shaders/restir_reproject.slang`) is the exact
    /// inverse of raygen's pixel→ray construction. The whole held-camera argument
    /// — that temporal reuse reduces to same-pixel reuse when the camera does not
    /// move — rests on a point seen at pixel p reprojecting back to p through the
    /// same camera. Each thread runs the *real* shader function on a world point
    /// synthesized as the hit of a known pixel (`origin + t·dir(p)`, raygen's own
    /// ray) and must recover exactly p; points behind the camera and off the
    /// screen must be rejected. A sign flip, a dropped aspect factor, or a swapped
    /// ndc→uv axis lands the wrong pixel here. (Reprojection is variance-only, not
    /// bias — D-094 — but a broken inverse would silently gut reuse quality, and
    /// the identity reduction the convergence gate leans on would be a fiction.)
    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "the camera, the synthesized ray-hit grid, and the reject cases \
                  are one round-trip gate — splitting them would scatter it"
    )]
    fn reprojection_inverts_raygen() {
        use glam::{Vec2, Vec3};

        use crate::scene::Camera;
        use crate::wavefront::{Reproject, ReprojectCamera};

        /// Mirrors `struct Params` in `shaders/restir_reproject_test.slang`.
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct ReprojectParams {
            reproject: vk::DeviceAddress,
            points: vk::DeviceAddress,
            pixel_out: vk::DeviceAddress,
            count: u32,
            _pad0: u32,
            _pad1: u32,
            _pad2: u32,
        }

        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let spirv = crate::shaders::compile_fixture("restir_reproject_test")
            .expect("compile restir_reproject_test");
        let pipeline = gpu
            .create_compute_pipeline(
                &spirv,
                c"restir_reproject_test",
                size_of::<ReprojectParams>() as u32,
                Bindings::None,
            )
            .expect("pipeline");

        let (width, height) = (64_u32, 48_u32);
        let camera = Camera {
            position: Vec3::new(0.0, 1.0, 4.0),
            look_at: Vec3::new(0.3, 0.5, 0.0),
            up: Vec3::Y,
            vfov_degrees: 45.0,
            lens: None,
        };
        let basis = camera.basis(width as f32 / height as f32);
        let reproject = Reproject::new(
            Some(ReprojectCamera {
                position: camera.position,
                right: basis.right,
                up: basis.up,
                forward: basis.forward,
            }),
            0, // G-buffer addresses are unused by reprojectPixel
            0,
            width,
            height,
        );

        // The world hit raygen's pixel-`p` ray would reach: NDC from the pixel
        // *centre* (jitter is immaterial — it stays inside the pixel and floor
        // recovers p), `dir = normalize(forward + ndc.x·right + ndc.y·up)`,
        // walked `t` along. Textually raygen's forward map, so this pins its
        // inverse.
        let hit_of = |x: u32, y: u32, t: f32| -> Vec3 {
            let uv = Vec2::new(
                (x as f32 + 0.5) / width as f32,
                (y as f32 + 0.5) / height as f32,
            );
            let ndc = Vec2::new(2.0 * uv.x - 1.0, 1.0 - 2.0 * uv.y);
            let dir = (basis.forward + ndc.x * basis.right + ndc.y * basis.up).normalize();
            camera.position + dir * t
        };

        // A grid of pixels, each at a different depth, must round-trip exactly.
        let mut points: Vec<[f32; 4]> = Vec::new();
        let mut expected: Vec<u32> = Vec::new();
        for y in (0..height).step_by(5) {
            for x in (0..width).step_by(3) {
                let t = 2.0 + (x + y) as f32 * 0.1; // vary the distance
                points.push(hit_of(x, y, t).extend(0.0).to_array());
                expected.push(y * width + x);
            }
        }
        // Rejects: a point behind the camera, and one off the side of the frame
        // (ndc.x ≫ 1). Both must come back 0xffffffff.
        let reject_from = points.len();
        points.push((camera.position - basis.forward * 2.0).extend(0.0).to_array());
        points.push(
            (camera.position + (basis.forward + basis.right * 5.0).normalize() * 3.0)
                .extend(0.0)
                .to_array(),
        );
        let count = points.len() as u32;

        let reproject_buf = gpu
            .upload_buffer(
                "test.reproject",
                bytemuck::bytes_of(&reproject),
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            )
            .expect("reproject buffer");
        let points_buf = gpu
            .upload_buffer(
                "test.reproject.points",
                bytemuck::cast_slice(&points),
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            )
            .expect("points buffer");
        let pixel_out = gpu
            .create_buffer(
                "test.reproject.out",
                u64::from(count) * 4,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::GpuOnly,
            )
            .expect("out buffer");

        gpu.dispatch(
            &pipeline,
            None,
            bytemuck::bytes_of(&ReprojectParams {
                reproject: reproject_buf.device_address(),
                points: points_buf.device_address(),
                pixel_out: pixel_out.device_address(),
                count,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            }),
            [count.div_ceil(64), 1, 1],
        )
        .expect("dispatch");

        let out: Vec<u32> = bytemuck::pod_collect_to_vec(
            &gpu.download_buffer(&pixel_out).expect("download"),
        );
        for (i, &want) in expected.iter().enumerate() {
            assert_eq!(
                out[i], want,
                "pixel {want} did not reproject to itself (got {})",
                out[i]
            );
        }
        assert_eq!(out[reject_from], 0xffff_ffff, "a point behind the camera must reject");
        assert_eq!(out[reject_from + 1], 0xffff_ffff, "an off-screen point must reject");
    }

    /// T0 (M6): the 96 B path reservoir record round-trips between the GPU's
    /// `StoredPathReservoir` and this host mirror, byte for byte. The struct is
    /// the milestone's central artifact — designed once (D-128) so no rung
    /// re-lays it out — and its three std430 `float3` lanes make the offsets, not
    /// just the size, load-bearing: a `float3` padded to a 16-byte slot on one
    /// side and packed on the other would shift every field after it and bias
    /// silently. Each scalar field gets a distinct transform (a plain copy would
    /// survive a symmetric layout slip unseen); the reserved flags/pad/`ReSTCV`
    /// lanes are seeded non-zero on input and asserted zero on output, proving
    /// `store` clears the step-6 slots and that those bytes exist where expected.
    /// This is the "struct allocates and round-trips" step-0 checkpoint.
    #[test]
    #[allow(clippy::too_many_lines, reason = "one assertion per reservoir field — the exhaustiveness is the test")]
    fn stored_path_reservoir_round_trips_through_the_gpu() {
        use super::{PathSample, ReconnectionVertex, StoredPathReservoir};

        const COUNT: u32 = 4096; // several workgroups, varied per-field values

        /// Mirrors `struct Params` in `shaders/reservoir_path_test.slang`.
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct PathRoundTripParams {
            input: vk::DeviceAddress,
            output: vk::DeviceAddress,
            loaded_weight_sum: vk::DeviceAddress,
            oct_agreement: vk::DeviceAddress,
            count: u32,
            _pad0: u32,
            _pad1: u32,
            _pad2: u32,
        }

        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };

        // Distinctive per-field inputs: any field read from the wrong offset
        // reads a neighbour's unrelated value, and the transform then misses.
        // The reserved lanes carry non-zero junk so the zero-on-output assert
        // proves `store` clears them rather than finding them already clear.
        let input: Vec<StoredPathReservoir> = (0..COUNT)
            .map(|i| {
                let f = i as f32;
                StoredPathReservoir {
                    sample: PathSample {
                        rc_vertex: ReconnectionVertex {
                            instance: i * 7 + 1,
                            primitive: i * 13 + 3,
                            barycentrics: [f * 0.001 + 0.1, f * 0.001 + 0.2],
                        },
                        rc_vertex_radiance: [f * 0.001 + 0.3, f * 0.001 + 0.4, f * 0.001 + 0.5],
                        rc_vertex_wi: i * 17 + 5,
                        f: [f * 0.001 + 0.6, f * 0.001 + 0.7, f * 0.001 + 0.8],
                        nee_light_pdf: f * 0.001 + 1.0,
                        init_random_seed: i * 19 + 7,
                        rc_vertex_random_seed: i * 23 + 9,
                        cached_jacobian: f * 0.001 + 2.0,
                        reserved: i * 29 + 11,
                    },
                    unbiased_weight: f * 0.5 + 1.0,
                    confidence: f * 0.25 + 2.0,
                    // Reserved lanes: non-zero junk `store` must overwrite with
                    // 0. The CV lane (live from 6b-i) instead round-trips with
                    // its own transform, like every live field.
                    flags: i * 31 + 13,
                    pad0: i * 37 + 15,
                    cv_accumulator: [f * 0.001 + 9.0, f * 0.001 + 9.3, f * 0.001 + 9.6],
                    cv_normalization: f * 0.001 + 9.9,
                }
            })
            .collect();

        let spirv = crate::shaders::compile_fixture("reservoir_path_test")
            .expect("compile reservoir_path_test");
        let pipeline = gpu
            .create_compute_pipeline(
                &spirv,
                c"reservoir_path_test",
                size_of::<PathRoundTripParams>() as u32,
                Bindings::None,
            )
            .expect("pipeline");

        let io_usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_SRC;
        let input_buffer = gpu
            .upload_buffer("test.reservoir.path.in", bytemuck::cast_slice(&input), io_usage)
            .expect("upload input");
        let output_buffer = gpu
            .create_buffer(
                "test.reservoir.path.out",
                u64::from(COUNT) * size_of::<StoredPathReservoir>() as u64,
                io_usage,
                MemoryLocation::GpuOnly,
            )
            .expect("output buffer");
        let weight_sum_buffer = gpu
            .create_buffer(
                "test.reservoir.path.wsum",
                u64::from(COUNT) * 4,
                io_usage,
                MemoryLocation::GpuOnly,
            )
            .expect("weight-sum buffer");
        let oct_buffer = gpu
            .create_buffer(
                "test.reservoir.path.oct",
                u64::from(COUNT) * 4,
                io_usage,
                MemoryLocation::GpuOnly,
            )
            .expect("oct buffer");

        let params = PathRoundTripParams {
            input: input_buffer.device_address(),
            output: output_buffer.device_address(),
            loaded_weight_sum: weight_sum_buffer.device_address(),
            oct_agreement: oct_buffer.device_address(),
            count: COUNT,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        gpu.dispatch(
            &pipeline,
            None,
            bytemuck::bytes_of(&params),
            [COUNT.div_ceil(64), 1, 1],
        )
        .expect("dispatch");

        let output: Vec<StoredPathReservoir> =
            bytemuck::pod_collect_to_vec(&gpu.download_buffer(&output_buffer).expect("download"));
        let weight_sum: Vec<f32> = bytemuck::pod_collect_to_vec(
            &gpu.download_buffer(&weight_sum_buffer).expect("download"),
        );

        for (i, (got, src)) in output.iter().zip(&input).enumerate() {
            let near = |a: f32, b: f32, what: &str| {
                assert!((a - b).abs() < 1e-4, "{what}[{i}]: {a} vs {b}");
            };
            // Integer fields must match exactly; a layout slip turns them into
            // reinterpreted float bytes, wildly wrong.
            assert_eq!(
                got.sample.rc_vertex.instance,
                src.sample.rc_vertex.instance + 1,
                "instance[{i}]"
            );
            assert_eq!(
                got.sample.rc_vertex.primitive,
                src.sample.rc_vertex.primitive + 2,
                "primitive[{i}]"
            );
            near(
                got.sample.rc_vertex.barycentrics[0],
                src.sample.rc_vertex.barycentrics[0] + 0.25,
                "bary.x",
            );
            near(
                got.sample.rc_vertex.barycentrics[1],
                src.sample.rc_vertex.barycentrics[1] + 0.5,
                "bary.y",
            );
            near(got.sample.rc_vertex_radiance[0], src.sample.rc_vertex_radiance[0] + 1.0, "Lk.x");
            near(got.sample.rc_vertex_radiance[1], src.sample.rc_vertex_radiance[1] + 2.0, "Lk.y");
            near(got.sample.rc_vertex_radiance[2], src.sample.rc_vertex_radiance[2] + 3.0, "Lk.z");
            assert_eq!(got.sample.rc_vertex_wi, src.sample.rc_vertex_wi + 4, "wi[{i}]");
            near(got.sample.f[0], src.sample.f[0] + 0.1, "F.x");
            near(got.sample.f[1], src.sample.f[1] + 0.2, "F.y");
            near(got.sample.f[2], src.sample.f[2] + 0.3, "F.z");
            near(got.sample.nee_light_pdf, src.sample.nee_light_pdf * 2.0, "neePdf");
            assert_eq!(
                got.sample.init_random_seed,
                src.sample.init_random_seed + 5,
                "initSeed[{i}]"
            );
            assert_eq!(
                got.sample.rc_vertex_random_seed,
                src.sample.rc_vertex_random_seed + 6,
                "rcSeed[{i}]"
            );
            near(got.sample.cached_jacobian, src.sample.cached_jacobian * 3.0, "jacobian");
            assert_eq!(got.sample.reserved, src.sample.reserved + 7, "reserved[{i}]");
            near(got.unbiased_weight, src.unbiased_weight * 2.0, "W");
            near(got.confidence, src.confidence + 100.0, "M");

            // The CV lane is live from 6b-i: it rides through store as an
            // argument, and its per-component transform pins the float3 at
            // bytes 80..92 by value, not just by the size assert.
            near(got.cv_accumulator[0], src.cv_accumulator[0] + 0.4, "cv.x");
            near(got.cv_accumulator[1], src.cv_accumulator[1] + 0.5, "cv.y");
            near(got.cv_accumulator[2], src.cv_accumulator[2] + 0.6, "cv.z");

            // store must clear the reserved flags/pad and the still-reserved
            // cvNormalization, regardless of the non-zero junk seeded on input.
            assert_eq!(got.flags, 0, "flags[{i}] not cleared");
            assert_eq!(got.pad0, 0, "pad0[{i}] not cleared");
            // `== 0.0` against a literal is the exact-zero idiom clippy exempts
            // (as the weightSum check below): store writes this lane to exactly 0.
            assert!(got.cv_normalization == 0.0, "cvNormalization[{i}] not cleared");
        }

        // load must zero weightSum regardless of what was stored — the
        // pass-local invariant, enforced at the read boundary.
        assert!(
            weight_sum.iter().all(|&w| w == 0.0),
            "loadPathReservoir left a non-zero weightSum"
        );

        // The ωₖ octahedral packing round-trips within its quantization cone
        // over a sphere of hashed directions (step 3a, D-137): a broken
        // hemisphere fold or swapped component decodes a direction degrees
        // away, far below this bound.
        let oct: Vec<f32> =
            bytemuck::pod_collect_to_vec(&gpu.download_buffer(&oct_buffer).expect("download"));
        assert!(
            oct.iter().all(|&d| d > 0.999_99),
            "packUnitDirection does not round-trip: worst dot {}",
            oct.iter().copied().fold(f32::INFINITY, f32::min)
        );
    }
}
