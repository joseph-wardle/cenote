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
//! together is the cache-friendly layout here (see `DirectLightSample` and the
//! packed-reservoir mirror below). The persistent per-pixel record is
//! {sample, `unbiasedWeight`, confidence} — `weightSum` is pass-local and never
//! stored, the one invariant `reservoir.slang` documents at length.

use ash::vk;
use bytemuck::{Pod, Zeroable};

use crate::error::Result;
use crate::gpu::{Buffer, Context};
use crate::lights::LightRecord;

pub(crate) use identity::{EmissiveLight, LightIdentityRegistry};
use identity::LightRemap;

/// The DI reservoir's concrete sample — the host mirror of `DirectLightSample`
/// in `shaders/reservoir_di.slang`. A light surface point in the re-evaluable
/// reconnection form (the `Hit` shape), which is what makes the DI shift map
/// the identity and the GRIS Jacobian 1. `light` is the *stable* light id (the
/// registry below), never the volatile TLAS custom index.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct DirectLightSample {
    /// Stable light id (see the identity registry).
    pub light: u32,
    /// Triangle index within the light's mesh.
    pub primitive: u32,
    /// The sampled point on that triangle.
    pub barycentrics: [f32; 2],
}

/// The persistent per-pixel reservoir record — the host mirror of
/// `StoredReservoir`. {sample, W, c} = 24 B exactly; `weightSum` is pass-local
/// and never stored (see `shaders/reservoir.slang`). The reservoir buffers are
/// arrays of this, one entry per pixel, row-major (Morton ordering deferred —
/// it would perturb the pixel↔slot mapping the determinism invariant rests on).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct StoredReservoir {
    /// Y — the held light sample.
    pub sample: DirectLightSample,
    /// `W_Y` — the unbiased contribution weight.
    pub unbiased_weight: f32,
    /// M — the generalized sample count.
    pub confidence: f32,
}

// The layout mirrors `shaders/reservoir_di.slang`; the sizes are the plan's
// 24 B/pixel figure. A drift on either side is a bias bug waiting to happen,
// so pin both at compile time — the GPU round-trip test (T3) then proves the
// field *offsets* agree, which sizes alone cannot.
const _: () = assert!(size_of::<DirectLightSample>() == 16);
const _: () = assert!(size_of::<StoredReservoir>() == 24);

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
        // Both remap tables are non-empty for any non-empty scene: `instance_to_id`
        // sizes to the instance count, `id_to_instance` to the reserved-plus-minted
        // id block (at least the environment sentinel's slot).
        let instance_to_id = gpu.upload_buffer(
            "restir.instance_to_id",
            bytemuck::cast_slice(&remap.instance_to_id),
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

// Per-view reservoir ownership — the three buffers a viewport's reuse reads
// and writes, keyed by a stable view identity — lives in its own file,
// colocated with its lifecycle test (T4).
mod view;

#[cfg(test)]
mod tests {
    use ash::vk;
    use bytemuck::{Pod, Zeroable};

    use super::{DirectLightSample, StoredReservoir};
    use crate::gpu::{Bindings, MemoryLocation};

    /// Mirrors `struct Params` in `shaders/reservoir_di_test.slang`.
    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    struct RoundTripParams {
        input: vk::DeviceAddress,
        output: vk::DeviceAddress,
        loaded_weight_sum: vk::DeviceAddress,
        count: u32,
        _pad0: u32,
        _pad1: u32,
        _pad2: u32,
    }

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

    /// T3: the persistent reservoir record round-trips between the GPU's
    /// `StoredReservoir` and this host mirror, byte for byte, and load/store
    /// are inverse. The fixture gives every scalar field a distinct transform
    /// (a plain copy would survive a symmetric layout mismatch unseen), so a
    /// swapped or mis-aligned field lands a value this side did not predict.
    /// The `AoS` reservoir buffer is exactly this array — proving it
    /// round-trips is proving the buffers allocate and read back correctly,
    /// the step-2 checkpoint.
    #[test]
    fn stored_reservoir_round_trips_through_the_gpu() {
        const COUNT: u32 = 4096; // several workgroups, varied per-field values

        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };

        // Distinctive per-field inputs: any field read from the wrong offset
        // reads a neighbour's unrelated value, and the transform then misses.
        let input: Vec<StoredReservoir> = (0..COUNT)
            .map(|i| StoredReservoir {
                sample: DirectLightSample {
                    light: i * 7 + 1,
                    primitive: i * 13 + 3,
                    barycentrics: [i as f32 * 0.001 + 0.1, i as f32 * 0.002 + 0.2],
                },
                unbiased_weight: i as f32 * 0.5 + 1.0,
                confidence: i as f32 * 0.25 + 2.0,
            })
            .collect();

        let spirv = crate::shaders::compile_fixture("reservoir_di_test")
            .expect("compile reservoir_di_test");
        let pipeline = gpu
            .create_compute_pipeline(
                &spirv,
                c"reservoir_di_test",
                size_of::<RoundTripParams>() as u32,
                Bindings::None,
            )
            .expect("pipeline");

        let io_usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_SRC;
        let input_buffer = gpu
            .upload_buffer(
                "test.reservoir.di.in",
                bytemuck::cast_slice(&input),
                io_usage,
            )
            .expect("upload input");
        let output_buffer = gpu
            .create_buffer(
                "test.reservoir.di.out",
                u64::from(COUNT) * size_of::<StoredReservoir>() as u64,
                io_usage,
                MemoryLocation::GpuOnly,
            )
            .expect("output buffer");
        let weight_sum_buffer = gpu
            .create_buffer(
                "test.reservoir.di.wsum",
                u64::from(COUNT) * 4,
                io_usage,
                MemoryLocation::GpuOnly,
            )
            .expect("weight-sum buffer");

        let params = RoundTripParams {
            input: input_buffer.device_address(),
            output: output_buffer.device_address(),
            loaded_weight_sum: weight_sum_buffer.device_address(),
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

        let output: Vec<StoredReservoir> =
            bytemuck::pod_collect_to_vec(&gpu.download_buffer(&output_buffer).expect("download"));
        let weight_sum: Vec<f32> = bytemuck::pod_collect_to_vec(
            &gpu.download_buffer(&weight_sum_buffer).expect("download"),
        );

        for (i, (got, src)) in output.iter().zip(&input).enumerate() {
            // Integer fields must match exactly; a layout slip turns them into
            // reinterpreted float bytes, wildly wrong.
            assert_eq!(got.sample.light, src.sample.light + 1, "light[{i}]");
            assert_eq!(
                got.sample.primitive,
                src.sample.primitive + 2,
                "primitive[{i}]"
            );
            let near = |a: f32, b: f32, what: &str| {
                assert!((a - b).abs() < 1e-4, "{what}[{i}]: {a} vs {b}");
            };
            near(
                got.sample.barycentrics[0],
                src.sample.barycentrics[0] + 0.25,
                "bary.x",
            );
            near(
                got.sample.barycentrics[1],
                src.sample.barycentrics[1] + 0.5,
                "bary.y",
            );
            near(got.unbiased_weight, src.unbiased_weight * 2.0, "W");
            near(got.confidence, src.confidence + 100.0, "M");
        }

        // load must zero weightSum regardless of what was stored — the
        // pass-local invariant, enforced at the read boundary.
        assert!(
            weight_sum.iter().all(|&w| w == 0.0),
            "loadReservoir left a non-zero weightSum"
        );
    }
}
