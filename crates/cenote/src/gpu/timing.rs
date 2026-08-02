//! GPU timing: two timestamps around every submission, and more inside the
//! ones that ask.
//!
//! The write is a *command*, not shader code — it never enters a kernel, so it
//! cannot cost a register, change occupancy, or otherwise perturb the thing it
//! is measuring. It is not free anyway. On an RTX 4070 Ti SUPER (NVIDIA
//! 580.173), brass-room at 512² — 71 passes over a 2.3 ms frame, the worst case
//! in this renderer — eight interleaved 800-sample runs per arm say:
//!
//! | what is stamped | vs `--no-gpu-timers` |
//! |---|---:|
//! | every pass boundary, every frame | **+49.9%** |
//! | every pass boundary, one frame in eight | **+7.8%** |
//! | every *span* boundary, one frame in 32, plus the two outer stamps every frame | **+1.9%** |
//! | the two outer stamps alone, every frame | **±0%**, inside the noise |
//!
//! That last row is what the design rests on. An interior stamp costs ~15 µs —
//! a real serialization point, which the memory barrier already in front of it
//! evidently does not absorb — while the two bracketing a submission cost
//! nothing measurable, because the GPU is draining at both ends anyway.
//! *Where* the time went is expensive; *how much* there was is free.
//!
//! So they are asked separately. Every submission is bracketed, which is what
//! lets [`crate::stats::Report`] weigh device time against wall clock over
//! every frame rather than a sample of them; the per-kernel breakdown runs one
//! frame in [`PassTimer::BREAKDOWN_INTERVAL`]. [`crate::stats::Recorder`] keeps
//! the two populations apart, so the frame-time distribution reports what the
//! renderer costs when nobody is looking.
//!
//! Two hypotheses died here and are worth not re-testing: resetting the query
//! pool from the host (`hostQueryReset`) rather than in the command buffer
//! changed nothing, and neither did the stamp's stage mask in any of its three
//! spellings. The cost is the write, not the wait in front of it.
//!
//! `cenote-cli --no-gpu-timers` is the A/B arm, kept so the claim stays
//! re-checkable. Re-check before trusting the table: absolute figures travel
//! badly between sittings on this machine, so only interleaved arms compare.
//!
//! A [`PassTimer`] owns one `VkQueryPool`, grown to fit the most spans it has
//! been handed; [`Context::submit_passes_timed`] is the one place that knows
//! how a wave is recorded. Submission blocks, so results are read straight back
//! after the fence.

use std::time::Duration;

use ash::vk;

use crate::error::Result;
use crate::gpu::submit::spans;
use crate::gpu::{Context, Pass};
use crate::stats::PassTimings;

/// How much of a submission one frame's timestamps resolve.
///
/// The two questions a stats line answers cost wildly different amounts to
/// ask, so they are asked separately: *how long was the device busy* needs
/// the two outer boundaries and nothing else, while *where did that go*
/// needs a stamp at every span boundary in between.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Detail {
    /// Two stamps around the whole submission: what it cost, not where.
    Total,
    /// A stamp at every span boundary: what each kernel cost.
    Breakdown,
}

impl Detail {
    /// Timestamp intervals this detail records over `passes`. Stamps are
    /// one more, since N intervals have N+1 boundaries.
    ///
    /// Recording and readback both ask here rather than each counting for
    /// themselves — the one way a stamp and a readback come to disagree
    /// about which queries were written is by counting them twice.
    pub(super) fn intervals(self, passes: &[Pass]) -> u32 {
        match self {
            Detail::Total => 1,
            Detail::Breakdown => u32::try_from(spans(passes).count()).unwrap_or(u32::MAX),
        }
    }
}

/// A query pool of GPU timestamps, one per boundary [`Detail`] asks for.
///
/// Created through [`Context::create_pass_timer`] and handed to
/// [`Context::submit_passes_timed`]. Not shareable across threads by
/// design: a pool records into one command buffer at a time, and cenote's
/// submissions are per-thread blocking anyway.
pub struct PassTimer {
    pool: vk::QueryPool,
    /// Timestamp slots the pool holds — one more than the intervals it can
    /// time, since N intervals have N+1 boundaries.
    slots: u32,
    /// Nanoseconds per timestamp tick, from `VkPhysicalDeviceLimits`.
    period_ns: f64,
    /// Mask of the bits the queue actually fills; the rest are undefined.
    valid_mask: u64,
    /// Scratch for readback, reused so timing allocates nothing per frame.
    raw: Vec<u64>,
    /// Frames since the last breakdown — see [`PassTimer::detail`].
    since_breakdown: u32,
    device: ash::Device,
}

impl PassTimer {
    /// Timestamp intervals a fresh timer starts with — enough for an
    /// ordinary wave (a fill and a handful of dispatches per bounce, per
    /// pool-sized pixel range) at a cost of 4 KB of query pool.
    ///
    /// It is a starting point, not a ceiling: a submission that outruns the
    /// pool grows it rather than going unmeasured. Guessing a ceiling here
    /// would be guessing wrong — the first real capture already found a
    /// corpus scene asking for 65 bounces, which is ~1000 passes at 1080p.
    pub const WAVE_CAPACITY: u32 = 512;

    /// Frames between breakdowns. The total is measured every frame at two
    /// stamps; only the per-kernel attribution runs on this cadence, and
    /// thirty-two of them still refreshes it several times a second at any
    /// interactive frame rate — far faster than a person reads it.
    pub const BREAKDOWN_INTERVAL: u32 = 32;

    /// What this frame's submission gets, advancing the cadence.
    ///
    /// The first call is due, so a render that only ever draws one frame
    /// still gets its breakdown.
    pub(super) fn detail(&mut self) -> Detail {
        let due = self.since_breakdown == 0;
        self.since_breakdown = (self.since_breakdown + 1) % Self::BREAKDOWN_INTERVAL;
        if due { Detail::Breakdown } else { Detail::Total }
    }

    /// How many intervals one submission may span and still be timed.
    #[must_use]
    pub fn capacity(&self) -> u32 {
        self.slots - 1
    }

    /// Record the reset that must precede any timestamp write in this
    /// command buffer.
    pub(super) fn reset(&self, cmd: vk::CommandBuffer, boundaries: u32) {
        unsafe {
            self.device
                .cmd_reset_query_pool(cmd, self.pool, 0, boundaries);
        }
    }

    /// Stamp boundary `index` once everything recorded before it has run.
    ///
    /// `ALL_COMMANDS` and not the narrower `COMPUTE_SHADER | ALL_TRANSFER`
    /// that would name the only two stages a [`Pass`] can occupy, for two
    /// reasons that agree. It is the legal spelling:
    /// `vkCmdWriteTimestamp2` takes a *single* stage
    /// (VUID-vkCmdWriteTimestamp2-stage-03859), and the two-bit mask is a
    /// validation error whose behaviour the spec does not define — so the
    /// narrowing was never a narrowing, only an unspecified one. And it is
    /// the correct one: a span of fills is transfer work, so a stamp that
    /// waited on compute alone would close the span before the fills
    /// finished and report a kernel as free.
    pub(super) fn stamp(&self, cmd: vk::CommandBuffer, index: u32) {
        unsafe {
            self.device.cmd_write_timestamp2(
                cmd,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                self.pool,
                index,
            );
        }
    }

    /// Read the boundaries `detail` wrote around `passes` back and
    /// difference them into device time. Call only after the submission's
    /// fence has signalled — every query is then available and the wait
    /// flag never blocks.
    pub(super) fn read(&mut self, passes: &[Pass], detail: Detail) -> Result<PassTimings> {
        let intervals = detail.intervals(passes) as usize;
        self.raw.clear();
        self.raw.resize(intervals + 1, 0);
        unsafe {
            self.device.get_query_pool_results(
                self.pool,
                0,
                &mut self.raw,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )?;
        }
        // The outer boundaries are the total either way — the first and
        // last stamp written, whatever lies between them.
        let mut timings = PassTimings::measured(self.elapsed(self.raw[0], self.raw[intervals]));
        if detail == Detail::Breakdown {
            for ((label, calls), pair) in spans(passes).zip(self.raw.windows(2)) {
                timings.add(label, self.elapsed(pair[0], pair[1]), calls);
            }
        }
        Ok(timings)
    }

    /// The time between two timestamps.
    ///
    /// Only `valid_mask`'s bits are defined and the counter can wrap within
    /// them, so the difference is taken masked. A wrap needs ~2 minutes of
    /// GPU time at typical periods — far longer than one submission — so
    /// the wrapped difference is the true one.
    fn elapsed(&self, from: u64, to: u64) -> Duration {
        let ticks = to.wrapping_sub(from) & self.valid_mask;
        // Tick counts stay far inside f64's exact integer range.
        Duration::from_secs_f64(ticks as f64 * self.period_ns / 1e9)
    }

    /// Make room for a submission of `intervals` timestamp intervals,
    /// replacing the pool with a larger one if it is short.
    ///
    /// Growing rather than declining to measure: a span count is a property
    /// of the scene (its bounce cap) and the target (how many pool-sized
    /// pixel ranges it takes), so any fixed ceiling is a guess that some
    /// scene will falsify — as one in the corpus immediately did. The pool
    /// only ever grows, and only between submissions, so this settles after
    /// the first wave of a given shape and costs nothing thereafter.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Vulkan`] if the larger pool cannot be created. The
    /// old pool is kept in that case, so the timer stays usable.
    pub(super) fn grow_to(&mut self, intervals: u32) -> Result<()> {
        if intervals < self.slots {
            return Ok(());
        }
        let slots = (intervals + 1).next_power_of_two();
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(slots);
        let pool = unsafe { self.device.create_query_pool(&info, None)? };
        log::debug!("pass timer grew from {} to {slots} timestamps", self.slots);
        unsafe { self.device.destroy_query_pool(self.pool, None) };
        self.pool = pool;
        self.slots = slots;
        self.raw.reserve(slots as usize);
        Ok(())
    }
}

impl Drop for PassTimer {
    fn drop(&mut self) {
        unsafe { self.device.destroy_query_pool(self.pool, None) };
    }
}

impl Context {
    /// A timer able to resolve submissions of up to `intervals` spans, or
    /// `None` where the device's compute queue does not support timestamps
    /// (some transfer-only and virtualized queues report zero valid bits).
    /// Callers treat `None` as "this build measures nothing" and carry on.
    pub fn create_pass_timer(&self, intervals: u32) -> Result<Option<PassTimer>> {
        if self.timestamp_valid_bits == 0 || self.timestamp_period == 0.0 {
            log::info!("device reports no queue timestamps; GPU pass timing is off");
            return Ok(None);
        }
        let slots = intervals + 1;
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(slots);
        let pool = unsafe { self.device().create_query_pool(&info, None)? };
        Ok(Some(PassTimer {
            pool,
            slots,
            period_ns: f64::from(self.timestamp_period),
            // 64 valid bits is the common case and shifting by 64 is UB-adjacent
            // in Rust (it panics in debug), so the full mask is spelled out.
            valid_mask: if self.timestamp_valid_bits >= 64 {
                u64::MAX
            } else {
                (1_u64 << self.timestamp_valid_bits) - 1
            },
            raw: Vec::with_capacity(slots as usize),
            since_breakdown: 0,
            device: self.device().clone(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use bytemuck::{Pod, Zeroable};

    use super::*;
    use crate::gpu::{Bindings, Buffer, ComputePipeline, MemoryLocation};

    /// Mirrors `struct Params` in `shaders/rng_test.slang` — borrowed here
    /// as a kernel that does real, sizeable work with no scene behind it.
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

    /// The `rng_test` dispatch both tests below measure, holding the pipeline
    /// and the two buffers it writes — one fixture because a [`Pass::Dispatch`]
    /// borrows the pipeline and the push constants borrow the addresses, so all
    /// three have to outlive the submission together.
    struct RngDispatch {
        pipeline: ComputePipeline,
        points: Buffer,
        values: Buffer,
    }

    impl RngDispatch {
        /// Points per dispatch — enough work that the span is measurable.
        const COUNT: u32 = 64;
        /// Byte size of the `float2` point buffer, which is also what the
        /// tests' `Pass::Fill` clears.
        const POINT_BYTES: u64 = Self::COUNT as u64 * 8;

        fn new(gpu: &Context) -> Self {
            let spirv = crate::shaders::compile_fixture("rng_test").expect("compile rng_test");
            let usage = vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_DST;
            let buffer = |name, bytes| {
                gpu.create_buffer(name, bytes, usage, MemoryLocation::GpuOnly)
                    .expect("fixture buffer")
            };
            Self {
                pipeline: gpu
                    .create_compute_pipeline(
                        &spirv,
                        c"rng_test",
                        size_of::<DumpParams>() as u32,
                        Bindings::None,
                    )
                    .expect("pipeline"),
                points: buffer("test.timing.points", Self::POINT_BYTES),
                values: buffer("test.timing.values", u64::from(Self::COUNT) * 4),
            }
        }

        /// Push constants aimed at this fixture's buffers.
        fn params(&self) -> DumpParams {
            DumpParams {
                points: self.points.device_address(),
                values: self.values.device_address(),
                pixel: 0,
                dimension: 0,
                count: Self::COUNT,
                _pad0: 0,
            }
        }

        /// The dispatch pass, and the fill that clears what it writes.
        fn passes<'a>(&'a self, params: &'a DumpParams) -> (Pass<'a>, Pass<'a>) {
            (
                Pass::Dispatch {
                    pipeline: &self.pipeline,
                    scene: None,
                    push_constants: bytemuck::bytes_of(params),
                    group_counts: [1, 1, 1],
                },
                Pass::Fill {
                    buffer: &self.points,
                    offset: 0,
                    size: Self::POINT_BYTES,
                    value: 0,
                },
            )
        }
    }

    /// The whole instrument, end to end on the GPU it ships on: a pool is
    /// created, a submission is stamped at every boundary, and the results
    /// come back as per-*kernel* durations under the kernel's own name.
    ///
    /// Three properties, and each one is a way the timer could be
    /// plausibly broken while still returning numbers: the labels must be
    /// the entry-point names (not indices, which a re-cut loop would
    /// scramble), repeated dispatches must fold into one entry carrying
    /// the call count, and the measured time must be non-zero and bounded
    /// by the wall-clock of the submission that produced it — a timer
    /// reporting microseconds for a millisecond of work is worse than no
    /// timer.
    #[test]
    fn a_timed_submission_reports_per_kernel_durations() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let Some(mut timer) = gpu.create_pass_timer(16).expect("timer") else {
            eprintln!("skipping: this queue reports no timestamps");
            return;
        };

        let fixture = RngDispatch::new(&gpu);
        let params = fixture.params();
        let (dispatch, clear) = fixture.passes(&params);
        // A fill, then the kernel twice — the shape a wave has, where one
        // kernel runs once per bounce.
        let passes = [clear, dispatch, dispatch];

        let started = std::time::Instant::now();
        let timings = gpu.submit_passes_timed(&passes, Some(&mut timer)).expect("timed");
        let wall = started.elapsed();

        let entries: Vec<_> = timings.iter().collect();
        assert_eq!(entries.len(), 2, "one entry per distinct kernel: {entries:?}");
        assert_eq!(entries[0].label, "fill", "data passes bucket by kind");
        assert_eq!(
            entries[1].label, "rng_test",
            "a dispatch reports under its entry-point name"
        );
        assert_eq!(entries[1].calls, 2, "both dispatches folded into one entry");
        assert!(
            !timings.total().is_zero(),
            "the GPU cannot have taken zero time"
        );
        assert!(
            timings.total() <= wall,
            "GPU time {:?} cannot exceed the wall-clock {wall:?} of the submission that ran it",
            timings.total()
        );
        // The total is read at the outer boundaries and the entries are
        // read span by span, so agreeing is a property of shared stamps
        // rather than of arithmetic — and the one way to notice if a span
        // ever stops sharing them. A microsecond of slack covers the
        // per-interval rounding into `Duration`.
        let attributed: Duration = entries.iter().map(|entry| entry.gpu).sum();
        assert!(
            timings.total().abs_diff(attributed) < Duration::from_micros(1),
            "the spans telescope to the whole: {:?} attributed against {:?} measured",
            attributed,
            timings.total(),
        );
    }

    /// Consecutive passes under one label are timed as a single span.
    ///
    /// The saving is the whole reason spans exist: the wave opens every
    /// bounce with three queue-clearing fills and the submission with as
    /// many as seven, and a stamp between two of them would buy a number
    /// [`PassTimings`] sums straight back together — at the price of a real
    /// timestamp write. The pool not growing is how the test sees it: four
    /// fills fit a timer built for one interval.
    #[test]
    fn consecutive_passes_under_one_label_share_a_span() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let Some(mut timer) = gpu.create_pass_timer(1).expect("timer") else {
            return;
        };
        let buffer = gpu
            .create_buffer(
                "test.timing.spans",
                256,
                vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuOnly,
            )
            .expect("buffer");
        let fill = Pass::Fill {
            buffer: &buffer,
            offset: 0,
            size: 256,
            value: 0,
        };
        let timings = gpu
            .submit_passes_timed(&[fill, fill, fill, fill], Some(&mut timer))
            .expect("the passes run");
        let entries: Vec<_> = timings.iter().collect();
        assert_eq!(entries.len(), 1, "one bucket for the fills");
        assert_eq!(entries[0].calls, 4, "all four were measured, not two");
        assert_eq!(
            timer.capacity(),
            1,
            "and four passes under one label cost one interval, so the \
             pool had no reason to grow"
        );
    }

    /// A submission past the timer's capacity in *spans* grows it and is
    /// measured anyway.
    ///
    /// This is not hypothetical: the very first baseline capture hit it —
    /// a corpus scene asking for 65 bounces records ~1000 passes at 1080p,
    /// past any starting capacity worth allocating up front. A timer that
    /// answered "no measurement" there would have gone quiet on exactly the
    /// scene worth measuring, and said so only in a log line.
    #[test]
    fn a_submission_past_the_timers_capacity_grows_it() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        // A pool that fits one span, then a submission of four.
        let Some(mut timer) = gpu.create_pass_timer(1).expect("timer") else {
            return;
        };
        let fixture = RngDispatch::new(&gpu);
        let params = fixture.params();
        // Alternating kinds, so every pass opens its own span — the shape a
        // wave has, where fills separate the stages they reset.
        let (dispatch, fill) = fixture.passes(&params);
        let timings = gpu
            .submit_passes_timed(&[fill, dispatch, fill, dispatch], Some(&mut timer))
            .expect("the passes run");
        let entries: Vec<_> = timings.iter().collect();
        assert_eq!(entries.len(), 2, "one bucket per kind: {entries:?}");
        assert_eq!(entries[0].calls, 2, "both fills were measured");
        assert_eq!(entries[1].calls, 2, "and both dispatches");
        assert!(timer.capacity() >= 4, "and the pool kept the room");
    }

    /// Between breakdowns a submission is still bracketed: it reports what
    /// the device spent without saying where, for two timestamps instead of
    /// one per span.
    ///
    /// This is what lets the run-level verdict cover every frame — see
    /// [`crate::stats::Bound::of`] — so a timer that quietly reported zero
    /// on the unresolved frames would leave the report weighing a sample
    /// against the whole.
    #[test]
    fn frames_between_breakdowns_report_a_total_alone() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let Some(mut timer) = gpu.create_pass_timer(4).expect("timer") else {
            return;
        };
        let buffer = gpu
            .create_buffer(
                "test.timing.cadence",
                1 << 20,
                vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuOnly,
            )
            .expect("buffer");
        let fill = Pass::Fill {
            buffer: &buffer,
            offset: 0,
            size: 1 << 20,
            value: 0,
        };

        // The first submission is due a breakdown; the next is not.
        let first = gpu
            .submit_passes_timed(&[fill], Some(&mut timer))
            .expect("the first submission runs");
        assert!(first.has_breakdown(), "the first frame is always resolved");

        let second = gpu
            .submit_passes_timed(&[fill], Some(&mut timer))
            .expect("the second submission runs");
        assert!(
            !second.has_breakdown(),
            "the frames in between say nothing about kernels"
        );
        assert!(
            !second.total().is_zero(),
            "but they do say what the device spent"
        );
    }
}
