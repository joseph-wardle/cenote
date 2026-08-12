//! What a frame cost, how the render is progressing, and where the memory
//! went — the renderer's one measurement surface.
//!
//! Everything here is **tier one**: always on, and cheap enough that leaving it
//! on cannot meaningfully change the number it reports. That constraint is what
//! admits each entry. GPU times come from timestamps written at pass
//! boundaries, by a command that never enters a kernel; bracketing a submission
//! for its total is free, stamping *inside* it is not, which is why only the
//! per-kernel breakdown is rationed (see [`crate::gpu::PassTimer`]). The
//! interactivity marks are host `Instant` differences off events the render loop
//! already has, and the [`Phases`] beneath them are a handful more of the same,
//! taken at boundaries the update path already crosses — some hundreds of
//! nanoseconds against a mark measured in tens of milliseconds. Memory is a
//! running total of allocations the renderer already makes, sampled rather than
//! recomputed.
//!
//! [`Phases`] is also where the tier-one line falls for latency, and it falls at
//! *per phase*. Per-object attribution — which mesh, which texture, which of
//! sixty-nine thousand instances — is a span per object, and a span per object at
//! that scale is milliseconds of instrument inside the thing it measures. When a
//! phase comes back bad and the next question is *which one*, the answer is an
//! opt-in trace, not an always-on one.
//!
//! Anything costing a register, a reduction pass, or a readback per frame —
//! ray counts, per-pixel variance — is tier two: opt-in, compiled out by
//! default, and not here. Occupancy, register pressure, and cache throughput
//! are absent on purpose: Vulkan cannot portably report them, and re-deriving
//! them means rebuilding Nsight or the Radeon GPU Profiler badly. When a
//! decision needs those numbers, attach the real tool.
//!
//! [`Recorder`] owns the running state; [`Stats`] is the plain snapshot it
//! produces, and the single source every consumer reads — the viewer's overlay,
//! the CLI's report, the Hydra delegate — so a number cannot mean two things in
//! two places. It travels beside the pixels, never inside them.

use std::collections::VecDeque;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// Samples after which an image is treated as readable for the
/// time-to-readable mark. Not a convergence claim — a true one needs
/// per-pixel variance, which is tier two. This is the cheap stand-in: the
/// point where a preview stops being obviously noise.
pub const READABLE_SAMPLES: u32 = 16;

/// Frames the rolling window keeps for the overlay's smoothed read.
const WINDOW: usize = 120;

/// Fraction of the CPU frame the GPU must be busy for the frame to count as
/// GPU-bound. Below it, the time went somewhere the dispatches are not —
/// recording, submission, driver, or the host half of the loop — which is
/// the signature of a wave that is launching more kernels than it is doing
/// work in.
const GPU_BOUND_FRACTION: f64 = 0.9;

/// When the process started, for [`Interactivity::to_first_ray`].
static STARTUP: OnceLock<Instant> = OnceLock::new();

/// Mark process start. A binary that wants an honest time-to-first-ray
/// calls this as its first statement in `main` — everything after it
/// (shader compile, scene load, pipeline creation, the acceleration-
/// structure build) then lands inside the measurement, which is the whole
/// point of the number.
///
/// Idempotent, and safe to skip: a [`Recorder`] built without it origins at
/// its own creation instead, which under-reports startup by whatever ran
/// before. It never over-reports.
pub fn mark_startup() {
    let _ = STARTUP.set(Instant::now());
}

/// Process start if [`mark_startup`] was called, else now — and now becomes
/// the origin for everyone after, so one run never mixes two origins. The
/// one zero in the crate: [`Recorder::new`] origins time-to-first-ray here,
/// and [`Scene::prep_timed`](crate::scene::Scene::prep_timed) origins the
/// wait in front of a load here, which is why the two add up.
pub(crate) fn startup() -> Instant {
    *STARTUP.get_or_init(Instant::now)
}

/// One kernel's contribution to a frame, summed over however many times the
/// wave dispatched it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PassTiming {
    /// The kernel's entry-point name — see [`crate::gpu::Pass::label`].
    pub label: String,
    /// Total GPU time across this frame's dispatches of it.
    #[serde(with = "millis")]
    pub gpu: Duration,
    /// How many times the frame dispatched it.
    pub calls: u32,
}

/// How long the device was busy, and — when it was asked — which kernels
/// it was busy in, in the order they first ran.
///
/// The two are separable because they cost so differently to measure: the
/// total is two timestamps, the attribution is one per span. Most frames
/// carry the total alone; see [`crate::gpu::PassTimer`].
///
/// The attribution is keyed by label rather than by dispatch index on
/// purpose: a wave dispatches `intersect` once per bounce and `fill` a
/// dozen times, and the useful question is what each *kernel* cost, not
/// what the seventh pass cost. It is also the form that survives the loop
/// being re-cut — split a kernel and two names appear; merge two and one
/// does.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PassTimings {
    /// Device time this measurement covers, whether or not anything below
    /// says where it went.
    #[serde(with = "millis")]
    total: Duration,
    entries: Vec<PassTiming>,
}

impl PassTimings {
    /// A submission that took `total` on the device, not yet attributed.
    /// The only way to state a total: it is *measured*, at the outer
    /// boundaries, and never accumulated from the parts.
    pub(crate) fn measured(total: Duration) -> Self {
        Self {
            total,
            entries: Vec::new(),
        }
    }

    /// Attribute one span's time to `label`'s entry, starting the entry if
    /// the kernel has not run yet.
    ///
    /// This says where the total went, so it leaves the total alone. The
    /// two agree without being made to: spans share their boundary stamps,
    /// so the attributed times telescope to exactly the outer interval.
    ///
    /// The scan is linear in the number of *distinct* kernels, which is
    /// around ten — the wrong side of a map's break-even — and it keeps the
    /// entries in dispatch order, which is the order a reader traces.
    ///
    /// Crate-internal, like [`PassTimings::merge`]: outside this crate a
    /// `PassTimings` is a measurement, and a measurement nobody can add to
    /// is one nobody can forge.
    pub(crate) fn add(&mut self, label: &str, gpu: Duration, calls: u32) {
        if let Some(entry) = self.entries.iter_mut().find(|entry| entry.label == label) {
            entry.gpu += gpu;
            entry.calls += calls;
        } else {
            self.entries.push(PassTiming {
                label: label.to_owned(),
                gpu,
                calls,
            });
        }
    }

    /// Fold another frame's measurement into these, for the session totals.
    pub(crate) fn merge(&mut self, other: &Self) {
        self.total += other.total;
        for entry in &other.entries {
            self.add(&entry.label, entry.gpu, entry.calls);
        }
    }

    /// The kernels, in the order they first ran.
    pub fn iter(&self) -> std::slice::Iter<'_, PassTiming> {
        self.entries.iter()
    }

    /// Device time this measurement covers.
    #[must_use]
    pub fn total(&self) -> Duration {
        self.total
    }

    /// Whether the time above is attributed to kernels. False on the frames
    /// that carried only the outer boundaries — thirty-one in thirty-two of
    /// them — and on a device without queue timestamps, where there is no
    /// time to attribute either. The common case, not a failure.
    #[must_use]
    pub fn has_breakdown(&self) -> bool {
        !self.entries.is_empty()
    }
}

impl<'a> IntoIterator for &'a PassTimings {
    type Item = &'a PassTiming;
    type IntoIter = std::slice::Iter<'a, PassTiming>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Which side of the loop a frame spent its time on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Bound {
    /// The dispatches account for the frame: the kernels are the cost.
    Gpu,
    /// They do not: the gap went to recording, submission, or the host.
    /// At low sample counts this is the shape of a wave paying more to
    /// launch its kernels than to run them.
    Cpu,
    /// Nothing was measured, so the question has no answer — the default,
    /// because an unmeasured frame must never read as a measured one.
    #[default]
    Unknown,
}

impl Bound {
    /// The verdict for a summed GPU time against the wall-clock of the
    /// frames that produced it.
    ///
    /// Both durations must cover the *same* frames. That is the whole
    /// discipline of this function: divide a GPU total drawn from some
    /// frames by the wall-clock of all of them and the answer is scaled by
    /// the ratio while still looking like a percentage. Measuring the
    /// device time of *every* frame — see [`crate::gpu::PassTimer`] — is
    /// what lets the run-level verdict use the whole run.
    #[must_use]
    pub fn of(cpu: Duration, gpu: Duration) -> Self {
        if cpu.is_zero() || gpu.is_zero() {
            return Bound::Unknown;
        }
        if gpu.as_secs_f64() >= cpu.as_secs_f64() * GPU_BOUND_FRACTION {
            Bound::Gpu
        } else {
            Bound::Cpu
        }
    }
}

/// The verdict as a word, spelled once so the overlay and the report cannot
/// disagree about what the same [`Bound`] is called.
impl std::fmt::Display for Bound {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.write_str(match self {
            Bound::Gpu => "gpu-bound",
            Bound::Cpu => "cpu-bound",
            Bound::Unknown => "not measured",
        })
    }
}

/// What one sample cost.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Frame {
    /// Wall-clock on the render thread: trace plus film accumulate, the
    /// whole round trip including submission and the fence wait.
    #[serde(with = "millis")]
    pub cpu: Duration,
    /// Per-kernel GPU time inside that.
    pub passes: PassTimings,
    /// What it rendered at.
    pub size: (u32, u32),
    /// Samples in the film's average after it.
    pub samples: u32,
    /// Whether `size` is a reduced rectangle rather than the target's own
    /// resolution. Only the render thread knows — [`Frame::size`] alone
    /// cannot say, the target size not being in here — and only
    /// [`Recorder::record`]'s readable mark reads it.
    pub preview: bool,
}

impl Frame {
    /// Summed GPU time across the frame's dispatches.
    #[must_use]
    pub fn gpu(&self) -> Duration {
        self.passes.total()
    }

    /// Whether the dispatches account for the frame — see [`Bound`]. Both
    /// numbers stay visible beside it so the size of the gap is readable,
    /// not just its verdict.
    #[must_use]
    pub fn bound(&self) -> Bound {
        Bound::of(self.cpu, self.gpu())
    }

    /// One line of a per-frame trace: compact RON, one value per line, so a
    /// settle curve is a file you can `grep` and a plotter can read a line
    /// at a time without holding the run in memory.
    pub fn to_ron_line(&self) -> crate::Result<String> {
        ron::ser::to_string(self).map_err(crate::Error::stats)
    }
}

/// Where an update's latency went, in the phases worth naming.
///
/// A fixed struct rather than the labelled list [`PassTimings`] carries, for
/// the reason [`Memory`]'s buckets are fixed: these eight *are* the
/// editorial claim about where an update's time can go, so a reader learns
/// the shape of the update path from this file, and a ninth span has to be
/// argued for here rather than added at a call site.
///
/// The phases account for their mark **exactly**. Seven are clocked at the
/// boundaries of the work they name; [`Phases::other`] is whatever the mark
/// covers that they do not, filled in when the mark lands. So
/// [`Phases::total`] equals the [`Interactivity`] mark it explains by
/// construction, and a phase that quietly stopped covering its work shows up
/// as `other` growing rather than as a total that no longer adds up.
///
/// Every phase is a host `Instant`, and that is exact rather than
/// approximate because every GPU call the update path makes blocks on a
/// fence — see [`Context::submit_once`](crate::gpu::Context). Nothing here
/// needs a query pool.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Phases {
    /// Before any of the work below could begin: an edit's wait for the next
    /// wave boundary, or — for a scene load — process start to the first
    /// prep call. Invisible from inside the renderer, and on a busy sample
    /// the largest term a person actually feels.
    #[serde(with = "millis")]
    pub before: Duration,
    /// Applying the change-set to the description: the clone it takes for
    /// atomicity, the patch, and validation. Zero for a load or a camera
    /// move, and the one phase here that is pure host work.
    #[serde(with = "millis")]
    pub apply: Duration,
    /// Host-side lowering: PLY reads, texture decode and block compression,
    /// the environment's sampling tables. Everything that can fail on user
    /// data happens in here, before the first GPU call.
    #[serde(with = "millis")]
    pub lower: Duration,
    /// Getting it onto the device: mesh buffers and the bottom-level
    /// acceleration structures over them, textures, the environment image.
    #[serde(with = "millis")]
    pub upload: Duration,
    /// The top-level acceleration structure over the instances, and the
    /// placement list it is built from. Zero on an edit that moved nothing.
    #[serde(with = "millis")]
    pub tlas: Duration,
    /// The instance tables (geometry, materials, lights) and the scene
    /// table. These rebuild wholesale on any edit, whatever it touched, so
    /// this phase is where a scene's instance count shows up whether or not
    /// the edit had anything to do with instances.
    #[serde(with = "millis")]
    pub tables: Duration,
    /// The first sample after the restart — the accumulate whose completion
    /// *is* the mark.
    #[serde(with = "millis")]
    pub sample: Duration,
    /// The remainder: whatever the mark covers that the phases above do not
    /// name. On an edit it is loop overhead and should stay near zero, so a
    /// large one means a phase has stopped covering its work. On a load it
    /// is honest bulk — the shader compile and pipeline creation that happen
    /// outside prep — and showing it beats hiding it in a phase that would
    /// then be lying.
    #[serde(with = "millis")]
    pub other: Duration,
    /// How many queued edits this drain covered. Zero for a load or a camera
    /// move; above one when a burst landed on the same wave boundary, in
    /// which case `before` is the *oldest* edit's wait — the longest anyone
    /// waited, which is the one worth reporting.
    pub batched: u32,
}

impl Phases {
    /// The interval these phases account for, which is the mark they
    /// explain.
    #[must_use]
    pub fn total(&self) -> Duration {
        self.measured() + self.other
    }

    /// The clocked phases alone — what [`Phases::other`] is the remainder
    /// of.
    fn measured(&self) -> Duration {
        self.before + self.apply + self.lower + self.upload + self.tlas + self.tables + self.sample
    }

    /// The phases in the order they run, for display.
    #[must_use]
    pub fn named(&self) -> [(&'static str, Duration); 8] {
        [
            ("before", self.before),
            ("apply", self.apply),
            ("lower", self.lower),
            ("upload", self.upload),
            ("tlas", self.tlas),
            ("tables", self.tables),
            ("sample", self.sample),
            ("other", self.other),
        ]
    }
}

/// Close a breakdown against the mark it explains: the sample that landed it
/// is the accumulate just recorded, and whatever the named phases still do
/// not cover is the remainder.
///
/// Saturating, because a breakdown claiming more time than its mark is a
/// caller whose clocks overlap, and a zero remainder beside an inflated
/// total is the reading that shows it.
fn close(mut phases: Phases, sample: Duration, mark: Duration) -> Phases {
    phases.sample = sample;
    phases.other = mark.saturating_sub(phases.measured());
    phases
}

/// How long the renderer took to become useful — the numbers a person
/// waiting on it actually feels.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Interactivity {
    /// Startup: process start to the first ray traced. Shader compile,
    /// pipeline creation, scene upload, and the acceleration-structure
    /// build all land in here. Measured once per scene load, not per frame.
    #[serde(with = "millis::option")]
    pub to_first_ray: Option<Duration>,
    /// Interaction: the camera move, edit, or resize that last restarted
    /// accumulation, to the first sample completed after it. What a click
    /// costs — origined at the *verb*, so the wait for the wave boundary is
    /// inside the number rather than in front of it.
    #[serde(with = "millis::option")]
    pub to_first_pixel: Option<Duration>,
    /// That same verb to [`READABLE_SAMPLES`] samples of the full-resolution
    /// image. The readability stand-in — see the constant.
    #[serde(with = "millis::option")]
    pub to_readable: Option<Duration>,
    /// Where [`Interactivity::to_first_ray`] went. `None` until the first
    /// sample lands, and on a caller that did not hand over a breakdown.
    pub load: Option<Phases>,
    /// Where the most recent [`Interactivity::to_first_pixel`] went. `None`
    /// until the first sample lands; after that it stays the last
    /// *completed* breakdown while the next one is in flight, so a consumer
    /// reading it never finds it blanked.
    pub interaction: Option<Phases>,
}

/// Where the device memory went, in the five buckets worth naming.
///
/// Five is the whole editorial claim: enough to answer *what would I have
/// to change to fit*, few enough to read at a glance. Per-allocation detail
/// is what a memory profiler is for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Memory {
    /// Geometry, materials, and the acceleration structures over them.
    pub scene: u64,
    /// The film: accumulation buffers, AOVs, and the published frames.
    pub film: u64,
    /// Decoded and compressed material textures.
    pub textures: u64,
    /// The wavefront's path pool and queues, plus host staging on its way
    /// to or from the device. This bucket is a receipt: it is mostly what a
    /// wave carries *in order to be* a wave, and it is expected to collapse
    /// when the queues do — down to the renderer's two global sampling
    /// tables, under a megabyte together, which stay.
    pub scratch: u64,
    /// The device-local heap, when the driver reports one — the headroom
    /// the totals above are spending against.
    pub budget: Option<u64>,
}

impl Memory {
    /// Everything the renderer is holding.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.scene + self.film + self.textures + self.scratch
    }

    /// The named buckets, largest concern first, for display.
    #[must_use]
    pub fn buckets(&self) -> [(&'static str, u64); 4] {
        [
            ("scene", self.scene),
            ("film", self.film),
            ("textures", self.textures),
            ("scratch", self.scratch),
        ]
    }
}

/// The rolling read of recent frame times.
///
/// A single frame's wall-clock is the noisiest number the renderer
/// produces, and an overlay showing it raw invites reading noise as
/// signal. The median is what the overlay shows; the 95th percentile is
/// the hitch beside it. Raw per-frame values are still what the trace
/// records — smoothing is for the eye, not for the log.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Smoothed {
    /// Median CPU frame time over the window.
    #[serde(with = "millis")]
    pub median: Duration,
    /// 95th percentile over the window — the hitch, not the average.
    #[serde(with = "millis")]
    pub p95: Duration,
}

/// One snapshot of everything tier one knows.
///
/// The single source of truth: the overlay renders a subset of it, the CLI
/// reports it, and it crosses the wire beside the pixels — never mixed into
/// them, so a framebuffer stays a framebuffer.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Stats {
    /// The most recent sample.
    pub frame: Frame,
    /// Recent frame times, smoothed for display.
    pub smoothed: Smoothed,
    /// How long it took to become useful.
    pub interactivity: Interactivity,
    /// Where the memory went.
    pub memory: Memory,
    /// What the last denoise cost, end to end — everything between the
    /// resolve and the post, copies and ownership transfers included, since
    /// the frame waits for all of it. `None` when nothing has been denoised.
    #[serde(with = "millis::option")]
    pub denoise: Option<Duration>,
}

/// The end-of-render summary: what the whole run cost, not what one frame
/// did.
///
/// Written beside the image as RON — the format the rest of the renderer
/// already speaks — so two runs diff as text and a piece of work is a
/// readable change rather than a remembered impression.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Report {
    /// The device that rendered it.
    pub device: String,
    /// The scene, as named on the command line.
    pub scene: String,
    /// What it rendered at.
    pub size: (u32, u32),
    /// Samples reached.
    pub samples: u32,
    /// Wall-clock from process start to the end of the render — startup
    /// and scene load included, because a person waited through those.
    #[serde(with = "millis")]
    pub wall: Duration,
    /// Summed wall-clock of the samples alone. The denominator the
    /// [`Report::bound`] verdict is against: comparing GPU time to `wall`
    /// would charge the dispatches for a scene load they did not do.
    #[serde(with = "millis")]
    pub sampling: Duration,
    /// Summed device time over those same samples — every one of them, not
    /// a subset, which is what makes the ratio against `sampling` a ratio.
    #[serde(with = "millis")]
    pub gpu: Duration,
    /// Samples rendered.
    pub frames: u64,
    /// How many of them also carried the per-kernel attribution. The kernel
    /// totals below are over these frames only — a per-call figure is
    /// exact, an absolute total is a sample of the run. Stated rather than
    /// hidden, because a stat that quietly means something other than it
    /// says is worse than no stat.
    pub breakdown_frames: u64,
    /// Summed GPU time per kernel across those frames — where the render
    /// went, in one list. Its own [`PassTimings::total`] is their device
    /// time, so a share-of-total reads against that and not against `gpu`.
    pub passes: PassTimings,
    /// Frame-time distribution over the run.
    pub smoothed: Smoothed,
    /// Mean CPU time per sample.
    #[serde(with = "millis")]
    pub mean_frame: Duration,
    /// Whether the dispatches accounted for the run.
    pub bound: Bound,
    /// The interactivity marks.
    pub interactivity: Interactivity,
    /// Peak memory per bucket over the run.
    pub peak_memory: Memory,
}

impl Report {
    /// The report as a sidecar file's contents.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Stats`] if serialization fails.
    pub fn to_ron(&self) -> crate::Result<String> {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default())
            .map_err(crate::Error::stats)
    }
}

/// The running state behind [`Stats`] — held by whoever drives the render
/// loop, and the only thing in this module that is not plain data.
pub struct Recorder {
    /// When the recorder was created, i.e. as close to process start as the
    /// caller could put it. The origin for [`Interactivity::to_first_ray`].
    origin: Instant,
    /// When the interaction that last restarted accumulation began — a
    /// camera move, an edit, a resize. The origin for the two
    /// per-interaction marks.
    reset: Instant,
    interactivity: Interactivity,
    /// Which of the two per-interaction marks the current reset still owes.
    /// Set by [`Recorder::restart`], cleared as each lands — so what is
    /// published stays the most recent *completed* measurement rather than
    /// blanking out for the duration of the next one.
    awaiting_first_pixel: bool,
    awaiting_readable: bool,
    /// The breakdown of the restart in flight, not yet closed: neither the
    /// sample nor the remainder is known until the mark it explains lands.
    /// Written by [`Recorder::restart`] and only there, so it can never
    /// describe a restart other than the current one.
    pending: Phases,
    /// The same for the load, kept separate because its mark is the
    /// once-only `to_first_ray` — which the same record can land alongside
    /// the first interaction's.
    pending_load: Option<Phases>,
    memory: Memory,
    peak_memory: Memory,
    /// What the last published frame's denoise cost, or `None` if none has
    /// been denoised. Latest rather than smoothed: it is one filter per
    /// publish, not a per-sample population.
    denoise: Option<Duration>,
    // The two populations are kept apart deliberately. A breakdown costs a
    // few percent of the frame it is taken on (see [`crate::gpu::PassTimer`]),
    // so a window that mixed those frames in would report a cost the
    // renderer only pays while being watched. The frame-time distribution
    // therefore comes from the plain frames alone. Everything else — the
    // run's device time, the bound verdict — is over both, because both
    // carry the outer boundary stamps and those are near enough to free.
    /// Recent plain frame times, for the smoothed read.
    window: VecDeque<Duration>,
    /// Count, summed wall-clock, and summed device time of the frames that
    /// carried only the outer boundaries. The mean frame time is theirs.
    plain_frames: u64,
    plain_cpu: Duration,
    plain_gpu: Duration,
    /// The same for the rarer frames that also carried the attribution —
    /// whose device time is `totals`' own, so it needs no field here.
    breakdown_frames: u64,
    breakdown_cpu: Duration,
    totals: PassTimings,
    /// The last frame that carried a breakdown, so [`Frame::bound`] on it
    /// compares a GPU time and a CPU time from the same sample.
    latest: Frame,
}

impl Recorder {
    /// Start recording, origining time-to-first-ray at [`mark_startup`] —
    /// so a recorder created on the render thread still measures the
    /// startup that happened before that thread existed.
    #[must_use]
    pub fn new() -> Self {
        Self::since(startup())
    }

    /// Start recording from an explicit origin, for a render whose "startup"
    /// is not the process's — a hot-reload re-render, where what a person
    /// waited through is the recompile, not the launch an hour ago.
    #[must_use]
    pub fn since(origin: Instant) -> Self {
        Self {
            origin,
            reset: origin,
            interactivity: Interactivity::default(),
            awaiting_first_pixel: true,
            awaiting_readable: true,
            pending: Phases::default(),
            pending_load: None,
            memory: Memory::default(),
            peak_memory: Memory::default(),
            denoise: None,
            window: VecDeque::with_capacity(WINDOW),
            plain_frames: 0,
            plain_cpu: Duration::ZERO,
            plain_gpu: Duration::ZERO,
            breakdown_frames: 0,
            breakdown_cpu: Duration::ZERO,
            totals: PassTimings::default(),
            latest: Frame::default(),
        }
    }

    /// Note that accumulation restarted: a camera move, an edit, a resize.
    /// The per-interaction marks re-arm from `origin`, and `phases` says
    /// where the time up to here went — as far as the caller has clocked
    /// it, since [`Phases::sample`] and [`Phases::other`] are not known
    /// until the mark lands.
    ///
    /// `origin` is when the *verb* happened, not when the restart did. An
    /// edit queued while the wave was mid-sample waits for the next wave
    /// boundary, and that wait is latency someone felt: origining here is
    /// what puts it inside [`Interactivity::to_first_pixel`] instead of in
    /// front of it. A caller with no verb to point at passes
    /// `Instant::now()`, which reports exactly what this measured before it
    /// took an origin.
    ///
    /// The origin and the breakdown arrive together because a breakdown
    /// adopted against the wrong restart would charge one interaction for
    /// another's cost — so a restart supersedes an in-flight breakdown by
    /// construction rather than by a rule someone has to remember.
    pub fn restart(&mut self, origin: Instant, phases: Phases) {
        self.reset = origin;
        self.awaiting_first_pixel = true;
        self.awaiting_readable = true;
        self.pending = phases;
    }

    /// The breakdown of the scene load, whose mark is
    /// [`Interactivity::to_first_ray`] and which happens once.
    pub fn attribute_load(&mut self, phases: Phases) {
        self.pending_load = Some(phases);
    }

    /// Record one finished sample.
    ///
    /// Takes the whole [`Frame`] rather than its four parts: a caller that
    /// also writes a per-frame trace then builds one value and hands the
    /// same one over, so the line in the log and the sample in the totals
    /// cannot be different frames — and there is no clone between them.
    ///
    /// A frame with no breakdown is one the timer only bracketed, which is
    /// thirty-one in thirty-two of them. That is not a failure; it is the
    /// common case, and the two populations are accounted apart for the
    /// reason in the struct's fields.
    pub fn record(&mut self, frame: Frame) {
        // The first frame to complete is the first ray traced, and startup
        // is only ever measured once. Each mark closes its own breakdown
        // against itself — the first record lands both, and they are
        // different intervals over the same sample.
        if self.interactivity.to_first_ray.is_none() {
            let mark = self.origin.elapsed();
            self.interactivity.to_first_ray = Some(mark);
            self.interactivity.load = self
                .pending_load
                .take()
                .map(|phases| close(phases, frame.cpu, mark));
        }
        if self.awaiting_first_pixel {
            let mark = self.reset.elapsed();
            self.interactivity.to_first_pixel = Some(mark);
            self.interactivity.interaction = Some(close(self.pending, frame.cpu, mark));
            self.awaiting_first_pixel = false;
        }
        // A reduced sample does not carry this mark. The asymmetry with the
        // first-pixel mark above is deliberate: a preview frame genuinely is
        // the first thing seen, so it closes that one, but sixteen samples of
        // a quarter-resolution image is not the image readability is defined
        // against — see [`READABLE_SAMPLES`]. Stamping it there would report
        // a wait a quarter of the true one, and report it as the render
        // getting faster.
        if self.awaiting_readable && !frame.preview && frame.samples >= READABLE_SAMPLES {
            self.interactivity.to_readable = Some(self.reset.elapsed());
            self.awaiting_readable = false;
        }

        if frame.passes.has_breakdown() {
            self.breakdown_frames += 1;
            self.breakdown_cpu += frame.cpu;
            self.totals.merge(&frame.passes);
            self.latest = frame;
        } else {
            self.plain_frames += 1;
            self.plain_cpu += frame.cpu;
            self.plain_gpu += frame.passes.total();
            if self.window.len() == WINDOW {
                self.window.pop_front();
            }
            self.window.push_back(frame.cpu);
            // The size and sample count are facts about the render, not
            // about the instrument, so they stay current between
            // breakdowns. The breakdown and the `cpu` beside it do not:
            // they belong to each other.
            self.latest.size = frame.size;
            self.latest.samples = frame.samples;
        }
    }

    /// Adopt a fresh memory reading. Sampled by the caller — roughly once a
    /// second is plenty, and per-frame would be a lie dressed as precision,
    /// since the allocations only move when the scene or the target does.
    pub fn memory(&mut self, memory: Memory) {
        self.memory = memory;
        self.peak_memory = Memory {
            scene: self.peak_memory.scene.max(memory.scene),
            film: self.peak_memory.film.max(memory.film),
            textures: self.peak_memory.textures.max(memory.textures),
            scratch: self.peak_memory.scratch.max(memory.scratch),
            budget: memory.budget.or(self.peak_memory.budget),
        };
    }

    /// What the last denoise of a published frame cost.
    pub fn denoise(&mut self, elapsed: Duration) {
        self.denoise = Some(elapsed);
    }

    /// A snapshot for a consumer — the overlay, the wire, the delegate.
    #[must_use]
    pub fn stats(&self) -> Stats {
        Stats {
            frame: self.latest.clone(),
            smoothed: self.smoothed(),
            interactivity: self.interactivity,
            memory: self.memory,
            denoise: self.denoise,
        }
    }

    /// The end-of-render summary.
    #[must_use]
    pub fn report(&self, device: String, scene: String) -> Report {
        // Every run-level total is a sum of the two populations rather than
        // a counter of its own: a third tally is a third thing that can
        // disagree with the other two. Both cover every sampled frame,
        // which is what lets `bound` weigh one against the other — see
        // [`Bound::of`].
        let sampling = self.plain_cpu + self.breakdown_cpu;
        let gpu = self.plain_gpu + self.totals.total();
        Report {
            device,
            scene,
            size: self.latest.size,
            samples: self.latest.samples,
            wall: self.origin.elapsed(),
            sampling,
            gpu,
            frames: self.plain_frames + self.breakdown_frames,
            breakdown_frames: self.breakdown_frames,
            passes: self.totals.clone(),
            smoothed: self.smoothed(),
            // The mean of the plain frames: what a sample costs when the
            // expensive instrument is off it. A `max(1)` rather than a
            // branch, since dividing zero frames by one is still zero.
            mean_frame: self.plain_cpu
                / u32::try_from(self.plain_frames).unwrap_or(u32::MAX).max(1),
            bound: Bound::of(sampling, gpu),
            interactivity: self.interactivity,
            peak_memory: self.peak_memory,
        }
    }

    fn smoothed(&self) -> Smoothed {
        if self.window.is_empty() {
            return Smoothed::default();
        }
        let mut sorted: Vec<Duration> = self.window.iter().copied().collect();
        sorted.sort_unstable();
        let at = |fraction: f64| {
            // An index into a window of at most WINDOW entries.
            let index = ((sorted.len() - 1) as f64 * fraction) as usize;
            sorted[index]
        };
        Smoothed {
            median: at(0.5),
            p95: at(0.95),
        }
    }
}

impl Default for Recorder {
    fn default() -> Self {
        Self::new()
    }
}

/// Durations serialize as milliseconds: a stats file is read by people as
/// often as by programs, and `1.42` beats `(secs: 0, nanos: 1420000)` for
/// both.
mod millis {
    use std::time::Duration;

    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(value: &Duration, out: S) -> Result<S::Ok, S::Error> {
        out.serialize_f64(value.as_secs_f64() * 1000.0)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(input: D) -> Result<Duration, D::Error> {
        let millis = f64::deserialize(input)?;
        Ok(Duration::from_secs_f64(millis / 1000.0))
    }

    pub mod option {
        use std::time::Duration;

        use serde::{Deserialize, Deserializer, Serializer};

        // serde's `with` contract hands the field by reference; the
        // idiomatic `Option<&T>` is not a signature it accepts.
        #[allow(clippy::ref_option, reason = "the shape serde's `with` requires")]
        pub fn serialize<S: Serializer>(
            value: &Option<Duration>,
            out: S,
        ) -> Result<S::Ok, S::Error> {
            match value {
                Some(value) => out.serialize_some(&(value.as_secs_f64() * 1000.0)),
                None => out.serialize_none(),
            }
        }

        pub fn deserialize<'de, D: Deserializer<'de>>(
            input: D,
        ) -> Result<Option<Duration>, D::Error> {
            Ok(Option::<f64>::deserialize(input)?
                .map(|millis| Duration::from_secs_f64(millis / 1000.0)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A breakdown as the timer builds one: the outer boundaries give the
    /// total, the spans attribute it.
    fn breakdown(spans: &[(&str, u64)]) -> PassTimings {
        let total = spans.iter().map(|&(_, micros)| micros).sum();
        let mut timings = PassTimings::measured(Duration::from_micros(total));
        for &(label, micros) in spans {
            timings.add(label, Duration::from_micros(micros), 1);
        }
        timings
    }

    /// One finished sample, at a size the tests below never care about, from
    /// the path tracer — which has no candidate count.
    fn sample(millis: u64, passes: PassTimings, samples: u32) -> Frame {
        Frame {
            cpu: Duration::from_millis(millis),
            passes,
            size: (8, 8),
            samples,
            preview: false,
        }
    }

    /// Repeated spans of one kernel collapse into a single entry that
    /// carries the sum and the count — the property that makes the timings
    /// survive a bounce count changing or a kernel being re-cut.
    #[test]
    fn repeated_dispatches_fold_into_one_entry() {
        let timings = breakdown(&[
            ("intersect", 100),
            ("shade_surface", 300),
            ("intersect", 50),
        ]);

        let entries: Vec<_> = timings.iter().collect();
        assert_eq!(entries.len(), 2, "two distinct kernels, two entries");
        assert_eq!(entries[0].label, "intersect", "first-run order is kept");
        assert_eq!(entries[0].gpu, Duration::from_micros(150));
        assert_eq!(entries[0].calls, 2);
        assert_eq!(timings.total(), Duration::from_micros(450));
    }

    /// A measurement with no attribution still carries its time. This is
    /// the shape thirty-one frames in thirty-two arrive in, so a consumer
    /// that reads `total` off an empty breakdown as zero would lose most of
    /// the render.
    #[test]
    fn a_total_without_a_breakdown_is_still_a_measurement() {
        let timings = PassTimings::measured(Duration::from_micros(1500));
        assert_eq!(timings.total(), Duration::from_micros(1500));
        assert!(!timings.has_breakdown());
        assert_eq!(timings.iter().count(), 0);

        // ...and merging one into a running total carries it, which is how
        // the report's device time covers every frame and not just the
        // measured few.
        let mut totals = breakdown(&[("intersect", 500)]);
        totals.merge(&timings);
        assert_eq!(totals.total(), Duration::from_millis(2));
        assert_eq!(totals.iter().count(), 1, "an unattributed total names nothing");
    }

    /// The bound verdict is the whole point of timing the passes at all: a
    /// frame whose dispatches do not account for its wall-clock is one
    /// whose cost is in launching them.
    #[test]
    fn a_frame_the_dispatches_do_not_account_for_is_cpu_bound() {
        let frame = Frame {
            cpu: Duration::from_millis(10),
            passes: breakdown(&[("intersect", 1000)]),
            size: (64, 64),
            samples: 1,
            preview: false,
        };
        assert_eq!(frame.bound(), Bound::Cpu);
        assert_eq!(frame.gpu(), Duration::from_millis(1));

        let frame = Frame {
            cpu: Duration::from_millis(10),
            passes: breakdown(&[("intersect", 9800)]),
            ..frame
        };
        assert_eq!(frame.bound(), Bound::Gpu);

        // Nothing measured is not a verdict of "CPU-bound" — it is no
        // verdict, and saying so is the difference between a stat and a
        // guess.
        let frame = Frame {
            cpu: Duration::from_millis(10),
            passes: PassTimings::default(),
            size: (64, 64),
            samples: 1,
            preview: false,
        };
        assert_eq!(frame.bound(), Bound::Unknown);
    }

    /// The marks arm off the right origins: startup is measured once and
    /// never re-armed, while the per-interaction marks re-arm on every
    /// restart.
    #[test]
    fn startup_is_measured_once_and_interaction_marks_re_arm() {
        let mut recorder = Recorder::new();
        recorder.record(sample(5, PassTimings::default(), 1));
        let first = recorder.stats().interactivity;
        assert!(first.to_first_ray.is_some());
        assert!(first.to_first_pixel.is_some());
        assert!(
            first.to_readable.is_none(),
            "one sample is not yet readable"
        );

        recorder.restart(Instant::now(), Phases::default());
        recorder.record(sample(5, PassTimings::default(), READABLE_SAMPLES));
        let second = recorder.stats().interactivity;
        assert_eq!(
            second.to_first_ray, first.to_first_ray,
            "startup is not re-measured"
        );
        assert!(second.to_readable.is_some(), "the readable mark landed");
    }

    /// The two interaction marks disagree about a reduced sample, on purpose.
    /// It is the first thing a person saw, so it closes the first-pixel mark;
    /// it is not the image readability is defined against, so no number of
    /// samples closes the other one. Without the second half, dropping
    /// resolution reads as the render reaching readable four times sooner.
    #[test]
    fn a_preview_sample_is_a_first_pixel_but_never_readable() {
        let mut recorder = Recorder::new();
        recorder.record(sample(5, PassTimings::default(), 1));
        recorder.restart(Instant::now(), Phases::default());

        recorder.record(Frame {
            preview: true,
            ..sample(5, PassTimings::default(), READABLE_SAMPLES * 4)
        });
        let marks = recorder.stats().interactivity;
        assert!(
            marks.to_first_pixel.is_some(),
            "a preview is still the first thing seen"
        );
        assert!(
            marks.to_readable.is_none(),
            "however many samples it carries"
        );

        // Still armed, so the full-resolution image that follows closes it —
        // the mark is deferred, not lost.
        recorder.record(sample(5, PassTimings::default(), READABLE_SAMPLES));
        assert!(
            recorder.stats().interactivity.to_readable.is_some(),
            "the full-resolution image closes it"
        );
    }

    /// A breakdown costs the frame it is taken on, so the frame-time
    /// distribution must come from the frames that carried none. Getting
    /// this backwards would make the renderer report a cost it only pays
    /// while being watched — the exact failure the whole module is written
    /// against.
    ///
    /// The run's *device* time is the other half of the same claim: every
    /// frame is bracketed, so it covers all eight here, and the verdict
    /// drawn from it is a verdict about the run rather than about the one
    /// frame that happened to be resolved.
    #[test]
    fn the_distribution_and_the_breakdown_come_from_different_frames() {
        let mut recorder = Recorder::new();
        // One resolved frame, deliberately the slow one...
        recorder.record(sample(10, breakdown(&[("intersect", 9500)]), 1));
        // ...and seven bracketed ones at the renderer's true cost.
        for samples in 2..=8 {
            let plain = PassTimings::measured(Duration::from_micros(5800));
            recorder.record(sample(6, plain, samples));
        }

        let stats = recorder.stats();
        assert_eq!(
            stats.smoothed.median,
            Duration::from_millis(6),
            "the distribution is the unresolved cost, not the measured one"
        );
        assert_eq!(
            stats.frame.gpu(),
            Duration::from_micros(9500),
            "the breakdown survives the frames that carried none"
        );
        assert_eq!(
            stats.frame.cpu,
            Duration::from_millis(10),
            "and keeps the CPU time of the frame it came from, so the \
             verdict compares like with like"
        );
        assert_eq!(stats.frame.bound(), Bound::Gpu);
        assert_eq!(
            stats.frame.samples, 8,
            "while the sample count stays current between measurements"
        );

        let report = recorder.report(String::new(), String::new());
        assert_eq!(report.frames, 8);
        assert_eq!(report.breakdown_frames, 1, "the sampling rate is stated");
        assert_eq!(
            report.mean_frame,
            Duration::from_millis(6),
            "the mean is over the frames that carried no breakdown"
        );
        assert_eq!(
            report.sampling,
            Duration::from_millis(52),
            "but the total wall-clock counts every frame, including what \
             measuring cost"
        );
        assert_eq!(
            report.gpu,
            Duration::from_micros(50_100),
            "and so does the device time — seven brackets plus the resolved \
             frame, not the resolved frame alone"
        );
        assert_eq!(
            report.passes.total(),
            Duration::from_micros(9500),
            "while the breakdown stays what it can answer for"
        );
        assert_eq!(report.bound, Bound::Gpu);
    }

    /// The phases account for the mark exactly, and the remainder is what
    /// makes that true: hand over a breakdown that names most of an
    /// interaction, and `other` closes the gap so the total *is* the number
    /// the mark reports. This is the property the whole struct exists for —
    /// a breakdown that only roughly added up would answer "where did the
    /// time go" with "somewhere near here".
    #[test]
    fn a_breakdown_accounts_for_the_mark_it_explains() {
        let mut recorder = Recorder::new();
        // An interaction that began 40 ms ago and named 25 ms of itself.
        let origin = Instant::now()
            .checked_sub(Duration::from_millis(40))
            .expect("the clock is not 40 ms old");
        recorder.restart(
            origin,
            Phases {
                before: Duration::from_millis(10),
                apply: Duration::from_millis(3),
                lower: Duration::from_millis(2),
                upload: Duration::from_millis(4),
                tlas: Duration::from_millis(1),
                tables: Duration::from_millis(5),
                batched: 2,
                ..Phases::default()
            },
        );
        recorder.record(sample(8, PassTimings::default(), 1));

        let interactivity = recorder.stats().interactivity;
        let mark = interactivity.to_first_pixel.expect("the mark landed");
        let phases = interactivity.interaction.expect("and carried its phases");
        assert_eq!(phases.total(), mark, "the phases are the mark, exactly");
        assert_eq!(
            phases.sample,
            Duration::from_millis(8),
            "the sample phase is the accumulate that landed the mark"
        );
        assert!(
            phases.other >= Duration::from_millis(7),
            "the unnamed remainder is real time, not rounding: {:?}",
            phases.other
        );
        assert_eq!(phases.batched, 2, "the batch count rides along");
        // The load's mark is the same sample over a different interval, and
        // its breakdown closes against that one instead.
        assert!(interactivity.load.is_none(), "no load breakdown was offered");
    }

    /// A restart supersedes the breakdown of the one it overtook. The work
    /// that one described never got its mark, and reporting it against the
    /// newer mark would charge one click for another's cost.
    #[test]
    fn an_overtaken_breakdown_is_dropped_rather_than_re_used() {
        let mut recorder = Recorder::new();
        recorder.restart(
            Instant::now(),
            Phases {
                tables: Duration::from_millis(20),
                ..Phases::default()
            },
        );
        recorder.restart(
            Instant::now(),
            Phases {
                apply: Duration::from_millis(1),
                ..Phases::default()
            },
        );
        recorder.record(sample(5, PassTimings::default(), 1));

        let phases = recorder
            .stats()
            .interactivity
            .interaction
            .expect("the mark still lands, and carries a breakdown");
        assert_eq!(
            phases.tables,
            Duration::ZERO,
            "the overtaken interaction's phases are gone, not folded into this one"
        );
        assert_eq!(phases.apply, Duration::from_millis(1), "these are the new ones");
    }

    /// Peaks are per bucket and never fall, so a report says what the run
    /// actually needed rather than what it happened to hold at the end.
    #[test]
    fn memory_peaks_are_kept_per_bucket() {
        let mut recorder = Recorder::new();
        recorder.memory(Memory {
            scene: 100,
            textures: 900,
            ..Memory::default()
        });
        recorder.memory(Memory {
            scene: 500,
            textures: 0,
            ..Memory::default()
        });
        let peak = recorder.report(String::new(), String::new()).peak_memory;
        assert_eq!(peak.scene, 500);
        assert_eq!(peak.textures, 900, "a freed bucket keeps its peak");
        assert_eq!(recorder.stats().memory.textures, 0, "live is live");
    }

    /// A report round-trips through RON, and durations read as plain
    /// milliseconds — the property that makes two runs diff as text.
    #[test]
    fn a_report_round_trips_as_readable_ron() {
        let mut recorder = Recorder::new();
        recorder.record(Frame {
            size: (128, 64),
            ..sample(5, breakdown(&[("shade_surface", 2500)]), 4)
        });
        let report = recorder.report("test device".into(), "brass-room".into());

        let text = ron::ser::to_string_pretty(&report, ron::ser::PrettyConfig::default())
            .expect("serialize");
        assert!(
            text.contains("2.5"),
            "durations serialize as milliseconds:\n{text}"
        );
        let back: Report = ron::from_str(&text).expect("deserialize");
        assert_eq!(back.scene, "brass-room");
        assert_eq!(back.passes.total(), Duration::from_micros(2500));
    }
}
