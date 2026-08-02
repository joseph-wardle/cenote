# Cenote — M7 Implementation Plan: Lean & Interactive

*Decisions locked 2026-08-01 via structured interview, after an honest look at where the
renderer sits: correct, well-tested, and **slower with ReSTIR on than off**. The
milestone's premise is the user's own: "a cool renderer, but needlessly verbose, poorly
architected, and correct but slow." M7 is the rearchitecture that makes the interactivity
claim true — and it opens with measurement, because everything after it is a bet that
needs a scoreboard. Everything consciously not built lives in
[deferrals.md](deferrals.md) with its revival trigger.*

Three framing notes, because they govern every choice below:

- **The gap is software, not hardware.** The target paper shows scenes readable in 12 ms
  where cenote takes seconds on a 4070 Ti SUPER — a card in the same family as the
  paper's. The cost is structural: one ReSTIR sample does 16–22× a plain sample's ray
  work (M=16 candidates, 5 spatial shifts, a temporal shift), hardcoded, unscaled by
  motion, and paid in full on the latency-critical first frame after a camera move.
- **The renderer is split-brained.** Plain PT runs through wavefront queues; ReSTIR
  traces inline with `rayQuery`, megakernel-style. Two loop shapes, one renderer.
- **Measure, then cut.** The appetite is a real rearchitecture plus aggressive debloat,
  but every rung after 7-0 is gated on a number 7-0 produced. That is what 7-0 is for.

---

## 1. Decisions locked in this session

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 1 | Metric | **Both** cheap-while-moving and fast-to-settle | The interactive claim is two claims; optimizing one alone is how the current state happened |
| 2 | Spine | **One always-on, motion-scaled ReSTIR path.** Plain PT is demoted to CI oracle and reference | Two first-class estimators is the split brain. The oracle role is the one PT must keep (D-090) |
| 3 | Moving preview | Fast GPU denoiser (NRD leads; research spike is 7b). OIDN stays for stills. **The final frame must converge without a denoiser** | A denoiser is how every production renderer makes motion readable — but it must never be load-bearing for correctness |
| 4 | Candidate count | `M` becomes a **tuned, motion-scaled parameter** (1–2 moving, 8+ still), not `const 16` | The single largest unnecessary cost, and the one most directly against the interactivity goal |
| 5 | Appetite | Rearchitect the loop, debloat hard — **gated on measurement first** | The user's call, stated plainly; the gate is what keeps it from being a rewrite on a hunch |
| 6 | Loop shape | **Measured kernel fission.** Kill the RAY/HIT/MISS/SHADOW queues; split ReSTIR at register-pressure and divergence seams | One loop shape, cut where the measurement says, not where the taxonomy says |
| 7 | Fission timing | Now, with clean stage boundaries — those seams are what volumes and SSS would need later | Doing it once, with the seams in the right places, beats doing it twice |
| 8 | Debloat | Strip the 373 D-number code cross-refs; tighten the `pub` surface; collapse the 30k-line `docs/` to a slim ARCHITECTURE.md plus a history archive — **after** extracting the load-bearing math (pairwise MIS, shift identities) | The decision log is the history; the code should read as code |
| 9 | Success bar | ≤33 ms moving at 1080p; lookdev-readable under 1 s after the camera stops; converged within relMSE of the PT reference **without a denoiser** | Three numbers, each falsifiable |
| 10 | Sequencing | Cheap wins first | Each rung's result informs the next |
| 11 | Hydra/USD | Delegate stays first-class; treated as a fixed API above the loop, goldens stay green | The delegate is shipped work and must not be collateral |

## 2. Build order

| Rung | Work | Gate |
|---|---|---|
| **7-0** | **Measurement spine + pre-refactor baseline + relax bit-exact goldens to relMSE/FLIP** | The scoreboard exists, and the instrument's own cost is measured rather than assumed |
| 7a | `M` → 1 with a motion ramp; measure | Moving frame time drops without the still tail regressing |
| 7b | Denoiser spike (NRD vs alternatives), then integrate | Moving preview is readable |
| 7c | Re-measure against 33 ms / <1 s | **Decision gate:** is fission still a *performance* need, or now only quality/debloat/future-work? |
| 7d | Kernel fission; kill the queues | Measured win, seams clean |
| 7e | Debloat sweep | Surface area down, docs collapsed, math preserved |

---

## 3. Step 7-0 plan — the measurement spine (interviewed 2026-08-01)

Ten decisions, walked one at a time. The through-line: **a statistic that changes what it
measures is worse than no statistic**, so every entry had to earn its place against that
test — and one of them failed it on measurement and was redesigned.

### 3.1 The ten decisions

1. **Two tiers.** An always-on tier cheap enough to trust, and an opt-in deep tier that
   accepts overhead and says so. Mirrors Karma's `--stats`, RenderMan's `-statslevel`,
   MoonRay's log verbosity.
2. **Per-dispatch granularity**, keyed to a pass-ID registry rather than hardcoded names,
   so the timers survive kernel fission re-splitting them. **Plus a hard deliverable: the
   instrument benchmarks itself.** If it is not near-free, this decision reopens.
3. **Cheap interactivity proxies**, tier one: time-to-first-ray (once per scene load, not
   a per-frame number), time-to-first-pixel per interaction, time-to-N-spp as the
   readability stand-in. All free host `Instant` deltas. True variance-based
   time-to-converged is tier two.
4. **Memory in five named buckets** — scene, film, reservoirs, textures, scratch — plus
   total and headroom. **Sampled, not per-frame.** The reservoir bucket prices reuse; the
   scratch bucket is a debloat receipt that should collapse when the queues die.
5. **No OS CPU%.** Instead a derived **CPU-bound vs GPU-bound verdict** from the frame's
   wall-clock against its summed GPU dispatch time, with both raw numbers shown so the
   size of the gap is visible, not just its verdict.
6. **Tier two is three families:** actual ray counts by type; per-pixel variance
   (true time-to-converged, and the ReSTIR-on-vs-off convergence-rate comparison that
   proves reuse earns its per-sample cost); and reuse quality — spatial acceptance rate,
   temporal survival fraction, mean confidence.
7. **One canonical `Stats` struct**, flowing to the viewer overlay, the headless CLI, and
   the Hydra delegate. It rides **beside** the pixels, never inside them, so the
   byte-exact framebuffer transport goldens stay green.
8. **Summary at render end by default** (human console block + machine sidecar from the
   same struct); **per-frame time-series opt-in** for settle curves. The overlay's frame
   time is a **rolling median with p95 beside it**; raw values live in the trace.
9. **Tier-two counters gated at compile time**, so the shipping kernel is byte-identical
   to a stats-free build — rather than a runtime uniform whose atomics would still
   inflate the register footprint and change the occupancy being measured.
10. **A fixed 3–4 scene benchmark set**, ReSTIR on and off, captured **pre-refactor**, and
    re-run at every rung as the M7 scoreboard. The 19-scene research corpus stays reserved
    for the final campaign, not the per-rung loop.

### 3.2 Explicit exclusions, and why

- **Occupancy, register pressure, cache throughput.** Not portably measurable from
  Vulkan. Re-deriving them would mean rebuilding a fraction of Nsight Graphics or the
  Radeon GPU Profiler, badly — which is precisely the bloat this milestone deletes. When
  the fission decision needs those numbers, attach the real tool.
- **OS CPU% and per-core load.** Noise plus syscall cost; the bound verdict answers the
  question it was standing in for.
- **Per-allocation memory detail.** Five buckets is the ceiling; a memory profiler is for
  the rest.

### 3.3 What was built

| Piece | Where |
|---|---|
| GPU pass timing (query pool, stamps, readback, cadence) | `crates/cenote/src/gpu/timing.rs` |
| Memory ledger, bucketed off allocation names | `crates/cenote/src/gpu/ledger.rs` |
| The canonical `Stats`/`Report` and the `Recorder` behind them | `crates/cenote/src/stats.rs` |
| Overlay block (four fixed lines + three collapsing sections) | `crates/cenote-viewer/src/ui.rs` |
| Console report, RON sidecar, `--stats-trace` | `crates/cenote-cli/src/main.rs` |

**The pass-ID registry is the kernel list itself.** Every pipeline is created from a
`Kernel` whose entry-point name is already unique and already `'static`, so
`ComputePipeline` simply carries it and `Pass::label()` reads it back. There is no table
to keep in sync: split a kernel in two and both halves appear under their own names the
moment they exist. That is decision 2's requirement met with no registry to maintain.

**The memory bucket is the allocation's name.** Every allocation already carried a dotted
name for the allocator's leak reports (`scene.geometry`, `film.albedo.sum`,
`restir.reservoir.curr`, `wavefront.queue.ray`); the ledger reads the bucket off whole
dotted segments. Two renames were needed to make the buckets mean what they say —
mesh buffers moved under `scene.mesh.*` (they were named by mesh and falling to scratch),
and the ReSTIR **light tables** moved to `scene.restir.*`, because the reservoir bucket's
whole job is to price *reuse*, not the light list the estimator happens to read.

### 3.4 The self-benchmark — and the decision it reopened

Decision 2 carried a condition: prove it is near-free, or reopen. **It was not free.**

The hypothesis was that timestamps would ride for nothing, because `submit_passes`
already places a full memory barrier between every pass — so the clock is read where the
GPU is provably quiet. The measurement disagreed, and kept disagreeing under a second,
more careful pass. Brass-room at 512² with the plain path tracer — 71 passes over a
2.3 ms frame, this renderer's worst case — eight interleaved 800-sample runs per arm:

| What is stamped | vs `--no-gpu-timers` |
|---|---:|
| Every pass boundary, every frame | **+49.9%** |
| Every pass boundary, one frame in eight *(the first design)* | **+7.8%** |
| Every **span** boundary, one frame in 32, plus the two outer stamps every frame *(shipped)* | **+1.9%** |
| The two outer stamps alone, every frame | **±0%**, inside the noise |

The last row is the finding the design rests on. An **interior** stamp costs ~15 µs — it
is a real serialization point, and the barrier in front of it evidently is not — while
the two that **bracket** a submission cost nothing measurable, because the GPU is already
draining at both ends. *Where* the time went is expensive; *how much* there was is free.

So the two questions are asked separately, at their own prices:

- **every** submission is bracketed, which is what lets the report weigh device time
  against wall-clock over the whole run rather than over a sample of it;
- the **per-kernel breakdown** runs one frame in 32, and is stamped per *span* — a maximal
  run of consecutive passes sharing a label — rather than per pass. `PassTimings` folds by
  label anyway, so a stamp inside a span buys a number the fold sums straight back
  together. The wave opens every bounce with three queue-clearing fills and the submission
  with five, so 71 passes are 52 spans: **24% fewer stamps, no information lost**.

And the `Recorder` keeps the two populations apart on purpose:

- the **frame-time distribution** (median, p95, mean) comes from the frames that carried
  no breakdown, so the number a person reads is what the renderer costs when nobody is
  looking;
- the **kernel breakdown** comes from the frames that carried one, where a GPU time and a
  CPU time can honestly be compared because they are the same frame.

Two hypotheses died on the way, both worth not re-testing. Resetting the query pool from
the host (`hostQueryReset`) rather than in the command buffer changed nothing, so the
reset is not the cost. Neither did the stamp's stage mask, in any of its three spellings.
The mask is now `ALL_COMMANDS` for a different reason: `vkCmdWriteTimestamp2` takes a
*single* stage (VUID-…-stage-03859), so the `COMPUTE_SHADER | ALL_TRANSFER` the first pass
shipped was a validation error whose behaviour the spec does not define.

`--no-gpu-timers` is the A/B arm, kept in the CLI so the claim stays re-checkable rather
than remembered — and it needs re-checking rather than quoting. The same `off` arm that
reads 1.85 s above read 1.38 s in the session that captured `m7-baseline/`, on the same
machine and the same command. Absolute figures travel badly between sittings; only
interleaved arms compare.

### 3.5 The benchmark set

Five scenes, each earning its slot on one of the two axes the user named — representative
of a production workflow, or a probe of a specific renderer property:

| Scene | Axis | What it stresses |
|---|---|---|
| `corpus/cornell-box.ron` | property | The cheapest possible frame — almost all cost is fixed overhead, so it is the **dispatch/launch-bound probe**. This is where a wave that launches more kernels than it runs shows up, and where M going to 1 must not be swamped by submission cost |
| `brass-room.ron` | property + workflow | **Heavy indirect GI** — the M6 pin scene, so the M7 numbers connect to the existing record. The ReSTIR-PT case |
| `many-lights.ron` | property + workflow | **Many lights / NEE-bound** — the ReSTIR-DI case, and the stage-lighting workflow |
| `corpus/bistro.ron` | workflow | **Production exterior at scale** — the heaviest asset load in the corpus. Probes scene memory, texture memory, TLAS build, and time-to-first-ray, which nothing else here does |
| `corpus/zero-day.ron` | workflow | **Production interior at scale, 283 area lights** — many-lights *and* heavy geometry at once, where `many-lights.ron` isolates the first and `bistro.ron` the second. Added 2026-08-02 because it is the scene the interactivity judgement is actually being made on, and a scoreboard that omits it measures something nobody is looking at |

Each captured ReSTIR-on and ReSTIR-off as RON sidecars, so a rung of work is a diff rather
than a remembered impression. `docs/m7-baseline.sh <dir>` captures the set; the pinned
pre-refactor capture is in [`m7-baseline/`](m7-baseline/).

**Captured at 1440p since 2026-08-02** (`WIDTH=1920 HEIGHT=1080` reproduces the old basis).
The pinned §3.5.1 table is 1080p and stays that way — 1440p is 1.78× the pixels, so the two
do not compare directly. Decision 9's 33 ms bar is stated at 1080p; whether to restate it at
the resolution the renderer is judged at is open, and belongs to 7c.

**Four arms per scene since the same date**, the fourth being `restir-moving-nospatial`.
The moving breakdown (§3.5.2) put `restir_spatial_gather` level with `restir_candidates`,
so the spatial pass needs its own price on the frame where it costs the most — before
anyone builds the motion-scaled `k` that §4.2 deferred.

### 3.5.1 The baseline, captured 2026-08-01 at 1080p, 64 spp

Pre-refactor, on the current code, before a line of the loop changed. RTX 4070 Ti SUPER,
NVIDIA 580.173.

| Scene | Mode | mean ms | p95 ms | in dispatches | 1st ray | 16 spp | VRAM | reservoirs | top kernel |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| cornell-box | PT | 11.8 | 13.9 | 67% | 0.7 s | 0.9 s | 602 MiB | — | `shade_surface` 0.02 ms/call |
| cornell-box | ReSTIR | 27.3 | 30.6 | 80% | 0.7 s | 1.2 s | 1551 MiB | 949 MiB | `restir_candidates` 5.1 ms/call |
| brass-room | PT | 9.4 | 11.5 | 88% | 0.6 s | 0.8 s | 602 MiB | — | `shade_surface` 0.27 ms/call |
| brass-room | ReSTIR | 44.9 | 50.1 | 89% | 0.7 s | 1.4 s | 1552 MiB | 949 MiB | `restir_candidates` 8.3 ms/call |
| many-lights | PT | 4.7 | 5.9 | 74% | 0.7 s | 0.8 s | 603 MiB | — | `shade_surface` 0.08 ms/call |
| many-lights | ReSTIR | 23.5 | 27.8 | 78% | 0.6 s | 1.0 s | 1552 MiB | 949 MiB | `restir_candidates` 4.1 ms/call |
| bistro | PT | 19.3 | 21.8 | 90% | 13.7 s | 14.0 s | 2016 MiB | — | `shade_surface` 0.78 ms/call |
| bistro | ReSTIR | 78.4 | 87.6 | **94%** | 14.9 s | 16.2 s | 2965 MiB | 949 MiB | `restir_candidates` 18.5 ms/call |

"In dispatches" is summed GPU time over the wall-clock of the same frames — the bound
verdict with its working shown. Only bistro-with-ReSTIR clears the 90% line.

**What it says, and what M7 does about it.**

- **Reuse costs 2.3–5.0× per frame, on every scene.** The premise of the milestone, now
  a number rather than an impression. Nothing clears the 33 ms bar with ReSTIR on except
  cornell-box, which is the trivial scene.
- **`restir_candidates` is the top kernel in every single ReSTIR run** — 4.1 to 18.5 ms
  per call, ahead of the spatial pass, ahead of everything. That is decision 4's
  `const M = 16` made visible, and it makes **7a the right next rung**: it is not a guess
  that M is the problem, it is the measurement.
- **Seven of eight runs are CPU-bound**, most of them by a wide margin — a third of the
  cornell-box frame is not in a dispatch at all. That gap is recording, submission, and
  the host half of the loop: the launch-cost signature decision 6's kernel fission has to
  answer for. Only bistro-with-ReSTIR — the heaviest scene and the slowest frame — keeps
  the GPU busy enough to count as GPU-bound.
- **Reuse costs a flat 949 MiB at 1080p** regardless of scene — the reservoir copies and
  G-buffers, ~3.8× the whole film. Worth knowing before 7d moves any of it.
- **bistro takes ~14 seconds to its first ray.** No rung above targets that yet; it is now
  on the board.

The capture also found a bug in the instrument on its first outing: cornell-box asks for
65 bounces, which records ~1000 passes at 1080p, past the timer's starting pool — so its
path-tracer run came back unmeasured. The pool now grows to fit rather than declining to
measure, because a pass count is a property of the scene and the target, and any fixed
ceiling is a guess some scene will falsify.

### 3.5.2 The 1440p capture, and the correction it forces (2026-08-02)

Captured on a quiet desktop, five scenes × four arms, at 2560×1440 / 64 spp — the
resolution the interactivity call is actually made at. Sidecars in
[`m7-baseline-1440/`](m7-baseline-1440/); the 1080p set in `m7-baseline/` is left alone.

| Scene | PT | ReSTIR still | ReSTIR moving | moving, no spatial |
|---|---:|---:|---:|---:|
| many-lights | 7.9 | 40.6 | 35.5 | 18.5 |
| brass-room | 15.5 | 80.4 | 77.2 | 40.8 |
| cornell-box | 17.9 | 46.3 | 45.7 | 27.2 |
| zero-day | **26.9** | 100.5 | **126.5** | **60.5** |
| bistro | 30.6 | 148.1 | 145.5 | 82.0 |

Mean ms/frame. The moving arms are M = 1 (7a live); the still arms settle to M = 16.

**zero-day, moving, device time per frame** — the interactive case on the scene the call is
made on:

| kernel | ms/frame | share |
|---|---:|---:|
| `restir_spatial_gather` | 50.87 | 46% |
| `restir_candidates` | 28.39 | 26% |
| `restir_temporal` | 16.04 | 14% |
| `restir_spatial` (combine) | 10.48 | 9% |
| everything else | 4.97 | 4% |
| **total in dispatches** | **110.75** | of a 126.5 ms frame (88%) |

Path-traced, the same frame is 25.9 ms in dispatches of a 26.9 ms wall (96%).

**Five corrections to §3.5.1, all of them load-bearing.**

- **The top kernel is the spatial gather, not the candidates.** §3.5.1 read
  *"`restir_candidates` is the top kernel in every single ReSTIR run — that is decision 4's
  `const M = 16` made visible"*, and 7a was sequenced on that sentence. At 1440p the gather
  is 1.8× the candidate kernel on zero-day and 0.9× on bistro; the spatial pass as a whole
  (gather + combine) is **55% of a moving frame**, which the no-spatial arm confirms
  independently at the wall-clock (126.5 → 60.5).
- **M is not what makes the candidate kernel expensive.** Still M=16 → moving M=1 takes it
  from 32.4 to 28.4 ms — **12%**, or 4 ms of a 111 ms frame. The rest is the inline indirect
  tail (`traceTail`, D-134), one full path per pixel at bounce 0, which M does not scale.
  That is the arithmetic behind D-155's measured 5.9–13.4%: 7a's win was real, correctly
  measured, and aimed at the third-largest term in the frame.
- **Reuse costs *more* while moving than settled** — 110.8 ms against 93.0 ms on zero-day,
  despite M being 16× lower. The decay ramp anneals history to nothing on a held camera, so
  `restir_temporal` falls 16.0 → 2.6 ms and the gather's shifts stop landing. **The
  estimator is at its most expensive exactly when the frame budget is tightest**, and every
  still measurement of temporal reuse is a measurement of the annealed case.
- **The frame is GPU-bound at 1440p** — 88% of a moving ReSTIR frame is inside a dispatch,
  96% path-traced. §3.5.1's *"seven of eight runs are CPU-bound, that gap is the launch-cost
  signature kernel fission has to answer for"* was 1080p plus a contended capture. At the
  resolution that matters there is ~12% of the frame outside dispatches, so **7d can no
  longer be justified as a frame-time rung** — its case is now divergence, register
  pressure, and the volumes/SSS seams, which is a different argument at a different price.
- **A 1440p frame is four dispatches per kernel, not one.** `DEFAULT_CAPACITY` is 1 << 20
  paths against 3.69 M pixels, so the wave chunks the frame — invisible at 1080p (two
  chunks) and never noticed. Harmless in itself, but it means launch count scales with
  resolution, and any per-frame reasoning that assumed one dispatch per stage was wrong.

**What this changes.** 7b reduces no frame time and 7c is a gate, so neither is the next
rung. Ranked by expected win against the 126.5 ms:

1. **`k` at the restart frame**, the twin §4.2 deferred — same machinery as 7a, aimed at the
   55% the spatial pass owns. If the gather is near-linear in `k`, 5 → 1 is worth ~40 ms and
   the combine another ~8: **126 → ~78 ms**. Whether it is linear is the first thing to
   measure, and the no-spatial arm now brackets it.
2. **The bounce-0 indirect tail** — 28 ms that M cannot touch. Capping tail depth in motion,
   or reusing the tail rather than re-tracing it.
3. **Reservoir footprint** — 480 B/pixel, 1.7 GiB at 1440p, `k+1` read per pixel by the
   gather. The target paper's 64 B reservoir cuts the traffic `k` also cuts, so it is second
   in line behind it rather than an independent win.

**The falsifiable exit.** PT is 26.9 ms on this scene and reuse is 4.7×. Levers 1 and 2
together project to ~60 ms — at the 33 ms bar restated for 1440p, and no better. If they
land and reuse is still >1.5× PT in motion, decision 2 (one always-on ReSTIR path, PT
demoted to oracle) is what the measurement is falsifying, not the tuning.

### 3.5.3 The equal-wall-clock measurement, and the exit it triggers (2026-08-02)

§3.5.2 left a falsifiable exit against decision 2. This is the measurement that settles it,
and it settles it against reuse harder than the projection there expected.

**Why it did not already exist.** `assert_reuse_gate` compares the two estimators at a
matched *sample* count. That is the correct protocol for the claim it makes — resampling
draws a better sample than one starved next-event draw — and it is silent on whether a
ReSTIR sample is worth what it costs. Every claim M3–M6 recorded about reuse "winning" is
a claim in that units. Nothing in the suite had ever divided by time.

**The protocol** (`restir_equal_time_efficiency_report`, beside the gates it corrects).
Both estimators, wall-clock timed over the sample loop only, at 1024² — on the two scenes
the gates assert the win on, so these are ReSTIR's best cases. Budgets overlap in *time*
rather than samples. References are cross-referenced exactly as the floor report:
ReSTIR against a deep brute reference, brute against a deep ReSTIR one, because
accumulation replays one sample sequence and an estimate sharing frames with a reference
from its own estimator measures its error against its own noise and reads low. Shipping
defaults on both sides, temporal reuse included. The verdict metric is Monte Carlo
efficiency, 1/(relMSE × s) — flat down a 1/N ladder, so it compares two ladders that never
land on the same second.

**many-lights** — 4.3 ms/sample brute, 22.3 ReSTIR (**5.2×**):

| budget | brute relMSE | ReSTIR relMSE | matched-sample win | brute eff | ReSTIR eff | **eff ratio** |
|---|---|---|---|---|---|---|
| 16 | 0.01987 | 0.00976 | 2.04× | 710 | 268 | **0.38** |
| 32 | 0.00937 | 0.00459 | 2.04× | 768 | 297 | **0.39** |
| 64 | 0.00393 | 0.00228 | 1.72× | 902 | 310 | **0.34** |

**indirect-glossy** — 3.8 ms/sample brute, 23.8 ReSTIR (**6.2×**):

| budget | brute relMSE | ReSTIR relMSE | matched-sample win | brute eff | ReSTIR eff | **eff ratio** |
|---|---|---|---|---|---|---|
| 16 | 0.06178 | 0.04460 | 1.39× | 254 | 59 | **0.23** |
| 32 | 0.02761 | 0.02116 | 1.30× | 293 | 61 | **0.21** |
| 64 | 0.01225 | 0.01037 | 1.18× | 328 | 64 | **0.20** |

**The verdict. Path tracing is 2.6× more efficient per second than ReSTIR on many-lights
and ~4.8× on indirect-glossy** — the starved many-light regime and the hard-GI regime, the
two the method exists for. The matched-sample win is real and reproduces the gate's
~1.7–2×; it is simply not large enough to cover a 5–6× price. Concretely, on many-lights
brute reaches relMSE 0.00089 in 1.09 s where ReSTIR reaches 0.00228 in 1.41 s.

Read the headline line as a *floor* on the gap. Brute's deepest points are where its
relMSE × N stops falling and starts climbing — it has run down to the reference's own
residual and is being charged error that belongs to the reference, while ReSTIR at a sixth
the sample count is nowhere near that floor. The shallow end, where both are still on 1/N,
is the clean read, and it says the same thing.

**Why nobody caught it: the suite is blind to cost by construction.** At 128² the same two
estimators cost **1.67 and 1.52 ms/sample — a 1.1× ratio**. Sixteen thousand pixels is not
pixel-bound, so ReSTIR's extra dispatches vanish into per-dispatch overhead. Every
convergence test in the repo runs there. That was the right call for testing convergence —
and it means reuse's price has never once been in frame. The megapixel ratio (5.2×) matches
the 1440p scoreboard's independent 5.1× on this scene, so the two measurements corroborate
each other across resolution, scene definition, and harness.

**The exit fires.** §3.5.2's condition was *"if levers 1 and 2 land and reuse is still >1.5×
PT in motion, decision 2 is what the measurement is falsifying."* That condition is now
strictly weaker than what has been measured: even at the projected ~60 ms floor, reuse
would need its efficiency ratio above 1.0, and to get there from 0.34 the *variance* side
would have to improve ~3×, which no rung on the ladder targets. **Decision 2 — one
always-on motion-scaled ReSTIR path, PT demoted to CI oracle — is falsified.** It was
locked before any equal-time number existed. PT is the interactive spine; ReSTIR is an
opt-in estimator, kept for what it is good at and for the M3–M6 validation it carries.

**One thing this does not settle, flagged rather than acted on.** Under the cross-referenced
protocol at 128², indirect-glossy shows ReSTIR at *no* matched-sample advantage (16 spp
0.0724 vs brute's 0.0694), where the gate — same resolution, shared ReSTIR reference —
measures 1.65×. Two candidate causes: the shared reference deflating ReSTIR's error, or
cross-referencing inflating it through a mean disagreement between the estimators. Neither
has been separated, and neither bears on the verdict above (which holds at every budget on
both scenes under either protocol). It does bear on how large the gate's headline margin
really is, and is worth its own measurement before that number is quoted again.

### 3.6 Definition of done for 7-0

- [x] Tier-one timing, interactivity marks, and memory buckets, on one canonical struct
- [x] Overlay, console report, and machine sidecar all rendered from that one struct
- [x] Stats ride beside the pixels; framebuffer goldens untouched
- [ ] **The delegate's share of decision 7.** `Frame::stats()` is available to the server,
      but the shared-memory header the C++ delegate reads is a fixed slot layout, so
      carrying `Stats` across it is an ABI extension with a drift guard on the far side
      (M4 step 1's shape). Deliberately not half-done here: the canonical struct exists
      and two of three consumers read it, and the third is a scoped piece of work rather
      than a missing design
- [x] The instrument benchmarks itself, and the result is recorded here **including the
      part where the first design failed**
- [x] Baseline captured on the four-scene set, ReSTIR on and off (§3.5.1, pinned in
      `m7-baseline/`)
- [ ] Bit-exact goldens relaxed to relMSE/FLIP thresholds (so fission's float-reduction
      reordering cannot false-fail)
- [ ] Tier two — ray counts, variance/convergence, reuse quality

---

## 4. Step 7a plan — the candidate ramp (interviewed 2026-08-01)

> **Shipped 2026-08-02 as D-155, with one decision overturned by measurement.** Decision 3's
> *ramp* — M climbing to the still count over the temporal decay window — cost 43% more
> error at 8 spp and erased `ReSTIR`'s whole margin over brute force on the many-lights
> reuse gate. What shipped is a **step**: the restart frame alone is cheap, and only when the
> history is warm. §4.0 below is why that keeps the entire moving win; D-155 is why the
> shape had to go, and carries the measured table. §4.3's golden migration proved
> unnecessary — every golden passed unchanged. Read §4.1 as the interview it was, not as a
> description of the code.

Eleven decisions, walked one at a time. The through-line: **7a moves exactly one number,
so that if anything regresses there is only one suspect.** Every decision below that
looked like a free improvement — lower the still ceiling, ramp `k` too, skip the golden
re-bake — was refused on that ground.

### 4.0 The decomposition the rest of the plan rests on

While the camera moves continuously the film resets every frame, so `film.samples()` is
**0 on every frame**. Therefore `M(0)` alone decides the moving cost, and the ramp's
*shape* is invisible to the moving claim — it only shapes the settle. 7a's two-part gate
splits along that line:

- the **floor** answers *"moving frame time drops"*;
- the **shape** answers *"the still tail doesn't regress"*.

They tune independently, and each has its own measurement. This decomposition is the one
thing in §4 that held up completely: because the floor alone carries the moving claim, the
shape could be deleted outright when it failed its own half — at no cost to the win.

### 4.1 The eleven decisions

1. **The ramp reads samples-since-reset.** Not an explicit `moving` flag (two code paths,
   and the CLI has no motion), and emphatically not a frame-time controller — that would
   make the image a function of background load, which D-152 already showed can double a
   number mid-session. Every restart path — resize, edit, camera adopt, both reuse
   toggles — funnels through `film.reset()` (`session.rs:710–773`), so one signal already
   covers all of them. It cannot distinguish a camera move from an edit; all three want a
   cheap first frame, so that is a feature.
2. **The ramp evaluates on the host.** The kernel is untouched and its SPIR-V stays
   byte-identical, so if the image moves, the ramp moved it — there is no second suspect.
   It also makes the knob a pure function with a unit test, and lets the resolved M be
   reported (decision 8), which the shader could not do.
3. **Linear, over `RESTIR_TEMPORAL_DECAY_FRAMES`** — sharing the temporal window, not
   merely reusing its number. The decay is `saturate(1 − s/decayFrames)`: at `s = 0`
   history is fully live, at `s ≥ 16` it is *exactly* dead. So the window over which
   history stops supplying confidence is precisely the window over which fresh candidates
   must start supplying it. A rising M against a falling decay keeps a pixel's total
   confidence roughly flat across the handoff — where today M is pinned at 16 through a
   regime in which history is already doing the work, which is the waste §3.5.1 measured.
   One constant, not two, so the two ramps cannot drift.
4. **Floor 1, ceiling 16.** The ceiling is *unchanged*, which buys a falsifiable property:
   at `s ≥ 16` every push constant is bit-identical to today, so **the still tail cannot
   regress by construction** and a regression there can only have come from the ramp. Is
   16 the right still M? Probably not — averaging may be paying for per-sample RIS twice —
   but that is a separate experiment (efficiency = variance × cost at fixed wall-clock),
   and folding it in costs exactly the property above. The floor is 1, not 2, because the
   light-sampling technique covers the support of *f* alone at any M ≥ 1 (what
   `_SUPPORT_COVERAGE` pins) and the internalized BSDF candidate rides alongside
   regardless — so M=1 is still a two-technique estimator. If it proves unacceptable, that
   is a finding about where reuse's floor sits, and 2 is one character away.
5. **The ramp is applied at the `Renderer`, not inside `Wavefront`.** `RestirInputs` gains
   a `candidates: u32` field the caller sets — the shape `decay_frames` already has, where
   the pinned-temporal gate passes `0` (`wavefront.rs:3186`). The 13 test construction
   sites each pin `RESTIR_CANDIDATES` and stay **bit-identical**. Had the ramp lived
   in `restir_wave_params`, two *timing* tests (`restir_frame_time_report`, and the §4e
   stage-pricing harness at `wavefront.rs:3162`, both looping `sample` 0..71 and averaging
   wall-clock) would have silently reported a cheaper number for reasons unrelated to what
   they measure.
6. **`--restart-every-sample` is the moving harness.** One `film.reset()` in the CLI loop.
   `Film::reset()` is literally `self.samples = 0` (`film.rs:321`) — the reservoirs
   deliberately survive, because that *is* the warm start — so a static camera with a reset
   every frame reproduces the moving regime's estimator state exactly: M at the floor,
   history live, decay at 1. Where it is wrong it is wrong in the safe direction for the
   claim it makes: reprojection is the identity and the disocclusion gate always passes,
   and a real disocclusion *drops* history before any shift work (§4c decision 3), so real
   motion is **cheaper** than this harness reports. It is an upper bound on moving frame
   time — and, for the same reason, **it proves the time claim and is structurally blind
   to the quality one.** That blindness is why decision 11 exists.
7. **The goldens are regenerated in the shipping default, behind a migration proof.**
   `ACCUM_SPP = 32`, so half of every ReSTIR golden's samples fall inside the window. Four
   of seven are affected; `restir-demo-survivor` and `restir-many-lights-survivor` go
   through `compare_bit_identical` and *will* fail. Pinning the ramp off in the golden
   tests would keep the diff at zero and quietly retire four goldens from their only job —
   pinning what ships. The procedure is in §4.3.
8. **The resolved M rides in `Stats`.** `Frame.candidates: Option<u32>` — `None` in
   path-tracer mode. It is derivable from `samples`, but only by a reader who knows both
   the ramp's shape and whether it was enabled; **`candidates` at sample 0 names which arm
   produced the sidecar** (1 = ramp on, 16 = pinned). That is the distinction §3.4 got
   burned by, where two sittings' `off` arms read 1.38 s and 1.85 s and the numbers alone
   could not say why. It also makes the kernel breakdown actionable: `restir_candidates
   8.3 ms/call` now varies frame to frame, and a figure without its M is one nobody can act
   on.
9. **One GPU wiring test, no timing assertions.** The arithmetic, the toggle, and the
   stats field are pure unit tests; the Renderer's wiring needs one accumulated sample,
   ramp on versus off, asserting the images *differ* — an exact assertion with no tolerance
   to age badly. A frame-time assertion in CI was rejected outright: D-152.
10. **The scoreboard gains a permanent moving column; the ramp A/B is 7a's own
    experiment.** The moving arm belongs in `m7-baseline.sh` on its own merits — decision
    9's bar is stated *for motion* and the scoreboard has only ever measured stills — and
    every later rung has a moving story it should not have to add the column for. ReSTIR
    only: a path-traced sample's cost is independent of the sample index, so a PT moving
    arm would reproduce the PT still column to within noise. The ramp A/B is interleaved,
    one-time, and recorded here rather than pinned into every future capture.
11. **The viewer gets a toggle and an overlay line.** `--no-cv` is CLI-only, and that
    precedent *supports* this rather than opposing it: CV's claim is about chroma noise in
    a converged still, which is exactly what the CLI produces. The ramp's claim cannot be
    judged from a still at all (decision 6). The only instrument that can answer "is M=1
    while orbiting acceptable?" is a person orbiting with a checkbox. It follows the
    temporal toggle exactly, `film.reset()` on change included, so a flip never blends two
    policies into one image.

### 4.2 Explicit exclusions, and why

- **`k` stays at 5.** The spatial neighbours are the other half of the 16–22× ray budget
  and the obvious "while we're here." Same argument as the ceiling: move two knobs and a
  regression has two suspects. It deserves its own rung and its own number.
- **The still ceiling stays 16.** See decision 4 — retuning it is a real experiment, just
  not this one.
- **No frame-time feedback controller.** Decision 1.

### 4.3 The golden migration

The invariant `restir_*-survivor` pins carry is a **relation** — cv-off must equal
survivor shading — not an absolute image, so both sides moving together leaves it intact.
What moves is the bytes it is anchored to, and the off switch lets that be *proved* rather
than asserted:

1. With the ramp **off**, run the full golden suite. All four must be green,
   `compare_bit_identical` included. **This is the proof that 7a changed nothing except
   M** — anything red here means the ramp is not the only thing that moved, and the rung
   stops.
2. Only then, `UPDATE_GOLDENS=1` with the ramp **on**; eyeball all four in `tev`.
3. Re-copy the two survivor pins as byte copies of the fresh `--no-cv` renders, restoring
   the invariant's construction.
4. Record in the D-number that the pins were re-anchored at 7a **and that step 1 was
   green**, so the lineage back to 6a is documented rather than lost.

Still to verify before implementing: whether `crates/cenote-pbrt/tests/corpus.rs` and the
`hydra/` goldens render through ReSTIR or the path-tracer default. If either is ReSTIR it
joins the list.

### 4.4 What changes, where

*As shipped — the names moved with the shape (`cheap_restart`, not `candidate_ramp`).*

| Piece | Where |
|---|---|
| `RESTIR_RESTART_CANDIDATES` beside the unchanged `RESTIR_CANDIDATES`; `_SUPPORT_COVERAGE` pinning both at ≥ 1 | `wavefront.rs` |
| `RestirInputs.candidates`; 13 test sites pin `RESTIR_CANDIDATES` | `wavefront.rs` |
| `ViewState::warm`, set at the first `swap` — the premise, enforced | `restir/view.rs` |
| `Film::restir_history_is_warm` | `render/film.rs` |
| `Renderer::candidate_count`, `set_cheap_restart`, and `restir_candidates` — the whole policy, one predicate | `render/mod.rs` |
| `Frame.candidates`, `Report.candidates` (the run's **last** sample) | `stats.rs` |
| `--no-cheap-restart`, `--restart-every-sample` | `cenote-cli/src/main.rs` |
| Toggle through the inputs lane + `film.reset()` branch | `render/session.rs` |
| Checkbox beside the reuse toggles; M on the overlay | `cenote-viewer/src/ui.rs` |
| Moving arm, ReSTIR only (8 → 12 runs) | `docs/m7-baseline.sh` |

### 4.5 Definition of done for 7a

- [x] Lands host-side; kernel SPIR-V byte-identical; 13 test sites pin `RESTIR_CANDIDATES`;
      the whole suite green before any golden was touched
- [x] Unit test — **retired in the post-rung review.** The shipped shape is one predicate,
      and a unit test over its two branches asserts an `if`; the test that actually caught
      the ramp is the many-lights reuse gate, which stands
- [x] One GPU wiring test, three cases: warm restart cheap ≠ pinned; cold film cheap ≡
      pinned; and — added by the review — warm reservoirs with temporal reuse *off*, cheap ≡
      pinned, which failed before `restir_candidates` learned to check it
- [x] Golden migration §4.3 — **not needed.** All seven goldens passed unchanged,
      `compare_bit_identical` included, because a batch render's only restart is its cold
      opening frame and that one is never cheapened
- [ ] `m7-baseline/` re-captured at 12 runs — **done but contended**, a viewer held the GPU
      at 94%; wants a quiet re-run. The pre-7a table in §3 is unaffected
- [x] The A/B, interleaved, min-of-median over clean reps, all four scenes — in D-155
- [x] Viewer toggle and overlay line
- [ ] **Motion quality judged by eye while orbiting**, on the disocclusion case the headless
      harness cannot see — user-owned, and the one claim no test here can make
- [x] D-155: the cheap restart, the warm-history premise, why the ramp died

### 4.6 Risks specific to 7a

- **A pixel disoccluded while moving has no history to warm-start from**, so it gets
  `M(0) = 1` candidates and nothing else — the known ReSTIR weak spot, and the floor
  sharpens it. Spatial reuse (k=5) is its only backstop. Watched in the debug views and by
  eye, not pre-solved; 7b's denoiser is the rung that actually owns moving-preview quality.
- **The moving harness flatters quality while bounding time.** Decision 6. Any claim about
  how motion *looks* that cites a `--restart-every-sample` capture is citing the wrong
  instrument.
- **The temporal M-cap needs no change, and that is load-bearing.**
  `restir_temporal.slang:230` is `min(prev.confidence, mCap · cCanon) · decay` — relative
  to *this* frame's candidate confidence, so the capped prev:cand ratio stays 20:1 at M = 1
  exactly as at M = 16. Had that cap been absolute, this design would have needed a second
  ramp to compensate. Anyone making it absolute breaks 7a.
- **The one that fired.** *Written before implementing, and it landed on decision 3 rather
  than here:* nothing in §4 asked what a thin candidate frame does to the **variance of an
  average**, only what it does to a preview's per-frame confidence. Those are different
  quantities, and the decay window is tuned for the first. The reuse gate caught it; had it
  not, 7a would have shipped a 43% error regression at 8 spp behind a green-looking rung.
  See D-155.

---

## 5. Risk watch

- **The instrument becomes the bloat.** Guard: five memory buckets is a ceiling, not a
  starting point; occupancy and cache counters are delegated to Nsight/RGP by decision,
  not by omission.
- **Sampled timestamps hide a spike.** One frame in eight cannot see a hitch shorter than
  eight frames. The p95 over *every* frame can, and does, which is why it is on the
  headline line and not behind a header.
- **The baseline drifts under background load.** D-152 already recorded that this desktop
  can double every number mid-session. Guard: interleave the arms of any A/B, and read the
  minimum as well as the median.
