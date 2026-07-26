# Cenote — M6 Implementation Plan

*Decisions locked 2026-07-24 via structured interview, preceded by a sourced research
pass over ReSTIR-PT/GRIS ("Foundations of ReSTIR", Lin 2022), the SIGGRAPH 2023 course
"A Gentle Introduction to ReSTIR" (Wyman et al.), the target paper **ReSTIR PT Enhanced**
(Lin, Kettunen, Wyman, I3D 2026 — read to the equation level against its supplemental),
and ReSTCV (Spatio-Temporal Control Variates with ReSTIR, SIGGRAPH 2026) — read against
the engine hygiene of MoonRay (HPG 2017, "Vectorized Production Path Tracing") and Cycles
X (the 2021 wavefront rewrite), whose architectures were each mined for what a
production wavefront tracer treats as load-bearing. Parent scope is charter §4 M6:
ReSTIR-PT — full path reuse via reconnection and hybrid shift maps, GRIS Jacobians,
temporal + spatial reuse of whole light paths, and a convergence study. M6 is the
reordered next milestone (geometry depth deferred — the charter's swap condition, a
rendering-research primary target, is met). It builds on M3's GRIS-DI (D-085…D-092), not
on geometry: the DI reservoir is the dormant-but-correct base case this milestone
generalizes. Decisions D-124…D-133 in [decisions.md](decisions.md) carry the full
rationale; this file is the working plan. Everything consciously *not* built lives in
[deferrals.md](deferrals.md) with its revival trigger.*

Three framing notes the research settled, because they govern every choice below:

- **The thesis extends unchanged: ReSTIR PT and brute-force PT are the same estimator.**
  As in M3, the oracle is cenote's own high-spp NEE+MIS accumulation — the whole claim is
  that reuse converges to the *same film*. Falcor's ReSTIR PT is the behaviour spot-check,
  never the numerical oracle. The M3 unbiasedness gate (reuse-on vs reuse-off → same
  image) carries forward as the spine of validation, from the first path-reuse step, not
  only at the end.
- **Bias and correlation are separate axes — and M6 stays on the unbiased side of it.**
  GRIS makes path reuse unbiased; what reuse costs is decorrelation (correlated unbiased
  frames do not average at 1/N). The converged still stays spatial-only + fresh-per-frame
  RNG (D-085). The plateau the user names — "converges fast early, slow late" — is that
  correlation cost, and M6's answer to it is **ReSTCV** (unbiased, verified by
  accumulating static frames), *not* Enhanced's duplication maps, which are the only
  biased contribution in the paper and plateau *above* the reference under accumulation —
  deferred to a preview-only future (deferrals.md, §7.4 of the paper says as much).
- **This is an accumulation-first lookdev tracer, not a real-time one.** Temporal reuse
  earns its cost only across a Hydra edit or camera move that resets the accumulator; on a
  held frame it anneals to spatial-only accumulation. The interactivity bet is fast early
  frames (ReSTIR) handing off to a clean tail (ReSTCV) — the decay ramp (D-089,
  decayFrames=16) is the mechanism, already built, that this milestone generalizes from DI
  to full paths.

---

## 1. Decisions locked in this session

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 1 | Spine (D-124) | **Full path reuse is the milestone** — the M3 reservoir generalizes from a direct-lighting sample to a whole light path. **Unified DI+GI reservoir** (Enhanced §6.1): one reservoir competes direct (a length-2 NEE path) and indirect candidates together and stores whichever wins, *retiring the separate DI reservoir*. c_cap exposure + ReSTCV fold in on top as the convergence study | A single lightweight reservoir over the full path space is the efficient, occupancy-friendly shape (Enhanced: 431→265 MB at 1080p, and the second-largest single speedup after Russian roulette). The DI reservoir was deliberately built (D-086) as the dormant base case of exactly this generalization; unifying is finishing that design, not bolting a second system beside it |
| 2 | Shift mappings (D-125) | **Hybrid shift is the deliverable; reconnection shift is the validated scaffold.** Reconnection first (cheap, correct on diffuse/rough, **same-pixel Jacobian = 1** — the validation invariant), then hybrid (random-replay the early specular segments, reconnect at the first sufficiently-rough+distant vertex pair — survives glossy/specular). Constant per-pixel storage = the reconnection vertex + a seed. Classic roughness/distance thresholds first; **footprint-based criteria** (Enhanced §4) replace them as an in-milestone rung | Reconnection alone fails on the glossy/specular paths lookdev lives on; hybrid is what makes it production-usable, and it is the paper's default. Reconnection-first is a validation ladder: the Jacobian-1 same-pixel invariant is checkable before any non-trivial Jacobian exists. Random replay is *free* here — the stateless RNG (rng.slang) was designed so any decision replays from its keys |
| 3 | Reuse axes (D-126) | **Both spatial and temporal**, clean and lightweight. Implementation order is **spatial-validated-first** (a same-pixel-topology shift has Jacobian 1, so spatial reuse is provable before temporal reprojection adds a variable), then temporal re-attached through M3's existing reproject + **decay-ramp** machinery. Reservoir storage is **temporal-ready and CV-ready from day one** so temporal and ReSTCV are additions, never re-layouts | Temporal is critical to the interactive feel and is in the final code path; but building spatial first means the shift and pairwise-MIS machinery are validated against a Jacobian-1 baseline before reprojection can hide a bug. Temporal and spatial share the same shift + defensive pairwise MIS (D-087), so spatial-first is a validation ladder, not architectural debt |
| 4 | Convergence — the tail (D-127) | **ReSTCV is first-class and in-milestone.** Architect it as the **general control-variate form with plain ReSTIR PT as the zero-CV degenerate** — one code path, not two. It stores a per-reservoir accumulated-colour estimate as a control variate and is *unbiased* (verified by accumulating static frames). It is activated as the last-validated estimator rung. The decay ramp stays as the ReSTIR-early → better-long-term handoff | The user's "ReSTIR early, better long-term after" is exactly ReSTCV — the published, unbiased realization. Making plain ReSTIR PT the zero-CV degenerate keeps the codebase lightweight (no estimator fork) and gives a free A/B (CV on vs off → same converged image). It is the principled answer to the plateau that duplication maps address only with bias |
| 5 | Reservoir layout (D-128) | **One unified ~64 B path reservoir, designed once** (Enhanced Alg 1): `W`; `float3 F` (the RGB integrand — target is scalar luminance, so F carries colour for the §6.3 colour-noise fix and exact resolve); two seeds (`initRandomSeed`, `rcVertexRandomSeed`) for random replay; `M` packed in flags; the reconnection vertex as the existing **`Hit`** (instance+prim+bary) + oct-encoded `wi` + radiance; cached Jacobian terms **pre-multiplied to one float** with the **NEE light PDF kept unpacked** (it feeds path MIS); plus the **ReSTCV accumulator slot** designed in from the start | The reservoir struct is the milestone's central artifact; getting it right once — holding every field the later rungs light up — is what prevents a re-layout mid-milestone. The `Hit` shape is reused verbatim from M1/M3 (pathstate.slang:27 already names it "the form later reservoir-based passes must hold"). 64 B keeps two temporal buffers + scratch inside a lean per-view budget |
| 6 | Validation & flagship demo (D-129) | Three artifacts: **(1)** an unbiasedness proof — ReSTIR-PT-on vs brute-force accumulation converge to the same image on a **GI-heavy** scene (the M3 gate, now on indirect paths); **(2)** an **equal-time** win, PT vs ReSTIR PT, matched seconds; **(3)** a **convergence-under-reuse study** — mean-error-vs-sample-count curves showing the layered tail: PT → decay-ramp → ReSTCV. Ground truth = cenote's own brute force; Falcor = behaviour spot-check | The oracle must be cenote's own estimator — the thesis is that ReSTIR PT and brute force are the same estimator, so anything else measures the wrong thing (carried from D-090). The convergence-under-reuse curves are the milestone's poster and the quantitative proof the plateau fix works |

## 1b. Amendments from the research review

The locked decisions were reviewed against MoonRay, Cycles X, and the Enhanced paper's
own dependency analysis — the axes a decisions interview under-weights (module
boundaries, what production tracers refuse to abstract, the bias/accumulation
interaction). No decision reversed; four amendments locked, and every one points toward a
*smaller* codebase.

| Amendment | Choice | Rationale |
|---|---|---|
| One integrator, no plugin seam (D-130) | **Do not build a pluggable-integrator abstraction.** cenote keeps ONE config-driven light-transport path; plain PT is the **zero-CV, no-reuse degenerate** of the ReSTIR-PT+ReSTCV path (D-127), reachable by config, not by a second integrator. The modularity that matters is the sampler / light-sampler / BSDF-closure seams cenote already has | MoonRay is one monolithic `PathIntegrator` (no base class, no virtuals, no registry — volumes/SSS are methods on it); Cycles is one data-driven stage machine. Both production tracers converge on one integrator. Building a swappable-estimator layer would be speculative generality — the named risk to resist (D-086) — and the zero-CV degenerate already gives the only fork that matters, for free |
| Shift re-shades from the `Hit` (D-131) | The shift map **re-resolves the `Closure` at the reconnection `Hit`** (a re-shade), storing only the re-evaluable key, never a serialized BSDF. `Closure` is already a pure function of `Hit`+`Material` (openpbr.slang) — POD, value-typed, GPU-resident. Budget one re-shade per reconnection per neighbour as the dominant per-shift cost; **guard its bitwise determinism** under stochastic texture filtering | Both MoonRay (persists `Intersection`, not `Bsdf`) and Cycles (rebuilds `ShaderData`, re-runs the shader graph) say the same thing: closures are transient; store the shader *input*, re-shade on demand. cenote's POD closure is the exact shape MoonRay recommends over its own arena/pointer/vtable graph — a structural advantage to exploit, not to fight |
| Duplication maps deferred; footprint & reciprocal kept (D-132) | **Duplication maps leave the milestone** → preview-only, biased, `deferrals.md`. **Footprint-based reconnection criteria** stay (Enhanced §4) — unbiased, isolated, computed from the **reciprocal area-density of PDFs/geometry cenote already has** (*not* ray differentials/cones), and they remove per-scene threshold tuning. **Reciprocal (paired) spatial reuse** stays (Enhanced §3) and is chosen **over** the deferred stochastic pairwise MIS — it nets ~1.63× *and lowers* FLIP (Gaussian neighbour concentration), a quality+perf win | Duplication maps are the only biased Enhanced contribution and plateau above the reference under accumulation (Fig 15; §7.4 says disable for offline) — dead weight for an accumulation-first renderer whose denoiser is a cadence-throttled view of the *accumulated* film. Footprint and reciprocal are separable, unbiased drop-ins ("generalizes to other reuse algs"). Splatting and CRIS are *not* built: gather reuse suffices and CRIS is subsumed by the GRIS formulation already in use |
| Colour-noise fix, RR trap, AOV seam (D-133) | Fold in the near-free **colour-noise fix** (Enhanced §6.3 — scalar-luminance target vs RGB `F`, fixed by accumulating vector-valued resampling weights for shading; matters for lookdev colour). Handle the **Russian-roulette + random-replay trap** (remove RR from replay, fold survival into the initial-sample PSS PDF — supp §6). Decide the **AOV-under-reuse seam now**: albedo/normal guides resolve from the **canonical path pre-resampling** (cenote already does this via guide throughput, pathstate.slang:89-106); LPE/light-group AOVs stay deferred but the seam is acknowledged, not retrofitted | These are the traps and cheap wins the equation-level read surfaced. The AOV seam is MoonRay's sharpest warning — radiance decomposition observes every scattering event and ReSTIR reuse complicates per-event attribution, so the "which pass does a *reused* path land in?" question is answered by construction (canonical-path guides) rather than discovered late |

## 2. Leaf defaults (stated, not interviewed — cheap to change)

- **Dependencies** (per D-011): M6 adds none to core's public surface. The path reservoir
  is more GPU-resident SoA beside the M3 reservoir buffers; the shift and ReSTCV are pure
  shader functions; random replay needs *no* new RNG state — rng.slang is stateless by
  design (every value a pure function of pixel/sample/dimension), which is precisely the
  property GRIS shift replay rests on. No new crate, no new external library.
- **Determinism** (D-085/D-089): path reuse preserves the bitwise-determinism invariant
  the same way DI reuse did — spatial reads a **committed prior-pass buffer** (ping-pong,
  a barrier between passes), neighbour and shift decisions come from **reserved
  deterministic RNG dimensions** (rng.slang already reserves the RESTIR blocks at
  256/512/768/1280 for exactly this), and no stage accumulates a reservoir with atomics.
  Random replay is a re-ask with the source path's keys — free and deterministic. The
  one new determinism obligation is the reconnection **re-shade** (D-131): it must return
  bit-identical closures across invocations, which the stateless sampler gives *unless* a
  texture-filter path injects per-call randomness — guarded by a re-shade-equality test.
- **Reservoir memory**: ~64 B per pixel per buffer (the Enhanced Alg 1 layout, D-128).
  **Three buffers per view** — prev/curr for the temporal ping-pong plus one scratch —
  row-major linear, per-view ownership keyed by stable view identity (the M3 substrate,
  reused). ~265 MB/view at 1080p (Enhanced's own figure for the unified reservoir),
  *down* from what a separate DI + path reservoir pair would cost — unification is a
  memory win, not a cost.
- **Shift mapping**: reconnection re-resolves the `Closure` at the reconnection `Hit` and
  evaluates the geometry-term-ratio Jacobian; hybrid random-replays the early
  (near-specular) segments from the source seed and reconnects at the first vertex pair
  that passes the reconnection criteria. Classic thresholds to start (roughness
  α_min ≈ 0.2, a world-distance floor); the footprint criteria (Enhanced Eq 5, constant
  c ≈ 0.02) replace them at their rung — a dual area-density test plus a single-vertex
  roughness guard for environment lights, skipping the inverse-footprint test for
  diffuse/emissive reconnection vertices.
- **Target function & colour**: p̂ = luminance of the unshadowed path contribution;
  `F` stores the RGB integrand so colour is exact at resolve and the §6.3 colour-noise fix
  (vector resampling weights for shading) is a stored-field away. The canonical-sample
  support condition (M3's support-coverage assert) generalizes to the path target — the
  hybrid shift's random-replay candidate covers the support the reconnection candidate
  cannot.
- **Reuse counts** (offline lookdev best-practice, Lin 2022 ReSTIR_PT README, tuned in
  validation): ~32 initial candidates, ~3 spatial rounds, ~6 neighbours at radius ~10 px;
  M-cap ≈ 20; the few-frame confidence decay on the motion/edit→hold handoff (decayFrames
  ≈ 16). Deeper bounces draw fewer candidates. Temporal reuse disabled at convergence, per
  the paper's explicit offline recommendation — which the decay ramp already realizes.
- **ReSTCV**: a per-reservoir accumulated-colour control variate; the estimator subtracts
  the CV mean and adds it back analytically, leaving the residual to resample. Unbiasedness
  is verified by the static-frame accumulation test (CV-on and CV-off converge to the same
  image). *This rung carries a research prerequisite* (§6): a focused deep-read of the
  ReSTCV reference (github.com/Hercier/ReSTCV) to pin the exact per-reservoir storage and
  update math before it is built — it is the one rung not yet read to the equation level.
- **Non-goals, graceful degradation**: **volumes** — volume vertices are *not*
  reconnection-eligible; a path through participating media falls back to random-replay or
  no-reuse (MoonRay couldn't keep volumes out of its integrator; M6 does not try to reuse
  through them). **Light sampling** — the power-alias table stays the candidate source;
  the light-BVH / presampled-tile / ReGIR upgrades stay deferred (the reservoir target
  carries the visibility/BSDF the source PDF ignores, so the cheap source is safe).
  **Splatting, CRIS, duplication maps, path guiding** — all out (deferrals.md), with
  triggers.

## 3. Layout additions

```
crates/
├── cenote/
│   ├── shaders/
│   │   ├── reservoir.slang        # grows: the unified path Sample instantiation
│   │   │                          # of the D-086 Reservoir<S> primitive (additive,
│   │   │                          # as its doc block foretold); Rosetta block updated
│   │   ├── shift.slang            # NEW: reconnection + hybrid shift maps, the
│   │   │                          # reconnection criteria (classic → footprint),
│   │   │                          # Jacobian eval; the re-shade-from-Hit path
│   │   ├── restcv.slang           # NEW: the control-variate estimator (general form;
│   │   │                          # zero-CV degenerate = plain ReSTIR PT)
│   │   ├── restir_candidates.slang # grows: path candidates + the unified d=2 NEE ray
│   │   ├── restir_temporal.slang   # grows: reproject + shift + pairwise-MIS on paths
│   │   ├── restir_spatial.slang    # grows: k-neighbour path gather; reciprocal pairing
│   │   ├── restir_resolve.slang    # grows: resolve the path survivor; colour-noise fix
│   │   ├── reservoir_di.slang       # RETIRED: absorbed into the unified reservoir
│   │   └── openpbr.slang            # Closure re-resolve entry for the reconnection re-shade
│   └── src/
│       ├── restir.rs              # grows: path reservoir lifecycle; reciprocal-pairing
│       │                          # texture (self-inverting, re-randomized per frame)
│       └── wavefront.rs           # the reuse stages carry paths, not just DI samples
└── crates/cenote/tests/          # GI-heavy reference scene; Jacobian-1 invariant;
                                   # unbiasedness-under-accumulation; re-shade-equality
```

Files earn existence (D-014). `shift.slang` and `restcv.slang` are the two genuinely new
concepts a reader traces; everything else is the M3 chain generalized from a DI sample to
a path. `reservoir_di.slang` is deleted, not left dormant — unification is the point.

## 4. Build order (~12–16 weeks at 10 h/wk)

Larger than M3's 8–10: full path reuse adds the shift maps, ReSTCV, footprint criteria,
and reciprocal reuse on top of machinery M3 only had to build for an identity shift. The
ordering is: **finish the shift entirely before reusing it across space and time, then
layer the estimator, then optimize** — so nothing is ever built on an unvalidated shift,
and each robustification is measured against a working baseline. Every step ends green:
compiles, clippy-clean (incl. `--features denoise`), tests pass on the GPU machine
(serially — `--test-threads=1`), committed.

0. **Plan docs + reservoir struct + validation harness** — this file; the decisions.md
   D-124…D-133 entries; the deferrals moves (done); README row. Design the ~64 B unified
   reservoir struct once (temporal-buffered, ReSTCV slot + two seeds designed in). Extend
   the test harness: the **Jacobian-1 same-pixel invariant**, converge-to-reference on a
   GI-heavy scene, and **unbiasedness-under-accumulation** (accumulate static frames).
   *Checkpoint: struct allocates and round-trips; the harness runs against the M3
   estimator as a baseline. Nothing renders differently.*
1. **Unified reservoir, DI-equivalent** — route the current DI through the new unified
   reservoir restricted to length-2 (NEE) paths; retire `reservoir_di.slang`. *Checkpoint:
   the existing DI goldens (restir-demo, restir-many-lights) still pass — same output,
   new architecture. The one architectural change, done as a reproduce-then-extend
   migration, risks nothing.*
2. **Reconnection shift + spatial, indirect** — extend the reservoir to multi-vertex
   paths; implement the reconnection shift (re-resolve `Closure` at the reconnection
   `Hit`, D-131); spatial reuse only. *Checkpoint: same-pixel Jacobian = 1 exactly;
   spatial-only path reuse converges to brute force on a **diffuse** GI scene. First true
   path reuse.*
3. **Hybrid shift (classic thresholds)** — random-replay the early specular segments +
   gated reconnection; handle the RR-replay trap (D-133). *Checkpoint: converges to brute
   force on a **glossy** GI scene — lookdev materials render right. Core ReSTIR PT.*
   Expanded to a three-rung ladder in §4a (interviewed 2026-07-25).
4. **Footprint reconnection criteria** — replace the classic thresholds with the dual
   area-density test (Enhanced §4). *Checkpoint: fewer dark/firefly artifacts on distant
   glossy reconnections; convergence equal-or-better; per-scene tuning gone.*
   Expanded to a two-rung ladder in §4b (interviewed 2026-07-26).
   **Done** (2026-07-26): T4a `04898a7` + T4b; checkpoint met — §4b's measurements.
5. **Temporal reuse** — re-point `restir_temporal` at the path reservoir through the
   hybrid shift; reprojection + the decay ramp on paths. *Checkpoint: converges with
   temporal on; the decay-ramp handoff (temporal early → spatial-only + accumulation late)
   is measured. Spatial + temporal complete, all unbiased.*
   Expanded to a three-rung ladder in §4c (interviewed 2026-07-26).
   **Done** (2026-07-26): T5a `66f86a1` + T5b `7245fcb` + T5c; checkpoint met —
   §4c's measurements.
6. **Colour-noise fix + ReSTCV** — first the near-free vector-resampling-weight fix
   (D-133); then, *after its deep-read prerequisite* (§6), ReSTCV as the general
   control-variate form (plain ReSTIR PT = zero-CV degenerate). *Checkpoint: CV-on and
   CV-off converge to the same image (unbiasedness); a measured constant-factor
   accumulated-relMSE win and the chroma improvement — the honest headline; the
   baseline floor measurement recorded either way.*
   Expanded to a four-rung ladder in §4d (interviewed 2026-07-26; the deep-read
   recalibrated the original "tail flatten / plateau fix" claim — §4d decision 1).
7. **Reciprocal (paired) spatial reuse** — self-inverting per-frame pairing textures;
   ~1.63× spatial speedup and lower FLIP. *Checkpoint: equal-quality at lower wall-clock;
   the equal-time figure improves; no structured pairing artifacts (textures
   re-randomized per frame).*
8. **Validation harness + flagship demo** — the three D-129 artifacts: the GI-heavy
   unbiasedness gate, the equal-time PT-vs-ReSTIR-PT figure, the layered
   convergence-under-reuse curves (PT → decay-ramp → ReSTCV). *Checkpoint: ReSTIR PT and
   brute force converge to the same golden; the curves are the poster.*
9. **Polish** — goldens regenerated and eyeballed, module headers and the reservoir
   Rosetta block current, README flagship section, decisions.md current. *M6 done.*

## 4a. Step-3 plan — the hybrid shift (interviewed 2026-07-25)

Step 3 was interviewed to the decision level after step 2 landed. Five decisions,
resolved in dependency order; each rung below ends green (compiles, clippy-clean incl.
`--features denoise`, GPU tests serial, committed), exactly like the milestone's steps.

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 1 | Candidate granularity | **Per-path streamed candidates** (the Lin 2022 / Enhanced shape). `traceTail`'s aggregate retires: every lighting event along the candidate walk — the NEE connection at each vertex, each emission hit — streams into the reservoir as its own candidate with its own F, rcVertex, and MIS fold. The stored winner is one concrete path | The T2 aggregate is sound only while every component shifts through the *same* reconnection at x₂. A movable reconnection vertex breaks that: a path terminating at x₂ cannot ride a reconnection at x₃. Streaming is the shape the target paper's Alg 1 assumes (steps 4–7 plug in rather than fight), bounds the per-shift cost (replay rays + one reconnection ray, never per-prefix-vertex shadow rays), and deletes a cenote-only divergence — including the T2 two-candidate MIS special case, which dissolves into the uniform per-depth NEE/BSDF fold. RIS over the streamed candidates recovers the variance the aggregate bought |
| 2 | Un-baking f₂ | **By construction, not as a feature.** A streamed candidate stores the *incident* radiance at its reconnection vertex plus `rcVertexWi`; f_k re-evaluates at shift time from the D-131 re-shade | Baked Lo(x₂) has no single wi — un-baking is only expressible per-path, so it falls out of decision 1. Reconnection eligibility reinterprets from bias guard to **variance guard**: a sharp vertex is now unbiased to reuse, merely firefly-prone, so the classic thresholds gate quality, not correctness |
| 3 | Support coverage | **Pure replay is in.** A path with no eligible pair anywhere stores only its seed; a shift replays the whole path — every bounce re-traced with the source's RNG keys, the terminal event re-drawn from the same dims. **Temporal holds at J = 0 for replay-kind and k>2 samples until step 5**; k=2 reconnections keep shifting temporally exactly as today | The replay walker must exist for the prefix anyway; run-to-termination is the same walker without a stopping pair — uniformity is the smaller code. Excluding it would forfeit reuse precisely on the sharp chains the milestone names as the point. Temporal-at-step-5 preserves the plan's sequencing: spatial proves the walker against a live, unbiased temporal baseline |
| 4 | Cost discipline | **Measurement-only, no knob.** Cheap rejection gates (neighbour acceptance, zero-W, eligibility) always run before any replay ray; each rung's checkpoint reports measured frame time on the demo + glossy scenes against the T2 baseline | Bidirectional pairwise MIS makes replay-kind shifts trace rays both ways per neighbour — the known price of hybrid, already bounded by `maxBounces`, with both legitimate reducers scheduled (footprint step 4 makes replay rarer; reciprocal step 7 halves the evaluations). A pre-measurement cap is speculative tuning surface (D-086's named risk); if numbers demand one, an unbiased J=0-over-depth knob is a five-line addition. The measurement gives step 7's 1.63× claim a real denominator |
| 5 | Proof obligations | Per-rung gates as listed below; **no FLIP goldens for gate-only scenes**; ReSTIR goldens regenerate at T3a (a different candidate stream is a different estimator realization — bit-stability with T2 is impossible); brute goldens stay bit-identical; D-137+ land append-only with their rungs' commits | The brute-force comparison anchors correctness; per-scene goldens would be maintenance weight without information. The convergence gate (below) exists because every unbiasedness gate here would also pass with reuse silently dead |

The ladder — each rung green and committable:

- **T3a — the streaming restructure, reconnection pinned at k=2.** Candidates stream
  per-path; the aggregate, the baked-radiance eligibility, and the two-candidate MIS
  special case are deleted; an ineligible (x₁,x₂) pair stays J=0; no replay, no seeds.
  *Gates: the existing diffuse unbiasedness scene AND a new glossy-x₂ scene (a glossy
  metal bounce surface at roughness 0.3 behind a diffuse primary — a vertex T2 provably
  refused; note the roughness sits **above** the 0.2 pair-criteria floor, or the gate
  would be vacuous at J=0 — sub-floor vertices are T3c's sharp-chain scene) both pass
  `assert_spatial_reuse_matches_brute_force`; same-pixel Jacobian-1 invariant; estimator
  microtests updated; ReSTIR goldens regenerated and eyeballed; brute goldens
  bit-identical; frame time ≈ T2 (the restructure should be roughly cost-neutral).*
  **Landed 2026-07-25 (D-137).**
- **T3b — the replay walker and the movable vertex.** Seeds light up (the stateless-RNG
  re-key: tail dims drawn from a per-path hash so generation and replay are the same
  draws); prefix replay (bounces 1…k−1) lands in the spatial shift; k chosen during the
  candidate walk by the classic pair criteria; the RR fold (walker never rolls — path
  length stored, survival probabilities stay folded into W). No-eligible-pair paths
  remain J=0 for one more rung. *Gates: **same-pixel replay bit-identity** (a GPU
  microtest: replaying a stored seed at its own pixel reproduces the exact path,
  bit-for-bit — the strongest invariant the hybrid offers, catching silent bias a
  convergence gate would smear); an RR-fold scene deep enough that roulette fires;
  the glossy-primary GI unbiasedness scene; frame time reported.*
  **Landed 2026-07-25 (D-138).** Gate refinements from the implementation: the
  replayed integrand F holds bit-for-bit; the *Jacobian* is pinned to 1 within one
  ulp, not bit-exact — its squared-distance ratio's last bit belongs to per-kernel
  fma contraction, which no source-level discipline controls. The glossy-primary
  scene is a sub-floor metal *panel* with the matte sphere-and-ground world in its
  reflection (a glossy ground alone leaves no diffuse-diffuse pair to lock, and the
  k = 3 gate would be vacuous); the RR-fold scene is a two-mirror corridor whose
  locks land at k ≥ 5, where the reconnection draw itself crosses a roulette roll.
- **T3c — the pure-replay kind.** The walker runs to termination; the terminal event
  re-draws from stored dims (an NEE redraw + shadow ray, or the replayed ray re-hitting
  its emitter). *Gates: a sharp-chain scene (all roughness below the 0.2 floor, so
  reconnection never fires) through the unbiasedness gate; the convergence harness
  extended to the glossy scene asserting ReSTIR beats plain PT at equal spp — the one
  test that proves reuse works rather than merely not lying; docs, D-entries, README
  row, frame time reported.*
  **Landed 2026-07-25 (D-139).** Refinements from the implementation: the generation
  walk unified (the x₂ preamble became the loop's first iteration and the T3a/T3b
  pinned-unshiftable context died — every pre-lock event is now a replay sample, so
  nothing waits on a re-derived x₂ segment); the same-pixel bit-identity gate covers
  the replay kind (F bit-for-bit, both terminal shapes, and a zero-locks coverage
  assert on the sharp scene); and the beats-plain-PT gate runs on an *indirect-only*
  variant of the glossy scene (one-sided emitter facing the panel, black environment
  — measured ≈2× at 8 spp). The direct-lit variant does not clear that bar: PT's
  summed per-vertex estimator is already low-variance there and the one-survivor
  resampling costs more than the default 5-neighbour reuse recovers — the hard-GI and
  many-light regimes are where reuse pays, which is the honest shape of the claim.

Leaf defaults settled with the interview (cheap to change, not re-interviewed): the
classic pair criteria are *both* endpoints past the 0.2 roughness floor (the existing
`reconnectionEligible` shape, now a variance guard) plus a segment-distance floor
relative to the primary hit's camera depth (`Surface.depth` — scale-free, and it dies at
step 4 when footprint criteria replace it). The stored sample packs k, path length,
terminal-event kind, and path kind into the reserved/flags words — no re-layout, as
designed (D-128). At shift time the path MIS weight re-evaluates in the destination
domain using the unpacked `neeLightPdf` against the re-shaded BSDF pdf (the reason that
field was kept unpacked). Resolve keeps its asymmetry deliberately: reconnection-kind
samples re-form at the pixel (the bit-exactness discipline), replay-kind samples trust
the shifted F the reuse stage wrote — re-forming would mean re-tracing the replay.

## 4b. Step-4 plan — footprint reconnection criteria (interviewed 2026-07-26)

Interviewed after step 3 landed (`cfd7fb8`), against the paper itself: Enhanced §4 +
supplemental §§2–4 were re-read to the equation level for this plan (the equation *is*
the step). Four decisions, resolved in dependency order; both rungs end green and
committed, exactly like the milestone's steps.

The criterion, in cenote's terms — Eq 5 with the supplemental's practical choices:

```
lock  (pair search, forward):   1/(e.reconPdf · e.partial)          ≥ rhs
demote (after scatter, inverse): 1/(vScatter.pdf · cosPrev/dist2)   < rhs
rhs   (once per pixel):         (c/100) · ‖x₀−x₁‖² · 4π / |⟨n₁, x̂₁x₀⟩|,  c = 0.02
guard (kept, §4.2):             reconnectionRough(x_{k−1}) only — x_k needs none
```

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 1 | Inverse-test placement | **Lock, then demote.** The pair search locks on [roughness guard at x_{k−1} + forward footprint] exactly where it locks today; the at-lock NEE event streams as-is; after the scatter out of x_k is drawn, the inverse test runs — skipped for diffuse-dominated x_k — and on failure demotes `ctx.shiftable` to false, re-arming the search at later pairs | The inverse footprint needs p_k(ω_k), which does not exist until the scatter draws ~60 lines below the lock. Demotion gives every streamed event the paper's verdict for *its own* continuation: the NEE-at-lock event's suffix is a light draw whose pdf ignores the incident direction, so skipping its inverse test is exact (footnote 6's emissive case), while deeper events ride the drawn ω_k. The alternative — reordering scatter before NEE — restructures walkTail's control flow (degenerate-scatter returns must not skip the NEE stream) and over-gates the NEE event. Demote is ~3 lines: ctx fields are dead while unlocked, prefix/suffix/roulette bookkeeping self-heals because the roll lands after the demote point |
| 2 | Classic fate + A/B evidence | **Measure first, then delete.** Classic's numbers are captured from the step-3 HEAD before any code changes; then footprint replaces classic outright — `RECONNECT_MIN_DISTANCE` and the x_k roughness gate die, `reconnectionRough` survives single-sided. No toggle ever exists | The checkpoint needs a classic baseline, and the committed step-3 tree *is* that baseline — measuring first brackets the diff with zero scaffolding. Git is the fallback seam §5 already names; a shipped knob would violate the measurement-only discipline (D-086's named risk) |
| 3 | Evidence scene | **One dedicated distant-glossy scene**, built in T4a's baseline phase so classic's numbers on it exist before classic dies: a sharp-glossy panel (roughness ~0.05–0.1, below classic's 0.2 floor) far from a diffuse, indirectly-lit wall — classic refuses the pair at x_k and falls to whole-path replay; footprint accepts on segment length | The existing fixtures were tuned so classic's floors *fire*; none can show the criterion's win. The scene is the discriminator and the permanent regression pin: convergence (ReSTIR ≡ brute) plus the ShiftCoverage kind mix (reconnection fraction up, replay fraction down vs classic's recorded numbers) is "earlier, cheaper reconnection" in measurable form. No FLIP golden — the assertions capture what a golden would (§4a decision 5) |
| 4 | Rung structure | **Two rungs, one commit each.** T4a — baseline capture (incl. the new scene) + the criterion swap, every existing gate green. T4b — evidence consolidation + docs | The step is a ~15-line criterion plus plumbing; three rungs would validate an intermediate hybrid criterion the paper never describes, one rung loses the bisection point between "behavior changed" and "evidence added" |

The ladder — each rung green and committable:

- **T4a — baseline, scene, and the criterion swap.** First the distant-glossy scene
  lands with classic still live, and the baseline is captured: convergence relMSE on
  the glossy scenes + the new scene, ShiftCoverage kind mixes, frame times on demo +
  glossy. Then the swap: the per-pixel RHS (primary footprint) replaces `floorDist`
  through walkTail's signature; the forward test joins the pair search (both factors
  already computed by `reconnectionEnd`); the inverse test + demote land after the
  scatter draw; the roughness guard goes single-sided; the classic constants die.
  *(Fixture correction from the T4a interview: the sharp-chain scene needs no re-tune —
  every surface fails `reconnectionRough`, and the surviving single-sided guard at
  x_{k−1} alone keeps it at zero locks; its comment updates. The fixture actually at
  risk is **mirror-chain**: its k ≥ 5 roulette-coverage assert leaned on classic's
  both-sided gate delaying locks to a rough-rough adjacent pair — single-sided, the
  lock lands one vertex after the first rough arrival. Re-tuned only if it goes red.)*
  *Gates: same-pixel bit-identity with full kind coverage; all unbiasedness scenes
  incl. the new one; brute goldens bit-identical; ReSTIR goldens re-eyeballed
  (criterion changes the estimator realization); clippy both variants.*
- **T4b — evidence and docs.** Footprint's numbers on everything T4a baselined;
  the checkpoint report (convergence equal-or-better, kind-mix shift, frame time —
  footprint makes replay rarer, so spatial should get *cheaper*); D-140 append-only;
  the §4 build-order entry checked off; the comment sweep — every "dies at step 4"
  note in restir_target.slang, restir_candidates.slang, and restir.rs resolves.

Leaf defaults settled with the interview (cheap to change, not re-interviewed): the RHS
uses the *geometric* normal at x₁ (the G-term convention `e.partial` already follows)
and is computed once per pixel in the kernel; c = 0.02 stays a named constant with the
/100 and 4π explicit, for paper-legibility; both tests use the *marginal* (all-lobe
mixture) pdfs — supplemental §3's practical choice, and exactly what `bsdfPdf` and
`sampleBsdf` already return; the diffuse-dominated inverse-skip reuses
`reconnectionRough`'s sharp-lobe-weight shape; the length-2 NEE (DI) kind and the
pure-replay kind are untouched — footprint gates only the geometry-pair lock, since an
NEE-sampled light point does not shift its sampling density across pixels (the Eq 4a
ratio is ≈1 by construction, which is why the paper never gates NEE vertices); grazing
primaries (cos → 0) push the RHS to ∞ and simply never lock, the paper's own behavior.

### T4a execution notes (interviewed 2026-07-26)

Five decisions from the T4a implementation interview:

1. **Baseline home** — classic's numbers land as a "Classic baseline — captured at
   cfd7fb8" table in this section, written before the swap and committed with T4a;
   T4b's checkpoint adds footprint's column beside it.
2. **Kind-mix scope** — the three shift-coverage scenes (glossy-primary, mirror-chain,
   sharp-chain) plus the new distant-glossy scene, all through the existing
   bit-identity harness: `ShiftCoverage` grows an NEE-kind/total counter so the mix has
   a denominator, and the harness eprintlns the mix per run. The demo scene contributes
   frame time only.
3. **Scene pins** — the distant-glossy scene joins `assert_spatial_reuse_matches_brute_force`
   and the bit-identity harness with a coverage assert that reconnection survivors
   exist through the sharp panel (`reconnection_ks` non-empty) — the acceptance classic
   structurally refused, pinned forever.
4. **Fixture policy** — re-tune only if red (see the T4a bullet's correction: sharp-chain
   stays guard-refused; mirror-chain's k ≥ 5 is the watched assert). No dedicated
   refusal fixture: the forward-refusal and demote branches fire naturally on the
   existing scenes (near-contact short segments; sharp x_k after a rough vertex), where
   unbiasedness + bit-identity catch any bookkeeping error and the baseline-vs-after
   mixes show them live.
5. **Timing seam** — an `#[ignore]`-by-default GPU test (run explicitly,
   `--test-threads=1`) accumulating N frames of demo + glossy + the new scene through
   the ReSTIR path, eprintln'ing ms/frame; runs identically on both trees, and step 7's
   1.63× claim inherits the same seam.

Leaf implementation defaults (accepted): both tests run **multiply-form** — lock when
`rhs · (e.reconPdf · e.partial) ≤ 1`, demote when `rhs · (vScatter.pdf · inverseG) > 1`
— with the paper's reciprocal form in the comment (IEEE-clean at grazing: rhs = ∞ never
locks, no per-pair division). The kernel's RHS replaces the `depth` line:
`FOOTPRINT_C/100 · ‖x₀−x₁‖² · 4π / |⟨n_geom, x̂₁x₀⟩|` with `FOOTPRINT_C = 0.02` named in
restir_target.slang where the dying constants lived; `floorDist` → `rhs` through
walkTail's signature. The inverse stash is one float at lock:
`inverseG = |⟨prev.normal, e.toRc⟩| / e.dist2` (cosine at x_{k−1} — the
G(x_k→x_{k−1}) convention). The demote sits in the scatter block *before* the
`justLocked` suffix assignment — on failure `ctx.shiftable = false; justLocked = false`,
so the suffix/rcWiWorld block skips, the deeper event arms fall back to replay
streaming, and roulette's division lands in `pendingSurvival` (never reset at lock, so
still correct); the already-streamed at-lock NEE event keeps its exempt verdict. The
inverse test runs only when x_k is sharp-capable —
`max(metalness, specularWeight, transmissionWeight) ≥ 0.25 || coatWeight ≥ 0.25`,
factored as a helper shared with `reconnectionRough`. The guard goes single-sided by
deleting `reconnectionRough(vMaterial)` from the pair-search condition; `prev.rough`
and its assignments stay. Execution order inside T4a's one commit: counters + mix
report + timed test + new scene land with classic live → run, write the classic column
here → swap → all gates serial, mirror-chain re-tuned only if red → ReSTIR goldens
regenerated and eyeballed, brute goldens bit-identical → clippy both variants → commit.

### T4a measurements — classic baseline (captured at cfd7fb8) vs footprint

Kind mixes (bit-identity harness, 32×32 × 8 frames; live survivors as
NEE / reconnection / replay):

| Scene | Classic (cfd7fb8) | Footprint (T4a) |
|---|---|---|
| glossy-primary | 6503 / 387 / 904 | 6503 / 484 / 807 |
| mirror-chain | 1228 / 597 / 6367 | 1228 / 881 / 6083 |
| sharp-chain | 1898 / 0 / 6294 | 1898 / 0 / 6294 — bit-identical, guard-carried |
| distant-glossy | 2146 / 0 / 155 | 2146 / **154** / 1 |

The distant-glossy row is the headline: the discriminating floor→panel→card
survivors flipped from whole-path replay to k = 2 reconnection nearly wholesale.

Convergence relMSE (ReSTIR, 8/32 spp): glossy 0.03316 / 0.00645 → 0.03313 / 0.00645
(equal); indirect-glossy 0.08172 / 0.02265 → 0.08572 / 0.02332, with the same-run
brute ratio holding 2.06× → 1.99× (the reference is ReSTIR-rendered, so its noise
floor moves with the estimator realization; the beats-PT gate's 1.3× floor clears
with margin on both sides).

Frame times (512², release, candidates + spatial, run-to-run ±0.5 ms): demo
6.5 → ~6.8 and glossy-primary 8.3 → ~8.8 (both within noise — the criterion is
cost-neutral where it changes nothing); distant-glossy 3.9 → ~3.2 (the predicted
win: replay-kind reuse became reconnection, and spatial got cheaper).

Outcomes against the plan above: the ReSTIR goldens came out **bit-identical** —
the criterion consumes no random dimensions and agrees with classic on every
golden-scene pair, so nothing regenerated; the mirror-chain k ≥ 5 assert held
without re-tune (locks moved earlier, 597 → 881, but the corridor still locks
deep); the demote branch is live in mirror-chain (a sharp x_k after a matte vertex
fails the inverse term and re-arms the search). The evidence scene settled at
roughness 0.12 / 16 m rather than the interview's 0.05–0.1 sketch: Eq 5's own
arithmetic prices a 0.05-roughness lobe's acceptance at ~80 m for these camera
depths, so 0.12 at 16 m is the same physics at a testable scale — still far under
classic's 0.2 floor, which is the discriminator. The scene needed one rig insight:
a forward-facing emitter's front halfspace inevitably direct-lights the floor, so
the card floats just *in front of the panel*, facing it — its front halfspace holds
the panel alone, no surface anywhere sees it directly, and the floor's light is
purely the panel's sharp reflection.

Checkpoint verdict (T4b), against the §4 entry's three claims: *fewer dark/firefly
artifacts on distant glossy reconnections* — delivered as the mechanism the paper
describes, the artifact-prone whole-path replays becoming k = 2 reconnections (the
kind-mix flip above), with the ~18% distant-glossy frame-time win as the visible
side of the same trade; *convergence equal-or-better* — equal, within the
reference's own noise floor on both glossy scenes; *per-scene tuning gone* —
`RECONNECT_MIN_DISTANCE` and the x_k roughness gate are deleted, and the one
surviving constant (c = 0.02) is the paper's own scene-independent fraction.
Step 4 is closed: D-140 records the criterion.

## 4c. Step-5 plan — temporal reuse (interviewed 2026-07-26)

Interviewed after step 4 landed (`8072816`). The deliverable was already pinned by the
step-2 scope comment in restir_temporal.slang, which names the three debts step 5 owes
at once: the shift across the reprojected surface, a stable identity for
`rcVertex.instance`, and a re-test of the visibility resolve deliberately skips — the
third falls out of the first (the shift re-traces the reconnection segment or terminal
connection at the destination pixel, exactly as spatial's block does). Five decisions,
resolved in dependency order; each rung below ends green (compiles, clippy-clean incl.
`--features denoise`, GPU tests serial, committed), exactly like the milestone's steps.

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 1 | Shift-block home | **Factor, not copy.** Spatial's per-sample shift-and-target block (kind dispatch → shift → visibility ray → shifted target, own-domain target, re-rooted sample) becomes two shared helpers beside the shifts in restir_target.slang — the forward form (neighbour-into-here, one ray) and the ray-free reverse (canonical-into-neighbour-domain). Spatial is rewritten to call them; temporal calls the same two with its one reprojected neighbour | Duplicating ~100 lines of shift dispatch plants the drift-into-silent-bias hazard restir_target.slang exists to prevent. The one stated obstacle — a shared module cannot own a TLAS binding — does not apply: the shift functions already take the TLAS as a parameter, so the helpers do too. The NEE arms stay per-stage (vis-aware in spatial, unshadowed in temporal) — that asymmetry is D-094's feed convention, not drift |
| 2 | `rcVertex.instance` across edits | **Epoch gate, not an instance registry.** `ViewState` remembers which scene build its `prev` was rendered against; when the current build differs, temporal drops the pair for non-NEE prev samples (`prevValid = false` *before* any dereference — the neighbour simply doesn't exist that frame). NEE history keeps surviving edits through the light-id registry, as in M3; both indirect kinds gate uniformly. The instance-identity registry goes to deferrals.md, revival trigger: editing-heavy interactive workflows | A reconnection sample at rest holds a raw TLAS custom index; an edit renumbers it into the wrong (or out-of-bounds) instance. The registry answer — per-instance stable ids, mesh fingerprints, at-rest remap, cross-build index composition — buys exactly one frame of indirect warm-start across an edit that resets the film anyway. The gate is ~3 kernel lines plus a build counter and dereferences nothing stale by construction. Camera-only motion rebuilds nothing, so the common temporal case (orbiting) never trips it. Replay is structurally index-free (seeds + dims, re-traced through the current TLAS) but gates anyway — one condition, both kinds |
| 3 | Cost discipline | **Measurement-only, again — no held-camera identity fast path.** One free ordering fix lands with the step: the ramped confidence folds into `prevValid` before any shift work, so a decayed-to-zero prev is no neighbour at all and the converged still provably spends zero shift rays | The fast path (trust the stored F at identity reprojection on a static scene — sound by the D-138/D-139 invariant) can only fire on a *held* camera, where the decay ramp anneals temporal to zero within decayFrames anyway, and can never fire during *motion*, where reprojection is non-identity and the real shift is irreducible — a second code path optimizing precisely the window where perf matters least (step-3 decision 4's discipline, D-086's named risk). The checkpoint reports temporal-on/off frame times; the fast path is the named follow-up if the ≤ decayFrames window ever shows in the numbers |
| 4 | Proof obligations | **Assert correctness, report performance** (the step-3c precedent). Pinned-temporal unbiasedness (decayFrames = 0 forces temporal live to convergence, D-094) on three kind-covering scenes — diffuse GI (NEE + k = 2), glossy-primary (k ≥ 3 + replay mix), sharp-chain (pure replay, both terminals) — through `matches_brute_force`; a **bitwise G-buffer-surface assertion** (the surface reconstructed from the G-buffer entry equals the path-pool reconstruction bit-for-bit); a host unit test on the epoch plumbing; ReSTIR goldens regenerated + eyeballed, brute goldens bit-identical. The handoff curve and frame times are reported, never asserted | The shift math is already pinned (shared functions, the bit-identity harness) and reprojection is M3-tested; the genuinely new surface is the temporal *wiring*, the epoch gate, and unbiasedness-on-paths. The G-buffer surface is the one input temporal sources differently from spatial — proving it bit-identical transfers the same-pixel invariant to the frame boundary for free. Indirect history crossing the boundary is a new estimator realization, so the ReSTIR goldens move (the §4a decision-5 precedent). No new analytic microtest: restir_temporal_test audits the M-cap/decay algebra, which kinds don't touch |
| 5 | Rung structure | **Three rungs, one commit each**: T5a the no-op refactor, T5b the behavior change, T5c evidence + docs | Two rungs can't tell "the refactor broke spatial" from "the temporal wiring is wrong" — T5a's bit-identical goldens make that distinction free. Four would split out the epoch gate, ~a dozen lines T5b's own gates already cover |

The ladder — each rung green and committable:

- **T5a — the shared shift block, a provable no-op.** The forward and reverse helpers
  land in restir_target.slang; restir_spatial's gather loop rewrites to call them;
  temporal is untouched. *Gates: every existing test green; **both golden sets
  bit-identical** — the rung is a pure refactor and proves it; clippy both variants.*
- **T5b — indirect samples cross the frame boundary.** The two helper calls land in
  restir_temporal (forward: prev → this surface, with its visibility ray; reverse:
  cand → the G-buffer-reconstructed prev surface, ray-free), both through the current
  TLAS; the `prevIsNee`/`candIsNee` scope guards and the step-2 J = 0 convention die;
  the decayed-zero fold into `prevValid` lands before any shift work; the epoch gate
  lands end-to-end (build counter on `ViewState` → push constant → guard-before-
  dereference). *Gates: the decision-4 set — pinned-temporal unbiasedness on the three
  scenes, the bitwise G-buffer-surface assertion, the epoch unit test, ReSTIR goldens
  regenerated + eyeballed, brute goldens bit-identical.*
- **T5c — evidence and docs.** The decay-handoff curve (relMSE at 1/2/4/8/16/32
  accumulated frames, temporal-on vs off, on the indirect-glossy scene — the warm-start
  win early, convergence-equality late) and temporal-on/off frame times (demo + glossy)
  through the `#[ignore]` timing seam, recorded here; D-141 append-only; the comment
  sweep — restir_temporal's step-2-scope block rewritten to what temporal now does,
  spatial's "temporal never passes a foreign indirect sample through" clause corrected
  to the re-rooting argument, stray "until step 5" notes resolved; the §4 build-order
  entry checked off; the instance-identity registry recorded in deferrals.md with its
  revival trigger.

Leaf defaults settled with the interview (cheap to change, not re-interviewed): both
directions of the cross-frame shift trace the *current* TLAS — no previous-frame TLAS
exists, and pairwise-MIS weights remain a valid partition for any deterministic targets,
the standard treatment; the M-cap and decay ramp are untouched (they scale
`prev.confidence` before the combine, kind-blind, and the microtest's cap-then-decay
ordering holds); the temporal NEE arm stays unshadowed and the W feed convention
(unshadowed into `prev`/spatial) is unchanged; a temporally-shifted indirect winner
still satisfies spatial's canonical-trust invariant because the temporal shift re-forms
F at this pixel this frame — visibility re-traced, the sample re-rooted — exactly what
generation would have produced; the helpers live beside the shifts in
restir_target.slang (one module for target + shift, as today); restir_temporal_test.slang
is unchanged.

### T5 measurements — captured at T5c

Rung outcomes against the ladder: **T5a** (`66f86a1`) landed the shared shift block with
both golden sets bit-identical — the refactor proven a no-op. **T5b** (`7245fcb`) crossed
the frame boundary: pinned-temporal unbiasedness (decay 0, 256 frames) green on all three
kind-covering scenes, the bitwise G-buffer-surface fixture green, the epoch plumbing
asserted host-side, the ReSTIR goldens re-pinned (the estimator realization moves — the
§4a decision-5 precedent) and the brute goldens proven pixel-identical. One gate moved:
the many-lights convergence floor 1.5× → 1.3× (the step-3c gate's own floor) —
cross-frame indirect correlation trades a little early-frame independence for the
warm-start (8 spp relMSE 0.0331 → 0.0362, 32 spp 0.0065 → 0.0066, mean unmoved), the
D-094 trade the decay ramp exists to anneal.

Decay-handoff curve (indirect-glossy scene, 128², relMSE of the accumulated image vs a
256-spp reference; fresh film per point, so frames-since-reset — what the ramp reads —
equals the frame count; `restir_decay_handoff_report`):

| Frames | Temporal on | Temporal off |
|---|---|---|
| 1 | 0.68410 | 0.68410 |
| 2 | 0.35122 | 0.30581 |
| 4 | 0.18766 | 0.16293 |
| 8 | 0.08896 | 0.08372 |
| 16 | 0.05322 | 0.05206 |
| 32 | 0.02364 | 0.02332 |

The measured shape corrects the interview's sketch ("the warm-start win early"): on a
*held camera from a cold start* the accumulated average shows **no** early temporal win —
history correlates the frames it folds, and correlated frames average slower (the §6
correlation floor, in miniature). The cost sits ~15% through 4 frames, then shrinks as
the 16-frame ramp hands off — +6% at 8, +2% at 16, +1.4% at 32 — converging to equality;
frame 1 is equal outright (a fresh film holds no history, and the zero-history combine
collapses to the candidate). The warm-start the stage exists for is *per-frame* quality
during motion — there the film resets every frame, so an accumulated-average metric
structurally cannot see it; the pinned-temporal gates hold that regime unbiased, and the
frame-time rows below price it.

Frame times (512², release, ms/frame over 64 timed frames, `restir_frame_time_report`;
"on" pins temporal live at decay 0 — the steady-state cost while the stage works; the
shipping ramp returns a held camera to the off row within 16 frames, and the
decayed-zero fold then spends no shift work at all):

| Scene | Temporal off | Temporal on (pinned live) |
|---|---|---|
| demo | 6.78 | 7.36 |
| glossy-primary | 8.66 | 9.66 |
| distant-glossy | 3.01 | 4.29 |

The live window costs ~0.6–1.3 ms/frame — the cross-frame shift's replay segments and
its one visibility ray. Decision 3's named follow-up (the held-camera identity fast
path) stays dormant: the window is bounded by the ramp, and the converged still is
provably free.

## 4d. Step-6 plan — colour-noise fix + ReSTCV (interviewed 2026-07-26)

Interviewed after step 5 closed (`73deaeb`), with the §6 research prerequisite
discharged first: an equation-level deep-read of ReSTCV (paper + supplemental +
the full Hercier/ReSTCV reference source — a Falcor fork of Lin's ReSTIR PT) and
of Enhanced §6.3, plus a seam map of this codebase. Two findings reshape the step.
**First**, ReSTCV is not the plateau fix D-127 credited it as: its stated problem is
*colour noise* (a scalar-luminance target resamples intensity, not chroma), and its
static-accumulation curves (paper Fig 8) show two parallel ~1/N lines — an unbiased
**constant-factor** variance win from frame 1, not a tail-slope change; no plateau is
exhibited or fixed. **Second**, §6.3 and ReSTCV are the same fix at two depths — §6.3
averages chroma over this frame's spatial candidates (Rao–Blackwellizing the
survivor-index choice); ReSTCV additionally carries the colour estimate across frames
and neighbours as a control variate, and its reference even ships §6.3-style decoupled
shading as its zero-CV mode. The D-127 "general form with degenerates" architecture is
thus *more* right than its rationale: survivor-only → vector-weight shading (6a) →
full ReSTCV (6b) is a genuine ladder where each rung is the next one's special case.
In both, resampling — target, p̂, W, M, MIS — is **100% untouched**; the colour
estimate is a passenger that consumes the shifted integrands, Jacobians, and pairwise
weights the passes already compute. The residual is never resampled, so no
negative-p̂ problem exists (pixel values may go transiently negative; accumulation
averages them out). Seven decisions, resolved in dependency order; each rung ends
green and committed.

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 1 | Headline | **Recalibrate the checkpoint and measure the floor.** Step 6's claim becomes what the literature supports: CV-on ≡ CV-off converged image (the hard gate), a measured constant-factor accumulated-relMSE win, and the chroma improvement. A baseline measurement (spatial-on fresh-RNG vs brute, accumulated) settles whether cenote even *has* a correlation floor in the still regime — recorded either way, the T5c honesty precedent. A decision entry corrects D-127's plateau rationale (append-only: correction, not rewrite) | The written checkpoint ("the tail flatten") is not what ReSTCV does; implementing against a miscalibrated claim reproduces the interview-sketch error T5c had to correct after the fact. cenote's own tail is likely already ~1/N on hold — the decay ramp turns temporal off and frames decorrelate (T5c: temporal-on annealed to +1.4% at 32 frames) — so the honest headline is chroma + constant factor + per-frame quality, and the floor question deserves a number, not an assumption |
| 2 | Rung structure | **Four rungs, one commit each**: 6-0 baseline measurement + D-127 correction (no shader change); 6a §6.3 vector-weight shading; 6b-i spatial CV terms (static-testable); 6b-ii temporal CV recurrence | The baseline must be captured *before* the estimator changes or it is gone. 6a builds the accumulate-in-spatial + shade-from-lane plumbing with zero cross-frame semantics; 6b-i proves the CV math under static accumulation with no reprojection variable (the milestone's validated-ladder ethos — no motion until the math is proven); 6b-ii adds only the cross-frame carry. Three rungs would debug the α/c spatial math and the reprojection recurrence entangled |
| 3 | 6a shape | **Spatial-only accumulation; the lane transports; resolve shades.** Only the spatial pass accumulates the vector weights — at each merge site, w⃗ = w_scalar·F/luminance(F), the chroma direction times the scalar weight already in hand (visibility, Jacobian, and defensive-pairwise m already folded) — into the CV lane of its output reservoir; the lane is pass-local (scratch never persists, so 6a has zero cross-frame semantics by construction). Resolve stays the sole film-writer: shades from the lane when spatial ran, survivor F·W when it didn't; the delta-light term is unchanged | Paper-faithful — Enhanced footnote 7: spatial neighbours carry uncorrelated chroma, temporal history doesn't, so spatial is where averaging wins. The w⃗ identity means no new visibility plumbing and no API widening (luminance(F) = 0 ⇔ F = 0, so the ratio is safe under a zero guard). Shading inside spatial would scatter the delta-light term, shadow queue, debug surface, and film wiring resolve owns |
| 4 | Default + goldens | **Default-on at 6a; the toggle is the degenerate.** Vector shading ships as the ReSTIR path's behaviour from 6a; a renderer-level toggle (the set_temporal_reuse shape) exposes survivor-only shading — the D-130 zero-CV degenerate — for the A/B gates, and 6b reuses it as CV-on/off. ReSTIR goldens regenerate + eyeball at each estimator-changing rung; toggle-off reproduces the prior goldens bit-identical (each rung carries its own no-op proof); brute goldens never move | §6.3 costs nothing measured (17.0 → 17.0 ms in the isolated ablation, HDR-FLIP 0.376 → 0.284) and is strictly better chroma — gating what ships is stronger evidence than gating a side path. No public code ships §6.3 (DQLin/ReSTIR_PT shades survivor-only; Enhanced released no code), so cenote implements from the equations — one more reason each rung carries a bit-identical degenerate check |
| 5 | 6b-i parameters | **Candidate-mean init; α in; c fixed at 1.6.** The candidate pass initializes the lane to the mean of all candidates' F/p (store stops zeroing it; the round-trip fixture starts pinning it); temporal passes the lane through untouched; spatial adds per-neighbour α⟨F_j⟩ + ⟨F_i − αF_j⟩ evaluated at the already-resampled samples with cenote's defensive pairwise m's, dead-shift fallbacks m→1 (backward dead) / m_c→0 (forward dead); α = per-RGB-channel primary-hit albedo ratio, NaN/Inf→1, clamped ≤2; central importance c fixed at the supplemental's closed-form 1.6; resolve outputs the CV | Defensive pairwise satisfies the partition-of-unity the unbiasedness induction needs, so cenote's own m's drop in. The α ablation (ratio MAPE 0.243 vs constant-α 0.344) says it carries real variance, so it lands with the rung — and any deterministic α is unbiased, so there is no correctness risk to isolate. The reference's live per-frame variance model for c (~5%, ad-hoc constants, cross-frame state in an otherwise static-testable rung) is deferred behind a measurement — D-043 discipline |
| 6 | 6b-ii decay | **The CV blend uses the decayed confidence** — q_init = M_c, q_j = the same min(M_prev, cap)·decay the temporal MIS already uses; temporal α ≡ 1; the epoch gate and reprojection failure reset the CV to candidate-mean init (spatial re-enriches the same frame); prev's CV rides the existing prev reservoir buffer | On hold the temporal CV term anneals to zero within decayFrames, so the still is shaded by candidate-mean + spatial CV only — frames stay independent under fresh RNG and the converged-still contract (D-085) holds by construction. Keeping CV history alive on hold would re-introduce exactly the frame correlation T5c measured as an accumulation cost, now on the shading estimate itself. One signal, one ramp, no new constant |
| 7 | Proof obligations | **Floors + reports + the chroma lens** (per-channel relMSE — no new metric machinery; the paper itself had only HDR-FLIP). 6-0: one `#[ignore]` report — the floor curve + per-channel baseline. 6a: existing unbiasedness gates green with vector shading on; chroma deltas recorded. 6b-i: static CV-on ≡ CV-off convergence gate; equal-frame relMSE recorded. 6b-ii: CV-on ≡ CV-off with temporal pinned live; the handoff and frame-time reports re-run | Assert correctness, report performance — the step-3c/5 precedent. Per-channel relMSE separates chroma from luminance well enough to see §6.3's effect without inventing a chromaticity metric with no literature baseline; the floor report answers decision 1's question with a ratio-vs-N curve rather than an argument |

The ladder — each rung green and committable:

- **6-0 — the baseline, captured before the estimator moves.** An `#[ignore]` report:
  spatial-on fresh-RNG vs brute-force, accumulated relMSE at 8/16/32/64/128 frames on
  the indirect-glossy scene (the ratio vs N answers "is there a floor?"), plus
  per-channel relMSE on many-lights (the chromatic scene) as 6a's before-side.
  The D-127-correcting decision entry appends. No shader change; suites trivially green.
  **Done (2026-07-26, D-142).** `restir_floor_and_chroma_report`: **no floor** — the
  converged-still configuration holds N·relMSE flat at ~1.02–1.15 over 16–128 frames.
  The interview's ratio lens turned out to be the wrong one: brute's per-pixel
  accumulation walks the Owen-scrambled Sobol sequence in order (a QMC estimator) and
  drifts mildly *below* 1/N (×N 1.25 → 0.94), a structure resampling forfeits, so the
  equal-frame ratio drifts brute-ward (0.65 → 1.23 by 128) with no floor anywhere — the
  report reads the floor off ReSTIR's own ×N column. Chroma before-side (32 spp,
  per-channel relMSE): ReSTIR R 0.00626 / G 0.00618 / B 0.00726, brute R 0.01039 /
  G 0.01063 / B 0.01235 — the +16% blue excess is the scene's, carried identically by
  both estimators, so 6a's win must show as a level drop or in the eyeball, not as a
  spread collapse.
- **6a — vector-weight shading (Enhanced §6.3).** Spatial accumulates
  Σᵢ mᵢ·F(Yᵢ)·W_{Xᵢ}·Jᵢ via the w⃗ = w·F/luminance(F) identity at every merge site
  (canonical share, neighbours, NEE arm) into its output reservoir's `cvAccumulator`;
  resolve shades from the lane when spatial ran and the toggle is on. *Gates: the
  unbiasedness pair green with vector shading on; toggle-off ReSTIR goldens
  bit-identical; new goldens regenerated + eyeballed; brute goldens untouched;
  per-channel relMSE deltas recorded against 6-0.*
  **Done (2026-07-26, D-143).** Landed as designed: the NEE arms re-form their RGB
  through the shared `evalTarget` (a `contribution` field beside the luminance it
  was always taking), the indirect arms read the shifted F already in hand, and the
  lane rides spatial's pass-local scratch; resolve shades `throughput·lane` under a
  `cv_shading` toggle (bit 10 of the packed flags word), default-on. All gates
  green: unbiasedness pair with vector shading on (many-lights ReSTIR 8/32 spp
  0.03602/0.00652, indirect 0.08893/0.02363); toggle-off pinned **bit-exactly**
  against the pre-6a ReSTIR goldens, kept as standing survivor pins (D-143); new
  goldens regenerated + eyeballed — the copper bounce speckle in the demo's sphere
  shadows visibly desaturates while the image otherwise stands (mean FLIP old→new
  0.020, carried by that speckle); brute goldens byte-untouched. Chroma deltas vs
  6-0 (many-lights, per-channel relMSE): 32 spp R 0.00626→0.00624, G
  0.00618→0.00617, B 0.00726→0.00714; 8 spp B 0.04115→0.04059. The many-lights
  level drop is small and blue-leaning, exactly as 6-0 predicted (the spread is the
  scene's); the demo eyeball is where §6.3 shows. A found identity explains why:
  per-pixel *luminance* is invariant between the two shadings — the survivor's
  luminance(F)·W and the lane's luminance both reduce to weightSum — so §6.3 is
  chroma-only by algebra, and the luminance-dominated relMSE columns barely moving
  is correctness, not a null result.
- **6b-i — the spatial control variate, static-proven.** Candidate-mean init in the
  lane (store stops zeroing; fixture re-pinned), temporal pass-through, the
  per-neighbour control terms with α and c = 1.6, resolve outputs the CV. Signed
  transients verified through accumulate/tonemap/auto-stop (the NaN/Inf guard is the
  only sanitization — no positivity clamp, matching the reference). *Gates: static
  CV-on ≡ CV-off — both converge to the brute reference within the existing floors;
  survivor-toggle goldens bit-identical; goldens regenerated + eyeballed; equal-frame
  relMSE recorded.*
- **6b-ii — the temporal recurrence.** The M-weighted blend lands in restir_temporal
  beside the merge it mirrors, weighted by the decayed confidence, reset by the epoch
  gate and reprojection failure; prev's CV rides the existing buffers — no new
  allocation, no layout change. *Gates: CV-on ≡ CV-off with temporal pinned live
  (decay 0) on the indirect-glossy scene; the pinned-temporal kind-covering gates
  re-run green; the decay-handoff and frame-time reports re-run and recorded here;
  goldens regenerated + eyeballed.*

Leaf defaults settled with the interview (cheap to change, not re-interviewed):
`cvNormalization` stays reserved-zero — the reference persists only the float3
(normalization is inline per pass, a local), and the 96 B layout and size asserts are
untouched; no new RNG dimensions (the CV consumes draws the passes already make; α and
c are deterministic); the guide/AOV seam is unchanged — guides resolve from the
canonical path pre-resampling (D-133), the same seam the reference leaves un-CV'd for
its denoiser inputs; spatial-off configs fall back to survivor shading, so the
degenerate stays reachable by construction; cenote's own luminance() is the chroma
denominator (the reference's Rec.601 vs 709 choice is immaterial — any fixed positive
weighting works); step 7's reciprocal pairing must carry the CV terms through the
paired shifts — a named seam for the step-7 interview, not this one.

## 5. Fallback seams (pre-agreed, in slip order)

- **Reciprocal spatial reuse (step 7)** → plain O(M) defensive pairwise MIS (M3's, on
  paths). First to go: it is a perf optimization on an already-correct, already-fast
  estimator; the milestone's correctness and the headline (ReSTCV) do not depend on it.
- **Footprint criteria (step 4)** → the classic roughness/distance thresholds from step 3
  (they work; footprint removes per-scene tuning and improves distant-glossy, but the
  estimator is unbiased and correct without it).
- **The colour-noise fix (step 6a)** → scalar-luminance resolve as-is; a lookdev colour
  refinement, not a correctness gate.
- **Steps 1, 2, 3, 5, 6b, and 8 are never compressed** — the unified reservoir, the
  reconnection scaffold, the hybrid shift, temporal reuse, ReSTCV, and validation *are*
  the milestone. ReSTCV in particular is first-class by the user's explicit direction; it
  is a defined later rung, never a cut.

## 6. Risk watch

The step-2/3 unknown is the **shift on glossy/specular paths** — the reconnection shift is
silently *wrong* there (it does not crash; it shifts the mean), which is exactly why the
hybrid shift and the reconnection-first validation ladder exist: the same-pixel Jacobian-1
invariant (D-125) is guarded by construction and checked from step 2, and the
unbiasedness gate (reuse-on vs brute force) runs from the first path-reuse step, as it did
in M3. Two concrete implementation traps the equation-level read surfaced, neither
improvised mid-step: (1) the **Russian-roulette + random-replay trap** — naive RR-replay
can turn a survived base path into a killed shifted path, invalidating the sample; the fix
(remove RR from replay, fold survival into the initial-sample PSS PDF) lands *with* step 3,
not after; (2) **re-shade determinism** — the shift re-resolves the `Closure` at the
reconnection vertex, and a stochastic texture-filter path would break the bit-exactness
random replay assumes; guarded by a re-shade-equality test from step 2. The **correlation
floor** is the expected step-5/6 incident (reuse correlates frames; they average slower
than 1/N): the pre-agreed answer is the converged-still contract (spatial-only fresh-RNG,
D-085) and then **ReSTCV** (the unbiased tail fix, step 6) — with the deferred
compatibility-guided-neighbours / MCMC decorrelation as the named next move if the floor
still bites. **Memory**: ~265 MB/view at 1080p is a win over separate reservoirs, but the
per-view × N-viewport multiplier (M3's substrate) still applies — watched, not a blocker.
One **research prerequisite gates step 6**: ReSTCV is the single rung not yet read to the
equation level; a focused deep-read of the reference implementation precedes it (it does
not block steps 0–5, which are independent of it). *(Discharged 2026-07-26 with the §4d
interview — and it earned its keep: the deep-read found ReSTCV is a colour-noise /
constant-factor fix, not the plateau fix D-127 credited, recalibrating the step before
any code was written.)* Build-side, the **AOV-under-reuse
seam** (D-133) is the piece most likely to reveal a hidden assumption — resolved early by
sourcing guides from the canonical path, not discovered late.

## 7. Definition of done

- A GI-heavy scene rendered with ReSTIR PT and with reuse disabled converge to the same
  image (FLIP under threshold) — the unbiasedness gate, on indirect paths, in CI on the
  GPU machine.
- The Jacobian-1 same-pixel invariant and the re-shade-equality check hold as always-on
  tripwires (the two silent-bias classes the shift can introduce), and CV-on vs CV-off
  converge to the same image (ReSTCV unbiasedness).
- Viewer: open a GI-heavy scene, orbit — the preview warm-starts through the move via
  temporal reuse and re-converges on hold via spatial-only + ReSTCV; the frame stops
  pinning the GPU once settled.
- The equal-time figure (PT vs ReSTIR PT, matched seconds) and the convergence-under-reuse
  curves (mean-error vs reference, layered PT → decay-ramp → ReSTCV) regenerate from the
  flagship scene — the poster.
- CI: existing demo and corpus FLIP goldens stay green — at converged spp ReSTIR PT ≡
  brute force, so they remain a free regression gate; the change-set, apply-order, and
  bitwise-determinism tests stay green through the path reservoir buffers.
- A stranger can read `wavefront.rs`'s stage sequence and see the reuse stages carry whole
  paths, read `shift.slang` and `reservoir.slang`'s Rosetta block to map the code to the
  Enhanced paper and the course, and read [deferrals.md](deferrals.md) to know exactly
  what — duplication maps, splatting/CRIS, the light-BVH, path guiding — was consciously
  left for later and when it returns.

## Appendix: primary sources

- **ReSTIR PT Enhanced** — Lin, Kettunen, Wyman, I3D 2026 (paper + supplemental).
  research.nvidia.com/labs/rtr/publication/lin2026restirptenhanced/ · DOI 10.1145/3804494.
- **Foundations of ReSTIR / GRIS** — Lin et al., SIGGRAPH 2022; reference implementation
  ReSTIR_PT (github.com/DQLin/ReSTIR_PT — the offline lookdev parameter defaults).
- **A Gentle Introduction to ReSTIR** — Wyman et al., SIGGRAPH 2023 course
  (intro-to-restir.cwyman.org) — the shift-map and confidence-cap practitioner reference.
- **ReSTCV** — Spatio-Temporal Control Variates with ReSTIR, SIGGRAPH 2026; code
  github.com/Hercier/ReSTCV (the step-6 deep-read target).
- **Engine hygiene oracles** — MoonRay (HPG 2017, "Vectorized Production Path Tracing";
  OpenMoonRay/moonray) and Cycles X (the 2021 wavefront rewrite, intern/cycles).
