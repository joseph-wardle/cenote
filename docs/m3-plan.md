# Cenote — M3 Implementation Plan

*Decisions locked 2026-07-11 via structured interview, preceded by a sourced research
pass over ReSTIR-DI (Bitterli 2020) and GRIS/"Foundations of ReSTIR" (Lin 2022), the
defensive pairwise-MIS formulation (Bitterli 2022 thesis), Wyman–Panteleev presampled
tiles (2021), ReGIR (RT Gems II 2021), and 2023–26 follow-on work — read against the
engine hygiene of Cycles X (kernel-per-stage SoA wavefront), MoonRay (first-class
adaptive sampling, strict scene/delegate separation), and pbrt-v4 (the textbook
estimator). The plan was then reviewed against those same three renderers for module
boundaries, naming, and observability, and amended (see §1b). Parent scope is charter
§4 M3: GRIS-DI — "reservoirs, unbiased contribution weights, generalized MIS; temporal
+ spatial reuse; convergence policy v1; validation harness." Decisions D-085…D-092 in
[decisions.md](decisions.md) carry the full rationale; this file is the working plan.
Everything consciously *not* built lives in [deferrals.md](deferrals.md) with its
revival trigger.*

Two framing notes the research settled, because they govern every choice below:

- **The estimator oracle is RTXDI/Falcor, not Moonray/Cycles.** Neither Cycles
  (light-tree) nor MoonRay (many-light BVH) ships screen-space ReSTIR as its primary
  direct-lighting path; cenote's bet — ReSTIR-DI *as* the primary DI accelerant — is
  the more modern one, and the charter names Falcor's ReSTIR PT as the correctness
  oracle. Moonray and Cycles are the oracle for everything *around* the estimator:
  discrete single-purpose kernels, SoA state, adaptive sampling, delegate separation.
- **Bias and correlation are separate axes.** GRIS makes temporal reuse *unbiased*.
  What temporal reuse costs is not correctness but decorrelation — correlated unbiased
  frames do not average at 1/N. This is why the converged still image (§D-085) is
  spatial-only with fresh per-frame RNG, and why temporal is annealed for *efficiency*,
  never suppressed for *correctness*.

---

## 1. Decisions locked in this session

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 1 | Convergence contract (D-085) | **Unbiased temporal warm-start across camera moves *and* edits** (film always resets on edit; reservoirs carry, re-target under the new scene, decay stale confidence); the **converged still is spatial-only + fresh-RNG** → independent unbiased frames → clean 1/N to today's exact ground truth. **Build the substrate now:** per-view reservoir/film ownership keyed by a stable view identity; stable light identity + index↔identity remap; a cold-start no-carry render setting | Hydra does not forbid cross-edit state carry (restart-on-dirty is a self-imposed delegate convention, via `IsConverged()`); carrying reservoirs across a dirty ≈ carrying the BVH — charter-native delta philosophy. Per-view ownership and stable light identity are high-retrofit-cost, so they land in M3, not M4. Retires the D-064/D-083 viewer-replica seam early |
| 2 | Reuse scope (D-086) | **Screen-space primary-hit DI reuse only**; secondary bounces keep M2 NEE/MIS. The reservoir + WRS + RIS + unbiased-weight + pairwise-MIS is built as an **index-agnostic primitive** — a `Reservoir<Sample>` type and a handful of pure functions, *not* a framework — so M6 path reuse and a future ReGIR grid instantiate it additively. **ReGIR deferred** (trigger: many local lights at secondary/volume/SSS vertices) | Industry baseline: ReSTIR is a primary-visibility accelerant on a converging renderer, not a replacement for path integration. Screen-space DI proves the machinery with zero speculative generality; the primitive stays small enough that reuse is free and over-abstraction is the named risk to resist. M6 (ReSTIR-PT) is *also* screen-space — ReGIR is an orthogonal, later axis |
| 3 | Estimator internals (D-087) | Reservoir domain = the **light-surface point** (stable light-id + primitive + barycentrics — the `Hit` shape, so the DI shift is identity and the **Jacobian is 1**); target p̂ = **unshadowed luminance of f·L·cosθ** over the full BSDF; reuse MIS = **defensive pairwise** (O(M)); bias correction = **unbiased ray-traced** (re-evaluate each neighbour's p̂ under its own visibility) | The Jacobian-1 property holds *iff* the reservoir stores the surface point, not a solid-angle direction — the single most important estimator invariant, and it falls straight out of reusing `Hit`. Pairwise MIS buys balance-heuristic-quality variance at O(M) storage-free cost where the generalized balance heuristic is O(M²). Ray-traced correction is what keeps reuse unbiased across differing visibility |
| 4 | Candidate generation & ownership (D-088) | The reservoir owns **100% of non-delta primary direct lighting**: M light candidates from the existing **power-alias table** (area emitters + HDRI) plus **one internalized BSDF-sampled candidate**; the BSDF continuation ray is **indirect only and suppresses first-hit emission**. **Delta lights stay outside** the reservoir on exact NEE (MIS weight 1). Default M ≈ 16 local + a few env + 1 BSDF, tuned in validation. **Presampled tiles deferred** | Reusing the alias table means no new light data structure. One general primary-DI owner (light + BSDF importance handled *inside* RIS) beats a reservoir-plus-separate-NEE double path; emission suppression on the continuation ray is what prevents double-counting. Delta lights can't be BSDF-hit and carry no MC noise, so a reservoir would only add variance to an exact term |
| 5 | Convergence policy v1 & interactivity (D-089) | M-cap ≈ 20; temporal active during motion + the first post-edit frame, then a **short confidence decay** handing off to spatial-only fresh-RNG accumulation (no pop). Rides along: **blue-noise sample-index ordering**, **interactive niceties** (sample cap / convergence idle, publish-interval growth, nav-resolution divider), and a **per-pixel variance substrate + global noise-threshold auto-stop**. Per-tile adaptive *steering* deferred | The camera-hold switch *is* the spatial-only converge mode. The variance substrate is the same quantity the validation convergence-curves need — one buffer, two consumers. Per-tile steering is deferred because ReSTIR residual is spatially correlated: a per-pixel white-noise variance estimate reads a converged-looking blotch as done, and stopping a pixel starves its neighbours' spatial reuse |
| 6 | Validation & flagship demo (D-090) | Ground truth = **cenote's own brute-force NEE+MIS accumulation** at high spp (proves ReSTIR converges to the *same* film — the thesis). Metrics: **FLIP + numerical mean-error-vs-reference convergence curves** (from the D-089 variance substrate). Unbiasedness gate = **ReSTIR-on vs ReSTIR-off converge to the same image**. Falcor = behaviour spot-check. Demo = **both**: a procedural many-light scene as the golden/validated primary (ReSTIR vs brute-force equal-time + convergence curves), plus one imported many-light showcase as an un-gated README beauty shot | The oracle must be cenote's own estimator, because the thesis is that ReSTIR and brute force are the same estimator — anything else measures the wrong thing. At converged spp ReSTIR ≡ brute force, so the existing FLIP goldens become a free regression gate (§D-090); low-spp ReSTIR noise is deliberately *not* golded |

## 1b. Amendments from the design review

The locked decisions were reviewed against Moonray/Cycles/RTXDI for module boundaries,
naming, and observability — the axes a decisions interview under-weights. No decision
reversed; four amendments locked, three of them about how M3 lands *as code* rather
than what it computes.

| Amendment | Choice | Rationale |
|---|---|---|
| Kernel & module layout (D-091) | The primary-hit reuse chain is **four discrete stages**, matching Cycles' one-purpose-per-kernel shape, never a mega-kernel bolted onto `shade_surface`: `restir_candidates` (initial RIS) → `restir_temporal` → `restir_spatial` → `restir_resolve`. Visibility/bias-correction rays reuse the `trace_shadow` kernel. The index-agnostic primitive lives in a new `shaders/reservoir.slang`; each stage is its own `shaders/restir_*.slang`; reservoir-buffer lifecycle, per-view ownership, and the light-identity remap live in a new `src/restir.rs`; `wavefront.rs` wires the stages beside its existing five | The wavefront stage sequence *is* the map of the renderer — a new reader traces it to understand a frame. Four named stages in that sequence keep the map honest where a fused kernel would hide the estimator. Discrete kernels also make §D-092's per-stage toggles free, and match the SoA queue-driven engine the code already is |
| Naming convention (D-091) | **Readable identifiers**, with a **Rosetta-stone doc block** at the head of `reservoir.slang` mapping each to its paper term: `Reservoir { sample, weightSum, confidence, unbiasedWeight }`; `reservoirUpdate` (WRS stream insert), `reservoirMerge` (pairwise-MIS combine), `targetFunction` (p̂), `unbiasedContributionWeight` (W_Y). The paper vocabulary (RIS, WRS, UCW, GRIS, Jacobian) appears in comments, never as bare identifiers | The literature's jargon is a wall for a new reader; cenote's house style is prose-heavy narration (see `rng.slang`). Readable names plus one Rosetta stone give discoverability without losing the ability to check the code against the papers |
| Debug/observability surface (D-092) | A **first-class debug workstream**, landing with step 3, not folded into final validation: false-colour **selected-light-id**, **confidence (M)** and **unbiased-weight (W)** heatmaps, **per-stage on/off toggles** (initial-only, +temporal, +spatial — the same switches the D-090 unbiasedness gate drives), and a **reuse-gain** view off the variance substrate. Gated behind a viewer debug mode / render setting, written by `restir_resolve` through a single enum-selected debug buffer — lean, not the full AOV registry | ReSTIR bias creeps in silently; the final converge-to-reference gate is an *end* check, and steps 4–5 are undebuggable without intermediate visibility. RTXDI ships exactly this surface for exactly this reason. It doubles as discoverability — the debug views *are* documentation of what each stage does |
| Thesis-honesty note (D-085) | The README's "the preview and the final frame are the same estimator" prose gains a **footnote**: temporal-on-motion and spatial-on-hold are *different* estimators frame-to-frame, both unbiased, converging to the same image. The claim stays true at the level that matters (same converged result, no biased preview mode) but stops overstating frame-level identity | The thesis is the project's defining promise; it must survive contact with an annealed temporal term. The footnote keeps it honest rather than quietly false |

## 2. Leaf defaults (stated, not interviewed — cheap to change)

- **Dependencies** (per D-011): M3 adds none to core's public surface. The reservoir
  buffers are more GPU-resident SoA beside the path pool; the variance substrate is one
  more accumulation buffer with the same pixel-owned determinism invariant; blue-noise
  ordering is a permutation on the existing Sobol-Burley sample index (D-021's earmarked
  drop-in). No new crate, no new external library.
- **Determinism** (D-085/D-089): reservoir reuse preserves the bitwise-determinism
  invariant for free — spatial reads a **committed prior-pass buffer** (ping-pong, a
  barrier between passes), neighbour selection comes from a **reserved deterministic RNG
  dimension** (the slot `rng.slang` already reserves for a GRIS shift decision), and no
  stage accumulates a reservoir with atomics. Async submits would break this, which is
  precisely why the timeline-semaphore pass stays deferred (D-089/D-043).
- **Reservoir memory**: ~16 B per pixel per buffer (sample = light-id + primitive +
  barycentrics + the scalar reservoir fields), two buffers for temporal ping-pong plus
  the spatial-pass pair, block-linear. Per-view, so a Hydra delegate driving N viewports
  owns N sets — the ownership keyed by stable view identity, not delegate-global.
- **Stable light identity**: the current GPU light index is order-derived *and*
  power-filtered in `scene/lower.rs` ("name order", "a powerless light is skipped
  outright"), so it is volatile across edits. M3 adds a stable identity (the source name,
  the same key the change-set API already uses) and an identity↔dense-slot remap rebuilt
  on light add/delete; reservoirs store the stable identity and translate at reuse. A
  reservoir referencing a deleted light is dropped; a remapped light keeps its history.
- **Target function**: p̂ = luminance(f·L·cosθ), unshadowed, evaluated over the full
  OpenPBR closure (the D-070 one-sample-MIS lobe machinery already gives a combined
  BSDF value). Luminance (not the RGB vector) keeps the reservoir scalar-weighted; the
  colour is recovered exactly at resolve where f·L·cosθ is evaluated for real.
- **Continuation ray**: after `restir_resolve` shades the primary-hit direct term, the
  path continues as an ordinary BSDF-sampled indirect ray into `intersect` — with
  first-hit emission suppressed on that ray, because the emitter it might hit is already
  a reservoir candidate. Secondary bounces are untouched M2 code: NEE + BSDF-hit power
  heuristic.
- **Reuse counts** (tunable in validation, these are the starting points): M ≈ 16 local
  + a few env candidates + 1 BSDF candidate at initial RIS; ~5 spatial neighbours at
  radius ~30 px, 1–2 spatial passes; M-cap ≈ 20; a few-frame confidence decay on the
  motion→hold handoff. Never feed a spatial result back into the temporal reservoir.
- **Variance substrate**: a per-pixel running mean + second moment of luminance, the
  standard online estimator, pixel-owned like every other film buffer. It powers the
  global auto-stop *and* the validation convergence curves. Per-tile *steering* is
  deferred (D-089) until the estimate's reliability on correlated ReSTIR residual is
  measured — the estimate assumes white noise, which the residual is not.
- **Interactive niceties** (picked up from the D-051/D-043 deferrals): `max_samples`
  and a convergence-idle that stops pinning the GPU on a settled frame; publish-interval
  growth as convergence slows; a navigation resolution divider during camera motion.
  These are frame-loop concerns, landing where the frame loop is the subject.

## 3. Layout additions

```
crates/
├── cenote/
│   ├── shaders/
│   │   ├── reservoir.slang        # NEW: the index-agnostic primitive —
│   │   │                          # Reservoir<Sample> + WRS/RIS/pairwise-MIS
│   │   │                          # pure fns; Rosetta-stone doc block up top
│   │   ├── restir_candidates.slang # NEW: initial RIS over primary hits
│   │   ├── restir_temporal.slang   # NEW: reproject + pairwise-MIS combine
│   │   ├── restir_spatial.slang    # NEW: k-neighbour gather + combine
│   │   ├── restir_resolve.slang    # NEW: shade the survivor; hand off indirect
│   │   ├── lights.slang            # grows the target-function eval + candidate feed
│   │   └── ...                     # shade_surface: primary-hit NEE moves into RIS
│   └── src/
│       ├── restir.rs              # NEW: reservoir buffers (ping-pong, per-view),
│       │                          # light-identity remap, stage-chain wiring inputs
│       ├── wavefront.rs           # four new stage pipelines beside the existing five
│       ├── scene/lower.rs         # gains the stable-identity↔dense-slot map
│       └── film / session         # per-view ownership; variance substrate buffer
├── cenote-viewer/
│   └── src/                       # debug-view selector; convergence-idle + nav divider
└── crates/cenote/tests/           # brute-force reference + ReSTIR-on-vs-off gate;
                                   # procedural many-light demo scene
```

Files earn existence (D-014); this is the expected shape, not a quota. The four
`restir_*.slang` stages are the ones a new reader will trace in `wavefront.rs`'s
sequence to understand a frame — they are named for what they do, in the order they run.

## 4. Build order (~8–10 weeks at 10 h/wk)

The charter sized M3 at 6–8 weeks; the interviewed scope adds the per-view/stable-identity
substrate, the variance substrate, the debug surface, and two demo scenes (all
charter-consistent), so 8–10 is the honest number — §5 lists what slips first. The
substrate lands first by deliberate choice (its per-view ownership and stable light
identity are the highest-retrofit-cost pieces); the cheap correctness proof follows
immediately so ReSTIR's value is demonstrated early.

Each step ends green: compiles, clippy-clean (including `--features denoise`), tests
pass on the GPU machine, committed.

1. **Plan docs** (this file, deferrals.md moves, decisions.md D-085…D-092, README row).
2. **Reservoir primitive + substrate** — `reservoir.slang` (the `Reservoir<Sample>`
   type + WRS/combine pure functions, Rosetta-stone doc block); `src/restir.rs` with
   ping-pong per-view buffers; the stable light-identity↔dense-slot remap in
   `lower.rs`; per-view ownership keyed by view identity. *Checkpoint: buffers allocate
   and round-trip per view; a light add/delete remaps without touching reservoir
   history. The riskiest step — see §6.*
3. **Initial RIS + resolve + debug surface** — `restir_candidates` (alias-table
   candidates + BSDF candidate, WRS, one shadow ray on the survivor) and
   `restir_resolve` (shade survivor, continuation ray with emission suppression);
   primary-hit NEE moves out of `shade_surface`; delta lights stay on exact NEE; the
   §D-092 debug views land here. *Checkpoint: single-frame RIS matches brute-force in
   expectation — the unbiasedness gate — and you can false-colour the selected light.*
4. **Spatial reuse** — `restir_spatial`: k-neighbour gather from the committed
   prior-pass buffer, defensive pairwise MIS, ray-traced visibility bias correction.
   *Checkpoint: spatial-only converges to ground truth; first convergence curves.*
5. **Temporal reuse** — `restir_temporal`: camera reprojection (scenes are static, so
   reuse is a ray-traced re-test against the same TLAS), M-cap, warm-start across moves
   *and* edits, the confidence-decay handoff to spatial-only. *Checkpoint: post-move and
   post-edit preview warm-starts; the frame still converges cleanly on hold.*
6. **Convergence policy + variance substrate** — per-pixel variance accumulation,
   global noise-threshold auto-stop, sample cap / convergence idle, nav-resolution
   divider, blue-noise sample-index ordering. *Checkpoint: a settled viewer stops
   pinning the GPU; early frames read cleaner under blue noise.*
7. **Validation harness + flagship demo** — the brute-force reference path, FLIP +
   mean-error convergence curves, the ReSTIR-on-vs-off unbiasedness gate, a Falcor
   behaviour spot-check, the procedural many-light golden scene, and the imported
   showcase stretch. *Checkpoint: ReSTIR and brute force converge to the same golden;
   equal-time figure shows the reuse win.*
8. **Polish** — goldens regenerated and eyeballed, module headers and the Rosetta-stone
   block current, README flagship section + the thesis footnote, decisions.md current.
   *M3 done. The flagship begins.*

## 5. Fallback seams (pre-agreed, in slip order)

- **Imported showcase (step 7)** → procedural golden scene only. First to go; the
  validated demo is the milestone, the beauty shot is not.
- **Blue-noise ordering + nav-resolution divider (step 6)** → Sobol as-is, full-res
  navigation; the variance auto-stop and sample cap stay.
- **Per-tile debug niceties (step 3)** → the two switches the gate actually needs
  (per-stage on/off) stay; the heatmaps are the trim.
- **Confidence-decay handoff (step 5)** → hard switch from temporal to spatial on
  camera hold (a visible one-frame settle instead of a smooth one); the substrate and
  correctness are untouched.
- **Steps 2, 3, 4, and 7 are never compressed** — the primitive + substrate, the
  single-frame unbiased RIS, spatial reuse, and validation *are* the milestone.

## 6. Risk watch

Step 2 carries the unknown-unknowns: it reshapes light identity (from the volatile
order-derived index to a stable-identity remap) and introduces per-view ownership at
once, both cross-cutting through `lower.rs`, the Session contract, and the delegate
seam — so it lands first, before anything depends on it, exactly as M2's step 3 did.
The silent-wrongness risk is **bias**: ReSTIR bias does not crash or NaN, it shifts the
converged mean by a few percent and looks plausible. Three defences, none improvised
mid-step: (1) the §D-092 debug surface lands *with* the first estimator (step 3), not
after, so every later step is inspectable; (2) the unbiasedness gate — ReSTIR-on vs
ReSTIR-off to the same image — runs from step 3 onward, not just at the end; (3) the
Jacobian-1 invariant (D-087) is guarded by construction — the reservoir stores the
surface point, so no Jacobian term can be silently dropped. The **correlation floor**
is the expected step-4/5 incident: reuse makes frames correlated, so they average slower
than 1/N and the residual is blotchy, not grainy — the pre-agreed answer is the
converged-still contract (spatial-only fresh-RNG, D-085), and if it still bites, the
deferred compatibility-guided neighbours / MCMC decorrelation upgrades (deferrals.md)
are the named next move, not a scramble. Build-side, the substrate's per-view ownership
is the piece most likely to reveal a Session-contract assumption; it is spiked before
step 2 commits to it.

## 7. Definition of done

- `cenote-cli render many-lights.ron --spp 4096` and the same scene with ReSTIR
  disabled converge to the same image (FLIP under threshold) — the unbiasedness gate,
  in CI on the GPU machine.
- Viewer: open the many-light scene, orbit — the preview warm-starts through the move
  and re-converges on hold; toggle each reuse stage and the debug views live; the frame
  stops pinning the GPU once settled.
- The equal-time figure (ReSTIR vs brute-force, matched seconds) and the convergence
  curves (mean-error vs reference) regenerate from the procedural golden scene.
- CI: existing demo and corpus FLIP goldens stay green — at converged spp ReSTIR ≡
  brute force, so they are a free regression gate; the change-set, apply-order, and
  bitwise-determinism tests stay green through the reservoir buffers.
- A stranger can read `wavefront.rs`'s stage sequence and see the four named reuse
  stages in the order they run, read `reservoir.slang`'s Rosetta-stone block to map the
  code to the papers, and read [deferrals.md](deferrals.md) to know exactly what reuse
  work was consciously left for later and when it returns.
