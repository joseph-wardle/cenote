# Deferral ledger

Every entry here is a *production solution we consciously decided not to build yet* —
the "right long-term answer, too much for now" option from a design interview. Each
records what we do instead today, what the production shape looks like, and the
trigger that revives it. The point of this file: when the trigger fires, the upgrade
is a plan we already made, not a rediscovery.

Unlike [decisions.md](decisions.md), this file is **not** append-only: when a
deferral is picked up, its entry moves into a new dated decision entry and is deleted
here. An entry's D-reference points at the decision that created the deferral.

---

## Scene API & formats

- **Bulk-data binary container** *(revisit: M5 geometry depth, or when load time
  hurts)* — Today: inline RON arrays or PLY-by-reference. Production shape: a
  memory-mappable companion payload (the role USDC/Alembic play). PLY references
  keep us honest until scene sizes demand it. (D-056)
- **Runtime attribute system** *(condition: a plugin SDK — a charter non-goal)* —
  Today: closed typed schema. Production shape: RDL2-style runtime-registered
  attribute tables with per-attribute metadata. Only a third-party-extensible
  renderer needs this; if the charter's no-plugin stance ever changes, this is the
  first consequence. (D-053)

## Importer coverage

Skipped pbrt features warn by token name at import; each maps to the milestone that
makes it honest to support. (D-057)

- **`curve` shapes** *(revisit: M5 geometry depth)* — needs a real curve primitive,
  not a tessellation hack.
- **`subsurface` materials** *(revisit: M7)* — today: warned and imported as the
  default surface; real random-walk SSS is M7's whole subject.
- **Participating media / `MakeNamedMedium`** *(revisit: M8 volumes)* — today the
  corpus teapot's tea imports as colorless glass, warned.
- **`spot` lights** *(revisit: first corpus scene that uses one — trivial)*.
- **`measured`/`mix` materials, `realistic` camera** *(no milestone; revisit on
  demand)* — measured BRDFs and lens tables serve research comparisons, not the
  production path.
- **Spectral light and IOR data** *(revisit: on demand)* — Today: named/file/inline
  spectra degrade with a warning (lights to white at their photometric scale,
  dispersive IORs to 1.5, conductor spectra outside the four-metal F0 table to
  copper). Production shape: spectral upsampling projected to RGB at import —
  meaningful only alongside the closure's own spectral ambitions (dispersion).
  (D-057)
- **Two-sided emission** *(revisit: a corpus or showcase scene with a `twosided
  true` area light that visibly needs back-face emission)* — Today: cenote emitters
  are one-sided (winding-front face only, matching pbrt's default); a pbrt light with
  `twosided true` loses its back faces and imports with a counted warning. Production
  shape: an emission-sidedness flag on the material, honored by both light-sampling
  strategies. (D-084)
- **Gzipped PLY (`.ply.gz`)** *(revisit: first showcase scene that ships one —
  lte-orb does)* — Today: the PLY reader reads plain files only. Production shape:
  a gzip wrapper over the same reader via the `miniz_oxide` already in the tree.
  (D-056)
- **Area-light `power` normalization** *(revisit: first scene that uses it)* —
  Today: warned, the plain photometric scale applies. Needs the shape's surface
  area (and an image integral for textured emitters) at import — pbrt's own
  `k_e` computation, straightforward once wanted. (D-057)

## Closure (OpenPBR)

Deferred lobes follow shipping-renderer precedent — Karma launched without
transmission scatter, Arnold disables dispersion when thin-walled, MaterialX
shadergen degrades SSS to diffuse. (D-059)

- **SSS random walk** *(revisit: M7)* — today: `subsurface_weight` degrades to the
  diffuse base (the MaterialX-shadergen fallback).
- **Nested dielectrics / priority stack** *(revisit: M8 volume stack)* — today: one
  current-medium slot in path state; overlapping interiors are undefined. The
  path-state schema seam is where the stack widens.
- **Dispersion** *(revisit: post-M6, needs spectral or 3-sample tint machinery)*.
- **Thin-film iridescence** *(revisit: on demand — self-contained Airy term)*.
- **Anisotropy** *(revisit: with tangent-frame quality work; needs authored tangents,
  which the normal-map path only derives per-hit)*.
- **Transmission scatter (`transmission_scatter`)** *(revisit: M8 — it is interior
  media by another name)*.
- **Transport-mode-aware refraction** *(revisit: bidirectional/light transport,
  if ever)* — Today: the BTDF uses the camera-path adjoint convention (no
  solid-angle-compression factor), which is what makes a VNDF sample's weight
  exactly (1−F)·G1 — the quantity the baked glass tables integrate — and closes
  the glass furnace by construction. Production shape: pbrt's `TransportMode`
  split, with η² compression on radiance-carrying paths. Unidirectional path
  tracing never sees the difference. (D-077)

## Texturing

- **Full mip chains + ray-cone LOD** *(revisit: pre-M3 perf pass, measured)* —
  Today: mip-cap at prep, one BC level, hardware bilinear — Cycles' shape for 15
  years; converged output is unbiased because jittered accumulation integrates the
  footprint. Production shape: ray-cone differentials (Cycles 5.2's dual-number
  approach) selecting mips — a bandwidth/cache optimization, adopted when profiling
  says textures are the bottleneck. (D-060)
- **Texture cache / out-of-core** *(revisit: far future; charter locks
  everything-resident-mip-capped through the flagship)* — Production shape: Cycles
  5.2-style demand-loaded tiles. Only scenes that exceed VRAM budgets force this.
- **Bump & displacement** *(bump: on demand; displacement: M5 geometry depth)* —
  Today: skipped at import with a warning; normal maps cover the corpus. (D-061)
- **UV transforms on texture references** *(revisit: first corpus scene that
  tiles — pbrt's `uscale`/`vscale`/`udelta`/`vdelta`)* — Today: warned, authored
  UVs sample directly (the vendored teapot pre-tiles its checkerboard instead).
  Production shape: a 2×3 UV transform on the texture reference, applied at
  sampling. (D-057)
- **Remap curves on textured roughness** *(revisit: first corpus scene whose look
  visibly needs it)* — Today: pbrt's `remaproughness` applies to constants only;
  a roughness *map* imports with its texels read as OpenPBR roughness directly,
  warned (the α conventions differ: pbrt remapped is `α = √r`, OpenPBR is
  `α = r²`). Production shape: a per-reference value transform baked at texture
  prep, alongside the existing usage classes. (D-079)
- **UDIM tiles + multiple UV sets** *(revisit: first production asset — M4/M5
  era)* — Today: one UV set, one image per reference. Production must-haves the
  corpus never exercises; the texture-reference schema grows a tile pattern and
  the mesh schema a second UV stream when a real asset arrives. (D-073)
- **Neural texture compression** *(revisit: VRAM pressure + cross-vendor
  maturity)* — Today: BC through `intel_tex_2`. NVIDIA's RTXNTC SDK is public
  beta (Vulkan-supported, ~85% VRAM reduction claimed) with no shipping adoption
  yet — watch, don't build. (D-073)

## Estimator & film

- **Firefly clamp** *(carried from D-051; revisit: when a corpus scene demands it,
  now that the denoiser exists)* — Today: NaN/Inf guard only. Production shape:
  direct/indirect clamp knobs (Cycles defaults indirect to 10.0). It is a bias knob;
  it arrives as an explicit decision, off by default, never silently.
- **Specular regularization** *(revisit: first corpus scene with specular
  fireflies — expected during M2 step 7)* — Today: nothing; no mips + normal maps
  + low roughness is the firefly recipe. Production shape, pre-agreed so the
  trigger firing mid-milestone is a plan and not an improvisation: Filter-Glossy
  path regularization (roughness clamp on glossy lobes after blurry bounces —
  Cycles ships it on by default at 1.0) plus Tokuyoshi–Kaplanyan specular AA (NDF
  filtering against normal-map variance). Both are bias knobs: explicit, off by
  default, the D-051 firefly-clamp template. (D-073)
- **Per-ray-type visibility flags** *(revisit: production lighting
  workflows)* — Today: `camera_visible` is wired (D-076) — a TLAS mask bit
  camera rays carry and every other ray ignores, so invisible emitters
  illuminate without appearing. Production shape: the full
  camera/diffuse/glossy/shadow set MoonRay and Cycles carry — more bits
  through the same mask seam. (D-073)
- **Sampler seed** *(revisit: when repeat batch renders need decorrelation —
  the CLI's `render` on scene files, M2 step 7 era)* — Today: `Settings.seed`
  is format data prep never reads. Production shape: a seed input hashed into
  the RNG stream, not a sample-index offset (overlapping index ranges share
  samples, which is not decorrelation). (D-075)
- **Cryptomatte / object-ID AOVs** *(revisit: the M4 compositing story)* — Today:
  beauty/albedo/normal/depth. Production compositing's first ask once real
  pipelines touch the output. (D-073)
- **Alpha as coverage (transparent sky, holdouts)** *(revisit: the M4 compositing
  story)* — Today: the beauty alpha is exactly 1 everywhere — it is the
  "every pixel finished once" counter, and the multi-layer EXR writes it as-is.
  Production shape: camera-ray escapes contribute alpha 0 and stochastic opacity
  its coverage fraction, so renders composite over backplates. Cheap at the
  shade_miss/intersect seam once something consumes it. (D-080)
- **Arbitrary AOVs / light-path expressions** *(revisit: production lighting
  workflows, alongside per-ray-type visibility)* — Today: the four fixed AOVs,
  each hand-wired through the film. Production shape: LPE-selected containers
  (MoonRay/Arnold) or Cycles' pass matrix — a registry the wavefront writes
  through, not four more hand-built buffer pairs. (D-080)
- **Per-pixel adaptive sample steering** *(revisit: when the variance estimate's
  reliability on ReSTIR residual is measured — post-M3)* — Today: M3 lands the
  per-pixel variance substrate and a *global* noise-threshold auto-stop (D-089), but
  not per-pixel/per-tile steering. Production shape: gate steering to the converged
  independent-frame phase (where the residual decorrelates and the white-noise variance
  estimate becomes reliable), make termination *tile*-based so no half-stopped pixel
  starves its neighbours' spatial reuse, keep the estimate and threshold deterministic.
  MoonRay's adaptive sampling is the reference; the ReSTIR-residual correlation is the
  reason it can't be a naive per-pixel copy. (D-089)

## ReSTIR & light reuse

The screen-space primary-hit DI reuse of M3 (D-085…D-090) is the first tenant of the
index-agnostic reservoir primitive (D-086). Everything below is a reuse axis it was
built to extend but that M3 consciously does not build.

- **ReGIR world-space reservoirs** *(revisit: many local lights illuminating
  secondary/volume/SSS vertices — M7/M8 era or a dedicated many-lights pass)* — Today:
  reuse is screen-space, primary-hit only; secondary bounces use M2 NEE+MIS. Production
  shape: world-space reservoirs in a hash grid (Boksansky/Wyman, RT Gems II 2021),
  *orthogonal* to screen-space ReSTIR — the accelerator for the vertices a screen-space
  reservoir can't see. It instantiates the same reservoir primitive; that is why the
  primitive is index-agnostic. (D-086)
- **Reservoir path reuse — splatting, CRIS** *(revisit: M6, ReSTIR-PT)* — Today: M6
  steps 2–5 landed spatiotemporal path reuse whole — the **reconnection shift**
  (D-134/D-135), the **hybrid shift** with the footprint pair criteria (D-137/D-140),
  and **temporal path reuse** across the frame boundary through the same shared shift
  block, epoch-gated across edits (D-141). Still pending from the family: reservoir
  **splatting** and continuous RIS. It reuses the D-086 primitive and the `Hit`-shaped
  reconnection vertex M1 chose; the DI shift built in M3 is the base case the
  reconnection shift generalized. (D-086, D-087, D-134, D-141)
- **Instance-identity registry — indirect history across scene edits** *(revisit:
  editing-heavy interactive workflows, where losing one frame of indirect warm-start
  per edit is felt)* — Today: temporal reuse gates indirect history on the scene build
  (the epoch gate, D-141): a reconnection sample at rest holds a raw TLAS custom index
  an edit may renumber, so when `prev` was rendered against an older build the pair is
  dropped before any dereference — the neighbour simply doesn't exist that frame — while
  NEE history keeps surviving edits through the light-id registry (M3), and camera-only
  motion (the common temporal case, orbiting) rebuilds nothing and never trips the gate.
  Production shape: per-instance stable ids with mesh fingerprints and an at-rest remap
  of stored `rcVertex.instance` across builds — cross-build index composition, the
  indirect twin of the light-id registry. It buys exactly one frame of indirect
  warm-start across an edit that resets the film anyway, which is why it waits for a
  workflow where that frame matters. (D-141)
- **Stochastic opacity in the reused indirect tail** *(revisit: a lookdev scene with
  cutout/fractional-opacity geometry along an indirect bounce — beside the tail's scope
  lines, D-136)* — Today: the inline tail the candidate stage traces for a reconnection
  sample (D-134) commits every crossing as opaque (nearest hit wins), the same shortcut the
  DI BSDF candidate takes; the hash-driven stochastic pass-through that `intersect`/`trace_shadow`
  run is not replayed there, so an alpha-cutout leaf in an indirect bounce reads solid. The
  diffuse/glossy checkpoint scenes carry none. Production shape: thread the same deterministic
  transparency split through the tail's continuation and NEE rays — out with volumes, which are
  likewise not reconnection-eligible (m6-plan §2). Two interior-media corners sit in the same
  bucket: an emitter *nested inside* a closed interior reaches the BSDF draw as a light
  candidate with no Beer–Lambert over the segment (the tail itself absorbs correctly since
  D-134's medium seeding), and a reconnection sample whose x₁→x₂ segment crossed an absorbing
  interior bakes the *source* pixel's absorption into Lo — exact in its own pixel, approximate
  at a neighbour. (D-134, D-136)
- **Presampled light tiles** *(revisit: measured per-candidate global-gather bottleneck
  at large light counts)* — Today: candidates are drawn directly from the
  power-proportional alias table. Production shape: Wyman–Panteleev (2021) RIS over
  presampled tiles — a memory-coherence win for millions of lights that injects
  intra-tile correlation, so it is a cost, not a free upgrade, and waits for the
  measurement that justifies it. (D-088)
- **Stochastic pairwise MIS** *(revisit: when neighbour counts grow enough that O(M)
  pairwise MIS is itself the cost — M6 path reuse, or ReGIR)* — Today: defensive
  pairwise MIS, O(M), evaluated in full. Production shape: the stochastic pairwise
  estimator (subsample the pairwise terms) when M is large. A drop-in refinement of the
  same combine function. (D-087)
- **Compatibility-guided neighbours + MCMC decorrelation** *(revisit: if the correlation
  floor bites — the converged-still contract proves insufficient in practice)* — Today:
  the converged still image decorrelates by construction (spatial-only, fresh per-frame
  RNG — D-085). Production shape, if correlated reuse still leaves a structured residual:
  compatibility-guided neighbour selection (bias reuse toward similar surfaces) and
  MCMC-style decorrelation of the reused stream. The named next move for the step-4/5
  correlation risk, so it is a plan and not a scramble. (D-085, D-089)
- **Duplication maps — spatiotemporal decorrelation for the interactive preview**
  *(revisit: only if a live per-frame denoised preview during motion is built — see the
  two-tier denoiser under Display & denoise)* — Today: M6 ships without them; correlation
  in the reused stream is fought unbiased and by construction — the decay ramp hands
  temporal → spatial-only + fresh-per-frame RNG (D-085), and the once-per-second OIDN
  view averages the temporal-correlated early window into a decorrelated tail before it
  ever samples the film. Production shape (Lin et al. 2026, *ReSTIR PT Enhanced* §5):
  each pixel counts how many reservoirs in a 17×17 neighbourhood share its
  `initRandomSeed` and throttles the temporal confidence cap where that count is high,
  killing the firefly "correlation blobs" a denoiser mistakes for signal. It is the
  *only biased* Enhanced contribution (~3.25 % worst-case, and it plateaus **above** the
  reference under accumulation — the paper itself says disable it for unbiased offline,
  §7.4), so it can never touch the accumulated film. Its one payoff — cleaner denoiser
  input — lands only for a denoiser that consumes the noisy early frames, which cenote's
  cadence-throttled view of the *accumulated* film does not. Free to add later:
  `initRandomSeed` is already stored for random replay, so no reservoir re-layout is
  needed. If it ever ships it is a preview-view-only feature, fenced from the estimator
  by the same seam that keeps denoising a view (never the film) — and should be A/B'd
  against the unbiased compatibility-guided + MCMC decorrelation above, which carry no
  bias into a preview that might then be let to accumulate. (D-085, D-089)

## Performance & sync (one measured pre-M3 pass, per D-043)

- **Timeline-semaphore pacing / async submits** — Today: blocking submits, sequential
  waves. The pass that removes the fences must also revisit the publish-buffer
  strong-count invariant (D-051) — the reuse protocol assumes blocking submits.
- **Wave-tail path regeneration** *(carried from D-051)* — Cycles X refills dead
  lanes mid-wave with the next sample's camera rays; we end the wave. Measure first.
- **Deform-only BLAS refit** *(revisit: animation — M5 era)* — Today: any topology
  or vertex change rebuilds the BLAS. Production shape: Cycles' split — refit for
  deformation, rebuild only on topology change. Matters the moment anything
  animates per frame. (D-073)
- **OIDN zero-copy interop** — Today: host-copy (download guides, denoise, upload).
  Production shape: `oidnNewSharedBufferFromFD` against exported VkDeviceMemory,
  vendor-matched device. It shares external-semaphore machinery with the timeline
  pass, so they ship together. (D-063)

## Display & denoise

- **OCIO/LUT display transforms (AgX, ACES 2.0)** *(revisit: when the analytic
  ACES fit's look becomes the limitation)* — the tonemap kernel is a swappable
  stage by design (D-029); ACES 2.0 has no shader-friendly form, so the upgrade is
  a baked 3D LUT through that same slot.
- **Prefiltered denoiser guides (`cleanAux`)** *(revisit: with zero-copy interop,
  which replaces the crate's filter plumbing anyway)* — Today: guides go in as
  noisy aux with the default weights, because the `oidn` crate can't express a
  guide-only prefilter and its `clean_aux` setter misspells the OIDN parameter.
  Production shape: each guide denoised through its own RT filter, then `cleanAux`
  on the beauty filter — OIDN's prescribed highest-quality path. (D-081)
- **Temporally-aware / in-flight denoising** *(revisit: M3+, with real-time
  interactivity)* — Today: OIDN on the accumulated film at a throttled cadence,
  Cycles' viewport pattern. Production shape only matters when frames stop being
  progressive accumulations.
- **Two-tier interactive denoising — fast GPU denoiser early, OIDN at convergence**
  *(revisit: next — the ~1 s cadence makes the denoised view feel non-interactive, and a
  fast interactive denoiser is critical to how a renderer's interactivity feels)* —
  Today: OIDN runs on the accumulated film at a ~1 s throttled cadence, a lagging *view*
  of the estimator (never part of it, `cenote-viewer/src/denoise.rs`). The CPU filter
  costs ~200 ms at 720p and trails the orbit by up to a second, which reads as
  non-interactive during motion. Production shape: a fast GPU denoiser for roughly the
  first 32 samples — NVIDIA **NRD** (ReLAX is the variant tuned for ReSTIR / path-traced
  signals; ReBLUR the general one) or the **OptiX** AI denoiser (GPU, temporal mode,
  driven by the albedo/normal guides cenote already emits) — handed off to OIDN once
  accumulation passes ~32 spp for the higher-quality still. The crossover mirrors the
  estimator's own ReSTIR-early / better-long-term shape and the decay ramp's frame-16
  handoff. Trade-off between the two candidates: NRD is the more interactive-tuned but
  integration-heavy path (needs motion vectors, normalised hit distance, and a G-buffer
  contract — much of which the ReSTIR/AOV plumbing already produces); the OptiX denoiser
  is the lighter lift (albedo/normal already in hand). Which denoiser, and whether the
  fast tier runs per-frame during motion, is the open sub-decision — and the per-frame
  case is the exact trigger that would revive duplication maps under ReSTIR & light
  reuse above. This is what makes ReSTIR PT's fast early frames *feel* fast. (D-081)

## Viewer & lookdev

- **Transform gizmos, object creation, scene authoring UI** *(revisit: M4 — usdview
  through the Hydra delegate supplies this wholesale)* — Today: material panel only.
  Building authoring UI ourselves duplicates what the M4 milestone gets for free.
  (D-064)

## Hydra delegate & render server

The M4 delegate (D-097…D-104, [m4-plan.md](m4-plan.md)) ships the smallest thing that
renders a real USD stage live in usdview. Everything the interview consciously left out
of that first delegate lands here.

- **primId / instanceId AOVs — interactive selection** *(revisit: when click-to-select
  in the usdview render viewport is wanted — a small, self-contained add)* — Today: the
  delegate exposes beauty + first-hit depth; the render viewport carries no object
  identity, so usdview cannot pick or highlight a rendered prim by click. Production
  shape: `primId`/`instanceId` integer AOVs written from the first-hit `Hit` (the
  geometry already carries the identity — it is the light-id's sibling), surfaced as
  `HdAovTokens->primId`/`instanceId` render buffers Hydra maps to selection. Distinct
  from the cryptomatte/object-ID compositing entry above (D-073): that is offline comp,
  this is live viewport picking. One extra cost the zero-Rprim shape adds: the render
  index's own primId→path table is populated at Rprim insertion, so with no Rprims
  `GetRprimPathFromPrimId` returns nothing — the revival also builds the delegate's own
  id→`SdfPath` table, plus the `primId` + `depth` buffers the pick-from-render-buffer
  task reads. (D-101)
- **GPU-shared framebuffer (dma-buf / external memory)** *(revisit: profiled
  motion-to-photon shows the CPU readback dominating at the target viewport resolution —
  4K interactive, or many-AOV interactive denoise)* — Today: CPU shared memory,
  double-buffered async readback, beauty-only across the wire. Production shape: cenote
  renders into exported external `VkDeviceMemory` (dma-buf fd on Linux, Win32 handle on
  Windows) imported into the host's Hgi backend (HgiGL/HgiVulkan) via
  `HdRenderBuffer::GetResource()` — zero PCIe crossing, zero CPU copy. Deliberately *not*
  first: it re-couples the two processes at the GPU-memory + graphics-backend level,
  undoing the clean byte-boundary that makes the split simple and portable, and it splits
  into per-platform interop branches. Shares external-memory machinery with the OIDN
  zero-copy and timeline-semaphore passes (D-063). (D-101)
- **Native `HdRenderer` implementation** *(revisit: the `HdRenderer` API gains real
  methods — the `HdLegacyRenderControlInterface` replacement lands — or a first in-tree
  implementation appears)* — Today: cenote runs under the Hydra 2.0 engine path (26.03+:
  `HdRendererPlugin::CreateRenderer` returns the adapter wrapping the delegate shell) —
  an `HdRenderer` in the only form that exists, with scene consumption already fully
  scene-index-native. The class itself is a verbatim stub ("TODO: Add API here to
  replace HdLegacyRenderControlInterface"); going "pure" today means hand-implementing
  ~25 pure virtuals of transitional task-execution machinery (AOV input, color
  correction, present, Hgi interop) that the adapter provides for free and Pixar is
  about to replace — nothing in the OpenUSD tree does it, hdPrman included. Production
  shape: implement the real API when it exists; the migration touches only the thin
  adapter ring, and the observer core, wire, and server never know. (D-098)
- **Windows host (transport + shm)** *(revisit: the first Windows host — Houdini or
  otherwise — materializes)* — Today: the control channel is loopback TCP + a spawn-time
  token, single-source-path on Windows by construction; the one deliberately-POSIX piece
  is the framebuffer's `shm_open`/`mmap` (~20 lines), whose Windows twin is named
  `CreateFileMapping`/`MapViewOfFile` (~20 more). The wire, protocol, and shm layout
  carry no platform assumptions — the port is those two dozen lines plus a CI lane.
  (D-100)
- **Automatic crash recovery / scene replay** *(revisit: the heavyweight-Houdini case,
  where re-populating a large stage is not a cheap renderer-toggle)* — Today: the process
  boundary gives crash *isolation* (the host survives a cenote crash); recovery is
  *manual* — the user re-populates via a renderer toggle or stage reload, and the
  delegate holds no scene state. Production shape: the delegate keeps a per-prim
  op-shadow (the `Op`s it last emitted per `SdfPath`), detects the dead socket, respawns
  `cenote-server`, and re-sends the accumulated genesis `ChangeSet` — the hdMoonray model
  (verified against source: its delegate retains a full `SceneContext` and lazily
  re-sends genesis on reconnect; Arras itself restarts nothing). The connection protocol
  is already genesis-then-deltas shaped, so this is a pure
  bolt-on with no wire change; it costs a second full copy of geometry in the delegate
  process, which is why it waits. (D-099)
- **Native analytic area lights** *(revisit: measured variance on sphere/disk-heavy
  lookdev scenes — a core-renderer feature, surfaced by the M4 UsdLux mapping)* — Today:
  area lights are emissive meshes (rect → 2 triangles, disk/sphere → tessellated emissive
  geometry), sampled by the existing power-alias NEE with MIS — correct, unbiased,
  ReSTIR-integrated, golden-covered. Production shape: parametric rect/disk/sphere/cylinder
  primitives with unbiased solid-angle importance sampling (Ureña spherical-rectangle,
  sphere-cone), a matching entry in the alias table, and — the real cost — a
  ray-vs-analytic-light intersection path so BSDF-sampled rays still hit the light and MIS
  stays symmetric (else glossy reflections of the light lose their strategy), plus the
  ReSTIR reservoir learning the new light type. LTC-style analytic shading is *not* the
  route: it is biased, and the preview=final thesis forbids it. Justified by variance
  (sphere lights first), never by "UsdLux has a RectLight prim." (D-103)
- **Subdivision refinement** *(revisit: subdiv-authored hero assets read visibly faceted
  in the lookdev viewport)* — Today: USD meshes default to `subdivisionScheme =
  catmullClark`, and the delegate triangulates the base cage via
  `HdMeshUtil::ComputeTriangleIndices` — the accepted floor at usdview's default
  complexity (refineLevel 0, where hdEmbree does the same; at higher refine levels
  hdEmbree hands Embree true subdivision geometry). Production shape: OpenSubdiv
  refinement of the cage at the display style's refineLevel, run delegate-side before the
  `MeshPatch`, so the wire and the renderer stay subdivision-ignorant. (D-098)
- **UsdLux fidelity extras — light textures, response multipliers, sun angle** *(revisit:
  an asset visibly depends on one; textured softboxes are the common case)* — Today: the
  M4 mapping carries `intensity·2^exposure·color`, `normalize`, `enableColorTemperature`,
  one-sided rect/disk emission, and `treatAsPoint`; it drops `texture:file` on rect and
  dome-adjacent area lights, dome `texture:format`s beyond `latlong`
  (mirroredBall/angular/cube-cross), the per-lobe `diffuse`/`specular` response
  multipliers (inexpressible for emissive geometry without shader support), the shaping
  API (cone/focus/IES), and DistantLight's angular diameter (default 0.53°, collapsed to
  a delta — sun-sized soft shadows lost). Production shape: rect-light textures ride the
  existing `emissionTexture` path on the synthesized mesh; the sun angle becomes a
  cone-sampled distant light; shaping and response multipliers wait on native analytic
  lights (above). (D-103)
- **`open_pbr_surface` / `standard_surface` node recognition** *(revisit: when the Hydra
  demo should exercise the full closure — coat/fuzz/transmission — which UsdPreviewSurface
  cannot express)* — Today: the material node switch handles UsdPreviewSurface only; a
  MaterialX surface arrives and maps to the default surface. Production shape: one more
  branch in the same switch reading the OpenPBR (or Standard Surface) root node's
  parameters straight into `MaterialPatch` — a near-identity copy, since cenote's closure
  *is* OpenPBR — reusing the existing texture-node handling and needing no MaterialX SDK.
  One prerequisite beyond the branch: the delegate must advertise `mtlx` in
  `GetMaterialRenderContexts()` to receive that terminal at all. Cheap and additive. (D-102)
- **Full MaterialX graph evaluation** *(condition: a shader-graph / codegen backend — not
  currently planned)* — Today: cenote is a fixed-closure renderer; it consumes known
  surface root nodes (UsdPreviewSurface now, OpenPBR later) with direct-value or
  simple-texture inputs. A MaterialX network whose inputs are driven by arbitrary
  math/procedural nodes is structurally unevaluable — there is no shader graph to run it
  through. Production shape: embed the MaterialX SDK and generate shader code per network
  (the hdStorm/hdPrman path). Only a renderer that grows a shader-graph backend needs
  this; like the runtime-attribute system (D-053), it is the first consequence if that
  architectural stance ever changes. (D-102)
- **Houdini / Solaris *production* integration** *(revisit: a Solaris demo is wanted, or a
  pipeline-TD / rendering-research role targets Solaris specifically)* — Today (step 6
  delivered, D-122): the same delegate source builds against the HDK's USD 25.05 through
  `CENOTE_USD_FLAVOR=hdk`, the 26.05↔25.05 drift isolated in one `usdCompat.hpp`; a
  top-level `UsdRenderers.json` lists cenote in `husk --list-renderers`; and `husk
  --renderer HdCenoteRendererPlugin` renders one frame of the golden stage against
  Houdini's own USD — a load-and-run proof, the usdrecord FLIP golden beside it. What
  remains deferred is husk as a *first-class production citizen*: the render-stat/progress,
  pause/resume, render-settings, deep-AOV, LOP-UI, and relocatable-packaging items below,
  plus stage-scale validation inside a licensed Solaris pipeline. A packaging-and-features
  milestone, not a rearchitecture, by construction of the M4 boundary. (D-097, D-122)
- **husk render stats & progress** *(revisit: husk-driven farm rendering, where the
  progress bar and per-frame stats are the operator's only feedback)* — Today: husk renders
  to completion with no `GetRenderStats`; convergence rides the shm status header the
  delegate already reads. Production shape: `HdRenderDelegate::GetRenderStats` populated from
  that header (samples, converged, rejected-edit count), surfaced as husk's `ALF_PROGRESS`
  and stat dictionary. (D-122)
- **`Stop`/`Pause`/`Resume` beyond the engine stub** *(revisit: an interactive husk /
  Solaris viewport, where a camera drag should pause and resume rather than restart from
  scratch)* — Today: `Stop(bool)` is the stub the 26.03+ engine requires; no true
  pause/resume. Production shape: the session's stop→apply→restart wave boundary wired to
  husk's pause/resume control surface, so an in-flight edit suspends the render instead of
  discarding it. (D-122)
- **`restartrendersettings` / `restartcamerasettings` + `HdRenderSettings` Bprim**
  *(revisit: a stage that authors a RenderSettings prim husk resolves and hands the
  delegate)* — Today: `CreateRenderDelegate(HdRenderSettingsMap const&)` forwards to the
  no-arg path and ignores the map — resolution reaches the delegate through
  `HdRenderBuffer::Allocate`, the only setting one frame needs. Production shape: consume the
  settings Bprim (sampling, AOV selection, product metadata) and declare which of its keys
  force a restart versus a live update. (D-122)
- **Deep AOVs** *(revisit: the Houdini compositing story, beside cryptomatte/object-ID above
  — D-073)* — Today: beauty + first-hit depth. Production shape: per-sample deep-`z` output
  husk writes as a deep EXR, the deep-composite pipeline's first ask. (D-122)
- **Dialog-script LOP parameter UIs** *(revisit: a Solaris artist should tune cenote from
  the LOP network, not the environment)* — Today: cenote is driven by scene data and env
  overrides; it exposes no Houdini parameter interface. Production shape: the `.ds` dialog
  scripts that give a Cenote ROP/LOP its parameter dialog inside Houdini. (D-122)
- **Self-contained relocatable package** *(revisit: shipping cenote to a machine that is not
  the build tree)* — Today: `PXR_PLUGINPATH_NAME`, `HOUDINI_PATH`, and `$CENOTE_SERVER` are
  three documented env exports, and the server lives in `target/`, not beside the `.so`.
  Production shape: `cenote-server` installed beside the plugin and located by `dladdr`
  self-location, so the whole thing is one relocatable directory a `HOUDINI_PATH` entry
  finds with no other env. The `$CENOTE_SERVER` override that makes discovery a one-liner
  today is exactly what keeps this deferrable. (D-122)
- **Delegate colorspace conversion for constant material colors** *(revisit: a front end
  that authors a constant `UsdPreviewSurface` `diffuseColor`/`specularColor` — the golden's
  sibling gap)* — Today: `_ReadDisplayColor` converts `displayColor` Rec.709→ACEScg (D-123),
  but `materialPrim`'s constant node colors still cross the wire raw; no golden catches it,
  because the preview-surface stage drives its base color from a *texture* (its own
  color-space path). Production shape: the one-line twin of D-123's conversion in the
  material-node switch. (D-123)
