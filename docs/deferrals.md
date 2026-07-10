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

- **C ABI** *(revisit: M4)* — Today: pure-Rust change-set API. Production shape: a
  small C ABI whose payload is *serialized change-sets*, MoonRay's RDLMessage
  pattern (manifest + payload + sync id) — not per-attribute FFI setters. The M4
  render server and Hydra delegate are the first real consumers. (D-052)
- **Binary change-set wire format** *(revisit: M4, with the ABI)* — Today: RON text.
  Production shape: the same serde value through a compact binary codec — a drop-in
  because file = wire = the same value by construction. Adopt when delta traffic is
  measured, not assumed, to be a bottleneck. (D-055)
- **Bulk-data binary container** *(revisit: M5 geometry depth, or when load time
  hurts)* — Today: inline RON arrays or PLY-by-reference. Production shape: a
  memory-mappable companion payload (the role USDC/Alembic play). PLY references
  keep us honest until scene sizes demand it. (D-056)
- **Runtime attribute system** *(condition: a plugin SDK — a charter non-goal)* —
  Today: closed typed schema. Production shape: RDL2-style runtime-registered
  attribute tables with per-attribute metadata. Only a third-party-extensible
  renderer needs this; if the charter's no-plugin stance ever changes, this is the
  first consequence. (D-053)
- **Array instancer op** *(revisit: M4 Hydra instancers / M5 landscape-class
  scenes)* — Today: N named instance objects. Production shape: a native op
  carrying per-instance transform arrays, the form Hydra instancers deliver and
  vegetation-scale scenes need. (D-073)

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
- **One-sided emission** *(revisit: a corpus scene where back-face emission visibly
  wrongs the light transport)* — Today: cenote emitters are two-sided by design
  (D-023); one-sided pbrt area lights import with a counted warning, and flush
  ceiling panels leak a little light above themselves. Production shape: an
  emission-sidedness flag on the material, honored by both light-sampling
  strategies. (D-079)
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
- **Sample cap / convergence idle** *(carried from D-051; revisit: M3 interactivity
  work)* — a long-converged viewer still pins the GPU at 100%; `max_samples`,
  publish-interval growth, and a navigation resolution divider belong where the
  frame loop is the subject.
- **Blue-noise sample-index ordering** *(revisit: M3, with the interactivity pass)*
  — the Sobol-Burley sampler was chosen with this as the known drop-in (D-021);
  it improves *perceived* early convergence, which matters most alongside ReSTIR.

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
- **Temporally-aware / in-flight denoising** *(revisit: M3+, with real-time
  interactivity)* — Today: OIDN on the accumulated film at a throttled cadence,
  Cycles' viewport pattern. Production shape only matters when frames stop being
  progressive accumulations.

## Viewer & lookdev

- **Transform gizmos, object creation, scene authoring UI** *(revisit: M4 — usdview
  through the Hydra delegate supplies this wholesale)* — Today: material panel only.
  Building authoring UI ourselves duplicates what the M4 milestone gets for free.
  (D-064)
