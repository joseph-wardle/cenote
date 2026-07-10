# Cenote — M2 Implementation Plan

*Decisions locked 2026-07-09 via structured interview, preceded by a sourced research
pass over Cycles X (GPU-first architecture), MoonRay/RDL2 (studio-grade scene API and
delta model), pbrt-v4 (format ground truth), the OpenPBR v1.1.1 spec, and OIDN 2.5 —
then amended the same day after a three-track adversarial review against Hydra's
delegate requirements, Cycles' shipped kernels, and current research (see §1b and
D-068…D-073). Parent scope is charter §4 M2: production I/O + full look. Decisions
D-052…D-073 in [decisions.md](decisions.md) carry the full rationale; this file is
the working plan. Everything consciously *not* built — the "production solution, but
not yet" options — lives in [deferrals.md](deferrals.md) with its revival trigger.*

---

## 1. Decisions locked in this session

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 1 | API boundary (D-052) | **C ABI deferred to M4**; M2 builds the change-set API in pure Rust | The ABI's real job is transporting serialized change-sets (MoonRay ships *no* C API — its boundary is RDLMessage deltas); the text format proves serializability without freezing an ABI two milestones early |
| 2 | Scene typing (D-053) | **Static typed schema**: closed set of object kinds (mesh, instance, material, light, camera, environment, settings) as ordinary Rust types; objects named, string ID → handle | RDL2's runtime attribute machinery serves a plugin SDK the charter rules out; a typed schema gets exhaustive matches and serde for free |
| 3 | Edit model (D-054) | **Change-sets as first-class values**: ordered typed patches (`Option` per attribute), get-or-create by name on apply, plus `Remove(kind, name)` ops (D-069); `apply()` is the *only* mutation path and accumulates the dirty state that drives re-prep | A file load IS a change-set (RDL2's insight); file = wire = undo = the same value; one mutation path means one dirty-tracking story. Remove is the wart RDL2 can't fix — Hydra requires real deletion (§1b) |
| 4 | Text format (D-055) | **RON via serde**, version field first | The derive *is* the parser — no drift between schema and format; serde+ron justified under D-011 (a hand parser would be 400+ lines that rot) |
| 5 | Bulk data (D-056) | Mesh payloads **inline or by relative-path PLY reference** (enum in the mesh op); environment stays EXR-by-reference; no bespoke binary container | Small scenes stay self-contained and diffable; big geometry stays in the format everyone already has; hand-rolled PLY reader (~200 lines) in core |
| 6 | Importer subset (D-057) | trianglemesh/plymesh/sphere/disk/ObjectInstance; diffuse/coateddiffuse/conductor/dielectric/thindielectric; area/infinite/distant lights; imagemap/constant/scale textures; perspective camera — **all five fidelity traps handled** (see §6) | Covers the corpus that actually exists (disk earns its ~20 lines because the killeroo family uses one); every skipped feature warns with the token name so silence never means "handled" |
| 7 | Estimator gaps (D-058) | **All four**: triangle emitters (the quad special case retires), distant + point delta lights, thin-lens DoF | pbrt scenes need them and M3's many-light work wants one general light path, not a quad special case plus patches |
| 8 | Closure cut (D-059) | **Coat** (base-IOR remap + analytic darkening), **fuzz** (Zeltner LTC sheen), **transmission** (Beer–Lambert, single interior medium, thin-walled mode), **variable specular IOR**, **opacity** (stochastic), emission in its stack slot. Deferred with named precedent: SSS, dispersion, thin-film, anisotropy, transmission scatter | Matches OpenPBR's own renderer-ready lobe-mixture decomposition; the deferrals mirror what Karma/Arnold shipped without at first |
| 9 | Mip policy (D-060) | **Mip-cap downscale at prep, upload one BC level, hardware bilinear** | Cycles shipped 15 years on exactly this; jittered accumulation integrates the pixel footprint, so converged output is unbiased — ray-cone LOD is a perf feature and waits for the perf pass |
| 10 | Texturable set (D-061) | base_color, specular_roughness, metalness, emission, opacity + **tangent-space normal maps** (UV-derived tangents per hit, BC5); pbrt bump/displacement skipped with a warning | The set real corpus scenes use; normal maps are the highest look-per-line feature in the pipeline |
| 11 | Denoiser guides (D-062) | **Cycles-style specular pass-through** with the 0–0.15 roughness ramp; albedo/normal/depth as separate pixel-owned accumulation buffers | Mirrors record what they show — OIDN's own guidance; pixel-owned buffers keep the bitwise-determinism invariant |
| 12 | OIDN (D-063) | **Host-copy interop** via the safe `oidn` crate (DEFAULT device), behind a `denoise` cargo feature | OIDN has no Vulkan device; zero-copy needs exported VkDeviceMemory + the sync machinery of the timeline-semaphore pass, so it ships with that pass |
| 13 | Lookdev (D-064) | Viewer gains **scene loading + a material panel emitting change-sets** through a Session edit channel (stop → apply → restart); no gizmos, no transform editing | The API's first interactive consumer proves the whole delta path; re-convergence after an edit is the thesis demo |
| 14 | Corpus (D-065) | **Tiered**: 3–4 small permissively-licensed pbrt scenes vendored with goldens, each license verified and its text vendored alongside (import → render → FLIP in CI); checksummed fetch script for showcase scenes; README pins the reference pbrt-v4 commit and states the spectral-vs-RGB caveat | CI stays fast and hermetic; the showcase tier stays out of the repo. Strictly-CC0 pbrt scenes barely exist — "permissive, license vendored" is the honest bar |
| 15 | Layout (D-066) | **`cenote-pbrt` leaf crate** (.pbrt in → ChangeSet out, consumes only core's public API); PLY reader in core (our own format references PLY) | The importer is a client of the scene API, and the crate boundary enforces that the public API is sufficient |

## 1b. Amendments from the adversarial review (same day)

Three review tracks attacked the locked plan against Hydra/MoonRay, Cycles' shipped
kernel source, and 2024–2026 research. No decision reversed; four amendments locked:

| Amendment | Choice | Rationale |
|---|---|---|
| Remove ops (D-069) | The op set gains **`Remove(kind, name)`**, with dirty semantics that retire GPU residency (BLAS slot, light-table entry, texture refs); the M2 viewer never emits one | Hydra's render index (M4) *requires* deletion — `DestroyRprim` is a mandatory delegate virtual, and renames arrive as remove + re-insert. RDL2 can't delete and hdMoonray fakes it with visibility flags — a wart, not a pattern |
| Lobe selection (D-070) | **One-sample MIS** over the lobe mixture: pick one closure proportional to its albedo-estimate weight via a CDF, *rescaling the used random number* to preserve stratification; evaluate **all** lobes and combine as the balance heuristic `pdf = Σ(pdfᵢ·wᵢ)/Σwᵢ`. The **sampled-lobe tag joins path state** | Cycles' shipped shape (`surface_shader.h`). The lobe tag drives the AOV pass-through ramp (D-062) and is the field M1 earmarked for M3's GRIS replay — one field, three consumers |
| Energy compensation mechanism (D-071, amends D-059) | Reflection lobes: Fresnel-free **E/E_avg directional-albedo tables + analytic multiple-scattering Fresnel** (`Fms = Fss·E_avg/(1−Fss(1−E_avg))`) — no IOR table axis at all. Transmission: precomputed **3D glass tables** (roughness × cosθ × IOR, 16³–32³, baked offline, embedded). Coat reuses the same tables and gains OpenPBR's **base-roughness remap** under nonzero coat roughness | Verified in Cycles `bsdf_microfacet.h`: closed-form Fss makes variable IOR free on reflection; only coupled reflection+refraction needs the IOR axis. Less work than the interviewed "fits gain an IOR axis", and furnace-provable |
| Format-freezing fields (D-072) | The format commits to **Y-up, right-handed, meters, vertical-fov degrees**; the camera op carries **full orientation (roll included), `focus_distance`, `aperture_radius`**; texture references carry a **color-space field** (slot-derived default, explicit override); emitters carry **`camera_visible`** (default true, matching pbrt) | Every one is a format version bump if discovered after step 2 ships: pbrt `LookAt` can roll, D-058's DoF needs the lens fields, someone must own sRGB-vs-linear, and lookdev always wants invisible lights eventually |

## 2. Leaf defaults (stated, not interviewed — cheap to change)

- **Dependencies** (per D-011): core gains `serde`, `ron`, `image` (PNG/JPEG decode),
  `intel_tex_2` (ISPC BC encoders), `ddsfile`, and `oidn` behind the `denoise`
  feature. `cenote-pbrt` gains nothing beyond core. Hand-rolled: PLY reader (core),
  pbrt tokenizer/parser (cenote-pbrt). RON hygiene: pin the version (it breaks
  semver-minor routinely) and keep the schema to plain structs and externally-tagged
  enums — RON's documented weak spots are `untagged`/`flatten`, so we don't use them.
- **The apply contract**: names are stable identities — rename is remove + create.
  Name references resolve after the whole set applies, so forward references within
  a change-set are legal. `apply()` is validate-then-apply: all I/O and reference
  resolution happens up front, mutation only after everything passes — a mid-set
  failure (missing PLY, bad EXR) leaves the description untouched. Relative paths
  resolve against the scene file's directory, never the CWD — enforced mechanically:
  `format::load` is the only place relative paths gain a meaning, and `apply()`
  rejects any path still relative. Unknown fields are parse errors, never skipped —
  a typo'd parameter silently ignored would be a wrong render with no error.
- **Render-settings op contents**: spp, max depth, resolution, seed — the minimal
  set, so the format doesn't churn in step 3. Material parameter names mirror
  OpenPBR's slugs exactly (`base_color`, `coat_weight`, …) — interop alignment as a
  commitment, not an accident.
- **Bindless textures**: the descriptor-indexing capability reserved since M0 finally
  pays — one variable-count sampled-image array; material params become
  constant-or-texture-index. BC7 for color, BC5 normals, BC4 scalar masks, BC6H HDR.
- **Color pipeline**: color textures stored BC7-sRGB, hardware sRGB decode, then the
  3×3 `acescg_from_rec709` in-shader (keeps sRGB quantization in storage — Cycles'
  in-kernel pattern). Data maps stay linear. Each texture reference carries its
  color space: derived from the slot by default (color slots sRGB for 8-bit inputs,
  linear for float; data and normal slots always linear), explicit override field
  for the exceptions — pbrt's 8-bit-defaults-sRGB rule maps straight onto it.
  Emission maps are LDR BC7-sRGB × a float emission scale (the glTF
  `emissive_strength` pattern; pbrt's `scale` texture maps onto it); BC6H is the
  escape hatch for genuinely HDR emission images. Color *constants* in the format
  are linear Rec.709, converted at prep — the same ownership rule as textures:
  storage stays in source space, conversion happens on the way into the renderer.
- **Prep pipeline**: decode → mip-cap downscale → BC encode → DDS cache written next
  to the source (Cycles `blender_tx` pattern); cache hit skips everything. Bindless
  slots are keyed by (canonical path, usage class) — two materials sharing a PNG
  share a slot; a color+data dual use gets two. Cache invalidation is by content
  hash of the source bytes + prep params in the cache header — mtimes break across
  git checkouts, and a hash is free next to a BC encode.
- **Film**: per-AOV accumulation buffers (beauty, albedo, normal, depth), same
  pixel-owned determinism invariant; the published `Frame` carries all four; CLI
  writes one multi-layer EXR. Denoised output is a second, labeled EXR — the raw
  estimator output is never silently replaced (estimator/view split, D-029's line).
- **AOV semantics**: the normal AOV is the world-space *shading* normal
  (post-normal-map — what OIDN wants); depth is camera-space perpendicular z at the
  first hit, lens-sample-averaged (defocused depth matches defocused beauty), f32,
  +∞ on miss. The denoiser guides follow D-062's specular pass-through; depth does
  not. OIDN itself takes no depth input — the depth AOV serves compositing only.
- **EXR layers**: Nuke-safe naming — beauty as bare `R/G/B/A`, depth as the bare
  de-facto-standard `Z`, `albedo.R/G/B`, `normal.X/Y/Z`; no dots in layer names;
  f16 for color layers, f32 for `Z`.
- **Edit channel**: Session inputs gain a pending change-set slot; edits arriving
  mid-wave merge in order and apply at the wave boundary — stop, apply with minimal
  re-prep (material → buffer upload; transform → TLAS; topology → BLAS; lights/env →
  their tables), reset accumulation. Restart-from-zero is the industry consensus
  (MoonRay restarts from sample 0 on any edit).
- **Denoise cadence**: viewer toggle at ~1 Hz, BALANCED quality; CLI `--denoise`
  final frame at HIGH with prefiltered guides (`cleanAux`) — Cycles' split. Caveat
  from the review: the `oidn` crate has no dedicated prefilter call (prefiltering
  is denoising each guide through its own RT filter) — step 9 spikes it; the honest
  fallback is `cleanAux` off, which selects OIDN's noisy-aux trained weights.
- **Transparency policy**: stochastic opacity applies to camera/bounce rays in the
  intersect loop (hash-based RNG, its own named dimension per transparent bounce);
  shadow rays use **deterministic** multiplicative attenuation in trace_shadow —
  Cycles' shipped split (`shade_shadow.h`) — so alpha cards cast correct shadows
  and the shadow kernel stays RNG-free. A transparent-bounce cap (default 8) is
  separate from path depth.
- **Interior tracking**: one current-medium slot in path state, via the schema seam —
  the seam M8's volume/priority stack widens.
- **Scene files** are `.ron`; `cenote-cli import scene.pbrt --out scene.ron` converts,
  and `render` accepts either format directly. Sphere shapes tessellate at import.
- **UVs** join the mesh schema (optional stream; absent means no textured lookups).

## 3. Layout additions

```
crates/
├── cenote/
│   ├── shaders/            # openpbr grows lobes; new: texture sampling + AOV writes
│   └── src/
│       ├── scene/          # scene.rs splits when the schema lands: description,
│       │                   # changeset (ops + apply + dirty), prep (GPU residency)
│       ├── format.rs       # RON round-trip: version field, serde impls
│       ├── ply.rs          # hand-rolled PLY reader
│       ├── texture.rs      # decode → mip-cap → BC encode → DDS cache; bindless table
│       └── denoise.rs      # OIDN host-copy (cfg(feature = "denoise"))
├── cenote-pbrt/
│   └── src/                # lexer, parser, map (pbrt semantics → ChangeSet)
├── cenote-viewer/
│   └── src/                # lookdev.rs: object list + material panel
└── tests/scenes/           # vendored CC0 corpus + goldens; fetch script for tier 2
```

Files earn existence (D-014); this is the expected shape, not a quota.

## 4. Build order (~7–9 weeks at 10 h/wk)

The charter sized M2 at 4–6 weeks; the interviewed scope added the estimator gaps and
the lookdev panel (both charter-consistent, neither in the original line item), so
7–9 is the honest number — §5 lists what slips first.

Each step ends green: compiles, clippy-clean, tests pass, committed.

1. **Plan docs** (this file, deferrals.md, decisions.md entries, README rows).
2. **Change-set schema + RON round-trip** — pure CPU: the value types (Remove ops
   included), apply semantics (get-or-create, validate-then-apply atomicity,
   after-the-set reference resolution, dirty accumulation), serde derives,
   round-trip and apply-order tests; `Scene::demo` re-expressed as a change-set.
   *Checkpoint: the demo scene exists as data on day one.*
3. **Prep rewire + edit channel**: SceneDescription → GPU residency becomes the one
   dirty-driven prep path; Session gains the pending change-set slot and
   stop → apply → restart; viewer loads `.ron` scenes. *Checkpoint: edit a material
   in a file, watch the viewer re-converge. The riskiest step — see §6.*
4. **Estimator gaps**: triangle emitters (alias table over (light, triangle), quad
   path retires), distant + point lights (NEE-only, MIS weight 1), thin-lens DoF.
   *Checkpoint: MIS-agreement and furnace stay green through a light-path rewrite.*
5. **Closure**: one-sample MIS lobe selection with the sampled-lobe path-state tag
   (D-070), variable IOR via analytic Fss through the E/E_avg tables, transmission
   with its embedded 3D glass tables + thin-walled + opacity, coat with darkening
   and base-roughness remap, fuzz LTC, emission in its slot. *Checkpoint: furnace
   matrix extends to the new lobes' white configurations.*
6. **Textures**: UVs through the geometry path, prep pipeline + DDS cache, bindless
   table, constant-or-texture params, in-shader IDT, normal maps. *Checkpoint: a
   textured material in the viewer, edited live.*
7. **`cenote-pbrt`**: tokenizer, parser, semantics mapping with the five fidelity
   traps, vendored corpus + FLIP harness, fetch script. *Checkpoint: first pbrt
   scene renders; CI runs the corpus.*
8. **AOVs**: per-AOV film buffers, specular pass-through guides (two path-state
   fields via the schema seam), multi-layer EXR, Frame extension.
9. **OIDN**: host-copy denoise, CLI `--denoise`, viewer toggle at cadence.
   *Checkpoint: corpus scene at 64 spp, denoised, next to its 4096-spp golden.*
10. **Lookdev panel**: object list, selected object's OpenPBR params as widgets
    emitting change-sets. *Checkpoint: drag coat weight on a pbrt scene, live.*
11. **Polish**: goldens regenerated and eyeballed, module headers current, README
    demo refreshed (side-by-side vs pbrt-v4), decisions.md current. *M2 done.*

## 5. Fallback seams (pre-agreed, in slip order)

- **Fuzz (step 5)** → params parse and store; lobe evaluates as absent. First to go.
- **Normal maps (step 6)** → geometry normals; UV plumbing and BC5 path stay.
- **Lookdev panel (step 10)** → viewer stays read-only; the edit channel is proven
  by file-reload instead.
- **Point lights (step 4)** → distant covers the corpus need.
- **Viewer denoise toggle (step 9)** → CLI-only denoise.
- **Steps 2–3 and 7 are never compressed** — the change-set API and the importer
  *are* the milestone.

## 6. Risk watch

Step 3 carries the unknown-unknowns: it reshapes `scene.rs` and the Session contract
at once, so it lands immediately after the schema exists and before anything depends
on the new prep path. The importer's five fidelity traps are the silent-wrongness
risk — each gets a targeted test: (1) pbrt's photometric light normalization (every
light scale is divided by `SpectrumToPhotometric`, so `L [1 1 1]` is ~1 nit, not
radiance 1); (2) `alpha = sqrt(roughness)` under default `remaproughness`; (3) `fov`
is the full angle of the *shorter* axis; (4) left-handed coordinates and
`ReverseOrientation` XOR handedness-swap on normals and emission side; (5) infinite
lights use equal-area octahedral square images — resampled to equirect at import.
Transmission correctness (interior tracking, refraction goldens through the furnace)
is the closure watch item. Specular fireflies are the *expected* step-7 incident:
no mips + normal maps + low roughness is the recipe, and the pre-agreed answer is
the specular-regularization ledger entry — adopt it explicitly when a corpus scene
demands, don't improvise a clamp mid-step. Build-side: `intel_tex_2`'s ISPC
binaries, the `oidn` crate's runtime library discovery, and the OIDN prefilter path
are checked in spikes before their steps commit to them (their fallbacks:
`ddsfile`-only uncompressed upload; CLI-only denoise; noisy-aux weights).

## 7. Definition of done

- `cenote-cli import scene.pbrt --out scene.ron && cenote-cli render scene.ron
  --spp 512 --denoise` → multi-layer EXR (beauty/albedo/normal/depth) plus a
  denoised EXR that stands next to pbrt-v4's render of the same scene.
- Viewer: open a corpus scene, orbit with DoF, select an object, drag its coat
  weight, watch instant re-convergence; toggle denoise.
- CI: vendored corpus import → render → FLIP green; furnace matrix green through
  the full closure; change-set round-trip, apply-order, and bitwise-determinism
  tests green.
- A stranger can read one `.ron` scene file and understand the entire scene model,
  and read [deferrals.md](deferrals.md) to know exactly what was consciously left
  for later and when it returns.
