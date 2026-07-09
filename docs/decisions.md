# Decision log

Append-only. One dated entry per decision, with enough rationale that future-us (or a
stranger) doesn't have to re-derive it. Charter-level decisions (language, execution
model, sampling theory, milestones) live in [charter.md](charter.md) §2 and are not
repeated here — this log starts where the charter stops: implementation decisions.

Newest entries at the bottom. Reversing a decision gets a *new* entry pointing at the
old one, never an edit.

---

## 2026-07-06 — M0 structural decisions (interview session)

### D-001: Name and crate namespace
**Cenote**; crates are prefixed: `cenote` (core lib), `cenote-cli`, later `cenote-viewer`.
*Why:* distinctive, greppable, portfolio-legible; the future C ABI (M2) gets a natural
`cenote_` prefix. Rejected bare `core`/`cli` dirs as ungreppable and colliding with
Rust's `core` conceptually.

### D-002: M0 is fully headless
Render → EXR on disk → view in [tev](https://github.com/Tom94/tev), which auto-refreshes
on file change. No winit, no swapchain, no surface extensions until the M1 viewer.
*Why:* keeps M0's Vulkan surface compute-only (which is the wavefront architecture's
shape anyway); a debug window would be rewritten in M1 regardless.

### D-003: Shaders live inside the core crate
`crates/cenote/shaders/`, not a workspace-root `shaders/` dir (diverges from the
charter's week-one sketch). *Why:* the core crate is fully self-contained — build.rs,
sources, and kernels travel together.

### D-004: `slangc` subprocess for both compile paths
build.rs shells out to `slangc` and embeds SPIR-V via `include_bytes!`; the runtime
hot-reload watcher shells out to the *same binary with the same flags*. *Why:* one
invocation shape, zero FFI, no build/runtime drift. A failed reload prints diagnostics
and keeps the old pipeline live — never kills the render. In-process Slang API
revisited when reflection-driven pipeline generation matters (M1+).

### D-005: Thin `gpu` module, no RHI
One `gpu` module owns the device context (instance/device/queue/allocator, RAII
teardown) plus purpose-built helpers extracted on the *second* use, never speculatively.
Everything else is direct `ash` at the call site. No traits, no generic resource
system, no render graph. *Why:* the charter locks single-backend Vulkan forever, so
backend abstraction is dead weight; a reader who knows Vulkan should read Vulkan.
Unsafe is quarantined behind `gpu` — code outside it doesn't touch raw `vk` handles.

### D-006: BDA-first binding model
All buffers reached via buffer device address in a push-constant struct; one tiny
descriptor set holds only what can't be an address (the TLAS; later the bindless
texture table). *Why:* scales directly to wavefront SoA path state (Cycles X pattern);
every kernel's data dependencies are visible in one struct at the top of its Slang
file. Descriptor indexing stays enabled-but-unused until textures exist (M2).

### D-007: Blocking one-shot submits in M0
Single compute queue; record → submit → fence-wait; barriers within the command
buffer. No timeline semaphores, no frames-in-flight. *Why:* M0's workload is strictly
sequential; M1's stage scheduler should *drive* the real sync design, not inherit
speculative plumbing.

### D-008: M0 scene is a procedural icosphere + ground plane
Two BLASes, two TLAS instances, fixed pinhole camera, zero file I/O. *Why:* faceted
normal rainbow instantly reveals winding/handedness bugs; two instances exercise
instancing from day one; scene file formats are M2's job.

### D-009: Golden tests via `cargo test` + nv-flip, GPU-gated
Integration tests render and FLIP-compare against checked-in 256² EXR goldens; skip
(not fail) without an RT GPU; failures dump actual + FLIP heatmap to `target/`;
goldens update only via explicit `UPDATE_GOLDENS=1`. *Why:* one-command
discoverability; FLIP-with-threshold survives legitimate FP reordering across
driver/compiler updates where byte-comparison turns to noise.

### D-010: `thiserror` core, `anyhow` bins
Core exposes one coarse `Error` enum (Vulkan, ShaderCompile, Io, NoCapableGpu, …);
variants are refined only when a caller matches on them. Binaries use `anyhow`.
Panics are for programmer bugs only — a missing GPU is an `Err`, never a panic.
*Why:* standard library/binary split; the enum maps mechanically to C error codes at M2.

### D-011: Dependency policy
Every new dependency needs a sentence of justification in the commit adding it;
anything replaceable by <100 lines gets written instead; dependencies land with their
first caller. Approved M0 set — core: ash, gpu-allocator, exr, glam, bytemuck,
thiserror, notify, log; cli: anyhow, clap, env_logger; dev: nv-flip.
*Why (glam):* de-facto Rust graphics standard, mirrors shader vocabulary.

### D-012: Public from first commit; MIT OR Apache-2.0; lean CI
CI on every push: rustfmt check, clippy `-D warnings`, build, non-GPU tests, and (from
the first shader onward) a pinned `slangc` compiling every kernel — shader breakage
fails CI even though runners have no GPU. GPU goldens are a documented local pre-push
ritual. *Why:* the from-scratch commit history is portfolio narrative; public repos
enforce hygiene.

### D-013: Documentation conventions
Root README carries vision + current demo image + repo map, refreshed each milestone.
Every module opens with a `//!` header explaining role and design rationale — skimming
`lib.rs` + module headers = understanding the architecture. This log is append-only.
Lints: rustfmt defaults; clippy pedantic with curated, individually-commented allows;
`missing_docs` warns on public items.

### D-014: Core crate layout
`gpu/` directory = the unsafe quarantine (mod/buffer/submit/accel); domain modules
stay flat and few (`shaders`, `scene`, `render`, `output`, `error`); a module earns a
file only when it exists — no empty homes for future milestones. M1's scheduler and
path state arrive as new top-level siblings of `render`.

### D-015: Leaf defaults
Edition 2024, MSRV = current stable. Right-handed, Y-up, camera looks −Z, meters.
Kernel output is a storage *buffer* of f32 RGBA (readback simplicity; M1 accumulation
wants a buffer anyway). M0 EXRs are linear with no transform (normals are data, not
color — ACEScg enters with actual radiance in M1). Device selection requires
rayQuery + accelerationStructure + BDA + descriptor indexing, prefers discrete, and
fails with a `NoCapableGpu` error listing what each rejected device lacked.
Validation layers on in debug, off in release, debug-utils messenger routed to `log`.

---

## 2026-07-07 — device bring-up

### D-016: Software rasterizers are rejected by device type, not capability
Discovered during step 3: Mesa's lavapipe (llvmpipe) genuinely implements
`VK_KHR_ray_query` + acceleration structures and passes every capability check —
"require ray tracing" does *not* exclude software Vulkan. Selection therefore rejects
`PhysicalDeviceType::CPU` explicitly. *Why:* a software path tracer is out of identity
(the charter's bet is extreme single-GPU performance), and silently "working" on
lavapipe in a GPU-less environment would make golden tests and perf numbers lie.
*Noted trade-off:* this forgoes running real render tests on CI runners via lavapipe;
if that ever becomes attractive, it needs its own decision entry reversing this one.

---

## 2026-07-07 — acceleration structures

### D-017: Geometric normals via buffer fetch, not `VK_KHR_ray_tracing_position_fetch`
The scene keeps every mesh's vertex/index buffers GPU-resident; the primary kernel
looks up the hit triangle's corners through buffer device addresses and computes the
geometric normal itself. The position-fetch extension would return hit-triangle
vertices directly, but adopting it would grow the device baseline beyond the D-015
set for a convenience — and it only covers *positions*: the moment shading needs UVs
or vertex normals (M2), resident geometry buffers are required anyway, so this is
the shape the renderer ends up with regardless. *Trade-off:* slightly more kernel
code and memory traffic in M0.

---

## 2026-07-07 — hot reload

### D-018: Hot reload is a dev-loop feature with a pinned interface
The D-004 no-drift promise is enforced structurally: the `slangc` invocation lives in
one file (`crates/cenote/slangc.rs`) that both `build.rs` and `src/shaders.rs`
`include!` — there is no second definition to drift. Shader source paths are baked
from `CARGO_MANIFEST_DIR` at compile time, so reload works from a source checkout and
a deployed binary just renders its embedded kernels. A reload swaps SPIR-V only: the
entry-point name and push-constant layout stay pinned by the embedded build, so hot
reload covers kernel *body* edits — changing a kernel's `Params` struct means a
`cargo build`, which is also the only correct response since the Rust mirror of that
struct must change in the same commit. *Why:* the alternative (runtime reflection of
recompiled SPIR-V to re-derive layouts) buys generality M0 doesn't need at the cost
of a second pipeline-creation path; revisit alongside D-004's in-process Slang API
when reflection-driven pipelines matter (M1+).

---

## 2026-07-07 — comment style

### D-019: Code comments are self-sufficient — no citations of this log
M0 code originally sprinkled `D-xxx` references through module headers and doc
comments. They forced readers to context-switch into this file for rationale that
belongs (succinctly) at the code site, and they read as noise once the numbered
entries stop being fresh in anyone's head. Comments now carry their own why in a
sentence; this log remains the deep archive with the full trade-off discussions,
discoverable through the README. Amends the D-013 conventions.

---

## 2026-07-07 — M1 structural decisions (interview session)

The plan these seed is [m1-plan.md](m1-plan.md); charter §4 M1 is the parent scope.

### D-020: M1 scope is the full charter list, staged as a walking skeleton
Nothing is cut up front, but the build order (m1-plan.md §4) is a walking skeleton
with pre-agreed fallback seams (§5): HDRI degrades to constant-sky, sliders to
presets; the wavefront core is never compressed. *Why:* the milestone bundles two
different risks — a novel engine and known-territory features — and staging keeps a
schedule slip from becoming a scope panic.

### D-021: Host-driven fixed-loop wavefront scheduler
The host records a fixed stage sequence per bounce for a max-depth number of
iterations, one command buffer per wave. GPU-side per-stage queues hold path indices;
kernels push survivors into the next stage's queue (that push *is* both compaction
and "each path records its next kernel"); every dispatch is indirect, sized by a
counter a prior kernel wrote. No mid-frame CPU↔GPU readbacks. *Why:* satisfies every
charter commitment (stages, queues, indirect dispatch, compaction) with the simplest
correct sync story. Cycles-X-style adaptive kernel selection reads the same counters
— it can layer on later without changing any kernel-facing contract.

### D-022: Fixed-capacity path pool + tile loop
The pool is capped (default 2²⁰ paths, configurable); a sample walks pixel tiles of
pool size. *Why:* path state will reach 100–200+ bytes/path once GRIS data arrives —
pool-per-pixel is 1–2 GB at 4K. Bounding the pool now costs one host-side loop level;
retrofitting tiles after accumulation and viewer code assume path==pixel would be a
structural rework. At viewer resolutions it degenerates to one tile.

### D-023: Path state is SoA behind a schema seam; "reserved" means structural
All path-state fields are declared in exactly one place — a Rust struct of buffers
with a mirrored Slang struct of buffer addresses handed to every kernel — so adding a
field (M3 reconnection vertex, M8 volume stack) is a two-line change no kernel
signature notices. M1 allocates only fields M1 reads; path flags are in (termination
and queue routing need them). *Why:* the charter pillar's intent is "adding these
later must not be a refactor"; allocating placeholder fields would be dead memory and
untestable layout guesses. Granularity: one buffer per logical field, 16-byte-friendly
packing — per-component splits are a measured optimization later.

### D-024: Six kernels — raygen, intersect, shade_miss, shade_surface, trace_shadow, accumulate
Per-bounce sequence `intersect → (shade_miss | shade_surface) → trace_shadow`,
bookended by raygen and a once-per-wave accumulate. Tracing and shading never share a
kernel: intersect stays pure traversal (the layer that later learns procedural
primitives), shadow rays are not inlined into shade_surface. *Why:* inlining is the
tempting collapse ray queries make easy, but it fuses the most divergent work into
one long-running kernel and deletes the exact queue boundaries where M3's GRIS
candidate/visibility passes insert.

### D-025: Sequential waves on one graphics+compute queue, timeline-semaphore pacing
One wave in flight; per wave: stage dispatches → tonemap → present, host paced by a
timeline semaphore (replacing M0's fence-blocked one-shots). The display image is
double-buffered from day one — the seam where present/compute overlap later slots in.
*Why:* path state has one copy, so waves can't overlap anyway; D-007's principle
holds — measured stalls, not speculation, drive concurrency. Async compute and
cross-queue transfers wait for a workload that demands them.

### D-026: Stateless pcg4d sampler with a named dimension registry
Sampling is a hash of (pixel, sample index, bounce, dimension) — nothing stored in
path state beyond keys the path already carries. Dimensions are named constants
(`CAMERA_JITTER`, `BSDF_LOBE`, `NEE_LIGHT`, …), never allocated by call order, behind
a `sample_1d/2d` API. *Why:* replayable-by-construction is the charter's GRIS
requirement — shift mappings re-trace with the same keys and get the same decisions.
Call-order dimensions would make any reordered sample call silently change every
downstream decision. Owen-scrambled Sobol is a pure drop-in behind the same seam
when convergence-per-sample matters.

### D-027: OpenPBR subset is three lobes with constant parameters
Lambert base + conductor GGX (metalness) + dielectric GGX specular layered by
OpenPBR's albedo-scaling approximation; parameters are per-instance constants in a
material buffer (textures are M2). Plain Slang lobe functions (`eval`/`sample`/`pdf`)
combined by a small ubershader evaluator — no Slang interfaces or dynamic dispatch.
Every lobe lands with a white furnace test. *Why:* the smallest set that exercises
MIS against sharp lobes (where MIS bugs actually live) and reads as a real renderer;
transmission is excluded because interior tracking is a disproportionate correctness
burden before M2. Parameters map onto named OpenPBR attributes so M2 grows the set
instead of rewriting it.

### D-028: Area lights are emissive mesh instances; alias-table selection; HDRI is M1's only texture
Quad lights are two-triangle mesh instances inside the TLAS, referenced by a light
list for NEE sampling; selection is a power-proportional alias table built at prep.
The environment is an equirect EXR uploaded as a sampled image (joining the TLAS in
the one small descriptor set — deliberately the only texture before M2's bindless
table), importance-sampled via a marginal/conditional CDF and MIS'd in shade_miss.
*Why:* analytic lights outside the BVH give MIS a second intersection code path to
keep honest; mesh lights make BSDF-sampled hits the ordinary path. The alias table is
~50 lines, chartered prep work, and M3's candidate generation wants it.

### D-029: Authored colors are linear Rec.709, converted to ACEScg at prep; display is an analytic ACES fit
The core is pure ACEScg. Human-authored values (material params, emission) and the
HDRI are taken as linear Rec.709 and converted by one 3×3 matrix at prep/load — the
first instance of the charter's IDT-at-prep pattern. The tonemap kernel applies the
Hill ACES RRT+ODT fit for display; EXRs stay linear ACEScg with chromaticity metadata
in the header. *Why:* every picker, tutorial, and copied reference value lives in
sRGB space — authoring in raw ACEScg makes them all silently wrong. The tonemap
kernel is the seam where M2's OCIO-baked 3D LUTs replace the formula without anything
upstream noticing.

### D-030: Viewer is a new `cenote-viewer` crate — egui on ash, blit presentation, single thread
Core stays windowless; no winit types cross into `cenote`. UI is egui via egui-winit +
egui-ash-renderer (dep-policy justification: an immediate-mode UI renderer is
thousands of lines, nowhere near the <100-line bar; egui is the de-facto Rust choice).
The tonemap kernel writes an offscreen RGBA8 storage image, blitted to the swapchain,
egui pass on top — direct storage writes to swapchain images have spotty driver
support; the blit always works. One winit event-loop thread drives one wave per
redraw; any camera/parameter edit resets accumulation. *Why:* re-convergence after an
edit is the thesis made visible; a render thread is a later optimization with real
ownership costs, not a day-one need.

### D-031: Four-layer test suite — goldens, furnace, MIS-agreement, determinism
(1) Goldens: fixed seed + fixed spp through the full wavefront, FLIP-compared —
D-009's threshold reasoning covers Monte Carlo with a pinned seed. (2) White furnace
per lobe: uniform environment, albedo-1 material, must converge to the environment
value. (3) MIS-agreement: NEE-only, BSDF-only, and MIS renders of one scene must
converge to the same mean — catches wrong-but-plausible weights that goldens would
normalize into the reference. (4) Determinism: same seed twice in-process must be
bitwise identical — the charter's replay guarantee (which GRIS shift mappings depend
on) made mechanical. Plus CPU unit tests for host-shared math (alias table, env CDF,
color matrices, camera rays). All GPU tests skip cleanly without hardware, per the
M0 pattern.

---

## 2026-07-07 — M1 plan review against Cycles X, MoonRay, and current practice

Before implementation began, the locked plan was reviewed against Cycles X's actual
source, MoonRay's paper and open source, and current research. Nine decisions
survived unchanged — several confirmed near-verbatim by Cycles (fixed pool + work
tiles, single-point-of-definition SoA with feature-gated allocation, intersect/shade
separation, zero evidence against the sequential sync model). The entries below
record what the review changed or surfaced.

### D-032: Sampler is hash-based Owen-scrambled Sobol (amends D-026)
M1 ships Sobol-Burley ("Practical Hash-based Owen Scrambling", Burley, JCGT 2020)
instead of a PCG hash. It is stateless and keyed (pixel hash, sample index,
dimension) exactly as D-026 required, ~200 lines (Cycles' whole implementation is
~180), and the production baseline — Cycles' current blue-noise default modes are
Sobol-Burley underneath, and pbrt-v4's default ZSobol is the same hashed-Owen
construction. Better convergence per sample serves the preview-predicts-final thesis
directly, and replayability is *cleaner* than the ReSTIR PT reference code, which
stores raw LCG state in reservoirs and burns dummy samples to keep streams aligned.
The named dimension registry and `sample_1d/2d` seam stand; blue-noise index
ordering (Morton-shuffled offsets, the Cycles/psychopath approach) is the documented
later drop-in. *Why the reversal:* "PCG now, Sobol later" priced the swap wrong —
Sobol-Burley costs roughly a day more now, while swapping later would cost
regenerating every golden.

### D-033: EON diffuse, Turquin energy compensation, spherical-caps VNDF (amends D-027)
Three upgrades, all evidence-forced. (1) The diffuse lobe is EON — energy-preserving
Oren-Nayar (Portsmouth et al. 2024) — because that is the lobe OpenPBR actually
specifies (Lambert is not in the spec), it is analytic and reciprocal, and it passes
the furnace by construction. (2) GGX lobes get Turquin-style multiple-scattering
energy compensation (Turquin 2019) via the Sforza-Pellacini analytic fits (2023,
tens of coefficients — no LUT-baking infrastructure): single-scatter GGX fails an
albedo-1 furnace test *by design*, and compensation is unanimous — Cycles 4.0+
(Turquin), MoonRay (Kulla-Conty), OpenPBR ("should"). (3) GGX sampling is named:
Dupuy-Benyoub spherical caps (HPG 2023) — identical distribution and PDF to Heitz
2018, simpler and faster, what Falcor ships. Bounded VNDF (Tokuyoshi & Eto 2024) is
a documented later option for opaque reflection lobes; it modifies the PDF, so it is
not a silent drop-in. OpenPBR's white-furnace section lists the exact configurations
to test; that list is the M1 furnace matrix.

### D-034: Forward emissive hits, hit encoding, shadow records (amends D-023/D-024 detail)
Cycles dedicates a shade_light kernel to BSDF rays that land on emitters; the plan
was silent. Resolution: shade_surface handles light-tagged instances — evaluate
emission, MIS-weight against the NEE pdf — which makes `prev_bsdf_pdf` a required
M1 path-state field. Folding into shade_surface is right at one-ubershader scale;
the queue boundary exists if it ever earns its own kernel. Two encodings recorded at
the same time: hits are stored as instance + primitive + barycentrics (re-evaluable
— the form M3 reservoirs must hold, per the ReSTIR PT reference's PathReservoir),
and shadow-queue entries are self-contained records (origin, direction, unshadowed
contribution, pixel) rather than main-path fields — simpler now, and already the
shape of the separate shadow-path pool Cycles uses and M3's multi-candidate NEE will
want. The per-bounce sampled-lobe/technique tag GRIS random replay needs is a known
future field; the schema seam makes it a two-line add.

### D-035: Robustness policy — rigorous ray offsets, unconditional finite guard, no default clamp
Self-intersection avoidance uses the rigorous-bounds method from van Antwerpen's
"Solving Self-Intersection Artifacts in DirectX Raytracing" (NVIDIA, 2023; reference
HLSL/GLSL published), with Wächter-Binder (Ray Tracing Gems 2019 ch. 6, Falcor's
choice) as fallback — never magic `TMin` epsilons. Every film contribution is
finite-guarded before accumulation, unconditionally (Cycles' `ensure_finite`).
Firefly clamping ships **off** by default: Cycles defaults indirect clamping to
10.0, but clamping changes the ground truth the thesis promises the artist; the
divergence is deliberate and gets revisited with the M2 denoiser.

### D-036: Interface seams — env pdf query, swappable tonemap
The environment light exposes `sample() → (direction, pdf, radiance)` and
`pdf(direction)` as separate entry points: BSDF-sampling MIS needs the pdf query in
M1, and every ReSTIR target-function and shift-Jacobian evaluation needs it in M3
(it is the piece RTXDI explicitly requires of host tracers). The tonemap is a
swappable stage, not a baked-in look: ACES 2.0 (finalized Sept 2024, in OCIO 2.4.2+)
has no shader-friendly form — the ACES community's own engine guidance is "bake a
3D LUT via OCIO" — and the DCC world is drifting to AgX (Blender's default since
4.0). The Hill fit is the built-in; the LUT slot is where ACES 2.0 or AgX land
without touching anything upstream.

### D-037: SER acknowledged; the wavefront bet stands; intersect is the seam
Since the charter was drafted, Shader Execution Reordering went cross-vendor:
`VK_EXT_ray_tracing_invocation_reorder` was ratified November 2025
(hardware-accelerated on NVIDIA RTX 40/50 and Intel Arc B; AMD committed), and DXR
1.2 SER shipped retail in early 2026. The spec is unambiguous — reordering exists
only in ray-tracing-pipeline raygen shaders, never in compute — so a
compute-wavefront tracer forgoes hardware SER entirely, and NVIDIA's ReSTIR
reference stack (Falcor, RTXPT) is a raygen loop + SER. The wavefront bet stands
anyway: Cenote's profile is Cycles' profile — offline-convergent, feature-staged
(curves/SSS/volumes as inserted stages), one fixed ubershader, divergence living
mostly in traversal that RT cores absorb, ReSTIR multi-pass regardless — and Cycles
remains wavefront. The escape hatch is architectural and cheap to keep true:
intersect is a pure-tracing stage behind a queue boundary, and the EXT's
`hitObjectRecordFromQueryEXT` lets a raygen shader wrap inline ray queries — a
SER-enabled trace stage would be a stage-implementation swap, not a rearchitecture.
This entry exists so the choice stays eyes-open rather than accidental.

---

## 2026-07-08 — GGX energy compensation (step 9 implementation)

### D-038: Albedo fits regenerated for the kernel's exact integrand; separable Smith pinned (amends D-033 detail)
D-033 chose Turquin-style compensation via the Sforza-Pellacini analytic fits.
Implementation surfaced two specifics worth recording. (1) **The published
coefficients underperform on our exact model** — validating them against Monte
Carlo integration of the kernel's own lobes measured up to 2.3% absolute error
for conductors and 9.5% for the glossy layer at IOR 1.5 (their 3-variable fit
spends its capacity across the full reflectivity range; we live on the 0.04
slice). Since the furnace test divides by these values, that error is the
furnace's error. Both fits were therefore **regenerated with their own
methodology** against QMC tables of this kernel's precise integrand: conductors
as a degree-4 rational in (roughness, μ) fit with relative-error weighting
(the compensation factor is 1/E, so relative error is what propagates —
max 1.3%); the glossy layer at fixed IOR 1.5 as a degree-3 rational in
(roughness, √μ) — the √μ warp absorbs the Fresnel rise at grazing that
defeated the unwarped form (max 1.4%, coefficients f32-safe). The full-mixture
white furnace closes to 0.6% worst-case, CPU-validated before any Slang was
written. (2) **Separable Smith G1·G1 is pinned by the fits**: the albedo
tables integrate that exact masking-shadowing form, so swapping in
height-correlated Smith (which reflects *more* energy) would silently turn the
compensation into over-compensation — a furnace that runs hot. Height-
correlated is a later upgrade that must land together with regenerated tables;
the shader comment on `smithG1` says so.

---

## 2026-07-08 — Environment sampling specifics (step 10 implementation)

### D-039: Env CDF weights are 3×3-max dilated; pdfs stored, not differenced; selection is power-proportional (implements D-028/D-036)
D-028 chose an equirect HDRI with marginal/conditional CDF importance sampling;
implementation pinned three specifics, all CPU-validated in the step's Python
prototype before any Slang. (1) **Sampling weights are the 3×3-neighborhood
maximum of texel luminance** (wrapping horizontally, clamping vertically — the
sampler's own address modes), times the row's sin θ. The kernel evaluates
radiance *bilinearly*, so a zero texel adjacent to a bright one still carries
radiance over its footprint; undilated weights give those regions sampling
probability zero, which biases the NEE-only estimator low and breaks the
MIS-agreement invariant exactly along zero/nonzero boundaries (the prototype
measured 3.3M unreachable quadrature points on a test image; the environment
MIS-agreement test pins this with a sun inside a hard-zero band). Slightly
fatter sun selection is the entire cost. (2) **Per-texel pdfs are stored as
their own table** rather than recovered as CDF differences at lookup:
adjacent `f32` CDF entries for dim texels under a 20 000× sun differ near the
representation's spacing, and the subtraction cancels catastrophically —
pbrt's layout, adopted for pbrt's reason. Sample and query read the same
table, so `sample()` and `pdf(dir)` (the D-036 split) agree exactly.
(3) **Environment-vs-quad selection is power-proportional**: quads weigh
π × luminance × area, the environment its luminance integral over the sphere —
dimensionally a flux per unit receiver area, so the comparison stands in a
~1 m² receiver. A heuristic, and deliberately so: selection probability
affects only noise, never the converged image, and both endpoints are pinned
exact (no quads → 1, black environment → 0) because the shader's quad branch
must never run without a light list. Poles report pdf 0 (the equirect
Jacobian is singular there): next-event skips such samples and an escaped
ray's MIS weight becomes 1 — no epsilon, no bias, measure zero.

---

## 2026-07-08 — The demo is a material chart (step 12 polish)

### D-040: Demo spheres form a roughness × metalness grid; the sliders edit the floor (amends D-030 detail)
The m1 plan's demo was a row of spheres sweeping metalness, with viewer sliders
applying roughness/metalness uniformly to the whole row. That arrangement was
self-defeating: dragging the metalness slider flattened the very sweep the row
existed to show, with no way back. Resolution: the demo becomes the standard
material chart — a 5 × 3 grid sweeping `specular_roughness` 0 → 1 left to right
and `metalness` 0 → 1 bottom to top — so the golden pins the whole parameter
plane (including the energy-compensation fits across the roughness range, which
the old golden sampled at a single roughness) and the README still shows the
entire material space at once. The sliders stay, repointed at the floor: they
remain the only demonstration that an in-place scene edit (a GPU material-buffer
update mid-accumulation) restarts the estimator — the lookdev half of the thesis,
and the DoD's "drag material sliders, watch the image re-converge" — and the
floor is the demo's one uniform surface, the only place a uniform edit is
coherent. The quad key light moved up out of the taller frame (and its emission
rose to keep the warm key comparable), preserving the original placement intent:
a visible blown-out quad reads as an artifact, not a light.
