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

---

## 2026-07-08 — Smooth shading normals; the demo lies down and loses its sliders

### D-041: Interpolated vertex shading normals, guarded by the geometric normal
The engine shaded exclusively with geometric normals, which made every sphere
a disco ball: faceting in specular reflections is a discontinuity in the
normal *field*, so no mesh resolution can smooth a mirror — only a smooth
normal field can. `Mesh` now carries one unit object-space shading normal per
vertex (the icosphere's are exact — a unit-sphere vertex is its own normal;
planes carry their face normal), the geometry record carries the buffer's
address, and `shade_surface` interpolates by barycentrics and builds the BSDF
frame on the result. The geometric normal keeps every job that must match
the actual triangle: the van Antwerpen spawn offset, and sidedness guards at
each consumer of a direction — next-event candidates and BSDF-sampled
continuations below the geometric horizon are rejected. That is the classic
shading-normal trade (a sliver of energy lost near silhouettes, never light
through walls), applied identically to every strategy so the MIS-agreement
tests still hold; the furnace tests are untouched by construction, since
their scenes are planes where both normals coincide.

### D-042: The demo chart lies on the floor at 5 × 5; the material sliders are gone (supersedes the slider half of D-040)
The vertical 5 × 3 chart floated its rows to fit a wall of spheres; with
smooth shading the showpiece is reflection, and a chart laid *across* the
floor gives every sphere a grounded contact shadow and a second reading of
the whole sweep in the glossy floor. Now 5 × 5 (roughness left → right,
metalness back → front, the full-metal mirror row nearest the camera), every
sphere resting on the floor, camera raised to separate the rows. The
roughness/metalness sliders — D-040's remaining justification was
demonstrating an in-place GPU material edit — are removed with their whole
machinery (`Scene::set_material`, the host material copy,
`Context::update_buffer`): M1 is complete, the DoD sentence they served is
history, and the live-edit story returns properly with M2's interactive
lookdev. The floor keeps the one good material the sliders defaulted to
(gray, `base_roughness` 0.1, `specular_roughness` 0.15). The viewer's
overlay keeps stats and exposure.

---

## 2026-07-08 — M1 code-review pass; timeline pacing formally deferred

### D-043: Timeline-semaphore frame pacing is deferred, not dropped (records a gap in D-025)
D-025's M1 plan described the render loop as sequential waves under
timeline-semaphore pacing. What shipped is the M0 blocking-submit model
throughout: every wave, every film pass, and every present is one
fence-waited submission, and `submit.rs` / `present.rs` now say so plainly in
their headers. That model is correct and bitwise-deterministic, but it
serializes stages that could overlap and idles the GPU on each fence — the
interactive thesis will eventually be bound by it. The decision: keep the
blocking model for now (it is the simple, obviously-correct baseline, and the
*estimator* — not the frame loop — is what M1 had to prove), and land timeline
pacing, narrowed per-stage barriers, and folding the accumulate/tonemap passes
into the wave submission as one measured performance pass before M3's ReSTIR
demo, where the interactive claim actually needs the frame loop. Recorded here
because this log is otherwise scrupulous about matching the code, and the
render loop was the one place it had drifted.

This entry closes a review of all of M1's code along three axes — readability,
architecture against Cycles X / MoonRay / recent research, and discoverability.
The review found the estimator correct by its existing tests (the furnace
matrix, MIS agreement under both light types, bitwise replay) and needing no
change; the ReSTIR seams the charter promises verifiably exist in the code.
Its other outputs were cleanups, not decisions, and shipped alongside this
entry: the `gpu` raw-handle quarantine is now compiler-enforced (`pub(super)`
instead of `pub`), pass submission moved out of `pipeline.rs` into the
like-named `submit.rs`, duplicated helpers (`image_barrier`, allocation-free,
the `powerHeuristic` MIS weight) were unified, cross-language mirror names were
aligned (`sample_index`, `select_prob`), stale milestone comments were swept,
and the `SceneTable`/`Environment` byte-mirror cross-references were corrected.
Two deeper follow-ups were logged for their milestones rather than done now:
reserving sampler-dimension headroom before more goldens exist, and designing
a per-pixel G-buffer once to serve both OIDN AOVs and ReSTIR neighbor
validation.

## 2026-07-08 — Sampler-dimension headroom reserved (D-043 follow-up)

### D-044: The dimension registry carries reserved headroom, paid for with one golden regen
The first of D-043's follow-ups, done now while only two goldens exist and the
regen is cheap. `rng.slang`'s registry numbered the per-bounce block tight —
five named slots at stride five, camera jitter alone at dimension zero, bounces
starting at one. Any new decision (a dielectric's Fresnel choice, a light BVH's
RIS candidates, a GRIS shift's randoms, or a pre-path lens/time draw for depth
of field and motion blur) would have to insert into that packing and renumber
every dimension after it — and every renumber silently changes every image ever
rendered. The registry now strides both blocks at eight: camera at 0 with room
reserved ahead of the bounces, per-bounce blocks of eight (five named, three
spare). Future decisions claim a reserved slot without shifting the ones after
them. The headroom is free at runtime — each dimension is an independently
hashed-and-scrambled copy of the same Sobol sequence (the padding
construction), so unused slots cost nothing and spacing decisions apart never
correlates them. The one price is a re-scramble of the current decisions'
noise, which moved both goldens; regenerated and confirmed a pure noise change,
not a bias one — the converged 64-spp frame-average shifted 0.42% and the 1-spp
1.9%, the ratio tracking the √-sample noise drop, with no directional offset.

The re-scramble also exposed that the MIS light-sampling agreement tests
(`assert_strategies_agree`) were under-margined: at 64 spp the worst-case sky —
a lone bright sun texel, high-variance for NEE — swings several percent between
sampler realizations, and the old numbering passed the 3% bound partly by luck
(the quad case sat at 2.29%, a whisker under). The new realization tripped it at
5.5%. The fix raises the shared sample budget to 256 spp, which converges the
frame-average enough that the worst case sits near 1% with the 3% bound intact —
so the bound still catches a real bias instead of tripping on noise. This is now
permanent: reserved-slot additions do not renumber, so the realization these
tests and the goldens pin stays fixed until someone deliberately renumbers.

## 2026-07-09 — Film passes folded into the wave submission (D-043 follow-up)

### D-045: Accumulate and tonemap ride the wave's one fence
The performance half of D-043's render-loop note, delivered now (its
timeline-pacing and narrowed-per-stage-barrier parts stay deferred to the
pre-M3 pass). `submit_passes` already records any number of passes into one
command buffer — a full memory barrier between each — and blocks on a single
fence at the end, so appending the film's accumulate and tonemap dispatches to
the wave's own pass list costs no new correctness machinery: only a `Pass: Copy`
derive and a `Wavefront::trace_then(trailing)` seam that concatenates the two
pass lists before submitting. The viewer's per-frame cost drops from three
blocking submissions (trace, accumulate, tonemap) to one; the batch CLI's
per-sample cost from two to one.

Correctness rests on the inter-pass barrier being as strong as the fence it
replaces: it flushes the wave's radiance writes before the accumulate reads
them, and the accumulate's sum writes before the tonemap reads them — exactly
the ordering the fence gave across separate submissions. So the output is
bit-identical. Both goldens pass unregenerated, and a new test
(`folded_frame_matches_separate_passes`) pins the folded viewer path to a
byte-identical display buffer against running the three passes apart. The fold
divides the average by the count *including* its own sample, since that
accumulate lands in the same submission the tonemap reads.

Folding the tonemap carries one viewer cost, taken deliberately: the tonemap
needs the exposure at record time, so the egui UI now runs *before* the combined
submission rather than between the accumulate and the tonemap. The stats it
shows are one frame stale as a result — imperceptible, and the price of the
single fence; the exposure itself still lands the frame it is dragged. The
viewer is vsync-paced (FIFO present), so fewer fences do not raise its frame
rate — the win there is latency. The real throughput win is the batch CLI,
which accumulates back-to-back with no vsync between samples.

## 2026-07-09 — The render loop decouples from the display (architecture)

### D-046: The renderer becomes an actor; the viewer and the future Hydra delegate are peer consumers of a linear frame
The render loop must accumulate as fast as the GPU allows, not at the display's
refresh rate. Today the viewer's redraw is single-threaded and vsync-paced: FIFO
`acquire_next_image` blocks at vblank, so accumulation is pinned to ~60 Hz no
matter how fast a sample is. Cycles X, MoonRay, and Karma all run the path
tracer on a dedicated thread and let the UI *peek* at its output; cenote will
too. The shape below was verified against those renderers, not assumed.

- A `render::Session` in the **core** owns the render thread, the
  `Renderer`/`Scene`/`Film`, and an `Arc<Context>`. It is the synchronization
  boundary. The viewer is its first consumer; the M2 Hydra delegate is a second
  — so the hard concurrency code is written once, in the core, not reimplemented
  per consumer. (Cycles' `Session`, not Blender, owns the loop.)
- Inputs cross in through an `Arc<Mutex<RenderInputs>>` latch (camera, size, a
  `generation` counter, a running flag), latest-wins, snapshotted once per
  sample.
- Output crosses out as the **linear** HDR average, published by a
  double-buffered pointer-swap under a short lock — never a lock held across a
  GPU submit, which would either deadlock against the queue lock or stall the
  render thread for a frame. (Cycles' double-buffered display driver; not the
  triple-buffer mailbox an earlier sketch reached for and this one rejected.)
- The view transform (tonemap + exposure) is the *consumer's*, applied
  downstream of the published linear frame — matching Hydra's `HdRenderBuffer` +
  `HdxColorCorrectionTask` split, and what the batch CLI already does (it writes
  the linear average to EXR with no tonemap at all). This moves the tonemap out
  of the render loop, superseding the viewer-arrangement half of D-045; that
  entry's throughput win for the CLI stands.

Delivered as green, committed batches: (1) the queue becomes a lock-guarded
handle [D-047]; (2) the viewer takes ownership of the tonemap and the `Film`
grows a linear-average resolve target; (3) the `render::Session` thread, the
input latch, and the double-buffered frame; (4) resize and shutdown hardening.

### D-047: The queue is a lock-guarded handle, not a raw `vk::Queue` (implements D-046; batch 1)
`vkQueueSubmit`/`vkQueuePresentKHR` require the queue to be externally
synchronized, yet `vk::Queue` is `Sync`, so nothing stops two threads racing it
once the render thread submits traces while the present thread blits. A
`submit::Queue` newtype wraps `Arc<Mutex<vk::Queue>>` and exposes
`submit`/`submit2`/`present`, each locking *only* around the one Vulkan call;
the fence wait that follows a submit runs with the lock released, so neither
thread blocks the other for a whole GPU frame.

It is a granular cloned handle, symmetric with how `Context` and `Presenter`
already share the allocator (`Arc<Mutex<Allocator>>`) and the device. The
alternative — a bare `Mutex<vk::Queue>` reached through an `Arc<Context>`
back-reference — would have made `Presenter` hold both a context handle and its
own device/allocator clones, three routes to the same object, and it fought
Rust's receiver rules at `create_presenter`. The one submission whose fence wait
is unavoidably inside the lock is the egui texture upload, which submits and
waits internally (`Queue::locked`); those uploads are rare and small. No
behavior change — both goldens pass unregenerated.

### D-048: The estimator ends at a linear average; the tonemap moves downstream to the consumer (implements D-046; batch 2)
D-046's estimator/view split, made concrete while still single-threaded, so
batch 3 only adds the thread. The renderer's output is now a **linear average**,
and the tonemap is a separate, consumer-owned step:

- A new `resolve.slang` kernel divides the film's running sums by the sample
  count into a new `Film` linear-average buffer (which replaces the film's old
  RGBA8 `display` buffer). `Renderer` swaps its tonemap pipeline for the resolve
  pipeline, gains `Renderer::resolve`, and drops `tonemap` and
  `accumulate_and_tonemap`.
- A new `render::Tonemap` type — the tonemap pipeline plus a lazily-sized
  display buffer — is the view transform: exposure, ACES, sRGB, pack. The viewer
  owns one permanently and drives it; the CLI never builds one (EXR stays
  linear). `tonemap.slang` re-points from the sums to the resolved average, and
  its scale drops the `÷ samples` (now the resolve kernel's job) to just
  `exp2(exposure)`.

Resolve is deliberately **separate** from accumulate, not folded into the wave
like D-045's tonemap was: batch 3's render thread accumulates flat out and
resolves only when it publishes, so resolve must not ride every sample. This
supersedes the viewer half of D-045 — the viewer's single fold becomes three
submissions (accumulate, resolve, tonemap), and `accumulate_and_tonemap` and its
`folded_frame_matches_separate_passes` test are gone. The CLI keeps its
trace+accumulate fold via `trace_then`, so D-045's throughput win for the batch
path stands.

One consequence of resolving on the GPU: Vulkan floating-point division is
correctly rounded only to ~2.5 ULP, so the GPU average and the host
`Film::average` readback (the batch EXR) agree to a few ULP, not bit for bit —
the same reason D-045 divided host-side into the scale. That is imperceptible in
a display image and irrelevant to the "same estimator" claim, which rests on the
identical sums, not the final normalize; the new `resolve_matches_host_average`
test asserts a ULP tolerance. The linear estimator itself is untouched, so both
goldens pass unregenerated.

### D-049: The render loop runs on its own thread; the viewer peeks a double-buffered frame (implements D-046; batch 3)
The render loop becomes an actor. A `render::Session` spawns a thread that
owns the `Renderer`, `Scene`, and `Film` and an `Arc<Context>` handle, and
accumulates flat out — no longer paced by the viewer's vsync'd redraw. Two
short-locked lanes cross the boundary, exactly as D-046 sketched:

- **Inputs in** — an `Arc<Mutex<RenderInputs>>` latch (camera, size, a
  `generation` counter, a running flag), latest-wins, snapshotted once per
  sample. The viewer writes the camera on orbit (bumping the generation) and
  the size on resize; the render thread adopts the camera and resets the film
  when the generation moves, and rebuilds the film when the size changes — the
  threaded stand-ins for the old direct `Film::reset` / film-replace. Exposure
  is deliberately *not* in the latch: it stays with the consumer's tonemap.
- **Frames out** — the resolved **linear** average behind a second mutex. The
  render thread resolves into whichever of its two frame buffers no one else
  references (an `Arc` strong count of one) and publishes an `Arc` to it; the
  viewer `peek`s the latest and tonemaps it. The lock spans only the pointer
  hand-off, never a GPU submit.

Two buffers, not the triple-buffered mailbox an earlier sketch reached for: the
render thread resolves only into a buffer with no outstanding reference, so a
slow viewer can never see a buffer torn by an in-flight resolve, and if both
are busy the thread simply skips that publish and keeps accumulating — it never
blocks on the consumer. `Renderer::resolve` now takes the target buffer as an
argument (the pair the session rotates through), so the `Film`'s own
resolve-target buffer from D-048 is gone; the host `Film::average` the CLI uses
is untouched. Publishing is throttled to just under a 60 Hz frame — resolving
every sample would burn GPU time no display can show.

The viewer becomes a thin consumer: feed inputs, `peek`, tonemap, present,
repeat, paced by its FIFO present while the renderer runs ahead. It holds the
last frame across redraws so an exposure drag re-tonemaps it even with no new
render frame. `render.rs` splits into a `render/` directory (`mod.rs` for the
renderer/film/tonemap, `session.rs` for the thread). A GPU-gated test spins a
session up and asserts it publishes frames whose sample count climbs — the
whole actor end to end, which no single-threaded test could reach.

One teardown ordering falls out of the new thread: `Presenter` teardown (and
its swapchain rebuild) waits for the device to idle, which Vulkan requires be
externally synchronized against queue submits. So the viewer drops the
`Session` *first* — joining the render thread stops its submits — before the
presenter tears down. The remaining `device_wait_idle` inside a *resize*'s
swapchain rebuild still overlaps the running render thread; hardening that seam
(and surfacing a render-thread panic through the join) is batch 4.

### D-050: Resize and shutdown hardening for the render thread (completes D-046; batch 4)
Batch 3 left two seams open where the render thread races or vanishes; batch 4
closes both.

**The resize-time device idle.** `Presenter::recreate_swapchain` waits for the
device to idle before destroying the old swapchain, and a *resize* runs that on
the viewer thread while the render thread submits to the same queue.
`vkDeviceWaitIdle` requires every queue be externally synchronized, exactly as
submission does — the queue mutex covered submits but not this wait. So the wait
now goes through a new `Queue::wait_device_idle`, which holds the queue lock
across `device_wait_idle`. It is the one place the lock spans a wait rather than
just the submit call (D-047's rule), and deliberately so: idling the device is
the point, and the render thread's next submit merely waits its brief turn,
which an occasional resize can afford. The presenter's *teardown* idle and the
`Context`'s final idle keep their raw calls — by then the render thread is
already joined (the viewer drops the `Session` first, D-049), so nothing races
them. This does not fix the loose-resize seam itself: the render thread keeps
tracing at the old size across a resize and the presenter's blit rescales the
mismatched frame until the film rebuilds — that visible-for-a-frame stretch is
intended (no cross-thread handshake, no stall), only the wait needed guarding.

**A render thread that fails or panics.** The loop returns `Ok` only when asked
to stop (Drop's job), so a thread that ends on its own has always failed — a GPU
call returned `Err`, or it panicked. Left alone, `peek` would just return `None`
forever and the viewer would freeze on the last frame. `Session::check`, called
at the top of every redraw, joins a thread that has already finished
(`JoinHandle::is_finished`, so it never blocks) and returns its outcome: a
returned error passes straight through, a panic becomes
`Error::RenderThreadPanicked` carrying the payload's message. That travels up
through `redraw` → `handle` into `App.error`, so `main` reports it and exits
non-zero instead of the window hanging. `Session::drop` is the shutdown
backstop: it still joins, logs any leftover error (Drop cannot return one), and
now recovers a poisoned input lock instead of `expect`-ing on it — a panic that
poisoned the lock must not double-panic in Drop and abort before the join can
name it. In practice the locks are held only across trivial `Copy`/move-assigns
that cannot panic, so poisoning is a defensive edge, not an expected path.

No new test: a resize race and a thread panic are both hard to provoke
deterministically without a fault-injection hook the crate doesn't have, and the
existing GPU-gated session test still exercises the actor end to end. The change
is in paths the type system and Vulkan validation now police (the queue lock)
and in an error path that reuses the join the actor already had.

## 2026-07-09 — Second M1 review pass, after the decoupling

### D-051: Review polish, and four deferrals recorded so they read as chosen
A second review of the whole M1 body — estimator, wavefront engine, film,
session actor, viewer — along D-043's three axes (readability, architecture
against Cycles X / MoonRay / current research, discoverability), now
covering everything the decoupling arc (D-044–D-050) added. The estimator
and the architecture came through unchanged: the algorithm choices (EON,
spherical-cap VNDF, Turquin compensation over regenerated fits,
Sobol-Burley padding, van Antwerpen offsets, alias-table selection, dilated
environment CDFs) are current practice, and the actor shape matches the
production pattern it was modeled on. The outputs were polish, shipped with
this entry:

- `Session::peek` became `Session::take_frame` — it consumes (two calls in
  a row answer differently), and `peek` in Rust means it wouldn't. The
  prose still calls the pattern peeking; the method now says what it does.
- `Tonemap` moved to its own `render/tonemap.rs` (with its params mirror
  and CPU-reference test), so the module tree states the estimator/view
  split the docs describe.
- The publish buffers are typed as the pair they are (`[Arc<Buffer>; 2]`,
  from `publish_buffers`), and the film and its pair now rebuild together
  as one value.
- The render thread logs its lifecycle at debug level — start, stop, film
  rebuild, camera adoption — the difference between a one-minute and a
  one-hour diagnosis of "why did the image stop updating".
- A panic whose payload isn't a string no longer reports "render thread
  panicked: render thread panicked".
- Stale comments swept: `FILM_WORKGROUP_SIZE` names all three film kernels;
  `submit.rs`'s header no longer implies the render loop hasn't arrived;
  `upload_environment`'s doc uses the field's post-rename name.
- The session test asserts samples strictly climb (`>`, matching its own
  doc), and `Tonemap::apply` validates its input buffer's size as
  `Renderer::resolve` already did.

Four deferrals were recorded rather than acted on:

- **The strong-count reuse protocol assumes blocking submits.** The render
  thread resolves into a publish buffer whose `Arc` strong count is 1 —
  sound today because every consumer submission blocks before its `Frame`
  drops, so a "free" buffer can have no in-flight GPU reader. The pre-M3
  timeline-pacing pass (D-043) must revisit this invariant along with the
  fences it removes; `session.rs`'s module doc now says so where the
  protocol is defined.
- **Wave tails idle without path regeneration.** Cycles X refills dead
  lanes mid-wave with the next sample's camera rays; cenote ends the wave
  and dispatches near-empty tail rounds. Same pre-M3 performance pass,
  measured before acted on.
- **The viewer session accumulates forever.** No sample cap, no
  convergence idle: a long-converged image still pins the GPU at 100%. A
  `max_samples` input (and possibly Cycles-style publish-interval growth
  and a resolution divider during navigation) belongs to M3's
  interactivity work, where the frame loop is the subject.
- **No firefly clamp, deliberately.** The estimator carries only the
  NaN/Inf guard; direct/indirect clamping is a knob every production
  renderer exposes, but it is a bias knob — cenote adds it when a real
  scene demands it, as an explicit decision, not silently.

## 2026-07-09 — M2 structural decisions (interview session)

Locked after a sourced research pass over Cycles X, MoonRay/RDL2, pbrt-v4, the
OpenPBR v1.1.1 spec, and OIDN 2.5. The working plan is
[m2-plan.md](m2-plan.md); the consciously-not-yet options are in
[deferrals.md](deferrals.md) (see D-067).

### D-052: C ABI deferred to M4; M2's boundary is the pure-Rust change-set API
Amends the charter's M2 line, which named the C ABI here. *Why:* the research
settled what the ABI's real job is — transporting *serialized change-sets*, not
exposing per-attribute setters. MoonRay ships no C API at all; its process
boundary is RDLMessage (a serialized delta: manifest + payload + sync id).
Freezing an ABI before its first real consumer (the M4 render server + Hydra
delegate) exists would lock in guesses; the text format proves the
serializability the ABI will rely on. M2's API consumers are the importer, the
CLI, and the viewer — all in-process Rust.

### D-053: Static typed scene schema, closed kind set, named objects
Object kinds (mesh, instance, material, light, camera, environment, render
settings) are ordinary Rust types; objects carry string names resolved to
handles; adding an attribute is a code change. *Why:* RDL2's runtime attribute
registry earns its complexity by serving a plugin SDK — a charter non-goal.
A closed schema gets exhaustive `match`, serde derives, and compiler-checked
refactors for free. The condition that would revive the dynamic option is
recorded in deferrals.md.

### D-054: Change-sets are first-class values; `apply()` is the only mutation path
A change-set is an ordered list of typed patches — one `Option` per attribute —
with get-or-create-by-name semantics on apply. Applying is the *sole* way a
SceneDescription changes, and it accumulates the dirty state (material /
transform / topology / lights / environment) that drives minimal re-prep.
*Why:* RDL2's load-is-a-delta insight — get-or-create makes loading a file and
applying an edit the same operation, so the file format, the future wire
format, and the viewer's edit stream are one code path with one dirty-tracking
story. The builder API (`set.material("floor").base_color(…)`) keeps call
sites readable.

### D-055: Scene text format is RON via serde, version field first
*Why:* the serde derive on the schema *is* the parser — format and schema
cannot drift. RON reads like Rust literals (right for a Rust-shaped schema)
and diffs cleanly. `serde` + `ron` clear the D-011 bar: a hand-rolled parser
of this schema is 400+ lines that must be updated with every schema change.
A binary codec for the M4 wire is a drop-in later because it serializes the
same value (deferrals.md).

### D-056: Bulk geometry inline or by PLY reference; hand-rolled PLY reader in core
The mesh op's payload is an enum: `Inline { positions, normals, uvs,
triangles }` or a relative-path PLY reference. Environments stay
EXR-by-reference. *Why:* small scenes stay single-file and diffable; heavy
geometry stays in the format the corpus already uses (pbrt scenes are
dominated by `plymesh`). The PLY reader is ~200 lines of well-specified
format — under the D-011 write-it-yourself bar — and lives in core because
cenote's own format references PLY, not just the importer.

### D-057: pbrt-v4 importer — the subset, and the five fidelity traps
Supported: `trianglemesh`/`plymesh`/`sphere` (tessellated at import)/
`ObjectInstance`; `diffuse`/`coateddiffuse`/`conductor`/`dielectric`/
`thindielectric`; `area`/`infinite`/`distant` (+`point`) lights;
`imagemap`/`constant`/`scale` textures; `perspective` camera. Everything else
warns by token name — silence never means "handled". Fidelity commitments,
each with a targeted test: (1) photometric normalization — pbrt divides every
light scale by `SpectrumToPhotometric(L)`, so `rgb L [1 1 1]` is ~1 nit, and
RGB illuminants are D65-tinted; (2) `alpha = sqrt(roughness)` under the
default `remaproughness`; (3) `fov` is the full angle of the *shorter* image
axis; (4) left-handed coordinates, with `ReverseOrientation` XOR
transform-swaps-handedness flipping normals and emission side; (5) infinite
lights use square equal-area *octahedral* images, resampled to equirect at
import. *Why this subset:* it covers the real corpus (pbrt-v4's published
scenes are overwhelmingly triangle meshes with these materials); each skip
maps to the milestone that makes it honest to support (deferrals.md).

### D-058: Estimator gains triangle emitters, delta lights, and thin-lens DoF
Triangle emitters replace the quad special case: the alias table is built over
(light, triangle) pairs weighted by area × power, sampling is uniform on the
triangle, and the parallelogram path retires. Distant and point lights are
NEE-only delta lights with MIS weight 1 (a BSDF sample hits them with
probability zero). Thin-lens DoF adds two named RNG dimensions and a lens
sample in raygen; pinhole is radius 0. *Why:* imported scenes need all four,
and M3's many-light work wants one general emissive-geometry path — growing
the quad hack would create exactly the second code path M1's light design
avoided (D-023's reasoning, continued).

### D-059: The full-look closure cut — six additions, five precedented deferrals
Added: coat (GGX with base-IOR remap η_s and the analytic darkening factor
from the spec), fuzz (Zeltner LTC sheen via the published 32×32 tables),
transmission (rough dielectric BTDF, Beer–Lambert interior via
μ_t = −ln(T)/λ, one current-medium slot in path state), thin-walled mode,
variable specular IOR (the Turquin energy-compensation fits gain an IOR axis
— Cycles' pattern), stochastic opacity in the intersect loop, and emission in
its OpenPBR stack position (coat-attenuated: L_e = lerp(1, T_coat, C)·E).
Deferred with shipping precedent (deferrals.md): SSS random walk (M7 —
`subsurface_color` degrades onto diffuse, the MaterialX-shadergen fallback),
dispersion, thin-film, anisotropy, transmission scatter. *Why:* this is
OpenPBR's own renderer-ready decomposition — the spec's slab tree flattens to
a lobe mixture with closed-form weights, and this cut is the portion whose
energy story we can prove in the furnace matrix now.

### D-060: Mip policy — cap at prep, one BC level, hardware bilinear
The mip-cap downscale happens at prep; exactly one BC level uploads; sampling
is hardware bilinear at LOD 0. *Why:* Cycles shipped this shape for 15 years
(mipmapping arrived only with the 2026 texture cache), and the estimator makes
it sound: jittered accumulation integrates the pixel footprint, so the
converged frame is unbiased — mip selection is a bandwidth optimization, not a
correctness feature. Ray-cone LOD and full chains are recorded for the
pre-M3 measured perf pass (deferrals.md).

### D-061: Texturable parameter set + tangent-space normal maps
Texturable: base_color, specular_roughness, metalness, emission, opacity.
Normal maps: tangent-space, BC5 two-channel with in-shader Z reconstruction,
per-hit UV-derived tangents, horizon-clamped perturbation. pbrt bump and
displacement are skipped with import warnings. *Why:* this is the set the
corpus actually uses, and normal maps are the highest look-per-line feature
in the whole milestone; authored-tangent quality work belongs with anisotropy
(deferrals.md), and displacement belongs to M5's geometry depth.

### D-062: Denoiser guides — Cycles-style specular pass-through AOVs
Albedo and normal guides pass through near-specular hits (roughness ramp
0–0.15), recording what mirrors and glass *show* rather than their own
surface; implemented as two path-state fields (feature throughput + written
flag) via the schema seam. Albedo/normal/depth accumulate in separate
pixel-owned film buffers — never atomics — preserving the bitwise-determinism
invariant. *Why:* OIDN's own guidance and Cycles' shipped behavior: a mirror
whose guide says "flat gray" denoises to mush; the ramp avoids a hard
roughness cliff in the guides.

### D-063: OIDN via host-copy, behind a `denoise` feature
Download resolved beauty + albedo + normal, run OIDN's DEFAULT device through
the safe `oidn` crate, upload the result. CLI `--denoise` runs final-frame at
HIGH quality with prefiltered guides (`cleanAux`); the viewer toggle runs
~1 Hz at BALANCED — Cycles' cadence split. Denoised output is a second,
labeled EXR; raw estimator output is never silently replaced. *Why:* OIDN has
no Vulkan device, so zero-copy means exported VkDeviceMemory + external
semaphores — machinery that belongs to the timeline-semaphore pass (D-043),
recorded together in deferrals.md. The feature gate keeps the heavy native
dependency out of default builds.

### D-064: Lookdev panel — the change-set API's first interactive consumer
The viewer loads scenes (.ron or .pbrt), lists objects, and exposes the
selected object's OpenPBR parameters as egui widgets that emit change-sets
into a Session edit channel: pending edits merge in order, apply at the wave
boundary (stop → apply with minimal re-prep → restart from sample 0). No
gizmos, no transform editing, no creation UI (deferrals.md: M4's usdview
supplies authoring wholesale). *Why:* this closes the loop the milestone
exists to prove — an edit path from UI event to converged pixels through the
same value type the file format serializes; restart-from-zero is the
industry consensus (MoonRay restarts on any edit), and instant re-convergence
is the thesis demo (D-042's promise lands here).

### D-065: Tiered regression corpus
Tier 1: 3–4 small CC0 pbrt scenes vendored under `tests/scenes` with goldens;
CI imports, renders, and FLIP-compares every run. Tier 2: a checksummed fetch
script for showcase scenes (bathroom-class) — never in the repo, never in CI.
The corpus README pins the reference pbrt-v4 commit and states the caveat:
pbrt renders spectrally, cenote in RGB ACEScg, so comparisons are perceptual
(FLIP), not pixel-exact. *Why:* hermetic fast CI with real end-to-end
coverage, showcase weight kept out of the clone, and the honesty caveat
written where a future reader will trip over the difference.

### D-066: `cenote-pbrt` is its own leaf crate
`.pbrt` in → `ChangeSet` out, consuming only cenote's public API; the pbrt
tokenizer/parser is hand-rolled there (~straightforward recursive descent over
a well-documented grammar — under the D-011 bar). Core's new dependencies:
`serde`, `ron`, `image` (PNG/JPEG decode), `intel_tex_2` (ISPC BC encoders),
`ddsfile`, `oidn` (feature-gated) — each cleared against D-011 in the plan.
*Why:* the importer is a *client* of the scene API, and the crate boundary
mechanically enforces that the public API is sufficient — the same forcing
function the M4 ABI will need, two milestones early and for free.

### D-067: The deferral ledger
[deferrals.md](deferrals.md) now holds every consciously-deferred production
solution — what we do instead today, the production shape, and the trigger
that revives it — including the four D-051 deferrals, which are carried there
unchanged. Unlike this log it is not append-only: picking up a deferral moves
the entry into a new dated decision here and deletes it there. *Why:* the
interview repeatedly produced "right long-term answer, too much now" options;
scattered across decision entries they rot, and a single living ledger turns
each trigger firing into a plan we already made rather than a rediscovery.

## 2026-07-09 — M2 plan review (adversarial, sourced)

The locked plan got the D-043 treatment before any code: three parallel review
tracks attacked it against Hydra's delegate requirements + MoonRay's actual
API surface, Cycles' shipped kernel source, and 2024–2026 research. No
decision was reversed; the findings were two missing decisions, one wrong
mechanism, a set of format-freezing fields, and seven deferral-ledger
entries — folded into [m2-plan.md](m2-plan.md) §1b/§2 and
[deferrals.md](deferrals.md) with this batch. Notably, every correction made
M2 smaller or safer: the review is why the plan is trusted, not a tax on it.

### D-068: The review itself, and what held
Method: one track per reference body, each instructed to *break* decisions,
not affirm them. Confirmed sound without amendment: C ABI deferral (D-052),
static schema (D-053), RON (D-055, with hygiene notes), PLY-by-reference
(D-056), triangle emitters/delta lights/DoF (D-058), mip-cap policy (D-060),
specular pass-through guides (D-062), OIDN host-copy (D-063), the edit
channel's batch-per-wave shape (D-064, matching hdMoonray's UpdateGuard
pattern), and the cenote-pbrt crate boundary (D-066). Checked and discarded:
OpenPBR 1.2 fields to reserve (spec is v1.1; in-progress additions have
unfinalized names), DLSS-RR-class denoising (covered by the temporal-denoise
deferral), pbrt-v4's stochastic layered BxDF (not the GPU shape).

### D-069: Change-sets gain Remove ops (amends D-054)
`Remove(kind, name)` joins the op set in step 2, with dirty semantics that
retire GPU residency (BLAS slot, light-table entry, texture references),
even though the M2 viewer never emits one. *Why:* the get-or-create + patch
set was RDL2's exact shape — including its most infamous wart: RDL2 cannot
delete objects, and hdMoonray fakes removal with visibility flags. Hydra,
our M4 milestone, requires real deletion (`DestroyRprim` is a mandatory
render-delegate virtual; renames arrive as remove + re-insert). Designing
residency retirement now is cheap; retrofitting deletion into an API whose
dirty tracking and file format assumed append-only is the expensive path.
With Remove in the schema, the identity contract is complete: names are
stable identities, rename = remove + create.

### D-070: Lobe selection is one-sample MIS with a path-state lobe tag
shade_surface picks one closure per bounce proportional to its
albedo-estimate weight via a CDF — rescaling the used random number to
preserve stratification — then evaluates *all* lobes and combines pdfs as
the one-sample balance heuristic `pdf = Σ(pdfᵢ·wᵢ)/Σwᵢ`. The sampled-lobe
tag becomes an M2 path-state field. *Why:* the closure grows from three
lobes to ~seven and the plan never said how one gets picked — this is
Cycles' shipped answer, verified in `surface_shader.h`
(`surface_shader_bsdf_bssrdf_pick` + `_surface_shader_bsdf_eval_mis`). The
lobe tag pays three times: it drives D-062's specular pass-through ramp, it
is the per-bounce technique record M1 earmarked for M3's GRIS random replay,
and it makes sampled-lobe debugging visualizations free.

### D-071: Energy compensation via E/E_avg tables + analytic Fresnel (amends D-059)
The interviewed "Turquin fits gain an IOR axis" was the wrong mechanism.
Verified in Cycles `bsdf_microfacet.h`: reflection lobes use Fresnel-free
directional-albedo tables E(roughness, cosθ) + E_avg(roughness), with
Fresnel entering analytically in the multiple-scattering term
`Fms = Fss·E_avg/(1 − Fss(1 − E_avg))` — closed-form `Fss` for both
dielectrics and conductors, so variable IOR costs *no* table axis on
reflection. Only the coupled reflection+refraction (transmission) lobe needs
IOR-dependent tables: 3D glass tables (roughness × cosθ × IOR-remap
√|(η−1)/(η+1)|, separate η<1 branch), 16³–32³ f32 baked offline and embedded
(≈16–128 KB; Blender's 2025 furnace fix showed 32³ needed at high
roughness). Coat reuses the same tables — darkening stays analytic — and
gains the spec's base-roughness remap under nonzero coat roughness, added
here so it isn't discovered in a conformance diff. *Why this is a win:*
less work than inventing an IOR-axis fit, furnace-provable, and exactly the
shipped shape of the renderer we benchmark against.

### D-072: Format-freezing fields locked before the schema ships
Four things that would each cost a format version bump if discovered after
step 2: (1) the native format commits to the code's conventions — Y-up,
right-handed, meters, vertical-fov degrees — stated in the schema module
doc, and the pbrt importer converts *into* them (including shorter-axis fov
→ vfov through the tangent when aspect < 1); (2) the camera op carries full
orientation (pbrt `LookAt` can roll; a position+look_at schema silently
drops it) plus `focus_distance` and `aperture_radius` — the fields D-058's
thin-lens DoF requires, which existed in no schema; (3) texture references
carry a color-space field: slot-derived default (color slots sRGB for 8-bit
inputs, linear for float; data/normal always linear) with an explicit
override — someone must own sRGB-vs-linear, and pbrt's rules need a target
to map onto; (4) emitters carry `camera_visible` (default true, matching
pbrt) — lookdev always wants invisible lights, and the full per-ray-type
set is ledgered. Companion contract sentences recorded in m2-plan §2:
validate-then-apply atomicity (a mid-set failure leaves the description
untouched), after-the-set name resolution (forward references legal),
scene-file-relative paths.

### D-073: Review leaf defaults and the ledger's seven additions
Leaf defaults recorded in m2-plan §2, each from a review finding:
shadow-ray transparency is *deterministic* multiplicative attenuation in
trace_shadow while bounce rays use stochastic pass-through (Cycles'
`shade_shadow.h` split — alpha cards cast correct shadows, the shadow
kernel stays RNG-free), with a transparent-bounce cap separate from path
depth; depth AOV = camera-space perpendicular z at first hit,
lens-sample-averaged, +∞ on miss — and OIDN takes no depth input, so the
AOV serves compositing only; EXR layers use the Nuke-safe convention (bare
`R/G/B/A` beauty, bare `Z`, no dots in layer names, f16 color / f32 depth);
emission maps are LDR BC7-sRGB × float emission scale with BC6H as the HDR
escape hatch; bindless slots key by (canonical path, usage class) and the
DDS cache invalidates by content hash, not mtime; the importer subset gains
`disk` (the killeroo scene family uses one; ~20 lines beside the sphere
tessellator); the corpus bar is "permissively licensed, license text
vendored" — strictly-CC0 pbrt scenes barely exist (amends D-065's wording);
RON is version-pinned and the schema avoids `untagged`/`flatten` (its
documented weak spots); the OIDN prefilter path is spiked in step 9 (the
Rust crate has no dedicated prefilter call; noisy-aux weights are the
honest fallback). Ledger additions (deferrals.md): specular regularization
(Filter-Glossy path regularization + Tokuyoshi–Kaplanyan specular AA — the
one deferral whose trigger is *expected to fire during M2 step 7*, so the
mechanism is pre-agreed), UDIM + multiple UV sets, neural texture
compression (RTXNTC is public beta — watch, don't build), per-ray-type
visibility flags, cryptomatte/object-ID AOVs, array instancer op, and
deform-only BLAS refit.

## 2026-07-09 — M2 step 2: the change-set schema lands

### D-074: Leaf decisions made while implementing the schema
The schema shipped as planned (D-052…D-056, D-069, D-072); these are the
calls the code forced that the plan hadn't spelled out. (1) *Format color
constants are linear Rec.709*, converted to ACEScg at prep — extending
D-072's texture color-space ownership rule to constants: storage stays in
source space, conversion happens on the way in. The demo change-set carries
raw authored values where the procedural builder converts in code. (2)
*Relative paths are rejected at apply*: `format::load` rebases against the
scene file's directory and is the only place a relative path gains meaning,
so the CWD can mechanically never leak into resolution. (3) *Unknown fields
are parse errors* (`deny_unknown_fields`): a typo'd parameter silently
skipped would be a wrong render with no error message — the worst outcome a
scene format can produce. The compatibility cost is nil because the version
field owns compatibility. (4) *No RON extensions*: `implicit_some` would
serialize the `Some(None)` patches that clear an optional field (normal
map, focus distance) as plain `None` and collapse two distinct meanings —
explicit `Some` everywhere is uglier and correct. (5) The settings field is
named `max_bounces` — the engine's actual quantity (`DEFAULT_MAX_BOUNCES`
is its default) — rather than the plan's looser "max depth". (6) Delta
lights patch *wholesale* (a light is a handful of numbers whose variant is
its identity) and state their radiometry in the schema: distant carries
irradiance (W/m² facing), point carries intensity (W/sr) — the pbrt
importer converts *into* these, keeping trap #1 (photometric
normalization) in one place. (7) `camera_visible` sits on the *instance*
(visibility is a placement property, and area emitters are instances), not
the material. (8) All seven kinds are uniform named maps; "exactly one
camera and settings, at most one environment" is prep's constraint at
render time, not the description's — Hydra delivers multi-camera scenes,
and the description shouldn't pre-reject them. (9) Apply is
clone-validate-swap: ops merge into a copy, validation sees the post-set
state (forward references legal by construction), and only a fully valid
outcome replaces the original — atomicity that is trivially correct, with
payload sharing (Arc) as the known optimization if lookdev edit-rate
profiling ever asks. Dirty state is two name sets — `changed` (rebuild)
and `removed` (retire, idempotent) — where a newer removal supersedes an
older change but remove-then-recreate keeps both.

## 2026-07-09 — M2 step 3: the prep rewire and the edit channel

### D-075: Leaf decisions made while rewiring prep
SceneDescription → GPU residency is now the one dirty-driven prep path
(`Scene::prep` fresh, `Scene::update` incremental), the Session carries the
edit channel, and the viewer loads and watches `.ron` scenes; `Scene::demo`
is `ChangeSet::demo` prepped, and the goldens passed *unregenerated* — the
data path renders the image the procedural builder did. The calls the code
forced: (1) *Prep errors split into recoverable vs fatal by construction*:
everything that can fail on user data (decodes, capability checks, shape
rules) runs host-side before the first GPU call and returns `Error::Scene`,
which guarantees residency untouched — so a live session keeps rendering
its last good scene through a bad edit; any other error is a device fault
and ends the render thread. Dirt whose re-prep was rejected is retained and
retried after the next applied edit, so nothing goes silently stale. (2)
*File reload is replace-diff, not overlay*: `SceneDescription::replace`
computes dirty as the per-object difference against the incoming
description, so deleting an object from a scene file retires its residency
(exercising D-069's removal semantics with a real client) and re-saving an
untouched file rebuilds nothing. `Session::apply` keeps the overlay shape
for the lookdev panel. (3) *Apply's dirty accounting is equality-gated*: a
patch that lands values already in place dirties nothing (creation always
dirties), so redundant edits force no re-prep and no accumulation restart.
(4) *Unwired features warn by name and render without* — textured slots,
delta lights, aperture, `camera_visible = false`, non-default closure
params — gated on the dirty set so a long edit session doesn't repeat
itself; things with no honest render (PLY geometry until the reader lands,
non-quad emitters until triangle emitters, singleton violations, an
environment that won't decode) are hard `Error::Scene`s. (5) *Rebuild
granularity*: per-name BLAS on mesh dirt, TLAS on any mesh/instance dirt,
environment image + tables on environment dirt; the small buffers
(geometry records, materials, lights, scene table) rebuild on any dirt —
they're the cheap tail, and light indices live inside geometry records
anyway. The scene retains the environment's power so a light edit can
recompute the NEE selection probability without reloading the image. (6)
*The core camera gained `up`* — D-072 committed the format to roll, and
honoring it is one basis change; aperture/focus stay warned until step 4's
thin-lens work. A camera edit snaps the view to the authored pose, but a
non-camera edit never touches the interactive camera. (7) *Settings-only
edits don't restart accumulation* (no residency, no visual change), though
they still validate — a second settings object is caught at update. (8)
Prep-time singleton rules (exactly one camera and settings, at most one
environment, at least one instance) live in prep per D-074(8); no
environment means a black sky, degenerating NEE cleanly to the quads. (9)
`Scene::new` (procedural objects + any `Environment`) survives as the
estimator-test path — furnace tests need constant-radiance environments and
exact GPU materials no scene file can express; it shares every assembly
helper with prep, so the two can't drift. `scenes/example.ron` is the
repo's hand-written walkthrough scene, pinned by a format test so it can't
rot. Ledger additions (deferrals.md): sampler seed wiring; the
`camera_visible` kernel gap folds into the existing per-ray-type visibility
entry.

## 2026-07-10 — M2 step 4: the estimator gaps

### D-076: Leaf decisions made while closing the estimator gaps
Triangle emitters retired the quad special case, distant and point lights
joined the estimator, thin-lens depth of field landed in raygen, and the
`camera_visible` flag got its kernel wiring — the deferral ledger had
scheduled that wiring for exactly this step, and the trigger fired. The
calls the code forced: (1) *One alias table for every light kind*: emissive
triangles and delta lights are records of a single power-proportional
Walker/Vose table, distinguished by a `kind` tag, so next-event estimation
keeps one selection path (M3's many-light work replaces the table, not the
shape). Per-kind power measures are frankly approximate — a triangle weighs
one face's exitance flux, a distant light its flux onto the environment's
conventional ~1 m² receiver, a point light its whole 4π sphere — and that
is fine: selection probabilities only steer noise, never the answer. (2)
*Hit-side pdf lookup is base + primitive*: an emissive instance gets one
record per triangle, contiguous in primitive order, and its geometry record
stores the first record's index — so a BSDF-sampled hit finds the exact pdf
its MIS weight competes against in O(1). Degenerate triangles keep their
slot (the indexing depends on it) with selection probability zero. (3)
*Shadow-ray identity extended to (instance, primitive)*: a ray aimed at a
point on a triangle meets that triangle's plane once, so the identity test
stays epsilon-free — and becomes *exact on closed emitters*, where a ray
toward a far-side sample hits the near side of the same instance and must
count as occluded. The sphere-emitter MIS-agreement test was verified to
catch the instance-only version (NEE-only biases high). Point lights bound
the shadow ray by the exact distance instead — the light is not geometry,
so anything committed nearer is a real occluder and no epsilon is needed.
(4) *Delta lights are NEE-only with MIS weight 1* (a BSDF sample hits zero
area with probability zero), which means BSDF-only mode cannot see them —
documented on `LightSampling`, and their correctness is pinned analytically
instead: a straight-down distant light on a white Lambert plane must render
exactly (albedo/π)·E per sample, a hoisted point light exactly
(albedo/π)·I/r². (5) *The thin lens is `Option<Lens>` on the core camera*
(pinhole = `None`, matching "aperture 0 is a pinhole" in the format);
prep resolves an unset `focus_distance` to |look_at − position|. The host
pre-scales the ray basis by the focus distance so `forward + x·right +
y·up` *is* each pixel's focal point, and raygen re-aims from a
concentric-disk lens sample (`CAMERA_LENS`, the reserved pre-path RNG
dimension) — the pinhole path is untouched down to the bit. A thin-lens
white furnace pins that the lens sample carries weight 1. (6)
*`camera_visible` is a TLAS mask bit*: camera rays (bounce 0) trace with
the camera bit, everything else with all bits, so an invisible emitter
still illuminates, occludes, and reflects — the intersect kernel gained a
per-bounce ray mask and the first real per-ray-type visibility bit; the
full set stays on the ledger. (7) *The viewer's orbit camera carries the
authored lens* through every move — orbiting holds the subject distance,
so authored focus stays meaningful. (8) The goldens were regenerated and
eyeballed: same scene, same light, a different (equally valid) noise
realization — the 64-spp golden had in fact survived unregenerated, only
the 1-spp realization moved past the FLIP bound.
