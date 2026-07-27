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

## 2026-07-10 — M2 step 5: the closure

### D-077: Leaf decisions made while building the full OpenPBR closure
The closure grew from three lobes to the full D-059 cut — coat, fuzz,
coupled glass with interior absorption, thin-walled mode, variable IOR,
stochastic opacity, emission in its stack slot — under D-070's one-sample
MIS and D-071's energy mechanism, and the furnace matrix now closes over
every white configuration of the new set. The calls the code forced:
(1) *The energy tables are baked in-repo, against this kernel's exact
integrand*: a committed QMC baker in `tables.rs` (regenerated by an
ignored test, spot-checked by fast ones that re-integrate entries from
scratch) produces the Fresnel-free E/E_avg reflection tables and the
16³ glass tables — roughness × cosθ × √((η−1)/(η+1)), a separate branch
for η < 1, exactly Cycles' shipped shape — replacing M1's rational fits.
Compensation is Cycles' verbatim: a multiplicative 1 + Fms·(1−E)/E on the
single-scatter lobe with the pdf never scaled, Fms = Fss·E_avg/(1 −
Fss·(1−E_avg)), and closed-form average Fresnels (Kulla-Conty's dielectric
fit, Schlick's F0 + (1−F0)/21). (2) *The layering albedo earned its own 3D
table*: D-071's "no IOR axis on reflection" holds for compensation, but
albedo-scaling needs the compensated dielectric reflection lobe's true
directional albedo at arbitrary IOR — baked from the finished lobe
(single-scatter with exact Fresnel times its own scale), so `1 − E_spec`
hands the base exactly what the interface passes and the furnace closes at
every IOR by construction. Cycles' `ggx_gen_schlick_ior_s` is the same
table wearing an interpolant. (3) *Fuzz is vendored data, not baked*: the
Zeltner–Burley–Chiang LTC volume fit (Apache-2.0, tizian/ltc-sheen
@9262411, full precision) — the table *is* the BSDF's definition, its
degenerate cells (A = 0 where reflectance ~0) read as "no fuzz lobe here",
and the kernel reproduces the published eval/sample verbatim, including
the rotation into the view-aligned frame and the density serving as both
lobe shape and perfect sampling pdf. (4) *Four techniques, and zero energy
means zero weight*: fuzz, coat, base-specular (whose rescaled technique
leftover drives the glass reflect-or-refract branch), and diffuse, picked
by tabulated-albedo weights and combined as the balance heuristic
Σwᵢpdfᵢ/Σw. A material whose interface carries no energy (specular_weight
0, no metal, no transmission) drops the interface lobes *entirely* —
otherwise the table lerp's overestimate of a ~0 F0 puts a stray weight on
the near-specular pdf spike, and the pure-diffuse furnaces lose their M1
per-sample exactness (found by exactly that test). (5) *Refraction uses
the camera-path adjoint convention* — Walter's BTDF without the
solid-angle-compression factor — so a VNDF sample's weight is exactly
(1−F)·G1, the quantity the glass tables integrate; ledgered for
bidirectional transport. Interior tracking is by instance identity: the
medium slot (packed with the lobe tag) says whose interior the segment
crossed, `medium == hit` means exiting at 1/η through the inverted-branch
tables, Beer–Lambert folds into throughput on arrival, and the spawn
point's error-bound offset now also nudges *below* the surface for
transmitted continuations. Next-event connections may cross the interface
(both-hemisphere eval), so refracted transport keeps matching strategy
support — the MIS-agreement scene gained a glass ball to hold that. (6)
*Thin-walled follows Cycles*: the two-interface Fresnel series
pre-integrated at the view angle, transmission modeled as a mirrored
reflection with Kulla-Conty's double-refraction roughening — with one
deviation: both lobes recover their full multiple-scattering energy
(Fss = 1), because a lossless sheet re-emits what inter-microfacet bounces
hold, and the white thin furnace must read exactly 1 (Cycles' color-tinted
Fms darkens it ~0.6%). (7) *OpenPBR's formulas adopted exactly*: darkening
Δ = (1−K)/(1−E_b·K) (vanishing against a white base — the furnace row),
the coat's variance-sum roughening of the base, specular_weight as an IOR
remap with the TIR-fixed coat–base ratio, coat_color applied once, gray
fuzz attenuation — and emission's coat tint folds in at prep, the one
place both the light table and the shading kernel read, so the strategies
can't disagree about an emitter's radiance. (8) *Opacity implements
D-073's split*: fractional-opacity instances carry FORCE_NO_OPAQUE, the
intersect stage resolves each candidate crossing stochastically from a
plain hash stream keyed (pixel, sample, bounce, crossing) — Sobol buys
nothing on visibility chains, and per-bounce keying keeps a path's
crossings independent, where a shared draw would bias their product —
while shadow rays attenuate deterministically and stay RNG-free. Triangle
next-event rays now stop just past their target, so nothing beyond the
light can occlude or attenuate a connection. Material dirt rebuilds the
TLAS: opacity is an instance flag. (9) *Push constants stay ≤ 128 bytes*:
the tables ride the scene table (one pointer for everything kernels
share), the lobe tag and medium pack into one path-state word (24-bit
medium, hence instance indexes below the 0xffffff sentinel), and raygen
names only the four fields it writes — bounce 0 owns the defaults.

## 2026-07-10 — M2 step 6: textures

### D-078: Leaf decisions made while building the texture pipeline
Textures now flow end to end: UVs through the geometry path, a prep
pipeline with a DDS cache, the bindless table the descriptor-indexing
baseline was reserved for, constant-or-texture material parameters, the
in-shader IDT, tangent-space normal maps, and per-texel emission and
opacity — with the demo goldens bit-identical, so the machinery provably
costs untextured scenes nothing. The calls the code forced: (1) *The prep
pipeline is decode → linear-light mip-cap → BC encode, cached as DDS
beside the source* (D-060's shape): `image` (PNG/JPEG, those features
only) and the existing EXR reader decode; anything past 4096 box-halves
in linear light (normals decode, average, renormalize); `intel_tex_2`'s
ISPC kernels encode BC7 for 8-bit color, BC6H for float color, BC4 for
scalars (red channel), BC5 for normals, over surfaces edge-padded to
whole 4×4 blocks (the upload skips the padding via its row length). The
planned `ddsfile` dependency died on inspection — it keeps the header's
reserved words private, which is exactly where the cache validity hash
belongs — so the cache is ~90 hand-rolled lines of DX10 DDS (D-011's
under-100-lines rule), with an FNV-1a hash of source bytes + prep
parameters + pipeline version in `reserved1`: invalidation is by content,
immune to the mtime churn of git checkouts, and any parse or hash
mismatch is simply a miss to re-encode over. (2) *The bindless table is
one partially-bound, fixed-capacity (1024) sampled-image array* on the
existing scene descriptor set (binding 2, beside the TLAS and
environment), written per submission like its neighbors — blocking
submits mean no set is in flight when written, so update-after-bind stays
unused. Slots are keyed by (path, usage, color-space override): two
materials sharing an image share a slot, a color and a mask use of one
file are two. Indices are the key set's sort order, deterministic per
prep; scene residency keeps each image's content hash, so an edit
re-preps only textures its dirty materials reference and re-uploads only
those whose source bytes actually changed — a repainted image reloads on
the next material touch, a coat-weight drag re-uploads nothing.
(3) *Constant-or-texture is constant + index*: each texturable slot
(base color, roughness, metalness, emission, opacity, normal) carries a
u32 table index, TEXTURE_NONE meaning constant-everywhere;
`resolveMaterial` replaces or multiplies per hit and everything
downstream reads one plain material. Textured slots lower their constants
to stand-ins — schema defaults where the kernel replaces, the identity
where it multiplies (emission, opacity). (4) *The color pipeline is
storage-stays-source-space*: BC7 with an sRGB view for 8-bit color
(hardware decode), the 3×3 Rec.709 → ACEScg matrix applied in-shader
after sampling — the same matrix prep applies to constants — float
sources linear BC6H (an sRGB override there is meaningless and ignored),
scalars linear by default with an explicit sRGB override honored by
linearizing at prep (BC4 has no sRGB view), normal maps always linear (a
stray override warns and is ignored rather than forking the cache).
(5) *Textured emission stays estimator-consistent by construction*: the
material record and light table carry the map's scale
(luminance × coat tint), and both strategies evaluate the map itself at
their own point — BSDF hits through the per-hit resolve, next-event
connections through the sampled point's barycentrics, which
`LightConnection` now carries (the light sampler's triangle
parameterization and ray-query barycentrics share a convention, weights
on the second and third corner). The MIS-agreement matrix gained a
textured emitter to hold it. Light *selection* still weighs the constant
scale — map variation steers noise, never the estimate. (6) *Textured
opacity resolves per crossing in traversal*: one `opacityAt` (constant ×
map at the candidate's UV) feeds both halves of D-073's split —
stochastic in intersect, multiplicative in trace_shadow — and an opacity
map forces the instance's non-opaque TLAS flag whatever its constant.
The agreement scene's perforated card holds the two policies to one
function. (7) *Normal-map tangents derive per hit from UVs* (D-061: no
authored tangents until anisotropy): dPdu orthonormalized against the
interpolated shading normal, the bitangent's sign from dPdv so mirrored
UV islands map correctly, z rebuilt from BC5's unit length, degenerate
parameterizations (including the all-zero stream of an unauthored mesh)
falling back to the unmapped normal, and the geometric-side flip
preserved. (8) *A UV-less mesh under a textured material warns and reads
texel (0, 0)* — constant, never out of bounds, never silent. A texture
that fails to read or decode is a scene error (the live session keeps its
last good residency); a cache that fails to write only warns. (9) *The
geometry record grew a fourth pointer* (per-vertex UVs, zeros when
unauthored) — which lands its transform rows on 16-byte alignment and
retires M1's PhysicalStorageBuffer validation nag as a side effect.

## 2026-07-10 — M2 step 7: the pbrt importer

### D-079: cenote-pbrt ships with the five traps pinned — and a sixth found
The importer landed as planned (D-057/D-065/D-066): a hand-rolled
tokenizer and recursive-descent parser (spans, `file:line` diagnostics,
`Include` spliced through a stack, every parameter's consumption tracked
so unconsumed ones warn by name), a graphics-state mapper onto the
change-set schema, the PLY reader in core, `cenote-cli import`/`render`
subcommands, the viewer opening `.pbrt` directly, and a three-scene CC0
corpus (Bitterli's cornell-box, veach-mis, teapot-full — converted from
his pbrt-v3 exports, changes documented in the corpus README) with FLIP
goldens in CI. Decisions the work forced:

(1) *Photometric normalization is subtler than "divide by luminance":*
pbrt's `SpectrumToPhotometric` on an RGB spectrum considers only the
color space's illuminant, never the RGB multiplier — so **RGB light
values import verbatim as nit-valued radiance** (`rgb L [4 4 4]` is 4
nits, color unnormalized), while `blackbody` emitters *are* normalized
(chromaticity at luminance 1 × scale). Verified against pbrt-v4's source
at `5f7a606`, along with every other contract this step encodes
(`RoughnessToAlpha = √r`, conductor defaults to copper, `eta` not `ior`,
instancing composes the full declaration-time CTM with no ObjectBegin
inverse, `k_e` factors for `power`/`illuminance`).

(2) *Handedness is a per-scene decision, not a constant.* pbrt's
left-handed projection imports through a `diag(1, 1, −1)` conjugation of
every world-space transform, which maps pbrt's camera space exactly onto
cenote's — for scenes whose camera transform is a proper rotation
(anything authored with `LookAt`). But Tungsten-converted exports (the
entire Bitterli corpus) bake their own handedness fix into a *reflective*
world-to-camera matrix, under which pbrt's projection already behaves
right-handed — conjugating those mirrors the image, which the vendored
cornell box catches on sight (red wall left, green right). The mapper
picks per scene at `WorldBegin` by the camera transform's determinant.
Found by the corpus within minutes of it existing — the reason D-065
wanted real scenes and not synthetic ones.

(3) *ReverseOrientation XOR mirror-CTM applies to the data, not the
shading:* the flip negates authored normals and reverses winding (which
also steers derived normals); since cenote shades and emits two-sided,
sidedness itself reduces to one honest, counted warning per import
(deferrals: one-sided emission).

(4) *The equal-area octahedral resample is Clarberg's mapping verbatim*
(exact `atan` in place of pbrt's polynomial), bilinear with pbrt's own
seam-wrap, light-space orientation and photometric scale baked into the
output equirect because the environment object carries neither; image-
less infinite lights become a two-texel constant EXR. Derived assets go
to an explicit generated-assets directory — beside the output `.ron` for
`import`, a temp directory for direct `.pbrt` rendering — never beside
vendored sources.

(5) *Fidelity warnings are values, not log lines:* `import` returns the
change-set plus its warning list; the CLI prints them, the viewer logs
them, and the corpus harness asserts the list matches exactly what the
corpus README documents — an undocumented degradation fails CI.

(6) *The importer proved the public API sufficient* (D-066's point): it
links only `cenote`'s public surface, and the one addition core needed
was `ChangeSet::relativize_paths` — the inverse of `rebase_paths`, so an
apply-ready absolute-path set becomes a portable scene file.

## 2026-07-10 — M2 step 8: AOVs

### D-080: Leaf decisions made while building the AOV film

Status: accepted. The film grows from one buffer to four — beauty plus
D-062's denoiser guides (albedo, normal) and first-hit depth — through
the same accumulate/resolve path, the same pixel-owned determinism
invariant, and zero change to the beauty estimator (the step-8 diff
leaves every golden byte-identical). Six leaf decisions:

(1) *The guides' state is pixel-indexed, not path-indexed.* The plan
predicted "two path-state fields via the schema seam"; reality:
`shade_surface`'s push constants already sat at exactly Vulkan's
guaranteed 128 bytes, so a seventh path-pool pointer couldn't fit. The
feature throughput (with the written flag folded in — weight zero means
recorded) lives instead in a per-pixel scratch inside the film's
`AovTable`, one indirection behind a single pointer that fits by packing
bounce/max-bounces/light-sampling into one word. Pixel-indexing is
equivalent by construction — a wave carries exactly one path per pixel —
and the scratch needs no clearing because bounce 0 knows a camera ray's
guides are open without reading.

(2) *Depth costs no camera data in any shading kernel:* raygen stashes
each ray's cosine against the camera forward axis in the direction
buffer's spare `w` lane, so first-hit camera-plane z is just hit distance
× w — exact under the thin lens too, since the lens disk spans the camera
plane itself. The AOV probe test pins this by construction: a
fronto-parallel quad must read the same depth at every pixel, which
Euclidean distance fails off-axis.

(3) *The pass-through ramp rides the sampled lobe* (D-070's tag, used at
the vertex that wrote it): a continuing path records `saturate(√α/0.15)`
of its open guides at this surface and passes the rest along the
scattered ray, tinted by the passed lobe's reflectance; any terminal
vertex records everything it still holds; escapes record white albedo
(OIDN's background convention) and no normal. Guides are never
roulette-reweighted — a guide wants low noise, not unbiasedness.

(4) *The guide albedo is the closure's own technique estimates, kept as
colors:* the same expressions the one-sample-MIS weights average into
scalars, summed as float3 and saturated — computed beside the weights but
never feeding them, so sampling is untouched by the guides existing.

(5) *Depth is the finite-guard's deliberate exception:* +∞ *is* its miss
value (only NaN drops), it accumulates and resolves as ±∞ correctly, and
the f32 `Z` channel carries it to disk exactly — pinned end-to-end by
tests.

(6) *One multi-layer EXR, Nuke-shaped:* bare `R/G/B/A` beauty in f16,
bare f32 `Z`, `albedo.RGB`/`normal.XYZ` in f16, zip scanlines, ACEScg
chromaticities in the header; the published `Frame` gained
albedo/normal/depth accessors resolved from the same publish slot as the
beauty, `TRANSFER_SRC` already in place for step 9's host-copy denoise.

## 2026-07-10 — M2 step 9: OIDN

### D-081: Leaf decisions made while wiring the denoiser

Status: accepted. D-063's shape held — the safe `oidn` crate, DEFAULT
device, host copies, a `denoise` cargo feature — and the estimator is
untouched by construction: denoising is a *view* (a second labeled EXR
from the CLI, a swapped tonemap input in the viewer), never a write back
into the film. Six leaf decisions from the flagged spike:

(1) *Library discovery is `OIDN_DIR`, documented, not automated.* The
crate's `bundled` feature is broken on crates.io (its pinned-hash files
didn't ship in the package — verified), and Fedora packages neither a
dev symlink nor a pkg-config file, so pkg-config can never fire there.
The honest path: `OIDN_DIR` in the *user-level* cargo config pointing at
a directory whose `lib/` holds a `libOpenImageDenoise.so` (a symlink to
the distro's versioned library suffices); runtime resolution then rides
the ordinary loader. The README carries the recipe; the feature stays
off by default so `cargo run` needs nothing new.

(2) *Guides go in as noisy aux — no `cleanAux`, no prefilter.* The
plan's caveat proved worse than feared: the crate cannot express OIDN's
prescribed guide prefilter (it unconditionally binds an input as the
color image), and its `clean_aux` setter misspells the parameter name
(`clean_aux` for `cleanAux`), which OIDN silently ignores — verified
empirically, output bit-identical either way. The pre-authorized
fallback stands: default noisy-aux weights, which only *under-trust*
our near-clean pass-through guides.

(3) *DEFAULT device honored; the filter is rebuilt per call.* On a
CPU-only OIDN install the DEFAULT device is the CPU (~200 ms at 720p;
measured: the CPU device treats `Balanced` and `High` identically, so
the quality knob matters only where GPU device runtimes exist — both
still passed as designed, viewer `Balanced`, CLI `High`). A fresh
filter per call costs ~10% of one 720p run and keeps `Denoiser` free of
self-borrowing; one device lives across frames.

(4) *The viewer's toggle is a cadence, not a mode:* one job in flight at
a time, at most one per second, downloaded from the published frame on
the redraw thread (all submits are queue-locked and fence-waited, so the
render thread is undisturbed), filtered on a worker thread, re-uploaded
through `Tonemap::upload_average`, and size-gated against the current
frame so a resize falls back to raw instead of stretching. The denoised
image trails the orbit by up to a second — the usual shape of an
interactive denoise preview (Cycles' viewport split, D-063).

(5) *libstdc++ gets named twice in the build script.* Linking
libOpenImageDenoise (C++) makes ld extract ISPC texture-compression
archive members it otherwise skips, whose C++-runtime symbols were only
ever satisfied transitively. Placement decides survival under
`--as-needed`: the propagated `rustc-link-lib` lands after the archives
for downstream binaries, while the crate's own test binaries put
root-crate libs first — discarded — so a trailing `rustc-link-arg`
covers them.

(6) *CI builds and runs the feature it can't render:* OIDN's CPU device
needs no GPU, so the runner installs the distro library, lints every
crate with the feature on, and runs the unit test that pushes a real
image through the real filter — the one step-9 seam CI can actually
exercise.

## 2026-07-10 — M2 step 10: the lookdev panel

### D-082: Leaf decisions made while building the lookdev panel

Status: accepted. The panel is a temporary closure-testing tool that goes
away when the M4 Hydra delegate moves lookdev into Solaris/Karma, so the
guiding bar was thin over complete. Its whole backend already existed —
the edit channel, `MaterialPatch`, `Op::Material`, and the stop → apply →
restart wave (D-064, D-075), proven by the session's edit tests — so step
10 was a pure viewer-UI task that added **zero** core API. Seven leaf
decisions:

(1) *The viewer holds its own `SceneDescription` replica as the panel's
model.* The description the renderer edits moves onto the render thread at
`Session::new` and is unreadable from the viewer, so the panel can't read
current material values from it. Rather than add a read-back channel, the
viewer keeps a second description applied from the same change-set: both
start identical, every panel edit is applied to *both* (the replica via
its public `apply`, the session via the edit channel), and a reload
rebuilds the replica from the new scene. The description's own `apply` is
the only mutation path, so the two can't drift.

(2) *Materials by name, not D-064's looser "object list".* The list is the
scene's materials (`description.materials()`); editing one affects every
instance wearing it — the right granularity for closure lookdev, and the
only kind the panel touches. Lights, camera, and environment stay out:
the camera is orbit-driven, and lights/environment ride the watched scene
file (D-064's "no authoring UI").

(3) *The full constant parameter set, grouped as OpenPBR groups it* (Base,
Specular, Transmission, Coat, Fuzz, Emission, Geometry) under collapsing
headers, one widget per field. Weights/roughness/metalness/darkening are
[0,1] sliders, IOR a 1–3 slider, luminance and transmission depth are
unbounded non-negative drag boxes, colors are linear-RGB squares (the
values are linear Rec.709, which is what egui's rgb picker edits).

(4) *A textured slot renders read-only.* The five `Texturable` fields and
the normal map, when bound to an image, show a muted "(textured)" label
rather than a widget — the panel edits constants, never overrides a map
with a constant. Constants-only keeps the temporary tool honest about what
it can and can't touch.

(5) *An edit sends the whole material, not a minimal diff.* On any widget
change the panel emits a `MaterialPatch` with every field `Some`. Apply's
equality gate (D-075(3)) makes this exact: only the field the widget
actually moved differs from what both replicas already hold, so only it
dirties and forces a re-prep — the panel needs no per-field change
tracking, and the wire behavior is identical to a minimal patch.

(6) *Reload wins over unsaved panel edits.* A watched-file save rebuilds
the replica and whole-scene-replaces the render scene; any in-flight
slider tweaks are ephemeral scratch, discarded. The reload is validated
against a fresh description locally *before* it's sent, so a file that
parses but doesn't apply keeps the previous scene on both sides and the
replica never diverges from what the renderer holds.

(7) *Lighting an object from the panel works, not just recoloring it.*
Dragging `emission_luminance` off zero makes the material an emitter, which
needs the light table rebuilt, not just the material buffer re-uploaded —
verified against `Scene::update`, which re-derives triangle lights over
the whole description and rebuilds the light table on every material wave
(D-075(5)), so the new emitter is sampled from the next frame.

## 2026-07-10 — M2 step 11: polish and the post-M2 refinement pass

### D-083: Leaf decisions made polishing M2

Status: accepted. Step 11 folded the plan's polish items (module headers
current, decisions current, goldens verified) together with a post-M2
code-review pass — six reviewers over the scene, prep, render, session,
importer, and support clusters, measured against the Cycles X / MoonRay
bar. What the review turned up split into refactors worth doing now and
seams worth naming for later. The refactors are all bit-identical: the
incremental-vs-fresh determinism test and the furnace matrix carry through
every one, so the goldens never moved and none needed regenerating.

Done:

(1) *Host lowering split out of prep.* `prep.rs` fused the fallible
host-side lowering (file reads, decodes, capability checks — everything
rejected on user data) with the GPU orchestration that consumes it, joined
at `HostScene`. The lowering moved to `scene/lower.rs`; `prep.rs` is now the
two orchestrators plus the two residency helpers. The boundary the module
doc described is now the boundary the code has, and the untouched-on-error
contract reads where it lives.

(2) *Film split out of render.* `Film`/`Accumulation`/`FilmAverages` moved
to `render/film.rs` — the split `Tonemap` already had — with fields
`pub(super)` so the renderer-side param builders read them without new
accessors, and the ~1500-line test suite moved to `render/tests.rs`.
`render/mod.rs` is the renderer and the film kernels, skimmable again.

(3) *One list for the textured slots.* The collection pass (`texture_keys`)
and the lowering pass (`lower_material`) each carried the slot-to-usage
pairing; a disagreement would panic the bindless-index lookup. Both now
read one `textured_slots` list, and `lower_material` destructures it through
an array pattern so the slot count is compiler-pinned — a new textured
parameter will not compile until it is assigned a field.

(4) *The PLY reader caps its pre-allocation.* A header-declared vertex count
was trusted straight into `reserve`, so a corrupt `element vertex
999999999999` could ask the allocator for terabytes before a byte was read.
The read loop was already EOF-bounded; the reserve now is too.

(5) *The docs sweep.* `Kind`'s unused `Hash` derive dropped (every use is a
`BTreeSet` keyed on `Ord`), `render_to_buffer` (one caller, a hypothetical
justification) inlined, four stale or dangling doc comments corrected, and
the importer's triply-explained reflective-camera rule trimmed to a single
owner (`FLIP_Z`).

Named, not fixed — the per-edit cost model is a watch item, not rot:

(6) *`validate()` re-stats every referenced file on every apply.* A lookdev
slider's one-field patch stats every texture and PLY in the scene. Scoping
the file-existence checks to dirtied paths is correct — paths are
per-object, and prep reads the file as the real guard — but it threads the
dirty set into the validate-then-apply contract for a cheap-`stat` latency
win. Left pristine; prep is the backstop, and interactive edits touch no
paths.

(7) *`lights::build` and `total_power` recompute the record list.* Scene
prep, update, and demo each call both, and each rebuilds `raw_records`. A
clean dedup threads `build`'s already-computed total through
`upload_instance_tables` to three call sites and reworks two unit tests —
cross-file churn for a prep-time linear pass over a handful of emissive
triangles. Deferred.

(8) *The viewer's replica can diverge from what renders.* D-082(6) claimed
the `ui_desc` replica never diverges from the rendered scene. It can: a
reload the render thread rejects at *residency* — a present-but-corrupt
asset, which apply's existence check cannot catch — leaves the replica
holding the new scene while the renderer keeps the old, until the next good
reload resyncs both. The lookdev edit path is safe (a material patch cannot
fail residency), so this is the reload seam only, and it is logged, not
silent. The proper fix is a confirmation channel so the replica advances
only on the render thread's acceptance; it belongs with the M4 rework that
makes the scene graph the single authority (deferral ledger, Viewer &
lookdev). The whole replica device is a temporary M2 stand-in for a
read-back the delegate supplies for free.

## 2026-07-11 — One-sided emission

### D-084: Emitters emit from the winding-front face only (picks up D-079)

Status: accepted. Reverses the two-sided-emission default the forward
emissive-hit path assumed (D-034), and picks up the deferred "one-sided
emission" (D-079). An emissive triangle now radiates from its winding-front
face only — the side `cross(edge1, edge2)` points to — matching pbrt's area
light, whose `twosided` defaults false.

*Why now:* the pbrt-v4 comparison demo made the gap visible and measurable.
A ceiling panel emitting from both faces leaks light up past itself, which
both looks wrong and inflates the frame: the vendored cornell box measured
19% brighter than pbrt in mean linear luminance (0.160 vs 0.135) purely from
back-face spill. One-sided brings it to 0.135 — agreement to under a
percent, with no scene edits. It is also the physically ordinary default; a
two-sided emitter is the special case.

*Mechanism, and why MIS stays consistent:* two gates, both keyed on the same
winding normal so the strategies agree on which face emits. Next-event
sampling (`lights.slang`) swaps the absolute light-side cosine for a signed
one — a receiver behind the emitter gets pdf zero. The BSDF-hit path
(`shade_surface.slang`) records `frontFace` from the raw geometric normal
*before* it is flushed to face the ray, and adds emission only on a
front-face hit. The MIS weight needed no change: its next-event pdf already
used the cosine *magnitude*, which is exactly right for a front-face hit (the
only case that now emits) and irrelevant for a back-face hit (which emits
nothing). BSDF surfaces stay two-sided — it is emission, not shading, that
picks a side.

*Scope, and the new deferral:* this is a global switch, not the per-light
flag D-079's production shape imagined. cenote is now one-sided everywhere;
a pbrt light with `twosided true` loses its back-face emission and imports
with a counted warning (the importer's sidedness warning flips polarity to
match). Honoring `twosided true` — an opt-in two-sided emitter — is the
residual, re-lodged in the ledger as "Two-sided emission" against this entry.

## M3 — GRIS-DI

*Locked 2026-07-11 via structured interview + a sourced research pass over ReSTIR-DI,
GRIS ("Foundations of ReSTIR"), defensive pairwise MIS, presampled tiles, and ReGIR,
read against Cycles X / MoonRay / pbrt-v4 engine hygiene, then reviewed against those
same renderers for module boundaries, naming, and observability (D-091, D-092). The
working plan is [m3-plan.md](m3-plan.md); these entries carry the rationale.*

### D-085: M3 is GRIS-DI; the unbiased convergence contract and the reservoir substrate
Status: accepted. M3 builds screen-space ReSTIR direct illumination on the GRIS
foundation, and settles the question a decisions interview must settle first: *what is
the same-estimator thesis under reuse?* The answer rests on separating two axes the
literature conflates in casual reading. **Bias:** GRIS (Lin 2022) proves reservoir reuse
— temporal included — is unbiased given the generalized unbiased contribution weight
`W_Y = (1/p̂(Y))·Σⱼ wⱼ` (the combined form, *not* the split `1/M`, which biases when the
resampled inputs differ) and ray-traced bias correction. **Correlation:** unbiased
frames that share reservoirs are *correlated*, and correlated frames do not average at
1/N — the effective sample count drops by the integrated autocorrelation time (~the
M-cap), leaving a spatially-structured, blotchy residual rather than white grain.

The contract that falls out: **temporal reuse warm-starts the preview after camera moves
*and* after light/material edits** — the film always resets to zero on an edit, but the
reservoirs carry over, are re-targeted under the new scene, and have stale confidence
decayed — while **the converged still image is spatial-only with fresh per-frame RNG**,
so it is a mean of *independent* unbiased frames and converges cleanly at 1/N to the
exact path-traced ground truth cenote produces today. Every mode is unbiased; temporal
is annealed for *efficiency*, never suppressed for *correctness*. This is the direct
answer to the interview's first challenge — yes, temporal can be included without bias,
and the reason to anneal it is decorrelation, not correctness.

*Why carry state across an edit, when production renderers restart from scratch:* the
restart-on-dirty reflex is a *delegate convention*, self-imposed via Hydra's
`IsConverged()` (which defaults true), not a Hydra requirement — nothing forbids a
delegate from carrying reservoirs across a dirty the way it already carries the BVH. For
an interactive lookdev renderer, carrying reservoirs across a material tweak is the
charter-native delta philosophy: re-prep exactly what changed, re-converge from a warm
start, film honestly reset. Correctness stays airtight because the film resets and every
mode is unbiased.

*Build the substrate now, not in M4:* the contract needs (a) **per-view reservoir/film
ownership** keyed by a stable view identity (render-buffer `SdfPath`-ready), because one
Hydra delegate drives multiple viewports and reservoirs are per-view; (b) **stable light
identity + an identity↔dense-slot remap**, because the current GPU light index is
order-derived *and* power-filtered in `lower.rs` — volatile across the very edits
temporal reuse must survive; and (c) a **cold-start / no-carry render setting** for batch
determinism (the CLI already cold-starts). Per-view ownership and stable identity are
high-retrofit-cost — retrofitting them into a shipped M3 would be a rewrite — so they
land now. This also retires the D-064/D-083 viewer-replica seam early: per-view state
owned by identity is the shape the Hydra delegate wanted anyway.

*Thesis honesty (the coherence note):* under this contract, temporal-on-motion and
spatial-on-hold are *different estimators frame-to-frame* — both unbiased, both
converging to the same image. The README's "the preview and the final are the same
estimator" gains a footnote saying so: the claim holds at the level that matters (same
converged result, no biased preview mode, no final-gather switch) but no longer asserts
frame-level identity once temporal is annealed. The promise stays true; it stops
overstating.

### D-086: Screen-space primary-hit DI reuse; the reservoir is an index-agnostic primitive
Status: accepted. Reuse scope is **screen-space, primary-hit direct illumination only**:
initial RIS → temporal → spatial → resolve, all at the first visible surface. Secondary
bounces keep the M2 NEE + BSDF-hit MIS path untouched. This is the industry baseline for
a converging renderer — ReSTIR is a primary-visibility *accelerant*, not a replacement
for path integration, and the charter's one-integrator commitment means the reuse layer
sits *in front of* the existing integrator, not around it.

The reservoir + weighted reservoir sampling + RIS + unbiased-weight + pairwise-MIS is
built as an **index-agnostic primitive** — a `Reservoir<Sample>` type and a handful of
pure functions with no hardcoded notion of "pixel" — so that M6 path reuse (ReSTIR-PT is
*also* screen-space, via reconnection/hybrid shift maps) and a future ReGIR world-space
hash grid both *instantiate* it rather than reimplementing it. The named risk, written
into the plan so it is resisted rather than rediscovered: this primitive is a data type
and three-or-four functions, **not** a framework. Designing a speculative abstraction
for M6/ReGIR requirements we cannot yet fully specify would produce a worse M3; the
primitive stays small precisely because a reservoir over a light-surface-point sample
genuinely *is* the same structure regardless of what indexes it.

*ReGIR deferred, and why it is not "the M6 thing":* the reframe that settled the
interview — ReGIR (world-space reservoirs in a hash grid) is *orthogonal* to screen-space
ReSTIR and *later*, not the content of M6. M6 is screen-space path reuse. ReGIR's trigger
is many local lights illuminating *non-primary* vertices — secondary bounces, volume
scatter points, SSS exit points — where a screen-space reservoir has nothing to reuse. It
lands in the M7/M8 era or a dedicated many-lights pass (deferrals.md).

### D-087: Estimator internals — surface-point domain, pairwise MIS, ray-traced correction
Status: accepted. Four internals, three of them derived-locked from D-085's
unbiasedness requirement plus the research:

**Reservoir domain = the light-surface point.** The reservoir stores a stable light-id +
primitive index + barycentrics — exactly the `Hit` shape M1 chose for its re-evaluable,
reconnection-ready form. This makes the DI reuse shift the *identity* map, so its
**Jacobian is 1**. This is the single most important estimator invariant and it is not
optional: the Jacobian is 1 *iff* the reservoir stores the surface *point*, not a
solid-angle *direction* — store the direction and every reuse across surfaces of
different distance/orientation needs a Jacobian that, dropped, biases silently. Reusing
`Hit` gives the correct domain for free, and leaves the shift dormant-but-correct for
M6, where the reconnection shift becomes non-trivial.

**Target p̂ = unshadowed luminance of f·L·cosθ** over the full OpenPBR closure (the
D-070 one-sample-MIS lobe machinery supplies the combined BSDF value). Luminance rather
than the RGB vector keeps the reservoir scalar-weighted; the true colour is recovered at
resolve, where f·L·cosθ is evaluated for real with the visibility ray. Unshadowed,
because folding visibility into the target would demand a shadow ray per candidate — the
whole point of RIS is one shadow ray on the *survivor*.

**Reuse MIS = defensive pairwise** (Bitterli 2022 thesis). It buys balance-heuristic-
quality variance at O(M) cost and ~zero storage, where the generalized balance heuristic
is O(M²) and the `1/Z` unbiased weight is correct but noisy. Pairwise is the right point
on the cost/variance curve for the neighbour counts M3 uses.

**Bias correction = unbiased ray-traced.** Each reused neighbour's p̂ is re-evaluated
*under its own visibility* — a shadow ray from the receiving pixel to the neighbour's
stored light point. This is what keeps reuse unbiased when neighbours see different
occlusion (the common case at a shadow boundary); the cheaper "assume visible" correction
is the classic ReSTIR bias source, and the charter's no-biased-modes thesis rules it out.

### D-088: The reservoir owns all non-delta primary DI; delta lights stay on exact NEE
Status: accepted. Resolves how ReSTIR and the existing MIS estimator share the primary
hit without double-counting. **The reservoir owns 100% of non-delta primary-hit direct
lighting.** Its candidates are M light samples drawn from the existing power-proportional
alias table (`lights.slang`'s `sampleLights()` — area emitters + the importance-sampled
HDRI, reused with no new light data structure) *plus one internalized BSDF-sampled
candidate*, so light-vs-BSDF importance is handled *inside* RIS rather than as a separate
MIS combination outside it. The BSDF **continuation** ray — the one that carries the path
to its next bounce — is therefore **indirect only, and suppresses emission at its first
hit**, because any emitter it strikes was already a reservoir candidate; letting it add
emission would double-count the light the reservoir just sampled.

**Delta lights stay outside the reservoir**, on exact regular NEE with MIS weight 1,
added separately. A delta light cannot be hit by a BSDF sample (zero solid angle), so it
carries no MIS partner and no Monte Carlo noise — a reservoir over it would only inject
resampling variance into a term that is already exact. This refines D-087's ownership to
"the reservoir owns all *non-delta* primary direct lighting."

*Candidate counts* start at M ≈ 16 local + a few environment + 1 BSDF, tuned in the
validation harness. *Presampled light tiles* (Wyman–Panteleev 2021) are deferred: they
are a memory-coherence optimization for millions of lights that injects intra-tile
correlation, and cenote's light counts do not need them — the trigger is a measured
per-candidate global-gather bottleneck (deferrals.md).

### D-089: Convergence policy v1, and the interactivity + variance bundle
Status: accepted. Convergence policy v1 is deliberately small: **M-cap ≈ 20** (tunable);
temporal reuse active during camera motion and the first post-edit frame, then a **short
confidence decay** over a few frames handing off to spatial-only fresh-RNG accumulation,
so the motion→hold transition has no visible pop. The spatial-only converge mode *is* the
D-085 camera-hold switch — one mechanism, described from two sides.

Three deferrals are **picked up** here and moved out of the ledger. **Blue-noise
sample-index ordering** (D-021 earmarked the Sobol-Burley sampler for exactly this
drop-in; it improves *perceived* early convergence, which matters most beside ReSTIR).
**Interactive niceties** (D-051's carried deferral): `max_samples` and a convergence-idle
so a settled viewer stops pinning the GPU, publish-interval growth as convergence slows,
a navigation-resolution divider during motion. And a **per-pixel variance substrate +
global noise-threshold auto-stop**: a running mean + second moment of luminance,
pixel-owned like every film buffer, driving a global stop *and* — the synergy that
justified landing it now — supplying the exact quantity the D-090 validation convergence
curves need. One buffer, two consumers.

*Per-tile adaptive steering is deferred*, and the reason is specific to ReSTIR: a
per-pixel variance estimate assumes *white* noise, but ReSTIR residual is spatially
correlated (blotchy), so (1) the estimate is unreliable — a correlated blotch reads as
converged — and (2) stopping a converged-looking pixel starves its neighbours' spatial
reuse. The clean resolution, when it is built: gate steering to the converged
independent-frame phase (where the residual *is* decorrelated and the estimate becomes
reliable), make termination *tile*-based (no half-stopped reuse neighbourhoods), keep the
estimate and threshold deterministic. Deferred until the estimate's reliability on real
ReSTIR residual is *measured*, not assumed (deferrals.md).

*Determinism is preserved for free:* spatial reuse reads a committed prior-pass buffer
(ping-pong, a barrier between passes), neighbour selection comes from a reserved
deterministic RNG dimension, and no stage accumulates a reservoir with atomics — so the
pixel-owned bitwise-determinism invariant survives. Async submits would break this, which
is exactly why the timeline-semaphore/wave-tail perf pass stays deferred (D-043): it is
an own, measured pass *after* ReSTIR is correct, not folded into M3.

### D-090: Validation harness and the dual flagship demo
Status: accepted. The ground-truth oracle is **cenote's own brute-force NEE+MIS
accumulation at high spp**, not an external renderer — because the thesis under test is
that ReSTIR and brute force are the *same estimator*, so the reference must *be* that
estimator; measuring against Falcor or pbrt would measure primaries/tonemap/closure
differences instead of the one thing that matters. Falcor is a behaviour *spot-check*
only. Metrics: **FLIP + numerical mean-error-vs-reference convergence curves** (read
straight off the D-089 variance substrate). The **unbiasedness gate** is generalized-MIS
agreement stated operationally: **ReSTIR-on and ReSTIR-off must converge to the same
image** — it runs from step 3 (the first estimator) onward, not only at the end, so bias
is caught the frame it appears.

*The golden strategy (the coherence note):* at converged spp ReSTIR ≡ brute force by
construction, so the **existing demo and corpus FLIP goldens become a free regression
gate** — ReSTIR must match them too, at converged spp. Low-spp ReSTIR *noise* is
deliberately **not** golded: it differs from brute-force noise (that is the whole point)
and would make the golden a noise-pattern test, not a correctness test.

*The demo is both.* A **procedural many-light scene** (dozens-to-hundreds of small
emitters with real occlusion, built via the `demo.rs` pattern) is the primary,
CI-golden, validated demo — ReSTIR vs brute-force equal-time plus convergence curves.
*Plus* one **imported many-light showcase** as an un-gated README beauty shot — the
flagship stretch, outside CI so a heavy asset never gates the build. "Flagship begins"
(charter §4) is this entry.

### D-091: The M3 kernel layout, module boundaries, and naming convention
Status: accepted. From the design review against Moonray/Cycles/RTXDI — the axes a
decisions interview under-weights. The estimator decisions (D-085…D-090) say *what* to
compute; this entry says how it lands *as code*, because for a silently-bug-prone feature
that is what separates a research-correct plan from a discoverable one.

*Kernels.* The primary-hit reuse chain is **four discrete, single-purpose stages**,
matching Cycles X's one-purpose-per-kernel shape and cenote's existing queue-driven
wavefront — never a mega-kernel fused into `shade_surface`: `restir_candidates` (initial
RIS) → `restir_temporal` → `restir_spatial` → `restir_resolve`. Visibility and
bias-correction rays reuse the existing `trace_shadow` kernel. The reasoning is
observability: the wavefront stage sequence in `wavefront.rs` *is* the map of the
renderer — a new reader traces it to understand a frame — and four named stages in that
sequence keep the map honest where a fused kernel would bury the estimator. Discrete
kernels also make the D-092 per-stage toggles fall out for free.

*Modules.* The index-agnostic primitive lives in a new `shaders/reservoir.slang`; each
stage is its own `shaders/restir_*.slang`; reservoir-buffer lifecycle, per-view
ownership, and the light-identity remap live in a new `src/restir.rs`; `wavefront.rs`
wires the four pipelines beside its existing five. This mirrors how the code already
separates the stage chain (wavefront.rs) from the resources it drives.

*Naming.* **Readable identifiers**, with a **Rosetta-stone doc block** at the head of
`reservoir.slang` mapping each to its paper term — the house style `rng.slang` already
sets. `Reservoir { sample, weightSum, confidence, unbiasedWeight }`; `reservoirUpdate`
(the WRS stream insert), `reservoirMerge` (the pairwise-MIS combine), `targetFunction`
(p̂), `unbiasedContributionWeight` (the GRIS W_Y). The paper vocabulary — RIS, WRS, UCW,
GRIS, Jacobian — appears in comments, never as bare identifiers, because that jargon is a
wall for a new reader and a Rosetta stone lets the code still be checked against the
papers.

### D-092: A first-class ReSTIR debug/observability surface
Status: accepted. From the same design review. ReSTIR bias does not crash or NaN — it
shifts the converged mean a few percent and looks plausible — so the D-090 converge-to-
reference gate, an *end* check, is not enough: steps 4–5 are undebuggable without
*intermediate* visibility, and RTXDI ships exactly this surface for exactly this reason.

The debug surface is its **own workstream, landing with step 3** (the first estimator),
not folded into final validation: false-colour **selected-light-id**, **confidence (M)**
and **unbiased-weight (W)** heatmaps, **per-stage on/off toggles** (initial-only,
+temporal, +spatial — the same switches the D-090 unbiasedness gate drives), and a
**reuse-gain** view off the variance substrate. It is gated behind a viewer debug mode /
render setting and written by `restir_resolve` through a single enum-selected debug
buffer — lean, deliberately *not* the full D-080 AOV registry, in keeping with the
charter's lightweight bias. It doubles as discoverability: the debug views *are*
documentation of what each stage does.

## 2026-07-12 — M3 step 4: spatial reuse

*The step-4 design was locked via a structured interview, then re-validated by a sourced
pass over the RTXDI source tree, the 2023 SIGGRAPH "A Gentle Introduction to ReSTIR"
course notes, and the GRIS supplemental, read against how production offline renderers
sample many lights (Cycles X / Arnold / MoonRay / RenderMan / Manuka). The validation
confirmed the plan is correctly sized — every choice maps to a documented reference
default or the course-recommended variant, nothing invented, nothing premature. It also
established the correct reference frame: **no shipping offline renderer does cross-pixel
reservoir reuse at all** — they converge on light-tree/BVH importance sampling + MIS +
adaptive sampling + shadow rays (Cycles and Arnold literally share the Conty & Kulla
light tree). ReSTIR spatial reuse is a real-time technique; its only offline cousins are
research ReSTIR-PT variants that drop temporal and average initial+spatial passes — which
is why cenote's own high-spp brute-force accumulation (D-090), not an external renderer,
is the only meaningful oracle. The three entries below carry the correctness hazards that
validation surfaced — implementation-time care, not design changes.*

### D-093: Step-4 spatial-reuse implementation watch-items
Status: accepted. Three places where the step-4 estimator is silently bug-prone, recorded
so the care is deliberate and the tests are aimed, not discovered late.

*The visibility-consistent pairwise MIS is the load-bearing hazard.* cenote takes the
course-recommended **visibility-aware target** (V inside p̂), which is a higher-quality,
still-unbiased variant of RTXDI's shipped split (unshadowed target + a separate
ray-traced bias-correction ray) — and is *required*, not gold-plating, because the step-4
checkpoint demands an unbiased converge-to-reference result that RTXDI's biased "Basic"
mode could not meet (D-087, D-090). Its two consistency obligations are exactly where
bias hides: (1) the pairwise MIS weights must satisfy **Σmᵢ(y) = 1** with V living in p̂
*and* the defensive/normalizer divisor keyed to the **accepted** neighbour count, not the
nominal k — because a geometrically pre-rejected neighbour cannot generate `y`, so it must
leave the count entirely (the canonical sample, covering full support, absorbs its
normalization); and (2) V must be re-evaluated at the **current pixel's** surface after
the shift — never the neighbour's stored V — and the *same* V-inclusive p̂ must appear in
both the resampling-weight numerator and the final 1/p̂ UCW, or the p̂ factors do not
cancel. Guard: the D-090 test suite's **analytic microtest with synthetic per-neighbour
visibility** is purpose-built for this and is the correctness spine; additionally
**numerically cross-check one scene config against RTXDI's "Ray Traced" mode** as the
reference unbiased estimator, since a residual Σm≠1 shows as a small consistent bias the
convergence curve would otherwise excuse as noise.

*Environment/distant-light samples in the shift are a second, quieter trap.* A stored env
sample is a **direction**, position-independent, so the identity shift and Jacobian = 1
still hold (unlike a stored *area* point, the trap D-087 already sidesteps) — but the p̂
recompute at the neighbour surface **must branch on light type**: an area emitter carries
the cosθ_light/d² geometry term, an environment sample does not. The shift must handle a
mixed area+env reservoir uniformly; the convergence gate (D-090) gets an explicit
env-light case so this path is exercised, not assumed.

*The geometric gate is deliberately stricter than the reference, and that is the first
knob — not a second pass.* cenote gates neighbours at **normal·normal > 0.9** (≈26°) where
RTXDI defaults to 0.5 (≈60°); the depth gate (<10% relative) matches RTXDI's 0.1 exactly.
The strict normal threshold is the conservative correctness-first choice — it rejects more
and reuses less. If spatial-only convergence lags the target, **loosen the normal
threshold before adding a second spatial pass**: the single pass keeps `curr` untouched
for step-5's temporal carry (D-091), and a second pass is a convergence optimization the
plan correctly defers, not a correctness fix.

## 2026-07-12 — M3 step 5: temporal reuse

*The step-5 design was locked via a structured interview walking the whole dependency
tree one question at a time, each answer grounded in the code the change touches rather
than assumed. The through-line: temporal reuse's **bias surface is small and nameable**.
Reprojection and disocclusion — the parts that look hard — are quality-only; a stale or
mismatched reprojected reservoir just contributes a bad sample that pairwise MIS
down-weights to correctness. Only two elements can actually shift the converged mean, and
the design pins each to a structural guarantee and an aimed test. The entry below carries
the resolved architecture and those two hazards, so the care is deliberate.*

### D-094: Step-5 temporal-reuse design and implementation watch-items
Status: accepted. The eight decisions the interview resolved, the two bias-critical
elements they isolate, and the buffer-by-buffer staging that lands them.

*Ownership rides the film's lifecycle.* The per-view reservoir state (`ViewState`) lives
in the [`Film`], not a separate registry, because the two share an identical lifecycle: a
film is per-resolution and rebuilt on resize — which is exactly when pixel-to-reservoir
correspondence breaks and the history *must* drop — and it persists across a `reset`,
which is exactly a camera move, where the history *must* carry (the warm-start). So
"rebuild on resize, carry across move" falls out of Film's own construction/reset for
free, with no generation-counter check. The `ViewId`-keyed `Views` map stays dormant
until a second viewport exists (step 6, multi-view); today one film owns one state under
`ViewId::PRIMARY`.

*The four-buffer fixed-role routing is the structural bias guarantee.* Four reservoirs per
view — `cand`, `prev`, `curr`, `scratch` — with candidates writing `cand`, the temporal
combine writing `curr` from `cand` + the reprojected `prev`, spatial writing `scratch`
from `curr`, resolve reading `scratch`, and a frame-end `swap(prev, curr)`. This makes the
plan's "**never feed a spatial result back into the temporal reservoir**" a property of
the wiring, not a discipline: `scratch` — carrying the *visibility-folded* W (D-093) — is
a dead-end only resolve reads, and `prev` only ever receives the `curr` (candidate /
temporal) lineage, which carries the *unshadowed* W. Mixing those two W conventions across
the ping-pong would bias silently; the routing makes that mix unrepresentable. The extra
`cand` buffer (distinct from `curr`, next frame's `prev`) is a deliberate **+1 over the
plan §2 three-buffer budget** — it buys the clean feed-convention rather than aliasing two
disjoint-lifetime roles onto one buffer, a footgun the first added reader would spring.

*A persisted prev-frame G-buffer is unavoidable, and it is the reprojection substrate.* A
`StoredReservoir` holds only the light point (id + primitive + barycentrics), nothing
about the receiving surface — so the disocclusion gate, which must compare this pixel's
current surface against the surface `prev` was resampled for, needs last frame's hit
persisted as its own prev/curr G-buffer pair. This is state the shift for spatial (which
reads live path state at the current pixel) never needed.

*Temporal stays unshadowed; visibility is still applied exactly once.* The combine adds no
shadow rays — it reuses `prev` under the same unshadowed target the candidates use — so
the committed visibility path (spatial's V-fold, or resolve's traced shadow ray) is
untouched. The `visibility_in_weight` signal still means "spatial ran".

*Two elements bias, and only two.* (1) **The M-cap** — `c_prev ← min(c_prev, M_CAP ·
c_cand)` — must be a *history-length multiplier* (≈20), not an absolute confidence,
because a candidates frame already credits ~16 confidence; an absolute cap near that would
throttle history to nothing, and an unbounded one lets a stale reservoir dominate forever.
The clamp is the one bias-critical numeric, guarded by the analytic temporal microtest.
(2) **The feed-convention** above. Everything else — reprojection accuracy, the
disocclusion normal/world-position gate, off-screen and dead-light drops — is variance and
quality: a wrong reuse is a bad candidate MIS corrects, never a bias. The RNG follows the
established registry pattern: a `RESTIR_TEMPORAL` dimension block reserved after
`RESTIR_SPATIAL`, so the combine's WRS acceptance draw never correlates with the candidate
or spatial streams (mirroring how the spatial block is kept clear of the per-bounce
dimensions, D-091).

*The decay handoff is what makes the converged still provable.* Temporal confidence is
scaled by a short ramp on the film's samples-since-`reset`; on a held camera the ramp
reaches zero, so the converged image is temporal-*free* — spatial-only with fresh RNG,
independent unbiased frames averaging by 1/N to the exact brute-force reference (the D-085
anchor). The corollary is a testing hazard: because the default estimator decays temporal
*off*, the ordinary converge-to-reference gate (D-090) is trivially blind to temporal
bias. So the CI spine is a **pinned-temporal gate** — decay off *and* spatial off, temporal
forced live to convergence, converging to the reference — plus the analytic microtest;
the default handoff gate and the determinism gate ride on top. Fallback if the smooth
handoff misbehaves (plan §5): a hard temporal→spatial switch on camera hold, a visible
one-frame settle, substrate unchanged.

*Staging: each buffer lands with its first reader, not all up front.* Step 5a wakes the
existing three-buffer `ViewState` into the film, routes the temporal-off path through it,
and plumbs the temporal toggle (`Renderer::set_temporal_reuse`, the CLI `--no-temporal`)
and the frame-end swap — all **behaviour-neutral**, since no stage reads `prev` yet, and
proven so by a byte-for-byte gate against the step-4 build across every ReSTIR config. 5b
adds `cand` and the `restir_temporal` combine (`prev`'s first reader) with the M-cap and
the analytic microtest; 5c adds the prev-Hit G-buffers, reprojection, and the disocclusion
gate with the pinned-temporal gate; 5d adds the decay ramp. Buffers tie to readers, and
each is verifiable the checkpoint it appears.

### D-095: Step-6 blue-noise sample-index ordering, and the STBN→2D reversal
Status: accepted. D-089 folded "blue-noise sample-index ordering" into the step-6 bundle
as a one-liner (D-021 earmarked the Sobol-Burley sampler for exactly this drop-in); the
implementation resolved *which* blue-noise construction, and reversed the obvious choice.
The win is perceptual: a per-pixel permutation of the Sobol sample index, keyed by a
screen-space blue-noise mask, so at low sample counts the Monte-Carlo error lands as **blue
noise across the frame** rather than white — far cleaner to the eye at equal spp,
converging to the identical image (Heitz et al. 2019).

*The temporal axis is the reversal.* Spatiotemporal blue noise (STBN, Wolfe 2021) varies
the permutation per sample to decorrelate consecutive frames — right for a renderer that
displays one sample per frame (real-time + TAA), wrong here. cenote **accumulates**: every
kernel takes a `sampleIndex` and the film averages 0,1,2,… into the displayed mean, and
Sobol's value is that a pixel walking indices 0,1,2,… covers the sequence *progressively*.
A per-sample-varying permutation would make each pixel draw a pseudo-random Sobol index
every sample, throwing away that progressive coverage and reverting per-pixel convergence
toward white-noise √N. A **fixed** per-pixel permutation keyed by a 2-D mask keeps both:
Sobol's convergence rate *and* blue screen-space error (the pure Heitz 2019 construction).
So the temporal axis is deliberately absent — the theoretically correct call for a renderer
that integrates over the sample axis.

*The mask is generated and committed, not downloaded.* A deterministic void-and-cluster
generator (Ulichney 1993) lives in-tree (`src/bluenoise.rs`); the 64×64 toroidal tile is
committed as `assets/bluenoise_64.bin` and loaded at startup. Provenance is the generator,
not an opaque blob — a test regenerates and asserts the committed bytes match, and a
spectral test asserts the tile is genuinely blue (high-band power dominates DC), not merely
a valid permutation. No new crate, no external library, fully reproducible (honours D-011).

*The seam keeps the path tracer untouched.* `rng.slang` splits *ranking* (which sample
index) from *scrambling* (the Owen-scrambled point) so ranking is pluggable: `sample_1d/2d`
keep their per-dimension padding shuffle **bit-identical** — the classic path tracer and
both FLIP goldens (which render in `PathTracer` mode) are unchanged, no regeneration — while
`sample_ranked_1d/2d` take a blue-ranked index shared across a sample's dimensions (Heitz's
shared-ranking, per-dimension-scrambling). The mask rides the scene descriptor set (set 0,
binding 3, beside the environment) rather than a push-constant device address, because
spatial and temporal already sit at Vulkan's guaranteed 128 push bytes — and a
renderer-global sampling resource belongs beside the environment anyway. Ranking is applied
in the three reservoir stages (`candidates`, `spatial`, `temporal` — the high-variance
direct-lighting streams), coherent on one per-pixel key; `resolve`'s lone delta-NEE draw
and raygen's AA jitter stay on the padded sampler (low variance, and it keeps raygen
descriptor-free). A fixed permutation is a bijection and still a pure function of
(pixel, sample, dimension), so the unbiasedness and bitwise-determinism gates
(`restir_matches_the_path_tracer`, `restir_is_bitwise_deterministic`) hold by construction.

## 2026-07-13 — M3 step 7: validation harness and the flagship demo

*Step 7 is the milestone's proof: the harness that shows `ReSTIR` converges to the same
image brute force does and gets there with less noise, plus the flagship scene that
demonstrates it. This entry is the validation write-up — what shipped, what each gate
proves, the one deliberate divergence from the DoD's literal wording, the Falcor
behaviour spot-check, and what step 7 consciously left for step 8. The care worth
recording is that the harness proves the estimator, not the scaffolding around it: every
gate is aimed at the silent-bias class §6 names, and the one place the implementation
reads differently from the plan is called out as a decision rather than left to be found
as a discrepancy.*

### D-096: Step-7 validation harness — what it proves, the Falcor spot-check, the residual items
Status: accepted. The four gates that close M3, the deliberate relMSE-for-FLIP divergence
from the DoD, the Falcor behavioural note, the committed flagship scene, and what stays
for step 8.

*The harness is four gates, each aimed at a different failure mode.* (1) The **`ReSTIR`
white furnace** (`restir_furnace_closes`, `render/tests.rs`) drives an albedo-1 plane
under a uniform emitter through the full resampling estimator — spatial and temporal both
on — and asserts energy neutrality; a missing π, a factor-of-2, or MIS weights that don't
sum to 1 (the exact silent-bias class) fail it fast, without waiting for a converged FLIP
run. It is the §6 cheap tripwire, now driven through `ReSTIR` rather than only the path
tracer. (2) The **numerical convergence gate**
(`tests/convergence.rs::restir_converges_faster_than_brute_force`) measures both
estimators' per-channel relMSE against a shared high-spp `ReSTIR` reference on the
many-light scene: both fall as samples accumulate, brute force closes to <0.05 relMSE by
32 spp — unbiasedness read off the *path tracer's* clean numbers, so the reference
privileging `ReSTIR`'s own noise floor cannot manufacture the result — and `ReSTIR` carries
≥1.5× less error at every matched budget, the reuse win measured rather than asserted. (3)
The **four FLIP goldens** (`many-lights`, `restir-demo`, `restir-many-lights`, and the
existing `demo`) are deterministic regression pins; because at converged spp `ReSTIR` ≡
brute force, the FLIP goldens are the free regression gate D-090 promised. (4) The
**step-3 on-vs-off and determinism gates** (`restir_matches_the_path_tracer`,
`restir_is_bitwise_deterministic`, D-094) stay green through the reservoir buffers. All GPU
gates skip cleanly without a capable GPU, and — per the standing hazard — run serially
(`--test-threads=1`).

*The unbiasedness gate is realized numerically, not as a 4096-spp FLIP — a deliberate
divergence from the DoD's literal wording, recorded as a decision.* §7's DoD names
"`render many-lights.ron --spp 4096`, `ReSTIR` on vs off, FLIP under threshold." What
shipped instead reads the same claim as relMSE against a shared reference at 8/32/256 spp,
because one relMSE test quantifies *both* halves of the thesis — the unbiasedness
(brute → reference) and the reuse win (`ReSTIR` vs brute at matched budget) — where a raw
FLIP at 4096 spp proves only the first, and slowly. The FLIP form's spirit (the two are
the same image) is still carried: the `many-lights` and `restir-many-lights` goldens agree
in the mean to ~1e-4/channel (the Part-2 pins), and convergence asserts brute force closes
on the `ReSTIR` image. The literal `--spp 4096` CLI invocation stays a documented manual
check, not a CI gate — a 4096-spp × 256² render is too slow for the serial-GPU suite. The
divergence is a choice about *how* to prove unbiasedness cheaply and completely, not a
weakening of the bar.

*The Falcor spot-check is a note, by design — not a gate.* D-090 fixed cenote's own brute
force as the *numerical* oracle (measuring against Falcor would measure primary-ray,
tonemap, and closure differences, not the estimator); Falcor is the *behaviour* oracle
only. The spot-check confirms cenote's `ReSTIR` shows the qualitative signatures Falcor's
published ReSTIR-DI does: variance falls as candidate count M rises; spatial reuse turns
per-pixel blotches into a smooth low-frequency residual rather than white noise; the
temporal warm-start converges a held camera faster than a cold start; and the equal-time
figure shows the many-light reuse win. It also records the one *intentional* behavioural
divergence — cenote uses **defensive** pairwise MIS (Bitterli 2022 thesis Eq. 7.8) where
RTXDI/Falcor ship the **non-defensive** variant, the higher-variance-floor but
provably-bounded choice (D-087, D-093). No Falcor build runs here or in CI; the spot-check
is a documented behavioural comparison matching D-090's "spot-check" wording, and a live
Falcor A/B stays deferred (deferrals.md).

*The flagship scene is committed as inspectable data and guarded against drift.*
`scenes/many-lights.ron` is `ChangeSet::many_lights()` serialized — the CLI renders a scene
*file* or the built-in `demo`, never a Rust builder, so committing the scene is what makes
`cenote-cli render many-lights.ron --restir --spp N` work and lets a stranger open the case
`ReSTIR` exists for. A CPU-only test (`committed_many_lights_ron_matches_the_builder`,
regenerated with `UPDATE_SCENES=1`) fails if the builder drifts from the committed file, so
the figures can never silently describe a stale scene. The two figures —
`docs/restir-convergence.png` (the relMSE curves) and `docs/restir-equal-sample.png` (brute
vs `ReSTIR` at 8 spp) — regenerate deterministically from the untracked `.git/demo`
pipeline (matplotlib plot + `magick montage`), and are skipped gracefully where those tools
are absent, exactly as the pbrt-v4 figures are.

*Each gate has one blind spot, covered by a sibling — worth naming so the harness isn't read
as tighter than it is.* The four FLIP goldens clamp linear HDR to [0, 1] before comparing
(`flip_image`, `golden.rs`), so a change confined to values already above white — a firefly
that brightens from 3 to 30, an emitter-power tweak that only lifts an already-clipped
highlight — moves no clamped pixel and passes the pin. That class is not unguarded: the
white furnace asserts absolute energy on the full unclamped range, and the convergence
gate's relMSE runs on the linear HDR directly (no clamp), so super-white drift surfaces as
changed energy or changed error even where the goldens are blind. Symmetrically, the
convergence reference is *empirical* — `ReSTIR` at 256 spp, not an analytic image — so that
gate proves the two estimators *agree* and measures the variance win; it does not, on its
own, pin *absolute* correctness, since a bias shared against analytic truth would move
reference and estimate together. Absolute unbiasedness is the furnace's job (a closed-form
energy target nothing empirical anchors); the goldens and the ~1e-4 mean agreement pin that
the two estimators land on the same image. No single gate is the whole proof — the four
interlock, and this paragraph is where the seams are.

*What step 7 consciously left for step 8, and what stays deferred.* The imported many-light
**showcase** — the un-gated README beauty shot (D-090) — is the pre-agreed first fallback
(§5) and did not land: the validated procedural demo is the milestone, the beauty shot is
not. The **README flagship section** and the **D-085 thesis footnote** (D-091) are step-8
polish, held deliberately so the prose is written once against the final figures rather
than twice. A live **Falcor A/B** and the **GRIS/CRIS** charter-scope question remain
deferred (deferrals.md). None of these gate M3's done-ness — the four validation gates
above do, and they are green.

## 2026-07-14 — M4 opening: the Hydra 2 delegate and the out-of-process render server

*M4's eight structural decisions, locked by a structured interview over two sourced
research passes — the Hydra render-delegate contract and the Arras/MoonRay
out-of-process precedent — then re-verified by an adversarial fact-checking pass, and
finally **pivoted to Hydra 2** by a third, source-level pass over OpenUSD 25.11–26.05
and dev (the scene-index observer machinery, hdPrman's experimental Riley observer back
end, the `HdRenderer` stub, the AOUSD deprecation roadmap). The working plan is
[m4-plan.md](m4-plan.md); this entry carries the rationale, the four standing deferrals
the milestone picks up, and what step 0 (the transport spine) actually shipped.*

### D-097: usdview-live is the milestone; Houdini-ready by rule; pinned to stock USD 26.05
Status: accepted. Done means cenote appears in usdview's Renderer menu and renders and
refreshes a real USD stage live. The render server and transport obey one standing rule:
nothing may assume usdview specifically, or bake in a stock-USD-only assumption an HDK
rebuild couldn't satisfy — so Houdini/Solaris later is a recompile-and-package step, not
a rearchitecture (its deferral, with the ABI details, lives in deferrals.md).

*Why usdview:* the charter names it as the legible pipeline-TD artifact, and stock USD
is one pinned version, compile- and GPU-testable without a DCC license; folding the HDK
ABI maze into M4 would roughly double the integration surface on the fragile step. *Why
26.05:* it is the newest release and the first where scene-index consumption is the
blessed default. Nothing older is calmer — an older pin just defers the same breaks to
the first upgrade.

### D-098: The delegate is scene-index-native — zero Rprims, an observer on the terminal scene index
Status: accepted. The delegate is a thin translator, out-of-process from the renderer.
It lists **no supported Rprim types** (back-end emulation then instantiates nothing) and
consumes the scene by overriding `SetTerminalSceneIndex()`: notices batch in a stock
`HdsiPrimTypeNoticeBatchingSceneIndex` and flush through a prim-managing observer in
`Update()` — Hydra's serial per-frame hook, run before task execution. Hydra's dirty
*locators* map onto cenote's patches (a locator-guarded pull is an `Option<…>` field;
clean attributes stay `None`); `PrimsRemoved` is `Op::Remove`, which was written for
exactly this. All validation, reference resolution, and prep stay server-side —
`SceneDescription::apply` is already validate-then-apply-atomic with an equality gate,
and re-implementing that in C++ is the exact duplication the thin shape avoids.

*Why the forefront bet is safe enough:* the classic Sync path was condemned while M4 was
being planned — 26.03 made scene-index mode the UsdImaging default and deprecated
scene-delegate mode, with removal on Team Hydra's stated roadmap — while the consuming
contract is verifiably stable (`HdSceneIndexObserver` unchanged in two years, the
prim-managing observer byte-identical since 24.03, the geometry schemas frozen). The
zero-Rprim shape is verified legal against source (`HdSceneIndexAdapterSceneDelegate`
silently skips unsupported prim types), and the pattern is hdPrman's own experimental
Riley observer back end, taken to its logical end. The pure `HdRenderer` is *not* the
target: the class is a verbatim stub whose only implementation is the adapter wrapping a
render delegate — which is exactly how cenote runs under the 26.03+ engine (deferral in
deferrals.md, trigger: the API gains real methods). Threading is retired as a risk class
by this choice: one `Update()` flush → one atomic `ChangeSet`, single-threaded by
construction; the classic design's parallel-Sync accumulator lock is deleted, not
ported. The pre-agreed fallback if the all-observer shape hits an unforeseen wall is a
classic Sync delegate *on the same wire* — only the C++ consumption ring swaps.

### D-099: Crash isolation, not recovery
Status: accepted. Isolation comes from the process boundary alone: on server death the
delegate degrades gracefully (dead-socket detection, no self-crash) and recovers on the
next destroy/recreate — a renderer toggle or stage reload hands the recreated delegate a
fresh terminal scene index, and the observer receives `PrimsAdded` for the whole stage.
No delegate-held replay state. The connection is genesis-then-deltas shaped
(`Replace` then `Apply`s), so automatic replay is a clean later bolt-on with no wire
change (deferral in deferrals.md).

*Why:* a render delegate can't cleanly self-trigger a Hydra repopulate, so the real
options are manual-zero-state or a full delegate-side scene mirror; the mirror costs a
second copy of *geometry* in the delegate process plus respawn orchestration —
unjustified when re-populating usdview is a two-second toggle. Verified against source:
hdMoonray pays exactly that mirror (its retained `SceneContext`, which its RDL delta
encoding needs as a baseline anyway; Arras restarts nothing), where cenote's patches
come straight from dirty locators and need no baseline at all.

### D-100: Control transport — loopback TCP + a spawn-time token, MessagePack change-sets, strict request/response
Status: accepted. The control channel is loopback TCP (`127.0.0.1`, ephemeral port) with
a spawn-time token, u32-LE length-prefixed MessagePack frames (`rmp-serde` defaults,
positional struct arrays), strict request/response — the server never speaks unprompted,
so the C++ client needs no reader thread. The payload is the existing serde `ChangeSet`
as a **full 1:1 wire mirror** (`cenote-wire`, deps `serde` + `rmp-serde` only), plus the
small control surface for what is not scene data: `SetCamera` (the inputs-lane fast
path — mandatory, because `Session`'s inputs-lane camera overwrites the scene camera at
every wave, so a `CameraPatch` would be silently dead), `Resize` (the shm handshake),
`Ping`. Continuous status (frame counter, samples, converged, rejected-edit count) rides
the shm header, never the socket. An `Ack` is a receipt, not a validation: edits apply
at wave boundaries, rejections accumulate onto the next response, and the header's
monotonic `rejected_edits` counter tells an idle client to `Ping` for the strings. EOF
is shutdown — the server is spawned per-delegate, so there is no shutdown message.
Cross-language agreement is pinned by a **byte-exact golden corpus** with Rust as the
authority: the corpus builder generates checked-in golden bytes (regenerated
`UPDATE_GOLDENS`-style, consciously, in the same commit as any wire change), the C++
encoder must reproduce them byte for byte, and a Rust test decodes the goldens and
asserts the values so the goldens themselves cannot rot. Not gRPC, not UDS.

*Why:* minimal C++ dependencies *is* a Houdini-ready requirement — gRPC drags in
protobuf + abseil, which a host ships its own ABI-incompatible copies of, and it would
force a second `.proto` schema beside the one source of truth. The C++ standard library
has no IPC at all, and loopback TCP is the one transport that is single-source-path on
every platform in both languages (`std::net`; BSD sockets ≈ Winsock to within four
lines), with no stale socket files; the token closes the any-local-process gap UDS
permissions covered. The corpus test replaces gRPC's codegen as the drift guard, aimed
squarely at the 20-field material patch, and runs in CI with no USD and no GPU. This
decision **realizes two standing deferrals**: the C ABI (D-052) arrives as serialized
change-sets over a socket rather than per-attribute FFI — the MoonRay `RDLMessage` shape
D-052 always named — and the binary change-set wire format (D-055) arrives as
MessagePack, a drop-in because file = wire = the same serde value by construction.

### D-101: Pixel transport — CPU shared memory, double-buffered, beauty + depth, server-side Rec.709
Status: accepted. Pixels cross by POSIX named shm (`shm_open` + `mmap` — the one
deliberately platform-specific piece; the Windows twin is a deferral), one segment per
size, named `/cenote-<pid>-<generation>` and carried in-band in `FbDesc`; the previous
segment is unlinked when the client's next request proves the reply was processed. The
layout is one 4 KiB header page (magic + layout version, dims, plane offsets,
`front_index`, `frame_counter`, `samples`, `converged`, `rejected_edits`) plus two
buffers, each beauty RGBA f32 + depth f32. The tear protocol is lock-free: the writer
fills the back buffer, flips the index, and release-increments the counter; a reader's
copy is valid iff the counter advanced ≤ 1 across it. No locks, no futexes — the client
can never block the render. The beauty is converted server-side from `ACEScg` to linear
`Rec.709` (one 3×3, `color::rec709_from_acescg()` — the runtime inverse of the one
authored constant, so a second hand-typed matrix never exists) before every shm write;
depth crosses unconverted. `HdRenderBuffer::GetResource()`'s GPU-texture path stays a
*measured* later upgrade (deferral).

*Why:* at lookdev viewport resolution the readback (~1.3 ms/1080p) hides behind the next
frame's render, and the high-res regime where it would bite is the converged still,
where per-frame latency stops mattering. The pivot re-verified the seam: `renderBuffer`
Bprims remain the only pixel path even under the 26.03+ engine (`HdxAovInputTask`
`Map()`s CPU pixels). The color conversion is not optional: usdview's default color
correction applies only the sRGB transfer curve — no gamut conversion exists anywhere in
its default path — so an `ACEScg` frame would render silently oversaturated; hdPrman on
dev ships the same delegate-converts fix. The step-0 integration test drives a saturated
`Rec.709` primary through the whole spine precisely so dropping the 3×3 fails loudly.

### D-102: Material scope — UsdPreviewSurface only
Status: accepted. A bounded switch from the `surface` terminal of the material network
schema (nodes, parameters, connections read as data sources; the universal render
context is the empty token, read explicitly — 26.03 removed the cross-context fallback)
covering `UsdPreviewSurface` + `UsdUVTexture` + `UsdPrimvarReader_*` into
`MaterialPatch`. Meshes with no bound material shade from `displayColor`. Four
documented exceptions: `useSpecularWorkflow=1`'s direct F0 approximates through the
specular tint; `displacement` and `occlusion` have no OpenPBR home; `opacityThreshold`'s
cutout applies delegate-side. `open_pbr_surface` recognition and MaterialX-graph
evaluation are deferrals.

*Why:* UsdPreviewSurface is the USD lingua franca every asset ships or falls back to,
and its default workflow maps near-losslessly onto cenote's shipped closure. cenote is a
fixed-closure renderer, so the MaterialX-SDK-codegen path is architecturally moot — not
deferred, nonexistent for this design — while `open_pbr_surface` (whose params *are*
cenote's) is a one-branch fast-follow on the same switch.

### D-103: Lights — UsdLux onto cenote's existing paths
Status: accepted. Params read lazily by name from the member-less light container
(UsdLux attribute names through `Get(name)`; enumeration is impossible by design;
`treatAsPoint` via raw-attribute fallthrough). distant → `Distant` delta (its default
0.53° angle collapsed — the stated floor); sphere+`treatAsPoint` → point delta; dome →
the equirect `Environment` (one active); rect/disk/sphere-area/cylinder → a synthesized
emissive mesh + emissive material + instance, placed by the light transform, radiance
`intensity·2^exposure·color`, × the blackbody RGB under `enableColorTemperature`, ÷
emitting area under `normalize`, rect/disk wound one-sided (−Z emits, matching UsdLux).
Light textures, the per-lobe response multipliers, the shaping API, and native analytic
area lights are deferrals.

*Why:* cenote already *has* area lights — emissive meshes, MIS-consistent,
ReSTIR-integrated, golden-covered, one-sided like UsdLux's — so synthesis is the correct
mechanism, not a workaround. Native analytic sampling is a core-estimator feature
justified by measured variance, never by the presence of a UsdLux prim type.

### D-104: Repo, build, and ABI shape — three components against system USD
Status: accepted. `cenote-wire` (the USD-free wire mirror + MessagePack + framing + the
shm layout constants — the Rust half of the drift guard), `cenote-server` (a binary
wrapping `render::Session`: TCP listener, request/response loop, the exhaustive
destructuring wire→`Op` translation — a field added on either side is a compile error —
and the shm framebuffer writer), and `hydra/` (a C++ CMake tree outside the Cargo
workspace: the adapter-ring delegate shell with zero Rprims, the scene-index observer,
a C++ wire mirror, transport client, and server spawn; the renderer-plugin bootstrap
glue isolated in one thin file, since that surface broke in 23.02 and 25.11 and dev
already carries 26.08's break). Built against **system-provided USD** (stock 26.05, or
the HDK — the Houdini pivot), never a vendored USD build.

*Why:* the server is plumbing around code that already exists; keeping the wire encoder
USD-free lets the drift guard run in CI with no USD and no GPU; stock ↔ HDK as a
build-root change on USD-version-agnostic source *is* the Houdini-ready rule made
concrete. Vendoring USD's enormous build is unjustified for a solo project. This
decision also hosts the remaining two deferral pickups: the **array instancer op**
(D-073) lands at step 5 as the form Hydra's instancer prims deliver, and the **viewer
single-source-of-truth scene graph** (D-064/D-082/D-083) is realized by the delegate's
model — the render index owns the scene and the delegate syncs deltas — retiring the
hand-mirrored `ui_desc` replica as the M4 answer to the M2 device.

### What step 0 shipped (the transport spine)
Status: shipped with this entry. The checkpoint holds: a Rust-only integration test
(`cenote-server/tests/spine.rs`) spawns the real binary, drives it over TCP — handshake,
resize (with the old segment's unlink proven), the `SetCamera` lane, a genesis
`Replace`, a live edit, a rejected edit surfacing through the header counter and
`Ping` — and reads correct frames out of shm under the tear protocol, including a
saturated `Rec.709` primary that fails if the 3×3 is ever dropped, plus EOF → exit 0 and
wrong-token → nonzero. Implementation notes that go beyond the plan's text, recorded so
they read as decisions rather than surprises:

- **Empty scenes render.** `Scene::prep` (and update) previously rejected a description
  with no instances; the server stands up its `Session` on exactly that (camera +
  settings singletons only), and a live edit may delete the last instance — so the core
  now preps zero instances: the empty TLAS and the instance tables pad one unread record
  each (the existing lightless-scene pattern), every ray misses, and the frame is black.
  Pinned by two GPU tests (`an_empty_scene_preps_and_renders_black`,
  `deleting_the_last_instance_updates_cleanly`).
- **Doubly-optional fields get an explicit wire spelling.** MessagePack cannot tell
  `Some(None)` from `None` — serde flattens both to nil — so the wire mirrors
  `Option<Option<T>>` as `Option<Reset<T>>` (`Reset::Clear` | `Reset::Set(v)`), and the
  server's translation unfolds it. Caught at design time by the corpus requirement that
  all three states round-trip distinctly; a bare mirror would have silently turned
  "clear the normal map" into "leave it alone".
- **The server owns the description's singletons.** A Hydra genesis carries no camera or
  settings op (the active camera is host view state on the `SetCamera` lane), so a
  `Replace` lacking them gets defaults injected; and because a camera-touching apply
  re-lowers the description camera over the inputs lane, the server re-asserts the last
  `SetCamera` after any camera-touching set.
- **Convergence is the sample cap, for now.** `Session` doesn't expose its park state,
  so the header's `converged` flag reports samples ≥ `CENOTE_SERVER_MAX_SAMPLES`
  (default 4096); a finer surface is step 2's business, alongside the refresh loop.

## 2026-07-14 — M4 step 1 opens: the wire's C++ half, drift-guarded before any USD

*Step 1's design tree was walked in a second structured interview — fourteen questions,
each resolved before the next — and the full locked detail lives in
[m4-plan.md](m4-plan.md)'s step-1 entry. Most of it is plan detail; five resolutions
are genuinely new decisions, two of them amending the plan's own leaves on inspection,
and those five are recorded here. Step 1 lands as two commits split at the USD
boundary; this entry ships with the first — the USD-free half — so the cross-language
drift guard is protecting `main` before the first line of USD-facing C++ exists.*

### D-105: The C++ baseline is C++23, under a two-part portability rule
Status: accepted. `CMAKE_CXX_STANDARD 23`, extensions off, `-Wall -Wextra -Werror`,
everywhere in `hydra/`. The self-imposed rule is two-part: portable core C++23 only —
no modules, no coroutines — and *inside the plugin `.so`* no library facilities that
demand new libstdc++ runtime symbols (`std::println`, `<stacktrace>`). Those live
freely in the USD-free tools (the corpus exe, tests); plugin logging is `TF_*` by
convention anyway. If the step-6 HDK build ever objects to the flag, downgrading the
plugin target alone is one line.

*Why:* C++23's readability wins (`std::println`, `std::expected`) are real and cheap —
the build-chain cost is a g++-14 pin in CI. The plugin-side ban exists because the
delegate is a shared library dlopen'd into a host's process, and a host like Houdini
launches with its bundled older libstdc++ on `LD_LIBRARY_PATH`: a plugin referencing
GLIBCXX_3.4.32+ symbols fails at load, in someone else's process, with someone else's
error message. The rule keeps the failure impossible rather than diagnosable.

### D-106: The wire's C++ half is hand-rolled — a minimal msgpack codec and 1:1 mirror structs
Status: accepted. No third-party msgpack library: `hydra/wire/` owns a minimal writer
plus a small response reader covering exactly the encodings the wire uses, zero
dependencies. On top of it, structs that mirror `cenote-wire`'s types 1:1 — Rust's
exact type and field names, `std::optional` for `Option`, `std::variant` for the
enums — with one `encode()` per type walking fields in Rust declaration order, and
designated-initializer construction throughout (C++ requires declaration order there,
so every construction site is a small field-order check). Decode stays asymmetric and
tiny: the client only ever reads `Welcome`/`Ack`/`Resized`, hand-decoded, strict —
unknown variant names, malformed frames, and trailing bytes are refused with an offset,
never skipped.

*Why hand-rolling is the conservative choice here:* the format subset is tiny and
frozen, the 12 checked-in goldens are a byte-exact conformance suite either way (a
library's output would still need pinning against them), and D-100/D-104 already treat
every C++ dependency as host-ABI liability — MoonRay's own Arras wire is a bespoke
in-tree encoder, not a framework. *Why mirror structs over a direct builder:* the
structs are the artifact that makes the drift guard meaningful — the corpus exe encodes
the same types the production observer will construct, so the guard exercises the real
encode path, not a replica. One honest limit, documented in the module header: C++ has
no exhaustive-destructuring trick, so a field added to a C++ struct but forgotten in
its `encode()` is not a compile error the way it is server-side — the corpus's "every
field `Some`" coverage rule is the only guard, which is why it is load-bearing on every
wire change.

### D-107: The step-1/2 line redrawn — step 1 carries a skeletal-but-honest pixel path, depth included
Status: accepted; amends the plan's step ordering. The plan put the render buffer,
render pass, resize, and convergence in step 2, yet step 1's checkpoint — "a mesh
renders in usdview under a real camera" — is impossible with zero pixel path. The
redraw: step 1 includes `Allocate()` → `Resize` → remap (usdview allocates
viewport-sized buffers immediately, against the server's 1280×720 boot default — this
is required for *first* pixels, not polish), the tear-protocol `Map()` from day one (a
"temporary" tear-ignoring version is the kind of placeholder that outlives its excuse),
camera → `SetCamera` on change, and `IsConverged()` = false (usdview just keeps
repainting — correct-enough and zero code). Depth folds in too: usdview's task
controller requests color+depth by default and the shm depth plane has existed since
step 0, so mapping it is one memcpy. Step 2 keeps its identity — *interactive and
unkillable*: honest convergence from the header, the throttle, resize robustness,
rejected-edit surfacing, dead-socket degradation.

### D-108: The distant light arrives in step 1
Status: accepted; amends the plan's step ordering. Step 1 as ordered rendered a black
frame: lights were step 4, materials step 3, the server boots black-sky empty, and
even usdview's default camera light arrives as a scene-index light prim — exactly the
translation step 1 lacked. The fix is the smallest honest one: pull the distant light
forward — direction is the prim's world −Z, radiance `intensity·2^exposure·color`, the
default 0.53° `angle` collapsed to the delta per the locked floor — real step-4 code
arriving early rather than scaffolding. The alternatives were worse: an emissive
checkpoint stage pulls material translation forward (a far bigger bite); a server-side
debug light pollutes the server's contract with a fiction the delegate would have to
undo; a black-frame checkpoint verified by depth alone guts the checkpoint's purpose.
Step 4 is now the *rest* of UsdLux, and the checkpoint stage carries its light
honestly.

### D-109: Visibility means removal, not camera invisibility
Status: accepted; corrects a plan leaf. The plan's leaf said "`camera_visible` from
visibility," and it is wrong on inspection: cenote's `camera_visible=false` is a
*primary-ray* flag — the object still shadows and reflects — while USD invisibility
means *gone*. Mapping one onto the other would leave ghost shadows from hidden prims.
Corrected: visibility=false **removes the instance** — the mesh payload stays
server-side, so re-showing is a cheap instance re-add with no geometry resend —
and `camera_visible` stays reserved for its true USD counterpart later.

### What the first commit shipped (the USD-free half)
Status: shipped with this entry. The `hydra/` skeleton (top-level CMake + a
standalone-buildable `hydra/wire/`, `.clang-format` scoped to the tree), the codec, the
mirror, the corpus conformance test, and the CI steps — every piece provable on a bare
runner, no USD, no GPU. Implementation notes that go beyond the interview's text,
recorded so they read as decisions rather than surprises:

- **The doubly-optional and texturable fields keep their wire spellings as types.**
  `Reset<T>` is `std::variant<Clear, Set<T>>` and `Texturable<T>` is
  `std::variant<Constant<T>, TextureRef>`, so the three states of a patch's
  `Option<Reset<T>>` stay distinct in C++ exactly as they do on the wire — the D-100
  lesson (serde flattens `Some(None)` to nil) carried across the language.
- **The corpus test runs both directions.** Beyond encode-vs-golden byte equality, the
  four response goldens are decoded and re-encoded to byte identity, and three
  strictness probes assert refusal: a request payload does not decode as a `Response`,
  an unknown variant name is an error, and trailing bytes after a complete message are
  an error. Twelve goldens, symmetric set equality, hex-window diagnostics at the first
  divergence.
- **CI pins by the mechanism each tool offers.** g++-14 by apt package name — Ubuntu
  ships exactly one g++-14 per release, so the name *is* the pin (GCC 14 because
  `std::println` needs its libstdc++, one past the runner default); clang-format by
  exact version through the PyPI wheel (`pipx install clang-format==22.1.8`, matching
  the local install — apt stops short of 22, and formatter output drifts across
  majors). The format check `find`s all of `hydra/` so the USD half is covered the day
  it lands. Every new step was verified in an `ubuntu:24.04` container before landing:
  g++-14 compiled the mirror `-Werror`-clean, ctest passed, the pinned clang-format
  reported the tree clean.

### D-110: Depth crosses the delegate in the projection's [0, 1], not in meters
Status: accepted (2026-07-14); corrects a D-107 leaf. D-107 said the depth AOV is "one
memcpy" of the shm plane, and the first real geometry proved that leaf wrong on
inspection: the shm plane carries camera-plane z in meters with +inf where every sample
missed, while Hydra's depth consumers assume the projection's [0, 1] — usdview's
visualizer min/max-normalizes the buffer, so a single +inf background pixel flattens
every real depth to black (the checkpoint cube vanished behind exactly this). The fix
keeps both contracts honest: the server's plane stays linear meters (+inf is the truthful
"no hit"), and the delegate remaps on the plane copy — only z is needed, so it is the
projection's z/w rows and a divide per pixel, hdEmbree's conversion without the full
point transform. Misses land exactly on the advertised 1.0 clear value. The render pass
hands each bound buffer the frame's projection; a rejected alternative was baking [0, 1]
into the shm contract, which would push a Hydra-ism into every non-Hydra client.

### D-111: Two light Sprim types are advertised so the camera light exists at all
Status: accepted (2026-07-14); an implementation note under D-108. D-108 counts on
usdview's default camera light arriving as a scene-index distantLight prim, but
HdxTaskController only injects its built-in lights when the delegate advertises Sprim
support for domeLight AND a camera light type (simpleLight or distantLight) — an
all-or-nothing gate, checked against the render index, that a scene-index-only delegate
still has to pass. So the delegate advertises distantLight and domeLight and backs both
with an inert HdLight subclass whose Sync does nothing: the render index gets the
vocabulary it demands, while the actual translation keeps reading the same prims from
the terminal scene index (zero Rprims stays true; the Sprims are decoys). The dome
light stays untranslated until step 4 — enabling it in usdview simply does nothing —
and simpleLight stays unadvertised so the task controller chooses the distantLight
spelling, the one translator step 1 carries. The camera light rides the free camera:
its transform is the view inverse, so every orbit re-patches the light's direction
alongside SetCamera, which is the honest reading of "a light attached to the camera".

### D-112: Honest convergence arrives with the smoke test, not with step 2
Status: accepted (2026-07-14); pulls one D-107 leaf forward. D-107 parked
`IsConverged()` at false — usdview just keeps repainting — and deferred the honest
answer to step 2. The step-1 test slice broke that parking spot: `usdrecord` loops
render-until-converged, so a never-converged delegate spins its recorder forever and
the checkpoint cannot exist. The honest answer was already sitting in the shm header
(`converged`, set when accumulation reaches `CENOTE_SERVER_MAX_SAMPLES`), so the
delegate now reads it: false until a first publish, true when the header says settled,
and true whenever the client is degraded — a dead server's picture will never improve,
and a render-until-converged host must not hang on it. A render buffer whose segment
trails its allocation is a resize still settling, and answers false. One recorded
caveat: the flag refreshes only at publish, so a poll racing an edit can read one
stale "converged" for at most a publish interval (~33 ms); usdview self-corrects on
its next paint, and the smoke run never sees it because its resize hands it a freshly
zeroed segment. The finer surface (per-edit scene epochs in the header) stays step 2's
business, as D-107 said.

### D-113: The session epoch — converged means settled *and* current
Status: accepted (2026-07-14); supersedes the ≤~33 ms stale-flag caveat D-112 recorded.
The shm header's `converged` flag says "accumulation settled"; it never said "at a
picture that includes your edit," and D-112 recorded the resulting lie as a caveat: a
poll racing an edit could read one stale converged. Step 2 retires it with a
session-owned epoch. An atomic counter in the session's shared lanes is bumped by
exactly the four wire verbs — `apply`, `replace`, `set_camera`, `resize` — *after*
each places its payload; viewer toggles restart accumulation anyway and do not bump.
At each wave boundary the render thread reads the counter *before* draining edits and
snapshotting inputs, and stamps every frame it subsequently publishes with that value:
"everything enqueued up to E is incorporated in this picture." A drained edit counts
as incorporated whether it applied or was rejected — rejection must not wedge
convergence. The one delivery hole is plugged deliberately: a visually inert edit
while parked (the equality gate means no restart, so no new publish) still republishes
the current buffers under the fresh stamp, so the epoch always arrives. On the wire,
`Ack` and `Resized` carry the server's post-request epoch (`Welcome` stays unchanged:
both sides start at 0, and the genesis `Replace` resynchronizes), and the shm header
carries the front frame's stamp — a protocol and layout version bump each. The client
remembers the largest epoch any `Ack` or `Resized` carried, and its `converged()`
becomes: degraded, or the front frame's epoch has reached that bar *and* the header
says settled. A settled stale picture no longer claims to be final, which is also the
viewer's edit-vs-converged story arriving as a side effect — usdview keeps painting
until the picture with the edit in it has settled, not merely until any picture has.

### D-114: The camera's vfov reads the conformed projection, not raw HdCamera
Status: accepted (2026-07-14); completes the symmetry D-110 started. The render pass
was decomposing HdCamera's apertures into a vertical field of view while the depth
remap read the pass state's *conformed* `GetProjectionMatrix()` — two camera stories
per frame, free to drift apart the moment a framing policy conforms the projection.
Now one matrix rules both: `P[1][1]` is `1/tan(vfov/2)` for any conformed perspective
projection, off-center included, so the pass reads `vfov = 2·atan(1/P[1][1])` and the
frame the server renders and the depth read back from it share one camera. HdCamera
still supplies what a projection cannot: the transform, the perspective check, and the
lens (focus distance, fStop, and focal length — the last only to turn the f-number
into an aperture radius). Rejected alternatives: consuming `CameraUtilFraming` in
full, because data windows and crop regions say more than one field of view can
express and cenote renders the full frame; and hand-rolling the conform math, which
would duplicate what the pass state has already done. Framing beyond a plain
full-frame viewport — a data window apart from the display window, or non-square
pixels — gets a one-time `TF_WARN` and renders the full frame at the vfov the
conformed projection implies.

### D-115: Material bindings and lifecycle — the registry and the symmetric hooks
Status: accepted (2026-07-14). Material prims get their own translator — the third,
beside mesh and light — publishing one wire material named by the prim's path, so a
shared material is one wire object referenced by many instances and a material edit
touches one patch. Every mesh keeps publishing its unconditional `<path>/displayColor`
companion as the permanent fallback. Binding resolution is registry-checked: the mesh
reads the pre-resolved all-purpose `HdMaterialBindingsSchema` binding and points its
instance at the bound path only when the material registry has it live on the wire
(a `Published()` guard, since a translator can exist before its first sync); a
dangling binding warns once naming mesh and target, and the instance wears the
companion. The check is load-bearing, not polite: server validation is post-set and
atomic, so an instance naming a missing material rejects the *whole wave* and takes
unrelated same-wave edits with it — the registry keeps every wave valid by
construction. Hydra generates no `materialBindings` dirty when a material prim itself
appears or dies, so the lifecycle holes are closed by symmetric hooks through the
registries: on material birth, walk the mesh registry and repoint every mesh whose
resolved binding names this path (the late-arrival hole); on death, unpublish, walk
the bound meshes back to their companions, *then* append the `Remove` — repoints and
removal land in one flush, so post-set validation can never see a dangling reference.
The mesh owns its repoint (`ResolveBinding()`, called by hooks and its own sync alike)
so its cached wear never drifts from the wire; the material translator never
fabricates patches for prims it doesn't own. Two supporting pieces: the notice-batching
priority functor flushes materials ahead of everything else — amending the interview's
one-bucket leaf, because a stage authoring geometry before its `Looks` scope would
otherwise spuriously warn on every binding at genesis and create-then-repoint every
instance — and a Storm-identical binding-purpose resolving scene index
(`{preview, allPurpose} → allPurpose`, inserted at start) so stages that author
`material:binding:preview` specifically still resolve. Rejected alternatives:
mesh-side resolution (N copies of shared materials, and no clean path for
material-prim invalidations to reach the meshes wearing them); never removing wire
materials (leaks texture residency for the session); parking material removals until
references drain (a deferred-op machine for a case the hooks handle in a dozen lines).

### D-116: The texture channel crosses the wire
Status: accepted (2026-07-14). `TextureRef` gains an optional source channel
(`r`/`g`/`b`/`a`; absent = red, the prior behavior) — the step's one wire growth. The
core's scalar textures are BC4, baked from a single source channel that was hardwired
red; real UsdPreviewSurface assets feed `roughness`/`metallic` from a packed ORM's
`outputs:g`/`outputs:b` and — the ubiquitous case — cutout `opacity` from the diffuse
texture's `outputs:a`, so read-red is silently wrong for exactly the assets the step-3
checkpoint names ("matching their authored look"). The ripple is contained and
mechanical: prep bakes BC4 from the chosen channel and the channel joins the DDS cache
tag and content hash — red deliberately spelled as the empty tag, so every cache
written before channels existed stays a hit; only the scalar usage reads the selector,
color and normal usages normalize it to red so a stray channel can never fork their
caches; both wire mirrors and the corpus goldens regenerated in the same commit (the
drift guard never goes red mid-history); the server's exhaustive destructuring picked
the field up by compile error, as designed; protocol 2 → 3, shm layout untouched.
Baked-channel BC4 was chosen over the runtime-swizzle school (sample RGBA, pick in
shader): half a byte per texel instead of a full BC7 for one used channel, and the
cache keyed honestly — a packed ORM preps the same PNG up to three times, a prep-time
cost, not a runtime one.

### D-117: Mapping fidelity — silent where native, one warning where lossy, never a rejected wave
Status: accepted (2026-07-14); amends one D-102 exception. The material translator's
whole degradation policy as one principle. **Silent** when the authored value is what
the core does anyway: wrap `repeat`/`useMetadata`, identity `scale`/`bias`, the
canonical normal-map `(2,…)/(−1,…)` remap (the BC5 path remaps natively),
`sourceColorSpace` `auto` spelled as *absence* (the core's per-slot auto — 8-bit color
reads sRGB, scalars and normals raw — is exactly `auto`'s spec semantics), an
unconnected `st` input, and both `opacityMode` spellings onto coverage. **One warning
naming the material and input** where fidelity is actually lost, rendering anyway:
foreign wraps render repeat; `UsdTransform2d` is skipped through to its reader
(placement drifts, the asset keeps its look); a reader with `varname ≠ "st"` degrades
that texture to its `fallback` constant (the mesh never shipped those coordinates —
sampling `st` anyway would be silently-wrong placement); readers connected directly to
surface inputs use the *reader's* fallback (per-vertex data has no wire home, and a
shared material can't resolve a per-mesh primvar); a constant non-default `normal` is
ignored; `displacement`/`occlusion` warn only when actually authored or connected;
textured opacity under a nonzero `opacityThreshold` is sent un-thresholded — soft
coverage instead of hard cutout — while constant opacity binarizes delegate-side.
**Never a rejected wave**: `validate_path` demands an absolute existing file and one
failure rejects the whole wave atomically, so the delegate pre-checks every resolved
path — broken paths and UDIMs degrade to the texture node's own `fallback` input as a
constant — and a foreign surface identifier (or no surface terminal at all) publishes
the wire material anyway wearing explicit OpenPBR defaults under one warning, keeping
every instance that binds it valid. The amendment: D-102 said `useSpecularWorkflow=1`'s
direct F0 "approximates through the specular tint," and that sentence is
unimplementable as written — no specular color/tint field exists on the wire, the
description, or the GPU material. Instead the F0's Rec.709 luminance collapses onto
`specular_ior` via ior = (1+√F0)/(1−√F0), clamped to [1, 5], the hue dropped under the
one warning — more honest than common practice, which silently ignores the flag
entirely.

### D-118: One environment on the wire, any number of domes in the stage
Status: accepted (2026-07-16). Two layers, split at the wire. **Below Hydra** (its own
commit, drift guard regenerated both directions): the environment's image becomes
optional — no path is a constant white sky through the same 1×1 code path the furnace
tests trust — and two dials ride beside it, a linear Rec.709 tint folded over the
image's radiance and a world-from-dome placement whose linear part turns the whole sky.
Both apply at sampling time, in the one place every kernel entry point (radiance,
sample, connect, locate, pdf) crosses the placement, so ReSTIR reconnection and MIS
stay exact under a turned sky; a tint or placement edit keeps the resident image by
source path. Radiance `.hdr` joins `.exr` as a decodable sky, told apart by magic bytes
rather than extension; a placement must invert (sampling maps world directions back
through it), validated at apply beside the instance rule; `EnvironmentPatch` gains the
doubly-optional path, the tint, and the transform in both mirrors, protocol 3 → 4.
**In the delegate**, a fourth translator — `HdCenoteDomePrim`, beside mesh, light, and
material — owns the fact that cenote renders exactly one environment while UsdLux
admits any number of domes. Its registry is an ordered map, and the order *is* the
arbitration: the lowest visible `SdfPath` holds the slot, every other eligible dome is
parked under a latched warning naming the winner (the latch re-arms when the parked
dome later publishes — a later demotion is a fresh parking). Handoffs are atomic —
demotion's `Remove` and the successor's patch ride one flush, so the wave never carries
two skies merged nor none while a contender stands — and a dying winner fails over from
its destructor: the survivors' cached payloads are re-arbitrated in the same flush the
`Remove` rides, so the sky never goes dark between domes. Placement bakes the 180° yaw
between the two equirect conventions innermost (USD centers the image on +Z, cenote on
−Z — the wire always speaks cenote's convention, the server never learns USD's),
composed with `domeOffset` (usdImaging's spelling of DomeLight_1's pole axis; the
original DomeLight serves none and gets identity) and the flattened world transform. A
degenerate placement falls back to the bare yaw instead of withdrawing — unlike a mesh,
a dome with a broken transform still lights the scene, the sky has no size to
collapse — under a warning gated on the *transition* into degeneracy (a latch would go
quiet forever; no gate would fire on every unrelated drag). Texture admission is the
exact inverse of the rect light's rule: float formats only (`.exr`/`.hdr` — the
environment is radiance), and `texture:format` `latlong` or `automatic` only, checked
last, once a file is actually at stake; every rejection degrades that one dome to its
constant-sky tint under a latched warning, never the flush and never the slot.

### D-119: Area lights are ordinary instances wearing a black absorber
Status: accepted (2026-07-16). The rect, disk, sphere, and cylinder lights become the
triple the plan promised — a `MeshPatch` of object-space triangles with the authored
dimensions baked in, a `MaterialPatch`, an `InstancePatch` carrying the flattened
matrix untouched — all named by the light's path, riding the same wire ops as any
geometry; the renderer keeps no analytic light shapes, emissive triangles being its
native area light. The load-bearing choices: **`camera_visible = false`, always** — the
light illuminates, its shape is a sampling device, not scenery — matching what Arnold
and MoonRay default to, since no USD-standard attribute exists to ask for the shape;
one divergence is accepted openly rather than papered over: the camera-invisible
emitter still blocks shadow rays (the triangles physically exist server-side), where
per-ray-type-visibility renderers hide area lights from occlusion too — physical, and
fixing it honestly needs a new ray-mask bit on the wire, deferred beyond M4.
**Geometry**: rect as two triangles restating the schema's texture frame in cenote's
v-down UV convention; disk a 64-segment fan; sphere a 32×16 UV sphere; cylinder a
64-segment capless tube along X, UsdLux's own reading of the shape; rect and disk wound
one-sided emitting −Z, and a negative-determinant placement flips every winding triple
so a mirroring transform keeps the emitting face on the side UsdLux lights. Dimensions
clamp non-negative — a negative width would silently flip the emitting side. **The
black absorber**: the camera never sees the surface, but reflections and shadow rays
still hit it, and the wire's default 80% gray would bounce light the light never
emitted — so black diffuse, zero specular weight, emission only. **Rect textures**:
LDR only — the emission slot rides the BC7 color pipeline, which would clip a float
source to display range rather than reject it; the map replaces the constant color on
the wire and the shader multiplies it onto the scalar luminance alone, so the tint has
one lane left: its Rec.709 luminance folds into the scalar and any hue is dropped under
a latched warning. **The normalize guard**: the area divided out is the world-space sum
over the very triangles being sent — what the renderer integrates is exactly what the
divisor weighed — and zero area keeps divisor 1, silently: the shape is already dark by
geometry, the reading UsdLux prescribes for its own sizeFactor. Lifecycle mirrors the
other translators: total re-read and total resend on any dirt in the lane, a spelling
flip (a sphere toggling `treatAsPoint`) withdrawing the old objects in the same atomic
flush the new ones ride, and a degenerate transform costing that one light — withdrawn
under a warning — never the flush it rides in.

### D-120: UsdLux radiometry — the formulas, and the distant correction
Status: accepted (2026-07-16); amends D-108's implicit-normalize shortcut. Every light
shares one radiometric spine: tint = `color`, times the luminance-normalized blackbody
of `colorTemperature` when `enableColorTemperature` — via `usdLux`'s own
`UsdLuxBlackbodyTemperatureAsRgb`, linked rather than re-derived — and scale =
`intensity·2^exposure`; `normalize` divides by the emitting quantity each shape defines.
**The distant correction**: D-108 read intensity as the delta's irradiance directly,
which is an implicit `normalize`. The schema's own constants refute it: the 50000
default intensity that "approximates sunlight" coheres only if intensity is the
*luminance* of the 0.53° default disk, whose π·sin²(θ/2) ≈ 1/15,000 sr multiplies out
to ≈ 3.4 units of irradiance — a sane sun; read as irradiance, the same defaults are a
15,000× blowout. So an unnormalized distant light delivers
`tint · scale · π·sin²(angle/2)`, and `normalize` (or a zero angle) makes intensity the
irradiance itself; `first-light.usda` now authors `normalize = 1` with a comment saying
why its 3 units survive. **treatAsPoint** (spheres only, the raw-attribute
fallthrough): radiant intensity is luminance times the projected area, `I = L·π·r²` —
or `intensity/4` when normalize has already divided the `4π·r²` surface out — with
`r_world = r·|det|^(1/3)`, the one isotropic reading of an anisotropic stretch a point
cannot express. The per-lobe `diffuse`/`specular` multipliers warn once and are
ignored — an artistic split cenote's one transport does not carry — under D-117's
warn-where-lossy policy, which this step extends without a new entry: silent where the
authored value is what the renderer does anyway, one warning naming the prim where
fidelity is actually lost, never a rejected wave.

### D-121: Instancing is arithmetic, not an object — the mesh carries the copies
Status: accepted (2026-07-17); realizes the array-instancer deferral (D-073). cenote
has no instancer: an instance is a mesh reference, a material, and a placements array —
the array the wire grew for it below Hydra, `InstancePatch.transforms` widened from a
single transform to a `Vec`, empty legal, each element its own stable light identity
`name#i` so a copy's reservoir history is its own, protocol 4 → 5, scene-file version 2,
goldens regenerated both directions. So a Hydra instancer translates to **nothing on the
wire** — a new translator, `HdCenoteInstancerPrim`, the only one of the five that
publishes no patch. It answers one question instead, `ComputePlacements(prototypeRoot)`:
the world-space matrices a prototype's prims should be copied to. A mesh inside a
prototype reads its own `instancedBy`, asks each named instancer, and authors the
concatenated array as its `InstancePatch.transforms`; an un-instanced mesh sends its one
placement as a length-1 array, unchanged.

**Composition is hdEmbree's, matrix for matrix**: per instance,
`matrix · S · R · T · instancerXform` in Gf's row-vector convention, reading both the
disaggregated `hydra:instanceTranslations/Rotations/Scales` primvars *and* the aggregated
`hydra:instanceTransforms` matrix form native instancing emits — any channel the
instancer omits, or an index past its end, is the identity for that component, so a bare
PointInstancer and a twice-referenced native instance walk the same arithmetic. The mask
(`inactiveIds` metadata, `invisibleIds` attribute) is not re-derived:
`HdInstancerTopologySchema::ComputeInstanceIndicesForProto` folds it in and returns the
surviving indices in order — a fully-masked prototype yields an empty array, resident and
placing nothing, which is exactly why the wire made empty legal rather than forcing a
remove/re-create round trip. **Nesting is the cartesian product** the `instancedBy`
parent chain implies: a mesh or a native instancer inside another instancer's prototype
composes through every level, child transform innermost, several parents concatenating —
`ComputePlacements` recurses up the chain, so a native instance scattered by a
PointInstancer folds two instancer levels at once with no special case.

**The poke, borrowed from materials** (D-115): an instancer edit dirties the instancer
prim but never the prototype prims it moves (their flattened transform is
prototype-root-relative and never sees the instancer), so the translator has no patch of
its own to dirty — birth, edit, and death all walk the mesh registry and every instanced
mesh recomposes from scratch. Broad on purpose: recomposing every instanced mesh on any
instancer change is what makes a nested edit correct without tracking who nests whom, and
the placements are a pure function of the whole prim, so any dirt at all pokes — no
per-locator surgery. Removal is RAII with nothing to withdraw server-side; the destructor
only pokes, unless a resync already handed the path to a successor. **What it drops**:
per-instance shading primvars — a color, a width authored at `instance` interpolation —
have no home when an instance is only a placement wearing a shared material, so they warn
once, named, and are dropped; the velocity family (`velocities`, `accelerations`,
`angularVelocities`) is dropped in silence, motion blur being deferred whole. Rejected
alternatives: a wire instancer object (a second scene-graph concept for what the renderer
already expresses as N placements — the estimator's TLAS is flat regardless, so the
object would only be un-flattened again server-side); mesh-side re-derivation of the mask
(a second copy of `ComputeInstanceIndicesForProto` drifting from usdImaging's); and
decomposing the aggregated native-instance matrix back into TRS (lossy, and the
composition consumes the matrix directly anyway).

### D-122: Houdini-ready means husk renders one frame — the proof the installed host makes cheap
Status: accepted (2026-07-18); the seventh structured interview, informed by
production-delegate research (hdPrman, hdMoonray, Karma/husk, hdCycles). Reinterprets
D-097's "usdview is the milestone, Houdini a later one" by one notch, and closes M4. The
trigger is a fact on the build machine: Houdini `hfs21.0.671` is installed, shipping USD
**25.05** with HDK headers — so the "HDK compile proof" the plan penciled as a
first-to-slip fallback is not a compile at all but a **load-and-run**: husk itself, the
host's own 25.05, rendering cenote's pixels. Buying that costs almost nothing over the
compile it replaces, so step 6's definition of done grows from "the source compiles
against the HDK" to **"`husk --renderer Cenote` renders one frame,"** the usdview FLIP
golden beside it and every heavier husk feature deferred by name.

**The build learns a second USD, not a second source.** `find_package(pxr CONFIG)` stays
exactly as it was for stock 26.05 (usdview, CI, the whole existing gate untouched); an
`-DCENOTE_USD_FLAVOR=hdk` branch enters instead through `toolkit/cmake/HoudiniConfig.cmake`
— the HDK ships **no `pxrConfig.cmake`**, only Houdini's, and its USD is the
`pxr_boost`-namespaced flavor built to Houdini's ABI. That blessed entry point is the whole
reason to prefer it over a hand-rolled include/lib prefix: it gets the
`_GLIBCXX_USE_CXX11_ABI` choice, the `dsolib` RPATH, and the pxr_boost namespace right for
free, and those three are exactly the "will the `.so` even load" landmines a by-hand prefix
reproduces and gets subtly wrong. The source files do not move; the delegate's three link
libraries (`hd hdsi usdLux`) map onto Houdini's imported targets and nothing else changes.

**The schema drift the risk-watch feared is already absorbed by the aliases.** 25.05 trails
the 26.05 pin by a release cycle, sitting *before* the 25.11 material-network rework — and
cenote reads materials through `HdMaterialNodeContainerSchema` and
`HdMaterialConnectionVectorSchema`, whose same-named `.h` files exist in neither USD because
both are `using` aliases in `schemaTypeDefs.h`. The rework renamed only the underlying
template (`HdSchemaBasedContainerSchema` in 25.05 → `HdContainerOfSchemasSchema` in 26.05);
the alias names and their `.Get()`/`.GetNodes()` surface are stable across both, and cenote
codes to the alias, never the template. So the material reader is expected to compile
against 25.05 **untouched** — the trial compile is the proof, and a single isolated
`usdCompat.hpp` shim is added only if that compile actually breaks, never pre-emptively. The
husk-facing API cenote calls — the `HdRenderSettingsMap` constructor overload, `Stop(bool)`,
`GetRenderStats`, `IsStopSupported` — is present in 25.05 already, so none of it forces a
version guard.

**The husk surface is the smallest thing that renders.** husk sends its initial settings *en
masse* through `CreateRenderDelegate(HdRenderSettingsMap const&)`, so that overload must
exist or the delegate never initializes — but it **forwards to the no-arg path and ignores
the map**, because resolution already reaches the delegate the way usdview delivers it,
through `HdRenderBuffer::Allocate`, not the settings dictionary. Beside it a three-line
`UsdRenderers.json` (`valid`, `menulabel`, `aovsupport:true`) is what makes husk list cenote
and treat its color AOV as a raster product. That is the entire husk-enabling diff;
everything else husk *can* consume, cenote consciously does not yet.

**The end-to-end golden guards the one thing the wire drift guard cannot see** — the
server-side Rec.709 conversion: a saturated primary rendered through the full usdview/26.05
path and FLIP-compared to a committed golden, so the day that conversion is dropped the
primary blows out and the golden fails loudly. FLIP is the metric already trusted repo-wide
(`nv-flip`, `crates/cenote/tests/golden.rs`), so rather than admit a second,
possibly-disagreeing FLIP into the Python test env — against the standing dependency policy
(D-100/D-104) — a ~80-line `cenote-flip` CLI wraps the same crate and the Python smoke shells
out to it. The **usdrecord/26.05 render is the pixel oracle**; the **husk/25.05 render is
not** — the server is byte-identical across both hosts, so a second golden would only assert
pixel-parity between two renderers that conform framing and handedness differently. husk
instead gets the lighter, flip/mirror-agnostic colour-bucket smoke the other stages use: its
job in this milestone is to prove *load-and-run against 25.05*, not to be a second oracle.

**Discovery stays env-driven, the package stays flat.** The `$CENOTE_SERVER` override already
makes the server a one-line export under husk, so step 6 adds no relocatable-package
machinery — `PXR_PLUGINPATH_NAME` for the plugin, `HOUDINI_PATH` for `UsdRenderers.json`,
`CENOTE_SERVER` for the binary, all documented, nothing installed beside the `.so`. And the
whole milestone degrades gracefully at its one external dependency: **if husk is unlicensed
on the machine, B becomes the compile proof it replaced** — the `find_package(Houdini)` build
still links a load-ready `.so`, the usdrecord golden still stands — with no design change,
only a slipped final assertion. The license check is the first implementation step, not a
surprise at the end.

**Deferred to the real Houdini-integration milestone, by name** (research-justified, so the
omissions are choices not gaps): `GetRenderStats`/progress reporting, `Stop(blocking)` beyond
the stub / `Pause` / `Resume`, the `restartrendersettings`/`restartcamerasettings`
declarations, `HdRenderSettings`-Bprim consumption, deep and cryptomatte AOVs, the
dialog-script LOP parameter UIs, and the self-contained relocatable package (server beside
the plugin, `dladdr` self-location). Rejected alternatives: a hand-rolled HDK prefix
(reproduces by hand the ABI flags `HoudiniConfig.cmake` sets correctly); a Python FLIP
dependency (a divergent second oracle and a new dep where an 80-line wrapper of the existing
one suffices); dual FLIP goldens across two host renderers (brittle pixel-parity for no
coverage the single oracle lacks); and consuming the husk settings map now (the render buffer
already carries the only setting — resolution — that one frame needs).

## 2026-07-24 — M4 step 6 lands: the golden caught the gamut bug it was built to catch

### D-123: The saturated-primary golden caught a real gamut bug — the delegate converts displayColor, the server clamps its Rec.709 output
Status: accepted (2026-07-24); implementing D-122's step-6 plan. D-122 framed the
end-to-end golden as guarding "the one thing the wire drift guard cannot see — the
server-side Rec.709 conversion." Built, it did exactly that on its first render, and
the thing it caught was not a *dropped* conversion but a *missing* one on the way in,
compounded by an unclamped one on the way out. Recording what the golden found and the
two fixes that make it pin correct pixels, because the plan predicted the mechanism in
the abstract and the implementation found it concrete.

**The bug: a saturated primary rendered as `NaN`.** The color pipeline has two sides.
On the way *in*, an authored linear-`Rec.709` color becomes `ACEScg` (the render space)
through `cenote::color::acescg_from_rec709` — the pbrt importer does it, the `.ron`
front end does it, and so the shader never sees a saturated-`ACEScg` albedo. On the way
*out*, `cenote-server` converts the `ACEScg` frame back to `Rec.709` for the display
host (D-101's hoisted 3×3). The delegate skipped the inbound half: `_ReadDisplayColor`
sent `displayColor` **raw** as the server's `base_color`, so a `displayColor` of pure
red reached the shader as `ACEScg` `(1,0,0)` — a color *more saturated than Rec.709 can
hold*. The outbound 3×3 then maps it to a `Rec.709` triple with two **negative**
components (green → `(-0.62, 1.14, -0.13)`), and a consumer's transfer curve raises each
component to a power — `powf` of a negative is `NaN`. The golden stage's three saturated
primaries turned ~300 pixels to `NaN`; the demo and every in-tree golden never did,
because their colors are converted on the way in and so are always in-gamut on the way
out. This is why the golden had to be *saturated primaries* and not the warmer palette
of `first-light`: only a color at the gamut edge exposes the missing conversion.

**Fix one — the delegate converts, like every other front end.** `_ReadDisplayColor`
now applies the `Rec.709 → ACEScg` 3×3 (the C++ twin of `ACESCG_FROM_REC709`, matrix
verbatim, drift pinned by the golden's own round-trip). A `displayColor` red round-trips
to a `Rec.709` red on screen, in gamut, no negatives — and, rendered through the OpenPBR
default dielectric specular layer, correctly *desaturated* by its white sheen rather
than the over-saturated primary the raw path produced.

**Fix two — the server clamps, completing D-101's silent-gamut defence.** Even with the
inbound conversion, genuinely wide-gamut `ACEScg` content (a future texture or authored
material outside `Rec.709`) can still map to negative `Rec.709`. A display cannot emit
negative light, so `rec709_texels` clamps the conversion output to `≥ 0` — the honest
gamut clip. In-gamut color is untouched and highlights above one are kept; only the
below-zero out-of-gamut lobe is clipped. Two fast unit tests pin it (saturated primaries
clamp non-negative; grey survives), so the guard no longer rides only on the GPU golden.

**Known follow-up, by name.** `materialPrim.cpp` reads a `UsdPreviewSurface`'s constant
`diffuseColor`/`specularColor` the same raw way `_ReadDisplayColor` used to — the same
missing inbound conversion. No golden covers it yet (the preview-surface stage drives its
base color from a *texture*, which carries its own color-space handling), so it is a live
gap, not a caught bug: the one-line twin of fix one, deferred until a constant-color
material front end is exercised.

## 2026-07-24 — M6 locked: full path reuse (ReSTIR PT), the plan and its amendments

M6 is the reordered next milestone (geometry depth deferred — the charter's swap condition,
a rendering-research primary target, is met). It builds on M3's GRIS-DI, not on geometry.
These entries lock the decisions the structured interview and the research review produced;
the working plan is [m6-plan.md](m6-plan.md), and everything consciously *not* built lives in
[deferrals.md](deferrals.md) with its revival trigger. The research pass read the target
paper **ReSTIR PT Enhanced** (Lin, Kettunen, Wyman, I3D 2026) to the equation level against
its supplemental, plus the SIGGRAPH 2023 course, ReSTCV (SIGGRAPH 2026), and the engine
hygiene of MoonRay (HPG 2017) and Cycles X (the 2021 wavefront rewrite).

### D-124: Full path reuse is the spine — the M3 reservoir generalizes to a whole light path
Status: accepted (2026-07-24). The milestone extends the DI reservoir from a length-2 direct
connection to a path of any length. A **unified DI+GI reservoir** (Enhanced §6.1) competes a
direct (length-2 NEE) candidate and indirect candidates in *one* reservoir and stores whichever
wins, **retiring the separate DI reservoir**. *Why:* a single lightweight reservoir over the
full path space is the occupancy-friendly shape (Enhanced: 431→265 MB at 1080p, the
second-largest single speedup after Russian roulette). The DI reservoir was deliberately built
(D-086) as the dormant base case of exactly this generalization; unifying finishes that design
rather than bolting a second system beside it. c_cap exposure + ReSTCV (D-127) fold in on top.

### D-125: Hybrid shift is the deliverable; reconnection shift is the validated scaffold
Status: accepted (2026-07-24). Reconnection shift first (cheap, correct on diffuse/rough,
**same-pixel Jacobian = 1** — the validation invariant), then the **hybrid shift**
(random-replay the early specular segments, reconnect at the first sufficiently-rough+distant
vertex pair). Classic roughness/distance thresholds first; **footprint-based criteria**
(Enhanced §4) replace them at their own rung. *Why:* reconnection alone is silently wrong on
the glossy/specular paths lookdev lives on; hybrid is the paper's default and what makes reuse
production-usable. Reconnection-first is a validation ladder — the Jacobian-1 invariant is
checkable before any non-trivial Jacobian exists. Random replay is *free*: the stateless RNG
(rng.slang) was built so any decision replays from its keys.

### D-126: Both spatial and temporal reuse — spatial validated first
Status: accepted (2026-07-24). Both axes ship. Implementation order is **spatial-first** (a
same-pixel-topology shift has Jacobian 1, so spatial reuse is provable before temporal
reprojection adds a variable), then temporal through M3's existing reproject + **decay-ramp**
machinery. The reservoir is **temporal- and CV-ready from day one** so both are additions, not
re-layouts. *Why:* temporal is critical to the interactive feel and is in the final path; but
building spatial first validates the shift and pairwise-MIS machinery against a Jacobian-1
baseline before reprojection can hide a bug. Spatial and temporal share the same shift +
defensive pairwise MIS (D-087), so spatial-first is a validation ladder, not architectural debt.

### D-127: ReSTCV is the unbiased convergence tail — first-class, in-milestone
Status: accepted (2026-07-24). ReSTCV (Spatio-Temporal Control Variates, SIGGRAPH 2026) is
architected as the **general control-variate form with plain ReSTIR PT as the zero-CV
degenerate** — one code path, not two. It stores a per-reservoir accumulated-colour control
variate and is *unbiased* (verified by accumulating static frames). *Why:* the user's "ReSTIR
early, better long-term after" is exactly ReSTCV — the published, unbiased realization of the
convergence-plateau fix. Making plain ReSTIR PT the zero-CV degenerate keeps the codebase
lightweight (no estimator fork) and gives a free A/B (CV on vs off → same converged image). It
is the principled answer to the plateau that Enhanced's duplication maps address only with bias.
Its exact per-reservoir storage/update math is the one rung not yet read to the equation level:
a focused deep-read of github.com/Hercier/ReSTCV gates the step that builds it (m6-plan.md §6).

### D-128: One unified ~64 B (cenote: 96 B) path reservoir, designed once
Status: accepted (2026-07-24). The reservoir struct — the milestone's central artifact — is
designed once (Enhanced Alg 1) holding every field the later rungs light up: `W`; `float3 F`
(RGB integrand — target is scalar luminance, so F carries colour for the §6.3 fix and exact
resolve); two seeds for random replay; `M`; the reconnection vertex as the existing **`Hit`**
(instance+prim+bary) + oct `wi` + radiance; cached Jacobian terms pre-multiplied to one float
with the **NEE light pdf kept unpacked** (it feeds path MIS); plus the **ReSTCV slot placed
last** so the step-6 deep-read pins it without moving any field before it. *Why:* getting the
layout right once is what prevents a re-layout mid-milestone. cenote's record is **96 B, not
Enhanced's 64 B**, by three deliberate choices — `Hit`'s float2 barycentrics (one shared vertex
form) over the paper's 2×16-bit unorm; the float `confidence` its reservoir primitive already
carries (fractional temporal decay) over the paper's 8-bit M; and the reserved CV lane the
paper lacks. Packing back toward 64 B (2×16u barycentrics, 8-bit M) is the deferred size
optimization if the per-view × N-viewport budget bites (deferrals.md). Implemented in T0:
`shaders/reservoir_path.slang`, the `src/restir.rs` mirror, and a GPU round-trip test pinning
the three std430 `float3`-lane offsets, not just the size.

### D-129: Validation & flagship demo — three artifacts, cenote's own brute force the oracle
Status: accepted (2026-07-24). **(1)** an unbiasedness proof — ReSTIR-PT-on vs brute-force
accumulation converge to the same image on a **GI-heavy** scene (the M3 gate, now on indirect
paths); **(2)** an **equal-time** win, PT vs ReSTIR PT, matched seconds; **(3)** a
**convergence-under-reuse study** — mean-error-vs-sample-count curves showing the layered tail
(PT → decay-ramp → ReSTCV). *Why:* the oracle must be cenote's own estimator — the thesis is
that ReSTIR PT and brute force are the same estimator, so anything else measures the wrong
thing (D-090). Falcor is the behaviour spot-check, never the numerical oracle.

### D-130: One integrator, no plugin seam
Status: accepted (2026-07-24); amendment from the research review. **No pluggable-integrator
abstraction.** cenote keeps ONE config-driven light-transport path; plain PT is the **zero-CV,
no-reuse degenerate** of the ReSTIR-PT+ReSTCV path (D-127), reachable by config, not by a
second integrator. *Why:* MoonRay is one monolithic `PathIntegrator` (no base class, no
virtuals, no registry); Cycles is one data-driven stage machine. Both production tracers
converge on one integrator. A swappable-estimator layer would be speculative generality (the
named risk, D-086); the zero-CV degenerate already gives the only fork that matters, for free.

### D-131: The shift re-shades from the `Hit`
Status: accepted (2026-07-24); amendment from the research review. The shift map **re-resolves
the `Closure` at the reconnection `Hit`** (a re-shade), storing only the re-evaluable key,
never a serialized BSDF. `Closure` is already a pure function of `Hit`+`Material` (openpbr.slang)
— POD, value-typed, GPU-resident. Budget one re-shade per reconnection per neighbour as the
dominant per-shift cost, and **guard its bitwise determinism** under stochastic texture
filtering (a re-shade-equality test from step 2). *Why:* MoonRay (persists `Intersection`, not
`Bsdf`) and Cycles (rebuilds `ShaderData`, re-runs the shader graph) both say closures are
transient — store the shader input, re-shade on demand. cenote's POD closure is the exact shape
MoonRay recommends over its own arena/pointer/vtable graph: a structural advantage to exploit.

### D-132: Duplication maps deferred; footprint & reciprocal reuse kept
Status: accepted (2026-07-24); amendment from the research review. **Duplication maps leave the
milestone** → preview-only, biased, deferrals.md. **Footprint-based reconnection criteria**
stay (Enhanced §4) — unbiased, isolated, computed from the **reciprocal area-density of
PDFs/geometry cenote already has** (*not* ray differentials/cones), removing per-scene
threshold tuning. **Reciprocal (paired) spatial reuse** stays (Enhanced §3), chosen **over**
the deferred stochastic pairwise MIS — it nets ~1.63× *and lowers* FLIP. Splatting and CRIS are
**not** built (gather suffices; CRIS is subsumed by the GRIS formulation). *Why:* duplication
maps are Enhanced's only biased contribution and plateau *above* the reference under
accumulation (Fig 15; §7.4 says disable for offline) — dead weight for an accumulation-first
renderer whose denoiser is a cadence-throttled view of the *accumulated* film. Footprint and
reciprocal are separable, unbiased drop-ins ("generalizes to other reuse algs").

### D-133: Colour-noise fix, the RR+random-replay trap, the AOV-under-reuse seam
Status: accepted (2026-07-24); amendment from the research review. Fold in the near-free
**colour-noise fix** (Enhanced §6.3 — scalar-luminance target vs RGB `F`, fixed by accumulating
vector-valued resampling weights for shading; F is a stored field for it, D-128). Handle the
**Russian-roulette + random-replay trap** (remove RR from replay, fold survival into the
initial-sample PSS PDF — supp §6) *with* the hybrid-shift step, not after. Decide the
**AOV-under-reuse seam now**: albedo/normal guides resolve from the **canonical path
pre-resampling** (cenote already does this via guide throughput, pathstate.slang:89-106);
LPE/light-group AOVs stay deferred but the seam is acknowledged, not retrofitted. *Why:* these
are the traps and cheap wins the equation-level read surfaced. The AOV seam is MoonRay's
sharpest warning — radiance decomposition observes every scattering event and reuse complicates
per-event attribution, so "which pass does a *reused* path land in?" is answered by construction.

## 2026-07-25 — M6 step 2: the reconnection shift, first true path reuse

### D-134: The unified reservoir owns the whole primary-hit path integral
Status: accepted (2026-07-25). The reservoir now estimates the *entire* reflected radiance at
the primary hit — direct **and** indirect — not just the direct term (D-124 finished). The
internalized BSDF draw (D-088) yields up to **two candidates with disjoint path supports**: the
**light** its scattered ray reaches (an emitter's emission or the environment — the length-2
sample, competing with the M light candidates), and the **continuation** past any surface hit —
the candidate stage traces the indirect tail inline (`traceTail` in restir_candidates.slang, the
same per-vertex estimator `shade_surface` runs via the shared `nee.slang`, accumulated into
`Lo(x₂→x₁)` instead of the film) and stores a length-≥3 **reconnection sample**: `rcVertex` = x₂,
`rcVertexRadiance` = the *reflected-only* Lo(x₂), MIS weight 1 (NEE cannot reach a length-≥3
path, so the balance-heuristic denominator keeps only the BSDF density). The two-candidate split
is load-bearing: **emitters reflect too** — an emissive-and-reflective surface's emission is the
light candidate and its reflection the tail, so neither is lost nor double-counted (guarded by
`restir_carries_reflection_off_emissive_surfaces`). `shade_surface`'s bounce-1+ continuation is
turned **off** in ReSTIR mode — the reservoir+resolve produce all indirect light (the ReSTIR PT
integrator, D-130). Resolve and the shift **re-form** the RGB integrand in *their* domain from
the cached Lo (no stored per-domain `F`, which would go stale under reuse). The two path kinds
are discriminated by `PathSample.reserved`'s low bit (zero = NEE, so a zero-init sample reads as
the base case). The tail seeds its interior medium from the draw's lobe: a closed-surface
refraction at x₁ starts the tail *inside* x₁'s own interior — Beer–Lambert over x₁→x₂, the far
wall's closure at the inverted IOR, and every later transmission toggling from inside — exactly
the state `shade_surface`'s continuation carried in the path state (guarded by
`restir_tracks_interior_media_through_the_indirect_tail`). *Why:* one reservoir over the whole
path space is Enhanced §6.1 and the D-124
spine; a surface hit is exactly where the tail begins, so the candidate that used to die there
now continues. Checkpoint green: initial RIS (own-pixel, identity shift) converges to the
brute-force path tracer, and the two ReSTIR goldens' means moved <0.2% (unbiased) with slightly
*lower* variance — regenerated, the FLIP pin now tracks the new estimator.

### D-135: The reconnection shift is a re-target plus a geometry-term Jacobian, folded into the shipped pairwise MIS
Status: accepted (2026-07-25). Reusing a reconnection sample across pixels re-forms its target at
the new surface — connect that surface to the stored x₂ at radiance Lo(x₂), shade f(x₁′;ω)·|cosθ|·Lo
(`connectSampleAt` handles both kinds; the reconnection segment's visibility is an *identity*
shadow test to x₂'s own triangle, exactly as a light point's) — and multiplies by the
**reconnection Jacobian** J = [|cosθ₂|/d²]_target / [|cosθ₂|/d²]_source (Lin 2022), the solid-angle
measure change of the moved connection. **Exactly 1 when the domains coincide** (the same-pixel
invariant, D-125 — by construction: identical inputs, ratio 1). It integrates into cenote's
*existing* defensive pairwise MIS (restir_mis.slang, D-087) with **no reconnection-aware variant**:
the two cross-domain targets `pcYi` (neighbour shifted to canonical) and `piYc` (canonical shifted
to neighbour) are each scaled by their shift's Jacobian before the ratios, which is precisely where
the measure change belongs; an NEE sample stores a domain-independent light point (identity shift,
J = 1), so the direct path is untouched. *Why:* the Jacobian needs *both* surfaces, which only the
spatial stage has — so it lives there, not in the per-surface target. Checkpoint green: spatial path
reuse converges to the path tracer on an all-matte GI scene (reconnection reuse dominant), and on
the mixed glossy gate.

### D-136: Step 2 bakes Lo direction-independent; undefined shifts carry Jacobian 0, never a dropped pair
Status: accepted (2026-07-25). `rcVertexRadiance` caches Lo(x₂) as a **direction-independent** exit
radiance (f₂ baked in). This is exact for a *diffuse* x₂ and exact for **any** x₂ in its *own*
pixel (the connection direction does not move), but reusing a *glossy* x₂ across pixels would bias
the mean (f₂ is directional). So a classic **reconnection eligibility guard** (α_min ≈ 0.2,
`reconnectionEligible` — mostly-diffuse material, weak specular/metal/glass) marks where the
baked-radiance shift is *undefined*, folded into one shared `reconnectionShiftJacobian`
(restir_target.slang): 1 for NEE (identity), the geometry ratio where defined, **0 where not** —
the partial-shift convention. The zero Jacobian nulls the sample's cross-domain target while
**the pair still counts** in V/c_tot and the canonical's share; skipping the pair instead would
make the pairwise-MIS weights a function of the neighbour's *realized* sample and overweight
every eligible point by the neighbour's ineligible probability — a small brightening bias on
mixed scenes the fixed-function form provably avoids (weights sum to 1 pointwise). A glossy
reconnection vertex still shades exactly in its own pixel (resolve re-forms it there); it just
carries nothing across domains. **The same convention guards the frame boundary**: a
reconnection sample's *temporal* shift is undefined until step 5 — it owes the Jacobian across
the reprojected surface, a stable identity for `rcVertex.instance` (a raw TLAS index, which a
scene edit renumbers — the hazard the light-id registry solves for NEE), and a visibility
re-test — so restir_temporal zeroes its cross-frame targets the same pair-preserving way, and a
stale instance is never dereferenced. Direct light still compounds temporally; path history
joins it at step 5. Un-baking f₂ (storing the incident-side radiance + `rcVertexWi`,
re-evaluating f₂ at the shifted connection) is exactly the **hybrid shift**, step 3 (D-125) —
the field is reserved, zero until then. This is a **no-op on the diffuse checkpoint** (every
vertex eligible). Two further scope lines, both deferred (deferrals.md): the inline tail treats
crossings as **opaque** (stochastic opacity in the reused tail is out, with volumes — the
checkpoint scenes carry none), and a **distance criterion** on the reconnection (the footprint
criteria, step 4) is not yet applied, so a very short reconnection can still spike the 1/d²
Jacobian — watched, not yet a firefly source at checkpoint spp.

### D-137: Step 3a streams candidates per path and un-bakes f₂; eligibility becomes a variance guard
Status: accepted (2026-07-25). The T2 aggregate retires: `traceTail`'s one baked Lo(x₂) candidate
becomes a **per-path streamed walk** (`walkTail`, the Lin 2022 / Enhanced Alg 1 shape) — every
lighting event along the BSDF draw's continuation (the NEE connection at each vertex, each
emission hit, the sky) enters the reservoir as its **own candidate** with its own suffix, MIS
fold, and acceptance draw, and the stored winner is one concrete path. The sample now holds the
**incident-side** suffix (`rcVertexRadiance`, with f₂·|cosθ₂| deliberately excluded) plus the
live `rcVertexWi`, so every domain re-forms Lo by **re-shading f₂ at the stored Hit**
(`reconnectionOutgoing`, one shared formula for generation, both reuse targets, and resolve —
D-131's re-shade made structural). The one MIS weight that involves x₂'s closure re-evaluates
per domain against the **unpacked `neeLightPdf`**, discriminated by a packed terminal kind
(NEE at x₂ / x₂'s scatter found the light / deeper-or-delta = baked); an inside-the-instance
bit reproduces the interior-IOR closure the walk resolved. With f₂ re-shaded, reusing any
vertex is **unbiased** — so the baked-radiance eligibility (D-136's material guard) is deleted
and replaced by the classic **pair criteria as a variance guard**: both endpoints past the 0.2
roughness floor (`reconnectionRough` — metals and rough glass now pass, vertices T2 refused
outright), the segment past a scale-free floor (`RECONNECT_MIN_DISTANCE` × camera depth, dies
at step 4's footprint criteria), and no interior medium on the reconnection segment (its
Beer–Lambert is baked from the source segment; media reuse is deferred with volumes). The
verdict is evaluated **once, at generation**, and stamped into the sample
(`reconnectionShiftable`), so the shift needs no material lookups and the MIS weights stay a
fixed function of each sample — a failing pair shifts with Jacobian 0 (D-136's partial-shift
convention) until T3b/T3c's replay kinds cover it. The walk keeps the path tracer's exact
dynamics: `rcFactor` folds the source-domain f₂·cosθ₂ back in for every termination and
roulette decision, so support and survival match `shade_surface` bounce for bounce while the
stored suffix stays f₂-free. ωₖ packs to 16-bit octahedral; the walk evaluates its own targets
through the **decoded** direction, so generation weighs exactly the sample every re-shading
domain sees (the ~1e-4 rad quantization is the industry-standard concession, negligible at the
≥0.2-roughness lobes the criteria admit). Reconnection pinned at k = 2 this rung; no replay, no
seeds — T3b moves the vertex, T3c adds the pure-replay kind (m6-plan §4a).

### D-138: Step 3b moves the reconnection vertex; the prefix replays from a per-path seed and roulette folds into the weight
Status: accepted (2026-07-25). The reconnection vertex k is no longer pinned at x₂: the candidate
walk carries a **reconnection context** — pinned at x₂ on entry, exactly D-137's shape — and moves
it once, to the **first pair along the walk that passes the variance criteria** (both endpoints
rough, the segment past the scale-free floor, exterior). From the lock on, events stream as k > 2
samples: the bounces before x_{k−1} become a **replayable prefix**, and the shift re-traces them at
the destination before reconnecting (`shiftReconnection`, restir_target.slang). Three mechanisms
land together, none separable. (1) **The seed re-key**: every walk sampling draw — the x₁ scatter
included, since it is the first replayed bounce — moves off the pixel-ranked stream onto a pure
function of (per-path seed, dimension) (`pathReplaySeed`/`replay_*`, rng.slang); the seed rides the
sample (`initRandomSeed` lights up), so generation and any domain's replay are *the same draws by
construction*. Reservoir acceptances stay ranked (pixel-local, never replayed). The price is
per-sample stratification on the walk — hash-independent draws, the reference-implementation trade;
resampling, not tail stratification, carries convergence. (2) **The roulette fold** (§6's RR-replay
trap, closed): generation's walk still rolls exactly as the path tracer (survival read off the true
running throughput), but a roll's division lands by where it sits relative to the context — into
the baked suffix always (a value, never replayed), and into a **pending survival** that a future
lock folds into its candidate weight's denominator (w = p̂ / (reconPdf · Πs)). The replayed prefix
therefore carries no 1/s, and **the replay walker never rolls**: a survived base path can never
become a killed shifted path. The prefix accumulator is kept roulette-free and
transmittance-inclusive, in replay's exact multiplication order. (3) **The cached Jacobian half**:
a k > 2 sample's x_{k−1} is a replayed point no reusing domain can recompute, so the source half
|cosθₖ|/d² is stamped at generation (`cachedJacobian` lights up) and the shift divides its own half
by it; the replayed prefix is an identity map in primary-sample space (uniforms copied), so only
the reconnection segment carries measure change. `F` lights up too: generation stores the
own-domain integrand, a winning spatial shift rewrites it with the destination's, and it is trusted
wherever re-forming would re-trace the replay — own-domain reuse targets (spatial's p̂_c(Y_c)/p̂_i(Y_i),
temporal's canonical), and **resolve at k > 2** (k = 2 keeps the D-134 own-pixel re-form; the
resolve asymmetry m6-plan §4a names). The walk itself now runs entirely on its *drawn* directions —
the chain replay reproduces — while the context re-derives its ends direction-free through one
shared seam (`reconnectionEnd`: re-derived segment, guards, f·|cos|, density, Jacobian half,
re-shaded x_k), used identically by generation's lock and the shift — the bit-identity discipline
made structural, and pinned by the new gate: **same-pixel replay reproduces the stored F bit for
bit** and Jacobian 1 to the ulp (the squared-distance's last bit belongs to per-kernel fma
contraction — the one honesty note on "exactly 1"). Scope holds the interviewed line: a path whose
terminal fires before its first qualifying pair — including an emitter that would itself be the
reconnection vertex, and every NEE-terminated no-pair path — stays J = 0 (own-pixel exact) until
T3c's pure-replay kind re-draws terminals from the stored dims. Temporal still holds reconnection
samples at J = 0 across frames until step 5. Storage: k packs into `reserved` bits 5..12 — no
re-layout, D-128's headroom.

### D-139: Step 3c adds the pure-replay kind; the walk unifies and every streamed event shifts
Status: accepted (2026-07-25). A path with no qualifying reconnection pair anywhere now stores a
**pure-replay sample**: its per-path seed, its shape — the terminal kind (an NEE connection, or a
scatter that found a light) and the walk bounce it fired at — and its own-domain integrand F.
Nothing else: the reconnection fields stay zero. The shift (`shiftReplay`, restir_target.slang)
replays every scatter from the seed at the destination's surfaces and re-forms the terminal in its
stored shape — an NEE terminal re-draws the connection from the *same dims* (light choice, point,
MIS, delta-ness all re-form locally; one shadow ray handed to the caller), a SCATTER terminal's
last segment finds its own light concretely (an emitter's front face, or the sky — no ray owed).
The whole map is a primary-sample-space identity — every uniform copied — so its Jacobian is
exactly 1; the source's roulette survivals live in its candidate weight (the D-138 fold), so the
replay never rolls and the path exists exactly as far as the stored shape says. A structural
mismatch (a dead lobe, a mid-chain escape, a non-emitter where the emitter stood) is the D-136
partial-shift convention: target zero, the sample still counts. **The generation walk unifies**:
the T3b x₂ preamble folds into the loop as its first iteration, the pinned-unshiftable context is
deleted — the context now starts unlocked and locks at the first qualifying pair, (x₁, x₂)
included — and every pre-lock event streams replay-kind with weight p̂/Πs (F carries the
roulette-free chain with every pdf divided in; only the survivals stay outside). The T3b
entry-invalid whole-walk drop disappears with the pinned context: no event now depends on a
re-derived x₂ segment existing. The `reserved` word re-lays out once: a 2-bit kind
(NEE/RECONNECTION/REPLAY — a replay sample must not be readable as either other kind), terminal at
bits 2..3, inside/shiftable at 4/5, and k *or* the replay terminal bounce at bits 6..13. The
replay walker is one shared seam (`ReplayWalker`/`replayStep`): the reconnection shift's prefix
and the pure-replay shift advance through the same code the generation walk mirrors
multiplication-for-multiplication — pinned by the same-pixel gate, now covering both kinds
(replay F bit-for-bit, Jacobian exactly the constructed 1). Reuse stages route by kind: spatial
shifts replay neighbours through `shiftReplay` at J = 1 (own-domain targets stay the stored
luminance(F)); resolve trusts F for every replay-carried sample (k > 2 and replay-kind alike);
temporal holds ALL indirect kinds at J = 0 until step 5. `rcVertexRandomSeed` stays reserved: the
per-path seed covers the terminal redraw, because dimensions are per-bounce. Gates: the sharp-chain
scene (every surface below the roughness floor, the emitter metal-based so no pair can ever lock —
the coverage assert proves zero locks) through the unbiasedness gate; and the convergence harness
extended with the **reuse-is-alive** gate — ReSTIR PT must *beat* plain PT at equal spp on an
indirect-only glossy GI scene (a one-sided emitter facing the sub-floor panel, black environment:
the good path is rare, the regime reuse exists for; measured ≈ 2× at 8 spp, 1.7× at 32 spp,
asserted at a 1.3× floor). The direct-lit variant measurably does NOT clear that bar — PT's summed
per-vertex estimator is already low-variance there and the one-survivor resampling costs more than
5-neighbour reuse recovers; the honest scope note, recorded so nobody re-litigates it from the
convergence curves alone.

### D-140: Step 4 prices reconnection by footprint area, not roughness — the classic thresholds die
Status: accepted (2026-07-26). The pair criteria (a variance guard, not a bias guard — D-131/D-137)
are now the **dual ray footprint test** of ReSTIR PT Enhanced §4 (Eq 5): a pair (x_{k−1}, x_k)
reconnects when the area each endpoint's lobe spreads over at the other end stays above a fixed
fraction of the primary ray's own spread — `1/(p·G) ≥ rhs` in both directions, with
`rhs = (c/100)·‖x₀−x₁‖²·4π/|⟨n₁,x̂₁x₀⟩|` computed once per pixel (geometric normal, the G-term
convention) and `c = FOOTPRINT_C = 0.02`, the paper's scene-independent fraction. The classic
per-scene knobs — `RECONNECT_MIN_DISTANCE` and the roughness floor at x_k — are deleted outright,
no toggle (the step-3 tree at `cfd7fb8` is the A/B seam); `reconnectionRough` survives
**single-sided**, at x_{k−1} only (§4.2 — its job is the re-shaded-closure firefly guard where the
lobe re-evaluates, and x_k's sharpness is now priced by the inverse term instead of banned).
Placement is **lock-then-demote**: the forward test joins the pair search on the two factors
`reconnectionEnd` already computes; the inverse test needs p_k(ω_k), which exists only after the
scatter out of x_k draws, so it runs there — only when x_k is sharp-capable (the shared
`sharpCapable` weight test; a diffuse lobe cannot fail it) — and on failure demotes
`ctx.shiftable`, re-arming the search at later pairs. The already-streamed at-lock NEE event keeps
its verdict exactly (footnote 6: a light draw's pdf ignores the incident direction, so its suffix
has no inverse lobe to test); deeper events ride the drawn ω_k. Both tests run multiply-form
(`rhs·(p·G_factor) ≤ 1`), IEEE-clean at grazing: cos → 0 pushes rhs to ∞ and the pair simply never
locks, the paper's own behavior. Evidence (§4b measurements, classic baseline at `cfd7fb8`): the
dedicated distant-glossy scene — a roughness-0.12 metal panel 16 m out, lit only through a
light-sandwich card so the floor sees nothing but the sharp reflection — flips from 0 reconnection
/ 155 replay under classic to **154 reconnection at k = 2 / 1 replay**, the acceptance classic's
0.2 floor structurally refused, pinned forever by a coverage assert; mirror-chain locks earlier
(597 → 881) with the k ≥ 5 corridor intact and the demote branch live; sharp-chain is bit-identical
(guard-carried); convergence is equal; the ReSTIR goldens came out bit-identical (the criterion
consumes no random dimensions and agrees with classic on every golden-scene pair); frame time is
cost-neutral where the mix doesn't move and ~18% faster on distant-glossy (replay-kind reuse became
reconnection, so spatial got cheaper). One scale note, from Eq 5's own arithmetic: at these camera
depths a roughness-0.05 lobe prices its acceptance at ~80 m of segment — the criterion is *strict*
on sharp lobes, which is the point; the evidence scene's 0.12/16 m is the same physics at a
testable scale.

### D-141: Step 5 carries paths across the frame boundary — the shared shift re-roots history, and an edit gates it
Status: accepted (2026-07-26). Temporal reuse now reuses *whole path samples* across frames. The
spatial per-sample shift-and-target block is **factored, not copied** (§4c decision 1): a forward
form (`shiftIntoDomain` — kind dispatch → shift → the one visibility ray → shifted target,
own-domain target, re-rooted sample) and a ray-free reverse (`targetInDomain`) live beside the
shifts in restir_target.slang; spatial was rewritten onto them first (T5a `66f86a1`, both golden
sets bit-identical — the refactor proven a no-op), then temporal calls the same two with its one
reprojected neighbour (T5b `7245fcb`), retiring the step-2 J = 0 convention (D-136). On a prev win
the **re-rooted** sample is stored — F re-formed at this pixel through the current TLAS, prefix
replayed, reconnection visibility re-traced, Jacobian priced — precisely what generation here
would have produced, so spatial's canonical trust (D-138/D-139) holds across the boundary; the
NEE arms stay per-stage and unshadowed (D-094's feed convention, not drift). `rcVertex.instance`
across edits is answered by an **epoch gate, not an instance registry** (decision 2):
`Scene::epoch` counts applied edits (camera moves never bump it), `ViewState` records at each
swap which build its `prev` rendered against, and one push-constant flag drops indirect history
before its raw TLAS index is ever dereferenced — NEE history keeps surviving edits through the
light-id registry (M3); the registry itself went to deferrals.md (revival: editing-heavy
interactive workflows). Cost discipline stayed **measurement-only** (decision 3) with one free
ordering fix: the capped-and-decayed confidence folds into `prevValid` before any shift work, so
a decayed-to-zero prev is no neighbour at all and the converged still provably spends zero shift
rays. Evidence (§4c's T5 measurements): pinned-temporal unbiasedness (decay 0, 256 frames) green
on the three kind-covering scenes; the temporal G-buffer surface bit-identical to the path-pool
reconstruction; the decay-handoff curve shows the honest shape — on a cold held camera the
accumulated average carries a correlation *cost* (~15% relMSE through 4 frames, annealing to
+1.4% at 32 as the 16-frame ramp hands off), not the interview's sketched early win, because the
warm-start's payoff is per-frame quality during motion, which an accumulated average structurally
cannot see; temporal live costs 0.6–1.3 ms/frame at 512² (demo 6.78 → 7.36, glossy-primary
8.66 → 9.66, distant-glossy 3.01 → 4.29) and the decayed still none. One gate moved: the
many-lights convergence floor 1.5× → 1.3× (the step-3c gate's own floor) — cross-frame indirect
correlation narrowed the 8-spp margin (relMSE 0.0331 → 0.0362, mean unmoved, brute-vs-reference
invariant), the D-094 trade the ramp exists to anneal.

### D-142: The deep-read corrects D-127 — ReSTCV is a colour-noise fix, and the still tail is already ~1/N
Status: accepted (2026-07-26); corrects D-127's *rationale* (append-only — the architecture
stands, the claim moves). The step-6 deep-read (paper + supplemental + the full Hercier/ReSTCV
reference source; m6-plan.md §4d) found ReSTCV is **not** the convergence-plateau fix D-127
credited: its stated problem is *colour noise* — a scalar-luminance target resamples intensity
and leaves chroma to luck — and its static-accumulation curves (paper Fig 8) are two parallel
~1/N lines, an unbiased **constant-factor** variance win from frame 1, not a tail-slope change;
the paper never claims a plateau fix and concedes covariance-blind combination weights as future
work. D-127's architecture survives *better* than its rationale: Enhanced §6.3 and ReSTCV are
the same fix at two depths (the reference ships §6.3-style decoupled shading as its zero-CV
mode), so survivor-only → vector-weight shading → full ReSTCV is a genuine ladder of special
cases — exactly the "general form with degenerates" D-127 asked for. Step 6's checkpoint
recalibrates accordingly (§4d decision 1): CV-on ≡ CV-off unbiasedness, a measured
constant-factor accumulated-relMSE win, and the chroma improvement — not "the tail flatten."
The floor question D-127 assumed answered got a number instead (the 6-0 baseline,
`restir_floor_and_chroma_report`, cross-referenced so shared deterministic frames cannot
deflate the tail): the converged-still configuration (spatial-only, fresh RNG, D-085) holds
N·relMSE flat at ~1.02–1.15 over 16–128 frames on indirect-glossy — **~1/N, no floor** — while
brute force's ×N drifts mildly *below* one (1.25 → 0.94), because per-pixel accumulation walks
the Owen-scrambled Sobol sequence in order (a QMC estimator) and resampling forfeits that
low-discrepancy structure; the equal-frame ratio therefore drifts brute-ward (0.65 → 1.23 by
128 frames) with no floor anywhere. The chroma before-side (many-lights, per-channel relMSE):
ReSTIR R 0.00626 / G 0.00618 / B 0.00726 at 32 spp vs brute R 0.01039 / G 0.01063 / B 0.01235 —
the +16% blue excess is the *scene's*, carried identically by both estimators, so 6a's win must
show as a level drop or an eyeball/golden difference, not a spread collapse. *Why correct rather
than silently proceed:* implementing against a miscalibrated claim reproduces the
interview-sketch error T5c had to correct after the fact; the honest-numbers precedent applies
to the milestone's own headline.

### D-143: 6a ships §6.3 vector-weight shading default-on, and the D-130 degenerate becomes a bit-exact pin
Status: accepted (2026-07-26). Rung 6a (m6-plan §4d decisions 3–4) landed as interviewed: the
spatial pass accumulates every merge candidate's w⃗ = w_scalar·F/luminance(F) into the
`cvAccumulator` lane of its pass-local output, and resolve shades the lane under a default-on
`cv_shading` toggle — survivor-only shading is the reachable degenerate. Two things the
implementation settled beyond the interview. **(1) The degenerate is pinned bit-exactly, not
perceptually**: the pre-6a ReSTIR goldens are kept as survivor-toggle goldens and the pin test
asserts float-for-float equality, unlike every other golden's FLIP threshold. The toggle's
claim is *identity* with the prior estimator — a claim bitwise determinism makes checkable and
a perceptual threshold would let a subtly different estimator pass; the cost (driver/compiler
FP sensitivity the FLIP goldens deliberately shed) is accepted because goldens only run on
GPU-capable local machines and regenerate together. Each later rung re-runs this pin as its
no-op proof (§4d decision 4). **(2) A luminance-invariance identity fell out of the algebra**:
the survivor's shaded luminance is luminance(F)·W = weightSum, and the lane's luminance is
Σ w_scalar·luminance(F/luminance(F)) = weightSum — identical. §6.3 therefore moves *only*
chroma, which the 6a numbers confirm: the many-lights per-channel relMSE drops are small and
blue-leaning (32 spp B 0.00726 → 0.00714) while the demo golden's copper bounce speckle
visibly desaturates (mean FLIP 0.020 against the old pin, carried by that speckle). The
constant-factor luminance-variance win D-142 recalibrated the headline toward is 6b's to
deliver — the control variate proper — not this rung's; 6a's deliverable is the chroma fix,
and the invariance is why measuring it needed the per-channel lens in the first place.

### D-144: 6b-i lands the spatial control variate on G-semantics, repaying visibility where the rays are
Status: accepted (2026-07-26). Rung 6b-i (m6-plan §4d decision 5) ships ReSTCV's spatial update
as interviewed — candidate-mean init (store now takes the lane as a required argument; the
round-trip fixture pins it by value), temporal pass-through, per-neighbour α⟨F_j⟩ + ⟨F_i − αF_j⟩
with cenote's pairwise ratios, α = the per-channel `aovAlbedo` ratio (non-finite → 1, clamp ≤ 2),
c fixed at 1.6, resolve shading the signed lane — with one divergence the interview never had to
face and the implementation had to settle. **The reference's integrands all embed traced
visibility** (its shifts always trace); **cenote's one-ray-per-candidate policy prices NEE
candidates, own-domain targets, and reverse targets unshadowed** (D-093/D-094), so the reference
formulas transplanted verbatim would be biased by every unshadowed evaluation paired against a
shadowed one. The resolution is that a control variate never needed the true integral on both
sides — only *matched* semantics: the lane transports **G, the pre-visibility integral the
candidate stream itself estimates** (NEE terms unshadowed, indirect terms concrete — their walks
traced every ray), the residual bracket α⊙(lane_j − estG_j) pairs G against G and is zero-mean
by construction, and the visibility-aware F-estimates use only evaluations whose rays the
spatial pass actually spends. Two consequences, both named in restir_spatial's header: an
indirect canonical's backshift — the one evaluation cenote never prices — gets pair-partition
weight zero on indirect points (the neighbour's own sample covers them at weight 1, the same
Kettunen coverage move the dead-shift fallbacks make; partition of unity holds pointwise, so
unbiasedness survives and only the residual's pairing weakens on indirect samples), and the
centre estimate repays the canonical lane's deficit at the one sample whose shadow ray the pass
traces: centre = lane_c + W_c·chroma_c·(p̂ᵛⁱˢ − p̂ᵘⁿˢʰ), whose correction has mean exactly
F_c − G_c and is identically zero for an indirect canonical. The numbers behave accordingly:
the chroma scene improves where the constant factor lives (many-lights 8 spp per-channel relMSE
R 0.03372 → 0.03219, G 0.03374 → 0.03120, B 0.04059 → 0.03796; scalar 0.03602 → 0.03378) and
the indirect-glossy scene pays the pairing cost (~3% at 8 spp, 0.08893 → 0.09164) — recorded,
not hidden, per the honest-numbers precedent. Signed transients are real and unclamped (demo
per-frame minima reach −0.96 in one channel; means agree with the survivor estimator to ~0.14%
at 32 spp): accumulate's finite-guard, the auto-stop floor, and tonemap's saturate were each
checked to carry them. The survivor pin (D-143) re-ran bit-exact — store's signature changed
but toggle-off's arithmetic did not. Named seam for 6b-ii: prev's persisted lane is today the
candidate mean (spatial's scratch never survives a frame), so the temporal recurrence must
either blend against that or persist the combined CV — and the unshadowed fraction of whatever
it blends must ride with it (cvNormalization is the natural carrier if one is needed).

### D-145: 6b-ii closes the ladder — the CV lane crosses the frame boundary on the decayed confidence, and G-semantics makes the carrier free
Status: accepted (2026-07-26). Rung 6b-ii (m6-plan §4d decision 6) lands the temporal
recurrence as interviewed: one M-weighted blend at restir_temporal's store site,
lane_out = (c_c·candMean + c_prev·lane_prev)/(c_c + c_prev), with c_prev the same
min(M_prev, 20·c_c)·decay confidence the pair's MIS and merge already fold — one signal, one
ramp, no new constant — temporal α ≡ 1 (the disocclusion gate already vouches the reprojected
surface is this surface), and every reset path (reprojection failure, the gate, decayed-to-zero
history, the epoch gate) falling back to this frame's candidate mean, which spatial re-enriches
the same frame. The implementation settled D-144's named seam in the cheap direction: **prev's
persisted lane is the candidate mean — a G-estimate — so blending G against G needs no
unshadowed-fraction carrier at all**; the recurrence stays a G-estimate by induction (the blend
weights are deterministic — confidence compounds by addition whatever the acceptance coins do,
and the cap and decay read only frame indices — so E[lane] = G survives any depth of history),
the spatial residual bracket keeps its zero mean unconditionally, and `cvNormalization` closes
the step **unclaimed and reserved-zero**. Two details beyond the interview: the epoch gate binds
the *lane* even when an NEE survivor rides on through resampling — the lane aggregates the whole
pixel's estimate, indirect terms priced against the old build included — and the decay-noop
endpoint now pins the lane too (at decay 0 the candidate mean passes through bit-for-bit, so the
converged still keeps independent frames on the shading estimate as well as the resampling,
D-085). The proof is the new pinned-live gate: the full pipeline — temporal live at decay 0,
spatial, CV resolve — held 256 frames on the indirect-glossy scene agrees with its zero-CV
degenerate to 0.57% and the path tracer to 0.28%, under maximal history compounding. The honest
numbers price the warm window: on a *held* camera the blend correlates the shading estimate the
way step 5 correlated resampling, so accumulated error inside the 16-frame window rises
(unbiasedness pair, temporal-on defaults: many-lights 8 spp 0.03378 → 0.04084, indirect
0.09164 → 0.10470; by 32 spp the gap anneals to +3–6%; the decay-handoff curve reads
+7.7%/+5.0% over temporal-off at 16/32 frames, where step 5 alone read +1.4%) — the cost
decision 6 accepted in exchange for warm-start per-frame colour under motion, bounded by the
same ramp that bounds resampling history and *zero* past the window. Named consequence: the
many-lights 8-spp convergence-gate margin narrows to 1.31× against its 1.3× floor —
deterministic (the suite replays exact sample sequences), but the thinnest margin in the suite,
and the first candidate to re-examine if a driver update ever moves it. Frame time is unchanged
within noise (demo 7.28 ms temporal-on vs 7.36 recorded at step 5); the demo golden moved
mean FLIP 0.045 carried almost entirely by 34 glint-firefly pixels (excluding 66 of 65536
pixels, channel means agree to 0.01–0.10%); the survivor pin re-ran bit-exact. With this rung
ReSTCV is whole: candidate mean → temporal blend → spatial combination → resolve, every stage
naming its lane, and step 7's reciprocal pairing inherits the one remaining named seam.

### D-146: 7-0 weighs the backshift — two thirds of the spatial pass, and 7b is go
Status: accepted (2026-07-26). Rung 7-0 (m6-plan.md §4e decisions 2–3) gave step 7's sharing
claim its denominator before any machinery: the frame-time report grew a **spatial-off** row
(candidates alone — the toggle already existed as `RestirInputs::scratch = None`), and the
backshift share came from an uncommitted, timing-only stub (a const-false around the spatial
kernel's backshift block, so the compiler deletes `targetInDomain`/`evalTarget` and their
replay machinery — the §4b precedent: numbers land, measurement hacks don't). The denominator
(temporal-off − spatial-off, 512², 64 timed frames): demo 6.51 − 2.92 = **3.59 ms**,
glossy-primary 8.70 − 2.32 = **6.38 ms**, distant-glossy 3.06 − 1.44 = **1.62 ms** of spatial
pass. The backshift (baseline − stubbed): **2.34 / 4.14 / 0.95 ms — 65% / 65% / 59%** of the
pass. The interview's caution inverted: cenote's backshift spends no *visibility* rays, but the
replay rays and closure re-evaluations it does spend are the majority of the pass — a
replay-kind canonical replays its whole suffix at each of five neighbours, and the glossy
scenes are replay-heavy, so the "cheap direction" is the expensive one. Decision 3's pre-agreed
rule (skip 7b under ~20% on the worst scene) reads ≥ ~20% everywhere by three-fold: **7b is
go**. The honest bound: the stub also lets dead-code elimination fold the downstream constant
arithmetic (`pairwiseMis` with a zero cross term, the NEE canonical's fBack path), and 7b buys
its deletion back with record traffic and a second dispatch — so 65% is the ceiling on 7b's
realized win, not the promise; the rung's own before/after is the number that counts. Nothing
renders differently: the committed change is report-only, and the stub is reverted.
