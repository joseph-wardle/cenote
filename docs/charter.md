# GPU-First Production Path Tracer — Project Charter

*Drafted July 2026. Decisions locked through structured interview; revisit only with cause.*

---

## 1. Vision statement

A portable, GPU-first, interactive-progressive production path tracer built on Vulkan ray tracing, with GRIS/ReSTIR as its theoretical core and a defining thesis: **the interactive lookdev preview and the converged final frame are the same estimator.** No biased preview modes, no "final gather" switch — reuse-accelerated progressive rendering that provably converges, so what the artist sees at one second is an honest prediction of the frame at one hour.

Where MoonRay optimizes for memory capacity and CPU-scale robustness on unbounded production scenes, this renderer optimizes for extreme single-GPU performance on scenes that fit in VRAM — the inverse bet, aimed at smaller-scale productions and lookdev.

**Success criteria, ranked (locked):**
1. Portfolio depth for studio recruiting (rendering + pipeline roles)
2. Deep learning of modern GPU rendering
3. Renders a real shot from *Sandwich Kwon Do* (stretch proof, not deadline)
4. ~~Long-lived tool others adopt~~ — explicitly out of scope; no robustness tax, no API stability promises, no docs-for-strangers

**Identity (locked):** interactive-progressive first. Final-frame batch is the same engine left running, not a separate mode.

---

## 2. Locked decisions

| Domain | Decision | Key rationale |
|---|---|---|
| Purpose | Portfolio + learning; film = stretch proof | Vertical slices beat completeness |
| Identity | Interactive-progressive, converging to final | Preview-matches-final thesis |
| USD/Hydra | Core deliverable, MoonRay pattern: own lean scene rep + separate Hydra delegate | Quarantines USD C++ swamp outside the core |
| Host language | **Rust** (ash + gpu-allocator + winit + egui) | Mature VK RT bindings, existing XPBD miles, language stability, recruiter legibility |
| Shader language | **Slang** → SPIR-V | RT support, generics/modules, prior experience |
| Execution | **Wavefront + ray queries in compute, from day one** | ReSTIR is multi-pass anyway; no SBT zoo; one programming model; matches Cycles X / MoonRay convergent architecture |
| Materials | **Fixed OpenPBR ubershader**; params = constant or texture; pattern-VM seam reserved | Single coherent shading kernel; industry-converging standard |
| Sampling | **GRIS as the theoretical core from day one**, staged GRIS-DI → ReSTIR PT; unbiased-by-construction; **convergence-under-reuse is a named design pillar** | Anything biased poisons the offline path; correlation management (M-caps, reuse decay, spatial-only converge mode) is the novel territory |
| Integrator | Exactly one: unidirectional PT + NEE/MIS, GRIS layered on | What Cycles X and MoonRay both do; features are stages, not systems |
| Geometry staging | Triangles + pre-tessellated subdivs → **curves** → **SSS (random walk)** → **volumes (NanoVDB)**; path state + scheduler designed for all four now | Film needs curves; SSS is cheap once wavefront exists; volumes are the biggest lift |
| Denoising | **OIDN only, both modes** (GPU backend, Vulkan interop); albedo/normal AOVs are core outputs | One denoiser = preview honestly predicts final; vendor-neutral |
| Process boundary | **Library-first**: Rust core behind a narrow C ABI designed around **transactional change-sets**; thin render-server wrapper later; Hydra delegate (C++) talks to server | Bento philosophy; change-set API = interactive edits + Hydra dirtying + file format, one mechanism |
| Memory | **Everything-resident, mip-capped**; BC7/BC5 transcode at prep; bindless indirection everywhere as future streaming seam; out-of-core geometry ruled out | The defining anti-MoonRay freedom |
| Hardware | **Any Vulkan RT GPU** (KHR core only, no vendor extensions in core paths) | Wavefront+ray-queries already avoids NV pipeline goodies |
| Color | **ACEScg working space in core**; built-in ACES output transforms + baked 3D LUT loading; sRGB→ACEScg texture IDT at prep; OCIO lives in the tooling/delegate layer later | Build simplicity; leverages existing LUT-baking expertise |
| Image I/O | `exr` crate (pure Rust) for EXR in/out; OIIO optional importer only, never a core dep | Clean build |
| Budget | ~10 hrs/week; first portfolio-visible artifact well before **April 2027** (~400 total hours to graduation) | Milestones sized in 2–8 week units |

---

## 3. Architecture overview

```
┌────────────────────────────── clients ──────────────────────────────┐
│  CLI batch app   interactive viewer (winit/egui)   Hydra delegate   │
│      (Rust, links core)        (Rust, links core)   (C++, own repo, │
│                                                      talks to ↓ )   │
├──────────────────────── render server (later) ──────────────────────┤
│        thin wrapper over C ABI · shared-mem framebuffer · IPC       │
├──────────────────────────── core (Rust cdylib) ─────────────────────┤
│  Scene: named objects, typed attributes, transactional change-sets  │
│  Render prep: subdiv tessellation · BC/mip transcode · IDT ·        │
│               BLAS/TLAS build · light list / alias tables           │
│  Orchestrator: wavefront stage scheduler · indirect dispatch ·      │
│                compaction · progressive accumulation · convergence  │
│                policy · OIDN interop · EXR/AOV output               │
├──────────────────────────── GPU (Slang kernels) ────────────────────┤
│  raygen/regen · intersect (ray queries; procedural-primitive-aware) │
│  shade_surface (OpenPBR ubershader) · NEE/shadow · GRIS passes      │
│  (candidates → temporal → spatial → resolve) · [reserved stages:    │
│  intersect_curve · sss_walk · volume_stack/shade_volume]            │
│  Path state: SoA IntegratorState-style, millions of paths,          │
│  volume-stack slot + flags reserved from day one                    │
└──────────────────────────────────────────────────────────────────────┘
```

**Design pillars (the five commitments):**
1. Extensible SoA path state — reserve volume stack, path flags, and GRIS reconnection data now.
2. Stage scheduler that tolerates inserted kernels (Cycles X model: each path records its next kernel).
3. Intersection layer that speaks procedural primitives even while only triangles exist.
4. Slang feature-flag kernel specialization — scenes without hair/volumes pay nothing.
5. One integrator; everything else is a stage or a sampling strategy inside it.

**GRIS-specific early commitments:** path state carries reconnection vertices + Jacobian inputs; RNG is replayable (counter-based, e.g., PCG/Philox keyed by pixel/sample/bounce) so shift mappings can re-trace; G-buffer layout chosen with reuse-domain MIS in mind; every reuse pass has an M-cap and a decay schedule wired to the convergence policy.

---

## 4. Milestone roadmap (~10 hrs/week)

Sizing is honest: **M0–M5 fit before April 2027 (~400 h)**. M6+ are post-graduation or capacity-permitting. Each milestone ends with a demo artifact.

**M0 — Skeleton** (4–6 wks)
Cargo workspace; ash device init with `VK_KHR_ray_query` + `acceleration_structure` + descriptor indexing; Slang compiled at build *and* hot-reloadable at runtime; BLAS/TLAS build; one compute kernel ray-querying a triangle mesh; EXR out via `exr`. *Demo: flat-shaded mesh render, hot-reloading shader.*

**M1 — Wavefront core + first light** (6–8 wks)
SoA path state; stage scheduler with indirect dispatch + compaction; OpenPBR subset (base + specular); NEE/MIS with area lights + HDRI importance sampling; progressive accumulation; ACEScg→display transform; winit/egui viewer with orbit camera and live parameter tweaks. *Demo: interactive path-traced viewer — the first portfolio-visible artifact.*

**M2 — Production I/O + full look** (4–6 wks)
Change-set scene API + C ABI; text scene format (serialized change-sets); PBRT-format importer subset → free regression corpus; full OpenPBR closure; BC transcode + mip-cap prep pipeline; OIDN interop; albedo/normal/depth AOVs. *Demo: PBRT scenes rendered, denoised, side-by-side vs pbrt-v4.*

**M3 — GRIS-DI** (6–8 wks)
Reservoirs, unbiased contribution weights, generalized MIS; temporal + spatial reuse; convergence policy v1 (M-caps, reuse decay, spatial-only converge mode); validation harness — FLIP + mean-error-vs-reference plots proving convergence to the M1 ground truth. *Demo: many-light scene, equal-time comparison + convergence curves. Flagship begins.*

**M4 — Hydra delegate + render server** (6–8 wks)
Thin server over the C ABI; shared-memory framebuffer; C++ delegate in a separate repo with its own (USD-infected) build. *Demo: your renderer live inside usdview. The single most legible artifact for pipeline-TD recruiting — which is why it outranks ReSTIR PT in sequence. Swap with M5 only if a rendering-research role becomes the primary target.*

**M5 — Geometry depth** (5–7 wks)
Subdiv tessellation at prep (uniform Catmull-Clark own-rolled, or OpenSubdiv behind the prep boundary); curves via procedural AABBs + custom hair intersector in the ray-query loop; hair BSDF lobe. *Demo: groomed character asset.*

**M6 — ReSTIR PT / GRIS full** (8–12 wks, post-grad likely)
Reconnection shift, hybrid shift, random replay; convergence-under-reuse study — this is the research-edge flagship and potential SIGGRAPH-poster territory.

**M7 — SSS random walk** (3–4 wks) → **M8 — Volumes** (8+ wks, NanoVDB via pnanovdb from Slang, volume stack, null scattering) → **Stretch: film shot proof** — one *Sandwich Kwon Do* shot re-lit and rendered (materials re-baked to OpenPBR textures), compared against the RenderMan frame.

---

## 5. First implementation steps (week one)

1. `cargo new` workspace: crates `core`, `viewer`, `cli`; `shaders/` with Slang.
2. Build script invoking `slangc` → SPIR-V, plus a runtime file-watch recompile path (this is your iteration loop — build it first).
3. ash device bring-up: pick physical device requiring `rayQuery`, `accelerationStructure`, `bufferDeviceAddress`, descriptor indexing; validation layers on.
4. gpu-allocator integration; staging-upload helper; BLAS/BLAS→TLAS build for one hardcoded mesh.
5. One Slang compute kernel: ray query per pixel, write normal-as-color to a storage buffer; readback → `exr` write.
6. Set up the validation harness *early*: a `tests/scenes/` dir and a script that renders + FLIP-compares against goldens on every change.

**Reference stack while building:** PBRT ch. 15 (wavefront), Cycles X kernel-scheduling docs, the MoonRay HPG'17 paper (bundled path tracing), the GRIS paper (Lin et al. 2022) + course notes, and the Falcor ReSTIR PT release as the correctness oracle for M3/M6.

---

## 6. Explicit non-goals (write these on the wall)

- No CPU fallback renderer. No OpenCL/CUDA/Metal backends.
- No arbitrary shader graphs / MaterialX codegen in v1 (pattern-VM seam only).
- No out-of-core anything. Scene fits in VRAM or fails loudly.
- No bidirectional/photon/VCM integrators.
- No API stability, plugin SDK, or third-party-user support.
- No path guiding in v1 (complementary axis, revisit post-M6).
- Biased shortcuts that break preview-predicts-final — never, that's the thesis.
