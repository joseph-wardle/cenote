# Cenote — M1 Implementation Plan

*Decisions locked 2026-07-07 via structured interview, then amended the same day after
a sourced review against Cycles X's source, MoonRay, and current research (see
D-032…D-037). Parent scope is charter §4 M1: wavefront core + first light. Decisions
D-020…D-037 in [decisions.md](decisions.md) carry the full rationale; this file is the
working plan.*

---

## 1. Decisions locked in this session

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 1 | Scope | **Full charter M1, staged as a walking skeleton** with named fallback seams; nothing cut up front | The wavefront engine and "first light" features carry different risk; staging isolates them |
| 2 | Scheduler | **Host-driven fixed loop**: fixed stage order per bounce recorded once per wave; GPU-side queues + counters; indirect dispatch; zero mid-frame readbacks | Satisfies every charter commitment; Cycles-X-style adaptivity layers on later without changing kernel contracts |
| 3 | Path pool | **Fixed capacity (~1M, configurable) + tile loop** per sample | Bounded VRAM at any resolution; degenerates to one tile at viewer sizes |
| 4 | Path state | **SoA behind a schema seam**: one point of definition (Rust struct of buffers ↔ mirrored Slang address struct); M1 allocates only fields M1 reads | "Reserved from day one" means adding an M3/M8 field is a two-line change, not that dead memory ships |
| 5 | Stage set | **Six kernels**: raygen / intersect / shade_miss / shade_surface / trace_shadow / accumulate. Forward emissive hits are MIS-weighted in shade_surface (so `prev_bsdf_pdf` is an M1 path-state field) | Tracing and shading never share a kernel; every M3 insertion point already exists as a queue boundary. Cycles gives emissive hits their own kernel — at one-ubershader scale, folding into shade_surface is right, and the queue boundary exists if it earns one |
| 6 | Sync | **Sequential waves, one graphics+compute queue, timeline-semaphore pacing**; display image double-buffered as the future overlap seam | Measured stalls, not speculation, drive concurrency (D-007's reasoning, continued) |
| 7 | Sampler | **Sobol-Burley** — hash-based Owen-scrambled Sobol (Burley, JCGT 2020) — keyed (pixel hash, sample, dimension); named dimension registry; `sample_1d/2d` API seam | Replayable by construction (the GRIS requirement); ~200 lines; the production baseline under Cycles' and pbrt-v4's current defaults. Blue-noise index ordering is the later drop-in |
| 8 | Materials | **Three OpenPBR lobes**: EON diffuse base (energy-preserving Oren-Nayar) + conductor GGX + dielectric specular layer (albedo-scaling layering). GGX sampled via Dupuy-Benyoub spherical caps, with Turquin-style energy compensation via the Sforza-Pellacini analytic fits; constant params per instance | Smallest set that stresses MIS and reads as a real renderer; grows additively in M2. EON is the lobe OpenPBR actually specifies; uncompensated GGX fails an albedo-1 furnace test by design |
| 9 | Lights | **Emissive mesh quads in the TLAS** + power-proportional alias table + **equirect HDRI as M1's only texture** (CDF importance sampling, MIS in shade_miss). The env light exposes `sample()` and `pdf(dir)` as separate entry points | BSDF paths hit lights through the ordinary intersect path — no second code path for MIS to keep honest. The `pdf(dir)` query serves MIS now and every ReSTIR target-function/Jacobian evaluation in M3 |
| 10 | Color | Authored colors are **linear Rec.709 → ACEScg at prep**; display via **analytic ACES fit** in the tonemap kernel — a swappable stage, not a baked-in look; EXRs stay linear ACEScg with chromaticity metadata | Pure ACEScg core. ACES 2.0 has no shader-friendly form (the ACES community's own engine guidance is "bake a 3D LUT via OCIO"), so Hill-fit-now + LUT-seam-later *is* the recommended shape; the same slot admits AgX or ACES 2.0 |
| 11 | Viewer | **`cenote-viewer` crate**: winit + egui (`egui-ash-renderer`), offscreen tonemap → blit → egui pass, single-threaded event loop, one wave per redraw; edits reset accumulation | Windowless core; re-convergence after an edit *is* the thesis demo |
| 12 | Testing | **Four layers**: goldens (fixed seed/spp, FLIP) + white furnace per lobe + MIS-agreement + bitwise determinism; CPU unit tests for host-shared math | Furnace explains energy bugs, MIS-agreement catches wrong-but-plausible weights, determinism enforces the replay guarantee |
| 13 | Build order | **Viewer-first** (§4) | The window becomes the iteration loop before the hard engine work starts — M0's build-the-loop-first lesson |

## 2. Leaf defaults (stated, not interviewed — cheap to change)

- **Dependencies** (per D-011 policy): core gains none (HDRI loads via the existing
  `exr`). `cenote-viewer`: winit, egui, egui-winit, egui-ash-renderer, anyhow,
  env_logger. An immediate-mode UI renderer is thousands of lines — nowhere near the
  <100-line write-it-yourself bar.
- **Pool default**: 2²⁰ paths. **Max depth** 8, Russian roulette from bounce 3.
  **CLI**: `--spp` (default 64) and `--depth` join the M0 flags; viewer accumulates
  indefinitely.
- **Accumulation**: f32 RGBA sum buffer + host-tracked sample count (uniform across
  pixels by construction); tonemap divides. Display image RGBA8.
- **SoA granularity**: one buffer per logical field, 16-byte-friendly packing;
  per-component splits — or packing hot fields AoS-within-SoA behind accessors, the
  lesson Cycles learned on Apple Silicon — are measured optimizations, not day-one
  guesses.
- **Hit encoding**: path state stores hits as instance + primitive + barycentrics —
  re-evaluable shading, and the form M3 reservoirs must hold. `prev_bsdf_pdf` is an
  M1 field (MIS for forward emissive hits); the per-bounce sampled-lobe/technique
  tag that GRIS random replay needs is a known future field via the schema seam.
- **Shadow queue entries are self-contained records** (origin, direction,
  unshadowed contribution, pixel) rather than main-path fields — simpler now, and
  already the shape of the separate shadow pool that M3's multi-candidate NEE wants.
- **Ray offsets**: self-intersection avoidance per van Antwerpen's rigorous-bounds
  method (NVIDIA 2023, reference HLSL/GLSL published), applied to bounce and shadow
  rays alike; Wächter-Binder (Ray Tracing Gems ch. 6) is the fallback. Never magic
  `TMin` epsilons.
- **Robustness at the film**: every contribution is finite-guarded (NaN/Inf dropped)
  before accumulation, unconditionally. Firefly clamping ships **off** by default —
  clamping changes the ground truth the thesis promises; deliberate divergence from
  Cycles (indirect clamp 10.0), revisited with the M2 denoiser.
- **Determinism invariant**: radiance is written only to path-owned or pixel-owned
  memory — never atomically accumulated — which is why the bitwise-determinism test
  holds despite nondeterministic queue-push order.
- **Shader layout**: one `.slang` per kernel stage plus shared modules (`pathstate`,
  `rng`, `openpbr`, `lights`, `colorspace`, `camera`). `primary.slang` serves the
  viewer skeleton (build step 2) until the wavefront replaces it (step 5), then
  retires with a regenerated golden.
- **Rust layout** (per D-014, files earn existence): core gains `wavefront.rs`
  (path state + queues + stage loop; splits into a directory when it earns it),
  `material.rs`, `lights.rs`, `color.rs`; `render.rs` becomes the wave/tile
  orchestrator; `scene.rs` grows materials, lights, and HDRI load. Viewer starts as
  `main.rs` + `camera.rs`.
- **Hot reload** extends across all kernels under the D-018 contract unchanged:
  layouts pinned by the embedded build, reload swaps SPIR-V bodies only.
- **Demo scene**: procedural still — ground plane + a row of spheres sweeping
  roughness/metalness + one emissive quad, under a small checked-in CC0 HDRI
  (≤512×256 EXR, attribution in-repo). **Demo artifact**: README beauty still +
  a short re-convergence screen capture.
- **Dimension registry**: named constants (`CAMERA_JITTER`, `BSDF_LOBE`, `BSDF_DIR`,
  `NEE_LIGHT`, `NEE_POINT`, `RR`, …) in `rng.slang`; call order never allocates.

## 3. Layout additions

```
crates/
├── cenote/
│   ├── shaders/            # kernels: raygen, intersect, shade_miss, shade_surface,
│   │                       #          trace_shadow, accumulate, tonemap
│   │                       # modules: pathstate, rng, openpbr, lights, colorspace, camera
│   └── src/
│       ├── wavefront.rs    # SoA path state, queues, stage loop
│       ├── material.rs     # OpenPBR params + GPU buffer layout
│       ├── lights.rs       # light list + alias table build
│       └── color.rs        # Rec.709→ACEScg, chromaticity metadata
└── cenote-viewer/
    └── src/                # main.rs, camera.rs (orbit); ui module when it earns a file
```

## 4. Build order (~6–8 weeks at 10 h/wk)

Each step ends green: compiles, clippy-clean, committed.

1. **Plan docs** (this file, decisions.md entries, README repo-map row).
2. **Viewer skeleton**: `cenote-viewer` crate — winit window, swapchain, blit of the
   M0 render; orbit camera re-renders on move. *Checkpoint: the M0 image live in a
   window. The iteration loop for everything after.*
3. **egui overlay**: device/frame stats, placeholder sliders.
4. **Accumulation + tonemap**: accumulation buffer, ACES-fit tonemap kernel, display
   path. (Deterministic input for now — proves the display plumbing in isolation.)
5. **Wavefront skeleton**: SoA path state, queues, indirect dispatch, all six kernels
   in degenerate form (shade_surface emits normal-as-color and terminates; shade_miss
   is constant sky). *Checkpoint: the M0 visual through the new engine; determinism
   test green. The riskiest step — see §6.*
6. **RNG + anti-aliasing**: Sobol-Burley, dimension registry, camera jitter.
   *Checkpoint: progressive refinement is real — edges converge.*
7. **EON diffuse + bounce loop**: BSDF sampling, Russian roulette, constant-sky
   lighting, van Antwerpen ray offsets. *Checkpoint: first GI (color bleeding);
   diffuse furnace test green.*
8. **NEE + MIS**: quad light, light list, alias table, trace_shadow live, MIS
   weights; forward emissive hits weighted via `prev_bsdf_pdf`. *Checkpoint:
   MIS-agreement test green.*
9. **GGX lobes**: conductor + dielectric specular layer, spherical-caps VNDF,
   Turquin energy compensation (Sforza-Pellacini fits); furnace matrix over
   OpenPBR's listed white configurations; sliders go live on real material params.
   *Checkpoint: the demo scene looks like a renderer.*
10. **HDRI**: EXR load → sampled image, marginal/conditional CDF (`sample()` +
    `pdf(dir)` split), MIS in shade_miss.
11. **Tiles + CLI**: tile loop engages when pixels > pool; `--spp`/`--depth`;
    batch output matches the viewer. *Checkpoint: batch/viewer golden agreement.*
12. **Polish**: goldens regenerated and eyeballed, module headers current, README
    refreshed with the M1 demo, decisions.md current. *M1 done.*

## 5. Fallback seams (pre-agreed, in slip order)

- **HDRI (step 10)** → ship constant-sky; the MIS plumbing and shade_miss stage all
  exist, so the upgrade later touches only env eval + its table. First to go.
- **Sliders (step 9)** → a handful of hardcoded parameter presets on number keys.
- **Steps 5–8 are never compressed** — they are the milestone.

## 6. Risk watch

Step 5 carries the unknown-unknowns (indirect dispatch + queue-counter validation
quirks, six pipelines sharing one push-constant discipline); it is deliberately placed
after the viewer exists so debugging happens against a live image. Step 10's CDF
edge cases (pole singularities, zero-radiance rows) are the second watch item.
`egui-ash-renderer` compatibility is checked in step 3 while stakes are low.

## 7. Definition of done

- `cargo run -p cenote-viewer`: orbit the demo scene, drag material sliders, watch
  the image re-converge in seconds — preview and final are visibly the same estimator.
- `cargo run -p cenote-cli -- --spp 256` writes the same image the viewer converges
  to (golden-verified), linear ACEScg EXR with chromaticity metadata.
- `cargo test` green with GPU: goldens, the furnace matrix (OpenPBR's white
  configurations), MIS-agreement, bitwise determinism; skips cleanly without one.
  CI green; hot reload still under a second.
- A stranger can: read `lib.rs` → find the wavefront in one hop; read any kernel →
  see its data dependencies in one struct at the top; read `docs/decisions.md` → know
  why every one of these choices was made.
