# Cenote — M0 Implementation Plan

*Decisions locked 2026-07-06 via structured interview. This file seeds `docs/decisions.md`
when the repo is scaffolded; the charter is the parent document.*

---

## 1. Decisions locked in this session

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 1 | Name | **Cenote**; crates `cenote` (core lib), `cenote-cli` | Distinctive, greppable; future C ABI gets `cenote_` prefix |
| 2 | M0 scope | **Fully headless** — render → EXR → view in `tev` | No swapchain/winit/surface code in M0; viewer is born in M1 as chartered |
| 3 | Layout | Workspace with `crates/`; **shaders live inside the core crate** (`crates/cenote/shaders/`) | Core crate fully self-contained; simplest build.rs paths |
| 4 | Shader compile | **`slangc` subprocess for both** build.rs and runtime hot-reload | One invocation shape, no FFI, no drift; failed reload keeps old pipeline live |
| 5 | Vulkan abstraction | **Thin `gpu` module, no RHI** — Context + purpose-built helpers; direct ash at call sites; helpers extracted on second use, never speculatively | Single-backend renderer by charter; a reader who knows Vulkan reads Vulkan |
| 6 | Binding model | **BDA-first**: all buffers via buffer device address in a push-constant struct; one tiny descriptor set holding only the TLAS | Scales directly to wavefront SoA path state; descriptor indexing enabled-but-unused until textures (M2) |
| 7 | Synchronization | **Blocking one-shot submits**, single compute queue, barriers in-buffer | M0 is strictly sequential; M1's scheduler drives the real sync design |
| 8 | Scene | **Procedural icosphere + ground plane**, two BLAS instances in one TLAS; fixed pinhole camera | Zero file I/O; faceted normals reveal winding/handedness bugs; instancing exercised day one |
| 9 | Golden tests | **`cargo test` + `nv-flip`**, GPU-gated with clean skips; goldens = 256² EXRs in-repo; update only via `UPDATE_GOLDENS=1`; failures dump actual + FLIP heatmap to `target/` | One-command discoverability; FLIP threshold survives driver/compiler FP reordering |
| 10 | Errors | **`thiserror` in core** (one coarse enum: Vulkan, ShaderCompile, Io, NoCapableGpu…), **`anyhow` in bins**; panics = programmer bugs only | Standard library/binary split; enum → C error codes mechanically at M2 |
| 11 | Dependencies | Core: ash, gpu-allocator, exr, glam, bytemuck, thiserror, notify, log. CLI: anyhow, clap (derive), env_logger. Dev: nv-flip. **Policy: every new dep needs one sentence of justification; <100-line replacements get written instead** | Everything listed has an M0 caller |
| 12 | Repo & CI | **Public GitHub from first commit; MIT OR Apache-2.0 dual.** CI: fmt check, clippy `-D warnings`, build, non-GPU tests, pinned-version slangc compiling every shader. GPU goldens = documented local pre-push ritual | The from-scratch history is portfolio narrative; shader breakage caught in CI without a GPU |
| 13 | Docs & lints | Root README (vision, demo image, quickstart, repo map — refreshed per milestone); `//!` header on every module; **append-only `docs/decisions.md` ADR log**; workspace lints: rustfmt defaults, clippy all + curated pedantic (each `allow` commented) | Module headers are the "find what you're looking for" mechanism; the whys must not evaporate |
| 14 | Core layout | `gpu/` = unsafe-Vulkan quarantine (mod/buffer/submit/accel); flat domain modules (shaders, scene, render, output); modules earn files only when they exist | Safe/unsafe boundary visible in the tree; M1 scheduler lands as a new sibling module |

## 2. Leaf defaults (stated, not interviewed — cheap to change)

- **Rust edition 2024**, MSRV = current stable, workspace-level `[workspace.lints]`.
- **Conventions** (recorded in decisions.md): right-handed, Y-up, camera looks −Z; distances in meters. glam types throughout host code; matching conventions stated in `primary.slang`'s header.
- **Kernel output**: storage *buffer* of `f32` RGBA (not a storage image) — simplest readback, and wavefront accumulation in M1 wants a buffer anyway.
- **EXR**: linear, no transform — M0 writes normals-as-color, which is data, not color. ACEScg enters in M1 with actual radiance.
- **CLI**: `cenote-cli --width 1280 --height 720 --out render.exr [--watch]`. `--watch` is the hot-reload demo: file-watch shaders, recompile, re-render, rewrite EXR; `tev` auto-refreshes.
- **Device selection**: require rayQuery + accelerationStructure + BDA + descriptor indexing; prefer discrete GPU; fail with a clear `NoCapableGpu` error listing what each device lacked (the machine has three Vulkan devices — llvmpipe must lose).
- **Validation layers**: on in debug builds, off in release, `VK_EXT_debug_utils` messenger routed to `log`.

## 3. Repo skeleton

```
cenote/
├── Cargo.toml              # workspace: members, [workspace.lints], shared deps
├── README.md               # vision ¶, demo image, quickstart, repo map
├── LICENSE-MIT / LICENSE-APACHE
├── .github/workflows/ci.yml
├── docs/
│   ├── charter.md          # renderer-charter.md moves here
│   ├── decisions.md        # ADR log, seeded from §1 above
│   └── m0-plan.md          # this file
├── crates/
│   ├── cenote/
│   │   ├── build.rs        # slangc → SPIR-V, rerun-if-changed=shaders/
│   │   ├── shaders/
│   │   │   └── primary.slang
│   │   ├── src/            # per layout decision #14
│   │   └── tests/
│   │       ├── golden.rs   # render + nv-flip vs goldens, GPU-gated
│   │       └── golden/     # 256² reference EXRs
│   └── cenote-cli/
│       └── src/main.rs
```

## 4. Build order (~4–6 weeks at 10 h/wk)

Each step ends green: compiles, clippy-clean, committed.

1. **Scaffold** (≈1 session): workspace, licenses, README stub, decisions.md seeded, CI up and passing on the empty skeleton. *Checkpoint: badge is green before any Vulkan exists.*
2. **Shader build path**: `primary.slang` (trivial fill-color kernel), build.rs invoking slangc, SPIR-V embedded via `include_bytes!`. CI compiles shaders from here on.
3. **Device bring-up** (`gpu/mod.rs`): instance, validation messenger, device selection per the leaf default, queue, gpu-allocator. *Checkpoint: `cenote-cli` prints the chosen device and exits 0; a unit-ish test asserts llvmpipe is never selected.*
4. **Buffers + submit** (`gpu/buffer.rs`, `gpu/submit.rs`): RAII buffer, staging upload, one-shot submit. *Checkpoint: upload→download round-trip test.*
5. **First dispatch** (`render.rs`, `output.rs`): bind output buffer via BDA push constant, dispatch fill kernel, read back, write EXR. *Checkpoint: solid-color EXR opens in tev — first image.*
6. **Acceleration structures** (`gpu/accel.rs`, `scene.rs`): icosphere + plane generation, BLAS each, TLAS with two instances.
7. **Ray-query kernel**: pinhole camera rays in `primary.slang`, `RayQuery` against TLAS, geometric normal → color, miss → black. *Checkpoint: the M0 demo image.*
8. **Hot reload** (`shaders.rs`, `--watch`): notify-watch shader dir → slangc subprocess → pipeline swap on success, stderr + keep old pipeline on failure. *Checkpoint: edit sky color in the kernel, tev updates in <1 s.*
9. **Golden harness**: first golden checked in, nv-flip comparison test, `UPDATE_GOLDENS=1` ritual + local pre-push ritual documented in README.
10. **Polish pass**: every module has its `//!` header, README gets the demo image, decisions.md is current. *M0 done.*

Risk watch: the only step with real unknown-unknowns is 6→7 (AS build validation errors, Slang↔ray-query codegen quirks). If the schedule slips, it slips there — steps 8–9 are safe to compress, never to skip.

## 5. Definition of done

- `cargo run -p cenote-cli -- --watch` renders sphere+plane normals to EXR; editing `primary.slang` re-renders in under a second; a broken edit prints the Slang error and keeps the last good image.
- `cargo test` passes with GPU (golden matches) and without (skips cleanly).
- CI green: fmt, clippy `-D warnings`, build, shader compile.
- A stranger can: read README → understand what Cenote is; read `lib.rs` → know where everything lives; read `docs/decisions.md` → know why.
