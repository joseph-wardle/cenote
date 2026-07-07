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
