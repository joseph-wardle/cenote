# Cenote — M4 Implementation Plan

*Decisions locked 2026-07-14 via structured interview, preceded by a sourced research
pass over two fronts: the modern Hydra render-delegate API (the `HdRenderDelegate` /
`HdRprim::Sync` contract, `hdTiny` and `hdEmbree` as the floor and the template, the
change-tracker dirty-bit model, and the Hydra-2.0 / scene-index **emulation** story that
keeps a classic Sync delegate the correct target); and the Arras/MoonRay out-of-process
render server (`arras:local` single-machine mode, `RDLMessage` scene deltas,
`ProgressiveFrame` streaming, and husk's process-level isolation), read against cenote's
own change-set scene API (`scene/changeset.rs`) and `render::Session`. Cross-renderer
survey: hdArnold / hdPrman / hdCycles / Karma are in-process; hdMoonray+Arras is the
out-of-process exception, chosen for exactly the crash-isolation reason M4 wants. The
locked decisions were then re-verified by an adversarial fact-checking pass against
OpenUSD source, the openmoonray repos, and the UsdLux / UsdPreviewSurface / OpenPBR
specs; its amendments are folded in below. A third pass (2026-07-14) then **pivoted the
delegate to Hydra 2**: source-level research against OpenUSD 25.11–26.05 and dev — the
scene-index observer machinery, hdPrman's experimental Riley observer back end, the
`HdRenderer` stub, and the AOUSD deprecation roadmap — established that scene-index
consumption is the blessed path (26.03 made it the UsdImaging default and deprecated
scene-delegate mode), so the delegate is now **scene-index-native**: zero Rprims, an
`HdSceneIndexObserver` on the terminal scene index, pinned to stock USD 26.05. Parent
scope is charter §4 M4: **Hydra render delegate + render server**. Decisions
D-097…D-104 in [decisions.md](decisions.md) carry the full rationale; this file is the
working plan. Everything consciously *not* built lives in [deferrals.md](deferrals.md)
with its revival trigger.*

Three framing facts the research settled, because they govern every choice below:

- **The hard half already exists.** `render::Session` already runs the render loop on
  its own thread, applies queued change-sets at wave boundaries (stop → apply → minimal
  re-prep → restart), and publishes frames for a consumer to peek. `ChangeSet` is
  already the serde wire value — "file = wire = edit are the same value by construction"
  — and `Op::Remove` was written *for* a scene-graph delegate ("renames arrive as remove
  + re-insert"). M4 is **three adapters** — a socket around `Session`, a C++ Hydra→`Op`
  translator, and a framebuffer transport back — not a new renderer and not a new sync
  protocol.
- **The delegate is scene-index-native — deliberately at the forefront.** The classic
  Sync path was condemned while M4 was being planned: 26.03 made scene-index mode the
  UsdImaging default and deprecated scene-delegate mode, with removal on Team Hydra's
  stated roadmap. The consuming contract, meanwhile, is verifiably stable —
  `HdSceneIndexObserver` unchanged in two years, `HdsiPrimManagingSceneIndexObserver`
  byte-identical since 24.03, the geometry schemas frozen. The reference implementation
  is hdPrman's experimental Riley observer back end (`SetTerminalSceneIndex` → notice
  batching → a prim-managing observer), the only scene-index-native renderer in the
  OpenUSD tree; cenote goes further — an *all*-observer delegate with **zero Rprims**,
  more scene-index-native than anything shipping, including hdPrman's default
  configuration. The pure `HdRenderer` is *not* the target: the class is a verbatim stub
  ("TODO: Add API here") whose only implementation is the adapter wrapping a render
  delegate — which is exactly how cenote runs under the 26.03+ engine. The deferral and
  its trigger live in [deferrals.md](deferrals.md).
- **The process boundary *is* the C++/Rust boundary.** Out-of-process is not exotic
  here — the language seam and the process seam are one line. It buys crash isolation
  (the host survives a renderer crash) and keeps the pixel path "just bytes," which is
  what keeps the whole split simple and portable.

---

## 1. Decisions locked in this session

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 1 | Target & definition of done (D-097) | **usdview-live is the milestone; the render server is built Houdini-*ready* but Houdini integration is a later milestone.** Done = cenote appears in usdview's Renderer menu and renders/refreshes a real USD stage live. Pinned to stock **USD 26.05**. The Houdini-ready *rule*: nothing in the transport or server may assume usdview specifically or bake in a stock-USD-only assumption an HDK rebuild couldn't satisfy | The charter names usdview-live as the legible pipeline-TD artifact. Stock USD is one pinned version, compile- and GPU-testable without a DCC license; folding the HDK ABI maze into M4 would roughly double the integration surface on the fragile step. The rule makes Houdini a recompile-and-package step later, not a rearchitecture. 26.05 because it is the newest release and the first where scene-index is the blessed default — nothing older is calmer, an older pin just defers the same breaks to the first upgrade |
| 2 | Process boundary & delegate thickness (D-098) | **Out-of-process render server + a thin scene-index-native translator.** The delegate lists **no supported Rprims** (back-end emulation then instantiates nothing) and attaches an `HdSceneIndexObserver` to the terminal scene index via `SetTerminalSceneIndex`, notices batched (`HdsiPrimTypeNoticeBatchingSceneIndex`) and flushed in `Update()` — Hydra's serial per-frame hook, run before task execution. All validation, reference resolution, and prep stay server-side. Hydra's dirty *locators* are cenote's patch — a locator-guarded pull maps 1:1 onto an `Option<…>` patch field, clean attributes stay `None`; `PrimsRemoved` is `Op::Remove`, which was written for exactly this | The language seam and process seam coincide, so isolation is nearly free. `SceneDescription::apply` is validate-then-apply-atomic with an equality gate *today*; re-implementing that in C++ is the exact duplication to avoid, and the equality gate means a redundant forward can't even cause a spurious re-prep. The zero-Rprim shape is verified legal against source — `HdSceneIndexAdapterSceneDelegate` silently skips unsupported prim types — and the observer pattern is hdPrman's own (its renderDelegate.cpp documents the rationale), kept to its thinnest. Dirty locators are finer-grained than the classic dirty bits, so the translator only gets thinner |
| 3 | Crash isolation, not recovery (D-099) | **Isolation via the process boundary; recovery is manual.** On server death the delegate degrades gracefully (dead-socket detection, no self-crash) and recovers on the next destroy/recreate — renderer toggle or stage reload — which hands the recreated delegate a fresh terminal scene index, and the observer receives `PrimsAdded` for the whole stage. No delegate-held replay state. The connection is **genesis-then-deltas** shaped so automatic replay is a clean later bolt-on with no wire change | Isolation is delivered by the boundary alone. A render delegate can't cleanly self-trigger a Hydra repopulate, so the only real options are manual-zero-state or a full delegate-side mirror; the mirror costs a second full copy of *geometry* in the delegate process plus respawn orchestration, unjustified for a usdview target where re-populate is a two-second toggle. Verified against source: hdMoonray pays exactly that mirror — its retained `SceneContext`, which its RDL delta-encoding needs as a baseline anyway (Arras itself restarts nothing; the delegate lazily reconnects and re-sends genesis) — where cenote's patches come straight from Hydra's dirty locators and need no baseline at all |
| 4 | Control transport (D-100) | **Loopback TCP (`127.0.0.1`, ephemeral port) + a spawn-time token — one code path in both languages on every platform; length-prefixed frames, strict request/response; payload = the existing serde `ChangeSet` as MessagePack, plus a small control surface for what is not scene data — `SetCamera` (the inputs-lane fast path), `Resize` (the shm handshake), `Ping`; continuous status (frame counter, samples, converged, rejected-edit count) rides the shm header, never the socket; explicit, documented wire types with a trivial wire→`Op` translation; C++/Rust agreement pinned by a USD-free byte-exact corpus test.** Not gRPC, not UDS | Minimal C++ dependencies *is* a Houdini-ready requirement: gRPC drags in protobuf + abseil, which the host ships its own ABI-incompatible copy of. The C++ standard library has no IPC at all, and loopback TCP is the one transport that is single-source-path everywhere (`std::net` in Rust; BSD sockets ≈ Winsock to within four lines), with no stale socket files — the token closes the any-local-process gap UDS permissions covered. Strict request/response deletes the C++ async-reader thread outright; status-in-shm makes `IsConverged()` a header read. The control surface is load-bearing, not optional: viewport resizes arrive as `HdRenderBuffer::Allocate` — never a scene edit — and `Session`'s inputs-lane camera always wins over the scene camera, so camera *must* travel outside the `ChangeSet`. The `ChangeSet` is already the wire value (one source of truth; gRPC would force a second `.proto` schema). The corpus test replaces gRPC's codegen as the drift guard, aimed at the 20-field material patch, and runs in CI with no USD and no GPU |
| 5 | Pixel transport (D-101) | **CPU shared memory, double-buffered async readback, beauty (+ first-hit depth) only — beauty converted server-side from `ACEScg` to linear `Rec.709` before the shm write.** Sit it behind `HdRenderBuffer`, whose `GetResource()` GPU-texture path stays a *measured* later upgrade | At lookdev viewport resolution the readback (~1.3 ms/1080p on PCIe 4, a fraction of one 2.6–8 ms sample) hides behind the next frame's render; the high-res regime where it bites is the converged still, where per-frame latency stops mattering — the two are anti-correlated. `hdEmbree` proves CPU-pixels→`HdRenderBuffer` is a fine interactive delegate, and the pivot re-verified the seam: `renderBuffer` Bprims remain the only pixel path even under the 26.03+ `HdRenderer` engine — `HdxAovInputTask` `Map()`s CPU pixels. GPU-share re-couples the two processes at the GPU-memory + Hgi-backend level, undoing the clean byte-boundary. The color conversion is not optional: usdview's default color correction applies *only* the sRGB transfer curve — no gamut conversion exists anywhere in the default path — so the color AOV must already carry `Rec.709` primaries or every frame is silently oversaturated. Pixar's own fix (hdPrman on OpenUSD dev, 2026: a display filter converting to hardcoded linear `Rec.709` inside the renderer) confirms delegate-converts is the de facto contract |
| 6 | Material scope (D-102) | **UsdPreviewSurface only** — a bounded switch from the `surface` terminal of the material network schema (`HdMaterialNetworkSchema`: nodes, parameters, connections read as data sources; the universal render context is the empty token, read explicitly — 26.03 removed the cross-context fallback) covering `UsdPreviewSurface` + `UsdUVTexture` + `UsdPrimvarReader_*` into `MaterialPatch`; meshes with no bound material shade from `displayColor` — real stages (layout, crowds) often carry no material at all. `open_pbr_surface` recognition and full MaterialX-graph evaluation deferred | UsdPreviewSurface is the USD lingua franca every asset ships or falls back to, and its default-workflow params map near-losslessly onto cenote's shipped closure — four documented exceptions: `useSpecularWorkflow=1`'s direct F0 (approximated through the specular tint), `displacement` and `occlusion` (no OpenPBR home), and `opacityThreshold` (cutout applied delegate-side before the patch). cenote is a **fixed-closure** renderer, so the heavy MaterialX-SDK-codegen path is architecturally moot — not deferred, nonexistent for this design. `open_pbr_surface` (whose params *are* cenote's, both being OpenPBR) is a one-branch fast-follow on the same switch |
| 7 | Light mapping (D-103) | **UsdLux → cenote's existing paths.** Params read lazily by name from the member-less light container (UsdLux attribute names through `Get(name)` — enumeration is impossible by design; `treatAsPoint` via raw-attribute fallthrough). distant → `Distant` delta; sphere+`treatAsPoint` → point delta; dome → `Environment` (one active); rect/disk/sphere-area/cylinder → a synthesized emissive mesh + emissive material + instance, placed by the light transform, radiance = `intensity·2^exposure·color`, × the blackbody RGB when `enableColorTemperature`, ÷ emitting area when `normalize`. Rect/disk wound so the front face emits −Z (cenote's emissive surfaces are already one-sided, matching UsdLux); distant's default 0.53° `angle` collapses to the delta — the stated floor. Light textures, the per-lobe `diffuse`/`specular` multipliers, and the shaping API deferred | cenote already *has* area lights — emissive meshes, MIS-consistent, ReSTIR-integrated, golden-covered, one-sided like UsdLux's rect/disk — so synthesis is the correct mechanism, not a workaround. The delegate's job is to translate to whatever cenote represents lights as. Native analytic sampling is a core-estimator feature justified by measured variance (sphere lights first), not by the presence of a UsdLux prim type |
| 8 | Repo / build / ABI shape (D-104) | **Three components:** `cenote-server` (Rust binary wrapping `render::Session` + transport + shm framebuffer), `cenote-wire` (Rust: the USD-free wire types + MessagePack + wire→`Op` translation), and `hydra/` (a C++ CMake tree: the delegate shell — the adapter ring, zero Rprims — plus the scene-index observer, a C++ wire mirror, transport client, and server spawn; the renderer-plugin bootstrap glue isolated in one thin file, since that surface has broken twice — 23.02, 25.11 — and dev already carries 26.08's). Built against **system-provided USD** (stock 26.05 for usdview, *or* HDK — the Houdini pivot), not a vendored USD build | The server is plumbing around code that already exists. Keeping the wire encoder USD-free lets the drift guard run in CI without USD/GPU. System-provided USD is light and *is* the Houdini pivot: stock ↔ HDK is a build-root change on one USD-version-agnostic source. Vendoring USD's enormous build is unjustified for a solo project |

M4 also **picks up four standing deferrals**, whose entries move from
[deferrals.md](deferrals.md) into dated decision entries at M4's opening commit (per the
ledger's non-append-only rule), not before: the **C ABI** (D-052) — realized as
serialized change-sets over the socket rather than per-attribute FFI, the MoonRay
`RDLMessage` shape it always named; the **binary change-set wire format** (D-055) —
realized as MessagePack; the **array instancer op** (D-073) — the form Hydra's
instancer prims deliver (step 5); and the **viewer single-source-of-truth scene graph**
(D-064/D-082/D-083) — the delegate's render-index-owns-the-scene model, which retires the
hand-mirrored `ui_desc` replica.

## 2. Leaf defaults (stated, not interviewed — cheap to change)

- **Dependencies**: the Rust side adds a MessagePack codec (`rmp-serde`) — and nothing
  else: the transport is `std::net` loopback TCP, so there is no IPC crate to justify. The
  core renderer's public surface gains exactly one function: `color.rs` grows
  `rec709_from_acescg()`, the runtime inverse of the same private authoring constant, so a
  second hand-typed matrix never exists. The C++ side links USD/HDK and
  a small header-only msgpack library — and *deliberately nothing heavier*, because every
  transitive C++ dependency is a host-ABI liability (the reason gRPC was rejected).
- **Threading**: scene-index notices arrive on whatever thread mutates the scene index;
  the notice-batching scene index holds them, and the delegate's `Update()` — called
  serially by `SyncAll` before task execution — flushes the batch through the observer.
  Translation is therefore **single-threaded by construction**: one flush → one atomic
  `ChangeSet`, mapping 1:1 onto cenote's wave-boundary apply. The classic design's
  parallel-Sync accumulator lock is deleted, not ported. `Update()` stays cheap —
  translate and send; the only wait is the local socket round-trip for the `Ack`.
- **Prim identity & decomposition**: the `SdfPath` string *is* the cenote object name.
  One mesh prim's translation emits a `MeshPatch` (geometry) **and** an `InstancePatch`
  (transform, mesh-ref = its own path, material-ref = the bound material's path,
  `camera_visible` from visibility); one material prim's translation emits a
  `MaterialPatch` named by its own path. Hydra already separates materials into their own
  prims with their own paths — exactly cenote's separate `Material` objects.
- **Color space**: the server converts the beauty from `ACEScg` to linear `Rec.709` —
  one 3×3, `rec709_from_acescg()`, hoisted out of the pixel loop — before every shm write;
  usdview's default sRGB mode then supplies only the transfer curve. The delegate never
  learns a color space: pixels stay "just bytes." Relying on usdview's OCIO mode was
  rejected — it requires `$OCIO` set, a menu switch, *and* an explicit input-space pick
  (the stock ACES configs' `default` role is AP0, so zero-config OCIO misreads ACEScg
  anyway). Depth crosses unconverted.
- **Subdivision & unbound meshes**: USD meshes default to `subdivisionScheme =
  catmullClark`; the delegate triangulates the base cage via
  `HdMeshUtil::ComputeTriangleIndices` (built from the topology schema's arrays) — the
  accepted floor at usdview's default
  complexity (refineLevel 0, where hdEmbree does the same), with refinement deferred
  (deferrals.md). Meshes with no bound material shade their `displayColor` (constant or
  vertex primvar) through the default closure's base color.
- **Schema reads**: transforms arrive world-flattened under usdview
  (`UsdImagingCreateSceneIndices` inserts the flattening scene index before the terminal);
  `displayColor` inheritance is likewise pre-resolved. The delegate registers the trimmed
  hdPrman convenience stack via `HdSceneIndexPluginRegistry` — `implicitSurface` (USD
  sphere/cube/… prims become meshes for free), `extComputationPrimvarPruning` (skinning
  becomes plain primvars), `nodeIdentifierResolving`, `dependencyForwarding`. Primvars are
  read pre-flattened (`primvarValue`); the indexed form is a later optimization.
- **Camera**: the active camera arrives through `HdRenderPassState` — which camera is
  active is a task parameter, not scene data — and travels on the wire's `SetCamera` fast
  lane, never in a `ChangeSet`: `Session`'s inputs-lane camera overwrites the scene camera
  at every wave, so a `CameraPatch` would be silently dead. The delegate decomposes the
  view transform into `position`/`look_at`/`up` and the projection into `vfov_degrees`;
  `focusDistance`/`fStop`/`focalLength` → `focus_distance`/`aperture_radius`, focal length
  and apertures arriving pre-scaled to scene units (UsdImaging applies the tenth-unit
  factor). Mechanical.
- **AOV scope**: beauty + first-hit depth (both already computed by the film).
  `primId`/`instanceId` — and therefore usdview click-to-select in the render viewport —
  are deferred (deferrals.md), the one asserted default with a visible consequence. With
  zero Rprims the render index's own primId→path table is empty, so the revival also
  builds the delegate's own (noted in the deferral).
- **Refresh cadence**: the server throttles frame pushes to a Max-FPS-style setting
  (sensible default, e.g. ~30) so it never floods the shm double-buffer swap; the delegate
  polls `IsConverged()` and repaints while unconverged, adding samples each pass — Hydra's
  standard progressive-refinement loop.
- **Determinism & goldens**: the estimator is untouched — CPU vs the display path is
  transport only — so the existing FLIP goldens and the ReSTIR bias/convergence gates stay
  valid unchanged. A *new* end-to-end "delegate renders a stage in usdview" check runs on
  the GPU machine, not in CI (usdview needs a GPU + display).

## 3. Layout additions

```
crates/
├── cenote/
│   └── src/render/            # Session already carries the edit channel + frame publish;
│                              # the server wraps it — nothing here changes shape
├── cenote-wire/              # NEW crate: explicit wire types (a full mirror of ChangeSet),
│   ├── src/                   #   MessagePack ser/de, the framed protocol — deps are serde +
│   └── tests/                #   rmp-serde, nothing else; Rust half of the byte-exact guard
└── cenote-server/            # NEW binary: loopback-TCP listener, request/response loop,
    └── src/                   #   wire→Op translation (exhaustive destructuring — a new field
                               #   is a compile error), double-buffered named-shm framebuffer;
                               #   owns a render::Session, applies ChangeSets at waves

hydra/                        # NEW C++ CMake tree, OUTSIDE the Cargo workspace
├── plugInfo.json             #   registers the plugin; displayName = usdview menu label
├── renderDelegate/           #   the adapter ring: delegate shell (zero Rprims), render pass,
│                             #   render buffer, bootstrap glue in its one thin file
├── observer/                 #   HdSceneIndexObserver + per-schema translators → wire ops
├── wire/                     #   C++ mirror of cenote-wire's types + msgpack encoder;
│                             #   a tiny USD-free encoder exe is the C++ half of the guard
└── transport/               #   TCP client + token handshake + cenote-server child spawn
```

Files earn existence (D-014); this is the expected shape, not a quota. The **drift
guard** spans `hydra/wire/` (a standalone C++ encoder, no USD) and `cenote-wire/tests/`,
with Rust as the authority: the Rust corpus builder generates the golden bytes (checked
in, regenerated `UPDATE_GOLDENS`-style — consciously, in the same commit that changes the
wire); the C++ encoder must reproduce them **byte-for-byte**; and a Rust test decodes the
goldens and asserts the values, so the goldens themselves cannot rot. It is the
compiler-substitute for cross-language wire agreement (D-100) and runs in CI without USD
or a GPU.

## 4. Build order (~10–14 weeks at 10 h/wk)

Larger than M3's 8–10: M4 stands up a second build system (C++/CMake against USD), a
cross-language wire with its own drift guard, and a process boundary — none of which M3
had. §5 lists what slips first. The transport lands *before* any Hydra code, so the
fragile cross-language seam is proven in isolation; the delegate then grows shell →
observer → schema by schema, in dependency order.

Each step ends green: Rust compiles + clippy-clean (including `--features denoise`) +
tests pass serially on the GPU machine; the C++ side compiles + its own lint; committed.

0. **Plan docs + transport spine** — this file, the deferrals.md entries, decisions.md
   D-097…D-104, the README status row; then the spine, whose shape a dedicated interview
   locked (2026-07-14). *Checkpoint: a Rust-only integration test spawns the
   `cenote-server` binary, drives it over TCP with a `ChangeSet`, and reads a correct
   frame out of shm — including a saturated primary chosen to fail if the 3×3 is ever
   dropped. The whole transport proven with no Hydra in sight; the riskiest seam, closed
   first. One commit, green.* The locked detail:
   - *`cenote-wire`*: deps `serde` + `rmp-serde` only — never the renderer. The wire
     types are a **full 1:1 mirror** of `Op` and its seven patches, `MeshSource::Ply`
     included — the wire's contract is exactly "a serialized `ChangeSet`," total and
     Hydra-agnostic — plus the request/response envelope and framing (u32-LE length
     prefix + MessagePack, `rmp-serde` defaults: positional struct arrays). The
     wire→`Op` translation lives in `cenote-server`, the one place both worlds are in
     scope, and exhaustively destructures every wire struct so a field added on either
     side is a compile error.
   - *Protocol*: strict request/response — every client message gets exactly one reply
     and the server never speaks unprompted, so the C++ client needs no reader thread.
     Vocabulary: `Hello{protocol, token}` → `Welcome{protocol, fb}`;
     `Replace(ChangeSet)` (genesis and stage-reload) and `Apply(ChangeSet)` → `Ack`;
     `SetCamera{…}` → `Ack` (the inputs-lane fast path — mandatory, see the camera
     leaf); `Resize{w, h}` → the new `FbDesc`, the server allocating the segment and
     calling `Session::resize` before replying; `Ping` → `Ack`. `Session::apply` is
     fire-and-forget, so an `Ack` is a receipt, not a validation: every `Ack` piggybacks
     whatever `take_edit_error` accumulated since the last exchange, and the shm
     header's monotonic `rejected_edits` counter tells an idle delegate to `Ping` for
     the strings. No shutdown message — the server is spawned per-delegate, so EOF *is*
     shutdown.
   - *Framebuffer*: POSIX named shm (`shm_open` + `mmap` — the one deliberately
     platform-specific piece, ~20 lines), name = pid + generation, carried in-band in
     `FbDesc`; the previous segment is unlinked when the client's next request proves
     the reply was processed. Layout: one 4 KiB header page (magic + layout version,
     dims, plane offsets, `front_index`, `frame_counter`, `samples`, `converged`,
     `rejected_edits`) + two buffers, each beauty RGBA f32 (exactly what
     `Context::download_buffer` yields) + depth f32. Tear protocol: the writer fills the
     back buffer, publishes the index, increments the counter; a reader's copy is valid
     iff the counter advanced ≤ 1 across it. No locks, no futexes — the C++ side can
     never block the render.
   - *Server binary*: binds `127.0.0.1:0`, prints exactly one stdout line —
     `cenote-server port=<N>` (`token=<hex>` appended when `CENOTE_SERVER_TOKEN` is
     unset and self-generated); logs go to stderr. One client; EOF → exit 0;
     render-thread fault (`Session`'s fault surface) → stderr + nonzero, so the
     delegate's dead-socket path handles both deaths identically. The `Session` (empty
     scene, default settings) is created *before* the port line prints, so a GPU failure
     fails the spawn legibly. Frame loop: `take_frame` at the throttle cadence →
     `download_buffer` → the 3×3 applied during the one copy into shm.
   - *Drift guard*: byte-exact, Rust as authority (§3). Corpus: every `Op` variant,
     every patch field `Some`, both `MeshSource` variants, the doubly-optional fields in
     all three states, a `Remove`, an empty set, unicode in names and paths.
1. **Delegate shell + observer + first triangles** — the `hydra/` CMake tree against USD
   26.05; `HdRendererPlugin` + `plugInfo.json` (cenote appears in usdview's Renderer
   menu; bootstrap glue in its one thin file); the delegate shell — supported types:
   **no Rprims**, `camera` Sprim (stock `HdCamera`), `renderBuffer` Bprim;
   `SetTerminalSceneIndex` → notice batching → the prim-managing observer, flushed in
   `Update()`; the C++ wire mirror + transport client + token + `cenote-server` spawn;
   mesh translation (points/topology/normals/st from the mesh and primvars schemas; base
   cage triangulated via `HdMeshUtil`; unbound meshes shade from `displayColor`) →
   `MeshPatch` + `InstancePatch`; genesis as `Replace`. *Checkpoint: an untextured mesh
   renders lit in usdview under a real camera and a real distant light, through cenote —
   zero Rprims instantiated. Lands as two commits, split at the USD boundary.* The
   locked detail (a second structured interview, 2026-07-14; the genuinely new decisions
   are D-105…D-109 in [decisions.md](decisions.md)):
   - *Provenance & toolchain*: stock USD 26.05 built once from source (`build_usd.py`,
     usdview on, extras off) into a read-only prefix; the exact invocation recorded in
     `hydra/README.md`. The C++ baseline is **C++23**, extensions off,
     `-Wall -Wextra -Werror`, under a two-part rule (D-105): portable core C++23 only —
     no modules, no coroutines — and *inside the plugin `.so`* no library facilities
     that demand new libstdc++ runtime symbols (`std::println`, `<stacktrace>`), because
     a host like Houdini launches with its bundled older libstdc++ on
     `LD_LIBRARY_PATH` and the dlopen would fail. `std::println` lives in the USD-free
     tools; plugin logging is `TF_*` by convention anyway.
   - *The wire, C++ half* (D-106): a **hand-rolled** minimal msgpack codec — a writer
     plus a small response reader, zero dependencies — and **mirror structs** 1:1 with
     `cenote-wire`'s types, keeping Rust's exact type and field names
     (`std::optional` for `Option`, `std::variant` for the enums), one `encode()` per
     type in Rust field-declaration order, designated-initializer construction
     throughout (C++ requires declaration order there, so every construction site is a
     small field-order check). Decode stays asymmetric and tiny: the client only ever
     reads `Welcome`/`Ack`/`Resized`, hand-decoded, strict. Documented limit: C++ has
     no exhaustive-destructuring trick, so the corpus's "every field `Some`" rule is
     the only guard on a forgotten encode line — load-bearing on every wire change.
   - *Drift guard & CI*: `hydra/wire/` is standalone-buildable (no USD anywhere; CI
     configures only this leaf). The corpus exe builds the same 12 cases the Rust
     corpus defines, encodes through the production `encode()` path, and asserts
     **symmetric set equality plus byte equality** — an unmirrored Rust case fails, a
     caseless C++ entry fails, and a mismatch prints the case, offset, and a hex window
     at the first divergence; registered with ctest. CI gains a pinned `g++-14`
     (C++23 `<print>` needs GCC 14's libstdc++, one past the runner default; apt
     carries exactly one g++-14 per Ubuntu release, so the package name is the pin),
     the wire configure/build/ctest, and `clang-format --dry-run -Werror` over
     `hydra/` (pinned to the local install's exact release via the PyPI wheel).
   - *CMake & discovery*: three targets — the static wire library, the corpus exe, and
     `hdCenote` as a **single `MODULE` library** holding everything else (one `.so` is
     the Hydra-plugin norm; the subdirectories stay source organization, never separate
     shared libraries). `find_package(pxr REQUIRED CONFIG)` against the 26.05 prefix —
     exactly the knob the step-6 HDK pivot swaps; nothing else is `find_package`'d.
     Install to gitignored `hydra/dist/hdCenote/` with a `configure_file`'d
     `plugInfo.json`; usdview finds it via `PXR_PLUGINPATH_NAME`.
   - *Shell contract*: Rprims **none**; Sprims `camera` only, as stock `HdCamera`;
     Bprims `renderBuffer` only — f32 RGBA color + f32 depth, exactly the shm planes,
     so `Map()` is a copy with no conversion. `GetRenderParam` → null (zero Rprims
     means nothing to thread), a plain resource registry, no render-settings
     descriptors yet; capability flags stay at defaults until a step needs them. Naming
     is two-dialect on purpose: the adapter ring follows USD plugin conventions, the
     wire structs keep Rust's names so a C++ struct literal reads token-for-token like
     the Rust one beside the goldens.
   - *Spawn & lifecycle*: `posix_spawn` at delegate construction — never `fork()` from
     a large threaded host. Binary lookup `$CENOTE_SERVER` → beside the plugin `.so` →
     `PATH`, failure naming all three; the token crosses in the child environment,
     never argv (`/proc/*/cmdline` is world-readable); the port line is read with a
     ~30 s deadline (GPU init is the slow part, and step 0 made slowness legible by
     creating the `Session` before the line prints). Any failure → degraded mode —
     alive, rendering nothing, saying why via `TF_WARN` — isolation applying to birth,
     not just death. Teardown: socket EOF (which *is* shutdown), `waitpid` grace, then
     `SIGKILL`. The one deliberately-POSIX file, mirroring the server's `shm.rs`.
   - *The step-1/2 line, redrawn* (D-107): step 1 carries a **skeletal-but-honest pixel
     path** — `Allocate()` → `Resize` → remap (usdview allocates viewport-sized buffers
     against the server's 1280×720 boot default, so this is required for *first*
     pixels), tear-protocol `Map()` from day one, camera → `SetCamera` on change,
     `IsConverged()` = false. Depth folds into step 1 too: usdview's task controller
     requests color+depth by default and the shm depth plane already exists — one
     memcpy. Step 2 keeps its identity as "interactive and unkillable": honest
     convergence, the throttle, resize robustness, rejected-edit surfacing, dead-socket
     degradation.
   - *Observer*: notice-batching scene index + `HdsiPrimManagingSceneIndexObserver` + a
     translator factory — mesh prims get a translator, unknown types a null handler
     (how unknown stays non-fatal forever); the managing observer owns the per-prim
     lifecycle (populate-on-attach, recursive subtree removal) so that bookkeeping is
     deleted, not ported. Translators never send — they append to a pending
     `ChangeSet`; `Update()` drains it: empty → no send, first flush → `Replace`
     (genesis), every later flush → `Apply`. Stage-reload correctness falls out free: a
     reload recreates the delegate, and `Replace` resets the scene.
   - *Mesh translator*: `HdMeshUtil::ComputeTriangleIndices` honoring `orientation`,
     base cage only. cenote's format is single-indexed (attributes per position), so
     vertex-interpolated `normals`/`st` copy through and **faceVarying attributes
     un-weld the mesh to per-corner vertices** — landed in step 3, where textures make
     `st` matter; meshes without faceVarying data keep the welded copy and its memory
     win. Absent normals → omitted, the server derives smooth.
     Unbound meshes synthesize a `<primPath>/displayColor` companion `MaterialPatch`
     (constant color used directly, vertex color approximated by its first element,
     neutral default otherwise) — per-prim companions keep removal trivial. The
     flattened world matrix goes straight into `Transform::Matrix`, never decomposed.
     **Visibility correction** (D-109): invisible → the *instance is removed* (the mesh
     payload stays server-side, so re-showing is a cheap re-add) — the plan's
     "camera_visible from visibility" leaf was wrong on inspection, since cenote's
     `camera_visible=false` is a primary-ray flag and mapping USD invisibility onto it
     would leave ghost shadows.
   - *The lighting hole, fixed* (D-108): step 1 as originally ordered renders a black
     frame — lights were step 4, materials step 3, and the server boots black-sky
     empty. The **distant light is pulled forward**: direction is the prim's world −Z,
     radiance `intensity·2^exposure·color`, the default `angle` collapsed to the delta
     per the locked floor — real step-4 code arriving early, not scaffolding. Step 4
     becomes the *rest* of UsdLux.
   - *Green gate & landing*: the Rust gate unchanged; plus the `-Werror` build, the
     ctest corpus (local and CI), `.clang-format` enforced in CI, and a curated
     `clang-tidy` (bugprone-\*, performance-\*, select readability — USD headers make
     the full firehose unusable) in the local pre-push ritual, since tidy needs USD
     headers CI doesn't have. Checkpoint artifact: `hydra/tests/stages/first-light.usda`
     (a couple of cubes at different depths, one distant light, one camera) rendered by
     a scriptable `usdrecord --renderer Cenote` smoke — success + non-black + the right
     silhouette, the seed that grows into step 6's end-to-end FLIP golden. **Two
     commits, split at the USD boundary**: first the USD-free half (skeleton, codec,
     mirror, corpus, CI, formatting) — fully CI-provable, so the drift guard protects
     `main` before the first line of USD-facing C++ exists — then everything USD, green
     on the GPU machine.
2. **Interactive progressive loop** — hardening the skeletal pixel path step 1 pulled
   in (see its locked detail): `HdRenderBuffer` (CPU, beauty + depth; `Map()`
   reads the shm front buffer under the tear protocol); `HdRenderPass::_Execute` (camera
   + framing from `HdRenderPassState` → the `SetCamera` lane; AOV bindings);
   `IsConverged` honest on **both** the pass and the buffers (usdview checks both), read
   from the shm header; the resize path (`HdRenderBuffer::Allocate` → `Resize` request →
   remap from the reply's `FbDesc`); the refresh throttle; rejected-edit surfacing (the
   header counter → `Ping`); dead-socket detection + graceful degradation.
   *Checkpoint: live progressive refinement, camera nav, and live edits in usdview; kill
   `cenote-server` mid-render and usdview survives, reports the disconnect, and recovers
   on a renderer toggle.* The locked detail (a third structured interview, 2026-07-14;
   the genuinely new decisions are D-113 and D-114 in [decisions.md](decisions.md)):
   - *Scope*: smaller than the sentence above suggests — D-107/D-112 pulled the pixel
     path, resize, and the first cut of honest convergence into step 1, and the refresh
     throttle has been the server pump's ~33 ms cadence since step 0. What remained is
     making the loop *honest and unkillable*: the per-edit epoch, the framing contract,
     dead-server detection, idle rejected-edit surfacing, and the committed checkpoint.
     `noise_threshold` stays unplumbed; every recorded deferral stays deferred.
   - *The epoch* (D-113): session-owned truth — an atomic counter bumped by exactly the
     four wire verbs; the render thread stamps each published frame with what it has
     incorporated (rejected edits included, so they cannot wedge convergence); a parked
     republish closes the visually-inert-edit case, turning the stamp into a delivery
     guarantee. `Ack`/`Resized` and the shm header carry it; the client's `converged()`
     becomes "the front frame's epoch has reached the last acked *and* the header says
     settled" — retiring the ≤~33 ms stale-flag caveat D-112 recorded, and giving the
     viewer a correct edit-vs-converged story as a side effect.
   - *Framing* (D-114): vfov read off the *conformed* `GetProjectionMatrix()` — the
     same matrix the depth remap already reads (D-110), so the whole camera contract
     shares one projection; exotic framing (a data window apart from the display
     window, non-square pixels) warns once and renders full-frame.
   - *Liveness*: a zero-timeout socket `poll()` per `_Execute` — strict
     request/response makes anything readable outside a call a death signal — plus a
     ~30 s receive timeout so a wedged-alive server degrades instead of hanging the
     host; one warning naming the D-099 recovery gesture. No new ledger entry: a
     consequence of D-099/D-100.
   - *Rejections*: the header's rejected-edit counter compared each `_Execute`; when it
     moves, one `Ping` — the existing acked path surfaces the strings and refreshes the
     epoch for free.
   - *Checkpoint, committed twice*: the epoch contract lands in the server integration
     test, and `hydra/tests/interactive_test.py` — edit honesty (a real edit drops
     convergence and returns with changed pixels; a visual no-op still drops and
     returns, the republish end to end), kill-survive (SIGKILL mid-render: the app
     survives, degraded reads converged, the warning posts), toggle-recover (the
     warning's own gesture: renderer away and back, a fresh server, the silhouette
     restored) — joins the pre-push ritual as its fifth command.
   - *Landing*: two commits split at the core/wire seam — the Rust-only session epoch
     first, then wire + server + delegate + test + docs together, green on the GPU
     machine.
3. **Materials & textures** — the material prim's network schema read as data sources
   (nodes / parameters / `inputConnections` / terminals; the universal empty-token
   render context read explicitly) → the UsdPreviewSurface node switch →
   `MaterialPatch`, the four documented exceptions handled as decided;
   `UsdUVTexture`/`UsdPrimvarReader_*` → texture refs through the existing bindless
   path; bindings from the pre-resolved, inherited bindings schema. *Checkpoint:
   textured UsdPreviewSurface assets render matching their authored look.*
4. **Lights & environment** — the *rest* of UsdLux (the distant light moved to step 1,
   D-108): the light prims over the six UsdLux tokens, params read
   lazily by name from the light container → delta / environment /
   synthesized emissive-mesh area lights; `intensity·2^exposure·color`, `normalize` (area
   division), `enableColorTemperature` (blackbody), `treatAsPoint`; rect/disk wound
   one-sided (−Z); distant `angle` collapsed to the delta; dome → the equirect environment
   (`latlong` format). *Checkpoint: a real lit, textured USD
   stage renders recognizably — the milestone's core artifact.*
5. **Instancing** — the instancer topology schema + `instancedBy` → the array-instancer
   op (D-073, picked up here); per-instance transforms composed from the
   `hydra:instanceTranslations/Rotations/Scales` primvars *and* the aggregated
   `hydra:instanceTransforms` matrix form (native instancing emits the latter);
   prototype-root-relative transforms composed with the instancer's world transform.
   *Checkpoint: a point-instanced stage renders.*
6. **Houdini-ready hardening + validation** — the build parameterized for stock-vs-HDK
   USD, *compiled against the HDK* to prove the pivot (the HDK's USD may trail the 26.05
   pin — the observer mechanism holds back to 23.11, so the guards are schema-semantic,
   not architectural); packaging (`cenote-server` beside
   the plugin, discovery via plugInfo/env); the corpus-test CI gate green; the end-to-end
   usdview render golden on the GPU machine; module headers and docs current. *Checkpoint:
   the same delegate source compiles against the HDK's USD. M4 done.*

## 5. Fallback seams (pre-agreed, in slip order)

- **HDK compile proof (step 6)** → the Houdini-ready *rule* is enforced by design
  throughout, so the pivot is documented even if not proven-in-CI this milestone. First to
  go; costs no correctness.
- **Instancing (step 5)** → single-instance only; the array op is the trim, deferred to
  its D-073 trigger (landscape-class scenes). Most lookdev assets are not instance-heavy.
- **Depth AOV (step 2)** → beauty only; depth is cheap but droppable if the schedule
  bites, and it takes nothing else with it.
- **The zero-Rprim shape itself** → not a schedule seam but a correctness one: if the
  all-observer delegate hits an unforeseen wall inside usdview, the pre-agreed fallback
  is a classic Sync delegate *on the same wire* — server, transport, and patches
  unchanged, only the C++ consumption ring swaps. The research says this won't fire
  (every shell requirement is verified against source); it is listed so the milestone is
  never hostage to the forefront bet.
- **Steps 0, 1, 2, and 4 are never compressed** — the transport spine, first pixels, the
  interactive loop, and a lit stage *are* the milestone.

## 6. Risk watch

The fragile axes here are not the estimator (untouched — transport can't perturb the
film, so the M3 bias/correlation risks do not recur and the existing goldens stay a free
regression gate) but the **seams**:

- **The cross-language wire**, concentrated in the 20-field material patch: a field added
  in Rust and forgotten in the C++ encoder is a *silent wrong material*, not a crash.
  Defence: the USD-free corpus drift guard runs from **step 0**, before any USD is in the
  build, and fails loudly on drift — the compiler-substitute D-100 chose in place of
  gRPC's codegen.
- **The silent gamut error**: `ACEScg` pixels through usdview's transfer-curve-only sRGB
  mode render oversaturated and hue-shifted — no crash, and the wire drift guard cannot
  see it. Defence: the 3×3 lives server-side from step 0, and the step-6 end-to-end
  usdview golden includes a saturated primary chosen to fail loudly if the conversion is
  ever dropped.
- **ABI + churn**: the delegate is a shared library loaded into the host's process, so
  USD version, TBB (hosts follow VFX Platform CY2025's oneTBB while USD itself still
  defaults to classic TBB — the mismatch is host-vs-plugin, not a USD release),
  `_GLIBCXX_USE_CXX11_ABI` (the VFX Platform CY2023 ecosystem switch; USD follows the
  compiler default), and Python minor must match the host *exactly*. And the scene-index
  APIs still move: ~one code-touching break per release, concentrated in the **material
  schemas** (25.11 interface rework, 26.03 fallback removal), the **render-settings
  schemas** (renamed as late as 26.05), and the **renderer-plugin bootstrap** — whose
  next break already sits on dev for 26.08. Defence: pin 26.05; isolate the bootstrap
  glue in one thin file; use the stable container-schema aliases (they survived 25.11's
  template renames); lean hardest on the contract verified frozen — the observer
  interface, the prim-managing observer, the geometry schemas — and prove the HDK build
  in step 6, keeping C++ deps minimal (D-100/D-104) to shrink the surface deliberately.
- **Process-boundary liveness**: a dead socket must never wedge or crash the delegate, or
  isolation is hollow — the host survives the crash but the next `Sync` faults and
  "restart from Houdini" becomes "restart Houdini." Defence: dead-socket detection +
  graceful degradation lands *with* the interactive loop (step 2), and recovery is
  exercised by killing the server in that step's checkpoint.
- **Threading**: retired as a risk class by the pivot — notices batch in the scene index
  and `Update()` flushes them serially before task execution, so the translator is
  single-threaded by construction. The residual care is only that `Update()` stays
  cheap: translate and send, never blocking past the local `Ack` round-trip.
- **The forefront bet**: no shipping renderer is all-observer — hdPrman's experimental
  mode converts only spheres as of dev — so the reference material is source code, not
  documentation. The unknowns were burned down by source-level verification before any
  code (zero-Rprim legality, the shell's exact obligations, every patch field's schema
  locator), the blast radius is confined by the process boundary to the thin C++ ring,
  and the classic-Sync fallback on the same wire is pre-agreed (§5).

## 7. Definition of done

- cenote appears in usdview's Renderer menu (via `plugInfo` `displayName`) and renders a
  real USD stage — scene consumption entirely through the terminal scene index observer,
  zero Rprims instantiated.
- Open a lit, textured, UsdPreviewSurface stage in usdview: it renders recognizably and
  in correct color (the server-side `Rec.709` conversion covered by the end-to-end
  golden), refines progressively, and updates live on camera moves and edits (stop →
  apply → restart through `Session`).
- Kill `cenote-server` mid-render: usdview survives, reports the disconnect, and recovers
  on a renderer toggle / stage reload.
- The cross-language corpus drift guard is green in CI (no USD, no GPU); the full delegate
  compiles against stock USD; an end-to-end usdview render golden passes on the GPU
  machine.
- The delegate source compiles against the HDK's USD — the Houdini pivot proven — even
  though Houdini integration itself is a later milestone.
- A stranger can read the three components (`cenote-wire`, `cenote-server`, `hydra/`) and
  the wire types + corpus test to see exactly what crosses the boundary, and read
  [deferrals.md](deferrals.md) to know what M4 consciously left for later — primId
  selection, the GPU-shared framebuffer, native analytic area lights, `open_pbr_surface`
  recognition, automatic crash recovery, the native `HdRenderer`, Windows hosts, and
  Houdini integration — and when each returns.
