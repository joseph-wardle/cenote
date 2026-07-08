# Cenote

A portable, GPU-first, interactive-progressive production path tracer built on Vulkan
ray tracing, with GRIS/ReSTIR as its theoretical core. The defining thesis: **the
interactive lookdev preview and the converged final frame are the same estimator** —
no biased preview modes, no "final gather" switch. What the artist sees at one second
is an honest prediction of the frame at one hour.

Where CPU production renderers optimize for memory capacity on unbounded scenes,
Cenote makes the inverse bet: extreme single-GPU performance on scenes that fit in
VRAM. Wavefront compute + ray queries, one integrator, everything resident.

**Status: M0 complete** — device bring-up, acceleration structures, an inline
ray-query kernel, sub-second shader hot reload, and golden-image tests.

![A faceted icosphere resting on a ground plane, shaded as geometric normals — a pastel rainbow sphere over a green floor against a black sky](docs/demo.png)

*The M0 demo image: geometric normals as color, traced with ray queries
against a real two-instance acceleration structure. Deliberately faceted —
a winding or handedness bug would scramble the rainbow.*

## Quickstart

Requires: stable Rust, [`slangc`](https://github.com/shader-slang/slang) on PATH
(CI pins 2026.9.1; any recent release should work), and a Vulkan GPU with
`VK_KHR_ray_query` support (any recent RT-capable card).

```sh
cargo run -p cenote-cli
```

## Tests and goldens

```sh
cargo test --workspace
```

runs everything; tests that need a GPU skip cleanly (with a note on stderr)
where there isn't one. The golden-image tests render the demo scene and
compare it against the reference EXRs in `crates/cenote/tests/golden/` with
[ꟻLIP](https://github.com/NVlabs/flip), a perceptual metric whose threshold
survives the floating-point reordering that driver and compiler updates cause.
A failure dumps the actual render and a FLIP heatmap (black = identical,
bright = different) into `target/tmp/` — open them in `tev` next to the golden.

After an **intentional** image change, regenerate the goldens and eyeball them
before committing:

```sh
UPDATE_GOLDENS=1 cargo test -p cenote --test golden
```

### Pre-push ritual

CI has no GPU, so everything image-shaped runs here, before pushing:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace   # on the GPU machine — includes the goldens
```

## Repo map

| Path | What lives there |
|---|---|
| `crates/cenote/` | The core renderer library — start at `src/lib.rs`, whose crate doc is the architecture map |
| `crates/cenote/shaders/` | Slang GPU kernels — the heart of the renderer |
| `crates/cenote-cli/` | Headless batch renderer binary |
| `docs/charter.md` | Project charter: vision, locked decisions, milestone roadmap |
| `docs/decisions.md` | Append-only log of every design decision and its rationale |
| `docs/m0-plan.md` | The M0 implementation plan |

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE), at your option.
