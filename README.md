# Cenote

A portable, GPU-first, interactive-progressive production path tracer built on Vulkan
ray tracing, with GRIS/ReSTIR as its theoretical core. The defining thesis: **the
interactive lookdev preview and the converged final frame are the same estimator** —
no biased preview modes, no "final gather" switch. What the artist sees at one second
is an honest prediction of the frame at one hour.

Where CPU production renderers optimize for memory capacity on unbounded scenes,
Cenote makes the inverse bet: extreme single-GPU performance on scenes that fit in
VRAM. Wavefront compute + ray queries, one integrator, everything resident.

**Status: M1 complete** — the six-kernel wavefront engine (indirect dispatch,
zero mid-frame readbacks), Sobol-Burley sampling, `OpenPBR` lobes with GGX
energy compensation, MIS-weighted next-event estimation of quad lights and an
importance-sampled HDRI, a progressive viewer, and a batch CLI that writes
exactly the image the viewer converges to.

![A grid of faceted terracotta spheres — roughness increasing left to right, metalness bottom to top — floating over a glossy gray floor under a blue sky](docs/demo.png)

*The M1 demo: a material chart sweeping `OpenPBR` roughness (left to right)
and metalness (bottom to top), path traced under the Kloofendal sky's sun
and a warm quad key light. Deliberately faceted — a shading-normal or
energy bug would show as a wrong or flat facet.*

![Four crops of the same render at 1, 8, 64, and 512 samples per pixel, the noise resolving away left to right](docs/convergence.png)

*The thesis in one strip: 1, 8, 64, and 512 spp are the same estimator —
noise is the only difference between preview and final.*

## Quickstart

Requires: stable Rust, [`slangc`](https://github.com/shader-slang/slang) on PATH
(CI pins 2026.9.1; any recent release should work), and a Vulkan GPU with
`VK_KHR_ray_query` support (any recent RT-capable card).

```sh
cargo run --release -p cenote-viewer   # orbit (drag), dolly (scroll), live material sliders
cargo run --release -p cenote-cli -- --spp 256 --out shot.exr
```

The viewer accumulates forever and re-converges after every camera move or
material edit. The CLI accumulates `--spp` samples of the same estimator
into the same film and writes the linear `ACEScg` average as an EXR
(chromaticities declared in the header); with `--watch` it re-renders on
every shader edit, recompiling from the source checkout in under a second.

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
| `crates/cenote-viewer/` | Interactive viewer binary: live render in a window, orbit camera, progressive accumulation, stats/controls overlay |
| `docs/charter.md` | Project charter: vision, locked decisions, milestone roadmap |
| `docs/decisions.md` | Append-only log of every design decision and its rationale |
| `docs/m0-plan.md` | The M0 implementation plan |
| `docs/m1-plan.md` | The M1 implementation plan |

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE), at your option.
