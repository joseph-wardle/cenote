# Cenote

A portable, GPU-first, interactive-progressive production path tracer built on
Vulkan ray tracing. One estimator, all the way down: what the artist sees at one
second is an honest prediction of the frame at one hour.[^estimator]

Where CPU production renderers optimize for memory capacity on unbounded scenes,
Cenote makes the inverse bet: extreme single-GPU performance on scenes that fit in
VRAM.

**Status: M7 in progress** — the wavefront engine
(indirect dispatch, zero mid-frame readbacks), Sobol-Burley sampling, the
full `OpenPBR` closure — coat, fuzz, rough glass with interior absorption,
thin-walled surfaces, variable IOR, fractional opacity — energy-compensated
against baked tables so a white furnace closes through every lobe, textured
materials through a bindless table (BC-compressed at prep with a DDS cache
beside each source, sRGB decoded in hardware, converted to the working
space in-shader) including tangent-space normal maps and per-texel opacity,
MIS-weighted next-event estimation of emissive meshes, delta lights, and an
importance-sampled HDRI, thin-lens depth of field, live-editable scene
files, a pbrt-v4 importer (`cenote-cli import`, or open a `.pbrt` in the
viewer directly) with a CC0 regression corpus rendered and FLIP-compared in
CI, AOVs (denoiser albedo/normal guides with Cycles-style specular
pass-through, first-hit depth) accumulated beside the beauty and written as
one multi-layer EXR, OIDN denoising over those guides (a CLI flag and a
viewer toggle, in builds with the `denoise` feature), a progressive viewer,
and a batch CLI that writes exactly the image the viewer converges to.

**M4 shipped the pipeline story**
([docs/m4-plan.md](docs/m4-plan.md)): `cenote-wire` (the explicit wire mirror
of the change-set schema), `cenote-server` (loopback TCP around
`render::Session`, pixels through a lock-free shared-memory framebuffer,
converted to `Rec.709` server-side), the byte-exact cross-language drift
guard, and `hdCenote` — a scene-index-native Hydra render delegate rendering
live inside `usdview` and batch through `usdrecord` and `husk` (see
[hydra/README.md](hydra/README.md)). M5 — geometry depth: subdivision, hair —
is deferred.

M3 and M6 built a second estimator on top of this one: **ReSTIR**, screen-space
reservoir resampling grown from a light sample (DI) to a whole light path (PT),
with reconnection and replay shifts and a ReSTCV control variate. It was
unbiased, gated, and genuinely better per *sample* — and, measured at equal wall
clock, 2.6–4.8× **less** efficient than the path tracer it sat on, because a
reservoir sample cost 5–6× what it saved (D-156). It has been removed from this
branch to keep the renderer lean; the complete implementation, its validation
harness, and the measurements that retired it live on the
[`restir-archive`](../../tree/restir-archive) branch.

![A 5×5 grid of terracotta spheres resting on a glossy gray floor — roughness increasing left to right, metalness back to front — under a blue sky](docs/demo.png)

*The M1 demo: a material chart sweeping `OpenPBR` roughness (left to right)
and metalness (back to front), path traced under the Kloofendal sky's sun
and a warm quad key light. The spheres are coarse meshes shaded smooth by
interpolated vertex normals — the mirror-sharp front row is where a
shading-normal or energy bug would show first.*

![Four crops of the same render at 1, 8, 64, and 512 samples per pixel, the noise resolving away left to right](docs/convergence.png)

*1, 8, 64, and 512 spp*

## Next to pbrt-v4

The importer's real test is a side-by-side. Below are the three
regression-corpus scenes (Benedikt Bitterli's, CC0) imported and rendered
by cenote, beside [pbrt-v4](https://github.com/mmp/pbrt-v4) rendering the
same source files. Both images take the same sRGB display transform.

![Cornell box, Veach MIS plates, and a glass teapot, each rendered by cenote and by pbrt-v4 at 64 samples per pixel](docs/pbrt-equal-samples.png)

*64 samples per pixel.  Cenote's `OpenPBR` conductor is not pbrt's spectral one, so
the Veach plates catch the light a little differently. And cenote has no volumetric 
medium yet, so the teapot's tea looks like plain glass. pbrt reconstructs with a 
triangle filter where cenote uses a box, so at matched samples its per-pixel noise 
sits slightly lower.*

![The same three scenes, each given about a quarter second of rendering: cenote resolves cleaner where pbrt-v4 still carries visible noise](docs/pbrt-equal-time.png)

*Equal time — about a quarter second of rendering each, both on one RTX 4070
Ti SUPER. In that quarter second cenote draws three to four times the
samples — 46, 96, and 31 spp against pbrt's 12, 33, and 11 — and its grain is
plainly the finer. The gap is throughput, not hardware: same GPU, same
scene, same seconds.*

| Scene | Resolution | pbrt-v4 | cenote | per sample |
|---|---|---|---|---|
| `cornell-box` | 1024² | 21.7 ms/spp | 5.5 ms/spp | 3.9× |
| `veach-mis` | 1280×720 | 7.6 ms/spp | 2.6 ms/spp | 2.9× |
| `teapot-full` | 1280×720 | 23.0 ms/spp | 8.0 ms/spp | 2.9× |

*Steady-state cost per sample, both engines on one NVIDIA RTX 4070 Ti SUPER
(pbrt-v4 through its OptiX wavefront back end).*

pbrt renders spectrally and writes linear `Rec.709`; cenote renders RGB in
`ACEScg`.The reference is pbrt-v4 at
[`5f7a606`](https://github.com/mmp/pbrt-v4/commit/5f7a606806a4ac7b939131ded9d7a30ebd02416e).

### The research corpus

Beyond the three CI scenes, `scenes/corpus/` carries eighteen of the scenes
the rendering literature actually measures on — Veach's ajar door and MIS
plates, the Bistro, San Miguel, Zero-Day, the Bitterli bathroom and kitchen,
Kroken, Watercolor — each imported once from its official pbrt-v4 source and
then curated into a first-class cenote `.ron` whose header tells the scene's
provenance, licence, and every knowing degradation with the feature that
unlocks it. Each landed only after matching a side-by-side pbrt-v4 reference;
the divergences that survived became the renderer's documented gap list. Two
of them were bugs the eyeball caught, fixed rather than curated around — a
mirrored-instance emitter lighting from the wrong side, and a cross-kind
texture-name collision the importer mis-bound. The full story is
[scenes/corpus/README.md](scenes/corpus/README.md); the campaign log, with
every rung's landing number, is [docs/corpus-plan.md](docs/corpus-plan.md).

## Quickstart

Requires: stable Rust, [`slangc`](https://github.com/shader-slang/slang) on PATH
(CI pins 2026.9.1; any recent release should work), and a Vulkan GPU with
`VK_KHR_ray_query` support (any recent RT-capable card).

```sh
cargo run --release -p cenote-viewer   # orbit (drag), dolly (scroll), live exposure
cargo run --release -p cenote-viewer -- scenes/example.ron   # open a scene file — and edit it live
cargo run --release -p cenote-cli -- render --spp 256 --out shot.exr
cargo run --release -p cenote-cli -- import scene.pbrt --out scene.ron   # pbrt-v4 in, cenote scene out
```

### Statistics

Every `render` ends with a summary — per-kernel GPU time, how much of the
frame was actually inside a dispatch, time-to-first-ray and
time-to-first-pixel, and device memory in four buckets (scene, film,
textures, scratch) — printed to the console and written beside
the image as `shot.stats.ron` so two runs diff as text. The viewer shows the
same numbers live, four lines plus three collapsing sections.

```sh
cargo run --release -p cenote-cli -- render --spp 64 --out shot.exr   # summary + shot.stats.ron
cargo run --release -p cenote-cli -- render --spp 64 --out shot.exr --stats-trace settle.ron
```

GPU timing is split by what it costs to ask. Bracketing a submission at its
two ends to learn *how long* the device was busy benchmarks at nothing
measurable, so every frame carries it. Stamping *inside* the submission to
learn *where* that time went costs ~15 µs a stamp — a real serialization
point — so the per-kernel breakdown is sampled, one frame in 32, and stamps
each run of same-named passes once rather than each pass. Together that puts
the whole instrument at **+1.9%** on the cheapest frames, against +49.9% for
stamping every boundary of every frame. The frame-time distribution is drawn
from the frames carrying no breakdown, so it reports what the renderer costs
when nothing is watching. `--no-gpu-timers` is the other arm of that A/B,
`--no-stats` turns the output off, and `--stats-trace` adds one line per
sample for plotting a settle curve.

### Denoising

Builds with the `denoise` feature add [Open Image
Denoise](https://www.openimagedenoise.org/), fed by the film's albedo and
normal AOV guides:

```sh
cargo run --release -p cenote-cli --features denoise -- render --spp 64 --denoise --out shot.exr
cargo run --release -p cenote-viewer --features denoise    # panel gains a denoise toggle
```

`--denoise` writes a second EXR (`shot.denoised.exr`) beside the raw one —
the estimator's output is never replaced. The viewer's toggle re-denoises
the accumulating frame about once a second.

The feature links the system OpenImageDenoise library. If your install has
a pkg-config file the build finds it alone; otherwise point `OIDN_DIR` at a
directory whose `lib/` contains `libOpenImageDenoise.so` — an extracted
[official release](https://github.com/RenderKit/oidn/releases), or a
symlink to your distro's versioned library (Setting it once in
`~/.cargo/config.toml` covers every build:

```toml
[env]
OIDN_DIR = "/home/you/.local/opt/oidn"
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
UPDATE_GOLDENS=1 cargo test -p cenote --test golden        # the demo scene
UPDATE_GOLDENS=1 cargo test -p cenote-pbrt --test corpus   # the pbrt corpus
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
| `crates/cenote-cli/` | Headless binary: batch renders, pbrt import |
| `crates/cenote-pbrt/` | pbrt-v4 importer library — a client of the core's public scene API |
| `crates/cenote-server/` | Out-of-process render server: loopback-TCP request/response around `render::Session`, shared-memory framebuffer (M4) |
| `crates/cenote-viewer/` | Interactive viewer binary: live render in a window, orbit camera, progressive accumulation, stats/controls overlay, live-editable scene files |
| `crates/cenote-wire/` | The render server's wire: explicit change-set mirror types, MessagePack framing, the shm layout — and the byte-exact cross-language drift guard |
| `hydra/` | The C++ half of M4 — the `hdCenote` scene-index-native Hydra render delegate, its transport client, and the C++ wire mirror with the drift-guard corpus test (see its README) |
| `scenes/` | Hand-written example scene — the scene model in one readable `.ron` file |
| `scenes/corpus/` | The research-scene corpus — the literature's benchmark scenes (Veach, Bistro, San Miguel, Zero-Day, …) as first-class `.ron`s, each header telling its provenance and every knowing degradation (see its README; campaign in `docs/corpus-plan.md`) |
| `tests/scenes/` | The vendored CC0 pbrt corpus wired into CI (see its README for provenance) |
| `docs/charter.md` | Project charter: vision, locked decisions, milestone roadmap |
| `docs/decisions.md` | Append-only log of every design decision and its rationale |
| `docs/m0-plan.md` | The M0 implementation plan |
| `docs/m1-plan.md` | The M1 implementation plan |
| `docs/m2-plan.md` | The M2 implementation plan |
| `docs/m3-plan.md` | The M3 implementation plan |
| `docs/m4-plan.md` | The M4 implementation plan |
| `docs/m6-plan.md` | The M6 implementation plan (ReSTIR PT — full path reuse, since removed; see `restir-archive`) |
| `docs/deferrals.md` | Living ledger of consciously deferred production features and their revival triggers |

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE), at your option.

[^estimator]: Literally one estimator: the preview and the final frame are the
    same path tracer at different sample counts. There is no biased preview
    mode, no final-gather switch, and no reuse to anneal away — the image at one
    second is the running mean of the same independent unbiased samples the
    image at one hour is the mean of.
