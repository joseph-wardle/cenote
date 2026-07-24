# hydra/ — the C++ half of M4

The Hydra 2 render delegate and everything around it that must be C++. Three
trees: `wire/`, a mirror of `cenote-wire`'s types over a hand-rolled
MessagePack codec, held byte-exact against the Rust goldens by the `corpus`
test; `transport/`, the client that spawns `cenote-server` and speaks to it —
spawn, socket, and the shm framebuffer reader, the deliberately-POSIX corner
of the tree; and `hdCenote/`, the scene-index-native render delegate plugin —
the half that needs USD. The plan is
[docs/m4-plan.md](../docs/m4-plan.md) (steps 1 through 6 carry their locked detail);
rationale lives in [docs/decisions.md](../docs/decisions.md) (D-097…D-123).

Baseline: C++23, extensions off, `-Wall -Wextra -Werror`. Two-part portability
rule (D-105): portable core C++23 only, and inside the plugin `.so` no library
facilities that need new libstdc++ runtime symbols — `std::println` and friends
live in the USD-free tools, plugin logging is `TF_*`.

## Building and testing the wire (no USD required)

```sh
cmake -S hydra/wire -B build/wire
cmake --build build/wire --parallel
ctest --test-dir build/wire --output-on-failure
```

This is exactly what CI runs (with its pinned `g++-14`). The full tree adds
the `hdCenote` plugin and therefore needs USD — `find_package(pxr REQUIRED
CONFIG)` finds it through `CMAKE_PREFIX_PATH` (or `pxr_DIR`) pointing at the
prefix below:

```sh
cmake -S hydra -B build/hydra -DCMAKE_PREFIX_PATH=~/opt/usd-26.05
cmake --build build/hydra --parallel
cmake --install build/hydra   # → hydra/dist/hdCenote/ (gitignored)
```

### The HDK flavor — Houdini's own USD (D-122)

The same source builds against the USD baked into Houdini's HDK (25.05 here) — a
second USD, not a second source — through `CENOTE_USD_FLAVOR=hdk`. It enters through
Houdini's own CMake config, because the HDK ships **no `pxrConfig.cmake`**: that
blessed entry point sets the `_GLIBCXX_USE_CXX11_ABI` choice, the `dsolib` RPATH, and
the `pxr_boost` namespace — the three "will the `.so` even load" landmines a hand-rolled
prefix reproduces and gets subtly wrong.

```sh
source /opt/hfs21.0.671/houdini_setup_bash        # puts Houdini's cmake/env on the path
cmake -S hydra -B build/hydra-hdk -DCENOTE_USD_FLAVOR=hdk \
      -DHoudini_DIR=/opt/hfs21.0.671/toolkit/cmake
cmake --build build/hydra-hdk --parallel
cmake --install build/hydra-hdk   # → hydra/dist/ (the hdk .so + UsdRenderers.json at root)
```

`stock` stays the default, so usdview, CI, and the whole existing gate are untouched. The
one release-cycle of 26.05↔25.05 API drift is absorbed in the single
[hdCenote/usdCompat.hpp](hdCenote/usdCompat.hpp) (`IsSupported`'s parameter list and one
`HdMeshUtil` return type), guarded on `PXR_VERSION`; the material reader compiles against
both untouched, its schemas being stable `using` aliases across the 25.11 rework.

## The pre-push ritual

CI proves the wire half on a bare runner; everything that needs USD or a GPU
is proven here instead, before every push. All six commands run from the
repo root:

```sh
cmake --build build/hydra --parallel && cmake --install build/hydra && ctest --test-dir build/hydra --output-on-failure
find hydra \( -name '*.cpp' -o -name '*.hpp' \) -print0 | xargs -0r clang-format --dry-run -Werror
find hydra -name '*.cpp' -print0 | xargs -0 -P"$(nproc)" -n1 clang-tidy -p build/hydra --quiet
python3 hydra/tests/usdrecord_smoke.py
python3 hydra/tests/flip_golden.py
python3 hydra/tests/interactive_test.py
```

Formatting and linting are pinned to the clang 22.1.8 PyPI wheels
(`pip install --user clang-format==22.1.8 clang-tidy==22.1.8`). CI runs the
same clang-format; clang-tidy stays a local ritual because it reads the USD
build's compile database. Its checks are curated in [.clang-tidy](.clang-tidy)
— bugprone + performance + selected readability, warnings as errors, each
opt-out justified in place.

The smoke renders [tests/stages/first-light.usda](tests/stages/first-light.usda)
— two cubes at different depths under one distant light and a real camera —
through `usdrecord --renderer Cenote` and asserts what a human would eyeball:
success, a non-black frame, and the expected silhouette (the warm near cube
left and larger, the cool far cube right and smaller, the background black).
It then renders
[tests/stages/preview-surface.usda](tests/stages/preview-surface.usda) — a
textured UsdPreviewSurface board whose two faces each map the full 0..1 st
range with faceVarying interpolation — and asserts the green/blue checker
alternates on both faces with the texture's transparent corner cells cut out:
the texture path, the alpha source channel, and the faceVarying un-weld,
eyeball-level. Finally it renders
[tests/stages/lit-stage.usda](tests/stages/lit-stage.usda) — the step-4
checkpoint: ground and board under a warm rect, a cool sphere, a cylinder
fill, and a generated dome sky, all three shaped lights inside the frustum —
and asserts the dome fills the background (blue sky above the horizon, its
warm ground glow below it), the warm and cool pools land on opposite sides of
the ground, and no pixel in the frame is dark: with sky behind everything,
the only route to darkness is a light's stand-in geometry turning
camera-visible and showing its black absorber. Last it renders
[tests/stages/instanced-stage.usda](tests/stages/instanced-stage.usda) — the step-5
checkpoint: a PointInstancer of orange bricks with authored orientations and scales,
its middle instance killed through `inactiveIds`; two green native-instance widgets
(the aggregated matrix form); two magenta gems, each a native instance nested inside a
PointInstancer prototype (recursion); all under a dome sky — and asserts each of the
three instancing hues stands as two mirror copies, the killed brick's centre reads as
sky, and no corner is dark. Buckets by colour and left/centre/right only, so a flipped
or mirrored render still passes; pixel equality is the FLIP golden's job (below). It needs
the USD prefix on `PATH`/`PYTHONPATH` (below) and a built `cenote-server`
(`target/{release,debug}`, or `$CENOTE_SERVER`).

The interactive test drives the same stage through `testusdview` and asserts
the loop is honest and unkillable: a real edit drops `IsConverged` and
reconverges with different pixels; a visually inert edit still drops and
returns (the epoch republish, D-113, end to end — a wedged epoch times out
right there); a SIGKILLed `cenote-server` leaves usdview alive, degraded, and
warning about the recovery gesture; and the gesture itself — the renderer
toggled away and back — spawns a fresh server and brings the silhouette home.
It needs everything the smoke needs, plus a display.

Where the colour-bucket smoke reads structure flip/mirror-agnostically, the FLIP
golden pins *pixels*. It renders
[tests/stages/golden-stage.usda](tests/stages/golden-stage.usda) — three matte cubes in
the saturated `Rec.709` primaries — through `usdrecord --renderer Cenote` to a linear EXR
and FLIP-compares it against [tests/golden/golden-stage.exr](tests/golden/golden-stage.exr),
shelling out to the `cenote-flip` binary (`cargo build -p cenote-cli --bin cenote-flip`;
an ~80-line wrapper of the repo's own `nv-flip`, so no second FLIP dependency enters the
Python env — D-122). usdrecord on stock 26.05 is the pixel oracle, never husk: an
Education-licence watermark would poison the golden, and the point is a clean,
licence-free, GPU-testable pin of the server's colour pipeline. The primaries are the
tripwire — they sit at the gamut edge, where the `Rec.709`→`ACEScg`→`Rec.709` round trip
(delegate in, server out) has the least slack, so a dropped or drifted conversion fails
the compare loudly. That round trip is exactly the bug the golden caught on its first
render (D-123). A saturated-primary gamut guard runs first as a readable sanity check;
FLIP absorbs the settled path-tracer noise a byte compare would trip on. Regenerate — and
eyeball in `tev` before committing — when the render legitimately changes:

```sh
UPDATE_GOLDENS=1 python3 hydra/tests/flip_golden.py
```

## Rendering in husk (Houdini)

husk builds its renderer list from `UsdRenderers.json` files at the **top level** of each
`HOUDINI_PATH` entry — not the pxr plugin registry — so the hdk install drops one at the
`hydra/dist/` root. The `.so` itself still loads through `PXR_PLUGINPATH_NAME`, and the
server still comes from `$CENOTE_SERVER`; the JSON's top-level key is the plugin's
registered type id, which is exactly what `--renderer` takes. `husk --list-renderers` then
shows `HdCenoteRendererPlugin (Cenote)` (untagged — i.e. supported). One frame of the
golden stage, headless:

```sh
QT_QPA_PLATFORM=offscreen \
HOUDINI_PATH=$PWD/hydra/dist:& \
PXR_PLUGINPATH_NAME=$PWD/hydra/dist/hdCenote/resources \
CENOTE_SERVER=$PWD/target/release/cenote-server \
husk --renderer HdCenoteRendererPlugin --camera /World/Camera --res 512 512 \
     --frame 1 --frame-count 1 -o out_\$F4.exr \
     --usd-input hydra/tests/stages/golden-stage.usda
```

husk here is a **load-and-run proof** against Houdini's own 25.05 — it drives the same
`cenote-server` the usdview path does and stamps `__delegate: HdCenoteRendererPlugin` on
the EXR — not a second pixel oracle (D-122): the server is byte-identical across hosts, so
pixel truth stays the usdrecord FLIP golden above. If husk is unlicensed the milestone
degrades to the compile-and-link it was upgraded from — the `find_package(Houdini)` build
still links a load-ready `.so` and the usdrecord golden still stands.

## USD 26.05 — the pinned build (record)

The delegate builds against stock OpenUSD 26.05 (D-097), built once from source
into a read-only prefix. Machine prep, one time:

```sh
python3 -m pip install --user PySide6 PyOpenGL   # usdview deps
git clone --branch v26.05 --depth 1 https://github.com/PixarAnimationStudios/OpenUSD ~/opt/src/OpenUSD
```

The build itself (recorded from the real invocation — Python 3.14, GCC 15.2,
CMake 4.3, Ninja):

```sh
python3 ~/opt/src/OpenUSD/build_scripts/build_usd.py \
  --generator Ninja \
  --no-examples --no-tutorials --no-tests \
  --no-materialx --no-openimageio --no-opencolorio --no-embree --no-alembic \
  --onetbb \
  ~/opt/usd-26.05
```

Notes on the flags:

- usdview, python, and usd-imaging are default-on on this tag and stay on.
- OpenImageIO/OpenColorIO/Embree/Alembic are default-off; passed explicitly so
  the invocation states intent rather than leaning on defaults.
- MaterialX is default-**on** on v26.05 — `--no-materialx` is the one flag doing
  real work among the extras (cenote is a fixed-closure renderer, D-102).
- `--onetbb` because legacy TBB 2020 is a known build casualty of newer GCCs.

Environment to use the prefix (delegate builds and usdview alike):

```sh
export PATH=~/opt/usd-26.05/bin:$PATH
export PYTHONPATH=~/opt/usd-26.05/lib/python:$PYTHONPATH
```

## Launching (once the plugin is installed to `hydra/dist/`)

Hydra discovers the plugin through `PXR_PLUGINPATH_NAME` pointing at the
installed `resources/` directory. The two launch commands:

```sh
PXR_PLUGINPATH_NAME=$PWD/hydra/dist/hdCenote/resources usdview stage.usda
PXR_PLUGINPATH_NAME=$PWD/hydra/dist/hdCenote/resources usdrecord --renderer Cenote stage.usda out.png
```

### Wayland desktop traps (this machine)

Storm needs desktop GL ≥ 4.5; two traps on a Wayland session, both fixed by env:

```sh
QT_QPA_PLATFORM=xcb PYOPENGL_PLATFORM=glx usdview <stage>
QT_QPA_PLATFORM=xcb usdrecord <stage> out.png
```

- Qt on the `wayland` platform creates a GLES 3.2 context, which HgiGL rejects
  ("No renderer plugins found"); `QT_QPA_PLATFORM=xcb` gets desktop GL via GLX.
  Applies to usdrecord too — its offscreen context also comes through PySide6.
- PyOpenGL picks its EGL platform whenever `WAYLAND_DISPLAY` is set, even under
  Qt-on-xcb, so the usdview HUD dies with "Attempt to retrieve context when no
  valid context"; `PYOPENGL_PLATFORM=glx` pins it. (usdrecord doesn't use
  PyOpenGL.)
