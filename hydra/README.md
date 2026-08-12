# hydra/ — the C++ half

The Hydra 2 render delegate and everything around it that must be C++. Three
trees: `wire/`, a mirror of `cenote-wire`'s types over a hand-rolled MessagePack
codec, held byte-exact against the Rust goldens by the `corpus` test;
`transport/`, the client that spawns `cenote-server` and speaks to it — spawn,
socket, and the shm framebuffer reader, the deliberately-POSIX corner of the
tree; and `hdCenote/`, the scene-index-native render delegate plugin — the half
that needs USD.

Baseline: C++23, extensions off, `-Wall -Wextra -Werror`. Two-part portability
rule: portable core C++23 only, and inside the plugin `.so` no library
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

### The HDK flavor — Houdini's own USD

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
`HdMeshUtil` return type), guarded on `PXR_VERSION`.

## The pre-push gate

```sh
scripts/hydra-check.sh   # build, ctest, clang-format, clang-tidy, the four Python tests
```

Formatting and linting are pinned to the clang 22.1.8 PyPI wheels
(`pip install --user clang-format==22.1.8 clang-tidy==22.1.8`). CI runs the same
clang-format; clang-tidy stays local because it reads the USD build's compile
database. The Python tests need the USD prefix on `PATH`/`PYTHONPATH` (below), a
built `cenote-server` (`target/{release,debug}`, or `$CENOTE_SERVER`), and — for
`interactive_test.py` — a display.

`render_settings_test.py` is the one that reads no pixels at all: it names each
`RenderSettings` prim on its stage in turn and asserts what the render *does* —
a budget nothing can finish keeps running, a threshold over it finishes anyway,
a depth past the ceiling is clamped rather than rejected — plus, where the host
lets the delegate speak, the line the delegate posts saying what it resolved.
That is what lets `--host husk` run the same file (see below).

`usdrecord_smoke.py` buckets by colour and left/centre/right only, so a flipped
render still passes; pixel equality is `flip_golden.py`'s job. That golden's
three cubes are the saturated `Rec.709` primaries because they sit at the gamut
edge, where the `Rec.709`→`ACEScg`→`Rec.709` round trip (delegate in, server
out) has the least slack, so a dropped or drifted conversion fails loudly. Its
oracle is usdrecord on stock 26.05, never husk — an Education-licence watermark
would poison the golden. Regenerate it, and eyeball it in `tev`, when the render
legitimately changes:

```sh
cargo build -p cenote-cli --bin cenote-flip   # the comparator the test shells out to
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

husk here is a **load-and-run proof** against Houdini's own 25.05, not a second
pixel oracle: the server is byte-identical across hosts, so pixel truth stays the
usdrecord FLIP golden above. The one behavioural check that does run under it is
the settings one, against a `dist/` installed from the **hdk** build (which
replaces the stock `.so` there, so reinstall the stock one afterwards):

```sh
cmake --install build/hydra-hdk
PATH=$HFS/bin:$PATH python3 hydra/tests/render_settings_test.py --host husk
```

Two husk behaviours matter here. husk populates the delegate's settings map,
with the `cenote:` namespace intact — `husk -s /Render/Foo -V 9` prints the map
it built, and an authored `cenote:samplesPerPixel` governs the render through
it. And Houdini installs a `TfDiagnosticMgr` delegate, so `TF_STATUS`/`TF_WARN`
go to its error manager rather than the stream: under husk nothing the delegate
*says* is readable, and only what the render does can be asserted.

## USD 26.05 — the pinned build

The delegate builds against stock OpenUSD 26.05, built once from source into a
read-only prefix. Machine prep, one time:

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

Only two of those flags do work: MaterialX is default-**on** at v26.05 and
cenote is a fixed-closure renderer, and `--onetbb` because legacy TBB 2020 is a
known build casualty of newer GCCs. usdview, python, and usd-imaging are
default-on and stay on; the rest are default-off, named to state intent.

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
