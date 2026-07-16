# hydra/ — the C++ half of M4

The Hydra 2 render delegate and everything around it that must be C++. Three
trees: `wire/`, a mirror of `cenote-wire`'s types over a hand-rolled
MessagePack codec, held byte-exact against the Rust goldens by the `corpus`
test; `transport/`, the client that spawns `cenote-server` and speaks to it —
spawn, socket, and the shm framebuffer reader, the deliberately-POSIX corner
of the tree; and `hdCenote/`, the scene-index-native render delegate plugin —
the half that needs USD. The plan is
[docs/m4-plan.md](../docs/m4-plan.md) (steps 1 and 2 carry their locked detail);
rationale lives in [docs/decisions.md](../docs/decisions.md) (D-097…D-114).

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

## The pre-push ritual

CI proves the wire half on a bare runner; everything that needs USD or a GPU
is proven here instead, before every push. All five commands run from the
repo root:

```sh
cmake --build build/hydra --parallel && cmake --install build/hydra && ctest --test-dir build/hydra --output-on-failure
find hydra \( -name '*.cpp' -o -name '*.hpp' \) -print0 | xargs -0r clang-format --dry-run -Werror
find hydra -name '*.cpp' -print0 | xargs -0 -P"$(nproc)" -n1 clang-tidy -p build/hydra --quiet
python3 hydra/tests/usdrecord_smoke.py
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
No pixel equality — that is step 6's FLIP golden. It needs the USD prefix on
`PATH`/`PYTHONPATH` (below) and a built `cenote-server`
(`target/{release,debug}`, or `$CENOTE_SERVER`).

The interactive test drives the same stage through `testusdview` and asserts
the loop is honest and unkillable: a real edit drops `IsConverged` and
reconverges with different pixels; a visually inert edit still drops and
returns (the epoch republish, D-113, end to end — a wedged epoch times out
right there); a SIGKILLed `cenote-server` leaves usdview alive, degraded, and
warning about the recovery gesture; and the gesture itself — the renderer
toggled away and back — spawns a fresh server and brings the silhouette home.
It needs everything the smoke needs, plus a display.

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
