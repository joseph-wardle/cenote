# hydra/ — the C++ half of M4

The Hydra 2 render delegate and everything around it that must be C++. Today the
tree holds the USD-free half: `wire/`, a mirror of `cenote-wire`'s types over a
hand-rolled MessagePack codec, held byte-exact against the Rust goldens by the
`corpus` test. The `hdCenote` delegate plugin — bootstrap, shell, observer,
translators, transport — lands in the following commit. The plan is
[docs/m4-plan.md](../docs/m4-plan.md) (step 1 carries the locked detail);
rationale lives in [docs/decisions.md](../docs/decisions.md) (D-097…D-109).

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

This is exactly what CI runs (with its pinned `g++-14`). The full tree —
`cmake -S hydra -B build/hydra` — builds the same targets today and gains the
`hdCenote` plugin once USD enters it, found via
`find_package(pxr REQUIRED CONFIG)` with `CMAKE_PREFIX_PATH` (or `pxr_DIR`)
pointing at the USD prefix below.

Formatting is `.clang-format`-enforced, pinned to clang-format 22.1.8 (the same
version CI installs from the PyPI wheel):

```sh
find hydra \( -name '*.cpp' -o -name '*.hpp' \) -print0 | xargs -0r clang-format --dry-run -Werror
```

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
