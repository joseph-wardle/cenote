#!/usr/bin/env python3
"""The end-to-end FLIP golden (step 6 T4): renders stages/golden-stage.usda
through the Cenote renderer with usdrecord — the full USD → delegate → server
→ EXR path — and FLIP-compares the result against the reference EXR checked in
at tests/golden/golden-stage.exr, shelling out to the `cenote-flip` binary.

usdrecord on stock USD 26.05 is the pixel oracle here, never husk: husk under an
Education licence stamps a watermark that would poison the golden, and the whole
point is a clean, licence-free, GPU-testable pin of the server's colour pipeline.
The golden stage is three saturated Rec.709 primaries, chosen because the primaries
are exactly where the server's Rec.709→ACEScg 3×3 moves a colour the most — so a
build that drops or corrupts that conversion (the one drift the USD-free wire corpus
cannot see) renders the primaries wrong and fails the FLIP compare loudly. A
saturated-primary gamut guard runs first as a human-readable sanity check: each
primary present in its own column, dominant in its own channel, and in-gamut
(non-negative) after the conversion.

FLIP absorbs the path tracer's settled Monte-Carlo noise a byte compare would trip
on. Run serially (the GPU-flake rule). Regenerate — and eyeball in tev before
committing — when the render legitimately changes:

    UPDATE_GOLDENS=1 python3 hydra/tests/flip_golden.py

Run from anywhere with the USD prefix on PATH/PYTHONPATH (README.md's environment)
and the plugin installed to hydra/dist/. The server binary comes from $CENOTE_SERVER
(falling back to target/{release,debug}); the FLIP binary from $CENOTE_FLIP (falling
back to the same), built with `cargo build -p cenote-cli --bin cenote-flip`.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STAGE = REPO / "hydra" / "tests" / "stages" / "golden-stage.usda"
GOLDEN = REPO / "hydra" / "tests" / "golden" / "golden-stage.exr"
RESOURCES = REPO / "hydra" / "dist" / "hdCenote" / "resources"

# The committed golden's width; the height follows the camera aperture. Small on
# purpose — enough pixels to pin the three primaries, small enough to live in the
# repo forever, fast enough for a stable render.
WIDTH = 256

# A generous sample cap for the golden: a flat-lit scene of three matte cubes
# settles fast, and the extra samples keep run-to-run noise well under the FLIP
# threshold so the pin does not flake. usdrecord loops until "converged", which
# arrives at this cap.
GOLDEN_SAMPLES = "256"


def fail(message):
    print(f"flip golden: FAIL — {message}", file=sys.stderr)
    sys.exit(1)


def locate(env_var, name, build_hint):
    """A built binary from $env_var, else target/{release,debug}/name."""
    if os.environ.get(env_var):
        return os.environ[env_var]
    for profile in ("release", "debug"):
        candidate = REPO / "target" / profile / name
        if os.access(candidate, os.X_OK):
            return str(candidate)
    fail(f"no {name}: set ${env_var} or `{build_hint}`")


def record(out_path):
    """Render the golden stage to a linear EXR through usdrecord/Cenote."""
    if not RESOURCES.is_dir():
        fail(f"{RESOURCES} missing — build and `cmake --install build/hydra` first")
    if shutil.which("usdrecord") is None:
        fail("usdrecord not on PATH — export the USD prefix (hydra/README.md)")
    env = os.environ.copy()
    env["PXR_PLUGINPATH_NAME"] = str(RESOURCES)
    env["CENOTE_SERVER"] = locate("CENOTE_SERVER", "cenote-server",
                                  "cargo build -p cenote-server")
    env.setdefault("CENOTE_SERVER_MAX_SAMPLES", GOLDEN_SAMPLES)
    # The Wayland trap (README.md): the offscreen GL context must come through
    # GLX, not a GLES-only Wayland surface.
    env.setdefault("QT_QPA_PLATFORM", "xcb")
    command = [
        "usdrecord",
        "--renderer", "Cenote",
        "--camera", "/World/Camera",
        "--disableCameraLight",
        "--imageWidth", str(WIDTH),
        str(STAGE),
        str(out_path),
    ]
    try:
        done = subprocess.run(command, env=env, timeout=300, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        fail("usdrecord timed out after 300s — convergence never reported?")
    if done.returncode != 0 or not out_path.is_file():
        sys.stderr.write(done.stdout[-2000:] + done.stderr[-2000:])
        fail(f"usdrecord exited {done.returncode} "
             f"(image {'written' if out_path.is_file() else 'missing'})")


def load_linear(path):
    """The EXR's pixels as a (height, width, channels) float array."""
    import OpenImageIO as oiio

    buf = oiio.ImageBuf(str(path))
    if buf.has_error:
        fail(f"{path} did not load as an EXR: {buf.geterror()}")
    pixels = buf.get_pixels(oiio.FLOAT)
    return pixels, buf.spec().width, buf.spec().height


def gamut_guard(exr_path):
    """The saturated primaries survive the server's Rec.709→ACEScg conversion:
    each column reads its own primary, dominant in its own channel and in-gamut
    (non-negative). Column-order-agnostic, so a mirrored frame still passes —
    pixel truth is the FLIP compare's job, this is the readable sanity check."""
    pixels, width, height = load_linear(exr_path)

    def patch_mean(fx, fy):
        cx, cy = int(fx * width), int(fy * height)
        radius = max(2, width // 48)
        region = pixels[cy - radius : cy + radius + 1, cx - radius : cx + radius + 1, :3]
        return region.reshape(-1, 3).mean(axis=0), region.min()

    def primary(rgb):
        r, g, b = rgb
        peak = max(rgb)
        if peak < 0.02:
            return "dark"
        # 1.4×, not a hard primary: the OpenPBR default dielectric specular
        # layer adds a white sheen that legitimately desaturates each cube, so
        # the guard asks for a clear channel lead, not a pure primary.
        if r > 1.4 * g and r > 1.4 * b:
            return "red"
        if g > 1.4 * r and g > 1.4 * b:
            return "green"
        if b > 1.4 * r and b > 1.4 * g:
            return "blue"
        return f"muddy {[round(float(c), 3) for c in rgb]}"

    # The three cubes project to the frame's left, middle, and right thirds
    # (world x = -2.2, 0, +2.2 through the first-light camera), centred
    # vertically.
    found = {}
    for fx in (0.19, 0.5, 0.81):
        mean, floor = patch_mean(fx, 0.5)
        name = primary(mean)
        if name == "dark":
            fail(f"the cube at x-fraction {fx} is unlit — a primary is missing")
        if name not in ("red", "green", "blue"):
            fail(f"the primary at x-fraction {fx} reads {name} — colour is not "
                 f"reaching the render cleanly")
        # A correct Rec.709→ACEScg keeps every in-gamut Rec.709 colour
        # non-negative (ACEScg's gamut contains Rec.709). A negative channel
        # means the conversion is wrong end-to-end, not merely shifted.
        if floor < -0.005:
            fail(f"the {name} cube has a negative channel ({floor:.4f}) — the "
                 f"colour conversion is out of gamut, not just off")
        found[name] = fx

    if set(found) != {"red", "green", "blue"}:
        fail(f"the three primaries are not each present once: found {sorted(found)}")

    # Background stays black — no environment in the stage, so any lift is a leak.
    for fx, fy in ((0.03, 0.03), (0.97, 0.03), (0.03, 0.97), (0.97, 0.97)):
        corner, _ = patch_mean(fx, fy)
        if max(corner) > 0.02:
            fail(f"the background at ({fx}, {fy}) is not black: "
                 f"{[round(float(c), 3) for c in corner]}")

    order = " ".join(name for name, _ in sorted(found.items(), key=lambda kv: kv[1]))
    print(f"flip golden: gamut guard OK — primaries {order} left→right, "
          f"each in-channel and in-gamut, background black")


def main():
    directory = Path(tempfile.mkdtemp(prefix="cenote-flip-golden-"))
    actual = directory / "golden-stage.exr"
    try:
        record(actual)

        if os.environ.get("UPDATE_GOLDENS"):
            gamut_guard(actual)
            GOLDEN.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(actual, GOLDEN)
            print(f"flip golden: updated {GOLDEN} — inspect it in tev before committing")
            shutil.rmtree(directory)
            return

        if not GOLDEN.is_file():
            fail(f"no golden at {GOLDEN} — regenerate with "
                 f"UPDATE_GOLDENS=1 python3 hydra/tests/flip_golden.py")

        gamut_guard(actual)

        heatmap = directory / "golden-stage.flip.exr"
        result = subprocess.run(
            [locate("CENOTE_FLIP", "cenote-flip",
                    "cargo build -p cenote-cli --bin cenote-flip"),
             str(GOLDEN), str(actual), "--heatmap", str(heatmap)],
            capture_output=True, text=True,
        )
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        if result.returncode != 0:
            print(f"flip golden: artifacts kept in {directory}", file=sys.stderr)
            fail("the render differs from the golden — see the FLIP heatmap above")
    except SystemExit:
        print(f"flip golden: artifacts kept in {directory}", file=sys.stderr)
        raise
    shutil.rmtree(directory)
    print("flip golden: OK — the end-to-end render matches the committed golden")


if __name__ == "__main__":
    main()
