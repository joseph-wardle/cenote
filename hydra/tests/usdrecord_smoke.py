#!/usr/bin/env python3
"""The usdrecord smoke check: renders stages/first-light.usda through the
Cenote renderer and asserts what a human would eyeball — the run succeeds,
the frame is not black, and the silhouette is the expected one (the warm
near cube left of center and larger, the cool far cube right and smaller,
the background black). No pixel equality; that is step 6's FLIP golden.

Run from anywhere with the USD prefix on PATH/PYTHONPATH (README.md's
environment) and the plugin installed to hydra/dist/. The server binary
comes from $CENOTE_SERVER, falling back to target/{release,debug}; the
sample cap defaults low ($CENOTE_SERVER_MAX_SAMPLES=64) so convergence —
which usdrecord waits for — arrives in seconds.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STAGE = REPO / "hydra" / "tests" / "stages" / "first-light.usda"
RESOURCES = REPO / "hydra" / "dist" / "hdCenote" / "resources"

# 8-bit thresholds: a pixel this bright counts as lit geometry, and the
# background must stay under the dark ceiling.
LIT = 40
DARK = 10


def fail(message):
    print(f"usdrecord smoke: FAIL — {message}", file=sys.stderr)
    sys.exit(1)


def locate_server():
    if os.environ.get("CENOTE_SERVER"):
        return os.environ["CENOTE_SERVER"]
    for profile in ("release", "debug"):
        candidate = REPO / "target" / profile / "cenote-server"
        if os.access(candidate, os.X_OK):
            return str(candidate)
    fail("no cenote-server: set $CENOTE_SERVER or `cargo build -p cenote-server`")


def record(out_path):
    if not RESOURCES.is_dir():
        fail(f"{RESOURCES} missing — build and `cmake --install build/hydra` first")
    if shutil.which("usdrecord") is None:
        fail("usdrecord not on PATH — export the USD prefix (hydra/README.md)")
    env = os.environ.copy()
    env["PXR_PLUGINPATH_NAME"] = str(RESOURCES)
    env["CENOTE_SERVER"] = locate_server()
    # A low cap so "converged" — the flag usdrecord loops on — is minutes
    # of margin away from the 4096 default.
    env.setdefault("CENOTE_SERVER_MAX_SAMPLES", "64")
    # The Wayland trap (README.md): the offscreen GL context must come
    # through GLX, not a GLES-only Wayland surface.
    env.setdefault("QT_QPA_PLATFORM", "xcb")
    command = [
        "usdrecord",
        "--renderer", "Cenote",
        "--camera", "/World/Camera",
        "--disableCameraLight",
        "--imageWidth", "640",
        str(STAGE),
        str(out_path),
    ]
    try:
        done = subprocess.run(command, env=env, timeout=300, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        fail("usdrecord timed out after 300s — convergence never reported?")
    if done.returncode != 0 or not out_path.is_file():
        sys.stderr.write(done.stdout[-2000:] + done.stderr[-2000:])
        fail(f"usdrecord exited {done.returncode} (image {'written' if out_path.is_file() else 'missing'})")


def analyze(out_path):
    from PySide6.QtGui import QImage

    image = QImage(str(out_path))
    if image.isNull():
        fail(f"{out_path} did not load as an image")
    image = image.convertToFormat(QImage.Format.Format_RGB888)
    width, height = image.width(), image.height()
    stride = image.bytesPerLine()
    pixels = image.constBits()

    # One pass: every lit pixel lands in a left-of-center or
    # right-of-center bucket as (count, sum R, sum B); the corners must
    # stay dark.
    buckets = [[0, 0, 0], [0, 0, 0]]
    for y in range(height):
        row = pixels[y * stride : y * stride + width * 3]
        for x in range(width):
            r, g, b = row[x * 3], row[x * 3 + 1], row[x * 3 + 2]
            if max(r, g, b) >= LIT:
                bucket = buckets[0 if x < width // 2 else 1]
                bucket[0] += 1
                bucket[1] += r
                bucket[2] += b
    for cx, cy in ((8, 8), (width - 9, 8), (8, height - 9), (width - 9, height - 9)):
        offset = cy * stride + cx * 3
        if max(pixels[offset : offset + 3]) > DARK:
            fail(f"background at ({cx}, {cy}) is not black")

    (near_count, near_r, near_b), (far_count, far_r, far_b) = buckets
    if near_count + far_count < width * height // 100:
        fail(f"black frame: only {near_count + far_count} lit pixels")
    if near_count == 0 or far_count == 0:
        fail(f"a cube is missing: {near_count} lit left, {far_count} lit right")
    if near_count <= far_count * 3 // 2:
        fail(f"the near cube ({near_count} px) should dwarf the far one ({far_count} px)")
    if near_r <= near_b:
        fail("the near cube reads cold — displayColor is not reaching the render")
    if far_b <= far_r:
        fail("the far cube reads warm — displayColor is not reaching the render")
    print(
        f"usdrecord smoke: OK — {width}x{height}, near cube {near_count} px (warm), "
        f"far cube {far_count} px (cool), background black"
    )


def main():
    directory = Path(tempfile.mkdtemp(prefix="cenote-usdrecord-"))
    out_path = directory / "first-light.png"
    try:
        record(out_path)
        analyze(out_path)
    except SystemExit:
        print(f"artifacts kept in {directory}", file=sys.stderr)
        raise
    shutil.rmtree(directory)


if __name__ == "__main__":
    main()
