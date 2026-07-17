#!/usr/bin/env python3
"""The usdrecord smoke check: renders stages/first-light.usda,
stages/preview-surface.usda, and stages/lit-stage.usda through the
Cenote renderer and asserts what a human would eyeball. For first-light:
the run succeeds, the frame is not black, and the silhouette is the
expected one (the warm near cube left of center and larger, the cool
far cube right and smaller, the background black). For preview-surface:
the textured board shows its checker — green and blue cells alternating
in both directions, the texture's transparent corner cells cut out, the
pattern appearing once per face — so the texture path, the alpha source
channel, and the faceVarying st un-weld are all visibly working. For
lit-stage: the dome fills the background (sky above the horizon, its
warm ground glow below it), the warm rect and cool sphere pool on
opposite sides of the ground, and no pixel in the frame is dark — with
sky behind everything, the only way to darkness is a light's stand-in
geometry turning camera-visible and showing its black absorber. No
pixel equality; that is step 6's FLIP golden.

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
STAGES = REPO / "hydra" / "tests" / "stages"
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


def record(stage, out_path):
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
        str(stage),
        str(out_path),
    ]
    try:
        done = subprocess.run(command, env=env, timeout=300, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        fail(f"usdrecord timed out after 300s on {stage.name} — convergence never reported?")
    if done.returncode != 0 or not out_path.is_file():
        sys.stderr.write(done.stdout[-2000:] + done.stderr[-2000:])
        fail(f"usdrecord exited {done.returncode} on {stage.name} "
             f"(image {'written' if out_path.is_file() else 'missing'})")


def load(out_path):
    from PySide6.QtGui import QImage

    image = QImage(str(out_path))
    if image.isNull():
        fail(f"{out_path} did not load as an image")
    return image.convertToFormat(QImage.Format.Format_RGB888)


def check_background(image):
    width, height = image.width(), image.height()
    for cx, cy in ((8, 8), (width - 9, 8), (8, height - 9), (width - 9, height - 9)):
        pixel = image.pixelColor(cx, cy)
        if max(pixel.red(), pixel.green(), pixel.blue()) > DARK:
            fail(f"background at ({cx}, {cy}) is not black")


def analyze_first_light(out_path):
    image = load(out_path)
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
    check_background(image)

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


def analyze_preview_surface(out_path):
    image = load(out_path)
    width, height = image.width(), image.height()

    # The board fills the frame's middle: camera at z=12 with the default
    # 50mm lens on the 20.955x15.2908mm aperture, so these are the
    # world-space half-extents of the frame at the board's plane.
    half_w = 12 * 20.955 / 2 / 50
    half_h = 12 * 15.2908 / 2 / 50

    def patch(wx, wy):
        """Mean RGB of a small patch at world (wx, wy) on the board."""
        cx = int(width / 2 + wx * (width / 2) / half_w)
        cy = int(height / 2 - wy * (height / 2) / half_h)
        total = [0, 0, 0]
        count = 0
        for y in range(cy - 6, cy + 7):
            for x in range(cx - 6, cx + 7):
                pixel = image.pixelColor(x, y)
                total[0] += pixel.red()
                total[1] += pixel.green()
                total[2] += pixel.blue()
                count += 1
        return [component / count for component in total]

    def classify(color):
        if max(color) < 2 * DARK:
            return "dark"
        if color[1] > 2 * color[0] and color[1] > 2 * color[2] and color[1] > LIT:
            return "green"
        if color[2] > 2 * color[0] and color[2] > 2 * color[1] and color[2] > LIT:
            return "blue"
        return f"unrecognized {[round(c, 1) for c in color]}"

    check_background(image)

    # Each face shows the full 4x4 checker (cells 0.5 world units): the
    # four corner cells cut out by the texture's alpha, the rest one
    # clean tint per checker parity, the parities in opposite tints, and
    # both faces in agreement — the second face's full pattern is the
    # faceVarying un-weld made visible, since its st shares no values
    # with the first across the welded edge. Deliberately orientation-
    # agnostic: parity sets and symmetric cutouts pin the structure and
    # the authored tints, not the handedness — that is step 6's FLIP
    # golden.
    tints = []
    for face, x0 in ((0, -2.0), (1, 0.0)):
        by_parity = {0: set(), 1: set()}
        for i in range(4):
            for j in range(4):
                look = classify(patch(x0 + (i + 0.5) * 0.5, -1 + (j + 0.5) * 0.5))
                if i in (0, 3) and j in (0, 3):
                    if look != "dark":
                        fail(f"face {face} corner cell ({i}, {j}) should be cut out by the "
                             f"texture's alpha, but reads {look}")
                elif look == "dark":
                    fail(f"face {face} cell ({i}, {j}) is unlit — the checker is not covering "
                         f"the face")
                else:
                    by_parity[(i + j) % 2].add(look)
        for parity in (0, 1):
            if len(by_parity[parity]) != 1 or not by_parity[parity] <= {"green", "blue"}:
                fail(f"face {face} parity-{parity} cells are not one clean tint: "
                     f"{sorted(by_parity[parity])}")
        if by_parity[0] == by_parity[1]:
            fail(f"face {face} does not alternate: every cell reads {by_parity[0].pop()}")
        tints.append((by_parity[0].pop(), by_parity[1].pop()))
    if tints[0] != tints[1]:
        fail(f"the faces disagree about the checker: {tints[0]} vs {tints[1]} — the second "
             f"face's st is not surviving the un-weld")
    print(
        f"usdrecord smoke: OK — {width}x{height}, checker {tints[0][0]}/{tints[0][1]} "
        f"alternating on both faces, corner cells cut out, background black"
    )


def analyze_lit_stage(out_path):
    image = load(out_path)
    width, height = image.width(), image.height()
    stride = image.bytesPerLine()
    pixels = image.constBits()

    def patch(fx, fy):
        """Mean RGB of a small patch at frame fractions (fx, fy)."""
        cx, cy = int(fx * width), int(fy * height)
        total = [0, 0, 0]
        count = 0
        for y in range(cy - 5, cy + 6):
            for x in range(cx - 5, cx + 6):
                pixel = image.pixelColor(x, y)
                total[0] += pixel.red()
                total[1] += pixel.green()
                total[2] += pixel.blue()
                count += 1
        return [component / count for component in total]

    # Every escaped ray lands on the dome, so nothing in the frame may be
    # dark: a dark blob could only be a light's stand-in geometry gone
    # camera-visible, silhouetting its black absorber against the sky.
    # All three shaped lights float inside the frustum to keep this
    # check honest.
    floor = 255
    for y in range(height):
        row = pixels[y * stride : y * stride + width * 3]
        for x in range(width):
            floor = min(floor, max(row[x * 3], row[x * 3 + 1], row[x * 3 + 2]))
    if floor <= 2 * DARK:
        fail(f"a pixel bottoms out at {floor} — with sky behind everything, that is "
             f"a black-absorber silhouette (a light shape turned camera-visible)")

    # The dome as background, found without trusting orientation: its
    # below-horizon ground glow is a warm-dark band (beyond the ground
    # plane's far edge) that neither the sky nor the lit ground can
    # imitate, at one of two vertically mirrored rows; the mirror row
    # must be bright blue sky. The left sky probe sits exactly where the
    # rect light's stand-in projects, so sky there re-proves invisibility.
    def band_like(color):
        return (color[0] > color[1] > color[2] and color[0] - color[2] >= 15
                and max(color) < 150)

    def sky_like(color):
        return color[2] - color[0] >= 25 and color[2] >= 180

    mirrored = (0.33, 0.67)
    rows = {}
    for fy in mirrored:
        sides = (patch(0.05, fy), patch(0.95, fy))
        rows[fy] = (all(band_like(side) for side in sides),
                    all(sky_like(side) for side in sides))
    band_rows = [fy for fy, (band, _) in rows.items() if band]
    if len(band_rows) != 1:
        fail(f"the dome's below-horizon band should sit at exactly one of the mirrored "
             f"rows, found it at {band_rows or 'neither'}")
    band_fy = band_rows[0]
    sky_fy = mirrored[band_fy == mirrored[0]]
    if not rows[sky_fy][1]:
        fail(f"the row mirroring the horizon band (y={sky_fy}) should be sky, but is "
             f"not bright blue-dominant")

    # The light pools, on the ground side of the band (the flip the band
    # just revealed), warm and cool on opposite sides — either side, so
    # a mirrored frame still passes.
    ground_fy = band_fy - 0.21 if band_fy < 0.5 else band_fy + 0.21
    sides = (patch(0.1, ground_fy), patch(0.85, ground_fy))
    warm = [c[0] - c[2] >= 25 and c[0] >= 200 for c in sides]
    cool = [c[2] - c[0] >= 30 and c[2] >= 210 for c in sides]
    if not ((warm[0] and cool[1]) or (warm[1] and cool[0])):
        fail(f"the ground should pool warm under the rect and cool under the sphere "
             f"on opposite sides, but reads {[[round(c, 1) for c in s] for s in sides]}")

    print(
        f"usdrecord smoke: OK — {width}x{height}, dome band and sky behind the scene, "
        f"warm and cool pools on opposite sides, darkest pixel {floor} (no silhouettes)"
    )


def main():
    directory = Path(tempfile.mkdtemp(prefix="cenote-usdrecord-"))
    try:
        for stage, analyze in (
            (STAGES / "first-light.usda", analyze_first_light),
            (STAGES / "preview-surface.usda", analyze_preview_surface),
            (STAGES / "lit-stage.usda", analyze_lit_stage),
        ):
            out_path = directory / f"{stage.stem}.png"
            record(stage, out_path)
            analyze(out_path)
    except SystemExit:
        print(f"artifacts kept in {directory}", file=sys.stderr)
        raise
    shutil.rmtree(directory)


if __name__ == "__main__":
    main()
