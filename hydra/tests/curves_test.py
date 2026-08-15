#!/usr/bin/env python3
"""The curves check: stages/curves-stage.usda through a batch host, asking
whether BasisCurves prims survive the trip from USD to the wire intact.

Three legs.

  1. Translation. The server logs one line per curve batch it resolved —
     strands, points, triangles — and this leg reads the same stage with
     pxr and holds the two accounts against each other. Every accepted
     prim has to appear with the curve count USD authored; the refused
     ones must not appear at all; and the points the server ends up with
     are reported against the control vertices it was given. That ratio is
     what a groom's memory footprint scales with, so it prints whether or
     not anything is wrong.

  2. Occupancy — usdrecord only. One band of pixels per prim, in the
     column the stage stands it in. Each accepted prim has to put ink in
     its own band, and the band the refused prims share has to be black.
     That second half is the real assertion: a change set is atomic, so a
     delegate that forwarded a refused prim instead of withdrawing it
     would take the whole scene down and every band would be black
     together.

  3. Colour — usdrecord only. Each contract prim's displayColor leads in
     its own channel, which is the companion-material path proven on
     curves rather than on meshes.

Legs 2 and 3 read pixels but pin none: what is asserted is which channel
leads and which band is empty, so noise, sampling, and a watermark change
nothing. That is also why the husk host runs leg 1 alone — Houdini's
diagnostic manager swallows the delegate's own warnings, but the server is
a child process either way and its log reaches the stream unchanged.

    python3 hydra/tests/curves_test.py               # usdrecord
    python3 hydra/tests/curves_test.py --host husk   # needs the hdk build

Run from anywhere with the USD prefix on PATH/PYTHONPATH (hydra/README.md's
environment) and the plugin installed to hydra/dist/. The husk host needs
that install to be the *hdk*-flavour one and $HFS on PATH.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STAGE = REPO / "hydra" / "tests" / "stages" / "curves-stage.usda"
RESOURCES = REPO / "hydra" / "dist" / "hdCenote" / "resources"

CAMERA = "/World/Camera"
WIDTH = 640
HEIGHT = 467
TIMEOUT_S = 300

# The server's own account of a resolved curve batch (scene/curves.rs).
RESOLVED = re.compile(r'curves "([^"]+)": (\d+) strands, (\d+) points, (\d+) triangles')

# The prims the stage exists to have refused — one per way the server
# refuses a topology, and neither may reach it.
REFUSED = (
    "/World/Contract/RefusedPeriodic",
    "/World/Contract/RefusedSpan",
)

# One band per column, in world x: wide enough to hold the prim and narrow
# enough to exclude its neighbours, with the share of lit pixels each must
# reach. The two refused prims share a band and must reach nothing. Bands
# are checked against where the stage actually stands each prim (see
# check_bands), so a prim that moves fails loudly instead of being measured
# in the wrong place.
BANDS = [
    ("/World/Groom/cubic/curve_0", -1.60, -0.92, 0.02),
    ("/World/Groom/linear/curve_0", -0.92, -0.38, 0.003),
    (REFUSED[0], -0.05, 0.25, 0.0),
    (REFUSED[1], -0.05, 0.25, 0.0),
    ("/World/Contract/PinnedCatmullRom", 0.30, 0.52, 0.02),
    ("/World/Contract/Bezier", 0.62, 0.98, 0.01),
    ("/World/Contract/VaryingWidth", 1.02, 1.38, 0.01),
    ("/World/Contract/UniformWidth", 1.50, 1.90, 0.02),
]

# What each contract prim's displayColor does to its column: three lead
# in a channel, and the yellow one is recognised by the channel it leaves
# behind. A modest margin, not a pure primary — OpenPBR's default
# specular layer puts a white sheen along every tube, and a curve is
# mostly silhouette, so the sheen is a large share of what the camera
# sees and it desaturates every column the same way.
LEAD = 1.15
COLOURS = [
    ("/World/Contract/PinnedCatmullRom", 0, "leads"),
    ("/World/Contract/Bezier", 2, "trails"),
    ("/World/Contract/VaryingWidth", 1, "leads"),
    ("/World/Contract/UniformWidth", 2, "leads"),
]

# A pixel counts as lit above this — well over the render's black, well
# under any strand's shaded value.
LIT = 0.01


def fail(message):
    print(f"curves: FAIL — {message}", file=sys.stderr)
    sys.exit(1)


def locate_server():
    if os.environ.get("CENOTE_SERVER"):
        return os.environ["CENOTE_SERVER"]
    for profile in ("release", "debug"):
        candidate = REPO / "target" / profile / "cenote-server"
        if os.access(candidate, os.X_OK):
            return str(candidate)
    fail("no cenote-server: set $CENOTE_SERVER or `cargo build -p cenote-server`")


def authored():
    """What the stage says, straight from USD: per BasisCurves prim, how
    many curves it holds, how many control vertices they carry between
    them, and where in world x it stands. The counts are the "in" half of
    the points-in/points-out diagnostic; the position is what the pixel
    bands are checked against. Read here rather than parsed out of the
    .usda so a reference, an override, or a Houdini regeneration all
    arrive composed.

    Returns the prims and the frame's own span in world x, taken off the
    stage camera at the plane the curves stand in — so the bands below
    stay written in the coordinates the stage is authored in."""
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(STAGE))
    if not stage:
        fail(f"{STAGE} did not open")
    camera = UsdGeom.Camera.Get(stage, CAMERA)
    if not camera:
        fail(f"{CAMERA} missing from {STAGE}")
    gf = camera.GetCamera(Usd.TimeCode.Default())
    eye = gf.transform.ExtractTranslation()
    # An axis-aligned camera looking down -z at the z = 0 plane the curves
    # stand in; similar triangles do the rest.
    if not Gf.IsClose(gf.transform.ExtractRotationMatrix(), Gf.Matrix3d(1), 1e-9):
        fail(f"{CAMERA} is rotated, and the band arithmetic assumes it is not")
    half = eye[2] * (gf.horizontalAperture / 2.0) / gf.focalLength
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    prims = {}
    for prim in stage.Traverse():
        if prim.GetTypeName() != "BasisCurves":
            continue
        curves = UsdGeom.BasisCurves(prim)
        counts = curves.GetCurveVertexCountsAttr().Get() or []
        box = bounds.ComputeWorldBound(prim).ComputeAlignedRange()
        if box.IsEmpty():
            fail(f"{prim.GetPath()} has no world bound; the stage cannot be framed")
        prims[str(prim.GetPath())] = {
            "curves": len(counts),
            "points": sum(counts),
            "type": curves.GetTypeAttr().Get(),
            "x": (box.GetMin()[0] + box.GetMax()[0]) / 2.0,
        }
    if not prims:
        fail(f"{STAGE} holds no BasisCurves prims")
    return prims, (eye[0] - half, eye[0] + half)


def check_bands(expected):
    """Every band names a prim the stage authors, and brackets where that
    prim actually stands. This is what keeps legs 2 and 3 measuring the
    column they think they are: move a prim on the stage and the test says
    so, instead of quietly reading its neighbour's pixels."""
    for name, left, right, _ in BANDS:
        if name not in expected:
            fail(f"the band for {name} names a prim {STAGE} does not author")
        centre = expected[name]["x"]
        if not left < centre < right:
            fail(f"{name} stands at x {centre:.2f}, outside its {left}…{right} band")


def render(host, out_path):
    """One frame through `host`, with the server talking. Returns whatever
    the pair of them wrote to the stream."""
    if not RESOURCES.is_dir():
        fail(f"{RESOURCES} missing — build and `cmake --install build/hydra` first")
    if shutil.which(host) is None:
        fail(f"{host} not on PATH — export the USD prefix (hydra/README.md)")
    env = os.environ.copy()
    env["PXR_PLUGINPATH_NAME"] = str(RESOURCES)
    env["CENOTE_SERVER"] = locate_server()
    # The whole of leg 1's evidence: the resolver's own per-batch line.
    env["RUST_LOG"] = "cenote=debug"
    if host == "husk":
        # husk finds its renderer through HOUDINI_PATH, and carries its own
        # USD — the stock prefix's python modules crash its older Sdf.
        env["HOUDINI_PATH"] = f"{RESOURCES.parents[1]}:&"
        env.pop("PYTHONPATH", None)
        command = [
            "husk",
            "--renderer", "HdCenoteRendererPlugin",
            "--camera", "/World/Camera",
            "--res", str(WIDTH), str(HEIGHT),
            "--frame", "1", "--frame-count", "1",
            "--output", str(out_path),
            "--usd-input", str(STAGE),
        ]
    else:
        command = [
            "usdrecord",
            "--renderer", "Cenote",
            "--camera", "/World/Camera",
            # The stage lights itself; usdview's fallback camera light
            # would wash the columns together.
            "--disableCameraLight",
            "--imageWidth", str(WIDTH),
            str(STAGE), str(out_path),
        ]
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        done = subprocess.run(command, env=env, capture_output=True, text=True,
                              timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        fail(f"{host} was still rendering after {TIMEOUT_S}s")
    output = done.stdout + done.stderr
    if done.returncode != 0:
        sys.stderr.write(output[-4000:])
        fail(f"{host} exited {done.returncode}")
    if not out_path.is_file():
        sys.stderr.write(output[-4000:])
        fail(f"{host} exited 0 but wrote no image")
    return output


def check_translation(output, expected):
    """Leg 1: every accepted prim reached the server with the topology USD
    authored, the refused one reached it not at all, and the points a
    groom costs are reported against the points it was authored with."""
    resolved = {}
    for name, strands, points, triangles in RESOLVED.findall(output):
        if name in resolved:
            fail(f"the server resolved {name} twice")
        resolved[name] = (int(strands), int(points), int(triangles))
    if not resolved:
        # No account at all is a broken observable, not a broken
        # translation — say which before blaming the delegate.
        fail("the server logged no resolved curve batch; is RUST_LOG reaching it?")
    for name in REFUSED:
        if name in resolved:
            fail(f"{name} reached the server; a topology it refuses must be withdrawn on the "
                 f"way in")
    accepted = {name: spec for name, spec in expected.items() if name not in REFUSED}
    missing = sorted(set(accepted) - set(resolved))
    if missing:
        fail(f"the server never saw {', '.join(missing)}")
    extra = sorted(set(resolved) - set(accepted))
    if extra:
        fail(f"the server saw prims the stage does not author: {', '.join(extra)}")
    print(f"{'prim':36s} {'curves':>7s} {'in':>7s} {'out':>7s} {'ratio':>6s} {'tris':>8s}")
    for name in sorted(accepted):
        strands, points, triangles = resolved[name]
        spec = accepted[name]
        if strands != spec["curves"]:
            fail(f"{name}: USD authors {spec['curves']} curves, the server resolved {strands}")
        ratio = points / spec["points"]
        print(f"{name:36s} {strands:7d} {spec['points']:7d} {points:7d} {ratio:6.2f} "
              f"{triangles:8d}")
        if spec["type"] == "linear":
            # A polyline passes through verbatim — no basis, no flattening.
            if points != spec["points"]:
                fail(f"{name} is linear: {spec['points']} points in, {points} out")
        elif not 1.0 <= ratio <= 8.0:
            # Cubic spans flatten adaptively, capped at MAX_SPAN_SEGMENTS
            # pieces each (scene/curves.rs), so the ratio is bounded on
            # both sides — under one would mean vertices went missing.
            fail(f"{name} flattened {ratio:.2f} points per control vertex, outside 1…8")
    withdrawn = ", ".join(name.split("/")[-1] for name in REFUSED)
    print(f"curves: {len(accepted)} prims translated, {withdrawn} withdrawn")


def load(path):
    import OpenImageIO as oiio

    buf = oiio.ImageBuf(str(path))
    if buf.has_error:
        fail(f"{path} did not load as an EXR: {buf.geterror()}")
    return buf.get_pixels(oiio.FLOAT), buf.spec().width


def band(frame, pixels, width, left, right):
    """The lit pixels of one column: their share of the band, and their
    mean colour."""
    span = frame[1] - frame[0]
    first = max(0, int((left - frame[0]) / span * width))
    last = min(width, int((right - frame[0]) / span * width))
    if last <= first:
        fail(f"the band {left}…{right} lands outside a {width}-pixel frame")
    region = pixels[:, first:last, :3]
    lit = region.max(axis=2) > LIT
    share = float(lit.mean())
    if not lit.any():
        return share, (0.0, 0.0, 0.0)
    return share, tuple(region.reshape(-1, 3)[lit.reshape(-1)].mean(axis=0))


def check_occupancy(frame, pixels, width):
    """Leg 2: each accepted prim renders in its own column, and the column
    the refused prims share is black."""
    for name, left, right, floor in BANDS:
        share, _ = band(frame, pixels, width, left, right)
        if name in REFUSED:
            if share > 0:
                fail(f"{name} rendered: {share * 100:.2f}% of its band is lit, and the server "
                     f"holds no such topology")
            continue
        if share < floor:
            fail(f"{name} covers {share * 100:.2f}% of its band, under the {floor * 100:.2f}% it "
                 f"has to reach — the prim is missing or misplaced")
    print(f"curves: {len(REFUSED)} refused prims left their band black")


def check_colour(frame, pixels, width):
    """Leg 3: each contract prim wears its own displayColor, so its column
    leads in the channel the stage authored."""
    bands = {name: (left, right) for name, left, right, _ in BANDS}
    channels = "red", "green", "blue"
    for name, channel, how in COLOURS:
        left, right = bands[name]
        _, rgb = band(frame, pixels, width, left, right)
        others = [rgb[index] for index in range(3) if index != channel]
        held = (rgb[channel] >= LEAD * max(others) if how == "leads"
                else LEAD * rgb[channel] <= min(others))
        if not held:
            fail(f"{name} reads {rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f}, which does not {how} in "
                 f"{channels[channel]} — the companion material did not reach its instance")
    print(f"curves: {len(COLOURS)} contract prims wear their own displayColor")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=("usdrecord", "husk"), default="usdrecord")
    parser.add_argument("--keep", action="store_true", help="leave the render on disk")
    arguments = parser.parse_args()

    expected, frame = authored()
    check_bands(expected)
    with tempfile.TemporaryDirectory() as directory:
        out_path = Path(directory) / "curves.exr"
        output = render(arguments.host, out_path)
        check_translation(output, expected)
        if arguments.host == "usdrecord":
            pixels, width = load(out_path)
            check_occupancy(frame, pixels, width)
            check_colour(frame, pixels, width)
        if arguments.keep:
            kept = REPO / "curves.exr"
            shutil.copy(out_path, kept)
            print(f"curves: kept {kept}")
    print(f"curves: PASS ({arguments.host})")


if __name__ == "__main__":
    main()
