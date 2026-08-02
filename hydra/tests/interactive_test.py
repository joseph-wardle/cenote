#!/usr/bin/env python3
"""The interactive-loop check: first-light.usda driven through testusdview,
asserting what a user at the viewport would live through. Three legs:

  1. Edit honesty — a real edit through the stage handle drops IsConverged
     (the epoch bar rising past the front frame) and reconverges
     with different pixels; a visually inert edit still travels the wire
     and comes back converged — a wedged epoch would time out right there.
  2. Kill-survive — SIGKILL the spawned cenote-server mid-session: the app
     survives, the client degrades (observed as the corpse being reaped),
     converged reads true so hosts do not spin, and the warning naming the
     recovery gesture reaches stderr.
  3. Toggle-recover — that very gesture: switch the renderer away and
     back, and a fresh delegate spawns a fresh server with the whole stage
     replayed; the cubes must come back.

One file, two roles. Run directly — from anywhere, with the USD prefix on
PATH/PYTHONPATH and the plugin installed to hydra/dist/ (README.md's
environment, GPU machine only) — it sets up the environment the way
usdrecord_smoke.py does and invokes testusdview with itself as the
--testScript. testusdview then execs this same file, with neither __name__
nor __file__ defined (hence the guards below), and calls
testUsdviewInputFunction(appController) inside the viewer process.
"""

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

# The stdout lines the viewer half prints and the wrapper half demands.
MARK_EDIT = "interactive: edit honesty OK"
MARK_NOOP = "interactive: no-op republish OK"
MARK_KILL = "interactive: kill-survive OK"
MARK_TOGGLE = "interactive: toggle-recover OK"

# The recovery gesture the disconnect warning names (transport/client.cpp):
# the kill leg asserts it reached stderr, then the toggle leg performs it.
KILL_WARNING = "toggle the renderer or reload the stage"

CENOTE = "HdCenoteRendererPlugin"

# 8-bit threshold: a pixel this bright counts as lit geometry.
LIT = 40


# ---------------------------------------------------------------------------
# The viewer half: everything below runs inside the testusdview process.


def _check(condition, message):
    if not condition:
        raise AssertionError(f"interactive test: {message}")


def _pump(view, predicate, what, kick=False, deadline_s=120):
    """Drives the Qt event loop until predicate() holds. An unconverged view
    schedules its own repaints, but a converged one goes idle — kick forces
    a paint each spin, so the client's per-frame health checks keep running
    when nothing else would run them."""
    from pxr.Usdviewq.qt import QtWidgets

    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        if kick:
            view.updateGL()
        QtWidgets.QApplication.processEvents()
        if predicate():
            return
    raise AssertionError(f"interactive test: timed out waiting for {what}")


def _rgb(shot):
    """A viewport QImage as comparable raw RGB bytes."""
    from pxr.Usdviewq.qt import QtGui

    return bytes(shot.convertToFormat(QtGui.QImage.Format.Format_RGB888).constBits())


def _silhouette(shot):
    """The smoke test's coarse eyeball: lit pixels bucketed left and right
    of center as (count, sum R, sum B)."""
    from pxr.Usdviewq.qt import QtGui

    image = shot.convertToFormat(QtGui.QImage.Format.Format_RGB888)
    width, height = image.width(), image.height()
    stride = image.bytesPerLine()
    pixels = image.constBits()
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
    return buckets


def _spawned_servers():
    """The cenote-server pids this very process spawned, by /proc walk: the
    delegate's client is in-process, so its child's ppid is our pid."""
    pids = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            stat = Path("/proc", entry, "stat").read_text()
        except OSError:
            continue  # gone between the listing and the read
        comm = stat[stat.index("(") + 1 : stat.rindex(")")]
        ppid = int(stat[stat.rindex(")") + 2 :].split()[1])
        if comm == "cenote-server" and ppid == os.getpid():
            pids.append(int(entry))
    return pids


def testUsdviewInputFunction(appController):
    from pxr import Gf

    view = appController._stageView
    stage = appController._dataModel.stage

    def converged():
        return view._renderer.IsConverged()

    # The HUD paints into the viewport grab; the pixel checks want geometry
    # only.
    appController._dataModel.viewSettings.showHUD = False
    _pump(view, converged, "the first converged frame")
    before = _rgb(view.grabFramebuffer())

    # Leg 1: edit honesty. A real edit through the stage handle must drop
    # convergence and come back with a different picture.
    near = stage.GetPrimAtPath("/World/Near")
    _check(near, "/World/Near missing from the stage")
    near.GetAttribute("xformOp:translate").Set(Gf.Vec3d(-1.2, -3.0, 1.2))
    _pump(view, lambda: not converged(), "convergence to drop after the edit")
    _pump(view, converged, "reconvergence after the edit")
    _check(
        _rgb(view.grabFramebuffer()) != before,
        "the edit reconverged without changing a pixel",
    )
    print(MARK_EDIT, flush=True)

    # Leg 1, the honest no-op: the same displayColor authored again is
    # visually inert but still travels the wire and bumps the epoch, so
    # convergence must drop and *return* — end to end through the server's
    # republish, where a wedged epoch would stick at never-converged.
    color = near.GetAttribute("primvars:displayColor")
    color.Set(color.Get())
    _pump(view, lambda: not converged(), "convergence to drop after the no-op edit")
    _pump(view, converged, "converged to return after the no-op edit")
    print(MARK_NOOP, flush=True)

    # Leg 2: kill-survive. SIGKILL the server; the next painted frame's
    # liveness probe must notice the hangup, warn, degrade, and reap the
    # corpse — /proc/<pid> outliving the kill only as a zombie, then
    # vanishing at the reap, is degradation observed from outside.
    pids = _spawned_servers()
    _check(len(pids) == 1, f"expected exactly one spawned cenote-server, found {pids}")
    server = pids[0]
    os.kill(server, signal.SIGKILL)
    _pump(
        view,
        lambda: not Path("/proc", str(server)).exists(),
        "the client to degrade and reap the dead server",
        kick=True,
    )
    _check(converged(), "degraded must still read converged, or hosts spin forever")
    print(MARK_KILL, flush=True)

    # Leg 3: toggle-recover — the gesture the warning names. Away to any
    # other renderer and back: a fresh delegate spawns a fresh server and
    # the whole stage is replayed into it.
    _check(
        view.GetCurrentRendererId() == CENOTE,
        f"expected {CENOTE} active, got {view.GetCurrentRendererId()}",
    )
    others = [p for p in view.GetRendererPlugins() if p != CENOTE]
    _check(others, "no other renderer plugin to toggle through")
    _check(view.SetRendererPlugin(others[0]), f"switching to {others[0]} failed")
    _pump(view, converged, f"{others[0]} to settle")
    _check(view.SetRendererPlugin(CENOTE), "switching back to Cenote failed")
    _pump(
        view,
        lambda: _spawned_servers() and converged(),
        "a fresh server and reconvergence after the toggle",
        kick=True,
    )
    (near_count, near_r, near_b), (far_count, far_r, far_b) = _silhouette(
        view.grabFramebuffer()
    )
    _check(
        near_count > 0 and far_count > 0,
        f"a cube is missing after recovery: {near_count} lit left, {far_count} lit right",
    )
    _check(near_r > near_b, "the near cube reads cold after recovery")
    _check(far_b > far_r, "the far cube reads warm after recovery")
    print(MARK_TOGGLE, flush=True)


# ---------------------------------------------------------------------------
# The wrapper half: everything below runs only when this file is executed
# directly, and drives testusdview with this same file as the test script.


def fail(message):
    print(f"interactive test: FAIL — {message}", file=sys.stderr)
    sys.exit(1)


def locate_server(repo):
    if os.environ.get("CENOTE_SERVER"):
        return os.environ["CENOTE_SERVER"]
    for profile in ("release", "debug"):
        candidate = repo / "target" / profile / "cenote-server"
        if os.access(candidate, os.X_OK):
            return str(candidate)
    fail("no cenote-server: set $CENOTE_SERVER or `cargo build -p cenote-server`")


def main():
    script = Path(__file__).resolve()
    repo = script.parents[2]
    stage = repo / "hydra" / "tests" / "stages" / "first-light.usda"
    resources = repo / "hydra" / "dist" / "hdCenote" / "resources"
    if not resources.is_dir():
        fail(f"{resources} missing — build and `cmake --install build/hydra` first")
    if shutil.which("testusdview") is None:
        fail("testusdview not on PATH — export the USD prefix (hydra/README.md)")
    env = os.environ.copy()
    env["PXR_PLUGINPATH_NAME"] = str(resources)
    env["CENOTE_SERVER"] = locate_server(repo)
    # A low cap so every reconvergence the legs wait on arrives in seconds.
    env.setdefault("CENOTE_SERVER_MAX_SAMPLES", "64")
    # The Wayland traps (README.md): Qt through xcb, GL through GLX.
    env.setdefault("QT_QPA_PLATFORM", "xcb")
    env.setdefault("PYOPENGL_PLATFORM", "glx")
    command = [
        "testusdview",
        "--renderer", "Cenote",
        "--camera", "/World/Camera",
        "--testScript", str(script),
        str(stage),
    ]
    try:
        done = subprocess.run(command, env=env, timeout=600, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        fail("testusdview timed out after 600s")
    if done.returncode != 0:
        sys.stderr.write(done.stdout[-2000:] + done.stderr[-2000:])
        fail(f"testusdview exited {done.returncode}")
    for marker in (MARK_EDIT, MARK_NOOP, MARK_KILL, MARK_TOGGLE):
        if marker not in done.stdout:
            sys.stderr.write(done.stdout[-2000:] + done.stderr[-2000:])
            fail(f"missing marker: {marker!r}")
    if KILL_WARNING not in done.stderr:
        sys.stderr.write(done.stderr[-2000:])
        fail("the kill never produced the disconnect warning")
    print("interactive test: OK — edit honesty, no-op republish, kill-survive, toggle-recover")


# testusdview execs this file with neither __name__ nor __file__ defined,
# so the usual bare __main__ check would NameError under it.
if globals().get("__name__") == "__main__":
    main()
