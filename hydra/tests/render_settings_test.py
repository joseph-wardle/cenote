#!/usr/bin/env python3
"""The render-settings check: stages/render-settings.usda driven through a
batch host, once per settings prim on it, asserting what a shot's authored
settings are supposed to do to the render.

Six legs, each naming a different RenderSettings prim (or none):

  1. Defaults — no prim named. The delegate's own 64/off/8 is what gets
     rendered, silently, and the unnamed prims on the stage stay inert.
  2. The prim wins — an impossible budget arrives and the render is still
     going long after the default would have settled. Wall clock only as a
     one-sided bound: the discriminator is "would 64 samples have finished
     by now", and the answer is calibrated from leg 1's own run.
  3. Early stop — that same impossible budget under a noise threshold
     finishes anyway. This is the leg a renderer that reports convergence
     by counting samples against the budget fails by hanging forever.
  4. Complaints — a depth past the ceiling, a budget of an unreadable
     type, and a typo'd key, all on a prim that must still render: the
     clamp is the delegate's job precisely because the server rejects a
     change-set whole, geometry included.
  5. Foreign — a prim authored for another renderer says so once, and
     that renderer's own keys pass without comment.
  6. The environment override — CENOTE_SERVER_MAX_SAMPLES beats the shot,
     which is the documented top of the precedence chain.

Behaviour, never pixels: the same file runs under husk, whose Education
watermark would poison any image check.

The two hosts prove different halves with the same stage. usdrecord's
`--renderSettingsPrimPath` names the prim to Hydra's scene globals, so the
prim path carries it; husk resolves the stage's settings itself and hands
the delegate a settings map, so the map path carries it — and husk really
does populate that map, with the `cenote:` namespace intact, which is what
this file was written to find out.

Husk answers in a narrower voice. Houdini installs a TfDiagnosticMgr
delegate, which routes TF_STATUS and TF_WARN into its own error manager and
out of the stream (`-V 9` does not bring them back), so under husk the only
account of the settings is what the render *does*: a budget nothing can
finish keeps running, a threshold over it finishes anyway. The legs that
read the delegate's own words run under usdrecord alone, and say so.

    python3 hydra/tests/render_settings_test.py               # usdrecord
    python3 hydra/tests/render_settings_test.py --host husk   # needs the hdk build

Run from anywhere with the USD prefix on PATH/PYTHONPATH (hydra/README.md's
environment) and the plugin installed to hydra/dist/. The husk host needs
that install to be the *hdk*-flavour one and $HFS on PATH.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STAGE = REPO / "hydra" / "tests" / "stages" / "render-settings.usda"
RESOURCES = REPO / "hydra" / "dist" / "hdCenote" / "resources"

# The delegate's status line (renderSettings.cpp) — the whole observable.
# A batch host has no settings UI, so what the renderer says it resolved is
# the only account of the settings anyone can read back.
STATUS = "cenote: rendering at "
DEFAULTS = "64 samples per pixel, no early stop, 8 bounces"

# The smallest frame that still exercises the whole path: nothing here
# looks at a pixel, and every leg pays for the width in seconds.
WIDTH = 320
HEIGHT = 240

# How long a leg may take before we call it hung. Generous — leg 2's real
# bound is computed from leg 1, not from this.
TIMEOUT_S = 120


def fail(message):
    print(f"render settings: FAIL — {message}", file=sys.stderr)
    sys.exit(1)


def locate_server():
    if os.environ.get("CENOTE_SERVER"):
        return os.environ["CENOTE_SERVER"]
    for profile in ("release", "debug"):
        candidate = REPO / "target" / profile / "cenote-server"
        if os.access(candidate, os.X_OK):
            return str(candidate)
    fail("no cenote-server: set $CENOTE_SERVER or `cargo build -p cenote-server`")


class Host:
    """One batch renderer, reduced to what this test needs of it: a command
    that renders the stage with a chosen settings prim active."""

    def __init__(self, name):
        if name not in ("usdrecord", "husk"):
            fail(f"unknown host {name!r} — usdrecord or husk")
        if shutil.which(name) is None:
            fail(f"{name} not on PATH — export the USD prefix (hydra/README.md)")
        self.name = name
        # Whether the delegate's own diagnostics reach the stream. Houdini
        # installs a TfDiagnosticMgr delegate and takes them into its error
        # manager instead, so under husk the render's behaviour is the
        # whole account.
        self.speaks = name == "usdrecord"

    def environment(self, overrides):
        if not RESOURCES.is_dir():
            fail(f"{RESOURCES} missing — build and `cmake --install build/hydra` first")
        env = os.environ.copy()
        env["PXR_PLUGINPATH_NAME"] = str(RESOURCES)
        env["CENOTE_SERVER"] = locate_server()
        # husk finds its renderer list through HOUDINI_PATH, not the pxr
        # registry (hydra/README.md).
        if self.name == "husk":
            env["HOUDINI_PATH"] = f"{RESOURCES.parents[1]}:&"
        # Deliberately *not* set: this test is about what the scene asks
        # for, and the override sits above everything the scene can say.
        # Leg 6 is the one that sets it, on purpose.
        env.pop("CENOTE_SERVER_MAX_SAMPLES", None)
        env.setdefault("QT_QPA_PLATFORM", "offscreen" if self.name == "husk" else "xcb")
        env.update(overrides)
        return env

    def command(self, settings_prim, out_path):
        if self.name == "usdrecord":
            command = [
                "usdrecord",
                "--renderer", "Cenote",
                "--camera", "/World/Camera",
                "--disableCameraLight",
                "--imageWidth", str(WIDTH),
            ]
            if settings_prim:
                command += ["--renderSettingsPrimPath", settings_prim]
            return command + [str(STAGE), str(out_path)]
        command = [
            "husk",
            "--renderer", "HdCenoteRendererPlugin",
            "--camera", "/World/Camera",
            "--res", str(WIDTH), str(HEIGHT),
            "--frame", "1", "--frame-count", "1",
            "--output", str(out_path),
        ]
        if settings_prim:
            command += ["--settings", settings_prim]
        return command + ["--usd-input", str(STAGE)]


class Run:
    """A finished (or abandoned) render: its output, its exit status, and
    how long it took."""

    def __init__(self, leg, output, code, seconds, finished, out_path):
        self.leg = leg
        self.output = output
        self.code = code
        self.seconds = seconds
        self.finished = finished
        self.out_path = out_path

    def dump(self):
        sys.stderr.write(self.output[-3000:])

    def succeeded(self):
        if not self.finished:
            self.dump()
            fail(f"{self.leg}: still rendering after {self.seconds:.1f}s")
        if self.code != 0:
            self.dump()
            fail(f"{self.leg}: exited {self.code}")
        if not self.out_path.is_file():
            self.dump()
            fail(f"{self.leg}: exited 0 but wrote no image")

    def status(self):
        """What the delegate said it resolved, as its one line. Every leg
        expects exactly one: the settings are re-resolved only when a
        source moves, and in batch nothing moves after the first frame."""
        lines = [
            line[line.index(STATUS) + len(STATUS):].strip()
            for line in self.output.splitlines()
            if STATUS in line
        ]
        if len(lines) != 1:
            self.dump()
            fail(f"{self.leg}: expected one resolved-settings line, found {len(lines)}")
        return lines[0]

    def complaints(self):
        """The delegate's own warnings, as the sentences it chose to say:
        TfDiagnostic prefixes each with the source context that raised it,
        and only what follows the dashes is the message. Narrowed to the
        settings vocabulary so the server's log — which names cenote on
        every line it prints — is not mistaken for a complaint."""
        said = []
        for line in self.output.splitlines():
            if "Warning" not in line or " -- " not in line:
                continue
            message = line.split(" -- ", 1)[1].strip()
            if "cenote:" in message or "render settings prim" in message:
                said.append(message)
        return said


def render(host, leg, settings_prim=None, overrides=None, window=None, directory=None):
    """Runs one leg. `window` turns the wait into a bound rather than a
    wait: the process is abandoned when it expires, and `finished` records
    which happened."""
    out_path = directory / f"{leg}.exr"
    command = host.command(settings_prim, out_path)
    log = directory / f"{leg}.log"
    started = time.monotonic()
    with log.open("w+") as sink:
        # A file, not a pipe: leg 2 abandons a process still writing, and a
        # pipe nobody is draining would deadlock it instead.
        process = subprocess.Popen(command, env=host.environment(overrides or {}),
                                   stdout=sink, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=window or TIMEOUT_S)
            finished = True
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            code, finished = None, False
    return Run(leg, log.read_text(errors="replace"), code, time.monotonic() - started,
               finished, out_path)


def check(condition, message):
    if not condition:
        fail(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="usdrecord", choices=("usdrecord", "husk"),
                        help="which batch renderer drives the stage")
    arguments = parser.parse_args()
    host = Host(arguments.host)

    directory = Path(tempfile.mkdtemp(prefix="cenote-render-settings-"))
    try:
        legs(host, directory)
    except SystemExit:
        print(f"artifacts kept in {directory}", file=sys.stderr)
        raise
    shutil.rmtree(directory)


def legs(host, directory):
    def run(leg, **kwargs):
        return render(host, leg, directory=directory, **kwargs)

    def resolved(run, expected):
        """What the delegate said it resolved — under a host that lets it
        speak. Every leg still asserts what the render *did*; this is the
        finer reading on top, and it is not husk's to give."""
        if host.speaks:
            check(run.status() == expected, f"{run.leg}: resolved {run.status()!r}, "
                                            f"expected {expected!r}")

    # 1. No prim named. The delegate answers for everything, and the five
    #    prims on the stage — one of them authoring a budget of 2 and a
    #    typo — are inert, because none of them is the active one.
    defaults = run("defaults")
    defaults.succeeded()
    resolved(defaults, DEFAULTS)
    if host.speaks:
        check(not defaults.complaints(),
              f"defaults: an unauthored stage should be silent, said {defaults.complaints()}")

    # 2. The shot's budget, and the one leg that proves it reached the
    #    *server* rather than merely the delegate. The bound is leg 1's own
    #    duration: four times over is far past the point a 64-sample render
    #    had settled, and 1,000,000 samples is minutes away on any machine
    #    that finishes the rest of this file at all.
    window = max(8.0, 4 * defaults.seconds)
    slow = run("slow", settings_prim="/Render/Slow", window=window)
    check(not slow.finished,
          f"slow: converged in {slow.seconds:.1f}s — a million samples cannot have "
          f"been rendered, so the budget never reached the server")
    resolved(slow, "1000000 samples per pixel, no early stop, 8 bounces")

    # 3. The early stop, which is also the regression: convergence has to
    #    come from the session's own verdict, because the sample count will
    #    never reach this budget and a host that loops on the flag would
    #    never return.
    early = run("early-stop", settings_prim="/Render/EarlyStop")
    early.succeeded()
    resolved(early, "1000000 samples per pixel, stopping early at 0.05 relative error, "
                    "8 bounces")
    check(early.seconds < window,
          f"early-stop: took {early.seconds:.1f}s, which is not an early stop")

    # 4. Everything wrong at once, on a render that must still arrive: the
    #    clamp especially, since server-side an authored 512 would take
    #    every mesh in the same flush down with it.
    loud = run("complaints", settings_prim="/Render/Complaints")
    loud.succeeded()
    resolved(loud, f"{DEFAULTS.split(',')[0]}, no early stop, 255 bounces")
    if host.speaks:
        for phrase in ("cenote:maxBounces is 512",
                       "cenote:samplesPerPixel is a",
                       "cenote:maxBouncez is not a cenote render setting"):
            check(any(phrase in line for line in loud.complaints()),
                  f"complaints: nothing said about {phrase!r} — said {loud.complaints()}")

    # 5. The stage authored for Karma or RenderMan: one sentence saying the
    #    shot's settings were not ours to obey, and not one word about the
    #    settings that were not addressed to us.
    foreign = run("foreign", settings_prim="/Render/Foreign")
    foreign.succeeded()
    resolved(foreign, DEFAULTS)
    if host.speaks:
        complaints = foreign.complaints()
        check(len(complaints) == 1 and "authors no cenote: setting" in complaints[0],
              f"foreign: expected one sentence about a stage that is not ours, "
              f"said {complaints}")
        check(not any("karma" in line or "ri:" in line for line in complaints),
              f"foreign: another renderer's settings are not ours to judge — {complaints}")

    # 6. The debug override, at the top of the chain: the same impossible
    #    budget as leg 2, settling at once because the environment says so.
    forced = run("forced", settings_prim="/Render/Slow",
                 overrides={"CENOTE_SERVER_MAX_SAMPLES": "8"}, window=window)
    forced.succeeded()
    # The delegate still resolves and sends the shot's budget: the override
    # is the server's, applied to whatever the scene said, so the line
    # underneath it is unchanged and only the render is shorter.
    resolved(forced, "1000000 samples per pixel, no early stop, 8 bounces")

    said = ("clamp and complaints posted, foreign stage named once"
            if host.speaks else "diagnostics unread (Houdini keeps them)")
    print(f"render settings: OK ({host.name}) — defaults {defaults.seconds:.1f}s, the shot's "
          f"budget still running at {window:.1f}s, early stop {early.seconds:.1f}s, "
          f"{said}, environment override {forced.seconds:.1f}s")


if __name__ == "__main__":
    main()
