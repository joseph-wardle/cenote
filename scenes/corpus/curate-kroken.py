#!/usr/bin/env python3
"""Regenerate scenes/corpus/kroken.ron locally — the ND-licensed scene the
repo cannot carry.

Kroken is CC-BY-ND 2.0 (Angelo Ferretti, via pbrt-v4-scenes): the license
grants format shifts "technically necessary to exercise the rights in other
media and formats" but no right to distribute Derivative Works, and a curated
RON (substituted materials, one picked camera, documented degradations) is a
derivative. So unlike every other corpus scene, the RON here is a *derived
asset*: this script rebuilds it from the untracked sources, and only the
script — which carries no scene content beyond material names and a handful
of interoperability constants — is committed. `./fetch.sh kroken` first.

Usage: python3 scenes/corpus/curate-kroken.py   (from anywhere)
"""

import re
import subprocess
import sys
from pathlib import Path

from field_order import FIELD_ORDER

CORPUS = Path(__file__).resolve().parent
ROOT = CORPUS.parent.parent
KROKEN = CORPUS / "sources/pbrt-v4-scenes/kroken"
RON = CORPUS / "kroken.ron"
DOTS = CORPUS / "kroken-dots.exr"

HEADER = """\
// Kroken — living-room interior (camera-1, the pbrt-v4-scenes showcase view).
//
// Provenance: pbrt-v4-scenes @ 30cf4a0, kroken/camera-1.pbrt; scene licensed
// from Angelo Ferretti (lucydreams.it), CC-BY-ND 2.0. THIS FILE IS NOT
// COMMITTED: the ND term withholds the right to distribute derivatives, and a
// curated conversion is one — regenerate it with scenes/corpus/curate-kroken.py
// (the script and this header are cenote's own text; the scene stays in the
// untracked sources). Oracle: pbrt volpath maxdepth 15, 1500x1500.
//
// Import-time degradations, documented (unlock in parentheses):
//  - planar texture mappings x14 and UV affine transforms x4 sample authored
//    UVs directly — books/magazines wear wrong-scaled art (texture mapping
//    modes); the two directionmix covers and two mix textures land as
//    mid-gray and are curated to linearized image means below.
//  - displacement / bump maps dropped scene-wide (rug, blanket, pillow,
//    concrete) — bathroom/kitchen/crown class.
//  - the x3/x4 value-scales on the exterior concrete import faithfully
//    (linearized texels stay under 0.25, so pbrt's reflectance clamp at 1
//    never fires and the unclamped sample-time scale agrees exactly) —
//    but the correctly-brighter pavement amplifies the portal gap below:
//    the full dome lights it where pbrt's portal keeps it dark, so its
//    bounce into the room is the dominant term of the pbrt divergence.
//  - diffusetransmission (Carpet, Blanket - Fringe) imports opaque diffuse.
//  - homogeneous media: the red-glass vases carry the source medium
//    exactly — sigma_t (0.8, 10, 10), sigma_s (0.4, 5, 5), isotropic —
//    as transmission_color exp(-0.2*sigma_t) at depth 0.2 with
//    transmission_scatter 0.2*sigma_s (a depth-1 bake of exp(-10) =
//    4.5e-5 sits under the importer's 1e-4 color clamp);
//    greenish-glass ("Glass - Set") extinction ~1e-4/unit, left clear.
//  - "Fabric - Pillow" is a pbrt mix material (blue coateddiffuse / white
//    diffuse by the Dots_Pillow mask): curated to one diffuse wearing a
//    baked lerp of the two reflectances (kroken-dots.exr, generated here).
//  - the window is a portal'd uniform 6500 K infinite light; cenote drops
//    the portal and lights a full dome. pbrt's portal RESTRICTS the env to
//    window directions (kroken leaks ~15% extra env light without it), so
//    the two env domains genuinely differ.
//  - the invisible sun ball (alpha 0 sphere, scale 50) imports faithfully
//    through the cutout fork (geometry_opacity 0 + emission), but
//    cenote's MIS discounts NEE against a BSDF strategy that can never hit
//    an invisible emitter — the sun lands at HALF strength (microtest:
//    2.160 vs pbrt 4.213, which matches its own visible-emitter case to 3
//    decimals). Renderer fix candidate.
//  - coateddiffuse is the dominant material family (36 of 59) — pbrt's
//    stochastically simulated coat darkens this white GI box 19% of scene
//    mean vs OpenPBR's analytic 4.4% (the bistro model gap, GI-amplified);
//    partly offset by cenote's conductors relaying less window light off
//    the metal shelving (the veach-ajar Fresnel class — an all-diffuse
//    bisect agrees to ±3% where the real material set differs 14%).
"""

# Linearized source-texture means (magick -colorspace RGB), used where an
# unsupported texture class (mix / directionmix / planar mapping) left a
# camera-visible prop mid-gray. Values are linear Rec.709 like every Constant.
PATCHES = {
    # pbrt: mix("Fabric - Pillow - blue", "... - white") by Dots_Pillow.
    "Fabric - Pillow": {
        "base_color": 'Some(Texture((\n'
        '                path: "kroken-dots.exr",\n'
        '                color_space: None,\n'
        '                channel: None,\n'
        '            )))',
        "specular_weight": "Some(0.0)",
        "geometry_normal": 'Some(Some((\n'
        '                path: "sources/pbrt-v4-scenes/kroken/textures/Pillow_bump.png",\n'
        '                color_space: None,\n'
        '                channel: None,\n'
        '            )))',
    },
    # Source medium exactly: sigma_a = sigma_s = 200*(0.002, 0.025, 0.025),
    # so sigma_t = (0.8, 10, 10), isotropic. Depth 0.2 because a depth-1
    # bake of exp(-10) = 4.5e-5 sits under the importer's 1e-4 color clamp
    # (sigma_t silently floored at 9.21); at 0.2 the bake exp(-(0.16, 2, 2))
    # clears it and -ln(color)/depth recovers the source sigma_t exactly.
    # scatter = sigma_s * depth.
    "Glass - Red": {
        "transmission_color": "Some((0.852144, 0.135335, 0.135335))",
        "transmission_depth": "Some(0.2)",
        "transmission_scatter": "Some((0.08, 1.0, 1.0))",
    },
    # directionmix covers: the visible face is the top, use the cover mean.
    "Magazine - version 1": {"base_color": "Some(Constant((0.672, 0.675, 0.681)))"},
    "Magazine - version 2-middle-left-top": {
        "base_color": "Some(Constant((0.660, 0.661, 0.680)))"
    },
    # mix(dark, light) by Book_Cover_2 (mean 0.035): stays near the dark arm.
    "Book 03 - Cover ": {"base_color": "Some(Constant((0.210, 0.164, 0.164)))"},
    # mix((.2,.2,.2), (.7,.55,.55)) by Box_Diffuse_2 (mean 0.117).
    "Box": {"base_color": "Some(Constant((0.258, 0.241, 0.241)))"},
    # mix((.85, .7, .6), (.7, .7, .7)) by the blanket mask (mean 0.583).
    "Fabric - Blanket": {"base_color": "Some(Constant((0.763, 0.700, 0.658)))"},
    # planar-projected pages.png has no usable UVs here — its mean.
    "Books - Pages": {"base_color": "Some(Constant((0.185, 0.169, 0.145)))"},
    "Books - Pages.1": {"base_color": "Some(Constant((0.185, 0.169, 0.145)))"},
}


def run(cmd, **kwargs):
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kwargs)


def patch_material(text, name, fields):
    start = text.index(f'name: "{name}",')
    end = text.index("Material((", start) if "Material((" in text[start:] else len(text)
    block = text[start:end]
    for field, value in fields.items():
        nxt = FIELD_ORDER[FIELD_ORDER.index(field) + 1] if field != "geometry_normal" else None
        tail = rf"\n            {nxt}:" if nxt else r"\n        \)\),"
        pattern = re.compile(rf"            {field}:.*?(?={tail})", re.DOTALL)
        block, n = pattern.subn(f"            {field}: {value},"[:-1] + ",", block, count=1)
        if n != 1:
            sys.exit(f"patch failed: {name} / {field}")
    return text[:start] + block + text[end:]


def main():
    if not KROKEN.is_dir():
        sys.exit("kroken sources missing — run scenes/corpus/fetch.sh kroken first")

    run(
        ["cargo", "run", "--release", "-p", "cenote-cli", "--", "import",
         str(KROKEN / "camera-1.pbrt"), "--out", str(RON)],
        cwd=ROOT,
    )

    # Bake the pillow: lerp(blue (0.02, 0.015, 0.1), white (0.8), mask), the
    # mask linearized the way pbrt reads an 8-bit float imagemap — to linear
    # Rec.709 named outright, never OCIO's `linear` role, which every
    # current config points at ACEScg (see curate-watercolor.py). This one
    # bake is indifferent: it keeps a single channel of a gray mask, and
    # gray is a fixed point of the conversion. Its neighbours were not.
    run(
        ["oiiotool", str(KROKEN / "textures/Dots_Pillow.png"),
         "--colorconvert", "srgb", "lin_rec709", "--ch", "0,0,0",
         "--chnames", "R,G,B",
         "--mulc", "0.78,0.785,0.7", "--addc", "0.02,0.015,0.1",
         "-d", "half", "-o", str(DOTS)],
    )

    text = RON.read_text()
    for name, fields in PATCHES.items():
        text = patch_material(text, name, fields)
    if "spp: Some(16)," not in text:
        sys.exit("patch failed: settings spp")
    text = text.replace("spp: Some(16),", "spp: Some(256),", 1)
    RON.write_text(HEADER + text)
    print(f"wrote {RON} (local only — see header for why it is not committed)")


if __name__ == "__main__":
    main()
