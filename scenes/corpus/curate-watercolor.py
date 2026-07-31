#!/usr/bin/env python3
"""Regenerate scenes/corpus/watercolor.ron locally — the second ND-licensed
scene the repo cannot carry (the pattern curate-kroken.py set).

Watercolor is CC-BY-ND 2.0 (Angelo Ferretti, via pbrt-v4-scenes): the license
grants format shifts "technically necessary to exercise the rights in other
media and formats" but no right to distribute Derivative Works, and a curated
RON is a derivative. So the RON and the handful of baked textures beside it
are derived assets this script rebuilds from the untracked sources; only the
script — material names and interoperability constants, no scene content —
is committed. `./fetch.sh watercolor` first.

Usage: python3 scenes/corpus/curate-watercolor.py   (from anywhere)
"""

import re
import subprocess
import sys
from pathlib import Path

CORPUS = Path(__file__).resolve().parent
ROOT = CORPUS.parent.parent
WATERCOLOR = CORPUS / "sources/pbrt-v4-scenes/watercolor"
RON = CORPUS / "watercolor.ron"

HEADER = """\
// Watercolor — attic art studio (camera-1, the landscape showcase view).
//
// Provenance: pbrt-v4-scenes @ 30cf4a0, watercolor/camera-1.pbrt; scene
// licensed from Angelo Ferretti (lucydreams.it), CC-BY-ND 2.0. THIS FILE IS
// NOT COMMITTED: the ND term withholds the right to distribute derivatives,
// and a curated conversion is one — regenerate it (and the watercolor-*.exr
// bakes beside it) with scenes/corpus/curate-watercolor.py. Oracle: pbrt
// volpath maxdepth 15, 1920x1080... (1920x1440).
//
// This rung's evidence landed the UV feature the campaign had parked: the
// easel drawing's affine remap, 28 texture value scales, and the three
// paint-splatter bilinear decals all import faithfully now.
//
// Import-time degradations, documented (unlock in parentheses):
//  - texture mapping modes x14 — cylindrical x9, planar x5 — sample authored
//    UVs instead of position-projected ones (mapping modes): wall pictures,
//    the paint plate, and small props wear differently-scaled art.
//  - mix textures land mid-gray and are curated below: the walls' near-white
//    dirt blend to its constant, the desk wood / jute rug / concrete floor
//    to bakes (watercolor-noce/carpet/concrete.exr) since they are the
//    scene's largest surfaces (procedural texture baker).
//  - inverted masks read direct (texture `invert`): the floor splatter
//    decals' cutout masks are baked pre-inverted (watercolor-spot01/10.exr)
//    so the paint spots show instead of their complement.
//  - mix materials (Tin 05, Case gold.16, Spot catcher, Paper 01-script,
//    the two drippy tins) import as OpenPBR defaults and are curated to
//    blends of their arms; Tin 05's dirt mask becomes textured metalness.
//  - the TiO2 conductor (white paint tubes) is not in the metal table and
//    falls back to copper — curated to a neutral F0 0.198.
//  - the brush-water medium ("liquid") is curated onto the cup's glass as
//    Beer-Lambert transmission exp(-80*(sigma_a+sigma_s)) at depth 1 —
//    APPROXIMATE, sigma_s is nonzero so pbrt also in-scatters (M8 volumes).
//  - "Leaves" diffusetransmission imports opaque (kitchen's Blinds class).
//  - displacement dropped scene-wide (bathroom/kitchen/crown class).
//  - the room is lit ENTIRELY by two portal'd copies of one infinite light
//    (one per skylight; the two blackbody area lights in the source have
//    their shapes commented out). cenote keeps the first light and lights a
//    full dome: the portal RESTRICTS pbrt's env to window directions, so the
//    domains genuinely differ (kroken's class, dominant here) — the full
//    dome adds blue overhead sky the portals never admit.
//  - coateddiffuse is the dominant material family (150 of 258) — pbrt's
//    stochastically simulated coat darkens multi-bounce interiors vs
//    OpenPBR's analytic coat (the bistro/kroken model gap).
"""

# Linear Rec.709 constants below derive from source-texture means
# (magick -colorspace RGB) pushed through the pbrt mix/scale arithmetic
# they replace; comments carry the recipe.
PATCHES = {
    # The walls: mix(0.1*Dirt_08, 0.88, amount .9) has texture amplitude
    # .01 — its constant, .1*(.1*.2366) + .9*.88.
    "Plaster": {"base_color": "Some(Constant((0.794, 0.794, 0.794)))"},
    "White - shiny _dupe": {"base_color": "Some(Constant((0.794, 0.794, 0.794)))"},
    "White - shiny _dupe_dupe": {"base_color": "Some(Constant((0.794, 0.794, 0.794)))"},
    # Stacked canvases: mix(.3*canvas, white, .6) = .4*.3*.2965 + .6.
    "Canvas 02": {"base_color": "Some(Constant((0.636, 0.636, 0.636)))"},
    # Floor cushion: mix(peach, .6 gray, .3).
    "Cushion Fabric": {"base_color": "Some(Constant((0.528, 0.364, 0.246)))"},
    # Art case: mix(fabric, (.35,.45,.4), .5).
    "Case Fabric": {"base_color": "Some(Constant((0.353, 0.307, 0.249)))"},
    # mix(cylindrical Dirt_09, .4 gray, .5) — the mapping is gone anyway.
    "Dirt_plastic": {"base_color": "Some(Constant((0.359, 0.359, 0.359)))"},
    # The desk (Noce Canaletto walnut): baked 1.7*noce + 0.06.
    "Table wood": {
        "base_color": 'Some(Texture((\n'
        '                path: "watercolor-noce.exr",\n'
        '                color_space: None,\n'
        '                channel: None,\n'
        '            )))',
    },
    "Table wood.3": {
        "base_color": 'Some(Texture((\n'
        '                path: "watercolor-noce.exr",\n'
        '                color_space: None,\n'
        '                channel: None,\n'
        '            )))',
    },
    # The jute rug: baked base * (0.5*AO + 0.5) — the AO ride was a
    # textured scale factor, inexpressible as a reference.
    "Carpet 02": {
        "base_color": 'Some(Texture((\n'
        '                path: "watercolor-carpet.exr",\n'
        '                color_space: None,\n'
        '                channel: None,\n'
        '            )))',
    },
    # The concrete floor: baked (concrete*.8 + (.1,.09,.08)) * AO.
    "Material.2": {
        "base_color": 'Some(Texture((\n'
        '                path: "watercolor-concrete.exr",\n'
        '                color_space: None,\n'
        '                channel: None,\n'
        '            )))',
    },
    # White paint tubes: TiO2 eta/k are not in the metal table; a neutral
    # F0 ((2.6-1)/(2.6+1))^2 replaces the copper fallback.
    "Metal - tins _dupe_dupe_dupe_dupe": {
        "base_color": "Some(Constant((0.198, 0.198, 0.198)))"
    },
    # The brush-water: exp(-80*(sigma_a+sigma_s)) = exp(-(.48,.88,.4)).
    "Liquid - glass": {
        "transmission_color": "Some((0.619, 0.415, 0.670))",
        "transmission_depth": "Some(1.0)",
    },
    "Liquid - glass_dupe": {
        "transmission_color": "Some((0.619, 0.415, 0.670))",
        "transmission_depth": "Some(1.0)",
    },
    # mix(dark diffuse .1, default-conductor tin, 1.5*Dirt_03): the mask
    # becomes textured metalness (crown's pattern, exact now that the
    # reference carries the 1.5), copper F0 as the metal arm.
    "Tin 05": {
        "base_color": "Some(Constant((0.95, 0.64, 0.54)))",
        "base_metalness": 'Some(Texture((\n'
        '                path: "sources/pbrt-v4-scenes/watercolor/textures/Dirt_03_i04042021.png",\n'
        '                color_space: None,\n'
        '                channel: None,\n'
        '                scale: Some(1.5),\n'
        '            )))',
        "specular_roughness": "Some(Constant(0.3))",
    },
    # mix(gold conductor, tarnish diffuse, cylindrical tarnish mask):
    # the clasps stay gold.
    "Case gold.16": {
        "base_color": "Some(Constant((1.0, 0.76, 0.33)))",
        "base_metalness": "Some(Constant(1.0))",
        "specular_roughness": "Some(Constant(0.15))",
    },
    # mix(paint-spot texture, beige base, splatter mask): the base arm.
    "Spot catcher": {
        "base_color": "Some(Constant((0.47, 0.405, 0.32)))",
        "specular_roughness": "Some(Constant(0.25))",
    },
    # mix(paper texture, handwriting overlay, script mask): the paper arm.
    "Paper 01-script": {
        "base_color": 'Some(Texture((\n'
        '                path: "sources/pbrt-v4-scenes/watercolor/textures/Paper_01_Diffuse_i04042021.png",\n'
        '                color_space: None,\n'
        '                channel: None,\n'
        '            )))',
    },
    # mix(white paint coateddiffuse .8, tin coatedconductor .25, drippy
    # cylindrical masks): mostly paint.
    "Metal - tins - drippy-lower-right": {
        "base_color": "Some(Constant((0.8, 0.8, 0.8)))",
        "base_metalness": "Some(Constant(0.3))",
        "specular_roughness": "Some(Constant(0.2))",
    },
    "Metal - tins - drippy-upper-left": {
        "base_color": "Some(Constant((0.8, 0.8, 0.8)))",
        "base_metalness": "Some(Constant(0.3))",
        "specular_roughness": "Some(Constant(0.2))",
    },
    # The splatter decals' masks are authored inverted ("bool invert"):
    # the bakes below carry 1 - mask so the spots show, not their
    # complement (spot10's 1.04 scale folds in pre-inversion, as pbrt
    # applies it).
    "coateddiffuse-0-cutout-0": {
        "geometry_opacity": 'Some(Texture((\n'
        '                path: "watercolor-spot01.exr",\n'
        '                color_space: None,\n'
        '                channel: None,\n'
        '            )))',
    },
    "coateddiffuse-2-cutout-0": {
        "geometry_opacity": 'Some(Texture((\n'
        '                path: "watercolor-spot10.exr",\n'
        '                color_space: None,\n'
        '                channel: None,\n'
        '            )))',
    },
    "Desk splatter-cutout-0": {
        "geometry_opacity": 'Some(Texture((\n'
        '                path: "watercolor-spot10.exr",\n'
        '                color_space: None,\n'
        '                channel: None,\n'
        '            )))',
    },
}

# Field order as the importer emits it — each replacement is anchored on the
# next field so multi-line values (texture refs) are replaced whole.
FIELD_ORDER = [
    "base_color", "base_diffuse_roughness", "base_metalness", "specular_weight",
    "specular_roughness", "specular_ior", "transmission_weight",
    "transmission_color", "transmission_depth", "coat_weight", "coat_color",
    "coat_roughness", "coat_ior", "coat_darkening", "fuzz_weight", "fuzz_color",
    "fuzz_roughness", "emission_luminance", "emission_color",
    "geometry_opacity", "geometry_thin_walled", "geometry_normal",
]


def run(cmd, **kwargs):
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kwargs)


def bake(args, out):
    run(["oiiotool", *args, "-d", "half", "-o", str(CORPUS / out)])


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
    if not WATERCOLOR.is_dir():
        sys.exit("watercolor sources missing — run scenes/corpus/fetch.sh watercolor first")

    run(
        ["cargo", "run", "--release", "-p", "cenote-cli", "--", "import",
         str(WATERCOLOR / "camera-1.pbrt"), "--out", str(RON)],
        cwd=ROOT,
    )

    tex = WATERCOLOR / "textures"
    lin = ["--colorconvert", "srgb", "linear"]
    # The AO masks are single-channel; broadcast them to R,G,B before a
    # per-channel --mul (a 3ch × 1ch multiply collapses to red alone).
    rgb = ["--ch", "0,0,0", "--chnames", "R,G,B"]
    # The desk walnut: pbrt's mix(2*noce, .4 gray, .15) = 1.7*noce + .06.
    bake([str(tex / "Noce_Canaletto_i04042021.png"), *lin,
          "--mulc", "1.7", "--addc", "0.06"], "watercolor-noce.exr")
    # The rug: base * (0.5*AO + 0.5) — pbrt's AO-biased textured scale.
    bake([str(tex / "Carpet_4_BaseColor_i04042021.png"), *lin,
          str(tex / "Carpet_4_AO_i04042021.png"), *lin, *rgb,
          "--mulc", "0.5", "--addc", "0.5", "--mul"], "watercolor-carpet.exr")
    # The floor: (concrete*.8 + (.1,.09,.08)) * AO.
    bake([str(tex / "Concrete_2_Base_Color_i04042021.png"), *lin,
          "--mulc", "0.8", "--addc", "0.1,0.09,0.08",
          str(tex / "Concrete_2_Ambient_Occlusion_i04042021.png"), *lin, *rgb,
          "--mul"], "watercolor-concrete.exr")
    # The splatter cutout masks, pre-inverted (pbrt inverts after its
    # scale): 1 - mask and 1 - 1.04*mask, floored at zero. The opacity
    # slot reads red, but cenote's EXR decode needs named R,G,B — the
    # single grayscale channel replicates across all three (kroken's
    # dots-bake fix).
    grayscale = ["--ch", "0,0,0", "--chnames", "R,G,B"]
    bake([str(tex / "Spot_Floor_01_i04042021.png"), *lin,
          "--mulc", "-1", "--addc", "1", "--clamp:min=0", *grayscale],
         "watercolor-spot01.exr")
    bake([str(tex / "Spot_Floor_10_i04042021.png"), *lin,
          "--mulc", "-1.04", "--addc", "1", "--clamp:min=0", *grayscale],
         "watercolor-spot10.exr")

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
