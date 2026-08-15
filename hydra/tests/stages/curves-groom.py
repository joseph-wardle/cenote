#!/usr/bin/env hython
"""Regenerates curves-groom.usda — the Houdini fixture — under hython.

    /opt/hfs21.0.671/bin/hython hydra/tests/stages/curves-groom.py

The point of this file is that nothing in it is hand-authored. A groom is
whatever Houdini decides to write, and what Houdini decides is the thing
under test: the delegate must read the tokens a DCC actually emits, not
the ones a spec-reading human would. Two habits of the exporter were found
this way and are pinned by the fixture:

  * Houdini leaves `basis` **empty** on a linear BasisCurves prim. USD says
    the basis is meaningless there, and Houdini takes it at its word. A
    reader that maps basis tokens before looking at the type refuses the
    whole groom.
  * The fur SOP's order-4 NURBS guides come out as `cubic`/`bspline`/
    `nonperiodic` with nine control vertices each — *not* pinned, so the
    strands pull away from the skin they grow out of. That is Houdini's
    choice, faithfully reproduced, and it is why a b-spline groom looks
    detached at the roots.

The two legs are the two shapes a groom arrives in: the fur SOP's cubic
guides, and hairgen's dense linear hairs. Both carry per-point widths that
taper root to tip, so the width stream lands on `vertex` interpolation —
the one interpolation that has to travel through the curve's own basis —
and the linear leg's taper reaches exactly zero, which is where the
tessellator collapses its tip ring to a point.
The rows Houdini will not write (pinned wrap, bezier, the other three
width interpolations, and a periodic curve to be refused) are hand-authored
alongside the reference in curves-stage.usda.

Kept small enough to live in the repository forever: about 120 curves a
leg, a couple of thousand points in total.
"""

import os

import hou

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "curves-groom.usda")

geo = hou.node("/obj").createNode("geo", "groom")

# The skin both legs grow from: a half-meter sphere, coarse enough that the
# scatter is even and the file stays small.
skin = geo.createNode("sphere")
skin.parm("type").set(2)  # polygon mesh
skin.parmTuple("rad").set((0.5, 0.5, 0.5))
skin.parm("rows").set(20)
skin.parm("cols").set(20)

# Leg one: the fur SOP's NURBS guides, which export as cubic b-splines.
fur = geo.createNode("fur")
fur.setFirstInput(skin)
fur.parm("density").set(90)
fur.parm("length").set(0.25)
fur_width = geo.createNode("attribwrangle", "fur_width")
fur_width.setFirstInput(fur)
fur_width.parm("class").set(1)  # primitives, so @primnum means something
# Root to tip along each guide, in the ratio a real fiber tapers by.
fur_width.parm("snippet").set(
    "int count = primvertexcount(0, @primnum);\n"
    "for (int i = 0; i < count; i++) {\n"
    "    setpointattrib(0, \"width\", primpoint(0, @primnum, i),\n"
    "                   fit01(float(i) / (count - 1), 0.012, 0.003));\n"
    "}"
)

# Leg two: hairgen's interpolated hairs, which export as linear polylines.
# On its own skin, a meter and a half to the right, so the two legs stand
# side by side in the frame and a render can tell which one is missing.
hair_skin = geo.createNode("xform", "hair_skin")
hair_skin.setFirstInput(skin)
hair_skin.parmTuple("t").set((1.5, 0, 0))
guides = geo.createNode("guideinit")
guides.setInput(1, hair_skin)
hair = geo.createNode("hairgen::2.0")
hair.setInput(0, hair_skin)
hair.setInput(1, guides)
hair.parm("density").set(40)
# hairgen writes its own `width`, tapering to exactly zero at the tip —
# a real groom's shape, and the tessellator's collapsed tip ring. The
# taper is kept and only its scale is raised, because at hairgen's own
# millimeter the hairs are thinner than a pixel of the checkpoint frame
# and the leg would be proven by an empty stretch of black.
hair_width = geo.createNode("attribwrangle", "hair_width")
hair_width.setFirstInput(hair)
hair_width.parm("class").set(2)  # points
hair_width.parm("snippet").set("f@width *= 6;")

stage = hou.node("/stage")
cubic = stage.createNode("sopimport", "cubic")
cubic.parm("soppath").set(fur_width.path())
cubic.parm("pathprefix").set("/groom/cubic")
linear = stage.createNode("sopimport", "linear")
linear.parm("soppath").set(hair_width.path())
linear.parm("pathprefix").set("/groom/linear")
linear.setInput(0, cubic)

rop = stage.createNode("usd_rop")
rop.setInput(0, linear)
rop.parm("lopoutput").set(OUT)
# One flat layer: a sopimport writes its geometry to a sidecar layer named
# after the SOP path otherwise, and the fixture has to be one committed file.
rop.parm("savestyle").set("flattenstage")
rop.parm("flattensoplayers").set(1)
rop.render()

for node, label in ((fur_width, "cubic"), (hair_width, "linear")):
    geometry = node.geometry()
    print(f"{label}: {len(geometry.prims())} curves, {len(geometry.points())} points")
print(f"wrote {OUT}")
