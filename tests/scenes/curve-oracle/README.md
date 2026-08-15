# The curve oracle

Four pbrt-v4 scenes and a comparator that hold cenote's `Curves` against a
second, independent implementation of `Shape "curve"`. Run them with
[`scripts/curve-oracle.sh`](../../../scripts/curve-oracle.sh); it needs a
pbrt-v4 binary and says so if it can't find one.

Every other gate in this repo compares cenote against cenote: the corpus
byte-compares its own renders, the goldens FLIP against its own images. Those
catch regression and nothing else — a misreading of the input that was wrong on
the day it was written passes all of them forever. This is the one gate where
something else does the reading.

## What is compared, and why it is not pixels

cenote renders a curve as a tessellated three-sided tube. pbrt intersects a
procedural swept curve. The two are *not* the same surface, and no threshold on
image difference can separate "cenote read the width wrong" from "a prism is
not a cylinder". So the scenes remove everything that is not geometry, and the
statistics are chosen to be blind to what is left:

- **The material reflects nothing** and the only light is a constant infinite
  dome, so every pixel is `L * (1 - coverage)`.
- **`L` cancels**, because the background level is read from each image
  separately. This measures geometry, not radiometry.
- **The strands are isolated** — 4 pixels wide on a 16-pixel pitch — so each
  contributes its own run of ink and can be located individually. A dense groom
  cannot do this: coverage saturates at the silhouette of the fur shell and
  goes nearly blind to how wide any one strand is.

`compare.py` states the four statistics, and its `LIMITS` are the gate.

## The scenes

| | exercises |
|---|---|
| `s1.pbrt` | straight, constant width, `cylinder` — width in its purest form: no cap in shot, so the image is a vertical grating |
| `s2.pbrt` | straight, 8:1 taper — `width0`/`width1` and their direction along u |
| `s3.pbrt` | strongly curved, 7 control vertices — two bezier spans and the stitch between them |
| `s4.pbrt` | b-spline, 6 control vertices, `flat` — the approximating basis, where the control polygon is not the curve |

All four are authored at half size under a `Scale 2 2 2`, so a dropped or
mis-composed CTM is a failure the instrument can see, and the scale is proven
to reach the *radii* and not only the control points. Each rank is 256 strands
because a single one is not a measurement: the ring's phase is random per
strand, which spreads its projected width across ±10%. The pitch is 16 pixels
and not 8 because at 8 adjacent strands touched, merging their runs and
partly saturating the ink — S2's root is 8 pixels across, S3 and S4 wander 2
either side of centre, and cenote's rings are circumscribed besides.

The scenes are vendored whole, one strand per line, rather than generated: both
renderers are pointed at the same bytes, and a statistic is only comparable
against an exact scene.

## Expected differences that are *not* failures

- Per-strand width scatter of ±10%, from the ring phase, as above.
- Silhouette shape at the sub-pixel level: a prism's edge is faceted.
- End caps. cenote closes a tube with a flat ring; pbrt's curve ends on the
  surface it sweeps. S1 and S2 keep both ends out of frame for that reason.
- Anything shading-related. That is deliberately not this gate's business.

## Limits worth knowing

The strands run off the top and bottom of every frame, so a pure vertical
translation of a whole rank is invisible here. A dropped or mis-composed
transform shows up as ink or `drift` instead, which is what the `Scale 2 2 2`
is for.

Below `FLATNESS` of the local radius, the tessellator discards centerline
detail before any pixel exists, so nothing image-based can resolve it. That
floor is pinned directly, in `scene::curves`.

## Upgrading pbrt

The numbers are an agreement between two implementations, so a new pbrt can
legitimately move them. They were last confirmed against pbrt-v4 `5f7a606`
(2026-06-14), the same commit `../README.md` cites. If a limit breaks after an
upgrade, attribute the move before widening anything.
