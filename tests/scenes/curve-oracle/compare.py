#!/usr/bin/env python3
"""Four numbers per scene, from two renders of a black strand under a white dome.

Each image is an occlusion mask and nothing else: a zero-reflectance material
under a constant infinite light makes every pixel `L * (1 - coverage)`, so the
BSDF leaves the comparison and the two renderers can be asked about geometry
alone. `L` cancels too, because the background level is read from each image
separately -- which is what lets pbrt's spectral pipeline and cenote's RGB one
be held to one standard without converting either.

Run through `scripts/curve-oracle.sh`, which renders both sides first.
"""

import os
import subprocess
import sys

import numpy as np

SCENES = ("s1", "s2", "s3", "s4")

# How far each statistic may sit from agreement.
#
# Neither run is noisy -- both renderers are deterministic, and pbrt against
# itself at two seeds measures every statistic at zero -- but the scenes still
# sit on a lottery. cenote gives each strand's ring a random phase, and a
# triangle's projected width swings 4.2% about its mean with that phase, so a
# rank of 256 draws a mean width with a 0.26% standard deviation. That is the
# floor these limits have to clear, and it is redrawn by any change to the
# per-strand hash rather than by anything the gate is looking for. So ink sits
# at about four of those, and the rest a comparable distance out from a
# measured 0.0024 / 0.0001 / 0.23 px. Still orders of magnitude inside a width,
# basis, taper or transform bug: none of those misses by less than fifteen
# times a limit.
LIMITS = {"ink": 0.01, "KS_x": 0.01, "KS_y": 0.01, "drift": 0.75}


def read(path):
    """Luminance plane of an EXR, via oiiotool -> pfm."""
    pfm = path + ".oracle.pfm"
    try:
        subprocess.run(
            ["oiiotool", path, "--ch", "R", "-o", pfm], check=True, capture_output=True
        )
    except FileNotFoundError:
        sys.exit("curve-oracle: needs `oiiotool` on PATH to read EXRs")
    with open(pfm, "rb") as f:
        assert f.readline().strip() == b"Pf"
        w, h = (int(v) for v in f.readline().split())
        scale = float(f.readline())
        order = "<f4" if scale < 0 else ">f4"
        data = np.frombuffer(f.read(w * h * 4), dtype=order).reshape(h, w)
    os.remove(pfm)
    return np.flipud(data).astype(np.float64)


def masked(images):
    """Coverage for several images at once, on their common support.

    Two nuisances, each on its own larger than what is being measured. Ink is
    a sum over a frame that is 99% background, so a 0.1% error in the
    background level moves it by 16 times that -- and pbrt writes half floats,
    whose spacing near 1.0 is already 0.1%. Hence the level is read only from
    pixels no strand comes near. And clipping coverage at zero would keep the
    pixels that fell below the background while discarding those above,
    rectifying pbrt's spectral noise into 1.7% of extra ink -- hence
    unclipped, and summed only over the support.
    """
    rough = []
    for image in images:
        upper = image[image > 0.5 * np.percentile(image, 99.0)]
        rough.append(1.0 - image / float(np.median(upper)))
    support = np.zeros_like(rough[0], dtype=bool)
    for cov in rough:
        support |= cov > 0.02
    # One pixel of slack, so a strand's own antialiased edge is inside the sum
    # rather than in the sample the background is read from.
    grown = support.copy()
    for axis in (0, 1):
        for shift in (-1, 1):
            grown |= np.roll(support, shift, axis=axis)

    out = []
    for image in images:
        background = float(image[~grown].mean())
        coverage = np.where(grown, 1.0 - image / background, 0.0)
        out.append(
            {
                "ink": float(coverage.sum()),
                "background": background,
                "x": coverage.sum(axis=0),
                "y": coverage.sum(axis=1),
            }
        )
    return out


def ks(a, b):
    """Largest gap between two normalised cumulative marginals."""
    ca, cb = np.cumsum(a), np.cumsum(b)
    return float(np.max(np.abs(ca / ca[-1] - cb / cb[-1])))


def centroids(marginal):
    """The coverage-weighted centre of every isolated run of ink.

    A rank of strands separated by empty columns gives one run per strand, so
    this locates each strand without knowing the scene's layout. The threshold
    is relative because pbrt samples four wavelengths per ray, which leaves
    its empty columns a little above zero rather than on it.
    """
    lit = marginal > 0.01 * marginal.max()
    edges = np.flatnonzero(np.diff(np.concatenate(([False], lit, [False]))))
    index = np.arange(len(marginal), dtype=np.float64)
    return [
        float((index[a:b] * marginal[a:b]).sum() / marginal[a:b].sum())
        for a, b in zip(edges[0::2], edges[1::2])
    ]


def drift(a, b):
    """RMS displacement between matched runs of ink, in pixels.

    KS compares distributions, and a comb of evenly spaced strands has very
    nearly the same distribution after being slid sideways -- measured, a
    one-pixel shift of a whole rank moves KS_x by 0.001. This is the statistic
    that sees it. NaN when the two images do not even hold the same number of
    strands, which no distance describes and every check below fails on.
    """
    ca, cb = centroids(a), centroids(b)
    if len(ca) != len(cb):
        return float("nan")
    return float(np.sqrt(np.mean((np.array(ca) - np.array(cb)) ** 2)))


def measure(directory, scene):
    """cenote against pbrt on one scene.

    ink     total coverage, cenote / pbrt. Widths, radii, taper and the
            diameter-versus-radius convention. Blind to placement.
    KS_x    largest gap between the cumulative coverage profiles across the
            frame: the shape of the rank.
    KS_y    the same down the frame. A swapped width0/width1 moves this and
            nothing else here does.
    drift   rigid displacement, in pixels.
    """
    cenote, pbrt = masked(
        [
            read(os.path.join(directory, f"{scene}-cenote.exr")),
            read(os.path.join(directory, f"{scene}-pbrt.exr")),
        ]
    )
    return {
        "ink": cenote["ink"] / pbrt["ink"],
        "KS_x": ks(cenote["x"], pbrt["x"]),
        "KS_y": ks(cenote["y"], pbrt["y"]),
        "drift": drift(cenote["x"], pbrt["x"]),
    }


def main():
    directory = sys.argv[1]
    print("scene    " + " ".join(f"{key:>9}" for key in LIMITS))
    failed = []
    for scene in SCENES:
        measured = measure(directory, scene)
        print(f"{scene:<8} " + " ".join(f"{measured[key]:9.4f}" for key in LIMITS))
        for key, limit in LIMITS.items():
            # Agreement is ink 1 and the rest 0. Written so that NaN fails.
            value = measured[key]
            distance = abs(value - 1.0) if key == "ink" else value
            if not distance <= limit:
                failed.append(f"{scene} {key} = {value:.4f}")

    for line in failed:
        print(f"FAIL    {line}")
    if failed:
        sys.exit(1)
    print("curve-oracle: pass")


if __name__ == "__main__":
    main()
