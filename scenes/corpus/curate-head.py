#!/usr/bin/env python3
"""Derive scenes/corpus/head-capped.ply from the pbrt-v4-scenes head scan.

The head ("Head of Infinite Realities", CC-BY — attribution in the PLY
header) is an open face scan: 19 boundary loops, from the whole back of
the scalp down to eye sockets and mouth. A membership-counted interior
cannot tolerate that — a path that enters through the face and finds no
back surface stays "inside" forever — so this script closes every loop
with a centroid fan: the front stays untouched, and the caps are crude
flat lids on surfaces the scan never had. One does show: head.ron drops
the source's cropwindow, and full-frame the chest lid catches the sky as
a cool rim along the silhouette. Every vertex property is carried
through (the albedo texture's UVs matter to rung 9e); a cap centroid
takes the mean of its loop's properties, except its normal, which is
computed from the loop's winding. The lids are then unwelded from the
shell so they shade by their own flat normal rather than by the scan's
outward one — a hard edge, and a large correction rather than a nicety:
the subsurface entry and exit reject any draw crossing the geometric
plane, so a welded rim was discarding about a third of them per lid
facet. Measured on the mesh, before and after, in the header comment.

Deepening the lids into domes was tried and measured: leak counts move
non-monotonically with cap depth, so the residual leaks are the scan's
own thin folds — the eyelid creases — and not a wedge at a cap rim.
Flat is what the evidence supports.

The output is a *derived asset* (untracked, like the resampled corpus
skies): only this script is committed. `./fetch.sh head` first.

Usage: python3 scenes/corpus/curate-head.py   (from anywhere)
"""

import struct
import sys
from collections import Counter
from pathlib import Path

CORPUS = Path(__file__).resolve().parent
SOURCE = CORPUS / "sources/pbrt-v4-scenes/head/geometry/head.ply"
OUT = CORPUS / "head-capped.ply"


def read_ply(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply"
        counts, props, current = {}, [], None
        while True:
            line = f.readline().decode().strip()
            if line == "end_header":
                break
            p = line.split()
            if p[0] == "element":
                counts[p[1]] = int(p[2])
                current = p[1]
            elif p[0] == "property" and p[1] != "list" and current == "vertex":
                props.append(p[2])
        assert props[:6] == ["x", "y", "z", "nx", "ny", "nz"], props
        nv, nf = counts["vertex"], counts["face"]
        stride = len(props)
        data = struct.unpack(f"<{nv * stride}f", f.read(nv * stride * 4))
        verts = [list(data[i * stride:(i + 1) * stride]) for i in range(nv)]
        faces = []
        for _ in range(nf):
            n, = struct.unpack("<B", f.read(1))
            assert n == 3
            faces.append(struct.unpack("<3i", f.read(12)))
    return props, verts, faces


def main():
    if not SOURCE.exists():
        sys.exit(f"{SOURCE} is not here — run scenes/corpus/fetch.sh head first")
    props, verts, faces = read_ply(SOURCE)
    stride = len(props)

    directed = Counter()
    for a, b, c in faces:
        directed[(a, b)] += 1
        directed[(b, c)] += 1
        directed[(c, a)] += 1
    assert all(n == 1 for n in directed.values()), "source is not orientable"
    nxt = dict(edge for edge in directed if (edge[1], edge[0]) not in directed)

    seen, loops = set(), []
    for start in nxt:
        if start in seen:
            continue
        loop, v = [start], nxt[start]
        seen.add(start)
        while v != start:
            loop.append(v)
            seen.add(v)
            v = nxt[v]
        loops.append(loop)

    cap_start = len(faces)
    capped = list(faces)
    caps = []
    for loop in loops:
        centroid = [sum(verts[v][i] for v in loop) / len(loop) for i in range(stride)]
        # The cap's normal: the loop's own winding, summed.
        normal = [0.0, 0.0, 0.0]
        for i, a in enumerate(loop):
            b = loop[(i + 1) % len(loop)]
            pa = [verts[a][j] - centroid[j] for j in range(3)]
            pb = [verts[b][j] - centroid[j] for j in range(3)]
            normal[0] += pa[1] * pb[2] - pa[2] * pb[1]
            normal[1] += pa[2] * pb[0] - pa[0] * pb[2]
            normal[2] += pa[0] * pb[1] - pa[1] * pb[0]
        length = max(sum(c * c for c in normal) ** 0.5, 1e-20)
        centroid[3:6] = [-c / length for c in normal]
        center = len(verts)
        verts.append(centroid)
        # Boundary edges run (a, b); the cap triangle takes (b, a) so every
        # edge gains its missing partner and the winding stays consistent.
        for i, a in enumerate(loop):
            b = loop[(i + 1) % len(loop)]
            capped.append((b, a, center))
        caps.append((loop, center, list(centroid[3:6])))

    check = Counter()
    for a, b, c in capped:
        check[(a, b)] += 1
        check[(b, c)] += 1
        check[(c, a)] += 1
    assert all(check.get((b, a), 0) == 1 for (a, b) in check), "cap failed"

    # Now split the lids' shading off the shell's, which the closure test
    # above had to see welded to be worth anything. A rim vertex carries
    # the scan's outward normal, and lending it to a flat lid leaves the
    # lid's own triangles shaded by a normal up to 90 degrees off their
    # geometry. That is not cosmetic here: cenote draws the subsurface
    # entry and exit as cosine lobes about the shading normal and discards
    # any draw landing on the far side of the geometric plane
    # (openpbr.slang, shade_subsurface.slang), so a welded fan threw away
    # ~32% of the draws at each end of the walk, varying per facet, with
    # steps up to 75% between neighbours. Each fan triangle gets its own
    # three vertices instead: same positions, same UVs, its own facet
    # normal. Flat is the honest shading for a lid the scan never had —
    # there is no smooth surface underneath for a soft normal to stand in
    # for. The surface is untouched and still closed; only the shading
    # attributes split, which is what a hard edge is.
    capped = capped[:cap_start]
    dupes = []
    for loop, center, lid in caps:
        for i, a in enumerate(loop):
            b = loop[(i + 1) % len(loop)]
            tri = (b, a, center)
            # The fan's own facet normal, not the lid's average: a boundary
            # loop is a 3D curve, so a centroid fan is not planar, and one
            # normal for the whole lid still left 45 degrees of disagreement.
            # Oriented to the lid so the mesh stays coherently outward.
            p = [verts[v][0:3] for v in tri]
            e1 = [p[1][j] - p[0][j] for j in range(3)]
            e2 = [p[2][j] - p[0][j] for j in range(3)]
            n = [e1[1] * e2[2] - e1[2] * e2[1],
                 e1[2] * e2[0] - e1[0] * e2[2],
                 e1[0] * e2[1] - e1[1] * e2[0]]
            length = max(sum(c * c for c in n) ** 0.5, 1e-20)
            n = [c / length for c in n]
            if sum(n[j] * lid[j] for j in range(3)) < 0.0:
                n = [-c for c in n]
            fresh = []
            for v in tri:
                copy = list(verts[v])
                copy[3:6] = n
                fresh.append(len(verts))
                dupes.append((v, len(verts)))
                verts.append(copy)
            capped.append(tuple(fresh))

    # The unweld is a shading change and nothing else: same triangle count,
    # and every copy stands exactly where its original stands.
    assert len(capped) == cap_start + sum(len(loop) for loop, _, _ in caps), \
        "unweld changed the triangulation"
    assert all(verts[v][0:3] == verts[w][0:3] for v, w in dupes), \
        "unweld moved a vertex"

    with open(OUT, "wb") as out:
        out.write(b"ply\nformat binary_little_endian 1.0\n")
        out.write(b"comment 'Head of Infinite Realities' scan (CC-BY, Infinite\n")
        out.write(b"comment Realities Inc. via pbrt-v4-scenes), boundary loops capped\n")
        out.write(b"comment closed by scenes/corpus/curate-head.py - a derived,\n")
        out.write(b"comment untracked asset; the script is what the repo carries.\n")
        out.write(f"element vertex {len(verts)}\n".encode())
        for p in props:
            out.write(f"property float {p}\n".encode())
        out.write(f"element face {len(capped)}\n".encode())
        out.write(b"property list uint8 int vertex_indices\nend_header\n")
        for v in verts:
            out.write(struct.pack(f"<{stride}f", *v))
        for tri in capped:
            out.write(struct.pack("<B3i", 3, *tri))
    print(f"capped {len(loops)} loops; wrote {OUT}")


if __name__ == "__main__":
    main()
