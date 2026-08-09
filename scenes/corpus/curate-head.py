#!/usr/bin/env python3
"""Derive scenes/corpus/head-capped.ply from the pbrt-v4-scenes head scan.

The head ("Head of Infinite Realities", CC-BY — attribution in the PLY
header) is an open face scan: 19 boundary loops, from the whole back of
the scalp down to eye sockets and mouth. A membership-counted interior
cannot tolerate that — a path that enters through the face and finds no
back surface stays "inside" forever — so this script closes every loop
with a centroid fan: the front stays untouched, the crude flat caps sit
where no driver-scene camera looks. Every vertex property is carried
through (the albedo texture's UVs matter to rung 9e); a cap centroid
takes the mean of its loop's properties, except its normal, which is
computed from the loop's winding.

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

    capped = list(faces)
    for loop in loops:
        centroid = [sum(verts[v][i] for v in loop) / len(loop) for i in range(stride)]
        # The cap's normal: the loop's own winding, summed — good enough
        # for shading a surface the driver cameras never see.
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

    check = Counter()
    for a, b, c in capped:
        check[(a, b)] += 1
        check[(b, c)] += 1
        check[(c, a)] += 1
    assert all(check.get((b, a), 0) == 1 for (a, b) in check), "cap failed"

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
