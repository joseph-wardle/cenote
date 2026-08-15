#!/usr/bin/env bash
# Render four controlled curve scenes in both pbrt-v4 and cenote, and compare
# them as occlusion masks. This is the repo's one *external* renderer oracle:
# every other gate compares cenote against cenote, so a shared misreading of
# the input would pass them all. Here a second, independent implementation of
# `Shape "curve"` says whether the centerlines, widths, taper and transforms
# are actually right.
#
#   scripts/curve-oracle.sh
#   PBRT=/path/to/pbrt scripts/curve-oracle.sh
#
# Needs a release cenote-cli, `oiiotool`, numpy, and pbrt-v4. Without pbrt it
# says so and exits 0: this gate proves an agreement with another renderer, and
# absent that renderer there is no weaker version of it worth running.
#
# What it does *not* claim: that a tessellated tube and pbrt's procedural curve
# make the same pixels. They do not, and the scenes are built so the ways they
# differ are separable — see tests/scenes/curve-oracle/README.md.
set -uo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
cli=$root/target/release/cenote-cli
oracle=$root/tests/scenes/curve-oracle
out=${OUT_DIR:-$root/target/curve-oracle}
pbrt=${PBRT:-pbrt}

if ! command -v "$pbrt" >/dev/null 2>&1; then
  echo "curve-oracle: SKIPPED — no pbrt-v4 (\`$pbrt\`); set PBRT=/path/to/pbrt"
  exit 0
fi
[ -x "$cli" ] || { echo "curve-oracle: build cenote-cli --release first"; exit 1; }

mkdir -p "$out"
for scene in s1 s2 s3 s4; do
  echo "== $scene"
  # pbrt reads the scene directly; cenote reads it through the importer, so
  # this arm is the import and the render at once.
  "$pbrt" --outfile "$out/$scene-pbrt.exr" "$oracle/$scene.pbrt" >/dev/null || exit 1
  "$cli" render "$oracle/$scene.pbrt" --out "$out/$scene-cenote.exr" --no-stats || exit 1
done

exec python3 "$oracle/compare.py" "$out"
