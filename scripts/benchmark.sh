#!/usr/bin/env bash
# Render the benchmark set and walk each scene's edit latency, leaving two RON
# sidecars per scene so two runs compare as a text diff:
#
#   scripts/benchmark.sh <out-dir>                       # 1440p
#   WIDTH=1920 HEIGHT=1080 scripts/benchmark.sh <out-dir>
#   diff -u <old-dir>/*.stats.ron <new-dir>/*.stats.ron
#   diff -u <old-dir>/*.latency.ron <new-dir>/*.latency.ron
#
# The two halves answer different questions: `.stats.ron` is what a frame
# costs once the render is running, `.latency.ron` is what an edit costs
# before a frame carrying it appears. Neither is a gate — no threshold fires
# here, because the run-to-run spread of both has never been measured.
#
# Run it on a quiet desktop: background load can double every number. The
# sidecars carry the frame-time median and p95 over uninstrumented frames, so
# a contended capture is visible rather than silently believed.
set -euo pipefail

out=${1:?usage: benchmark.sh <out-dir>}
mkdir -p "$out"

root=$(cd "$(dirname "$0")/.." && pwd)
cli=$root/target/release/cenote-cli
[[ -x $cli ]] || { echo "build it first: cargo build --release -p cenote-cli" >&2; exit 1; }

width=${WIDTH:-2560}
height=${HEIGHT:-1440}
# Long enough for a stable median, short enough that the whole set is minutes.
spp=64

# The scene list lives in scenes/manifest.sh, one entry per cost regime.
source "$root/scenes/manifest.sh"

for entry in "${benchmark_scenes[@]}"; do
  scene=$root/${entry%%:*}
  label=${entry##*:}
  [[ -f $scene ]] || { echo "skipping $label — $scene is not here" >&2; continue; }
  echo "== $label =="
  "$cli" render "$scene" \
    --spp "$spp" --width "$width" --height "$height" \
    --out "$out/$label.exr" 2>/dev/null
done

# The same scenes again, edited rather than rendered: one pass down the edit
# vocabulary per scene, each edit issued against a settled image and timed to
# the first pixel that shows it. A scene with nothing for an edit to target
# records the skip and moves on, so the walk is the same length everywhere.
echo
echo "== edit latency =="
for entry in "${scenes[@]}"; do
  scene=${entry%%:*}
  label=${entry##*:}
  [[ -f $scene ]] || { echo "skipping $label — $scene is not here" >&2; continue; }
  "$cli" edit-latency "$scene" \
    --width "$width" --height "$height" \
    --out "$out/$label.latency.ron" 2>/dev/null
done

echo
echo "sidecars in $out:"
ls -1 "$out"/*.stats.ron "$out"/*.latency.ron
