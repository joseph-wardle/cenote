#!/usr/bin/env bash
# Render the benchmark set, leaving one RON stats sidecar per scene so two
# runs compare as a text diff:
#
#   scripts/benchmark.sh <out-dir>                       # 1440p
#   WIDTH=1920 HEIGHT=1080 scripts/benchmark.sh <out-dir>
#   diff -u <old-dir>/*.stats.ron <new-dir>/*.stats.ron
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

# Each scene probes one cost regime; label after the colon.
scenes=(
  "$root/scenes/corpus/cornell-box.ron:cornell-box"   # cheapest frame — dispatch-bound
  "$root/scenes/brass-room.ron:brass-room"            # heavy indirect GI
  "$root/scenes/many-lights.ron:many-lights"          # NEE-bound
  "$root/scenes/corpus/bistro.ron:bistro"             # production exterior at scale
  "$root/scenes/corpus/zero-day.ron:zero-day"         # 283 lights in a dark interior
)

for entry in "${scenes[@]}"; do
  scene=${entry%%:*}
  label=${entry##*:}
  [[ -f $scene ]] || { echo "skipping $label — $scene is not here" >&2; continue; }
  echo "== $label =="
  "$cli" render "$scene" \
    --spp "$spp" --width "$width" --height "$height" \
    --out "$out/$label.exr" 2>/dev/null
done

echo
echo "sidecars in $out:"
ls -1 "$out"/*.stats.ron
