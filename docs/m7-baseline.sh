#!/usr/bin/env bash
# The M7 scoreboard: render the five-scene benchmark set and leave one RON
# sidecar per run, so a rung of work is a diff rather than a remembered
# impression.
#
#   docs/m7-baseline.sh <out-dir>          # capture at 1440p
#   WIDTH=1920 HEIGHT=1080 docs/m7-baseline.sh <out-dir>
#   diff -u <old-dir>/*.stats.ron <new-dir>/*.stats.ron
#
# Each scene earns its slot on one of the two axes M7 named — representative of
# a production workflow, or a probe of a specific renderer property. See
# docs/m7-plan.md §3.5 for what each one is for.
#
# Run it on a quiet desktop. D-152 already recorded that background load on this
# machine can double every number mid-session; the sidecars carry the frame-time
# distribution (median and p95, over uninstrumented frames) precisely so a
# contended capture is visible rather than silently believed.
set -euo pipefail

out=${1:?usage: m7-baseline.sh <out-dir>}
mkdir -p "$out"

root=$(cd "$(dirname "$0")/.." && pwd)
cli=$root/target/release/cenote-cli
[[ -x $cli ]] || { echo "build it first: cargo build --release -p cenote-cli" >&2; exit 1; }

# Captured at 1440p: the resolution the renderer is actually judged at.
# Decision 9's bar is stated at 1080p — set WIDTH and HEIGHT to reproduce that
# basis, and the pinned §3.5.1 table with it.
width=${WIDTH:-2560}
height=${HEIGHT:-1440}
# Enough samples for a stable median and to reach the 16-spp readability mark;
# short enough that the whole set is a few minutes.
spp=64

# scene:label
scenes=(
  "$root/scenes/corpus/cornell-box.ron:cornell-box"   # cheapest frame — the dispatch-bound probe
  "$root/scenes/brass-room.ron:brass-room"            # heavy indirect GI — the hard-transport case
  "$root/scenes/many-lights.ron:many-lights"          # many lights — the NEE-bound case
  "$root/scenes/corpus/bistro.ron:bistro"             # production exterior at scale
  "$root/scenes/corpus/zero-day.ron:zero-day"         # 283 lights in a dark interior — the scene the interactivity call is made on
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
