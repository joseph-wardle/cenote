#!/usr/bin/env bash
# The M7 scoreboard: render the five-scene benchmark set — ReSTIR on and off,
# plus two moving arms for ReSTIR — and leave one RON sidecar per run so a rung
# of work is a diff rather than a remembered impression.
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

# Captured at 1440p: the resolution the renderer is actually judged at, and
# where reuse's 480 B/pixel of reservoir state starts to bite (1.7 GiB here
# against 949 MiB at 1080p). Decision 9's bar is stated at 1080p — set WIDTH and
# HEIGHT to reproduce that basis, and the pinned §3.5.1 table with it.
width=${WIDTH:-2560}
height=${HEIGHT:-1440}
# Enough samples for a stable median and to reach the 16-spp readability mark;
# short enough that the whole set is a few minutes.
spp=64

# scene:label
scenes=(
  "$root/scenes/corpus/cornell-box.ron:cornell-box"   # cheapest frame — the dispatch-bound probe
  "$root/scenes/brass-room.ron:brass-room"            # heavy indirect GI — the ReSTIR-PT case
  "$root/scenes/many-lights.ron:many-lights"          # many lights / NEE-bound — the ReSTIR-DI case
  "$root/scenes/corpus/bistro.ron:bistro"             # production exterior at scale
  "$root/scenes/corpus/zero-day.ron:zero-day"         # 283 lights in a dark interior — the scene the interactivity call is made on
)

for entry in "${scenes[@]}"; do
  scene=${entry%%:*}
  label=${entry##*:}
  [[ -f $scene ]] || { echo "skipping $label — $scene is not here" >&2; continue; }
  # Four arms per scene. `pt` and `restir` are the settled still — what a batch
  # render costs. `restir-moving` resets the film between samples, so every
  # frame renders as the frame after a restart: the estimator state a camera
  # being dragged is actually in, which the two still arms never visit and where
  # M7's interactive bar is stated. ReSTIR only — a path-traced sample's cost
  # does not depend on the sample index, so a PT moving arm would reproduce the
  # PT still column to within noise.
  #
  # The moving arms are an *upper* bound on real motion: reprojection is the
  # identity with a held camera, so nothing ever disoccludes, and a disocclusion
  # drops history before any shift work. They say nothing about how motion looks.
  #
  # `restir-moving-nospatial` prices the spatial pass as a whole — gather plus
  # combine, k = 5 neighbours — on the frame where it costs the most. It is the
  # arm that says how much a motion-scaled k could be worth before anyone builds
  # one, and it doubles as the blotchiness reference: this is the image temporal
  # reuse produces alone.
  for mode in pt restir restir-moving restir-moving-nospatial; do
    case $mode in
      pt)                      flags="" ;;
      restir)                  flags="--restir" ;;
      restir-moving)           flags="--restir --restart-every-sample" ;;
      restir-moving-nospatial) flags="--restir --restart-every-sample --no-spatial" ;;
    esac
    echo "== $label / $mode =="
    "$cli" render "$scene" $flags \
      --spp "$spp" --width "$width" --height "$height" \
      --out "$out/$label.$mode.exr" 2>/dev/null
  done
done

echo
echo "sidecars in $out:"
ls -1 "$out"/*.stats.ron
