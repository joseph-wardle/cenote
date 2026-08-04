#!/usr/bin/env bash
# B7's gate captures, in two halves:
#
#   scripts/b7-gates.sh timing      # 3 runs x 128 spp per gate scene -> .stats.ron
#   scripts/b7-gates.sh reference   # one 4096-spp bake per gate scene -> reference EXR
#   scripts/b7-gates.sh all
#
# Everything lands in /b7-baselines/ (gitignored — the numbers are this
# machine's). `timing` is the regression side: frame medians and the kernel
# breakdown, three repeats so the run-to-run band is measured rather than
# guessed. `reference` is the error side: the images B7's relMSE oracle
# compares candidates against at equal sample counts. Renders are
# deterministic per build, so one EXR per timing config is enough — repeats
# only exist for the clock.
#
# Run on a quiet desktop; background load can double every number.
set -euo pipefail

mode=${1:?usage: b7-gates.sh timing|reference|all}
root=$(cd "$(dirname "$0")/.." && pwd)
cli=$root/target/release/cenote-cli
[[ -x $cli ]] || { echo "build it first: cargo build --release -p cenote-cli" >&2; exit 1; }

out=$root/b7-baselines
mkdir -p "$out/timing" "$out/reference"

width=2560
height=1440
# The oracle's "equal samples" count: long enough that relMSE is stable,
# short enough that seven scenes are minutes.
gate_spp=128
# 32x the gate count, so the reference's own noise is a rounding error on
# both sides of the relMSE ratio.
reference_spp=4096

# Drivers first (shading-bound), then the overhead controls, then the
# transmission/coat coverage the drivers lack.
scenes=(
  "$root/scenes/corpus/bistro.ron:bistro"
  "$root/scenes/corpus/sanmiguel.ron:sanmiguel"
  "$root/scenes/corpus/zero-day.ron:zero-day"
  "$root/scenes/corpus/cornell-box.ron:cornell-box"
  "$root/scenes/brass-room.ron:brass-room"
  "$root/scenes/corpus/glass-of-water.ron:glass-of-water"
  "$root/scenes/corpus/spaceship.ron:spaceship"
)

if [[ $mode == timing || $mode == all ]]; then
  for entry in "${scenes[@]}"; do
    scene=${entry%%:*}
    label=${entry##*:}
    for run in 1 2 3; do
      echo "== $label timing run $run =="
      "$cli" render "$scene" \
        --spp $gate_spp --width $width --height $height \
        --out "$out/timing/$label.exr" 2>/dev/null
      mv "$out/timing/$label.stats.ron" "$out/timing/$label.r$run.stats.ron"
    done
  done
fi

if [[ $mode == reference || $mode == all ]]; then
  for entry in "${scenes[@]}"; do
    scene=${entry%%:*}
    label=${entry##*:}
    echo "== $label reference bake =="
    "$cli" render "$scene" \
      --spp $reference_spp --width $width --height $height \
      --out "$out/reference/$label.exr" 2>/dev/null
  done
fi

echo
ls -1 "$out"/timing "$out"/reference
