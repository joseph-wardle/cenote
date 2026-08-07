#!/usr/bin/env bash
# B7's gate captures. Artifacts land in /b7-baselines/ (gitignored — the
# numbers are this machine's), each directory stamped with the commit,
# binary, and driver that produced it.
#
#   scripts/b7-gates.sh timing [scene]     # 3 runs x 128 spp -> .stats.ron + EXR
#   scripts/b7-gates.sh reference [scene]  # one 4096-spp bake -> reference EXR
#   scripts/b7-gates.sh relmse [scene]     # timing EXRs vs references -> baseline file
#   scripts/b7-gates.sh all
#   scripts/b7-gates.sh ab <control-cli> <candidate-cli> [scene]
#
# `timing` measures the regression side: frame medians and the kernel
# breakdown, three repeats so the run-to-run band is measured rather than
# guessed. `reference` bakes the images the relMSE oracle scores against;
# `relmse` does that scoring (renders are deterministic per build, so the
# timing EXR doubles as the oracle input, and repeats only exist for the
# clock). `ab` is the required protocol for comparing two builds:
# session-to-session clock drift exceeds the run-to-run band, so control
# and candidate must interleave within one session — never against stored
# numbers. Its EXRs also hand Class A rungs their `cmp` evidence for free.
#
# Run on a quiet desktop; background load can double every number.
set -euo pipefail

usage="usage: b7-gates.sh timing|reference|relmse|all [scene] | ab <control-cli> <candidate-cli> [scene]"
mode=${1:?$usage}
root=$(cd "$(dirname "$0")/.." && pwd)
out=$root/b7-baselines
cli=$root/target/release/cenote-cli
relmse=$root/target/release/cenote-relmse

width=2560
height=1440
# Long enough that relMSE is stable, short enough that seven scenes are
# minutes; the reference is 32x this, so its own noise is a rounding error
# on both sides of the ratio.
gate_spp=128
reference_spp=4096

# Drivers first (shading-bound), then the overhead controls, then the
# transmission/coat coverage the drivers lack — ending with the one scene
# that authors a nesting priority and has no media at all, which is the
# only place the volume stage's cost to a media-free scene can be seen.
scenes=(
  "$root/scenes/corpus/bistro.ron:bistro"
  "$root/scenes/corpus/sanmiguel.ron:sanmiguel"
  "$root/scenes/corpus/zero-day.ron:zero-day"
  "$root/scenes/corpus/cornell-box.ron:cornell-box"
  "$root/scenes/brass-room.ron:brass-room"
  "$root/scenes/corpus/glass-of-water.ron:glass-of-water"
  "$root/scenes/corpus/spaceship.ron:spaceship"
  "$root/scenes/nested-glass.ron:nested-glass"
)

filter=
case $mode in
  timing | reference | relmse | all) filter=${2-} ;;
  ab)
    control=${2:?$usage}
    candidate=${3:?$usage}
    filter=${4-}
    ;;
  *)
    echo "$usage" >&2
    exit 1
    ;;
esac

# Returns success when the scene entry should be skipped under $filter.
skipped() { [[ -n $filter && ${1##*:} != "$filter" ]]; }

require() { [[ -x $1 ]] || { echo "build it first: cargo build --release -p cenote-cli ($1)" >&2; exit 1; }; }

# stamp <file> <binary>... — record what produced a capture directory.
stamp() {
  {
    date --iso-8601=seconds
    git -C "$root" describe --always --dirty
    stat -c '%y  %n' "${@:2}"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true
  } >"$1"
}

if [[ $mode == timing || $mode == all ]]; then
  require "$cli"
  mkdir -p "$out/timing"
  stamp "$out/timing/capture-meta.txt" "$cli"
  for entry in "${scenes[@]}"; do
    skipped "$entry" && continue
    scene=${entry%%:*} label=${entry##*:}
    for run in 1 2 3; do
      echo "== $label timing run $run =="
      "$cli" render "$scene" --spp $gate_spp --width $width --height $height \
        --out "$out/timing/$label.exr"
      mv "$out/timing/$label.stats.ron" "$out/timing/$label.r$run.stats.ron"
    done
  done
fi

if [[ $mode == reference || $mode == all ]]; then
  require "$cli"
  mkdir -p "$out/reference"
  stamp "$out/reference/capture-meta.txt" "$cli"
  for entry in "${scenes[@]}"; do
    skipped "$entry" && continue
    scene=${entry%%:*} label=${entry##*:}
    echo "== $label reference bake =="
    "$cli" render "$scene" --spp $reference_spp --width $width --height $height \
      --out "$out/reference/$label.exr"
  done
fi

if [[ $mode == relmse || $mode == all ]]; then
  require "$relmse"
  baseline_file=$out/relmse-baseline-${gate_spp}spp.txt
  # A scene filter prints without rewriting the (whole-set) baseline file.
  [[ -z $filter ]] && : >"$baseline_file"
  for entry in "${scenes[@]}"; do
    skipped "$entry" && continue
    label=${entry##*:}
    score=$("$relmse" "$out/reference/$label.exr" "$out/timing/$label.exr")
    printf '%-16s %s\n' "$label" "$score"
    [[ -z $filter ]] && printf '%-16s %s\n' "$label" "$score" >>"$baseline_file"
  done
fi

if [[ $mode == ab ]]; then
  require "$control"
  require "$candidate"
  mkdir -p "$out/ab"
  stamp "$out/ab/capture-meta.txt" "$control" "$candidate"
  for entry in "${scenes[@]}"; do
    skipped "$entry" && continue
    scene=${entry%%:*} label=${entry##*:}
    for run in 1 2 3; do
      for side in control candidate; do
        if [[ $side == control ]]; then bin=$control; else bin=$candidate; fi
        echo "== $label $side run $run =="
        "$bin" render "$scene" --spp $gate_spp --width $width --height $height \
          --out "$out/ab/$label.$side.exr"
        mv "$out/ab/$label.$side.stats.ron" "$out/ab/$label.$side.r$run.stats.ron"
      done
    done
  done
fi
