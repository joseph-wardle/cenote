#!/usr/bin/env bash
# Gate captures for the shading (B7) and subsurface-walk (B9) sets.
# Artifacts land in /b7-baselines/ and /b9-baselines/ (gitignored — the
# numbers are this machine's), each directory stamped with the commit,
# binary, and driver that produced it.
#
#   scripts/b7-gates.sh timing [scene]     # 3 runs x 128 spp -> .stats.ron + EXR
#   scripts/b7-gates.sh reference [scene]  # one 4096-spp bake -> reference EXR
#   scripts/b7-gates.sh quality [scene]    # the relMSE candidate, where the
#                                          # cost and quality resolutions differ
#   scripts/b7-gates.sh relmse [scene]     # candidate EXRs vs references
#   scripts/b7-gates.sh all
#   scripts/b7-gates.sh ab <control-cli> <candidate-cli> [scene]
#   scripts/b7-gates.sh b9 <verb> ...      # the same verbs over the walk set
#   scripts/b7-gates.sh b9 probes <probes-cli> [scene]
#
# `timing` measures the regression side: frame medians and the kernel
# breakdown, three repeats so the run-to-run band is measured rather than
# guessed. `reference` bakes the images the relMSE oracle scores against;
# `relmse` does that scoring (renders are deterministic per build, so the
# candidate EXR is rendered once and repeats only exist for the clock).
# `ab` is the required protocol for comparing two builds: session-to-session
# clock drift exceeds the run-to-run band, so control and candidate must
# interleave within one session — never against stored numbers. Its EXRs
# also hand Class A rungs their `cmp` evidence for free.
#
# The two sets differ in what they are for, and the resolutions are pinned
# by the ladders that produced them, not chosen here:
#
# * `b7` — nine scenes at 2560x1440. Cost and quality share a resolution,
#   so the timing EXR doubles as the relMSE candidate.
# * `b9` — the four subsurface walk drivers. Cost is measured at 720p,
#   because that is where the 9-0 and 9b ladders measured it and the three
#   are only comparable at one resolution; everything else runs at
#   512x288, likewise. `probes` is the set's hard gate and the only mode
#   that is exact: the walk's histograms are deterministic, so a
#   difference is a transport change, not drift. It is excluded from `all`
#   and takes its own binary, because a `--features probes` build carries
#   atomics the timing runs must not measure:
#
#       cargo build --release -p cenote-cli --features probes \
#           --target-dir target/probes
#       scripts/b7-gates.sh b9 probes target/probes/release/cenote-cli
#
#   It diffs each capture against /b9-baselines/probes-pinned/ and fails
#   on any difference. Promote a reviewed capture with
#   `cp b9-baselines/probes/*.probes.ron b9-baselines/probes-pinned/`.
#
# Run on a quiet desktop; background load can double every number.
set -euo pipefail

usage="usage: b7-gates.sh [b9] timing|reference|quality|relmse|all [scene]
       b7-gates.sh [b9] ab <control-cli> <candidate-cli> [scene]
       b7-gates.sh b9 probes <probes-cli> [scene]"

set_name=b7
if [[ ${1-} == b7 || ${1-} == b9 ]]; then
  set_name=$1
  shift
fi

mode=${1:?$usage}
root=$(cd "$(dirname "$0")/.." && pwd)
out=$root/$set_name-baselines
cli=$root/target/release/cenote-cli
relmse=$root/target/release/cenote-relmse

# Long enough that relMSE is stable, short enough that a whole set is
# minutes; the reference is 32x this, so its own noise is a rounding error
# on both sides of the ratio.
gate_spp=128
reference_spp=4096
# Where the relMSE candidate comes from: the timing EXR when cost and
# quality share a resolution, a render of its own when they do not.
oracle_dir=timing

if [[ $set_name == b7 ]]; then
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
    "$root/scenes/milk-glass.ron:milk-glass"
  )
  width=2560
  height=1440
  cost_width=$width
  cost_height=$height
else
  # The three optical-depth tiers, then the skin-density head: the tiers
  # share geometry and differ only in sigma, so a number that moves on one
  # and not the others is the medium's doing rather than the scene's.
  scenes=(
    "$root/scenes/sss-teapot-wax-walk.ron:wax"
    "$root/scenes/sss-teapot-marble-walk.ron:marble"
    "$root/scenes/sss-teapot-skin-walk.ron:skin"
    "$root/scenes/sss-head-walk.ron:head"
  )
  width=512
  height=288
  cost_width=1280
  cost_height=720
  probe_spp=8
  oracle_dir=quality
fi

filter=
case $mode in
  timing | reference | quality | relmse | all) filter=${2-} ;;
  probes)
    [[ $set_name == b9 ]] || { echo "probes is the b9 set's gate" >&2; exit 1; }
    probes_cli=${2:?$usage}
    filter=${3-}
    ;;
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
      "$cli" render "$scene" --spp $gate_spp --width $cost_width --height $cost_height \
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

if [[ $oracle_dir == quality && ($mode == quality || $mode == all) ]]; then
  require "$cli"
  mkdir -p "$out/quality"
  stamp "$out/quality/capture-meta.txt" "$cli"
  for entry in "${scenes[@]}"; do
    skipped "$entry" && continue
    scene=${entry%%:*} label=${entry##*:}
    echo "== $label quality render =="
    "$cli" render "$scene" --spp $gate_spp --width $width --height $height \
      --out "$out/quality/$label.exr"
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
    score=$("$relmse" "$out/reference/$label.exr" "$out/$oracle_dir/$label.exr")
    printf '%-16s %s\n' "$label" "$score"
    [[ -z $filter ]] && printf '%-16s %s\n' "$label" "$score" >>"$baseline_file"
  done
fi

if [[ $mode == probes ]]; then
  require "$probes_cli"
  mkdir -p "$out/probes"
  stamp "$out/probes/capture-meta.txt" "$probes_cli"
  drifted=0
  for entry in "${scenes[@]}"; do
    skipped "$entry" && continue
    scene=${entry%%:*} label=${entry##*:}
    echo "== $label probes =="
    "$probes_cli" render "$scene" --spp $probe_spp --width $width --height $height \
      --out "$out/probes/$label.exr"
    [[ -f $out/probes/$label.probes.ron ]] || {
      echo "no sidecar — build the CLI with --features probes" >&2
      exit 1
    }
    pinned=$out/probes-pinned/$label.probes.ron
    if [[ -f $pinned ]]; then
      if diff -q "$pinned" "$out/probes/$label.probes.ron" >/dev/null; then
        echo "   pinned: unchanged"
      else
        echo "   pinned: CHANGED — the walk's histogram is deterministic, so this is transport"
        diff "$pinned" "$out/probes/$label.probes.ron" | head -20
        drifted=1
      fi
    else
      echo "   pinned: none yet"
    fi
  done
  [[ $drifted == 0 ]] || exit 1
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
        "$bin" render "$scene" --spp $gate_spp --width $cost_width --height $cost_height \
          --out "$out/ab/$label.$side.exr"
        mv "$out/ab/$label.$side.stats.ron" "$out/ab/$label.$side.r$run.stats.ron"
      done
    done
  done
fi
