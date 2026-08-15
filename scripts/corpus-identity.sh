#!/usr/bin/env bash
# Re-render the whole corpus at its capture settings and byte-compare
# against a stored sweep. This is the repo's blunt regression gate: cenote
# renders deterministically per build, so an EXR that differs by one byte
# means the images moved, and the only question left is whether that was
# intended.
#
#   scripts/corpus-identity.sh                    # vs baselines/corpus
#   scripts/corpus-identity.sh <baseline-dir>     # vs some other sweep
#   scripts/corpus-identity.sh <baseline-dir> <out-dir>
#
# Exits nonzero if any scene differs, fails to render, or is missing. A
# difference is a defect to bisect until something says otherwise — never
# a reason to re-mint the baseline in the same breath.
#
# The output EXRs carry no timestamp or provenance attribute (see
# `write_exr` in output.rs), which is what makes `cmp` a valid identity
# test rather than an approximation of one.
#
# Minting a new sweep is deliberate and separate: render into a fresh
# directory, review why it moved, and only then point later runs at it.
# Each capture directory carries its account in its own
# `capture-meta.txt`. A scene with no baseline yet is counted loudly as
# unbaselined, not failed — it joins the gate at its first minted sweep.
# The scene list lives in scenes/manifest.sh, beside its prerequisites.
set -uo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
cli=$root/target/release/cenote-cli
base=${1:-$root/baselines/corpus}
out=${2:-$root/corpus-identity}

width=2560
height=1440
spp=64

source "$root/scenes/manifest.sh"
scenes=("${identity_scenes[@]}")

[[ -x $cli ]] || { echo "build it first: cargo build --release -p cenote-cli" >&2; exit 1; }
mkdir -p "$out"

fail=0
compared=0
unbaselined=0
for entry in "${scenes[@]}"; do
  scene=$root/${entry%%:*} label=${entry##*:}
  echo "== $label =="
  if [[ ! -f $scene ]]; then
    echo "RESULT $label MISSING-SCENE"
    fail=1
    continue
  fi
  if ! "$cli" render "$scene" --spp $spp --width $width --height $height \
      --out "$out/$label.exr" >"$out/$label.log" 2>&1; then
    echo "RESULT $label RENDER-FAILED"
    tail -5 "$out/$label.log"
    fail=1
    continue
  fi
  if [[ ! -f $base/$label.exr ]]; then
    # A scene the baseline predates — the head, until a sweep is minted
    # that includes it. Loud, counted, and not a failure: there is no
    # earlier image for it to differ from.
    echo "RESULT $label NO-BASELINE"
    unbaselined=$((unbaselined + 1))
  elif cmp -s "$out/$label.exr" "$base/$label.exr"; then
    echo "RESULT $label IDENTICAL"
    compared=$((compared + 1))
  else
    echo "RESULT $label DIFFERS"
    compared=$((compared + 1))
    fail=1
  fi
done

# Nothing compared means the baseline directory is wrong, not that every
# scene is new — the one way this gate could pass by accident.
if [[ $compared == 0 ]]; then
  echo "no scene had a baseline in $base — wrong directory?"
  fail=1
fi

echo "== sweep done: $compared compared, $unbaselined unbaselined, fail=$fail =="
exit $fail
