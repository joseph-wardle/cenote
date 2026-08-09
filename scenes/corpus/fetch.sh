#!/usr/bin/env sh
# Fetch research-corpus scene sources into scenes/corpus/sources/ (which
# stays untracked — only the curated .ron files and README.md are in git).
#
#   ./fetch.sh                # everything (~8.5 GB, mostly pbrt-v4-scenes)
#   ./fetch.sh veach-ajar     # just the named scenes
#
# Two sources, one recipe each. Bitterli's zips download whole (a few MB
# to ~40 MB, each carrying its own license text). pbrt-v4-scenes comes as
# one sparse partial clone at a pinned commit — git verifies every blob
# against the commit hash, so the pin is the checksum — and each scene
# fetch just widens the sparse cone. Both are idempotent: a scene that already materialized is
# skipped, so re-running after adding a name is cheap.
set -eu

PIN=30cf4a0346ae5a80a2d7a530a3ef7d0fa4f70572
SOURCES="$(cd "$(dirname "$0")" && pwd)/sources"

BITTERLI="bathroom coffee cornell-box glass-of-water kitchen spaceship
          teapot-full veach-ajar veach-bidir veach-mis volumetric-caustic
          water-caustic"
PBRT_V4="bistro bmw-m6 crown head kroken sanmiguel watercolor zero-day"

bitterli() { # <slug>
    dir="$SOURCES/bitterli/$1"
    [ -e "$dir/scene-v4.pbrt" ] && { echo "already fetched: $1"; return; }
    mkdir -p "$SOURCES/bitterli"
    curl -fL --progress-bar -o "$dir.zip" \
        "https://benedikt-bitterli.me/resources/pbrt-v4/$1.zip"
    unzip -oq "$dir.zip" -d "$SOURCES/bitterli"
    rm "$dir.zip"
    echo "fetched bitterli: $1"
}

pbrt_v4() { # <dir>
    repo="$SOURCES/pbrt-v4-scenes"
    [ -e "$repo/$1/README.md" ] || [ -n "$(ls "$repo/$1" 2>/dev/null)" ] && {
        echo "already fetched: $1"; return; }
    mkdir -p "$repo"
    git -C "$repo" init -q
    git -C "$repo" remote add origin \
        https://github.com/mmp/pbrt-v4-scenes.git 2>/dev/null || true
    git -C "$repo" sparse-checkout add "$1" 2>/dev/null \
        || git -C "$repo" sparse-checkout set "$1"
    git -C "$repo" fetch -q --filter=blob:none origin "$PIN"
    git -C "$repo" checkout -q "$PIN"
    echo "fetched pbrt-v4-scenes@$PIN: $1"
}

listed() { # <name> <word...>
    name="$1"; shift
    for word in "$@"; do [ "$word" = "$name" ] && return 0; done
    return 1
}

fetch() { # <name>
    if   listed "$1" $BITTERLI; then bitterli "$1"
    elif listed "$1" $PBRT_V4;  then pbrt_v4  "$1"
    else
        echo "unknown scene: $1 (see README.md for the corpus list)" >&2
        exit 1
    fi
}

if [ $# -eq 0 ]; then
    for scene in $BITTERLI $PBRT_V4; do fetch "$scene"; done
else
    for scene in "$@"; do fetch "$scene"; done
fi
