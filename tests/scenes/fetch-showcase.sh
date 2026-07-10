#!/usr/bin/env sh
# Fetch the tier-2 showcase scene (BMW M6, CC0) from pbrt-v4-scenes at a
# pinned commit — git verifies every blob against the commit hash, so
# the pin is the checksum. Cloned sparse and shallow-ish: only the
# bmw-m6 directory materializes (~72 MB with its sky image).
#
#   ./fetch-showcase.sh
#   cargo run --release -p cenote-cli -- import \
#       tests/scenes/showcase/bmw-m6/bmw-m6.pbrt --out /tmp/bmw.ron
set -eu

COMMIT=30cf4a0346ae5a80a2d7a530a3ef7d0fa4f70572
DEST="$(dirname "$0")/showcase"

if [ -e "$DEST/bmw-m6/bmw-m6.pbrt" ]; then
    echo "already fetched: $DEST/bmw-m6"
    exit 0
fi

mkdir -p "$DEST"
git -C "$DEST" init -q
git -C "$DEST" remote add origin https://github.com/mmp/pbrt-v4-scenes.git 2>/dev/null || true
git -C "$DEST" sparse-checkout set bmw-m6
git -C "$DEST" fetch -q --filter=blob:none origin "$COMMIT"
git -C "$DEST" checkout -q "$COMMIT"
echo "fetched pbrt-v4-scenes@$COMMIT: $DEST/bmw-m6"
