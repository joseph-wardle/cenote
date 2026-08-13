#!/usr/bin/env sh
# Fetch the B11 volume assets into scenes/vdb/sources/ (which stays
# untracked — see .gitignore). Mirrors scenes/corpus/fetch.sh.
#
#   ./fetch.sh              # everything (~3.2 GB, mostly wdas_cloud)
#   ./fetch.sh bunny_cloud  # just the named assets
#
# The openvdb.org sample grids (CC-BY 4.0) come from the ASWF artifact
# registry. wdas_cloud is Disney Animation's data set, licensed for
# research and non-commercial use only — like the kroken sources, it may
# never be committed or redistributed from here; the zip carries its
# license text.
set -eu

SOURCES="$(cd "$(dirname "$0")" && pwd)/sources"
ASWF="https://artifacts.aswf.io/io/aswf/openvdb/models"

OPENVDB_ORG="bunny_cloud smoke2 explosion"

aswf() { # <grid>
    [ -e "$SOURCES/$1.vdb" ] && { echo "already fetched: $1"; return; }
    mkdir -p "$SOURCES"
    curl -fL --progress-bar -o "$SOURCES/$1.zip" \
        "$ASWF/$1.vdb/1.0.0/$1.vdb-1.0.0.zip"
    unzip -oq "$SOURCES/$1.zip" -d "$SOURCES"
    rm "$SOURCES/$1.zip"
    echo "fetched openvdb.org: $1"
}

wdas() {
    [ -e "$SOURCES/wdas_cloud/wdas_cloud.vdb" ] && {
        echo "already fetched: wdas_cloud"; return; }
    mkdir -p "$SOURCES"
    curl -fL --progress-bar -o "$SOURCES/wdas_cloud.zip" \
        "https://assets.disneyanimation.com/wdas_cloud.zip"
    unzip -oq "$SOURCES/wdas_cloud.zip" -d "$SOURCES"
    rm "$SOURCES/wdas_cloud.zip"
    echo "fetched wdas_cloud (research/non-commercial — never commit)"
}

listed() { # <name> <word...>
    name="$1"; shift
    for word in "$@"; do [ "$word" = "$name" ] && return 0; done
    return 1
}

fetch() { # <name>
    if   listed "$1" $OPENVDB_ORG;    then aswf "$1"
    elif [ "$1" = "wdas_cloud" ];     then wdas
    else
        echo "unknown asset: $1 (openvdb.org: $OPENVDB_ORG; wdas_cloud)" >&2
        exit 1
    fi
}

if [ $# -eq 0 ]; then
    for name in $OPENVDB_ORG wdas_cloud; do fetch "$name"; done
else
    for name in "$@"; do fetch "$name"; done
fi
