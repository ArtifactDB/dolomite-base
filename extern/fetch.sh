#!/bin/bash

set -e
set -u

includedir=../src/dolomite_base/include
mkdir -p $includedir
transplant_headers() {
    local src=$1
    dest=$includedir/$(basename $src)
    rm -rf $dest
    cp -r sources/$src $dest
}

# Simple fetching, when a submodule can be directly added and the
# header files are directly available from the repository.

mkdir -p sources
fetch_simple() {
    local name=$1
    local url=$2
    local version=$3

    if [ ! -e sources/$name ]
    then
        git clone $url sources/$name
    else
        cd sources/$name
        git fetch --all
        git checkout $version
        cd -
    fi
}

fetch_simple comservatory https://github.com/ArtifactDB/comservatory v1.0.0
transplant_headers comservatory/include/comservatory

fetch_simple uzuki2 https://github.com/ArtifactDB/uzuki2 v1.1.0
transplant_headers uzuki2/include/uzuki2
