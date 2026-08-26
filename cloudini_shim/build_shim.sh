#!/usr/bin/env bash
# Builds libcloudini_shim.so — a self-contained ctypes-friendly decoder for
# cloudini-compressed point clouds (point_cloud_interfaces/CompressedPointCloud2).
#
# Usage: ./build_shim.sh [path-to-cloudini-checkout]
# Requires: cmake, g++ (C++20), libzstd-dev, liblz4-dev
# (all present in helhest-jazzy.sif; run inside the container if unsure).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CLOUDINI_SRC="${1:-$HERE/cloudini}"

if [ ! -d "$CLOUDINI_SRC" ]; then
    git clone --depth 1 https://github.com/facontidavide/cloudini "$CLOUDINI_SRC"
fi

BUILD_DIR="$HERE/build"
cmake -B "$BUILD_DIR" -S "$CLOUDINI_SRC/cloudini_lib" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCLOUDINI_BUILD_TOOLS=OFF \
    -DCLOUDINI_BUILD_BENCHMARKS=OFF \
    -DCLOUDINI_FORCE_VENDORED_DEPS=OFF \
    -DBUILD_TESTING=OFF
cmake --build "$BUILD_DIR" --target cloudini_lib -j"$(nproc)"

g++ -shared -fPIC -std=c++20 -O2 \
    -I"$CLOUDINI_SRC/cloudini_lib/include" \
    "$HERE/cloudini_shim.cpp" \
    "$BUILD_DIR/libcloudini_lib.a" \
    -lzstd -llz4 \
    -o "$HERE/../dataset_tools/libcloudini_shim.so"

echo "Built $(realpath "$HERE/../dataset_tools/libcloudini_shim.so")"
