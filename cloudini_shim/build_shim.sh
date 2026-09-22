#!/usr/bin/env bash
# Builds libcloudini_shim.so — a self-contained ctypes-friendly decoder for
# cloudini-compressed point clouds (point_cloud_interfaces/CompressedPointCloud2).
#
# Usage: ./build_shim.sh [path-to-cloudini-checkout]
# Requires: cmake, g++ (C++20). libzstd-dev / liblz4-dev are used if present;
# otherwise cloudini's vendored copies are built instead, so the script works
# on a machine without those headers.
#
# ABI FLOOR -- THE THING THAT ACTUALLY BITES. The .so records the glibc of the
# machine that built it and will not load on anything older. A shim built in
# a ROS 2 Jazzy (Ubuntu 24.04) container came out needing GLIBC_2.38 and loaded
# on NEITHER an Ubuntu 22.04 workstation (2.35) NOR a cluster's RHEL 8 nodes
# (2.28), which is silent until the first decode() and then surfaces as a bare
# OSError from ctypes. So BUILD ON THE OLDEST MACHINE YOU NEED TO SUPPORT. The
# committed binary was built on Ubuntu 22.04 (g++ 11.4, glibc 2.35) and needs
# GLIBC_2.32 / GLIBCXX_3.4.30 -- fine there and newer, too new for glibc 2.28.
# Check with:
#   objdump -T libcloudini_shim.so | grep -oE "GLIBC[A-Z_]*_[0-9.]+" | sort -uV | tail
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CLOUDINI_SRC="${1:-$HERE/cloudini}"

if [ ! -d "$CLOUDINI_SRC" ]; then
    git clone --depth 1 https://github.com/facontidavide/cloudini "$CLOUDINI_SRC"
fi

BUILD_DIR="$HERE/build"

# Use the system zstd/lz4 when their headers are installed, otherwise let
# cloudini vendor them. Without this the configure step succeeds and the final
# link fails on -lzstd/-llz4, which reads like a cloudini problem and is not.
if [ -f /usr/include/zstd.h ] && [ -f /usr/include/lz4.h ]; then
    VENDORED=OFF
    SYSLIBS="-lzstd -llz4"
else
    echo "libzstd/liblz4 headers not found -- building cloudini's vendored copies"
    VENDORED=ON
    SYSLIBS=""
fi

cmake -B "$BUILD_DIR" -S "$CLOUDINI_SRC/cloudini_lib" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCLOUDINI_BUILD_TOOLS=OFF \
    -DCLOUDINI_BUILD_BENCHMARKS=OFF \
    -DCLOUDINI_FORCE_VENDORED_DEPS=$VENDORED \
    -DBUILD_TESTING=OFF
cmake --build "$BUILD_DIR" --target cloudini_lib -j"$(nproc)"

# When vendored, the static archives land in the build tree and must be linked
# explicitly; when not, the system -l flags cover it.
VENDORED_ARCHIVES=""
if [ "$VENDORED" = "ON" ]; then
    VENDORED_ARCHIVES=$(find "$BUILD_DIR" \
        \( -name "liblibzstd_static.a" -o -name "libzstd*.a" -o -name "liblz4*.a" \) \
        | tr "\n" " ")
fi

OUT="$HERE/../dataset_tools/libcloudini_shim.so"
g++ -shared -fPIC -std=c++20 -O2 \
    -I"$CLOUDINI_SRC/cloudini_lib/include" \
    "$HERE/cloudini_shim.cpp" \
    "$BUILD_DIR/libcloudini_lib.a" \
    $VENDORED_ARCHIVES $SYSLIBS \
    -o "$OUT"

echo "Built $(realpath "$OUT")"
echo "ABI floor: $(objdump -T "$OUT" | grep -oE "GLIBC[A-Z_]*_[0-9.]+" | sort -uV | tail -2 | tr "\n" " ")"
