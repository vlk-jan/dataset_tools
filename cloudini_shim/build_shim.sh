#!/usr/bin/env bash
# Builds libcloudini_shim.so — a self-contained ctypes-friendly decoder for
# cloudini-compressed point clouds (point_cloud_interfaces/CompressedPointCloud2).
#
# Usage: ./build_shim.sh [path-to-cloudini-checkout]
# Requires: cmake, g++ (C++20). System libzstd/liblz4 if present, else vendored.
#
# BUILD ON THE OLDEST MACHINE YOU NEED TO SUPPORT: the .so won't load on an older
# glibc (a bare ctypes OSError). The committed binary (Ubuntu 22.04) needs
# GLIBC_2.32 / GLIBCXX_3.4.30. Check with:
#   objdump -T libcloudini_shim.so | grep -oE "GLIBC[A-Z_]*_[0-9.]+" | sort -uV | tail
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CLOUDINI_SRC="${1:-$HERE/cloudini}"

if [ ! -d "$CLOUDINI_SRC" ]; then
    git clone --depth 1 https://github.com/facontidavide/cloudini "$CLOUDINI_SRC"
fi

BUILD_DIR="$HERE/build"

# Without this check, a missing zstd/lz4 only fails at the final link.
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

# Vendored static archives must be linked explicitly.
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
