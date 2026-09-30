"""ctypes wrapper for libcloudini_shim.so -- decodes cloudini-compressed point
clouds (point_cloud_interfaces/msg/CompressedPointCloud2, as produced by the
Odin1 unit on HELHEST). Build the shim with cloudini_shim/build_shim.sh."""

import ctypes
import functools
from pathlib import Path

_LIB_PATH = Path(__file__).resolve().parent / "libcloudini_shim.so"


@functools.cache
def _get_lib():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found. Build it with cloudini_shim/build_shim.sh "
            "(needs cmake and a C++20 g++; zstd/lz4 headers are used when present)."
        )
    try:
        lib = ctypes.CDLL(str(_LIB_PATH))
    except OSError as exc:
        # Almost always an ABI floor, not a missing file: the .so records
        # the glibc of its build machine and refuses to load on anything
        # older. The committed binary was built on Ubuntu 22.04 (glibc
        # 2.35) and needs GLIBC_2.32 / GLIBCXX_3.4.30; a host on an older
        # glibc (2.28, say) cannot load it and has to rebuild.
        raise RuntimeError(
            f"could not load {_LIB_PATH}: {exc}\n"
            "If this mentions a GLIBC/GLIBCXX version, the shim was built "
            "on a newer machine than this one. Rebuild it here with "
            "cloudini_shim/build_shim.sh -- and note that the result then "
            "carries THIS machine's floor, so build on the oldest host you "
            "need to support."
        ) from exc
    lib.cldn_shim_decoded_size.restype = ctypes.c_uint64
    lib.cldn_shim_decoded_size.argtypes = [ctypes.c_char_p, ctypes.c_uint64]
    lib.cldn_shim_decode.restype = ctypes.c_uint64
    lib.cldn_shim_decode.argtypes = [
        ctypes.c_char_p,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_uint64,
    ]
    return lib


def decode(compressed_data: bytes) -> bytes:
    """Decode a cloudini blob (the `compressed_data` field of a
    CompressedPointCloud2 message) into the raw PointCloud2 data buffer."""
    lib = _get_lib()
    size = lib.cldn_shim_decoded_size(compressed_data, len(compressed_data))
    if size == 0:
        raise ValueError("cloudini: failed to decode header")
    out = ctypes.create_string_buffer(int(size))
    written = lib.cldn_shim_decode(compressed_data, len(compressed_data), out, size)
    if written == 0:
        raise ValueError("cloudini: failed to decompress point cloud")
    return out.raw[:written]
