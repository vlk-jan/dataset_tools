"""ctypes wrapper for libcloudini_shim.so — decodes cloudini-compressed point
clouds (point_cloud_interfaces/msg/CompressedPointCloud2, as produced by the
Odin1 unit on HELHEST). Build the shim with cloudini_shim/build_shim.sh."""

import ctypes
from pathlib import Path

_LIB_PATH = Path(__file__).resolve().parent / "libcloudini_shim.so"
_lib = None


def _get_lib():
    global _lib
    if _lib is None:
        if not _LIB_PATH.exists():
            raise RuntimeError(
                f"{_LIB_PATH} not found. Build it with cloudini_shim/build_shim.sh "
                "(inside helhest-jazzy.sif or any machine with cmake/g++/zstd/lz4)."
            )
        lib = ctypes.CDLL(str(_LIB_PATH))
        lib.cldn_shim_decoded_size.restype = ctypes.c_uint64
        lib.cldn_shim_decoded_size.argtypes = [ctypes.c_char_p, ctypes.c_uint64]
        lib.cldn_shim_decode.restype = ctypes.c_uint64
        lib.cldn_shim_decode.argtypes = [
            ctypes.c_char_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_uint64,
        ]
        _lib = lib
    return _lib


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
