"""Raw message decoding: header stamps out of CDR bytes, PointCloud2 to xyzi."""

from __future__ import annotations

import struct

import numpy as np

from dataset_tools import cloudini


def header_stamp_from_cdr(raw: bytes) -> float | None:
    """``header.stamp`` read straight out of a CDR buffer, without deserialising.

    Valid only for messages whose first field is a ``std_msgs/Header``: after
    the 4-byte CDR encapsulation (whose second byte says little- or big-endian)
    the header starts with ``stamp.sec`` (int32) and ``stamp.nanosec``
    (uint32). Images and clouds are megabytes each and the first pass needs
    nothing but this number, so skipping the full decode is what makes that
    pass cheap. Returns None for a buffer too short to hold a stamp.
    """
    if len(raw) < 12:
        return None
    fmt = "<iI" if raw[1] == 1 else ">iI"
    sec, nanosec = struct.unpack_from(fmt, raw, 4)
    return sec + nanosec * 1e-9


# sensor_msgs/PointField datatype enum -> numpy type
PC2_DTYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}


def fields_to_dtype(fields, point_step: int, big_endian: bool = False) -> np.dtype:
    endian = ">" if big_endian else "<"
    return np.dtype(
        {
            "names": [f.name for f in fields],
            "formats": [endian + PC2_DTYPES[f.datatype] for f in fields],
            "offsets": [f.offset for f in fields],
            "itemsize": point_step,
        }
    )


def cloud_to_xyzi(msg) -> np.ndarray:
    """PointCloud2 or CompressedPointCloud2 message -> (N, 4) float32 [x y z i]."""
    # point_cloud_interfaces/CompressedPointCloud2 carries a cloudini blob instead of raw data
    data = cloudini.decode(bytes(msg.compressed_data)) if hasattr(msg, "compressed_data") else bytes(msg.data)
    dtype = fields_to_dtype(msg.fields, msg.point_step, msg.is_bigendian)
    points = np.frombuffer(data, dtype=dtype, count=msg.width * msg.height)
    names = points.dtype.names
    intensity = (
        points["intensity"].astype(np.float32) if "intensity" in names else np.zeros(len(points), dtype=np.float32)
    )
    return np.stack(
        [
            points["x"].astype(np.float32),
            points["y"].astype(np.float32),
            points["z"].astype(np.float32),
            intensity,
        ],
        axis=1,
    )


def drop_invalid_returns(xyzi: np.ndarray) -> np.ndarray:
    """Drop the two things a sensor emits for a shot that returned nothing.

    An organised cloud ships every element every sweep (Odin1: 256x192 = 49152,
    constant), so a missed shot has to be encoded rather than omitted. Both
    encodings occur in the same bag: NaN, and exactly (0, 0, 0).

    This MUST run before the cloud is transformed into base_link: afterwards
    the zeros land on the mount translation, where they read as a dense real
    object rather than a sentinel. That is HELHEST v1, where 54-70% of every
    published cloud is (-0.170, 0, 0.145) repeated ten to fourteen thousand
    times. Measured per sweep on the 2026-08-21 bags: 49152 elements, 11.2%
    NaN, 47.1% of the rest dropped shots, ~23k real returns left.

    Exact zeros only -- (0, 0, 1e-6) is a real return against the sensor, and
    range gating is the consumer's decision, not this function's.
    """
    xyzi = xyzi[~np.isnan(xyzi).any(axis=1)]
    return xyzi[np.any(xyzi[:, :3] != 0, axis=1)]
