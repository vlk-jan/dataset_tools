"""Raw message decoding: header stamps out of CDR bytes, PointCloud2 to xyzi."""

from __future__ import annotations

import struct

import numpy as np

from dataset_tools import cloudini
from dataset_tools.transforms import MIN_VALID_STAMP


def header_stamp_from_cdr(raw: bytes) -> float | None:
    """``header.stamp`` read straight out of a CDR buffer, without deserialising.

    Valid only for messages whose first field is a ``std_msgs/Header``: after
    the 4-byte CDR encapsulation (whose second byte says little- or big-endian)
    the header starts with ``stamp.sec`` (int32) and ``stamp.nanosec``
    (uint32). Images and clouds are megabytes each and the first pass needs
    nothing but this number, so skipping the full decode is what makes that
    pass cheap. Returns None for a buffer too short to hold a stamp, and for a
    stamp that was never set (:data:`~dataset_tools.transforms.MIN_VALID_STAMP`),
    so the caller falls back exactly as :func:`~dataset_tools.transforms.header_stamp` does.
    """
    if len(raw) < 12:
        return None
    fmt = "<iI" if raw[1] == 1 else ">iI"
    sec, nanosec = struct.unpack_from(fmt, raw, 4)
    t = sec + nanosec * 1e-9
    return t if t >= MIN_VALID_STAMP else None


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


def _points(msg) -> np.ndarray:
    """The message's point records, one structured row per element."""
    # point_cloud_interfaces/CompressedPointCloud2 carries a cloudini blob instead of raw data
    data = cloudini.decode(bytes(msg.compressed_data)) if hasattr(msg, "compressed_data") else bytes(msg.data)
    dtype = fields_to_dtype(msg.fields, msg.point_step, msg.is_bigendian)
    return np.frombuffer(data, dtype=dtype, count=msg.width * msg.height)


def _xyzi(points: np.ndarray) -> np.ndarray:
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


def cloud_to_xyzi(msg) -> np.ndarray:
    """PointCloud2 or CompressedPointCloud2 message -> (N, 4) float32 [x y z i]."""
    return _xyzi(_points(msg))


def cloud_to_xyzit(msg, stamp: float) -> tuple[np.ndarray, np.ndarray | None]:
    """As :func:`cloud_to_xyzi`, plus each point's capture time relative to
    ``stamp`` (the header stamp) in seconds -- or None when the cloud carries
    no usable per-point time. Rows correspond, so one mask fits both."""
    points = _points(msg)
    return _xyzi(points), point_time_offsets(points, stamp)


#: Per-point time fields by the names drivers give them. A relative field holds
#: seconds when it is a float and nanoseconds when it is an integer (Odin1:
#: ``offset_time`` f32 s; Livox and Ouster: u32 ns); an absolute one holds the
#: capture time itself, in seconds or -- Livox's double -- in nanoseconds.
RELATIVE_TIME_FIELDS = ("offset_time", "time_offset", "time", "t")
ABSOLUTE_TIME_FIELDS = ("timestamp", "time_stamp")
#: No sweep is longer than this; a field whose values fall outside is not a time.
MAX_SWEEP_S = 1.0


def point_time_offsets(points: np.ndarray, stamp: float) -> np.ndarray | None:
    """Each point's capture time minus the header ``stamp``, in seconds.

    None when no known field is present or its values are not a plausible
    sweep -- a cloud without this is a cloud nothing can deskew, and the
    caller says so once rather than guessing.
    """
    names = points.dtype.names or ()
    for name in RELATIVE_TIME_FIELDS:
        if name in names:
            v = points[name]
            off = v.astype(np.float64) * (1e-9 if np.issubdtype(v.dtype, np.integer) else 1.0)
            return off if _plausible_offsets(off) else None
    for name in ABSOLUTE_TIME_FIELDS:
        if name in names:
            v = points[name]
            t = v.astype(np.float64)
            if np.issubdtype(v.dtype, np.integer) or np.nanmedian(t) > 1e12:
                t = t * 1e-9  # nanoseconds, whether counted in an integer or in a double
            off = t - stamp
            return off if _plausible_offsets(off) else None
    return None


def _plausible_offsets(off: np.ndarray) -> bool:
    finite = off[np.isfinite(off)]
    return len(finite) > 0 and float(finite.min()) >= -MAX_SWEEP_S and float(finite.max()) <= MAX_SWEEP_S


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
    return xyzi[valid_returns_mask(xyzi)]


def valid_returns_mask(xyzi: np.ndarray) -> np.ndarray:
    """The rows :func:`drop_invalid_returns` keeps, as a mask -- for the
    per-point columns that have to follow the same rows."""
    return ~np.isnan(xyzi).any(axis=1) & np.any(xyzi[:, :3] != 0, axis=1)
