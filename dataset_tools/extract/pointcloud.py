"""Raw message decoding: header stamps out of CDR bytes, PointCloud2 to xyzi."""

from __future__ import annotations

import struct

import numpy as np

from dataset_tools import cloudini
from dataset_tools.transforms import MIN_VALID_STAMP


def header_stamp_from_cdr(raw: bytes) -> float | None:
    """``header.stamp`` read straight out of a CDR buffer, without deserialising.

    Only for messages starting with a ``std_msgs/Header``. None for a short buffer
    or an unset stamp, matching :func:`~dataset_tools.transforms.header_stamp`.
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


def cloud_to_xyzit(msg, stamp: float) -> tuple[np.ndarray, np.ndarray | None]:
    """(Compressed)PointCloud2 -> (N, 4) float32 [x y z i] and per-point time offsets
    from ``stamp`` in seconds (or None), row-aligned."""
    points = _points(msg)
    return _xyzi(points), point_time_offsets(points, stamp)


#: Per-point time fields. Relative: float seconds or integer ns; absolute: seconds or ns.
RELATIVE_TIME_FIELDS = ("offset_time", "time_offset", "time", "t")
ABSOLUTE_TIME_FIELDS = ("timestamp", "time_stamp")
#: No sweep is longer than this; a field whose values fall outside is not a time.
MAX_SWEEP_S = 1.0


def point_time_offsets(points: np.ndarray, stamp: float) -> np.ndarray | None:
    """Each point's capture time minus ``stamp`` in seconds; None without a plausible time field."""
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


def valid_returns_mask(xyzi: np.ndarray) -> np.ndarray:
    """Mask off missed shots in an organised cloud: NaN rows and exact (0, 0, 0).

    Apply BEFORE transforming to base_link, or the zeros become a dense fake
    object at the mount translation (as in HELHEST v1).
    """
    return ~np.isnan(xyzi).any(axis=1) & np.any(xyzi[:, :3] != 0, axis=1)
