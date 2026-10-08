"""Readers for the per-scene files an extracted dataset root carries.

Readers never raise on a malformed or absent file; they return ``None`` or empty.
Poses are ``(4, 4)`` with ``p_world = T @ p_body``; quaternions are ``[w, x, y, z]``.
Height maps are ``[row=x, col=y]``; SimView's terrain wants ``[row=y, col=x]``, so transpose.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from .. import gt_io, gt_map
from ..gt_io import SegmentRow as Segment

logger = logging.getLogger("dataset_tools.viewer.loaders")

__all__ = ["Segment"]

# Per-frame BEV height map geometry, as DRIFT's height-map generator writes it.
HM_RES = 0.1
HM_EXTENT = (-6.4, 6.4)

# HELHEST geometry (DRIFT's robot model): base_link IS the front axle. Hubs and
# radius come from ``gt_map`` so the drawn robot is the one the contact check uses.
# The chassis box carries base_link's trajectory, its centre shifted back by
# CHASSIS_CENTER (body frame) to span the real chassis. Shared by build and render.
CHASSIS = dict(hx=0.35, hy=0.25, hz=0.15)
CHASSIS_CENTER = (-0.15, 0.0, 0.0)
WHEEL_RADIUS = gt_map.WHEEL_RADIUS
WHEEL_WIDTH = 0.142
#: name -> (x, y, z) hub in base frame, joint_states column, x-align sign.
#: ``gt_map.WHEELS`` rows are left, right, rear; the bodies are made in
#: joint_states order (left, rear, right).
_LEFT, _RIGHT, _REAR = (tuple(float(v) for v in row) for row in gt_map.WHEELS)
WHEELS = {
    "wheel_left": (*_LEFT, 0.0, 0, -1.0),
    "wheel_rear": (*_REAR, 0.0, 1, -1.0),
    "wheel_right": (*_RIGHT, 0.0, 2, 1.0),
}

VALIDITY_LABELS = {
    0: "extrapolated",
    1: "interpolated",
    2: "measured",
    3: "anchored",
}

#: "How founded in lidar returns is this cell", indexed by validity code. Codes are
#: provenance, not a scale: 3 (wheel-contact anchor) ranks with extrapolation, so
#: average/threshold through this, never the raw code. Mirrors DRIFT's ``utils.pcd.VALIDITY_RANK``.
VALIDITY_RANK = np.array([0, 1, 2, 0], dtype=np.uint8)


# --------------------------------------------------------------------------
# small file readers
# --------------------------------------------------------------------------


def _unreadable(what: str, path: Path, exc: Exception) -> None:
    """Log a failed read: debug for a missing (optional) file, warning otherwise."""
    level = logger.debug if isinstance(exc, FileNotFoundError) else logger.warning
    level("cannot %s %s: %s", what, path, exc)


def read_timestamps(path: Path) -> np.ndarray | None:
    """Read a one-timestamp-per-line file (``data_ts.txt`` / ``pcd_ts.txt``)."""
    try:
        with open(path) as fh:
            vals = [float(line) for line in fh if line.strip() and not line.startswith("#")]
    except (OSError, ValueError) as exc:
        _unreadable("read timestamps", path, exc)
        return None
    return np.asarray(vals, dtype=np.float64) if vals else None


def count_lines(path: Path) -> int:
    """Line count of a text file, ignoring blanks and comments. 0 if unreadable."""
    try:
        with open(path, "rb") as fh:
            return sum(1 for line in fh if line.strip() and not line.lstrip().startswith(b"#"))
    except OSError:
        return 0


def read_matrix(path: Path, shape: tuple[int, ...] | None = None) -> np.ndarray | None:
    """Read a whitespace-separated numeric matrix (K, T_cam_to_base, ...)."""
    try:
        arr = np.loadtxt(path, dtype=np.float64, comments="#")
    except (OSError, ValueError) as exc:
        _unreadable("read matrix", path, exc)
        return None
    if shape is not None:
        try:
            arr = arr.reshape(shape)
        except ValueError:
            logger.warning("matrix %s has shape %s, expected %s", path, arr.shape, shape)
            return None
    return arr


def load_npy(path: Path) -> np.ndarray | None:
    """``np.load`` that returns None instead of raising."""
    try:
        return np.load(path)
    except (OSError, ValueError) as exc:
        _unreadable("load", path, exc)
        return None


def load_npz_key(path: Path, key: str) -> np.ndarray | None:
    """Load one array out of an ``.npz``, tolerating a differently-named key
    (falls back to the file's single array when ``key`` is absent)."""
    try:
        with np.load(path) as data:
            if key in data:
                return np.asarray(data[key])
            names = list(data.keys())
            if len(names) == 1:
                return np.asarray(data[names[0]])
            logger.warning("%s has no key %r (has %s)", path, key, names)
            return None
    except (OSError, ValueError) as exc:
        _unreadable("load", path, exc)
        return None


def read_text(path: Path, max_bytes: int = 200_000) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read(max_bytes)
    except OSError:
        return None


def _scalar(text: str) -> Any:
    text = text.strip()
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text


def read_camera_info(path: Path) -> dict[str, Any] | None:
    """The flat subset of a ROS ``camera_info.yaml``.

    Hand-rolled (no YAML dependency) for the extractor's flat output. Nested
    blocks become ``<parent>`` (the data list) and ``<parent>_rows`` / ``_cols``.
    """
    text = read_text(path, 200_000)
    if not text:
        return None
    info: dict[str, Any] = {}
    parent: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, val = line.partition(":")
        if not sep:
            continue
        key, val = key.strip(), val.strip()
        if len(raw) - len(raw.lstrip()) == 0:
            parent = key
            if val:
                info[key] = _scalar(val)
        elif parent and key in ("rows", "cols"):
            info[f"{parent}_{key}"] = _scalar(val)
        elif parent and key == "data":
            try:
                info[parent] = [float(v) for v in val.strip("[] ").split(",") if v.strip()]
            except ValueError:
                continue
    return info or None


# --------------------------------------------------------------------------
# odometry
# --------------------------------------------------------------------------


def read_gps(path: Path) -> np.ndarray | None:
    """``gps.npy`` as ``(M, >= 4)`` sorted by time, or None if it is unusable.

    Rows with non-finite time/lat/lon are dropped.
    """
    arr = load_npy(path)
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 4:
        logger.warning("gps %s: expected (M, >=4), got %s", path, getattr(arr, "shape", None))
        return None
    good = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1]) & np.isfinite(arr[:, 2])
    arr = arr[good]
    if not len(arr):
        return None
    return arr[np.argsort(arr[:, 0], kind="stable")]


def read_odometry(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Parse ``odometry.txt`` (:func:`dataset_tools.gt_io.read_trajectory`).

    Returns ``(ts (N,), poses (N, 4, 4))`` sorted by timestamp, or None for a
    missing, malformed or empty file.
    """
    try:
        ts, poses = gt_io.read_trajectory(path)
    except (OSError, ValueError) as exc:
        logger.warning("cannot read odometry %s: %s", path, exc)
        return None
    if ts.size == 0:
        return None
    return ts, poses


# Segments share no common frame (each is anchored at its own first pose), so
# nothing may ever be built across a segment boundary.


def read_segments(path: Path) -> list[Segment]:
    """Parse ``odometry_segments.txt``, sorted by start time.

    Unknown trailing columns are folded into ``reason``; unparseable lines are
    skipped. Returns [] when the file is missing or empty.
    """
    try:
        segments = gt_io.read_segments(path)
    except OSError as exc:
        logger.warning("cannot read segments %s: %s", path, exc)
        return []
    segments.sort(key=lambda s: s.start_ts)
    return segments


# --------------------------------------------------------------------------
# rotations
# --------------------------------------------------------------------------


def roll_pitch_from_gravity_align(R_ga: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Roll/pitch (radians) implied by the per-frame gravity-alignment rotation.

    ``R_ga`` maps base-frame points into the gravity-aligned frame, so the
    world's up axis expressed in the base frame is ``R_ga.T @ [0, 0, 1]``.
    """
    R_ga = np.asarray(R_ga, dtype=np.float64)
    up_base = R_ga[:, 2, :]  # == (R_ga.T @ e_z) per frame
    roll = np.arctan2(up_base[:, 1], up_base[:, 2])
    pitch = -np.arctan2(up_base[:, 0], np.hypot(up_base[:, 1], up_base[:, 2]))
    return roll, pitch


# --------------------------------------------------------------------------
# per-frame heavy data
# --------------------------------------------------------------------------


def load_cloud(path: Path) -> np.ndarray | None:
    """Load one cloud, returning (N, >=3) float32 with NaN rows dropped."""
    arr = load_npy(path)
    if arr is None or arr.ndim != 2 or arr.shape[1] < 3:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr[:, :3]).all(axis=1)
    return arr[finite]


def height_map_cell_centers(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Local (x, y) cell-centre coordinates of a BEV height map.

    ``shape`` is ``(rows, cols)`` with row -> x (forward) and col -> y (left),
    starting at ``HM_EXTENT[0]`` with :data:`HM_RES` spacing.
    """
    rows, cols = shape
    x = HM_EXTENT[0] + HM_RES * (np.arange(rows) + 0.5)
    y = HM_EXTENT[0] + HM_RES * (np.arange(cols) + 0.5)
    return x, y


#: A resampled value further than this (s) from every source sample is NaN.
RESAMPLE_MAX_GAP = 1.0


def resample_series(src_ts: np.ndarray, src_vals: np.ndarray, query_ts: np.ndarray) -> np.ndarray:
    """Linearly resample ``src_vals`` (T, K) onto ``query_ts``.

    Samples further than :data:`RESAMPLE_MAX_GAP` seconds from any source
    sample become NaN rather than being invented by extrapolation.
    """
    src_ts = np.asarray(src_ts, dtype=np.float64)
    src_vals = np.asarray(src_vals, dtype=np.float64)
    if src_vals.ndim == 1:
        src_vals = src_vals[:, None]
    order = np.argsort(src_ts, kind="stable")
    src_ts, src_vals = src_ts[order], src_vals[order]

    out = np.empty((len(query_ts), src_vals.shape[1]), dtype=np.float64)
    for k in range(src_vals.shape[1]):
        out[:, k] = np.interp(query_ts, src_ts, src_vals[:, k])

    nearest = np.searchsorted(src_ts, query_ts)
    nearest = np.clip(nearest, 1, len(src_ts) - 1)
    gap = np.minimum(np.abs(query_ts - src_ts[nearest - 1]), np.abs(src_ts[nearest] - query_ts))
    out[gap > RESAMPLE_MAX_GAP] = np.nan
    return out


def frame_indices_dir(path: Path, suffix: str) -> list[int]:
    """Sorted integer frame indices of ``NNNNN<suffix>`` files in a directory."""
    try:
        names = os.listdir(path)
    except OSError:
        return []
    out = []
    for name in names:
        if not name.endswith(suffix):
            continue
        stem = name[: -len(suffix)]
        if stem.isascii() and stem.isdigit():  # "⑤".isdigit() is true; int() refuses it
            out.append(int(stem))
    out.sort()
    return out
