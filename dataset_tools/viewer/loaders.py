"""Readers for the per-scene files an extracted dataset root carries.

Every reader is defensive: a malformed or absent file yields ``None`` (or an
empty result) rather than raising, because the viewer's whole contract is that
it must render whatever subset of the files a scene happens to have.

Geometry conventions used throughout:

* Poses are ``(4, 4)`` homogeneous matrices, ``p_world = T @ p_body``.
* Quaternions are ``[w, x, y, z]`` (scalar-first), matching SimView.
* Per-frame BEV height maps are ``height_map[row, col]`` with ``row`` indexing
  **x** (forward) and ``col`` indexing **y** (left) -- the layout
  ``scipy.stats.binned_statistic_2d(points[:, 0], points[:, 1], ...)`` produces
  in the generator these files come from. SimView's terrain wants the opposite
  (``[row=y, col=x]``), so anything handed to it is transposed.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from .. import gt_io
from ..gt_io import SegmentRow as Segment

logger = logging.getLogger("dataset_tools.viewer.loaders")

__all__ = ["Segment"]

# Per-frame BEV height map geometry. These are the values the DRIFT height-map
# generator uses; they are re-derived per file from the array shape where
# possible so a differently-sized map still lands in the right place.
HM_RES = 0.1
HM_EXTENT = (-6.4, 6.4)

VALIDITY_LABELS = {
    0: "extrapolated",
    1: "interpolated",
    2: "measured",
    3: "anchored",
}

#: "How founded in lidar returns is this cell", indexed by validity code.
#:
#: The codes are a provenance, not a scale: state 3 (a cell decided by a
#: wheel-contact anchor, in a root built with DRIFT's
#: ``build_heightmaps.py --anchors``) is invention with a kinematic basis, not
#: a better measurement, so it ranks with the extrapolation. Anything that
#: averages or thresholds validity has to go through this first -- taking the
#: raw code would read the anchors as the best-founded cells in the map. The
#: lookup is the identity on the three states every published root ships, so
#: it changes nothing for existing data. Mirrors ``utils.pcd.VALIDITY_RANK``
#: in DRIFT, which is where the masks are written.
VALIDITY_RANK = np.array([0, 1, 2, 0], dtype=np.uint8)


# --------------------------------------------------------------------------
# small file readers
# --------------------------------------------------------------------------


def _unreadable(what: str, path: Path, exc: Exception) -> None:
    """Log a failed read. A file that is simply not there is the ordinary
    shape of an optional artefact and gets debug; anything else is a warning."""
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


def load_npy(path: Path, mmap: bool = False) -> np.ndarray | None:
    """``np.load`` that returns None instead of raising."""
    try:
        return np.load(path, mmap_mode="r" if mmap else None)
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

    Hand-rolled rather than a YAML dependency: the file is machine-written by
    the extractor and is always flat scalars plus ``data: [...]`` rows. Nested
    blocks collapse to ``<parent>`` (the data list) and ``<parent>_rows`` /
    ``<parent>_cols``. Anything unparseable is simply left out.
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

    Rows whose latitude or longitude is not finite are dropped rather than
    drawn: a receiver reporting no position at all is a hole in the track, and
    a hole is not a place the robot was.
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


# ``Segment`` is :class:`dataset_tools.gt_io.SegmentRow`: one independently
# anchored stretch of ground truth. Poses from different segments share **no**
# common frame -- each is re-anchored to identity at its own first pose -- so
# nothing (trajectory, aggregated cloud, terrain) may ever be built across a
# segment boundary.


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


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) rotation matrices -> (..., 4) ``[w, x, y, z]`` quaternions.

    Shepperd's method: pick the branch with the largest denominator so the
    conversion stays numerically stable through 180-degree rotations.
    """
    R = np.asarray(R, dtype=np.float64)
    batch = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    n = R.shape[0]
    q = np.empty((n, 4), dtype=np.float64)

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    diag = np.stack([R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]], axis=1)
    use_trace = trace > 0.0

    # branch 0: trace positive
    if np.any(use_trace):
        idx = np.flatnonzero(use_trace)
        s = np.sqrt(trace[idx] + 1.0) * 2.0
        q[idx, 0] = 0.25 * s
        q[idx, 1] = (R[idx, 2, 1] - R[idx, 1, 2]) / s
        q[idx, 2] = (R[idx, 0, 2] - R[idx, 2, 0]) / s
        q[idx, 3] = (R[idx, 1, 0] - R[idx, 0, 1]) / s

    rest = np.flatnonzero(~use_trace)
    if rest.size:
        major = np.argmax(diag[rest], axis=1)
        for axis in (0, 1, 2):
            idx = rest[major == axis]
            if idx.size == 0:
                continue
            i, j, k = axis, (axis + 1) % 3, (axis + 2) % 3
            s = np.sqrt(1.0 + R[idx, i, i] - R[idx, j, j] - R[idx, k, k]) * 2.0
            q[idx, 0] = (R[idx, k, j] - R[idx, j, k]) / s
            q[idx, 1 + i] = 0.25 * s
            q[idx, 1 + j] = (R[idx, j, i] + R[idx, i, j]) / s
            q[idx, 1 + k] = (R[idx, k, i] + R[idx, i, k]) / s

    norm = np.linalg.norm(q, axis=1, keepdims=True)
    norm[norm == 0.0] = 1.0
    q /= norm
    # Canonical sign (w >= 0) so consecutive frames don't flip hemisphere.
    q[q[:, 0] < 0.0] *= -1.0
    return q.reshape(*batch, 4)


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Batched slerp between ``[w, x, y, z]`` quaternions; ``t`` is (N,)."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64).copy()
    t = np.asarray(t, dtype=np.float64)[:, None]

    dot = np.sum(q0 * q1, axis=1, keepdims=True)
    # Shortest arc.
    flip = dot < 0.0
    q1 = np.where(flip, -q1, q1)
    dot = np.abs(dot)

    out = np.empty_like(q0)
    close = (dot > 0.9995).ravel()
    if np.any(close):  # nearly parallel -> lerp, slerp is ill-conditioned
        out[close] = q0[close] + t[close] * (q1[close] - q0[close])
    far = ~close
    if np.any(far):
        theta0 = np.arccos(np.clip(dot[far], -1.0, 1.0))
        sin0 = np.sin(theta0)
        s0 = np.sin((1.0 - t[far]) * theta0) / sin0
        s1 = np.sin(t[far] * theta0) / sin0
        out[far] = s0 * q0[far] + s1 * q1[far]
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    norm[norm == 0.0] = 1.0
    return out / norm


def interpolate_poses(
    src_ts: np.ndarray, src_poses: np.ndarray, query_ts: np.ndarray, tol: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate a pose track at arbitrary times.

    Translation is linear, rotation is slerp. Returns
    ``(positions (M, 3), quaternions (M, 4), valid (M,) bool)``; ``valid`` is
    False for query times outside ``[src_ts[0], src_ts[-1]]`` (those entries
    hold the clamped endpoint pose so callers can still render something, but
    must not treat them as ground truth). ``tol`` widens that span: the pose and
    frame clocks are unrelated, so a frame a few ms outside it is still on the
    track, not off it.
    """
    query_ts = np.asarray(query_ts, dtype=np.float64)
    valid = (query_ts >= src_ts[0] - tol) & (query_ts <= src_ts[-1] + tol)
    hi = np.clip(np.searchsorted(src_ts, query_ts, side="left"), 1, len(src_ts) - 1)
    lo = hi - 1
    span = src_ts[hi] - src_ts[lo]
    alpha = np.where(span > 0, (query_ts - src_ts[lo]) / np.where(span > 0, span, 1.0), 0.0)
    alpha = np.clip(alpha, 0.0, 1.0)

    pos = src_poses[lo, :3, 3] + alpha[:, None] * (src_poses[hi, :3, 3] - src_poses[lo, :3, 3])
    quat = quat_slerp(mat_to_quat(src_poses[lo, :3, :3]), mat_to_quat(src_poses[hi, :3, :3]), alpha)
    return pos, quat, valid


def poses_from_quat(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """(M, 3) + (M, 4) [w,x,y,z] -> (M, 4, 4) homogeneous poses."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    R = np.empty((len(quat), 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    T = np.zeros((len(quat), 4, 4), dtype=np.float64)
    T[:, :3, :3] = R
    T[:, :3, 3] = pos
    T[:, 3, 3] = 1.0
    return T


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

    ``shape`` is ``(rows, cols)`` with row -> x (forward) and col -> y (left).
    The extent is assumed symmetric about the robot at :data:`HM_RES`
    resolution, which is what a square map of the documented size implies; a
    map of a different size keeps the same resolution and grows its extent.
    """
    rows, cols = shape
    x = HM_EXTENT[0] + HM_RES * (np.arange(rows) + 0.5)
    y = HM_EXTENT[0] + HM_RES * (np.arange(cols) + 0.5)
    return x, y


def resample_series(src_ts: np.ndarray, src_vals: np.ndarray, query_ts: np.ndarray, max_gap: float = 1.0) -> np.ndarray:
    """Linearly resample ``src_vals`` (T, K) onto ``query_ts``.

    Samples further than ``max_gap`` seconds from any source sample become NaN
    rather than being invented by extrapolation.
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
    out[gap > max_gap] = np.nan
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
        if stem.isdigit():
            out.append(int(stem))
    out.sort()
    return out
