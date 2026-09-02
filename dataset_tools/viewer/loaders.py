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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger("dataset_tools.viewer.loaders")

# Per-frame BEV height map geometry. These are the values the DRIFT height-map
# generator uses; they are re-derived per file from the array shape where
# possible so a differently-sized map still lands in the right place.
HM_RES = 0.1
HM_EXTENT = (-6.4, 6.4)

VALIDITY_LABELS = {0: "extrapolated", 1: "interpolated", 2: "measured"}


# --------------------------------------------------------------------------
# small file readers
# --------------------------------------------------------------------------


def read_timestamps(path: Path) -> np.ndarray | None:
    """Read a one-timestamp-per-line file (``data_ts.txt`` / ``pcd_ts.txt``)."""
    try:
        with open(path) as fh:
            vals = [float(line) for line in fh if line.strip() and not line.startswith("#")]
    except (OSError, ValueError) as exc:
        logger.warning("cannot read timestamps %s: %s", path, exc)
        return None
    return np.asarray(vals, dtype=np.float64) if vals else None


def count_lines(path: Path) -> int:
    """Line count of a text file, ignoring blanks and comments. 0 if unreadable."""
    try:
        with open(path, "rb") as fh:
            return sum(
                1 for line in fh if line.strip() and not line.lstrip().startswith(b"#")
            )
    except OSError:
        return 0


def read_matrix(path: Path, shape: tuple[int, ...] | None = None) -> np.ndarray | None:
    """Read a whitespace-separated numeric matrix (K, T_cam_to_base, ...)."""
    try:
        arr = np.loadtxt(path, dtype=np.float64, comments="#")
    except (OSError, ValueError) as exc:
        logger.warning("cannot read matrix %s: %s", path, exc)
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
        logger.warning("cannot load %s: %s", path, exc)
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
        logger.warning("cannot load %s: %s", path, exc)
        return None


def read_text(path: Path, max_bytes: int = 200_000) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read(max_bytes)
    except OSError:
        return None


# --------------------------------------------------------------------------
# odometry
# --------------------------------------------------------------------------


def read_odometry(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Parse ``odometry.txt``: ``ts`` + a row-major 4x4 per line.

    Accepts both the 17-column form (timestamp + full 4x4, trailing ``0 0 0 1``
    included) and the 13-column form (timestamp + the top 3x4 only), so a rig
    that writes the shorter row still loads.

    Returns ``(ts (N,), poses (N, 4, 4))`` sorted by timestamp, or None.
    """
    try:
        raw = np.loadtxt(path, dtype=np.float64, comments="#", ndmin=2)
    except (OSError, ValueError) as exc:
        logger.warning("cannot read odometry %s: %s", path, exc)
        return None
    if raw.size == 0:
        return None
    if raw.shape[1] == 17:
        ts, flat = raw[:, 0], raw[:, 1:]
        poses = flat.reshape(-1, 4, 4)
    elif raw.shape[1] == 13:
        ts, flat = raw[:, 0], raw[:, 1:]
        poses = np.zeros((raw.shape[0], 4, 4), dtype=np.float64)
        poses[:, :3, :] = flat.reshape(-1, 3, 4)
        poses[:, 3, 3] = 1.0
    else:
        logger.warning("odometry %s: unexpected %d columns", path, raw.shape[1])
        return None
    order = np.argsort(ts, kind="stable")
    return ts[order], poses[order]


@dataclass
class Segment:
    """One independently-anchored stretch of ground-truth odometry.

    Poses from different segments share **no** common frame -- each segment is
    re-anchored to identity at its own first pose -- so nothing (trajectory,
    aggregated cloud, terrain) may ever be built across a segment boundary.
    """

    index: int
    start_ts: float
    end_ts: float
    n_poses: int = 0
    reason: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end_ts - self.start_ts)


def read_segments(path: Path) -> list[Segment]:
    """Parse ``odometry_segments.txt`` (``id start_ts end_ts n_poses reason``).

    Unknown trailing columns are folded into ``reason``; unparseable lines are
    skipped. Returns [] when the file is missing or empty.
    """
    segments: list[Segment] = []
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    idx = int(float(parts[0]))
                    start, end = float(parts[1]), float(parts[2])
                except ValueError:
                    continue
                n_poses = 0
                reason = ""
                if len(parts) >= 4:
                    try:
                        n_poses = int(float(parts[3]))
                        reason = " ".join(parts[4:])
                    except ValueError:
                        reason = " ".join(parts[3:])
                segments.append(Segment(idx, start, end, n_poses, reason))
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
    src_ts: np.ndarray, src_poses: np.ndarray, query_ts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate a pose track at arbitrary times.

    Translation is linear, rotation is slerp. Returns
    ``(positions (M, 3), quaternions (M, 4), valid (M,) bool)``; ``valid`` is
    False for query times outside ``[src_ts[0], src_ts[-1]]`` (those entries
    hold the clamped endpoint pose so callers can still render something, but
    must not treat them as ground truth).
    """
    query_ts = np.asarray(query_ts, dtype=np.float64)
    valid = (query_ts >= src_ts[0]) & (query_ts <= src_ts[-1])
    hi = np.clip(np.searchsorted(src_ts, query_ts, side="left"), 1, len(src_ts) - 1)
    lo = hi - 1
    span = src_ts[hi] - src_ts[lo]
    alpha = np.where(span > 0, (query_ts - src_ts[lo]) / np.where(span > 0, span, 1.0), 0.0)
    alpha = np.clip(alpha, 0.0, 1.0)

    pos = src_poses[lo, :3, 3] + alpha[:, None] * (
        src_poses[hi, :3, 3] - src_poses[lo, :3, 3]
    )
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


@dataclass
class ProprioTrack:
    """A resampled proprioception channel set, one value per requested time."""

    names: list[str] = field(default_factory=list)
    values: dict[str, np.ndarray] = field(default_factory=dict)


def resample_series(
    src_ts: np.ndarray, src_vals: np.ndarray, query_ts: np.ndarray, max_gap: float = 1.0
) -> np.ndarray:
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
    gap = np.minimum(
        np.abs(query_ts - src_ts[nearest - 1]), np.abs(src_ts[nearest] - query_ts)
    )
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
