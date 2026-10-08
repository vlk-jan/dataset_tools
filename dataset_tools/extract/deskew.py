"""Motion compensation: every return of a sweep moved onto one instant.

The Odin1 scans bottom to top over 67 ms and each point carries its own
``offset_time``, so a turning robot shears the cloud by degrees. Each point is
carried from the pose at its capture time to the pose at the reference instant::

    p_ref = R(t_ref)^T ( R(t_p) p + x(t_p) - x(t_ref) ),   t_p = t_stamp + offset

with poses interpolated from the ground truth (rotation optionally from the gyro,
:class:`GyroTrack`). Points must already be in ``base_link``. Offsets are binned
to :data:`BIN_S` so poses are looked up once per bin. Sweeps that are already an
instant, or that the pose stream does not bracket, are left as recorded.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from dataset_tools.time_align import PoseTrack
from dataset_tools.transforms import rotation_angle_deg

#: Offsets are quantised to this before the poses are looked up.
BIN_S = 0.001
#: Max spacing of the pose samples bracketing a sweep.
MAX_GAP_S = 0.5
#: Gyro samples inside a sweep may be at most this far apart (the Odin1 IMU: 2.5 ms).
GYRO_MAX_GAP_S = 0.02
#: Max disagreement between gyro and slerp rotation over a sweep for the gyro to be used.
GYRO_TOL_DEG = 1.0


def deskew(
    points: np.ndarray,
    offsets: np.ndarray,
    stamp: float,
    t_ref: float,
    track: PoseTrack,
    gyro: GyroTrack | None = None,
) -> tuple[np.ndarray, str, str | None]:
    """``points`` (N, 3) taken at ``stamp + offsets``, re-expressed at ``t_ref``.

    Returns the points, a status (``deskewed``, ``instant``, ``no_pose``, ``empty``;
    points untouched unless ``deskewed``) and the rotation source (``"gyro"``,
    ``"slerp"``, or None when nothing moved).
    """
    pts = np.asarray(points)
    off = np.asarray(offsets, dtype=np.float64)
    if len(pts) == 0:
        return pts, "empty", None
    lo, hi = float(off.min()), float(off.max())
    if hi - lo < BIN_S and abs(stamp - t_ref) < BIN_S:
        return pts, "instant", None
    if not covered(track.ts, min(t_ref, stamp + lo), max(t_ref, stamp + hi)):
        return pts, "no_pose", None

    bins = np.round((stamp - t_ref + off) / BIN_S).astype(np.int64)
    uniq, inv = np.unique(bins, return_inverse=True)
    times = t_ref + uniq * BIN_S
    pos_u, rot_u = track.at(times)
    pos_r, rot_r = track.at(np.array([t_ref]))
    R_ref_T = rot_r[0].T
    R_rel = np.einsum("ij,njk->nik", R_ref_T, rot_u)  # (U, 3, 3): base at t_p -> base at t_ref
    method = "slerp"
    if gyro is not None:
        R_gyro = gyro.relative_rotations(t_ref, times)
        if R_gyro is not None:
            # Check agreement at the farthest bin.
            far = int(np.argmax(np.abs(times - t_ref)))
            if rotation_angle_deg(R_rel[far].T @ R_gyro[far]) <= GYRO_TOL_DEG:
                R_rel, method = R_gyro, "gyro"
    t_rel = (pos_u - pos_r[0]) @ R_ref_T.T  # (U, 3)
    out = np.einsum("nij,nj->ni", R_rel[inv], pts.astype(np.float64)) + t_rel[inv]
    return out.astype(pts.dtype), "deskewed", method


@dataclass
class GyroTrack:
    """Body angular rates in the base frame, integrated (zero-order hold) for in-sweep rotation."""

    ts: np.ndarray
    omega: np.ndarray

    @classmethod
    def build(cls, rows, R_to_base: np.ndarray | None = None) -> GyroTrack | None:
        """From ``[(t, wx, wy, wz), ...]`` in the sensor frame, rotated into the base frame."""
        if rows is None or len(rows) < 2:
            return None
        a = np.asarray(rows, dtype=np.float64)
        a = a[np.argsort(a[:, 0], kind="stable")]
        w = a[:, 1:4] if R_to_base is None else a[:, 1:4] @ np.asarray(R_to_base, float).T
        return cls(a[:, 0], w)

    def relative_rotations(self, t_ref: float, times: np.ndarray) -> np.ndarray | None:
        """``R[k] = R_world(t_ref)^T R_world(times[k])``, or None when the
        samples do not cover the span with gaps of at most :data:`GYRO_MAX_GAP_S`."""
        lo, hi = min(t_ref, float(times.min())), max(t_ref, float(times.max()))
        i0 = int(np.searchsorted(self.ts, lo, side="right")) - 1
        i1 = int(np.searchsorted(self.ts, hi, side="left"))
        if i0 < 0 or i1 >= len(self.ts) or np.any(np.diff(self.ts[i0 : i1 + 1]) > GYRO_MAX_GAP_S):
            return None
        seg_t, seg_w = self.ts[i0 : i1 + 1], self.omega[i0 : i1 + 1]
        cum = [np.eye(3)]
        for j in range(len(seg_t) - 1):
            cum.append(cum[-1] @ cv2.Rodrigues(seg_w[j] * (seg_t[j + 1] - seg_t[j]))[0])

        def at(t: float) -> np.ndarray:
            j = min(int(np.searchsorted(seg_t, t, side="right")) - 1, len(seg_t) - 1)
            return cum[j] @ cv2.Rodrigues(seg_w[j] * (t - seg_t[j]))[0]

        R_ref_T = at(t_ref).T
        return np.array([R_ref_T @ at(float(t)) for t in times])


def covered(sample_ts: np.ndarray, t_lo: float, t_hi: float, max_gap: float = MAX_GAP_S) -> bool:
    """Whether sorted ``sample_ts`` bracket ``[t_lo, t_hi]`` with no gap over ``max_gap``."""
    ts = np.asarray(sample_ts, dtype=np.float64)
    if len(ts) < 2 or t_lo < ts[0] or t_hi > ts[-1]:
        return False
    i0 = int(np.searchsorted(ts, t_lo, side="right")) - 1
    i1 = int(np.searchsorted(ts, t_hi, side="left"))
    return bool(np.all(np.diff(ts[i0 : i1 + 1]) <= max_gap))
