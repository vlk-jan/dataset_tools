"""Motion compensation: every return of a sweep moved onto one instant.

A lidar frame is not a snapshot. The Odin1 scans its 256 x 192 raster bottom
to top over 67 ms -- almost the whole 69 ms frame period -- and stamps the
frame at the start of that sweep; each point carries the time it was actually
taken (``offset_time``, 0 .. 0.067 s). While the robot turns, the top of the
cloud therefore shows the world a few degrees further round than the bottom:
at 60 deg/s the raster is sheared by 4 deg, at the 160 deg/s of a spin by 11.

Measured on HELHEST v2.2 (2026-10-01), before this module: the extractor wrote
the sweep as if every point were simultaneous, :mod:`dataset_tools.time_align`
then found the clouds' effective instant 27-52 ms after the stamp (the mean of
the offsets of the points it looks at) and moved the frame clock there -- and
the camera, exposed at the stamp, read as 30-50 ms *early* against the frame
clock. Nothing was wrong with the camera.

What this does
--------------
Each point is carried from the pose the robot held when the point was taken
to the pose it held at the frame's reference instant::

    p_ref = R(t_ref)^T ( R(t_p) p + x(t_p) - x(t_ref) ),   t_p = t_stamp + offset

with the pose interpolated from the ground-truth stream (translation linear,
rotation slerp -- or, given the Odin1's gyro, integrated from it, see
:class:`GyroTrack`). The points must already be in the frame the poses describe
(``base_link``), so this runs after the lidar mount transform. Offsets are
quantised to :data:`BIN_S` and the pose looked up once per bin: the Odin1
itself ticks in 2.2 ms slots, so 32 lookups serve 49 152 points.

A sweep whose span is below one bin is already an instant and is left alone,
as is one the pose stream does not bracket closely enough (a hole the GNSS
guard opened, or the stream's ends): guessing a motion there would smear a
cloud that is merely raw.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from dataset_tools.time_align import PoseTrack

#: Offsets are quantised to this before the poses are looked up.
BIN_S = 0.001
#: Pose samples bracketing a sweep must be at most this far apart for its
#: motion to be interpolated; the Odin1 odometry runs at 14 Hz, the Fixposition at 10.
MAX_GAP_S = 0.5
#: Gyro samples inside a sweep may be at most this far apart (the Odin1 IMU: 2.5 ms).
GYRO_MAX_GAP_S = 0.02
#: The gyro's rotation over the sweep must agree with the pose track's slerp to
#: this, or the gyro is not believed for that sweep. Constant-rate slerp is off
#: by ~0.2 deg at the robot's angular accelerations; a broken gyro by much more.
GYRO_TOL_DEG = 1.0

#: What :func:`deskew` can report back, in the order a summary lists them.
STATUSES = ("deskewed", "instant", "no_pose", "empty")


def deskew(
    points: np.ndarray,
    offsets: np.ndarray,
    stamp: float,
    t_ref: float,
    track: PoseTrack,
    gyro: GyroTrack | None = None,
    gyro_tol_deg: float = GYRO_TOL_DEG,
) -> tuple[np.ndarray, str, str | None]:
    """``points`` (N, 3) taken at ``stamp + offsets``, re-expressed at ``t_ref``.

    ``stamp`` is the header stamp the offsets count from; ``t_ref`` the frame's
    instant on the dataset's clock (on the Odin1 the two differ by under a
    millisecond, the camera's half of the pairing). Returns the moved points, a
    status from :data:`STATUSES` and how the rotation inside the sweep was
    found: ``"gyro"`` when a :class:`GyroTrack` covered the sweep and agreed
    with the pose track's slerp to ``gyro_tol_deg`` over it, ``"slerp"``
    otherwise, None when nothing was moved. The points come back untouched for
    every status but ``deskewed``. A sweep is "instant" when its offsets span
    less than one bin *and* the stamp is the reference: then there is no
    motion to undo.
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
            # The farthest bin is where the two disagree most; the gyro is
            # believed only where it agrees with the pose track to the tolerance.
            far = int(np.argmax(np.abs(times - t_ref)))
            if rotation_angle_deg(R_rel[far].T @ R_gyro[far]) <= gyro_tol_deg:
                R_rel, method = R_gyro, "gyro"
    t_rel = (pos_u - pos_r[0]) @ R_ref_T.T  # (U, 3)
    out = np.einsum("nij,nj->ni", R_rel[inv], pts.astype(np.float64)) + t_rel[inv]
    return out.astype(pts.dtype), "deskewed", method


def rotation_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


@dataclass
class GyroTrack:
    """Body angular rates in the base frame, for the rotation inside a sweep.

    A pose track sampled at 14 Hz can only slerp across a sweep, which assumes
    the angular rate held constant for 67 ms. The Odin1's IMU reports it 400
    times a second; integrating that (zero-order hold between samples) gives
    the rotation the robot actually went through between any two instants.
    """

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


def build_track(odom) -> PoseTrack | None:
    """A pose track from ``[(t, T), ...]``, or None when there is too little to interpolate."""
    if odom is None or len(odom) < 2:
        return None
    ts = np.array([x[0] for x in odom], dtype=np.float64)
    poses = np.array([x[1] for x in odom], dtype=np.float64)
    return PoseTrack.build(ts, poses)
