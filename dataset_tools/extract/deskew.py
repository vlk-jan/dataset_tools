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
rotation slerp). The points must already be in the frame the poses describe
(``base_link``), so this runs after the lidar mount transform. Offsets are
quantised to :data:`BIN_S` and the pose looked up once per bin: the Odin1
itself ticks in 2.2 ms slots, so 32 lookups serve 49 152 points.

A sweep whose span is below one bin is already an instant and is left alone,
as is one the pose stream does not bracket closely enough (a hole the GNSS
guard opened, or the stream's ends): guessing a motion there would smear a
cloud that is merely raw.
"""

from __future__ import annotations

import numpy as np

from dataset_tools.time_align import PoseTrack

#: Offsets are quantised to this before the poses are looked up.
BIN_S = 0.001
#: Pose samples bracketing a sweep must be at most this far apart for its
#: motion to be interpolated; the Fixposition publishes at 10 Hz.
MAX_GAP_S = 0.5

#: What :func:`deskew` can report back, in the order a summary lists them.
STATUSES = ("deskewed", "instant", "no_pose", "empty")


def deskew(
    points: np.ndarray, offsets: np.ndarray, stamp: float, t_ref: float, track: PoseTrack
) -> tuple[np.ndarray, str]:
    """``points`` (N, 3) taken at ``stamp + offsets``, re-expressed at ``t_ref``.

    ``stamp`` is the header stamp the offsets count from; ``t_ref`` the frame's
    instant on the dataset's clock (on the Odin1 the two differ by under a
    millisecond, the camera's half of the pairing). Returns the moved points
    and a status from :data:`STATUSES`; the points come back untouched for
    every status but ``deskewed``. A sweep is "instant" when its offsets span
    less than one bin *and* the stamp is the reference: then there is no
    motion to undo.
    """
    pts = np.asarray(points)
    off = np.asarray(offsets, dtype=np.float64)
    if len(pts) == 0:
        return pts, "empty"
    lo, hi = float(off.min()), float(off.max())
    if hi - lo < BIN_S and abs(stamp - t_ref) < BIN_S:
        return pts, "instant"
    if not covered(track.ts, min(t_ref, stamp + lo), max(t_ref, stamp + hi)):
        return pts, "no_pose"

    bins = np.round((stamp - t_ref + off) / BIN_S).astype(np.int64)
    uniq, inv = np.unique(bins, return_inverse=True)
    pos_u, rot_u = track.at(t_ref + uniq * BIN_S)
    pos_r, rot_r = track.at(np.array([t_ref]))
    R_ref_T = rot_r[0].T
    R_rel = np.einsum("ij,njk->nik", R_ref_T, rot_u)  # (U, 3, 3): base at t_p -> base at t_ref
    t_rel = (pos_u - pos_r[0]) @ R_ref_T.T  # (U, 3)
    out = np.einsum("nij,nj->ni", R_rel[inv], pts.astype(np.float64)) + t_rel[inv]
    return out.astype(pts.dtype), "deskewed"


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
