"""Camera/lidar pairing and sampling a pose stream onto the frame clock."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from dataset_tools import gnss_guard, transforms


def nearest_idx(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For each value in ``a``, the index of the nearest value in *sorted* ``b``."""
    if len(b) == 1:
        return np.zeros(len(a), dtype=int)
    pos = np.clip(np.searchsorted(b, a), 1, len(b) - 1)
    return np.where(a - b[pos - 1] <= b[pos] - a, pos - 1, pos)


def interpolate_poses(
    query_ts: np.ndarray,
    sample_ts: np.ndarray,
    poses: np.ndarray,
    max_gap: float = 0.5,
    max_speed: float = gnss_guard.MAX_SPEED,
) -> tuple[np.ndarray, int]:
    """Pose at each query time: linear in translation, SLERP in rotation.

    Nearest-neighbour lookup onto a 10 Hz stream quantises every frame by up to
    half a sample period -- 50 ms, which is 5 cm at 1 m/s. Interpolating removes
    that, but only where interpolating means something. The nearest sample is
    used instead when the query is outside the sample range, when the bracketing
    samples are more than ``max_gap`` apart, or when the bracket implies a speed
    above ``max_speed`` -- a GNSS teleport, where blending the two sides would
    fabricate a pose the robot never held. ``max_speed`` matches the gate
    :func:`~dataset_tools.extract.ground_truth.write_ground_truth` breaks segments on, so the two agree about which
    steps are real motion.

    Returns the poses and how many of them were interpolated rather than copied.
    """
    query = np.asarray(query_ts, dtype=float)
    ts = np.asarray(sample_ts, dtype=float)
    poses = np.asarray(poses, dtype=float)
    nearest = poses[nearest_idx(query, ts)]
    if len(ts) < 2:
        return nearest, 0

    pos, quat, _ = transforms.interpolate_poses(ts, poses, query)
    out = transforms.poses_from_quat(pos, quat)
    j = np.clip(np.searchsorted(ts, query), 1, len(ts) - 1)
    i = j - 1
    dt = ts[j] - ts[i]
    step = np.linalg.norm(poses[j, :3, 3] - poses[i, :3, 3], axis=1)
    ok = (ts[i] <= query) & (query <= ts[j]) & (dt > 0.0) & (dt <= max_gap) & (step <= max_speed * dt)
    out[~ok] = nearest[~ok]
    return out, int(ok.sum())


def sync_timestamps(
    cam_timestamps: Sequence[float], lidar_timestamps: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Mutual nearest-neighbour match between camera and lidar timestamps.

    Both inputs are sorted first (bags may contain out-of-order messages). A
    pair is kept only if each stamp is the other's nearest neighbour, which
    drops frames that have no counterpart. Uses ``searchsorted``, so memory
    stays O(n) instead of the O(n*m) full distance matrix.

    Returns the matched (camera, lidar) timestamps, in chronological order.
    """
    cam_ts = np.sort(np.asarray(cam_timestamps, dtype=float))
    pcd_ts = np.sort(np.asarray(lidar_timestamps, dtype=float))
    if len(cam_ts) == 0 or len(pcd_ts) == 0:
        return cam_ts[:0], pcd_ts[:0]

    cam_to_pcd = nearest_idx(cam_ts, pcd_ts)
    pcd_to_cam = nearest_idx(pcd_ts, cam_ts)
    mask = pcd_to_cam[cam_to_pcd] == np.arange(len(cam_ts))
    return cam_ts[mask], pcd_ts[cam_to_pcd[mask]]
