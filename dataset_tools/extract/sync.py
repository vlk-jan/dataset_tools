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

    Falls back to the nearest sample outside the range, across gaps over
    ``max_gap``, or across a bracket faster than ``max_speed`` (a teleport).
    Returns the poses and how many were interpolated.
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
    """Mutual nearest-neighbour match of (sorted) camera and lidar stamps, chronological."""
    cam_ts = np.sort(np.asarray(cam_timestamps, dtype=float))
    pcd_ts = np.sort(np.asarray(lidar_timestamps, dtype=float))
    if len(cam_ts) == 0 or len(pcd_ts) == 0:
        return cam_ts[:0], pcd_ts[:0]

    cam_to_pcd = nearest_idx(cam_ts, pcd_ts)
    pcd_to_cam = nearest_idx(pcd_ts, cam_ts)
    mask = pcd_to_cam[cam_to_pcd] == np.arange(len(cam_ts))
    return cam_ts[mask], pcd_ts[cam_to_pcd[mask]]
