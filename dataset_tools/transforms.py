"""Small rotation and pose helpers ('sxyz' convention), numpy only.

Shared by the extractor, the viewer and the ROS ``odom_recorder`` node, which
is why this module imports nothing heavier than numpy: the node runs where
``cv2`` and ``rosbags`` may not be installed, and the extractor where the
viewer's extra is not.

Single quaternions are ``[x, y, z, w]``, as ROS messages carry them. The
batched helpers at the bottom are ``[w, x, y, z]`` (scalar-first), matching
SimView, which is where the viewer hands them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def quaternion_matrix(q: Sequence[float]) -> np.ndarray:
    """4x4 homogeneous rotation matrix from quaternion [x, y, z, w]."""
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(4)
    s = 2.0 / n
    T = np.eye(4)
    T[:3, :3] = [
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ]
    return T


def euler_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """4x4 matrix R = Rz(yaw) @ Ry(pitch) @ Rx(roll) ('sxyz')."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    return T


def roll_pitch_from_matrix(R: np.ndarray) -> tuple[float, float]:
    """Roll and pitch ('sxyz') of a rotation matrix (3x3 or 4x4)."""
    roll = math.atan2(R[2, 1], R[2, 2])
    pitch = math.asin(float(np.clip(-R[2, 0], -1.0, 1.0)))
    return roll, pitch


def pose_to_matrix(pose) -> np.ndarray:
    """4x4 pose matrix from a ``geometry_msgs/Pose``-shaped object."""
    T = quaternion_matrix([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
    T[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return T


def header_stamp(msg, fallback: float) -> float:
    """``header.stamp`` of a deserialised message in seconds, or ``fallback``."""
    if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    return fallback


# ---------------------------------------------------------------------------
# Batched, [w, x, y, z]
# ---------------------------------------------------------------------------
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
