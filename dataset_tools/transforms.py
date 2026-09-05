"""Small rotation and pose helpers ('sxyz' convention), numpy only.

Shared by the extractor and the ROS ``odom_recorder`` node, which is why this
module imports nothing heavier than numpy: the node runs where ``cv2``,
``networkx`` and ``rosbags`` may not be installed.
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


def roll_pitch_from_quat(q: Sequence[float]) -> tuple[float, float]:
    """Roll and pitch ('sxyz') of quaternion [x, y, z, w]."""
    return roll_pitch_from_matrix(quaternion_matrix(q))


def pose_to_matrix(pose) -> np.ndarray:
    """4x4 pose matrix from a ``geometry_msgs/Pose``-shaped object."""
    T = quaternion_matrix([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
    T[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return T
