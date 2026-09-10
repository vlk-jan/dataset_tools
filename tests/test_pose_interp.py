"""Interpolating the pose stream onto the frame clock.

Nearest-neighbour lookup onto a 10 Hz stream quantises every frame by up to
50 ms; these cover the interpolation that replaced it, and the three cases where
interpolating would invent a pose and the nearest sample is used instead.
"""

import math

import numpy as np
import pytest

from dataset_tools import extract_data as ed


def _pose(x=0.0, y=0.0, yaw=0.0):
    T = ed.euler_matrix(0.0, 0.0, yaw)
    T[:3, 3] = [x, y, 0.0]
    return T


@pytest.mark.parametrize(
    "roll,pitch,yaw",
    [(0.0, 0.0, 0.0), (0.3, -0.2, 1.1), (math.pi, 0.0, 0.0), (0.0, math.pi, 0.0), (-2.9, 0.4, 3.0)],
)
def test_quaternion_from_matrix_round_trips(roll, pitch, yaw):
    """Including the 180-degree cases, where the naive trace formula collapses."""
    R = ed.euler_matrix(roll, pitch, yaw)
    q = ed.quaternion_from_matrix(R)
    assert np.allclose(np.linalg.norm(q), 1.0)
    assert np.allclose(ed.quaternion_matrix(q)[:3, :3], R[:3, :3], atol=1e-12)


def test_slerp_endpoints_and_midpoint():
    q0 = ed.quaternion_from_matrix(ed.euler_matrix(0, 0, 0.0))
    q1 = ed.quaternion_from_matrix(ed.euler_matrix(0, 0, 1.0))
    assert np.allclose(ed.slerp(q0, q1, 0.0), q0)
    assert np.allclose(ed.slerp(q0, q1, 1.0), q1)
    mid = ed.quaternion_matrix(ed.slerp(q0, q1, 0.5))
    assert np.allclose(mid[:3, :3], ed.euler_matrix(0, 0, 0.5)[:3, :3], atol=1e-12)


def test_slerp_takes_the_short_way_round():
    """q and -q are the same rotation; the interpolation must not go the long way."""
    q0 = ed.quaternion_from_matrix(ed.euler_matrix(0, 0, 0.1))
    q1 = -ed.quaternion_from_matrix(ed.euler_matrix(0, 0, 0.2))
    mid = ed.quaternion_matrix(ed.slerp(q0, q1, 0.5))
    assert np.allclose(mid[:3, :3], ed.euler_matrix(0, 0, 0.15)[:3, :3], atol=1e-9)


def test_interpolation_beats_nearest_on_a_ramp():
    ts = np.arange(0.0, 1.001, 0.1)  # 10 Hz, as the Fixposition
    poses = np.array([_pose(x=t, yaw=t) for t in ts])  # 1 m/s, 1 rad/s
    query = np.array([0.05, 0.17, 0.94])  # nothing lands on a sample

    out, n = ed.interpolate_poses(query, ts, poses)

    assert n == 3
    assert np.allclose(out[:, 0, 3], query, atol=1e-12)
    for T, q in zip(out, query, strict=True):  # the assert above already pins the lengths
        assert np.allclose(T[:3, :3], ed.euler_matrix(0, 0, q)[:3, :3], atol=1e-9)
    # what nearest-neighbour would have produced, for the same queries
    nearest = poses[ed.nearest_idx(query, ts)]
    assert np.abs(nearest[:, 0, 3] - query).max() > 0.02


def test_samples_are_reproduced_exactly():
    ts = np.arange(0.0, 0.51, 0.1)
    poses = np.array([_pose(x=t, yaw=0.5 * t) for t in ts])
    out, _ = ed.interpolate_poses(ts, ts, poses)
    assert np.allclose(out, poses, atol=1e-12)


def test_nearest_is_used_across_a_gap():
    """A 2 s hole is not motion to interpolate through."""
    ts = np.array([0.0, 0.1, 2.1, 2.2])
    poses = np.array([_pose(x=0.0), _pose(x=0.1), _pose(x=2.1), _pose(x=2.2)])
    out, n = ed.interpolate_poses(np.array([1.0]), ts, poses, max_gap=0.5)
    assert n == 0
    assert out[0][0, 3] == pytest.approx(0.1)  # the nearer edge of the hole


def test_nearest_is_used_across_a_teleport():
    """A GNSS jump: blending the two sides would fabricate a pose never held."""
    ts = np.array([0.0, 0.1, 0.2])
    poses = np.array([_pose(x=0.0), _pose(x=0.1), _pose(x=900.0)])
    out, n = ed.interpolate_poses(np.array([0.15]), ts, poses, max_speed=4.0)
    assert n == 0
    assert out[0][0, 3] in (pytest.approx(0.1), pytest.approx(900.0))
    assert out[0][0, 3] == pytest.approx(0.1)


def test_queries_outside_the_sample_range_clamp():
    ts = np.array([1.0, 1.1, 1.2])
    poses = np.array([_pose(x=1.0), _pose(x=1.1), _pose(x=1.2)])
    out, n = ed.interpolate_poses(np.array([0.5, 5.0]), ts, poses)
    assert n == 0
    assert out[0][0, 3] == pytest.approx(1.0)
    assert out[1][0, 3] == pytest.approx(1.2)


def test_a_single_sample_is_used_everywhere():
    ts = np.array([3.0])
    poses = np.array([_pose(x=7.0)])
    out, n = ed.interpolate_poses(np.array([0.0, 3.0, 9.0]), ts, poses)
    assert n == 0
    assert np.allclose(out[:, 0, 3], 7.0)
