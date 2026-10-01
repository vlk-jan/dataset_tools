"""Rotation helpers and static-TF graph composition."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from dataset_tools.extract.tf_graph import build_tf_graph, get_transform
from dataset_tools.transforms import euler_matrix, pose_to_matrix, quaternion_matrix, roll_pitch_from_matrix


def quat_from_euler(roll: float, pitch: float, yaw: float) -> list[float]:
    """[x, y, z, w] for the 'sxyz' convention used by euler_matrix()."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


def make_tf(parent: str, child: str, xyz, rpy) -> SimpleNamespace:
    x, y, z, w = quat_from_euler(*rpy)
    return SimpleNamespace(
        header=SimpleNamespace(frame_id=parent),
        child_frame_id=child,
        transform=SimpleNamespace(
            translation=SimpleNamespace(x=xyz[0], y=xyz[1], z=xyz[2]),
            rotation=SimpleNamespace(x=x, y=y, z=z, w=w),
        ),
    )


ANGLES = [
    (0.0, 0.0, 0.0),
    (0.1, -0.2, 0.3),
    (-0.9, 0.4, 2.5),
    (0.0, 1.2, -1.7),
]


@pytest.mark.parametrize("rpy", ANGLES)
def test_quaternion_matrix_matches_euler_matrix(rpy):
    np.testing.assert_allclose(quaternion_matrix(quat_from_euler(*rpy)), euler_matrix(*rpy), atol=1e-12)


@pytest.mark.parametrize("rpy", ANGLES)
def test_roll_pitch_round_trip(rpy):
    roll, pitch, _ = rpy
    got_roll, got_pitch = roll_pitch_from_matrix(quaternion_matrix(quat_from_euler(*rpy)))
    assert got_roll == pytest.approx(roll, abs=1e-9)
    assert got_pitch == pytest.approx(pitch, abs=1e-9)


def test_quaternion_matrix_is_orthonormal_and_degenerate_safe():
    R = quaternion_matrix([0.3, -0.2, 0.5, 0.7])[:3, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)
    np.testing.assert_allclose(quaternion_matrix([0.0, 0.0, 0.0, 0.0]), np.eye(4))


def test_euler_matrix_composition_order_is_zyx():
    roll, pitch, yaw = 0.3, -0.4, 1.1
    expected = euler_matrix(0, 0, yaw) @ euler_matrix(0, pitch, 0) @ euler_matrix(roll, 0, 0)
    np.testing.assert_allclose(euler_matrix(roll, pitch, yaw), expected, atol=1e-12)


def test_pose_to_matrix():
    pose = SimpleNamespace(
        position=SimpleNamespace(x=1.0, y=2.0, z=3.0),
        orientation=SimpleNamespace(**dict(zip("xyzw", quat_from_euler(0.0, 0.0, math.pi / 2), strict=False))),
    )
    T = pose_to_matrix(pose)
    np.testing.assert_allclose(T[:3, 3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(T[:3, :3], euler_matrix(0, 0, math.pi / 2)[:3, :3], atol=1e-12)


def test_tf_graph_edges_hold_point_maps_along_their_direction():
    # TF message: pose of "lidar" in "base_link" (p_base = M @ p_lidar).
    tf = make_tf("base_link", "lidar", (0.1, 0.0, 0.5), (0.0, 0.0, math.pi / 2))
    G = build_tf_graph([tf])

    M = euler_matrix(0.0, 0.0, math.pi / 2)
    M[:3, 3] = [0.1, 0.0, 0.5]
    # child -> parent edge maps lidar points into base_link, parent -> child the inverse
    np.testing.assert_allclose(G["lidar"]["base_link"], M, atol=1e-12)
    np.testing.assert_allclose(G["base_link"]["lidar"], np.linalg.inv(M), atol=1e-12)


def test_get_transform_maps_points_from_src_into_dst():
    # A lidar mounted 1 m above base_link, upside down (roll = pi), as on HELHEST.
    tf = make_tf("base_link", "lidar", (0.2, 0.0, 1.0), (math.pi, 0.0, 0.0))
    G = build_tf_graph([tf])
    T_lidar_to_base = get_transform(G, "lidar", "base_link")

    # The lidar origin lands at its mount position in base_link ...
    np.testing.assert_allclose((T_lidar_to_base @ [0, 0, 0, 1])[:3], [0.2, 0.0, 1.0], atol=1e-12)
    # ... and a ground return seen 1 m "up" in the flipped lidar frame is at z = 0.
    np.testing.assert_allclose((T_lidar_to_base @ [1.0, 0, 1.0, 1])[:3], [1.2, 0.0, 0.0], atol=1e-12)
    # Same convention as tf2's lookup_transform(base_link, lidar).
    np.testing.assert_allclose(get_transform(G, "base_link", "lidar"), np.linalg.inv(T_lidar_to_base), atol=1e-12)


def test_tf_graph_composes_multi_hop_paths():
    # base_link -> mount -> lidar, each with a translation and a rotation.
    tfs = [
        make_tf("base_link", "mount", (0.1, 0.0, 0.5), (0.0, 0.0, math.pi / 2)),
        make_tf("mount", "lidar", (0.2, 0.0, 0.0), (0.0, 0.1, 0.0)),
    ]
    G = build_tf_graph(tfs)

    M_base_mount = euler_matrix(0.0, 0.0, math.pi / 2)
    M_base_mount[:3, 3] = [0.1, 0.0, 0.5]
    M_mount_lidar = euler_matrix(0.0, 0.1, 0.0)
    M_mount_lidar[:3, 3] = [0.2, 0.0, 0.0]

    # Pose of the lidar in base_link is the chained poses; that is the
    # lidar -> base_link point map.
    lidar_to_base = get_transform(G, "lidar", "base_link")
    np.testing.assert_allclose(lidar_to_base, M_base_mount @ M_mount_lidar, atol=1e-12)
    # Reversing the endpoints inverts the composed transform.
    np.testing.assert_allclose(get_transform(G, "base_link", "lidar"), np.linalg.inv(lidar_to_base), atol=1e-12)


def test_tf_graph_path_composition_is_associative():
    tfs = [
        make_tf("a", "b", (0.3, -0.1, 0.0), (0.2, 0.0, 0.4)),
        make_tf("b", "c", (0.0, 0.7, 0.2), (0.0, -0.3, 0.1)),
        make_tf("c", "d", (1.0, 0.0, -0.4), (0.5, 0.5, 0.0)),
    ]
    G = build_tf_graph(tfs)
    direct = get_transform(G, "a", "d")
    # Point maps chain right-to-left: p_d = T_cd @ T_bc @ T_ab @ p_a.
    stepwise = get_transform(G, "c", "d") @ get_transform(G, "b", "c") @ get_transform(G, "a", "b")
    np.testing.assert_allclose(direct, stepwise, atol=1e-12)
    # Still a valid rigid transform after three compositions.
    np.testing.assert_allclose(direct[:3, :3] @ direct[:3, :3].T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(direct[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-12)


def test_tf_graph_takes_the_shortest_path_through_a_diamond():
    tfs = [
        make_tf("root", "left", (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        make_tf("left", "leaf", (0.0, 1.0, 0.0), (0.0, 0.0, 0.0)),
        make_tf("root", "leaf", (2.0, 2.0, 0.0), (0.0, 0.0, 0.0)),
    ]
    G = build_tf_graph(tfs)
    # The direct root->leaf edge is one hop, so it wins over root->left->leaf:
    # the leaf origin sits at (2, 2, 0) in root, not (1, 1, 0).
    np.testing.assert_allclose(get_transform(G, "leaf", "root")[:3, 3], [2.0, 2.0, 0.0])


def test_tf_graph_missing_paths_and_identity():
    G = build_tf_graph([make_tf("a", "b", (1.0, 0.0, 0.0), (0.0, 0.0, 0.0))])
    G.update(build_tf_graph([make_tf("c", "d", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))]))
    np.testing.assert_allclose(get_transform(G, "a", "a"), np.eye(4))
    assert get_transform(G, "a", "missing") is None
    assert get_transform(G, "missing", "a") is None
    assert get_transform(G, "a", "d") is None  # disconnected component
