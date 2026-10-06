import json
import math

import numpy as np
import pytest

from dataset_tools import odom_frame
from dataset_tools.gt_io import anchor_to_first, read_trajectory, write_trajectory
from dataset_tools.transforms import euler_matrix, roll_pitch_from_matrix


def _world(n=30):
    rng = np.random.default_rng(3)
    W = []
    for k in range(n):
        T = euler_matrix(*rng.normal(0, 0.05, 2), 0.1 * k)
        T[:3, 3] = [0.3 * k, 0.05 * k, 0.01 * k]
        W.append(T)
    return np.array(W)


def _scene(tmp_path, W):
    d = tmp_path / "0000"
    d.mkdir()
    write_trajectory(d / "odometry.txt", np.arange(len(W)) * 0.1, anchor_to_first(W))
    np.save(d / "transforms.npy", anchor_to_first(W))
    g = np.array([euler_matrix(*roll_pitch_from_matrix(T), 0.0)[:3, :3] for T in W])
    np.save(d / "gravity_align.npy", g)
    return d


def test_apply_equals_rebuilding_from_the_rotated_world(tmp_path):
    W = _world()
    d = _scene(tmp_path, W)
    C = euler_matrix(math.radians(0.2), math.radians(-0.7), 0.0)[:3, :3]
    odom_frame.apply(d, {"C": C.tolist(), "roll_deg": 0.2, "pitch_deg": -0.7})
    X = np.eye(4)
    X[:3, :3] = C
    WC = W @ X
    _, poses = read_trajectory(d / "odometry.txt")
    np.testing.assert_allclose(poses, anchor_to_first(WC), atol=2e-6)
    np.testing.assert_allclose(np.load(d / "transforms.npy"), anchor_to_first(WC), atol=1e-12)
    g = np.array([euler_matrix(*roll_pitch_from_matrix(T), 0.0)[:3, :3] for T in WC])
    np.testing.assert_allclose(np.load(d / "gravity_align.npy"), g, atol=1e-12)
    rec = (d / odom_frame.RECORD).read_text().splitlines()[-1]
    assert json.loads(rec)["pitch_deg"] == -0.7


def test_apply_refuses_twice_and_through_a_link(tmp_path):
    d = _scene(tmp_path, _world())
    fit = {"C": np.eye(3).tolist()}
    odom_frame.apply(d, fit)
    with pytest.raises(RuntimeError, match="already applied"):
        odom_frame.apply(d, fit)
    e = tmp_path / "0001"
    e.mkdir()
    for f in odom_frame.FILES:
        (e / f).symlink_to(d / f)
    with pytest.raises(RuntimeError, match="symlink"):
        odom_frame.apply(e, fit)


def test_apply_refuses_a_large_rotation(tmp_path):
    d = _scene(tmp_path, _world())
    with pytest.raises(RuntimeError, match="not a calibration residual"):
        odom_frame.apply(d, {"C": euler_matrix(0, math.radians(5), 0)[:3, :3].tolist()})
