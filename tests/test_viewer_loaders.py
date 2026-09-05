"""Viewer: geometry, file readers (dataset_tools.viewer.loaders).

Every test builds a synthetic dataset root (``helpers_viewer``), so nothing
here touches a real dataset or the network. The end-to-end HTTP tests skip
cleanly when the viewer extra (SimView + torch) is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from dataset_tools.viewer import loaders

# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def test_quaternion_round_trips_through_rotation_matrices():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(64, 3, 3))
    u, _, vt = np.linalg.svd(a)
    R = u @ vt
    R[np.linalg.det(R) < 0] *= -1
    q = loaders.mat_to_quat(R)
    back = loaders.poses_from_quat(np.zeros((64, 3)), q)[:, :3, :3]
    assert np.allclose(R, back, atol=1e-9)
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0)


def test_quaternion_of_identity_is_scalar_one():
    assert np.allclose(loaders.mat_to_quat(np.eye(3)[None]), [[1, 0, 0, 0]])


def test_slerp_midpoint_is_the_half_angle():
    q0 = np.array([[1.0, 0, 0, 0]])
    angle = np.pi / 2
    q1 = np.array([[np.cos(angle / 2), 0, 0, np.sin(angle / 2)]])
    mid = loaders.quat_slerp(q0, q1, np.array([0.5]))
    expected = np.array([np.cos(angle / 4), 0, 0, np.sin(angle / 4)])
    assert np.allclose(mid[0], expected, atol=1e-12)


def test_interpolate_poses_marks_queries_outside_the_track_invalid():
    ts = np.array([10.0, 11.0, 12.0])
    poses = np.stack([np.eye(4)] * 3)
    for i, x in enumerate([0.0, 1.0, 2.0]):
        poses[i, 0, 3] = x
    pos, _quat, valid = loaders.interpolate_poses(ts, poses, np.array([9.0, 10.5, 12.0, 13.0]))
    assert valid.tolist() == [False, True, True, False]
    assert pos[1, 0] == pytest.approx(0.5)
    # Outside queries are clamped, not extrapolated.
    assert pos[0, 0] == pytest.approx(0.0)
    assert pos[3, 0] == pytest.approx(2.0)


def test_interpolate_poses_counts_a_query_within_tolerance_as_on_the_track():
    ts = np.array([10.0, 11.0, 12.0])
    poses = np.stack([np.eye(4)] * 3)
    just_early = np.array([9.99])
    assert not loaders.interpolate_poses(ts, poses, just_early)[2][0]
    assert loaders.interpolate_poses(ts, poses, just_early, tol=0.05)[2][0]


def test_resample_series_does_not_invent_values_across_a_long_gap():
    src_ts = np.array([0.0, 1.0, 50.0, 51.0])
    vals = np.array([[0.0], [1.0], [2.0], [3.0]])
    out = loaders.resample_series(src_ts, vals, np.array([0.5, 25.0]), max_gap=1.0)
    assert out[0, 0] == pytest.approx(0.5)
    assert np.isnan(out[1, 0])


def test_roll_pitch_of_an_identity_gravity_rotation_is_zero():
    roll, pitch = loaders.roll_pitch_from_gravity_align(np.eye(3)[None])
    assert roll[0] == pytest.approx(0.0)
    assert pitch[0] == pytest.approx(0.0)


def test_odometry_parses_both_the_17_and_13_column_forms(tmp_path):
    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, 3.0]
    long_form = tmp_path / "long.txt"
    long_form.write_text("# header\n" + " ".join(["100.0", *[str(v) for v in T.reshape(-1)]]) + "\n")
    short = tmp_path / "short.txt"
    short.write_text("# header\n" + " ".join(["100.0", *[str(v) for v in T[:3, :].reshape(-1)]]) + "\n")
    for path in (long_form, short):
        result = loaders.read_odometry(path)
        assert result is not None
        ts, poses = result
        assert ts[0] == pytest.approx(100.0)
        assert np.allclose(poses[0], T)


def test_segments_parse_and_sort(tmp_path):
    path = tmp_path / "segments.txt"
    path.write_text(
        "# id start_ts end_ts n_poses reason\n"
        "1 20.0 30.0 100 filtered:pose_jump\n"
        "0 0.0 10.0 50 v11_fill\n"
        "garbage line\n"
    )
    segs = loaders.read_segments(path)
    assert [s.index for s in segs] == [0, 1]
    assert segs[0].reason == "v11_fill"
    assert segs[1].duration == pytest.approx(10.0)


def test_missing_files_return_none_rather_than_raising(tmp_path):
    assert loaders.read_odometry(tmp_path / "nope.txt") is None
    assert loaders.read_segments(tmp_path / "nope.txt") == []
    assert loaders.read_timestamps(tmp_path / "nope.txt") is None
    assert loaders.load_npy(tmp_path / "nope.npy") is None
    assert loaders.read_matrix(tmp_path / "nope.txt") is None
