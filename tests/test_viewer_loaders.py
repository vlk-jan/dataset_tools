"""Viewer: geometry, file readers (dataset_tools.viewer.loaders)."""

from __future__ import annotations

import numpy as np
import pytest

from dataset_tools.transforms import interpolate_poses
from dataset_tools.viewer import loaders


def test_interpolate_poses_marks_queries_outside_the_track_invalid():
    ts = np.array([10.0, 11.0, 12.0])
    poses = np.stack([np.eye(4)] * 3)
    for i, x in enumerate([0.0, 1.0, 2.0]):
        poses[i, 0, 3] = x
    pos, _quat, valid = interpolate_poses(ts, poses, np.array([9.0, 10.5, 12.0, 13.0]))
    assert valid.tolist() == [False, True, True, False]
    assert pos[1, 0] == pytest.approx(0.5)
    # Outside queries are clamped, not extrapolated.
    assert pos[0, 0] == pytest.approx(0.0)
    assert pos[3, 0] == pytest.approx(2.0)


def test_interpolate_poses_counts_a_query_within_tolerance_as_on_the_track():
    ts = np.array([10.0, 11.0, 12.0])
    poses = np.stack([np.eye(4)] * 3)
    just_early = np.array([9.99])
    assert not interpolate_poses(ts, poses, just_early)[2][0]
    assert interpolate_poses(ts, poses, just_early, tol=0.05)[2][0]


def test_resample_series_does_not_invent_values_across_a_long_gap():
    src_ts = np.array([0.0, 1.0, 50.0, 51.0])
    vals = np.array([[0.0], [1.0], [2.0], [3.0]])
    out = loaders.resample_series(src_ts, vals, np.array([0.5, 25.0]))
    assert out[0, 0] == pytest.approx(0.5)
    assert np.isnan(out[1, 0])


def test_resample_series_of_an_empty_source_is_all_nan():
    """A zero-row joint_states.npy used to raise in np.interp and 500 every window."""
    out = loaders.resample_series(np.zeros(0), np.zeros((0, 3)), np.arange(5) * 0.1)
    assert out.shape == (5, 3) and np.isnan(out).all()


def test_odometry_reads_as_timestamps_and_poses(tmp_path):
    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, 3.0]
    path = tmp_path / "odometry.txt"
    path.write_text("# header\n" + " ".join(["100.0", *[str(v) for v in T.reshape(-1)]]) + "\n")
    result = loaders.read_odometry(path)
    assert result is not None
    ts, poses = result
    assert ts[0] == pytest.approx(100.0)
    assert np.allclose(poses[0], T)


def test_segments_parse_and_sort(tmp_path):
    path = tmp_path / "segments.txt"
    path.write_text("# id start_ts end_ts n_poses reason\n1 20.0 30.0 100 teleport\n0 0.0 10.0 50 gap\ngarbage line\n")
    segs = loaders.read_segments(path)
    assert [s.index for s in segs] == [0, 1]
    assert segs[0].reason == "gap"
    assert segs[1].duration == pytest.approx(10.0)


def test_missing_files_return_none_rather_than_raising(tmp_path):
    assert loaders.read_odometry(tmp_path / "nope.txt") is None
    assert loaders.read_segments(tmp_path / "nope.txt") == []
    assert loaders.read_timestamps(tmp_path / "nope.txt") is None
    assert loaders.load_npy(tmp_path / "nope.npy") is None
    assert loaders.read_matrix(tmp_path / "nope.txt", (3, 3)) is None


def test_frame_indices_take_ascii_digits_only(tmp_path):
    for name in ("00001.npy", "⑤.npy"):  # "⑤".isdigit() is true; int() refuses it
        (tmp_path / name).write_bytes(b"")
    assert loaders.frame_indices_dir(tmp_path, ".npy") == [1]
