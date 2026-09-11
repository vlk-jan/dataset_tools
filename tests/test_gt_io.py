"""The ground-truth file format, read and written by one module."""

from __future__ import annotations

import numpy as np
import pytest

from dataset_tools import gt_io


def _poses(n: int) -> np.ndarray:
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, 0, 3] = np.arange(n) * 0.5
    T[:, 1, 3] = 1.0
    return T


def test_trajectory_round_trips_and_sorts(tmp_path):
    ts = np.array([3.0, 1.0, 2.0])
    poses = _poses(3)
    gt_io.write_trajectory(tmp_path / "odometry.txt", ts, poses)
    text = (tmp_path / "odometry.txt").read_text()
    assert text.startswith(gt_io.GT_HEADER + "\n")
    assert len(text.splitlines()) == 4

    got_ts, got_poses = gt_io.read_trajectory(tmp_path / "odometry.txt")
    assert got_ts.tolist() == [1.0, 2.0, 3.0]
    assert np.allclose(got_poses[:, 0, 3], [0.5, 1.0, 0.0])


def test_trajectory_accepts_the_13_column_form_and_rejects_others(tmp_path):
    row = np.concatenate([[5.0], _poses(1)[0, :3].ravel()])
    np.savetxt(tmp_path / "short.txt", row[None])
    ts, poses = gt_io.read_trajectory(tmp_path / "short.txt")
    assert ts.tolist() == [5.0] and poses.shape == (1, 4, 4) and poses[0, 3, 3] == 1.0

    np.savetxt(tmp_path / "odd.txt", np.ones((2, 9)))
    with pytest.raises(ValueError):
        gt_io.read_trajectory(tmp_path / "odd.txt")


def test_an_empty_trajectory_is_empty_arrays_not_an_error(tmp_path):
    (tmp_path / "odometry.txt").write_text(gt_io.GT_HEADER + "\n")
    ts, poses = gt_io.read_trajectory(tmp_path / "odometry.txt")
    assert ts.shape == (0,) and poses.shape == (0, 4, 4)


def test_segments_round_trip_and_fold_trailing_columns_into_the_reason(tmp_path):
    rows = [gt_io.SegmentRow(0, 1.0, 2.5, 16, "filtered:pose_jump"), gt_io.SegmentRow(1, 3.0, 4.0, 11, "end_of_run")]
    gt_io.write_segments(tmp_path / "seg.txt", rows)
    text = (tmp_path / "seg.txt").read_text()
    assert text.startswith(gt_io.SEGMENTS_HEADER + "\n")
    # Everything but the number survives the round trip; the number is assigned
    # by the writer (see test_segments_are_numbered_from_one).
    got = gt_io.read_segments(tmp_path / "seg.txt")
    assert [(r.start_ts, r.end_ts, r.n_poses, r.reason) for r in got] == [
        (r.start_ts, r.end_ts, r.n_poses, r.reason) for r in rows
    ]

    (tmp_path / "odd.txt").write_text("# header\n0 1.0 2.0 5 two words\n1 2.0 3.0 notanumber\nbad line\n2 4.0\n")
    got = gt_io.read_segments(tmp_path / "odd.txt")
    assert [(r.index, r.n_poses, r.reason) for r in got] == [(0, 5, "two words"), (1, 0, "notanumber")]


def test_segments_are_numbered_from_one(tmp_path):
    """ "Segment 3 of 4" should be the third, not the last."""
    rows = [gt_io.SegmentRow(9, 1.0, 2.0, 5, "gap"), gt_io.SegmentRow(9, 3.0, 4.0, 5, "end_of_run")]
    gt_io.write_segments(tmp_path / "seg.txt", rows)
    assert [r.index for r in gt_io.read_segments(tmp_path / "seg.txt")] == [1, 2]


def test_a_table_written_before_the_renumbering_still_reads_as_written(tmp_path):
    """Reading stays faithful: HELHEST v1.2 through v1.5 are numbered from zero
    and a segment is addressed by the number in its own file."""
    (tmp_path / "seg.txt").write_text(gt_io.SEGMENTS_HEADER + "\n0 1.0 2.0 5 gap\n1 3.0 4.0 5 end_of_run\n")
    assert [r.index for r in gt_io.read_segments(tmp_path / "seg.txt")] == [0, 1]


def test_anchor_to_first_puts_the_identity_first():
    anchored = gt_io.anchor_to_first(_poses(4))
    assert np.allclose(anchored[0], np.eye(4))
    assert np.allclose(anchored[:, 0, 3], [0.0, 0.5, 1.0, 1.5])
    assert np.allclose(anchored[:, 1, 3], 0.0)
