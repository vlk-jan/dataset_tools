"""The post-hoc /odom filter's segment tags.

The tag on a segment says why *that* segment ended. The filter splits one input
segment into several runs, so only the run that reaches the input segment's end
may inherit its tag; the rest end where the filter cut them.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from dataset_tools.gt_gaps import TAG_GAP, cut_segments
from dataset_tools.gt_io import SegmentRow

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("filter_segments", REPO_ROOT / "scripts" / "filter_segments.py")
filter_segments = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(filter_segments)

HEADER = "# Timestamp(sec) r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz 0 0 0 1"


def _write_traj(path: Path, ts, xs) -> None:
    with open(path, "w") as f:
        f.write(HEADER + "\n")
        for t, x in zip(ts, xs, strict=False):
            T = np.eye(4)
            T[0, 3] = x
            f.write(" ".join(f"{v:.6f}" for v in (t, *T.ravel())) + "\n")


def _segments(path: Path):
    return [line.split() for line in path.read_text().splitlines() if not line.startswith("#")]


def test_only_the_run_reaching_the_end_inherits_the_tag(tmp_path):
    """One input segment, cut in the middle -> two runs, two different reasons."""
    ts = np.arange(0, 20, 0.1)
    ref = ts * 1.0
    poses = ref.copy()
    poses[80:100] += 5.0  # a 2 s stretch that jumps away from live odometry

    regen, out = tmp_path / "regen", tmp_path / "out"
    regen.mkdir()
    _write_traj(regen / "odometry.txt", ts, poses)
    _write_traj(tmp_path / "odom_ref.txt", ts, ref)
    with open(regen / "odometry_segments.txt", "w") as f:
        f.write("# segment start_ts end_ts n_poses reason\n")
        f.write(f"0 {ts[0]:.6f} {ts[-1]:.6f} {len(ts)} end_of_run\n")

    sys.argv = ["filter_segments.py", str(regen), str(tmp_path / "odom_ref.txt"), str(out)]
    filter_segments.main()

    segs = _segments(out / "odometry_segments.txt")
    assert len(segs) == 2
    # The first run stops where the filter dropped poses, not where the
    # recording stopped -- it ends well before the input segment's end.
    assert segs[0][4] == "filtered:odom_mismatch"
    assert float(segs[0][2]) < ts[-1] - 1.0
    # The second reaches the input segment's end, so it keeps its tag.
    assert segs[1][4] == "filtered:end_of_run"
    assert float(segs[1][2]) == float(f"{ts[-1]:.6f}")


def test_an_uncut_segment_keeps_its_tag(tmp_path):
    """Nothing dropped -> one run, and it is the one that ended for the reason."""
    ts = np.arange(0, 20, 0.1)
    xs = ts * 1.0

    regen, out = tmp_path / "regen", tmp_path / "out"
    regen.mkdir()
    _write_traj(regen / "odometry.txt", ts, xs)
    _write_traj(tmp_path / "odom_ref.txt", ts, xs)
    with open(regen / "odometry_segments.txt", "w") as f:
        f.write("# segment start_ts end_ts n_poses reason\n")
        f.write(f"0 {ts[0]:.6f} {ts[-1]:.6f} {len(ts)} pose_jump\n")

    sys.argv = ["filter_segments.py", str(regen), str(tmp_path / "odom_ref.txt"), str(out)]
    filter_segments.main()

    segs = _segments(out / "odometry_segments.txt")
    assert len(segs) == 1
    assert segs[0][4] == "filtered:pose_jump"


def test_a_hole_the_wheels_drove_through_cuts_the_run(tmp_path):
    """No window closes inside a 10 s hole, so the window test is blind to it;
    the wheel-distance rule cuts there because /odom travelled 10 m while the
    GT pose moved 1 m."""
    ts = np.concatenate([np.arange(0, 10, 0.1), np.arange(20, 30, 0.1)])
    ref_ts = np.arange(0, 30, 0.1)
    ref = ref_ts * 1.0
    poses = np.where(ts < 15, ts, ts - 9.0)  # 10 s hole, the GT pose advanced only 1 m across it

    regen, out = tmp_path / "regen", tmp_path / "out"
    regen.mkdir()
    _write_traj(regen / "odometry.txt", ts, poses)
    _write_traj(tmp_path / "odom_ref.txt", ref_ts, ref)
    with open(regen / "odometry_segments.txt", "w") as f:
        f.write("# segment start_ts end_ts n_poses reason\n")
        f.write(f"0 {ts[0]:.6f} {ts[-1]:.6f} {len(ts)} end_of_run\n")

    sys.argv = ["filter_segments.py", str(regen), str(tmp_path / "odom_ref.txt"), str(out), "--max-gap", "100"]
    filter_segments.main()

    segs = _segments(out / "odometry_segments.txt")
    assert [s[4] for s in segs] == ["filtered:wheel_gap", "filtered:end_of_run"]
    assert float(segs[0][2]) == float(f"{ts[99]:.6f}")
    assert float(segs[1][1]) == float(f"{ts[100]:.6f}")
    stats = dict(line.split("=") for line in (regen / "filter_stats.txt").read_text().split())
    assert stats["seams_wheel_gap"] == "1" and stats["seams_gap"] == "0"


def test_a_long_gap_cuts_even_when_the_endpoints_agree(tmp_path):
    """The GT pose and /odom agree across a 3 s hole; nothing in between is known."""
    ts = np.concatenate([np.arange(0, 10, 0.1), np.arange(13, 23, 0.1)])
    xs = ts * 1.0

    regen, out = tmp_path / "regen", tmp_path / "out"
    regen.mkdir()
    _write_traj(regen / "odometry.txt", ts, xs)
    _write_traj(tmp_path / "odom_ref.txt", np.arange(0, 23, 0.1), np.arange(0, 23, 0.1) * 1.0)
    with open(regen / "odometry_segments.txt", "w") as f:
        f.write("# segment start_ts end_ts n_poses reason\n")
        f.write(f"0 {ts[0]:.6f} {ts[-1]:.6f} {len(ts)} pose_jump\n")

    sys.argv = ["filter_segments.py", str(regen), str(tmp_path / "odom_ref.txt"), str(out)]
    filter_segments.main()

    segs = _segments(out / "odometry_segments.txt")
    assert [s[4] for s in segs] == ["filtered:gap", "filtered:pose_jump"]


def test_a_dense_stream_is_never_cut_by_the_gap_rules(tmp_path):
    """0.1 s steps cannot accumulate 1 m of wheel disagreement: one run, untouched."""
    ts = np.arange(0, 20, 0.1)
    xs = ts * 2.0  # 2 m/s

    regen, out = tmp_path / "regen", tmp_path / "out"
    regen.mkdir()
    _write_traj(regen / "odometry.txt", ts, xs)
    _write_traj(tmp_path / "odom_ref.txt", ts, xs * 1.1)  # 10 % wheel slip
    with open(regen / "odometry_segments.txt", "w") as f:
        f.write("# segment start_ts end_ts n_poses reason\n")
        f.write(f"0 {ts[0]:.6f} {ts[-1]:.6f} {len(ts)} end_of_run\n")

    sys.argv = ["filter_segments.py", str(regen), str(tmp_path / "odom_ref.txt"), str(out)]
    filter_segments.main()

    assert [s[4] for s in _segments(out / "odometry_segments.txt")] == ["filtered:end_of_run"]


def test_poses_no_segment_covers_are_counted(tmp_path):
    """poses_in == poses_out + dropped_short + dropped_outside, so nothing vanishes silently."""
    ts = np.arange(0, 20, 0.1)
    xs = ts * 1.0

    regen, out = tmp_path / "regen", tmp_path / "out"
    regen.mkdir()
    _write_traj(regen / "odometry.txt", ts, xs)
    _write_traj(tmp_path / "odom_ref.txt", ts, xs)
    with open(regen / "odometry_segments.txt", "w") as f:
        f.write("# segment start_ts end_ts n_poses reason\n")
        f.write(f"0 {ts[50]:.6f} {ts[-1]:.6f} {len(ts) - 50} end_of_run\n")  # the first 5 s belong to no segment

    sys.argv = ["filter_segments.py", str(regen), str(tmp_path / "odom_ref.txt"), str(out)]
    filter_segments.main()

    stats = dict(line.split("=") for line in (regen / "filter_stats.txt").read_text().split())
    assert stats["dropped_outside"] == "50"
    assert int(stats["poses_in"]) == int(stats["poses_out"]) + int(stats["dropped_short"]) + int(
        stats["dropped_outside"]
    )


def test_split_segments_at_gaps_never_writes_through_a_linked_regen(tmp_path):
    """A derived root links its parent's regen/: the backup and the digest must not land there."""
    _spec2 = importlib.util.spec_from_file_location(
        "split_segments_at_gaps", REPO_ROOT / "scripts" / "split_segments_at_gaps.py"
    )
    split_at_gaps = importlib.util.module_from_spec(_spec2)
    _spec2.loader.exec_module(split_at_gaps)

    ts = np.concatenate([np.arange(0, 10, 0.1), np.arange(13, 23, 0.1)])  # a 3 s hole
    parent_regen, scene = tmp_path / "parent" / "regen", tmp_path / "derived" / "0000"
    parent_regen.mkdir(parents=True)
    scene.mkdir(parents=True)
    (scene / "regen").symlink_to(parent_regen)
    _write_traj(scene / "odometry.txt", ts, ts * 1.0)
    with open(scene / "odometry_segments.txt", "w") as f:
        f.write("# segment start_ts end_ts n_poses reason\n")
        f.write(f"1 {ts[0]:.6f} {ts[-1]:.6f} {len(ts)} end_of_run\n")

    stats = split_at_gaps.split_scene(scene, max_gap=1.0, max_err=1.0, min_poses=20, check=False)

    assert stats["segments_out"] == 2 and len(_segments(scene / "odometry_segments.txt")) == 2
    assert list(parent_regen.iterdir()) == []


def test_a_stamp_two_segments_share_is_not_taken_twice():
    """v1 odometry.txt is not strictly monotonic: a segment may start on the stamp the
    previous one ended on. An inclusive time mask would give both poses to both segments."""
    ts = np.r_[np.arange(0, 10.01, 0.1), np.arange(10, 20.01, 0.1)]  # 101 + 101, two poses at t = 10
    T = np.tile(np.eye(4), (len(ts), 1, 1))
    T[:, 0, 3] = np.r_[np.arange(101) * 0.1, 50 + np.arange(101) * 0.1]  # a jump between the two
    segs = [SegmentRow(0, 0.0, 10.0, 101, "teleport"), SegmentRow(1, 10.0, 20.0, 101, "end_of_run")]

    def no_cuts(t, _):
        return np.zeros(len(t) - 1, bool), [TAG_GAP] * (len(t) - 1), None

    out_t, out_T, rows, st = cut_segments(ts, T, segs, no_cuts, lambda reason, cut: reason, 20)

    assert np.array_equal(out_t, ts) and st["poses_out"] == len(ts) and st["dropped_outside"] == 0
    assert [r.n_poses for r in rows] == [101, 101]
    assert np.allclose(out_T[101], np.eye(4))  # the second segment starts on its own pose, re-anchored
