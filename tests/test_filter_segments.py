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


# --- the one-off relabel of roots the old filter already wrote ----------------

_rspec = importlib.util.spec_from_file_location("relabel_segments", REPO_ROOT / "scripts" / "relabel_segments.py")
relabel_segments = importlib.util.module_from_spec(_rspec)
_rspec.loader.exec_module(relabel_segments)


def _write_table(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# segment start_ts end_ts n_poses reason\n")
        for i, (s0, s1, n, reason) in enumerate(rows):
            f.write(f"{i} {s0:.6f} {s1:.6f} {n} {reason}\n")


def _relabel_root(tmp_path: Path) -> Path:
    """A root whose scene 0000 shows every case the relabel has to tell apart."""
    scene = tmp_path / "0000"
    _write_table(
        scene / "regen" / "odometry_segments_prefilter.txt",
        [(10.0, 20.0, 100, "pose_jump"), (30.0, 60.0, 300, "end_of_run")],
    )
    _write_table(
        scene / "odometry_segments.txt",
        [
            (10.0, 20.0, 100, "filtered:pose_jump"),  # reaches its parent's end
            (30.0, 40.0, 100, "filtered:end_of_run"),  # cut by the filter
            (42.0, 44.0, 20, "v11_fill"),  # merged from another root, inside a parent
            (45.0, 60.0, 150, "filtered:end_of_run"),  # reaches its parent's end
            (70.0, 80.0, 100, "v11_fill"),  # merged from another root, no parent
        ],
    )
    return tmp_path


def test_relabel_retags_only_the_runs_the_filter_cut(tmp_path):
    root = _relabel_root(tmp_path)
    sys.argv = ["relabel_segments.py", str(root), "--apply"]
    relabel_segments.main()

    segs = _segments(root / "0000" / "odometry_segments.txt")
    assert [s[4] for s in segs] == [
        "filtered:pose_jump",
        "filtered:odom_mismatch",
        "v11_fill",
        "filtered:end_of_run",
        "v11_fill",
    ]
    # Nothing but the reason column moved.
    assert [(s[1], s[2], s[3]) for s in segs] == [
        ("10.000000", "20.000000", "100"),
        ("30.000000", "40.000000", "100"),
        ("42.000000", "44.000000", "20"),
        ("45.000000", "60.000000", "150"),
        ("70.000000", "80.000000", "100"),
    ]


def test_relabel_is_idempotent(tmp_path):
    root = _relabel_root(tmp_path)
    sys.argv = ["relabel_segments.py", str(root), "--apply"]
    relabel_segments.main()
    first = (root / "0000" / "odometry_segments.txt").read_text()
    relabel_segments.main()
    assert (root / "0000" / "odometry_segments.txt").read_text() == first
    assert relabel_segments.plan_scene(root / "0000") == []


def test_relabel_without_apply_changes_nothing(tmp_path):
    root = _relabel_root(tmp_path)
    before = (root / "0000" / "odometry_segments.txt").read_text()
    sys.argv = ["relabel_segments.py", str(root)]
    relabel_segments.main()
    assert (root / "0000" / "odometry_segments.txt").read_text() == before


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
