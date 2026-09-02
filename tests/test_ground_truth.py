import numpy as np

from dataset_tools.extract_data import write_ground_truth


def _traj(ts, xs):
    out = []
    for t, x in zip(ts, xs):
        T = np.eye(4)
        T[0, 3] = x
        out.append((t, T))
    return out


def test_continuous_run_is_one_anchored_segment(tmp_path):
    ts = np.arange(0, 10, 0.1)
    assert write_ground_truth(_traj(ts, ts * 1.0), tmp_path, min_poses=20)
    rows = np.loadtxt(tmp_path / "odometry.txt", comments="#")
    segs = [l.split() for l in open(tmp_path / "odometry_segments.txt") if not l.startswith("#")]
    assert len(segs) == 1 and int(segs[0][3]) == len(ts)
    T0 = rows[0, 1:].reshape(4, 4)
    assert np.allclose(T0, np.eye(4))  # anchored


def test_teleport_splits_and_spike_dropped(tmp_path):
    ts = np.arange(0, 20, 0.1)
    xs = ts * 1.0
    xs[100] += 750.0  # one-sample out-and-back GNSS spike
    assert write_ground_truth(_traj(ts, xs), tmp_path, min_poses=20)
    segs = [l.split() for l in open(tmp_path / "odometry_segments.txt") if not l.startswith("#")]
    assert len(segs) == 2  # spike sample's 1-pose segment discarded
    assert segs[1][4] == "teleport"
    n = sum(int(s[3]) for s in segs)
    assert n == len(ts) - 1  # only the spike sample lost
    rows = np.loadtxt(tmp_path / "odometry.txt", comments="#")
    # each segment starts at identity
    starts = {float(s[1]) for s in segs}
    for r in rows:
        if r[0] in starts:
            assert np.allclose(r[1:].reshape(4, 4), np.eye(4))


def test_gap_reason_and_min_poses(tmp_path):
    ts = np.concatenate([np.arange(0, 5, 0.1), np.arange(30, 30.5, 0.1), np.arange(60, 70, 0.1)])
    assert write_ground_truth(_traj(ts, np.zeros(len(ts))), tmp_path, min_poses=20)
    segs = [l.split() for l in open(tmp_path / "odometry_segments.txt") if not l.startswith("#")]
    assert len(segs) == 2  # the 5-sample middle chunk is discarded
    assert segs[1][4] == "gap"


def test_too_few_samples(tmp_path):
    assert not write_ground_truth(_traj([0, 0.1], [0, 0.1]), tmp_path)
