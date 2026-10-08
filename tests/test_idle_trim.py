import numpy as np
import pytest

from dataset_tools import idle_trim
from dataset_tools.transforms import euler_matrix


def _odom(segments, hz=14.0):
    """[(t, T)] from [(duration_s, speed_mps, yaw_rate_dps)] driven in turn."""
    out, t, x, yaw = [], 0.0, 0.0, 0.0
    for dur, v, w in segments:
        for _ in range(int(dur * hz)):
            T = euler_matrix(0, 0, np.radians(yaw))
            T[:3, 3] = [x, 0, 0]
            out.append((t, T))
            t += 1 / hz
            x += v / hz
            yaw += w / hz
    return out


def test_a_long_stop_keeps_its_first_seconds_and_a_short_one_stays():
    odom = _odom([(20, 1.0, 0), (100, 0, 0), (20, 1.0, 0), (30, 0, 0), (10, 1.0, 0)])
    ts = np.array([t for t, _ in odom])
    keep, v = idle_trim.trim(ts, odom, 60, 10)
    assert len(v.runs) == 1
    a, b, n = v.runs[0]
    assert 19.5 < a < 21 and 119 < b < 121
    dropped = ts[~keep]
    assert dropped.min() > a + 10 - 0.1 and dropped.max() <= b
    assert abs(v.frames_dropped - (b - a - 10) * 14) < 30
    assert keep[(ts > 140) & (ts < 170)].all()  # the 30 s stop is under 60 s


def test_turning_on_the_spot_is_driving():
    odom = _odom([(10, 1.0, 0), (90, 0, 30), (10, 1.0, 0)])
    keep, v = idle_trim.trim(np.array([t for t, _ in odom]), odom, 60, 10)
    assert keep.all() and not v.runs


def test_a_frame_clock_hole_splits_a_run():
    odom = _odom([(200, 0, 0)])
    ts = np.array([t for t, _ in odom])
    ts = ts[(ts < 50) | (ts > 60)]
    keep, v = idle_trim.trim(ts, odom, 60, 10)
    assert len(v.runs) == 1 and v.runs[0][0] > 59  # the first 50 s run is under 60 s, the second is cut


def test_keep_longer_than_max_is_refused():
    with pytest.raises(ValueError):
        idle_trim.trim(np.arange(10.0), _odom([(10, 0, 0)]), 5, 10)
