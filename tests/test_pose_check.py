"""The scene's odometry checked against the rig's other pose stream."""

from __future__ import annotations

import numpy as np

from dataset_tools.extract import pose_check


def _track(ts, lag=0.0, scale=1.0, jumps=(), tilt_deg=0.0):
    """A robot driving a wandering curve; ``lag`` delays the content against the stamps."""
    poses = []
    for t in ts:
        tau = t + lag
        yaw = 0.4 * np.sin(0.5 * tau) + 0.1 * tau
        x = scale * (2.0 * tau + 0.5 * np.sin(0.3 * tau))
        y = scale * 1.5 * np.cos(0.2 * tau)
        T = np.eye(4)
        c, s = np.cos(yaw), np.sin(yaw)
        T[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
        if tilt_deg:
            a = np.radians(tilt_deg)
            T[:3, :3] = T[:3, :3] @ np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        T[:3, 3] = [x + sum(10.0 for j in jumps if tau > j), y, 0.1 * np.sin(0.1 * tau)]
        poses.append(T)
    return np.asarray(ts, float), np.array(poses)


def test_two_copies_of_the_same_motion_agree():
    tp, pp = _track(np.arange(0.0, 60.0, 0.069))
    tr, pr = _track(np.arange(0.0, 60.0, 0.1))
    r = pose_check.compare(tp, pp, tr, pr, max_speed=6.0)
    assert r["primary_jumps"]["steps"] == 0 and r["reference_jumps"]["steps"] == 0
    assert r["displacement"]["5s"]["median_cm"] < 1.0 and abs(r["displacement"]["5s"]["scale"] - 1.0) < 0.01
    assert abs(r["clock_lag_s"]) < 0.004 and r["clock_corr"] > 0.99
    assert abs(r["tilt_diff_deg"]["pitch"]) < 0.1
    assert r["warnings"] == []


def test_a_clock_lag_a_scale_error_and_jumps_are_each_reported():
    tp, pp = _track(np.arange(0.0, 60.0, 0.069), lag=0.04, scale=0.97, jumps=(20.0, 40.0), tilt_deg=4.0)
    tr, pr = _track(np.arange(0.0, 60.0, 0.1))
    r = pose_check.compare(tp, pp, tr, pr, max_speed=6.0)
    assert r["clock_lag_s"] == pytest.approx(0.04, abs=0.006)
    assert r["primary_jumps"]["steps"] == 2 and r["primary_jumps"]["metres"] == pytest.approx(20.0, abs=1.0)
    assert r["displacement"]["5s"]["scale"] == pytest.approx(0.97, abs=0.01)
    assert r["tilt_diff_deg"]["pitch"] == pytest.approx(4.0, abs=0.3)
    assert any("clock lag" in w for w in r["warnings"]) and any("tilt" in w for w in r["warnings"])
    assert any("jumps" in w for w in r["warnings"])


def test_the_report_is_written_and_short_streams_are_a_note(tmp_path):
    tp, pp = _track(np.arange(0.0, 30.0, 0.069))
    tr, pr = _track(np.arange(0.0, 30.0, 0.1))
    r = pose_check.compare(tp, pp, tr, pr, max_speed=6.0)
    pose_check.write_report(tmp_path / "pose_check.txt", r, "/odin1/odometry", "/fixposition/odometry_enu")
    text = (tmp_path / "pose_check.txt").read_text()
    assert "/odin1/odometry" in text and "clock lag" in text and "5s" in text
    short = pose_check.compare(tp[:2], pp[:2], tr, pr, max_speed=6.0)
    assert "note" in short
    pose_check.write_report(tmp_path / "short.txt", short, "a", "b")
    assert "too few" in (tmp_path / "short.txt").read_text()


import pytest  # noqa: E402
