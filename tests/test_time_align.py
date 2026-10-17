"""The frame-clock estimator on synthetic scenes whose ``data_ts`` is stamped late by a known amount."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dataset_tools import time_align as ta
from dataset_tools.gt_io import write_trajectory

FRAMES = 700
FRAME_RATE = 14.0
POSE_RATE = 10.0
T0 = 1_700_000_000.0


def _pose(t: float) -> np.ndarray:
    """A curve at varying speed; constant velocity would make the offset unobservable."""
    yaw = 0.22 * t
    dist = 0.9 * t - (0.5 / 0.55) * (np.cos(0.55 * t) - 1.0)  # 0.4 - 1.4 m/s
    T = np.eye(4)
    T[:3, :3] = [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    T[:3, 3] = [dist * np.cos(0.5 * yaw), dist * np.sin(0.5 * yaw), 0.0]
    return T


@pytest.fixture(scope="module")
def world() -> np.ndarray:
    """Points on a ground plane plus scattered posts, in the world frame."""
    rng = np.random.default_rng(7)
    ground = np.column_stack([rng.uniform(-40, 90, 90000), rng.uniform(-70, 70, 90000)])
    ground = np.column_stack([ground, rng.normal(-0.35, 0.02, len(ground))])
    posts = [
        np.column_stack([np.full(120, cx), np.full(120, cy), rng.uniform(-0.3, 2.2, 120)])
        for cx, cy in rng.uniform(-35, 85, size=(220, 2))
    ]
    return np.vstack([ground, *posts])


def build_scene(root: Path, world: np.ndarray, latency) -> tuple[Path, np.ndarray]:
    """A scene stamped late by ``latency`` seconds (a number, or a function of t);
    the recovered profile must come out negative."""
    scene = root / "0000"
    (scene / "clouds").mkdir(parents=True)
    tau = T0 + np.arange(FRAMES) / FRAME_RATE
    lat = np.full(FRAMES, float(latency)) if np.isscalar(latency) else np.array([latency(t - T0) for t in tau])
    np.savetxt(scene / "data_ts.txt", tau + lat)

    pose_ts = T0 + np.arange(-20, FRAMES / FRAME_RATE * POSE_RATE + 20) / POSE_RATE
    write_trajectory(scene / "odometry.txt", pose_ts, [_pose(t - T0) for t in pose_ts])

    for k, t in enumerate(tau):
        T = _pose(t - T0)
        local = (world - T[:3, 3]) @ T[:3, :3]
        r = np.hypot(local[:, 0], local[:, 1])
        keep = (r > 1.0) & (r < 16.0) & (local[:, 0] > 0)  # forward-looking, like the real unit
        pts = local[keep]
        np.save(scene / "clouds" / f"{k:05d}.npy", np.column_stack([pts, np.ones(len(pts))]).astype(np.float32))
    return scene, -lat


def _estimate(scene: Path, limit: int = 12):
    data_ts = np.loadtxt(scene / "data_ts.txt")
    from dataset_tools.gt_io import read_trajectory

    ts, poses = read_trajectory(scene / "odometry.txt")
    return data_ts, ta.estimate(scene, data_ts, ts, poses, limit=limit, span=1.0)


@pytest.mark.parametrize("latency", [0.70, -0.35])
def test_a_constant_latency_comes_back(tmp_path, world, latency):
    scene, want = build_scene(tmp_path / f"c{latency}", world, latency)
    data_ts, profile = _estimate(scene)

    assert not profile.is_identity
    assert profile.at(data_ts) == pytest.approx(want, abs=0.05)


def test_a_latency_that_changes_is_followed(tmp_path, world):
    """Half the drive 0.70 s late, half not (v2 beta 0000): the curve follows both."""
    edge = 25.0  # seconds into the drive

    scene, want = build_scene(tmp_path / "step", world, lambda t: 0.70 if t < edge else 0.0)
    data_ts, profile = _estimate(scene, limit=24)
    got = profile.at(data_ts)

    assert not profile.is_identity
    # Windows are 80 frames, so one straddles the edge; judge away from it.
    early = (data_ts - data_ts[0]) < edge - 8
    late = (data_ts - data_ts[0]) > edge + 8
    assert got[early] == pytest.approx(want[early], abs=0.07)
    assert got[late] == pytest.approx(want[late], abs=0.07)
    assert got[early].mean() - got[late].mean() == pytest.approx(-0.70, abs=0.10)


def test_an_aligned_scene_is_left_alone(tmp_path, world):
    """A rig that stamps at capture must come back untouched, not near-zero."""
    scene, _ = build_scene(tmp_path / "aligned", world, 0.0)
    _, profile = _estimate(scene)

    assert profile.is_identity and "already aligned" in profile.summary
    assert np.median([w["gain"] for w in profile.windows]) < 10.0  # sharp as it stands


def test_the_estimate_does_not_scale_with_speed(tmp_path, world):
    """A real latency is recovered independent of drive speed (unlike a spatial offset)."""
    scene, _ = build_scene(tmp_path / "speeds", world, 0.50)
    data_ts = np.loadtxt(scene / "data_ts.txt")
    from dataset_tools.gt_io import read_trajectory

    ts, poses = read_trajectory(scene / "odometry.txt")
    track = ta.PoseTrack.build(ts, poses)

    step = np.hypot(*np.diff(poses[:, :2, 3], axis=0).T)
    seen = []
    for frames in ta.pick_windows(data_ts, track, 10):
        w = ta.measure_window(scene, frames, track, data_ts, span=1.0)
        lo = np.searchsorted(ts, data_ts[frames[0]])
        hi = np.searchsorted(ts, data_ts[frames[-1]])
        seen.append((step[lo:hi].sum() / (data_ts[frames[-1]] - data_ts[frames[0]]), w["delta"]))

    seen.sort()
    assert seen[-1][0] > 1.3 * seen[0][0], "the windows must actually differ in speed"
    assert seen[0][1] == pytest.approx(seen[-1][1], abs=0.06)


def test_a_wild_window_cannot_bend_the_curve(tmp_path, world):
    """One bad measurement among good ones is filtered, not averaged in."""
    good = [{"t": float(t), "delta": -0.70, "gain": 40.0, "depth": 40.0} for t in range(0, 100, 10)]
    good[5]["delta"] = +1.90  # a window that found the wrong minimum
    ts, d = ta._knots(good)

    assert len(ts) == len(good)
    assert d == pytest.approx(np.full(len(good), -0.70), abs=1e-9)


def test_a_wild_first_or_last_window_is_filtered_too():
    """At the ends the median is of the nearest three, not a mean of two."""
    good = [{"t": float(t), "delta": -0.70, "gain": 40.0, "depth": 40.0} for t in range(0, 100, 10)]
    good[0]["delta"] = +1.90
    good[-1]["delta"] = +1.90
    _, d = ta._knots(good)
    assert d == pytest.approx(np.full(len(good), -0.70), abs=1e-9)


def test_a_window_whose_clouds_are_all_empty_fills_no_voxel():
    """Clouds cropped to nothing must not fail the scene late with a ValueError."""
    assert ta.occupied([np.zeros((0, 3))], np.zeros((1, 3)), np.eye(3)[None]) == 0


def test_a_backwards_clock_is_forced_into_order(tmp_path):
    """A frame clock that ran backwards would break every consumer downstream."""
    frame_ts = T0 + np.arange(50) / 10.0
    # A profile no real latency produces: falling faster than time passes.
    profile = ta.Profile([], np.array([frame_ts[0], frame_ts[-1]]), np.array([0.0, -20.0]), "synthetic")
    out, forced = ta.apply_to_clock(frame_ts, profile)

    assert forced
    assert np.all(np.diff(out) > 0)


def test_a_long_scene_is_measured_to_its_end(tmp_path):
    """Thinning windows makes coverage sparser, never shorter."""
    n = 14_000
    frame_ts = T0 + np.arange(n) / FRAME_RATE
    t = np.arange(2000) / 2.0  # 1000 s of pose at 2 Hz
    yaw = 0.01 * t
    poses = np.tile(np.eye(4), (2000, 1, 1))
    poses[:, 0, 0] = poses[:, 1, 1] = np.cos(yaw)
    poses[:, 0, 1] = -np.sin(yaw)
    poses[:, 1, 0] = np.sin(yaw)
    poses[:, 0, 3] = np.cumsum(np.full(2000, 0.5) * np.cos(yaw))  # ~1 m/s, turning
    poses[:, 1, 3] = np.cumsum(np.full(2000, 0.5) * np.sin(yaw))
    track = ta.PoseTrack.build(T0 + t, poses)

    frames = ta.pick_windows(frame_ts, track, limit=ta.MAX_WINDOWS)

    assert len(frames) <= ta.MAX_WINDOWS
    # The last window has to reach the last stretch of the scene, not stop at
    # the point where the cap ran out.
    assert frames[-1][-1] > n - 3 * ta.WINDOW


def test_a_small_but_consistent_offset_is_a_measurement_not_noise():
    """Small knots all on one side are an offset, not 'already aligned' (v2.2 0011)."""
    from dataset_tools.time_align import ZERO_TOL, already_aligned

    weak = [{"gain": 9.0}, {"gain": 12.0}, {"gain": 10.0}, {"gain": 6.0}]
    assert not already_aligned(weak, np.array([0.040, 0.039, 0.044, 0.048]))
    assert already_aligned(weak, np.array([0.004, -0.003, 0.006, -0.002]))
    assert not already_aligned(weak, np.array([0.002, 0.003, 0.080, 0.001]))  # one wild knot: not aligned
    assert not already_aligned([{"gain": 30.0}], np.array([0.004]))  # a strong gain is an offset
    assert not already_aligned(weak, np.zeros(0))
    assert ZERO_TOL == 0.010
