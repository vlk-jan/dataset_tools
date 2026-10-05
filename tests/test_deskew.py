"""Motion compensation of a sweep onto one instant (``dataset_tools.extract.deskew``)."""

from __future__ import annotations

import numpy as np
import pytest

from dataset_tools.extract import deskew
from dataset_tools.time_align import PoseTrack
from dataset_tools.transforms import rotation_angle_deg


def _yaw(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _track(ts, yaw_rate=0.0, speed=0.0):
    """A pose stream: constant yaw rate about the origin, constant speed along the heading."""
    poses = []
    for t in ts:
        T = np.eye(4)
        T[:3, :3] = _yaw(yaw_rate * t)
        T[:3, 3] = [speed * t, 0.0, 0.0]
        poses.append(T)
    return PoseTrack.build(np.asarray(ts, float), np.array(poses))


def _seen_from(world, yaw, pos):
    """World points in the base frame of a robot at ``pos`` heading ``yaw``."""
    return (world - pos) @ _yaw(yaw)  # R^T (w - x), as a row product


def _raw_sweep(world, offsets, t_ref, yaw_rate, speed):
    """What a sensor that sees each point at ``t_ref + offset`` records, in its frame at that moment."""
    out = np.empty_like(world)
    for k, (w, o) in enumerate(zip(world, offsets, strict=True)):
        t = t_ref + o
        out[k] = _seen_from(w[None], yaw_rate * t, np.array([speed * t, 0.0, 0.0]))[0]
    return out


@pytest.fixture
def wall():
    """A vertical wall 6 m ahead, scanned bottom to top over 67 ms like the Odin1 raster."""
    rng = np.random.default_rng(3)
    y = rng.uniform(-3, 3, 400)
    z = np.sort(rng.uniform(-0.5, 2.5, 400))
    world = np.column_stack([np.full(400, 6.0), y, z])
    offsets = np.linspace(0.0, 0.067, 400)  # the top of the wall is seen last
    return world, offsets


def test_a_sweep_taken_while_turning_comes_out_straight(wall):
    world, offsets = wall
    rate = np.radians(60.0)
    track = _track(np.arange(0.0, 3.0, 0.1), yaw_rate=rate)
    t_ref = 1.5
    raw = _raw_sweep(world, offsets, t_ref, rate, 0.0)
    want = _seen_from(world, rate * t_ref, np.zeros(3))
    # The raw sweep is sheared: the top of the wall sits 4 degrees off the bottom.
    assert np.abs(raw - want).max() > 0.3

    out, status, _ = deskew.deskew(raw.astype(np.float32), offsets, t_ref, t_ref, track)
    assert status == "deskewed"
    assert out.dtype == np.float32
    assert np.abs(out - want).max() < 0.01  # a millimetre of binning, no more


def test_translation_during_the_sweep_is_undone_too(wall):
    world, offsets = wall
    track = _track(np.arange(0.0, 3.0, 0.1), yaw_rate=np.radians(30.0), speed=2.0)
    t_ref = 1.0
    raw = _raw_sweep(world, offsets, t_ref, np.radians(30.0), 2.0)
    want = _seen_from(world, np.radians(30.0) * t_ref, np.array([2.0 * t_ref, 0.0, 0.0]))
    out, status, _ = deskew.deskew(raw, offsets, t_ref, t_ref, track)
    assert status == "deskewed" and np.abs(out - want).max() < 0.01


def test_a_robot_holding_still_leaves_the_cloud_as_it_is(wall):
    world, offsets = wall
    track = _track(np.arange(0.0, 3.0, 0.1))
    raw = _seen_from(world, 0.0, np.zeros(3))
    out, status, _ = deskew.deskew(raw, offsets, 1.5, 1.5, track)
    assert status == "deskewed" and np.allclose(out, raw, atol=1e-9)


def test_an_instantaneous_cloud_is_not_touched(wall):
    world, _ = wall
    track = _track(np.arange(0.0, 3.0, 0.1), yaw_rate=1.0)
    out, status, _ = deskew.deskew(world, np.zeros(len(world)), 1.5, 1.5, track)
    assert status == "instant" and out is world


def test_a_sweep_the_poses_do_not_cover_is_left_raw(wall):
    world, offsets = wall
    track = _track(np.arange(0.0, 3.0, 0.1), yaw_rate=1.0)
    for t_ref in (-0.5, 2.98, 7.0):  # before the stream, over its end, after it
        out, status, _ = deskew.deskew(world, offsets, t_ref, t_ref, track)
        assert status == "no_pose" and out is world
    # A hole in the stream wider than MAX_GAP_S is no better than its end.
    ts = np.r_[np.arange(0.0, 1.0, 0.1), np.arange(2.0, 3.0, 0.1)]
    gappy = _track(ts, yaw_rate=1.0)
    assert deskew.deskew(world, offsets, 1.5, 1.5, gappy)[1] == "no_pose"
    assert deskew.deskew(world, offsets, 0.5, 0.5, gappy)[1] == "deskewed"


def test_the_reference_instant_may_differ_from_the_stamp(wall):
    """The dataset's frame clock is the camera/lidar midpoint, a hair off the
    lidar stamp the offsets count from; the cloud lands on the frame clock."""
    world, offsets = wall
    rate = np.radians(60.0)
    track = _track(np.arange(0.0, 3.0, 0.1), yaw_rate=rate, speed=1.0)
    stamp, t_ref = 1.5, 1.515
    raw = _raw_sweep(world, offsets, stamp, rate, 1.0)
    want = _seen_from(world, rate * t_ref, np.array([t_ref, 0.0, 0.0]))
    out, status, _ = deskew.deskew(raw, offsets, stamp, t_ref, track)
    assert status == "deskewed" and np.abs(out - want).max() < 0.01
    # An instantaneous cloud still moves when the stamp is not the reference.
    snap = _seen_from(world, rate * stamp, np.array([stamp, 0.0, 0.0]))
    out, status, _ = deskew.deskew(snap, np.zeros(len(world)), stamp, t_ref, track)
    assert status == "deskewed" and np.abs(out - want).max() < 0.01


def test_an_empty_cloud_reports_empty():
    track = _track(np.arange(0.0, 1.0, 0.1))
    out, status, _ = deskew.deskew(np.zeros((0, 3)), np.zeros(0), 0.5, 0.5, track)
    assert status == "empty" and len(out) == 0


def test_covered_requires_a_bracket_without_a_wide_gap():
    ts = np.array([0.0, 0.1, 0.2, 1.0, 1.1])
    assert deskew.covered(ts, 0.05, 0.15)
    assert deskew.covered(ts, 0.0, 0.2)
    assert not deskew.covered(ts, 0.15, 0.95)  # spans the 0.8 s hole
    assert not deskew.covered(ts, -0.1, 0.1)
    assert not deskew.covered(ts, 1.05, 1.2)
    assert not deskew.covered(ts[:1], 0.0, 0.0)


# --------------------------------------------------------------------------
# the gyro inside the sweep
# --------------------------------------------------------------------------


def _gyro(ts, rate_fn):
    return deskew.GyroTrack.build([(t, 0.0, 0.0, rate_fn(t)) for t in ts])


def _track_from_rate(ts, rate_fn):
    """A pose stream whose yaw is the integral of ``rate_fn`` (fine steps, then sampled)."""
    fine = np.arange(ts[0], ts[-1] + 1e-9, 0.001)
    yaw = np.concatenate([[0.0], np.cumsum([rate_fn(t) * 0.001 for t in fine[:-1]])])
    poses = []
    for t in ts:
        T = np.eye(4)
        T[:3, :3] = _yaw(float(np.interp(t, fine, yaw)))
        poses.append(T)
    return PoseTrack.build(np.asarray(ts, float), np.array(poses)), (fine, yaw)


def test_the_gyro_beats_the_slerp_when_the_rate_changes_inside_the_sweep(wall):
    """Slerp across two 14 Hz samples assumes a constant rate; a robot that is
    braking its turn hard inside the sweep is caught only by the gyro."""
    world, offsets = wall
    rate = lambda t: np.radians(90.0) * np.sin(2 * np.pi * 1.5 * t)  # noqa: E731  -- rate swings within 70 ms
    ts = np.arange(0.0, 3.0, 0.069)
    track, (fine, yaw) = _track_from_rate(ts, rate)
    gyro = _gyro(np.arange(0.0, 3.0, 0.0025), rate)
    t_ref = float(ts[20])
    yaw_at = lambda t: float(np.interp(t, fine, yaw))  # noqa: E731
    raw = np.array(
        [_seen_from(w[None], yaw_at(t_ref + o), np.zeros(3))[0] for w, o in zip(world, offsets, strict=True)]
    )
    want = _seen_from(world, yaw_at(t_ref), np.zeros(3))

    by_slerp, status, method = deskew.deskew(raw, offsets, t_ref, t_ref, track)
    assert status == "deskewed" and method == "slerp"
    by_gyro, status, method = deskew.deskew(raw, offsets, t_ref, t_ref, track, gyro)
    assert status == "deskewed" and method == "gyro"
    assert np.abs(by_gyro - want).max() < 0.01
    assert np.abs(by_gyro - want).max() < 0.25 * np.abs(by_slerp - want).max()


def test_a_gyro_that_disagrees_with_the_poses_is_not_believed(wall):
    world, offsets = wall
    rate = np.radians(60.0)
    track = _track(np.arange(0.0, 3.0, 0.1), yaw_rate=rate)
    broken = _gyro(np.arange(0.0, 3.0, 0.0025), lambda t: rate * 1.5)  # 50 % scale error: 2 deg over the sweep
    raw = _raw_sweep(world, offsets, 1.5, rate, 0.0)
    out, status, method = deskew.deskew(raw, offsets, 1.5, 1.5, track, broken)
    assert status == "deskewed" and method == "slerp"
    assert np.abs(out - _seen_from(world, rate * 1.5, np.zeros(3))).max() < 0.01
    good = _gyro(np.arange(0.0, 3.0, 0.0025), lambda t: rate)
    assert deskew.deskew(raw, offsets, 1.5, 1.5, track, good)[2] == "gyro"


def test_a_gyro_with_a_hole_under_the_sweep_falls_back_to_slerp(wall):
    world, offsets = wall
    rate = np.radians(60.0)
    track = _track(np.arange(0.0, 3.0, 0.1), yaw_rate=rate)
    ts = np.r_[np.arange(0.0, 1.5, 0.0025), np.arange(1.6, 3.0, 0.0025)]
    gappy = _gyro(ts, lambda t: rate)
    raw = _raw_sweep(world, offsets, 1.5, rate, 0.0)
    assert deskew.deskew(raw, offsets, 1.5, 1.5, track, gappy)[2] == "slerp"
    assert deskew.deskew(raw, offsets, 2.0, 2.0, track, gappy)[2] == "gyro"


def test_the_gyro_is_rotated_into_the_base_frame():
    rows = [(t, 0.0, 1.0, 0.0) for t in np.arange(0.0, 1.0, 0.01)]  # the sensor's y is the base's z
    R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    g = deskew.GyroTrack.build(rows, R)
    assert np.allclose(g.omega[:, 2], 1.0) and np.allclose(g.omega[:, :2], 0.0)
    Rr = g.relative_rotations(0.5, np.array([0.6]))
    assert rotation_angle_deg(Rr[0]) == pytest.approx(np.degrees(0.1), abs=1e-6)
    assert Rr[0][1, 0] > 0  # a positive yaw
    assert deskew.GyroTrack.build([]) is None
