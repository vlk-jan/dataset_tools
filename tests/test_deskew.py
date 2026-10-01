"""Motion compensation of a sweep onto one instant (``dataset_tools.extract.deskew``)."""

from __future__ import annotations

import numpy as np
import pytest

from dataset_tools.extract import deskew
from dataset_tools.time_align import PoseTrack


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

    out, status = deskew.deskew(raw.astype(np.float32), offsets, t_ref, t_ref, track)
    assert status == "deskewed"
    assert out.dtype == np.float32
    assert np.abs(out - want).max() < 0.01  # a millimetre of binning, no more


def test_translation_during_the_sweep_is_undone_too(wall):
    world, offsets = wall
    track = _track(np.arange(0.0, 3.0, 0.1), yaw_rate=np.radians(30.0), speed=2.0)
    t_ref = 1.0
    raw = _raw_sweep(world, offsets, t_ref, np.radians(30.0), 2.0)
    want = _seen_from(world, np.radians(30.0) * t_ref, np.array([2.0 * t_ref, 0.0, 0.0]))
    out, status = deskew.deskew(raw, offsets, t_ref, t_ref, track)
    assert status == "deskewed" and np.abs(out - want).max() < 0.01


def test_a_robot_holding_still_leaves_the_cloud_as_it_is(wall):
    world, offsets = wall
    track = _track(np.arange(0.0, 3.0, 0.1))
    raw = _seen_from(world, 0.0, np.zeros(3))
    out, status = deskew.deskew(raw, offsets, 1.5, 1.5, track)
    assert status == "deskewed" and np.allclose(out, raw, atol=1e-9)


def test_an_instantaneous_cloud_is_not_touched(wall):
    world, _ = wall
    track = _track(np.arange(0.0, 3.0, 0.1), yaw_rate=1.0)
    out, status = deskew.deskew(world, np.zeros(len(world)), 1.5, 1.5, track)
    assert status == "instant" and out is world


def test_a_sweep_the_poses_do_not_cover_is_left_raw(wall):
    world, offsets = wall
    track = _track(np.arange(0.0, 3.0, 0.1), yaw_rate=1.0)
    for t_ref in (-0.5, 2.98, 7.0):  # before the stream, over its end, after it
        out, status = deskew.deskew(world, offsets, t_ref, t_ref, track)
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
    out, status = deskew.deskew(raw, offsets, stamp, t_ref, track)
    assert status == "deskewed" and np.abs(out - want).max() < 0.01
    # An instantaneous cloud still moves when the stamp is not the reference.
    snap = _seen_from(world, rate * stamp, np.array([stamp, 0.0, 0.0]))
    out, status = deskew.deskew(snap, np.zeros(len(world)), stamp, t_ref, track)
    assert status == "deskewed" and np.abs(out - want).max() < 0.01


def test_an_empty_cloud_reports_empty():
    track = _track(np.arange(0.0, 1.0, 0.1))
    out, status = deskew.deskew(np.zeros((0, 3)), np.zeros(0), 0.5, 0.5, track)
    assert status == "empty" and len(out) == 0


def test_covered_requires_a_bracket_without_a_wide_gap():
    ts = np.array([0.0, 0.1, 0.2, 1.0, 1.1])
    assert deskew.covered(ts, 0.05, 0.15)
    assert deskew.covered(ts, 0.0, 0.2)
    assert not deskew.covered(ts, 0.15, 0.95)  # spans the 0.8 s hole
    assert not deskew.covered(ts, -0.1, 0.1)
    assert not deskew.covered(ts, 1.05, 1.2)
    assert not deskew.covered(ts[:1], 0.0, 0.0)


def test_build_track_needs_two_poses():
    assert deskew.build_track([]) is None
    assert deskew.build_track([(0.0, np.eye(4))]) is None
    track = deskew.build_track([(1.0, np.eye(4)), (0.0, np.eye(4))])
    assert track is not None and list(track.ts) == [0.0, 1.0]
