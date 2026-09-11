"""The guard that keeps a reset GNSS/INS solution out of the exported poses."""

from __future__ import annotations

import numpy as np
import pytest

from dataset_tools import gnss_guard as gg


def pose(x=0.0, y=0.0, z=0.0, R=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    T[:3, 3] = (x, y, z)
    return T


def drive(n, t0=1000.0, rate=10.0, speed=1.0, x0=0.0):
    """A robot going straight at ``speed``, with the dither a real fix has."""
    rng = np.random.default_rng(0)
    ts = t0 + np.arange(n) / rate
    xs = x0 + speed * (ts - t0) + rng.normal(0, 1e-3, n)
    return [(float(t), pose(float(x))) for t, x in zip(ts, xs, strict=True)]


def reset(n, t0, rate=10.0, at=(-1147.98, 357.94, 27.03)):
    """What the receiver publishes with no solution: one position, held."""
    return [(float(t0 + i / rate), pose(*at)) for i in range(n)]


def test_a_reset_at_the_end_of_a_scene_is_removed():
    odom = drive(600) + reset(460, 1060.0)
    frame_ts = np.arange(1000.0, 1106.0, 0.08)
    kept, keep, verdict = gg.guard(odom, frame_ts)

    assert len(verdict.spans) == 1 and verdict.spans[0].reason == "frozen"
    assert len(kept) == 600
    # every surviving frame is inside the driving, none inside the reset
    assert frame_ts[keep].max() < 1060.0


def test_a_round_trip_excursion_is_removed_and_the_scene_closes_over_it():
    before = drive(400)
    after = drive(400, t0=1043.0, x0=40.0)
    odom = before + reset(30, 1040.0, at=(-1028.5, -531.8, 61.6)) + after
    frame_ts = np.arange(1000.0, 1083.0, 0.08)
    kept, keep, verdict = gg.guard(odom, frame_ts)

    kept_ts = np.array([r[0] for r in kept])
    assert not ((kept_ts >= 1040.0) & (kept_ts < 1043.0)).any()
    assert len(kept) == 800
    surviving = frame_ts[keep]
    assert not ((surviving > 1039.0) & (surviving < 1043.9)).any()


def test_a_roll_over_is_data_and_is_left_alone():
    """The robot really does flip. The flip is the measurement, not the fault:
    nothing in the guard reads the rotation."""
    odom = drive(600)
    flipped = np.diag([1.0, -1.0, -1.0])  # 180 deg roll
    odom = [(t, T if i < 200 else pose(*T[:3, 3], R=flipped)) for i, (t, T) in enumerate(odom)]
    kept, keep, verdict = gg.guard(odom, np.arange(1000.0, 1060.0, 0.08))

    assert verdict.clean
    assert len(kept) == len(odom) and keep.all()


def test_standing_still_is_not_a_reset():
    """A real solution dithers. Across HELHEST v2 beta no genuine run of
    bit-identical positions lasts longer than 0.3 s."""
    odom = drive(300) + drive(300, t0=1030.0, speed=0.0, x0=30.0) + drive(300, t0=1060.0, x0=30.0)
    _, keep, verdict = gg.guard(odom, np.arange(1000.0, 1090.0, 0.08))
    assert verdict.clean and keep.all()


def test_a_long_dropout_the_robot_drove_through_is_not_a_teleport():
    """5.2 m between two samples 62 s apart. That is driving, not a jump."""
    odom = drive(400) + drive(400, t0=1102.0, x0=45.0)
    kept, keep, verdict = gg.guard(odom, np.arange(1000.0, 1142.0, 0.08))

    # No span: the stream said nothing for a minute, and the ground the robot
    # covered meanwhile is not a jump. The poses all survive.
    assert not verdict.spans and len(kept) == len(odom)
    # The frames inside the dropout do not, because nothing measured them.
    frame_ts = np.arange(1000.0, 1142.0, 0.08)
    gone = frame_ts[~keep]
    assert gone.min() > 1040.0 and gone.max() < 1102.0
    assert keep[frame_ts < 1039.9].all() and keep[frame_ts > 1102.0].all()


def test_a_hole_in_the_fix_stream_widens_a_span():
    """The receiver went quiet a second before it froze and stayed quiet a
    second after it recovered; the span is the whole of what it missed."""
    odom = drive(400) + reset(30, 1040.0) + drive(400, t0=1043.0, x0=40.0)
    fixes = np.r_[np.arange(1000.0, 1039.0, 0.1), np.arange(1044.4, 1083.0, 0.1)]
    _, _, verdict = gg.guard(odom, np.arange(1000.0, 1083.0, 0.08), fixes)

    assert len(verdict.spans) == 1
    span = verdict.spans[0]
    assert "no_fix" in span.reason
    assert span.start == pytest.approx(1038.9) and span.end == pytest.approx(1044.4)


def test_an_outage_the_solution_coasted_through_costs_nothing():
    """An inertial solution rides out a short outage; throwing that away would
    cost good data for nothing."""
    odom = drive(800)
    fixes = np.r_[np.arange(1000.0, 1030.0, 0.1), np.arange(1040.0, 1080.0, 0.1)]
    _, keep, verdict = gg.guard(odom, np.arange(1000.0, 1080.0, 0.08), fixes)
    assert verdict.clean and keep.all()


def test_frames_are_dropped_with_a_margin_for_the_uncorrected_clock():
    """The guard runs before the capture-to-publish latency is measured, so a
    frame stamp may sit a latency away from the pose clock."""
    spans = [gg.Span(1040.0, 1042.0, "frozen")]
    frame_ts = np.array([1038.5, 1039.5, 1041.0, 1042.5, 1043.5])
    keep = gg.frame_mask(frame_ts, spans, pad=1.0)
    assert list(keep) == [True, False, False, False, True]


def test_a_stream_too_short_to_judge_is_left_alone():
    kept, keep, verdict = gg.guard([(1000.0, pose())], np.arange(1000.0, 1001.0, 0.1))
    assert verdict.clean and len(kept) == 1 and keep.all()


def test_frames_past_the_end_of_the_poses_are_not_exported():
    """Scene 0016: the receiver stops 57 s before the camera does. Those frames
    would carry the last sample and show a robot that parked itself."""
    odom = drive(400)
    frame_ts = np.arange(1000.0, 1100.0, 0.08)
    kept, keep, verdict = gg.guard(odom, frame_ts)

    assert not verdict.spans and len(kept) == len(odom)
    assert not verdict.clean and verdict.n_frames_unsupported == int((~keep).sum())
    # 1039.9 is the last pose, plus the margin for the uncorrected frame clock.
    assert frame_ts[keep].max() <= 1040.9 and frame_ts[keep].max() > 1040.0


def test_a_hole_shorter_than_the_interpolator_minds_costs_nothing():
    """A dropped message or two still brackets a frame closely enough to
    interpolate between, which is the only question being asked."""
    odom = drive(200) + drive(200, t0=1020.3, x0=20.3)
    _, keep, verdict = gg.guard(odom, np.arange(1000.0, 1040.0, 0.08))
    assert verdict.clean and keep.all()


def test_the_margin_is_spent_towards_keeping_the_frame():
    """frame_mask drops a frame that *might* sit in a span; frame_support keeps
    a frame that *might* sit on a pose. Both err away from fabricating one."""
    odom_ts = np.r_[np.arange(1000.0, 1010.0, 0.1), np.arange(1015.0, 1025.0, 0.1)]
    frame_ts = np.array([1010.5, 1011.0, 1012.5, 1014.0, 1014.5])
    keep = gg.frame_support(frame_ts, odom_ts, pad=1.0)
    assert list(keep) == [True, False, False, True, True]


def test_the_report_counts_the_frames_that_had_no_pose(tmp_path):
    odom = drive(400)
    _, _, verdict = gg.guard(odom, np.arange(1000.0, 1100.0, 0.08))
    gg.write_report(tmp_path / "gnss_guard.txt", verdict, 1000.0)
    assert "further frames had no pose under them" in (tmp_path / "gnss_guard.txt").read_text()


def test_the_report_names_every_span_and_what_it_cost(tmp_path):
    odom = drive(600) + reset(460, 1060.0)
    _, _, verdict = gg.guard(odom, np.arange(1000.0, 1106.0, 0.08))
    gg.write_report(tmp_path / "gnss_guard.txt", verdict, 1000.0)
    text = (tmp_path / "gnss_guard.txt").read_text()
    assert "frozen" in text and "60.000" in text


def test_a_span_covers_the_samples_that_are_fabricated_and_no_more():
    """Neither detector reaches past what it found: a teleport elsewhere in the
    scene is a different event, and the samples beside a fabricated stretch are
    real poses."""
    odom = drive(200) + [(1020.0, pose(60.0))] + drive(200, t0=1020.1, x0=60.0)
    odom += reset(30, 1040.1) + drive(200, t0=1043.1, x0=80.0)
    ts = np.array([r[0] for r in odom])
    poses = np.array([r[1] for r in odom])
    (span,) = gg.find_spans(ts, poses)
    assert span.start == pytest.approx(1040.1) and span.end == pytest.approx(1043.0)


def test_a_robot_that_is_simply_stuck_keeps_its_frames():
    """Scene 0009: the position is bit-identical for seconds at a time while
    the driver commands 1.3 m/s into a robot that will not move. The fix is
    continuous, the covariance is the scene's best, and the GNSS latitude and
    longitude are just as still as the odometry. Nothing was fabricated -- a
    Fixposition quantises to a micrometre, so a stopped robot publishes the
    same number, and a freeze with no jump and no admitted outage is not
    evidence of anything."""
    parked = [(1030.0 + i / 10.0, pose(30.0)) for i in range(30)]  # 3 s, bit-identical
    odom = drive(300) + parked + drive(300, t0=1033.0, x0=30.0)
    kept, keep, verdict = gg.guard(odom, np.arange(1000.0, 1063.0, 0.08))
    assert verdict.clean
    assert len(kept) == len(odom) and keep.all()


def test_the_same_freeze_is_removed_once_a_jump_agrees_with_it():
    """The only difference from the scene above: the position got there by
    moving 900 m in 100 ms."""
    frozen = [(1030.0 + i / 10.0, pose(900.0)) for i in range(30)]
    odom = drive(300) + frozen + drive(300, t0=1033.0, x0=30.0)
    _, _, verdict = gg.guard(odom, np.arange(1000.0, 1063.0, 0.08))
    assert len(verdict.spans) == 1 and "frozen" in verdict.spans[0].reason


def test_a_lone_jump_inside_an_outage_is_removed():
    """Scene 0014 left a 3.29 m step at 53 m/s standing: nothing froze and no
    second jump fenced it, so neither of the other rules saw it. What it sits
    inside is a 12.7 s hole in the fix stream -- and that hole is a silence,
    not null island, so it has to be read off the good fixes around it."""
    odom = drive(400) + [(1040.0 + i / 10.0, pose(43.3 + 0.1 * i)) for i in range(120)]
    fixes = np.r_[np.arange(1000.0, 1039.0, 0.1), np.arange(1051.8, 1053.0, 0.1)]
    frame_ts = np.arange(1000.0, 1053.0, 0.08)
    _, keep, verdict = gg.guard(odom, frame_ts, fixes)

    assert len(verdict.spans) == 1 and verdict.spans[0].reason == "no_fix+teleport"
    assert not ((frame_ts[keep] > 1039.0) & (frame_ts[keep] < 1051.7)).any()


def test_an_outage_with_no_jump_in_it_is_still_left_alone():
    odom = drive(800)
    fixes = np.r_[np.arange(1000.0, 1030.0, 0.1), np.arange(1043.0, 1080.0, 0.1)]
    _, keep, verdict = gg.guard(odom, np.arange(1000.0, 1080.0, 0.08), fixes)
    assert verdict.clean and keep.all()
