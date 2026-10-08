"""Viewer: the map's arithmetic (dataset_tools.viewer.geo), checked against hand-worked numbers."""

from __future__ import annotations

import numpy as np
import pytest

from dataset_tools.viewer import geo

#: Prague, where the recordings this was written for were made.
LAT0, LON0 = 50.109, 14.418


def track(n: int = 100, *, dlat: float = 1e-5, dlon: float = 0.0, t0: float = 1_700_000_000.0) -> np.ndarray:
    """A straight, evenly-timed track at 10 Hz, RTK throughout."""
    rows = np.zeros((n, 8))
    rows[:, geo.T] = t0 + 0.1 * np.arange(n)
    rows[:, geo.LAT] = LAT0 + dlat * np.arange(n)
    rows[:, geo.LON] = LON0 + dlon * np.arange(n)
    rows[:, geo.ALT] = 227.0
    rows[:, geo.STATUS] = 2
    rows[:, geo.COV] = [0.09, 0.16, 0.25]
    return rows


def test_a_degree_of_latitude_is_a_hundred_and_eleven_kilometres():
    # 99 steps of 1e-5 deg: 99 * 1.1132 m, to a tenth of a percent.
    assert geo.track_length_m(track()) == pytest.approx(99 * 1.1132, rel=1e-3)


def test_longitude_shrinks_with_the_cosine_of_the_latitude():
    """The same number of degrees east is shorter on the ground than north."""
    north = geo.track_length_m(track(dlat=1e-5, dlon=0.0))
    east = geo.track_length_m(track(dlat=0.0, dlon=1e-5))
    assert east == pytest.approx(north * np.cos(np.radians(LAT0)), rel=1e-4)


def test_extent_and_bounds_describe_the_same_box():
    rows = track(dlat=1e-5, dlon=1e-5)
    (south, west), (north, east) = geo.bounds(rows)
    assert south == pytest.approx(LAT0) and north == pytest.approx(LAT0 + 99e-5)
    assert west == pytest.approx(LON0) and east == pytest.approx(LON0 + 99e-5)
    ew, ns = geo.span_m(rows)
    assert ns == pytest.approx(99 * 1.1132, rel=1e-3)
    assert ew < ns  # a degree east is shorter than a degree north, at 50 deg


def test_decimation_keeps_both_ends_and_the_cap():
    thinned = geo.decimate(track(1000), 100)
    assert len(thinned) <= 101
    assert np.array_equal(thinned[0], track(1000)[0])
    assert np.array_equal(thinned[-1], track(1000)[-1])
    # A track already short enough is handed back untouched.
    short = track(10)
    assert geo.decimate(short, 100) is short


def test_a_time_is_located_between_the_fixes_around_it():
    rows = track()
    t0 = rows[0, geo.T]
    lat, lon = geo.locate(rows, t0 + 0.55)
    # Halfway between fix 5 and fix 6.
    assert lat == pytest.approx(LAT0 + 5.5e-5)
    assert lon == pytest.approx(LON0)


def test_a_time_in_a_receiver_dropout_is_not_located():
    """A straight line across a gap would be a claim about where the robot went."""
    rows = np.vstack([track(20), track(20, t0=1_700_000_000.0 + 30.0)])
    assert geo.locate(rows, rows[19, geo.T] + 5.0) is None
    # Just outside the track, but within a frame period of its end: still fine.
    assert geo.locate(rows, rows[-1, geo.T] + 0.2) is not None
    assert geo.locate(rows, rows[-1, geo.T] + 10.0) is None


def test_fix_quality_and_accuracy_come_out_of_the_columns():
    rows = track(100)
    rows[:40, geo.STATUS] = 0  # the receiver spent 40 % on a single-point fix
    shares = geo.fix_shares(rows)
    assert shares[0] == ("RTK / GBAS", pytest.approx(0.6))
    assert shares[1] == ("single", pytest.approx(0.4))
    # sqrt(cov_ee + cov_nn) = sqrt(0.25): the one-sigma circle's radius.
    assert geo.sigma_m(rows) == pytest.approx(0.5)


def test_an_unknown_covariance_reports_no_accuracy_rather_than_zero():
    rows = track(10)
    rows[:, geo.COV] = np.nan  # COVARIANCE_TYPE_UNKNOWN, as the extractor writes it
    assert geo.sigma_m(rows) is None


def test_a_slice_never_reaches_outside_its_segment():
    rows = track(100)
    t0 = rows[0, geo.T]
    inside = geo.slice_time(rows, t0 + 1.0, t0 + 2.0)
    assert len(inside) == 11  # inclusive of both edges, at 10 Hz
    assert inside[0, geo.T] == pytest.approx(t0 + 1.0)
    assert not len(geo.slice_time(rows, t0 + 100.0, t0 + 200.0))


def test_a_summary_survives_a_track_of_one_fix():
    """A one-fix scene is a point on the map, not a crash."""
    one = track(1)
    assert geo.track_length_m(one) == 0.0 and geo.span_m(one) == (0.0, 0.0)
    assert len(geo.points(geo.decimate(one, 2000))) == 1
    bounds = geo.bounds(one)
    assert bounds is not None and bounds[0] == bounds[1]


def test_what_is_not_a_track_is_rejected():
    assert not geo.is_track(None)
    assert not geo.is_track(np.zeros((0, 8)))
    assert not geo.is_track(np.zeros((5, 2)))  # no longitude column
    assert geo.is_track(track(2))
