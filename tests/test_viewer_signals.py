"""Viewer: the signal charts on the scene and segment pages (dataset_tools.viewer.signals)."""

from __future__ import annotations

import json

import numpy as np
from helpers_viewer import make_root

from dataset_tools.viewer import probe, signals


def access_for(tmp_path, **kw):
    from dataset_tools.viewer.build import SceneAccess

    root = probe.DatasetRoot(make_root(tmp_path, **kw), cache_dir=tmp_path / "cache")
    return SceneAccess(root, "0000")


def test_resample_means_bins_bridges_short_gaps_and_leaves_holes():
    t = np.array([0.0, 0.1, 0.2, 5.0, 5.1])
    v = np.array([1.0, 3.0, 5.0, 2.0, 2.0])
    out = signals.resample(t, v, 0.0, 6.0, n=6)
    assert out[0] == 3.0  # the mean of the three samples in [0, 1)
    assert np.isnan(out[1:5]).all()  # 4.8 s without a sample is a hole, not a ramp
    assert out[5] == 2.0
    # A sparse signal is interpolated between samples that sit close together.
    bridged = signals.resample(np.array([0.0, 1.0]), np.array([0.0, 10.0]), 0.0, 1.0, n=4)
    assert np.allclose(bridged, [0.0, 3.75, 6.25, 8.75])  # t=1.0 falls just past the last column
    peak = signals.resample(np.array([0.1, 0.2, 0.3]), np.array([1.0, 9.0, 1.0]), 0.0, 1.0, n=1, how="max")
    assert peak[0] == 9.0


def test_a_scene_offers_one_chart_per_unit_from_the_files_it_has(tmp_path):
    access = access_for(tmp_path)
    ts = access.frame_ts
    sig = signals.signal_charts(access, float(ts[0]), float(ts[-1]))
    keys = [c["key"] for c in sig["charts"]]
    assert keys == ["speed", "wheel_left", "wheel_rear", "wheel_right", "effort", "attitude", "frame_dt"]
    speed = next(c for c in sig["charts"] if c["key"] == "speed")
    assert [s["label"] for s in speed["series"]] == ["ground truth", "robot odometry", "commanded"]
    # The three wheels share one y-range, so they read against each other.
    wheels = [c for c in sig["charts"] if c["key"].startswith("wheel_")]
    assert len({(c["lo"], c["hi"]) for c in wheels}) == 1
    # Every chart is drawn and read on the same columns, and a click opens a real frame.
    frames = json.loads(sig["frames"])
    assert len(frames) == signals.BINS and frames[0] == 0 and frames[-1] == len(ts) - 1
    for c in sig["charts"]:
        for s in json.loads(c["data"])["series"]:
            assert len(s["v"]) == signals.BINS
    # The fixture's two segments tile the scene, and are shaded as such.
    assert len(sig["shade"]) == 2


def test_absent_files_drop_their_charts_rather_than_draw_them_empty(tmp_path):
    access = access_for(tmp_path, with_joints=False, with_cmd_vel=False, with_gravity=False, with_transforms=False)
    ts = access.frame_ts
    sig = signals.signal_charts(access, float(ts[0]), float(ts[-1]))
    assert [c["key"] for c in sig["charts"]] == ["speed", "frame_dt"]
    assert [s["label"] for s in sig["charts"][0]["series"]] == ["ground truth"]


def test_ground_truth_speed_never_steps_across_a_segment_boundary(tmp_path):
    access = access_for(tmp_path)
    seg = access.segments[1]
    t, v = signals._gt_speed(access, seg.start_ts, seg.end_ts)
    assert (t >= seg.start_ts).all() and (t <= seg.end_ts).all()
    # 0.3 m per 0.1 s along x, 0.02 along y: ~3 m/s everywhere, no jump at the edge.
    assert np.allclose(v, np.hypot(3.0, 0.2), atol=1e-3)


def test_both_pages_render_the_charts(client):
    for url in ("/scene/0000", "/scene/0000/segment/1"):
        body = client.get(url).text
        assert "data-signals" in body and "<b>Left wheel speed</b>" in body and "charts.js" in body
        # Every chart folds under its own arrow, and all start open.
        assert body.count('<details class="sig" open') == body.count('<summary class="sighead">') > 0
    assert client.get("/viewer-static/charts.js").status_code == 200


def test_the_time_axis_ticks_at_round_times_on_the_clock_it_reads():
    ticks = signals.time_ticks(447.0)  # a 7 min 27 s scene
    assert [t["label"] for t in ticks] == ["0:00", "2:00", "4:00", "6:00"]
    assert ticks[0]["edge"] == "l" and ticks[0]["pct"] == 0.0
    # A frame page's 30 s window, 97 s into the scene: ticks on the scene's clock.
    ticks = signals.time_ticks(30.0, offset=97.0)
    assert [t["label"] for t in ticks] == ["1:40", "1:45", "1:50", "1:55", "2:00", "2:05"]
    assert abs(ticks[0]["pct"] - 3 / 30 * 100) < 1e-9
    assert [t["label"] for t in signals.time_ticks(12.0)] == ["0 s", "2 s", "4 s", "6 s", "8 s", "10 s", "12 s"]
    assert signals.time_ticks(12.0)[-1]["edge"] == "r"


def test_a_bogus_stamp_does_not_flood_the_time_axis():
    ticks = signals.time_ticks(3.0e8)  # one stamp a decade off
    assert 1 < len(ticks) <= signals.MAX_TICKS + 1
