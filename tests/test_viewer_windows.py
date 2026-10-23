"""Viewer: cutting a scene into windows (dataset_tools.viewer.windows), on synthetic roots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from helpers_viewer import (
    make_root,
    viewer_client,
    write_image,
)

from dataset_tools.viewer import loaders, probe, windows

# --------------------------------------------------------------------------
# windowing
# --------------------------------------------------------------------------


def test_windows_never_span_two_segments():
    ts = np.arange(100, dtype=np.float64)
    segs = [loaders.Segment(0, 0.0, 39.0, 40, "a"), loaders.Segment(1, 60.0, 99.0, 40, "b")]
    result = windows.windows_for_scene(ts, segs, len(ts), has_transforms=True)
    seg_windows = [w for w in result if w.kind == "segment"]
    assert len(seg_windows) == 2
    # The gap between the segments belongs to neither window.
    assert seg_windows[0].frame_end <= 40
    assert seg_windows[1].frame_start >= 60


def test_long_segments_are_split_into_even_chunks():
    """On a uniform clock the two divisions coincide: even in time is even in
    frames, and the boundary lands in the same place either way."""
    ts = np.arange(1000, dtype=np.float64)
    segs = [loaders.Segment(0, 0.0, 999.0, 1000, "x")]
    result = windows.windows_for_scene(ts, segs, len(ts), has_transforms=False, max_frames=300, max_seconds=1e9)
    chunks = [w for w in result if w.kind == "segment"]
    assert len(chunks) == 4
    assert all(w.n_frames <= 300 for w in chunks)
    # Chunks are contiguous and cover the whole segment.
    assert chunks[0].frame_start == 0
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert a.frame_end == b.frame_start


def test_the_parts_of_a_split_segment_are_equal_in_time_not_in_frame_count():
    """The cap is in seconds, so the split is by time, not by index range."""
    # 200 s at 10 Hz, but only every fourth frame survives the first 100 s.
    ts = np.concatenate([np.arange(0, 100, 0.4), np.arange(100, 200, 0.1)])
    segs = [loaders.Segment(0, float(ts[0]), float(ts[-1]), len(ts), "x")]
    ws = windows.windows_for_scene(ts, segs, len(ts), has_transforms=False, max_seconds=100.0)

    assert len(ws) == 2
    assert all(w.duration <= 100.0 for w in ws)
    assert abs(ws[0].duration - ws[1].duration) < 1.0
    # Which is emphatically not an even split of the frames.
    assert ws[0].n_frames * 3 < ws[1].n_frames


def test_a_hole_in_the_clock_never_makes_an_empty_window():
    """A time slice can contain no frame at all when the clock has a hole wider
    than the slice. Every part still has to be something the viewer can build."""
    ts = np.concatenate([np.arange(0, 10, 0.1), np.arange(600, 610, 0.1)])
    segs = [loaders.Segment(0, float(ts[0]), float(ts[-1]), len(ts), "x")]
    ws = windows.windows_for_scene(ts, segs, len(ts), has_transforms=False, max_seconds=100.0)

    assert [w.n_frames for w in ws] == [100, 100]  # the five cuts inside the hole are one cut
    assert sum(w.n_frames for w in ws) == len(ts)


def test_a_hole_in_the_clock_does_not_shatter_a_segment_into_one_frame_windows():
    """A camera outage inside a segment (v2_4 0030/0032/0033) used to turn every
    equal-time cut inside it into a one-frame window."""
    ts = np.concatenate([np.arange(3000) * 0.1, 1800.0 + np.arange(3000) * 0.1])
    segs = [loaders.Segment(0, float(ts[0]), float(ts[-1]), len(ts), "x")]
    ws = windows.windows_for_scene(ts, segs, len(ts), has_transforms=False, max_frames=1500, max_seconds=600.0)

    assert [w.n_frames for w in ws] == [1500, 1500, 1500, 1500]
    assert all(w.n_frames >= windows.MIN_WINDOW_FRAMES for w in ws)


def test_the_frame_ceiling_survives_a_division_by_time():
    """A frame burst raises the piece count until every part is under the frame ceiling."""
    # 300 s: sparse for the first 200, then 100 s of 100 Hz.
    ts = np.concatenate([np.arange(0, 200, 1.0), np.arange(200, 300, 0.01)])
    segs = [loaders.Segment(0, float(ts[0]), float(ts[-1]), len(ts), "x")]
    ws = windows.windows_for_scene(ts, segs, len(ts), has_transforms=False, max_seconds=150.0, max_frames=4000)

    assert len(ws) > 2  # the seconds cap alone asked for two
    assert all(w.n_frames <= 4000 for w in ws)
    assert all(w.duration <= 150.0 for w in ws)


def test_two_scenes_at_different_frame_rates_are_cut_at_the_same_length():
    """The cap counts seconds, not frames, so 3.2 Hz and 10 Hz scenes cut alike."""
    cut = []
    for hz in (3.2, 10.0):
        n = int(round(400 * hz))  # 400 s of recording, either way
        ts = np.arange(n, dtype=np.float64) / hz
        segs = [loaders.Segment(0, float(ts[0]), float(ts[-1]), n, "x")]
        ws = windows.windows_for_scene(ts, segs, n, has_transforms=False, max_seconds=150.0)
        assert all(w.duration <= 150.0 + 1.0 for w in ws)
        cut.append([round(w.duration) for w in ws])
    assert cut[0] == cut[1] == [133, 133, 133]


def test_a_segment_inside_the_cap_is_never_cut():
    """A 462 s segment at 3.2 Hz -- 1479 frames, over the old 1500-frame cap
    once the boundary slack is counted -- stays one window."""
    ts = np.arange(1479, dtype=np.float64) / 3.2
    segs = [loaders.Segment(0, float(ts[0]), float(ts[-1]), 1479, "x")]
    ws = windows.windows_for_scene(ts, segs, len(ts), has_transforms=False)
    assert len(ws) == 1 and ws[0].n_chunks == 1
    assert ws[0].n_frames == 1479


def test_the_frame_ceiling_still_bites_on_a_fast_clock():
    """The frame count backstops the seconds cap for very fast recordings."""
    ts = np.arange(20_000, dtype=np.float64) / 200.0  # 200 Hz, 100 s in total
    segs = [loaders.Segment(0, float(ts[0]), float(ts[-1]), 20_000, "x")]
    ws = windows.windows_for_scene(ts, segs, len(ts), has_transforms=False, max_frames=6000)
    assert len(ws) == 4  # the seconds cap alone would have left it whole
    assert all(w.n_frames <= 6000 for w in ws)


def test_a_scene_with_no_clock_is_cut_on_frames_alone():
    """Without ``data_ts.txt`` the fallback window uses the frame ceiling."""
    ws = windows.windows_for_scene(None, [], 900, has_transforms=True, max_frames=300)
    assert [w.n_frames for w in ws] == [300, 300, 300]


def test_a_frame_just_before_a_segment_starts_still_belongs_to_it():
    """A frame a few ms before a segment's first pose still belongs to that segment."""
    ts = np.arange(20, dtype=np.float64) * 0.32
    # The segment opens 3 ms after frame 5 -- a hundredth of a frame period.
    segs = [loaders.Segment(0, float(ts[5]) + 0.003, float(ts[15]), 30, "a")]
    assert windows.segment_of_frame(ts, segs, 5) is segs[0]
    seg_window = [w for w in windows.windows_for_scene(ts, segs, len(ts), False) if w.kind == "segment"][0]
    assert seg_window.frame_start == 5
    # A frame a whole period early is genuinely outside it.
    assert windows.segment_of_frame(ts, segs, 4) is None


def test_the_boundary_slack_never_gives_one_frame_to_two_segments():
    """Segments share no reference frame, so a frame in both would be in neither."""
    ts = np.arange(40, dtype=np.float64) * 0.32
    # Two segments whose gap is narrower than the slack a frame period allows.
    segs = [
        loaders.Segment(0, float(ts[2]), float(ts[10]) + 0.02, 20, "a"),
        loaders.Segment(1, float(ts[10]) + 0.08, float(ts[20]), 20, "b"),
    ]
    owners = [windows.segment_of_frame(ts, segs, i) for i in range(len(ts))]
    assert all(o is None or o in segs for o in owners)
    lo, hi = windows.segment_bounds(ts, segs)
    assert lo[1] <= hi[0]  # the widened spans meet, they do not overlap
    seg_windows = [w for w in windows.windows_for_scene(ts, segs, len(ts), False) if w.kind == "segment"]
    assert seg_windows[0].frame_end <= seg_windows[1].frame_start


def test_whole_scene_window_reports_its_pose_source():
    ts = np.arange(50, dtype=np.float64)
    with_tf = windows.windows_for_scene(ts, [], 50, has_transforms=True)
    assert with_tf[0].pose_source == "transforms"
    without = windows.windows_for_scene(ts, [], 50, has_transforms=False)
    assert without[0].pose_source == "none"


# --------------------------------------------------------------------------
# single-frame windows
# --------------------------------------------------------------------------


def test_a_frame_window_key_round_trips():
    assert windows.parse_frame_window_key(windows.frame_window_key(42)) == 42
    assert windows.parse_frame_window_key("seg001") is None
    assert windows.parse_frame_window_key("framex") is None
    assert windows.parse_frame_window_key("frame⑤") is None  # isdigit, yet not an int
    assert windows.parse_frame_window_key("frame" + "9" * 5000) is None  # past int()'s digit limit


def test_a_single_frame_window_takes_its_pose_from_the_segment_it_is_in():
    ts = np.arange(20, dtype=np.float64)
    segs = [loaders.Segment(0, 0.0, 5.0, 6, "a")]
    inside = windows.single_frame_window(ts, segs, 20, 3)
    assert inside.pose_source == "odometry" and inside.segment_index == 0
    assert inside.frame_start == 3 and inside.n_frames == 1
    outside = windows.single_frame_window(ts, segs, 20, 12)
    assert outside.pose_source == "none" and outside.segment_index is None
    assert "robot-centric" in outside.reason
    assert windows.single_frame_window(ts, segs, 20, 99) is None


# --------------------------------------------------------------------------
# a scene whose per-frame directories outrun its timestamp file
# --------------------------------------------------------------------------


def _root_with_untimestamped_frames(tmp_path: Path, timestamped: int = 8, extra: int = 6) -> Path:
    """A root mid-extraction: more images on disk than lines in data_ts.txt."""
    root_path = make_root(tmp_path, frames=timestamped, segments=False, with_transforms=False)
    scene = root_path / "0000"
    for i in range(timestamped, timestamped + extra):
        write_image(scene / "images" / f"{i:05d}.png")
    return root_path


def test_no_window_reaches_past_the_end_of_the_timestamp_file(tmp_path):
    """No window may contain a frame without a timestamp (used to 500)."""
    root = probe.DatasetRoot(_root_with_untimestamped_frames(tmp_path), cache_dir=tmp_path / "c")
    p = root.probe("0000")
    assert p.n_frames == 14 and p.counts["data_ts"] == 8  # the disagreement
    assert any("counts disagree" in w for w in p.warnings)

    ws = windows.windows_for_scene(np.arange(8, dtype=np.float64), [], p.n_frames, has_transforms=False, max_frames=4)
    assert ws, "the scene still offers windows"
    assert max(w.frame_end for w in ws) == 8


def test_a_frame_with_no_timestamp_has_no_single_frame_window(tmp_path):
    root = probe.DatasetRoot(_root_with_untimestamped_frames(tmp_path), cache_dir=tmp_path / "c")
    from dataset_tools.viewer.build import SceneAccess

    access = SceneAccess(root, "0000")
    assert access.window(windows.frame_window_key(3)) is not None
    assert access.window(windows.frame_window_key(10)) is None


def test_an_untimestamped_frame_is_a_404_and_its_page_says_why(tmp_path):
    """Not a 500: the frame page still renders (the image is on disk), it just
    does not offer a 3D view it cannot build."""
    pytest.importorskip("simview")
    pytest.importorskip("fastapi")

    root_path = _root_with_untimestamped_frames(tmp_path)
    c = viewer_client(root_path, tmp_path / "cache")

    page = c.get("/scene/0000/frame/10")
    assert page.status_code == 200
    assert "Visualise this frame" not in page.text
    assert "past the end of" in page.text
    assert c.get("/view/0000/frame00010").status_code == 404

    ok = c.get("/scene/0000/frame/3")
    assert "Visualise this frame" in ok.text
    assert c.get("/view/0000/frame00003").status_code == 200


def test_every_offered_window_actually_builds(tmp_path):
    """The windows table is a list of buttons; every one of them must work."""
    pytest.importorskip("simview")
    pytest.importorskip("torch")

    from dataset_tools.viewer.build import SceneAccess
    from dataset_tools.viewer.probe import DatasetRoot

    root_path = _root_with_untimestamped_frames(tmp_path)
    c = viewer_client(root_path, tmp_path / "cache", max_frames=4)
    offered = SceneAccess(DatasetRoot(root_path, cache_dir=tmp_path / "cache"), "0000", max_frames=4).windows()
    assert offered
    for w in offered:
        assert c.get(f"/view/0000/{w.key}").status_code == 200, w.key


def test_a_cached_bundle_is_not_served_for_a_window_that_has_been_recut(tmp_path):
    """A window key like ``raw.0`` means different frames under another cap,
    so the cache key must include the range."""
    pytest.importorskip("simview")
    pytest.importorskip("torch")

    from dataset_tools.viewer.build import BuildOptions

    root_path = make_root(tmp_path, frames=8, segments=False, with_transforms=True)
    cache = tmp_path / "cache"
    url = f"/api/data/0000/raw.0/{BuildOptions().cache_key()}/stamp/meta.json"

    first = viewer_client(root_path, cache, max_frames=2).get(url)
    assert first.json()["window"]["frame_end"] == 2

    # Same key, same cache, a cap that cuts the scene differently.
    second = viewer_client(root_path, cache, max_frames=3).get(url)
    assert second.json()["window"]["frame_end"] == 3
    assert second.json()["frames"] == [0, 1, 2]


# --------------------------------------------------------------------------
# how densely a segment is anchored, and saying so
# --------------------------------------------------------------------------


def test_a_window_carries_the_pose_count_of_its_segment(tmp_path):
    """The ground-truth pose count travels with the window; interpolation hides it."""
    ts = np.arange(40, dtype=np.float64) * 0.1
    segs = [
        loaders.Segment(0, ts[0], ts[19], n_poses=200, reason="dense"),
        loaders.Segment(1, ts[20], ts[-1], n_poses=2, reason="sparse"),
    ]
    ws = windows.windows_for_scene(ts, segs, len(ts), has_transforms=False)
    dense, sparse = ws[0], ws[1]
    assert dense.n_poses == 200 and sparse.n_poses == 2
    assert dense.pose_rate > 100 and sparse.pose_rate < 2


def test_a_split_segment_keeps_its_pose_rate(tmp_path):
    """Chunking a long segment must not make it look sparser than it is."""
    ts = np.arange(100, dtype=np.float64) * 0.1
    segs = [loaders.Segment(0, ts[0], ts[-1], n_poses=100, reason="x")]
    ws = windows.windows_for_scene(ts, segs, len(ts), has_transforms=False, max_frames=30, max_seconds=1e9)
    assert len(ws) == 4
    whole = windows.windows_for_scene(ts, segs, len(ts), has_transforms=False)[0]
    for w in ws:
        assert abs(w.pose_rate - whole.pose_rate) < 0.15 * whole.pose_rate


def test_neighbour_windows_walk_a_scene_in_frame_order():
    """The panel's < / > page through a scene one window at a time, and stop
    at both ends rather than wrapping."""
    ts = np.arange(100, dtype=np.float64)
    segs = [loaders.Segment(0, 0.0, 39.0, 40, "a"), loaders.Segment(1, 60.0, 99.0, 40, "b")]
    ws = windows.windows_for_scene(ts, segs, len(ts), has_transforms=True)

    assert windows.neighbour_windows(ws, ws[0]) == (None, ws[1])
    assert windows.neighbour_windows(ws, ws[1]) == (ws[0], None)


def test_a_single_frame_window_pages_into_the_segment_holding_it():
    """A ``frame00042`` window's neighbours come from where its frame sits."""
    ts = np.arange(100, dtype=np.float64)
    segs = [loaders.Segment(0, 0.0, 39.0, 40, "a"), loaders.Segment(1, 60.0, 99.0, 40, "b")]
    ws = windows.windows_for_scene(ts, segs, len(ts), has_transforms=True)

    inside = windows.single_frame_window(ts, segs, len(ts), 70)
    assert windows.neighbour_windows(ws, inside) == (ws[1], None)

    # A frame in the gap between two segments belongs to neither, and pages to
    # the one either side of it.
    between = windows.single_frame_window(ts, segs, len(ts), 50)
    assert windows.neighbour_windows(ws, between) == (ws[0], ws[1])


def test_the_windows_table_states_the_ground_truth_pose_rate(client):
    body = client.get("/scene/0000?tab=windows").text
    assert "GT poses" in body
    assert "Hz" in body


def test_a_reason_tag_is_translated_tag_by_tag():
    """Reasons accumulate tags; none of them may reach a page as raw jargon."""
    d = windows.describe_reason
    assert d("filtered:pose_jump").startswith("the GT pose jumped")
    # Where the poses came from reads as "filled ...", which is what the pages
    # branch on to avoid saying "ends because filled".
    assert d("v11_fill").startswith("filled from v1.1 ground truth")
    assert d("regen_fill:t194").startswith("filled by replaying")
    # A "+" tail says what was done to the poses afterwards.
    assert d("regen_fill:t194+refined").endswith("(refined against the wheel odometry)")
    assert "spliced from the earlier root" in d("filtered:gap+coverage_guard")
    assert d("filtered:gap+coverage_guard").startswith("a gap of over 1 s")
    # A filled stretch that also carries why it ended says both.
    assert "ends because the recording ended" in d("v11_fill:end_of_run")
    # An unknown tag is shown as it is rather than dropped.
    assert d("something_new") == "something_new"
    assert d("") == ""


def test_a_joined_segment_says_what_was_joined_and_out_of_what():
    """v1.5 joins pieces across a proven standstill; the tag names the sources."""
    d = windows.describe_reason
    assert "joined across a standstill" in d("filtered:gap+joined")
    # Pieces from different estimators: the reason keeps only the last piece's
    # body, so the mark is where a reader learns the segment is a composite.
    both = d("filtered:pose_jump+joined:native,v11_fill")
    assert "this root's own estimator run and the v1.1 fill" in both
    assert "native" not in both and "v11_fill" not in both
    assert windows.describe_marks("regen_fill:t194+refined+joined") == [
        "refined against the wheel odometry",
        "joined across a standstill the wheels, the live odometry and the IMU all confirm",
    ]
    # The scene card counts marks across a scene, so it asks for the short form:
    # the sources vary segment to segment and would splinter every count.
    assert windows.describe_marks("filtered:gap+joined:native,v11_fill", brief=True) == ["joined across a standstill"]
