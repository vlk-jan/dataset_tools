"""Viewer: cutting a scene into windows (dataset_tools.viewer.windows).

Every test builds a synthetic dataset root (``helpers_viewer``), so nothing
here touches a real dataset or the network. The end-to-end HTTP tests skip
cleanly when the viewer extra (SimView + torch) is not installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from helpers_viewer import (
    _write_png,
    make_root,
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
    ts = np.arange(1000, dtype=np.float64)
    segs = [loaders.Segment(0, 0.0, 999.0, 1000, "x")]
    result = windows.windows_for_scene(ts, segs, len(ts), has_transforms=False, max_frames=300)
    chunks = [w for w in result if w.kind == "segment"]
    assert len(chunks) == 4
    assert all(w.n_frames <= 300 for w in chunks)
    # Chunks are contiguous and cover the whole segment.
    assert chunks[0].frame_start == 0
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert a.frame_end == b.frame_start


def test_a_frame_just_before_a_segment_starts_still_belongs_to_it():
    """Segment edges are on the ground-truth pose clock, frames on their own.

    A frame a few ms short of a segment's first pose is that segment's first
    frame; treated strictly it belongs to nothing and the frame page reports no
    ground truth for a frame that plainly has some.
    """
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
    """A root mid-extraction: more images on disk than lines in data_ts.txt.

    Exactly what ``probe`` warns about ("per-frame counts disagree"), and what
    the module docstrings promise the viewer degrades gracefully for.
    """
    root_path = make_root(tmp_path, frames=timestamped, segments=False, with_transforms=False)
    scene = root_path / "0000"
    for i in range(timestamped, timestamped + extra):
        _write_png(scene / "images" / f"{i:05d}.png")
    return root_path


def test_no_window_reaches_past_the_end_of_the_timestamp_file(tmp_path):
    """A frame with no timestamp has no place on the playback clock, so no
    window may contain one -- a window made only of such frames used to be
    offered, built zero frames and died with an IndexError (HTTP 500)."""
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
    assert access.window(windows.frame_window_key(3), 1500) is not None
    assert access.window(windows.frame_window_key(10), 1500) is None


def test_an_untimestamped_frame_is_a_404_and_its_page_says_why(tmp_path):
    """Not a 500: the frame page still renders (the image is on disk), it just
    does not offer a 3D view it cannot build."""
    pytest.importorskip("simview")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from dataset_tools.viewer.app import create_app

    root_path = _root_with_untimestamped_frames(tmp_path)
    c = TestClient(create_app(root_path, cache_dir=tmp_path / "cache"))

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
    from fastapi.testclient import TestClient

    from dataset_tools.viewer.app import create_app

    root_path = _root_with_untimestamped_frames(tmp_path)
    c = TestClient(create_app(root_path, max_frames=4, cache_dir=tmp_path / "cache"))
    offered = c.get("/api/scene/0000").json()["windows"]
    assert offered
    for w in offered:
        assert c.get(f"/view/0000/{w['key']}").status_code == 200, w["key"]


# --------------------------------------------------------------------------
# how densely a segment is anchored, and saying so
# --------------------------------------------------------------------------


def test_a_window_carries_the_pose_count_of_its_segment(tmp_path):
    """Two segments of the same length and frame count can rest on 10 Hz poses
    or on one every two seconds; interpolated onto the frame clock they look
    identical, so the count has to travel with the window."""
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
    ws = windows.windows_for_scene(ts, segs, len(ts), has_transforms=False, max_frames=30)
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
    """A ``frame00042`` window is addressable but never enumerated, so its
    neighbours come from where its frame sits, not from a position in the list:
    paging back from a frame inside a segment opens that whole segment."""
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
