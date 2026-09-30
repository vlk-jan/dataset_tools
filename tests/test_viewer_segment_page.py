"""Viewer: the per-segment page (dataset_tools.viewer.segment_page).

A segment is the unit the ground truth is anchored in, so the page is checked
the way the segment is defined: its frames are the ones its windows carry, its
numbers stop at its own boundaries, and it is addressed by the number in
``odometry_segments.txt`` rather than by its position in the file.

Every test builds a synthetic dataset root (``helpers_viewer``); the end-to-end
HTTP tests skip cleanly when the viewer extra is not installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from helpers_viewer import make_root

from dataset_tools.viewer import probe, segment_page


def access_for(tmp_path: Path, root_path: Path, scene: str = "0000"):
    from dataset_tools.viewer.build import SceneAccess

    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")
    return SceneAccess(root, scene)


def card_of(page, title: str) -> dict | None:
    return next((c for c in page["cards"] if c["title"] == title), None)


def value_of(card, label: str):
    return next((s["value"] for s in card["stats"] if s["label"] == label), None)


# --------------------------------------------------------------------------
# resolution and frame ranges
# --------------------------------------------------------------------------


def test_a_segment_is_found_by_its_own_number_not_its_position(tmp_path):
    """filter_segments renumbers; window keys carry the file's number."""
    root_path = make_root(tmp_path)
    scene = root_path / "0000"
    ts = np.loadtxt(scene / "data_ts.txt")
    (scene / "odometry_segments.txt").write_text(
        "# id start_ts end_ts n_poses reason\n"
        f"4 {ts[0]:.6f} {ts[5]:.6f} 6 filtered:pose_jump\n"
        f"9 {ts[6]:.6f} {ts[-1]:.6f} 6 filtered:end_of_run\n"
    )
    access = access_for(tmp_path, root_path)

    assert segment_page.find(access, 0) is None
    position, seg = segment_page.find(access, 9)
    assert position == 1 and seg.index == 9
    assert segment_page.frame_range(access, position) == (6, 12)


def test_a_segments_frames_are_the_frames_its_windows_carry(tmp_path):
    """The page and the window it links to must not disagree about the range."""
    access = access_for(tmp_path, make_root(tmp_path))
    for position, seg in enumerate(access.segments):
        lo, hi = segment_page.frame_range(access, position)
        windows = segment_page.segment_windows(access, seg, max_frames=1500)
        assert windows, "the fixture's segments are long enough to build"
        assert (windows[0].frame_start, windows[-1].frame_end) == (lo, hi)


# --------------------------------------------------------------------------
# the cards
# --------------------------------------------------------------------------


def test_the_segment_card_states_the_range_the_rate_and_the_gaps(tmp_path):
    access = access_for(tmp_path, make_root(tmp_path))
    position, seg = segment_page.find(access, 1)
    page = segment_page.build_segment_page(access, position, seg, max_frames=1500)

    card = card_of(page, "Segment")
    assert value_of(card, "frames") == "6 – 11"
    assert value_of(card, "GT poses") == "6"
    # The 0.1 s gap to segment 0, not a number out of thin air.
    assert value_of(card, "gap before") == "0.1 s"
    assert "gap after" not in [s["label"] for s in card["stats"]]
    assert "the recording ended" in card["note"]


def test_a_segment_never_covers_more_than_the_whole_scene(tmp_path):
    """Ground truth outlives the synced frames; a share of it cannot exceed 100 %."""
    root_path = make_root(tmp_path)
    scene = root_path / "0000"
    ts = np.loadtxt(scene / "data_ts.txt")
    (scene / "odometry_segments.txt").write_text(
        f"# id start_ts end_ts n_poses reason\n0 {ts[0] - 3.0:.6f} {ts[-1] + 3.0:.6f} 12 filtered:end_of_run\n"
    )
    access = access_for(tmp_path, root_path)
    position, seg = segment_page.find(access, 0)
    page = segment_page.build_segment_page(access, position, seg, max_frames=1500)
    assert value_of(card_of(page, "Segment"), "of the scene") == "100%"


def test_a_thinly_anchored_segment_says_so(tmp_path):
    """Below 2 Hz the trajectory between poses is a straight guess."""
    root_path = make_root(tmp_path)
    scene = root_path / "0000"
    ts = np.loadtxt(scene / "data_ts.txt")
    (scene / "odometry_segments.txt").write_text(
        f"# id start_ts end_ts n_poses reason\n0 {ts[0]:.6f} {ts[-1]:.6f} 2 filtered:end_of_run\n"
    )
    access = access_for(tmp_path, root_path)
    position, seg = segment_page.find(access, 0)
    page = segment_page.build_segment_page(access, position, seg, max_frames=1500)

    card = card_of(page, "Segment")
    assert card["badge_class"] == "lo"
    assert "cuts every corner" in card["note"]


def test_motion_is_measured_inside_the_segment_only(tmp_path):
    """The fixture drives 0.3 m per frame; a six-frame segment is five steps."""
    access = access_for(tmp_path, make_root(tmp_path))
    position, seg = segment_page.find(access, 0)
    page = segment_page.build_segment_page(access, position, seg, max_frames=1500)

    card = card_of(page, "Path & motion")
    assert value_of(card, "path") == "2 m"  # 5 steps of ~0.3 m
    assert value_of(card, "moving") == "100%"
    assert value_of(card, "roll · pitch") == "0.0° · 0.0°"


def test_a_scene_without_ground_truth_poses_still_renders_its_cards(tmp_path):
    root_path = make_root(tmp_path)
    (root_path / "0000" / "odometry.txt").unlink()
    access = access_for(tmp_path, root_path)
    position, seg = segment_page.find(access, 0)
    page = segment_page.build_segment_page(access, position, seg, max_frames=1500)

    assert card_of(page, "Segment") is not None
    motion = card_of(page, "Path & motion")
    assert value_of(motion, "path") is None
    assert value_of(motion, "roll · pitch") is not None  # gravity_align is still there


def test_the_hero_frames_come_from_inside_the_segment(tmp_path):
    access = access_for(tmp_path, make_root(tmp_path))
    position, seg = segment_page.find(access, 1)
    page = segment_page.build_segment_page(access, position, seg, max_frames=1500)

    indices = [int(t["sub"].split()[1]) for t in page["hero"]]
    assert indices == [6, 8, 10]  # 10 / 50 / 90 % of frames 6..11
    assert all("/frame/" in t["href"] for t in page["hero"])


def test_the_map_is_absent_without_gps_and_scoped_to_the_segment_with_it(tmp_path):
    access = access_for(tmp_path, make_root(tmp_path))
    position, seg = segment_page.find(access, 1)
    assert segment_page.build_segment_page(access, position, seg, 1500)["map"] is None

    (tmp_path / "gps").mkdir()
    gps_root = make_root(tmp_path / "gps", with_gps=True)
    access = access_for(tmp_path, gps_root)
    position, seg = segment_page.find(access, 1)
    page = segment_page.build_segment_page(access, position, seg, 1500)
    assert page["map"]["src"].endswith("gps.json?segment=1")
    # The fixes inside frames 6..11, not the scene's 48: the numbers beside the
    # map are the segment's, not just the drawn track.
    assert value_of(page["map"], "fixes") == "22"


# --------------------------------------------------------------------------
# over HTTP
# --------------------------------------------------------------------------


def test_the_segment_page_renders_and_links_where_it_should(client):
    r = client.get("/scene/0000/segment/1")
    assert r.status_code == 200
    assert "segment 1" in r.text
    # Its own windows, its own frames, and the way back to both neighbours.
    assert "/view/0000/seg001" in r.text
    assert "/scene/0000/segment/0" in r.text
    assert "/scene/0000/frame/6" in r.text
    assert "/scene/0000/frame/5" not in r.text  # segment 0's frames


def test_an_unknown_segment_is_a_404(client):
    assert client.get("/scene/0000/segment/7").status_code == 404
    assert client.get("/api/scene/0000/segment/7").status_code == 404
    assert client.get("/scene/9999/segment/0").status_code == 404


def test_the_scene_page_links_every_segment_to_its_own_page(client):
    r = client.get("/scene/0000")
    assert "/scene/0000/segment/0" in r.text and "/scene/0000/segment/1" in r.text


def test_the_frame_page_links_the_segment_the_frame_falls_in(client):
    assert "/scene/0000/segment/0" in client.get("/scene/0000/frame/2").text


def test_the_segment_api_carries_the_range_and_the_windows(client):
    data = client.get("/api/scene/0000/segment/0").json()
    assert data["segment"]["index"] == 0 and data["position"] == 0
    assert data["frames"] == [0, 6]
    assert [w["key"] for w in data["windows"]] == ["seg000"]


def test_only_the_notes_touching_the_segment_are_listed(client):
    client.post("/api/scene/0000/notes", json={"frames": [0, 2], "topic": "odometry", "text": "early wobble"})
    client.post("/api/scene/0000/notes", json={"frames": [9, 11], "topic": "camera", "text": "late dropout"})

    first = client.get("/scene/0000/segment/0").text
    assert "early wobble" in first and "late dropout" not in first
    second = client.get("/scene/0000/segment/1").text
    assert "late dropout" in second and "early wobble" not in second
    # The compose form opens pinned to the segment being looked at.
    assert 'name="frame_start"' in second and 'value="6"' in second


# --------------------------------------------------------------------------
# GNSS against the ground truth
# --------------------------------------------------------------------------


def test_a_gnss_track_that_disagrees_with_the_ground_truth_flags_the_segment(tmp_path):
    """The fixture's fixes walk ~1 m a frame while the odometry drives 0.3 m,
    so the receiver describes three times the path the poses do."""
    access = access_for(tmp_path, make_root(tmp_path, frames=40, with_gps=True))
    position, seg = segment_page.find(access, 0)
    card = card_of(segment_page.build_segment_page(access, position, seg, 1500), "Path & motion")
    track = next(s for s in card["stats"] if s["label"] == "GNSS track")
    assert track["sub"].endswith("% of the GT path")
    assert "disagree" in card["note"]


def test_a_gnss_track_that_matches_the_ground_truth_is_a_quiet_check(tmp_path):
    root_path = make_root(tmp_path, frames=40, with_gps=True)
    scene = root_path / "0000"
    # Fixes spaced exactly as the odometry drives: 0.3 m per frame, due north.
    ts = np.loadtxt(scene / "data_ts.txt")
    gps = np.zeros((len(ts), 8))
    gps[:, 0] = ts
    gps[:, 1] = 50.109 + 0.3 / 111_320.0 * np.arange(len(ts))
    gps[:, 2] = 14.418
    gps[:, 4] = 2
    np.save(scene / "gps.npy", gps)
    access = access_for(tmp_path, root_path)
    position, seg = segment_page.find(access, 0)
    card = card_of(segment_page.build_segment_page(access, position, seg, 1500), "Path & motion")
    assert value_of(card, "GNSS track") == value_of(card, "path")
    assert "disagree" not in card["note"]
    assert "not ground truth" in card["note"]


def test_a_segment_without_fixes_says_nothing_about_gnss(tmp_path):
    access = access_for(tmp_path, make_root(tmp_path))
    position, seg = segment_page.find(access, 0)
    card = card_of(segment_page.build_segment_page(access, position, seg, 1500), "Path & motion")
    assert value_of(card, "GNSS track") is None
