"""Viewer: the scene page's cards (dataset_tools.viewer.scene_page)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from helpers_viewer import (
    CAMERA_INFO,
    CURATED_THUMBS,
    card_of,
    make_root,
    scene_access,
    value_of,
    viewer_client,
    write_atlas,
    write_image,
)

from dataset_tools.viewer import probe
from dataset_tools.viewer.probe import ARTEFACTS
from dataset_tools.viewer.scene_page import frame_gaps

# --------------------------------------------------------------------------
# the scene page: four cards assembled from probe + atlas + a few small files
# --------------------------------------------------------------------------


def scene_page(tmp_path, root_path, scene: str = "0000"):
    from dataset_tools.viewer.scene_page import build_scene_page

    access = scene_access(tmp_path, root_path, scene)
    return build_scene_page(access.root, access)


def sub_of(card, label: str):
    return next((s["sub"] for s in card["stats"] if s["label"] == label), None)


def files_of(page) -> list[dict]:
    """The file card's entries, across its groups."""
    return [f for g in page["file_groups"] for f in g["entries"]]


def test_camera_info_reads_the_fields_the_page_shows(tmp_path):
    from dataset_tools.viewer.loaders import read_camera_info

    path = tmp_path / "camera_info.yaml"
    path.write_text(CAMERA_INFO)
    info = read_camera_info(path)
    assert info["image_width"] == 640 and info["image_height"] == 400
    assert info["distortion_model"] == "plumb_bob"
    assert info["frame_id"] == "test_camera_optical_frame"
    # A nested block collapses to its data list plus its declared shape.
    assert len(info["distortion_coefficients"]) == 5
    assert info["camera_matrix_rows"] == 3
    assert info["camera_matrix"][0] == 700.0


def test_camera_info_of_a_missing_or_malformed_file_is_simply_absent(tmp_path):
    from dataset_tools.viewer.loaders import read_camera_info

    assert read_camera_info(tmp_path / "nope.yaml") is None
    bad = tmp_path / "bad.yaml"
    bad.write_text("!!! not: [a, yaml\nfile at all\n")
    read_camera_info(bad)  # must not raise; whatever it salvages is fine


def test_the_proprioception_card_reports_each_files_own_count_and_rate(tmp_path):
    root_path = make_root(tmp_path)
    arr = np.zeros((10_000, 7))
    arr[:, 0] = 1_700_000_000.0 + 0.01 * np.arange(10_000)  # 100 Hz
    arr[:, 1] = -2.0
    np.save(root_path / "0000" / "cmd_vel.npy", arr)

    proprio = card_of(scene_page(tmp_path, root_path), "Proprioception")
    assert value_of(proprio, "cmd_vel") == "10,000 × 7"
    assert sub_of(proprio, "cmd_vel") == "100.0 Hz"
    assert value_of(proprio, "commanded speed") == "2.00 m/s"


def test_the_four_cards_state_what_the_scene_has(tmp_path):
    root_path = make_root(tmp_path)
    write_atlas(root_path)
    page = scene_page(tmp_path, root_path)
    assert [c["title"] for c in page["cards"]] == [
        "Recording",
        "Ground truth",
        "Calibration",
        "Proprioception",
    ]
    # No card is a single entry: that was the point of reworking them.
    assert all(len(c["stats"]) >= 2 for c in page["cards"])

    recording = card_of(page, "Recording")
    assert value_of(recording, "synced pairs") == "12"
    assert value_of(recording, "frame rate") == "10.00 Hz"
    assert value_of(recording, "image size") == "640 × 400"
    assert "25.1 GB" in recording["note"] and "synthetic_session" in recording["note"]

    calib = card_of(page, "Calibration")
    assert value_of(calib, "focal length") == "700.0 · 700.0"
    assert value_of(calib, "camera → base") == "0.000, 0.000, 0.000"
    assert "plumb_bob" in sub_of(calib, "image")
    assert [m["label"] for m in calib["matrices"]] == [
        "camera_intrinsics.txt (K)",
        "T_cam_to_base.txt (4×4)",
    ]


def test_wheel_speeds_keep_the_units_the_file_carries(tmp_path):
    """No rig constant: a wheel radius belongs to one robot, rad/s to all of them."""
    page = scene_page(tmp_path, make_root(tmp_path))
    proprio = card_of(page, "Proprioception")
    assert value_of(proprio, "wheel speed") == "1.50 rad/s"  # the fixture's velocity
    assert value_of(proprio, "wheels turning") == "100%"
    assert value_of(proprio, "commanded speed") == "0.40 m/s"
    assert value_of(proprio, "roll · pitch") == "0.0° · 0.0°"  # identity gravity align


def test_a_card_with_nothing_to_say_is_dropped_rather_than_shown_empty(tmp_path):
    root_path = make_root(tmp_path, with_joints=False, with_cmd_vel=False, with_gravity=False)
    titles = [c["title"] for c in scene_page(tmp_path, root_path)["cards"]]
    assert "Proprioception" not in titles
    assert "Recording" in titles and "Ground truth" in titles


def test_the_ground_truth_card_reads_the_segment_file_over_the_atlas(tmp_path):
    """The file is the truth: a scene may have been re-filtered since curation."""
    root_path = make_root(tmp_path)
    write_atlas(root_path)  # claims nine segments
    card = card_of(scene_page(tmp_path, root_path), "Ground truth")
    assert value_of(card, "segments") == 2  # what the file actually holds
    assert card["badge"].endswith("GT")
    assert card["timeline"]["spans"]


def test_every_timeline_block_opens_its_own_segment(tmp_path):
    """A block on the ground-truth timeline is a link to the segment it draws."""
    root_path = make_root(tmp_path)
    scene = root_path / "0000"
    ts = np.loadtxt(scene / "data_ts.txt")
    with open(scene / "odometry_segments.txt", "w") as fh:
        # Numbered 3 and 7: segments are addressed by their own number, not position.
        fh.write(f"3 {ts[0]:.6f} {ts[5]:.6f} 6 teleport\n")
        fh.write(f"7 {ts[6]:.6f} {ts[-1]:.6f} 6 end_of_run\n")
    marks = card_of(scene_page(tmp_path, root_path), "Ground truth")["timeline"]["marks"]
    assert [m["index"] for m in marks] == [3, 7]
    assert [m["href"] for m in marks] == ["/scene/0000/segment/3", "/scene/0000/segment/7"]
    # The tooltip describes the block but not why the segment ended.
    assert "segment 3" in marks[0]["title"] and "6 GT poses" in marks[0]["title"]
    assert "ends because" not in marks[0]["title"]


def test_a_segment_too_short_to_see_is_still_wide_enough_to_click(tmp_path):
    from dataset_tools.viewer.scene_page import MIN_MARK_FRAC

    root_path = make_root(tmp_path)
    scene = root_path / "0000"
    ts = np.loadtxt(scene / "data_ts.txt")
    with open(scene / "odometry_segments.txt", "w") as fh:
        fh.write(f"0 {ts[0]:.6f} {ts[0] + 0.001:.6f} 2 gap\n")
        fh.write(f"1 {ts[6]:.6f} {ts[-1]:.6f} 6 end_of_run\n")
    marks = card_of(scene_page(tmp_path, root_path), "Ground truth")["timeline"]["marks"]
    assert marks[0]["width"] == pytest.approx(MIN_MARK_FRAC * 100)
    assert marks[-1]["left"] + marks[-1]["width"] <= 100.0


def test_a_scene_with_no_segment_file_still_draws_its_curated_timeline(tmp_path):
    """No segments to link to is not a reason to lose the timeline."""
    root_path = make_root(tmp_path)
    (root_path / "0000" / "odometry_segments.txt").unlink()
    atlas = write_atlas(root_path) / "atlas.json"
    data = json.loads(atlas.read_text())
    data["scenes"]["0000"]["seg_spans"] = [[0.0, 0.4], [0.6, 1.0]]
    atlas.write_text(json.dumps(data))
    timeline = card_of(scene_page(tmp_path, root_path), "Ground truth")["timeline"]
    assert timeline["marks"] == []
    assert len(timeline["spans"]) == 2


def test_the_file_inventory_says_what_each_artefact_is_for(tmp_path):
    page = scene_page(tmp_path, make_root(tmp_path, with_cmd_vel=False, with_transforms=False))
    entries = {f["key"]: f for f in files_of(page)}
    assert entries["odometry"]["state"] == "yes" and "ground truth" in entries["odometry"]["purpose"].lower()
    assert entries["odometry"]["relpath"] == "odometry.txt"
    assert entries["images"]["relpath"] == "images" and entries["images"]["group"] == "Per frame"
    # An absent file keeps its entry: the page says what the scene is missing,
    # and tells a missing file from one that is only written sometimes.
    assert entries["transforms"]["state"] == "no" and entries["transforms"]["purpose"]
    assert entries["tf_pins"]["state"] == "optional"
    # Teleoperated scenes have no cmd_vel: normal, not missing.
    assert entries["cmd_vel"]["state"] == "optional"
    assert page["files_summary"]["badge"].endswith("missing")
    assert "transforms.npy" in page["files_summary"]["lines"][0]
    assert "cmd_vel.npy" not in page["files_summary"]["lines"][0]
    assert "tf_pins.txt" not in page["files_summary"]["lines"][0]
    # Per-frame things are counted against the frame count, not shown bare.
    assert entries["images"]["stats"][0]["label"] == "frames"
    assert entries["images"]["stats"][0]["sub"] == "every frame"
    # Every artefact is in exactly one group.
    assert sorted(entries) == sorted(a.key for a in ARTEFACTS)


def test_frame_gaps_are_runs_of_missing_indices():
    assert frame_gaps({0, 1, 4, 6}, 8) == [(2, 3), (5, 5), (7, 7)]
    assert frame_gaps({0, 1}, 2) == []


def test_path_length_is_derived_from_odometry_when_no_atlas_supplies_it(tmp_path):
    """Rig-agnostic fallback: distance inside segments, never across them."""
    page = scene_page(tmp_path, make_root(tmp_path))  # no atlas/
    card = card_of(page, "Ground truth")
    # Two segments of six poses, each step 0.3 m forward and 0.02 m sideways.
    assert value_of(card, "path in GT") == "3 m"
    assert sub_of(card, "path in GT") == "from odometry.txt"
    assert value_of(card, "moving") == "100%"


def test_the_hero_thumbnails_are_the_index_slots_and_link_to_their_frames(tmp_path):
    page = scene_page(tmp_path, make_root(tmp_path))
    assert [t["label"] for t in page["hero"]] == ["10%", "50%", "90%"]
    assert [t["url"] for t in page["hero"]] == [f"/api/scene/0000/thumb/{slot}.jpg" for slot in ("a", "b", "c")]
    assert [t["href"] for t in page["hero"]] == [
        "/scene/0000/frame/1",
        "/scene/0000/frame/6",
        "/scene/0000/frame/10",
    ]


def test_a_curated_hero_thumbnail_links_to_the_frame_it_was_cut_from(tmp_path):
    """Thumbnail picture and link come from the atlas's frame number, not the fraction."""
    root_path = make_root(tmp_path)
    write_atlas(root_path)
    page = scene_page(tmp_path, root_path)
    assert [t["href"] for t in page["hero"]] == [f"/scene/0000/frame/{i}" for i in CURATED_THUMBS.values()]
    assert [t["sub"].split(" ·")[0] for t in page["hero"]] == [f"frame {i}" for i in CURATED_THUMBS.values()]


def test_a_curated_thumbnail_with_no_stored_file_renders_the_frame_it_names(tmp_path):
    """The stored JPEG is a cache of the frame, not the definition of it: delete
    it and the endpoint renders exactly the same frame at the same width."""
    pytest.importorskip("simview")
    pytest.importorskip("torch")

    root_path = make_root(tmp_path)
    atlas_dir = write_atlas(root_path)
    for stored in (atlas_dir / "thumbs").glob("*.jpg"):
        stored.unlink()
    # Mark the frame the 50 % fraction would pick, so the two rules differ.
    write_image(root_path / "0000" / "images" / "00006.png", rgb=(200, 30, 30))

    client = viewer_client(root_path, tmp_path / "cache")
    thumb = client.get("/api/scene/0000/thumb/b.jpg")
    assert thumb.status_code == 200
    frame = client.get(f"/api/scene/0000/frame/{CURATED_THUMBS['b']}/image.jpg?w=384")
    assert thumb.content == frame.content
    assert thumb.content != client.get("/api/scene/0000/frame/6/image.jpg?w=384").content


def test_the_scene_page_leads_with_its_thumbnails_and_the_four_cards(atlas_client):
    r = atlas_client.get("/scene/0000")
    assert r.status_code == 200
    body = r.text
    for slot in ("a", "b", "c"):
        assert f"/api/scene/0000/thumb/{slot}.jpg" in body
    for title in ("Recording", "Ground truth", "Calibration", "Proprioception"):
        assert f"<h2>{title}</h2>" in body
    # The thumbnails come before the cards, which come before the tabs.
    assert body.index('class="hero"') < body.index('class="grid cards"') < body.index('class="tabs"')
    # And the absent-artefact chips survived the rework.
    assert 'class="chip no"' in body and "joint_setpoints.npy" in body


def test_the_rendered_timeline_links_into_the_segment_pages(atlas_client):
    body = atlas_client.get("/scene/0000").text
    bar = body[body.index('class="bar"') : body.index('class="barcap"')]
    assert bar.count('class="seg') == 2  # the fixture's two segments
    assert "/scene/0000/segment/0" in bar and "/scene/0000/segment/1" in bar


def test_the_rendered_file_strip_opens_each_artefact_in_place(atlas_client):
    body = atlas_client.get("/scene/0000").text
    # Each chip opens the panel that describes it as a native popover, and the
    # panel is rendered by the server.
    assert 'data-file="odometry"' in body and 'popovertarget="fileinfo-odometry"' in body
    assert 'id="fileinfo-odometry" popover' in body
    assert "re-anchored to the identity at its own first pose" in body
    # Absent files keep their entry and say so.
    assert 'data-file="gps"' in body and 'class="chip no"' in body


def test_a_scene_page_on_a_root_with_almost_nothing_still_renders(tmp_path):
    pytest.importorskip("simview")
    pytest.importorskip("torch")

    root_path = make_root(
        tmp_path,
        with_height_maps=False,
        with_cmd_vel=False,
        with_transforms=False,
        with_gravity=False,
        with_joints=False,
        with_camera_info=False,
    )
    client = viewer_client(root_path, tmp_path / "cache")
    body = client.get("/scene/0000").text
    assert "<h2>Recording</h2>" in body and "<h2>Ground truth</h2>" in body
    assert "<h2>Proprioception</h2>" not in body  # nothing to fill it
    assert "field of view" not in body  # no camera_info, so no image size to use
    assert 'class="chip no"' in body


def test_the_index_and_the_scene_page_share_one_palette():
    """The index may restyle its *type*; its colours are the viewer's colours."""
    import dataset_tools.viewer as viewer  # the package, not the FastAPI app

    css = (Path(viewer.__file__).parent / "static" / "style.css").read_text()
    index_scope = "".join(re.findall(r"body\.atlas[^{}()]*\{([^}]*)\}", css))
    for token in ("--bg", "--panel", "--fg", "--accent", "--ok", "--warn", "--line"):
        assert token not in index_scope, f"{token} is redefined for the index"
    assert "--display" in index_scope  # only the type stacks are the index's own
    assert '@import url("theme.css")' in css
    # The palette is theme.css's: every hue a light-dark() pair on :root, so
    # no token leaks from one theme into the other.
    theme = (Path(viewer.__file__).parent / "static" / "theme.css").read_text()
    # So are the type stacks and scales, which the 3D panel shares.
    assert "--mono:" in theme and "--fs-sm:" in theme and "--sp-3:" in theme
    palette = theme[theme.index(":root {") :].split("}", 1)[0]
    for token in ("--bg", "--panel", "--fg", "--accent", "--ok", "--warn", "--line", "--track", "--shade"):
        assert f"{token}: light-dark(" in palette, token
    assert ':root[data-theme="dark"]' in theme and ':root[data-theme="light"]' in theme
    assert "--ok-soft" in theme


def test_a_measured_scene_gets_a_disk_card_and_sized_files(tmp_path):
    from dataset_tools import disk_usage

    root_path = make_root(tmp_path)
    page = scene_page(tmp_path, root_path)
    assert page["disk"] is None
    assert all(value_of(f, "on disk") is None for f in files_of(page))

    doc = disk_usage.update(root_path)
    page = scene_page(tmp_path / "again", root_path)
    card = page["disk"]
    entry = doc["scenes"]["0000"]
    assert [p["bytes"] for p in card["parts"]] == sorted((p["bytes"] for p in card["parts"]), reverse=True)
    assert sum(p["bytes"] for p in card["parts"]) == entry["bytes"]
    assert "images" in sub_of(card, "per frame")

    files = {f["relpath"]: f for f in files_of(page)}
    assert value_of(files["images"], "on disk") and value_of(files["odometry.txt"], "on disk")
    # An absent file has no size, whatever the measurement said.
    assert value_of(files["gps.npy"], "on disk") is None


# --------------------------------------------------------------------------
# the Extraction card: the extractor's own three report files
# --------------------------------------------------------------------------


def _write_reports(scene: Path) -> None:
    """The three files exactly as the extractor writes them, through its own writers."""
    from dataset_tools import gnss_guard, tf_pins, time_align

    ts = np.loadtxt(scene / "data_ts.txt")
    t0 = float(ts[0])
    windows = [
        {"t": t0 + 1.0, "delta": -0.69, "gain": 30.0, "depth": 25.0},
        {"t": t0 + 2.0, "delta": -0.71, "gain": 28.0, "depth": 22.0},
        {"t": t0 + 3.0, "delta": -0.05, "gain": 3.0, "depth": 4.0},
    ]
    profile = time_align.Profile(windows, np.array([t0 + 1.0, t0 + 2.0]), np.array([-0.69, -0.71]), "measured")
    time_align.write_report(scene / "time_align.txt", profile, t0)
    verdict = gnss_guard.Verdict(
        spans=[gnss_guard.Span(t0 + 5.0, t0 + 8.0, "frozen"), gnss_guard.Span(t0 + 20.0, t0 + 21.0, "island")],
        n_poses_dropped=40,
        n_frames_dropped=30,
        n_poses=1000,
        n_frames=900,
        n_frames_unsupported=7,
    )
    gnss_guard.write_report(scene / "gnss_guard.txt", verdict, t0)
    pin = tf_pins.PINS[0]
    bag_xyz = tuple(v + d for v, d in zip(pin.xyz, (0.03, 0.04, 0.0), strict=True))  # 5 cm off, far
    tf_pins.write_report(scene / tf_pins.FILE, [tf_pins.Record(pin, bag_xyz, pin.rpy_deg)])


def test_the_extraction_card_sums_up_the_three_report_files(tmp_path):
    from dataset_tools import tf_pins

    root_path = make_root(tmp_path)
    _write_reports(root_path / "0000")
    page = scene_page(tmp_path, root_path)
    card = card_of(page, "Extraction")
    assert card is not None and card["warn"] is None

    assert value_of(card, "frame clock") == "-0.700 s"  # the median of the two knots
    assert sub_of(card, "frame clock") == "2/3 windows (measured)"
    assert value_of(card, "measured offset") == "-690 ms"
    assert sub_of(card, "measured offset") == "median over 3 windows, largest -710 ms"

    assert value_of(card, "fabricated stretches") == 2
    assert value_of(card, "cost") == "40 poses · 30 frames"
    assert sub_of(card, "cost") == "of 1,000 · 900"
    assert value_of(card, "no pose under them") == "7 frames"
    assert "frozen, island" in card["note"]

    pin = tf_pins.PINS[0]
    assert value_of(card, f"{pin.parent} → {pin.child}") == "5.0 cm · 0.00°"
    assert sub_of(card, f"{pin.parent} → {pin.child}") == "bag from pin, far"
    assert "negative moves a frame earlier" in card["note"]

    # And the inventory knows the files by name.
    present = {f["relpath"] for f in files_of(page) if f["state"] in ("yes", "partial")}
    assert {"time_align.txt", "gnss_guard.txt", "tf_pins.txt"} <= present


def test_a_scene_without_the_report_files_has_no_extraction_card(tmp_path):
    page = scene_page(tmp_path, make_root(tmp_path))
    assert card_of(page, "Extraction") is None
    absent = {f["relpath"] for f in files_of(page) if f["state"] not in ("yes", "partial")}
    assert {"time_align.txt", "gnss_guard.txt", "tf_pins.txt"} <= absent


def test_an_uncorrected_clock_and_a_constant_read_as_such(tmp_path):
    from dataset_tools import time_align

    root_path = make_root(tmp_path)
    scene = root_path / "0000"
    t0 = float(np.loadtxt(scene / "data_ts.txt")[0])
    time_align.write_report(
        scene / "time_align.txt", time_align.Profile([], np.zeros(0), np.zeros(0), "too little"), t0
    )
    card = card_of(scene_page(tmp_path, root_path), "Extraction")
    assert value_of(card, "frame clock") == "not corrected"
    assert sub_of(card, "frame clock") == "too little"
    assert value_of(card, "measured offset") is None

    constant = time_align.Profile([], np.array([t0]), np.array([0.69]), "given on the command line")
    time_align.write_report(scene / "time_align.txt", constant, t0)
    card = card_of(scene_page(tmp_path / "again", root_path), "Extraction")
    assert value_of(card, "frame clock") == "+0.690 s"
    assert "given on the command line" in sub_of(card, "frame clock")
    assert "No gnss_guard.txt, tf_pins.txt" in card["note"]


def test_a_malformed_report_costs_its_part_and_says_so(tmp_path):
    root_path = make_root(tmp_path)
    scene = root_path / "0000"
    _write_reports(scene)
    (scene / "gnss_guard.txt").write_text("# nothing the card can count\n")
    (scene / "tf_pins.txt").write_text("base_link odin1_lidar pin 0.1 not-a-number\n")
    card = card_of(scene_page(tmp_path, root_path), "Extraction")
    assert value_of(card, "frame clock") is not None
    assert value_of(card, "cost") is None
    assert card["warn"] == "gnss_guard.txt could not be read; tf_pins.txt could not be read"


def test_the_window_tables_carry_the_attitude_verdict(client):
    for url in ("/scene/0000", "/scene/0000/segment/0"):
        text = client.get(url).text
        assert "GT attitude" in text
        # The fixture's ground truth is level and so is its IMU: a green 0.0°.
        assert "median tilt gap" in text
        assert 'class="chip yes"' in text and "0.0°" in text


def test_the_scene_page_follows_a_root_level_report_without_a_scene_change(tmp_path):
    """validation.csv is rewritten beside a running viewer; its card must not
    wait for the scene directory's fingerprint to move."""
    from dataset_tools.viewer.build import SceneAccess
    from dataset_tools.viewer.scene_page import build_scene_page

    root_path = make_root(tmp_path)
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")
    access = SceneAccess(root, "0000")
    assert card_of(build_scene_page(root, access), "Validation") is None
    (root_path / "validation.csv").write_text("scene,status,errors,warnings\n0000,WARN,0,2\n")
    assert card_of(build_scene_page(root, access), "Validation") is not None
