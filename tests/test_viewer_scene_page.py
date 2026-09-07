"""Viewer: the scene page's cards (dataset_tools.viewer.scene_page).

Every test builds a synthetic dataset root (``helpers_viewer``), so nothing
here touches a real dataset or the network. The end-to-end HTTP tests skip
cleanly when the viewer extra (SimView + torch) is not installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from helpers_viewer import (
    CAMERA_INFO,
    CURATED_THUMBS,
    make_root,
    write_atlas,
)

from dataset_tools.viewer import probe

# --------------------------------------------------------------------------
# the scene page: four cards assembled from probe + atlas + a few small files
# --------------------------------------------------------------------------


def scene_page(tmp_path, root_path, scene: str = "0000"):
    from dataset_tools.viewer.build import SceneAccess
    from dataset_tools.viewer.scene_page import build_scene_page

    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")
    return build_scene_page(root, SceneAccess(root, scene))


def card_of(page, title: str) -> dict | None:
    return next((c for c in page["cards"] if c["title"] == title), None)


def value_of(card, label: str):
    return next((s["value"] for s in card["stats"] if s["label"] == label), None)


def sub_of(card, label: str):
    return next((s["sub"] for s in card["stats"] if s["label"] == label), None)


def test_camera_info_reads_the_fields_the_page_shows(tmp_path):
    from dataset_tools.viewer.scene_page import read_camera_info

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
    from dataset_tools.viewer.scene_page import read_camera_info

    assert read_camera_info(tmp_path / "nope.yaml") is None
    bad = tmp_path / "bad.yaml"
    bad.write_text("!!! not: [a, yaml\nfile at all\n")
    read_camera_info(bad)  # must not raise; whatever it salvages is fine


def test_a_series_reports_the_files_own_rate_not_the_sampled_one(tmp_path):
    from dataset_tools.viewer.scene_page import read_series

    arr = np.zeros((10_000, 4))
    arr[:, 0] = 1_700_000_000.0 + 0.01 * np.arange(10_000)  # 100 Hz
    arr[:, 1] = 2.0
    path = tmp_path / "series.npy"
    np.save(path, arr)

    series = read_series(path, max_rows=50)
    assert series.n == 10_000  # the header count, not the sampled one
    assert len(series.rows) <= 51
    assert series.rate == pytest.approx(100.0, rel=0.01)
    assert series.mean_abs(1) == pytest.approx(2.0)


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
    # Curated numbers that cannot be recomputed are still shown.
    assert value_of(card, "2 s err med / p95") == "0.12 / 0.25 m"
    assert value_of(card, "divergence flags") == 24


def test_a_filled_segment_is_marked_on_the_scene_timeline(tmp_path):
    root_path = make_root(tmp_path)
    scene = root_path / "0000"
    ts = np.loadtxt(scene / "data_ts.txt")
    with open(scene / "odometry_segments.txt", "w") as fh:
        fh.write(f"0 {ts[0]:.6f} {ts[5]:.6f} 6 filtered:pose_jump\n")
        fh.write(f"1 {ts[6]:.6f} {ts[-1]:.6f} 6 v11_fill\n")
    card = card_of(scene_page(tmp_path, root_path), "Ground truth")
    assert len(card["timeline"]["fill"]) == 1
    assert len(card["timeline"]["spans"]) == 1
    assert "filled from v1.1" in card["note"]


def test_the_card_never_claims_more_raw_poses_than_the_scene_has(tmp_path):
    """A filled scene has more poses than this root's own run produced."""
    root_path = make_root(tmp_path)
    atlas = write_atlas(root_path) / "atlas.json"
    data = json.loads(atlas.read_text())
    data["scenes"]["0000"]["poses_unf"] = 3  # fewer than the segments carry
    atlas.write_text(json.dumps(data))
    card = card_of(scene_page(tmp_path, root_path), "Ground truth")
    assert value_of(card, "poses") == "12"
    assert sub_of(card, "poses") is None


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
    """Picture and caption come from one number, so they cannot drift apart:
    before the atlas named the frame, the link was recomputed from the fraction
    and simply assumed the stored JPEG had been cut there."""
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
    from fastapi.testclient import TestClient

    from dataset_tools.viewer.app import create_app

    root_path = make_root(tmp_path)
    atlas_dir = write_atlas(root_path)
    for stored in (atlas_dir / "thumbs").glob("*.jpg"):
        stored.unlink()
    # The fixture writes one flat colour per frame, so mark the frame the 50 %
    # fraction would have picked -- otherwise every frame renders alike and the
    # test cannot tell the two rules apart.
    from PIL import Image

    Image.new("RGB", (16, 10), (200, 30, 30)).save(root_path / "0000" / "images" / "00006.png")

    client = TestClient(create_app(root_path, cache_dir=tmp_path / "cache"))
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


def test_a_scene_page_on_a_root_with_almost_nothing_still_renders(tmp_path):
    pytest.importorskip("simview")
    pytest.importorskip("torch")
    from fastapi.testclient import TestClient

    from dataset_tools.viewer.app import create_app

    root_path = make_root(
        tmp_path,
        with_height_maps=False,
        with_cmd_vel=False,
        with_transforms=False,
        with_gravity=False,
        with_joints=False,
        with_camera_info=False,
    )
    client = TestClient(create_app(root_path, cache_dir=tmp_path / "cache"))
    body = client.get("/scene/0000").text
    assert "<h2>Recording</h2>" in body and "<h2>Ground truth</h2>" in body
    assert "<h2>Proprioception</h2>" not in body  # nothing to fill it
    assert "field of view" not in body  # no camera_info, so no image size to use
    assert 'class="chip no"' in body


def test_the_index_and_the_scene_page_share_one_palette():
    """The index may restyle its *type*; its colours are the viewer's colours."""
    import dataset_tools.viewer as viewer  # the package, not the FastAPI app

    css = (Path(viewer.__file__).parent / "static" / "style.css").read_text()
    index_scope = css[css.index("body.atlas {") :].split("}", 1)[0]
    for token in ("--bg", "--panel", "--fg", "--accent", "--ok", "--warn", "--line"):
        assert token not in index_scope, f"{token} is redefined for the index"
    assert "--sans" in index_scope  # only the type stacks are the index's own
    root_scope = css[css.index(":root {") :].split("}", 1)[0]
    for token in ("--track", "--ok-soft", "--shade", "--mono"):
        assert token in root_scope
