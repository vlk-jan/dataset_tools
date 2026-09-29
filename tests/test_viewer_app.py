"""Viewer: every route, end to end over HTTP (dataset_tools.viewer.app).

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
    _write_png,
    make_root,
)

from dataset_tools.viewer import loaders, probe, render, windows

# --------------------------------------------------------------------------
# end to end over HTTP
# --------------------------------------------------------------------------


@pytest.fixture
def test_index_and_scene_pages_render(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "0000" in r.text and "0001" in r.text
    r = client.get("/scene/0000")
    assert r.status_code == 200
    assert "Segment 0" in r.text


def test_unknown_scene_is_a_404(client):
    assert client.get("/scene/9999").status_code == 404
    assert client.get("/api/scene/9999").status_code == 404


def test_scene_api_lists_windows_and_missing_files(client):
    data = client.get("/api/scene/0000").json()
    assert data["probe"]["n_frames"] == 12
    assert "cmd_vel" in data["probe"]["missing"]
    assert any(w["pose_source"] == "odometry" for w in data["windows"])


def test_building_a_window_produces_a_loadable_simview_scene(client, tmp_path):
    """The generated bundle must be a scene SimView itself can read back."""
    from simview.scene import SimulationScene

    opts = "40-160000-80-500000-0.1-height"
    meta = client.get(f"/api/data/0000/seg000/{opts}/x/meta.json")
    assert meta.status_code == 200, meta.text
    body = meta.json()
    assert body["pose_source"] == "odometry"
    assert body["build"]["terrain_source"] in ("height_maps", "clouds")
    # The panel's GT-against-IMU comparison travels with the window.
    assert set(body["attitude"]) >= {"gt_roll", "imu_roll", "error", "median_error", "bands"}
    assert len(body["attitude"]["error"]) == len(body["frames"])

    model = client.get(f"/api/data/0000/seg000/{opts}/x/model.json").json()
    states = client.get(f"/api/data/0000/seg000/{opts}/x/states.json").json()
    assert [b["name"] for b in model["bodies"]][0] == "robot"

    # Re-inline the blobs and hand the document to SimView's own loader.
    import base64

    blobs = {}

    def resolve(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, str) and v.startswith("/blob/"):
                    n = v.rsplit("/", 1)[-1]
                    if n not in blobs:
                        raw = client.get(f"/api/data/0000/seg000/{opts}/x/blob/{n}").content
                        blobs[n] = "__b64__" + base64.b64encode(raw).decode()
                    node[k] = blobs[n]
                else:
                    resolve(v)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                if isinstance(v, str) and v.startswith("/blob/"):
                    n = v.rsplit("/", 1)[-1]
                    if n not in blobs:
                        raw = client.get(f"/api/data/0000/seg000/{opts}/x/blob/{n}").content
                        blobs[n] = "__b64__" + base64.b64encode(raw).decode()
                    node[i] = blobs[n]
                else:
                    resolve(v)

    resolve(model)
    resolve(states)
    scene = SimulationScene.from_dict({"model": model, "states": states})
    assert scene.model.terrain is not None
    assert "robot" in scene.model.bodies
    assert len(scene.states) == len(body["frames"])


def test_blob_endpoint_supports_byte_ranges(client):
    opts = "40-160000-80-500000-0.1-height"
    client.get(f"/api/data/0000/seg000/{opts}/x/meta.json")
    r = client.get(f"/api/data/0000/seg000/{opts}/x/blob/0", headers={"Range": "bytes=0-15"})
    assert r.status_code == 206
    assert len(r.content) == 16


def test_per_frame_endpoints_serve_images_and_404_past_the_end(client):
    assert client.get("/api/scene/0000/frame/0/image.jpg").status_code == 200
    assert client.get("/api/scene/0000/frame/0/heightmap.png").status_code == 200
    assert client.get("/api/scene/0000/frame/0/validity.png").status_code == 200
    assert client.get("/api/scene/0000/frame/0/cloud.png").status_code == 200
    assert client.get("/api/scene/0000/frame/9999/image.jpg").status_code == 404
    info = client.get("/api/scene/0000/frame/3/info").json()
    assert info["frame"] == 3
    assert "timestamp" in info


def test_viewer_page_wires_the_static_base_and_defers_simview(client):
    r = client.get("/view/0000/seg000")
    assert r.status_code == 200
    assert "__simviewStaticBase" in r.text
    assert "__dtSimviewEntry" in r.text  # SimView's entry point is deferred
    assert "dt-panel" in r.text
    assert 'id="dt-attitude"' in r.text  # the GT-against-IMU readout
    assert client.get("/view/0000/nosuchwindow").status_code == 404


def test_the_viewer_page_pages_between_a_scene_s_windows(client):
    """The panel's < / > next to the title: a link to the neighbouring window,
    a dead end at either end of the scene, and the build options carried over
    so paging keeps the view you set up."""
    first = client.get("/view/0000/seg000").text
    assert '/view/0000/seg001"' in first
    assert 'aria-disabled="true"' in first  # nothing before the first window

    last = client.get("/view/0000/seg001").text
    assert '/view/0000/seg000"' in last

    # ?frame= names a frame of *this* window, so it is not carried over; the
    # build options are.
    carried = client.get("/view/0000/seg000?cloud_frames=40&frame=3").text
    assert "/view/0000/seg001?cloud_frames=40" in carried
    assert "frame=3" not in carried.split("dt-winnav")[1].split("</nav>")[0]


def test_a_scene_without_optional_files_still_builds(tmp_path):
    """The degradation contract: no height maps, no gravity alignment, no
    proprioception, no wheel odometry -- the window must still build."""
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
    )
    client = TestClient(create_app(root_path, cache_dir=tmp_path / "cache"))
    opts = "40-160000-80-500000-0.1-height"
    meta = client.get(f"/api/data/0000/seg000/{opts}/x/meta.json")
    assert meta.status_code == 200, meta.text
    body = meta.json()
    assert body["build"]["terrain_source"] == "clouds"
    # Only pose-derived scalars survive when there is no proprioception.
    assert set(body["scalar_names"]) <= {"speed", "pose_z", "yaw_rate"}
    assert "height_maps" not in body["available"]


# --------------------------------------------------------------------------
# index, scene tabs and frame pages over HTTP
# --------------------------------------------------------------------------


def test_the_index_shows_curated_descriptions_and_serves_stored_thumbnails(atlas_client):
    r = atlas_client.get("/")
    assert r.status_code == 200
    assert "Synthetic Atlas" in r.text
    assert "A synthetic place for scene 0000" in r.text
    assert "An outing" in r.text and "What happened here." in r.text
    assert 'loading="lazy"' in r.text
    thumb = atlas_client.get("/api/scene/0000/thumb/a.jpg")
    assert thumb.status_code == 200
    assert "immutable" in thumb.headers["cache-control"]


def test_the_index_renders_without_any_curated_metadata(client):
    """The rig-agnostic contract: no atlas/, same page, fewer facts."""
    r = client.get("/")
    assert r.status_code == 200
    assert "0000" in r.text and "0001" in r.text
    assert 'class="env"' not in r.text  # no curated description to show
    assert "/api/scene/0000/thumb/a.jpg" in r.text
    # And that thumbnail endpoint renders a live frame instead of a stored file.
    thumb = client.get("/api/scene/0000/thumb/b.jpg")
    assert thumb.status_code == 200
    assert thumb.headers["content-type"] == "image/jpeg"


def test_the_frames_tab_pages_through_frames_and_jumps(client):
    r = client.get("/scene/0000?tab=frames")
    assert r.status_code == 200
    assert "/scene/0000/frame/0" in r.text
    assert r.text.count("image.jpg?w=192") == 12  # the whole scene fits one page
    assert 'loading="lazy"' in r.text
    # A page past the end clamps rather than 404s.
    assert client.get("/scene/0000?tab=frames&page=99").status_code == 200
    jump = client.get("/scene/0000/jump?index=7", follow_redirects=False)
    assert jump.headers["location"] == "/scene/0000/frame/7"
    assert (
        client.get("/scene/0000/jump?index=999", follow_redirects=False).headers["location"] == "/scene/0000/frame/11"
    )


def test_the_frame_page_states_what_the_frame_has(client):
    r = client.get("/scene/0000/frame/3")
    assert r.status_code == 200
    assert "Camera stamp" in r.text and "Lidar stamp" in r.text
    assert "Segment</dt>" in r.text
    assert "/view/0000/frame00003" in r.text  # the static single-frame view
    assert "?frame=3" in r.text  # and the motion context around it
    assert client.get("/scene/0000/frame/9999").status_code == 404


def test_a_frame_with_no_ground_truth_pose_still_has_a_page(tmp_path):
    pytest.importorskip("simview")
    pytest.importorskip("torch")
    from fastapi.testclient import TestClient

    from dataset_tools.viewer.app import create_app

    root_path = make_root(tmp_path, segments=False, with_height_maps=False)
    client = TestClient(create_app(root_path, cache_dir=tmp_path / "cache"))
    r = client.get("/scene/0000/frame/4")
    assert r.status_code == 200
    assert "No ground-truth pose" in r.text
    assert "robot-centric" in r.text
    assert "/view/0000/frame00004" in r.text


def test_a_single_frame_bundle_builds_and_simview_can_read_it_back(client):
    """One frame, one state: a static scene, still a valid SimView document."""
    from simview.scene import SimulationScene

    opts = "40-160000-80-500000-0.1-height"
    meta = client.get(f"/api/data/0000/frame00003/{opts}/x/meta.json")
    assert meta.status_code == 200, meta.text
    body = meta.json()
    assert body["frames"] == [3]
    assert body["pose_source"] == "odometry"
    assert body["window"]["kind"] == "frame"

    model = client.get(f"/api/data/0000/frame00003/{opts}/x/model.json").json()
    states = client.get(f"/api/data/0000/frame00003/{opts}/x/states.json").json()
    import base64

    def resolve(node):
        items = node.items() if isinstance(node, dict) else enumerate(node)
        for key, value in list(items):
            if isinstance(value, str) and value.startswith("/blob/"):
                n = value.rsplit("/", 1)[-1]
                raw = client.get(f"/api/data/0000/frame00003/{opts}/x/blob/{n}").content
                node[key] = "__b64__" + base64.b64encode(raw).decode()
            elif isinstance(value, dict | list):
                resolve(value)

    resolve(model)
    resolve(states)
    scene = SimulationScene.from_dict({"model": model, "states": states})
    assert len(scene.states) == 1
    assert "robot" in scene.model.bodies
    assert scene.model.terrain is not None


def test_a_single_frame_view_page_opens(client):
    r = client.get("/view/0000/frame00003")
    assert r.status_code == 200
    assert "__simviewStaticBase" in r.text
    r = client.get("/view/0000/seg000?frame=4")
    assert r.status_code == 200
    assert '"initialFrame": 4' in r.text


# --------------------------------------------------------------------------
# the 3D view: what the wrapper adds on top of SimView
# --------------------------------------------------------------------------


def test_the_tile_robot_is_helhest_drawn_to_scale():
    """The BEV tiles carry a top-down robot, not an anonymous dot.

    Checked geometrically rather than by eye: the drawn footprint has to be
    the real one, so it can be used as a ruler for the map around it.
    """
    pytest.importorskip("PIL")
    import io

    from PIL import Image

    # A flat map: everything that differs from the background is the robot.
    png = render.render_height_map(np.zeros((128, 128)), size=384)
    img = Image.open(io.BytesIO(png)).convert("RGB")
    assert img.size == (384, 384)
    px_per_m = (img.width / 128) / loaders.HM_RES

    arr = np.asarray(img).astype(int)
    background = arr[0, 0]
    marked = np.abs(arr - background).sum(axis=2) > 12
    rows, cols = np.nonzero(marked)
    assert marked.any(), "nothing was drawn on the tile"

    # Extent: the chassis box runs x = -0.5 .. +0.2, the rear wheel reaches
    # back to x = -1.057 and the heading wedge sticks 0.3 m past the nose; the
    # front tyres span y = +-0.431.
    height_m = (rows.max() - rows.min() + 1) / px_per_m
    width_m = (cols.max() - cols.min() + 1) / px_per_m
    assert 1.45 < height_m < 1.7, height_m  # -1.057 .. +0.5 forward
    assert 0.75 < width_m < 1.0, width_m  # +-0.431 across the wheels

    # The tile's centre is base_link, which is the FRONT AXLE: the front hubs
    # sit on the middle row, so most of the robot is drawn behind it, and the
    # ink is symmetric left-right.
    front_m = (img.height / 2 - rows.min()) / px_per_m
    rear_m = (rows.max() + 1 - img.height / 2) / px_per_m
    assert 0.4 < front_m < 0.6, front_m  # nose + wedge
    assert 0.95 < rear_m < 1.15, rear_m  # rear wheel
    assert abs((cols.min() + cols.max()) / 2 - img.width / 2) < 0.12 * px_per_m
    assert marked[img.height // 2, :].sum() > 0  # a front tyre crosses the middle row

    # Subtle: a marker that covered the map would defeat the point of the map.
    assert marked.mean() < 0.03, marked.mean()


def test_every_bev_tile_carries_the_same_robot():
    """Height map, validity and cloud are all the same robot-centred frame, so
    the robot has to be drawn the same real size on each of them."""
    pytest.importorskip("PIL")
    import io

    from PIL import Image

    # The silhouette's outline is the only near-white ink on any of these
    # tiles (turbo's low end, the validity green, and the cloud's black
    # background with its grey range rings are all far from white).
    tiles = {
        128 * loaders.HM_RES: render.render_height_map(np.zeros((128, 128))),
        128 * loaders.HM_RES + 1e-9: render.render_validity(np.full((128, 128), 2)),
        24.0: render.render_cloud_bev(np.zeros((32, 4))),
    }
    sizes = set()
    for span_m, png in tiles.items():
        img = Image.open(io.BytesIO(png)).convert("RGB")
        white = np.asarray(img).min(axis=2) >= 190
        assert white.any(), span_m
        rows, cols = np.nonzero(white)
        px_per_m = img.width / span_m
        sizes.add(round((cols.max() - cols.min() + 1) / px_per_m, 1))
        # Centred on the robot, whatever the tile's scale.
        assert abs((cols.min() + cols.max()) / 2 - img.width / 2) < 0.15 * px_per_m
    assert len(sizes) == 1, sizes


def test_the_tile_robot_matches_the_3d_robot():
    """render.py duplicates build.py's geometry (it cannot import it: build
    imports render). The two must not be allowed to drift apart."""
    pytest.importorskip("torch")
    from dataset_tools.viewer import build

    assert (build.CHASSIS["hx"], build.CHASSIS["hy"]) == render.ROBOT_CHASSIS_HALF
    assert tuple(build.CHASSIS_CENTER[:2]) == render.ROBOT_CHASSIS_CENTER
    assert render.ROBOT_WHEEL_RADIUS == build.WHEEL_RADIUS
    assert render.ROBOT_WHEEL_WIDTH == build.WHEEL_WIDTH
    assert sorted(render.ROBOT_WHEEL_HUBS) == sorted((x, y) for x, y, _z, _col, _sign in build.WHEELS.values())


def test_the_robot_model_is_drifts():
    """base_link is HELHEST's front axle: the front hubs sit on the origin and
    the box must not reach ahead of the sensors (x <= +0.2). A stale copy once
    drew the axle at +0.4 and the chassis to +0.5, which put the lidar in the
    middle of the robot."""
    pytest.importorskip("torch")
    from dataset_tools.viewer import build

    hubs = {name: (x, y) for name, (x, y, _z, _c, _s) in build.WHEELS.items()}
    assert hubs["wheel_left"] == (0.0, 0.36) and hubs["wheel_right"] == (0.0, -0.36)
    assert hubs["wheel_rear"] == (-0.697, 0.0)
    assert build.CHASSIS_CENTER[0] + build.CHASSIS["hx"] <= 0.2


def test_every_terrain_colour_mode_is_explained():
    """SimView offers a "Color Mode" per attached property and explains none
    of them, so the descriptions travel with the layer names."""
    pytest.importorskip("torch")
    from dataset_tools.viewer import build

    modes = build.terrain_color_modes(["coverage", "validity", "cloud_points"])
    assert [m["name"] for m in modes] == [
        "height",
        "coverage",
        "validity",
        "cloud_points",
    ]
    assert all(m["description"] for m in modes)
    # Every layer the builder can attach has prose written for it.
    produced = {"coverage", "validity", "cloud_points", "cloud_intensity", "cloud_spread"}
    assert produced <= set(build.TERRAIN_COLOR_MODE_HELP)


def test_meta_json_carries_the_colour_mode_legend(client):
    opts = "40-160000-80-500000-0.1-height"
    meta = client.get(f"/api/data/0000/seg000/{opts}/x/meta.json").json()
    modes = meta["build"]["terrain_color_modes"]
    assert modes[0]["name"] == "height"
    assert all(m["description"] for m in modes)
    # The legend has to describe exactly the modes SimView will offer, which
    # is "height" plus whatever properties the terrain actually carries.
    model = client.get(f"/api/data/0000/seg000/{opts}/x/model.json").json()
    attached = list((model["terrain"].get("properties") or {}).keys())
    assert [m["name"] for m in modes] == ["height", *attached]


def test_a_bundle_from_an_older_viewer_is_rebuilt_not_served(client, tmp_path):
    """The cache is keyed by what was asked for, not by the code that answers,
    so a stale bundle would otherwise outlive a viewer upgrade forever."""
    from dataset_tools.viewer import app as app_module

    opts = "40-160000-80-500000-0.1-height"
    first = client.get(f"/api/data/0000/seg000/{opts}/x/meta.json").json()
    assert first["meta_version"] == app_module.META_VERSION

    bundles = list(tmp_path.rglob("seg000/*/meta.json"))
    assert bundles, "the built bundle was not found on disk"
    stale = json.loads(bundles[0].read_text())
    stale["meta_version"] = 0
    stale["build"].pop("terrain_color_modes", None)
    bundles[0].write_text(json.dumps(stale))

    again = client.get(f"/api/data/0000/seg000/{opts}/x/meta.json").json()
    assert again["meta_version"] == app_module.META_VERSION
    assert again["build"]["terrain_color_modes"]


def test_the_overlay_offers_the_cloud_and_the_ghost_as_separate_toggles(client):
    """The old control was one either/or switch that put SimView in "points"
    mode, which hid the robot. They are independent checkboxes now."""
    page = client.get("/view/0000/seg000").text
    assert 'id="dt-cloud"' in page and 'id="dt-ghost"' in page
    assert "dt-seg" not in page  # the either/or switch is gone
    # Each toggle says what it is; the ghost is the one nobody recognises.
    assert "wheel-odometry ghost" in page
    assert "wheel_odom_drift" in page
    # And the panel's own toggle is a single edge tab, not the old pair.
    assert 'id="dt-edge"' in page
    assert 'id="dt-expand"' not in page and 'id="dt-collapse"' not in page


def test_the_viewer_page_installs_the_simview_patches(client):
    page = client.get("/view/0000/seg000").text
    assert "/viewer-static/overlay.js" in page
    patch = client.get("/viewer-static/simview_patch.js")
    assert patch.status_code == 200
    body = patch.text
    # The stopgaps have to stay findable and deletable as whole blocks.
    for marker in (
        "STOPGAP for simview <= 4.2: origin-anchored frustum culling",
        'STOPGAP for simview <= 4.2: the "points" body-visualization mode',
        "STOPGAP for simview <= 4.2.0: ScalarPlotter plots never draw (3 causes)",
    ):
        at = body.find(marker)
        assert at >= 0, marker
        # Each one has to announce, right there, that it is deletable.
        assert "Delete this block on upgrade" in body[at : at + 400], marker
    # And the permanent parts name the bodies they are about.
    assert "lidar_cloud" in body and "wheel_odom_ghost" in body


def test_the_stopgaps_switch_themselves_off_when_simview_is_upgraded(client):
    """A workaround left running over a fixed SimView would override the real
    fix, so each one is gated on the version the page reports."""
    from dataset_tools.viewer.app import _simview_version

    page = client.get("/view/0000/seg000").text
    assert f'"simviewVersion": "{_simview_version()}"' in page

    body = client.get("/viewer-static/simview_patch.js").text
    assert 'LAST_SIMVIEW_NEEDING_STOPGAPS = "4.2.0"' in body
    # The two SimView has fixed are behind the gate...
    gated = body[body.index("const stopgaps = versionAtMost") :]
    assert "stopgapOriginCulling(app);" in gated.split("hideHousekeepingBodies")[0]
    assert "stopgapDropPointsMode(app);" in gated.split("hideHousekeepingBodies")[0]
    # ...and the scalar-plot one is now gated with them (simview 4.2.1 fixes it),
    # so on a newer package none of the three run and simview handles them.
    assert "if (stopgaps) stopgapScalarXScale(app);" in body
    # It still adopts each chart once and only pins a still-null x scale.
    assert "if (_dtScalarFixed.has(chart)) continue;" in body


def test_the_initial_frame_is_opened_through_simviews_own_seek(client):
    """Assigning `currentTime` does not move the scene.

    Nothing in SimView reads that field until playback advances it, so a window
    opened from a frame's page drew the window's *first* frame and only jumped
    to the requested one once play was pressed. `goToTime` is what every one of
    SimView's own seek affordances calls: it moves `currentStateIndex` with the
    clock, redraws the scene and refreshes the playback bar.
    """
    body = client.get("/viewer-static/overlay.js").text
    assert "app.animationController.goToTime(seekTo);" in body
    # The old write is what the bug was; it must not come back.
    assert "animationController.currentTime =" not in body


def test_the_view_opens_with_the_robot_in_shot(client):
    """SimView's camera is fixed at the world origin, which is not where every
    window starts the robot -- a later part of a split segment carries on from
    the previous part, and `?frame=N` seeks into the middle of a traverse."""
    body = client.get("/viewer-static/simview_patch.js").text
    assert 'ROBOT_BODY = "robot"' in body
    assert "export function centerOnRobot(app)" in body
    # Camera and orbit target move by the same delta, so the angle and the
    # distance the view was framed at survive the re-centring.
    fn = body[body.index("export function centerOnRobot(app)") :]
    fn = fn[: fn.index("\n}\n")]
    assert "camera.position.set(" in fn and "controls.target.set(x, y, z);" in fn
    assert "controls.update();" in fn
    # It reads the body's world position the way SimView's own tracking does.
    assert "app.batchManager.getBatchOffset(batch)" in fn
    # Falsy until the robot has a resolved position, so the caller can retry.
    assert "return false;" in fn

    overlay = client.get("/viewer-static/overlay.js").text
    assert "if (centered || centerOnRobot(app)) {" in overlay
    # Once, at load -- following it every frame is SimView's own trackBody.
    assert overlay.count("centerOnRobot(app)") == 1


def test_the_aggregated_cloud_carries_no_per_frame_pose(client):
    """It is already in the world frame, so a per-frame identity transform is
    a blob that says nothing -- and is what would keep it in the Body states
    list once SimView starts filtering that list by trajectory."""
    opts = "40-160000-80-500000-0.1-height"
    client.get(f"/api/data/0000/seg000/{opts}/x/meta.json")
    model = client.get(f"/api/data/0000/seg000/{opts}/x/model.json").json()
    states = client.get(f"/api/data/0000/seg000/{opts}/x/states.json").json()
    names = [b["name"] for b in model["bodies"]]
    if "lidar_cloud" not in names:
        pytest.skip("this synthetic window aggregated no cloud")
    posed = {b["name"] for b in states["bodies"]}
    assert "robot" in posed
    assert "lidar_cloud" not in posed


# --------------------------------------------------------------------------
# smaller corrections
# --------------------------------------------------------------------------


def test_the_frame_page_reads_the_segment_tag_out_in_words(client):
    """Every other page renders `filtered:pose_jump` through `describe_reason`;
    the frame page printed the raw tag."""
    import html

    body = html.unescape(client.get("/scene/0000/frame/2").text)
    assert "Ends because" in body
    assert windows.describe_reason("filtered:pose_jump") in body
    assert "filtered:pose_jump" not in body


def test_back_to_the_frame_strip_lands_on_the_page_the_frame_is_on(tmp_path):
    """Not `index // 100`: a scene with a sparse per-frame directory pages
    through the frames that exist, so a frame's page is its position in that
    list, not its own number."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from dataset_tools.viewer.app import FRAMES_PER_PAGE, create_app

    # A sparse scene: only every 40th frame has an image on disk.
    root_path = make_root(tmp_path, frames=12, segments=False)
    scene = root_path / "0000"
    ts = 1_700_000_000.0 + 0.1 * np.arange(600)
    np.savetxt(scene / "data_ts.txt", ts, fmt="%.6f")
    for i in range(0, 600, 40):
        _write_png(scene / "images" / f"{i:05d}.png")
    for stale in scene.glob("images/*.png"):
        if int(stale.stem) % 40:
            stale.unlink()

    c = TestClient(create_app(root_path, cache_dir=tmp_path / "cache"))
    body = c.get("/scene/0000/frame/560").text
    assert f"tab=frames&amp;page={560 // FRAMES_PER_PAGE}" not in body
    assert "tab=frames&amp;page=0" in body  # 15 sparse frames: all on page 0


def test_probing_a_root_writes_its_cache_once_not_once_per_scene(tmp_path):
    """`probe_all` rewrote the whole growing cache file after every scene."""
    root_path = make_root(tmp_path, scenes=("0000", "0001", "0002", "0003"), frames=4)
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")

    writes = []
    original = probe.DatasetRoot.save_probe_cache

    def counting(self):
        if self._probe_dirty:
            writes.append(1)
        original(self)

    probe.DatasetRoot.save_probe_cache = counting
    try:
        assert len(root.probe_all()) == 4
        assert len(writes) == 1
        root.probe_all()  # everything cached and unchanged: nothing to write
        assert len(writes) == 1
        root.probe("0000", refresh=True)  # a fresh probe still gets persisted
        assert len(writes) == 2
    finally:
        probe.DatasetRoot.save_probe_cache = original


def test_the_bundle_stamp_covers_every_module_that_shapes_a_bundle(client):
    """The stamp is what makes a browser-cached bundle unaddressable after a
    code change, so it has to cover windowing and pose interpolation too --
    not only the three modules that write the files out."""
    import inspect

    from dataset_tools.viewer import app as appmod

    source = inspect.getsource(appmod.create_app)
    for module in ("build.py", "wire.py", "app.py", "windows.py", "loaders.py", "render.py"):
        assert f'"{module}"' in source, module


def test_the_status_endpoint_agrees_with_what_a_fetch_would_do(client, tmp_path):
    """Reporting "ready" for a bundle an upgrade has superseded promises a
    fetch that in fact blocks on a full rebuild."""
    opts = "40-160000-80-500000-0.1-height"
    base = f"/api/data/0000/seg000/{opts}/x"
    assert client.get(f"{base}/status").json()["ready"] is False
    client.get(f"{base}/meta.json")
    status = client.get(f"{base}/status").json()
    assert status["ready"] is True

    meta = Path(status["path"]) / "meta.json"
    stale = json.loads(meta.read_text())
    stale["meta_version"] = 0
    meta.write_text(json.dumps(stale))
    assert client.get(f"{base}/status").json()["ready"] is False


def test_a_bundle_key_that_is_not_canonical_is_refused(client):
    """The options key names a cache directory. Only the spelling
    ``BuildOptions.cache_key`` writes is accepted: ``..`` never becomes a path,
    and a variant spelling of the same options cannot mint a fresh build."""
    for opts in ("%2e%2e", "..", "40-160000-80-500000-0.10-height", "9999-160000-80-500000-0.1-height", "x"):
        r = client.get(f"/api/data/0000/seg000/{opts}/x/meta.json")
        assert r.status_code == 404, (opts, r.status_code)
    assert client.get("/api/data/0000/seg000/40-160000-80-500000-0.1-height/x/meta.json").status_code == 200


def test_the_viewer_starts_and_says_so_when_simview_is_missing(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from helpers_viewer import make_root

    from dataset_tools.viewer import app as appmod

    monkeypatch.setattr(appmod, "_simview_static_dir", lambda: None)
    root_path = make_root(tmp_path, scenes=("0000",), frames=4)
    client = TestClient(appmod.create_app(root_path, cache_dir=tmp_path / "cache"))
    r = client.get("/")
    assert r.status_code == 200
    assert "3D views unavailable" in r.text


def test_an_empty_root_says_what_a_scene_is(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from dataset_tools.viewer.app import create_app

    root_path = tmp_path / "root"
    (root_path / "run_2024_05_01").mkdir(parents=True)
    r = TestClient(create_app(root_path, cache_dir=tmp_path / "cache")).get("/")
    assert r.status_code == 200
    assert "No scenes found" in r.text and "0000" in r.text


def test_a_loopback_viewer_answers_only_to_loopback_hosts(tmp_path):
    """DNS rebinding: a page elsewhere must not reach a 127.0.0.1 viewer through
    a hostname it made resolve there."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from helpers_viewer import make_root

    from dataset_tools.viewer.app import create_app

    root_path = make_root(tmp_path, scenes=("0000",), frames=4)
    app = create_app(root_path, cache_dir=tmp_path / "cache", allowed_hosts=["127.0.0.1", "localhost"])
    assert TestClient(app, base_url="http://evil.example").get("/").status_code == 400
    assert TestClient(app, base_url="http://127.0.0.1:8080").get("/").status_code == 200
    assert TestClient(app, base_url="http://localhost").get("/healthz").status_code == 200


def test_the_scene_page_does_not_read_arrays_it_never_renders(client, tmp_path):
    """`_scene_details` loads joint_states/cmd_vel whole (no memory map) and
    the scene template renders none of it -- the four cards carry the same
    facts, read strided. It stays what /api/scene/{id} returns."""
    from dataset_tools.viewer import app as appmod

    calls = []
    original = appmod._scene_details
    appmod._scene_details = lambda access: (calls.append(access.scene_id), original(access))[1]
    try:
        assert client.get("/scene/0000").status_code == 200
        assert calls == []
        assert client.get("/api/scene/0000").status_code == 200
        assert calls == ["0000"]
    finally:
        appmod._scene_details = original


# --------------------------------------------------------------------------
# GNSS: the track endpoint, and the map that is not there without it
# --------------------------------------------------------------------------


def test_a_root_without_gps_has_no_map_anywhere(client):
    """The promise: a scene with no gps.npy looks exactly as it always did.

    Not "an empty map" and not "a map that fails to load" -- no map markup, no
    Leaflet, no fetch. Checked on every page that can carry one.
    """
    for path in ("/scene/0000", "/scene/0000/segment/0", "/scene/0000/frame/3"):
        page = client.get(path).text
        assert "leaflet" not in page.lower()
        assert "data-map" not in page and 'class="pair' not in page
    assert client.get("/api/scene/0000/gps.json").status_code == 404


def test_the_track_endpoint_serves_what_the_map_draws(gps_client):
    data = gps_client.get("/api/scene/0000/gps.json").json()
    assert data["scope"] == "scene"
    assert data["n_fixes"] == 48 and len(data["points"]) == 48
    assert data["bounds"][0][0] < data["bounds"][1][0]  # south of north
    assert data["fix"][0]["label"] == "RTK / GBAS"
    assert data["sigma_m"] == 0.5  # sqrt(0.09 + 0.16)
    assert data["marker"] is None and data["context"] is None


def test_a_long_track_is_thinned_before_it_is_shipped(gps_client):
    data = gps_client.get("/api/scene/0000/gps.json?max_points=10").json()
    assert data["n_fixes"] == 48 and len(data["points"]) <= 11
    # The numbers still describe every fix, not the ten that were drawn.
    assert data["n_points"] < data["n_fixes"]


def test_a_segments_track_comes_with_the_rest_of_the_scene_behind_it(gps_client):
    data = gps_client.get("/api/scene/0000/gps.json?segment=1").json()
    assert data["scope"] == "segment 1"
    assert 0 < data["n_fixes"] < 48
    assert data["context"], "the whole scene, faint, for context"
    assert gps_client.get("/api/scene/0000/gps.json?segment=7").status_code == 404


def test_a_frames_position_is_marked_on_the_track(gps_client):
    data = gps_client.get("/api/scene/0000/gps.json?frame=6").json()
    lat, lon = data["marker"]
    assert data["bounds"][0][0] <= lat <= data["bounds"][1][0]
    assert data["bounds"][0][1] <= lon <= data["bounds"][1][1]


def test_the_map_sits_beside_the_files_card_when_there_is_a_track(gps_client):
    page = gps_client.get("/scene/0000").text
    assert 'class="pair withmap"' in page
    assert 'data-src="/api/scene/0000/gps.json"' in page
    assert "/viewer-static/vendor/leaflet.js" in page
    # The file inventory now lists gps.npy as present, like any other artefact.
    assert "gps.npy" in page


def test_the_frame_page_marks_the_frame_it_is_showing(gps_client):
    page = gps_client.get("/scene/0000/frame/6").text
    assert 'data-src="/api/scene/0000/gps.json?frame=6"' in page
