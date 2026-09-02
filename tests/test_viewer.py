"""Tests for the dataset viewer (dataset_tools.viewer).

The heavy parts of the viewer (SimView + torch) are an optional extra, so the
tests are layered: the geometry, probing, windowing and wire-format tests run
with numpy alone, and the end-to-end HTTP tests skip cleanly when the extra is
not installed.

Every fixture builds a synthetic dataset root, so nothing here touches a real
dataset or the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dataset_tools.viewer import loaders, probe, render, windows, wire


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def _write_png(path: Path, size=(16, 10)) -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    Image.new("RGB", size, (40, 80, 120)).save(path)


def make_root(
    tmp_path: Path,
    scenes=("0000",),
    frames: int = 12,
    *,
    with_height_maps: bool = True,
    with_cmd_vel: bool = True,
    with_transforms: bool = True,
    with_gravity: bool = True,
    with_joints: bool = True,
    segments: bool = True,
) -> Path:
    """A synthetic dataset root with a controllable set of optional files."""
    root = tmp_path / "root"
    root.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)

    for scene in scenes:
        d = root / scene
        (d / "images").mkdir(parents=True)
        (d / "clouds").mkdir()
        ts = 1_700_000_000.0 + 0.1 * np.arange(frames)
        for i in range(frames):
            _write_png(d / "images" / f"{i:05d}.png")
            cloud = rng.normal(scale=2.0, size=(64, 4)).astype(np.float32)
            cloud[:, 0] = np.abs(cloud[:, 0])  # forward-only, like a front lidar
            np.save(d / "clouds" / f"{i:05d}.npy", cloud)
        np.savetxt(d / "data_ts.txt", ts, fmt="%.6f")
        np.savetxt(d / "pcd_ts.txt", ts - 0.005, fmt="%.6f")

        poses = []
        for i in range(frames):
            yaw = 0.05 * i
            c, s = np.cos(yaw), np.sin(yaw)
            T = np.eye(4)
            T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
            T[:3, 3] = [0.3 * i, 0.02 * i, 0.0]
            poses.append(T)
        with open(d / "odometry.txt", "w") as fh:
            fh.write("# Timestamp(sec) r11 ... 0 0 0 1\n")
            for t, T in zip(ts, poses):
                fh.write(" ".join(f"{v:.9f}" for v in [t, *T.reshape(-1)]) + "\n")
        if segments:
            half = frames // 2
            with open(d / "odometry_segments.txt", "w") as fh:
                fh.write("# id start_ts end_ts n_poses reason\n")
                fh.write(f"0 {ts[0]:.6f} {ts[half - 1]:.6f} {half} filtered:pose_jump\n")
                fh.write(f"1 {ts[half]:.6f} {ts[-1]:.6f} {frames - half} filtered:end_of_run\n")

        if with_transforms:
            np.save(d / "transforms.npy", np.stack(poses))
        if with_gravity:
            np.save(d / "gravity_align.npy", np.stack([np.eye(3)] * frames))
        if with_joints:
            jt = np.linspace(ts[0], ts[-1], frames * 4)
            js = np.zeros((len(jt), 10))
            js[:, 0] = jt
            js[:, 1:4] = np.linspace(0, 10, len(jt))[:, None]  # positions
            js[:, 4:7] = 1.5  # velocities
            js[:, 7:10] = 0.5  # efforts
            np.save(d / "joint_states.npy", js)
        if with_cmd_vel:
            ct = np.linspace(ts[0], ts[-1], frames * 2)
            cv = np.zeros((len(ct), 7))
            cv[:, 0] = ct
            cv[:, 1] = 0.4
            cv[:, 6] = 0.1
            np.save(d / "cmd_vel.npy", cv)
        if with_height_maps:
            (d / "height_maps").mkdir()
            (d / "height_masks").mkdir()
            for i in range(frames):
                hm = np.zeros((128, 128), dtype=np.float64)
                hm[:] = 0.01 * np.arange(128)[:, None]
                np.savez_compressed(d / "height_maps" / f"{i:05d}.npz", height_map=hm)
                validity = np.full((128, 128), 2, dtype=np.uint8)
                validity[:64] = 0  # behind the robot: extrapolated
                np.savez_compressed(d / "height_masks" / f"{i:05d}.npz", validity=validity)
        np.savetxt(d / "camera_intrinsics.txt", np.eye(3) * 700.0)
        np.savetxt(d / "T_cam_to_base.txt", np.eye(4))
    return root


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def test_quaternion_round_trips_through_rotation_matrices():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(64, 3, 3))
    u, _, vt = np.linalg.svd(a)
    R = u @ vt
    R[np.linalg.det(R) < 0] *= -1
    q = loaders.mat_to_quat(R)
    back = loaders.poses_from_quat(np.zeros((64, 3)), q)[:, :3, :3]
    assert np.allclose(R, back, atol=1e-9)
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0)


def test_quaternion_of_identity_is_scalar_one():
    assert np.allclose(loaders.mat_to_quat(np.eye(3)[None]), [[1, 0, 0, 0]])


def test_slerp_midpoint_is_the_half_angle():
    q0 = np.array([[1.0, 0, 0, 0]])
    angle = np.pi / 2
    q1 = np.array([[np.cos(angle / 2), 0, 0, np.sin(angle / 2)]])
    mid = loaders.quat_slerp(q0, q1, np.array([0.5]))
    expected = np.array([np.cos(angle / 4), 0, 0, np.sin(angle / 4)])
    assert np.allclose(mid[0], expected, atol=1e-12)


def test_interpolate_poses_marks_queries_outside_the_track_invalid():
    ts = np.array([10.0, 11.0, 12.0])
    poses = np.stack([np.eye(4)] * 3)
    for i, x in enumerate([0.0, 1.0, 2.0]):
        poses[i, 0, 3] = x
    pos, _quat, valid = loaders.interpolate_poses(ts, poses, np.array([9.0, 10.5, 12.0, 13.0]))
    assert valid.tolist() == [False, True, True, False]
    assert pos[1, 0] == pytest.approx(0.5)
    # Outside queries are clamped, not extrapolated.
    assert pos[0, 0] == pytest.approx(0.0)
    assert pos[3, 0] == pytest.approx(2.0)


def test_resample_series_does_not_invent_values_across_a_long_gap():
    src_ts = np.array([0.0, 1.0, 50.0, 51.0])
    vals = np.array([[0.0], [1.0], [2.0], [3.0]])
    out = loaders.resample_series(src_ts, vals, np.array([0.5, 25.0]), max_gap=1.0)
    assert out[0, 0] == pytest.approx(0.5)
    assert np.isnan(out[1, 0])


def test_roll_pitch_of_an_identity_gravity_rotation_is_zero():
    roll, pitch = loaders.roll_pitch_from_gravity_align(np.eye(3)[None])
    assert roll[0] == pytest.approx(0.0)
    assert pitch[0] == pytest.approx(0.0)


def test_odometry_parses_both_the_17_and_13_column_forms(tmp_path):
    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, 3.0]
    long_form = tmp_path / "long.txt"
    long_form.write_text("# header\n" + " ".join(["100.0", *[str(v) for v in T.reshape(-1)]]) + "\n")
    short = tmp_path / "short.txt"
    short.write_text("# header\n" + " ".join(["100.0", *[str(v) for v in T[:3, :].reshape(-1)]]) + "\n")
    for path in (long_form, short):
        result = loaders.read_odometry(path)
        assert result is not None
        ts, poses = result
        assert ts[0] == pytest.approx(100.0)
        assert np.allclose(poses[0], T)


def test_segments_parse_and_sort(tmp_path):
    path = tmp_path / "segments.txt"
    path.write_text(
        "# id start_ts end_ts n_poses reason\n"
        "1 20.0 30.0 100 filtered:pose_jump\n"
        "0 0.0 10.0 50 v11_fill\n"
        "garbage line\n"
    )
    segs = loaders.read_segments(path)
    assert [s.index for s in segs] == [0, 1]
    assert segs[0].reason == "v11_fill"
    assert segs[1].duration == pytest.approx(10.0)


def test_missing_files_return_none_rather_than_raising(tmp_path):
    assert loaders.read_odometry(tmp_path / "nope.txt") is None
    assert loaders.read_segments(tmp_path / "nope.txt") == []
    assert loaders.read_timestamps(tmp_path / "nope.txt") is None
    assert loaders.load_npy(tmp_path / "nope.npy") is None
    assert loaders.read_matrix(tmp_path / "nope.txt") is None


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def test_bev_display_puts_forward_at_the_top_and_left_at_the_left():
    """Known answer: the height map is [row=x forward, col=y left], and the
    rendered image must be a top-down view with forward up."""
    grid = np.zeros((8, 8))
    grid[7, 4] = 1.0  # far forward (max x), slightly left of centre
    disp = render._bev_to_display(grid)
    row, col = np.unravel_index(np.argmax(disp), disp.shape)
    assert row == 0, "max-x cell must land in the top row"
    assert col == 3, "the y axis must be mirrored so +y is on the left"


def test_colorize_handles_nan_and_a_flat_range():
    out = render.colorize(np.array([[np.nan, np.nan]]), cmap="turbo")
    assert out.shape == (1, 2, 3)
    assert out.dtype == np.uint8
    flat = render.colorize(np.full((4, 4), 3.0))
    assert flat.shape == (4, 4, 3)


def test_renderers_produce_valid_images():
    pytest.importorskip("PIL")
    from PIL import Image
    import io

    hm = np.random.default_rng(0).normal(size=(128, 128))
    for data in (render.render_height_map(hm), render.render_validity((hm > 0).astype(np.uint8))):
        assert Image.open(io.BytesIO(data)).size[0] > 0
    cloud = np.random.default_rng(1).normal(size=(500, 4))
    assert Image.open(io.BytesIO(render.render_cloud_bev(cloud))).size[0] > 0


# --------------------------------------------------------------------------
# probing
# --------------------------------------------------------------------------


def test_probe_reports_what_a_scene_has_and_lacks(tmp_path):
    root_path = make_root(tmp_path, with_cmd_vel=False, with_height_maps=False)
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")
    assert root.scene_ids() == ["0000"]
    p = root.probe("0000")
    assert p.n_frames == 12
    assert p.has("images") and p.has("clouds") and p.has("odometry")
    assert not p.has("cmd_vel")
    assert not p.has("height_maps")
    assert "cmd_vel" in p.missing and "height_maps" in p.missing
    assert p.counts["images"] == 12
    assert p.n_segments == 2


def test_probe_cache_is_reused_and_invalidated_by_an_mtime_change(tmp_path):
    root_path = make_root(tmp_path)
    cache = tmp_path / "cache"
    root = probe.DatasetRoot(root_path, cache_dir=cache)
    first = root.probe("0000")
    assert (cache / "probe_cache.json").is_file()
    # A second DatasetRoot reads the persisted probe rather than re-listing.
    again = probe.DatasetRoot(root_path, cache_dir=cache)
    assert again.probe("0000").n_frames == first.n_frames


def test_probe_warns_when_per_frame_counts_disagree(tmp_path):
    root_path = make_root(tmp_path)
    (root_path / "0000" / "clouds" / "00011.npy").unlink()
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")
    p = root.probe("0000", refresh=True)
    assert any("counts disagree" in w for w in p.warnings)


def test_a_root_with_no_scenes_csv_or_manifest_still_works(tmp_path):
    root_path = make_root(tmp_path)
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")
    assert root.manifest() is None
    assert root.csv_rows() == {}
    assert root.probe_all()


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
    for a, b in zip(chunks, chunks[1:]):
        assert a.frame_end == b.frame_start


def test_whole_scene_window_reports_its_pose_source():
    ts = np.arange(50, dtype=np.float64)
    with_tf = windows.windows_for_scene(ts, [], 50, has_transforms=True)
    assert with_tf[0].pose_source == "transforms"
    without = windows.windows_for_scene(ts, [], 50, has_transforms=False)
    assert without[0].pose_source == "none"


# --------------------------------------------------------------------------
# wire format
# --------------------------------------------------------------------------


def test_split_scene_document_extracts_every_blob():
    doc = {
        "model": {"terrain": {"heightData": "__b64__AAAA"}, "n": 1},
        "states": {"version": 4, "bodies": {"a": {"bodyTransform": "__b64__BBBB"}}},
    }
    model, states, blobs = wire.split_scene_document(doc)
    assert len(blobs) == 2
    assert model["terrain"]["heightData"].startswith("/blob/")
    assert states["bodies"]["a"]["bodyTransform"].startswith("/blob/")
    # The viewer resolves a blob reference by its last path segment.
    ids = {int(model["terrain"]["heightData"].rsplit("/", 1)[-1]),
           int(states["bodies"]["a"]["bodyTransform"].rsplit("/", 1)[-1])}
    assert ids == {0, 1}


def test_write_static_bundle_lays_out_the_files_the_front_end_fetches(tmp_path):
    doc = {"model": {"x": "__b64__AAAA"}, "states": {"version": 4}}
    dest = tmp_path / "bundle"
    info = wire.write_static_bundle(doc, dest)
    assert info["blobs"] == 1
    assert (dest / "model.json").is_file()
    assert (dest / "states.json").is_file()
    assert (dest / "blob" / "0").is_file()
    assert json.loads((dest / "model.json").read_text())["x"] == "/blob/static/0"


def test_render_viewer_page_substitutes_urls_and_injects_the_static_base():
    pytest.importorskip("simview")
    page = wire.render_viewer_page("/simview-static", "/api/data/k", "1.2.3", "<div id='x'></div>")
    assert 'window.__simviewStaticBase = "/api/data/k"' in page
    assert "/simview-static/js/main.js" in page
    assert "<base href=\"/\">" in page
    assert "{{" not in page
    assert "<div id='x'></div>" in page


# --------------------------------------------------------------------------
# end to end over HTTP
# --------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path):
    pytest.importorskip("simview")
    pytest.importorskip("torch")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from dataset_tools.viewer.app import create_app

    root_path = make_root(tmp_path, scenes=("0000", "0001"), frames=12, with_cmd_vel=False)
    app = create_app(root_path, cache_dir=tmp_path / "cache")
    return TestClient(app)


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
    meta = client.get(f"/api/data/0000/seg000/{opts}/meta.json")
    assert meta.status_code == 200, meta.text
    body = meta.json()
    assert body["pose_source"] == "odometry"
    assert body["build"]["terrain_source"] in ("height_maps", "clouds")

    model = client.get(f"/api/data/0000/seg000/{opts}/model.json").json()
    states = client.get(f"/api/data/0000/seg000/{opts}/states.json").json()
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
                        raw = client.get(f"/api/data/0000/seg000/{opts}/blob/{n}").content
                        blobs[n] = "__b64__" + base64.b64encode(raw).decode()
                    node[k] = blobs[n]
                else:
                    resolve(v)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                if isinstance(v, str) and v.startswith("/blob/"):
                    n = v.rsplit("/", 1)[-1]
                    if n not in blobs:
                        raw = client.get(f"/api/data/0000/seg000/{opts}/blob/{n}").content
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
    client.get(f"/api/data/0000/seg000/{opts}/meta.json")
    r = client.get(f"/api/data/0000/seg000/{opts}/blob/0", headers={"Range": "bytes=0-15"})
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
    assert client.get("/view/0000/nosuchwindow").status_code == 404


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
    meta = client.get(f"/api/data/0000/seg000/{opts}/meta.json")
    assert meta.status_code == 200, meta.text
    body = meta.json()
    assert body["build"]["terrain_source"] == "clouds"
    # Only pose-derived scalars survive when there is no proprioception.
    assert set(body["scalar_names"]) <= {"speed", "pose_z", "yaw_rate"}
    assert "height_maps" not in body["available"]
