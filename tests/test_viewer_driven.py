"""Viewer: the ``driven_similarity`` modality -- DRIFT's driven cells and feature
similarity -- through probe, tile, terrain layers and routes.

Every test builds a synthetic dataset root (``helpers_viewer``), so nothing
here touches a real dataset or the network. The end-to-end HTTP tests skip
cleanly when the viewer extra (SimView + torch) is not installed.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from helpers_viewer import make_root, viewer_client

from dataset_tools.viewer import loaders, probe, render


@pytest.fixture
def driven_client(tmp_path):
    # Its own subtree: a test that also takes the plain ``client`` shares tmp_path.
    (tmp_path / "driven").mkdir()
    root = make_root(tmp_path / "driven", scenes=("0000",), with_driven_similarity=True)
    return viewer_client(root, tmp_path / "driven_cache")


def _sidecar():
    rows, cols = np.mgrid[0:128, 0:128]
    driven = np.zeros((128, 128), dtype=bool)
    driven[72:75, 63:66] = True
    max_sim = np.clip(1.0 - np.hypot(rows - 73.0, cols - 64.0) * 0.1 / 3.0, 0.0, 1.0)
    max_sim[driven] = 1.0
    return driven, max_sim.astype(np.float16)


# --------------------------------------------------------------------------
# tile
# --------------------------------------------------------------------------


def test_the_driven_tile_paints_driven_cells_in_their_own_colour():
    """The driven cells are the one thing on the tile that is a fact rather
    than a similarity, so they get a colour the ramp never produces."""
    pytest.importorskip("PIL")
    from PIL import Image

    driven, max_sim = _sidecar()
    data = render.render_driven_similarity(driven, max_sim, 0.9)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    assert img.size[0] >= 384
    px = np.asarray(img)
    driven_px = np.all(px == render.DRIVEN_COLOR, axis=-1)
    assert driven_px.sum() > 0
    # Forward is up: the patch one metre ahead lands above the tile's centre.
    ys, _ = np.nonzero(driven_px)
    assert ys.mean() < img.size[1] / 2
    # Cells beyond the ramp's reach are the ramp's floor, not black or turbo.
    far = np.all(px == render.DRIVEN_SIM_ANCHORS[0].astype(np.uint8), axis=-1)
    assert far.sum() > 0
    # The trust gate is a visible step: some cells brighter than the ramp gives.
    ramp = set(map(tuple, render.LUTS["ember"].tolist()))
    boosted = [tuple(c) for c in px.reshape(-1, 3) if tuple(c) not in ramp and tuple(c) != tuple(render.DRIVEN_COLOR)]
    assert boosted, "no cell above the threshold was brightened"


def test_the_ramp_is_not_the_height_map_s():
    assert "ember" in render.LUTS
    assert not np.array_equal(render.LUTS["ember"], render.LUTS["turbo"])


# --------------------------------------------------------------------------
# probe
# --------------------------------------------------------------------------


def test_the_probe_counts_the_modality_where_it_exists(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    with_it = probe.DatasetRoot(make_root(tmp_path / "a", with_driven_similarity=True), cache_dir=tmp_path / "c1")
    p = with_it.probe("0000")
    assert p.has("driven_similarity") and p.counts["driven_similarity"] == 12
    assert "driven_similarity" in probe.COUNTED_DIRS
    assert probe.ARTEFACTS_BY_KEY["driven_similarity"].purpose
    without = probe.DatasetRoot(make_root(tmp_path / "b"), cache_dir=tmp_path / "c2")
    assert not without.probe("0000").has("driven_similarity")
    assert "driven_similarity" in without.probe("0000").missing


# --------------------------------------------------------------------------
# terrain layers
# --------------------------------------------------------------------------


def test_the_driven_layer_is_one_under_the_robot_s_path_and_zero_beside_it(tmp_path):
    """Splatted with the height maps' own placement, so the driven cell of a
    frame lands on the terrain cell the robot stood on in that frame."""
    pytest.importorskip("torch")
    from dataset_tools.viewer import build

    root_path = make_root(tmp_path, with_driven_similarity=True)
    access = build.SceneAccess(probe.DatasetRoot(root_path), "0000")
    window = access.window("seg000", 1500)
    frames = np.arange(window.frame_start, window.frame_end)
    track = build.resolve_poses(access, window, access.frame_ts[frames])
    poses = loaders.poses_from_quat(track.positions, track.quats)
    opts = build.BuildOptions().sanitized()
    grid = build._make_grid(track.positions, opts)
    build.splat_height_maps(access, grid, frames, poses, opts)
    stats = build.splat_driven_similarity(access, grid, frames, poses, opts)
    assert stats["used"] == len(frames)
    height, layers, bounds = build._finish_terrain(grid, None)

    assert bounds["driven"] == (0.0, 1.0) and bounds["drivable"] == (0.0, 1.0)
    row, col, inside = grid.cell_of(track.positions[:, 0], track.positions[:, 1])
    assert inside.all()
    assert np.all(layers["driven"][row, col] == 1.0)
    assert np.all(layers["drivable"][row, col] == 1.0)
    # Four metres to the robot's left nothing was driven and nothing resembles it.
    left = (track.positions[0] + np.array([0.0, 4.0, 0.0]))[None]
    r, c, ok = grid.cell_of(left[:, 0], left[:, 1])
    assert ok.all()
    assert layers["driven"][r, c][0] == 0.0
    assert layers["drivable"][r, c][0] == 0.0
    assert layers["driven"].max() == 1.0 and layers["drivable"].min() >= 0.0
    assert "driven" in build.TERRAIN_COLOR_MODE_HELP and "drivable" in build.TERRAIN_COLOR_MODE_HELP
    assert build.META_VERSION >= 8


def test_a_root_without_the_modality_attaches_no_such_layer(tmp_path):
    pytest.importorskip("torch")
    from dataset_tools.viewer import build

    root_path = make_root(tmp_path)
    access = build.SceneAccess(probe.DatasetRoot(root_path), "0000")
    window = access.window("seg000", 1500)
    frames = np.arange(window.frame_start, window.frame_end)
    track = build.resolve_poses(access, window, access.frame_ts[frames])
    poses = loaders.poses_from_quat(track.positions, track.quats)
    opts = build.BuildOptions().sanitized()
    grid = build._make_grid(track.positions, opts)
    build.splat_height_maps(access, grid, frames, poses, opts)
    assert build.splat_driven_similarity(access, grid, frames, poses, opts) == {
        "requested": len(frames),
        "available": 0,
        "used": 0,
    }
    _, layers, bounds = build._finish_terrain(grid, None)
    assert "driven" not in layers and "drivable" not in layers
    assert "driven" not in bounds and "drivable" not in bounds


# --------------------------------------------------------------------------
# routes and pages
# --------------------------------------------------------------------------


def test_the_route_serves_a_png_where_the_file_is_and_404s_where_it_is_not(driven_client):
    r = driven_client.get("/api/scene/0000/frame/0/driven.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert driven_client.get("/api/scene/0000/frame/9999/driven.png").status_code == 404


def test_the_route_404s_on_a_root_without_the_modality(client):
    assert client.get("/api/scene/0000/frame/0/driven.png").status_code == 404


def test_the_frame_page_shows_the_tile_only_where_the_modality_exists(driven_client, client):
    r = driven_client.get("/scene/0000/frame/3")
    assert r.status_code == 200
    assert "Driven &amp; drivable" in r.text
    assert "driven.png" in r.text
    plain = client.get("/scene/0000/frame/3")
    assert plain.status_code == 200
    assert "Driven &amp; drivable" not in plain.text
    assert "driven.png" not in plain.text
    assert "driven_similarity" not in plain.text


def test_the_window_s_metadata_carries_the_layers_and_the_availability(driven_client):
    opts = "40-160000-80-500000-0.1-height"
    meta = driven_client.get(f"/api/data/0000/seg000/{opts}/x/meta.json")
    assert meta.status_code == 200, meta.text
    body = meta.json()
    assert body["build"]["driven_similarity"]["used"] > 0
    assert "driven_similarity" in body["available"]
    modes = [m["name"] for m in body["build"]["terrain_color_modes"]]
    assert "driven" in modes and "drivable" in modes
    model = driven_client.get(f"/api/data/0000/seg000/{opts}/x/model.json").json()
    attached = set((model["terrain"].get("properties") or {}).keys())
    assert {"driven", "drivable"} <= attached


def test_the_index_lists_the_modality_only_where_it_exists(driven_client, client):
    assert "driven &amp; similar" in driven_client.get("/").text or "driven_similarity" in driven_client.get("/").text
    assert "driven_similarity" not in client.get("/").text


def test_the_grid_the_zoom_probe_reads_keeps_the_files_layout(driven_client):
    g = driven_client.get("/api/scene/0000/frame/0/grid.json").json()
    assert (g["rows"], g["cols"], g["res"], g["x0"]) == (128, 128, 0.1, -6.4)
    at = lambda name, i, j: g["layers"][name][i * g["cols"] + j]  # noqa: E731
    # [row=x, col=y], row-major: the helper's height rises with the row only.
    assert at("height", 100, 5) == 1.0 and at("height", 100, 120) == 1.0
    assert at("validity", 10, 64) == 0 and at("validity", 100, 64) == 2
    assert g["validity_labels"]["2"] == "measured"
    assert at("driven", 63, 63) == 1 and at("driven", 0, 0) == 0
    assert at("max_sim", 73, 64) == 1.0 and g["threshold"] == pytest.approx(0.9)
    assert "max_sim_geom" not in g["layers"]
    assert driven_client.get("/api/scene/0000/frame/9999/grid.json").status_code == 404


def test_the_zoom_asks_for_bigger_tiles_within_bounds(driven_client):
    import io

    from PIL import Image

    r = driven_client.get("/api/scene/0000/frame/0/heightmap.png?size=1024")
    assert Image.open(io.BytesIO(r.content)).size == (1024, 1024)
    assert driven_client.get("/api/scene/0000/frame/0/cloud.png?size=99999").status_code == 422
    page = driven_client.get("/scene/0000/frame/0").text
    assert 'data-grid="/api/scene/0000/frame/0/grid.json"' in page and "zoom.js" in page
