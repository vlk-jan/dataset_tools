"""Viewer: colour maps and the per-frame tiles (dataset_tools.viewer.render).

Every test builds a synthetic dataset root (``helpers_viewer``), so nothing
here touches a real dataset or the network. The end-to-end HTTP tests skip
cleanly when the viewer extra (SimView + torch) is not installed.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from dataset_tools.viewer import render

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

    hm = np.random.default_rng(0).normal(size=(128, 128))
    for data in (render.render_height_map(hm), render.render_validity((hm > 0).astype(np.uint8))):
        assert Image.open(io.BytesIO(data)).size[0] > 0
    cloud = np.random.default_rng(1).normal(size=(500, 4))
    assert Image.open(io.BytesIO(render.render_cloud_bev(cloud))).size[0] > 0


def test_the_cameras_own_jpeg_is_served_untouched_when_no_downscale_is_needed(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    from dataset_tools.viewer import render

    path = tmp_path / "00000.jpg"
    Image.new("RGB", (64, 40), (40, 80, 120)).save(path, format="JPEG", quality=95)
    assert render.render_camera_image(path, width=640) == path.read_bytes()
    smaller = render.render_camera_image(path, width=32)
    assert smaller != path.read_bytes()
    assert Image.open(io.BytesIO(smaller)).size == (32, 20)


def test_a_known_point_lands_on_its_pixel_when_projected_into_the_image(tmp_path):
    pytest.importorskip("PIL")
    import io

    from PIL import Image

    path = tmp_path / "00000.png"
    Image.new("RGB", (200, 100), (0, 0, 0)).save(path)
    K = np.array([[100.0, 0.0, 100.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    # The camera's optical frame (z forward, x right, y down) expressed in the
    # base frame (x forward, y left, z up), 0.2 m above the base origin.
    T = np.eye(4)
    T[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    T[2, 3] = 0.2
    # A return 5 m ahead, 1 m to the left, at the camera's height: x_cam = -1,
    # y_cam = 0, z_cam = 5 -> u = 100 - 100 / 5 = 80, v = 50.
    cloud = np.array([[5.0, 1.0, 0.2, 10.0]], dtype=np.float32)
    jpeg = render.render_cloud_on_image(path, cloud, K, T, width=1280)
    px = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))
    assert px[50, 80].sum() > 100  # the dot
    assert px[50, 150].sum() < 30  # nowhere else
    # A return behind the camera is not drawn at all.
    behind = render.render_cloud_on_image(path, np.array([[-5.0, 1.0, 0.2, 10.0]]), K, T)
    assert np.asarray(Image.open(io.BytesIO(behind)).convert("RGB")).sum() < 3000
