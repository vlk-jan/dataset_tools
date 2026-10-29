"""Viewer: colour maps and the per-frame tiles (dataset_tools.viewer.render)."""

from __future__ import annotations

import numpy as np
from helpers_viewer import decode_rgb, write_image

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
    hm = np.random.default_rng(0).normal(size=(128, 128))
    for data in (render.render_height_map(hm), render.render_validity((hm > 0).astype(np.uint8))):
        assert decode_rgb(data).shape[1] > 0
    cloud = np.random.default_rng(1).normal(size=(500, 4))
    assert decode_rgb(render.render_cloud_bev(cloud)).shape[1] > 0


def test_the_cameras_own_jpeg_is_served_untouched_when_no_downscale_is_needed(tmp_path):
    from dataset_tools.viewer import render

    path = tmp_path / "00000.jpg"
    write_image(path, (64, 40))
    assert render.render_camera_image(path, width=640) == path.read_bytes()
    smaller = render.render_camera_image(path, width=32)
    assert smaller != path.read_bytes()
    assert decode_rgb(smaller).shape[:2] == (20, 32)


def test_a_thumbnail_decoded_at_reduced_size_matches_a_full_decode(tmp_path):
    import cv2

    rng = np.random.default_rng(0)
    # Smooth content, like a camera frame; noise would differ by aliasing alone.
    small = rng.integers(0, 255, (13, 16, 3), dtype=np.uint8)
    frame = cv2.resize(small, (1600, 1296), interpolation=cv2.INTER_CUBIC)
    for name, params in (("base.jpg", []), ("prog.jpg", [cv2.IMWRITE_JPEG_PROGRESSIVE, 1])):
        path = tmp_path / name
        path.write_bytes(cv2.imencode(".jpg", frame, params)[1].tobytes())
        assert render._jpeg_width(path.read_bytes()) == 1600
        thumb = decode_rgb(render.render_camera_image(path, width=192))
        assert thumb.shape[:2] == (156, 192)
        want = decode_rgb(render._jpeg(render._shrink(render._decode(path.read_bytes()), 192)))
        assert np.abs(thumb.astype(int) - want.astype(int)).mean() < 2.0
    assert render._jpeg_width(b"\xff\xd8\xff\xd9") is None


def test_a_known_point_lands_on_its_pixel_when_projected_into_the_image(tmp_path):
    path = tmp_path / "00000.png"
    write_image(path, (200, 100), (0, 0, 0))
    K = np.array([[100.0, 0.0, 100.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    # The camera's optical frame (z forward, x right, y down) expressed in the
    # base frame (x forward, y left, z up), 0.2 m above the base origin.
    T = np.eye(4)
    T[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    T[2, 3] = 0.2
    # A return 5 m ahead, 1 m to the left, at the camera's height: x_cam = -1,
    # y_cam = 0, z_cam = 5 -> u = 100 - 100 / 5 = 80, v = 50.
    cloud = np.array([[5.0, 1.0, 0.2, 10.0]], dtype=np.float32)
    jpeg, span = render.render_cloud_on_image(path, cloud, K, T, width=1280)
    assert span is not None and span[0] < span[1]  # the depth span the page's legend states
    px = decode_rgb(jpeg)
    assert px[50, 80].sum() > 100  # the dot
    assert px[50, 150].sum() < 30  # nowhere else
    # A return behind the camera is not drawn at all.
    behind, no_span = render.render_cloud_on_image(path, np.array([[-5.0, 1.0, 0.2, 10.0]]), K, T)
    assert no_span is None
    assert decode_rgb(behind).sum() < 3000


def test_fisheye_and_distortion_projection_lands_at_fisheye_coordinates(tmp_path):
    path = tmp_path / "00000.png"
    write_image(path, (300, 100), (0, 0, 0))
    K = np.array([[100.0, 0.0, 100.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    T[2, 3] = 0.2
    # Point at 45 degrees to the right: x_cam = 5, y_cam = 0, z_cam = 5 -> theta = pi/4
    # Pinhole projection: u = 100 + 100 * (5 / 5) = 200
    # Equidistant fisheye (D=0): u = 100 + 100 * (pi / 4) ≈ 179
    cloud = np.array([[5.0, -5.0, 0.2, 10.0]], dtype=np.float32)

    # 1. Pinhole: lands around u=200
    jpeg_pin, _ = render.render_cloud_on_image(path, cloud, K, T, width=1280)
    px_pin = decode_rgb(jpeg_pin)
    assert px_pin[50, 200].sum() > 100
    assert px_pin[50, 179].sum() < 30

    # 2. Equidistant fisheye: lands around u=179
    jpeg_fish, _ = render.render_cloud_on_image(
        path, cloud, K, T, width=1280, distortion_model="equidistant", distortion_coeffs=[0.0, 0.0, 0.0, 0.0]
    )
    px_fish = decode_rgb(jpeg_fish)
    assert px_fish[50, 179].sum() > 100
    assert px_fish[50, 200].sum() < 30
