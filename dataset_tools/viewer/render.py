"""Colour maps and small PNG/JPEG renderers for the per-frame side panel.

SimView shows 3D geometry; the panel next to it shows what the scene carries
per frame that is *not* geometry -- the camera image, the BEV height map, its
validity mask and a top-down view of the raw cloud. Everything is rendered on
demand for one frame at a time, so a 20k-frame scene never costs more than the
frames actually looked at.

Only numpy and Pillow are used (no matplotlib): the colour maps are small
piecewise-linear look-up tables built once at import.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("dataset_tools.viewer.render")

# Anchor colours (0-255) for a few perceptually reasonable ramps.
_TURBO_ANCHORS = np.array(
    [
        [48, 18, 59], [70, 107, 227], [37, 180, 248], [70, 227, 161],
        [166, 249, 76], [231, 214, 46], [253, 141, 39], [220, 47, 12],
        [122, 4, 3],
    ],
    dtype=np.float64,
)
_VIRIDIS_ANCHORS = np.array(
    [
        [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142],
        [38, 130, 142], [31, 158, 137], [53, 183, 121], [110, 206, 88],
        [181, 222, 43], [253, 231, 37],
    ],
    dtype=np.float64,
)
_GREY_ANCHORS = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.float64)

#: Fixed, meaningful colours for the three height-map validity states.
VALIDITY_COLORS = np.array(
    [
        [200, 60, 60],    # 0 extrapolated -- invented
        [230, 190, 70],   # 1 interpolated -- inside the hull, no return
        [60, 170, 90],    # 2 measured -- a lidar return landed here
    ],
    dtype=np.uint8,
)


def _build_lut(anchors: np.ndarray, n: int = 256) -> np.ndarray:
    idx = np.linspace(0.0, len(anchors) - 1.0, n)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(anchors) - 1)
    frac = (idx - lo)[:, None]
    return (anchors[lo] * (1 - frac) + anchors[hi] * frac).astype(np.uint8)


LUTS = {
    "turbo": _build_lut(_TURBO_ANCHORS),
    "viridis": _build_lut(_VIRIDIS_ANCHORS),
    "grey": _build_lut(_GREY_ANCHORS),
}


def colorize(
    values: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "turbo",
) -> np.ndarray:
    """Map an array of scalars to uint8 RGB using a named LUT.

    NaNs (and an empty value range) map to the LUT's low end rather than
    producing garbage.
    """
    lut = LUTS.get(cmap, LUTS["turbo"])
    vals = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(vals)
    if vmin is None:
        vmin = float(np.nanpercentile(vals[finite], 2)) if finite.any() else 0.0
    if vmax is None:
        vmax = float(np.nanpercentile(vals[finite], 98)) if finite.any() else 1.0
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    norm = np.clip((vals - vmin) / (vmax - vmin), 0.0, 1.0)
    norm = np.where(finite, norm, 0.0)
    idx = (norm * (len(lut) - 1)).astype(np.int32)
    return lut[idx]


def unit_colors(values: np.ndarray, cmap: str = "turbo") -> np.ndarray:
    """Like :func:`colorize` but returns float RGB in [0, 1], which is what
    SimView's point-cloud ``color`` field expects."""
    return colorize(values, cmap=cmap).astype(np.float32) / 255.0


# --------------------------------------------------------------------------
# PNG / JPEG encoders
# --------------------------------------------------------------------------


def _to_png(rgb: np.ndarray, upscale_to: int | None = None) -> bytes:
    from PIL import Image

    img = Image.fromarray(np.ascontiguousarray(rgb), mode="RGB")
    if upscale_to:
        scale = max(1, upscale_to // max(img.width, img.height))
        if scale > 1:
            img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=3)
    return buf.getvalue()


def _bev_to_display(grid: np.ndarray) -> np.ndarray:
    """Orient a ``[row=x forward, col=y left]`` BEV grid for screen display.

    Screen convention: +x (forward) is up, +y (left) is left, i.e. a top-down
    view with the robot at the centre facing up.

    ``grid[i, j]`` has ``i`` indexing x and ``j`` indexing y, both increasing,
    so display row ``r`` must map to ``x = N-1-r`` (forward at the top) and
    display column ``c`` to ``y = N-1-c`` (+y at the left). That is a double
    flip and **no** transpose -- transposing here swaps the axes and rotates
    the map 90 degrees, which shows up as the lidar's measured cone pointing
    sideways instead of up.
    """
    return grid[::-1, ::-1]


def render_height_map(hm: np.ndarray, size: int = 384) -> bytes:
    """Colour-mapped BEV height map, robot centred, forward up."""
    disp = _bev_to_display(np.asarray(hm, dtype=np.float64))
    rgb = colorize(disp, cmap="turbo")
    rgb = _draw_robot_marker(rgb)
    return _to_png(rgb, upscale_to=size)


def render_validity(mask: np.ndarray, size: int = 384) -> bytes:
    """Three-state validity mask in fixed colours (see VALIDITY_COLORS)."""
    disp = _bev_to_display(np.asarray(mask))
    idx = np.clip(disp.astype(np.int32), 0, len(VALIDITY_COLORS) - 1)
    rgb = VALIDITY_COLORS[idx]
    rgb = _draw_robot_marker(rgb.copy())
    return _to_png(rgb, upscale_to=size)


def _draw_robot_marker(rgb: np.ndarray) -> np.ndarray:
    """Mark the grid centre (the robot) and a short heading tick."""
    h, w = rgb.shape[:2]
    cy, cx = h // 2, w // 2
    rgb[max(0, cy - 1) : cy + 2, max(0, cx - 1) : cx + 2] = [255, 255, 255]
    rgb[max(0, cy - 6) : cy, cx : cx + 1] = [255, 255, 255]  # forward = up
    return rgb


def render_cloud_bev(
    cloud: np.ndarray,
    extent: float = 12.0,
    size: int = 384,
    color_by: str = "z",
) -> bytes:
    """Top-down scatter of one raw cloud, in the robot's own frame.

    Rendered by binning rather than plotting, so it costs one pass over the
    points. Empty cells stay dark, which makes the sensor's blind wedge (no
    returns behind the robot on a forward-only lidar) immediately visible.
    """
    cloud = np.asarray(cloud)
    n = size
    rgb = np.zeros((n, n, 3), dtype=np.uint8)
    if cloud.size == 0:
        return _to_png(rgb)

    x, y, z = cloud[:, 0], cloud[:, 1], cloud[:, 2]
    keep = (np.abs(x) < extent) & (np.abs(y) < extent)
    if not keep.any():
        return _to_png(rgb)
    x, y, z = x[keep], y[keep], z[keep]
    values = z
    if color_by == "intensity" and cloud.shape[1] > 3:
        values = cloud[keep, 3]

    # Screen: +x forward -> up (row 0 at top), +y left -> left (col 0 at left).
    row = np.clip(((extent - x) / (2 * extent) * n).astype(np.int32), 0, n - 1)
    col = np.clip(((extent - y) / (2 * extent) * n).astype(np.int32), 0, n - 1)
    flat = row * n + col

    acc = np.zeros(n * n, dtype=np.float64)
    cnt = np.zeros(n * n, dtype=np.int32)
    np.add.at(acc, flat, values)
    np.add.at(cnt, flat, 1)
    filled = cnt > 0
    mean = np.full(n * n, np.nan)
    mean[filled] = acc[filled] / cnt[filled]

    colors = colorize(mean.reshape(n, n), cmap="turbo").reshape(-1, 3)
    rgb = np.zeros((n * n, 3), dtype=np.uint8)
    rgb[filled] = colors[filled]
    rgb = rgb.reshape(n, n, 3)

    # Range rings every 5 m plus the robot marker, for scale.
    yy, xx = np.mgrid[0:n, 0:n]
    cx = cy = n / 2.0
    dist = np.hypot(xx - cx, yy - cy) * (2 * extent / n)
    for radius in range(5, int(extent) + 1, 5):
        ring = np.abs(dist - radius) < (extent / n)
        rgb[ring] = np.maximum(rgb[ring], [70, 70, 70])
    return _to_png(_draw_robot_marker(rgb))


def render_camera_image(path: Path, width: int = 640, quality: int = 78) -> bytes:
    """Downscaled JPEG of a camera frame (the PNGs are 1280x800 and slow to
    ship one per playback frame)."""
    from PIL import Image

    with Image.open(path) as img:
        img = img.convert("RGB")
        if width and img.width > width:
            height = max(1, round(img.height * width / img.width))
            img = img.resize((width, height), Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=False)
    return buf.getvalue()


def placeholder_png(text_hint: str = "") -> bytes:
    """A small dark tile used when a per-frame artefact is absent."""
    rgb = np.full((64, 64, 3), 24, dtype=np.uint8)
    rgb[::8, :] = 40
    rgb[:, ::8] = 40
    return _to_png(rgb)
