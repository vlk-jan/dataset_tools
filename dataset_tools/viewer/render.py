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

from .loaders import HM_RES

logger = logging.getLogger("dataset_tools.viewer.render")

# Anchor colours (0-255) for a few perceptually reasonable ramps.
_TURBO_ANCHORS = np.array(
    [
        [48, 18, 59],
        [70, 107, 227],
        [37, 180, 248],
        [70, 227, 161],
        [166, 249, 76],
        [231, 214, 46],
        [253, 141, 39],
        [220, 47, 12],
        [122, 4, 3],
    ],
    dtype=np.float64,
)
#: Fixed, meaningful colours for the height-map validity states.
#:
#: Anchored reads as a cousin of extrapolated rather than of measured, because
#: that is what it is: a cell the lidar never saw, filled from the robot's own
#: wheel contacts instead of from a distant return. Warm = invented, green =
#: measured; see :data:`loaders.VALIDITY_RANK`.
VALIDITY_COLORS = np.array(
    [
        [200, 60, 60],  # 0 extrapolated -- invented
        [230, 190, 70],  # 1 interpolated -- inside the hull, no return
        [60, 170, 90],  # 2 measured -- a lidar return landed here
        [150, 90, 190],  # 3 anchored -- the robot's own wheel contacts
    ],
    dtype=np.uint8,
)


#: HELHEST's footprint in its own frame (metres), forward = +x, left = +y.
#:
#: Deliberately duplicated from :mod:`build`'s ``CHASSIS``/``WHEELS`` rather
#: than imported: ``build`` imports *this* module, so importing back would be a
#: cycle. ``tests/test_viewer_app.py`` asserts the two stay in agreement.
ROBOT_CHASSIS_HALF = (0.5, 0.3)  # half length (x), half width (y)
ROBOT_WHEEL_RADIUS = 0.3685
ROBOT_WHEEL_WIDTH = 0.142
#: (x, y) of each wheel hub -- two front wheels and one rear centre wheel.
ROBOT_WHEEL_HUBS = ((0.4, 0.52), (0.4, -0.52), (-0.5, 0.0))


def _build_lut(anchors: np.ndarray, n: int = 256) -> np.ndarray:
    idx = np.linspace(0.0, len(anchors) - 1.0, n)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(anchors) - 1)
    frac = (idx - lo)[:, None]
    return (anchors[lo] * (1 - frac) + anchors[hi] * frac).astype(np.uint8)


#: The one colour map every renderer here uses; ``colorize`` falls back to it
#: for a name it does not know.
LUTS = {"turbo": _build_lut(_TURBO_ANCHORS)}


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

    ``grid[i, j]`` indexes x by ``i`` and y by ``j``, both increasing, so row
    ``r`` maps to ``x = N-1-r`` and column ``c`` to ``y = N-1-c``: a double flip
    and **no** transpose. Transposing rotates the map 90 degrees, which shows up
    as the lidar's cone pointing sideways.
    """
    return grid[::-1, ::-1]


def _bev_png(rgb: np.ndarray, size: int, res_m: float = HM_RES) -> bytes:
    """Encode a robot-centred BEV grid, upscaled, with the robot drawn on top.

    The silhouette is drawn *after* the nearest-neighbour upscale so it is
    smooth at display resolution while the data cells stay honestly chunky.
    """
    from PIL import Image

    img = Image.fromarray(np.ascontiguousarray(rgb), mode="RGB")
    scale = max(1, size // max(img.width, img.height))
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    draw_robot_silhouette(img, px_per_m=scale / res_m)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=3)
    return buf.getvalue()


def render_height_map(hm: np.ndarray, size: int = 384) -> bytes:
    """Colour-mapped BEV height map, robot centred, forward up."""
    disp = _bev_to_display(np.asarray(hm, dtype=np.float64))
    return _bev_png(colorize(disp, cmap="turbo"), size)


def render_validity(mask: np.ndarray, size: int = 384) -> bytes:
    """Three-state validity mask in fixed colours (see VALIDITY_COLORS)."""
    disp = _bev_to_display(np.asarray(mask))
    idx = np.clip(disp.astype(np.int32), 0, len(VALIDITY_COLORS) - 1)
    return _bev_png(VALIDITY_COLORS[idx], size)


def draw_robot_silhouette(img, px_per_m: float):
    """Draw HELHEST top-down at the centre of a robot-centred, forward-up tile.

    A single pixel says "the robot is here" but not how big it is or which way
    it faces -- which is what a BEV tile is read for. This draws the footprint
    to scale (1.0 x 0.6 m chassis, two front wheels, one rear) so features can
    be sized against it by eye.

    ``px_per_m`` is the tile's scale *after* upscaling, so this draws at display
    resolution rather than in chunky grid cells. Translucent with a thin opaque
    outline, to stay readable over both dark and bright colour maps.
    """
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img, "RGBA")
    cx, cy = img.width / 2.0, img.height / 2.0

    def pt(x: float, y: float) -> tuple[float, float]:
        # Screen: +x (forward) is up, +y (left) is left.
        return (cx - y * px_per_m, cy - x * px_per_m)

    def box(x0: float, x1: float, y0: float, y1: float, fill, outline, width=1):
        draw.polygon([pt(x0, y0), pt(x1, y0), pt(x1, y1), pt(x0, y1)], fill=fill)
        draw.line(
            [pt(x0, y0), pt(x1, y0), pt(x1, y1), pt(x0, y1), pt(x0, y0)],
            fill=outline,
            width=width,
        )

    r, half_w = ROBOT_WHEEL_RADIUS, ROBOT_WHEEL_WIDTH / 2.0
    for wx, wy in ROBOT_WHEEL_HUBS:
        box(wx - r, wx + r, wy - half_w, wy + half_w, (0, 0, 0, 150), (255, 255, 255, 210))

    hx, hy = ROBOT_CHASSIS_HALF
    box(-hx, hx, -hy, hy, (255, 255, 255, 70), (255, 255, 255, 235))

    # Heading wedge just ahead of the chassis: which end is the front is not
    # inferable from a symmetric-looking box.
    nose = [pt(hx + 0.30, 0.0), pt(hx + 0.02, 0.16), pt(hx + 0.02, -0.16)]
    draw.polygon(nose, fill=(255, 255, 255, 235))
    return img


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
    from PIL import Image

    img = Image.fromarray(np.ascontiguousarray(rgb), mode="RGB")
    draw_robot_silhouette(img, px_per_m=n / (2 * extent))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=3)
    return buf.getvalue()


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
