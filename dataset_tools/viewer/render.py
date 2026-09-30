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
from collections.abc import Sequence
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


#: Ramp for the ``driven_similarity`` tile's ``max_sim`` (0 -> 1): dark grey
#: through ember to a pale orange, so the tile cannot be mistaken for the
#: turbo-coloured height map beside it.
DRIVEN_SIM_ANCHORS = np.array(
    [
        [38, 38, 42],  # 0.0  nothing like anything driven
        [92, 52, 40],
        [168, 72, 30],
        [232, 130, 40],
        [255, 205, 120],  # 1.0  a driven cell's own feature
    ],
    dtype=np.float64,
)
#: Cells the wheels actually rolled over, one fixed colour: cyan, which the ramp
#: never produces.
DRIVEN_COLOR = np.array([40, 220, 235], dtype=np.uint8)
#: Brightening applied to cells above the sidecar's trust threshold, so the gate
#: DRIFT actually applies is visible as a step inside the smooth ramp.
DRIVEN_TRUST_BOOST = 1.18


#: HELHEST's footprint in its own frame (metres), forward = +x, left = +y.
#:
#: Deliberately duplicated from :mod:`build`'s ``CHASSIS``/``WHEELS`` rather
#: than imported: ``build`` imports *this* module, so importing back would be a
#: cycle. ``tests/test_viewer_app.py`` asserts the two stay in agreement.
#: base_link is the front axle (DRIFT's helhest_robot_model), so the chassis
#: box sits mostly behind the origin.
ROBOT_CHASSIS_HALF = (0.35, 0.25)  # half length (x), half width (y)
ROBOT_CHASSIS_CENTER = (-0.15, 0.0)  # box centre in the base frame
ROBOT_WHEEL_RADIUS = 0.334
ROBOT_WHEEL_WIDTH = 0.142
#: (x, y) of each wheel hub -- two front wheels on the axle, one rear centre wheel.
ROBOT_WHEEL_HUBS = ((0.0, 0.36), (0.0, -0.36), (-0.697, 0.0))


def _build_lut(anchors: np.ndarray, n: int = 256) -> np.ndarray:
    idx = np.linspace(0.0, len(anchors) - 1.0, n)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(anchors) - 1)
    frac = (idx - lo)[:, None]
    return (anchors[lo] * (1 - frac) + anchors[hi] * frac).astype(np.uint8)


#: The one colour map every renderer here uses; ``colorize`` falls back to it
#: for a name it does not know.
LUTS = {"turbo": _build_lut(_TURBO_ANCHORS), "ember": _build_lut(DRIVEN_SIM_ANCHORS)}


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


def render_driven_similarity(driven: np.ndarray, max_sim: np.ndarray, threshold: float, size: int = 384) -> bytes:
    """The ``driven_similarity`` tile: similarity ramp, trust gate, driven cells.

    ``max_sim`` in [0, 1] runs through the ``ember`` ramp (negative similarity
    is as unlike as zero and is shown so); cells above ``threshold`` are
    brightened by :data:`DRIVEN_TRUST_BOOST` so the step where DRIFT starts
    trusting its prediction is visible; the cells the wheels rolled over are
    painted :data:`DRIVEN_COLOR` on top of everything, and the robot silhouette
    on top of that. Robot centred, forward up, like every BEV tile.
    """
    sim = np.nan_to_num(np.asarray(max_sim, dtype=np.float64), nan=0.0)
    disp_sim = _bev_to_display(sim)
    disp_driven = _bev_to_display(np.asarray(driven, dtype=bool))
    rgb = colorize(disp_sim, vmin=0.0, vmax=1.0, cmap="ember").astype(np.float64)
    trusted = disp_sim > float(threshold)
    rgb[trusted] = np.clip(rgb[trusted] * DRIVEN_TRUST_BOOST, 0, 255)
    rgb = rgb.astype(np.uint8)
    rgb[disp_driven] = DRIVEN_COLOR
    return _bev_png(rgb, size)


def draw_robot_silhouette(img, px_per_m: float):
    """Draw HELHEST top-down at the centre of a robot-centred, forward-up tile.

    A single pixel says "the robot is here" but not how big it is or which way
    it faces -- which is what a BEV tile is read for. This draws the footprint
    to scale (0.7 x 0.5 m chassis, two front wheels on the axle at the tile's
    centre, one rear) so features can be sized against it by eye.

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
    cx0, cy0 = ROBOT_CHASSIS_CENTER
    box(cx0 - hx, cx0 + hx, cy0 - hy, cy0 + hy, (255, 255, 255, 70), (255, 255, 255, 235))

    # Heading wedge just ahead of the chassis: which end is the front is not
    # inferable from a symmetric-looking box.
    front = cx0 + hx
    nose = [pt(front + 0.30, 0.0), pt(front + 0.02, 0.16), pt(front + 0.02, -0.16)]
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
    """JPEG of a camera frame, downscaled to at most ``width`` pixels across.

    The frames are 1280x800 and one is shipped per playback frame, so the
    panel asks for them small. A root that holds the camera's own JPEG
    (``images.extension: jpg``, roots made from 2026-09-29) is served the file
    as it is whenever no downscale is needed: re-encoding it would only cost
    CPU and a generation of JPEG loss.
    """
    from PIL import Image

    with Image.open(path) as img:
        if img.format == "JPEG" and not (width and img.width > width):
            return path.read_bytes()
        img = img.convert("RGB")
        if width and img.width > width:
            height = max(1, round(img.height * width / img.width))
            img = img.resize((width, height), Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=False)
    return buf.getvalue()


#: Returns closer to the camera than this (metres, along its optical axis) are
#: not drawn: they are the robot's own body, or behind the lens.
OVERLAY_MIN_DEPTH = 0.1


def _project_points(
    cam: np.ndarray,
    K: np.ndarray,
    scale: float,
    distortion_model: str | None = None,
    distortion_coeffs: Sequence[float] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-frame 3D points (x, y, z) into pixel coordinates (u, v).

    Applies the camera's distortion model when available (e.g. equidistant fisheye
    or plumb_bob/rational_polynomial), falling back to pinhole projection when
    unspecified or unsupported.
    """
    if not len(cam):
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    K_scaled = np.asarray(K, dtype=np.float64).copy()
    K_scaled[:2, :] *= scale
    D = np.asarray(distortion_coeffs, dtype=np.float64).ravel() if distortion_coeffs is not None else None

    if distortion_model in ("equidistant", "fisheye") and D is not None and len(D) == 4:
        try:
            import cv2

            pts = cam.reshape(-1, 1, 3).astype(np.float64)
            proj, _ = cv2.fisheye.projectPoints(pts, np.zeros((3, 1)), np.zeros((3, 1)), K_scaled, D)
            proj = proj.reshape(-1, 2)
            return np.round(proj[:, 0]).astype(np.int64), np.round(proj[:, 1]).astype(np.int64)
        except ImportError:
            R = np.sqrt(cam[:, 0] ** 2 + cam[:, 1] ** 2)
            theta = np.arctan2(R, cam[:, 2])
            theta2 = theta**2
            theta_d = theta * (1.0 + D[0] * theta2 + D[1] * theta2**2 + D[2] * theta2**3 + D[3] * theta2**4)
            s = np.where(R > 1e-8, theta_d / R, 1.0)
            xp = cam[:, 0] * s
            yp = cam[:, 1] * s
            u = np.round(K_scaled[0, 0] * xp + K_scaled[0, 1] * yp + K_scaled[0, 2]).astype(np.int64)
            v = np.round(K_scaled[1, 1] * yp + K_scaled[1, 2]).astype(np.int64)
            return u, v

    if (
        distortion_model in ("plumb_bob", "rational_polynomial")
        and D is not None
        and len(D) in (4, 5, 8, 12, 14)
    ):
        try:
            import cv2

            pts = cam.reshape(-1, 1, 3).astype(np.float64)
            proj, _ = cv2.projectPoints(pts, np.zeros((3, 1)), np.zeros((3, 1)), K_scaled, D)
            proj = proj.reshape(-1, 2)
            return np.round(proj[:, 0]).astype(np.int64), np.round(proj[:, 1]).astype(np.int64)
        except ImportError:
            pass

    fx, fy, cx, cy = K_scaled[0, 0], K_scaled[1, 1], K_scaled[0, 2], K_scaled[1, 2]
    u = np.round(fx * cam[:, 0] / cam[:, 2] + cx).astype(np.int64)
    v = np.round(fy * cam[:, 1] / cam[:, 2] + cy).astype(np.int64)
    return u, v


def render_cloud_on_image(
    image_path: Path,
    cloud: np.ndarray,
    K: np.ndarray,
    T_cam_to_base: np.ndarray,
    width: int = 1280,
    quality: int = 78,
    distortion_model: str | None = None,
    distortion_coeffs: Sequence[float] | np.ndarray | None = None,
) -> bytes:
    """The frame's lidar returns projected into its camera image, as a JPEG.

    A calibration check, not a product: a road that lands beside the road, or
    a trunk that lands a metre from its picture, is a wrong ``T_cam_to_base``
    -- the kind of error that shifted every v1 cloud by 0.34 m and that the
    Odin1's own extrinsic estimate wanders by. Each return is taken from the
    base frame into the camera's optical frame with the inverse of
    ``T_cam_to_base`` (``p_base = T @ p_cam_optical``), kept if it lies in
    front of the lens, and projected through the camera model; the dots are
    coloured by depth, near to far.

    When ``distortion_model`` and ``distortion_coeffs`` are provided (from
    ``camera_info.yaml``), the camera's distortion model is applied (e.g.
    equidistant fisheye for Odin1, plumb_bob / rational_polynomial for pinhole
    lenses). If absent or unknown, uncorrected pinhole projection is used as a
    fallback. ``K`` is scaled with the image when it is downscaled to ``width``.
    """
    from PIL import Image

    with Image.open(image_path) as src:
        img = src.convert("RGB")
    scale = 1.0
    if width and img.width > width:
        scale = width / img.width
        img = img.resize((width, max(1, round(img.height * scale))), Image.BILINEAR)
    rgb = np.asarray(img).copy()
    h, w = rgb.shape[:2]

    pts = np.asarray(cloud, dtype=np.float64)
    if pts.ndim == 2 and pts.shape[1] >= 3 and len(pts):
        T = np.linalg.inv(np.asarray(T_cam_to_base, dtype=np.float64))
        cam = pts[:, :3] @ T[:3, :3].T + T[:3, 3]
        cam = cam[cam[:, 2] > OVERLAY_MIN_DEPTH]
        K = np.asarray(K, dtype=np.float64)
        u, v = _project_points(cam, K, scale, distortion_model, distortion_coeffs)
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v, z = u[inside], v[inside], cam[inside, 2]
        if len(z):
            # Far to near, so where dots overlap the nearer return is on top.
            order = np.argsort(-z)
            u, v, colors = u[order], v[order], colorize(z[order], cmap="turbo")
            # A 3x3 dot per return, painted in nine vectorised strokes rather
            # than one Pillow call per point: a cloud has tens of thousands.
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    vv, uu = v + dy, u + dx
                    ok = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
                    rgb[vv[ok], uu[ok]] = colors[ok]

    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="JPEG", quality=quality, optimize=False)
    return buf.getvalue()
