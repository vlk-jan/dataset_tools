"""Colour maps and small PNG/JPEG renderers for the per-frame side panel.

Rendered on demand, one frame at a time. No matplotlib: colour maps are small
piecewise-linear LUTs built at import.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from . import remote
from .loaders import CHASSIS, CHASSIS_CENTER, HM_RES, WHEEL_RADIUS, WHEEL_WIDTH, WHEELS

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
#: Fixed colours for the validity states: warm = invented, green = measured.
#: Anchored is coloured as a cousin of extrapolated (see :data:`loaders.VALIDITY_RANK`).
VALIDITY_COLORS = np.array(
    [
        [200, 60, 60],  # 0 extrapolated -- invented
        [230, 190, 70],  # 1 interpolated -- inside the hull, no return
        [60, 170, 90],  # 2 measured -- a lidar return landed here
        [150, 90, 190],  # 3 anchored -- the robot's own wheel contacts
    ],
    dtype=np.uint8,
)


#: Ramp for ``driven_similarity``'s ``max_sim`` (0 -> 1), distinct from turbo.
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
#: Cells the wheels rolled over: cyan, which the ramp never produces.
DRIVEN_COLOR = np.array([40, 220, 235], dtype=np.uint8)
#: Brightening of cells above the trust threshold, so DRIFT's gate shows as a step.
DRIVEN_TRUST_BOOST = 1.18


def _build_lut(anchors: np.ndarray, n: int = 256) -> np.ndarray:
    idx = np.linspace(0.0, len(anchors) - 1.0, n)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(anchors) - 1)
    frac = (idx - lo)[:, None]
    return (anchors[lo] * (1 - frac) + anchors[hi] * frac).astype(np.uint8)


#: Named LUTs for ``colorize``.
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
    lut = LUTS[cmap]
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


def unit_colors(values: np.ndarray) -> np.ndarray:
    """:func:`colorize` as float RGB in [0, 1], for SimView's point-cloud ``color``."""
    return colorize(values).astype(np.float32) / 255.0


# --------------------------------------------------------------------------
# PNG / JPEG encoders
# --------------------------------------------------------------------------


#: JPEG quality of the camera and overlay renders.
JPEG_QUALITY = 78


def _png(rgb: np.ndarray) -> bytes:
    return cv2.imencode(".png", np.ascontiguousarray(rgb[..., ::-1]), [cv2.IMWRITE_PNG_COMPRESSION, 3])[1].tobytes()


def _jpeg(bgr: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    return cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def _decode(data: bytes) -> np.ndarray:
    """An image file's bytes as 8-bit BGR."""
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("not a decodable image")
    return bgr


def _shrink(bgr: np.ndarray, width: int) -> np.ndarray:
    """``bgr`` downscaled to at most ``width`` pixels across; INTER_AREA antialiases."""
    h, w = bgr.shape[:2]
    if not width or w <= width:
        return bgr
    return cv2.resize(bgr, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)


def _bev_to_display(grid: np.ndarray) -> np.ndarray:
    """Orient a ``[row=x forward, col=y left]`` BEV grid for screen display.

    Forward up, left left: a double flip and **no** transpose (a transpose
    rotates the map 90 degrees).
    """
    return grid[::-1, ::-1]


def _bev_png(rgb: np.ndarray, size: int) -> bytes:
    """Encode a robot-centred BEV grid, upscaled, with the robot drawn on top.

    The silhouette is drawn after the nearest-neighbour upscale, at display resolution.
    """
    scale = max(1, size // max(rgb.shape[:2]))
    if scale > 1:
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return _png(draw_robot_silhouette(np.ascontiguousarray(rgb), px_per_m=scale / HM_RES))


def render_height_map(hm: np.ndarray, size: int = 384) -> bytes:
    """Colour-mapped BEV height map, robot centred, forward up."""
    disp = _bev_to_display(np.asarray(hm, dtype=np.float64))
    return _bev_png(colorize(disp, cmap="turbo"), size)


def render_validity(mask: np.ndarray, size: int = 384) -> bytes:
    """Four-state validity mask in fixed colours (see VALIDITY_COLORS)."""
    disp = _bev_to_display(np.asarray(mask))
    idx = np.clip(disp.astype(np.int32), 0, len(VALIDITY_COLORS) - 1)
    return _bev_png(VALIDITY_COLORS[idx], size)


def render_driven_similarity(driven: np.ndarray, max_sim: np.ndarray, threshold: float, size: int = 384) -> bytes:
    """The ``driven_similarity`` tile: similarity ramp, trust gate, driven cells.

    ``max_sim`` on the ``ember`` ramp (negatives clip to 0), cells above
    ``threshold`` brightened, driven cells painted :data:`DRIVEN_COLOR` on top.
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


def draw_robot_silhouette(rgb: np.ndarray, px_per_m: float) -> np.ndarray:
    """Draw HELHEST top-down, in place, at the centre of a robot-centred, forward-up tile.

    Footprint to scale, with a heading wedge. ``px_per_m`` is the scale after upscaling.
    """
    cy, cx = rgb.shape[0] / 2.0, rgb.shape[1] / 2.0

    def poly(corners, rgba, outline: bool = False) -> None:
        # Screen: +x (forward) is up, +y (left) is left.
        pts = np.round([(cx - y * px_per_m, cy - x * px_per_m) for x, y in corners]).astype(np.int32)
        layer = rgb.copy()
        if outline:
            cv2.polylines(layer, [pts], True, rgba[:3], 1)
        else:
            cv2.fillPoly(layer, [pts], rgba[:3])
        cv2.addWeighted(layer, rgba[3] / 255, rgb, 1 - rgba[3] / 255, 0, dst=rgb)

    def box(x0: float, x1: float, y0: float, y1: float, fill, outline) -> None:
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        poly(corners, fill)
        poly(corners, outline, outline=True)

    r, half_w = WHEEL_RADIUS, WHEEL_WIDTH / 2.0
    for wx, wy, *_ in WHEELS.values():
        box(wx - r, wx + r, wy - half_w, wy + half_w, (0, 0, 0, 150), (255, 255, 255, 210))

    hx, hy = CHASSIS["hx"], CHASSIS["hy"]
    cx0, cy0, _ = CHASSIS_CENTER
    box(cx0 - hx, cx0 + hx, cy0 - hy, cy0 + hy, (255, 255, 255, 70), (255, 255, 255, 235))

    # Heading wedge: the box alone does not show which end is the front.
    front = cx0 + hx
    poly([(front + 0.30, 0.0), (front + 0.02, 0.16), (front + 0.02, -0.16)], (255, 255, 255, 235))
    return rgb


#: Half the side of the cloud tile (m): it shows ``[-12, 12]`` m about the robot.
CLOUD_BEV_EXTENT = 12.0


def render_cloud_bev(cloud: np.ndarray, size: int = 384) -> bytes:
    """Top-down scatter of one raw cloud, in the robot's own frame, coloured by z.

    Binned in one pass; empty cells stay dark, so the blind wedge is visible.
    """
    extent = CLOUD_BEV_EXTENT
    cloud = np.asarray(cloud)
    n = size
    rgb = np.zeros((n, n, 3), dtype=np.uint8)
    if cloud.size == 0:
        return _png(rgb)

    x, y, z = cloud[:, 0], cloud[:, 1], cloud[:, 2]
    keep = (np.abs(x) < extent) & (np.abs(y) < extent)
    if not keep.any():
        return _png(rgb)
    x, y, z = x[keep], y[keep], z[keep]

    # Screen: +x forward -> up (row 0 at top), +y left -> left (col 0 at left).
    row = np.clip(((extent - x) / (2 * extent) * n).astype(np.int32), 0, n - 1)
    col = np.clip(((extent - y) / (2 * extent) * n).astype(np.int32), 0, n - 1)
    flat = row * n + col

    acc = np.zeros(n * n, dtype=np.float64)
    cnt = np.zeros(n * n, dtype=np.int32)
    np.add.at(acc, flat, z)
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
    return _png(draw_robot_silhouette(rgb, px_per_m=n / (2 * extent)))


def render_camera_image(path: Path, width: int = 640, quality: int = JPEG_QUALITY) -> bytes:
    """JPEG of a camera frame, downscaled to at most ``width`` pixels across.

    A JPEG source that needs no downscale is served as is (no re-encode loss).
    """
    data = remote.local(path).read_bytes()
    bgr = _decode(data)
    if data[:3] == b"\xff\xd8\xff" and not (width and bgr.shape[1] > width):
        return data
    return _jpeg(_shrink(bgr, width), quality)


#: Returns closer than this (m, along the optical axis) are the robot itself or behind the lens.
OVERLAY_MIN_DEPTH = 0.1


def _project_points(
    cam: np.ndarray,
    K: np.ndarray,
    scale: float,
    distortion_model: str | None = None,
    distortion_coeffs: Sequence[float] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-frame 3D points (x, y, z) into pixel coordinates (u, v).

    Uses the distortion model when known (fisheye, plumb_bob, rational_polynomial),
    otherwise plain pinhole.
    """
    if not len(cam):
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    K_scaled = np.asarray(K, dtype=np.float64).copy()
    K_scaled[:2, :] *= scale
    D = np.asarray(distortion_coeffs, dtype=np.float64).ravel() if distortion_coeffs is not None else None

    if distortion_model in ("equidistant", "fisheye") and D is not None and len(D) == 4:
        project = cv2.fisheye.projectPoints
    else:
        project = cv2.projectPoints
        if distortion_model not in ("plumb_bob", "rational_polynomial") or D is None or len(D) not in (4, 5, 8, 12, 14):
            D = None  # plain pinhole
    pts = cam.reshape(-1, 1, 3).astype(np.float64)
    proj, _ = project(pts, np.zeros((3, 1)), np.zeros((3, 1)), K_scaled, D)
    proj = proj.reshape(-1, 2)
    return np.round(proj[:, 0]).astype(np.int64), np.round(proj[:, 1]).astype(np.int64)


def render_cloud_on_image(
    image_path: Path,
    cloud: np.ndarray,
    K: np.ndarray,
    T_cam_to_base: np.ndarray,
    width: int = 1280,
    distortion_model: str | None = None,
    distortion_coeffs: Sequence[float] | np.ndarray | None = None,
) -> tuple[bytes, tuple[float, float] | None]:
    """The frame's lidar returns projected into its camera image, as a JPEG.

    A calibration check for ``T_cam_to_base`` (``p_base = T @ p_cam_optical``);
    dots coloured by depth. Returns the JPEG and the colour depth span (2nd-98th
    percentile, m), or ``None`` when no return landed.
    """
    full = _decode(remote.local(image_path).read_bytes())
    bgr = _shrink(full, width)
    scale = bgr.shape[1] / full.shape[1]
    h, w = bgr.shape[:2]

    span = None
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
            # Computed here so the legend can report it.
            near, far = (float(q) for q in np.percentile(z, (2, 98)))
            far = max(far, near + 0.1)
            u, v, colors = u[order], v[order], colorize(z[order], near, far, cmap="turbo")
            colors = colors[:, ::-1]  # BGR
            # 3x3 dots in nine vectorised strokes, not one draw call per point.
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    vv, uu = v + dy, u + dx
                    ok = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
                    bgr[vv[ok], uu[ok]] = colors[ok]
            span = (near, far)

    return _jpeg(bgr), span
