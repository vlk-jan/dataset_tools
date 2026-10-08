"""Pure functions turning a scene's ``gps.npy`` into something a map can draw.

Coordinates stay in degrees; distances use the local equirectangular
approximation. GNSS is provenance, not ground truth, so it never enters a window bundle.
"""

from __future__ import annotations

import math

import numpy as np

#: Column layout of ``gps.npy`` (see the extractor's ``gps_row``).
T, LAT, LON, ALT, STATUS = 0, 1, 2, 3, 4
COV = slice(5, 8)

#: Metres per degree of latitude. Longitude scales this by ``cos(lat)``.
M_PER_DEG = 111_320.0

#: ``sensor_msgs/NavSatStatus.status`` in words; unknown values show as their number.
FIX_LABELS = {-2: "unknown", -1: "no fix", 0: "single", 1: "SBAS", 2: "RTK / GBAS"}

#: Max seconds from a fix a timestamp may sit and still be located by it
#: (both rigs publish at 10 Hz, so a longer gap is a dropout).
MAX_INTERP_GAP = 1.0


def is_track(gps: np.ndarray | None) -> bool:
    """Whether an array is usable as a GNSS track at all."""
    return gps is not None and gps.ndim == 2 and gps.shape[0] > 0 and gps.shape[1] > LON


def slice_time(gps: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """The fixes inside ``[t0, t1]``, in file order."""
    return gps[(gps[:, T] >= t0) & (gps[:, T] <= t1)]


def decimate(track: np.ndarray, max_points: int) -> np.ndarray:
    """At most ``max_points`` fixes, evenly spaced, both ends kept."""
    if max_points <= 0 or len(track) <= max_points:
        return track
    step = -(-len(track) // max_points)  # ceil
    out = track[::step]
    if not np.array_equal(out[-1], track[-1]):
        out = np.vstack([out, track[-1]])
    return out


def bounds(track: np.ndarray) -> list[list[float]] | None:
    """``[[south, west], [north, east]]`` -- Leaflet's own bounds order."""
    if not len(track):
        return None
    lat, lon = track[:, LAT], track[:, LON]
    return [[float(lat.min()), float(lon.min())], [float(lat.max()), float(lon.max())]]


def track_length_m(track: np.ndarray) -> float:
    """Distance along the track, in metres."""
    if len(track) < 2:
        return 0.0
    lat, lon = track[:, LAT], track[:, LON]
    mid = np.radians(0.5 * (lat[1:] + lat[:-1]))
    dy = np.diff(lat) * M_PER_DEG
    dx = np.diff(lon) * M_PER_DEG * np.cos(mid)
    return float(np.hypot(dx, dy).sum())


def span_m(track: np.ndarray) -> tuple[float, float]:
    """Extent of the bounding box, ``(east-west, north-south)`` in metres."""
    if not len(track):
        return (0.0, 0.0)
    lat, lon = track[:, LAT], track[:, LON]
    ns = float(lat.max() - lat.min()) * M_PER_DEG
    ew = float(lon.max() - lon.min()) * M_PER_DEG * math.cos(math.radians(float(lat.mean())))
    return (ew, ns)


def fix_shares(track: np.ndarray) -> list[tuple[str, float]]:
    """Fraction of the track spent at each fix quality, commonest first."""
    if not len(track) or track.shape[1] <= STATUS:
        return []
    values, counts = np.unique(track[:, STATUS], return_counts=True)
    total = counts.sum()
    shares = [
        (FIX_LABELS.get(int(v), f"status {int(v)}"), float(n / total)) for v, n in zip(values, counts, strict=True)
    ]
    return sorted(shares, key=lambda kv: -kv[1])


def sigma_m(track: np.ndarray) -> float | None:
    """Median horizontal one-sigma radius ``sqrt(cov_ee + cov_nn)`` in metres, or None.

    Fixes with unknown (NaN) covariance drop out.
    """
    if not len(track) or track.shape[1] < COV.stop:
        return None
    horizontal = track[:, COV.start] + track[:, COV.start + 1]
    good = horizontal[np.isfinite(horizontal) & (horizontal > 0)]
    return float(np.sqrt(np.median(good))) if good.size else None


def locate(gps: np.ndarray, t: float) -> list[float] | None:
    """``[lat, lon]`` at time ``t``, interpolated between the fixes around it.

    None when ``t`` sits outside the track or in a gap longer than
    :data:`MAX_INTERP_GAP`; a dropout is not interpolated across.
    """
    if not is_track(gps) or not math.isfinite(t):
        return None
    ts = gps[:, T]
    i = int(np.searchsorted(ts, t))
    if i == 0:
        return [float(gps[0, LAT]), float(gps[0, LON])] if abs(ts[0] - t) <= MAX_INTERP_GAP else None
    if i >= len(ts):
        return [float(gps[-1, LAT]), float(gps[-1, LON])] if abs(t - ts[-1]) <= MAX_INTERP_GAP else None
    t0, t1 = float(ts[i - 1]), float(ts[i])
    if t1 - t0 > MAX_INTERP_GAP:
        return None
    w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
    lat = float(gps[i - 1, LAT] + w * (gps[i, LAT] - gps[i - 1, LAT]))
    lon = float(gps[i - 1, LON] + w * (gps[i, LON] - gps[i - 1, LON]))
    return [lat, lon]


def points(track: np.ndarray) -> list[list[float]]:
    """The track as ``[[lat, lon], ...]``, rounded to 7 decimals (~1 cm)."""
    return [[round(float(lat), 7), round(float(lon), 7)] for lat, lon in track[:, [LAT, LON]]]
