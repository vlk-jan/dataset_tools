"""Turning a scene's GNSS fixes into something a map can draw.

Pure functions over the ``(M, >= 4)`` array ``gps.npy`` holds -- no I/O and no
file knowledge -- so the map's arithmetic is testable without a dataset.

Coordinates stay in degrees from end to end: the front end wants latitude and
longitude, and projecting here would only mean unprojecting there. Distances
use the local equirectangular approximation (a degree of latitude is ~111.3 km
everywhere; a degree of longitude is that times ``cos(lat)``), which is good to
a fraction of a percent over the few kilometres one recording covers and needs
no projection library.

Nothing here is ground truth. A fix says where the robot was on the Earth; the
scene's poses say where it was relative to its own segment anchor, and on the
old rig the two do not even come from the same device. The map is *provenance*,
which is why it is never mixed into a window bundle.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

#: Column layout of ``gps.npy`` (see the extractor's ``gps_row``).
T, LAT, LON, ALT, STATUS = 0, 1, 2, 3, 4
COV = slice(5, 8)

#: Metres per degree of latitude. Longitude scales this by ``cos(lat)``.
M_PER_DEG = 111_320.0

#: ``sensor_msgs/NavSatStatus.status`` in words. Both rigs' receivers use the
#: standard constants, so this needs no per-rig knowledge; an unknown value is
#: reported as its own number rather than dropped.
FIX_LABELS = {-2: "unknown", -1: "no fix", 0: "single", 1: "SBAS", 2: "RTK / GBAS"}

#: How far from a fix a timestamp may sit and still be located by it. Both rigs
#: publish at 10 Hz, so a second of silence means the receiver dropped out, not
#: that the robot was somewhere between two samples.
MAX_INTERP_GAP = 1.0


def is_track(gps: np.ndarray | None) -> bool:
    """Whether an array is usable as a GNSS track at all."""
    return gps is not None and gps.ndim == 2 and gps.shape[0] > 0 and gps.shape[1] > LON


def slice_time(gps: np.ndarray, t0: float | None = None, t1: float | None = None) -> np.ndarray:
    """The fixes inside ``[t0, t1]`` (either end optional), in file order."""
    keep = np.ones(len(gps), dtype=bool)
    if t0 is not None:
        keep &= gps[:, T] >= t0
    if t1 is not None:
        keep &= gps[:, T] <= t1
    return gps[keep]


def decimate(track: np.ndarray, max_points: int) -> np.ndarray:
    """At most ``max_points`` fixes, evenly spaced, both ends kept.

    An even stride rather than a shape-preserving simplification: a 10 Hz track
    thinned to one sample every half second still draws every corner a robot
    driving at walking pace actually turned, and the arithmetic is obvious
    enough to trust without a tolerance to tune.
    """
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
    """Median horizontal standard deviation, in metres, or None.

    ``sqrt(cov_ee + cov_nn)``: the radius of the one-sigma circle around a fix,
    which is the number a person reads as "how well does it know where it is".
    Fixes whose covariance the receiver called unknown carry NaN and drop out.
    """
    if not len(track) or track.shape[1] < COV.stop:
        return None
    horizontal = track[:, COV.start] + track[:, COV.start + 1]
    good = horizontal[np.isfinite(horizontal) & (horizontal > 0)]
    return float(np.sqrt(np.median(good))) if good.size else None


def locate(gps: np.ndarray, t: float, max_gap: float = MAX_INTERP_GAP) -> list[float] | None:
    """``[lat, lon]`` at time ``t``, interpolated between the fixes around it.

    None when ``t`` sits outside the track or in a gap longer than ``max_gap``:
    a straight line drawn across a receiver dropout is a claim about where the
    robot went, and this file does not know that.
    """
    if not is_track(gps) or not math.isfinite(t):
        return None
    ts = gps[:, T]
    i = int(np.searchsorted(ts, t))
    if i == 0:
        return [float(gps[0, LAT]), float(gps[0, LON])] if abs(ts[0] - t) <= max_gap else None
    if i >= len(ts):
        return [float(gps[-1, LAT]), float(gps[-1, LON])] if abs(t - ts[-1]) <= max_gap else None
    t0, t1 = float(ts[i - 1]), float(ts[i])
    if t1 - t0 > max_gap:
        return None
    w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
    lat = float(gps[i - 1, LAT] + w * (gps[i, LAT] - gps[i - 1, LAT]))
    lon = float(gps[i - 1, LON] + w * (gps[i, LON] - gps[i - 1, LON]))
    return [lat, lon]


def points(track: np.ndarray) -> list[list[float]]:
    """The track as ``[[lat, lon], ...]``, rounded to ~1 cm.

    Seven decimals is about a centimetre of latitude -- far finer than any of
    these receivers -- and halves the JSON a long scene ships.
    """
    return [[round(float(lat), 7), round(float(lon), 7)] for lat, lon in track[:, [LAT, LON]]]


def summarise(track: np.ndarray, max_points: int) -> dict[str, Any]:
    """Everything one drawn track needs: its points, its extent, its quality."""
    thinned = decimate(track, max_points)
    ew, ns = span_m(track)
    return {
        "points": points(thinned),
        "n_points": int(len(thinned)),
        "n_fixes": int(len(track)),
        "bounds": bounds(track),
        "start": [round(float(track[0, LAT]), 7), round(float(track[0, LON]), 7)] if len(track) else None,
        "end": [round(float(track[-1, LAT]), 7), round(float(track[-1, LON]), 7)] if len(track) else None,
        "t0": float(track[0, T]) if len(track) else None,
        "t1": float(track[-1, T]) if len(track) else None,
        "length_m": round(track_length_m(track), 1),
        "span_m": [round(ew, 1), round(ns, 1)],
        "fix": [{"label": label, "share": round(share, 4)} for label, share in fix_shares(track)],
        "sigma_m": round(sigma, 2) if (sigma := sigma_m(track)) is not None else None,
        "alt_m": (
            [round(float(np.nanmin(track[:, ALT])), 1), round(float(np.nanmax(track[:, ALT])), 1)]
            if len(track) and track.shape[1] > ALT and np.isfinite(track[:, ALT]).any()
            else None
        ),
    }
