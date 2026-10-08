"""Time-series charts for a scene or segment page, on one shared time axis.

Each chart is resampled onto ``BINS`` columns (column mean, else interpolated
unless the gap exceeds ``MAX_GAP``). One chart per unit. Reads only the small
arrays cached on the scene handle.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from . import geo
from .build import WHEEL_LABELS, SceneAccess
from .loaders import roll_pitch_from_gravity_align

#: Columns a chart is resampled onto. About one per pixel of a card's width.
BINS = 600

#: Source samples further apart than this (s) are a hole, not a slow signal.
MAX_GAP = 1.5

#: The y-range covers this central share of the values, so one spike does not
#: flatten a chart; the spike is clipped at the edge and still read on hover.
Y_PERCENTILES = (0.5, 99.5)

#: A GT step faster than this (m/s) is a pose jump, not driving. Generous: the
#: gate that deleted real sprints was 4 m/s.
MAX_GT_SPEED = 8.0


def resample(t: np.ndarray, v: np.ndarray, t0: float, t1: float, n: int = BINS, how: str = "mean") -> np.ndarray:
    """``v(t)`` on ``n`` equal columns of ``[t0, t1]``; NaN across holes."""
    t, v = np.asarray(t, dtype=np.float64), np.asarray(v, dtype=np.float64)
    good = np.isfinite(t) & np.isfinite(v)
    t, v = t[good], v[good]
    out = np.full(n, np.nan)
    if len(t) == 0 or t1 <= t0:
        return out
    order = np.argsort(t, kind="stable")
    t, v = t[order], v[order]
    col = np.floor((t - t0) / (t1 - t0) * n).astype(np.int64)
    inside = (col >= 0) & (col < n)
    count = np.bincount(col[inside], minlength=n)
    if how == "max":
        agg = np.full(n, -np.inf)
        np.maximum.at(agg, col[inside], v[inside])
    else:
        agg = np.bincount(col[inside], v[inside], minlength=n) / np.maximum(count, 1)
    centres = t0 + (np.arange(n) + 0.5) * (t1 - t0) / n
    right = np.clip(np.searchsorted(t, centres), 0, len(t) - 1)
    left = np.clip(right - 1, 0, len(t) - 1)
    spanned = (t[left] <= centres) & (t[right] >= centres) & (t[right] - t[left] <= MAX_GAP)
    out = np.where(spanned, np.interp(centres, t, v), np.nan)
    return np.where(count > 0, agg, out)


def _path(values: np.ndarray, lo: float, hi: float, height: float = 100.0) -> str:
    """An SVG path over ``values`` in a ``BINS x height`` box, broken at NaN."""
    parts, pen_down = [], False
    span = hi - lo if hi > lo else 1.0
    for x, v in enumerate(values):
        if not np.isfinite(v):
            pen_down = False
            continue
        y = height - (v - lo) / span * height
        parts.append(f"{'L' if pen_down else 'M'}{x + 0.5:.1f} {y:.1f}")
        pen_down = True
    return "".join(parts)


def _fmt(v: float) -> str:
    return f"{v:.0f}" if abs(v) >= 100 else f"{v:.3g}"


def _y_range(values: list[np.ndarray]) -> tuple[float, float]:
    """The padded y-range covering ``Y_PERCENTILES`` of every finite value."""
    every = np.concatenate(values)
    every = every[np.isfinite(every)]
    if not every.size:
        return (-1.0, 1.0)
    lo, hi = (float(x) for x in np.percentile(every, Y_PERCENTILES))
    if hi - lo < 1e-9:
        lo, hi = lo - 1.0, hi + 1.0
    pad = 0.06 * (hi - lo)
    return (lo - pad, hi + pad)


def _chart(
    key: str,
    title: str,
    unit: str,
    series: list[dict],
    note: str | None = None,
    span: tuple[float, float] | None = None,
) -> dict | None:
    """A chart from ``[{label, values, cls, dash?}]``, or None if nothing in it.

    ``span`` fixes the y-range, for charts meant to be read against each other.
    """
    series = [s for s in series if np.isfinite(s["values"]).any()]
    if not series:
        return None
    lo, hi = span or _y_range([s["values"] for s in series])
    return {
        "key": key,
        "title": title,
        "unit": unit,
        "note": note,
        "lo": _fmt(lo),
        "hi": _fmt(hi),
        "zero": 100 - (0 - lo) / (hi - lo) * 100 if lo < 0 < hi else None,
        "series": [
            {"label": s["label"], "cls": s["cls"], "dash": s.get("dash", False), "d": _path(s["values"], lo, hi)}
            for s in series
        ],
        # What the hover readout reads: the same columns the lines are drawn from.
        "data": json.dumps(
            {
                "unit": unit,
                "series": [
                    {"label": s["label"], "v": [round(float(v), 3) if np.isfinite(v) else None for v in s["values"]]}
                    for s in series
                ],
            },
            separators=(",", ":"),
        ),
    }


#: Candidate spacings of the time axis's ticks (s): the first giving at most
#: ``MAX_TICKS`` gaps across the chart is used.
TICK_STEPS = (1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200)
MAX_TICKS = 6


def time_ticks(duration: float, offset: float = 0.0) -> list[dict[str, Any]]:
    """Round-numbered ticks for a time axis ``[offset, offset + duration]``.

    Each is ``{pct, label, edge}``: where it sits across the chart, what it
    reads (``m:ss`` once the axis passes a minute, plain seconds before), and
    ``l`` / ``r`` for a label too close to an edge to be centred on its tick.
    """
    if duration <= 0:
        return []
    # Past the table (one bogus stamp can span years) the step grows in whole hours: still MAX_TICKS.
    step = next((s for s in TICK_STEPS if duration / s <= MAX_TICKS), math.ceil(duration / MAX_TICKS / 3600) * 3600)
    minutes = offset + duration >= 60
    out = []
    t = np.ceil(offset / step - 1e-9) * step
    while t <= offset + duration + 1e-9:
        pct = (t - offset) / duration * 100
        sec = int(round(t))
        label = f"{sec // 60}:{sec % 60:02d}" if minutes else f"{sec} s"
        out.append({"pct": pct, "label": label, "edge": "l" if pct < 4 else ("r" if pct > 96 else "")})
        t += step
    return out


def _speed(ts: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Speed between consecutive positions, at their midpoints; sub-ms steps dropped."""
    dt = np.diff(ts)
    ok = dt > 1e-3
    return (ts[1:][ok] + ts[:-1][ok]) / 2, np.linalg.norm(np.diff(xy, axis=0), axis=1)[ok] / dt[ok]


def _gt_speed(access: SceneAccess, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Ground-truth speed, differentiated inside each segment, never across a boundary."""
    odom = access.odometry
    if odom is None:
        return None
    ts, poses = odom
    order = np.argsort(ts, kind="stable")
    ts, xy = ts[order], poses[order, :2, 3]
    spans = [(s.start_ts, s.end_ts) for s in access.segments] or [(float(ts[0]), float(ts[-1]))]
    out_t, out_v = [], []
    for s0, s1 in spans:
        if s1 < t0 or s0 > t1:
            continue
        take = (ts >= s0) & (ts <= s1)
        if take.sum() < 2:
            continue
        mid, speed = _speed(ts[take], xy[take])
        speed[speed > MAX_GT_SPEED] = np.nan
        out_t.append(mid)
        out_v.append(speed)
    if not out_t:
        return None
    return np.concatenate(out_t), np.concatenate(out_v)


def _frame_speed(access: SceneAccess) -> tuple[np.ndarray, np.ndarray] | None:
    """The robot's own odometry speed (``transforms.npy``), on the frame clock."""
    tf, ts = access.transforms, access.frame_ts
    if tf is None or ts is None or np.ndim(tf) != 3:
        return None
    n = min(len(tf), len(ts))
    if n < 2:
        return None
    return _speed(ts[:n], np.asarray(tf[:n, :2, 3]))


def signal_charts(
    access: SceneAccess, t0: float, t1: float, shade: bool = True, offset: float = 0.0
) -> dict[str, Any] | None:
    """Every chart the scene can offer over ``[t0, t1]``, or None if none.

    Also returns the frame nearest each column (what a click opens) and, with
    ``shade``, segment spans in % of the width. ``offset`` labels ``t0`` on the axis (s).
    """
    if not (np.isfinite(t0) and np.isfinite(t1)) or t1 <= t0:
        return None
    charts: list[dict | None] = []

    speed = []
    if (gt := _gt_speed(access, t0, t1)) is not None:
        speed.append({"label": "ground truth", "cls": "s1", "values": resample(*gt, t0, t1)})
    if (fs := _frame_speed(access)) is not None:
        speed.append({"label": "robot odometry", "cls": "s2", "values": resample(*fs, t0, t1)})
    cv = access.cmd_vel
    if cv is not None and np.ndim(cv) == 2 and cv.shape[1] >= 7:
        speed.append({"label": "commanded", "cls": "s3", "values": resample(cv[:, 0], np.abs(cv[:, 1]), t0, t1)})
    charts.append(
        _chart(
            "speed",
            "Ground speed",
            "m/s",
            speed,
            "Ground truth is differentiated inside each segment and never across a boundary; the robot's "
            "own odometry is continuous but drifts. Where they part, one of them is wrong.",
        )
    )

    js = access.joint_states
    if js is not None and np.ndim(js) == 2 and js.shape[1] >= 10:
        sp = access.joint_setpoints
        has_sp = sp is not None and np.ndim(sp) == 2 and sp.shape[1] >= 7
        # One chart per wheel on a shared y-range, each in its effort-chart colour.
        per_wheel = [
            [{"label": "measured", "cls": f"s{k + 1}", "values": resample(js[:, 0], js[:, 4 + k], t0, t1)}]
            + (
                [
                    {
                        "label": "commanded",
                        "cls": f"s{k + 1}",
                        "dash": True,
                        "values": resample(sp[:, 0], sp[:, 4 + k], t0, t1),
                    }
                ]
                if has_sp
                else []
            )
            for k in range(len(WHEEL_LABELS))
        ]
        span = _y_range([s["values"] for wheel in per_wheel for s in wheel])
        for k, (label, series) in enumerate(zip(WHEEL_LABELS, per_wheel, strict=True)):
            last = k == len(WHEEL_LABELS) - 1
            charts.append(
                _chart(
                    f"wheel_{label}",
                    f"{label.capitalize()} wheel speed",
                    "rad/s",
                    series,
                    (
                        (
                            "The three wheels share one scale. Measured solid, commanded dashed: a wheel running "
                            "away from its dashed line is slipping, one lagging below it is loaded."
                            if has_sp
                            else "The three wheels share one scale; measured angular velocity, in the file's units."
                        )
                        if last
                        else None
                    ),
                    span=span,
                )
            )
        charts.append(
            _chart(
                "effort",
                "Wheel effort",
                "as recorded",
                [
                    {"label": label, "cls": f"s{k + 1}", "values": resample(js[:, 0], js[:, 7 + k], t0, t1)}
                    for k, label in enumerate(WHEEL_LABELS)
                ],
                "The drive's own effort reading (current or torque, whatever the controller publishes). "
                "It climbs on slopes, in mud and against obstacles.",
            )
        )

    ga, ts = access.gravity_align, access.frame_ts
    if ga is not None and ts is not None and np.ndim(ga) == 3:
        n = min(len(ga), len(ts))
        roll, pitch = roll_pitch_from_gravity_align(np.asarray(ga[:n]))
        charts.append(
            _chart(
                "attitude",
                "Roll & pitch",
                "deg",
                [
                    {"label": "roll", "cls": "s1", "values": resample(ts[:n], np.degrees(roll), t0, t1)},
                    {"label": "pitch", "cls": "s2", "values": resample(ts[:n], np.degrees(pitch), t0, t1)},
                ],
                "From the IMU (gravity_align.npy): absolute, against gravity.",
            )
        )

    gps = access.gps
    if geo.is_track(gps) and gps.shape[1] > geo.ALT:
        charts.append(
            _chart(
                "altitude",
                "GNSS altitude",
                "m",
                [{"label": "altitude", "cls": "s1", "values": resample(gps[:, geo.T], gps[:, geo.ALT], t0, t1)}],
            )
        )
        if gps.shape[1] >= geo.COV.stop:
            horiz = gps[:, geo.COV.start] + gps[:, geo.COV.start + 1]
            sigma = np.where(horiz > 0, np.sqrt(np.abs(horiz)), np.nan)
            charts.append(
                _chart(
                    "gnss_sigma",
                    "GNSS horizontal σ",
                    "m",
                    [{"label": "σ", "cls": "s1", "values": resample(gps[:, geo.T], sigma, t0, t1, how="max")}],
                    "The receiver's own one-sigma radius; a step up is a fix lost or downgraded.",
                )
            )

    if ts is not None and len(ts) > 2:
        charts.append(
            _chart(
                "frame_dt",
                "Frame interval",
                "ms",
                [
                    {
                        "label": "interval",
                        "cls": "s1",
                        "values": resample(ts[1:], np.diff(ts) * 1000.0, t0, t1, how="max"),
                    }
                ],
                "The longest gap between synced frames in each column: a spike is frames dropped.",
            )
        )

    charts = [c for c in charts if c]
    if not charts:
        return None
    centres = t0 + (np.arange(BINS) + 0.5) * (t1 - t0) / BINS
    frames = None
    if ts is not None and len(ts):
        right = np.clip(np.searchsorted(ts, centres), 0, len(ts) - 1)
        left = np.clip(right - 1, 0, len(ts) - 1)
        frames = np.where(np.abs(ts[left] - centres) <= np.abs(ts[right] - centres), left, right).tolist()
    spans = [
        [(max(s.start_ts, t0) - t0) / (t1 - t0) * 100, (min(s.end_ts, t1) - max(s.start_ts, t0)) / (t1 - t0) * 100]
        for s in access.segments
        if shade and s.end_ts > t0 and s.start_ts < t1
    ]
    return {
        "charts": charts,
        "duration": t1 - t0,
        "offset": offset,
        "ticks": time_ticks(t1 - t0, offset),
        "bins": BINS,
        "frames": json.dumps(frames, separators=(",", ":")) if frames else None,
        "shade": spans,
    }
