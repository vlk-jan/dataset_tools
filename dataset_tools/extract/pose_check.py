"""One pose stream checked against another: ``pose_check.txt``.

From v2.3 the ground truth is the Odin1 odometry, checked against the
Fixposition GNSS/INS. Measured at ``base_link`` in both streams:

- **jumps**: steps over ``max_speed``, their count and metres carried;
- **displacement**: |distance| disagreement over 1/2/5 s jump-free windows (median, p90, scale);
- **clock**: the lag best aligning the yaw rates (positive: primary content later than its stamps);
- **tilt**: median roll/pitch difference;
- **z**: the vertical extent of each.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dataset_tools.time_align import refine
from dataset_tools.transforms import roll_pitch_from_matrix

WINDOWS_S = (1.0, 2.0, 5.0)
LAGS_S = np.arange(-0.1, 0.1001, 0.005)
#: Beyond these the scene is flagged in the log and in the report.
WARN_LAG_S = 0.030
WARN_JUMP_FRACTION = 0.10
WARN_P90_5S_M = 0.5
WARN_TILT_DEG = 3.0


def _steps(ts: np.ndarray, pos: np.ndarray, max_speed: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-step distance and the mask of steps faster than ``max_speed``."""
    d = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    dt = np.maximum(np.diff(ts), 1e-3)
    return d, d / dt > max_speed


def _yaw_rate(ts: np.ndarray, poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yaw = np.unwrap(np.arctan2(poses[:, 1, 0], poses[:, 0, 0]))
    dt = np.diff(ts)
    ok = (dt > 1e-3) & (dt < 0.5)
    return ((ts[1:] + ts[:-1]) / 2)[ok], (np.diff(yaw) / dt)[ok]


def _lag(tm_p, r_p, tm_r, r_r) -> tuple[float | None, float | None]:
    """The lag maximising the correlation of the primary's yaw rate with the reference's."""
    lo, hi = max(tm_p[0], tm_r[0]) + 0.2, min(tm_p[-1], tm_r[-1]) - 0.2
    m = (tm_p > lo) & (tm_p < hi)
    if m.sum() < 20 or np.std(r_p[m]) < 1e-3:
        return None, None
    cs = np.array([np.corrcoef(r_p[m], np.interp(tm_p[m] + L, tm_r, r_r))[0, 1] for L in LAGS_S])
    if not np.all(np.isfinite(cs)):
        return None, None
    return refine(LAGS_S, -cs), float(cs.max())


def _at(ts: np.ndarray, pos: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(q, ts, pos[:, i]) for i in range(3)])


def compare(ts_p, poses_p, ts_r, poses_r, max_speed: float, step_s: float = 0.5) -> dict:
    """``primary`` against ``reference``; see the module docstring for the keys."""
    ts_p, ts_r = np.asarray(ts_p, float), np.asarray(ts_r, float)
    poses_p, poses_r = np.asarray(poses_p, float), np.asarray(poses_r, float)
    out: dict = {
        "primary_span_s": round(float(ts_p[-1] - ts_p[0]), 1) if len(ts_p) else 0.0,
        "reference_span_s": round(float(ts_r[-1] - ts_r[0]), 1) if len(ts_r) else 0.0,
    }
    if len(ts_p) < 3 or len(ts_r) < 3:
        out["note"] = "too few poses to compare"
        return out
    pos_p, pos_r = poses_p[:, :3, 3], poses_r[:, :3, 3]
    d_p, jump_p = _steps(ts_p, pos_p, max_speed)
    d_r, jump_r = _steps(ts_r, pos_r, max_speed)
    out["primary_path_m"] = round(float(d_p.sum()), 1)
    out["reference_path_m"] = round(float(d_r.sum()), 1)
    out["primary_jumps"] = {"steps": int(jump_p.sum()), "metres": round(float(d_p[jump_p].sum()), 1)}
    out["reference_jumps"] = {"steps": int(jump_r.sum()), "metres": round(float(d_r[jump_r].sum()), 1)}
    out["primary_z_range_m"] = round(float(np.ptp(pos_p[:, 2])), 2)
    out["reference_z_range_m"] = round(float(np.ptp(pos_r[:, 2])), 2)

    lo, hi = max(ts_p[0], ts_r[0]), min(ts_p[-1], ts_r[-1])
    out["overlap_s"] = round(float(max(hi - lo, 0.0)), 1)
    jumps = np.r_[ts_p[1:][jump_p], ts_r[1:][jump_r]]
    disp = {}
    for w in WINDOWS_S:
        q0 = np.arange(lo, hi - w, step_s)
        if not len(q0):
            continue
        q1 = q0 + w
        clean = np.array([not np.any((jumps > a) & (jumps < b)) for a, b in zip(q0, q1, strict=True)])
        dp = np.linalg.norm(_at(ts_p, pos_p, q1) - _at(ts_p, pos_p, q0), axis=1)
        dr = np.linalg.norm(_at(ts_r, pos_r, q1) - _at(ts_r, pos_r, q0), axis=1)
        use = clean & (dr > 0.2)
        if use.sum() < 5:
            continue
        err = np.abs(dp - dr)[use]
        disp[f"{w:g}s"] = {
            "windows": int(use.sum()),
            "median_cm": round(float(np.median(err) * 100), 1),
            "p90_cm": round(float(np.percentile(err, 90) * 100), 1),
            "scale": round(float(np.median(dp[use] / dr[use])), 3),
        }
    out["displacement"] = disp

    tm_p, r_p = _yaw_rate(ts_p, poses_p)
    tm_r, r_r = _yaw_rate(ts_r, poses_r)
    lag, corr = (None, None) if len(tm_p) < 20 or len(tm_r) < 20 else _lag(tm_p, r_p, tm_r, r_r)
    out["clock_lag_s"] = None if lag is None else round(lag, 4)
    out["clock_corr"] = None if corr is None else round(corr, 3)

    m = (ts_p >= lo) & (ts_p <= hi)
    if m.sum() >= 3:
        j = np.clip(np.searchsorted(ts_r, ts_p[m]), 0, len(ts_r) - 1)
        diff = np.degrees(
            np.median(np.subtract(roll_pitch_from_matrix(poses_p[m]), roll_pitch_from_matrix(poses_r[j])), axis=1)
        )
        out["tilt_diff_deg"] = {"roll": round(float(diff[0]), 2), "pitch": round(float(diff[1]), 2)}
    out["warnings"] = warnings(out)
    return out


def warnings(result: dict) -> list[str]:
    w = []
    lag = result.get("clock_lag_s")
    if lag is not None and abs(lag) > WARN_LAG_S:
        w.append(f"clock lag {lag:+.3f} s against the reference")
    jumps = result.get("primary_jumps") or {}
    path = result.get("primary_path_m") or 0.0
    if path and jumps.get("metres", 0.0) > WARN_JUMP_FRACTION * path:
        w.append(f"{jumps['steps']} jumps carry {jumps['metres']} of {path} m of path")
    five = (result.get("displacement") or {}).get("5s") or {}
    if five.get("p90_cm", 0.0) > WARN_P90_5S_M * 100:
        w.append(f"5 s displacement disagrees by {five['p90_cm']} cm at p90")
    tilt = result.get("tilt_diff_deg") or {}
    if any(abs(v) > WARN_TILT_DEG for v in tilt.values()):
        w.append(f"tilt differs by roll {tilt.get('roll')} / pitch {tilt.get('pitch')} deg")
    return w


def write_report(path: Path | str, result: dict, primary: str, reference: str) -> None:
    lines = [
        f"# {primary} (the odometry this scene is built on) checked against {reference}",
        f"# spans: primary {result.get('primary_span_s')} s, reference {result.get('reference_span_s')} s, "
        f"overlap {result.get('overlap_s')} s",
    ]
    if "note" in result:
        lines.append(f"# {result['note']}")
    else:
        pj, rj = result["primary_jumps"], result["reference_jumps"]
        lines += [
            f"# path: primary {result['primary_path_m']} m ({pj['steps']} jumps carrying {pj['metres']} m), "
            f"reference {result['reference_path_m']} m ({rj['steps']} jumps carrying {rj['metres']} m)",
            f"# z range: primary {result['primary_z_range_m']} m, reference {result['reference_z_range_m']} m",
            f"# clock lag of the primary against the reference: {result['clock_lag_s']} s (r {result['clock_corr']})",
            f"# tilt difference (roll, pitch): {result.get('tilt_diff_deg')} deg",
            "# window_s  windows  |displacement| diff median_cm  p90_cm  scale",
        ]
        lines += [
            f"{k:>8s}  {v['windows']:7d}  {v['median_cm']:26.1f}  {v['p90_cm']:6.1f}  {v['scale']:.3f}"
            for k, v in result["displacement"].items()
        ]
        lines += [f"# WARNING: {w}" for w in result.get("warnings", [])]
    Path(path).write_text("\n".join(lines) + "\n")
