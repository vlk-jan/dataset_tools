#!/usr/bin/env python3
"""Compare the regenerated HELHEST v1.2 root against the published v1 root.

    compare_v1_v12.py <old_root> <new_root> <out.csv> [scenes...]

Per scene: odometry coverage (old file vs new filtered file, fraction of
data_ts frames inside a covered span), 2 s local consistency of each
trajectory against the live /odom (old = published file conjugated to
base_link, i.e. the level-1 fix), motion while the wheels were stopped,
yaw-rate correlation with the wheel-speed differential, 10 s relative-pose
error between old-conjugated and new on their overlap, and cloud statistics
(points per frame, x offset of the zero-return blob / centroid shift).
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the package, when it is not installed
from dataset_tools.gt_io import read_segments, read_trajectory  # noqa: E402

X = np.eye(4)
X[:3, :3] = np.diag([1, -1, -1])
X[:3, 3] = [0.17, 0, 0.145]
L = np.eye(4)
L[:3, 3] = [0.011, 0.02329, -0.04412]
XP = X @ L
WHEEL_EPS = 0.05


def conjugate(T):
    Tp = T @ np.linalg.inv(XP)
    return np.einsum("ij,njk->nik", np.linalg.inv(Tp[0]), Tp)


def segment_spans(p):
    """``[(start, end), ...]`` of a segment table, or None when the scene has none."""
    if not Path(p).exists():
        return None
    return [(s.start_ts, s.end_ts) for s in read_segments(p)]


def coverage(ts, t, segs):
    if len(t) == 0:
        return 0.0
    cov = np.zeros(len(ts), bool)
    for a, b in segs if segs else [(t[0], t[-1])]:
        cov |= (ts >= a - 0.15) & (ts <= b + 0.15)
    return cov.mean()


def rel(T, i, j):
    return np.linalg.inv(T[i]) @ T[j]


def windows(t, T, rt, rT, segs, w):
    errs = []
    for i in range(len(t)):
        j = np.searchsorted(t, t[i] + w)
        if j >= len(t) or t[j] - t[i] > w + 0.5:
            continue
        if segs and not any(a <= t[i] and t[j] <= b for a, b in segs):
            continue
        ri, rj = np.searchsorted(rt, t[i]), np.searchsorted(rt, t[j])
        if ri >= len(rt) or rj >= len(rt) or abs(rt[ri] - t[i]) > 0.2 or abs(rt[rj] - t[j]) > 0.2:
            continue
        errs.append(np.linalg.norm(rel(T, i, j)[:3, 3] - rel(rT, ri, rj)[:3, 3]))
    return np.array(errs)


def stationary_drift(t, T, wheels):
    """Max translation of the estimate over intervals where all wheels are stopped >= 0.5 s."""
    if len(t) == 0 or wheels is None:
        return np.nan, 0
    wt = wheels[:, 0]
    stopped = (np.abs(wheels[:, 1:]) < WHEEL_EPS).all(1)
    edges = np.diff(np.concatenate([[0], stopped.astype(int), [0]]))
    worst, n = 0.0, 0
    for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0] - 1, strict=True):
        if wt[b] - wt[a] < 0.5:
            continue
        m = (t >= wt[a]) & (t <= wt[b])
        if m.sum() < 2:
            continue
        n += 1
        p = T[m][:, :3, 3]
        worst = max(worst, np.linalg.norm(p[-1] - p[0]))
    return worst, n


def yaw_corr(t, T, wheels):
    if len(t) < 10 or wheels is None:
        return np.nan
    yaw = np.unwrap(np.arctan2(T[:, 1, 0], T[:, 0, 0]))
    yr = np.gradient(yaw, t)
    wt = wheels[:, 0]
    diff = wheels[:, 2] - wheels[:, 1]  # right - left (signed columns absent -> abs)
    d = np.interp(t, wt, diff)
    m = np.isfinite(yr) & np.isfinite(d)
    if m.sum() < 10 or d[m].std() == 0 or yr[m].std() == 0:
        return np.nan
    return float(np.corrcoef(yr[m], d[m])[0, 1])


def cloud_stats(scene_dir, k=5):
    files = sorted((scene_dir / "clouds").glob("*.npy"))
    if not files:
        return np.nan, np.nan
    pick = files[:: max(1, len(files) // k)][:k]
    n, blob = [], []
    for f in pick:
        p = np.load(f)
        n.append(len(p))
        u, c = np.unique(np.round(p[:, :3], 3), axis=0, return_counts=True)
        blob.append(c.max() / len(p))
    return float(np.mean(n)), float(np.mean(blob))


def main():
    old, new, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    scenes = sys.argv[4:] or sorted(d.name for d in new.iterdir() if d.is_dir() and d.name.isdigit())
    rows = []
    for s in scenes:
        n, o = new / s, old / s
        if not (n / "data_ts.txt").exists():
            continue
        ts = np.loadtxt(n / "data_ts.txt")
        ts = ts if ts.ndim == 1 else ts[:, 0]
        r = dict(scene=s, frames=len(ts))
        ref = (
            read_trajectory(n / "regen" / "odom_ref.txt")
            if (n / "regen" / "odom_ref.txt").exists()
            else (np.zeros(0), None)
        )
        wheels = np.loadtxt(n / "regen" / "wheels.txt") if (n / "regen" / "wheels.txt").exists() else None
        # old
        if (o / "odometry.txt").exists():
            ot, oT = read_trajectory(o / "odometry.txt")
            oC = conjugate(oT)
            r["old_cov"] = coverage(ts, ot, None)
            e = windows(ot, oC, *ref, None, 2.0) if ref[1] is not None else np.array([])
            r["old_win_med"] = np.median(e) if len(e) else np.nan
            r["old_win_p95"] = np.percentile(e, 95) if len(e) else np.nan
            r["old_win_gt1"] = float(np.mean(e > 1)) if len(e) else np.nan
            r["old_stat_drift"], r["stat_intervals"] = stationary_drift(ot, oC, wheels)
            r["old_yaw_corr"] = yaw_corr(ot, oC, wheels)
            r["old_yaw_corr_raw"] = yaw_corr(ot, oT, wheels)
        else:
            ot = np.zeros(0)
            r["old_cov"] = 0.0
        # new
        if (n / "odometry.txt").exists():
            nt, nT = read_trajectory(n / "odometry.txt")
            segs = segment_spans(n / "odometry_segments.txt")
            r["new_cov"] = coverage(ts, nt, segs)
            r["new_segments"] = len(segs) if segs else 1
            e = windows(nt, nT, *ref, segs, 2.0) if ref[1] is not None else np.array([])
            r["new_win_med"] = np.median(e) if len(e) else np.nan
            r["new_win_p95"] = np.percentile(e, 95) if len(e) else np.nan
            r["new_win_gt1"] = float(np.mean(e > 1)) if len(e) else np.nan
            r["new_stat_drift"], _ = stationary_drift(nt, nT, wheels)
            r["new_yaw_corr"] = yaw_corr(nt, nT, wheels)
            if len(ot):
                # 10 s relative-pose error old-conjugated vs new on the overlap
                errs = []
                for i in range(len(nt)):
                    j = np.searchsorted(nt, nt[i] + 10.0)
                    if j >= len(nt) or nt[j] - nt[i] > 10.5:
                        continue
                    if segs and not any(a <= nt[i] and nt[j] <= b for a, b in segs):
                        continue
                    oi, oj = np.searchsorted(ot, nt[i]), np.searchsorted(ot, nt[j])
                    if oi >= len(ot) or oj >= len(ot) or abs(ot[oi] - nt[i]) > 0.2 or abs(ot[oj] - nt[j]) > 0.2:
                        continue
                    errs.append(np.linalg.norm(rel(nT, i, j)[:3, 3] - rel(oC, oi, oj)[:3, 3]))
                r["old_vs_new_10s_med"] = np.median(errs) if errs else np.nan
        else:
            r["new_cov"] = 0.0
        r["old_pts"], r["old_blob"] = cloud_stats(o) if o.exists() else (np.nan, np.nan)
        r["new_pts"], r["new_blob"] = cloud_stats(n)
        r["camera_info_yaml"] = int((n / "camera_info.yaml").exists())
        rows.append(r)
        print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "scene", k))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
