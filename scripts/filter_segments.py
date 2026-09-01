#!/usr/bin/env python3
"""Post-hoc filter for a regenerated, segmented odometry.txt.

The divergence detector cannot judge the first seconds after a FAST-LIO
restart, so degenerate stretches leave short segments whose poses are wrong
but were never flagged. This pass checks every segment after the fact against
the robot's live /odom over short windows, drops the poses that disagree,
splits the remainder into runs, keeps runs long enough to matter and
re-anchors each run at its first pose.

    filter_segments.py <regen_dir> <odom_ref.txt> <out_dir> [--window 2.0]
                       [--max-err 1.0] [--min-poses 20]

Reads  <regen_dir>/odometry.txt and odometry_segments.txt (run_scene.sh output)
Writes <out_dir>/odometry.txt            filtered runs, each anchored at identity
       <out_dir>/odometry_segments.txt   "segment start_ts end_ts n_poses reason"
       <out_dir>/odometry_unfiltered.txt copy of the input trajectory
       <out_dir>/odometry_filter.txt     key=value summary
Prints one FILTER line: poses_in poses_out segments_in segments_out span_in span_out
"""
import argparse
import shutil
from pathlib import Path

import numpy as np

HEADER = "# Timestamp(sec) r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz 0 0 0 1"


def load_traj(p):
    d = np.loadtxt(p, comments="#")
    if d.size == 0:
        return np.zeros(0), np.zeros((0, 4, 4))
    if d.ndim == 1:
        d = d[None, :]
    o = np.argsort(d[:, 0], kind="stable")
    return d[o, 0], d[o, 1:].reshape(-1, 4, 4)


def load_segments(p):
    segs = []
    for line in open(p):
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        segs.append((float(f[1]), float(f[2]), int(f[3]), f[4] if len(f) > 4 else ""))
    return segs


def rel(T, i, j):
    return np.linalg.inv(T[i]) @ T[j]


def bad_poses(t, T, rt, rT, window, max_err):
    """Boolean mask of poses that belong to a window disagreeing with the reference."""
    bad = np.zeros(len(t), bool)
    for i in range(len(t)):
        j = np.searchsorted(t, t[i] + window)
        if j >= len(t) or t[j] - t[i] > window + 0.5:
            continue
        ri = np.searchsorted(rt, t[i])
        rj = np.searchsorted(rt, t[j])
        if ri >= len(rt) or rj >= len(rt) or abs(rt[ri] - t[i]) > 0.2 or abs(rt[rj] - t[j]) > 0.2:
            continue
        err = np.linalg.norm(rel(T, i, j)[:3, 3] - rel(rT, ri, rj)[:3, 3])
        if err > max_err:
            bad[i:j + 1] = True
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("regen_dir")
    ap.add_argument("odom_ref")
    ap.add_argument("out_dir")
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument("--max-err", type=float, default=1.0)
    ap.add_argument("--min-poses", type=int, default=20)
    a = ap.parse_args()

    regen, out = Path(a.regen_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t, T = load_traj(regen / "odometry.txt")
    segs = load_segments(regen / "odometry_segments.txt") if (regen / "odometry_segments.txt").exists() else [(t[0], t[-1], len(t), "whole")] if len(t) else []
    rt, rT = load_traj(a.odom_ref)

    kept_rows, kept_segs = [], []
    span_in = span_out = 0.0
    for s0, s1, _, reason in segs:
        idx = np.where((t >= s0) & (t <= s1))[0]
        if len(idx) == 0:
            continue
        span_in += t[idx[-1]] - t[idx[0]]
        ts, Ts = t[idx], T[idx]
        bad = bad_poses(ts, Ts, rt, rT, a.window, a.max_err)
        # runs of consecutive good poses
        good = ~bad
        edges = np.diff(np.concatenate([[0], good.astype(int), [0]]))
        for r0, r1 in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
            if r1 - r0 < a.min_poses:
                continue
            run_t, run_T = ts[r0:r1], Ts[r0:r1]
            anchored = np.einsum("ij,njk->nik", np.linalg.inv(run_T[0]), run_T)
            for k in range(len(run_t)):
                kept_rows.append(np.concatenate([[run_t[k]], anchored[k].ravel()]))
            kept_segs.append((run_t[0], run_t[-1], len(run_t), f"filtered:{reason}"))
            span_out += run_t[-1] - run_t[0]

    shutil.copy(regen / "odometry.txt", out / "odometry_unfiltered.txt")
    with open(out / "odometry.txt", "w") as f:
        f.write(HEADER + "\n")
        for row in kept_rows:
            f.write(" ".join(f"{v:.6f}" for v in row) + "\n")
    with open(out / "odometry_segments.txt", "w") as f:
        f.write("# segment start_ts end_ts n_poses reason\n")
        for k, (s0, s1, n, reason) in enumerate(kept_segs):
            f.write(f"{k} {s0:.6f} {s1:.6f} {n} {reason}\n")
    summary = dict(poses_in=len(t), poses_out=len(kept_rows), segments_in=len(segs),
                   segments_out=len(kept_segs), span_in_s=round(span_in, 1), span_out_s=round(span_out, 1),
                   window=a.window, max_err=a.max_err, min_poses=a.min_poses)
    with open(out / "odometry_filter.txt", "w") as f:
        for k, v in summary.items():
            f.write(f"{k}={v}\n")
    print("FILTER " + " ".join(f"{k}={v}" for k, v in summary.items()))


if __name__ == "__main__":
    main()
