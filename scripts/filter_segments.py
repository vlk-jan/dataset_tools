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
       <out_dir>/odometry_segments.txt   "segment start_ts end_ts n_poses reason",
                                         the reason naming why that run ENDED:
                                         the input segment's tag for the run
                                         that reaches its end, `odom_mismatch`
                                         for a run this filter cut short
       <regen_dir>/filter_stats.txt      key=value summary (the pre-filter
                                         trajectory already lives in regen_dir)
Prints one FILTER line: poses_in poses_out segments_in segments_out span_in span_out
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the package, when it is not installed
from dataset_tools.gt_io import (  # noqa: E402
    SegmentRow,
    anchor_to_first,
    read_segments,
    read_trajectory,
    write_segments,
    write_trajectory,
)


def load_segments(p):
    """``(start, end, n_poses, reason)`` per row of a segment table."""
    return [(s.start_ts, s.end_ts, s.n_poses, s.reason) for s in read_segments(p)]


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
            bad[i : j + 1] = True
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
    src = regen / "odometry_prefilter.txt"
    if not src.exists():
        src = regen / "odometry.txt"
    t, T = read_trajectory(src)
    segp = regen / "odometry_segments.txt"
    if not segp.exists():
        segp = regen / "odometry_segments_prefilter.txt"
    segs = load_segments(segp) if segp.exists() else [(t[0], t[-1], len(t), "whole")] if len(t) else []
    rt, rT = read_trajectory(a.odom_ref)

    kept_ts, kept_T, kept_segs = [], [], []
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
        for r0, r1 in zip(np.where(edges == 1)[0], np.where(edges == -1)[0], strict=True):
            if r1 - r0 < a.min_poses:
                continue
            run_t, run_T = ts[r0:r1], Ts[r0:r1]
            kept_ts.append(run_t)
            kept_T.append(anchor_to_first(run_T))
            # The input tag says why the input segment ENDED, so only the run
            # that reaches that end inherits it; every earlier run ends where
            # this filter cut it, which is a different reason and its own tag.
            # Without the distinction a segment could read "the recording
            # ended" 97 s before the recording ended (scene 0000, seg 7).
            tag = reason if r1 == len(ts) else "odom_mismatch"
            kept_segs.append(SegmentRow(len(kept_segs), run_t[0], run_t[-1], len(run_t), f"filtered:{tag}"))
            span_out += run_t[-1] - run_t[0]

    poses_out = int(sum(len(x) for x in kept_ts))
    write_trajectory(
        out / "odometry.txt",
        np.concatenate(kept_ts) if kept_ts else t[:0],
        np.concatenate(kept_T) if kept_T else T[:0],
    )
    write_segments(out / "odometry_segments.txt", kept_segs)
    summary = dict(
        poses_in=len(t),
        poses_out=poses_out,
        segments_in=len(segs),
        segments_out=len(kept_segs),
        span_in_s=round(span_in, 1),
        span_out_s=round(span_out, 1),
        window=a.window,
        max_err=a.max_err,
        min_poses=a.min_poses,
    )
    with open(regen / "filter_stats.txt", "w") as f:
        for k, v in summary.items():
            f.write(f"{k}={v}\n")
    print("FILTER " + " ".join(f"{k}={v}" for k, v in summary.items()))


if __name__ == "__main__":
    main()
