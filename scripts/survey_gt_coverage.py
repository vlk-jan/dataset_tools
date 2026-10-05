#!/usr/bin/env python3
"""How much of a root's frame time has ground truth, and where it does not.

    survey_gt_coverage.py <root> [--min-gap 1.0] [--min-frozen 1.0]

Three tables: per-scene coverage of the frame window by odometry segments; every
gap over ``--min-gap`` with how far the recorded ``/odom`` says the wheels moved
through it (a gap the robot drove through is lost data, one it stood through is
not); and every frozen run -- consecutive poses at an identical position -- with
the same wheel distance, which is how a stuck estimator or an over-eager ZUPT
shows up.

The frozen scan runs INSIDE each segment, never across a seam. Segments are
re-anchored at the identity, so a scan over the whole file reports the anchor
change as a fake multi-minute freeze; that trap cost a day on v1.3.

Reads only. Written for the v1.3 campaign, where it was called survey.py.
"""

import argparse

import numpy as np

from dataset_tools.gt_io import read_segments, read_trajectory
from dataset_tools.image_format import scenes


def frozen_runs(t, P, lo, hi, min_frozen, frame_ts, odom_p):
    """Frozen runs strictly inside one segment's [lo, hi] span."""
    idx = np.where((t >= lo) & (t <= hi))[0]
    if len(idx) < 3:
        return []
    tt, pp = t[idx], P[idx, :3, 3]
    same = np.linalg.norm(np.diff(pp, axis=0), axis=1) == 0.0
    out, i = [], 0
    while i < len(same):
        if not same[i]:
            i += 1
            continue
        j = i
        while j < len(same) and same[j]:
            j += 1
        dur = tt[j] - tt[i]
        if dur >= min_frozen:
            m = (frame_ts >= tt[i]) & (frame_ts <= tt[j])
            wheel = np.linalg.norm(np.diff(odom_p[m], axis=0), axis=1).sum() if m.sum() >= 2 else np.nan
            out.append((tt[i], dur, j - i + 1, wheel))
        i = j
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--min-gap", type=float, default=1.0, help="report gaps longer than this (s)")
    ap.add_argument("--min-frozen", type=float, default=1.0, help="report frozen runs longer than this (s)")
    a = ap.parse_args()

    rows, gaps, frozen = [], [], []
    for d in scenes(a.root):
        s = d.name
        frame_ts = np.loadtxt(d / "data_ts.txt")
        seg = [(r.start_ts, r.end_ts) for r in read_segments(d / "odometry_segments.txt")]
        t, P = read_trajectory(d / "odometry.txt")
        odom_p = np.load(d / "transforms.npy")[:, :3, 3]  # /odom, one pose per frame
        first, last = frame_ts[0], frame_ts[-1]
        span = last - first
        cov = sum(min(e, last) - max(b, first) for b, e in seg if e > first and b < last)

        scene_gaps, prev = [], first
        for b, e in seg:
            if b - prev > a.min_gap:
                scene_gaps.append((prev, b))
            prev = e
        if last - prev > a.min_gap:
            scene_gaps.append((prev, last))
        for g0, g1 in scene_gaps:
            m = (frame_ts >= g0) & (frame_ts <= g1)
            dist = np.linalg.norm(np.diff(odom_p[m], axis=0), axis=1).sum() if m.sum() >= 2 else np.nan
            gaps.append((s, g0 - first, g1 - first, g1 - g0, dist))

        for b, e in seg:
            for t0, dur, n, wheel in frozen_runs(t, P, b, e, a.min_frozen, frame_ts, odom_p):
                frozen.append((s, t0 - first, dur, n, wheel))

        rows.append((s, span, cov, cov / span, len(seg), len(scene_gaps), sum(y - x for x, y in scene_gaps)))

    print("scene    span     cov frac nseg ngaps  gapsec")
    for r in rows:
        print("%s %7.0f %7.0f %.2f %4d %5d %7.0f" % r)
    tot, cv = sum(r[1] for r in rows), sum(r[2] for r in rows)
    print(f"TOTAL span {tot:.0f} cov {cv:.0f} frac {cv / tot:.3f} gapsec {sum(r[6] for r in rows):.0f}")

    print(f"\nGAPS >{a.min_gap:.0f}s (scene, t0, t1, dur, wheel_m) sorted by duration")
    for g in sorted(gaps, key=lambda x: -x[3]):
        print("{} {:7.1f} {:7.1f} {:6.1f} {:7.1f}".format(*g))

    print(f"\nFROZEN runs >={a.min_frozen:.0f}s inside a segment (scene, t0, dur, nposes, wheel_m)")
    for f in sorted(frozen, key=lambda x: -(x[4] if not np.isnan(x[4]) else 0)):
        print("%s %7.1f %6.1f %4d %6.2f" % f)


if __name__ == "__main__":
    main()
