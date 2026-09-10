#!/usr/bin/env python3
"""Inventory the ground-truth holes in a root and say which ones are worth replaying.

    find_gt_holes.py <root> [--min-gap 5] [--csv out.csv]

A hole is a stretch of frame time no odometry segment covers. For each one this
reports how alive the rest of the recording was there -- frame rate, cloud size,
how much of the time FAST-LIO published anything, the longest stretch where it
published nothing, and how far the wheels actually travelled per the recorded
``/odom`` -- and classifies it:

  silent_stall            FAST-LIO went quiet for over ``--max-silence`` seconds
                          while frames and /odom kept coming. The estimator
                          stalled at run time; replaying the bag at an offset
                          recovers these (see docs/GT_HOLE_REPLAY.md).
  restart_storm_or_gate   short silences only: FAST-LIO was running and either
                          restarting or refusing to track. A re-run reproduces
                          the same refusal, so these need a different estimator,
                          not another pass.

Reads only; ``--csv`` writes the table docs/v13_gt_holes.csv was made from.
Written for the v1.3 campaign, where it was called holes.py.
"""

import argparse
import os

import numpy as np

STILL_SPEED = 0.05  # m per 1 s bin below which the robot counts as not moving
FASTLIO_HZ = 10.0


def scene_holes(root, scene, min_gap, max_silence):
    d = f"{root}/{scene}"
    ts = np.loadtxt(f"{d}/data_ts.txt")
    first, last = ts[0], ts[-1]
    seg = np.loadtxt(f"{d}/odometry_segments.txt", usecols=(1, 2), ndmin=2)
    raw = np.loadtxt(f"{d}/regen/odometry_imu_raw.txt", usecols=(0,))
    ref = np.loadtxt(f"{d}/regen/odom_ref.txt")
    ref_t, ref_p = ref[:, 0], ref[:, [4, 8, 12]]

    gaps, prev = [], first
    for b, e in seg:
        if b - prev > min_gap:
            gaps.append((prev, b))
        prev = e
    if last - prev > min_gap:
        gaps.append((prev, last))

    rows = []
    for a, b in gaps:
        dur = b - a
        idx = np.where((ts >= a) & (ts <= b))[0]
        fps = len(idx) / dur
        pts = [np.load(f"{d}/clouds/{i:05d}.npy", mmap_mode="r").shape[0] for i in idx[:: max(1, len(idx) // 5)][:5]]
        raw_cov = ((raw >= a) & (raw <= b)).sum() / FASTLIO_HZ / dur
        # the longest stretch inside the hole where FAST-LIO published nothing
        edges = np.concatenate([[a], raw[(raw > a) & (raw < b)], [b]])
        max_silence_s = float(np.diff(edges).max())

        m = (ref_t >= a) & (ref_t <= b)
        if m.sum() > 10:
            binned = np.clip(np.searchsorted(ref_t[m], np.arange(a, b + 1, 1.0)), 0, m.sum() - 1)
            step = np.linalg.norm(np.diff(ref_p[m][binned], axis=0), axis=1)
            moving_frac, wheel_m = float((step > STILL_SPEED).mean()), float(step.sum())
        else:
            moving_frac = wheel_m = np.nan

        rows.append(
            dict(
                scene=scene,
                t0_rel_s=a - first,
                dur_s=dur,
                fps=fps,
                med_cloud_pts=int(np.median(pts)) if pts else -1,
                fastlio_raw_cov=raw_cov,
                max_fastlio_silence_s=max_silence_s,
                moving_frac=moving_frac,
                wheel_path_m=wheel_m,
                cls="silent_stall" if max_silence_s > max_silence else "restart_storm_or_gate",
            )
        )
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--min-gap", type=float, default=5.0, help="ignore holes shorter than this (s)")
    ap.add_argument("--max-silence", type=float, default=20.0, help="silence above this makes it a silent stall (s)")
    ap.add_argument("--csv")
    a = ap.parse_args()

    scenes = sorted(d for d in os.listdir(a.root) if d.isdigit())
    rows = []
    for scene in scenes:
        rows += scene_holes(a.root, scene, a.min_gap, a.max_silence)

    print("scene   t0     dur  fps  medpts rawFLIO maxsilent movef wheel_m  class")
    for r in sorted(rows, key=lambda x: -x["dur_s"]):
        print(
            f"{r['scene']} {r['t0_rel_s']:7.1f} {r['dur_s']:6.1f} {r['fps']:4.1f} {r['med_cloud_pts']:6d}"
            f"  {r['fastlio_raw_cov']:5.2f}  {r['max_fastlio_silence_s']:7.1f}"
            f"  {r['moving_frac']:5.2f} {r['wheel_path_m']:7.1f}  {r['cls']}"
        )

    stalls = [r for r in rows if r["cls"] == "silent_stall"]
    print(
        f"\ntotal gap>{a.min_gap:.0f}s: {sum(r['dur_s'] for r in rows):.0f} s in {len(rows)} holes; "
        f"silent stalls: {len(stalls)}, {sum(r['dur_s'] for r in stalls):.0f} s, "
        f"wheel {np.nansum([r['wheel_path_m'] for r in stalls]):.0f} m"
    )
    print(f"moving (moving_frac>0.3): {sum(r['dur_s'] for r in rows if r['moving_frac'] > 0.3):.0f} s")

    if a.csv:
        cols = [
            "scene",
            "t0_rel_s",
            "dur_s",
            "fps",
            "med_cloud_pts",
            "fastlio_raw_cov",
            "max_fastlio_silence_s",
            "moving_frac",
            "wheel_path_m",
        ]
        with open(a.csv, "w") as fh:
            fh.write(",".join(cols) + ",class\n")
            for r in sorted(rows, key=lambda x: -x["dur_s"]):
                fh.write(
                    f"{r['scene']},{r['t0_rel_s']:.1f},{r['dur_s']:.1f},{r['fps']:.1f},{r['med_cloud_pts']},"
                    f"{r['fastlio_raw_cov']:.2f},{r['max_fastlio_silence_s']:.1f},{r['moving_frac']:.2f},"
                    f"{r['wheel_path_m']:.1f},{r['cls']}\n"
                )
        print(f"wrote {a.csv}")


if __name__ == "__main__":
    main()
