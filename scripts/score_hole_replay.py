#!/usr/bin/env python3
"""Score one re-run of a ground-truth hole before it is merged into a root.

    score_hole_replay.py <scene> <replay_dir> <t0> <t1> [--root ROOT]

``t0``/``t1`` are seconds from the scene's first frame and name the hole the
replay was supposed to fill; ``replay_dir`` is a raw ``run_scene_offset.sh``
output (``odometry.txt`` + ``odometry_segments.txt``, prefilter).

Applies the SAME acceptance the builder applies -- filter_segments' 2 s windows
against the recorded ``/odom`` (drop over 1 m), gt_gaps seams, runs of at least
20 poses, re-anchored at identity -- and then reports what survives:

  * the 2 s window error against /odom, so a diverged run is visible as a tail
  * hole coverage raw and after the gate, which is the yield that matters
  * the height-map check on measured cells (4 s windows), the external referee
  * how many poses the root already has inside the replay's span, so an overlap
    that would double-cover a stretch is caught before the merge

Prints only; nothing is written. Written for the v1.3 silent-stall campaign
(docs/GT_HOLE_REPLAY.md), where it was called score_pilot.py.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the package, when it is not installed
from filter_segments import bad_poses  # noqa: E402

from dataset_tools.gt_gaps import ReferencePath, gap_seams, split_runs  # noqa: E402
from dataset_tools.gt_io import anchor_to_first, read_segments, read_trajectory  # noqa: E402
from dataset_tools.gt_map import window_offset  # noqa: E402

WINDOW, MAX_ERR, MIN_POSES, MAX_GAP = 2.0, 1.0, 20, 1.0
MAP_HORIZON, MAP_STRIDE, MAP_MIN_POSES = 4.0, 5, 5


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("replay_dir")
    ap.add_argument("t0", type=float, help="hole start, seconds from the first frame")
    ap.add_argument("t1", type=float, help="hole end, seconds from the first frame")
    ap.add_argument("--root", default="/mnt/personal/vlkjan6/HELHEST_v1_3", help="root the hole belongs to")
    a = ap.parse_args()

    scene = f"{a.root}/{a.scene}"
    ts = np.loadtxt(f"{scene}/data_ts.txt")
    first = ts[0]
    hole0, hole1 = first + a.t0, first + a.t1

    t, T = read_trajectory(f"{a.replay_dir}/odometry.txt")
    segs = read_segments(f"{a.replay_dir}/odometry_segments.txt")
    rt, rT = read_trajectory(f"{scene}/regen/odom_ref.txt")
    ref = ReferencePath.from_trajectory(rt, rT)

    print(
        f"replay poses {len(t)}, segments {len(segs)}, "
        f"span {t[0] - first:.1f}..{t[-1] - first:.1f} s; hole {a.t0:.1f}..{a.t1:.1f}"
    )
    for s in segs:
        print(f"  seg {s.index} {s.start_ts - first:7.1f}..{s.end_ts - first:7.1f} n={s.n_poses} {s.reason}")

    kept, errs = [], []
    for s in segs:
        idx = np.where((t >= s.start_ts) & (t <= s.end_ts))[0]
        if len(idx) == 0:
            continue
        tt, TT = t[idx], T[idx]
        bad = bad_poses(tt, TT, rt, rT, WINDOW, MAX_ERR)
        seam, tags = gap_seams(tt, TT, ref, MAX_GAP, MAX_ERR)
        for r0, r1, _cut in split_runs(len(tt), seam, tags, bad):
            if r1 - r0 < MIN_POSES:
                continue
            kept.append((tt[r0:r1], anchor_to_first(TT[r0:r1])))
        # the same 2 s comparison, kept per window so the distribution is visible
        for i in range(len(tt)):
            j = np.searchsorted(tt, tt[i] + WINDOW)
            if j >= len(tt) or tt[j] - tt[i] > WINDOW + 0.5:
                continue
            ri, rj = np.searchsorted(rt, tt[i]), np.searchsorted(rt, tt[j])
            if ri >= len(rt) or rj >= len(rt):
                continue
            moved = (np.linalg.inv(TT[i]) @ TT[j])[:3, 3]
            ref_moved = (np.linalg.inv(rT[ri]) @ rT[rj])[:3, 3]
            errs.append(float(np.linalg.norm(moved - ref_moved)))

    errs = np.array(errs)
    if len(errs):
        print(
            f"{WINDOW:.0f} s window err vs /odom: n {len(errs)} median {np.median(errs):.3f} "
            f"p95 {np.percentile(errs, 95):.3f} max {errs.max():.2f}  "
            f">{MAX_ERR:.0f} m {100 * (errs > MAX_ERR).mean():.1f}%"
        )
    else:
        print("no 2 s window could be compared against /odom -- the reference does not reach this span")

    raw_cov = max(0.0, min(t[-1], hole1) - max(t[0], hole0)) if t[-1] > hole0 and t[0] < hole1 else 0.0
    gate_cov = sum(max(0.0, min(k[0][-1], hole1) - max(k[0][0], hole0)) for k in kept)
    span = a.t1 - a.t0
    print(
        f"hole coverage: raw {raw_cov:.1f} s ({100 * raw_cov / span:.0f}%), "
        f"after gate {gate_cov:.1f} s ({100 * gate_cov / span:.0f}%) in {len(kept)} runs"
    )

    ga = np.load(f"{scene}/gravity_align.npy")
    pcd_ts = np.loadtxt(f"{scene}/pcd_ts.txt")
    offs = []
    for kt, kT in kept:
        for f in np.where((pcd_ts >= kt[0]) & (pcd_ts + MAP_HORIZON <= kt[-1]))[0][::MAP_STRIDE]:
            hmap, hmask = f"{scene}/height_maps/{f:05d}.npz", f"{scene}/height_masks/{f:05d}.npz"
            if not (os.path.exists(hmap) and os.path.exists(hmask)):
                continue
            m = (kt >= pcd_ts[f]) & (kt <= pcd_ts[f] + MAP_HORIZON)
            if m.sum() < MAP_MIN_POSES:
                continue
            res = window_offset(kT[m], ga[f], np.load(hmap)["height_map"], np.load(hmask)["validity"])
            if res:
                offs.append(res[0])
    offs = np.array(offs)
    if len(offs):
        print(
            f"map check ({MAP_HORIZON:.0f} s, measured cells): n {len(offs)} median {np.median(offs):+.3f} "
            f"|.| p90 {np.percentile(np.abs(offs), 90):.3f} >0.3 m {100 * (np.abs(offs) > 0.3).mean():.1f}%"
        )
    else:
        print("map check: no window had a measured cell under the wheels -- this hole cannot be map-scored")

    root_t, _root_T = read_trajectory(f"{scene}/odometry.txt")
    overlap = ((root_t >= t[0]) & (root_t < hole0)).sum()
    print(f"poses the root already has inside the replay span before the hole: {overlap}")


if __name__ == "__main__":
    main()
