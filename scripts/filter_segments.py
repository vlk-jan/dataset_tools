#!/usr/bin/env python3
"""Post-hoc filter for a regenerated, segmented odometry.txt.

The divergence detector cannot judge the first seconds after a FAST-LIO
restart, so degenerate stretches leave short segments whose poses are wrong
but were never flagged. This pass checks every segment after the fact against
the robot's live /odom over short windows, drops the poses that disagree,
splits the remainder into runs, keeps runs long enough to matter and
re-anchors each run at its first pose.

The window test cannot see a HOLE -- FAST-LIO publishing nothing for seconds to
minutes while the robot drives on -- because no window closes inside it. So
every consecutive pose pair is also judged as a gap (``dataset_tools.gt_gaps``):
the run is cut where the live odometry travelled more than --max-err further
across the gap than the GT pose moved (tag ``wheel_gap``), or where the gap is
longer than --max-gap regardless (tag ``gap``).

    filter_segments.py <regen_dir> <odom_ref.txt> <out_dir> [--window 2.0]
                       [--max-err 1.0] [--min-poses 20] [--max-gap 1.0]

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
from pathlib import Path

import numpy as np

from dataset_tools.gt_gaps import ReferencePath, cut_segments, gap_seams, window_errors
from dataset_tools.gt_io import SegmentRow, read_segments, read_trajectory, write_segments, write_trajectory


def bad_poses(t, T, rt, rT, window, max_err):
    """Boolean mask of poses that belong to a window disagreeing with the reference."""
    bad = np.zeros(len(t), bool)
    for i, j, err in window_errors(t, T, rt, rT, window):
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
    ap.add_argument("--max-gap", type=float, default=1.0, help="s; a longer pose gap always ends the run")
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
    segs = read_segments(segp) if segp.exists() else [SegmentRow(0, t[0], t[-1], len(t), "whole")] if len(t) else []
    rt, rT = read_trajectory(a.odom_ref)
    ref_path = ReferencePath.from_trajectory(rt, rT)

    seams = {"wheel_gap": 0, "gap": 0}
    hole_s = 0.0

    def mark(ts, Ts):
        """Poses the window test drops, and the seams the gap rules cut at."""
        nonlocal hole_s
        seam, tags = gap_seams(ts, Ts, ref_path, a.max_gap, a.max_err)
        for k in np.where(seam)[0]:
            seams[tags[k]] += 1
            hole_s += ts[k + 1] - ts[k]
        return seam, tags, bad_poses(ts, Ts, rt, rT, a.window, a.max_err)

    # The input tag says why the input segment ENDED, so only the run that
    # reaches that end inherits it; every earlier run ends where this filter
    # cut it, which is a different reason and its own tag. Without the
    # distinction a segment could read "the recording ended" 97 s before the
    # recording ended (scene 0000, seg 7).
    out_t, out_T, kept_segs, st = cut_segments(
        t, T, segs, mark, lambda reason, cut: f"filtered:{reason if cut is None else cut}", a.min_poses
    )
    write_trajectory(out / "odometry.txt", out_t, out_T)
    write_segments(out / "odometry_segments.txt", kept_segs)
    summary = dict(
        poses_in=len(t),
        poses_out=st["poses_out"],
        segments_in=len(segs),
        segments_out=len(kept_segs),
        span_in_s=round(st["span_in"], 1),
        span_out_s=round(st["span_out"], 1),
        window=a.window,
        max_err=a.max_err,
        min_poses=a.min_poses,
        max_gap=a.max_gap,
        seams_wheel_gap=seams["wheel_gap"],
        seams_gap=seams["gap"],
        hole_s=round(hole_s, 1),
    )
    with open(regen / "filter_stats.txt", "w") as f:
        for k, v in summary.items():
            f.write(f"{k}={v}\n")
    print("FILTER " + " ".join(f"{k}={v}" for k, v in summary.items()))


if __name__ == "__main__":
    main()
