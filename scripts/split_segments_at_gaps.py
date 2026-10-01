#!/usr/bin/env python3
"""Cut the segments of an existing root at the holes filter_segments.py missed.

Applies the gap rules of ``dataset_tools.gt_gaps`` to a scene's published
``odometry.txt`` / ``odometry_segments.txt`` without redoing the window test
(those poses already passed it) and without needing the v1.2 inputs or the gyro
extracts a full refinement re-run would: cutting a segment and re-anchoring
each side is a rigid transform of the poses on that side, so every relative
motion inside a piece is exactly what it was. What goes is the pose pair that
bridged the hole -- the one step nothing could have justified.

    split_segments_at_gaps.py <root> [--scenes 0000 0001] [--max-gap 1.0]
                              [--max-err 1.0] [--min-poses 20] [--check]

Per scene it reads
    <scene>/odometry.txt, <scene>/odometry_segments.txt, <scene>/regen/odom_ref.txt
and rewrites the first two in place, keeping the input as
    <scene>/regen/odometry_pregapsplit.txt, odometry_segments_pregapsplit.txt
(a re-run starts from those, so the pass is idempotent). --check prints the
numbers and writes nothing. One line per scene plus a TOTAL line:
segments/poses/span in -> out, seams per rule, and the reference path length
the robot drove inside the removed holes.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the package, when it is not installed
from dataset_tools.gt_gaps import ReferencePath, gap_seams, split_runs  # noqa: E402
from dataset_tools.gt_io import (  # noqa: E402
    SegmentRow,
    anchor_to_first,
    changed_since_written,
    read_segments,
    read_trajectory,
    record_written,
    write_segments,
    write_trajectory,
)

BACKUP_TRAJ = "odometry_pregapsplit.txt"
BACKUP_SEGS = "odometry_segments_pregapsplit.txt"


def split_scene(scene: Path, max_gap: float, max_err: float, min_poses: int, check: bool, force: bool = False) -> dict:
    regen = scene / "regen"
    if (regen / BACKUP_TRAJ).exists() and not check and not force and changed_since_written(scene, BACKUP_TRAJ):
        raise SystemExit(
            f"{scene}: odometry.txt changed since this pass last wrote it; rebuilding from "
            f"regen/{BACKUP_TRAJ} would undo that. Move the backup aside, or --force"
        )
    src_traj = regen / BACKUP_TRAJ if (regen / BACKUP_TRAJ).exists() else scene / "odometry.txt"
    src_segs = regen / BACKUP_SEGS if (regen / BACKUP_SEGS).exists() else scene / "odometry_segments.txt"
    t, T = read_trajectory(src_traj)
    segs = read_segments(src_segs)
    ref = None
    if (regen / "odom_ref.txt").exists():
        ref = ReferencePath.from_trajectory(*read_trajectory(regen / "odom_ref.txt"))

    kept_ts, kept_T, rows = [], [], []
    stats = dict(
        segments_in=len(segs),
        poses_in=len(t),
        span_in=0.0,
        span_out=0.0,
        poses_out=0,
        seams_wheel_gap=0,
        seams_gap=0,
        hole_s=0.0,
        ref_m_in_holes=0.0,
        dropped_short=0,
    )
    for seg in segs:
        idx = np.where((t >= seg.start_ts) & (t <= seg.end_ts))[0]
        if len(idx) == 0:
            continue
        ts, Ts = t[idx], T[idx]
        stats["span_in"] += ts[-1] - ts[0]
        seam, tags = gap_seams(ts, Ts, ref, max_gap, max_err)
        for k in np.where(seam)[0]:
            stats["seams_" + tags[k]] += 1
            stats["hole_s"] += ts[k + 1] - ts[k]
            if ref is not None:
                d = ref.length_between(ts[k], ts[k + 1])
                stats["ref_m_in_holes"] += float(d) if np.isfinite(d) else 0.0
        for r0, r1, cut in split_runs(len(ts), seam, tags):
            if r1 - r0 < min_poses:
                stats["dropped_short"] += r1 - r0
                continue
            run_t, run_T = ts[r0:r1], Ts[r0:r1]
            kept_ts.append(run_t)
            kept_T.append(anchor_to_first(run_T))
            # The run reaching the input segment's end keeps its tag (why that
            # segment ended); a run cut at a seam ends because of the seam.
            if cut is None:
                reason = seg.reason
            else:
                prefix = seg.reason.split(":", 1)[0] if ":" in seg.reason else "filtered"
                reason = f"{prefix}:{cut}"
            rows.append(SegmentRow(len(rows), run_t[0], run_t[-1], len(run_t), reason))
            stats["span_out"] += run_t[-1] - run_t[0]
            stats["poses_out"] += len(run_t)
    stats["segments_out"] = len(rows)

    if not check:
        if src_traj == scene / "odometry.txt":
            regen.mkdir(exist_ok=True)
            shutil.copy2(scene / "odometry.txt", regen / BACKUP_TRAJ)
            shutil.copy2(scene / "odometry_segments.txt", regen / BACKUP_SEGS)
        write_trajectory(
            scene / "odometry.txt",
            np.concatenate(kept_ts) if kept_ts else t[:0],
            np.concatenate(kept_T) if kept_T else T[:0],
        )
        write_segments(scene / "odometry_segments.txt", rows)
        record_written(scene, BACKUP_TRAJ)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--max-gap", type=float, default=1.0)
    ap.add_argument("--max-err", type=float, default=1.0)
    ap.add_argument("--min-poses", type=int, default=20)
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    ap.add_argument(
        "--force", action="store_true", help="rebuild from the regen/ backup even if odometry.txt changed since"
    )
    a = ap.parse_args()

    scenes = a.scenes or sorted(p.name for p in a.root.iterdir() if p.is_dir() and p.name.isdigit())
    tot: dict[str, float] = {}
    print(
        f"{'scene':5s} {'segs':>9s} {'poses':>13s} {'span_s':>15s} {'wheel':>5s} {'gap':>4s} {'hole_s':>7s} {'ref_m':>6s} {'short':>5s}"
    )
    for sc in scenes:
        s = split_scene(a.root / sc, a.max_gap, a.max_err, a.min_poses, a.check, a.force)
        for k, v in s.items():
            tot[k] = tot.get(k, 0) + v
        print(
            f"{sc:5s} {s['segments_in']:3d}->{s['segments_out']:<4d} {s['poses_in']:6d}->{s['poses_out']:<6d} "
            f"{s['span_in']:6.0f}->{s['span_out']:<7.0f} {s['seams_wheel_gap']:5d} {s['seams_gap']:4d} "
            f"{s['hole_s']:7.0f} {s['ref_m_in_holes']:6.0f} {s['dropped_short']:5d}"
        )
    print(
        f"TOTAL {int(tot['segments_in']):3d}->{int(tot['segments_out']):<4d} {int(tot['poses_in']):6d}->{int(tot['poses_out']):<6d} "
        f"{tot['span_in']:6.0f}->{tot['span_out']:<7.0f} {int(tot['seams_wheel_gap']):5d} {int(tot['seams_gap']):4d} "
        f"{tot['hole_s']:7.0f} {tot['ref_m_in_holes']:6.0f} {int(tot['dropped_short']):5d}"
        + ("  (check only, nothing written)" if a.check else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
