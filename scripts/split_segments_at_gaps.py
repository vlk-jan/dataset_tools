#!/usr/bin/env python3
"""Cut the segments of an existing root at the holes filter_segments.py missed.

Applies the gap rules of ``dataset_tools.gt_gaps`` to the published odometry
without a full re-run. Re-anchoring is rigid, so relative motion within a piece
is unchanged; only the pose pair bridging each hole is lost.

    split_segments_at_gaps.py <root> [--scenes 0000 0001] [--max-gap 1.0]
                              [--max-err 1.0] [--min-poses 20] [--check]

Rewrites odometry.txt and odometry_segments.txt in place, backing the input up to
regen/*_pregapsplit.txt; re-runs start from that backup, so the pass is idempotent.
--check writes nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from dataset_tools.gt_gaps import ReferencePath, cut_reason, cut_segments, gap_seams
from dataset_tools.gt_io import changed_since_written, publish_with_backup, read_segments, read_trajectory
from dataset_tools.image_format import scenes as scene_dirs

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

    seams = dict(seams_wheel_gap=0, seams_gap=0, hole_s=0.0, ref_m_in_holes=0.0)

    def mark(ts, Ts):
        seam, tags = gap_seams(ts, Ts, ref, max_gap, max_err)
        for k in np.where(seam)[0]:
            seams["seams_" + tags[k]] += 1
            seams["hole_s"] += ts[k + 1] - ts[k]
            if ref is not None:
                d = ref.length_between(ts[k], ts[k + 1])
                seams["ref_m_in_holes"] += float(d) if np.isfinite(d) else 0.0
        return seam, tags, None

    # Only the run reaching the input segment's end keeps its tag; a seam cut gets the seam's.
    out_t, out_T, rows, st = cut_segments(
        t, T, segs, mark, lambda reason, cut: reason if cut is None else cut_reason(reason, cut), min_poses
    )
    stats = dict(segments_in=len(segs), poses_in=len(t), **st, **seams, segments_out=len(rows))

    if not check:
        publish_with_backup(scene, src_traj, (BACKUP_TRAJ, BACKUP_SEGS), out_t, out_T, rows)
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

    scenes = a.scenes or [p.name for p in scene_dirs(a.root)]
    tot: dict[str, float] = {}
    print(
        f"{'scene':5s} {'segs':>9s} {'poses':>13s} {'span_s':>15s} {'wheel':>5s} {'gap':>4s} {'hole_s':>7s} {'ref_m':>6s} {'short':>5s} {'outside':>7s}"
    )
    for sc in scenes:
        s = split_scene(a.root / sc, a.max_gap, a.max_err, a.min_poses, a.check, a.force)
        for k, v in s.items():
            tot[k] = tot.get(k, 0) + v
        print(
            f"{sc:5s} {s['segments_in']:3d}->{s['segments_out']:<4d} {s['poses_in']:6d}->{s['poses_out']:<6d} "
            f"{s['span_in']:6.0f}->{s['span_out']:<7.0f} {s['seams_wheel_gap']:5d} {s['seams_gap']:4d} "
            f"{s['hole_s']:7.0f} {s['ref_m_in_holes']:6.0f} {s['dropped_short']:5d} {s['dropped_outside']:7d}"
        )
    print(
        f"TOTAL {int(tot['segments_in']):3d}->{int(tot['segments_out']):<4d} {int(tot['poses_in']):6d}->{int(tot['poses_out']):<6d} "
        f"{tot['span_in']:6.0f}->{tot['span_out']:<7.0f} {int(tot['seams_wheel_gap']):5d} {int(tot['seams_gap']):4d} "
        f"{tot['hole_s']:7.0f} {tot['ref_m_in_holes']:6.0f} {int(tot['dropped_short']):5d} {int(tot['dropped_outside']):7d}"
        + ("  (check only, nothing written)" if a.check else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
