#!/usr/bin/env python3
"""Cut the segments of a root where the ground-truth path leaves the lidar's ground.

Applies :mod:`dataset_tools.gt_map` to every keyframe of a scene, then removes the
poses inside each mismatch window (plus a margin) and re-anchors the pieces, the
way ``split_segments_at_gaps.py`` handles holes. Rigid re-anchoring keeps every
relative motion inside a piece exactly what it was.

    cut_segments_at_map_mismatch.py <root> [--scenes ...] [--horizon 4.0] [--max-off 0.3]
                                    [--margin 0.5] [--min-poses 20] [--jobs 8] [--check]

Reads <scene>/odometry.txt, odometry_segments.txt, gravity_align.npy, pcd_ts.txt,
height_maps/, height_masks/ and rewrites the first two in place, keeping the input
as <scene>/regen/odometry_premapcut.txt, odometry_segments_premapcut.txt (a re-run
starts from those, so the pass is idempotent; the windows are scored from the
backup, so the check is idempotent too). --check writes nothing. A JSON of every
flagged window goes to --report.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_tools.gt_gaps import split_runs  # noqa: E402
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
from dataset_tools.gt_map import TAG_MAP, Mismatch, mismatch_intervals, window_offset  # noqa: E402

BACKUP_TRAJ = "odometry_premapcut.txt"
BACKUP_SEGS = "odometry_segments_premapcut.txt"


USE_BACKUPS = True  # set False (--live) to score the files as published, ignoring regen/ backups


def _sources(scene: Path):
    regen = scene / "regen"
    if USE_BACKUPS and (regen / BACKUP_TRAJ).exists() and (regen / BACKUP_SEGS).exists():
        return regen / BACKUP_TRAJ, regen / BACKUP_SEGS
    return scene / "odometry.txt", scene / "odometry_segments.txt"


def scan_scene(args):
    scene, horizon, max_off, use_backups = args
    global USE_BACKUPS
    USE_BACKUPS = use_backups
    scene = Path(scene)
    src_traj, src_segs = _sources(scene)
    t, T = read_trajectory(src_traj)
    segs = read_segments(src_segs)
    ga = np.load(scene / "gravity_align.npy")
    pts = np.loadtxt(scene / "pcd_ts.txt")
    hits, n_win = [], 0
    for seg in segs:
        for f in np.where((pts >= seg.start_ts) & (pts + horizon <= seg.end_ts + 1e-3))[0]:
            if f >= len(ga):
                break
            mp = scene / "height_maps" / f"{f:05d}.npz"
            mk = scene / "height_masks" / f"{f:05d}.npz"
            if not (mp.exists() and mk.exists()):
                continue
            t0 = float(pts[f])
            m = (t >= t0) & (t <= t0 + horizon)
            if m.sum() < 5:
                continue
            res = window_offset(T[m], ga[f], np.load(mp)["height_map"], np.load(mk)["validity"])
            if res is None:
                continue
            n_win += 1
            if abs(res[0]) > max_off:
                hits.append(Mismatch(int(f), t0, res[0], res[1]))
    return scene.name, n_win, hits


def cut_scene(scene: Path, hits, horizon, margin, min_poses, check) -> dict:
    src_traj, src_segs = _sources(scene)
    t, T = read_trajectory(src_traj)
    segs = read_segments(src_segs)
    cuts = mismatch_intervals(hits, horizon, margin)
    kept_ts, kept_T, rows = [], [], []
    st = dict(
        segments_in=len(segs),
        poses_in=len(t),
        span_in=0.0,
        span_out=0.0,
        poses_out=0,
        cut_s=0.0,
        dropped_short=0,
        cuts=len(cuts),
    )
    for seg in segs:
        idx = np.where((t >= seg.start_ts) & (t <= seg.end_ts))[0]
        if len(idx) == 0:
            continue
        ts, Ts = t[idx], T[idx]
        st["span_in"] += ts[-1] - ts[0]
        bad = np.zeros(len(ts), dtype=bool)
        for a, b in cuts:
            bad |= (ts >= a) & (ts <= b)
        if len(ts) > 1:
            st["cut_s"] += float(np.sum(np.diff(ts)[bad[1:] | bad[:-1]]))
        for r0, r1, ended in split_runs(len(ts), np.zeros(len(ts) - 1, bool), [], bad):
            run_t, run_T = ts[r0:r1], Ts[r0:r1]
            ends_at_cut = ended is not None  # a cut pose follows inside this input segment
            if len(run_t) < min_poses:
                st["dropped_short"] += len(run_t)
            else:
                kept_ts.append(run_t)
                kept_T.append(anchor_to_first(run_T))
                if ends_at_cut:
                    prefix = seg.reason.split(":", 1)[0] if ":" in seg.reason else "filtered"
                    reason = f"{prefix}:{TAG_MAP}"
                else:
                    reason = seg.reason
                rows.append(SegmentRow(len(rows), run_t[0], run_t[-1], len(run_t), reason))
                st["span_out"] += run_t[-1] - run_t[0]
                st["poses_out"] += len(run_t)
    st["segments_out"] = len(rows)
    if not check:
        regen = scene / "regen"
        if src_traj == scene / "odometry.txt" and not regen.is_symlink():
            regen.mkdir(exist_ok=True)
            for name in (BACKUP_TRAJ, BACKUP_SEGS):
                # A derived root links its parent's regen/ files; copying THROUGH
                # such a link would overwrite the parent's history. Unlink first
                # so the backup is this root's own file.
                if (regen / name).is_symlink():
                    (regen / name).unlink()
            shutil.copy2(scene / "odometry.txt", regen / BACKUP_TRAJ)
            shutil.copy2(scene / "odometry_segments.txt", regen / BACKUP_SEGS)
        write_trajectory(
            scene / "odometry.txt",
            np.concatenate(kept_ts) if kept_ts else t[:0],
            np.concatenate(kept_T) if kept_T else T[:0],
        )
        write_segments(scene / "odometry_segments.txt", rows)
        if not regen.is_symlink():
            record_written(scene, BACKUP_TRAJ)
    return st


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--horizon", type=float, default=4.0, help="s of path scored per keyframe (DRIFT's label horizon)")
    ap.add_argument(
        "--max-off", type=float, default=0.3, help="m; median wheel-bottom minus map beyond this is a mismatch"
    )
    ap.add_argument("--margin", type=float, default=0.5, help="s cut either side of a flagged window")
    ap.add_argument("--min-poses", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--report", type=Path, default=None, help="JSON of every flagged window")
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    ap.add_argument(
        "--live",
        action="store_true",
        help="score/cut the published files even if regen/ holds a pre-cut backup "
        "(a root whose regen/ is a symlink to another root's must use this)",
    )
    ap.add_argument(
        "--force", action="store_true", help="rebuild from the regen/ backup even if odometry.txt changed since"
    )
    a = ap.parse_args()
    global USE_BACKUPS
    USE_BACKUPS = not a.live
    scenes = a.scenes or sorted(p.name for p in a.root.iterdir() if p.is_dir() and p.name.isdigit())
    if USE_BACKUPS and not a.check and not a.force:
        changed = [sc for sc in scenes if changed_since_written(a.root / sc, BACKUP_TRAJ)]
        if changed:
            print(
                f"odometry.txt changed since this pass last wrote it in {', '.join(changed)}; rebuilding from "
                f"regen/{BACKUP_TRAJ} would undo that. Move the backups aside, or --force",
                file=sys.stderr,
            )
            return 1
    with Pool(a.jobs) as pool:
        scanned = pool.map(scan_scene, [(str(a.root / sc), a.horizon, a.max_off, USE_BACKUPS) for sc in scenes])
    report, tot = {}, {}
    print(
        f"{'scene':5s} {'windows':>7s} {'flagged':>7s} {'segs':>9s} {'poses':>13s} {'span_s':>15s} {'cut_s':>6s} {'short':>5s}"
    )
    for sc, n_win, hits in scanned:
        st = cut_scene(a.root / sc, hits, a.horizon, a.margin, a.min_poses, a.check)
        report[sc] = [dict(frame=h.frame, t0=h.t0, offset=round(h.offset, 3), n=h.n) for h in hits]
        for k, v in st.items():
            tot[k] = tot.get(k, 0) + v
        tot["windows"] = tot.get("windows", 0) + n_win
        tot["flagged"] = tot.get("flagged", 0) + len(hits)
        print(
            f"{sc:5s} {n_win:7d} {len(hits):7d} {st['segments_in']:3d}->{st['segments_out']:<4d} "
            f"{st['poses_in']:6d}->{st['poses_out']:<6d} {st['span_in']:6.0f}->{st['span_out']:<7.0f} "
            f"{st['cut_s']:6.0f} {st['dropped_short']:5d}",
            flush=True,
        )
    print(
        f"TOTAL {int(tot['windows']):7d} {int(tot['flagged']):7d} {int(tot['segments_in']):3d}->{int(tot['segments_out']):<4d} "
        f"{int(tot['poses_in']):6d}->{int(tot['poses_out']):<6d} {tot['span_in']:6.0f}->{tot['span_out']:<7.0f} "
        f"{tot['cut_s']:6.0f} {int(tot['dropped_short']):5d}" + ("  (check only, nothing written)" if a.check else "")
    )
    if a.report:
        with open(a.report, "w") as fh:
            json.dump(report, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
