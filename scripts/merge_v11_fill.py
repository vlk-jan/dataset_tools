#!/usr/bin/env python3
"""Fill v1.2's ground-truth gaps with quality-gated v1.1 odometry.

    merge_v11_fill.py <v11_root> <v12_root> [scenes...]

Keeps every v1.2 segment (the paired comparison shows v1.2 wins wherever both
tracked), takes v1.1 restricted to v1.2's gaps, applies THE SAME gate as
filter_segments.py (2 s windows vs recorded /odom, drop > 1 m, keep runs >= 20
poses, re-anchor at identity), and appends the survivors tagged ``v11_fill`` so
provenance stays visible per segment.

Idempotent: the pure-v1.2 files are preserved as ``regen/*_v12only.txt`` before
the first merge, and a table already containing ``v11_fill`` is skipped. Every
processed scene gets a ``regen/merge_v11_fill.txt`` record, so a scene that was
processed but not filled stays distinguishable from one never processed.
"""
import datetime
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from filter_segments import HEADER, bad_poses, load_traj  # noqa: E402

WINDOW, MAX_ERR, MIN_POSES, MARGIN, MAX_GAP = 2.0, 1.0, 20, 0.15, 1.0


def load_segment_rows(p):
    rows = []
    for line in open(p):
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        rows.append([float(f[1]), float(f[2]), int(f[3]), f[4] if len(f) > 4 else ""])
    return rows


def fill_segments(t1, T1, segs12, rt, rT):
    """Return [(ts_array, anchored_T_array)] of gated v1.1 runs in v1.2 gaps."""
    covered = np.zeros(len(t1), bool)
    for a, b, *_ in segs12:
        covered |= (t1 >= a - MARGIN) & (t1 <= b + MARGIN)
    idx = np.where(~covered)[0]
    out = []
    if len(idx) == 0:
        return out
    splits = np.where(np.diff(t1[idx]) > MAX_GAP)[0] + 1
    for run in np.split(idx, splits):
        if len(run) < MIN_POSES:
            continue
        tt, TT = t1[run], T1[run]
        bad = bad_poses(tt, TT, rt, rT, WINDOW, MAX_ERR)
        good = ~bad
        edges = np.diff(np.concatenate([[0], good.astype(int), [0]]))
        for r0, r1 in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
            if r1 - r0 < MIN_POSES:
                continue
            anchored = np.einsum(
                "ij,njk->nik", np.linalg.inv(TT[r0]), TT[r0:r1]
            )
            out.append((tt[r0:r1], anchored))
    return out


def record(d12, outcome, params="window=2.0 max_err=1.0 min_poses=20"):
    with open(d12 / "regen" / "merge_v11_fill.txt", "w") as f:
        f.write(f"processed={datetime.datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"params={params}\noutcome={outcome}\n")


def merge_scene(v11, v12, scene):
    d12 = Path(v12) / scene
    seg_path = d12 / "odometry_segments.txt"
    odo_path = d12 / "odometry.txt"
    p11 = Path(v11) / scene / "odometry.txt"
    ref_path = d12 / "regen" / "odom_ref.txt"
    if not p11.exists():
        record(d12, "no_v11_odometry")
        return f"{scene}: no v1.1 odometry, nothing to fill"
    if "v11_fill" in seg_path.read_text():
        return f"{scene}: already merged, skipping"
    t1, T1 = load_traj(p11)
    segs12 = load_segment_rows(seg_path)
    rt, rT = load_traj(ref_path)
    fills = fill_segments(t1, T1, segs12, rt, rT)
    if not fills:
        record(d12, "no_fill_survives_gate")
        return f"{scene}: no fill survives the gate"
    # preserve the pure-v1.2 files once
    for src, dst in ((odo_path, "odometry_v12only.txt"), (seg_path, "odometry_segments_v12only.txt")):
        target = d12 / "regen" / dst
        if not target.exists():
            shutil.copy(src, target)
    # existing rows grouped per segment, then merge in time order
    t2, T2 = load_traj(odo_path)
    pieces = []
    for a, b, n, reason in segs12:
        m = (t2 >= a) & (t2 <= b)
        pieces.append((t2[m], T2[m], reason))
    for tt, TT in fills:
        pieces.append((tt, TT, "v11_fill"))
    pieces.sort(key=lambda p: p[0][0])
    with open(odo_path, "w") as fo, open(seg_path, "w") as fsg:
        fo.write(HEADER + "\n")
        fsg.write("# segment start_ts end_ts n_poses reason\n")
        for k, (tt, TT, reason) in enumerate(pieces):
            for t, T in zip(tt, TT):
                fo.write(" ".join(f"{v:.6f}" for v in (t, *T.ravel())) + "\n")
            fsg.write(f"{k} {tt[0]:.6f} {tt[-1]:.6f} {len(tt)} {reason}\n")
    span = sum(p[0][-1] - p[0][0] for p in pieces if p[2] == "v11_fill")
    record(d12, f"filled:{len(fills)}_segments_{span:.1f}s")
    return (f"{scene}: MERGED +{len(fills)} fill segments, +{span:.1f} s "
            f"({sum(len(p[0]) for p in pieces if p[2]=='v11_fill')} poses)")


def main():
    v11, v12 = sys.argv[1], sys.argv[2]
    scenes = sys.argv[3:] or sorted(
        d.name for d in Path(v12).iterdir() if d.is_dir() and d.name.isdigit()
    )
    for s in scenes:
        print(merge_scene(v11, v12, s))


if __name__ == "__main__":
    main()
