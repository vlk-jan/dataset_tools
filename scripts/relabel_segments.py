#!/usr/bin/env python3
"""Retag published segments whose `reason` was inherited rather than measured.

Until the fix in ``filter_segments.py``, that filter stamped every run it cut
out of an input segment with the *input* segment's tag. Only the run reaching
the input segment's end actually ended for that reason; the runs before it end
where the filter's cross-check against live ``/odom`` dropped poses. So a
segment could claim "the recording ended" 97 s before the recording ended
(HELHEST_v1_3 scene 0000, segment 7).

Fixing the filter does not fix the roots it already wrote, and those roots no
longer carry the pre-filter trajectory the filter would need to run again. They
do not have to: the two segment tables alone say which run reached its parent's
end, so this is a pure relabel of the `reason` column --

    <scene>/odometry_segments.txt              the published table
    <scene>/regen/odometry_segments_prefilter.txt   its input

A published segment ending before its parent's end gets ``filtered:odom_mismatch``.
Nothing else changes: no pose, no boundary, no count. Only ``filtered:`` rows are
touched, since those are the ones that filter wrote; a ``v11_fill`` stretch
merged in from another root names its provenance rather than an end, and one can
sit inside a prefilter segment's span, so the parent check alone would not spare
it.

    relabel_segments.py <root> [--apply] [--scenes 0000 0003 ...]

Without ``--apply`` it only reports. It is idempotent: a relabelled segment
still ends before its parent's end, so a second run finds nothing to do.
"""

import argparse
from pathlib import Path

#: Tag for a segment that ends where the post-hoc /odom filter cut it, under the
#: prefix ``filter_segments.py`` puts on every row it writes.
PREFIX = "filtered:"
CUT_TAG = PREFIX + "odom_mismatch"
#: A published end may differ from its parent's by a printing round-trip, not
#: by a dropped pose; both tables print %.6f from the same timestamps.
END_TOL = 1e-3


def read_segments(path: Path):
    """``(index, start_ts, end_ts, n_poses, reason)`` per data row, in file order."""
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        rows.append((len(rows), float(f[1]), float(f[2]), int(f[3]), f[4] if len(f) > 4 else ""))
    return rows


def plan_scene(scene_dir: Path):
    """Segments of one scene whose reason is inherited, as ``(index, old, new)``."""
    post_p = scene_dir / "odometry_segments.txt"
    pre_p = scene_dir / "regen" / "odometry_segments_prefilter.txt"
    if not (post_p.exists() and pre_p.exists()):
        return []
    parents = read_segments(pre_p)
    out = []
    for i, s0, s1, _, reason in read_segments(post_p):
        if not reason.startswith(PREFIX) or reason == CUT_TAG:
            continue
        # The parent is the input segment this run was carved out of: the one
        # whose span contains it. They are disjoint, so at most one matches.
        parent = next((p for p in parents if p[1] - END_TOL <= s0 and s1 <= p[2] + END_TOL), None)
        if parent is not None and abs(parent[2] - s1) > END_TOL:
            out.append((i, reason, CUT_TAG))
    return out


def apply_scene(scene_dir: Path, changes) -> None:
    """Rewrite only the reason column of the named rows, in place."""
    path = scene_dir / "odometry_segments.txt"
    new_reason = {i: new for i, _, new in changes}
    lines, row = [], 0
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            lines.append(line)
            continue
        if row in new_reason:
            f = line.split()
            f[4] = new_reason[row]
            line = " ".join(f)
        lines.append(line)
        row += 1
    # Rename over the original so a reader never sees a half-written table.
    tmp = path.with_suffix(".txt.tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--apply", action="store_true", help="write the tables (default: report only)")
    ap.add_argument("--scenes", nargs="*", help="scene ids to consider (default: all)")
    a = ap.parse_args()

    root = Path(a.root)
    scenes = [root / s for s in a.scenes] if a.scenes else sorted(d for d in root.glob("[0-9]" * 4) if d.is_dir())
    total = 0
    for scene_dir in scenes:
        changes = plan_scene(scene_dir)
        if not changes:
            continue
        total += len(changes)
        for i, old, new in changes:
            print(f"{scene_dir.name} segment {i}: {old} -> {new}")
        if a.apply:
            apply_scene(scene_dir, changes)
    print(f"{'relabelled' if a.apply else 'would relabel'} {total} segment(s) in {len(scenes)} scene(s)")


if __name__ == "__main__":
    main()
