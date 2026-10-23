"""odometry.txt + odometry_segments.txt from an absolute pose source."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dataset_tools import gnss_guard
from dataset_tools.gt_io import SegmentRow, anchor_to_first, write_segments, write_trajectory


def _auto(mode: str, odom: str) -> bool:
    """``on``/``off`` as given; ``auto`` is on only for a ground-truth source (Odin1, Fixposition).

    Wheel /odom drifts and holds still when parked, which the guard would read as fabricated.
    """
    return mode == "on" or (mode == "auto" and any(key in odom for key in GROUND_TRUTH_SOURCES))


#: Odometry topics taken as ground truth: Odin1 (v2.3 on), Fixposition (v2 beta to v2.2).
GROUND_TRUTH_SOURCES = ("odin1/odometry", "fixposition")


#: Segments break at gaps over MAX_GAP_S s; shorter than MIN_POSES poses they are dropped.
MAX_GAP_S, MIN_POSES = 1.0, 20


def write_ground_truth(odom_data, scene_dir: Path, spans=None) -> bool:
    """Write odometry.txt + odometry_segments.txt in the HELHEST v1.2 contract.

    Segments break at gaps > ``MAX_GAP_S`` and steps faster than ``gnss_guard.MAX_SPEED``;
    segments under ``MIN_POSES`` are dropped. Each ``reason`` says why the segment
    ended: ``gap``, ``teleport``, ``fix_lost`` (a guard span) or ``end_of_run``.
    """
    if len(odom_data) < MIN_POSES:
        return False
    data = sorted(odom_data, key=lambda x: x[0])
    ts = np.array([x[0] for x in data])
    Ts = np.array([x[1] for x in data])
    dt = np.diff(ts)
    step = np.linalg.norm(np.diff(Ts[:, :3, 3], axis=0), axis=1)
    breaks = np.where((dt > MAX_GAP_S) | (step > gnss_guard.MAX_SPEED * np.maximum(dt, 1e-3)))[0] + 1
    spans = list(spans or [])

    def _reason(b: int) -> str:
        if any(ts[b - 1] <= sp.end and sp.start <= ts[b] for sp in spans):
            return "fix_lost"
        return "gap" if dt[b - 1] > MAX_GAP_S else "teleport"

    reasons = [_reason(int(b)) for b in breaks]
    starts = [0, *breaks]
    ends = [*breaks, len(ts)]
    kept_ts: list[np.ndarray] = []
    kept_poses: list[np.ndarray] = []
    rows: list[SegmentRow] = []
    for k, (a, b) in enumerate(zip(starts, ends, strict=True)):
        if b - a < MIN_POSES:
            continue
        kept_ts.append(ts[a:b])
        kept_poses.append(anchor_to_first(Ts[a:b]))
        # Why this segment ENDED, as in the regenerated (FAST-LIO) tables.
        reason = reasons[k] if k < len(reasons) else "end_of_run"
        rows.append(SegmentRow(len(rows), float(ts[a]), float(ts[b - 1]), b - a, reason))
    write_trajectory(
        scene_dir / "odometry.txt", np.concatenate(kept_ts or [ts[:0]]), np.concatenate(kept_poses or [Ts[:0]])
    )
    write_segments(scene_dir / "odometry_segments.txt", rows)
    return bool(rows)
