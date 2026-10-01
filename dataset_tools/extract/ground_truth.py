"""odometry.txt + odometry_segments.txt from an absolute pose source."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dataset_tools import gnss_guard
from dataset_tools.gt_io import SegmentRow, anchor_to_first, write_segments, write_trajectory


def _auto(mode: str, odom: str) -> bool:
    """``on``/``off`` as given; ``auto`` is on only for an absolute GNSS/INS source.

    --gt: the odometry is its own ground truth only on the new rig, whose
    sources are the Odin1's lidar-inertial odometry and the Fixposition
    GNSS/INS; wheel-IMU /odom drifts and the old rig's ground truth comes from
    the offline FAST-LIO regeneration. --gnss_guard: the full guard (reset
    solutions) is for the Fixposition; the Odin1 odometry gets its coverage
    half only (:func:`dataset_tools.gnss_guard.coverage_guard`). Wheel-IMU
    /odom, being an integration, legitimately holds a bit-identical pose for
    as long as the robot stands still -- which is the guard's own signature
    for a fabricated stretch, so running it there would delete exactly the
    frames where the robot was parked.
    """
    return mode == "on" or (mode == "auto" and is_ground_truth_source(odom))


#: Odometry topics the dataset takes as ground truth as recorded: the Odin1's
#: own odometry (v2.3 on) and the Fixposition GNSS/INS (v2 beta to v2.2).
GROUND_TRUTH_SOURCES = ("odin1/odometry", "fixposition")


def is_ground_truth_source(odom: str) -> bool:
    return any(key in odom for key in GROUND_TRUTH_SOURCES)


def write_ground_truth(
    odom_data, scene_dir: Path, max_gap=1.0, max_speed=gnss_guard.MAX_SPEED, min_poses=20, spans=None
) -> bool:
    """Write odometry.txt + odometry_segments.txt in the HELHEST v1.2 contract.

    For rigs whose odometry source is trustworthy on its own (Fixposition
    GNSS/INS), the recorded poses ARE the ground truth -- but consumers expect
    the v1.2 shape: pose rows at full rate, split into segments that are each
    re-anchored to the identity, with a segment table that must never be
    bridged. Segments break at time gaps > ``max_gap`` and at teleports
    (displacement implying > ``max_speed`` m/s, e.g. GNSS out-and-back spikes);
    spike samples end up in tiny segments that ``min_poses`` discards.

    Each segment's ``reason`` names why *that* segment ended: ``gap``,
    ``teleport``, ``fix_lost`` for a hole :mod:`dataset_tools.gnss_guard` left
    behind when it removed a fabricated stretch, or ``end_of_run`` for the last
    one.
    """
    if len(odom_data) < min_poses:
        return False
    data = sorted(odom_data, key=lambda x: x[0])
    ts = np.array([x[0] for x in data])
    Ts = np.array([x[1] for x in data])
    dt = np.diff(ts)
    step = np.linalg.norm(np.diff(Ts[:, :3, 3], axis=0), axis=1)
    breaks = np.where((dt > max_gap) | (step > max_speed * np.maximum(dt, 1e-3)))[0] + 1
    spans = list(spans or [])

    def _reason(b: int) -> str:
        # A hole the guard opened is named for what opened it, not for the fact
        # that it is a hole: "gap" would say the recording stopped, which is
        # not what happened.
        if any(ts[b - 1] <= sp.end and sp.start <= ts[b] for sp in spans):
            return "fix_lost"
        return "gap" if dt[b - 1] > max_gap else "teleport"

    reasons = [_reason(int(b)) for b in breaks]
    starts = [0, *breaks]
    ends = [*breaks, len(ts)]
    kept_ts: list[np.ndarray] = []
    kept_poses: list[np.ndarray] = []
    rows: list[SegmentRow] = []
    for k, (a, b) in enumerate(zip(starts, ends, strict=True)):
        if b - a < min_poses:
            continue
        kept_ts.append(ts[a:b])
        kept_poses.append(anchor_to_first(Ts[a:b]))
        # The tag names why this segment ENDED -- the break that closes it,
        # not the one that opened it -- so it means the same thing here as
        # in the regenerated (FAST-LIO) tables the viewer also reads.
        reason = reasons[k] if k < len(reasons) else "end_of_run"
        rows.append(SegmentRow(len(rows), float(ts[a]), float(ts[b - 1]), b - a, reason))
    write_trajectory(
        scene_dir / "odometry.txt", np.concatenate(kept_ts or [ts[:0]]), np.concatenate(kept_poses or [Ts[:0]])
    )
    write_segments(scene_dir / "odometry_segments.txt", rows)
    return bool(rows)
