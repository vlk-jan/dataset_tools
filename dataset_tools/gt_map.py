"""Stretches where the ground-truth path leaves the lidar's own ground.

The ground truth is FAST-LIO. Where it degenerates (snow, featureless
vegetation) it keeps publishing a smooth, plausible-looking pose stream whose z
and attitude are wrong by metres and tens of degrees; the wheel checks in
DRIFT's ``audit_odometry.py`` cannot see it (nothing kinematically impossible
happens), the attitude check stops just under its gate, and the gap rules in
:mod:`dataset_tools.gt_gaps` see no hole. The height map is the one witness
that is independent of the pose stream: it is built from a single scan,
levelled by the IMU, and it says where the ground was.

The rule, per keyframe: level the next ``horizon`` seconds of poses in the
keyframe's own frame (exactly as DRIFT's loader does, ``R_align @ inv(T0)``),
put the three wheels on the ground under each pose and read the height map of
that keyframe at the cells they cross. Only cells the lidar MEASURED count
(validity 2 -- never anchored or interpolated cells, whose height is not
independent). A window whose median wheel-bottom-minus-map exceeds ``max_off``
is a *mismatch*; the poses it covers are cut out of the segment, like a hole.

Found on HELHEST v1.3 (2026-09-09): 0017 seg 17 t 4-32 s sits 0.4-0.9 m BELOW
the ground it measured, 0036 seg 17 t 58-98 s ~1 m above, with 14 more
segments carrying one to three such windows; 63 of 4900 sampled windows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Tag written into ``odometry_segments.txt`` for a run that ends at a mismatch cut.
TAG_MAP = "map_mismatch"

#: HELHEST wheel contacts in base_link (URDF: base_link is the front axle) and the wheel radius.
WHEELS = np.array([[0.0, 0.36], [0.0, -0.36], [-0.697, 0.0]])
WHEEL_RADIUS = 0.36
#: Height-map layout DRIFT ships: 128x128 cells of 0.1 m, robot at the centre, row = x, col = y.
MAP_RES = 0.1
MAP_N = 128
VALIDITY_MEASURED = 2


@dataclass
class Mismatch:
    frame: int
    t0: float
    offset: float  # median wheel-bottom minus map height over the window, metres
    n: int  # measured cells behind the median


def window_offset(T_window, R_align, height_map, validity, min_cells: int = 5):
    """Median wheel-bottom-minus-map over one window, or None with too few measured cells."""
    lev = np.eye(4)
    lev[:3, :3] = R_align
    P = np.einsum("ij,njk->nik", lev @ np.linalg.inv(T_window[0]), T_window)
    offs = []
    for k in range(0, len(P), 2):
        for wx, wy in WHEELS:
            p = P[k] @ np.array([wx, wy, 0.0, 1.0])
            r = int(round(p[0] / MAP_RES + (MAP_N - 1) / 2))
            c = int(round(p[1] / MAP_RES + (MAP_N - 1) / 2))
            if 0 <= r < MAP_N and 0 <= c < MAP_N and validity[r, c] == VALIDITY_MEASURED:
                offs.append(p[2] - WHEEL_RADIUS - height_map[r, c])
    if len(offs) < min_cells:
        return None
    return float(np.median(offs)), len(offs)


def mismatch_intervals(hits, horizon: float, margin: float):
    """Merge flagged windows into absolute-time intervals to cut, with a margin either side."""
    iv = sorted((h.t0 - margin, h.t0 + horizon + margin) for h in hits)
    out = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]
