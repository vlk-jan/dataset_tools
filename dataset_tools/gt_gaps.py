"""Holes in a ground-truth pose stream, judged by what the wheels did meanwhile.

The post-hoc /odom filter (``scripts/filter_segments.py``) scores 2 s windows of
poses against the live odometry. A *hole* -- FAST-LIO publishing nothing for
seconds to minutes while the robot keeps driving -- has no window to score: the
only pose pair that spans it is skipped because the window cannot be closed in
time, so the hole survives inside one segment and every consumer that
interpolates within a segment (the viewer, DRIFT's trajectory loader) draws a
straight line through it. On HELHEST v1.3 that was 11 % of the GT span, with
1.6 km of driving inside the holes (2026-09-07).

The recorder's live detector misses the same holes for a related reason: its
jump allowance is ``reference_speed * dt`` and grows without bound with the gap,
and its speed test needs the mismatch to persist for a second, which a single
gap-closing sample never does.

Two rules, applied to every pair of consecutive poses inside a segment:

* **wheel-distance**: the reference (live /odom, wheel-derived) travelled more
  than ``max_err`` further across the gap than the GT pose moved. Inert on a
  dense stream (a 0.1 s step cannot accumulate 1 m of disagreement), it bites
  exactly on holes the robot drove through.
* **max-gap**: the gap is longer than ``max_gap`` seconds. Even when the
  endpoints happen to agree with the wheels, nothing between them is known.

A pair failing either rule is a *seam*: the segment is cut there, and the two
sides are re-anchored as separate segments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Tags written into ``odometry_segments.txt`` for a run that ends at a seam.
TAG_WHEEL = "wheel_gap"
TAG_GAP = "gap"
#: Tag for a run that ends because the window test dropped the pose after it.
TAG_MISMATCH = "odom_mismatch"


@dataclass
class ReferencePath:
    """Cumulative path length of a reference trajectory, sampled on its own clock.

    The live /odom is logged at ~100 Hz; differencing it at that rate turns
    sensor jitter into path length while the robot stands still, so the
    trajectory is sub-sampled to at least ``min_dt`` between samples first.
    """

    ts: np.ndarray
    cum: np.ndarray

    @classmethod
    def from_trajectory(cls, rt: np.ndarray, rT: np.ndarray, min_dt: float = 0.1) -> ReferencePath:
        rt = np.asarray(rt, dtype=np.float64)
        p = np.asarray(rT)[:, :3, 3]
        if len(rt) == 0:
            return cls(rt, np.zeros(0))
        keep = [0]
        last = rt[0]
        for i in range(1, len(rt)):
            if rt[i] - last >= min_dt:
                keep.append(i)
                last = rt[i]
        idx = np.asarray(keep)
        steps = np.linalg.norm(np.diff(p[idx], axis=0), axis=1)
        return cls(rt[idx], np.concatenate([[0.0], np.cumsum(steps)]))

    def length_between(self, t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
        """Path length travelled in ``[t0, t1]``; NaN where the reference does not cover the interval."""
        t0 = np.asarray(t0, dtype=np.float64)
        t1 = np.asarray(t1, dtype=np.float64)
        if len(self.ts) < 2:
            return np.full(np.shape(t0), np.nan)
        out = np.interp(t1, self.ts, self.cum) - np.interp(t0, self.ts, self.cum)
        covered = (t0 >= self.ts[0]) & (t1 <= self.ts[-1])
        return np.where(covered, out, np.nan)


def gap_seams(
    t: np.ndarray,
    T: np.ndarray,
    ref: ReferencePath | None,
    max_gap: float = 1.0,
    max_err: float = 1.0,
) -> tuple[np.ndarray, list[str]]:
    """Which consecutive pose pairs of one segment must not be bridged.

    Returns ``(seam, tags)``, both of length ``len(t) - 1``: ``seam[k]`` is True
    when poses ``k`` and ``k + 1`` belong to different segments; ``tags[k]`` is
    ``TAG_WHEEL`` when the wheel-distance rule fired there, else ``TAG_GAP``.
    """
    t = np.asarray(t, dtype=np.float64)
    n = len(t)
    if n < 2:
        return np.zeros(0, bool), []
    dt = np.diff(t)
    step = np.linalg.norm(np.diff(np.asarray(T)[:, :3, 3], axis=0), axis=1)
    too_long = dt > max_gap
    if ref is not None:
        travelled = ref.length_between(t[:-1], t[1:])
        with np.errstate(invalid="ignore"):
            wheels = (travelled - step) > max_err
        wheels &= np.isfinite(travelled)
    else:
        wheels = np.zeros(n - 1, bool)
    seam = too_long | wheels
    tags = [TAG_WHEEL if w else TAG_GAP for w in wheels]
    return seam, tags


def split_runs(
    n: int,
    seam: np.ndarray,
    tags: list[str],
    bad: np.ndarray | None = None,
) -> list[tuple[int, int, str | None]]:
    """Runs ``[r0, r1)`` of poses that stay together, with why each run ended.

    ``bad[k]`` marks poses to drop (from the window test); ``seam[k]`` cuts
    between ``k`` and ``k + 1`` for the reason ``tags[k]``. The third field is
    ``None`` for the run that reaches the end of the input (it inherits the
    input segment's tag), ``TAG_MISMATCH`` when the next pose was dropped, or
    the seam's tag.
    """
    if bad is None:
        bad = np.zeros(n, bool)
    runs: list[tuple[int, int, str | None]] = []
    r0: int | None = None
    for k in range(n):
        if bad[k]:
            if r0 is not None:
                runs.append((r0, k, TAG_MISMATCH))
                r0 = None
            continue
        if r0 is None:
            r0 = k
        if k < n - 1 and seam[k]:
            runs.append((r0, k + 1, tags[k]))
            r0 = None
    if r0 is not None:
        runs.append((r0, n, None))
    return runs
