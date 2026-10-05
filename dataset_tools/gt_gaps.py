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

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from dataset_tools.gt_io import SegmentRow, anchor_to_first

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


def window_errors(
    t: np.ndarray, T: np.ndarray, rt: np.ndarray, rT: np.ndarray, window: float
) -> list[tuple[int, int, float]]:
    """``(i, j, err)`` for every ``window``-second stretch of ``T`` the reference covers.

    ``j`` is the first pose at least ``window`` after ``i`` (skipped when that is
    over half a second late); ``err`` is the distance between the translation
    ``T`` and ``rT`` each report across ``[t_i, t_j]``. Both reference
    endpoints must lie within 0.2 s of the poses'. The window test of
    ``scripts/filter_segments.py`` and the atlas's 2 s error both read this.
    """
    out = []
    for i in range(len(t)):
        j = np.searchsorted(t, t[i] + window)
        if j >= len(t) or t[j] - t[i] > window + 0.5:
            continue
        ri, rj = np.searchsorted(rt, t[i]), np.searchsorted(rt, t[j])
        if ri >= len(rt) or rj >= len(rt) or abs(rt[ri] - t[i]) > 0.2 or abs(rt[rj] - t[j]) > 0.2:
            continue
        rel = np.linalg.inv(T[i]) @ T[j]
        rel_ref = np.linalg.inv(rT[ri]) @ rT[rj]
        out.append((i, int(j), float(np.linalg.norm(rel[:3, 3] - rel_ref[:3, 3]))))
    return out


def cut_reason(reason: str, tag: str) -> str:
    """The tag of a run a post-processing pass cut short: the pass prefix of the
    input's ``reason`` (``filtered`` when it has none) and why this cut."""
    prefix = reason.split(":", 1)[0] if ":" in reason else "filtered"
    return f"{prefix}:{tag}"


def cut_segments(
    t: np.ndarray,
    T: np.ndarray,
    segs: list[SegmentRow],
    mark: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, list[str], np.ndarray | None]],
    retag: Callable[[str, str | None], str],
    min_poses: int,
) -> tuple[np.ndarray, np.ndarray, list[SegmentRow], dict]:
    """Split every segment into runs, drop the short ones and re-anchor the rest.

    ``mark(ts, Ts)`` returns ``(seam, tags, bad)`` for one segment's poses, as
    :func:`split_runs` takes them; ``retag(reason, cut)`` names a kept run from
    its input segment's reason and :func:`split_runs`' third field. Returns the
    kept poses, their segment rows, and ``span_in``/``span_out``/``poses_out``/
    ``dropped_short``. Re-anchoring is rigid, so every relative motion inside a
    run is what it was.
    """
    kept_ts, kept_T, rows = [], [], []
    st = dict(span_in=0.0, span_out=0.0, poses_out=0, dropped_short=0)
    for seg in segs:
        idx = np.where((t >= seg.start_ts) & (t <= seg.end_ts))[0]
        if len(idx) == 0:
            continue
        ts, Ts = t[idx], T[idx]
        st["span_in"] += ts[-1] - ts[0]
        seam, tags, bad = mark(ts, Ts)
        for r0, r1, cut in split_runs(len(ts), seam, tags, bad):
            if r1 - r0 < min_poses:
                st["dropped_short"] += r1 - r0
                continue
            run_t = ts[r0:r1]
            kept_ts.append(run_t)
            kept_T.append(anchor_to_first(Ts[r0:r1]))
            rows.append(SegmentRow(len(rows), run_t[0], run_t[-1], len(run_t), retag(seg.reason, cut)))
            st["span_out"] += run_t[-1] - run_t[0]
            st["poses_out"] += len(run_t)
    out_t = np.concatenate(kept_ts) if kept_ts else t[:0]
    out_T = np.concatenate(kept_T) if kept_T else T[:0]
    return out_t, out_T, rows, st
