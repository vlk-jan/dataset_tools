"""Long stops cut down to their start (``--trim_idle MAX_S KEEP_S``).

A robot standing still records the same view at 14 Hz: a three-minute stop is
2,600 frames of one scene, which a training sampler draws as often as 2,600
frames of driving. The 2026-10-07 Stromovka bags stand for 74 % of one bag
(four stops of 97-222 s) and end another with 215 s on the asphalt.

A frame is *idle* when, over a :data:`WINDOW_S` window centred on it, the
odometry moved less than :data:`SPEED_MPS` and turned less than
:data:`TURN_DPS` -- turning on the spot is driving. A run of idle frames longer
than ``max_s`` keeps its first ``keep_s`` and loses the rest. The odometry is
not touched (``odometry.txt`` still covers the stop); only frames go, before
any image or cloud is written, so indices stay contiguous and the stop reads as
a frame-clock hole.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

SPEED_MPS = 0.15
TURN_DPS = 5.0
WINDOW_S = 1.0
#: Frames further apart than this do not belong to one run (a hole in the frame clock).
MAX_FRAME_GAP_S = 1.0


@dataclass
class IdleTrim:
    max_s: float
    keep_s: float
    frames_dropped: int = 0
    seconds_dropped: float = 0.0
    #: ``[start, end, frames dropped]`` of each trimmed run, seconds on the frame clock
    runs: list = field(default_factory=list)

    def report(self, t0: float) -> dict:
        return {
            "max_s": self.max_s,
            "keep_s": self.keep_s,
            "frames_dropped": self.frames_dropped,
            "seconds_dropped": round(self.seconds_dropped, 1),
            "runs": [[round(a - t0, 1), round(b - t0, 1), n] for a, b, n in self.runs],
        }


def idle_frames(frame_ts: np.ndarray, odom_ts: np.ndarray, odom_T: np.ndarray) -> np.ndarray:
    """Per frame: whether the odometry stood still around it. Frames the odometry does not bracket are not idle."""
    frame_ts = np.asarray(frame_ts, dtype=np.float64)
    odom_ts = np.asarray(odom_ts, dtype=np.float64)
    if len(odom_ts) < 2:
        return np.zeros(len(frame_ts), bool)
    half = WINDOW_S / 2
    i0 = np.clip(np.searchsorted(odom_ts, frame_ts - half), 0, len(odom_ts) - 1)
    i1 = np.clip(np.searchsorted(odom_ts, frame_ts + half) - 1, 0, len(odom_ts) - 1)
    dt = odom_ts[i1] - odom_ts[i0]
    ok = (dt > 0.5 * WINDOW_S) & (frame_ts >= odom_ts[0]) & (frame_ts <= odom_ts[-1])
    dist = np.linalg.norm(odom_T[i1, :3, 3] - odom_T[i0, :3, 3], axis=1)
    rel = np.einsum("nji,njk->nik", odom_T[i0, :3, :3], odom_T[i1, :3, :3])
    ang = np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    with np.errstate(divide="ignore", invalid="ignore"):
        return ok & (dist / dt < SPEED_MPS) & (ang / dt < TURN_DPS)


def trim(frame_ts: np.ndarray, odom, max_s: float, keep_s: float) -> tuple[np.ndarray, IdleTrim]:
    """``(keep, verdict)``: the frames to keep, every idle run longer than ``max_s`` cut to its first ``keep_s``.

    ``odom`` is ``[(t, T), ...]`` sorted by time, any frame (only speeds and turn rates are read).
    """
    if keep_s > max_s:
        raise ValueError(f"--trim_idle: keep {keep_s} s is longer than the {max_s} s a stop may last")
    frame_ts = np.asarray(frame_ts, dtype=np.float64)
    verdict = IdleTrim(max_s, keep_s)
    keep = np.ones(len(frame_ts), bool)
    if not odom or len(frame_ts) == 0:
        return keep, verdict
    idle = idle_frames(frame_ts, np.array([t for t, _ in odom]), np.array([T for _, T in odom]))
    edges = np.diff(np.r_[0, idle.astype(np.int8), 0])
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        # a run broken by a hole in the frame clock is two runs
        cuts = np.flatnonzero(np.diff(frame_ts[a:b]) > MAX_FRAME_GAP_S) + 1
        for lo, hi in zip(np.r_[0, cuts] + a, np.r_[cuts, b - a] + a):
            t_lo, t_hi = frame_ts[lo], frame_ts[hi - 1]
            if t_hi - t_lo <= max_s:
                continue
            drop = np.arange(lo, hi)[frame_ts[lo:hi] - t_lo > keep_s]
            keep[drop] = False
            verdict.frames_dropped += len(drop)
            verdict.seconds_dropped += float(t_hi - t_lo - keep_s)
            verdict.runs.append([float(t_lo), float(t_hi), int(len(drop))])
    return keep, verdict


def describe(verdict: IdleTrim, t0: float) -> str:
    if not verdict.runs:
        return f"no stop longer than {verdict.max_s:g} s"
    runs = ", ".join(f"{a - t0:.0f}-{b - t0:.0f} s" for a, b, _ in verdict.runs)
    return (
        f"{len(verdict.runs)} stop(s) over {verdict.max_s:g} s cut to their first {verdict.keep_s:g} s "
        f"({runs}): {verdict.frames_dropped} frames, {math.floor(verdict.seconds_dropped)} s dropped"
    )
