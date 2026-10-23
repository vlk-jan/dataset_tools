"""Keep a reset GNSS/INS solution out of the exported poses.

A Fixposition that loses its fix (e.g. on a roll-over) keeps publishing the
reset state: a far-away position held still, attitude at identity, unflagged.
Signatures (attitude is deliberately ignored: real roll-overs flip it):

``teleport``  step faster than ``MAX_SPEED`` over an interval <= ``MAX_GAP``; an edge, never a span alone.
``island``    run fenced by teleports at both ends, shorter than ``MIN_ISLAND``.
``frozen``    translation perfectly still for ``MIN_FREEZE``; only with a teleport or
              no-fix agreeing, since a stuck robot also publishes a constant position.
``no_fix``    hole between usable fixes; a span only if the position also jumped in it,
              and it widens spans that reach into it.

Samples in a span are dropped and frames within it are not exported.
:func:`frame_support` also drops frames no real samples bracket, which
``interpolate_poses`` would otherwise fill with a copied nearest pose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: Speed, m/s, above which a step is not motion; shared by ground truth, interpolation
#: and validation. Real sprints reach 5.35 m/s; the slowest fabricated step is 53 m/s.
MAX_SPEED = 6.0

#: Sample interval, s, beyond which a step is a stream hole, not a teleport.
MAX_GAP = 1.0

#: Translation, m, below which consecutive samples count as the same position.
FREEZE_STEP = 1e-6

#: Seconds a position must hold still to be the reset state (3x the longest genuine still run).
MIN_FREEZE = 1.0

#: Seconds; a teleport-fenced run shorter than this is an excursion.
MIN_ISLAND = 5.0

#: Seconds without a usable fix that count as a lost solution (fix rate is 10 Hz).
MIN_OUTAGE = 2.0

#: Seconds; also ``interpolate_poses``' ``max_gap`` default.
SUPPORT_GAP = 0.5

#: Seconds of margin around a span: the frame clock is not latency-corrected yet when this runs.
FRAME_PAD = 1.0


@dataclass
class Span:
    """A stretch of the pose stream that the receiver fabricated."""

    start: float
    end: float
    reason: str

    def __contains__(self, t: float) -> bool:
        return self.start <= t <= self.end


@dataclass
class Verdict:
    """What the guard found, and what it cost."""

    spans: list[Span] = field(default_factory=list)
    n_poses_dropped: int = 0
    n_frames_dropped: int = 0
    n_poses: int = 0
    n_frames: int = 0
    #: Frames dropped for having no pose under them, beyond those a span cost.
    n_frames_unsupported: int = 0
    #: ``n_frames_unsupported`` split by position relative to the surviving poses.
    n_unsupported_before: int = 0
    n_unsupported_after: int = 0
    n_unsupported_inside: int = 0
    #: Frames dropped after clock correction for lacking a bracketing pose.
    n_frames_trimmed: int = 0

    @property
    def clean(self) -> bool:
        return not self.spans and not self.n_frames_unsupported

    def summary(self, t0: float) -> str:
        parts = []
        if self.spans:
            where = ", ".join(f"{s.start - t0:.1f}-{s.end - t0:.1f}s ({s.reason})" for s in self.spans)
            parts.append(
                f"{len(self.spans)} fabricated stretch(es) dropped -- {where}; "
                f"{self.n_poses_dropped}/{self.n_poses} poses and "
                f"{self.n_frames_dropped}/{self.n_frames} frames removed"
            )
        if self.n_frames_unsupported:
            parts.append(f"{self.n_frames_unsupported}/{self.n_frames} frames dropped for having no pose under them")
        return "; ".join(parts) if parts else "no fabricated poses"


# ---------------------------------------------------------------------------
# The four signatures
# ---------------------------------------------------------------------------
def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Maximal half-open index ranges where ``flags`` is true."""
    edges = np.flatnonzero(np.diff(np.r_[0, flags.astype(np.int8), 0]))
    return list(zip(edges[0::2], edges[1::2], strict=True))


def teleport_edges(ts: np.ndarray, pos: np.ndarray):
    """Indices ``i`` whose step to ``i + 1`` exceeds ``MAX_SPEED`` (holes over ``MAX_GAP`` excluded)."""
    if len(ts) < 2:
        return np.empty(0, dtype=int)
    dt = np.diff(ts)
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    return np.flatnonzero((dt > 0) & (dt <= MAX_GAP) & (step > MAX_SPEED * dt))


def frozen_spans(ts: np.ndarray, pos: np.ndarray) -> list[Span]:
    """Stretches where the translation does not move at all for long enough."""
    if len(ts) < 2:
        return []
    still = np.linalg.norm(np.diff(pos, axis=0), axis=1) <= FREEZE_STEP
    out = []
    for a, b in _runs(still):
        # still[k] is the step k -> k+1, so samples a..b inclusive hold the position.
        if ts[b] - ts[a] >= MIN_FREEZE:
            out.append(Span(float(ts[a]), float(ts[b]), "frozen"))
    return out


def island_spans(ts: np.ndarray, edges: np.ndarray) -> list[Span]:
    """Short stretches fenced in by a teleport at both ends."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        a, b = int(lo) + 1, int(hi)  # samples strictly between the two jumps
        if a <= b and ts[b] - ts[a] < MIN_ISLAND:
            out.append(Span(float(ts[a]), float(ts[b]), "island"))
    return out


def outage_spans(fix_ts) -> list[Span]:
    """Holes between usable fix stamps (null-island fixes are already dropped).

    Late starts and early ends don't count: that's an ordinary recording.
    """
    t = np.sort(np.asarray(fix_ts, dtype=float))
    if len(t) < 2:
        return []
    return [Span(float(t[i]), float(t[i + 1]), "no_fix") for i in np.flatnonzero(np.diff(t) > MIN_OUTAGE)]


# ---------------------------------------------------------------------------
# Putting them together
# ---------------------------------------------------------------------------
def _merge(spans: list[Span]) -> list[Span]:
    """Union of overlapping spans, keeping every reason that contributed."""
    out: list[Span] = []
    for s in sorted(spans, key=lambda s: s.start):
        if out and s.start <= out[-1].end:
            last = out[-1]
            reasons = sorted({*last.reason.split("+"), *s.reason.split("+")})
            out[-1] = Span(last.start, max(last.end, s.end), "+".join(reasons))
        else:
            out.append(Span(s.start, s.end, s.reason))
    return out


def _outages_with_a_jump(ts: np.ndarray, edges: np.ndarray, outages: list[Span]) -> list[Span]:
    """Outages during which the position also teleported (either alone is inconclusive)."""
    out = []
    for o in outages:
        jumped = any(o.start <= ts[int(i)] <= o.end or o.start <= ts[int(i) + 1] <= o.end for i in edges)
        if jumped:
            out.append(Span(o.start, o.end, "no_fix+teleport"))
    return out


def _corroborated(span: Span, ts: np.ndarray, edges: np.ndarray, outages: list[Span], fix_ts=None) -> bool:
    """Does an outage, a missing fix or a teleport edge confirm this frozen run?

    A stuck robot also publishes a constant position, so freezing alone is not enough.
    """
    if any(span.start <= o.end and o.start <= span.end for o in outages):
        return True
    # No usable fix under the run at all (outages miss runs before/after all fixes).
    if fix_ts is not None and not np.any((fix_ts >= span.start) & (fix_ts <= span.end)):
        return True
    inside = np.flatnonzero((ts >= span.start) & (ts <= span.end))
    if not len(inside):
        return False
    first, last = int(inside[0]), int(inside[-1])
    return bool((first - 1) in edges or last in edges)


def find_spans(ts, poses, fix_ts=None) -> list[Span]:
    """Every fabricated stretch of a sorted pose stream; ``fix_ts`` may be None (no fix topic).

    Fails towards keeping data.
    """
    ts = np.asarray(ts, dtype=float)
    if len(ts) < 2:
        return []
    pos = np.asarray(poses, dtype=float)[:, :3, 3]
    edges = teleport_edges(ts, pos)
    outages = [] if fix_ts is None else outage_spans(fix_ts)
    fixes = None if fix_ts is None else np.asarray(fix_ts, dtype=float)

    frozen = [s for s in frozen_spans(ts, pos) if _corroborated(s, ts, edges, outages, fixes)]
    spans = _merge(frozen + island_spans(ts, edges) + _outages_with_a_jump(ts, edges, outages))

    # Widen spans to outages they reach into; a lone outage is just coasting.
    for outage in outages:
        if any(s.start <= outage.end and outage.start <= s.end for s in spans):
            spans = _merge([*spans, outage])
    return spans


def drop_poses(odom_data, spans: list[Span]):
    """``odom_data`` with every sample inside a span removed."""
    return [row for row in odom_data if not any(row[0] in s for s in spans)]


def frame_mask(frame_ts: np.ndarray, spans: list[Span]) -> np.ndarray:
    """True for the frames that keep a trustworthy pose."""
    keep = np.ones(len(frame_ts), dtype=bool)
    for s in spans:
        keep &= ~((frame_ts >= s.start - FRAME_PAD) & (frame_ts <= s.end + FRAME_PAD))
    return keep


def frame_support(frame_ts, odom_ts, pad: float = FRAME_PAD) -> np.ndarray:
    """True for frames real pose samples bracket closely enough to interpolate.

    False before/after the stream or inside a hole wider than ``SUPPORT_GAP``.
    ``pad`` widens each frame stamp in favour of keeping it; ``pad=0`` once the
    clock is corrected.
    """
    t = np.asarray(frame_ts, dtype=float)
    o = np.sort(np.asarray(odom_ts, dtype=float))
    if not len(o):
        return np.zeros(len(t), dtype=bool)
    lo, hi = t - pad, t + pad
    keep = (hi >= o[0]) & (lo <= o[-1])
    for i in np.flatnonzero(np.diff(o) > SUPPORT_GAP):
        keep &= ~((lo > o[i]) & (hi < o[i + 1]))
    return keep


def guard(odom_data, frame_ts: np.ndarray, fix_ts=None) -> tuple[list, np.ndarray, Verdict]:
    """Drop the fabricated poses, and the frames they leave unsupported.

    Returns the surviving odometry, the keep-mask over ``frame_ts``, and what
    it cost. The odometry must be sorted by time.
    """
    verdict = Verdict(n_poses=len(odom_data), n_frames=len(frame_ts))
    if not odom_data:
        return list(odom_data), np.ones(len(frame_ts), dtype=bool), verdict
    ts = np.array([row[0] for row in odom_data], dtype=float)
    poses = np.array([row[1] for row in odom_data], dtype=float)
    frames = np.asarray(frame_ts, dtype=float)

    verdict.spans = find_spans(ts, poses, fix_ts)
    kept = drop_poses(odom_data, verdict.spans)
    mask = frame_mask(frames, verdict.spans)
    verdict.n_poses_dropped = len(odom_data) - len(kept)
    verdict.n_frames_dropped = int(np.count_nonzero(~mask))

    # Measured against the survivors, so span holes are covered too.
    support = _support(verdict, frames, [row[0] for row in kept], mask)
    return kept, mask & support, verdict


def _support(verdict: Verdict, frames: np.ndarray, kept_ts: list, mask: np.ndarray) -> np.ndarray:
    """:func:`frame_support` over ``kept_ts``, counting the frames it costs into ``verdict``."""
    support = frame_support(frames, kept_ts)
    lost = mask & ~support
    verdict.n_frames_unsupported = int(np.count_nonzero(lost))
    if kept_ts:
        verdict.n_unsupported_before = int(np.count_nonzero(lost & (frames < kept_ts[0])))
        verdict.n_unsupported_after = int(np.count_nonzero(lost & (frames > kept_ts[-1])))
    else:
        verdict.n_unsupported_before = verdict.n_frames_unsupported
    verdict.n_unsupported_inside = (
        verdict.n_frames_unsupported - verdict.n_unsupported_before - verdict.n_unsupported_after
    )
    return support


def coverage_guard(odom_data, frame_ts: np.ndarray) -> tuple[list, np.ndarray, Verdict]:
    """The support half of :func:`guard` alone, for a source with no reset solution (Odin1)."""
    verdict = Verdict(n_poses=len(odom_data), n_frames=len(frame_ts))
    frames = np.asarray(frame_ts, dtype=float)
    if not odom_data:
        return list(odom_data), np.ones(len(frames), dtype=bool), verdict
    support = _support(verdict, frames, [row[0] for row in odom_data], np.ones(len(frames), dtype=bool))
    return list(odom_data), support, verdict


def write_report(path, verdict: Verdict, t0: float) -> None:
    """Write one line per span (start/end relative to the first frame, reason)."""
    lines = [
        "# stretches of fabricated GNSS/INS pose removed from this scene",
        "# start_s end_s reason   (seconds into the scene)",
        f"# {verdict.n_poses_dropped}/{verdict.n_poses} poses, "
        f"{verdict.n_frames_dropped}/{verdict.n_frames} frames removed",
        f"# {verdict.n_frames_unsupported}/{verdict.n_frames} further frames had no pose under them "
        f"({verdict.n_unsupported_before} before the pose stream, {verdict.n_unsupported_after} after it, "
        f"{verdict.n_unsupported_inside} inside holes)",
        f"# after the clock correction, {verdict.n_frames_trimmed}/{verdict.n_frames} more frames "
        "had no pose bracketing them, dropped as well",
    ]
    lines += [f"{s.start - t0:.3f} {s.end - t0:.3f} {s.reason}" for s in verdict.spans]
    Path(path).write_text("\n".join(lines) + "\n")
