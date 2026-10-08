"""Splitting a scene into bounded, self-consistent viewing windows.

A window never spans two segments (segments share no frame). Long segments are
cut by *seconds* (sampling density is a per-second cost), with a frame cap as a
backstop. ``pose_source`` says what the poses are: ``odometry`` (ground truth),
``transforms`` (drifty wheel odometry) or ``none``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .loaders import Segment

#: Seconds per window: the cap that normally decides.
DEFAULT_MAX_SECONDS = 600.0
#: Frames per window: a backstop that bites only above ~20 Hz.
DEFAULT_MAX_FRAMES = 12000
#: A window with fewer frames than this is not worth building.
MIN_WINDOW_FRAMES = 2


@dataclass
class Window:
    """One buildable view of a scene: a contiguous frame range plus its pose source."""

    key: str
    kind: str  # "segment" | "raw" | "frame"
    label: str
    frame_start: int  # inclusive
    frame_end: int  # exclusive
    pose_source: str  # "odometry" | "transforms" | "none"
    segment_index: int | None = None
    chunk: int = 0
    n_chunks: int = 1
    reason: str = ""
    t_start: float = 0.0
    t_end: float = 0.0
    #: Ground-truth poses behind this window (0 without a segment). Pose rates vary
    #: wildly and look identical once interpolated, so the count travels along.
    n_poses: int = 0

    @property
    def n_frames(self) -> int:
        return self.frame_end - self.frame_start

    @property
    def duration(self) -> float:
        return max(0.0, self.t_end - self.t_start)

    @property
    def pose_rate(self) -> float | None:
        """Ground-truth poses per second, or None when there is no segment."""
        if not self.n_poses or self.duration <= 0:
            return None
        return self.n_poses / self.duration

    @property
    def note(self) -> str:
        """Concrete, plain-language description shown in the windows table."""
        if self.kind == "segment":
            d = describe_reason(self.reason)
            if d and not d.startswith("filled"):
                return f"ends because {d}"
            return d
        return self.reason

    def to_json(self) -> dict:
        d = asdict(self)
        d["n_frames"] = self.n_frames
        d["duration"] = self.duration
        return d


def _chunk_edges(start: int, end: int, n: int, frame_ts: np.ndarray | None) -> list[int]:
    """The ``n + 1`` frame indices that cut ``[start, end)`` into ``n`` chunks.

    Cut at equal times when there is a clock, else equal frame counts. Every
    chunk keeps at least one frame, even across a hole in the clock.
    """
    if n <= 1:
        return [start, end]
    if frame_ts is None:
        return [int(x) for x in np.linspace(start, end, n + 1).round()]
    cuts = np.linspace(float(frame_ts[start]), float(frame_ts[end - 1]), n + 1)[1:-1]
    edges = [start]
    for i, cut in enumerate(np.searchsorted(frame_ts, cuts, side="left")):
        edges.append(int(min(max(int(cut), edges[-1] + 1), end - (n - 1 - i))))
    edges.append(end)
    return edges


def _chunk_ranges(
    start: int,
    end: int,
    max_frames: int,
    frame_ts: np.ndarray | None = None,
    max_seconds: float = DEFAULT_MAX_SECONDS,
) -> list[tuple[int, int]]:
    """Split ``[start, end)`` into consecutive chunks the caps allow.

    Both caps are honoured (the seconds cap needs ``frame_ts``). The count is
    chosen first and the range divided evenly, so 10% over the cap gives two
    halves, not a window and a stub; the count grows until no chunk exceeds the frame cap.
    """
    total = end - start
    if total <= 0:
        return [(start, end)]
    have_clock = frame_ts is not None and max_seconds > 0 and end - 1 < len(frame_ts)
    n = -(-total // max_frames) if max_frames > 0 else 1  # ceil
    if have_clock:
        span = float(frame_ts[end - 1]) - float(frame_ts[start])
        n = max(n, int(math.ceil(span / max_seconds)) if span > 0 else 1)
    n = min(max(n, 1), total)  # never more chunks than frames
    while True:
        edges = _chunk_edges(start, end, n, frame_ts if have_clock else None)
        widest = max(b - a for a, b in zip(edges, edges[1:], strict=False))
        if max_frames <= 0 or widest <= max_frames or n >= total:
            break
        n += 1
    return [(edges[i], edges[i + 1]) for i in range(n)]


def _playable_frames(frame_ts: np.ndarray | None, n_frames: int) -> int:
    """How many leading frames can actually be played back.

    A frame past the end of the timestamp track has no place on the clock.
    """
    if frame_ts is None:
        return n_frames
    return min(n_frames, len(frame_ts))


def frame_slack(frame_ts: np.ndarray | None) -> float:
    """Half a frame period: how far off a segment edge a frame may still fall.

    The pose and frame clocks are unrelated, so a frame a few ms before a
    segment's first pose still belongs to it.
    """
    if frame_ts is None or len(frame_ts) < 2:
        return 0.0
    return 0.5 * float(np.median(np.diff(frame_ts)))


def segment_bounds(frame_ts: np.ndarray | None, segments: list[Segment]) -> list[tuple[float, float]]:
    """Each segment's time span widened to nearest-frame edges.

    Widened by :func:`frame_slack`, never past the midpoint to a neighbour, so
    no frame is claimed by two segments.
    """
    slack = frame_slack(frame_ts)
    bounds: list[tuple[float, float]] = []
    for i, seg in enumerate(segments):
        lo = slack if i == 0 else min(slack, max(0.0, (seg.start_ts - segments[i - 1].end_ts) / 2.0))
        hi = slack if i == len(segments) - 1 else min(slack, max(0.0, (segments[i + 1].start_ts - seg.end_ts) / 2.0))
        bounds.append((seg.start_ts - lo, seg.end_ts + hi))
    return bounds


def frames_within(frame_ts: np.ndarray, lo_ts: float, hi_ts: float) -> tuple[int, int]:
    """The ``[start, end)`` frames whose stamps fall inside ``[lo_ts, hi_ts]``."""
    return int(np.searchsorted(frame_ts, lo_ts, side="left")), int(np.searchsorted(frame_ts, hi_ts, side="right"))


def windows_for_scene(
    frame_ts: np.ndarray | None,
    segments: list[Segment],
    n_frames: int,
    has_transforms: bool,
    max_frames: int = DEFAULT_MAX_FRAMES,
    max_seconds: float = DEFAULT_MAX_SECONDS,
) -> list[Window]:
    """Enumerate every window a scene offers, ground truth first.

    Without ``frame_ts`` (``data_ts.txt``) only the whole-scene fallback is
    produced, cut on the frame cap alone.
    """
    windows: list[Window] = []

    if frame_ts is not None and len(frame_ts) and segments:
        for seg, bounds in zip(segments, segment_bounds(frame_ts, segments), strict=False):
            lo, hi = frames_within(frame_ts, *bounds)
            if hi - lo < MIN_WINDOW_FRAMES:
                continue
            windows += _chunked(
                lo,
                hi,
                max_frames,
                frame_ts,
                max_seconds,
                key=f"seg{seg.index:03d}",
                kind="segment",
                label=f"Segment {seg.index}",
                pose_source="odometry",
                segment_index=seg.index,
                reason=seg.reason,
                n_poses=seg.n_poses,
            )

    # Whole-scene wheel-odometry fallback, only when there is no ground-truth
    # window (it is not GT). Bounded by the timestamp track, not ``n_frames``.
    playable = _playable_frames(frame_ts, n_frames)
    if not windows and playable >= MIN_WINDOW_FRAMES:
        windows += _chunked(
            0,
            playable,
            max_frames,
            frame_ts,
            max_seconds,
            key="raw",
            kind="raw",
            label="Whole scene, wheel odometry" if has_transforms else "Whole scene, no pose",
            pose_source="transforms" if has_transforms else "none",
            reason="wheel odometry, drifts; not ground truth" if has_transforms else "no pose data in this scene",
        )
    return windows


def _chunked(
    start: int,
    end: int,
    max_frames: int,
    frame_ts: np.ndarray | None,
    max_seconds: float,
    key: str,
    label: str,
    n_poses: int = 0,
    **fields: Any,
) -> list[Window]:
    """One window per chunk of ``[start, end)`` (:func:`_chunk_ranges`), keyed and labelled in turn.

    A lone chunk keeps the bare ``key`` and ``label``; several get ``.i`` and
    ``(i/n)``. ``n_poses`` (per segment) is shared evenly between chunks.
    """
    chunks = _chunk_ranges(start, end, max_frames, frame_ts, max_seconds)
    n = len(chunks)
    return [
        Window(
            key=key + (f".{ci}" if n > 1 else ""),
            label=label + (f" ({ci + 1}/{n})" if n > 1 else ""),
            frame_start=cs,
            frame_end=ce,
            chunk=ci,
            n_chunks=n,
            n_poses=round(n_poses / n),
            t_start=float(frame_ts[cs]) if frame_ts is not None and len(frame_ts) > cs else 0.0,
            t_end=float(frame_ts[ce - 1]) if frame_ts is not None and len(frame_ts) >= ce else 0.0,
            **fields,
        )
        for ci, (cs, ce) in enumerate(chunks)
    ]


def neighbour_windows(windows: list[Window], current: Window) -> tuple[Window | None, Window | None]:
    """The windows immediately before and after ``current``, in scene order.

    Ordered by frame range, so a one-frame window (never enumerated) has
    neighbours too. Either end is ``None`` where the scene runs out.
    """

    def span(w: Window) -> tuple[int, int]:
        return (w.frame_start, w.frame_end)

    ordered = sorted(windows, key=span)
    before = [w for w in ordered if span(w) < span(current)]
    after = [w for w in ordered if span(w) > span(current)]
    return (before[-1] if before else None, after[0] if after else None)


#: Plain-language text for the machine tags in ``odometry_segments.txt``.
#: A tag names why the segment ENDED (except ``v11_fill``, which names where
#: the whole segment came from). Unknown tags pass through verbatim.
_REASON_TEXT = {
    "pose_jump": "the GT pose jumped against the robot's live odometry",
    "speed_mismatch": "the GT speed disagreed with the robot's live odometry",
    "yawrate_mismatch": "the GT yaw rate disagreed with the robot's live odometry",
    "odom_mismatch": "the post-hoc check against live odometry rejected the poses that follow",
    "end_of_run": "the recording ended",
    "gap": "a gap of over 1 s in the GT stream",
    "wheel_gap": "the wheels drove further across a gap in the GT stream than the GT pose moved",
    "teleport": "a step faster than the robot can move (GT jump)",
    "fix_lost": "the GNSS fix was lost and the poses the receiver fabricated without it were removed",
}


#: Where a segment's poses came from when not this root's own run. Each must
#: start with "filled": the pages branch on it.
_FILL_TEXT = {
    "v11_fill": "filled from v1.1 ground truth, through the same quality gate",
    "regen_fill": "filled by replaying that stretch of the bag through the estimator, through the same quality gate",
}

#: ``+`` tags: what was done to the poses after the stretch was cut.
_MARK_TEXT = {
    "refined": "refined against the wheel odometry",
    "coverage_guard": "spliced from the earlier root, which covered this stretch",
    "joined": "joined across a standstill",
}

#: Where the pieces of a joined segment came from (``+joined:native,v11_fill``).
_SOURCE_TEXT = {
    "native": "this root's own estimator run",
    "v11_fill": "the v1.1 fill",
    "regen_fill": "a replay of the bag",
}


def describe_reason(reason: str) -> str:
    """A concrete sentence fragment for a segment-reason tag.

    E.g. ``regen_fill:t194+refined``: the body says where the segment came
    from or why it ended; ``+`` tails are appended in parentheses.
    """
    if not reason:
        return ""
    described = _describe_body(reason.split("+")[0])
    extra = "; ".join(describe_marks(reason))
    return f"{described} ({extra})" if described and extra else (described or reason)


def describe_marks(reason: str, brief: bool = False) -> list[str]:
    """The ``+`` tails of a reason, translated; ``[]`` when it carries none.

    ``brief`` drops per-segment detail so a page can count them across a scene.
    """
    return [_mark_text(m, brief) for m in reason.split("+")[1:] if m]


def _mark_text(mark: str, brief: bool = False) -> str:
    """One ``+`` tail, translated. ``joined`` also carries the sources it joined."""
    if mark.startswith("joined"):
        if brief:
            return _MARK_TEXT["joined"]
        text = f"{_MARK_TEXT['joined']} the wheels, the live odometry and the IMU all confirm"
        if ":" not in mark:
            return text
        pieces = [_SOURCE_TEXT.get(x, x) for x in mark.split(":", 1)[1].split(",") if x]
        return f"{text}, out of pieces from {' and '.join(pieces)}"
    return _MARK_TEXT.get(mark, mark)


def _describe_body(body: str) -> str:
    for prefix, text in _FILL_TEXT.items():
        if body.startswith(prefix):
            rest = body.split(":", 1)[1] if ":" in body else ""
            # A fill may also carry the reason it ends.
            tail = _REASON_TEXT.get(rest.split(":")[-1], "")
            return f"{text}; ends because {tail}" if tail else text
    return _REASON_TEXT.get(body.split(":")[-1], body)


#: Key prefix of the one-frame window (``frame00042``), built on demand from the key.
FRAME_KEY_PREFIX = "frame"


def frame_window_key(index: int) -> str:
    return f"{FRAME_KEY_PREFIX}{index:05d}"


def parse_frame_window_key(key: str) -> int | None:
    """The frame index a ``frame00042``-style key names, or ``None``."""
    if not key.startswith(FRAME_KEY_PREFIX):
        return None
    tail = key[len(FRAME_KEY_PREFIX) :]
    # ASCII only: "⑤".isdigit() is true and int() refuses it; the length cap keeps int() cheap.
    return int(tail) if tail.isascii() and tail.isdigit() and len(tail) <= 12 else None


def segment_of_frame(frame_ts: np.ndarray | None, segments: list[Segment], index: int) -> Segment | None:
    """The ground-truth segment a frame falls inside, if any.

    Uses the widened :func:`segment_bounds`.
    """
    if frame_ts is None or index < 0 or index >= len(frame_ts):
        return None
    t = float(frame_ts[index])
    for seg, (lo, hi) in zip(segments, segment_bounds(frame_ts, segments), strict=False):
        if lo <= t <= hi:
            return seg
    return None


def single_frame_window(
    frame_ts: np.ndarray | None,
    segments: list[Segment],
    n_frames: int,
    index: int,
) -> Window | None:
    """A window holding exactly one frame, for static inspection of that frame.

    Inside a segment the robot stands at its GT pose; elsewhere the view is
    robot-centric. ``pose_source`` says which.
    """
    if index < 0 or index >= _playable_frames(frame_ts, n_frames):
        # No clock to place the frame on; the caller returns 404.
        return None
    seg = segment_of_frame(frame_ts, segments, index)
    t = float(frame_ts[index]) if frame_ts is not None and index < len(frame_ts) else 0.0
    return Window(
        key=frame_window_key(index),
        kind="frame",
        label=f"Frame {index}",
        frame_start=index,
        frame_end=index + 1,
        pose_source="odometry" if seg is not None else "none",
        segment_index=seg.index if seg is not None else None,
        reason=(
            f"single frame, inside segment {seg.index}"
            if seg is not None
            else "single frame, outside every ground-truth segment: robot-centric"
        ),
        n_poses=seg.n_poses if seg is not None else 0,
        t_start=t,
        t_end=t,
    )
