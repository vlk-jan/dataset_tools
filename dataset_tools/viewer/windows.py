"""Splitting a scene into bounded, self-consistent viewing windows.

Two hard constraints shape this module:

1. **Segments do not share a frame.** ``odometry_segments.txt`` marks stretches
   of ground truth that were each re-anchored to identity at their own first
   pose, so a trajectory, an aggregated cloud or a splatted terrain may never
   cross a segment boundary. One window therefore never spans two segments.
2. **Scenes are long.** Up to ~20k frames, and one segment of HELHEST v1.3 runs
   half an hour. Long segments are cut into consecutive chunks the UI can page
   through.

   The cut is measured in *seconds*, not frames. A build already reads a
   bounded number of per-frame files whatever the window's length
   (``BuildOptions.max_hm_frames`` and ``max_cloud_frames`` pick a fixed subset
   of it), and the bytes a window ships barely move with it: measured on that
   half-hour segment, twelve times the frames cost 1.9x the bundle and no extra
   build time. What a longer window really costs is sampling density -- the
   same number of height maps and clouds spread over more ground -- which is a
   cost per second driven, so seconds are what the cap counts.

   Counting frames made the cap mean two different things in one root: v1.3
   holds scenes recorded at 3.2 Hz and at 10 Hz, so a 1500-frame cap left a
   462 s window whole in one scene while halving a 180 s segment in the next.
   The frame ceiling survives underneath as a backstop.

A window also records where its poses come from, which changes what the view
means: ``odometry`` is ground truth, ``transforms`` the drifty wheel-odometry
fallback, ``none`` no pose at all (per-frame data only).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .loaders import Segment

#: Seconds per window: the cap that actually bites. Long enough that across
#: HELHEST v1.3 exactly one segment (0019/0, 29 min) is cut at all.
DEFAULT_MAX_SECONDS = 600.0
#: Frames per window, a backstop under the seconds cap rather than the rule: it
#: bites only on a root recorded above ~20 Hz, where a window of
#: ``DEFAULT_MAX_SECONDS`` would carry more frames than a trajectory should.
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
    #: Ground-truth poses the underlying segment carries, 0 when there is no
    #: segment behind this window. A segment's pose rate varies by more than an
    #: order of magnitude across one root -- some are anchored at ~10 Hz, others
    #: at well under 1 Hz -- and the two look identical once interpolated onto
    #: the frame clock, so the count has to travel with the window.
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

    Cut at equal *times* when there is a clock to read, so the pieces are the
    equal parts of a stretch of driving rather than equal counts of frames --
    the cap is in seconds, and the division should be in the same unit. Only a
    non-uniform frame clock tells the two apart, but that is exactly when it
    matters: frames dropped in the first half would otherwise push the cut
    late and hand the second window more seconds than the cap allows.

    Each edge is then held at least one frame past the one before it and far
    enough from the end to leave every later chunk a frame of its own, so a
    time slice that happens to contain no frame at all -- a hole in the clock
    wider than the slice -- yields a thin window rather than an empty one.
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

    Both caps are honoured -- whichever asks for more pieces wins -- and the
    seconds cap is the one that normally does. It needs ``frame_ts`` to know
    how long the range is; without timestamps only the frame ceiling applies.

    The number of chunks is chosen first, then the range is divided into that
    many equal parts (see :func:`_chunk_edges`), so a segment 10% over the cap
    becomes two even halves rather than one full window and a stub. Dividing by
    time can leave a chunk over the frame ceiling where the clock is uneven --
    the same seconds hold more frames in one part than another -- so the count
    is raised until neither cap is exceeded.
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

    ``n_frames`` is the largest of the scene's several per-frame counts, so it
    can exceed the number of timestamps (a directory built in flight, a
    truncated ``data_ts.txt``). A frame past the end of the timestamp track has
    no place on the clock, so no window may contain one.
    """
    if frame_ts is None:
        return n_frames
    return min(n_frames, len(frame_ts))


def frame_slack(frame_ts: np.ndarray | None) -> float:
    """Half a frame period: how far off a segment edge a frame may still fall.

    Segment edges come from the ground-truth pose clock, frames from their own,
    and the two are unrelated. A frame a few ms before a segment's first pose is
    that segment's first frame; without the slack it belongs to nothing, and the
    frame page reports no ground truth for it.
    """
    if frame_ts is None or len(frame_ts) < 2:
        return 0.0
    return 0.5 * float(np.median(np.diff(frame_ts)))


def segment_bounds(frame_ts: np.ndarray | None, segments: list[Segment]) -> list[tuple[float, float]]:
    """Each segment's time span widened to nearest-frame edges.

    Widened by :func:`frame_slack`, but never past the midpoint of the gap to a
    neighbour, so no frame is ever claimed by two segments -- segments share no
    reference frame, so a frame in both would be in neither.
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

    ``frame_ts`` is the per-frame timestamp array (``data_ts.txt``). Without it
    frames cannot be matched to segments at all, so only the whole-scene
    fallback windows are produced -- and, having no clock, they are cut on the
    frame ceiling alone.
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

    # Whole-scene wheel-odometry fallback: drifty and NOT ground truth, so it is
    # offered only when the scene yields no ground-truth window at all, where it
    # is the only motion view there is. Alongside ground truth it would invite
    # reading a wrong trajectory (user decision 2026-09-02).
    #
    # Bounded by the timestamp track, not ``n_frames``: the frame count is the
    # largest of several disagreeing counts, and a frame with no timestamp has
    # no place on the playback clock -- offering it built a window that read
    # zero frames and died with an IndexError.
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
    ``(i/n)``. ``n_poses`` is the whole stretch's: the segment file counts poses
    per segment, not per chunk, and the split is even, so each chunk gets an
    even share -- the honest estimate, and the rate it gives is unchanged.
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

    Ordered by frame range rather than by position in ``windows``, so a
    one-frame window -- addressable but never enumerated, see
    ``single_frame_window`` -- has neighbours at all: the previous window is
    the last one starting before it, which for a frame inside a segment is
    that segment, so paging back from a single frame opens the stretch it
    belongs to. Either end is ``None`` where the scene runs out.
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


#: Where a segment's poses came from, when they did not come out of this root's
#: own odometry run. Every one of these reads as a sentence starting "filled",
#: which is what the pages branch on to say "filled ..." instead of "ends
#: because ...".
_FILL_TEXT = {
    "v11_fill": "filled from v1.1 ground truth, through the same quality gate",
    "regen_fill": "filled by replaying that stretch of the bag through the estimator, through the same quality gate",
}

#: Tags a reason accumulates after a ``+``: what was done to the poses after
#: the stretch was cut, rather than why it was cut.
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

    Tags accumulate: a stretch can be filled from elsewhere, then refined, and
    is written ``regen_fill:t194+refined``. The part before the first ``+`` says
    where the segment came from or why it ended; each ``+`` tail says what was
    then done to its poses, and those are appended in parentheses rather than
    left as raw tags -- a reader of the ground-truth panel should never be shown
    a tag the tooling invented for itself.
    """
    if not reason:
        return ""
    described = _describe_body(reason.split("+")[0])
    extra = "; ".join(describe_marks(reason))
    return f"{described} ({extra})" if described and extra else (described or reason)


def describe_marks(reason: str, brief: bool = False) -> list[str]:
    """The ``+`` tails of a reason, translated; ``[]`` when it carries none.

    Separate from :func:`describe_reason` so a page can count them across a
    scene ("10 of them refined ...") instead of repeating a sentence that is
    identical but for its tail. ``brief`` drops the detail that varies from one
    segment to the next, so those counts group instead of splintering.
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
            # A filled stretch may carry the reason it *ends* as well; where it
            # carries something else (the hole it filled), there is nothing to
            # translate and the fill sentence stands alone.
            tail = _REASON_TEXT.get(rest.split(":")[-1], "")
            return f"{text}; ends because {tail}" if tail else text
    return _REASON_TEXT.get(body.split(":")[-1], body)


#: Key prefix for the synthetic one-frame window (``frame00042``). These are
#: never enumerated -- a 20k-frame scene would mean 20k windows -- they are
#: built on demand from the key, which is why the prefix has to round-trip.
FRAME_KEY_PREFIX = "frame"


def frame_window_key(index: int) -> str:
    return f"{FRAME_KEY_PREFIX}{index:05d}"


def parse_frame_window_key(key: str) -> int | None:
    """The frame index a ``frame00042``-style key names, or ``None``."""
    if not key.startswith(FRAME_KEY_PREFIX):
        return None
    tail = key[len(FRAME_KEY_PREFIX) :]
    return int(tail) if tail.isdigit() else None


def segment_of_frame(frame_ts: np.ndarray | None, segments: list[Segment], index: int) -> Segment | None:
    """The ground-truth segment a frame falls inside, if any.

    Most frames of most scenes are *not* inside one: divergence splits the run,
    and the gaps between segments have no trustworthy pose at all.
    """
    if frame_ts is None or index < 0 or index >= len(frame_ts):
        return None
    t = float(frame_ts[index])
    # On the widened bounds (see ``segment_bounds``), so a frame just off an
    # edge is still that segment's. They contain every raw span and never reach
    # into a later segment's, so a frame inside one segment is that segment's.
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

    There is no motion to show, so the only thing the pose decides is *where*
    the robot stands: inside a ground-truth segment it stands at its real pose
    in that segment's frame, and everywhere else -- most of most scenes -- the
    view is robot-centric, with the robot at the origin and the frame's own
    cloud and height map around it. Both are useful; only one is ground truth,
    and ``pose_source`` says which you are looking at.
    """
    if index < 0 or index >= _playable_frames(frame_ts, n_frames):
        # Past the timestamp track there is no clock to place the frame on, so
        # there is nothing to build -- the caller turns this into a 404 rather
        # than letting the build fail on an empty frame range.
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
