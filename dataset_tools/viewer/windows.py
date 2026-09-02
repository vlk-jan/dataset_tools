"""Splitting a scene into bounded, self-consistent viewing windows.

Two hard constraints shape this module:

1. **Segments do not share a frame.** ``odometry_segments.txt`` marks stretches
   of ground truth that were each re-anchored to identity at their own first
   pose, so a trajectory, an aggregated cloud or a splatted terrain may never
   cross a segment boundary. One window therefore never spans two segments.
2. **Scenes are long.** Up to ~20k frames, on contended network storage. A
   window is capped at a few thousand frames so a build reads a bounded number
   of files; longer segments are cut into consecutive chunks the UI can page
   through.

A window also records *where its poses come from*, because that changes what
the view means: ``odometry`` is ground truth, ``transforms`` is the drifty
wheel-odometry fallback, and ``none`` means the scene has no pose at all and
only per-frame data is shown.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .loaders import Segment

#: Frames per window. Bounds the number of per-frame files one build touches.
DEFAULT_MAX_FRAMES = 1500
#: A window with fewer frames than this is not worth building.
MIN_WINDOW_FRAMES = 2


@dataclass
class Window:
    """One buildable view of a scene: a contiguous frame range plus its pose source."""

    key: str
    kind: str  # "segment" | "raw" | "frames"
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

    @property
    def n_frames(self) -> int:
        return self.frame_end - self.frame_start

    @property
    def duration(self) -> float:
        return max(0.0, self.t_end - self.t_start)

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


def _chunk_ranges(start: int, end: int, max_frames: int) -> list[tuple[int, int]]:
    """Split ``[start, end)`` into consecutive chunks of at most ``max_frames``.

    The number of chunks is chosen first, then the range is divided evenly, so
    a 1600-frame segment becomes two 800-frame windows rather than 1500 + 100.
    """
    total = end - start
    if total <= max_frames:
        return [(start, end)]
    n = -(-total // max_frames)  # ceil
    edges = np.linspace(start, end, n + 1).round().astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n)]


def windows_for_scene(
    frame_ts: np.ndarray | None,
    segments: list[Segment],
    n_frames: int,
    has_transforms: bool,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> list[Window]:
    """Enumerate every window a scene offers, ground truth first.

    ``frame_ts`` is the per-frame timestamp array (``data_ts.txt``). Without it
    frames cannot be matched to segments at all, so only the whole-scene
    fallback windows are produced.
    """
    windows: list[Window] = []

    if frame_ts is not None and len(frame_ts) and segments:
        for seg in segments:
            lo = int(np.searchsorted(frame_ts, seg.start_ts, side="left"))
            hi = int(np.searchsorted(frame_ts, seg.end_ts, side="right"))
            if hi - lo < MIN_WINDOW_FRAMES:
                continue
            chunks = _chunk_ranges(lo, hi, max_frames)
            for ci, (cs, ce) in enumerate(chunks):
                suffix = f".{ci}" if len(chunks) > 1 else ""
                label = f"Segment {seg.index}"
                if len(chunks) > 1:
                    label += f" (part {ci + 1}/{len(chunks)})"
                windows.append(
                    Window(
                        key=f"seg{seg.index:03d}{suffix}",
                        kind="segment",
                        label=label,
                        frame_start=cs,
                        frame_end=ce,
                        pose_source="odometry",
                        segment_index=seg.index,
                        chunk=ci,
                        n_chunks=len(chunks),
                        reason=seg.reason,
                        t_start=float(frame_ts[cs]),
                        t_end=float(frame_ts[ce - 1]),
                    )
                )

    # Whole-scene wheel-odometry fallback: drifty and NOT ground truth. It is
    # offered only when the scene yields no ground-truth window at all -- there
    # it is the only motion view that exists. Where ground truth exists, a
    # whole-scene wheel-odometry reconstruction would simply be wrong, and
    # showing it invites reading a wrong trajectory (user decision 2026-09-02).
    if not windows and n_frames >= MIN_WINDOW_FRAMES:
        source = "transforms" if has_transforms else "none"
        chunks = _chunk_ranges(0, n_frames, max_frames)
        for ci, (cs, ce) in enumerate(chunks):
            suffix = f".{ci}" if len(chunks) > 1 else ""
            label = (
                "Whole scene, wheel odometry"
                if source == "transforms"
                else "Whole scene, no pose"
            )
            if len(chunks) > 1:
                label += f" (part {ci + 1}/{len(chunks)})"
            windows.append(
                Window(
                    key=f"raw{suffix}",
                    kind="raw",
                    label=label,
                    frame_start=cs,
                    frame_end=ce,
                    pose_source=source,
                    chunk=ci,
                    n_chunks=len(chunks),
                    reason="wheel odometry, drifts; not ground truth"
                    if source == "transforms"
                    else "no pose data in this scene",
                    t_start=float(frame_ts[cs]) if frame_ts is not None and len(frame_ts) > cs else 0.0,
                    t_end=float(frame_ts[ce - 1]) if frame_ts is not None and len(frame_ts) >= ce else 0.0,
                )
            )
    return windows


#: Plain-language text for the machine tags in ``odometry_segments.txt``.
#: A tag names why the segment ENDED (except ``v11_fill``, which names where
#: the whole segment came from). Unknown tags pass through verbatim.
_REASON_TEXT = {
    "pose_jump": "the GT pose jumped against the robot's live odometry",
    "speed_mismatch": "the GT speed disagreed with the robot's live odometry",
    "end_of_run": "the recording ended",
    "start": "the GT stream started",
    "gap": "a gap of over 1 s in the GT stream",
    "teleport": "an implied speed over 4 m/s (GT jump)",
    "fix_lost": "the GNSS fix was lost",
}


def describe_reason(reason: str) -> str:
    """A concrete sentence fragment for a segment-reason tag."""
    if not reason:
        return ""
    if reason.startswith("v11_fill"):
        rest = reason.split(":", 1)[1] if ":" in reason else ""
        tail = _REASON_TEXT.get(rest.split(":")[-1], "")
        out = "filled from v1.1 ground truth (same quality gate)"
        return f"{out}; ends because {tail}" if tail else out
    key = reason.split(":")[-1]
    return _REASON_TEXT.get(key, reason)


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
    tail = key[len(FRAME_KEY_PREFIX):]
    return int(tail) if tail.isdigit() else None


def segment_of_frame(
    frame_ts: np.ndarray | None, segments: list[Segment], index: int
) -> Segment | None:
    """The ground-truth segment a frame falls inside, if any.

    Most frames of most scenes are *not* inside one: divergence splits the run,
    and the gaps between segments have no trustworthy pose at all.
    """
    if frame_ts is None or index < 0 or index >= len(frame_ts):
        return None
    t = float(frame_ts[index])
    for seg in segments:
        if seg.start_ts <= t <= seg.end_ts:
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
    if index < 0 or (n_frames and index >= n_frames):
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
        t_start=t,
        t_end=t,
    )


def frames_covered_by_segments(frame_ts: np.ndarray, segments: list[Segment]) -> np.ndarray:
    """Boolean mask of frames that fall inside some ground-truth segment."""
    covered = np.zeros(len(frame_ts), dtype=bool)
    for seg in segments:
        lo = np.searchsorted(frame_ts, seg.start_ts, side="left")
        hi = np.searchsorted(frame_ts, seg.end_ts, side="right")
        covered[lo:hi] = True
    return covered
