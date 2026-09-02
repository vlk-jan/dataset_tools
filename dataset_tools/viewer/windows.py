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

    # Whole-scene fallback: continuous across the entire recording, but drifty.
    # Always offered when it exists -- it is the only view that shows the parts
    # of a scene no ground-truth segment covers.
    if n_frames >= MIN_WINDOW_FRAMES:
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


def frames_covered_by_segments(frame_ts: np.ndarray, segments: list[Segment]) -> np.ndarray:
    """Boolean mask of frames that fall inside some ground-truth segment."""
    covered = np.zeros(len(frame_ts), dtype=bool)
    for seg in segments:
        lo = np.searchsorted(frame_ts, seg.start_ts, side="left")
        hi = np.searchsorted(frame_ts, seg.end_ts, side="right")
        covered[lo:hi] = True
    return covered
