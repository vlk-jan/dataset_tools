"""What a segment page states about one ground-truth segment.

The segment is the unit the ground truth is actually anchored in: each one was
re-anchored to the identity at its own first pose, so it is the largest stretch
whose poses mean anything together. The scene page lists segments and the 3D
view plays one; between the two there was nowhere to *read* about one -- how
long it is, how densely it is anchored, how far the robot drove inside it, why
it ended, where on the Earth it happened -- and that is this page.

The reading budget is the scene page's: the probe, arrays bounded by the frame
count (``odometry.txt``, ``gravity_align.npy``, ``gps.npy``, all cached on the
scene handle and shared with every other page of the same scene), and no
clouds, images or height maps. The frame thumbnails are separate requests the
browser makes for itself.

Every number here is restricted to the segment's own span, and nothing is ever
computed across a boundary: poses either side of one share no reference frame.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .build import SceneAccess
from .index_page import THUMB_FRACTIONS, THUMB_SLOTS, fmt_count, fmt_duration, stat
from .loaders import Segment, roll_pitch_from_gravity_align
from .scene_page import MOVING_M_S, ODOMETRY_READ_LIMIT, map_card, utc_stamp
from .windows import DEFAULT_MAX_SECONDS, Window, describe_reason, segment_bounds

logger = logging.getLogger("dataset_tools.viewer.segment_page")

#: Ground-truth poses per second below which a segment's trajectory is a guess
#: between anchors rather than a measurement. Matches the window table's flag.
THIN_POSE_RATE = 2.0

#: A segment anchored at least this densely is anchored well.
GOOD_POSE_RATE = 5.0


def find(access: SceneAccess, index: int) -> tuple[int, Segment] | None:
    """``(position, segment)`` for the segment *numbered* ``index``, or None.

    Segments are addressed by the number in their own file rather than by their
    position in it: ``scripts/filter_segments.py`` rewrites the table, and the
    window keys (``seg003``) already name the file's number, so a page reached
    from a window has to resolve the same way the window did.
    """
    for position, seg in enumerate(access.segments):
        if seg.index == index:
            return position, seg
    return None


def frame_range(access: SceneAccess, position: int) -> tuple[int, int]:
    """The segment's ``[start, end)`` frames, on the window list's own edges.

    ``windows.segment_bounds`` widens each segment by half a frame period
    without ever letting two segments claim one frame; using it here is what
    keeps this page, the window it links to and the frame page agreeing about
    which frames belong to the segment.
    """
    ts = access.frame_ts
    segments = access.segments
    if ts is None or not len(ts) or position >= len(segments):
        return (0, 0)
    lo_ts, hi_ts = segment_bounds(ts, segments)[position]
    lo = int(np.searchsorted(ts, lo_ts, side="left"))
    hi = int(np.searchsorted(ts, hi_ts, side="right"))
    return (lo, min(hi, len(ts)))


def poses_in(access: SceneAccess, seg: Segment) -> tuple[np.ndarray, np.ndarray] | None:
    """``(ts, poses)`` of the segment's own ground truth, or None.

    Guarded by the same file-size limit the scene page uses: ``odometry.txt``
    is parsed text, and a root whose scenes carry hundreds of thousands of
    poses should not pay for a full parse to render a summary card.
    """
    try:
        if (access.dir / "odometry.txt").stat().st_size > ODOMETRY_READ_LIMIT:
            return None
    except OSError:
        return None
    odom = access.odometry
    if odom is None:
        return None
    ts, poses = odom
    take = (ts >= seg.start_ts) & (ts <= seg.end_ts)
    if take.sum() < 2:
        return None
    return ts[take], poses[take]


def _rate_class(rate: float | None) -> str:
    if rate is None:
        return "none"
    if rate >= GOOD_POSE_RATE:
        return "hi"
    return "mid" if rate >= THIN_POSE_RATE else "lo"


def segment_card(access: SceneAccess, seg: Segment, position: int, frames: tuple[int, int]) -> dict[str, Any]:
    """What the segment *is*: when, how long, how well anchored, why it ended."""
    p = access.probe
    segments = access.segments
    lo, hi = frames
    stats: list[dict[str, Any]] = []
    notes: list[str] = []

    stamp = utc_stamp(seg.start_ts)
    if stamp:
        stats.append(stat("starts", stamp[0], f"{stamp[1]} UTC"))
    stats.append(stat("duration", fmt_duration(seg.duration)))
    rate = seg.n_poses / seg.duration if seg.duration > 0 and seg.n_poses else None
    stats.append(stat("GT poses", fmt_count(seg.n_poses), f"{rate:.1f} Hz" if rate else None))
    stats.append(stat("frames", f"{lo} – {hi - 1}" if hi > lo else "—", f"{hi - lo} of {p.n_frames}"))
    # The *overlap* with the frame clock, not the raw ratio of durations: the
    # ground-truth stream routinely runs a second or two past the last synced
    # frame, and "105% of the scene" is not a thing a segment can be.
    if p.duration and p.start_ts is not None and p.end_ts is not None:
        overlap = max(0.0, min(seg.end_ts, p.end_ts) - max(seg.start_ts, p.start_ts))
        stats.append(stat("of the scene", f"{min(1.0, overlap / p.duration) * 100:.0f}%", "by time"))

    # The silence either side. A segment does not end because the recording
    # did; it ends because the ground truth stopped being trustworthy, and how
    # long it stayed that way is the thing to know before trusting the next one.
    if position > 0:
        stats.append(stat("gap before", f"{max(0.0, seg.start_ts - segments[position - 1].end_ts):.1f} s"))
    if position + 1 < len(segments):
        stats.append(stat("gap after", f"{max(0.0, segments[position + 1].start_ts - seg.end_ts):.1f} s"))

    # Only worth counting when a per-frame directory is known to be incomplete:
    # otherwise every frame in the range has its file and a listing says nothing.
    missing: list[str] = []
    for kind in ("images", "clouds", "height_maps", "height_masks"):
        if not p.has(kind):
            continue
        if p.counts.get(kind, 0) == p.n_frames:
            continue
        have = sum(1 for i in access.available_frames(kind) if lo <= i < hi)
        missing.append(f"{kind} {have}/{hi - lo}")
    if missing:
        stats.append(stat("per-frame files", " · ".join(missing), "inside this segment"))

    reason = describe_reason(seg.reason)
    if reason:
        notes.append(
            f"Ends because {reason}." if not reason.startswith("filled") else f"{reason[0].upper()}{reason[1:]}."
        )
    if rate and rate < THIN_POSE_RATE:
        notes.append(
            f"Anchored at {rate:.1f} Hz: the trajectory between two poses is a straight line over "
            "half a second or more of real driving, so it cuts every corner the robot actually drove."
        )
    notes.append("Poses are anchored to this segment's own first pose, so they mean nothing outside it.")

    return {
        "kind": "segment",
        "title": "Segment",
        "badge": f"{rate:.1f} Hz" if rate else None,
        "badge_class": _rate_class(rate),
        "badge_title": "ground-truth poses per second inside this segment",
        "stats": stats,
        "note": " ".join(notes),
    }


def motion_card(access: SceneAccess, seg: Segment, frames: tuple[int, int]) -> dict[str, Any] | None:
    """What the robot did inside the segment: how far, how fast, how steep."""
    stats: list[dict[str, Any]] = []
    notes: list[str] = []

    track = poses_in(access, seg)
    if track is not None:
        ts, poses = track
        xyz = poses[:, :3, 3]
        steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        dt = np.diff(ts)
        path = float(steps.sum())
        stats.append(stat("path", f"{path:.0f} m", "along the ground truth"))
        with np.errstate(divide="ignore", invalid="ignore"):
            speed = np.where(dt > 0, steps / np.where(dt > 0, dt, 1.0), 0.0)
        moving = speed > MOVING_M_S
        stats.append(stat("moving", f"{moving.mean() * 100:.0f}%", f"over {MOVING_M_S} m/s"))
        if moving.any():
            stats.append(
                stat("speed", f"{float(speed[moving].mean()):.2f} m/s", f"while moving, max {float(speed.max()):.2f}")
            )
        net = float(np.linalg.norm(xyz[-1] - xyz[0]))
        # How much of the driving got the robot somewhere: 100 % is a straight
        # line, and a loop that comes back to its start is near 0 %.
        sub = f"{net / path * 100:.0f}% of the path" if path > 0 else None
        stats.append(stat("net displacement", f"{net:.0f} m", sub))
        climb = float(xyz[:, 2].max() - xyz[:, 2].min())
        stats.append(stat("z range", f"{climb:.1f} m", "as the ground truth has it"))
    else:
        notes.append("No ground-truth poses could be read for this segment.")

    lo, hi = frames
    ga = access.gravity_align
    if ga is not None and np.ndim(ga) == 3 and hi > lo:
        window = np.asarray(ga[lo : min(hi, len(ga))])
        if len(window):
            roll, pitch = roll_pitch_from_gravity_align(window)
            roll, pitch = np.degrees(np.abs(roll)), np.degrees(np.abs(pitch))
            stats.append(
                stat(
                    "roll · pitch",
                    f"{roll.mean():.1f}° · {pitch.mean():.1f}°",
                    f"mean, max {roll.max():.0f}° · {pitch.max():.0f}°",
                )
            )
            notes.append(
                "Attitude comes from gravity_align.npy, which the anchoring did not touch: "
                "roll and pitch are absolute, measured against gravity."
            )

    if not stats:
        return None
    return {"kind": "motion", "title": "Path & motion", "stats": stats, "note": " ".join(notes) or None}


def hero_thumbs(access: SceneAccess, frames: tuple[int, int]) -> list[dict[str, Any]]:
    """Three frames from inside the segment, at 10 / 50 / 90 % of it.

    The same fractions and the same look as the scene page's hero, sampled
    within this stretch rather than across the whole recording -- so what the
    segment looked like is on the page before anything is opened.
    """
    lo, hi = frames
    if not access.probe.has("images") or hi <= lo:
        return []
    ts = access.frame_ts
    out = []
    for slot in THUMB_SLOTS:
        index = min(hi - 1, lo + int(round((hi - 1 - lo) * THUMB_FRACTIONS[slot])))
        at = f"{float(ts[index]) - float(ts[lo]):.0f} s" if ts is not None and index < len(ts) else None
        out.append(
            {
                "url": f"/api/scene/{access.scene_id}/frame/{index}/image.jpg?w=384",
                "href": f"/scene/{access.scene_id}/frame/{index}",
                "alt": f"scene {access.scene_id} frame {index}",
                "label": f"{THUMB_FRACTIONS[slot] * 100:.0f}%",
                "sub": f"frame {index}" + (f" · {at} in" if at else ""),
            }
        )
    return out


def segment_windows(
    access: SceneAccess, seg: Segment, max_frames: int, max_seconds: float = DEFAULT_MAX_SECONDS
) -> list[Window]:
    """The buildable windows this segment offers -- its chunks, in order."""
    return [w for w in access.windows(max_frames, max_seconds) if w.kind == "segment" and w.segment_index == seg.index]


def build_segment_page(
    access: SceneAccess,
    position: int,
    seg: Segment,
    max_frames: int,
    max_seconds: float = DEFAULT_MAX_SECONDS,
) -> dict[str, Any]:
    """Everything the segment page renders, cached on the scene handle."""

    def work() -> dict[str, Any]:
        frames = frame_range(access, position)
        segments = access.segments
        cards = [segment_card(access, seg, position, frames), motion_card(access, seg, frames)]
        return {
            "frames": frames,
            "hero": hero_thumbs(access, frames),
            "cards": [c for c in cards if c],
            # Narrowed to this segment: the drawn track, and the numbers beside it.
            "map": map_card(access, segment=seg),
            "windows": segment_windows(access, seg, max_frames, max_seconds),
            "prev": segments[position - 1] if position > 0 else None,
            "next": segments[position + 1] if position + 1 < len(segments) else None,
            "position": position,
            "n_segments": len(segments),
        }

    return access.cached(f"segment_page_{seg.index}_{max_frames}_{max_seconds}", work)
