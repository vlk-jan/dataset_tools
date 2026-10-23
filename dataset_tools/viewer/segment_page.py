"""What a segment page states about one ground-truth segment.

Reads only small cached per-scene arrays (no clouds, images or height maps).
Every number is restricted to the segment's own span: poses never cross a boundary.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import geo
from .build import ATTITUDE_BANDS, SceneAccess, window_attitude
from .index_page import THUMB_FRACTIONS, THUMB_SLOTS, band, fmt_count, fmt_duration, stat, thumb
from .loaders import Segment
from .probe import COUNTED_DIRS, OPTIONAL_ARTEFACTS
from .scene_page import MOVING_M_S, map_card, path_motion, poses_between, roll_pitch_stat, small_odometry, utc_stamp
from .signals import signal_charts
from .windows import Window, describe_reason, frames_within, segment_bounds

#: GT poses per second below which the trajectory is a guess between anchors.
THIN_POSE_RATE = 2.0

#: A segment anchored at least this densely is anchored well.
GOOD_POSE_RATE = 5.0

#: Header tooltips shared by the scene, segment and segments window tables.
COLUMN_TITLES: dict[str, str] = {
    "Scene": "the recording this window comes from",
    "Window": "one buildable 3D view: a ground-truth segment, or a part of one when it runs past the length cap",
    "Frames": "first and last frame index of the window, both inclusive",
    "Count": "how many camera frames the window covers",
    "Duration": "time from the window's first frame to its last",
    "Pose source": (
        "where the 3D view takes the robot's pose from: odometry is ground truth, "
        "transforms is the per-frame odometry track (not ground truth), none has no pose at all"
    ),
    "GT poses": (
        "ground-truth poses behind the window, and their rate; a window cut from a longer segment gets "
        f"its share of the segment's poses. Under {THIN_POSE_RATE:.0f} Hz the path between poses is "
        "interpolated over a second or more"
    ),
    "GT attitude": (
        "median tilt gap between the ground truth's orientation and the IMU's gravity over the window: "
        f"under {ATTITUDE_BANDS['ok']:.0f}° they agree, over {ATTITUDE_BANDS['warn']:.0f}° the ground truth "
        "has diverged; - when there is no ground truth or no gravity to compare"
    ),
    "Notes": "notes whose frame range touches this window, with any verdict they carry",
    "Note": "for a ground-truth segment, why it ends; otherwise what the window's pose is",
    "Ends": "why the ground-truth segment stops where it does",
}

#: GNSS and GT path lengths disagree when they differ by BOTH this fraction and
#: this many metres (the fraction alone cries wolf on short paths).
GNSS_MISMATCH_FRAC = 0.25
GNSS_MISMATCH_M = 5.0


def frame_range(access: SceneAccess, position: int) -> tuple[int, int]:
    """The segment's ``[start, end)`` frames, on the window list's own edges.

    Uses ``windows.segment_bounds`` so this page, its windows and the frame page agree.
    """
    ts = access.frame_ts
    segments = access.segments
    if ts is None or not len(ts) or position >= len(segments):
        return (0, 0)
    return frames_within(ts, *segment_bounds(ts, segments)[position])


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
    stats.append(stat("frames", f"{lo} - {hi - 1}" if hi > lo else "-", f"{hi - lo} of {p.n_frames}"))
    # Overlap with the frame clock: GT often runs past the last frame.
    if p.duration and p.start_ts is not None and p.end_ts is not None:
        overlap = max(0.0, min(seg.end_ts, p.end_ts) - max(seg.start_ts, p.start_ts))
        stats.append(stat("of the scene", f"{min(1.0, overlap / p.duration) * 100:.0f}%", "by time"))

    # How long the ground truth was untrustworthy either side.
    if position > 0:
        stats.append(stat("gap before", f"{max(0.0, seg.start_ts - segments[position - 1].end_ts):.1f} s"))
    if position + 1 < len(segments):
        stats.append(stat("gap after", f"{max(0.0, segments[position + 1].start_ts - seg.end_ts):.1f} s"))

    # Only listed for per-frame directories known to be incomplete; optional ones are sparse by design.
    missing: list[str] = []
    for kind in (k for k in COUNTED_DIRS if k not in OPTIONAL_ARTEFACTS):
        if not p.has(kind):
            continue
        if p.counts.get(kind, 0) == p.n_frames:
            continue
        have = sum(1 for i in access.available_frames(kind) if lo <= i < hi)
        missing.append(f"{kind} {have}/{hi - lo}")
        stats.append(stat(kind.replace("_", " "), f"{have}/{hi - lo}", "frames with the file"))
    if any(m.startswith("height_map") for m in missing) and p.counts.get("clouds", 0) == p.n_frames:
        notes.append(
            "A frame whose sweep has too few returns left after the ground-plane clip gets no height map; "
            "its cloud is still there."
        )

    reason = describe_reason(seg.reason)
    if reason:
        notes.append(f"Ends because {reason}.")
    if rate and rate < THIN_POSE_RATE:
        notes.append(
            f"Anchored at {rate:.1f} Hz: the trajectory between two poses is a straight line over "
            "half a second or more of real driving, so it cuts every corner the robot actually drove."
        )
    notes.append("Poses are anchored to this segment's own first pose, so they mean nothing outside it.")

    return {
        "title": "Segment",
        "badge": f"{rate:.1f} Hz" if rate else None,
        "badge_class": band(rate, GOOD_POSE_RATE, THIN_POSE_RATE),
        "badge_title": "ground-truth poses per second inside this segment",
        "stats": stats,
        "note": " ".join(notes),
    }


def motion_card(access: SceneAccess, seg: Segment, frames: tuple[int, int]) -> dict[str, Any] | None:
    """What the robot did inside the segment: how far, how fast, how steep."""
    stats: list[dict[str, Any]] = []
    notes: list[str] = []

    track = poses_between(small_odometry(access), seg.start_ts, seg.end_ts)
    if track is not None:
        ts, poses = track
        xyz = poses[:, :3, 3]
        path, speed = path_motion(ts, poses)
        stats.append(stat("path", f"{path:.0f} m", "along the ground truth"))
        moving = speed > MOVING_M_S
        stats.append(stat("moving", f"{moving.mean() * 100:.0f}%", f"over {MOVING_M_S} m/s"))
        if moving.any():
            stats.append(
                stat("speed", f"{float(speed[moving].mean()):.2f} m/s", f"while moving, max {float(speed.max()):.2f}")
            )
        net = float(np.linalg.norm(xyz[-1] - xyz[0]))
        sub = f"{net / path * 100:.0f}% of the path" if path > 0 else None
        stats.append(stat("net displacement", f"{net:.0f} m", sub))
        climb = float(xyz[:, 2].max() - xyz[:, 2].min())
        stats.append(stat("z range", f"{climb:.1f} m", "as the ground truth has it"))
        # GNSS is an independent device, so agreement with GT is a check.
        gps = access.gps
        fixes = geo.slice_time(gps, seg.start_ts, seg.end_ts) if geo.is_track(gps) else None
        if fixes is not None and len(fixes) >= 2:
            along = geo.track_length_m(fixes)
            stats.append(
                stat("GNSS track", f"{along:.0f} m", f"{along / path * 100:.0f}% of the GT path" if path > 0 else None)
            )
            gap = abs(along - path)
            if gap > GNSS_MISMATCH_M and gap > GNSS_MISMATCH_FRAC * max(path, along):
                notes.append(
                    f"The GNSS track and the ground truth disagree by {gap:.0f} m over this segment - "
                    "a fabricated stretch or a lost fix."
                )
            else:
                notes.append(
                    "The GNSS fixes are the receiver's own position, not ground truth; their agreement is a check."
                )
    else:
        notes.append("No ground-truth poses could be read for this segment.")

    lo, hi = frames
    ga = access.gravity_align
    if ga is not None and np.ndim(ga) == 3 and hi > lo:
        window = np.asarray(ga[lo : min(hi, len(ga))])
        if len(window):
            stats.append(roll_pitch_stat(window))
            notes.append(
                "Attitude comes from gravity_align.npy, which the anchoring did not touch: "
                "roll and pitch are absolute, measured against gravity."
            )

    if not stats:
        return None
    return {"title": "Path & motion", "stats": stats, "note": " ".join(notes) or None}


def hero_thumbs(access: SceneAccess, frames: tuple[int, int]) -> list[dict[str, Any]]:
    """Three frames from inside the segment, at 10 / 50 / 90 % of it.

    Same fractions as the scene page's hero, sampled within the segment.
    """
    lo, hi = frames
    if not access.probe.has("images") or hi <= lo:
        return []
    ts = access.frame_ts
    out = []
    for slot in THUMB_SLOTS:
        index = min(hi - 1, lo + int(round((hi - 1 - lo) * THUMB_FRACTIONS[slot])))
        at = f"{float(ts[index]) - float(ts[lo]):.0f} s in" if ts is not None and index < len(ts) else None
        out.append(
            thumb(access.scene_id, index, f"/api/scene/{access.scene_id}/frame/{index}/image.jpg?w=384", slot, at)
        )
    return out


#: Windows per page above which attitude verdicts are skipped (a few ms each).
#: ponytail: raise it, or compute per row on demand, if a root cuts scenes that finely.
MAX_WINDOW_VERDICTS = 200


def attitude_per_window(access: SceneAccess, windows: list[Window]) -> dict[str, dict]:
    """The GT-versus-IMU verdict for each window of a table, keyed by window key.

    Each entry carries its tooltip. A window with no verdict has no entry.
    """
    if len(windows) > MAX_WINDOW_VERDICTS:
        return {}
    title = (
        "median tilt gap between the ground truth's attitude and the IMU over this window: "
        f"under {ATTITUDE_BANDS['ok']:.0f}° the ground truth's orientation tracks the IMU, "
        f"over {ATTITUDE_BANDS['warn']:.0f}° it has diverged"
    )
    out: dict[str, dict] = {}
    for w in windows:
        verdict = window_attitude(access, w)
        if verdict is not None:
            out[w.key] = {**verdict, "title": f"{title} (95th percentile {verdict['p95']:.1f}°)"}
    return out


def segment_windows(access: SceneAccess, seg: Segment) -> list[Window]:
    """The buildable windows this segment offers -- its chunks, in order."""
    return [w for w in access.windows() if w.kind == "segment" and w.segment_index == seg.index]


def build_segment_page(access: SceneAccess, seg: Segment) -> dict[str, Any]:
    """Everything the segment page renders, cached on the scene handle."""

    def work() -> dict[str, Any]:
        position = access.segments.index(seg)
        frames = frame_range(access, position)
        segments = access.segments
        cards = [segment_card(access, seg, position, frames), motion_card(access, seg, frames)]
        return {
            "frames": frames,
            "hero": hero_thumbs(access, frames),
            "cards": [c for c in cards if c],
            "map": map_card(access, segment=seg),
            "windows": segment_windows(access, seg),
            "signals": signal_charts(access, seg.start_ts, seg.end_ts, shade=False),
            "prev": segments[position - 1] if position > 0 else None,
            "next": segments[position + 1] if position + 1 < len(segments) else None,
            "position": position,
            "n_segments": len(segments),
        }

    return access.cached(f"segment_page_{seg.index}", work)
