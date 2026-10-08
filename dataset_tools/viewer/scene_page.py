"""What a scene page states about one scene, above the tabs.

Cards from the probe, the optional ``atlas.json`` entry and small per-scene
arrays; clouds, images and height maps are never opened, ``odometry.txt`` only
when small. Each card degrades alone and is dropped when empty.
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter
from typing import Any
from urllib.parse import urlencode

import numpy as np

from . import geo
from .build import SceneAccess
from .index_page import (
    content_bytes,
    coverage_class,
    disk_parts,
    fmt_bytes,
    fmt_count,
    fmt_disk_rate,
    fmt_duration,
    fmt_share,
    images_clouds_stat,
    segment_timeline,
    shown_slots,
    split_mark,
    stat,
    thumb,
    thumb_index,
    validation_mark,
)
from .loaders import read_text, roll_pitch_from_gravity_align
from .probe import ARTEFACTS_BY_KEY, DatasetRoot, _npy_shape
from .signals import _speed, signal_charts
from .splits import DECLARED_FILES
from .windows import describe_marks, describe_reason, span_fractions

logger = logging.getLogger("dataset_tools.viewer.scene_page")

#: A card with fewer entries than this is not worth a panel of its own.
MIN_CARD_STATS = 2

#: Minimum timeline block width (fraction of the scene), so short segments stay clickable.
MIN_MARK_FRAC = 0.006

#: A wheel counts as turning above this angular speed (rad/s), above encoder noise.
TURNING_RAD_S = 0.1

#: Max ``odometry.txt`` size (bytes) parsed for a path length when the atlas has none.
ODOMETRY_READ_LIMIT = 8_000_000

#: A frame counts as moving above this speed (m/s), as in the atlas's ``moving_frac``.
MOVING_M_S = 0.1


def utc_stamp(ts: float | None) -> tuple[str, str] | None:
    """``(date, time)`` in UTC for an epoch timestamp, or None."""
    if not ts or ts < 1e8:  # not a wall-clock stamp (a synthesised 0-based one)
        return None
    try:
        parts = time.gmtime(ts)
    except (OverflowError, OSError, ValueError):
        return None
    return time.strftime("%Y-%m-%d", parts), time.strftime("%H:%M", parts)


# --------------------------------------------------------------------------
# the four cards
# --------------------------------------------------------------------------


def _sampled_cloud_points(access: SceneAccess) -> tuple[int, int] | None:
    """``(points, frame)`` from one cloud's ``.npy`` header, or None.

    Stands in for a curated ``pts_mean``.
    """
    if not access.probe.has("clouds") or not access.probe.n_frames:
        return None
    index = access.nearest_frame("clouds", access.probe.n_frames // 2)
    if index is None:
        return None
    shape = _npy_shape(access.frame_path("clouds", index))
    return (int(shape[0]), index) if shape else None


def recording_card(access: SceneAccess, a: dict, info: dict | None) -> dict[str, Any]:
    """What the recording *is*: when, how long, how much of it there is."""
    p = access.probe
    stats: list[dict[str, Any]] = []

    stamp = utc_stamp(p.start_ts)
    if stamp:
        stats.append(stat("recorded", stamp[0], f"{stamp[1]} UTC"))
    elif a.get("date"):
        stats.append(stat("recorded", a["date"], "UTC"))

    duration = p.duration or a.get("dur_s")
    stats.append(stat("duration", fmt_duration(duration)))
    stats.append(stat("synced pairs", fmt_count(p.n_frames or a.get("frames"))))

    ts = access.frame_ts
    if ts is not None and len(ts) > 1:
        step = float(np.median(np.diff(ts)))
        if step > 0:
            stats.append(stat("frame rate", f"{1.0 / step:.2f} Hz"))

    if counted := images_clouds_stat(p):
        stats.append(counted)

    if a.get("pts_mean"):
        stats.append(stat("pts / cloud", fmt_count(a["pts_mean"]), "mean over the scene"))
    else:
        sample = _sampled_cloud_points(access)
        if sample:
            stats.append(stat("pts / cloud", fmt_count(sample[0]), f"frame {sample[1]}"))

    if info and info.get("image_width") and info.get("image_height"):
        stats.append(stat("image size", f"{info['image_width']} × {info['image_height']}", "px"))

    bits = []
    if p.bag:
        bits.append(p.bag)
    if a.get("bag_gb"):
        bits.append(f"{a['bag_gb']:.1f} GB")
    if a.get("session"):
        bits.append(f"session {a['session']}")
    return {
        "kind": "recording",
        "title": "Recording",
        "stats": stats,
        "note": " · ".join(bits) or None,
        "warn": "; ".join(p.warnings) or None,
    }


def path_and_motion(access: SceneAccess, bounds: list[tuple[float, float]] | None = None) -> tuple[float, float] | None:
    """``(path metres, moving fraction)`` from this scene's own odometry.

    Accumulated inside segments, never across one. ``bounds`` overrides the
    segment list with explicit ``(start, end)`` pairs.
    """
    odom = small_odometry(access)
    if odom is None:
        return None
    ts, poses = odom
    if len(ts) < 2:
        return None
    if bounds is None:
        bounds = [(s.start_ts, s.end_ts) for s in access.segments] or [(ts[0], ts[-1])]
    total, moving, count = 0.0, 0, 0
    for lo, hi in bounds:
        stretch = poses_between(odom, lo, hi)
        if stretch is None:
            continue
        path, speed = path_motion(*stretch)
        total += path
        moving += int((speed > MOVING_M_S).sum())
        count += len(speed)
    if not count:
        return None
    return total, moving / count


def small_odometry(access: SceneAccess) -> tuple[np.ndarray, np.ndarray] | None:
    """``access.odometry``, only when ``odometry.txt`` is at most :data:`ODOMETRY_READ_LIMIT`."""
    try:
        if (access.dir / "odometry.txt").stat().st_size > ODOMETRY_READ_LIMIT:
            return None
    except OSError:
        return None
    return access.odometry


def poses_between(
    odom: tuple[np.ndarray, np.ndarray] | None, lo: float, hi: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """``(ts, poses)`` of the odometry inside ``[lo, hi]``, or None under two poses."""
    if odom is None:
        return None
    ts, poses = odom
    take = (ts >= lo) & (ts <= hi)
    if take.sum() < 2:
        return None
    return ts[take], poses[take]


def path_motion(ts: np.ndarray, poses: np.ndarray) -> tuple[float, np.ndarray]:
    """``(path metres, speed per step)`` along one stretch of poses.

    Speeds are :func:`signals._speed`'s, so cards and charts agree.
    """
    xyz = poses[:, :3, 3]
    return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()), _speed(ts, xyz)[1]


def timeline_marks(access: SceneAccess) -> list[dict[str, Any]]:
    """One clickable block per ground-truth segment, in scene-time fractions.

    Carries each segment's file number. Empty without segments or a time span;
    the timeline then falls back to plain spans.
    """
    p, segments = access.probe, access.segments
    if not segments or p.start_ts is None or not p.duration or p.duration <= 0:
        return []
    base, span = p.start_ts, p.duration
    marks: list[dict[str, Any]] = []
    for seg in segments:
        lo, hi = span_fractions(seg, base, span)
        width = min(max(hi - lo, MIN_MARK_FRAC), 1.0 - lo)
        # The hover names where filled poses came from, not why the segment ended.
        described = describe_reason(seg.reason)
        why = described.split("; ends because", 1)[0] if described.startswith("filled") else ""
        title = f"segment {seg.index} - {fmt_duration(seg.duration)}, {fmt_count(seg.n_poses)} GT poses"
        marks.append(
            {
                "index": seg.index,
                "left": round(lo * 100, 3),
                "width": round(width * 100, 3),
                "fill": "fill" in (seg.reason or "").lower(),
                "href": f"/scene/{p.scene_id}/segment/{seg.index}",
                "title": f"{title}; {why}" if why else title,
            }
        )
    return marks


def ground_truth_card(access: SceneAccess, a: dict) -> dict[str, Any]:
    """Where trustworthy poses exist inside the scene, and how good they are.

    The live segment file is preferred over the atlas; only what is expensive
    to recompute comes from the atlas.
    """
    p = access.probe
    segments = access.segments
    stats: list[dict[str, Any]] = []
    notes: list[str] = []

    # Same source as the index badge.
    coverage = a.get("cov", p.segment_coverage)

    if segments:
        lens = sorted((s.duration for s in segments), reverse=True)
        longest = ", ".join(f"{v:.0f}" for v in lens[:3])
        stats.append(stat("segments", len(segments), f"longest {longest} s"))
        stats.append(
            stat(
                "GT span",
                fmt_duration(sum(lens)),
                f"median {float(np.median(lens)):.1f} s",
            )
        )
        poses = sum(s.n_poses for s in segments)
        if poses:
            # "of N raw" only when N >= poses (filled scenes have more than their own run).
            raw = a.get("poses_unf") or a.get("poses_raw")
            sub = f"of {fmt_count(raw)} raw" if raw and raw >= poses else None
            stats.append(stat("poses", fmt_count(poses), sub))
        reasons = sorted(p.segment_reasons.items(), key=lambda kv: -kv[1])
        # Fills are not end reasons; the fill note covers them.
        ends = [(describe_reason(r), n) for r, n in reasons]
        ends = [(d, n) for d, n in ends if d and not d.startswith("filled")]
        if ends:
            parts = [f"{d} ({n}×)" for d, n in ends[:3]]
            notes.append("Segments end because " + "; ".join(parts) + ".")
    elif p.has("odometry"):
        notes.append("odometry.txt is present but unsegmented: the whole scene is one stretch.")
    else:
        notes.append("No ground-truth odometry in this scene.")

    if a.get("path_m") is not None:
        stats.append(stat("path in GT", f"{a['path_m']:.0f} m"))
        if a.get("moving_frac") is not None:
            stats.append(stat("moving", f"{a['moving_frac'] * 100:.0f}%", f"over {MOVING_M_S} m/s"))
    else:
        derived = path_and_motion(access)
        if derived:
            stats.append(stat("path in GT", f"{derived[0]:.0f} m", "from odometry.txt"))
            stats.append(stat("moving", f"{derived[1] * 100:.0f}%", f"over {MOVING_M_S} m/s"))

    if a.get("win_med") is not None and a.get("win_p95") is not None:
        stats.append(
            stat("2 s err med / p95", f"{a['win_med']:.2f} / {a['win_p95']:.2f} m", "vs the robot's own odometry")
        )
    if a.get("flags") is not None:
        stats.append(stat("divergence flags", a["flags"]))

    if not p.has("odometry") and p.has("transforms"):
        notes.append("Wheel odometry only (transforms.npy): continuous, drifty, not ground truth.")

    spans, fill = segment_timeline(p, a)
    marks = timeline_marks(access)
    if fill:
        # Name the fill sources this root holds, then count the marks.
        filled = [s for s in segments if "fill" in (s.reason or "").lower()]
        sources = sorted({describe_reason((s.reason or "").split("+")[0]) for s in filled})
        mark_counts = Counter(m for s in filled for m in describe_marks(s.reason or "", brief=True))
        detail = "; ".join(sources) if sources else "filled from another root through the same quality gate"
        for text, n in sorted(mark_counts.items(), key=lambda kv: -kv[1]):
            detail += f"; {n} of them {text}"
        notes.append(f"{len(fill)} segment{'s' if len(fill) != 1 else ''} (amber): {detail}.")

    return {
        "kind": "gt",
        "title": "Ground truth",
        "badge": f"{coverage * 100:.0f}% GT" if coverage is not None else None,
        "badge_class": coverage_class(coverage),
        "badge_title": "fraction of the scene inside a ground-truth segment",
        "stats": stats,
        "timeline": (
            {
                "spans": spans,
                "fill": fill,
                "marks": marks,
                "label": fmt_duration(p.duration or a.get("dur_s")),
            }
            if spans
            else None
        ),
        "note": " ".join(notes) or None,
    }


def calibration_card(access: SceneAccess, info: dict | None) -> dict[str, Any] | None:
    """The camera: what it saw, and where it sat on the robot."""
    K = access.K
    T = access.T_cam
    if K is None and T is None and not info:
        return None

    stats: list[dict[str, Any]] = []
    matrices: list[dict[str, Any]] = []
    fx = fy = None
    if K is not None:
        K = np.asarray(K, dtype=float)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        stats.append(stat("focal length", f"{fx:.1f} · {fy:.1f}", "fx · fy, px"))
        stats.append(stat("principal point", f"{cx:.1f} · {cy:.1f}", "cx · cy, px"))
        matrices.append({"label": "camera_intrinsics.txt (K)", "rows": K.round(3).tolist(), "fmt": "%10.3f"})

    width = info.get("image_width") if info else None
    height = info.get("image_height") if info else None
    if width and height:
        model = (info or {}).get("distortion_model")
        coeffs = (info or {}).get("distortion_coefficients") or []
        # All-zero coefficients differ from none: say so.
        described = f"{len(coeffs)} coeff{'' if any(coeffs) else ', all zero'}" if coeffs else None
        sub = " · ".join(x for x in (model, described) if x)
        stats.append(stat("image", f"{width} × {height}", sub or "px"))
        if fx and fy:
            hfov = math.degrees(2 * math.atan(width / (2 * fx)))
            vfov = math.degrees(2 * math.atan(height / (2 * fy)))
            stats.append(stat("field of view", f"{hfov:.0f}° × {vfov:.0f}°", "horizontal × vertical"))

    if T is not None:
        T = np.asarray(T, dtype=float)
        t = T[:3, 3]
        stats.append(stat("camera → base", f"{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}", "x, y, z (m)"))
        source = "T_cam_to_base.txt" if access.probe.has("T_cam_to_base") else "T_cam_to_lidar.txt"
        matrices.append({"label": f"{source} (4×4)", "rows": T.round(4).tolist(), "fmt": "%9.4f"})

    if len(stats) < MIN_CARD_STATS and not matrices:
        return None
    frame = (info or {}).get("frame_id") or (info or {}).get("camera_name")
    return {
        "kind": "calib",
        "title": "Calibration",
        "stats": stats,
        "matrices": matrices,
        "note": f"frame {frame}" if frame else None,
    }


def _series(arr: np.ndarray | None) -> np.ndarray | None:
    """A timestamped ``(T, K)`` array (column 0 the time) with rows in it, or None."""
    return arr if arr is not None and np.ndim(arr) == 2 and len(arr) and arr.shape[1] >= 2 else None


def _rate(arr: np.ndarray) -> float | None:
    """The file's own rate (Hz): its rows over the span its first and last stamps cover."""
    span = float(arr[-1, 0] - arr[0, 0])
    return len(arr) / span if span > 0 else None


def _mean_abs(arr: np.ndarray, cols) -> float:
    return float(np.nanmean(np.abs(arr[:, cols])))


def _max_abs(arr: np.ndarray, cols) -> float:
    return float(np.nanmax(np.abs(arr[:, cols])))


def roll_pitch_stat(ga: np.ndarray) -> dict[str, Any]:
    """The ``roll · pitch`` cell over a stack of gravity-alignment rotations: mean and max tilt."""
    roll, pitch = roll_pitch_from_gravity_align(np.asarray(ga))
    roll, pitch = np.degrees(np.abs(roll)), np.degrees(np.abs(pitch))
    return stat(
        "roll · pitch", f"{roll.mean():.1f}° · {pitch.mean():.1f}°", f"mean, max {roll.max():.0f}° · {pitch.max():.0f}°"
    )


def proprioception_card(access: SceneAccess) -> dict[str, Any] | None:
    """What the robot reported about itself: wheels, commands, attitude."""
    p = access.probe
    stats: list[dict[str, Any]] = []
    notes: list[str] = []

    js = _series(access.joint_states)
    if js is not None and js.shape[1] >= 10:
        vel = slice(4, 7)
        stats.append(stat("wheel speed", f"{_mean_abs(js, vel):.2f} rad/s", f"mean |ω|, max {_max_abs(js, vel):.1f}"))
        turning = float((np.abs(js[:, vel]) > TURNING_RAD_S).any(axis=1).mean())
        stats.append(stat("wheels turning", f"{turning * 100:.0f}%", f"any |ω| > {TURNING_RAD_S} rad/s"))
        effort = slice(7, 10)
        stats.append(stat("wheel effort", f"{_mean_abs(js, effort):.1f}", f"mean, max {_max_abs(js, effort):.0f}"))
    elif js is not None:
        notes.append(f"joint_states.npy is {js.shape[1]} columns wide, not the expected 10.")
    if js is not None:
        rate = _rate(js)
        stats.append(stat("joint states", f"{fmt_count(len(js))} × {js.shape[1]}", f"{rate:.1f} Hz" if rate else None))

    cv = _series(access.cmd_vel)
    if cv is not None and cv.shape[1] >= 7:
        stats.append(
            stat("commanded speed", f"{_mean_abs(cv, 1):.2f} m/s", f"mean |v|, |ω| {_mean_abs(cv, 6):.2f} rad/s")
        )
    if cv is not None:
        rate = _rate(cv)
        stats.append(stat("cmd_vel", f"{fmt_count(len(cv))} × {cv.shape[1]}", f"{rate:.1f} Hz" if rate else None))

    sp = _series(access.joint_setpoints)
    if sp is not None and sp.shape[1] >= 7:
        rate = _rate(sp)
        stats.append(
            stat(
                "commanded wheels",
                f"{_mean_abs(sp, slice(4, 7)):.2f} rad/s",
                f"setpoints, {rate:.1f} Hz" if rate else "setpoints",
            )
        )

    ga = access.gravity_align
    if ga is not None and np.ndim(ga) == 3:
        stats.append(roll_pitch_stat(ga))

    if len(stats) < MIN_CARD_STATS:
        return None
    if not p.has("cmd_vel") and not p.has("joint_setpoints"):
        notes.append("Nothing recorded what was commanded, only what the wheels did.")
    return {
        "kind": "proprio",
        "title": "Proprioception",
        "stats": stats,
        "note": " ".join(notes) or None,
    }


def _report_lines(access: SceneAccess, name: str) -> tuple[list[str], list[str]]:
    """``(comment lines, data lines)`` of one of the extractor's report files."""
    text = read_text(access.dir / name) or ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return [ln for ln in lines if ln.startswith("#")], [ln for ln in lines if not ln.startswith("#")]


def _time_align_stats(access: SceneAccess) -> list[dict[str, Any]]:
    """What ``time_align.txt`` says the frame clock was moved by.

    Second comment line is ``time_align.Profile.summary``; rows are
    ``t delta gain depth`` per window (none for a constant offset).
    """
    comments, rows = _report_lines(access, "time_align.txt")
    summary = comments[1].lstrip("# ").strip() if len(comments) > 1 else ""
    if summary.startswith("no correction"):
        return [stat("frame clock", "not corrected", summary[len("no correction") :].strip("() "))]
    shape, _, rest = summary.partition(" from ")
    stats = [stat("frame clock", shape or "corrected", rest.strip() or None)]
    deltas = np.array([float(r.split()[1]) for r in rows]) if rows else np.zeros(0)
    if len(deltas):
        biggest = deltas[np.argmax(np.abs(deltas))]
        stats.append(
            stat(
                "measured offset",
                f"{np.median(deltas) * 1000:+.0f} ms",
                f"median over {len(deltas)} windows, largest {biggest * 1000:+.0f} ms",
            )
        )
    return stats


def _gnss_guard_stats(access: SceneAccess) -> tuple[list[dict[str, Any]], str | None]:
    """What ``gnss_guard.txt`` says was removed: two counted comment lines and one row per stretch."""
    comments, rows = _report_lines(access, "gnss_guard.txt")
    counts = re.findall(r"(\d+)/(\d+) (poses|frames|further frames|more frames)", " ".join(comments))
    by = {what: (int(a), int(b)) for a, b, what in counts}
    poses, frames, unsupported = by.get("poses"), by.get("frames"), by.get("further frames")
    if poses is None or frames is None or unsupported is None:
        raise ValueError("gnss_guard.txt has no counts in its header")
    stats = [
        stat("fabricated stretches", len(rows), "removed by the GNSS guard"),
        stat("cost", f"{poses[0]:,} poses · {frames[0]:,} frames", f"of {poses[1]:,} · {frames[1]:,}"),
        stat("no pose under them", f"{unsupported[0]:,} frames", f"of {unsupported[1]:,}, dropped as well"),
    ]
    if "more frames" in by:  # written since 2026-10-01
        trimmed = by["more frames"]
        stats.append(
            stat("no pose after the clock correction", f"{trimmed[0]:,} frames", f"of {trimmed[1]:,}, dropped")
        )
    reasons = sorted({r.split()[2] for r in rows if len(r.split()) > 2})
    note = f"Stretches caught as {', '.join(reasons)}." if reasons else None
    return stats, note


def _tf_pins_stats(access: SceneAccess) -> list[dict[str, Any]]:
    """How far each pinned transform sat from what the bag itself published.

    Recomputed from the pin/bag rows through ``tf_pins.Record``.
    """
    from .. import tf_pins

    _, rows = _report_lines(access, "tf_pins.txt")
    said: dict[tuple[str, str], dict[str, tuple]] = {}
    for row in rows:
        parent, child, which, *nums = row.split()
        xyz, rpy = tuple(map(float, nums[:3])), tuple(map(float, nums[3:6]))
        said.setdefault((parent, child), {})[which] = (xyz, rpy)
    stats = []
    for (parent, child), have in said.items():
        if "pin" not in have or "bag" not in have:
            continue
        pin = tf_pins.Pin(parent, child, have["pin"][0], have["pin"][1])
        record = tf_pins.Record(pin, have["bag"][0], have["bag"][1])
        stats.append(
            stat(
                f"{parent} → {child}",
                f"{record.translation_m * 100:.1f} cm · {record.rotation_deg:.2f}°",
                "bag from pin" + (", far" if record.far else ""),
            )
        )
    return stats


def extraction_card(access: SceneAccess) -> dict[str, Any] | None:
    """What the extractor did to the recording on its way to a scene.

    Three optional reports; a malformed one costs only its part. ``None`` without any.
    """
    p = access.probe
    stats: list[dict[str, Any]] = []
    notes: list[str] = []
    warnings: list[str] = []
    parts = (
        ("time_align", "time_align.txt", lambda: (_time_align_stats(access), None)),
        ("gnss_guard", "gnss_guard.txt", lambda: _gnss_guard_stats(access)),
        ("tf_pins", "tf_pins.txt", lambda: (_tf_pins_stats(access), None)),
    )
    for key, name, read in parts:
        if not p.has(key):
            continue
        try:
            more, note = read()
        except (ValueError, IndexError) as exc:
            logger.warning("cannot read %s of scene %s: %s", name, p.scene_id, exc)
            warnings.append(f"{name} could not be read")
            continue
        stats.extend(more)
        if note:
            notes.append(note)
    if not stats and not warnings:
        return None
    if p.has("time_align"):
        notes.append(
            "The offset is what was added to the frame stamps: negative moves a frame earlier, "
            "to when it was captured rather than published."
        )
    absent = [name for key, name, _ in parts if not p.has(key)]
    if absent:
        notes.append(f"No {', '.join(absent)} in this scene.")
    return {
        "kind": "extraction",
        "title": "Extraction",
        "stats": stats,
        "note": " ".join(notes) or None,
        "warn": "; ".join(warnings) or None,
    }


def validation_card(access: SceneAccess) -> dict[str, Any] | None:
    """What ``helhest-validate`` made of the scene; None without a report row.

    Messages come from ``validation.json`` when present; the CSV holds only counts.
    """
    row = access.root.validation_scene(access.scene_id)
    mark = validation_mark(row)
    if mark is None:
        return None
    messages = access.root.validation_messages(access.scene_id)
    stats = [
        stat("errors", row.get("errors") or "0", "structural: the scene contract"),
        stat("warnings", row.get("warnings") or "0", "data quality: NaNs, jumps, tilt"),
    ]
    for key, label, unit in (
        ("max_step_m", "max step", "m between frames"),
        ("max_speed_mps", "max speed", "m/s implied by it"),
        ("tilt_max_deg", "max tilt", "degrees"),
        ("max_sync_offset_s", "max sync offset", "s, camera vs lidar"),
        ("nan_frac", "non-finite", "fraction of cloud points"),
    ):
        if row.get(key):
            stats.append(stat(label, row[key], unit))
    note = None
    if messages is None and mark["cls"] != "yes":
        note = (
            f"For the messages, run helhest-validate {access.scene_id} on the root, or write them next to "
            "validation.csv with --json validation.json."
        )
    card = {
        "kind": "validation",
        "title": "Validation",
        "badge": mark["status"],
        "badge_class": {"yes": "hi", "partial": "lo", "no": "lo"}[mark["cls"]],
        "badge_title": "the status helhest-validate gave this scene",
        "stats": stats,
        "note": note,
    }
    if messages:
        card.update(_validation_findings(access, messages))
    return card


#: Most severe first in the card's list.
SEVERITY_ORDER = {"ERROR": 0, "WARN": 1, "INFO": 2}


def _validation_findings(access: SceneAccess, messages: list[dict]) -> dict[str, Any]:
    """The validator's messages, plus timeline marks for those with a time."""
    ts = access.frame_ts
    duration = float(ts[-1] - ts[0]) if ts is not None and len(ts) > 1 else 0.0

    def frame_at(t: float) -> int:
        return int(min(np.searchsorted(ts, ts[0] + t), len(ts) - 1))

    findings, marks = [], []
    for m in sorted(messages, key=lambda m: (SEVERITY_ORDER.get(m.get("severity"), 3), m.get("t0") or 0)):
        severity = "expected" if m.get("expected") else str(m.get("severity", ""))
        href = None
        if m.get("t0") is not None and duration > 0:
            href = f"/scene/{access.scene_id}/frame/{frame_at(float(m['t0']))}"
            t0, t1 = float(m["t0"]), float(m.get("t1", m["t0"]))
            marks.append(
                {
                    "left": round(max(0.0, t0 / duration) * 100, 3),
                    "width": round(max(0.4, (t1 - t0) / duration * 100), 3),
                    "href": href,
                    "title": f"{m.get('code')}: {m.get('msg')}",
                    "fill": severity != "INFO",
                }
            )
        findings.append({"severity": severity, "code": m.get("code"), "text": m.get("msg"), "href": href})
    out: dict[str, Any] = {"findings": findings}
    if marks:
        out["timeline"] = {
            "marks": marks,
            "label": fmt_duration(duration),
            "title": "where the validator's findings are in the scene",
            "hint": "Each block is a finding with a place in time - click it to open its first frame.",
        }
    return out


def _gps_src(scene_id: str, segment, frame: int | None) -> str:
    """The endpoint one map fetches its track from."""
    params = {"segment": segment.index if segment is not None else None, "frame": frame}
    query = urlencode({k: v for k, v in params.items() if v is not None})
    return f"/api/scene/{scene_id}/gps.json" + (f"?{query}" if query else "")


def map_card(access: SceneAccess, segment=None, frame: int | None = None) -> dict[str, Any] | None:
    """Where on the Earth this was recorded -- ``None`` when nothing says.

    ``segment`` narrows the track and numbers to its span; ``frame`` adds a
    marker. GNSS is not ground truth.
    """
    gps = access.gps
    if not geo.is_track(gps):
        return None
    track = geo.slice_time(gps, segment.start_ts, segment.end_ts) if segment is not None else gps
    if not len(track):
        return None

    stats: list[dict[str, Any]] = []
    if frame is not None:
        ts = access.frame_ts
        here = geo.locate(gps, float(ts[frame])) if ts is not None and 0 <= frame < len(ts) else None
        stats.append(
            stat("this frame", f"{here[0]:.6f}, {here[1]:.6f}", "latitude, longitude")
            if here
            else stat("this frame", "-", "no fix within a second of it")
        )
    span = float(track[-1, geo.T] - track[0, geo.T])
    rate = f"{len(track) / span:.1f} Hz" if span > 0 else None
    stats.append(stat("fixes", fmt_count(len(track)), rate))
    stats.append(stat("track", f"{geo.track_length_m(track):.0f} m", "along the fixes"))
    ew, ns = geo.span_m(track)
    stats.append(stat("extent", f"{ew:.0f} × {ns:.0f} m", "east-west × north-south"))
    sigma = geo.sigma_m(track)
    if sigma is not None:
        stats.append(stat("accuracy", f"± {sigma:.2f} m", "median horizontal 1σ"))
    shares = geo.fix_shares(track)
    if len(shares) > 1:
        rest = "; ".join(f"{label} {share * 100:.0f}%" for label, share in shares[1:4])
        stats.append(stat("also", rest, "of the fixes"))

    best = shares[0] if shares else None
    return {
        "kind": "map",
        "title": "Where",
        "badge": f"{best[0]} {best[1] * 100:.0f}%" if best else None,
        "badge_class": coverage_class(best[1]) if best else "none",
        "badge_title": "the commonest GNSS fix quality over this track",
        "stats": stats,
        "src": _gps_src(access.scene_id, segment, frame),
        "note": "GNSS as the receiver reported it - where the robot was on the Earth, not ground truth.",
    }


# --------------------------------------------------------------------------
# the page
# --------------------------------------------------------------------------


def hero_thumbs(access: SceneAccess, a: dict) -> list[dict[str, Any]]:
    """The same three frames the index card shows, at the top of the scene page.

    Same endpoint and ``thumb_index`` as the index; each links to its frame.
    """
    p = access.probe
    slots = shown_slots(p, a)
    ts = access.frame_ts if slots else None
    out = []
    for slot in slots:
        index = thumb_index(a, slot, p.n_frames)
        at = f"{float(ts[index]) - float(ts[0]):.0f} s" if ts is not None and index < len(ts) else None
        url = f"/api/scene/{p.scene_id}/thumb/{slot}.jpg"
        out.append(thumb(p.scene_id, index, url, slot, at, link=p.has("images")))
    return out


def disk_card(access: SceneAccess) -> dict[str, Any] | None:
    """Where this scene's bytes go, from ``disk_usage.json``; None without an entry."""
    d = access.root.disk_usage_scene(access.scene_id)
    parts = disk_parts([d])
    if not parts:
        return None
    total = sum(p["bytes"] for p in parts)
    stats = [stat("on disk", fmt_bytes(total), fmt_disk_rate(total, d.get("span_s")))]
    if d.get("frames"):
        mods = d.get("modalities") or {}
        count = {name: (m.get("files") or 0) + (m.get("linked_files") or 0) for name, m in mods.items()}
        each = [
            f"{name.replace('_', ' ')} {fmt_bytes(content_bytes(mods[name]) / count[name])}"
            for name in ("images", "clouds")
            if count.get(name)
        ]
        stats.append(stat("per frame", fmt_bytes(total / d["frames"]), " · ".join(each) or None))
    return {
        "stats": stats,
        "parts": parts,
        "note": f"measured {d['scanned'][:10]} by helhest-disk-usage" if d.get("scanned") else None,
    }


def file_sizes(access: SceneAccess) -> dict[str, int]:
    """Bytes per top-level entry of the scene, directories included, as last measured."""
    d = access.root.disk_usage_scene(access.scene_id)
    sizes = {name: content_bytes(m) for name, m in (d.get("modalities") or {}).items() if name != "other"}
    sizes.update(d.get("files") or {})
    sizes.update({name: f.get("bytes") for name, f in (d.get("linked_files") or {}).items()})
    return {k: v for k, v in sizes.items() if isinstance(v, int)}


#: The file card's rows: what each artefact is for, in the order a reader asks.
FILE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Per frame", ("images", "clouds", "height_maps", "height_masks", "driven_similarity",
                   "data_ts", "pcd_ts", "transforms", "gravity_align")),
    ("Ground truth", ("odometry", "odometry_segments")),
    ("Robot & GNSS", ("joint_states", "joint_setpoints", "cmd_vel", "gps")),
    ("Calibration", ("camera_intrinsics", "camera_info", "T_cam_to_base", "T_cam_to_lidar")),
    ("Extraction", ("source_bag", "time_align", "gnss_guard", "tf_pins")),
)  # fmt: skip

#: Arrays with one row per frame, whose count is read against the frame count.
_PER_FRAME_FILES = ("data_ts", "pcd_ts", "transforms", "gravity_align")
#: Arrays on their own clock, whose count is read as a rate over the scene.
_TIME_SERIES = ("joint_states", "joint_setpoints", "cmd_vel", "gps")

_STATE_WORD = {"yes": "present", "partial": "partial", "no": "missing", "optional": "not written"}


def frame_gaps(have: set[int], n_frames: int) -> list[tuple[int, int]]:
    """The runs of frame indices ``[first, last]`` not in ``have``."""
    have = np.array(sorted(have), dtype=int)
    missing = np.ones(n_frames, dtype=np.int8)
    missing[have[have < n_frames]] = 0
    # Where "missing" switches on and off: each run's first index, then one past its last.
    edges = np.flatnonzero(np.diff(missing, prepend=0, append=0))
    return [(int(a), int(b) - 1) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def _gap_text(gaps: list[tuple[int, int]], shown: int = 6) -> str:
    runs = [f"{a}" if a == b else f"{a}-{b}" for a, b in gaps[:shown]]
    more = len(gaps) - shown
    return ", ".join(runs) + (f" and {more} more gap{'s' if more > 1 else ''}" if more > 0 else "")


def _coverage(access: SceneAccess, art: Any, count: int) -> dict[str, Any] | None:
    """Where a per-frame directory has no file: a bar of the gaps and a line naming them.

    Lists the directory; only called for partial ones.
    """
    n = access.probe.n_frames
    gaps = frame_gaps(access.available_frames(art.key), n)
    if not gaps:
        return None
    # Merge runs closer than ~a pixel; CSS gives marks a minimum width.
    marks: list[list[float]] = []
    for a, b in gaps:
        left, right = a / n * 100, (b + 1) / n * 100
        if marks and left - marks[-1][1] < 0.3:
            marks[-1][1] = right
        else:
            marks.append([left, right])
    return {
        "marks": [{"left": lo, "width": hi - lo} for lo, hi in marks],
        "text": _gap_text(gaps),
        "n_missing": n - count,
    }


def file_entries(access: SceneAccess) -> list[dict[str, Any]]:
    """The file inventory in group order, each entry carrying what it is and what it is for.

    State is ``SceneProbe.state``, as on the index page.
    """
    p = access.probe
    sizes = file_sizes(access)
    grouped = {k for _, keys in FILE_GROUPS for k in keys}
    groups = [*FILE_GROUPS, ("Other", tuple(k for k in ARTEFACTS_BY_KEY if k not in grouped))]
    out: list[dict[str, Any]] = []
    for title, keys in groups:
        for art in (ARTEFACTS_BY_KEY[k] for k in keys if k in ARTEFACTS_BY_KEY):
            state = p.state(art.key)
            count = p.counts.get(art.key)
            size = sizes.get(art.relpath) if p.has(art.key) else None
            stats: list[dict[str, Any]] = []
            coverage = None
            if count is not None and (art.kind == "dir" or art.key in _PER_FRAME_FILES) and p.n_frames:
                share = count / p.n_frames
                stats.append(
                    stat(
                        "frames",
                        f"{count:,} of {p.n_frames:,}",
                        (
                            "every frame"
                            if count >= p.n_frames
                            else f"{fmt_share(share)}, {p.n_frames - count:,} without one"
                        ),
                    )
                )
                if state == "partial":
                    coverage = _coverage(access, art, count)
            elif count is not None:
                rate = (
                    f"≈ {count / p.duration:.1f} Hz over the scene" if art.key in _TIME_SERIES and p.duration else None
                )
                stats.append(stat(art.count_unit, f"{count:,}", rate))
            if size is not None:
                each = f"{fmt_bytes(size / count)} per file" if art.kind == "dir" and count else None
                stats.append(stat("on disk", fmt_bytes(size), each))
            out.append(
                {
                    "key": art.key,
                    "label": art.label,
                    "relpath": art.relpath,
                    "group": title,
                    "present": p.has(art.key),
                    "size": fmt_bytes(size) if size is not None else None,
                    "state": state,
                    "state_word": _STATE_WORD[state],
                    "stats": stats,
                    "coverage": coverage,
                    "shape": art.description,
                    # The prose is written with ASCII "--"; the card sets it as a plain dash.
                    "purpose": art.purpose.replace(" -- ", " - "),
                }
            )
    return out


def file_groups(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The inventory as the card's rows, in ``FILE_GROUPS`` order."""
    titles = list(dict.fromkeys(e["group"] for e in entries))
    return [{"title": t, "entries": [e for e in entries if e["group"] == t]} for t in titles]


def file_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """The file card's badge, and a line naming what is missing (the badge counts what is short)."""
    missing = [e["relpath"] for e in entries if e["state"] == "no"]
    partial = [e["relpath"] for e in entries if e["state"] == "partial"]
    bits = [f"{len(missing)} missing"] * bool(missing) + [f"{len(partial)} partial"] * bool(partial)
    lines = []
    if missing:
        lines.append(f"Missing: {', '.join(missing)}.")
    return {
        "badge": " · ".join(bits) or "complete",
        "badge_class": "lo" if bits else "hi",
        "lines": lines,
        "present": sum(e["state"] in ("yes", "partial") for e in entries),
        "total": len(entries),
    }


#: Root-level files the scene page shows beside the scene's own: a change to one
#: must show without waiting for the scene directory's fingerprint to move.
PAGE_ROOT_FILES = (
    "atlas/atlas.json",
    "validation.csv",
    "validation.json",
    "disk_usage.json",
    "splits.json",
    *DECLARED_FILES,
)


def build_scene_page(root: DatasetRoot, access: SceneAccess) -> dict[str, Any]:
    """Thumbnails and the cards, cached on the scene handle until one of
    :data:`PAGE_ROOT_FILES` changes.

    Cached: small-array reads on network storage add up.
    """

    def work() -> dict[str, Any]:
        a = root.atlas_scene(access.scene_id)
        info = access.camera_info
        cards = [
            recording_card(access, a, info),
            ground_truth_card(access, a),
            calibration_card(access, info),
            proprioception_card(access),
            extraction_card(access),
            validation_card(access),
        ]
        return {
            "hero": hero_thumbs(access, a),
            "cards": [c for c in cards if c],
            "split": split_mark(root, access.scene_id),
            # map and disk are None (not empty) without gps.npy / disk_usage.json.
            "map": map_card(access),
            "disk": disk_card(access),
            "files": (files := file_entries(access)),
            "file_groups": file_groups(files),
            "files_summary": file_summary(files),
            "signals": scene_signals(access),
        }

    stamp = root.root_stamp(PAGE_ROOT_FILES)
    hit = access._cache.get("scene_page")
    if hit is None or hit[0] != stamp:
        hit = access._cache["scene_page"] = (stamp, work())
    return hit[1]


def scene_signals(access: SceneAccess) -> dict[str, Any] | None:
    """The signal charts over the whole recording, segments shaded behind."""
    ts = access.frame_ts
    if ts is None or len(ts) < 2:
        return None
    return signal_charts(access, float(ts[0]), float(ts[-1]))
