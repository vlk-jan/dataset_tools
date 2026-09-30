"""What a scene page states about one scene, above the tabs.

Four cards -- what was recorded, what ground truth is inside it, how the camera
was calibrated, what the robot reported -- from the directory probe, the curated
``atlas.json`` entry if there is one, and a few per-scene arrays read only here.
Where the scene's bytes go sits beside the file inventory when the root has a
``disk_usage.json``; the sizes in the inventory come from the same entry.

The reading budget is deliberate: everything is either already in memory or in
a file bounded by the frame count (the proprioception arrays, read strided
through a memory map; ``camera_info.yaml``). Clouds, images and height maps are
never opened, and ``odometry.txt`` only when no curated path length exists and
the file is small.

Nothing here is rig-specific -- wheel speeds stay in the file's own rad/s,
never divided through a wheel radius that belongs to one robot. Each card
degrades alone, and one left with nothing to say is dropped, not shown empty.
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import geo
from .build import SceneAccess
from .index_page import (
    THUMB_FRACTIONS,
    THUMB_SLOTS,
    coverage_class,
    disk_parts,
    fmt_bytes,
    fmt_count,
    fmt_disk_rate,
    fmt_duration,
    segment_timeline,
    split_mark,
    stat,
    thumb_index,
    thumb_slots,
    validation_mark,
)
from .loaders import _scalar, read_camera_info, read_text, roll_pitch_from_gravity_align
from .probe import ARTEFACTS, DatasetRoot
from .signals import signal_charts

logger = logging.getLogger("dataset_tools.viewer.scene_page")

#: A card with fewer entries than this is not worth a panel of its own.
MIN_CARD_STATS = 2

#: A timeline block narrower than this fraction of the scene cannot be hit with
#: a pointer, so a very short segment is drawn -- and clicked -- at this width
#: rather than at its own. It overstates nothing else: the tooltip and the
#: segment page both carry the real duration.
MIN_MARK_FRAC = 0.006

#: Rows read out of a proprioception array. The array is memory-mapped and
#: strided, so this bounds the work rather than the bytes the kernel faults in.
PROPRIO_SAMPLES = 4000

#: A wheel counts as turning above this angular speed (rad/s) -- fast enough to
#: separate driving from a robot sitting still with noise on the encoders.
TURNING_RAD_S = 0.1

#: ``odometry.txt`` is parsed to fill in a path length only when it is at most
#: this large. A curated root supplies the number and never gets here.
ODOMETRY_READ_LIMIT = 8_000_000

#: A frame counts as moving above this speed (m/s), matching the atlas's own
#: ``moving_frac`` definition closely enough to stand in for it.
MOVING_M_S = 0.1


# --------------------------------------------------------------------------
# series
# --------------------------------------------------------------------------


@dataclass
class Series:
    """A timestamped ``(T, K)`` array, sampled down to a readable number of rows."""

    rows: np.ndarray
    n: int
    t0: float
    t1: float

    @property
    def duration(self) -> float:
        return max(0.0, self.t1 - self.t0)

    @property
    def rate(self) -> float | None:
        return self.n / self.duration if self.duration > 0 else None

    def mean_abs(self, cols) -> float:
        return float(np.nanmean(np.abs(self.rows[:, cols])))

    def max_abs(self, cols) -> float:
        return float(np.nanmax(np.abs(self.rows[:, cols])))


def read_series(path: Path, max_rows: int = PROPRIO_SAMPLES) -> Series | None:
    """Evenly-spaced rows of a timestamped ``.npy``, read through a memory map.

    The header gives the true row count and the first and last row give the
    time span, so the reported rate is the file's real rate and not the sampled
    one. Column 0 is assumed to be the timestamp, as every proprioception array
    an extraction writes has it there.
    """
    try:
        arr = np.load(path, mmap_mode="r")
    except (OSError, ValueError) as exc:
        logger.warning("cannot read series %s: %s", path, exc)
        return None
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        return None
    step = max(1, arr.shape[0] // max_rows)
    rows = np.asarray(arr[::step], dtype=np.float64)
    return Series(rows=rows, n=int(arr.shape[0]), t0=float(arr[0, 0]), t1=float(arr[-1, 0]))


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

    One header read, no cloud data: enough to say how dense a scan is on a root
    that has no curated ``pts_mean``.
    """
    if not access.probe.has("clouds") or not access.probe.n_frames:
        return None
    index = access.probe.n_frames // 2
    path = access.frame_path("clouds", index)
    if not path.is_file():
        available = sorted(access.available_frames("clouds"))
        if not available:
            return None
        index = available[len(available) // 2]
        path = access.frame_path("clouds", index)
    try:
        shape = np.load(path, mmap_mode="r").shape
    except (OSError, ValueError):
        return None
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

    images, clouds = p.counts.get("images"), p.counts.get("clouds")
    if images is not None or clouds is not None:
        stats.append(
            stat(
                "images · clouds",
                f"{fmt_count(images) if images is not None else '—'}"
                f" · {fmt_count(clouds) if clouds is not None else '—'}",
            )
        )

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

    Only ever called when the atlas did not supply the numbers. Distances are
    accumulated **inside** a segment and never across one: consecutive segments
    are separately anchored, so the jump between them is not motion.

    ``bounds`` overrides the segment list with explicit ``(start, end)`` pairs.
    The caller uses it to clip to the frame window: ground truth recorded before
    the camera and lidar came up is real odometry, but it is not distance this
    dataset covers, and counting it makes the total disagree with the frames.
    """
    path = access.dir / "odometry.txt"
    try:
        if path.stat().st_size > ODOMETRY_READ_LIMIT:
            return None
    except OSError:
        return None
    odom = access.odometry
    if odom is None:
        return None
    ts, poses = odom
    if len(ts) < 2:
        return None
    if bounds is None:
        bounds = [(s.start_ts, s.end_ts) for s in access.segments] or [(ts[0], ts[-1])]
    total, moving, count = 0.0, 0, 0
    for lo, hi in bounds:
        take = (ts >= lo) & (ts <= hi)
        if take.sum() < 2:
            continue
        xyz, stamps = poses[take, :3, 3], ts[take]
        steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        dt = np.diff(stamps)
        total += float(steps.sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            speed = np.where(dt > 0, steps / np.where(dt > 0, dt, 1.0), 0.0)
        moving += int((speed > MOVING_M_S).sum())
        count += len(speed)
    if not count:
        return None
    return total, moving / count


def timeline_marks(access: SceneAccess) -> list[dict[str, Any]]:
    """One clickable block per ground-truth segment, in scene-time fractions.

    Built from the segment file rather than from the probe's flat span pairs,
    because a block is only worth clicking if it knows *which* segment it is:
    the number carried here is the one in ``odometry_segments.txt``, which is
    what the segment page resolves and what the window keys already name.

    Empty when the scene has no readable segments or no time span to scale them
    against; the timeline then falls back to the plain spans, which is all a
    curated-only scene ever had.
    """
    from .windows import describe_reason

    p, segments = access.probe, access.segments
    if not segments or p.start_ts is None or not p.duration or p.duration <= 0:
        return []
    base, span = p.start_ts, p.duration
    marks: list[dict[str, Any]] = []
    for seg in segments:
        lo = min(1.0, max(0.0, (seg.start_ts - base) / span))
        hi = min(1.0, max(0.0, (seg.end_ts - base) / span))
        width = min(max(hi - lo, MIN_MARK_FRAC), 1.0 - lo)
        # Why a segment *ended* is not what a reader wants off a hover: it is a
        # property of the boundary, it is the same sentence on block after
        # block, and the segment page states it anyway. Where the poses came
        # from stays -- that is what makes an amber block different.
        described = describe_reason(seg.reason)
        why = described.split("; ends because", 1)[0] if described.startswith("filled") else ""
        title = f"segment {seg.index} \u2014 {fmt_duration(seg.duration)}, {fmt_count(seg.n_poses)} GT poses"
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

    The segment file is read live and preferred over the curated copy: the file
    is the truth, and a scene may have been re-filtered since the atlas was
    written. Only what cannot be recomputed cheaply -- the disagreement
    percentiles against the robot's own odometry, the divergence flag count,
    the previous root's coverage -- is taken from the atlas.
    """
    p = access.probe
    segments = access.segments
    stats: list[dict[str, Any]] = []
    notes: list[str] = []

    # The same source the index badge uses, so one scene does not carry two
    # different coverage numbers on two pages of the same viewer.
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
            # Only when the curated count is of the *same* poses: a scene whose
            # gaps were filled from another root has more poses in its segments
            # than this root's own run produced, and "of N raw" would be a lie.
            raw = a.get("poses_unf") or a.get("poses_raw")
            sub = f"of {fmt_count(raw)} raw" if raw and raw >= poses else None
            stats.append(stat("poses", fmt_count(poses), sub))
        from .windows import describe_reason

        reasons = sorted(p.segment_reasons.items(), key=lambda kv: -kv[1])
        # A filled stretch is not a reason a segment *ended*; it is where its
        # poses came from, and the fill note below says so. Mixing the two read
        # as "segments end because filled by replaying the bag".
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
        # Say which fill this root actually holds rather than naming one: v1.3's
        # holes were filled by replaying their bags, v1.2's from v1.1, and a
        # root can carry both.
        from .windows import describe_marks, describe_reason

        filled = [s for s in segments if "fill" in (s.reason or "").lower()]
        # The fill sentence, then what was done to the poses afterwards as a
        # count -- a scene whose fills were refined one at a time would
        # otherwise read the same sentence twice, once per tail.
        sources = sorted({describe_reason((s.reason or "").split("+")[0]) for s in filled})
        mark_counts: dict[str, int] = {}
        for s in filled:
            for m in describe_marks(s.reason or "", brief=True):
                mark_counts[m] = mark_counts.get(m, 0) + 1
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
                # One block per segment, each knowing which segment it is. The
                # flat spans above stay: they are what a scene with no readable
                # segment file (a curated-only root) has to be drawn from, and
                # they are what the index card draws.
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
        # An all-zero coefficient vector is not the same fact as no vector at
        # all: the camera declared a model and then reported no distortion.
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


def proprioception_card(access: SceneAccess) -> dict[str, Any] | None:
    """What the robot reported about itself: wheels, commands, attitude."""
    p = access.probe
    stats: list[dict[str, Any]] = []
    notes: list[str] = []

    js = read_series(access.dir / "joint_states.npy") if p.has("joint_states") else None
    if js is not None and js.rows.shape[1] >= 10:
        vel = slice(4, 7)
        stats.append(stat("wheel speed", f"{js.mean_abs(vel):.2f} rad/s", f"mean |ω|, max {js.max_abs(vel):.1f}"))
        turning = float((np.abs(js.rows[:, vel]) > TURNING_RAD_S).any(axis=1).mean())
        stats.append(stat("wheels turning", f"{turning * 100:.0f}%", f"any |ω| > {TURNING_RAD_S} rad/s"))
        stats.append(
            stat("wheel effort", f"{js.mean_abs(slice(7, 10)):.1f}", f"mean, max {js.max_abs(slice(7, 10)):.0f}")
        )
    elif js is not None:
        notes.append(f"joint_states.npy is {js.rows.shape[1]} columns wide, not the expected 10.")
    if js is not None:
        stats.append(
            stat("joint states", f"{fmt_count(js.n)} × {js.rows.shape[1]}", f"{js.rate:.1f} Hz" if js.rate else None)
        )

    cv = read_series(access.dir / "cmd_vel.npy") if p.has("cmd_vel") else None
    if cv is not None and cv.rows.shape[1] >= 7:
        stats.append(stat("commanded speed", f"{cv.mean_abs(1):.2f} m/s", f"mean |v|, |ω| {cv.mean_abs(6):.2f} rad/s"))
    if cv is not None:
        stats.append(
            stat("cmd_vel", f"{fmt_count(cv.n)} × {cv.rows.shape[1]}", f"{cv.rate:.1f} Hz" if cv.rate else None)
        )

    sp = read_series(access.dir / "joint_setpoints.npy") if p.has("joint_setpoints") else None
    if sp is not None and sp.rows.shape[1] >= 7:
        stats.append(
            stat(
                "commanded wheels",
                f"{sp.mean_abs(slice(4, 7)):.2f} rad/s",
                f"setpoints, {sp.rate:.1f} Hz" if sp.rate else "setpoints",
            )
        )

    ga = access.gravity_align
    if ga is not None and np.ndim(ga) == 3:
        roll, pitch = roll_pitch_from_gravity_align(np.asarray(ga))
        roll, pitch = np.degrees(np.abs(roll)), np.degrees(np.abs(pitch))
        stats.append(
            stat(
                "roll · pitch",
                f"{roll.mean():.1f}° · {pitch.mean():.1f}°",
                f"mean, max {roll.max():.0f}° · {pitch.max():.0f}°",
            )
        )

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

    Its second comment line is ``time_align.Profile.summary`` -- "no correction
    (why)" or "<shape> from K/N windows (note)" -- and its rows are the windows
    measured, ``t delta gain depth``. A constant given on the command line has
    a summary and no rows.
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
    counts = re.findall(r"(\d+)/(\d+) (poses|frames|further frames)", " ".join(comments))
    by = {what: (int(a), int(b)) for a, b, what in counts}
    poses, frames, unsupported = by.get("poses"), by.get("frames"), by.get("further frames")
    if poses is None or frames is None or unsupported is None:
        raise ValueError("gnss_guard.txt has no counts in its header")
    stats = [
        stat("fabricated stretches", len(rows), "removed by the GNSS guard"),
        stat("cost", f"{poses[0]:,} poses · {frames[0]:,} frames", f"of {poses[1]:,} · {frames[1]:,}"),
        stat("no pose under them", f"{unsupported[0]:,} frames", f"of {unsupported[1]:,}, dropped as well"),
    ]
    reasons = sorted({r.split()[2] for r in rows if len(r.split()) > 2})
    note = f"Stretches caught as {', '.join(reasons)}." if reasons else None
    return stats, note


def _tf_pins_stats(access: SceneAccess) -> list[dict[str, Any]]:
    """How far each pinned transform sat from what the bag itself published.

    Rows come in pin/bag pairs per edge; the numbers are recomputed from the
    rows through ``tf_pins.Record`` rather than read off the report's own
    comment, so the card and the extractor cannot disagree about a distance.
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
        pin = tf_pins.Pin(parent, child, have["pin"][0], have["pin"][1], "")
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

    Three report files, each optional, each read on its own: a malformed one
    costs the card its part and a warning, never the page. ``None`` when the
    scene carries none of them (a root extracted before they existed).
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

    The report is ``<root>/validation.csv``, which carries the status, the
    numbers and counts of findings -- never the messages, which only the
    validator's own output has. The card says so instead of pretending.
    """
    row = access.root.validation_scene(access.scene_id)
    mark = validation_mark(row)
    if mark is None:
        return None
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
    note = "Counts only: the file holds how many findings, not what they said. "
    note += (
        f"For the messages, run helhest-validate {access.scene_id} on the root."
        if mark["cls"] != "yes"
        else "Nothing was found."
    )
    return {
        "kind": "validation",
        "title": "Validation",
        "badge": mark["status"],
        "badge_class": {"yes": "hi", "partial": "lo", "no": "lo"}[mark["cls"]],
        "badge_title": "the status helhest-validate gave this scene",
        "stats": stats,
        "note": note,
    }


def _gps_src(scene_id: str, segment, frame: int | None) -> str:
    """The endpoint one map fetches its track from."""
    query = []
    if segment is not None:
        query.append(f"segment={segment.index}")
    if frame is not None:
        query.append(f"frame={frame}")
    return f"/api/scene/{scene_id}/gps.json" + ("?" + "&".join(query) if query else "")


def map_card(access: SceneAccess, segment=None, frame: int | None = None) -> dict[str, Any] | None:
    """Where on the Earth this was recorded -- ``None`` when nothing says.

    A scene with no ``gps.npy`` gets no card, and the template then renders no
    map, no grid and no Leaflet at all: a root from a rig without GNSS looks
    exactly as it did before any of this existed.

    ``segment`` narrows the numbers (not just the drawn track) to one segment's
    time span, which is what makes the same card honest on the segment page.
    ``frame`` marks where one frame was, for the frame page; the numbers then
    still describe the whole track the marker sits on.

    The fixes are the receiver's own position -- on the new rig the fusion
    point, a few tens of centimetres from ``base_link``, well inside its own
    sigma -- and they are not ground truth: the poses in ``odometry.txt`` are
    anchored per segment and say nothing about the Earth.
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
            stat("this frame", f"{here[0]:.6f}, {here[1]:.6f}" if here else "—", "latitude, longitude")
            if here
            else stat("this frame", "—", "no fix within a second of it")
        )
    span = float(track[-1, geo.T] - track[0, geo.T])
    rate = f"{len(track) / span:.1f} Hz" if span > 0 else None
    stats.append(stat("fixes", fmt_count(len(track)), rate))
    stats.append(stat("track", f"{geo.track_length_m(track):.0f} m", "along the fixes"))
    ew, ns = geo.span_m(track)
    stats.append(stat("extent", f"{ew:.0f} × {ns:.0f} m", "east–west × north–south"))
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
        "note": "GNSS as the receiver reported it — where the robot was on the Earth, not ground truth.",
    }


# --------------------------------------------------------------------------
# the page
# --------------------------------------------------------------------------


def hero_thumbs(access: SceneAccess, a: dict) -> list[dict[str, Any]]:
    """The same three frames the index card shows, at the top of the scene page.

    Same endpoint and same resolver as the index, so a scene looks the same in
    both places; here each one also links to the frame it was cut from. The link
    and the picture come from one ``thumb_index`` call: a curated thumbnail cut
    at some other frame than the default fraction still captions itself right.
    """
    p = access.probe
    stored = thumb_slots(a)
    slots = stored or (list(THUMB_SLOTS) if p.has("images") and p.n_frames else [])
    ts = access.frame_ts if slots else None
    t0 = float(ts[0]) if ts is not None and len(ts) else None

    out = []
    for slot in slots:
        index = thumb_index(a, slot, p.n_frames)
        at = None
        if ts is not None and t0 is not None and index < len(ts):
            at = f"{float(ts[index]) - t0:.0f} s"
        out.append(
            {
                "url": f"/api/scene/{p.scene_id}/thumb/{slot}.jpg",
                "href": f"/scene/{p.scene_id}/frame/{index}" if p.has("images") else None,
                "alt": f"scene {p.scene_id} frame {index}",
                "label": f"{THUMB_FRACTIONS[slot] * 100:.0f}%",
                "sub": f"frame {index}" + (f" · {at}" if at else ""),
            }
        )
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
        each = [
            f"{name.replace('_', ' ')} {fmt_bytes(mods[name]['bytes'] / mods[name]['files'])}"
            for name in ("images", "clouds")
            if (mods.get(name) or {}).get("files")
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
    sizes = {name: m.get("bytes") for name, m in (d.get("modalities") or {}).items() if name != "other"}
    sizes.update(d.get("files") or {})
    return {k: v for k, v in sizes.items() if isinstance(v, int)}


def file_entries(access: SceneAccess) -> list[dict[str, Any]]:
    """The file inventory, each entry carrying what it is and what it is for.

    The chips used to say only a path and a count, with the rest in a tooltip
    that a touch device never shows at all. Everything a reader needs about one
    artefact is assembled here instead, so the page can open it in place: no
    extra request, and nothing read off disk that the probe did not already
    have.
    """
    p = access.probe
    sizes = file_sizes(access)
    entries: list[dict[str, Any]] = []
    for art in ARTEFACTS:
        count = p.counts.get(art.key)
        entries.append(
            {
                "key": art.key,
                "label": art.label,
                "relpath": art.relpath,
                "kind": "directory" if art.kind == "dir" else "file",
                "present": p.has(art.key),
                "count": count,
                "count_unit": art.count_unit,
                "size": fmt_bytes(sizes[art.relpath]) if p.has(art.key) and art.relpath in sizes else None,
                "shape": art.description,
                "purpose": art.purpose,
            }
        )
    return entries


def build_scene_page(root: DatasetRoot, access: SceneAccess) -> dict[str, Any]:
    """Thumbnails and the four cards, cached on the scene handle.

    Cached because a page view costs a few megabytes of small-array reads on
    network storage, and switching tabs on the same scene must not pay it twice.
    """

    def work() -> dict[str, Any]:
        a = root.atlas_scene(access.scene_id)
        info = read_camera_info(access.dir / "camera_info.yaml") if access.probe.has("camera_info") else None
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
            # Which side of the train/val split this scene is on. The same mark
            # the index card is rimmed with, from the same function, so the two
            # pages cannot disagree about a held-out scene.
            "split": split_mark(root, access.scene_id),
            "camera_info": info,
            # Not one of the four: it sits beside the file inventory, and it is
            # absent -- not empty -- on a scene without GNSS.
            "map": map_card(access),
            # Between the inventory and the map, likewise absent on a root
            # with no disk_usage.json.
            "disk": disk_card(access),
            # The inventory strip: one entry per artefact the viewer knows,
            # present or not, each with the prose the page opens under it.
            "files": file_entries(access),
            # The same arrays over time, on the frame clock's own span.
            "signals": scene_signals(access),
        }

    return access.cached("scene_page", work)


def scene_signals(access: SceneAccess) -> dict[str, Any] | None:
    """The signal charts over the whole recording, segments shaded behind."""
    ts = access.frame_ts
    if ts is None or len(ts) < 2:
        return None
    return signal_charts(access, float(ts[0]), float(ts[-1]))
