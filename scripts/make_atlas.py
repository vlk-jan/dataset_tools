#!/usr/bin/env python3
"""Generate ``<root>/atlas/atlas.json`` and its thumbnails from the root itself.

    make_atlas.py <root> [--check] [--materialize] [--size]

``atlas/curation.json`` is hand-edited prose (titles, groups, descriptions,
thumbnail frames); ``atlas/atlas.json`` is generated: that prose plus every
number computed from this root, using the same code the viewer pages use, so
the two cannot disagree. Rerun after anything changes a scene's odometry,
frames or clouds; ``--check`` reports staleness without writing.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from dataset_tools.gt_gaps import window_errors as scan_windows
from dataset_tools.gt_io import _replace_atomically, read_trajectory
from dataset_tools.viewer import render
from dataset_tools.viewer.build import SceneAccess
from dataset_tools.viewer.index_page import (
    THUMB_SLOTS,
    fmt_count,
    fmt_duration,
    thumb_index,
)
from dataset_tools.viewer.probe import DatasetRoot, SceneProbe
from dataset_tools.viewer.scene_page import path_and_motion

logger = logging.getLogger("make_atlas")

ATLAS_SCHEMA = "dataset_tools.viewer/atlas@2"

#: Same as the live thumbnail endpoint (``app.api_scene_thumb``).
THUMB_WIDTH = 384
THUMB_QUALITY = 80

#: Window length of the local-consistency metric, in seconds.
WINDOW_S = 2.0

#: Clouds sampled for ``pts_mean`` when ``regen/validation.csv`` has no count.
CLOUD_SAMPLES = 5


# --------------------------------------------------------------------------
# small readers over <scene>/regen/
# --------------------------------------------------------------------------


def read_kv(path: Path) -> dict[str, str]:
    """``key=value`` lines (``regen/filter_stats.txt``)."""
    out: dict[str, str] = {}
    try:
        text = path.read_text()
    except OSError:
        return out
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            out[key.strip()] = value.strip()
    return out


def count_rows(path: Path) -> int | None:
    """Non-comment, non-blank lines, or None when the file is absent."""
    try:
        with open(path) as fh:
            return sum(1 for line in fh if line.strip() and not line.startswith("#"))
    except OSError:
        return None


def read_validation(path: Path) -> dict[str, str]:
    """The scene's row of ``regen/validation.csv``.

    Only its cloud/frame columns are trusted: the pose ones may come from another odometry run.
    """
    try:
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                return row
    except OSError:
        pass
    return {}


def as_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


# --------------------------------------------------------------------------
# the computed half
# --------------------------------------------------------------------------


def windows(t, T, rt, rT, segs, w):
    """Translation error of every ``w``-second window of ``T`` against ``rT``, inside one of ``segs``."""
    return np.array(
        [
            err
            for i, j, err in scan_windows(t, T, rt, rT, w)
            if not segs or any(a <= t[i] and t[j] <= b for a, b in segs)
        ]
    )


def window_errors(access: SceneAccess) -> tuple[float, float] | None:
    """``(median, p95)`` of the 2 s window error against the live /odom, in m.

    Only within segments. ``regen/odom_ref.txt`` (~100 Hz) is the reference;
    frame-rate ``transforms.npy`` is too coarse for a 2 s window.
    """
    ref_path = access.dir / "regen" / "odom_ref.txt"
    odom = access.odometry
    if odom is None or not ref_path.is_file():
        return None
    ts, poses = odom
    if len(ts) < 2:
        return None
    try:
        rt, rT = read_trajectory(ref_path)
    except (OSError, ValueError) as exc:
        logger.warning("%s: cannot read odom_ref.txt: %s", access.scene_id, exc)
        return None
    if not len(rt):
        return None
    bounds = [(s.start_ts, s.end_ts) for s in access.segments] or None
    errs = windows(np.asarray(ts), np.asarray(poses), rt, rT, bounds, WINDOW_S)
    if not len(errs):
        return None
    return float(np.median(errs)), float(np.percentile(errs, 95))


def sampled_points(access: SceneAccess) -> int | None:
    """Mean points per cloud from a few ``.npy`` headers (no ``validation.csv`` fallback)."""
    probe = access.probe
    if not probe.has("clouds") or not probe.n_frames:
        return None
    available = sorted(access.available_frames("clouds"))
    if not available:
        return None
    step = max(1, len(available) // CLOUD_SAMPLES)
    counts = []
    for index in available[::step][:CLOUD_SAMPLES]:
        try:
            shape = np.load(access.frame_path("clouds", index), mmap_mode="r").shape
        except (OSError, ValueError):
            continue
        if shape:
            counts.append(int(shape[0]))
    return round(statistics.mean(counts)) if counts else None


def scene_date(probe: SceneProbe, csv_row: dict) -> str | None:
    """``YYYY-MM-DD HH:MM`` UTC of the scene's first frame."""
    stamp = probe.start_ts or as_float(csv_row.get("start_time"))
    if not stamp or stamp < 1e8:  # a synthesised 0-based clock, not wall time
        return None
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.gmtime(stamp))
    except (OverflowError, OSError, ValueError):
        return None


def frame_window(probe: SceneProbe) -> tuple[float, float] | None:
    """The scene's first and last frame stamp, or None if it has no clock."""
    if probe.start_ts is None or probe.end_ts is None or probe.end_ts <= probe.start_ts:
        return None
    return float(probe.start_ts), float(probe.end_ts)


def clip(span: tuple[float, float], win: tuple[float, float] | None) -> float:
    """How much of ``span`` lies inside ``win``; the whole of it when unwindowed."""
    if win is None:
        return max(0.0, span[1] - span[0])
    return max(0.0, min(span[1], win[1]) - max(span[0], win[0]))


def scene_entry(
    root: DatasetRoot,
    access: SceneAccess,
    curated: dict,
    bags: dict[str, dict],
) -> dict[str, Any]:
    """One scene's atlas entry: curated prose first, then everything computed."""
    probe = access.probe
    regen = access.dir / "regen"
    filter_stats = read_kv(regen / "filter_stats.txt")
    validation = read_validation(regen / "validation.csv")
    bag = bags.get(probe.scene_id, {})

    entry: dict[str, Any] = {}

    date = scene_date(probe, root.csv_rows().get(probe.scene_id, {}))
    if date:
        entry["date"] = date
    if bag.get("session"):
        entry["session"] = bag["session"]
    if bag.get("bag_gb") is not None:
        entry["bag_gb"] = bag["bag_gb"]

    entry["frames"] = probe.n_frames
    if probe.duration:
        entry["dur_s"] = round(probe.duration, 1)
    for kind in ("images", "clouds"):
        if probe.counts.get(kind) is not None:
            entry[kind] = probe.counts[kind]
    points = as_int(validation.get("points_mean")) or sampled_points(access)
    if points:
        entry["pts_mean"] = points

    # GT stats are clipped to the frame window: odometry can start well before
    # the first frame (Odin boots after the Fixposition), and "with GT" must not
    # exceed the scene duration.
    win = frame_window(probe)
    segments = access.segments
    odom = access.odometry
    if odom is not None:
        ts = odom[0]
        inside = ts if win is None else ts[(ts >= win[0]) & (ts <= win[1])]
        entry["poses"] = int(len(inside))
    poses_unf = as_int(filter_stats.get("poses_in"))
    if poses_unf:
        entry["poses_unf"] = poses_unf
    raw = count_rows(regen / "odometry_imu_raw.txt")
    if raw:
        entry["poses_raw"] = raw

    entry["segments"] = len(segments) or probe.n_segments
    lens = sorted((v for s in segments if (v := clip((s.start_ts, s.end_ts), win)) > 0.0), reverse=True)
    if lens:
        entry["seg_lens"] = [round(v, 1) for v in lens]
        entry["seg_med_s"] = round(float(statistics.median(lens)), 1)
        entry["gt_span_s"] = round(float(sum(lens)), 1)

    flags = count_rows(regen / "odometry_flags.txt")
    if flags is not None:
        entry["flags"] = flags

    clipped_bounds = [(max(s.start_ts, win[0]), min(s.end_ts, win[1])) for s in segments] if win and segments else None
    if clipped_bounds is not None:
        clipped_bounds = [(a, b) for a, b in clipped_bounds if b > a]
    motion = path_and_motion(access, clipped_bounds)
    if motion:
        entry["path_m"] = round(motion[0], 1)
        entry["moving_frac"] = round(motion[1], 3)

    if probe.segment_coverage is not None:
        entry["cov"] = round(probe.segment_coverage, 4)
    errors = window_errors(access)
    if errors:
        entry["win_med"] = round(errors[0], 4)
        entry["win_p95"] = round(errors[1], 4)

    entry["seg_spans"] = [[round(v, 4) for v in span] for span in probe.segment_spans]
    entry["fill_spans"] = [[round(v, 4) for v in span] for span in probe.fill_spans]
    entry["fill_segments"] = len(probe.fill_spans)

    if curated.get("description"):
        entry["description"] = curated["description"]
    thumbs = curated_thumbs(curated, probe)
    if thumbs:
        entry["thumbs"] = thumbs
    return entry


def curated_thumbs(curated: dict, probe: SceneProbe) -> dict[str, int]:
    """``{slot: frame index}``; the index defines the thumbnail, the JPEG only caches it."""
    if not probe.has("images") or not probe.n_frames:
        return {}
    return {slot: thumb_index(curated, slot, probe.n_frames) for slot in THUMB_SLOTS}


def group_range(scenes: list[str]) -> str:
    """``0000–0011 · 12 scenes`` for a group's header."""
    count = f"{len(scenes)} scene{'s' if len(scenes) != 1 else ''}"
    if not scenes:
        return count
    span = scenes[0] if len(scenes) == 1 else f"{min(scenes)}–{max(scenes)}"
    return f"{span} · {count}"


def totals(entries: dict[str, dict], size_gb: float | None, previous: list | None = None) -> list[dict[str, Any]]:
    """The masthead tiles, summed over this root's scenes.

    Size is slow to compute, so only under ``--size``; otherwise carried over from ``previous``.
    """
    values = list(entries.values())
    frames = sum(e.get("frames") or 0 for e in values)
    gt_span = sum(e.get("gt_span_s") or 0.0 for e in values)
    segments = sum(e.get("segments") or 0 for e in values)
    poses = sum(e.get("poses") or 0 for e in values)
    path_m = sum(e.get("path_m") or 0.0 for e in values)
    # Frame-weighted, not the mean of per-scene fractions.
    weighted = sum((e.get("cov") or 0.0) * (e.get("frames") or 0) for e in values)

    out = [
        {"label": "scenes", "value": str(len(values)), "sub": None},
        {"label": "duration", "value": fmt_duration(sum(e.get("dur_s") or 0.0 for e in values)), "sub": None},
        {"label": "frames", "value": fmt_count(frames), "sub": None},
    ]
    if gt_span:
        out.append(
            {
                "label": "with GT",
                "value": fmt_duration(gt_span),
                "sub": f"in {fmt_count(segments)} segments" if segments else None,
            }
        )
    if poses:
        out.append({"label": "GT poses", "value": fmt_count(poses), "sub": None})
    if path_m:
        out.append({"label": "path with GT", "value": f"{path_m / 1000:.1f} km", "sub": None})
    if frames:
        out.append({"label": "GT coverage", "value": f"{weighted / frames * 100:.0f}%", "sub": None})
    if size_gb:
        out.append({"label": "size", "value": f"{size_gb:.0f} GB", "sub": None})
    else:
        out.extend(t for t in previous or () if isinstance(t, dict) and t.get("label") == "size")
    return out


def root_size_gb(root: Path) -> float:
    """Size of the root in GB, following symlinks (a symlinked scene is still
    part of what the root ships), each real directory counted once."""
    seen: set[str] = set()
    total = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in seen:
            dirnames[:] = []
            continue
        seen.add(real)
        for name in filenames:
            try:
                total += os.stat(os.path.join(dirpath, name)).st_size
            except OSError:
                continue
    return total / 1e9


# --------------------------------------------------------------------------
# thumbnails
# --------------------------------------------------------------------------


def thumb_path(atlas_dir: Path, spec: dict, scene_id: str, slot: str) -> Path:
    pattern = spec.get("pattern") or "{scene}{slot}.jpg"
    return atlas_dir / (spec.get("dir") or "thumbs") / pattern.format(scene=scene_id, slot=slot)


def stored_width(path: Path) -> int | None:
    try:
        from PIL import Image

        with Image.open(path) as img:
            return img.width
    except Exception:  # noqa: BLE001 - a corrupt or missing file just gets rewritten
        return None


def write_thumbs(
    access: SceneAccess,
    atlas_dir: Path,
    spec: dict,
    indexes: dict[str, int],
    previous: dict,
) -> tuple[int, int]:
    """Render this scene's thumbnails. Returns ``(written, skipped)``.

    Skips a thumbnail whose frame and width are unchanged since the last run.
    """
    old = previous.get("thumbs") if isinstance(previous.get("thumbs"), dict) else {}
    written = skipped = 0
    for slot, index in indexes.items():
        out = thumb_path(atlas_dir, spec, access.scene_id, slot)
        if out.is_file() and old.get(slot) == index and stored_width(out) == THUMB_WIDTH:
            skipped += 1
            continue
        source = access.frame_path("images", index)
        if not source.is_file():
            available = sorted(access.available_frames("images"))
            if not available:
                continue
            source = access.frame_path("images", min(available, key=lambda i: abs(i - index)))
        try:
            data = render.render_camera_image(source, THUMB_WIDTH, quality=THUMB_QUALITY)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s%s: cannot render %s: %s", access.scene_id, slot, source, exc)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        written += 1
    return written, skipped


# --------------------------------------------------------------------------
# curation
# --------------------------------------------------------------------------


def read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        if path.exists():
            raise SystemExit(f"cannot read {path}: {exc}") from exc
        return {}
    return data if isinstance(data, dict) else {}


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------


def read_bags(path: Path | None) -> dict[str, dict]:
    """``{scene: {session, bag_gb}}`` from a bag list CSV; empty without one."""
    if not path or not path.is_file():
        return {}
    out: dict[str, dict] = {}
    try:
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                scene = (row.get("scene") or "").strip()
                if not scene:
                    continue
                bag_dir = (row.get("bag_dir_on_subtdata") or "").strip()
                out[scene] = {
                    "session": Path(bag_dir).parent.name if bag_dir else None,
                    "bag_gb": as_float(row.get("bag_size_gb")),
                }
    except (OSError, csv.Error) as exc:
        logger.warning("cannot read %s: %s", path, exc)
    return out


def build_atlas(root: DatasetRoot, args, curation: dict, previous: dict) -> dict:
    """The whole ``atlas.json``: curated prose, then this root's own numbers."""
    probes = [p for p in root.probe_all() if not args.scenes or p.scene_id in args.scenes]
    bags = read_bags(args.bags)
    curated_scenes = curation.get("scenes") or {}
    old_scenes = previous.get("scenes") or {}
    spec = {
        "dir": "thumbs",
        "slots": list(THUMB_SLOTS),
        "pattern": "{scene}{slot}.jpg",
        "width": THUMB_WIDTH,
        "note": (
            f"{THUMB_WIDTH} px wide jpeg of the frame each slot names; the viewer "
            "renders that same frame live when the file is absent"
        ),
    }

    entries: dict[str, dict] = {}
    written = skipped = 0
    for probe in probes:
        access = SceneAccess(root, probe.scene_id)
        curated = curated_scenes.get(probe.scene_id) or {}
        entry = scene_entry(root, access, curated, bags)
        entries[probe.scene_id] = entry
        if not args.no_thumbs and not args.check and entry.get("thumbs"):
            w, s = write_thumbs(access, root.atlas_dir, spec, entry["thumbs"], old_scenes.get(probe.scene_id, {}))
            written += w
            skipped += s
        logger.info("%s  %s", probe.scene_id, summarise(entry))
    if written or skipped:
        logger.info("thumbnails: %d written, %d already current", written, skipped)

    groups = []
    for group in curation.get("groups") or []:
        scenes = [s for s in group.get("scenes") or [] if s in entries]
        if not scenes:
            continue
        groups.append(
            {
                "title": group.get("title"),
                "range": group_range(scenes),
                "blurb": group.get("blurb"),
                "scenes": scenes,
            }
        )

    manifest = root.manifest() or {}
    size_gb = root_size_gb(root.path) if args.size else None
    atlas = {
        "schema": ATLAS_SCHEMA,
        "generated": time.strftime("%Y-%m-%d"),
        "dataset_version": manifest.get("dataset_version"),
        "root": str(root.path),
        "source": (
            "Generated by dataset_tools scripts/make_atlas.py from this root: counts, "
            "spans and coverage from the scene probe, path and motion from odometry.txt, "
            "pose and flag counts from <scene>/regen/, 2 s window error against "
            "<scene>/regen/odom_ref.txt. Prose and thumbnail frames come from "
            "atlas/curation.json, the only hand-edited file."
        ),
        "note": (
            "Optional. A dataset root without atlas/ still renders in the viewer: "
            "descriptions are omitted, thumbnails fall back to live frames, and the "
            "stats reduce to what the probe supplies. Do not edit -- regenerate."
        ),
    }
    for key in ("title", "kicker", "intro", "root_note", "footer"):
        if curation.get(key):
            atlas[key] = curation[key]
    atlas["totals"] = totals(entries, size_gb, previous.get("totals"))
    atlas["thumbs"] = spec
    atlas["groups"] = groups
    atlas["scenes"] = entries
    return atlas


def summarise(entry: dict) -> str:
    bits = [f"{entry.get('frames', 0)} frames"]
    if entry.get("segments"):
        bits.append(f"{entry['segments']} segments")
    if entry.get("cov") is not None:
        bits.append(f"{entry['cov'] * 100:.0f}% GT")
    if entry.get("path_m") is not None:
        bits.append(f"{entry['path_m']:.0f} m")
    if entry.get("win_med") is not None:
        bits.append(f"2 s err {entry['win_med']:.2f} m")
    return " · ".join(bits)


def diff_atlas(new: dict, old: dict, only: list[str] | None = None) -> list[str]:
    """Human-readable differences, root keys first then per scene.

    ``only`` limits it to those scenes and skips the root keys (totals would differ).
    """
    out: list[str] = []
    skip = {"generated", "scenes", "source", "note"}
    for key in sorted(set(new) | set(old)) if not only else ():
        if key in skip:
            continue
        if new.get(key) != old.get(key):
            out.append(f"{key}: {old.get(key)!r} -> {new.get(key)!r}")
    new_scenes, old_scenes = new.get("scenes") or {}, old.get("scenes") or {}
    for scene in sorted(only if only else set(new_scenes) | set(old_scenes)):
        a, b = new_scenes.get(scene) or {}, old_scenes.get(scene) or {}
        changed = [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
        if changed:
            out.append(f"{scene}: " + ", ".join(f"{k} {b.get(k)!r} -> {a.get(k)!r}" for k in changed))
    return out


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _replace_atomically(path, lambda fh: fh.write(json.dumps(data, indent=1, ensure_ascii=False) + "\n"))


def resolve_thumbs_dir(atlas_dir: Path, materialize: bool) -> None:
    """The same refusal for a linked ``atlas/thumbs/``; ``--materialize`` copies the files across."""
    thumbs = atlas_dir / "thumbs"
    if not thumbs.is_symlink():
        return
    target = thumbs.resolve()
    if not materialize:
        raise SystemExit(
            f"{thumbs} is a symlink to {target}\n"
            "Generating here would overwrite that root's thumbnails. Re-run with --materialize "
            "to replace the link with a real directory (the thumbnails are copied across)."
        )
    thumbs.unlink()
    if target.is_dir():
        shutil.copytree(target, thumbs)
    else:
        thumbs.mkdir()
    logger.info("materialised %s (was a symlink to %s)", thumbs, target)


def resolve_atlas_dir(root: DatasetRoot, materialize: bool) -> None:
    """Refuse to write through a symlinked ``atlas/`` (it is another release's); ``--materialize`` breaks it."""
    atlas_dir = root.atlas_dir
    if not atlas_dir.is_symlink():
        resolve_thumbs_dir(atlas_dir, materialize)
        return
    target = atlas_dir.resolve()
    if not materialize:
        raise SystemExit(
            f"{atlas_dir} is a symlink to {target}\n"
            "Generating here would overwrite that root's atlas. Re-run with --materialize "
            "to replace the link with a real directory (curation.json is copied across)."
        )
    curation = read_json(target / "curation.json")
    atlas_dir.unlink()
    atlas_dir.mkdir()
    if curation:
        write_json(atlas_dir / "curation.json", curation)
    logger.info("materialised %s (was a symlink to %s)", atlas_dir, target)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="dataset root to generate an atlas for")
    ap.add_argument("--check", action="store_true", help="report what would change; write nothing")
    ap.add_argument("--materialize", action="store_true", help="replace a symlinked atlas/ with a real directory")
    ap.add_argument("--no-thumbs", action="store_true", help="do not render thumbnails")
    ap.add_argument("--size", action="store_true", help="walk the root for the size total (slow on network storage)")
    ap.add_argument(
        "--bags",
        type=Path,
        help="the root's bag list (docs/scene_bags.csv for v1, docs/v2_<n>_bags.csv for v2) for session/bag size; "
        "without it those two fields are left out",
    )
    ap.add_argument("--scenes", nargs="*", default=[], help="limit to these scene ids")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    root = DatasetRoot(args.root)

    curation = read_json(root.atlas_dir / "curation.json")
    if not curation:
        logger.warning(
            "no atlas/curation.json in this root: generating numbers only, with no prose. "
            "Write one (see docs/VIEWER.md) to add it."
        )
    previous = read_json(root.atlas_dir / "atlas.json")
    if not args.check:
        resolve_atlas_dir(root, args.materialize)

    atlas = build_atlas(root, args, curation, previous)

    if args.check:
        changes = diff_atlas(atlas, previous, args.scenes or None)
        if not changes:
            logger.info("atlas.json is current")
            return 0
        logger.info("atlas.json is out of date -- %d differences:", len(changes))
        for line in changes:
            logger.info("  %s", line)
        return 1

    write_json(root.atlas_dir / "atlas.json", atlas)
    logger.info("wrote %s (%d scenes)", root.atlas_dir / "atlas.json", len(atlas["scenes"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
