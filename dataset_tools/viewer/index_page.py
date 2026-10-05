"""What the scene index shows, assembled once per page render.

Two halves, not to be confused. *Probed*: listing a scene directory and reading
its timestamp and segment files, so it exists for any root, from any rig.
*Curated*: prose, groupings into outings, numbers that took a full pass over the
raw bags -- optional, from ``atlas/atlas.json`` beside the data.

They merge into one row shape here so the template need not ask which root it
is looking at; without the curated half a row is simply shorter. Nothing reads
a cloud, an image or a height map -- the probe is cached and the atlas is one
small JSON read.
"""

from __future__ import annotations

from typing import Any

from .probe import DatasetRoot, SceneProbe
from .splits import SPLIT_NAMES, SPLIT_ORDER, SPLIT_TITLES, split_warning

#: Thumbnail slots and where in a scene they sample it when the atlas does not
#: say. A curated root names the frame each slot was cut from, so the picture,
#: its caption and the frame it links to are one number; these fractions are the
#: fallback for a root with no atlas at all.
THUMB_SLOTS: tuple[str, ...] = ("a", "b", "c")
THUMB_FRACTIONS: dict[str, float] = {"a": 0.1, "b": 0.5, "c": 0.9}

#: Ground-truth coverage bands, used only to colour the badge.
COVERAGE_HIGH = 0.85
COVERAGE_MID = 0.5

#: Modalities the disk bar gives a colour of their own, in the order of the
#: palette slots (style.css ``.dk-*``); any other directory counts as "other".
DISK_MODALITIES: tuple[str, ...] = ("images", "clouds", "height_maps", "height_masks", "driven_similarity", "other")


def band(value: float | None, high: float, mid: float) -> str:
    """``hi`` / ``mid`` / ``lo`` against two thresholds, ``none`` when unknown."""
    if value is None:
        return "none"
    if value >= high:
        return "hi"
    return "mid" if value >= mid else "lo"


def coverage_class(coverage: float | None) -> str:
    return band(coverage, COVERAGE_HIGH, COVERAGE_MID)


def fmt_duration(seconds: float | None) -> str:
    """``39 s`` / ``10 min 53 s`` / ``1 h 3 min`` -- the atlas's own phrasing."""
    if not seconds or seconds <= 0:
        return "-"
    total = int(seconds)  # truncate: a scene is "10 min 53 s" long, not 54
    if total < 60:
        return f"{total} s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes} min {secs} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes} min"


def fmt_count(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "-"


def fmt_bytes(n: float) -> str:
    """``3.6 MB`` / ``812 MB`` / ``80.2 GB`` / ``1.45 TB``, decimal units like ``df -H``."""
    for unit in ("B", "kB", "MB", "GB"):
        if n < 1000:
            return f"{n:.1f} {unit}" if unit == "GB" or (unit != "B" and n < 10) else f"{n:.0f} {unit}"
        n /= 1000
    return f"{n:.2f} TB"


def fmt_disk_rate(nbytes: float, seconds: float | None) -> str | None:
    """Bytes per hour of recording: what one more hour in the field costs on disk."""
    return f"{nbytes / seconds * 3600 / 1e9:.0f} GB/h" if seconds else None


def content_bytes(entry: dict) -> int:
    """A ``disk_usage.json`` scene or modality entry's size, linked bytes included.

    A root built from links into another (v2.4 into v2.3) stores almost nothing
    itself, but its scenes still hold everything they link to; showing only the
    own ``bytes`` would read a fully linked scene as empty.
    """
    return int(entry.get("bytes") or 0) + int(entry.get("linked_bytes") or 0)


def fmt_share(f: float) -> str:
    """``91%`` / ``7.4%`` / ``<0.1%``: a sliver of the bar still reads as present."""
    if f >= 0.1:
        return f"{f * 100:.0f}%"
    return f"{f * 100:.1f}%" if f >= 0.001 else "<0.1%"


def stat(label: str, value: Any, sub: str | None = None) -> dict[str, Any]:
    """One cell of a stat grid. The scene page builds its cards out of these too."""
    return {"label": label, "value": value, "sub": sub}


def _curated_stats(a: dict) -> list[dict[str, Any]]:
    """The stat grid the atlas shows, from curated numbers only."""
    stats: list[dict[str, Any]] = [
        stat("duration", fmt_duration(a.get("dur_s"))),
        stat("frames", fmt_count(a.get("frames"))),
    ]
    images, clouds = a.get("images"), a.get("clouds")
    # Redundant when everything matches the frame count -- show only a mismatch.
    if images is not None and clouds is not None and (images != clouds or images != a.get("frames")):
        stats.append(stat("images · clouds", f"{fmt_count(images)} · {fmt_count(clouds)}"))
    if a.get("pts_mean"):
        stats.append(stat("points / cloud", fmt_count(a["pts_mean"])))
    if a.get("poses") is not None:
        # "of N raw" only when N counts the same poses: a scene whose gaps were
        # filled from another root has more of them than its own run produced.
        raw = a.get("poses_unf") or a.get("poses_raw")
        keep = raw and raw >= a["poses"]
        stats.append(stat("poses", fmt_count(a["poses"]), f"of {fmt_count(raw)} raw" if keep else None))
    if a.get("segments") is not None:
        longest = ", ".join(f"{v:.0f}" for v in (a.get("seg_lens") or [])[:3])
        stats.append(stat("segments", a["segments"], f"longest {longest} s" if longest else None))
    if a.get("gt_span_s"):
        med = a.get("seg_med_s")
        stats.append(stat("time with GT", fmt_duration(a["gt_span_s"]), f"median segment {med:.1f} s" if med else None))
    if a.get("path_m") is not None:
        stats.append(stat("path in GT", f"{a['path_m']:.0f} m"))
    if a.get("moving_frac") is not None:
        stats.append(stat("moving", f"{a['moving_frac'] * 100:.0f}%"))
    return stats


def _probed_stats(p: SceneProbe) -> list[dict[str, Any]]:
    """The stat grid for a root with no curated metadata: probe output only."""
    stats = [
        stat("duration", fmt_duration(p.duration)),
        stat("frames", fmt_count(p.n_frames)),
    ]
    images, clouds = p.counts.get("images"), p.counts.get("clouds")
    if images is not None or clouds is not None:
        stats.append(
            stat(
                "images · clouds",
                f"{fmt_count(images) if images is not None else '-'}"
                f" · {fmt_count(clouds) if clouds is not None else '-'}",
            )
        )
    if p.n_segments:
        reasons = ", ".join(f"{n}× {r}" for r, n in sorted(p.segment_reasons.items(), key=lambda kv: -kv[1])[:2])
        stats.append(stat("segments", p.n_segments, reasons or None))
    if p.segment_coverage is not None:
        stats.append(stat("GT coverage", f"{p.segment_coverage * 100:.0f}%"))
    for key, label in (
        ("height_maps", "height maps"),
        ("height_masks", "validity masks"),
        ("driven_similarity", "driven & similar"),
    ):
        if p.counts.get(key):
            stats.append(stat(label, fmt_count(p.counts[key])))
    if p.start_ts:
        stats.append(stat("first stamp", f"{p.start_ts:.0f}"))
    return stats


def segment_timeline(probe: SceneProbe, atlas_entry: dict) -> tuple[list, list]:
    """``(normal spans, filled spans)`` for a scene's ground-truth timeline.

    Both halves come from ONE source. The live segment file wins whenever it
    produced any span at all, because that file is the truth and the atlas may
    predate a re-filtering; but "the live file has no filled segments" is an
    answer, not a hole, so it must not fall through to the atlas's list. Mixing
    them drew an amber "filled from v1.1" band at atlas coordinates over live
    spans -- a scene whose fills had since been dropped still claimed them, on
    the index card, on the scene page's timeline and in its prose.
    """
    if probe.segment_spans:
        # `fill_spans` is a subset of `segment_spans`, so the plain half has to
        # have it taken out or every filled stretch is drawn twice.
        spans, fill = list(probe.segment_spans), list(probe.fill_spans)
    else:
        fill = [list(s) for s in (atlas_entry.get("fill_spans") or [])]
        spans = [list(s) for s in (atlas_entry.get("seg_spans") or [])] + fill
    return [s for s in spans if s not in fill], fill


def thumb_slots(atlas_entry: dict) -> list[str]:
    """The slots the atlas curates for a scene, in slot order.

    Two spellings are accepted: ``{"a": 196}`` names the frame each slot was cut
    from (what ``scripts/make_atlas.py`` writes), and a bare ``["a", "b", "c"]``
    is the older form that only claimed a file exists -- there the frame falls
    back to the fraction, which is what those files were cut at anyway.
    """
    raw = atlas_entry.get("thumbs")
    keys = raw.keys() if isinstance(raw, dict) else (raw or ())
    return [s for s in THUMB_SLOTS if s in keys]


def thumb_index(atlas_entry: dict, slot: str, n_frames: int) -> int:
    """The frame one thumbnail slot shows: the curated index, else the fraction.

    One function for the picture, its caption and its link, so a curated
    thumbnail cannot be captioned with a frame it was not cut from.
    """
    raw = atlas_entry.get("thumbs")
    index = raw.get(slot) if isinstance(raw, dict) else None
    if not isinstance(index, int) or isinstance(index, bool):
        index = round(THUMB_FRACTIONS[slot] * (n_frames - 1)) if n_frames else 0
    return min(max(0, index), max(0, n_frames - 1))


def _thumbs(scene_id: str, probe: SceneProbe, atlas_entry: dict) -> list[dict[str, str]]:
    """Three thumbnail URLs for a scene, stored ones preferred.

    Both cases go through the same endpoint: it serves the curated file when the
    root has one and renders a live frame when it does not, so the template does
    not branch and a root that later grows an ``atlas/`` needs no template change.
    """
    stored = thumb_slots(atlas_entry)
    slots = stored or (list(THUMB_SLOTS) if probe.has("images") and probe.n_frames else [])
    return [
        {
            "url": f"/api/scene/{scene_id}/thumb/{slot}.jpg",
            "alt": f"scene {scene_id} frame {thumb_index(atlas_entry, slot, probe.n_frames)}",
        }
        for slot in slots
    ]


def split_mark(root: DatasetRoot, scene_id: str) -> dict[str, str] | None:
    """The train/val mark for one scene, or ``None`` when the root declares none.

    One function for the index card's band and the scene-page tag, so a scene
    cannot be banded as held out on the index and tagged as training on its own
    page. The
    tooltip names the fold and the tree, because a scene id only means something
    inside one tree and the badge is the only place a reader of the scene page
    would find out which fold put it there.
    """
    splits = root.splits()
    which = splits.of(scene_id)
    if which is None:
        return None
    where = ", ".join(
        x for x in (splits.name, f"dataset {splits.dataset_version}" if splits.dataset_version else None) if x
    )
    return {
        "which": which,
        "title": f"{SPLIT_TITLES[which]}{f' ({where})' if where else ''}",
    }


def validation_mark(row: dict[str, str]) -> dict[str, str] | None:
    """The chip a scene's ``validation.csv`` row earns, or ``None`` without one.

    One function for the index chip and the scene page's badge, so a scene
    cannot pass on one page and fail on the other. The tooltip carries the
    numbers the row has, since the file holds counts of findings and not
    their text -- for the messages, run ``helhest-validate`` on the scene.
    """
    if not row.get("status"):
        return None
    errors, warnings = _count(row.get("errors")), _count(row.get("warnings"))
    if errors:
        cls, label = "no", f"{errors} error{'s' if errors != 1 else ''}"
    elif warnings:
        cls, label = "partial", f"{warnings} warning{'s' if warnings != 1 else ''}"
    else:
        cls, label = "yes", "validated ok"
    detail = "; ".join(f"{k} {row[k]}" for k in VALIDATION_DETAIL if row.get(k))
    status = row["status"]
    if status == "FAIL" and not errors:
        status = "FAIL (--strict)"  # a warning-only scene the run was told to fail on
    return {"cls": cls, "label": label, "status": status, "title": f"helhest-validate: {status}; {detail}"}


#: The columns of ``validation.csv`` worth reading off a chip, in this order.
VALIDATION_DETAIL: tuple[str, ...] = (
    "frames",
    "max_step_m",
    "max_speed_mps",
    "tilt_max_deg",
    "max_sync_offset_s",
    "nan_frac",
    "cloud_dtype",
)


def _count(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def validation_totals(root: DatasetRoot, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """One masthead cell: how the scenes the index shows fared, or ``None``.

    Counted over the index's rows, as the disk and split panels are: the file
    may name scenes this root no longer has, or miss ones added since.
    """
    if root.validation() is None:
        return None
    tally = {"yes": 0, "partial": 0, "no": 0}
    for row in rows:
        mark = row.get("validation")
        if mark:
            tally[mark["cls"]] += 1
    seen = sum(tally.values())
    if not seen:
        return None
    w, e = tally["partial"], tally["no"]
    sub = f"{w} warning{'s' if w != 1 else ''}, {e} error{'s' if e != 1 else ''}"
    if seen < len(rows):
        sub += f"; {len(rows) - seen} not in the report"
    return stat("validated", f"{tally['yes']} ok", sub)


def build_row(root: DatasetRoot, probe: SceneProbe) -> dict[str, Any]:
    """One scene card's worth of content."""
    a = root.atlas_scene(probe.scene_id)
    coverage = a.get("cov", probe.segment_coverage)

    normal, fill = segment_timeline(probe, a)
    stats = _curated_stats(a) if a else _probed_stats(probe)
    disk = root.disk_usage_scene(probe.scene_id)
    if isinstance(disk.get("bytes"), int):
        total = content_bytes(disk)
        stats.append(stat("on disk", fmt_bytes(total), fmt_disk_rate(total, disk.get("span_s"))))

    return {
        "probe": probe,
        "split": split_mark(root, probe.scene_id),
        "validation": validation_mark(root.validation_scene(probe.scene_id)),
        "atlas": a,
        "description": a.get("description"),
        "date": a.get("date"),
        "coverage": coverage,
        "coverage_class": coverage_class(coverage),
        "thumbs": _thumbs(probe.scene_id, probe, a),
        "stats": stats,
        "spans": normal,
        "fill_spans": fill,
        "span_label": fmt_duration(a.get("dur_s") or probe.duration),
    }


def _derived_totals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Header totals for a root with no curated ones: sums over the probes."""
    probes = [r["probe"] for r in rows]
    frames = sum(p.n_frames for p in probes)
    seconds = sum(p.duration or 0.0 for p in probes)
    segments = sum(p.n_segments for p in probes)
    covered = [p.segment_coverage for p in probes if p.segment_coverage is not None]
    totals = [
        stat("scenes", str(len(probes))),
        stat("recorded", fmt_duration(seconds)),
        stat("frames", fmt_count(frames)),
    ]
    if segments:
        totals.append(stat("GT segments", fmt_count(segments)))
    if covered:
        totals.append(
            stat("GT coverage", f"{sum(covered) / len(covered) * 100:.0f}%", f"mean over {len(covered)} scenes")
        )
    return totals


#: ``/?status=`` values and the validation statuses each one shows.
STATUS_FILTERS = {"ok": ("OK",), "warn": ("WARN",), "fail": ("FAIL", "FAIL (--strict)")}


def build_index(
    root: DatasetRoot,
    probes: list[SceneProbe],
    status: str | None = None,
    notes: dict[str, list] | None = None,
    verdict: str | None = None,
) -> dict[str, Any]:
    """Everything the index template needs, curated half included when present.

    ``status`` keeps only the scenes ``validation.csv`` gives that status
    (``STATUS_FILTERS``); the totals then describe the scenes shown.
    ``notes`` is every scene's notes (``NoteStore.list_all``): each card then
    says how many it has and what verdicts they carry, and ``verdict`` keeps
    only the scenes with a note of that verdict.
    """
    atlas = root.atlas() or {}
    rows = [build_row(root, p) for p in probes]
    for r in rows:
        mine = (notes or {}).get(r["probe"].scene_id, [])
        r["notes"] = {
            "count": len(mine),
            "verdicts": {v: n for v in ("drop", "fix", "keep") if (n := sum(1 for x in mine if x.verdict == v))},
        }
    if verdict:
        rows = [r for r in rows if verdict in r["notes"]["verdicts"]]
    counts = {k: sum(1 for r in rows if (r["validation"] or {}).get("status") in v) for k, v in STATUS_FILTERS.items()}
    status_filter = {"current": status, "counts": counts} if any(r["validation"] for r in rows) else None
    if status in STATUS_FILTERS:
        rows = [r for r in rows if (r["validation"] or {}).get("status") in STATUS_FILTERS[status]]
    by_id = {r["probe"].scene_id: r for r in rows}

    groups: list[dict[str, Any]] = []
    taken: set[str] = set()
    for group in atlas.get("groups") or []:
        members = [by_id[sid] for sid in group.get("scenes", []) if sid in by_id]
        if not members:
            continue  # a group whose scenes are not in this root: not our root
        taken.update(r["probe"].scene_id for r in members)
        groups.append(
            {
                "title": group.get("title"),
                "range": group.get("range"),
                "blurb": group.get("blurb"),
                "rows": members,
            }
        )
    # Anything the atlas did not mention still gets shown -- a scene added to a
    # root after the atlas was written must not vanish from the index.
    rest = [r for r in rows if r["probe"].scene_id not in taken]
    if rest:
        groups.append(
            {
                "title": "Scenes" if not groups else "Not in the atlas",
                "range": f"{len(rest)} scene{'s' if len(rest) != 1 else ''}",
                "blurb": None if not groups else "Present in this root but absent from atlas/atlas.json.",
                "rows": rest,
            }
        )

    totals = list(atlas.get("totals") or _derived_totals(rows))
    disk = disk_panel(root, rows)
    if disk:
        totals.append(stat("on disk", disk["total"], disk["rate"]))
    validated = validation_totals(root, rows)
    if validated:
        totals.append(validated)

    return {
        "rows": rows,
        "groups": groups,
        "atlas": atlas or None,
        "atlas_warning": atlas_warning(root, atlas),
        "totals": totals,
        # Where the bytes go, as one bar under the totals; None without a
        # disk_usage.json, and the masthead is as it always was.
        "disk_panel": disk,
        # The split panel under the masthead: what the card bands mean, and the
        # filter over them. Absent -- not empty -- on a root with no splits.json,
        # so the index of an unsplit root gains nothing at all.
        "split_panel": split_panel(root, rows),
        # Links that show only the scenes validation.csv passed, warned on or
        # failed; None on a root that was never validated.
        "status_filter": status_filter,
        "split_warning": split_warning(root.splits(), root.path, [r["probe"].scene_id for r in rows], root.manifest()),
    }


def disk_panel(root: DatasetRoot, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The root's bytes by modality, summed over the scenes the index shows.

    Summed here rather than read from the file's own totals, for the reason
    `split_panel` counts rows: the file can lag the root (a scene extracted
    after the last scan) or name scenes it no longer has, and the scenes it
    has not measured are listed instead of silently left out of the sum.
    """
    ids = [r["probe"].scene_id for r in rows]
    scanned = {s: root.disk_usage_scene(s) for s in ids}
    scanned = {s: e for s, e in scanned.items() if e.get("modalities")}
    parts = disk_parts(scanned.values())
    if not parts:
        return None
    total = sum(p["bytes"] for p in parts)
    seconds = sum(e.get("span_s") or 0 for e in scanned.values())
    return {
        "total": fmt_bytes(total),
        "rate": fmt_disk_rate(total, seconds),
        "generated": (root.disk_usage() or {}).get("generated"),
        "missing": [s for s in ids if s not in scanned],
        "parts": parts,
    }


def disk_parts(entries: Any) -> list[dict[str, Any]]:
    """The disk bar's slices, largest first, summed over ``disk_usage.json``
    scene entries, linked bytes included. One function for the index's bar and a scene page's, so a
    modality has the same colour and name on both."""
    by_mod: dict[str, int] = {}
    for entry in entries:
        for name, m in (entry.get("modalities") or {}).items():
            key = name if name in DISK_MODALITIES else "other"
            by_mod[key] = by_mod.get(key, 0) + content_bytes(m)
    total = sum(by_mod.values())
    if not total:
        return []
    return [
        {
            "key": key,
            "label": key.replace("_", " "),
            "bytes": b,
            "size": fmt_bytes(b),
            "share": b / total,
            "pct": fmt_share(b / total),
        }
        for key, b in sorted(by_mod.items(), key=lambda kv: -kv[1])
    ]


def split_panel(root: DatasetRoot, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """What the card bands mean and the filter over them, for this root's scenes.

    Counted over `rows` rather than over the file's own lists because the file
    may name scenes this root does not have (it is written once and carried
    between root versions), and a panel that claims ten held-out scenes where
    the index shows eight is worse than no panel.
    """
    splits = root.splits()
    if not splits:
        return None
    tally: dict[str, int] = {}
    for row in rows:
        mark = row.get("split")
        if mark:
            tally[mark["which"]] = tally.get(mark["which"], 0) + 1
    if not tally:
        return None
    fractions = splits.shares({r["probe"].scene_id: r["probe"].n_frames for r in rows}) or {}
    sides = [which for which in SPLIT_ORDER if which in tally]
    return {
        "name": splits.name,
        "dataset_version": splits.dataset_version,
        "note": splits.note,
        "source": splits.source,
        "total": len(rows),
        # Named unit, never a bare percentage: the same fold is a different
        # share of the frames, the labels and the training windows.
        "unit": "frames" if fractions else None,
        # "parts", not "items": Jinja resolves `panel.items` to dict.items and
        # the template would iterate the bound method instead of the list.
        "parts": [
            {
                "which": which,
                "name": SPLIT_NAMES[which],
                "title": SPLIT_TITLES[which],
                "count": tally[which],
                "share": fractions.get(which),
                "pct": f"{fractions[which] * 100:.0f}%" if which in fractions else None,
            }
            for which in sides
        ],
        "measures": split_measures(root, splits, rows, sides),
    }


def split_measures(
    root: DatasetRoot, splits: Any, rows: list[dict[str, Any]], sides: list[str]
) -> list[dict[str, Any]]:
    """How big each side of the split is, in every unit the root can count.

    One row per unit -- scenes, frames, recorded time, keyframes, proprio
    labels, training windows -- with a cell per side and the total over every
    scene shown. A unit the root does not carry (no keyframes.lst, no label
    counts) is left out rather than shown as zeros.
    """
    keyframes = root.keyframe_counts()
    units: list[tuple[str, Any, Any]] = [
        ("Scenes", lambda r: 1, fmt_count),
        ("Frames", lambda r: r["probe"].n_frames, fmt_count),
        ("Recorded", lambda r: r["probe"].duration or 0, fmt_duration),
    ]
    for label, counts in (("Keyframes", keyframes), ("Proprio labels", splits.labels), ("Windows", splits.windows)):
        if counts:
            units.append((label, lambda r, c=counts: c.get(r["probe"].scene_id, 0), fmt_count))
    out = []
    for label, value, fmt in units:
        per_side = {which: 0.0 for which in sides}
        total = 0.0
        for row in rows:
            v = value(row) or 0
            total += v
            mark = row.get("split")
            if mark and mark["which"] in per_side:
                per_side[mark["which"]] += v
        if not total:
            continue
        out.append(
            {
                "label": label,
                "cells": [{"value": fmt(per_side[w]), "pct": fmt_share(per_side[w] / total)} for w in sides],
                "total": fmt(total),
            }
        )
    return out


def atlas_warning(root: DatasetRoot, atlas: dict) -> str | None:
    """Why this root's atlas should not be believed, if there is a reason.

    A curated atlas carries rendered numbers, not just prose, so one copied from
    an earlier release describes that release: HELHEST v1.3 shipped with its
    ``atlas/`` symlinked to v1.2's and the index quietly showed v1.2's poses,
    segments and coverage under a v1.2 title. The generator stamps the dataset
    version it was run against; a mismatch says so on the page instead.
    """
    if not atlas:
        return None
    manifest = root.manifest() or {}
    theirs, ours = atlas.get("dataset_version"), manifest.get("dataset_version")
    if theirs and ours and str(theirs) != str(ours):
        return (
            f"atlas/atlas.json describes dataset {theirs}, this root is {ours} - "
            "its prose and numbers are stale; regenerate with scripts/make_atlas.py"
        )
    return None
