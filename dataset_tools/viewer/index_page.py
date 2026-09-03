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

#: Thumbnail slots and where in a scene they sample it. The stored thumbnails
#: in ``atlas/thumbs/`` were cut at these same fractions, so a root with an
#: atlas and a root without it show the same three moments of a scene.
THUMB_SLOTS: tuple[str, ...] = ("a", "b", "c")
THUMB_FRACTIONS: dict[str, float] = {"a": 0.1, "b": 0.5, "c": 0.9}

#: Ground-truth coverage bands, used only to colour the badge.
COVERAGE_HIGH = 0.85
COVERAGE_MID = 0.5


def coverage_class(coverage: float | None) -> str:
    if coverage is None:
        return "none"
    if coverage >= COVERAGE_HIGH:
        return "hi"
    return "mid" if coverage >= COVERAGE_MID else "lo"


def fmt_duration(seconds: float | None) -> str:
    """``39 s`` / ``10 min 53 s`` / ``1 h 3 min`` -- the atlas's own phrasing."""
    if not seconds or seconds <= 0:
        return "—"
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
        return "—"


def stat(label: str, value: Any, sub: str | None = None) -> dict[str, Any]:
    """One cell of a stat grid. The scene page builds its cards out of these too."""
    return {"label": label, "value": value, "sub": sub}


def _curated_stats(a: dict) -> list[dict[str, Any]]:
    """The stat grid the atlas shows, from curated numbers only."""
    stats: list[dict[str, Any]] = [
        stat("duration", fmt_duration(a.get("dur_s"))),
        stat("frames", fmt_count(a.get("frames"))),
    ]
    if a.get("images") is not None and a.get("clouds") is not None:
        # Redundant when everything matches the frame count -- show only a mismatch.
        if a["images"] != a["clouds"] or a["images"] != a.get("frames"):
            stats.append(stat("images · clouds", f"{fmt_count(a['images'])} · {fmt_count(a['clouds'])}"))
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
                f"{fmt_count(images) if images is not None else '—'}"
                f" · {fmt_count(clouds) if clouds is not None else '—'}",
            )
        )
    if p.n_segments:
        reasons = ", ".join(f"{n}× {r}" for r, n in sorted(p.segment_reasons.items(), key=lambda kv: -kv[1])[:2])
        stats.append(stat("segments", p.n_segments, reasons or None))
    if p.segment_coverage is not None:
        stats.append(stat("GT coverage", f"{p.segment_coverage * 100:.0f}%"))
    for key, label in (("height_maps", "height maps"), ("height_masks", "validity masks")):
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


def _thumbs(scene_id: str, probe: SceneProbe, atlas_entry: dict) -> list[dict[str, str]]:
    """Three thumbnail URLs for a scene, stored ones preferred.

    Both cases go through the same endpoint: it serves the curated file when the
    root has one and renders a live frame when it does not, so the template does
    not branch and a root that later grows an ``atlas/`` needs no template change.
    """
    stored = [s for s in (atlas_entry.get("thumbs") or []) if s in THUMB_SLOTS]
    slots = stored or (list(THUMB_SLOTS) if probe.has("images") and probe.n_frames else [])
    return [
        {
            "url": f"/api/scene/{scene_id}/thumb/{slot}.jpg",
            "alt": f"scene {scene_id} at {THUMB_FRACTIONS[slot] * 100:.0f}%",
            "stored": bool(stored),
        }
        for slot in slots
    ]


def build_row(root: DatasetRoot, probe: SceneProbe) -> dict[str, Any]:
    """One scene card's worth of content."""
    a = root.atlas_scene(probe.scene_id)
    coverage = a.get("cov", probe.segment_coverage)

    normal, fill = segment_timeline(probe, a)

    return {
        "probe": probe,
        "csv": root.csv_rows().get(probe.scene_id, {}),
        "atlas": a,
        "description": a.get("description"),
        "date": a.get("date"),
        "coverage": coverage,
        "coverage_class": coverage_class(coverage),
        "thumbs": _thumbs(probe.scene_id, probe, a),
        "stats": _curated_stats(a) if a else _probed_stats(probe),
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
        {"label": "scenes", "value": str(len(probes)), "sub": None},
        {"label": "recorded", "value": fmt_duration(seconds), "sub": None},
        {"label": "frames", "value": fmt_count(frames), "sub": None},
    ]
    if segments:
        totals.append({"label": "GT segments", "value": fmt_count(segments), "sub": None})
    if covered:
        totals.append(
            {
                "label": "GT coverage",
                "value": f"{sum(covered) / len(covered) * 100:.0f}%",
                "sub": f"mean over {len(covered)} scenes",
            }
        )
    return totals


def build_index(root: DatasetRoot, probes: list[SceneProbe]) -> dict[str, Any]:
    """Everything the index template needs, curated half included when present."""
    atlas = root.atlas() or {}
    rows = [build_row(root, p) for p in probes]
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

    return {
        "rows": rows,
        "groups": groups,
        "atlas": atlas or None,
        "totals": atlas.get("totals") or _derived_totals(rows),
    }
