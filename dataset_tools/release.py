"""The root-level files of a release, from its spec and its scenes.

    python -m dataset_tools.release ROOT --spec docs/releases/v2_4.json [--out_dir DIR]

Writes ``scenes.csv``, the three keyframe lists (:mod:`dataset_tools.keyframes`),
``manifest.json``, ``splits.json``, ``atlas/curation.json`` and ``README.md``.
Run it once every scene of the bag list is extracted (``slurm/extract_v2.batch``),
then validate and draw the atlas (``slurm/finalize_v2.sh`` does all three).

Two sources, kept apart:

- **The spec** (``docs/releases/<release>.json``): what only a person can say --
  the bag list, the held-out scenes and why, the sessions, what was left out,
  known defects, open items, the atlas prose. ``"inherit"`` names a file of
  prose shared by every release of a rig (``docs/releases/helhest_v2.json``):
  the sensors, the frames, the odometry contract. The spec wins key by key.
- **The scenes**: everything measured -- frame counts, spans, paths, segments,
  the pose check, deskew, the odom_frame correction, stop trimming, tilted
  stretches, height-map settings and coverage, list counts.

``proprio_labels`` and ``derived`` belong to the tools that write those
products (DRIFT); an existing manifest's entries are carried over untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from dataset_tools import heightmap, image_format, keyframes
from dataset_tools.extract.scene_io import merge_scenes_csv
from dataset_tools.gnss_guard import MAX_SPEED
from dataset_tools.gt_io import _replace_atomically, read_segments, read_trajectory
from dataset_tools.transforms import tilt_deg

#: Manifest keys other tools write; carried over from an existing manifest.
OWNED_ELSEWHERE = ("proprio_labels", "derived")
#: Spec keys copied into the manifest as they are.
PROSE_KEYS = (
    "rig",
    "sessions",
    "excluded",
    "cloud_frame",
    "cloud_dtype",
    "odometry",
    "gravity_align",
    "camera",
    "gps",
    "terrain_notes",
    "open_items",
    "not_present",
)
TILT_REPORT_DEG = 60.0


def _git(*argv) -> str | None:
    try:
        return subprocess.run(
            ["git", *argv], cwd=Path(__file__).resolve().parent, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_spec(path: Path) -> dict:
    """The spec with its ``inherit`` chain merged under it (relative to the spec's directory)."""
    path = Path(path)
    spec = json.loads(path.read_text())
    if parent := spec.pop("inherit", None):
        base = load_spec(path.parent / parent)
        spec = {**base, **spec}
    return spec


def read_bag_list(path: Path) -> dict[str, dict]:
    """``{scene: row}`` of a bag list CSV (``docs/v2_*_bags.csv``)."""
    with open(path, newline="") as fh:
        return {row["scene"].zfill(4): row for row in csv.DictReader(fh)}


def _runs(idx: np.ndarray, d: np.ndarray) -> dict:
    """Index runs with their seconds into the scene: ``[first, last, t_first, t_last]``."""
    if not len(idx):
        return {"count": 0, "runs": []}
    runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    return {
        "count": int(len(idx)),
        "runs": [
            [int(r[0]), int(r[-1]), round(float(d[r[0]] - d[0]), 1), round(float(d[r[-1]] - d[0]), 1)] for r in runs
        ],
    }


def _storage(scene_dir: Path) -> str:
    """'real files', or which root the entries link into."""
    targets = set()
    for p in scene_dir.iterdir():
        if p.is_symlink():
            parts = Path(os.readlink(p)).parts
            targets.add(next((x for x in parts if x not in ("..", ".")), "?"))
    return f"symlinks into ../{', ../'.join(sorted(targets))}" if targets else "real files"


def _tf_pins(path: Path) -> dict:
    """``{parent->child: {bag_vs_pin_mm, bag_vs_pin_deg}}`` from the comment lines of tf_pins.txt."""
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        m = re.match(r"#\s*(\S+) -> (\S+): bag (\d+) mm, ([\d.]+) deg", line)
        if m:
            out[f"{m.group(1)}->{m.group(2)}"] = {"bag_vs_pin_mm": int(m.group(3)), "bag_vs_pin_deg": float(m.group(4))}
    return out


def _odom_frame(path: Path) -> dict | None:
    if not path.exists():
        return None
    rec = json.loads(path.read_text().strip().splitlines()[-1])
    return {
        "roll": round(rec["roll_deg"], 3),
        "pitch": round(rec["pitch_deg"], 3),
        "applied_utc": rec.get("applied_utc"),
    }


def survey_scene(scene_dir: Path) -> dict:
    """What a scene's own files say about it."""
    d = np.loadtxt(scene_dir / "data_ts.txt", ndmin=1)
    ts, poses = read_trajectory(scene_dir / "odometry.txt")
    meta = json.loads((scene_dir / "extract_meta.json").read_text())
    frames = meta.get("frames") or {}
    seg = read_segments(scene_dir / "odometry_segments.txt")
    pos = poses[:, :3, 3]
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    dt = np.diff(ts)
    # Steps inside one segment only: each segment is re-anchored at the identity,
    # and a relocalisation across a hole can look like a slow drive.
    sid = np.searchsorted(np.array([s.start_ts for s in seg]), ts, side="right")
    # dt > 0: a repeated stamp is no speed at all (NaN would reach the manifest).
    in_seg = (np.diff(sid) == 0) & (dt > 0) & (step <= MAX_SPEED * np.maximum(dt, 1e-3))
    tilt = tilt_deg(np.load(scene_dir / "gravity_align.npy", mmap_mode="r"))
    pc = frames.get("pose_check") or {}
    guard = frames.get("guard") or {}
    holes = np.diff(d) > 1.0
    ta = scene_dir / "time_align.txt"
    maps_dir = scene_dir / "height_maps"
    return {
        "bag": (scene_dir / "source_bag.txt").read_text().strip(),
        "storage": _storage(scene_dir),
        "frames": int(len(d)),
        "span_s": round(float(d[-1] - d[0]), 1),
        "path_m": round(float(step[in_seg].sum()), 1),
        "max_speed_mps": round(float((step[in_seg] / dt[in_seg]).max()), 2) if in_seg.any() else None,
        "frame_holes_over_1s": int(holes.sum()),
        "hole_time_s": round(float(np.diff(d)[holes].sum()), 1),
        "odometry_segments": len(seg),
        "frames_without_pose": int(guard.get("n_frames_unsupported", 0) + guard.get("n_frames_trimmed", 0)),
        "frame_clock": ta.read_text().splitlines()[1].strip("# ") if ta.exists() else None,
        "deskew": {k: v for k, v in (frames.get("deskew") or {}).items() if k != "mode"},
        "pose_check_vs_fixposition": {
            k: pc.get(k)
            for k in ("clock_lag_s", "clock_corr", "primary_jumps", "overlap_s", "tilt_diff_deg", "warnings")
            if k in pc
        }
        | ({"displacement_5s": pc["displacement"]["5s"]} if "5s" in (pc.get("displacement") or {}) else {}),
        "idle_trim": frames.get("idle_trim"),
        "odom_frame_deg": _odom_frame(scene_dir / "odom_frame.txt"),
        "height_maps": sum(1 for f in os.listdir(maps_dir) if f.endswith(".npz")) if maps_dir.is_dir() else 0,
        "tilt_max_deg": round(float(tilt.max()), 1),
        f"frames_past_{TILT_REPORT_DEG:.0f}deg": _runs(np.nonzero(tilt > TILT_REPORT_DEG)[0], d),
        "odin_tf_bag_vs_pin": _tf_pins(scene_dir / "tf_pins.txt"),
        "has": {f: (scene_dir / f).exists() for f in ("cmd_vel.npy", "gps.npy", "joint_states.npy")},
        "extraction": {
            "dataset_tools": (meta.get("tool") or {}).get("git_commit"),
            "dirty": (meta.get("tool") or {}).get("git_dirty"),
            "args": [a for a in (meta.get("argv") or [])[3:] if not a.startswith("/")],
        },
    }


def _height_maps_block(root: Path, scenes: dict[str, dict]) -> dict:
    params, recorded = {}, []
    for sc in scenes:
        p = heightmap.scene_params(root / sc)
        if p is not None:
            recorded.append(sc)
            params.setdefault(json.dumps(p.to_dict(), sort_keys=True), []).append(sc)
    if len(params) == 1:
        parameters = json.loads(next(iter(params)))
    elif params:
        parameters = {"per_scene": {json.dumps(k): v for k, v in params.items()}}
    else:
        parameters = heightmap.Params().to_dict()
    built = sum(e["height_maps"] for e in scenes.values())
    frames = sum(e["frames"] for e in scenes.values())
    return {
        "generator": "dataset_tools.heightmap (per scene: height_maps.json)",
        "parameters": parameters,
        "scenes_without_a_settings_record": sorted(set(scenes) - set(recorded)),
        "coverage": {
            "built": built,
            "of_frames": frames,
            "frames_without_a_map": {
                s: e["frames"] - e["height_maps"] for s, e in scenes.items() if e["frames"] > e["height_maps"]
            },
            "note": "a frame with fewer than min_points returns outside min_radius (below the canopy clip) has no map",
        },
        "validity": "0 extrapolated, 1 interpolated, 2 measured, 3 anchored (wheel contact)",
    }


def build_manifest(root: Path, spec: dict, spec_path: Path, bags: dict, scenes: dict, lists: dict, old: dict) -> dict:
    held = sorted(spec["held_out"]["scenes"])
    for sc, e in scenes.items():
        row = bags.get(sc, {})
        e.update(
            {
                "session": row.get("session"),
                "extract_args": (row.get("extract_args") or "").strip() or None,
                "held_out": sc in held,
            }
        )
        if row and Path(row["bag_dir_on_subtdata"]).name != e["bag"]:
            raise SystemExit(
                f"{sc}: holds {e['bag']} but the bag list puts {Path(row['bag_dir_on_subtdata']).name} there"
            )
    commits = sorted({e["extraction"]["dataset_tools"] for e in scenes.values() if e["extraction"]["dataset_tools"]})
    tilted = {
        s: e[f"frames_past_{TILT_REPORT_DEG:.0f}deg"]
        for s, e in scenes.items()
        if e[f"frames_past_{TILT_REPORT_DEG:.0f}deg"]["count"]
    }
    m = {
        "dataset_version": spec["dataset_version"],
        "title": spec.get("title"),
        "summary": spec.get("summary"),
        "generated": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": {
            "tool": "dataset_tools.release",
            "dataset_tools": _git("describe", "--always", "--dirty"),
            "spec": str(spec_path),
            "extraction_commits": commits,
            "driver": "slurm/extract_v2.batch (one bag per array task: extract, odom_frame, height maps, validate), "
            "then slurm/finalize_v2.sh",
            "per_scene_record": "extract_meta.json (tool commit, arguments, bag topics, frame counts, deskew, pose check, guard), "
            "odom_frame.txt, height_maps.json, tf_pins.txt, time_align.txt, gnss_guard.txt, pose_check.txt",
        },
        "source": {"bag_list": spec["bags"], "bags_on": spec.get("bag_host"), "scenes": len(scenes)},
        **{k: spec[k] for k in PROSE_KEYS if k in spec},
        "images": {**(spec.get("images") or {}), "extension": lists["extension"]},
        "held_out": {k: v for k, v in spec["held_out"].items() if k != "scenes"} | {"scenes": held},
        "scenes": scenes,
        "keyframes": {
            "files": lists["lists"],
            "gate": f"one line per frame with a cloud, an image, a height map and a mask (dataset_tools.keyframes), minus every "
            f"frame whose next {keyframes.TILT_HORIZON_S:g} s hold a frame tilted more than {keyframes.MAX_TILT_DEG:g} deg "
            "from level; keyframes_train.lst leaves out held_out.scenes, keyframes_heldout.lst is only them",
        },
        "height_maps": _height_maps_block(root, scenes),
        "corrections": {
            **(spec.get("corrections") or {}),
            "odom_frame_per_scene": {s: e["odom_frame_deg"] for s, e in scenes.items()},
        },
        "defects": {**(spec.get("defects") or {}), f"tilted_past_{TILT_REPORT_DEG:.0f}deg": tilted},
    }
    for k in OWNED_ELSEWHERE:
        if k in old:
            m[k] = old[k]
    return m


def build_splits(root: Path, manifest: dict, lists: dict) -> dict:
    held = manifest["held_out"]["scenes"]
    scenes = sorted(manifest["scenes"])
    out = {
        "split_mode": "held_out",
        "dataset_version": manifest["dataset_version"],
        "note": manifest["held_out"].get("decision"),
        "val_scenes": held,
        "train_scenes": [s for s in scenes if s not in held],
        "keyframes_per_scene": {s: c["listed"] for s, c in lists["scenes"].items()},
        "source": "manifest.json held_out.scenes; keyframes from dataset_tools.keyframes",
    }
    labels = {s: len(os.listdir(root / s / "proprio_labels")) for s in scenes if (root / s / "proprio_labels").is_dir()}
    if labels:
        out["labels_per_scene"] = labels
    return out


def _table(rows: list[list], head: list[str]) -> str:
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join("" if c is None else str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def build_readme(manifest: dict, spec: dict) -> str:
    sc = manifest["scenes"]
    held = set(manifest["held_out"]["scenes"])
    frames = sum(e["frames"] for e in sc.values())
    path_km = sum(e["path_m"] for e in sc.values()) / 1000
    hours = sum(e["span_s"] for e in sc.values()) / 3600
    kf = manifest["keyframes"]["files"]
    parts = [
        f"# {manifest.get('title') or 'HELHEST ' + manifest['dataset_version']}",
        "",
        "*Generated by `dataset_tools.release` from "
        f"`{manifest['generator']['spec']}` at {manifest['generated']}; do not edit by hand -- edit the spec and rebuild.*",
        "",
        manifest.get("summary") or "",
        "",
        f"{len(sc)} scenes, {frames:,} frames, {hours:.1f} h, {path_km:.1f} km driven. "
        f"Keyframe lists: `keyframes.lst` {kf['keyframes.lst']:,} / `keyframes_train.lst` {kf['keyframes_train.lst']:,} / "
        f"`keyframes_heldout.lst` {kf['keyframes_heldout.lst']:,} lines.",
        "",
        "## Rebuilding it",
        "",
        "From a checkout of dataset_tools (https://github.com/vlk-jan/dataset_tools), with the bags reachable:",
        "",
        "```bash",
        f"CSV={spec['bags']} sbatch --array=0-{len(sc) - 1}%4 slurm/extract_v2.batch <ROOT>",
        f"slurm/finalize_v2.sh <ROOT> {spec['bags']} {manifest['generator']['spec']}",
        "```",
        "",
        'The file formats are described in the dataset_tools README ("Output format"). Per-frame products other '
        "tools add later (proprio labels, driven similarity) are not part of the rebuild.",
        "",
        "## Split",
        "",
        f"Validation (`manifest.json` -> `held_out.scenes`): {', '.join(sorted(held))}. "
        f"{manifest['held_out'].get('decision') or ''}",
        "",
        "## Sessions",
        "",
    ]
    for s in manifest.get("sessions") or []:
        parts.append(f"- **{s.get('date')}**, scenes {s.get('scenes')}: {s.get('site')}")
    parts += ["", "## Scenes", ""]
    rows = []
    for k, e in sc.items():
        rows.append(
            [
                k,
                "val" if k in held else "train",
                e["bag"],
                e["frames"],
                f"{e['span_s']:.0f}",
                f"{e['path_m']:.0f}",
                e["odometry_segments"],
                e["height_maps"],
                e["extract_args"] or "",
            ]
        )
    parts.append(
        _table(rows, ["scene", "split", "bag", "frames", "span s", "path m", "segments", "maps", "extractor options"])
    )
    if excl := manifest.get("excluded"):
        parts += ["", "## Left out", ""] + [f"- **{k}**: {v}" for k, v in excl.items()]
    defects = {k: v for k, v in (manifest.get("defects") or {}).items() if not k.startswith("tilted_past")}
    if defects:
        parts += ["", "## Known defects", ""]
        for k, v in defects.items():
            text = v.get("what") if isinstance(v, dict) else v
            parts.append(f"- **{k}**: {text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)}")
    if tilted := manifest["defects"].get(f"tilted_past_{TILT_REPORT_DEG:.0f}deg"):
        parts.append(
            f"- **tilted past {TILT_REPORT_DEG:.0f} deg** (the robot tipped or on its back; kept, gated out of the lists): "
            + "; ".join(f"{s} ({v['count']} frames)" for s, v in tilted.items())
        )
    if items := manifest.get("open_items"):
        parts += ["", "## Open items", ""] + [f"- {x}" for x in items]
    return "\n".join(parts) + "\n"


def _write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False: a NaN stat fails here instead of writing invalid JSON.
    _replace_atomically(
        path, lambda fh: fh.write(json.dumps(doc, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    )


def build(root: Path, spec_path: Path, out_dir: Path | None = None) -> dict:
    root = Path(root)
    out = Path(out_dir) if out_dir else root
    out.mkdir(parents=True, exist_ok=True)
    spec = load_spec(spec_path)
    repo = Path(__file__).resolve().parent.parent
    bags = read_bag_list(repo / spec["bags"] if not Path(spec["bags"]).is_absolute() else Path(spec["bags"]))
    on_disk = sorted(d for d in os.listdir(root) if d.isdigit() and (root / d / "source_bag.txt").exists())
    if missing := sorted(set(bags) - set(on_disk)):
        raise SystemExit(f"{root}: scenes {missing} of {spec['bags']} are not extracted (no source_bag.txt)")
    if extra := sorted(set(on_disk) - set(bags)):
        raise SystemExit(f"{root}: scenes {extra} are not in {spec['bags']}")
    if bad := sorted(set(spec["held_out"]["scenes"]) - set(on_disk)):
        raise SystemExit(f"held_out.scenes names {bad}, which are not in the root")

    old_path = root / "manifest.json"
    old = json.loads(old_path.read_text()) if old_path.exists() else {}
    if out == root:
        merge_scenes_csv(root)
    ext = image_format.declared(root) or image_format.of_root(root)
    lists = keyframes.write_lists(root, out, held=spec["held_out"]["scenes"], ext=ext)

    scenes = {sc: survey_scene(root / sc) for sc in on_disk}
    manifest = build_manifest(root, spec, Path(spec_path), bags, scenes, lists, old)
    _write_json(out / "manifest.json", manifest)
    _write_json(out / "splits.json", build_splits(root, manifest, lists))
    if cur := spec.get("curation"):
        _write_json(out / "atlas" / "curation.json", cur)
    (out / "README.md").write_text(build_readme(manifest, spec))
    return {"scenes": len(scenes), "lists": lists["lists"], "out": str(out)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=None, help="write the root-level files here instead (a dry run)")
    a = ap.parse_args(argv)
    r = build(a.root, a.spec, a.out_dir)
    print(f"{r['out']}: {r['scenes']} scenes; " + ", ".join(f"{k} {v}" for k, v in r["lists"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
