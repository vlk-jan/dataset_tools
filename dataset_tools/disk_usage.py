#!/usr/bin/env python3
"""Disk usage of a root per scene and modality, as ``<root>/disk_usage.json``.

    helhest-disk-usage <root> [scene ...] [--all]

Without scene ids it scans only the scenes the file does not have yet, so a
grown root costs one scan of the new scenes; name scenes to rescan them, or
``--all`` for everything. A file written under an older schema is rescanned
whole. ``helhest-extract`` calls `update` for the scenes it just wrote. Scenes whose directory is gone are dropped. Keys a
person added by hand to a scene entry (a ``note``, say) survive a rescan; the
measured keys are overwritten. Totals are recomputed on every run.

Every subdirectory of a scene is one modality (``images``, ``clouds``, ...);
files at the scene's top level count as ``other``, and each of them is also
listed by name under ``files``. Sizes are apparent bytes.

Layout, flat so a chart reads it without reshaping:

    {"schema": 2, "root": "HELHEST_v2_0", "generated": "<iso>", "unit": "bytes",
     "modalities": ["images", "clouds", ...],            # largest first
     "totals": {"bytes": .., "files": .., "frames": .., "span_s": ..,
                "modalities": {"images": {"bytes": .., "files": ..}, ...}},
     "scenes": {"0000": {"bytes": .., "frames": .., "span_s": .., "scanned": "<iso>",
                         "modalities": {"images": {"bytes": .., "files": ..}, ...},
                         "files": {"odometry.txt": <bytes>, ...}}}}

Reads the scenes only; writes the one file, atomically.
"""

import argparse
import json
import logging
import os
from datetime import datetime, timezone

NAME = "disk_usage.json"
SCHEMA = 2
LOG = logging.getLogger(__name__)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def dir_usage(path):
    """(bytes, files) of every regular file under ``path``."""
    size = files = 0
    for dirpath, _, names in os.walk(path):
        for n in names:
            st = os.lstat(os.path.join(dirpath, n))
            size += st.st_size
            files += 1
    return size, files


def span_s(scene_dir):
    """Seconds between the first and last frame stamp in ``data_ts.txt``; None without one."""
    try:
        with open(os.path.join(scene_dir, "data_ts.txt")) as f:
            ts = [float(line) for line in f if line.strip()]
    except OSError:
        return None
    return round(ts[-1] - ts[0], 3) if ts else None


def scan_scene(scene_dir):
    mods, files = {}, {}
    for e in os.scandir(scene_dir):
        if e.is_dir(follow_symlinks=False):
            b, n = dir_usage(e.path)
            mods[e.name] = {"bytes": b, "files": n}
        elif e.is_file(follow_symlinks=False):
            files[e.name] = e.stat(follow_symlinks=False).st_size
    mods["other"] = {"bytes": sum(files.values()), "files": len(files)}
    frames = mods.get("images", {}).get("files")
    return {
        "bytes": sum(m["bytes"] for m in mods.values()),
        "frames": frames,
        "span_s": span_s(scene_dir),
        "scanned": now(),
        "modalities": dict(sorted(mods.items(), key=lambda kv: -kv[1]["bytes"])),
        "files": dict(sorted(files.items(), key=lambda kv: -kv[1])),
    }


def totals(scenes):
    mods = {}
    for s in scenes.values():
        for k, m in s["modalities"].items():
            t = mods.setdefault(k, {"bytes": 0, "files": 0})
            t["bytes"] += m["bytes"]
            t["files"] += m["files"]
    mods = dict(sorted(mods.items(), key=lambda kv: -kv[1]["bytes"]))
    return {
        "bytes": sum(m["bytes"] for m in mods.values()),
        "files": sum(m["files"] for m in mods.values()),
        "frames": sum(s.get("frames") or 0 for s in scenes.values()),
        "span_s": round(sum(s.get("span_s") or 0 for s in scenes.values()), 3),
        "modalities": mods,
    }


def update(root, rescan=(), rescan_all=False):
    """Bring ``<root>/disk_usage.json`` up to date and return its contents."""
    path = os.path.join(root, NAME)
    try:
        with open(path) as f:
            doc = json.load(f)
        old = doc.get("scenes", {})
        rescan_all = rescan_all or doc.get("schema") != SCHEMA
    except FileNotFoundError:
        old = {}
    on_disk = sorted(d for d in os.listdir(root) if d.isdigit() and os.path.isdir(os.path.join(root, d)))
    scenes = {}
    for s in on_disk:
        if rescan_all or s in rescan or s not in old:
            LOG.info("disk usage: scanning %s", s)
            scenes[s] = {**old.get(s, {}), **scan_scene(os.path.join(root, s))}
        else:
            scenes[s] = old[s]
    tot = totals(scenes)
    doc = {
        "schema": SCHEMA,
        "root": os.path.basename(os.path.abspath(root)),
        "generated": now(),
        "unit": "bytes",
        "modalities": list(tot["modalities"]),
        "totals": tot,
        "scenes": scenes,
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)
    return doc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("scenes", nargs="*", help="rescan these scenes even if the file has them")
    ap.add_argument("--all", action="store_true", help="rescan every scene")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    missing = [s for s in a.scenes if not os.path.isdir(os.path.join(a.root, s))]
    if missing:
        ap.error(f"no such scene: {' '.join(missing)}")
    doc = update(a.root, set(a.scenes), a.all)
    t = doc["totals"]
    print(f"{NAME}: {len(doc['scenes'])} scenes, {t['bytes'] / 1e9:.1f} GB")
    for k, m in t["modalities"].items():
        print(f"  {k:14s}{m['bytes'] / 1e9:9.1f} GB {100 * m['bytes'] / t['bytes']:5.1f}%")


if __name__ == "__main__":
    main()
