#!/usr/bin/env python3
"""Disk usage of a root per scene and modality, as ``<root>/disk_usage.json``.

    helhest-disk-usage <root> [scene ...] [--all]

Scans only scenes the file lacks, unless named or ``--all`` (or the schema
changed). Hand-added keys in a scene entry survive a rescan. Each scene
subdirectory is a modality; top-level files count as ``other``. Content reached
through symlinks is counted apart as ``linked_bytes``/``linked_files``; dangling
links as ``broken_links``. Sizes are apparent bytes; the file is written atomically.

    {"schema": 3, "root": .., "generated": "<iso>", "unit": "bytes",
     "modalities": [...],  # largest first
     "totals": {"bytes", "files", "linked_bytes", "linked_files", "broken_links",
                "frames", "span_s", "modalities": {name: {"bytes", "files"}}},
     "scenes": {"0000": {"bytes", "linked_bytes", "broken_links", "frames", "span_s",
                         "scanned", "modalities", "files", "linked_files"}}}
"""

import argparse
import json
import logging
import os
import stat
from datetime import UTC, datetime

from dataset_tools.gt_io import _replace_atomically
from dataset_tools.image_format import scenes as scene_dirs

NAME = "disk_usage.json"
SCHEMA = 3
LOG = logging.getLogger(__name__)


def now():
    return datetime.now(UTC).isoformat(timespec="seconds")


def dir_usage(path, linked=False):
    """Own and linked regular files under ``path``, and broken links (cycles skipped)."""
    use = {"bytes": 0, "files": 0, "linked_bytes": 0, "linked_files": 0, "broken_links": 0}
    seen = {os.path.realpath(path)}
    stack = [(path, linked)]
    while stack:
        here, via_link = stack.pop()
        with os.scandir(here) as entries:
            for e in entries:
                via = via_link or e.is_symlink()
                try:
                    st = e.stat()  # of the target, for a link
                except OSError:
                    use["broken_links"] += 1
                    continue
                if stat.S_ISDIR(st.st_mode):
                    real = os.path.realpath(e.path)
                    if real not in seen:
                        seen.add(real)
                        stack.append((e.path, via))
                elif stat.S_ISREG(st.st_mode):
                    use["linked_bytes" if via else "bytes"] += st.st_size
                    use["linked_files" if via else "files"] += 1
    return use


def modality(use, link=None):
    """A modality entry: the two own figures, and the linked ones only where there are any."""
    m = {"bytes": use["bytes"], "files": use["files"]}
    if use["linked_files"] or link is not None:
        m["linked_bytes"] = use["linked_bytes"]
        m["linked_files"] = use["linked_files"]
    if link is not None:
        m["link"] = link
    return m


def span_s(scene_dir):
    """Seconds between the first and last frame stamp in ``data_ts.txt``; None without one."""
    try:
        with open(os.path.join(scene_dir, "data_ts.txt")) as f:
            ts = [float(line) for line in f if line.strip()]
    except OSError:
        return None
    return round(ts[-1] - ts[0], 3) if ts else None


def scan_scene(scene_dir):
    mods, files, linked_files, broken = {}, {}, {}, 0
    for e in os.scandir(scene_dir):
        link = os.readlink(e.path) if e.is_symlink() else None
        try:
            st = e.stat()  # of the target, for a link
        except OSError:
            broken += 1
            continue
        if stat.S_ISDIR(st.st_mode):
            use = dir_usage(e.path, linked=link is not None)
            broken += use["broken_links"]
            mods[e.name] = modality(use, link)
        elif stat.S_ISREG(st.st_mode):
            if link is None:
                files[e.name] = st.st_size
            else:
                linked_files[e.name] = {"bytes": st.st_size, "link": link}
    other = {"bytes": sum(files.values()), "files": len(files), "linked_files": len(linked_files)}
    other["linked_bytes"] = sum(f["bytes"] for f in linked_files.values())
    mods["other"] = modality(other)
    images = mods.get("images")
    return {
        "bytes": sum(m["bytes"] for m in mods.values()),
        "linked_bytes": sum(m.get("linked_bytes", 0) for m in mods.values()),
        "broken_links": broken,
        "frames": images["files"] + images.get("linked_files", 0) if images else None,
        "span_s": span_s(scene_dir),
        "scanned": now(),
        "modalities": dict(sorted(mods.items(), key=lambda kv: -(kv[1]["bytes"] + kv[1].get("linked_bytes", 0)))),
        "files": dict(sorted(files.items(), key=lambda kv: -kv[1])),
        "linked_files": dict(sorted(linked_files.items(), key=lambda kv: -kv[1]["bytes"])),
    }


def totals(scenes):
    mods = {}
    for s in scenes.values():
        for k, m in s["modalities"].items():
            t = mods.setdefault(k, {"bytes": 0, "files": 0, "linked_bytes": 0, "linked_files": 0})
            for key in t:
                t[key] += m.get(key, 0)
    mods = dict(sorted(mods.items(), key=lambda kv: -(kv[1]["bytes"] + kv[1]["linked_bytes"])))
    return {
        "bytes": sum(m["bytes"] for m in mods.values()),
        "files": sum(m["files"] for m in mods.values()),
        "linked_bytes": sum(m["linked_bytes"] for m in mods.values()),
        "linked_files": sum(m["linked_files"] for m in mods.values()),
        "broken_links": sum(s.get("broken_links") or 0 for s in scenes.values()),
        "frames": sum(s.get("frames") or 0 for s in scenes.values()),
        "span_s": round(sum(s.get("span_s") or 0 for s in scenes.values()), 3),
        "modalities": {k: modality(m) for k, m in mods.items()},
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
    on_disk = [d.name for d in scene_dirs(root)]
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
    _replace_atomically(path, lambda f: f.write(json.dumps(doc, indent=2) + "\n"))
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
    print(f"{NAME}: {len(doc['scenes'])} scenes, {t['frames']} frames, {t['bytes'] / 1e9:.1f} GB of its own")
    if t["linked_files"]:
        print(f"  and {t['linked_bytes'] / 1e9:.1f} GB in {t['linked_files']} files reached through links")
    if t["broken_links"]:
        print(f"  {t['broken_links']} broken links")
    for k, m in t["modalities"].items():
        share = f"{100 * m['bytes'] / t['bytes']:5.1f}%" if t["bytes"] else "     -"
        linked = f"   + {m['linked_bytes'] / 1e9:.1f} GB linked" if m.get("linked_files") else ""
        print(f"  {k:14s}{m['bytes'] / 1e9:9.1f} GB {share}{linked}")


if __name__ == "__main__":
    main()
