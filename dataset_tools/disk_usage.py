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

A root may be built from symbolic links into another root (HELHEST v1.3 into
v1.2, v2.1 into v2.0). What a link leads to is not this root's disk usage, but
it is this root's content, so the two are counted apart: ``bytes`` and
``files`` are what the root stores itself, ``linked_bytes`` and
``linked_files`` what it reaches through a link -- a linked directory, a linked
file, or a link inside a directory of its own. ``frames`` counts both, because
a linked frame is a frame. A link whose target is gone is counted under
``broken_links`` and nowhere else.

Layout, flat so a chart reads it without reshaping:

    {"schema": 3, "root": "HELHEST_v2_0", "generated": "<iso>", "unit": "bytes",
     "modalities": ["images", "clouds", ...],            # largest first, own + linked
     "totals": {"bytes": .., "files": .., "linked_bytes": .., "linked_files": ..,
                "broken_links": .., "frames": .., "span_s": ..,
                "modalities": {"images": {"bytes": .., "files": ..}, ...}},
     "scenes": {"0000": {"bytes": .., "linked_bytes": .., "broken_links": ..,
                         "frames": .., "span_s": .., "scanned": "<iso>",
                         "modalities": {"images": {"bytes": .., "files": ..}, ...},
                         "files": {"odometry.txt": <bytes>, ...},
                         "linked_files": {"gps.npy": {"bytes": .., "link": "<target>"}}}}}

A modality that reaches anything through a link carries ``linked_bytes`` and
``linked_files`` beside ``bytes`` and ``files``, and ``link`` with the target
when the directory itself is the link; one that does not has the two keys only.

Reads the scenes only; writes the one file, atomically.
"""

import argparse
import json
import logging
import os
import stat
from datetime import datetime, timezone

NAME = "disk_usage.json"
SCHEMA = 3
LOG = logging.getLogger(__name__)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def dir_usage(path, linked=False):
    """What is under ``path``: own and linked regular files, and broken links.

    ``linked`` says ``path`` was itself reached through a link, so everything
    under it is linked. Directory links are followed; one that leads back into
    a directory already visited is not, so a cycle ends.
    """
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
