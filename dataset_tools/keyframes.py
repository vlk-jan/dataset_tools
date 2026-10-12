"""The root's three keyframe lists: ``keyframes.lst``, ``keyframes_train.lst``, ``keyframes_heldout.lst``.

    python -m dataset_tools.keyframes ROOT [--max_tilt_deg 45 --tilt_horizon 4]

One line per frame that has a cloud, an image, a height map and a mask::

    <scene>/clouds/NNNNN.npy <scene>/images/NNNNN.<ext>

minus every frame whose next ``tilt_horizon`` seconds hold a frame tilted more
than ``max_tilt_deg`` from level (``gravity_align``): the robot tipped or on
its back, whose wheel contacts are above the ground. ``keyframes.lst`` lists
every scene, ``keyframes_train.lst`` all but ``manifest.json`` ->
``held_out.scenes``, ``keyframes_heldout.lst`` only those.

Two columns always, though consumers read the first: ``np.loadtxt(...,
dtype=str)[:, 0]`` on a one-column file raises. The image extension is the one
the manifest declares (``images.extension``; none declared means png), and a
declared extension that matches no file is refused rather than written as an
empty list.

Until 2026-10 this was DRIFT's ``scripts/data/write_keyframes.py``, run three
times per root with ``--max_tilt_deg 45 --tilt_horizon 4.0``; same gate, same
lines.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from dataset_tools import image_format

LISTS = ("keyframes.lst", "keyframes_train.lst", "keyframes_heldout.lst")
MAX_TILT_DEG = 45.0
TILT_HORIZON_S = 4.0


def held_out(root: Path) -> list[str]:
    """``manifest.json`` -> ``held_out.scenes``, as four-digit ids; empty when there is none."""
    try:
        m = json.loads((Path(root) / "manifest.json").read_text())
    except (OSError, ValueError):
        return []
    scenes = (m.get("held_out") or {}).get("scenes") or []
    return [str(s).zfill(4) for s in scenes]


def tilt_exposed(scene_dir: Path, max_tilt_deg: float, horizon_s: float) -> np.ndarray:
    """Per frame: whether ``[t, t + horizon_s]`` holds a frame tilted more than ``max_tilt_deg``."""
    g = np.load(scene_dir / "gravity_align.npy", mmap_mode="r")
    tilt = np.degrees(np.arccos(np.clip(np.asarray(g[:, 2, 2]), -1.0, 1.0)))
    bad = tilt > max_tilt_deg
    if horizon_s <= 0 or not bad.any():
        return bad
    clock = scene_dir / "data_ts.txt"
    ts = np.loadtxt(clock if clock.exists() else scene_dir / "pcd_ts.txt")
    bad_t = ts[bad]
    j = np.minimum(np.searchsorted(bad_t, ts, side="left"), len(bad_t) - 1)
    return bad | (bad_t[j] <= ts + horizon_s) & (bad_t[j] >= ts)


def scene_lines(scene_dir: Path, ext: str, max_tilt_deg: float | None, horizon_s: float) -> tuple[list[str], dict]:
    """The lines of one scene and what was left out."""
    sc = scene_dir.name

    def stems(sub, suffix):
        d = scene_dir / sub
        return {f[: -len(suffix)] for f in os.listdir(d) if f.endswith(suffix)} if d.is_dir() else set()

    clouds = sorted(stems("clouds", ".npy"))
    have = stems("images", "." + ext) & stems("height_maps", ".npz") & stems("height_masks", ".npz")
    exposed = tilt_exposed(scene_dir, max_tilt_deg, horizon_s) if max_tilt_deg is not None else None
    lines, missing, tilted = [], 0, 0
    for stem in clouds:
        if stem not in have:
            missing += 1
        elif exposed is not None and int(stem) < len(exposed) and exposed[int(stem)]:
            tilted += 1
        else:
            lines.append(f"{sc}/clouds/{stem}.npy {sc}/images/{stem}.{ext}")
    return lines, {"listed": len(lines), "no_map_mask_or_image": missing, "tilted": tilted}


def write_lists(
    root: Path,
    out_dir: Path | None = None,
    max_tilt_deg: float | None = MAX_TILT_DEG,
    horizon_s: float = TILT_HORIZON_S,
    held: list[str] | None = None,
    ext: str | None = None,
) -> dict:
    """Write the three lists into ``out_dir`` (the root by default); returns counts per list and scene.

    ``held`` and ``ext`` default to what the root's manifest declares.
    """
    root = Path(root)
    out_dir = Path(out_dir) if out_dir else root
    ext = ext or image_format.declared(root)
    declared = ext is not None
    ext = (ext or "png").lstrip(".")
    val = set(held_out(root) if held is None else (str(s).zfill(4) for s in held))
    scenes = sorted(d for d in os.listdir(root) if len(d) == 4 and d.isdigit() and (root / d / "clouds").is_dir())
    if missing := sorted(val - set(scenes)):
        raise SystemExit(f"{root}: held_out.scenes names {missing}, which are not on disk")
    per, lines = {}, {}
    for sc in scenes:
        lines[sc], per[sc] = scene_lines(root / sc, ext, max_tilt_deg, horizon_s)
    if declared and not any(lines.values()):
        raise SystemExit(
            f"{root}/manifest.json declares images.extension={ext!r} but no frame has an images/<stem>.{ext}; "
            "refusing to write lists that name nothing"
        )
    chosen = {
        "keyframes.lst": scenes,
        "keyframes_train.lst": [s for s in scenes if s not in val],
        "keyframes_heldout.lst": [s for s in scenes if s in val],
    }
    totals = {}
    for name, ss in chosen.items():
        body = [ln for s in ss for ln in lines[s]]
        path = out_dir / name
        tmp = path.with_suffix(".lst.tmp")
        tmp.write_text("\n".join(body) + "\n")
        os.replace(tmp, path)
        totals[name] = len(body)
    return {"lists": totals, "scenes": per, "held_out": sorted(val), "extension": ext}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--out_dir", type=Path, default=None, help="write the lists here instead of into the root")
    ap.add_argument("--max_tilt_deg", type=float, default=MAX_TILT_DEG, help="negative: no tilt gate")
    ap.add_argument("--tilt_horizon", type=float, default=TILT_HORIZON_S)
    a = ap.parse_args(argv)
    r = write_lists(a.root, a.out_dir, a.max_tilt_deg if a.max_tilt_deg >= 0 else None, a.tilt_horizon)
    print(f"images .{r['extension']}; held out: {', '.join(r['held_out']) or 'none'}")
    for name, n in r["lists"].items():
        print(f"{name}: {n} lines")
    gaps = {s: c for s, c in r["scenes"].items() if c["no_map_mask_or_image"] or c["tilted"]}
    for s, c in gaps.items():
        print(f"  {s}: {c['no_map_mask_or_image']} without map, mask or image; {c['tilted']} tilt-gated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
