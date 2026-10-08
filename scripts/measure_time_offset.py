#!/usr/bin/env python3
"""Audit an extracted scene's frame clock against its poses.

The extractor applies this itself (``--time_align auto``); this checks a root on
disk. A corrected scene reads as a flat profile at zero. See :mod:`dataset_tools.time_align`.

    scripts/measure_time_offset.py <root-or-scene>... [--json out.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dataset_tools import time_align as ta
from dataset_tools.gt_io import read_trajectory


def describe(profile: ta.Profile) -> str:
    """One line: what the scene's clock is doing, and how firmly that is known."""
    if profile.is_identity:
        return f"no correction wanted -- {profile.note}"
    lo, hi = profile.knot_delta.min(), profile.knot_delta.max()
    knots = f"{len(profile.knot_ts)}/{len(profile.windows)} windows"
    if hi - lo < 0.05:
        return f"{float(np.median(profile.knot_delta)):+.3f} s, flat ({knots})"
    return f"{lo:+.3f} .. {hi:+.3f} s, VARYING by {hi - lo:.3f} s over the scene ({knots})"


def measure(scene: Path, limit: int, span: float) -> ta.Profile | None:
    if not all((scene / n).exists() for n in ("data_ts.txt", "odometry.txt", "clouds")):
        return None
    frame_ts = np.loadtxt(scene / "data_ts.txt")
    ts, poses = read_trajectory(scene / "odometry.txt")
    return ta.estimate(scene, frame_ts, ts, poses, limit=limit, span=span)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenes", nargs="+", type=Path, help="Scene directories, or a dataset root")
    ap.add_argument("--windows", type=int, default=ta.MAX_WINDOWS, help="Windows per scene")
    ap.add_argument("--span", type=float, default=1.5, help="Half-width of the coarse sweep, seconds")
    ap.add_argument("--per-window", action="store_true", help="Print every window, not just the summary")
    ap.add_argument("--json", type=Path, default=None, help="Write the per-window detail here")
    args = ap.parse_args(argv)

    scenes: list[Path] = []
    for p in args.scenes:
        if (p / "scenes.csv").is_file():
            scenes.extend(sorted(q for q in p.iterdir() if (q / "data_ts.txt").is_file()))
        else:
            scenes.append(p)

    report = {}
    for scene in scenes:
        profile = measure(scene, args.windows, args.span)
        if profile is None:
            print(f"{scene.name}: not an extracted scene with ground truth, skipped")
            continue
        t0 = float(np.loadtxt(scene / "data_ts.txt")[0])
        print(f"{scene.name}: {describe(profile)}")
        if args.per_window:
            for t, delta, gain, depth in profile.rows(t0):
                knot = "K" if depth >= ta.MIN_DEPTH and abs(delta) <= ta.MAX_ABS_OFFSET else " "
                print(f"    t={t:8.1f}s {knot} {delta:+.3f} s   gain {gain:5.1f}%   depth {depth:5.1f}%")
        report[scene.name] = {
            "summary": profile.summary,
            "is_identity": profile.is_identity,
            "knots": [[float(a), float(b)] for a, b in zip(profile.knot_ts - t0, profile.knot_delta, strict=True)],
            "windows": profile.rows(t0),
        }

    if args.json:
        args.json.write_text(json.dumps(report, indent=1))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
