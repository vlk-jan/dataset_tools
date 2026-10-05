#!/usr/bin/env python3
"""Every publisher's clock against the recorder's, straight out of a bag.

A rosbag2 message carries two times: the ``header.stamp`` its publisher wrote,
and the time the recorder received it. Their difference is transport latency
plus however far apart the two clocks are, so a publisher whose clock steps
shows up as a step in that difference at the moment it happens -- no
estimation, no geometry, nothing to interpret.

This is the direct reading of what ``scripts/measure_time_offset.py`` can only
infer from the data. Use it to find out *when* a sensor's clock moved, and use
that to decide whether one offset per bag is a fair model of the recording.

    scripts/probe_clock_skew.py <bag> --bin 5 --csv skew.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader

from dataset_tools import rosbags_compat  # noqa: F401  (patches rosbags)
from dataset_tools.extract.pointcloud import header_stamp_from_cdr
from dataset_tools.transforms import header_stamp

#: Topics worth probing on the 2026-09 rig: the sensor unit, the pose source,
#: and the robot's own controller, which is the recorder's own clock domain.
DEFAULT_TOPICS = (
    "/odin1/image/compressed",
    "/odin1/cloud_filtered/cloudini",
    "/fixposition/odometry_enu",
    "/joint_states",
)
#: A step this large is a clock moving, not jitter in the transport.
STEP_S = 0.15


def collect(bag: Path, topics: list[str]) -> dict[str, np.ndarray]:
    """``{topic: (N, 2) [recv_time, recv - header]}``, in recording order."""
    rows: dict[str, list[tuple[float, float]]] = {t: [] for t in topics}
    with AnyReader([bag]) as reader:
        conns = [c for c in reader.connections if c.topic in rows]
        if not conns:
            raise SystemExit(f"none of {topics} is in {bag}")
        for conn, recv_ns, raw in reader.messages(conns):
            recv = recv_ns * 1e-9
            stamp = header_stamp_from_cdr(raw)
            if stamp is None:
                stamp = header_stamp(reader.deserialize(raw, conn.msgtype), float("nan"))
            if stamp and np.isfinite(stamp):
                rows[conn.topic].append((recv, recv - stamp))
    return {t: np.asarray(v, dtype=float) for t, v in rows.items() if v}


def binned(skew: np.ndarray, t0: float, width: float) -> list[tuple[float, float, int]]:
    """``(seconds into the bag, median skew, count)`` per time bin."""
    if not len(skew):
        return []
    idx = ((skew[:, 0] - t0) // width).astype(int)
    out = []
    for b in np.unique(idx):
        m = idx == b
        out.append((b * width, float(np.median(skew[m, 1])), int(m.sum())))
    return out


def steps(series: list[tuple[float, float, int]], threshold: float) -> list[tuple[float, float]]:
    """``(seconds into the bag, jump)`` wherever the median skew moves."""
    return [
        (series[k][0], series[k][1] - series[k - 1][1])
        for k in range(1, len(series))
        if abs(series[k][1] - series[k - 1][1]) >= threshold
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", type=Path)
    ap.add_argument("--topics", nargs="+", default=list(DEFAULT_TOPICS))
    ap.add_argument("--bin", type=float, default=5.0, metavar="SECONDS", help="Bin width (default 5 s)")
    ap.add_argument(
        "--step", type=float, default=STEP_S, metavar="SECONDS", help=f"Report jumps this big (default {STEP_S})"
    )
    ap.add_argument("--csv", type=Path, default=None, help="Write every bin here")
    args = ap.parse_args(argv)

    data = collect(args.bag, args.topics)
    t0 = min(v[0, 0] for v in data.values())

    series = {t: binned(v, t0, args.bin) for t, v in data.items()}
    for topic, s in series.items():
        med = np.array([x[1] for x in s])
        print(f"\n{topic}  ({len(data[topic])} messages)")
        print(f"  recv - header: median {np.median(med):+.3f} s, range {med.min():+.3f} .. {med.max():+.3f} s")
        found = steps(s, args.step)
        if not found:
            print("  no step: this clock held for the whole bag")
        for when, jump in found:
            print(f"  STEP at t = {when:7.1f} s: {jump:+.3f} s")

    if args.csv:
        with args.csv.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["topic", "t_into_bag_s", "median_recv_minus_header_s", "messages"])
            for topic, s in series.items():
                for when, med, n in s:
                    w.writerow([topic, f"{when:.1f}", f"{med:.6f}", n])
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
