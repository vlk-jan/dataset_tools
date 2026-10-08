#!/usr/bin/env python3
"""Rewrite a ROS 2 bag with its messages ordered by header timestamp."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from rosbags.highlevel import AnyReader
from rosbags.rosbag2 import Writer

from dataset_tools import rosbags_compat  # noqa: F401  (patches rosbags)
from dataset_tools.transforms import MIN_VALID_STAMP


def sort_bag(input_path: Path, output_path: Path, overwrite: bool = False) -> int:
    """Sort ROS2 bag messages by header timestamp (whole bag held in memory).

    Args:
        input_path: Path to the input bag (directory or mcap file).
        output_path: Path to the output bag directory. Refused if it exists,
            unless ``overwrite`` is set, in which case it is removed first.

    Returns:
        The number of messages written.
    """
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")
        print(f"Removing existing {output_path}...")
        shutil.rmtree(output_path)

    messages = []

    print(f"Reading messages from {input_path}...")
    with AnyReader([input_path]) as reader, Writer(output_path, version=8) as writer:
        # Connections first, so a schema nothing can define fails before the bag is loaded;
        # the source's definition, hash, QoS and format carry over (/tf_static stays latched).
        conns = {}
        for c in reader.connections:
            if (c.topic, c.msgtype) in conns:
                continue
            kw = dict(serialization_format=c.ext.serialization_format, offered_qos_profiles=c.ext.offered_qos_profiles)
            if c.msgdef.data and c.digest:
                kw.update(msgdef=c.msgdef.data, rihs01=c.digest)
            else:
                kw["typestore"] = reader.typestore
            conns[c.topic, c.msgtype] = writer.add_connection(c.topic, c.msgtype, **kw)

        for connection, timestamp, rawdata in reader.messages():
            # Readers order by the bag timestamp, so the header stamp is written there.
            # An unset stamp (zero or boot clock) keeps the receive time, as the extractor does.
            t = timestamp
            try:
                msg = reader.deserialize(rawdata, connection.msgtype)
                stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
                if stamp >= MIN_VALID_STAMP * 10**9:
                    t = stamp
            except Exception:  # noqa: BLE001  (no header, or undecodable: receive time)
                pass
            messages.append((t, connection.topic, connection.msgtype, rawdata))

        print(f"Loaded {len(messages)} messages. Sorting...")
        messages.sort(key=lambda x: x[0])

        print(f"Writing sorted messages to {output_path}...")
        for t, topic, msgtype, rawdata in messages:
            writer.write(conns[topic, msgtype], t, rawdata)

    print(f"Successfully wrote sorted bag to: {output_path}")
    return len(messages)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sort a ROS2 bag by header or bag timestamp. "
        "The whole bag is held in memory while it is sorted.",
    )
    parser.add_argument("input", type=str, help="Path to input ROS2 bag")
    parser.add_argument("output", type=str, help="Path to output sorted bag (directory)")
    parser.add_argument("--overwrite", action="store_true", help="replace the output directory if it exists")

    args = parser.parse_args(argv)

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        print(f"Error: Input path {in_path} does not exist.", file=sys.stderr)
        return 2

    try:
        sort_bag(in_path, out_path, overwrite=args.overwrite)
    except FileExistsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
