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


def sort_bag(input_path: Path, output_path: Path, overwrite: bool = False) -> int:
    """Sort ROS2 bag messages by timestamp.

    Every message is held in memory until the last one has been read, so the
    machine needs RAM for the whole bag; the tool is meant for the odd
    recording whose order is wrong, not for a pipeline.

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
    with AnyReader([input_path]) as reader:
        typestore = reader.typestore

        for connection, timestamp, rawdata in reader.messages():
            # Deserialization is needed to check for header timestamps
            try:
                msg = reader.deserialize(rawdata, connection.msgtype)
                # Prefer header timestamp if present (nanoseconds)
                if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
                    t = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
                else:
                    t = timestamp
            except Exception:
                # Fallback to bag timestamp if deserialization fails
                t = timestamp

            messages.append((t, connection.topic, connection.msgtype, rawdata))

    print(f"Loaded {len(messages)} messages. Sorting...")
    # Sort by the determined timestamp
    messages.sort(key=lambda x: x[0])

    print(f"Writing sorted messages to {output_path}...")

    with Writer(output_path, version=8) as writer:
        # Create connections in the new bag
        topic_to_conn = {}

        for t, topic, msgtype, rawdata in messages:
            if topic not in topic_to_conn:
                topic_to_conn[topic] = writer.add_connection(topic, msgtype, typestore=typestore)

            # Write the raw data directly to avoid unnecessary re-serialization
            writer.write(topic_to_conn[topic], t, rawdata)

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
