#!/usr/bin/env python3
import argparse
from pathlib import Path

from rosbags.highlevel import AnyReader
from rosbags.rosbag2 import Writer


def sort_bag(input_path: Path, output_path: Path):
    """Sort ROS2 bag messages by timestamp.

    Args:
        input_path: Path to the input bag (directory or mcap file).
        output_path: Path to the output bag directory.
    """
    messages = []

    print(f"Reading messages from {input_path}...")
    with AnyReader([input_path]) as reader:
        # Map to keep track of connection metadata for the writer
        connections_info = {}
        for conn in reader.connections:
            connections_info[conn.id] = (conn.topic, conn.msgtype)

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

    # Ensure output directory doesn't exist to avoid rosbags Writer errors
    if output_path.exists():
        import shutil

        print(f"Warning: Output path {output_path} exists. Removing it.")
        shutil.rmtree(output_path)

    with Writer(output_path) as writer:
        # Create connections in the new bag
        topic_to_conn = {}

        for t, topic, msgtype, rawdata in messages:
            if topic not in topic_to_conn:
                topic_to_conn[topic] = writer.add_connection(topic, msgtype)

            # Write the raw data directly to avoid unnecessary re-serialization
            writer.write(topic_to_conn[topic], t, rawdata)

    print(f"Successfully wrote sorted bag to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Sort a ROS2 bag by header or bag timestamp."
    )
    parser.add_argument("input", type=str, help="Path to input ROS2 bag")
    parser.add_argument(
        "output", type=str, help="Path to output sorted bag (directory)"
    )

    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        print(f"Error: Input path {in_path} does not exist.")
        return

    sort_bag(in_path, out_path)


if __name__ == "__main__":
    main()
