#!/usr/bin/env python3
import sys
from rosbag2_py import (
    SequentialReader,
    SequentialWriter,
    StorageOptions,
    ConverterOptions,
    TopicMetadata,
)
from rclpy.serialization import serialize_message, deserialize_message
from rosidl_runtime_py.utilities import get_message


def sort_bag(in_bag, out_bag):
    # --- Open reader ---
    reader = SequentialReader()
    storage_options_in = StorageOptions(uri=in_bag, storage_id="mcap")
    converter_options = ConverterOptions(
        input_serialization_format="cdr", output_serialization_format="cdr"
    )
    reader.open(storage_options_in, converter_options)

    topic_types = reader.get_all_topics_and_types()

    def typename(topic_name):
        for topic_type in topic_types:
            if topic_type.name == topic_name:
                return topic_type.type
        raise ValueError(f"topic {topic_name} not in bag")

    # --- Load all messages ---
    messages = []
    topics = {}
    while reader.has_next():
        topic, data, ts = reader.read_next()

        try:
            topic_type = typename(topic)
            msg_type = get_message(topic_type)
        except ModuleNotFoundError:
            continue
        if topic not in topics.keys():
            topics[topic] = topic_type
        msg = deserialize_message(data, msg_type)

        # Prefer header timestamp if present
        if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
            t = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
        else:
            t = ts

        messages.append((t, topic, msg))

    print(f"Loaded {len(messages)} messages. Sorting…")

    # Sort by timestamp
    messages.sort(key=lambda x: x[0])

    # --- Write new sorted bag ---
    writer = SequentialWriter()
    storage_options_out = StorageOptions(uri=out_bag, storage_id="mcap")
    writer.open(storage_options_out, converter_options)

    # Register topics
    for idx, topic in enumerate(topics.keys()):
        writer.create_topic(
            TopicMetadata(
                id=idx,
                name=topic,
                type=topics[topic],
                serialization_format="cdr",
            )
        )

    # Write sorted messages
    for t, topic, msg in messages:
        writer.write(topic, serialize_message(msg), int(t))

    print("Done. Output written to:", out_bag)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: sort_mcap_bag.py input.mcap output_sorted.mcap")
        sys.exit(1)

    sort_bag(sys.argv[1], sys.argv[2])
