"""Compatibility patches for the ``rosbags`` library.

Some HELHEST mcap bags (old sensor generation, recorded with rosbag2 on ROS 2
Jazzy) contain a schema record whose encoding is ``"unknown"`` (seen for
``action_msgs/msg/GoalStatusArray``). ``rosbags`` <= 0.11 raises ``KeyError``
for any encoding it does not know while *opening* the bag, which makes the
whole recording unreadable even though the topic is never used.

Importing this module makes ``rosbags`` treat unrecognised schema encodings as
"no definition" so the bag opens; such topics deserialise via the built-in
typestore if the type is known there, and are simply unusable otherwise.
"""

from __future__ import annotations

from rosbags.rosbag2 import storage_mcap

_KNOWN_ENCODINGS = {"ros2msg", "ros2idl", "omgidl", ""}


def _patch_schema() -> None:
    orig = storage_mcap.Schema
    if getattr(orig, "_helhest_patched", False):
        return

    class Schema(orig):  # type: ignore[misc,valid-type]
        """``rosbags`` Schema that tolerates unknown encodings."""

        __slots__ = ()
        _helhest_patched = True

        def __new__(cls, id: int, name: str, encoding: str, data: str):  # noqa: A002
            if encoding not in _KNOWN_ENCODINGS:
                encoding = ""
            return super().__new__(cls, id, name, encoding, data)

    storage_mcap.Schema = Schema


_patch_schema()
