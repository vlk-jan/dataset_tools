"""Compatibility patch for ``rosbags``: some old-rig mcap bags carry a schema with
encoding ``"unknown"``, on which ``rosbags`` <= 0.11 refuses to open the bag.
Importing this module treats unknown encodings as "no definition".
"""

from __future__ import annotations

from rosbags.rosbag2 import storage_mcap

_KNOWN_ENCODINGS = {"ros2msg", "ros2idl", "omgidl", ""}


def _patch_schema() -> None:
    orig = storage_mcap.Schema

    class Schema(orig):  # type: ignore[misc,valid-type]
        """``rosbags`` Schema that tolerates unknown encodings."""

        __slots__ = ()

        def __new__(cls, id: int, name: str, encoding: str, data: str):  # noqa: A002
            if encoding not in _KNOWN_ENCODINGS:
                encoding = ""
            return super().__new__(cls, id, name, encoding, data)

    storage_mcap.Schema = Schema


_patch_schema()
