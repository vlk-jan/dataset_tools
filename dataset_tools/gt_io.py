"""Reading and writing the ground-truth odometry files of a HELHEST root.

``odometry.txt``: ``timestamp`` + row-major 4x4 pose, 17 columns per line.
``odometry_segments.txt``: ``segment start_ts end_ts n_poses reason``. Each
segment is anchored at its own first pose; never interpolate across segments.
"""

from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np

LOG = logging.getLogger("helhest.gt_io")

GT_HEADER = "# Timestamp(sec) r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz 0 0 0 1"
SEGMENTS_HEADER = "# segment start_ts end_ts n_poses reason"

#: First segment number in written tables. Tables that start at zero are read as written;
#: segments are always addressed by their number in the file, not their position.
FIRST_SEGMENT = 1


@dataclass
class SegmentRow:
    """One line of ``odometry_segments.txt``."""

    index: int
    start_ts: float
    end_ts: float
    n_poses: int = 0
    reason: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end_ts - self.start_ts)


# ---------------------------------------------------------------------------
# odometry.txt
# ---------------------------------------------------------------------------
def read_trajectory(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """``(ts (N,), poses (N, 4, 4))`` sorted by timestamp; empty arrays for an empty file.

    Raises ``OSError`` for an unreadable file and ``ValueError`` for a file
    that is not 17 columns wide.
    """
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", UserWarning
        )  # numpy's "input contained no data": an empty file is a result here
        raw = np.loadtxt(path, dtype=np.float64, comments="#", ndmin=2)
    if raw.size == 0:
        return np.zeros(0), np.zeros((0, 4, 4))
    if raw.shape[1] != 17:
        raise ValueError(f"{path}: expected 17 columns, got {raw.shape[1]}")
    order = np.argsort(raw[:, 0], kind="stable")
    return raw[order, 0], raw[order, 1:].reshape(-1, 4, 4)


def write_tmp(dest: Path | str, writer: Callable[[TextIO], None], mode: str = "w") -> Path:
    """``dest``'s content in a sibling temp file, fsynced; the caller ``os.replace``s it in."""
    dest = Path(dest)
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, mode) as fh:
            writer(fh)
            fh.flush()
            os.fsync(fh.fileno())
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return tmp


def _replace_atomically(dest: Path | str, writer: Callable[[TextIO], None]) -> None:
    """Write through a temp file and ``os.replace`` it in, so concurrent readers never see a partial file."""
    os.replace(write_tmp(dest, writer), dest)


def write_trajectory(dest: Path | str, ts: Iterable[float], poses: Iterable[np.ndarray]) -> None:
    """Write header + one line per pose, atomically; lengths must match."""
    ts, poses = list(ts), list(poses)
    if len(ts) != len(poses):
        raise ValueError(f"{len(ts)} timestamps for {len(poses)} poses")
    _replace_atomically(dest, lambda fh: _write_trajectory(fh, ts, poses))


def _write_trajectory(fh: TextIO, ts: Iterable[float], poses: Iterable[np.ndarray]) -> None:
    fh.write(GT_HEADER + "\n")
    for t, T in zip(ts, poses, strict=True):  # %.6f throughout
        fh.write(" ".join(f"{v:.6f}" for v in (t, *np.asarray(T).ravel())) + "\n")


def anchor_to_first(poses: np.ndarray) -> np.ndarray:
    """Re-express a ``(N, 4, 4)`` run so its first pose is the identity."""
    return np.einsum("ij,njk->nik", np.linalg.inv(poses[0]), poses)


# ---------------------------------------------------------------------------
# odometry_segments.txt
# ---------------------------------------------------------------------------
def parse_segment_line(line: str) -> SegmentRow | None:
    """One data line -> ``SegmentRow``, or None for a comment, blank or malformed line.

    Trailing columns beyond the fourth are folded into ``reason``.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split()
    if len(parts) < 3:
        return None
    try:
        index = int(float(parts[0]))
        start, end = float(parts[1]), float(parts[2])
    except ValueError:
        return None
    n_poses, reason = 0, ""
    if len(parts) >= 4:
        try:
            n_poses = int(float(parts[3]))
            reason = " ".join(parts[4:])
        except ValueError:
            reason = " ".join(parts[3:])
    return SegmentRow(index, start, end, n_poses, reason)


def read_segments(path: Path | str) -> list[SegmentRow]:
    """Every parseable row, in file order (malformed lines warned about). Raises ``OSError`` if unreadable."""
    rows = []
    with open(path) as fh:
        for number, line in enumerate(fh, 1):
            row = parse_segment_line(line)
            if row is not None:
                rows.append(row)
            elif line.strip() and not line.lstrip().startswith("#"):
                LOG.warning("%s:%d: not a segment row, skipped: %s", path, number, line.rstrip())
    return rows


def write_segments(dest: Path | str, rows: Iterable[SegmentRow]) -> None:
    """Write header + one line per segment, atomically."""
    rows = list(rows)
    _replace_atomically(dest, lambda fh: _write_segments(fh, rows))


def _write_segments(fh: TextIO, rows: Iterable[SegmentRow]) -> None:
    """Renumbers contiguously from :data:`FIRST_SEGMENT`, whatever the callers filled in."""
    fh.write(SEGMENTS_HEADER + "\n")
    for k, row in enumerate(rows):
        fh.write(f"{FIRST_SEGMENT + k} {row.start_ts:.6f} {row.end_ts:.6f} {row.n_poses} {row.reason}".rstrip() + "\n")
