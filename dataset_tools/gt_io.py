"""Reading and writing the ground-truth odometry files, in one place.

Two text files make up a scene's ground truth (the HELHEST v1.2 contract):

``odometry.txt``
    One pose per line: ``timestamp`` followed by a row-major 4x4 matrix. The
    17-column form is what every writer here produces; the 13-column form
    (timestamp + the top 3x4 only) is accepted on read.
``odometry_segments.txt``
    ``segment start_ts end_ts n_poses reason`` per line. Poses from different
    segments share no reference frame -- each segment is re-anchored to the
    identity at its own first pose -- so a reader must never interpolate across
    a segment boundary.

The extractor, the validator, the viewer and the post-processing scripts all
go through these functions, so the format is spelled out exactly once.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import os

import numpy as np

GT_HEADER = "# Timestamp(sec) r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz 0 0 0 1"
SEGMENTS_HEADER = "# segment start_ts end_ts n_poses reason"


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
    that is neither 17 nor 13 columns wide.
    """
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", UserWarning
        )  # numpy's "input contained no data": an empty file is a result here
        raw = np.loadtxt(path, dtype=np.float64, comments="#", ndmin=2)
    if raw.size == 0:
        return np.zeros(0), np.zeros((0, 4, 4))
    if raw.shape[1] == 17:
        poses = raw[:, 1:].reshape(-1, 4, 4)
    elif raw.shape[1] == 13:
        poses = np.zeros((raw.shape[0], 4, 4), dtype=np.float64)
        poses[:, :3, :] = raw[:, 1:].reshape(-1, 3, 4)
        poses[:, 3, 3] = 1.0
    else:
        raise ValueError(f"{path}: expected 17 or 13 columns, got {raw.shape[1]}")
    order = np.argsort(raw[:, 0], kind="stable")
    return raw[order, 0], poses[order]


def format_pose_line(t: float, T: np.ndarray) -> str:
    """One ``odometry.txt`` line, ``%.6f`` throughout (byte-compatible with v1)."""
    return " ".join(f"{v:.6f}" for v in (t, *np.asarray(T).ravel()))


def _replace_atomically(dest, writer) -> None:
    """Write through a temp file in the same directory and ``os.replace`` it in.

    A root that is being read by a training job at the time of an in-place edit
    must never expose a half-written file: a DataLoader worker that opened
    ``odometry.txt`` mid-truncate on 2026-09-09 got zero bytes and killed a
    nine-epoch arm. ``os.replace`` is atomic on POSIX, so a reader sees the old
    file or the new one, never a partial.
    """
    dest = Path(dest)
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as fh:
        writer(fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, dest)


def write_trajectory(dest: Path | str | TextIO, ts: Iterable[float], poses: Iterable[np.ndarray]) -> None:
    """Write header + one line per pose to a path or an open text handle."""
    if hasattr(dest, "write"):
        _write_trajectory(dest, ts, poses)  # type: ignore[arg-type]
        return
    ts, poses = list(ts), list(poses)
    _replace_atomically(dest, lambda fh: _write_trajectory(fh, ts, poses))


def _write_trajectory(fh: TextIO, ts: Iterable[float], poses: Iterable[np.ndarray]) -> None:
    fh.write(GT_HEADER + "\n")
    for t, T in zip(ts, poses, strict=False):
        fh.write(format_pose_line(t, T) + "\n")


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
    """Every parseable row, in file order. Raises ``OSError`` if unreadable."""
    rows = []
    with open(path) as fh:
        for line in fh:
            row = parse_segment_line(line)
            if row is not None:
                rows.append(row)
    return rows


def format_segment_line(row: SegmentRow) -> str:
    return f"{row.index} {row.start_ts:.6f} {row.end_ts:.6f} {row.n_poses} {row.reason}".rstrip()


def write_segments(dest: Path | str | TextIO, rows: Iterable[SegmentRow]) -> None:
    """Write header + one line per segment to a path or an open text handle."""
    if hasattr(dest, "write"):
        _write_segments(dest, rows)  # type: ignore[arg-type]
        return
    rows = list(rows)
    _replace_atomically(dest, lambda fh: _write_segments(fh, rows))


def _write_segments(fh: TextIO, rows: Iterable[SegmentRow]) -> None:
    fh.write(SEGMENTS_HEADER + "\n")
    for row in rows:
        fh.write(format_segment_line(row) + "\n")
