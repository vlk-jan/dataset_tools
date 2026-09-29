#!/usr/bin/env python3
"""Extract HELHEST scenes from ROS 2 bags (rosbag2 sqlite3 or mcap).

One scene directory is written per input bag; see README.md for the exact
output format. Bags are read with the pure-Python ``rosbags`` library, so no
ROS installation is required.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import logging
import math
import re
import shutil
import struct
import sys
import traceback
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
from rosbags.highlevel import AnyReader
from tqdm import tqdm

try:
    from dataset_tools import (  # noqa: F401  (rosbags_compat patches rosbags)
        cloudini,
        disk_usage,
        gnss_guard,
        rosbags_compat,
        time_align,
    )
    from dataset_tools.gt_io import SegmentRow, anchor_to_first, write_segments, write_trajectory
    from dataset_tools.transforms import (  # noqa: F401  (re-exported: the transform helpers used to live here)
        euler_matrix,
        pose_to_matrix,
        quaternion_matrix,
        roll_pitch_from_matrix,
        roll_pitch_from_quat,
    )
except ImportError as exc:  # when the file is run directly from dataset_tools/
    # Only the package itself being unimportable takes the fallback; an import
    # failing *inside* one of these modules must surface as what it is rather
    # than as "No module named 'cloudini'".
    if exc.name != "dataset_tools":
        raise
    import cloudini
    import disk_usage
    import gnss_guard
    import rosbags_compat  # noqa: F401
    import time_align
    from gt_io import SegmentRow, anchor_to_first, write_segments, write_trajectory
    from transforms import (  # noqa: F401
        euler_matrix,
        pose_to_matrix,
        quaternion_matrix,
        roll_pitch_from_matrix,
        roll_pitch_from_quat,
    )

LOG = logging.getLogger("helhest.extract")

SCENE_DIR_RE = re.compile(r"^\d{4,}$")
#: What ``--cam`` may point at: pass 2 decodes the payload with OpenCV.
IMAGE_MSGTYPES = ("sensor_msgs/msg/CompressedImage",)
#: Written as the very last step of a scene, so its presence marks completion.
SENTINEL_FILE = "source_bag.txt"
SCENES_CSV = "scenes.csv"
SCENES_PARTS_DIR = ".scenes_parts"
SCENES_CSV_HEADER = ["scene", "bag", "start_time", "frames"]

STATUS_OK = "ok"
STATUS_SKIPPED = "skipped"
STATUS_EMPTY = "empty"
STATUS_FAILED = "failed"

#: Everything one extraction run writes into a scene directory. A re-extraction
#: (``--overwrite``, or resuming a scene that never finished) removes these
#: first, so frames of a previous run cannot outlive it and skew the counts.
SCENE_OUTPUT_DIRS = ("images", "clouds")
#: The record of what the frame clock was corrected by, and on what evidence.
TIME_ALIGN_FILE = "time_align.txt"
#: The record of what the pose guard threw away, and why (gnss_guard).
GNSS_GUARD_FILE = "gnss_guard.txt"
SCENE_OUTPUT_FILES = (
    "data_ts.txt",
    "pcd_ts.txt",
    "transforms.npy",
    "gravity_align.npy",
    "camera_intrinsics.txt",
    "camera_info.yaml",
    "T_cam_to_base.txt",
    "joint_states.npy",
    "joint_setpoints.npy",
    "cmd_vel.npy",
    "gps.npy",
    TIME_ALIGN_FILE,
    GNSS_GUARD_FILE,
    SENTINEL_FILE,
)
#: Written only when the odometry source is its own ground truth (``--gt``).
GROUND_TRUTH_FILES = ("odometry.txt", "odometry_segments.txt")


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------
def header_stamp(msg, fallback: float) -> float:
    """``header.stamp`` of a deserialised message in seconds, or ``fallback``."""
    if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    return fallback


def header_stamp_from_cdr(raw: bytes) -> float | None:
    """``header.stamp`` read straight out of a CDR buffer, without deserialising.

    Valid only for messages whose first field is a ``std_msgs/Header``: after
    the 4-byte CDR encapsulation (whose second byte says little- or big-endian)
    the header starts with ``stamp.sec`` (int32) and ``stamp.nanosec``
    (uint32). Images and clouds are megabytes each and the first pass needs
    nothing but this number, so skipping the full decode is what makes that
    pass cheap. Returns None for a buffer too short to hold a stamp.
    """
    if len(raw) < 12:
        return None
    fmt = "<iI" if raw[1] == 1 else ">iI"
    sec, nanosec = struct.unpack_from(fmt, raw, 4)
    return sec + nanosec * 1e-9


# ---------------------------------------------------------------------------
# PointCloud2 parsing (dynamic field layout; handles cloudini compression)
# ---------------------------------------------------------------------------
# sensor_msgs/PointField datatype enum -> numpy type
PC2_DTYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}


def fields_to_dtype(fields, point_step: int, big_endian: bool = False) -> np.dtype:
    endian = ">" if big_endian else "<"
    return np.dtype(
        {
            "names": [f.name for f in fields],
            "formats": [endian + PC2_DTYPES[f.datatype] for f in fields],
            "offsets": [f.offset for f in fields],
            "itemsize": point_step,
        }
    )


def cloud_to_xyzi(msg) -> np.ndarray:
    """PointCloud2 or CompressedPointCloud2 message -> (N, 4) float32 [x y z i]."""
    # point_cloud_interfaces/CompressedPointCloud2 carries a cloudini blob instead of raw data
    data = cloudini.decode(bytes(msg.compressed_data)) if hasattr(msg, "compressed_data") else bytes(msg.data)
    dtype = fields_to_dtype(msg.fields, msg.point_step, msg.is_bigendian)
    points = np.frombuffer(data, dtype=dtype, count=msg.width * msg.height)
    names = points.dtype.names
    intensity = (
        points["intensity"].astype(np.float32) if "intensity" in names else np.zeros(len(points), dtype=np.float32)
    )
    return np.stack(
        [
            points["x"].astype(np.float32),
            points["y"].astype(np.float32),
            points["z"].astype(np.float32),
            intensity,
        ],
        axis=1,
    )


def drop_invalid_returns(xyzi: np.ndarray) -> np.ndarray:
    """Drop the two things a sensor emits for a shot that returned nothing.

    An organised cloud ships every element every sweep (Odin1: 256x192 = 49152,
    constant), so a missed shot has to be encoded rather than omitted. Both
    encodings occur in the same bag: NaN, and exactly (0, 0, 0).

    This MUST run before the cloud is transformed into base_link: afterwards
    the zeros land on the mount translation, where they read as a dense real
    object rather than a sentinel. That is HELHEST v1, where 54-70% of every
    published cloud is (-0.170, 0, 0.145) repeated ten to fourteen thousand
    times. Measured per sweep on the 2026-08-21 bags: 49152 elements, 11.2%
    NaN, 47.1% of the rest dropped shots, ~23k real returns left.

    Exact zeros only -- (0, 0, 1e-6) is a real return against the sensor, and
    range gating is the consumer's decision, not this function's.
    """
    xyzi = xyzi[~np.isnan(xyzi).any(axis=1)]
    return xyzi[np.any(xyzi[:, :3] != 0, axis=1)]


# ---------------------------------------------------------------------------
# GNSS
# ---------------------------------------------------------------------------
#: The only message type ``gps.npy`` is ever built from. Both rig generations
#: publish one -- Fixposition on the new stack, Septentrio on the old -- under
#: names that have nothing in common, which is why the topic is found by type.
GPS_MSGTYPE = "sensor_msgs/msg/NavSatFix"

#: Ordered substrings preferred when ``--gps auto`` has several NavSatFix
#: topics to choose between. A fused solution (Fixposition's ``odometry_llh``,
#: which is its ENU odometry expressed geodetically) says more about where the
#: robot was than one antenna's raw fix, and a receiver's own ``navsatfix`` /
#: ``fix`` outranks whatever else a rig happens to publish.
GPS_TOPIC_PREFERENCE = ("odometry_llh", "navsatfix", "fix", "gnss", "gps")


def pick_gps_topic(connections, requested: str) -> str | None:
    """Which topic ``gps.npy`` comes from: the one asked for, or the best fix.

    ``off`` writes no GPS; an explicit topic is taken as given (and simply
    finds nothing if the bag does not carry it); ``auto`` searches by message
    type, so it needs no knowledge of a rig's naming. Topics recorded with
    zero messages are never chosen -- the new bags declare an empty
    ``/fixposition/gnss1`` next to the fix that is actually populated.
    """
    if requested == "off":
        return None
    if requested != "auto":
        return requested
    counts: dict[str, int] = {}
    for conn in connections:
        if conn.msgtype != GPS_MSGTYPE:
            continue
        counts[conn.topic] = counts.get(conn.topic, 0) + int(getattr(conn, "msgcount", 1) or 0)
    populated = {topic: n for topic, n in counts.items() if n}
    if not populated:
        return None

    def rank(topic: str) -> tuple[int, int, str]:
        for i, token in enumerate(GPS_TOPIC_PREFERENCE):
            if token in topic:
                return (i, -populated[topic], topic)
        return (len(GPS_TOPIC_PREFERENCE), -populated[topic], topic)

    return min(populated, key=rank)


def gps_row(t: float, msg) -> np.ndarray | None:
    """One ``gps.npy`` row from a ``NavSatFix``, or None for a fix with no position.

    ``[t, lat, lon, alt, status, cov_ee, cov_nn, cov_uu]``. The covariance is
    the diagonal of the message's row-major 3x3, in m^2, and is NaN when the
    message declares ``COVARIANCE_TYPE_UNKNOWN`` -- the numbers mean nothing
    then, and NaN says so without spending a column on the type constant.
    """
    lat, lon = float(msg.latitude), float(msg.longitude)
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return None
    # A Fixposition that has lost its solution publishes lat = lon = alt = 0
    # with position_covariance_type UNKNOWN -- while status still claims
    # GBAS_FIX, so the status field cannot be the filter. Exact null island is
    # never a fix a ground robot holds. Seen in the 2026-09-08 recordings:
    # 466 such rows in scene 0010 and 29 in 0014, each run spanning exactly one
    # ground-truth segment (the solution resets, the ENU pose teleports and the
    # attitude flips at the same moment).
    if lat == 0.0 and lon == 0.0:
        return None
    cov = np.asarray(msg.position_covariance, dtype=np.float64).ravel()
    unknown = int(getattr(msg, "position_covariance_type", 0)) == 0
    diag = np.full(3, np.nan) if unknown or cov.size < 9 else cov[[0, 4, 8]]
    return np.r_[t, lat, lon, float(msg.altitude), float(msg.status.status), diag]


# ---------------------------------------------------------------------------
# TF helpers
# ---------------------------------------------------------------------------
def build_tf_graph(tf_msgs: Iterable) -> nx.DiGraph:
    """Build a bidirectional frame graph from TF messages.

    A TF message with ``header.frame_id = parent`` and ``child_frame_id = child``
    carries the pose of the child in the parent frame, i.e. the matrix ``M``
    with ``p_parent = M @ p_child``. Every edge ``a -> b`` of the graph stores
    the *point map* from ``a`` to ``b`` (``p_b = E @ p_a``), so the edge
    ``parent -> child`` holds ``inv(M)`` and ``child -> parent`` holds ``M``.
    """
    G = nx.DiGraph()
    for tf in tf_msgs:
        q = tf.transform.rotation
        M = quaternion_matrix([q.x, q.y, q.z, q.w])
        M[:3, 3] = [
            tf.transform.translation.x,
            tf.transform.translation.y,
            tf.transform.translation.z,
        ]
        G.add_edge(tf.header.frame_id, tf.child_frame_id, transform=np.linalg.inv(M))
        G.add_edge(tf.child_frame_id, tf.header.frame_id, transform=M)
    return G


def get_transform(G: nx.DiGraph, src_frame, dst_frame) -> np.ndarray | None:
    """Transform T such that ``p_dst = T @ p_src``, or None if no TF path exists.

    Edges store point maps along their direction (see :func:`build_tf_graph`),
    so the maps are chained right-to-left along the shortest path.
    """
    if src_frame == dst_frame:
        return np.eye(4)
    if src_frame not in G or dst_frame not in G:
        return None
    try:
        path = nx.shortest_path(G, source=src_frame, target=dst_frame)
    except nx.NetworkXNoPath:
        return None
    T = np.eye(4)
    for i in range(len(path) - 1):
        T = G[path[i]][path[i + 1]]["transform"] @ T
    return T


# ---------------------------------------------------------------------------
# Camera/lidar synchronisation
# ---------------------------------------------------------------------------
def nearest_idx(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For each value in ``a``, the index of the nearest value in *sorted* ``b``."""
    if len(b) == 1:
        return np.zeros(len(a), dtype=int)
    pos = np.clip(np.searchsorted(b, a), 1, len(b) - 1)
    return np.where(a - b[pos - 1] <= b[pos] - a, pos - 1, pos)


def quaternion_from_matrix(R) -> np.ndarray:
    """Unit quaternion ``(x, y, z, w)`` of a 3x3 rotation (Shepperd's method).

    The inverse of :func:`quaternion_matrix`. Branching on the largest diagonal
    term keeps the divisor away from zero for every rotation, including the
    180-degree cases where the naive trace formula loses all precision.
    """
    m = np.asarray(R, dtype=float)[:3, :3]
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        d = math.sqrt(tr + 1.0) * 2.0
        return np.array([(m[2, 1] - m[1, 2]) / d, (m[0, 2] - m[2, 0]) / d, (m[1, 0] - m[0, 1]) / d, 0.25 * d])
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        d = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        return np.array([0.25 * d, (m[0, 1] + m[1, 0]) / d, (m[0, 2] + m[2, 0]) / d, (m[2, 1] - m[1, 2]) / d])
    if m[1, 1] > m[2, 2]:
        d = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        return np.array([(m[0, 1] + m[1, 0]) / d, 0.25 * d, (m[1, 2] + m[2, 1]) / d, (m[0, 2] - m[2, 0]) / d])
    d = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
    return np.array([(m[0, 2] + m[2, 0]) / d, (m[1, 2] + m[2, 1]) / d, 0.25 * d, (m[1, 0] - m[0, 1]) / d])


def slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    """Spherical linear interpolation between two ``(x, y, z, w)`` quaternions."""
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:  # same rotation, opposite hemisphere: take the short way round
        q1, dot = -q1, -dot
    if dot > 0.9995:  # nearly parallel; slerp is ill-conditioned, lerp is exact enough
        q = q0 + u * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = math.acos(max(-1.0, min(1.0, dot))) * u
    q2 = q1 - q0 * dot
    q2 /= np.linalg.norm(q2)
    return q0 * math.cos(theta) + q2 * math.sin(theta)


def interpolate_poses(
    query_ts: np.ndarray,
    sample_ts: np.ndarray,
    poses: np.ndarray,
    max_gap: float = 0.5,
    max_speed: float = gnss_guard.MAX_SPEED,
) -> tuple[np.ndarray, int]:
    """Pose at each query time: linear in translation, SLERP in rotation.

    Nearest-neighbour lookup onto a 10 Hz stream quantises every frame by up to
    half a sample period -- 50 ms, which is 5 cm at 1 m/s. Interpolating removes
    that, but only where interpolating means something. The nearest sample is
    used instead when the query is outside the sample range, when the bracketing
    samples are more than ``max_gap`` apart, or when the bracket implies a speed
    above ``max_speed`` -- a GNSS teleport, where blending the two sides would
    fabricate a pose the robot never held. ``max_speed`` matches the gate
    :func:`write_ground_truth` breaks segments on, so the two agree about which
    steps are real motion.

    Returns the poses and how many of them were interpolated rather than copied.
    """
    query = np.asarray(query_ts, dtype=float)
    ts = np.asarray(sample_ts, dtype=float)
    poses = np.asarray(poses, dtype=float)
    out = np.empty((len(query), 4, 4))
    nearest = nearest_idx(query, ts)
    if len(ts) < 2:
        out[:] = poses[nearest]
        return out, 0

    hi = np.clip(np.searchsorted(ts, query), 1, len(ts) - 1)
    n_interp = 0
    for k, q in enumerate(query):
        j = int(hi[k])
        i = j - 1
        dt = ts[j] - ts[i]
        step = float(np.linalg.norm(poses[j][:3, 3] - poses[i][:3, 3]))
        if not (ts[i] <= q <= ts[j]) or not (0.0 < dt <= max_gap) or step > max_speed * dt:
            out[k] = poses[nearest[k]]
            continue
        u = (q - ts[i]) / dt
        T = np.eye(4)
        T[:3, 3] = poses[i][:3, 3] + u * (poses[j][:3, 3] - poses[i][:3, 3])
        qi = quaternion_from_matrix(poses[i])
        qj = quaternion_from_matrix(poses[j])
        T[:3, :3] = quaternion_matrix(slerp(qi, qj, u))[:3, :3]
        out[k] = T
        n_interp += 1
    return out, n_interp


def sync_timestamps(
    cam_timestamps: Sequence[float], lidar_timestamps: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Mutual nearest-neighbour match between camera and lidar timestamps.

    Both inputs are sorted first (bags may contain out-of-order messages). A
    pair is kept only if each stamp is the other's nearest neighbour, which
    drops frames that have no counterpart. Uses ``searchsorted``, so memory
    stays O(n) instead of the O(n*m) full distance matrix.

    Returns the matched (camera, lidar) timestamps, in chronological order.
    """
    cam_ts = np.sort(np.asarray(cam_timestamps, dtype=float))
    pcd_ts = np.sort(np.asarray(lidar_timestamps, dtype=float))
    if len(cam_ts) == 0 or len(pcd_ts) == 0:
        return cam_ts[:0], pcd_ts[:0]

    cam_to_pcd = nearest_idx(cam_ts, pcd_ts)
    pcd_to_cam = nearest_idx(pcd_ts, cam_ts)
    mask = pcd_to_cam[cam_to_pcd] == np.arange(len(cam_ts))
    return cam_ts[mask], pcd_ts[cam_to_pcd[mask]]


# ---------------------------------------------------------------------------
# Bag discovery, scene bookkeeping
# ---------------------------------------------------------------------------
@dataclass
class SceneResult:
    """Outcome of one bag -> scene extraction (returned by worker processes)."""

    index: int
    bag: str
    status: str
    frames: int = 0
    start_time: float = float("nan")
    message: str = ""


def is_bag_dir(path: Path) -> bool:
    """A rosbag2 directory is anything that carries a ``metadata.yaml``."""
    return path.is_dir() and (path / "metadata.yaml").is_file()


def find_bags(input_dir: Path) -> list[Path]:
    """Bags under ``input_dir``; ``input_dir`` itself may also be a single bag."""
    if not input_dir.exists():
        raise FileNotFoundError(f"input path does not exist: {input_dir}")
    if is_bag_dir(input_dir):
        return [input_dir]
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input path is not a directory: {input_dir}")

    subdirs = sorted(d for d in input_dir.iterdir() if d.is_dir())
    bags = [d for d in subdirs if is_bag_dir(d)]
    ignored = [d.name for d in subdirs if not is_bag_dir(d)]
    if ignored:
        LOG.warning(
            "ignoring %d directory/ies without metadata.yaml in %s: %s",
            len(ignored),
            input_dir,
            ", ".join(ignored[:10]) + (" ..." if len(ignored) > 10 else ""),
        )
    return bags


def parse_scene_selection(tokens: Sequence[str] | None):
    """Build a predicate from ``--scenes`` tokens.

    Tokens may be scene indices (``7``), inclusive index ranges (``3-9``) or
    bag directory names; they may also be comma-separated inside one argument.
    Returns ``None`` when nothing was requested (i.e. everything is selected).
    """
    if not tokens:
        return None

    indices: set[int] = set()
    names: set[str] = set()
    for raw in tokens:
        for token in str(raw).split(","):
            token = token.strip()
            if not token:
                continue
            match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", token)
            if match:
                lo, hi = int(match.group(1)), int(match.group(2))
                indices.update(range(min(lo, hi), max(lo, hi) + 1))
            elif token.isdigit():
                indices.add(int(token))
            else:
                names.add(token.rstrip("/"))

    def selected(index: int, bag_name: str) -> bool:
        return index in indices or bag_name in names

    return selected


def _count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _count_files(directory: Path, suffix: str) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for p in directory.iterdir() if p.suffix == suffix)


def scene_frame_counts(scene_dir: Path) -> dict[str, int]:
    """Per-modality frame counts of an extracted scene (missing -> absent key)."""
    counts: dict[str, int] = {}
    counts["images"] = _count_files(scene_dir / "images", ".png")
    counts["clouds"] = _count_files(scene_dir / "clouds", ".npy")
    for name in ("data_ts.txt", "pcd_ts.txt"):
        if (scene_dir / name).is_file():
            counts[name] = _count_lines(scene_dir / name)
    return counts


def scene_is_complete(scene_dir: Path) -> bool:
    """True if the scene was extracted to the end and its counts line up."""
    if not (scene_dir / SENTINEL_FILE).is_file():
        return False
    counts = scene_frame_counts(scene_dir)
    if len(counts) < 4:  # a timestamp file is missing entirely
        return False
    values = set(counts.values())
    return len(values) == 1 and values != {0}


def _scene_row(scene_dir: Path) -> list[str] | None:
    """Reconstruct a ``scenes.csv`` row from an already extracted scene."""
    sentinel = scene_dir / SENTINEL_FILE
    if not sentinel.is_file():
        return None
    bag = sentinel.read_text().strip()
    frames = _count_files(scene_dir / "images", ".png")
    start = ""
    data_ts = scene_dir / "data_ts.txt"
    if data_ts.is_file():
        with data_ts.open() as handle:
            first = handle.readline().strip()
        if first:
            start = f"{float(first):.6f}"
    return [scene_dir.name, bag, start, str(frames)]


def write_scene_part(parts_dir: Path, result: SceneResult) -> None:
    """One CSV line per worker; the parent merges them (multiprocessing-safe)."""
    parts_dir.mkdir(parents=True, exist_ok=True)
    start = "" if math.isnan(result.start_time) else f"{result.start_time:.6f}"
    row = [f"{result.index:04d}", result.bag, start, str(result.frames)]
    with (parts_dir / f"{result.index:04d}.csv").open("w", newline="") as handle:
        csv.writer(handle).writerow(row)


def merge_scenes_csv(output_dir: Path) -> Path:
    """Rewrite ``<output_dir>/scenes.csv`` from worker parts + finished scenes."""
    parts_dir = output_dir / SCENES_PARTS_DIR
    rows: dict[str, list[str]] = {}

    if parts_dir.is_dir():
        for part in sorted(parts_dir.glob("*.csv")):
            with part.open(newline="") as handle:
                for row in csv.reader(handle):
                    if row:
                        rows[row[0]] = row

    for scene_dir in sorted(output_dir.iterdir()):
        if not scene_dir.is_dir() or not SCENE_DIR_RE.match(scene_dir.name):
            continue
        if scene_dir.name in rows:
            continue
        row = _scene_row(scene_dir)
        if row is not None:
            rows[row[0]] = row

    csv_path = output_dir / SCENES_CSV
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(SCENES_CSV_HEADER)
        for key in sorted(rows):
            writer.writerow(rows[key])

    if parts_dir.is_dir():
        for part in parts_dir.glob("*.csv"):
            part.unlink()
        with contextlib.suppress(OSError):  # someone else is still writing; harmless
            parts_dir.rmdir()
    return csv_path


def clear_scene_outputs(scene_dir: Path, ground_truth: bool) -> None:
    """Remove what a previous extraction of this scene wrote.

    Only the extractor's own outputs go; anything else in the directory (a
    ``regen/`` provenance folder, height maps, reviewer notes) is left alone.
    ``odometry.txt`` and its segment table are removed only when this run will
    write them itself: on the old rig they come from the FAST-LIO regeneration,
    which runs *before* the extraction and has to survive it.
    """
    if not scene_dir.is_dir():
        return
    for name in SCENE_OUTPUT_DIRS:
        shutil.rmtree(scene_dir / name, ignore_errors=True)
    files = SCENE_OUTPUT_FILES + (GROUND_TRUTH_FILES if ground_truth else ())
    for name in files:
        (scene_dir / name).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def _ensure_logging() -> None:
    """Worker processes started without fork inherit no logging configuration."""
    if not logging.getLogger().handlers:
        setup_logging(False)


def _worker(bag_path: Path, output_dir: Path, idx: int, args) -> SceneResult:
    _ensure_logging()
    scene_dir = output_dir / f"{idx:04d}"
    try:
        with AnyReader([bag_path]) as reader:
            result = process_scene(reader, bag_path, scene_dir, args)
        result.index = idx
    except Exception as exc:  # noqa: BLE001 - one bad bag must not kill the run
        LOG.error("failed to process %s:\n%s", bag_path, traceback.format_exc())
        result = SceneResult(
            index=idx,
            bag=bag_path.name,
            status=STATUS_FAILED,
            message=f"{type(exc).__name__}: {exc}",
        )
    # Only finished scenes go into scenes.csv; failures are reported in the
    # end-of-run summary instead.
    if result.status == STATUS_OK:
        write_scene_part(output_dir / SCENES_PARTS_DIR, result)
    return result


def time_align_mode(value: str) -> str:
    """Normalise a ``--time_align`` value: ``auto``, ``off`` or a number as text.

    Also the parser's ``type``, so a typo is refused with a usage message before
    any bag is opened rather than inside a worker process, where a ``SystemExit``
    would take the whole pool down.
    """
    mode = str(value or "auto").strip().lower()
    if mode in ("auto", "off"):
        return mode
    try:
        float(mode)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected 'auto', 'off' or seconds, not {value!r}") from None
    return mode


def align_frame_clock(scene_dir: Path, frame_ts: np.ndarray, odom_data, args, bag: str):
    """Take the sensor unit's capture-to-publish latency off the frame clock.

    ``--time_align auto`` measures it against the clouds just written (see
    :mod:`dataset_tools.time_align`); a number applies that constant instead,
    for a rig already characterised or a scene too poor in geometry to measure;
    ``off`` leaves the stamps alone. Returns the corrected clock and the profile
    to record, or ``(frame_ts, None)`` when nothing was applied.
    """
    mode = time_align_mode(args.time_align)
    if mode == "off":
        return frame_ts, None

    if mode != "auto":
        constant = float(mode)
        LOG.info("%s: frame clock shifted by %+.3f s (--time_align %s)", bag, constant, mode)
        profile = time_align.Profile([], np.array([frame_ts[0]]), np.array([constant]), "given on the command line")
        shifted, _ = time_align.apply_to_clock(frame_ts, profile)
        return shifted, profile

    if not odom_data:
        LOG.info("%s: no odometry to align the frame clock against; stamps left as recorded", bag)
        return frame_ts, None

    odom = sorted(odom_data, key=lambda x: x[0])
    profile = time_align.estimate(
        scene_dir,
        frame_ts,
        np.array([x[0] for x in odom]),
        np.array([x[1] for x in odom]),
    )
    LOG.info("%s: frame clock -- %s", bag, profile.summary)
    if profile.is_identity:
        return frame_ts, profile
    shifted, forced = time_align.apply_to_clock(frame_ts, profile)
    if forced:
        LOG.warning("%s: the measured offset ran backwards faster than the clock; forced monotonic", bag)
    return shifted, profile


def process_scene(reader, bag_path: Path, scene_dir: Path, args) -> SceneResult:
    camera_topic, lidar_topic = args.cam, args.lidar
    # Added to the camera stamp in both passes, in one expression, so the two
    # produce bit-identical floats and pass 2's set lookups hit. This is the
    # camera's stamping against the LIDAR's, and nothing else: the offset both
    # of them share against the pose clock is a latency, not a stamp error, and
    # is measured after the clouds exist (see --time_align).
    shift_cam = float(args.cam_time_offset or 0.0)
    shift_lidar = 0.0
    result = SceneResult(index=-1, bag=bag_path.name, status=STATUS_OK)

    # Pre-cache connections
    conns = {c.topic: c for c in reader.connections}
    cam_conn = conns.get(camera_topic)
    lidar_conn = conns.get(lidar_topic)

    if not cam_conn or not lidar_conn:
        missing = [topic for topic, conn in ((camera_topic, cam_conn), (lidar_topic, lidar_conn)) if conn is None]
        LOG.warning("%s: missing topic(s) %s; scene skipped", bag_path.name, missing)
        result.status = STATUS_EMPTY
        result.message = f"missing topic(s): {', '.join(missing)}"
        return result
    # Pass 2 hands the payload to cv2.imdecode, which cannot read a raw
    # sensor_msgs/Image; said here, once, instead of as an OpenCV assertion on
    # the first frame.
    if cam_conn.msgtype not in IMAGE_MSGTYPES:
        raise ValueError(f"--cam {camera_topic} is {cam_conn.msgtype}; expected one of {', '.join(IMAGE_MSGTYPES)}")

    # The two 'auto' modes are decided by the odometry topic, which is worth a
    # line in the log: a rig this tool has not met gets neither ground truth
    # nor the pose guard, and would otherwise not be told why.
    if args.gt == "auto" or args.gnss_guard == "auto":
        LOG.info(
            "%s: from --odom %s: ground truth %s, GNSS guard %s",
            bag_path.name,
            args.odom,
            "on" if gt_enabled(args) else "off (--gt on to force)",
            "on" if guard_enabled(args) else "off (--gnss_guard on to force)",
        )

    # A scene is extracted from scratch every time, so whatever an earlier run
    # left behind must go first (see clear_scene_outputs).
    clear_scene_outputs(scene_dir, ground_truth=gt_enabled(args))

    # Pass 1: Collect timestamps for synchronization
    cam_timestamps = []
    lidar_timestamps = []

    # We also collect odom and joint states in this pass if they are small
    odom_data = []  # (ts, transform of odom child frame in odom fixed frame)
    odom_child_frame = None
    imu_data = []  # (ts, rotation matrix of imu frame in gravity-referenced frame)
    imu_frame = None
    joint_states = []
    joint_setpoints = []
    cmd_vels = []
    gps_rows = []  # (t, lat, lon, alt, status, cov diagonal) of a NavSatFix

    # Found by message type rather than by name, so the same run works on both
    # rig generations; None when the bag carries no fix at all.
    gps_topic = pick_gps_topic(reader.connections, args.gps)

    relevant_conns = [
        cam_conn,
        lidar_conn,
        conns.get(args.odom),
        conns.get(args.imu),
        conns.get("/joint_states"),
        conns.get("/joint_setpoints"),
        conns.get("/cmd_vel"),
        conns.get(gps_topic),
        conns.get("/tf_static"),
    ]
    # Deduplicate (e.g. when --odom and --imu are the same topic)
    relevant_conns = list({id(c): c for c in relevant_conns if c is not None}.values())

    tf_msgs = []

    # Images and clouds are the bulk of the bag and this pass needs only their
    # stamps, which are read out of the raw bytes (header_stamp_from_cdr). The
    # shortcut is checked against a full decode of each connection's first
    # message and dropped for that connection if the two disagree.
    stamp_only = {id(cam_conn), id(lidar_conn)}
    verified: set[int] = set()

    for conn, timestamp, raw in reader.messages(relevant_conns):
        if id(conn) in stamp_only and id(conn) in verified:
            t = header_stamp_from_cdr(raw)
            if t is None:
                t = header_stamp(reader.deserialize(raw, conn.msgtype), timestamp)
        else:
            msg = reader.deserialize(raw, conn.msgtype)
            t = header_stamp(msg, timestamp)
            if id(conn) in stamp_only:
                if header_stamp_from_cdr(raw) == t:
                    verified.add(id(conn))
                else:
                    stamp_only.discard(id(conn))

        if conn.topic == camera_topic:
            cam_timestamps.append(t + shift_cam)
        elif conn.topic == lidar_topic:
            lidar_timestamps.append(t + shift_lidar)
        elif conn.topic in (args.odom, args.imu):
            # The same topic may serve as both odometry and orientation source
            if conn.topic == args.odom:
                odom_child_frame = msg.child_frame_id
                odom_data.append((t, pose_to_matrix(msg.pose.pose)))
            if conn.topic == args.imu:
                # nav_msgs/Odometry also works as an orientation source
                if hasattr(msg, "orientation"):  # sensor_msgs/Imu
                    q = msg.orientation
                    imu_frame = msg.header.frame_id
                else:
                    q = msg.pose.pose.orientation
                    imu_frame = msg.child_frame_id
                imu_data.append((t, quaternion_matrix([q.x, q.y, q.z, q.w])))
        elif conn.topic == "/joint_states":
            joint_states.append(np.r_[t, msg.position, msg.velocity, msg.effort])
        elif conn.topic == "/joint_setpoints":
            joint_setpoints.append(np.r_[t, msg.position, msg.velocity, msg.effort])
        elif conn.topic == "/cmd_vel":
            tw = msg.twist if hasattr(msg, "twist") else msg
            cmd_vels.append(
                np.r_[
                    t,
                    tw.linear.x,
                    tw.linear.y,
                    tw.linear.z,
                    tw.angular.x,
                    tw.angular.y,
                    tw.angular.z,
                ]
            )
        elif conn.topic == gps_topic:
            row = gps_row(t, msg)
            if row is not None:
                gps_rows.append(row)
        elif conn.topic == "/tf_static":
            tf_msgs.extend(msg.transforms)

    # Sync timestamps (mutual nearest neighbour)
    valid_cam_ts, valid_pcd_ts = sync_timestamps(cam_timestamps, lidar_timestamps)

    if len(valid_cam_ts) == 0:
        LOG.warning(
            "%s: no synchronised camera/lidar pairs (%d camera, %d lidar messages); scene left empty",
            bag_path.name,
            len(cam_timestamps),
            len(lidar_timestamps),
        )
        result.status = STATUS_EMPTY
        result.message = f"no synced pairs ({len(cam_timestamps)} cam, {len(lidar_timestamps)} lidar)"
        return result

    # One clock per frame. transforms.npy and gravity_align.npy used to be
    # sampled at the camera and the lidar stamps respectively, while data_ts.txt
    # was their midpoint -- three clocks behind one frame index, which is not
    # what a consumer reading "frame k" from all three expects.
    #
    # This is the clock as the sensors stamped it. Where the unit stamps at
    # publish rather than at capture it is late by the pipeline's latency, which
    # is measured and taken off below, once there are clouds to measure with.
    frame_ts = (valid_cam_ts + valid_pcd_ts) / 2

    # Whatever the GNSS/INS fabricated goes now, before a single image or cloud
    # is written, so the frames that survive are exactly the frames the dataset
    # can put a pose on and the indices below stay contiguous. The poses are
    # still the receiver's own output here, not yet re-expressed at the base
    # frame: a fabricated stretch is a property of what the receiver published,
    # and moving it down a lever arm only lets a fast rotation masquerade as a
    # jump. The same stage drops frames the pose stream never covered at all,
    # before it starts, after it ends or inside a hole too wide to interpolate
    # across: those would be exported carrying the nearest sample, which is a
    # fabricated pose with a different author. See dataset_tools.gnss_guard.
    guard_verdict = None
    if guard_enabled(args) and odom_data:
        odom_data.sort(key=lambda x: x[0])
        # The stamps of the fixes that carried a position. The gaps in them are
        # when the receiver had no solution, whether it said so by publishing
        # null island (gps_row drops those) or by publishing nothing at all.
        # A rig with a fix topic but not one usable fix is an empty list, not
        # None: the receiver never had a solution, which is evidence too.
        fix_ts = [float(r[0]) for r in gps_rows] if gps_topic else None
        odom_data, keep, guard_verdict = gnss_guard.guard(odom_data, frame_ts, fix_ts)
        if not guard_verdict.clean:
            LOG.warning("%s: %s", bag_path.name, guard_verdict.summary(float(frame_ts[0])))
            valid_cam_ts, valid_pcd_ts, frame_ts = valid_cam_ts[keep], valid_pcd_ts[keep], frame_ts[keep]
            if len(valid_cam_ts) == 0:
                LOG.warning("%s: no frame of this bag has a trustworthy pose; scene left empty", bag_path.name)
                result.status = STATUS_EMPTY
                result.message = "no frame of this bag has a pose the dataset can stand behind"
                return result
        else:
            LOG.info("%s: pose guard -- %s", bag_path.name, guard_verdict.summary(float(frame_ts[0])))

    (scene_dir / "images").mkdir(parents=True, exist_ok=True)
    (scene_dir / "clouds").mkdir(parents=True, exist_ok=True)

    # Map timestamps to their chronological frame indices
    cam_ts_to_idx = {ts: i for i, ts in enumerate(valid_cam_ts)}
    pcd_ts_to_idx = {ts: i for i, ts in enumerate(valid_pcd_ts)}

    for name, rows in [
        ("joint_states", joint_states),
        ("joint_setpoints", joint_setpoints),
        ("cmd_vel", cmd_vels),
    ]:
        if rows:
            arr = np.vstack(rows)
            np.save(scene_dir / f"{name}.npy", arr[np.argsort(arr[:, 0])])
        else:
            LOG.info("%s: no messages on /%s; %s.npy not written", bag_path.name, name, name)

    # GPS is an optional extra, not part of the scene contract: a bag without a
    # fix is a scene without gps.npy, exactly as one without /cmd_vel is.
    if gps_rows:
        arr = np.vstack(gps_rows)
        np.save(scene_dir / "gps.npy", arr[np.argsort(arr[:, 0])])
        LOG.info("%s: %d GNSS fixes from '%s'", bag_path.name, len(gps_rows), gps_topic)
    elif gps_topic:
        LOG.info("%s: no usable fixes on '%s'; gps.npy not written", bag_path.name, gps_topic)
    else:
        LOG.info("%s: no %s topic in the bag; gps.npy not written", bag_path.name, GPS_MSGTYPE)

    # Process TF Static
    tf_graph = build_tf_graph(tf_msgs) if tf_msgs else nx.DiGraph()

    lidar_frame = get_first_frame(reader, lidar_conn)
    T_lidar_to_base = get_transform(tf_graph, lidar_frame, args.base_frame)
    if T_lidar_to_base is None:
        if args.lidar_extrinsic is not None:
            x, y, z, roll, pitch, yaw = args.lidar_extrinsic
            T_lidar_to_base = euler_matrix(roll, pitch, yaw)
            T_lidar_to_base[:3, 3] = [x, y, z]
        else:
            LOG.warning(
                "%s: no TF path from '%s' to '%s' and no --lidar_extrinsic given; using identity",
                bag_path.name,
                lidar_frame,
                args.base_frame,
            )
            T_lidar_to_base = np.eye(4)

    # Odom poses track the odometry child frame (e.g. FP_POI for Fixposition);
    # re-express them as poses of the base frame when TF allows it.
    T_child_to_base = get_transform(tf_graph, odom_child_frame, args.base_frame) if odom_child_frame else None
    if T_child_to_base is not None and not np.allclose(T_child_to_base, np.eye(4)):
        T_base_in_child = np.linalg.inv(T_child_to_base)
        odom_data = [(t, T @ T_base_in_child) for t, T in odom_data]
    elif odom_child_frame and T_child_to_base is None:
        LOG.warning(
            "%s: no TF path from odom child '%s' to '%s'; storing raw odometry poses",
            bag_path.name,
            odom_child_frame,
            args.base_frame,
        )

    # Re-express the orientation source at the base frame too, so gravity
    # alignment reflects the base tilt rather than the sensor mounting.
    T_imu_to_base = get_transform(tf_graph, imu_frame, args.base_frame) if imu_frame else None
    if T_imu_to_base is not None and not np.allclose(T_imu_to_base, np.eye(4)):
        T_base_in_imu = np.linalg.inv(T_imu_to_base)
        imu_data = [(t, R @ T_base_in_imu) for t, R in imu_data]
    elif imu_frame and T_imu_to_base is None:
        LOG.warning(
            "%s: no TF path from orientation frame '%s' to '%s'; using raw orientations for gravity alignment",
            bag_path.name,
            imu_frame,
            args.base_frame,
        )

    # Pass 2: Extract heavy data (Images, Pointclouds)
    cam_set = set(valid_cam_ts)
    pcd_set = set(valid_pcd_ts)

    intrinsics_saved = False

    for conn, timestamp, raw in reader.messages([cam_conn, lidar_conn]):
        # Same stamp arithmetic as pass 1, so the set lookups below hit exactly
        # -- including the fallback to the bag's own timestamp for a message
        # type without a header.
        is_cam = conn.topic == camera_topic
        shift = shift_cam if is_cam else shift_lidar
        t = header_stamp_from_cdr(raw) if id(conn) in verified else None
        if t is None:
            msg = reader.deserialize(raw, conn.msgtype)
            t = header_stamp(msg, timestamp) + shift
        else:
            t += shift
            if t not in (cam_set if is_cam else pcd_set):
                continue  # not a synchronised frame: never decoded at all
            msg = reader.deserialize(raw, conn.msgtype)

        if conn.topic == camera_topic and t in cam_set:
            img_idx = cam_ts_to_idx[t]  # Get correct chronological index

            # Use OpenCV directly
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_img is None:
                # A frame nothing can be written for leaves a hole in an
                # index-aligned scene, so the scene fails -- but with the frame
                # named, not with an OpenCV assertion from cv2.imwrite.
                raise ValueError(f"image {img_idx} (stamp {t:.6f}, {len(np_arr)} bytes) could not be decoded")
            cv2.imwrite(str(scene_dir / "images" / f"{img_idx:05d}.png"), cv_img)

            # Extract Intrinsics once
            if not intrinsics_saved:
                save_intrinsics(reader, conns, camera_topic, scene_dir)
                save_camera_extrinsic(tf_graph, msg.header.frame_id, args.base_frame, scene_dir)
                intrinsics_saved = True

            cam_set.remove(t)  # Handle duplicates

        elif conn.topic == lidar_topic and t in pcd_set:
            pcd_idx = pcd_ts_to_idx[t]  # Get correct chronological index
            # Before T_lidar_to_base, never after -- see drop_invalid_returns.
            xyzi = drop_invalid_returns(cloud_to_xyzi(msg))
            pcd = np.hstack((xyzi[:, :3], np.ones((len(xyzi), 1), dtype=np.float32)))
            pcd = (pcd @ T_lidar_to_base.T.astype(np.float32))[:, :3]
            np.save(
                scene_dir / "clouds" / f"{pcd_idx:05d}.npy",
                np.hstack((pcd, xyzi[:, 3:4])),
            )
            pcd_set.remove(t)

    # The frame clock, corrected now that there are clouds to measure against.
    # The lidar's own clock takes the same correction -- it is the same unit's
    # latency, and pcd_ts.txt has to stay comparable with data_ts.txt -- so the
    # camera/lidar offset within a frame survives it untouched. Everything
    # sampled onto the clock below follows from here.
    corrected, align = align_frame_clock(scene_dir, frame_ts, odom_data, args, bag_path.name)
    applied = corrected - frame_ts
    frame_ts = corrected
    np.savetxt(scene_dir / "data_ts.txt", frame_ts)
    np.savetxt(scene_dir / "pcd_ts.txt", valid_pcd_ts + applied)
    if align is not None:
        time_align.write_report(scene_dir / TIME_ALIGN_FILE, align, float(frame_ts[0]))

    if guard_verdict is not None:
        gnss_guard.write_report(scene_dir / GNSS_GUARD_FILE, guard_verdict, float(frame_ts[0]))

    # Process Odom Transforms (relative to first frame)
    if gt_enabled(args) and odom_data:
        if write_ground_truth(odom_data, scene_dir, spans=guard_verdict.spans if guard_verdict else None):
            LOG.info("%s: ground truth written (odometry.txt + segments)", bag_path.name)
        else:
            LOG.warning("%s: too few odometry samples for ground truth", bag_path.name)

    if odom_data:
        # The odometry interpolated onto the frame clock, relative to frame 0.
        odom_data.sort(key=lambda x: x[0])
        odom_ts = np.array([x[0] for x in odom_data])
        odom_poses = np.array([x[1] for x in odom_data])
        matched, n_interp = interpolate_poses(frame_ts, odom_ts, odom_poses)
        LOG.info(
            "%s: transforms.npy -- %d/%d interpolated, %d fell back to the nearest pose (gap or teleport)",
            bag_path.name,
            n_interp,
            len(frame_ts),
            len(frame_ts) - n_interp,
        )
        rel_transforms = anchor_to_first(matched)
        rel_transforms[0] = np.eye(4)
        np.save(scene_dir / "transforms.npy", rel_transforms)
    else:
        LOG.warning(
            "%s: no odometry messages on '%s'; transforms.npy not written",
            bag_path.name,
            args.odom,
        )

    # Process gravity alignment
    if imu_data:
        # The orientation interpolated onto the frame clock; yaw is dropped.
        imu_data.sort(key=lambda x: x[0])
        imu_ts = np.array([x[0] for x in imu_data])
        imu_mats = np.array([x[1] for x in imu_data])
        oriented, _ = interpolate_poses(frame_ts, imu_ts, imu_mats)
        gravity_aligns = np.empty((len(frame_ts), 3, 3))
        for k, M in enumerate(oriented):
            roll, pitch = roll_pitch_from_matrix(M)
            gravity_aligns[k] = euler_matrix(roll, pitch, 0.0)[:3, :3]
        np.save(scene_dir / "gravity_align.npy", gravity_aligns)
    else:
        LOG.warning(
            "%s: no orientation messages on '%s'; gravity_align.npy not written",
            bag_path.name,
            args.imu,
        )

    # Provenance; written last, so it doubles as the "scene finished" marker.
    (scene_dir / SENTINEL_FILE).write_text(f"{bag_path.name}\n")

    result.frames = int(len(valid_cam_ts))
    result.start_time = float(frame_ts[0])
    return result


def get_first_frame(reader, conn):
    for _, _, raw in reader.messages([conn]):
        return reader.deserialize(raw, conn.msgtype).header.frame_id
    return None


def save_intrinsics(reader, conns, cam_topic: str, scene_dir: Path) -> bool:
    # e.g. /luxonis/oak/rgb/image_raw/compressed -> /luxonis/oak/rgb/camera_info
    #      /odin1/image/compressed               -> /odin1/camera_info
    candidates = [
        cam_topic.replace("image_raw/compressed", "camera_info"),
        cam_topic.replace("image/compressed", "camera_info"),
        cam_topic.rsplit("/", 2)[0] + "/camera_info",
    ]
    candidates = [c for c in candidates if c != cam_topic]
    conn = next((conns[c] for c in candidates if c in conns), None)
    if conn is None:
        LOG.warning(
            "no camera_info topic for %s (tried %s); camera_intrinsics.txt not written",
            cam_topic,
            ", ".join(candidates),
        )
        return False
    for _, _, raw in reader.messages([conn]):
        msg = reader.deserialize(raw, conn.msgtype)
        np.savetxt(scene_dir / "camera_intrinsics.txt", np.array(msg.k).reshape(3, 3))
        # The full CameraInfo: v1 scenes kept only K and lost the distortion
        # coefficients, so anything that needs undistortion has to go back to the
        # bags. Write everything, in ROS camera_info_manager YAML layout.
        write_camera_info_yaml(msg, scene_dir / "camera_info.yaml")
        return True
    LOG.warning("camera_info topic %s carries no messages", conn.topic)
    return False


def write_camera_info_yaml(msg, path: Path) -> None:
    """Dump a sensor_msgs/CameraInfo as camera_info_manager-style YAML (no yaml dep)."""

    def mat(name, rows, cols, data):
        vals = ", ".join(f"{float(v):.10g}" for v in data)
        return f"{name}:\n  rows: {rows}\n  cols: {cols}\n  data: [{vals}]\n"

    d = list(msg.d)
    text = (
        f"image_width: {int(msg.width)}\n"
        f"image_height: {int(msg.height)}\n"
        f"frame_id: {msg.header.frame_id}\n"
        f"camera_name: {msg.header.frame_id}\n"
        f"distortion_model: {msg.distortion_model}\n"
        + mat("camera_matrix", 3, 3, msg.k)
        + mat("distortion_coefficients", 1, len(d), d)
        + mat("rectification_matrix", 3, 3, msg.r)
        + mat("projection_matrix", 3, 4, msg.p)
        + f"binning_x: {int(msg.binning_x)}\nbinning_y: {int(msg.binning_y)}\n"
    )
    path.write_text(text)


def save_camera_extrinsic(tf_graph, cam_frame, base_frame, scene_dir: Path) -> bool:
    """Write T_cam_to_base.txt: p_base = T @ p_cam_optical, from the static TF.

    v1 scenes instead carry a hand-entered ``T_cam_to_lidar.txt`` expressed in
    the RAW lidar frame; the two conventions differ by the lidar mount X and
    (empirically) 4.5 cm of x, so the file is named by its target frame.
    """
    if not cam_frame:
        return False
    T = get_transform(tf_graph, cam_frame, base_frame)
    if T is None:
        LOG.warning("no TF path %s -> %s; T_cam_to_base.txt not written", cam_frame, base_frame)
        return False
    np.savetxt(scene_dir / "T_cam_to_base.txt", T)
    return True


def gt_enabled(args) -> bool:
    """--gt auto: the odometry is its own ground truth only on rigs where the
    source is absolute (Fixposition GNSS/INS); wheel-IMU /odom drifts and the
    old rig's ground truth comes from the offline FAST-LIO regeneration."""
    if args.gt == "on":
        return True
    if args.gt == "off":
        return False
    return "fixposition" in args.odom


def guard_enabled(args) -> bool:
    """--gnss_guard auto: screen the poses for a reset solution only where a
    reset is a thing that happens. An absolute GNSS/INS source (Fixposition)
    answers with a fabricated position when it loses its fix; wheel-IMU /odom
    has nothing to lose and, being an integration, legitimately holds a
    bit-identical pose for as long as the robot stands still -- which is the
    guard's own signature for a fabricated stretch, so running it there would
    delete exactly the frames where the robot was parked."""
    mode = str(args.gnss_guard or "auto").lower()
    if mode == "on":
        return True
    if mode == "off":
        return False
    return "fixposition" in args.odom


def write_ground_truth(
    odom_data, scene_dir: Path, max_gap=1.0, max_speed=gnss_guard.MAX_SPEED, min_poses=20, spans=None
) -> bool:
    """Write odometry.txt + odometry_segments.txt in the HELHEST v1.2 contract.

    For rigs whose odometry source is trustworthy on its own (Fixposition
    GNSS/INS), the recorded poses ARE the ground truth -- but consumers expect
    the v1.2 shape: pose rows at full rate, split into segments that are each
    re-anchored to the identity, with a segment table that must never be
    bridged. Segments break at time gaps > ``max_gap`` and at teleports
    (displacement implying > ``max_speed`` m/s, e.g. GNSS out-and-back spikes);
    spike samples end up in tiny segments that ``min_poses`` discards.

    Each segment's ``reason`` names why *that* segment ended: ``gap``,
    ``teleport``, ``fix_lost`` for a hole :mod:`dataset_tools.gnss_guard` left
    behind when it removed a fabricated stretch, or ``end_of_run`` for the last
    one.
    """
    if len(odom_data) < min_poses:
        return False
    data = sorted(odom_data, key=lambda x: x[0])
    ts = np.array([x[0] for x in data])
    Ts = np.array([x[1] for x in data])
    dt = np.diff(ts)
    step = np.linalg.norm(np.diff(Ts[:, :3, 3], axis=0), axis=1)
    breaks = np.where((dt > max_gap) | (step > max_speed * np.maximum(dt, 1e-3)))[0] + 1
    spans = list(spans or [])

    def _reason(b: int) -> str:
        # A hole the guard opened is named for what opened it, not for the fact
        # that it is a hole: "gap" would say the recording stopped, which is
        # not what happened.
        if any(ts[b - 1] <= sp.end and sp.start <= ts[b] for sp in spans):
            return "fix_lost"
        return "gap" if dt[b - 1] > max_gap else "teleport"

    reasons = [_reason(int(b)) for b in breaks]
    starts = [0, *breaks]
    ends = [*breaks, len(ts)]
    kept_ts: list[np.ndarray] = []
    kept_poses: list[np.ndarray] = []
    rows: list[SegmentRow] = []
    for k, (a, b) in enumerate(zip(starts, ends, strict=True)):
        if b - a < min_poses:
            continue
        kept_ts.append(ts[a:b])
        kept_poses.append(anchor_to_first(Ts[a:b]))
        # The tag names why this segment ENDED -- the break that closes it,
        # not the one that opened it -- so it means the same thing here as
        # in the regenerated (FAST-LIO) tables the viewer also reads.
        reason = reasons[k] if k < len(reasons) else "end_of_run"
        rows.append(SegmentRow(len(rows), float(ts[a]), float(ts[b - 1]), b - a, reason))
    write_trajectory(
        scene_dir / "odometry.txt", np.concatenate(kept_ts or [ts[:0]]), np.concatenate(kept_poses or [Ts[:0]])
    )
    write_segments(scene_dir / "odometry_segments.txt", rows)
    return bool(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract HELHEST scenes from ROS 2 bags.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_dir", type=str, help="Directory of bags, or a single bag directory")
    parser.add_argument("output_dir", type=str, help="Dataset root to write scenes to")
    parser.add_argument(
        "--cam",
        type=str,
        default="/odin1/image/compressed",
        help="Compressed image topic (old bags: /luxonis/oak/rgb/image_raw/compressed)",
    )
    parser.add_argument(
        "--lidar",
        type=str,
        default="/odin1/cloud_filtered/cloudini",
        help="PointCloud2 or CompressedPointCloud2 topic (old bags: /livox/lidar_front)",
    )
    parser.add_argument(
        "--odom",
        type=str,
        default="/fixposition/odometry_enu",
        help="Odometry topic used for transforms.npy (old bags: /odom). Note: "
        "the _smooth variant's child frame FP_POISH is not connected to "
        "base_link via static TF, plain odometry_enu (FP_POI) is.",
    )
    parser.add_argument(
        "--imu",
        type=str,
        default="/fixposition/odometry_enu",
        help="Orientation source for gravity alignment; accepts sensor_msgs/Imu "
        "or nav_msgs/Odometry topics (old bags: /ekf/imu/data). Note: the new "
        "bags' /imu/data is mounted tilted (~5 deg roll, ~11 deg pitch) with "
        "no TF to base_link, so the Fixposition orientation is used instead.",
    )
    parser.add_argument(
        "--gps",
        type=str,
        default="auto",
        metavar="TOPIC",
        help="GNSS fix topic (sensor_msgs/NavSatFix) written to gps.npy. "
        "'auto' takes the NavSatFix topic that carries messages, preferring a "
        "fused solution (/fixposition/odometry_llh) over a single antenna's "
        "raw fix; 'off' writes no GPS.",
    )
    parser.add_argument(
        "--time_align",
        type=time_align_mode,
        default="auto",
        metavar="auto|off|SECONDS",
        help="How to put the frame clock on the same clock as the poses. A "
        "unit that stamps at publish rather than at capture hands out headers "
        "that are late by its own pipeline latency, which makes the poses run "
        "ahead of the imagery and the clouds; nothing in the bag records it, "
        "because recv-minus-header measures publish-to-receive. 'auto' (the "
        "default) measures it against the clouds this run just wrote, as a "
        "curve over the scene -- a latency moves with load, and on the 2026-09 "
        "Odin1 rig one scene spends 80 of its 290 s at 0.05 s and the rest at "
        "0.69 s. A number applies that constant instead; 'off' leaves the "
        "stamps as recorded. What it changes is data_ts.txt, pcd_ts.txt and "
        "every pose sampled onto them; the camera/lidar pairing and the clouds "
        "themselves are untouched. Recorded per scene in time_align.txt",
    )
    parser.add_argument(
        "--gnss_guard",
        type=str,
        default="auto",
        metavar="auto|on|off",
        help="Drop stretches where the GNSS/INS published a reset solution "
        "instead of a pose. A Fixposition that has lost its fix -- which is "
        "what rolling the robot over does -- keeps publishing odometry at full "
        "rate, filled with a position hundreds of metres away, held perfectly "
        "still, with the attitude snapped to identity, and nothing in the "
        "recording flags it. Those samples are removed, and so are the frames "
        "they leave without a pose -- along with any frame the pose stream "
        "never covered, which would otherwise be exported carrying the nearest "
        "sample -- before any image or cloud is written. "
        "'auto' (the default) turns it on for an absolute source and off for "
        "wheel odometry, which legitimately holds still. Recorded per scene in "
        "gnss_guard.txt; the holes read as 'fix_lost' in odometry_segments.txt",
    )
    parser.add_argument(
        "--cam_time_offset",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Added to the camera header stamp before synchronisation, for a "
        "camera that stamps differently from the lidar beside it. On the "
        "2026-09 Odin1 rig the two agree to a median 4 ms, so this stays 0.0 "
        "there. Unlike --time_align this one re-pairs camera frames with "
        "sweeps, since it moves only one side of the match",
    )
    parser.add_argument("--base_frame", type=str, default="base_link")
    parser.add_argument(
        "--lidar_extrinsic",
        type=float,
        nargs=6,
        default=None,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        help="Manual lidar->base transform, used when TF does not connect the "
        "lidar frame to the base frame (the case for the 2026-08 Odin1 bags)",
    )
    parser.add_argument("--jobs", type=int, default=4, help="Worker processes")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract scenes whose output directory is already complete",
    )
    parser.add_argument(
        "--gt",
        choices=["auto", "on", "off"],
        default="auto",
        help="Write odometry.txt + odometry_segments.txt (v1.2 ground-truth "
        "contract) from the odometry stream, segmented at gaps and GNSS "
        "teleports. 'auto' enables it when --odom is a Fixposition topic.",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        nargs="+",
        default=None,
        metavar="SEL",
        help="Only process these scenes: indices (7), ranges (3-9) or bag "
        "directory names; comma-separated values are accepted too",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Index of the first scene directory (offsets the numbering)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)

    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    try:
        bags = find_bags(input_dir)
    except (FileNotFoundError, NotADirectoryError) as exc:
        LOG.error("%s", exc)
        return 2
    if not bags:
        LOG.error("no rosbag2 directories (with metadata.yaml) found under %s", input_dir)
        return 2

    jobs = [(args.start_index + i, bag) for i, bag in enumerate(bags)]
    selected = parse_scene_selection(args.scenes)
    if selected is not None:
        jobs = [(idx, bag) for idx, bag in jobs if selected(idx, bag.name)]
        if not jobs:
            LOG.error("--scenes %s selected none of the %d bags found", args.scenes, len(bags))
            return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    todo = []
    results: list[SceneResult] = []
    for idx, bag in jobs:
        scene_dir = output_dir / f"{idx:04d}"
        if not args.overwrite and scene_is_complete(scene_dir):
            LOG.info("scene %04d already complete (%s); skipping", idx, bag.name)
            results.append(SceneResult(index=idx, bag=bag.name, status=STATUS_SKIPPED))
            continue
        todo.append((idx, bag))

    LOG.info(
        "%d bag(s) found, %d to extract, %d skipped -> %s",
        len(bags),
        len(todo),
        len(jobs) - len(todo),
        output_dir,
    )

    if todo and args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(_worker, bag, output_dir, idx, args) for idx, bag in todo]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing bags"):
                results.append(future.result())
    else:
        for idx, bag in tqdm(todo, desc="Processing bags"):
            results.append(_worker(bag, output_dir, idx, args))

    csv_path = merge_scenes_csv(output_dir)
    # The viewer's disk figures. Every scene this run touched is rescanned,
    # failed ones too, since an overwrite may have left them smaller; a failure
    # here costs the figures, never the extraction.
    try:
        disk_usage.update(output_dir, {f"{idx:04d}" for idx, _ in todo})
    except (OSError, ValueError) as exc:
        LOG.warning("could not update %s: %s", disk_usage.NAME, exc)

    failed = [r for r in results if r.status == STATUS_FAILED]
    empty = [r for r in results if r.status == STATUS_EMPTY]
    ok = [r for r in results if r.status == STATUS_OK]
    skipped = [r for r in results if r.status == STATUS_SKIPPED]

    LOG.info(
        "done: %d extracted (%d frames), %d skipped, %d empty, %d failed; index -> %s",
        len(ok),
        sum(r.frames for r in ok),
        len(skipped),
        len(empty),
        len(failed),
        csv_path,
    )
    for result in empty:
        LOG.warning(
            "scene %04d (%s) produced no frames: %s",
            result.index,
            result.bag,
            result.message,
        )
    for result in failed:
        LOG.error("scene %04d (%s) FAILED: %s", result.index, result.bag, result.message)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
