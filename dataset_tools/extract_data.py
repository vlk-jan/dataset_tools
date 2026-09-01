#!/usr/bin/env python3
"""Extract HELHEST scenes from ROS 2 bags (rosbag2 sqlite3 or mcap).

One scene directory is written per input bag; see README.md for the exact
output format. Bags are read with the pure-Python ``rosbags`` library, so no
ROS installation is required.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import networkx as nx
from tqdm import tqdm
from rosbags.highlevel import AnyReader

try:
    from dataset_tools import cloudini, rosbags_compat  # noqa: F401  (patches rosbags)
except ImportError:  # when the file is run directly from dataset_tools/
    import cloudini
    import rosbags_compat  # noqa: F401


LOG = logging.getLogger("helhest.extract")

SCENE_DIR_RE = re.compile(r"^\d{4,}$")
#: Written as the very last step of a scene, so its presence marks completion.
SENTINEL_FILE = "source_bag.txt"
SCENES_CSV = "scenes.csv"
SCENES_PARTS_DIR = ".scenes_parts"
SCENES_CSV_HEADER = ["scene", "bag", "start_time", "frames"]

STATUS_OK = "ok"
STATUS_SKIPPED = "skipped"
STATUS_EMPTY = "empty"
STATUS_FAILED = "failed"


# ---------------------------------------------------------------------------
# Small transform helpers (replacing tf_transformations, 'sxyz' convention)
# ---------------------------------------------------------------------------
def quaternion_matrix(q: Sequence[float]) -> np.ndarray:
    """4x4 homogeneous rotation matrix from quaternion [x, y, z, w]."""
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(4)
    s = 2.0 / n
    T = np.eye(4)
    T[:3, :3] = [
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ]
    return T


def euler_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """4x4 matrix R = Rz(yaw) @ Ry(pitch) @ Rx(roll) ('sxyz')."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    return T


def roll_pitch_from_quat(q: Sequence[float]) -> tuple[float, float]:
    """Roll and pitch ('sxyz') of quaternion [x, y, z, w]."""
    R = quaternion_matrix(q)
    roll = math.atan2(R[2, 1], R[2, 2])
    pitch = math.asin(np.clip(-R[2, 0], -1.0, 1.0))
    return roll, pitch


def pose_to_matrix(pose) -> np.ndarray:
    T = quaternion_matrix(
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    )
    T[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return T


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
    if hasattr(msg, "compressed_data"):  # point_cloud_interfaces/CompressedPointCloud2
        data = cloudini.decode(bytes(msg.compressed_data))
    else:
        data = bytes(msg.data)
    dtype = fields_to_dtype(msg.fields, msg.point_step, msg.is_bigendian)
    points = np.frombuffer(data, dtype=dtype, count=msg.width * msg.height)
    names = points.dtype.names
    intensity = (
        points["intensity"].astype(np.float32)
        if "intensity" in names
        else np.zeros(len(points), dtype=np.float32)
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

    This MUST run before the cloud is transformed into base_link. Afterwards
    the zeros are no longer at the origin -- they land on the mount
    translation, where they stop looking like a sentinel and start looking
    like a dense real object a few centimetres from the sensor. That is what
    happened to HELHEST v1: 54-70% of every published cloud is one coordinate
    repeated ten to fourteen thousand times, at (-0.170, 0, 0.145).

    Measured on the 2026-08-21 Odin1 bags, per sweep: 49152 elements, 11.2%
    NaN, then 47.1% of the remainder dropped shots, leaving ~23k real returns.

    Exact zeros only. A point at (0, 0, 1e-6) is a real return from something
    against the sensor and is kept -- range gating is the consumer's decision,
    not this function's.
    """
    xyzi = xyzi[~np.isnan(xyzi).any(axis=1)]
    return xyzi[np.any(xyzi[:, :3] != 0, axis=1)]


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
        try:
            parts_dir.rmdir()
        except OSError:  # someone else is still writing; harmless
            pass
    return csv_path


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


def process_scene(reader, bag_path: Path, scene_dir: Path, args) -> SceneResult:
    camera_topic, lidar_topic = args.cam, args.lidar
    result = SceneResult(index=-1, bag=bag_path.name, status=STATUS_OK)

    # Pre-cache connections
    conns = {c.topic: c for c in reader.connections}
    cam_conn = conns.get(camera_topic)
    lidar_conn = conns.get(lidar_topic)

    if not cam_conn or not lidar_conn:
        missing = [
            topic
            for topic, conn in ((camera_topic, cam_conn), (lidar_topic, lidar_conn))
            if conn is None
        ]
        LOG.warning("%s: missing topic(s) %s; scene skipped", bag_path.name, missing)
        result.status = STATUS_EMPTY
        result.message = f"missing topic(s): {', '.join(missing)}"
        return result

    (scene_dir / "images").mkdir(parents=True, exist_ok=True)
    (scene_dir / "clouds").mkdir(parents=True, exist_ok=True)

    # Pass 1: Collect timestamps for synchronization
    # We only deserialize headers for efficiency
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

    relevant_conns = [
        cam_conn,
        lidar_conn,
        conns.get(args.odom),
        conns.get(args.imu),
        conns.get("/joint_states"),
        conns.get("/joint_setpoints"),
        conns.get("/cmd_vel"),
        conns.get("/tf_static"),
    ]
    # Deduplicate (e.g. when --odom and --imu are the same topic)
    relevant_conns = list({id(c): c for c in relevant_conns if c is not None}.values())

    tf_msgs = []

    for conn, timestamp, raw in reader.messages(relevant_conns):
        msg = reader.deserialize(raw, conn.msgtype)

        # Get header timestamp if available
        t = timestamp
        if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if conn.topic == camera_topic:
            cam_timestamps.append(t)
        elif conn.topic == lidar_topic:
            lidar_timestamps.append(t)
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
        elif conn.topic == "/tf_static":
            tf_msgs.extend(msg.transforms)

    # Sync timestamps (mutual nearest neighbour)
    valid_cam_ts, valid_pcd_ts = sync_timestamps(cam_timestamps, lidar_timestamps)

    if len(valid_cam_ts) == 0:
        LOG.warning(
            "%s: no synchronised camera/lidar pairs (%d camera, %d lidar messages); "
            "scene left empty",
            bag_path.name,
            len(cam_timestamps),
            len(lidar_timestamps),
        )
        result.status = STATUS_EMPTY
        result.message = f"no synced pairs ({len(cam_timestamps)} cam, {len(lidar_timestamps)} lidar)"
        return result

    # Map timestamps to their chronological frame indices
    cam_ts_to_idx = {ts: i for i, ts in enumerate(valid_cam_ts)}
    pcd_ts_to_idx = {ts: i for i, ts in enumerate(valid_pcd_ts)}

    # Save metadata
    np.savetxt(scene_dir / "data_ts.txt", (valid_cam_ts + valid_pcd_ts) / 2)
    np.savetxt(scene_dir / "pcd_ts.txt", valid_pcd_ts)

    for name, rows in [
        ("joint_states", joint_states),
        ("joint_setpoints", joint_setpoints),
        ("cmd_vel", cmd_vels),
    ]:
        if rows:
            arr = np.vstack(rows)
            np.save(scene_dir / f"{name}.npy", arr[np.argsort(arr[:, 0])])
        else:
            LOG.info(
                "%s: no messages on /%s; %s.npy not written", bag_path.name, name, name
            )

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
                "%s: no TF path from '%s' to '%s' and no --lidar_extrinsic given; "
                "using identity",
                bag_path.name,
                lidar_frame,
                args.base_frame,
            )
            T_lidar_to_base = np.eye(4)

    # Odom poses track the odometry child frame (e.g. FP_POI for Fixposition);
    # re-express them as poses of the base frame when TF allows it.
    T_child_to_base = (
        get_transform(tf_graph, odom_child_frame, args.base_frame)
        if odom_child_frame
        else None
    )
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
    T_imu_to_base = (
        get_transform(tf_graph, imu_frame, args.base_frame) if imu_frame else None
    )
    if T_imu_to_base is not None and not np.allclose(T_imu_to_base, np.eye(4)):
        T_base_in_imu = np.linalg.inv(T_imu_to_base)
        imu_data = [(t, R @ T_base_in_imu) for t, R in imu_data]
    elif imu_frame and T_imu_to_base is None:
        LOG.warning(
            "%s: no TF path from orientation frame '%s' to '%s'; using raw "
            "orientations for gravity alignment",
            bag_path.name,
            imu_frame,
            args.base_frame,
        )

    # Pass 2: Extract heavy data (Images, Pointclouds)
    cam_set = set(valid_cam_ts)
    pcd_set = set(valid_pcd_ts)

    # For odom interpolation/matching
    odom_ts = np.array([x[0] for x in odom_data])
    odom_poses = [x[1] for x in odom_data]

    transforms_dict = {}
    intrinsics_saved = False

    for conn, timestamp, raw in reader.messages([cam_conn, lidar_conn]):
        msg = reader.deserialize(raw, conn.msgtype)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if conn.topic == camera_topic and t in cam_set:
            img_idx = cam_ts_to_idx[t]  # Get correct chronological index

            # Use OpenCV directly
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            cv2.imwrite(str(scene_dir / "images" / f"{img_idx:05d}.png"), cv_img)

            # Extract Intrinsics once
            if not intrinsics_saved:
                save_intrinsics(reader, conns, camera_topic, scene_dir)
                intrinsics_saved = True

            # Match Odom and save via dict instead of list append
            if len(odom_ts):
                idx = np.argmin(np.abs(odom_ts - t))
                transforms_dict[img_idx] = odom_poses[idx]

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

    # Process Odom Transforms (relative to first frame)
    if transforms_dict:
        missing = [i for i in range(len(valid_cam_ts)) if i not in transforms_dict]
        if missing:
            raise RuntimeError(
                f"odometry pose missing for {len(missing)} of {len(valid_cam_ts)} "
                "frames (duplicate camera timestamps in the bag?)"
            )
        # Reconstruct sequential list based on chronological indices
        transforms = [transforms_dict[i] for i in range(len(valid_cam_ts))]
        rel_transforms = [np.eye(4)]
        first_inv = np.linalg.inv(transforms[0])
        for i in range(1, len(transforms)):
            rel_transforms.append(first_inv @ transforms[i])
        np.save(scene_dir / "transforms.npy", np.array(rel_transforms))
    else:
        LOG.warning(
            "%s: no odometry messages on '%s'; transforms.npy not written",
            bag_path.name,
            args.odom,
        )

    # Process gravity alignment
    if imu_data:
        imu_ts = np.array([x[0] for x in imu_data])
        imu_rolls = np.empty(len(imu_data))
        imu_pitches = np.empty(len(imu_data))
        for i, (_, R) in enumerate(imu_data):
            imu_rolls[i] = math.atan2(R[2, 1], R[2, 2])
            imu_pitches[i] = math.asin(np.clip(-R[2, 0], -1.0, 1.0))

        gravity_aligns = []
        for pcd_t in valid_pcd_ts:
            # Find the closest IMU message to the Lidar timestamp
            idx = np.argmin(np.abs(imu_ts - pcd_t))

            roll = imu_rolls[idx]
            pitch = imu_pitches[idx]

            # Gravity alignment rotation (corrects pitch/roll, ignores yaw)
            R_align = euler_matrix(roll, pitch, 0.0)[:3, :3]
            gravity_aligns.append(R_align)

        np.save(scene_dir / "gravity_align.npy", np.array(gravity_aligns))
    else:
        LOG.warning(
            "%s: no orientation messages on '%s'; gravity_align.npy not written",
            bag_path.name,
            args.imu,
        )

    # Provenance; written last, so it doubles as the "scene finished" marker.
    (scene_dir / SENTINEL_FILE).write_text(f"{bag_path.name}\n")

    result.frames = int(len(valid_cam_ts))
    result.start_time = float((valid_cam_ts[0] + valid_pcd_ts[0]) / 2)
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
    parser.add_argument(
        "input_dir", type=str, help="Directory of bags, or a single bag directory"
    )
    parser.add_argument("output_dir", type=str, help="Dataset root to write scenes to")
    parser.add_argument(
        "--cam",
        type=str,
        default="/odin1/image/compressed",
        help="Compressed image topic (old bags: "
        "/luxonis/oak/rgb/image_raw/compressed)",
    )
    parser.add_argument(
        "--lidar",
        type=str,
        default="/odin1/cloud_filtered/cloudini",
        help="PointCloud2 or CompressedPointCloud2 topic (old bags: "
        "/livox/lidar_front)",
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
        LOG.error(
            "no rosbag2 directories (with metadata.yaml) found under %s", input_dir
        )
        return 2

    jobs = [(args.start_index + i, bag) for i, bag in enumerate(bags)]
    selected = parse_scene_selection(args.scenes)
    if selected is not None:
        jobs = [(idx, bag) for idx, bag in jobs if selected(idx, bag.name)]
        if not jobs:
            LOG.error(
                "--scenes %s selected none of the %d bags found", args.scenes, len(bags)
            )
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
            futures = [
                executor.submit(_worker, bag, output_dir, idx, args)
                for idx, bag in todo
            ]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing bags"
            ):
                results.append(future.result())
    else:
        for idx, bag in tqdm(todo, desc="Processing bags"):
            results.append(_worker(bag, output_dir, idx, args))

    csv_path = merge_scenes_csv(output_dir)

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
        LOG.error(
            "scene %04d (%s) FAILED: %s", result.index, result.bag, result.message
        )

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
