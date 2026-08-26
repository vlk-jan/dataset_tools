#!/usr/bin/env python3
import argparse
import math
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import networkx as nx
from tqdm import tqdm
from rosbags.highlevel import AnyReader

try:
    from dataset_tools import cloudini
except ImportError:
    import cloudini


# ---------------------------------------------------------------------------
# Small transform helpers (replacing tf_transformations, 'sxyz' convention)
# ---------------------------------------------------------------------------
def quaternion_matrix(q):
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


def euler_matrix(roll, pitch, yaw):
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


def roll_pitch_from_quat(q):
    """Roll and pitch ('sxyz') of quaternion [x, y, z, w]."""
    R = quaternion_matrix(q)
    roll = math.atan2(R[2, 1], R[2, 2])
    pitch = math.asin(np.clip(-R[2, 0], -1.0, 1.0))
    return roll, pitch


def pose_to_matrix(pose):
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


def fields_to_dtype(fields, point_step, big_endian=False):
    endian = ">" if big_endian else "<"
    return np.dtype(
        {
            "names": [f.name for f in fields],
            "formats": [endian + PC2_DTYPES[f.datatype] for f in fields],
            "offsets": [f.offset for f in fields],
            "itemsize": point_step,
        }
    )


def cloud_to_xyzi(msg):
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


# ---------------------------------------------------------------------------
# TF helpers
# ---------------------------------------------------------------------------
def build_tf_graph(tf_msgs):
    G = nx.DiGraph()
    for tf in tf_msgs:
        q = tf.transform.rotation
        Tr = quaternion_matrix([q.x, q.y, q.z, q.w])
        Tr[:3, 3] = [
            tf.transform.translation.x,
            tf.transform.translation.y,
            tf.transform.translation.z,
        ]
        G.add_edge(tf.header.frame_id, tf.child_frame_id, transform=Tr)
        G.add_edge(tf.child_frame_id, tf.header.frame_id, transform=np.linalg.inv(Tr))
    return G


def get_transform(G, src_frame, dst_frame):
    """Transform T such that p_dst = T @ p_src, or None if no TF path exists."""
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
        T = T @ G[path[i]][path[i + 1]]["transform"]
    return T


# ---------------------------------------------------------------------------
def _worker(bag_path, output_dir, idx, args):
    scene_dir = output_dir / f"{idx:04d}"
    try:
        with AnyReader([bag_path]) as reader:
            process_scene(reader, bag_path, scene_dir, args)
        return True
    except Exception as e:
        print(f"Error processing {bag_path}: {e}\n{traceback.format_exc()}")
        return False


def process_scene(reader, bag_path, scene_dir, args):
    camera_topic, lidar_topic = args.cam, args.lidar

    # Pre-cache connections
    conns = {c.topic: c for c in reader.connections}
    cam_conn = conns.get(camera_topic)
    lidar_conn = conns.get(lidar_topic)

    if not cam_conn or not lidar_conn:
        print(
            f"Missing topics in {bag_path}. Cam: {bool(cam_conn)}, Lidar: {bool(lidar_conn)}"
        )
        return

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
                np.r_[t, tw.linear.x, tw.linear.y, tw.linear.z,
                      tw.angular.x, tw.angular.y, tw.angular.z]
            )
        elif conn.topic == "/tf_static":
            tf_msgs.extend(msg.transforms)

    # Sync Timestamps (Mutual Nearest Neighbor)
    cam_ts = np.array(cam_timestamps)
    pcd_ts = np.array(lidar_timestamps)

    if len(cam_ts) == 0 or len(pcd_ts) == 0:
        return

    # Work in sorted order (also handles out-of-order messages); searchsorted
    # keeps memory O(n) instead of the O(n*m) full distance matrix.
    cam_ts = np.sort(cam_ts)
    pcd_ts = np.sort(pcd_ts)

    def nearest_idx(a, b):
        """For each value in `a`, index of the nearest value in sorted `b`."""
        if len(b) == 1:
            return np.zeros(len(a), dtype=int)
        pos = np.clip(np.searchsorted(b, a), 1, len(b) - 1)
        return np.where(a - b[pos - 1] <= b[pos] - a, pos - 1, pos)

    ts1_to_ts2 = nearest_idx(cam_ts, pcd_ts)
    ts2_to_ts1 = nearest_idx(pcd_ts, cam_ts)
    mask = ts2_to_ts1[ts1_to_ts2] == np.arange(len(cam_ts))

    valid_cam_ts = cam_ts[mask]
    valid_pcd_ts = pcd_ts[ts1_to_ts2[mask]]

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
            print(
                f"No TF path from '{lidar_frame}' to '{args.base_frame}' in {bag_path} "
                "and no --lidar_extrinsic given; using identity."
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
        print(
            f"No TF path from odom child '{odom_child_frame}' to '{args.base_frame}'; "
            "storing raw odometry poses."
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
        print(
            f"No TF path from orientation frame '{imu_frame}' to '{args.base_frame}'; "
            "using raw orientations for gravity alignment."
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
            xyzi = cloud_to_xyzi(msg)
            xyzi = xyzi[~np.isnan(xyzi).any(axis=1)]
            # Drop all-zero points (unreturned rays in organized clouds)
            xyzi = xyzi[np.any(xyzi[:, :3] != 0, axis=1)]
            pcd = np.hstack((xyzi[:, :3], np.ones((len(xyzi), 1), dtype=np.float32)))
            pcd = (pcd @ T_lidar_to_base.T.astype(np.float32))[:, :3]
            np.save(
                scene_dir / "clouds" / f"{pcd_idx:05d}.npy",
                np.hstack((pcd, xyzi[:, 3:4])),
            )
            pcd_set.remove(t)

    # Process Odom Transforms (relative to first frame)
    if transforms_dict:
        # Reconstruct sequential list based on chronological indices
        transforms = [transforms_dict[i] for i in range(len(valid_cam_ts))]
        rel_transforms = [np.eye(4)]
        first_inv = np.linalg.inv(transforms[0])
        for i in range(1, len(transforms)):
            rel_transforms.append(first_inv @ transforms[i])
        np.save(scene_dir / "transforms.npy", np.array(rel_transforms))

    # Process gravity alignment
    if not imu_data:
        print(f"No orientation messages on '{args.imu}'; skipping gravity_align.npy")
        return
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

        # Create the gravity alignment rotation matrix (corrects pitch/roll, ignores yaw)
        R_align = euler_matrix(roll, pitch, 0.0)[:3, :3]
        gravity_aligns.append(R_align)

    np.save(scene_dir / "gravity_align.npy", np.array(gravity_aligns))


def get_first_frame(reader, conn):
    for _, _, raw in reader.messages([conn]):
        return reader.deserialize(raw, conn.msgtype).header.frame_id


def save_intrinsics(reader, conns, cam_topic, scene_dir):
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
        print(f"No camera_info topic found for {cam_topic} (tried {candidates})")
        return
    for _, _, raw in reader.messages([conn]):
        msg = reader.deserialize(raw, conn.msgtype)
        np.savetxt(scene_dir / "camera_intrinsics.txt", np.array(msg.k).reshape(3, 3))
        break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--cam", type=str, default="/odin1/image/compressed")
    parser.add_argument("--lidar", type=str, default="/odin1/cloud_filtered/cloudini")
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
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bag_files = sorted([d for d in input_dir.iterdir() if d.is_dir()])

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = [
            executor.submit(_worker, b, output_dir, i, args)
            for i, b in enumerate(bag_files)
        ]
        for _ in tqdm(
            as_completed(futures), total=len(futures), desc="Processing bags"
        ):
            pass


if __name__ == "__main__":
    main()
