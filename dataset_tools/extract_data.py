#!/usr/bin/env python3
import argparse
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import networkx as nx
from tqdm import tqdm
from cv_bridge import CvBridge
from rosbags.highlevel import AnyReader
from tf_transformations import quaternion_matrix, translation_matrix


def _worker(bag_path, output_dir, idx):
    scene_dir = output_dir / f"{idx:04d}"
    try:
        with AnyReader([bag_path]) as reader:
            process_scene(reader, bag_path, scene_dir)
        return True
    except Exception as e:
        print(f"Error processing {bag_path}: {e}\n{traceback.format_exc()}")
        return False


def main(args):
    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    bag_files = sorted(input_dir.iterdir())
    for file in bag_files:
        if not file.is_dir():
            bag_files.remove(file)

    num_threads = 1

    futures = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for idx, bag_path in enumerate(bag_files):
            futures.append(executor.submit(_worker, bag_path, output_dir, idx))

        for _ in tqdm(
            as_completed(futures), total=len(futures), desc="Processing scenes"
        ):
            pass


def process_scene(reader, bag_path, scene_dir):
    cv_bridge = CvBridge()

    camera_topic = "/luxonis/oak/rgb/image_raw/compressed"
    lidar_topic = "/livox/lidar_front"

    camera_topic = "/luxonis/rear/rgb/image_raw/compressed"
    lidar_topic = "/livox/lidar_rear"

    # Pre-cache connections
    conns = {c.topic: c for c in reader.connections}
    cam_conn = conns.get(camera_topic)
    lidar_conn = conns.get(lidar_topic)
    if not cam_conn or not lidar_conn:
        print(f"Missing camera or lidar topic in {bag_path}")
        return

    (scene_dir / "images").mkdir(parents=True, exist_ok=True)
    (scene_dir / "clouds").mkdir(parents=True, exist_ok=True)

    cam_ts, pcd_ts = get_synced_timestamps(reader, cam_conn, lidar_conn)
    data_ts_mean = (cam_ts + pcd_ts) / 2
    np.savetxt(scene_dir / "data_ts.txt", data_ts_mean)
    np.savetxt(scene_dir / "pcd_ts.txt", pcd_ts)

    extract_images(cv_bridge, reader, cam_conn, scene_dir, cam_ts)
    extract_pointclouds(reader, lidar_conn, scene_dir, pcd_ts)

    save_calibration(reader, conns, cam_conn, lidar_conn, scene_dir)
    save_joint_states(reader, conns, scene_dir)

    odom_conn = conns.get("/odom")
    odom_ts = extract_timestamps(reader, odom_conn)

    diff = np.abs(pcd_ts[:, None] - odom_ts[None, :])
    ts1_to_ts2 = np.argmin(diff, axis=1)
    timestamps = odom_ts[ts1_to_ts2]

    extract_odom_poses(reader, odom_conn, timestamps, scene_dir)


def extract_timestamps(reader, conn):
    ts = []
    for _, _, raw in reader.messages([conn]):
        try:
            msg = reader.deserialize(raw, conn.msgtype)
        except UnicodeDecodeError as _:
            continue
        ts.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
    return np.array(ts, dtype=np.float64)


def get_synced_timestamps(reader, conn1, conn2):
    ts1 = extract_timestamps(reader, conn1)
    ts2 = extract_timestamps(reader, conn2)

    # Vectorized mutual nearest matching
    diff = np.abs(ts1[:, None] - ts2[None, :])
    ts1_to_ts2 = np.argmin(diff, axis=1)
    ts2_to_ts1 = np.argmin(diff, axis=0)

    mask = np.array([ts2_to_ts1[j] == i for i, j in enumerate(ts1_to_ts2)])
    return ts1[mask], ts2[ts1_to_ts2[mask]]


def extract_images(cv_bridge, reader, cam_conn, save_dir, valid_ts):
    ts_set = set(valid_ts)  # tolerate float error
    img_dir = save_dir / "images"

    idx = 0
    used_set = set()
    for _, _, raw in reader.messages([cam_conn]):
        try:
            msg = reader.deserialize(raw, cam_conn.msgtype)
        except UnicodeDecodeError as _:
            continue
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if ts not in ts_set:
            continue
        if ts in used_set:
            continue
        used_set.add(ts)
        cv_img = cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
        cv2.imwrite(str(img_dir / f"{idx:05d}.png"), cv_img)
        idx += 1

    # Extract intrinsics once
    cam_info_topic = cam_conn.topic.replace("image_raw/compressed", "camera_info")
    cam_info_conn = next(
        (c for c in reader.connections if c.topic == cam_info_topic), None
    )
    if cam_info_conn:
        _, _, raw = next(reader.messages([cam_info_conn]))
        msg = reader.deserialize(raw, cam_info_conn.msgtype)
        intrinsics = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        np.savetxt(save_dir / "camera_intrinsics.txt", intrinsics)


def extract_odom_poses(reader, odom_conn, timestamps, save_dir):
    ts_set = set(timestamps)
    prev_pose = None
    transforms = []

    used_ts = []
    for _, _, raw in reader.messages([odom_conn]):
        try:
            msg = reader.deserialize(raw, odom_conn.msgtype)
        except UnicodeDecodeError as _:
            continue
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if ts not in ts_set:
            continue

        # Extract current pose
        pose = msg.pose.pose
        Tt = translation_matrix([pose.position.x, pose.position.y, pose.position.z])
        Tr = quaternion_matrix(
            [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ]
        )

        T = np.dot(Tt, Tr)
        if prev_pose is None:
            transform = np.eye(4)
        else:
            transform = np.linalg.inv(prev_pose) @ T

        transforms.append(transform)
        used_ts.append(ts)
        prev_pose = T

    transforms = np.array(transforms)
    mask = [np.where(used_ts == x)[0] for x in timestamps]
    transforms = transforms[mask].squeeze()
    np.save(save_dir / "transforms.npy", transforms)


def extract_pointclouds(reader, lidar_conn, save_dir, valid_ts):
    ts_set = set(valid_ts)
    cloud_dir = save_dir / "clouds"

    endian_char = ">" if lidar_conn.msgtype.endswith("BigEndian") else "<"
    dtype = np.dtype(
        [
            ("x", endian_char + "f4"),
            ("y", endian_char + "f4"),
            ("z", endian_char + "f4"),
            ("intensity", endian_char + "f4"),
            ("tag", endian_char + "u1"),
            ("line", endian_char + "u1"),
            ("timestamp", endian_char + "f8"),
        ]
    )

    idx = 0
    used_set = set()
    for _, _, raw in reader.messages([lidar_conn]):
        try:
            msg = reader.deserialize(raw, lidar_conn.msgtype)
        except UnicodeDecodeError as _:
            continue
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if ts not in ts_set:
            continue
        if ts in used_set:
            continue
        used_set.add(ts)

        points = np.frombuffer(msg.data, dtype=dtype)
        pcd = np.vstack((points["x"], points["y"], points["z"], points["intensity"])).T
        pcd = pcd[~np.isnan(pcd).any(axis=1)]
        np.savez_compressed(cloud_dir / f"{idx:05d}.npz", points=pcd)
        idx += 1


def save_calibration(reader, conns, cam_conn, lidar_conn, save_dir):
    tf_conn = conns.get("/tf_static")
    if not tf_conn:
        print("No /tf_static topic found — skipping calibration.")
        return

    # Extract one message from each to get frame IDs
    _, _, raw = next(reader.messages([cam_conn]))
    cam_frame = reader.deserialize(raw, cam_conn.msgtype).header.frame_id
    _, _, raw = next(reader.messages([lidar_conn]))
    lidar_frame = reader.deserialize(raw, lidar_conn.msgtype).header.frame_id

    tf_frames = [
        t
        for _, _, raw in reader.messages([tf_conn])
        for msg in [reader.deserialize(raw, tf_conn.msgtype)]
        for t in msg.transforms
    ]

    G = build_tf_tree(tf_frames)
    T = compute_transform(G, cam_frame, lidar_frame)
    if T is not None:
        np.savetxt(save_dir / "T_cam_to_lidar.txt", T)
    else:
        print(f"Calibration failed between {cam_frame} and {lidar_frame}.")


def build_tf_tree(tf_msgs):
    G = nx.DiGraph()
    for tf in tf_msgs:
        q = tf.transform.rotation
        Tr = quaternion_matrix([q.x, q.y, q.z, q.w])
        Tr[:3, 3] = [
            tf.transform.translation.x,
            tf.transform.translation.y,
            tf.transform.translation.z,
        ]
        parent, child = tf.header.frame_id, tf.child_frame_id
        G.add_edge(parent, child, transform=Tr)
        G.add_edge(child, parent, transform=np.linalg.inv(Tr))
    return G


def compute_transform(G, source, target):
    try:
        path = nx.shortest_path(G, source=target, target=source)
    except nx.NetworkXNoPath:
        return None
    T = np.eye(4)
    for i in range(len(path) - 1):
        T = T @ G[path[i]][path[i + 1]]["transform"]
    return T


def save_joint_states(reader, conns, save_dir):
    js_conn = conns.get("/joint_states")
    data = []
    for _, _, raw in reader.messages([js_conn]):
        msg = reader.deserialize(raw, js_conn.msgtype)
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        data.append(np.r_[ts, msg.position, msg.velocity, msg.effort])
    data = np.vstack(data)
    np.save(save_dir / "joint_states.npy", data)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir", type=str, help="Directory containing ROS2 bag files."
    )
    parser.add_argument(
        "output_dir", type=str, help="Directory to save extracted data."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
