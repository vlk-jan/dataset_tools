#!/usr/bin/env python3
import argparse
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import networkx as nx
from tqdm import tqdm
from rosbags.highlevel import AnyReader
from tf_transformations import (
    quaternion_matrix,
    translation_matrix,
    euler_matrix,
    euler_from_matrix,
)


def _worker(bag_path, output_dir, idx, cam_topic, lidar_topic):
    scene_dir = output_dir / f"{idx:04d}"
    try:
        with AnyReader([bag_path]) as reader:
            process_scene(reader, bag_path, scene_dir, cam_topic, lidar_topic)
        return True
    except Exception as e:
        print(f"Error processing {bag_path}: {e}\n{traceback.format_exc()}")
        return False


def process_scene(reader, bag_path, scene_dir, camera_topic, lidar_topic):
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
    odom_data = []  # (ts, transform)
    imu_data = []  # (ts, roll, pitch)
    joint_states = []

    relevant_conns = [
        cam_conn,
        lidar_conn,
        conns.get("/odom"),
        conns.get("/ekf/imu/data"),
        conns.get("/joint_states"),
        conns.get("/tf_static"),
    ]
    relevant_conns = [c for c in relevant_conns if c is not None]

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
        elif conn.topic == "/odom":
            # Pre-calculate odom transform
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
            odom_data.append((t, Tt @ Tr))
        elif conn.topic == "/ekf/imu/data":
            q = msg.orientation
            roll, pitch, _ = euler_from_matrix(quaternion_matrix([q.x, q.y, q.z, q.w]))
            imu_data.append((t, roll, pitch))
        elif conn.topic == "/joint_states":
            joint_states.append(np.r_[t, msg.position, msg.velocity, msg.effort])
        elif conn.topic == "/tf_static":
            tf_msgs.extend(msg.transforms)

    # Sync Timestamps (Mutual Nearest Neighbor)
    cam_ts = np.array(cam_timestamps)
    pcd_ts = np.array(lidar_timestamps)

    if len(cam_ts) == 0 or len(pcd_ts) == 0:
        return

    diff = np.abs(cam_ts[:, None] - pcd_ts[None, :])
    ts1_to_ts2 = np.argmin(diff, axis=1)
    ts2_to_ts1 = np.argmin(diff, axis=0)
    mask = np.array([ts2_to_ts1[j] == i for i, j in enumerate(ts1_to_ts2)])

    valid_cam_ts = cam_ts[mask]
    valid_pcd_ts = pcd_ts[ts1_to_ts2[mask]]

    # Save metadata
    np.savetxt(scene_dir / "data_ts.txt", (valid_cam_ts + valid_pcd_ts) / 2)
    np.savetxt(scene_dir / "pcd_ts.txt", valid_pcd_ts)

    if joint_states:
        np.save(scene_dir / "joint_states.npy", np.vstack(joint_states))

    # Process TF Static
    if tf_msgs:
        T_lidar_to_base = get_lidar_transform(tf_msgs, reader, lidar_conn)
    else:
        T_lidar_to_base = np.eye(4)

    # Pass 2: Extract heavy data (Images, Pointclouds)
    cam_set = set(valid_cam_ts)
    pcd_set = set(valid_pcd_ts)

    # Pointcloud dtype
    endian = ">" if lidar_conn.msgtype.endswith("BigEndian") else "<"
    pcd_dtype = np.dtype(
        [
            ("x", f"{endian}f4"),
            ("y", f"{endian}f4"),
            ("z", f"{endian}f4"),
            ("intensity", f"{endian}f4"),
            ("tag", f"{endian}u1"),
            ("line", f"{endian}u1"),
            ("timestamp", f"{endian}f8"),
        ]
    )

    img_idx = 0
    pcd_idx = 0

    # For odom interpolation/matching
    odom_ts = np.array([x[0] for x in odom_data])
    odom_poses = [x[1] for x in odom_data]
    transforms = []

    for conn, timestamp, raw in reader.messages([cam_conn, lidar_conn]):
        msg = reader.deserialize(raw, conn.msgtype)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if conn.topic == camera_topic and t in cam_set:
            # Use OpenCV directly
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            cv2.imwrite(str(scene_dir / "images" / f"{img_idx:05d}.png"), cv_img)

            # Extract Intrinsics once
            if img_idx == 0:
                save_intrinsics(reader, conns, camera_topic, scene_dir)

            # Match Odom
            idx = np.argmin(np.abs(odom_ts - t))
            transforms.append(odom_poses[idx])

            img_idx += 1
            cam_set.remove(t)  # Handle duplicates

        elif conn.topic == lidar_topic and t in pcd_set:
            points = np.frombuffer(msg.data, dtype=pcd_dtype)
            pcd = np.vstack(
                (points["x"], points["y"], points["z"], np.ones_like(points["x"]))
            ).T
            pcd = (pcd[~np.isnan(pcd).any(axis=1)] @ T_lidar_to_base.T)[:, :3]
            pcd = np.hstack((pcd, points["intensity"][:, None]))
            np.save(
                scene_dir / "clouds" / f"{pcd_idx:05d}.npy",
                pcd,
            )
            pcd_idx += 1
            pcd_set.remove(t)

    # Process Odom Transforms (relative to first frame)
    if transforms:
        rel_transforms = [np.eye(4)]
        first_inv = np.linalg.inv(transforms[0])
        for i in range(1, len(transforms)):
            rel_transforms.append(first_inv @ transforms[i])
        np.save(scene_dir / "transforms.npy", np.array(rel_transforms))

    # Process gravity alignment
    imu_ts = np.array([x[0] for x in imu_data])
    imu_rolls = np.array([x[1] for x in imu_data])
    imu_pitches = np.array([x[2] for x in imu_data])

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


def save_intrinsics(reader, conns, cam_topic, scene_dir):
    info_topic = cam_topic.replace("image_raw/compressed", "camera_info")
    conn = conns.get(info_topic)
    if conn:
        for _, _, raw in reader.messages([conn]):
            msg = reader.deserialize(raw, conn.msgtype)
            np.savetxt(
                scene_dir / "camera_intrinsics.txt", np.array(msg.k).reshape(3, 3)
            )
            break


def get_lidar_transform(tf_msgs, reader, lidar_conn):
    # Extract frame IDs from first messages
    def get_frame(conn):
        for _, _, raw in reader.messages([conn]):
            return reader.deserialize(raw, conn.msgtype).header.frame_id

    l_frame = get_frame(lidar_conn)

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

    # Extract Lidar to Base Link transform
    base_frame = "base_link"
    T_lidar_to_base = np.eye(4)
    if base_frame in G:
        try:
            path = nx.shortest_path(G, source=l_frame, target=base_frame)
            for i in range(len(path) - 1):
                T_lidar_to_base = T_lidar_to_base @ G[path[i]][path[i + 1]]["transform"]
        except Exception:
            print(f"Failed to find TF path from {l_frame} to {base_frame}")
    else:
        print(f"Frame '{base_frame}' not found in TF static tree.")

    return T_lidar_to_base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument(
        "--cam", type=str, default="/luxonis/oak/rgb/image_raw/compressed"
    )
    parser.add_argument("--lidar", type=str, default="/livox/lidar_front")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bag_files = sorted([d for d in input_dir.iterdir() if d.is_dir()])

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = [
            executor.submit(_worker, b, output_dir, i, args.cam, args.lidar)
            for i, b in enumerate(bag_files)
        ]
        for _ in tqdm(
            as_completed(futures), total=len(futures), desc="Processing bags"
        ):
            pass


if __name__ == "__main__":
    main()
