#!/usr/bin/env python3

import os

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np


class OdomRecorder(Node):
    def __init__(self):
        super().__init__("odom_recorder")

        # --- Parameters ---
        self.declare_parameter("topic", "/Odometry")
        self.declare_parameter("file_path", "trajectory_matrices.txt")

        topic_name = self.get_parameter("topic").get_parameter_value().string_value
        self.file_path = self.get_parameter("file_path").get_parameter_value().string_value

        # --- Subscriber ---
        self.subscription = self.create_subscription(Odometry, topic_name, self.odom_callback, 10)

        # --- File Initialization ---
        # Create directory if it doesn't exist
        directory = os.path.dirname(self.file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        # Open file in write mode to clear it, then close
        with open(self.file_path, "w") as f:
            f.write("# Timestamp(sec) r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz 0 0 0 1\n")

        self.get_logger().info(f"Recording odometry from '{topic_name}' to '{self.file_path}'")

    def odom_callback(self, msg):
        # 1. Extract Position
        tx = msg.pose.pose.position.x
        ty = msg.pose.pose.position.y
        tz = msg.pose.pose.position.z

        # 2. Extract Orientation (Quaternion)
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w

        # 3. Convert Quaternion to Rotation Matrix
        R = np.array(
            [
                [
                    1 - 2 * qy**2 - 2 * qz**2,
                    2 * qx * qy - 2 * qz * qw,
                    2 * qx * qz + 2 * qy * qw,
                ],
                [
                    2 * qx * qy + 2 * qz * qw,
                    1 - 2 * qx**2 - 2 * qz**2,
                    2 * qy * qz - 2 * qx * qw,
                ],
                [
                    2 * qx * qz - 2 * qy * qw,
                    2 * qy * qz + 2 * qx * qw,
                    1 - 2 * qx**2 - 2 * qy**2,
                ],
            ]
        )

        # 4. Construct 4x4 Transformation Matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[0, 3] = tx
        T[1, 3] = ty
        T[2, 3] = tz

        # 5. Get Timestamp
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # 6. Save to file
        # Format: timestamp r11 r12 r13 tx ... 0 0 0 1 (Flattened row-major)
        flattened_matrix = T.flatten()

        line_str = f"{timestamp:.6f} " + " ".join([f"{val:.6f}" for val in flattened_matrix]) + "\n"

        with open(self.file_path, "a") as f:
            f.write(line_str)


def main(args=None):
    rclpy.init(args=args)

    recorder = OdomRecorder()

    try:
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        pass
    finally:
        recorder.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
