#!/usr/bin/env python3
"""ROS 2 node that records a ``nav_msgs/Odometry`` topic as a text trajectory.

One line per message: ``timestamp`` followed by the row-major 4x4 pose, the
same layout as a scene's ``odometry.txt`` (see :mod:`dataset_tools.gt_io`).
"""

import os

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node

from dataset_tools.gt_io import GT_HEADER, format_pose_line
from dataset_tools.transforms import pose_to_matrix


class OdomRecorder(Node):
    def __init__(self):
        super().__init__("odom_recorder")

        # --- Parameters ---
        self.declare_parameter("topic", "/Odometry")
        self.declare_parameter("file_path", "trajectory_matrices.txt")

        topic_name = self.get_parameter("topic").get_parameter_value().string_value
        self.file_path = self.get_parameter("file_path").get_parameter_value().string_value

        # --- File ---
        directory = os.path.dirname(self.file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        # One handle for the whole run, flushed per message: a line is on disk
        # as soon as it is written, without reopening the file at 100 Hz.
        self._file = open(self.file_path, "w")  # noqa: SIM115 - closed in destroy_node
        self._file.write(GT_HEADER + "\n")
        self._file.flush()

        # --- Subscriber ---
        self.subscription = self.create_subscription(Odometry, topic_name, self.odom_callback, 10)
        self.get_logger().info(f"Recording odometry from '{topic_name}' to '{self.file_path}'")

    def odom_callback(self, msg):
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._file.write(format_pose_line(timestamp, pose_to_matrix(msg.pose.pose)) + "\n")
        self._file.flush()

    def destroy_node(self):
        self._file.close()
        return super().destroy_node()


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
