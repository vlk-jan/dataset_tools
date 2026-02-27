#!/usr/bin/env python3

from launch_ros.actions import Node
from launch import LaunchDescription


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="dataset_tools",
                executable="odom_recorder",
                name="odom_recorder",
                respawn=True,
                respawn_delay=1,
                parameters=[
                    {
                        "topic": "/Odometry",
                        "file_path": "./trajectory.txt",
                    }
                ],
            )
        ]
    )
