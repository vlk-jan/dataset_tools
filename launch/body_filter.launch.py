#!/usr/bin/env python3

from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch import LaunchDescription


def generate_launch_description():
    return LaunchDescription(
        [
            ComposableNodeContainer(
                name="lidar_filter_container",
                namespace="",
                package="rclcpp_components",
                executable="component_container",
                composable_node_descriptions=[
                    # Filter 1: Main Body Box
                    ComposableNode(
                        package="pcl_ros",
                        plugin="pcl_ros::CropBox",
                        name="livox_filter_front1",
                        remappings=[
                            ("input", "/livox/lidar_front"),
                            ("output", "/livox/lidar_front_filtered_tmp"),
                        ],
                        parameters=[
                            {
                                "min_x": -0.6,
                                "max_x": 0.3,
                                "min_y": -0.5,
                                "max_y": 0.5,
                                "min_z": -1.0,
                                "max_z": 1.0,
                                "negative": True,
                            }
                        ],
                    ),
                    # Filter 2: Rear/Tail Box
                    ComposableNode(
                        package="pcl_ros",
                        plugin="pcl_ros::CropBox",
                        name="livox_filter_front2",
                        remappings=[
                            ("input", "/livox/lidar_front_filtered_tmp"),
                            (
                                "output",
                                "/livox/lidar_front_filtered",
                            ),
                        ],
                        parameters=[
                            {
                                "min_x": -1.5,
                                "max_x": -0.6,
                                "min_y": -0.3,
                                "max_y": 0.3,
                                "min_z": -1.0,
                                "max_z": 1.0,
                                "negative": True,
                            }
                        ],
                    ),
                ],
                output="screen",
            )
        ]
    )
