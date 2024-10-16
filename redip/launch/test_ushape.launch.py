import os
from typing import Final, List, Optional

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

# ROS Pkg Share
SWERVE_NAV2_PKG: Final[str] = "swerve_nav2"  

# File Names
NAV2_LAUNCH_FILE: Final[str] = "bringup_semantic_map_test_u_shape.launch.py"
# NAV2_LAUNCH_FILE: Final[str] = "bringup_semantic_navigation.launch.py"

# Directory Names
LAUNCH_DIR: Final[str] = "launch"

# File Paths
nav2_launch_path = os.path.join(
    get_package_share_directory(SWERVE_NAV2_PKG), LAUNCH_DIR, NAV2_LAUNCH_FILE
)

# Launch Arguments
ARGUMENTS: Optional[List[DeclareLaunchArgument]] = []

def generate_launch_description() -> LaunchDescription:

    nav2_launch_description = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([nav2_launch_path]),
    )

    yolo_segmentation_lidar_projection_node = Node(
        package='swerve_drive_semantic_mapping',
        executable='yolo_segmentation_lidar_projection',
        name='yolo_segmentation_lidar_projection',
    )

    # detect_dynamic_obs_2d_lidar_node = Node(
    #     package='swerve_drive_semantic_mapping',
    #     executable='detect_dynamic_obs_2d_lidar',
    #     name='detect_dynamic_obs_2d_lidar',
    # )

    # for Diffusion Policy
    test_dynamic_obs_avoidance_node = Node(
        package='swerve_drive_semantic_mapping',
        executable='test_dynamic_obs_avoidance',
        name='test_dynamic_obs_avoidance'
    )

    # for Just Use Nav2
    test_only_nav2_node = Node(
        package='swerve_drive_semantic_mapping',
        executable='test_only_nav2',
        name='test_only_nav2'
    )


    ld = LaunchDescription(ARGUMENTS)

    ld.add_action(nav2_launch_description)
    # ld.add_action(detect_dynamic_obs_2d_lidar_node)
    ld.add_action(yolo_segmentation_lidar_projection_node)
    ld.add_action(test_dynamic_obs_avoidance_node)
    # ld.add_action(test_only_nav2_node)
    return ld