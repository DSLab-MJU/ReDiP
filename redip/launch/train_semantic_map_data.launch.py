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
NAV2_LAUNCH_FILE: Final[str] = "bringup_semantic_map_train_data.launch.py"

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

    detect_dynamic_obs_2d_lidar_node = Node(
        package='swerve_drive_semantic_mapping',
        executable='detect_dynamic_obs_2d_lidar',
        name='detect_dynamic_obs_2d_lidar',
    )
    global_planner_node = Node(
        package='swerve_drive_semantic_mapping',
        executable='global_planner',
        name='global_planner',
    )
    make_train_semantic_data_node = Node(
        package='swerve_drive_semantic_mapping',
        executable='make_train_semantic_data',
        name='make_train_semantic_data'
    )
    keyboard_state_input_node = Node(
        package='swerve_drive_semantic_mapping',
        executable='keyboard_state_input',
        name='keyboard_state_input',
    )


    ld = LaunchDescription(ARGUMENTS)

    ld.add_action(nav2_launch_description)
    ld.add_action(detect_dynamic_obs_2d_lidar_node)
    ld.add_action(global_planner_node )
    ld.add_action(make_train_semantic_data_node )
    ld.add_action(keyboard_state_input_node)
    return ld