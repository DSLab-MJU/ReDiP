import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition
from typing import Final

# ROS Pkg Share
WORLD_PKG: Final[str] = "gazebo_sfm_plugin"
SWERVE_ROBOT_GAZEBO_PKG: Final[str] = "urdf_tutorial"

# File Names
WORLD_LAUNCH_FILE: Final[str] = "spawn_dynamic_obs.launch.py"
SPAWN_LAUNCH_FILE: Final[str] = "swerve_robot_spawn.launch.py"

# Directory Names
LAUNCH_DIR: Final[str] = "launch"

# File Paths
world_launch_path = os.path.join(
    get_package_share_directory(WORLD_PKG), LAUNCH_DIR, WORLD_LAUNCH_FILE
)
spawn_launch_path = os.path.join(
    get_package_share_directory(SWERVE_ROBOT_GAZEBO_PKG), LAUNCH_DIR, SPAWN_LAUNCH_FILE
)

def generate_launch_description():
    start_rviz = LaunchConfiguration('start_rviz')
    use_sim = LaunchConfiguration('use_sim')
    use_composition = LaunchConfiguration('use_composition')
    params_file = LaunchConfiguration('params_file')
    map_yaml_file = LaunchConfiguration('map_yaml_file')
    autostart = LaunchConfiguration('autostart')
    map_yaml_file = LaunchConfiguration(
        'map_yaml_file',
        default=PathJoinSubstitution(
            [
                FindPackageShare('swerve_nav2'),
                'map/U_shape_map',
                'map.yaml'
            ]
        )
    )

    params_file = LaunchConfiguration(
        'params_file',
        default=PathJoinSubstitution(
            [
                FindPackageShare('swerve_nav2'),
                'param',
                'swerve_robot.yaml'
            ]
        )
    )

    nav2_launch_file_dir = PathJoinSubstitution(
        [
            FindPackageShare('nav2_bringup'),
            'launch',
        ]
    )

    rviz_config_file = PathJoinSubstitution(
        [
            FindPackageShare('swerve_nav2'),
            'rviz',
            # 'nav2.rviz'
            'test_diffusion.rviz'
        ]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'start_rviz',
            default_value='true',
            description='Whether to execute rviz2'),

        DeclareLaunchArgument(
            'use_sim',
            default_value='true',
            description='Start robot in Gazebo simulation'),

        DeclareLaunchArgument(
            'map_yaml_file',
            default_value=map_yaml_file,
            description='Full path to map file to load'),

        DeclareLaunchArgument(
            'params_file',
            default_value=params_file,
            description='Full path to the ROS2 parameters file to use for all launched nodes'),
        
        DeclareLaunchArgument(
            'use_composition',
            default_value='True',
            description='Whether to use composed bringup'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([nav2_launch_file_dir, '/bringup_launch.py']),
            launch_arguments={
                'map': map_yaml_file,
                'use_sim_time': use_sim,
                'params_file': params_file,
                # 'use_composition': use_composition,
            }.items(),
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(world_launch_path)
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(spawn_launch_path)
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_file],
            parameters=[{'use_sim_time': use_sim}],
            output='screen',
            condition=IfCondition(start_rviz)),

    ])