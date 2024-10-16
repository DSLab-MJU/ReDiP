import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition

def generate_launch_description():
    # Launch configuration variables
    start_rviz = LaunchConfiguration('start_rviz')
    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')
    map_yaml_file = LaunchConfiguration('map_yaml_file')

    # Default paths to the parameters and map yaml files
    map_yaml_file = LaunchConfiguration(
        'map_yaml_file',
        default=PathJoinSubstitution(
            [
                FindPackageShare('swerve_nav2'),
                'map',
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

    rviz_config_file = PathJoinSubstitution(
        [
            FindPackageShare('swerve_nav2'),
            'rviz',
            'nav2.rviz'
        ]
    )

    # Lifecycle managed nodes
    lifecycle_nodes = ['map_server', 'amcl', 'costmap/costmap']

    # Launch description
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'start_rviz',
            default_value='true',
            description='Whether to execute rviz2'),

        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'),

        DeclareLaunchArgument(
            'map_yaml_file',
            default_value=map_yaml_file,
            description='Full path to map file to load'),

        DeclareLaunchArgument(
            'params_file',
            default_value=params_file,
            description='Full path to the ROS2 parameters file to use for all launched nodes'),

        # Map Server Node
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time, 'yaml_filename': map_yaml_file, 'params_file' : params_file}],
            remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')]),

        # AMCL Node
        Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time, 'params_file' : params_file}],
            remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')]),


        # Costmap Node
        Node(
            package='nav2_costmap_2d',
            executable='nav2_costmap_2d_markers',
            name='voxel_visualizer',
            remappings=[('voxel_grid', 'costmap/voxel_grid')]
        ),

        # Run the costmap node
        Node(
            package='nav2_costmap_2d',
            executable='nav2_costmap_2d',
            name='costmap_node',
            parameters=[params_file],
            namespace='costmap'
        ),

        # Lifecycle Manager Node
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_localization',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time},
                        {'autostart': True},
                        {'node_names': lifecycle_nodes}]),


        # RViz2 Node
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_file],
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen',
            condition=IfCondition(start_rviz)),
    ])
