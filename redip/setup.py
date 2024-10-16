import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'swerve_drive_semantic_mapping'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kmg',
    maintainer_email='k012123600@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'interpolate_dynamic_obs = swerve_drive_semantic_mapping.interpolate_dynamic_obs:main',
            'mapping_dynamic_obs = swerve_drive_semantic_mapping.mapping_dynamic_obs:main',
            'global_planner = swerve_drive_semantic_mapping.global_planner:main',
            'keyboard_state_input = swerve_drive_semantic_mapping.keyboard_state_input:main',
            'make_train_semantic_data = swerve_drive_semantic_mapping.make_train_semantic_data:main',
            'detect_dynamic_obs_2d_lidar = swerve_drive_semantic_mapping.detect_dynamic_obs_2d_lidar:main',
            'detect_dynamic_obs_3d_lidar = swerve_drive_semantic_mapping.detect_dynamic_obs_3d_lidar:main',
            'test_dynamic_obs_avoidance = swerve_drive_semantic_mapping.test_dynamic_obs_avoidance:main',
            'test_only_nav2 = swerve_drive_semantic_mapping.test_only_nav2:main',
            'test_only_drl_node = swerve_drive_semantic_mapping.test_only_drl_node:main',
            'yolo_segmentation_lidar_projection = swerve_drive_semantic_mapping.yolo_segmentation_lidar_projection:main',
        ],
    },
)
