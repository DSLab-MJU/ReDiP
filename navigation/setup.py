import os
from glob import glob
from setuptools import setup

package_name = 'swerve_nav2'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # (os.path.join('share', package_name, 'swerve_drive_navigation'), glob('swerve_drive_navig?ation/*.py')),
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
        ],
    },
)
