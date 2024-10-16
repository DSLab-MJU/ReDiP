cmake_minimum_required(VERSION 3.5)
project(swerve_drive_semantic_mapping)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(gazebo_dev REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Include directories
include_directories(
  include
  ${GAZEBO_INCLUDE_DIRS}
  ${rclcpp_INCLUDE_DIRS}
  ${geometry_msgs_INCLUDE_DIRS}
  ${nav_msgs_INCLUDE_DIRS}
  ${tf2_ros_INCLUDE_DIRS}
)

# Libraries
add_library(${PROJECT_NAME} SHARED src/swerve_drive_plugin.cpp)
ament_target_dependencies(${PROJECT_NAME}
  "rclcpp"
  "pluginlib"
  "geometry_msgs"
  "nav_msgs"
  "tf2_ros"
  "gazebo_dev"
)

# Install
install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

install(DIRECTORY urdf
  DESTINATION share/${PROJECT_NAME})

# Include Python scripts
install(PROGRAMS
  swerve_drive_semantic_mapping/interpolate_dynamic_obs.py
  swerve_drive_semantic_mapping/mapping_dynamic_obs.py
  swerve_drive_semantic_mapping/global_planner.py
  swerve_drive_semantic_mapping/keyboard_state_input.py
  swerve_drive_semantic_mapping/make_train_semantic_data.py
  swerve_drive_semantic_mapping/detect_dynamic_obs_2d_lidar.py
  swerve_drive_semantic_mapping/detect_dynamic_obs_3d_lidar.py
  swerve_drive_semantic_mapping/diffusion.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_dependencies(rclcpp geometry_msgs nav_msgs tf2_ros gazebo_dev pluginlib)
ament_export_libraries(${PROJECT_NAME})

ament_package()

# Suppress dev warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-dev")
