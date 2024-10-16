#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav2_simple_commander.robot_navigator import BasicNavigator
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav2_msgs.msg import Costmap  # Costmap 메시지 타입 임포트

class GlobalPathPlanner(Node):

    def __init__(self):
        super().__init__('global_path_planner')
        self.robot_pose = None
        self.goal_pose = None
        self.navigator = BasicNavigator()
        self.path_generated = False

        # Subscribe to goal topic
        self.goal_subscription = self.create_subscription(
            PoseStamped,
            '/goal',
            self.goal_callback,
            10
        )

        self.pose_subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.pose_callback,
            10  
        )

        # Publisher for the global path and costmap
        self.path_publisher = self.create_publisher(Path, '/global_path', 10)
        self.costmap_publisher = self.create_publisher(Costmap, '/global_costmap', 10)  # Costmap 형식으로 변경

        # Timer to periodically generate global path and costmap
        self.timer = self.create_timer(2.0, self.publish_global_path_and_costmap)

    def goal_callback(self, msg):
        self.goal_pose = msg
        self.path_generated = False  # 새로운 목표 지점이 설정되면 경로 재생성 플래그를 리셋
        self.get_logger().info(f"Received new goal: {self.goal_pose}")

    def pose_callback(self, msg):
        pose_stamped = PoseStamped()
        pose_stamped.header = msg.header
        pose_stamped.pose = msg.pose.pose
        self.robot_pose = pose_stamped

    def publish_global_path_and_costmap(self):
        # Generate and publish the global costmap
        costmap = self.navigator.getGlobalCostmap()

        if costmap is not None:
            # self.get_logger().info("Global costmap retrieved.")
            self.costmap_publisher.publish(costmap)
        # else:
            # self.get_logger().info("Failed to retrieve global costmap.")


        # Generate and publish the global path if goal_pose is set and path has not been generated
        if self.goal_pose is not None and self.robot_pose is not None and not self.path_generated:
            path = self.navigator.getPath(self.robot_pose, self.goal_pose, use_start=True)  # use_start=True 추가
            if path is not None:
                self.get_logger().info("Global path generated.")
                self.path_publisher.publish(path)
                self.path_generated = True  # 경로가 생성되면 플래그를 설정하여 재생성 방지
            else:
                self.get_logger().info("Failed to generate global path.")
        elif self.path_generated:
            self.get_logger().info("Global path already generated. Waiting for new goal.")
        else:
            self.get_logger().info("Goal pose or robot pose is not set yet.")

    def destroy_node(self):
        if self.navigator is not None:
            self.navigator.lifecycleShutdown()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    global_path_planner = GlobalPathPlanner()

    try:
        rclpy.spin(global_path_planner)
    except KeyboardInterrupt:
        pass
    finally:
        global_path_planner.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
