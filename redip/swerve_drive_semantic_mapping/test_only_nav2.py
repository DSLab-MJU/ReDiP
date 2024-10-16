#!/usr/bin/env python3

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Pose
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

from sensor_msgs.msg import LaserScan
from rclpy.node import Node
import os
import pandas as pd
import cv2
import math
import time

from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

import numpy as np

from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup


#필요 한것
# 자동 현재 위치 지정
# 1. simple commander api로 일정 시간이 지나면 지정된 위치로 이동(똑같은 테스트에서 시험하기 위함.)
# 2. 


# 2. 시나리오 시작 타이밍
# - 일정 범위 내 동적 장애물이 탐지되고 일정 범위 내 들어 왔을 때 회피 기동 호출 함수가 있어야 함.
# - 호출 시 예측을 위한 입력 시퀀스 내 사진들을 담아뒀다가 전달하는 기능이 필요함.
# - 

class TestOnlyNav2(Node):

    def __init__(self):
        super().__init__("test_dynamic_obs_avoidance")
        
        nav2_cb_group = MutuallyExclusiveCallbackGroup()

        topic_cb_group = MutuallyExclusiveCallbackGroup()

        publish_cb_group = MutuallyExclusiveCallbackGroup()

        self.goal_subscription = self.create_subscription(
            PoseStamped,
            '/goal',
            self.goal_callback,
            10,
            callback_group = nav2_cb_group
        )
        self.pose_subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.pose_callback,
            10,
            callback_group = publish_cb_group
        )
        # 라이다 센서 값 받아야 할듯.

        # set publishers
        self.initial_pose_publisher = self.create_publisher(
            PoseWithCovarianceStamped,
            '/initialpose',
            10
        )

        self.laser_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.get_collision_status,
            10,
            callback_group=publish_cb_group
        )

        self.goal_publisher = self.create_publisher(PoseStamped, '/goal', 10)

        # 얘를 목적지 발행하는 로직으로 수정하기
        self.local_goal_marker_publisher = self.create_publisher(Marker, '/local_goal', 10)
        
        self.initial_time = time.time()

        # 처음 로봇이 주어질 때 초기 위치를 발행하는 타이머 함수인데 일단 삭제하진 않았음.
        self.initial_pose_timer = self.create_timer(3.0, 
                                                    self.publish_initial_pose)
 
        self.local_goal_marker_publish_timer = self.create_timer(0.2, self.goal_marking, callback_group = publish_cb_group)
        # self.get_collision_status_timer = self.create_timer(0.1, self.get_collision_status, callback_group = topic_cb_group)
        # self.get_path_timer = self.create_timer(0.5, self.get_path_callback, callback_group = publish_cb_group)

        # 15초 후에 목적지를 설정하고 주행을 시작하는 타이머
        self.timer = self.create_timer(seed, self.set_goal_and_start_navigation, callback_group=nav2_cb_group)

        # 3m 이내 장애물 탐지 시 기록해야 할 위치 정보 시퀀스 
        self.episode_data = [] 

        self.initial_robot_pose = None  # 초기 로봇 위치 속성 추가

        # 로봇의 위치
        self.robot_pose = None
        
        # 라이다 센서 값
        self.laser_scan = None
        self.is_driving = False
        
        # 목적지(한 시나리오에서의 주행을 위한 목적지)
        self.goal_pose = None
        
        self.detected_status = False
        self.dynamic_in_area_status = False
        self.detected_dynamic_objects = []

        self.navigator = BasicNavigator()

    def publish_initial_pose(self):
        initial_pose_msg = PoseWithCovarianceStamped()
        initial_pose_msg.header.stamp = self.get_clock().now().to_msg()
        initial_pose_msg.header.frame_id = 'map'
        initial_pose_msg.pose.pose.position.x = 0.0
        initial_pose_msg.pose.pose.position.y = -0.0
        initial_pose_msg.pose.pose.position.z = 0.0
        initial_pose_msg.pose.pose.orientation.x = 0.0
        initial_pose_msg.pose.pose.orientation.y = 0.0
        initial_pose_msg.pose.pose.orientation.z = 0.02357536340486449
        initial_pose_msg.pose.pose.orientation.w = 0.9997220624955361
        initial_pose_msg.pose.covariance = [0.0] * 36
    # def publish_initial_pose(self):
    #     initial_pose_msg = PoseWithCovarianceStamped()
    #     initial_pose_msg.header.stamp = self.get_clock().now().to_msg()
    #     initial_pose_msg.header.frame_id = 'map'
    #     initial_pose_msg.pose.pose.position.x = 0.0
    #     initial_pose_msg.pose.pose.position.y = -0.0
    #     initial_pose_msg.pose.pose.position.z = 0.0
    #     initial_pose_msg.pose.pose.orientation.x = 0.0
    #     initial_pose_msg.pose.pose.orientation.y = 0.0

        # initial_pose_msg.pose.pose.orientation.z = -0.0  # 90 degrees clockwise rotation
        # initial_pose_msg.pose.pose.orientation.w = 0.0   # 90 degrees clockwise rotation
        
        # initial_pose_msg.pose.pose.orientation.z = -0.7071  # 90 degrees clockwise rotation
        # initial_pose_msg.pose.pose.orientation.w = 0.7071   # 90 degrees clockwise rotation
    #     initial_pose_msg.pose.covariance = [0.0] * 36

        self.initial_pose_publisher.publish(initial_pose_msg)
        self.get_logger().info("초기 로봇 위치가 설정되었습니다.")
        self.initial_pose_timer.cancel()
    
    def pose_callback(self, msg):
        # self.get_logger().info(f"Received new robot pose: {msg}")
        self.robot_pose = msg.pose.pose
        global robot_current_pose
        robot_current_pose = self.robot_pose
    
    def set_goal_and_start_navigation(self):
        # 15초 후에 호출될 함수로 목표 지점 설정 및 주행 시작
        self.get_logger().info(f"15초 후에 설정된 목표 지점으로 주행을 시작합니다.")

        # 설정된 목표 지점 (여기서는 예시로 제공된 좌표)
        self.goal_pose = PoseStamped()
        self.goal_pose.header.frame_id = "map"
        self.goal_pose.header.stamp = self.get_clock().now().to_msg()
        
        # for just forward
        # self.goal_pose.pose = Pose(
        #     position=Point(x=-0.17435789108276367, y=10.21552562713623, z=0.0),
        #     orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        # )

        # for u shape test
        # self.goal_pose.pose = Pose(
        #     position=Point(x=15.624313354492188, y=-0.3219001293182373, z=0.0),
        #     orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        # )

        # for crowd
        self.goal_pose.pose = Pose(
            position=Point(x= -20.2978439331054 , y= 9.513629913330078, z=0.0),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        )
        global navigation_start_status, start_time
        navigation_start_status = True  
        self.is_driving = True
        self.navigator.goToPose(self.goal_pose)

        # 타이머를 멈추기 위해 타이머 객체를 파괴
        self.timer.cancel()

        start_time = time.time()
        
        while not self.navigator.isTaskComplete():

            if self.detected_status:
                self.get_logger().info(f"충돌 발생. 주행에 실패했습니다.")
                self.navigator.cancelTask()
                break

            distance_to_goal = self.calc_goal_distance(self.robot_pose, self.goal_pose)
            
            if distance_to_goal <= 0.25:
                self.get_logger().info("목적지에 도달하였습니다.")
                self.is_driving = False
                self.navigator.cancelTask()
                self.get_logger().info(f"is_driving set to False")
                break
            
            time.sleep(0.1)

        end_time = time.time()
        drive_time = end_time - start_time
        navigation_start_status = False
        self.get_logger().info(f'주행 시간 : {drive_time:.2f} 초')


    def goal_callback(self, msg):
        self.get_logger().info(f"테스트 목적지로 주행을 시작합니다.")

        
        # self.goal_pose = msg
        if not isinstance(msg, PoseStamped):
            goal_pose_stamped = PoseStamped()
            goal_pose_stamped.header.frame_id = "map"
            goal_pose_stamped.header.stamp = self.get_clock().now().to_msg()
            goal_pose_stamped.pose = msg.pose
            self.goal_pose = goal_pose_stamped
        else:
            self.goal_pose = msg

        self.is_driving = True
        self.get_logger().info(f"목적지 정보 : {self.goal_pose}")
        self.navigator.goToPose(self.goal_pose)
        
        # 주행 시작 시간 기록
        start_time = time.time()
        
        while not self.navigator.isTaskComplete():

            if self.detected_status:
                self.get_logger().info(f"충돌 발생. 주행에 실패했습니다.")
                self.navigator.cancelTask()
            
            distance_to_goal = self.calc_goal_distance(self.robot_pose, self.goal_pose)
            
            if distance_to_goal <= 0.25:
                self.get_logger().info("목적지에 도달하였습니다.")
                self.is_driving = False
                self.navigator.cancelTask()
                self.get_logger().info(f"is_driving set to False")
            
            time.sleep(0.1)

        # 주행 종료 시간 기록 및 경과 시간 계산
        end_time = time.time()
        drive_time = end_time - start_time

        self.get_logger().info(f'주행 시간 : {drive_time}')

    def calc_goal_distance(self,robot_pose, goal_pose):
        distance_to_goal = np.sqrt((robot_pose.position.x - goal_pose.pose.position.x) ** 2 +
                                    (robot_pose.position.y - goal_pose.pose.position.y) ** 2)
        return distance_to_goal
    
    def goal_marking(self):
        if not self.is_driving:
            # self.get_logger().info('로컬 목적 지점 지정 안됨.')
            return
        
        # 위치 각도 transformed 수정 안함 [ ]
        # 원본 수정 [ ] 
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "goal_marker"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # 설정할 점의 위치
        marker.pose.position.x = float(self.goal_pose.pose.position.x)
        marker.pose.position.y = float(self.goal_pose.pose.position.y)
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3

        marker.color.a = 1.0
        marker.color.r = 0.4
        marker.color.g = 1.0
        marker.color.b = 1.0

        self.local_goal_marker_publisher.publish(marker)
        # self.get_logger().info('로컬 목적 지점 발행됨')

    def get_collision_status(self, msg: LaserScan):
        """
        라이다 센서 데이터를 받아 일정 거리 이내에 장애물이 있는지 확인합니다.
        """
        self.laser_scan = msg
        min_distance = 0.25  # 이 거리 이내에 장애물이 있을 경우 충돌 상태로 간주

        # 라이다 센서의 최소 거리 값을 확인
        if min(msg.ranges) <= min_distance and self.is_driving:
            self.detected_status = True
        
class RecordRobotPosition(Node):
    def __init__(self):
        super().__init__('record_robot_position')
        self.previous_x = None
        self.previous_y = None
        self.total_distance = 0.0
        self.record_robot_distance_timer = self.create_timer(0.1, self.record_robot_distance)
        # 위치 기록 리스트에 저장하는 로직 필요
        self.record_position = []
        self.previous_navigation_status = False
        self.root_dir = f'{save_log_dir}{str(int(seed))}/'
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
            self.get_logger().info(f'로그 디렉토리가 생성되었습니다: {self.root_dir}')

    def record_robot_distance(self):
        if navigation_start_status:
            #이동거리 기록 로직
            current_x = robot_current_pose.position.x
            current_y = robot_current_pose.position.y
            if self.previous_x is not None and self.previous_y is not None:
                distance_increment = math.sqrt(
                    (current_x - self.previous_x) ** 2 + (current_y - self.previous_y) ** 2 
                )
                self.total_distance += distance_increment
        
            self.previous_x = current_x
            self.previous_y = current_y

            image_x = int((current_x - origin_x) / resolution)
            image_y = int((current_y - origin_y) / resolution)
            
            self.record_position.append([self.previous_x, self.previous_y, image_x, image_y])
        else:
            # 이전에 참이였지만 현재 거짓인 경우 주행이 종료되었다고 간주.
            if self.previous_navigation_status:
                self.get_logger().info("주행이 종료되었습니다. 기록을 시작합니다.")
                self.save_navigation_logs()
                
        self.previous_navigation_status = navigation_start_status
    # 위치 맵에 기록하는 로직 필요 #얘는 종료 요청시 발행됨.
    def record_robot_position_on_map(self):
        # original_map_in_trajectory = map_image
        # cost_map_in_trajectory = cost_map_image

        # 좌우 반전된 맵 생성
        original_map_in_trajectory = cv2.flip(map_image, 0)  # 좌우 반전 (flipCode=1)
        cost_map_in_trajectory = cv2.flip(cost_map_image, 0)  # 좌우 반전 (flipCode=1)

        # map_in_trajectory 이미지에 로봇 위치를 빨간색 점으로 그리는 로직
        for position in self.record_position:
            image_x = int((position[0] - origin_x) / resolution)
            image_y = int((position[1] - origin_y) / resolution)
            image_y = height - image_y  # y축 반전

            # 이미지 범위를 벗어나는 좌표 처리 (optional)
            image_x = max(0, min(image_x, width - 1))
            image_y = max(0, min(image_y, height - 1))
            
            # 빨간색 점 그리기
            cv2.circle(cost_map_in_trajectory, (image_x, image_y), 2, (0, 0, 255), -1)  # 빨간색 점 (2 픽셀 크기)
            cv2.circle(original_map_in_trajectory, (image_x, image_y), 2, (0, 0, 255), -1)  # 빨간색 점 (2 픽셀 크기)

        # 최종 이미지를 파일로 저장
        cv2.imwrite(f'{self.root_dir}robot_trajectory_map.png', cost_map_in_trajectory)
        cv2.imwrite(f'{self.root_dir}robot_trajectory_original_map.png', original_map_in_trajectory)
        
        self.get_logger().info('로봇의 궤적이 robot_trajectory_map.png 파일로 저장되었습니다.')        

    # 전체 이동거리 걸린 시간 누적된 위치 정보 맵 저장하는 로직 필요 + 이동 거리 위치 정보 리스트도 csv형식으로 저장하자.
    def save_navigation_logs(self):
        end_time = time.time()
        duration = end_time - start_time
        self.get_logger().info(f'목적지 도달까지 총 누적 시간 {duration} 초.')
        self.get_logger().info(f'총 이동한 거리 {self.total_distance}')

        # 위치 정보 기록한 맵 저장.
        # self.record_robot_position_on_map()

        # 위치 정보 CSV 파일로 저장
        position_df = pd.DataFrame(self.record_position, columns=['x', 'y', 'image_x','image_y'])

        position_df.to_csv(f'{self.root_dir}robot_positions.csv', index=False)
        self.get_logger().info('위치 정보가 robot_positions.csv 파일로 저장되었습니다.')

        # 이동 거리 및 시간 정보 CSV 파일로 저장
        log_data = {'duration': [duration], 'total_distance': [self.total_distance]}
        log_df = pd.DataFrame(log_data)
        log_df.to_csv(f'{self.root_dir}navigation_logs.csv', index=False)
        self.get_logger().info('이동 거리 및 시간 정보가 navigation_logs.csv 파일로 저장되었습니다.')


def main():
    # rclpy.init(args=sys.argv)
    # test_dynamic_obs_avoidance = TestDynamicObsAvoidance()
    # rclpy.spin(test_dynamic_obs_avoidance)

    rclpy.init()
    test_only_nav2 = TestOnlyNav2()
    record_robot_position  = RecordRobotPosition()
    executor = MultiThreadedExecutor()
    executor.add_node(test_only_nav2)
    executor.add_node(record_robot_position)
    try:
        test_only_nav2.get_logger().info('Beginning client, shut down with CTRL-C')
        executor.spin()
    except KeyboardInterrupt:
        test_only_nav2.get_logger().info('Keyboard interrupt, shutting down.\n')
    test_only_nav2.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

seed = 34.0 # seed는 엑셀 마지막 부분부터 수행해 주세요 34까지
test_approach = 'dwa_crowd_09302346'
save_log_dir = f'/home/kmg/nav2_swerve/src/swerve_drive_semantic_navigation/navigation_log/{test_approach}/'
navigation_start_status = False
robot_current_pose = None
#이동 거리 측정을 위함.
start_time = None

cost_map_image = None
map_image = None
origin_x = -3.54
origin_y = -2.33
width = -3.54
height = -2.33
resolution = 0.05