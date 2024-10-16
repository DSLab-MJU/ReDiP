#!/usr/bin/env python3

import sys
import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, Point, Pose
from visualization_msgs.msg import MarkerArray, Marker
from sensor_msgs.msg import LaserScan, Image
from nav2_msgs.msg import Costmap 

from rclpy.node import Node
import tf_transformations
import hdbscan
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import euclidean
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from sklearn.linear_model import RANSACRegressor
from cv_bridge import CvBridge
import cv2

class KalmanBoxTracker:
    count = 0

    def __init__(self, centroid, R_scale=1.0, Q_scale=0.01, P_scale=10.0):
        self.kf = self.kalman_filter_init(R_scale, Q_scale, P_scale)
        self.kf.x[:2] = centroid[:2].reshape((2, 1))
        self.kf.x[2:6] = 0  # 초기 속도 및 가속도는 0으로 설정
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.age = 0
        self.time_since_update = 0

    def kalman_filter_init(self, R_scale, Q_scale, P_scale):
        kf = KalmanFilter(dim_x=6, dim_z=2)  # 상태 벡터 [x, y, vx, vy, ax, ay]
        kf.F = np.array([[1, 0, 1, 0, 0.5, 0],
                         [0, 1, 0, 1, 0, 0.5],
                         [0, 0, 1, 0, 1, 0],
                         [0, 0, 0, 1, 0, 1],
                         [0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0]])
        kf.R[0:, 0:] *= R_scale  # 관측 노이즈 공분산
        kf.P[4:, 4:] *= 1000.  # 상태 노이즈 공분산
        kf.P *= P_scale
        kf.Q[-1, -1] *= Q_scale  # 프로세스 노이즈 공분산
        kf.Q[4:, 4:] *= Q_scale
        return kf

    def update(self, centroid):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.kf.update(centroid[:2].reshape((2, 1)))

    def predict(self):
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hits = 0
        self.time_since_update += 1
        self.history.append(self.kf.x.copy())
        return self.kf.x.reshape((1, 6))

    def get_state(self):
        return self.kf.x.reshape((1, 6))
    
    def get_velocity(self):
        return np.linalg.norm(self.kf.x[2:4])  # 속도의 크기 계산

class Sort:
    def __init__(self, node):
        self.trackers = []
        self.node = node  # Node 객체를 멤버 변수로 저장

    def update(self, detections):
        trks = np.zeros((len(self.trackers), 6))
        to_del = []
        ret = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict().reshape(-1)
            # self.node.get_logger().info(f"poses: {pos}")
            trks[t, :] = pos[:6]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matches, unmatched_detections, unmatched_trackers = self.associate_detections_to_trackers(detections, trks)

        for t in unmatched_trackers:
            if t < len(self.trackers):
                self.trackers.pop(t)

        for m in matches:
            if m[1] < len(self.trackers) and m[0] < len(detections):
                self.trackers[m[1]].update(np.array(detections[m[0]]))

        for i in unmatched_detections:
            if i < len(detections):
                self.trackers.append(KalmanBoxTracker(np.array(detections[i])))

        for trk in self.trackers:
            d = trk.get_state().reshape(-1)
            if trk.time_since_update < 1:
                ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 7))

    def associate_detections_to_trackers(self, detections, trackers, iou_threshold=0.6):
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 1), dtype=int)

        cost_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                cost_matrix[d, t] = np.linalg.norm(det[:2] - trk[:2])  # det에서 (x, y) 좌표만 사용하도록 수정

        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        matched_indices = np.array(list(zip(row_indices, col_indices)))

        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)

        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        matches = []
        for m in matched_indices:
            if cost_matrix[m[0], m[1]] > iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class YoloSegmentationLidarProjection(Node):
    def __init__(self):
        super().__init__("yolo_segmentation_lidar_projection")
        
        # subscribers
        self.amcl_pose_subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.amcl_pose_callback,
            10)
        self.laser_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )
        self.laser_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )
        self.raw_image_subscription = self.create_subscription(
            Image,
            '/rgb_camera/image_raw',  
            self.image_callback,
            10
        )
        self.mask_image_subscription = self.create_subscription(
            Image,
            '/mask_image', 
            self.mask_image_callback,
            10
        )

        # publishers 
        self.detected_obs_publisher = self.create_publisher(
            MarkerArray,
            '/detected_objects_in_map',
            10
        )
        self.kf_prediction_publisher = self.create_publisher(
            MarkerArray,
            '/kf_predictions',
            10
        )
        self.image_publisher = self.create_publisher(
            Image,
            '/projected_image',  # 투영된 이미지를 발행할 토픽 이름
            10
        )
        self.amcl_pose = None
        self.laser_scan = None
        self.robot_position = None
        self.yaw = None
        self.detected_clusters = None
        self.tracker = Sort(self)  # Node 객체를 전달하여 Sort 객체를 초기화
        self.detect_dynamic_obs_timer = self.create_timer(0.2, self.projection_segmentation_lidar)
        self.current_image = None  # 이미지를 저장할 변수 추가
        
        self.first_initialize_mask = True
        self.current_mask_image = None

        # gazebo camera calibration camera
        self.fx = 1384.647695
        self.cx = 959.066302
        self.fy = 1384.696748
        self.cy = 540.008542
        self.camera_matrix = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])  # 카메라 매트릭스 초기화
        self.rvec, _ = cv2.Rodrigues(np.array([[ 1.21418319], [-1.19095043], [ 1.17409947]]))
        self.tvec = np.array([[-0.13813353], [0.28884489], [0.04034645]])
        
        self.bridge = CvBridge()

    def amcl_pose_callback(self, msg):
        self.amcl_pose = msg
        self.robot_position = self.amcl_pose.pose.pose.position
        robot_orientation = self.amcl_pose.pose.pose.orientation
        orientation_q = [robot_orientation.x, robot_orientation.y, robot_orientation.z, robot_orientation.w]
        (_, _, yaw) = tf_transformations.euler_from_quaternion(orientation_q)
        self.yaw = yaw

    def laser_callback(self, msg):
        self.laser_scan = msg
    
    def image_callback(self, msg):
        self.current_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        if self.first_initialize_mask:
            self.first_initialize_mask =  False
            self.current_mask_image = np.zeros_like(self.current_image)

    def mask_image_callback(self, msg):
        self.current_mask_image = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')

    def projection_segmentation_lidar(self):
        laser_scan = self.laser_scan
        if laser_scan is None or self.robot_position is None:
            return

        angle_min = laser_scan.angle_min
        angle_increment = laser_scan.angle_increment
        ranges = laser_scan.ranges
        points = []
        for i, r in enumerate(ranges):
            if r < laser_scan.range_max and r > 0.05:  # 유효한 범위 필터링
                angle = angle_min + i * angle_increment
                scan_x = r * np.cos(angle)
                scan_y = r * np.sin(angle)
                points.append([scan_x, scan_y, 0])
                # points.append([scan_x + 0.55, scan_y,0])
                # points.append([scan_x - 0.055, scan_y - 0.055,0])
                # points.append([scan_x - 0.055, scan_y + 0.055,0])
                # points.append([scan_x + 0.044, scan_y,0])
                # points.append([scan_x - 0.044, scan_y - 0.044,0])
                # points.append([scan_x - 0.044, scan_y + 0.044,0])
                # points.append([scan_x + 0.05, scan_y,0])
                # points.append([scan_x - 0.05, scan_y - 0.05,0])
                # points.append([scan_x - 0.05, scan_y + 0.05,0])
                # points.append([scan_x + 0.04, scan_y,0])
                # points.append([scan_x - 0.04, scan_y - 0.04,0])
                # points.append([scan_x - 0.04, scan_y + 0.04,0])
                points.append([scan_x + 0.01, scan_y,0])
                points.append([scan_x - 0.01, scan_y - 0.01,0])
                points.append([scan_x - 0.01, scan_y + 0.01,0])

        points = np.array(points, dtype=np.float32)

        # LiDAR 포인트들을 이미지 좌표로 투영
        img_points, _ = cv2.projectPoints(
            points, self.rvec, self.tvec, self.camera_matrix, np.zeros((4, 1)))

        # 흰색 영역만 마스킹
        mask = cv2.inRange(self.current_mask_image, np.array([255, 255, 255]), np.array([255, 255, 255]))

        # 객체 분리 (connected components)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        detected_targets = []

        for label in range(1, num_labels):  # 0은 배경이므로 제외
            component_mask = (labels == label)

            # 각 객체의 포인트만 필터링
            component_points = []
            for i in range(len(img_points)):
                x = int(img_points[i][0][0])
                y = int(img_points[i][0][1])
                # 현재 라벨의 객체에 포함된 포인트만 고려
                if 0 <= x < self.current_mask_image.shape[1] and 0 <= y < self.current_mask_image.shape[0]:
                    if component_mask[y, x]:
                        component_points.append(points[i])
                        # 마스크 내에 포함된 포인트들을 그림
                        cv2.circle(self.current_mask_image, (x, y), 12, (0, 0, 255), -1)


            if not component_points:
                continue

            component_points = np.array(component_points, dtype=np.float32)

            # 가장 가까운 포인트 찾기
            distances = np.linalg.norm(component_points[:, :2], axis=1)
            min_idx = np.argmin(distances)
            closest_point = component_points[min_idx]

            # 뒤로 25cm 떨어진 곳에 중점 정의
            direction_vector = closest_point[:2] / np.linalg.norm(closest_point[:2])
            target_point = closest_point[:2] - 0.35 * direction_vector  # 25cm 뒤로 이동

            # 타겟 포인트를 발행할 리스트에 추가
            target_point_3d = np.array([target_point[0], target_point[1], 0.0])  # z=0으로 설정
            detected_targets.append(target_point_3d)
        
            # if not component_points:
            #     continue

            # component_points = np.array(component_points, dtype=np.float32)

            # # 가장 가까운 상위 3개의 포인트 찾기
            # distances = np.linalg.norm(component_points[:, :2], axis=1)
            # min_indices = np.argsort(distances)[:3]  # 가장 가까운 3개의 포인트 인덱스 선택

            # # 상위 3개의 포인트의 평균 계산
            # closest_points = component_points[min_indices]
            # average_point = np.mean(closest_points, axis=0)

            # # 뒤로 25cm 떨어진 곳에 중점 정의
            # direction_vector = average_point[:2] / np.linalg.norm(average_point[:2])
            # target_point = average_point[:2] - 0.25 * direction_vector  # 25cm 뒤로 이동

            # # 타겟 포인트를 발행할 리스트에 추가
            # target_point_3d = np.array([target_point[0], target_point[1], 0.0])  # z=0으로 설정
            # detected_targets.append(target_point_3d)


        if detected_targets:
            self.publish_detected_objects(detected_targets)
        else :
            far_point = np.array([1000.0, 1000.0, 0.0])  # 멀리 있는 좌표 지정
            self.publish_detected_objects([far_point])
        # detected_targets = np.array(detected_targets)

        # 각 타겟 포인트를 발행
        # if len(detected_targets) > 0:
        #     tracked_objects = self.tracker.update(detected_targets)
        #     self.detected_clusters = {int(obj[6]): obj[:6] for obj in tracked_objects}

        # self.interpolate_dynamic_obs_in_map()
        
        # 결과 이미지를 ROS Image 메시지로 변환하여 발행
        projected_image_msg = self.bridge.cv2_to_imgmsg(self.current_mask_image, encoding="bgr8")
        self.image_publisher.publish(projected_image_msg)

    def publish_detected_objects(self, clusters):
        marker_array = MarkerArray()

        # 기존 마커 삭제 명령 추가
        delete_marker = Marker()
        delete_marker.header.frame_id = "map"
        delete_marker.header.stamp = self.get_clock().now().to_msg()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        # 새로운 마커 추가
        for i, cluster in enumerate(clusters):
            world_point = self.transform_point(self.robot_position, self.yaw, cluster)
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = world_point.x
            marker.pose.position.y = world_point.y
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.6
            marker.scale.y = 0.6
            marker.scale.z = 0.6
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker_array.markers.append(marker)

        # 마커 배열 발행
        self.detected_obs_publisher.publish(marker_array)


    def transform_point(self, origin, yaw, point):
        rotation_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)]
        ])
        point_map = rotation_matrix.dot(np.array([point[0], point[1]])) + np.array([origin.x, origin.y])
        transformed_point = Point()
        transformed_point.x = float(point_map[0])
        transformed_point.y = float(point_map[1])
        transformed_point.z = 0.0  # z 값은 0으로 설정
        return transformed_point

    def interpolate_dynamic_obs_in_map(self):
        if self.amcl_pose is None or self.detected_clusters is None:
            self.get_logger().warning('AMCL pose or detected clusters not available')
            return
        marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.header.frame_id = "map"
        delete_marker.header.stamp = self.get_clock().now().to_msg()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        color_map = plt.get_cmap("tab10")
        
        self.static_obstacles = []  # 초기화
        for cluster_id, state in self.detected_clusters.items():
            try:
                tracker = next((trk for trk in self.tracker.trackers if trk.id == cluster_id), None)
                if tracker is not None:
                    velocity = tracker.get_velocity()
                    color = color_map(cluster_id % 10)

                    centroid_marker = Marker()
                    centroid_marker.header.frame_id = "map"
                    centroid_marker.header.stamp = self.get_clock().now().to_msg()
                    centroid_marker.id = int(cluster_id)
                    centroid_marker.type = Marker.SPHERE
                    centroid_marker.action = Marker.ADD
                    centroid_marker.pose.position.x = state[0]
                    centroid_marker.pose.position.y = state[1]
                    centroid_marker.pose.position.z = 0.0
                    centroid_marker.pose.orientation.w = 1.0
                    centroid_marker.scale.x = 0.6
                    centroid_marker.scale.y = 0.6
                    centroid_marker.scale.z = 0.6

                    if velocity > 0.05:
                        centroid_marker.color.r = 1.0
                        centroid_marker.color.g = 0.0
                        centroid_marker.color.b = 0.0
                    else:
                        centroid_marker.color.r = 0.0
                        centroid_marker.color.g = 1.0
                        centroid_marker.color.b = 0.0
                    centroid_marker.color.a = 1.0
                    marker_array.markers.append(centroid_marker)
            except Exception as e:
                self.get_logger().error(f"Error creating centroid marker: {e}")
        self.detected_obs_publisher.publish(marker_array)

    def transform_point(self, origin, yaw, point):
        rotation_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)]
        ])
        point_map = rotation_matrix.dot(np.array([point[0], point[1]])) + np.array([origin.x, origin.y])
        transformed_point = Point()
        transformed_point.x = float(point_map[0])
        transformed_point.y = float(point_map[1])
        transformed_point.z = 0.0  # z 값은 0으로 설정
        return transformed_point


def main():
    rclpy.init(args=sys.argv)
    yolo_segmentation_lidar_projection = YoloSegmentationLidarProjection()
    rclpy.spin(yolo_segmentation_lidar_projection) 

if __name__ == "__main__":
    main()