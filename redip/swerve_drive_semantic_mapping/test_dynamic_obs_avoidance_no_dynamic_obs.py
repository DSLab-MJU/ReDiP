#!/usr/bin/env python3

import cv2
import numpy as np
import os
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Pose
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import LaserScan
from rclpy.node import Node
from nav_msgs.msg import Path
from sensor_msgs.msg import Image  as im
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist

import time
import tf_transformations
import pandas as pd

import math
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from nav_msgs.msg import Path
import copy

from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from typing import Tuple, Sequence, Dict, Union, Optional, Callable

import math
import torch
import torch.nn as nn
import torchvision
from diffusers import DDIMScheduler

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
# from torch2trt import TRTModule

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)

class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            cond_dim, #256 + 514*obs_horizon
            kernel_size=3,
            n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:,0,...]
        bias = embed[:,1,...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out

class ConditionalUnet1D(nn.Module):
    def __init__(self,
        input_dim,
        global_cond_dim, #global_cond_dim :  obs_dim(514=vision_feature_dim(512)+agentpos(2)) * obs_horizon
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=5,
        n_groups=8
        ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim #256
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim #256 + 514*obs_horizon

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            global_cond=None):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim) / shape : (B, obs_dim(514=vision_feature_dim(512)+agentpos(2))*obs_horizaon)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1,-2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            #print('x.dtype, global_feature.dtype:',x.dtype, global_feature.dtype)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1,-2)
        # (B,T,C)
        return x

def get_resnet(name:str, weights=None, **kwargs) -> nn.Module:
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", None
    """
    # Use standard ResNet implementation from torchvision
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)

    # remove the final fully connected layer
    # for resnet18, the output dim should be 512
    resnet.fc = torch.nn.Identity()
    return resnet

def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module

#필요 한것
# 1. simple commander api  
# - global planner(근데 이건 그냥 global_planner node 쓰면 될듯?)
# - local planner(이건 새로 추가해야 할듯)
# - custom plan을 local planner가 따라가게 하는 기능

# 2. 시나리오 시작 타이밍
# - 일정 범위 내 동적 장애물이 탐지되고 일정 범위 내 들어 왔을 때 회피 기동 호출 함수가 있어야 함.
# - 호출 시 예측을 위한 입력 시퀀스 내 사진들을 담아뒀다가 전달하는 기능이 필요함.
# - 

class TestDynamicObsAvoidance(Node):

    def __init__(self):
        super().__init__("test_dynamic_obs_avoidance")
        
        nav2_cb_group = MutuallyExclusiveCallbackGroup()

        topic_cb_group = MutuallyExclusiveCallbackGroup()

        publish_cb_group = MutuallyExclusiveCallbackGroup()


        # Subscribers
        self.detected_obs_publisher = self.create_subscription(
            MarkerArray,
            '/detected_objects_in_map',
            self.get_dynamic_obs_pose,
            10,
            callback_group = topic_cb_group
        )
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
        self.laser_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10,
            callback_group=publish_cb_group
        )

        # set publishers
        self.initial_pose_publisher = self.create_publisher(
            PoseWithCovarianceStamped,
            '/initialpose',
            10
        )

        self.goal_publisher = self.create_publisher(PoseStamped, '/goal', 10)

        self.path_marker_publisher = self.create_publisher(MarkerArray, '/diffusion_plan', 10)

        self.local_goal_marker_publisher = self.create_publisher(Marker, '/local_goal', 10)

        self.semantic_map_publisher = self.create_publisher(im, '/semantic_map', 10)
        
        # 녹화가 시작되는 위치 시각화
        self.crop_center_marker_publisher = self.create_publisher(Marker, '/crop_center_marker', 10)
        self.marker_publisher = self.create_publisher(Marker, '/record_boundary_marker', 10)
        
        self.initial_time = time.time()

        # 처음 로봇이 주어질 때 초기 위치를 발행하는 타이머 함수인데 일단 삭제하진 않았음.
        self.initial_pose_timer = self.create_timer(3.0, 
                                                    self.publish_initial_pose)
        self.inference_dummy_path_for_first_step_timer = self.create_timer(4.0, 
                                                    self.inference_dummy_path_for_first_step)
        
        # self.get_global_costmap_timer = self.create_timer(1.0, self.get_global_costmap)
        # self.publish_record_boundary_marker_timer = self.create_timer(0.2, 
        #                                                               self.publish_boundary_marker,
        #                                                               callback_group = publish_cb_group)

        # 0.2초마다 맵을 생성하는 타이머 콜백 추가
        self.semantic_map_timer = self.create_timer(0.1, 
                                                    self.update_semantic_map,
                                                    callback_group = topic_cb_group)

        # Navigator for navigation
        self.costmap_publish_timer = self.create_timer(6.0, self.publish_global_costmap)
        # self.rotate_timer = self.create_timer(10.0, lambda: self.rotate_in_place(100), callback_group=nav2_cb_group)

        self.local_goal_marker_publish_timer = self.create_timer(0.2, self.local_goal_marking, callback_group= publish_cb_group)


        
        # 15초 후에 목적지를 설정하고 주행을 시작하는 타이머
        self.seed = seed
        self.go_to_set_goal_timer = self.create_timer(self.seed, self.set_goal_and_start_navigation, callback_group=nav2_cb_group)

        # self.get_path_timer = self.create_timer(0.5, self.get_path_callback, callback_group = publish_cb_group)

        self.laser_scan = None
        self.detected_collision = False
        self.detected_status = False
        # 맵 메타데이터 가져오기
        self.width = None
        self.height = None
        self.resolution = None
        self.origin_x = None
        self.origin_y = None
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None

        # 크롭 영역 크기 설정 (6미터를 픽셀 단위로 변환)
        self.area = 4
        self.crop_size = None

        # 크롭 영역의 좌표 계산
        self.half_crop_size = None
        
        # 3m 이내 장애물 탐지 시 기록해야 할 위치 정보 시퀀스 
        self.episode_data = [] 
        self.linear_x =  None
        self.linear_y =  None
        self.angular_z = None

        self.pred_horizon = None
        self.obs_horizon = None
        self.action_horizon  = None
        self.test_nets = None
        self.action_dim = None
        self.device = None
        self.num_diffusion_iters = None
        self.noise_scheduler = None
        
        self.initial_robot_pose = None  # 초기 로봇 위치 속성 추가
        
        # 탐지되었을때 저장해 둘 cost map 
        self.fixed_cost_map = None
        self.cropped_map_size = None
        self.scale_factor = None
        # 저장할 시멘틱 맵 리스트
        # self.semantic_map_input_dim = 3

        self.semantic_maps = []
        self.semantic_map_save_count = 0
        self.undetectable_count_total = 0
        self.undetected_dynamic_obs_during_generative_plan = 0
        self.semantic_record_status = False
        self.record_first_state = True
        # 로봇의 위치
        self.robot_pose = None
        
        # 목적지(한 시나리오에서의 주행을 위한 목적지)
        self.goal_pose = None
        self.final_goal_pose = None
        # 한 회피 동작 에피소드 내에서 주행을 위한 목적지 
        # (3m 이내 장애물 탐지될 때 그 당시의 global_plan의 잘린지점이 해당 목적지가 됨.)
        self.cropped_goal_pose = None

        # semantic map의 위치 및 생성된 액션 시퀀스의 위치를 복원하기 위해 필요.
        self.original_size = None

        # 탐지 될 때 저장되는 global_plan
        self.initial_path = None
        
        self.init_yaw = None   
        
        self.map_data = None
        
        # 탐지시 발동되는 semantic mapping 로직.
        self.initial_crop_center_x = None
        self.initial_crop_center_y = None

        self.initial_crop_goal = None
        self.transformed_initial_crop_goal = None

        self.detected_status = False
        self.dynamic_in_area_status = False
        self.detected_dynamic_objects = []

        self.custom_plan_in_progress = False
        self.center_x = None
        self.center_y = None
        self.map_count = 0

        self.distance_to_goal = None
        # self.get_semantic_map_timer = self.create_timer(0.2, self.get_semantic_map)

        self.navigator = BasicNavigator()
        self.is_dynamic_obs_in_area_status = False
        
        self.costmap = None
        self.costmap_np = None
        self.is_driving = False
        self.global_path_while_one_episode = None
        self.original_local_goal = None
        self.accumulate_semantic_map = None
        self.during_create_custom_path = False
        self.first_inference = True
        self.nav2_start_time = None

        self.initialize_diffuison_model() # diffusion model 초기화

        self.semantic_map_input_dim = self.action_horizon # 이미지 입력 수는 10개로 지정.
        

    def initialize_diffuison_model(self):
        #load models
        #crowd
        # model_save_path = '/home/kmg/nav2_swerve/src/swerve_drive_semantic_navigation/trained_model_datav13oa+crowd_goalcond_20seq5img_t50_ep100.pth'
        
        #umap
        # model_save_path = '/home/kmg/nav2_swerve/src/swerve_drive_semantic_navigation/models/umap/trained_model_datav13_goalcond_20seq5img_t50_ep100.pth'
        # model_save_path = '/home/kmg/nav2_swerve/src/swerve_drive_semantic_navigation/trained_model_datav13+16_umap_goalcond_20seq5img_t50_ep100.pth'
        
        # umap에서 쓰던 모델
        model_save_path = '/home/kmg/nav2_swerve/src/swerve_drive_semantic_navigation/trained_model_datav13oa_goalcond_20seq5img_t50_ep100.pth'
        
        #just forward crowd
        # model_save_path = '/home/kmg/nav2_swerve/src/swerve_drive_semantic_navigation/trained_model_datav17_umap_oa_goalcond_20seq5img_t50_ep100.pth'
        
        # parameters
        self.num_diffusion_iters = 40  #Default 50

        self.noise_scheduler = DDIMScheduler( #DDIMScheduler , DDPMScheduler
            num_train_timesteps=self.num_diffusion_iters,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule='squaredcos_cap_v2',
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type='epsilon'
        )

        self.pred_horizon = 20 #20
        # self.pred_horizon = 8

        self.obs_horizon = 5 #이미지 시퀀스의 길이 = 연속적인 이미지 n장
        
        # self.obs_horizon = 3

        self.action_horizon = self.obs_horizon

        # self.action_horizon = 3

        vision_encoder = get_resnet('resnet18')

        vision_encoder = replace_bn_with_gn(vision_encoder)

        # ResNet18 has output dim of 512
        vision_feature_dim = 512
        # agent_pos is 2 dimensional
        lowdim_obs_dim = 4
        # observation feature has 514 dims in total per step
        obs_dim = vision_feature_dim + lowdim_obs_dim
        self.action_dim = 2

        # create network object
        # 유넷 정의
        noise_test_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=obs_dim*self.obs_horizon
        )

        # the final arch has 2 parts
        self.test_nets_before_tensorrt = nn.ModuleDict({
            'vision_encoder': vision_encoder,
            'noise_pred_net': noise_test_net
        })

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        
        # Load the saved model state dictionary
        loaded_state_dict = torch.load(model_save_path, map_location="cuda:0")
        # 로드를 한다.
        self.test_nets_before_tensorrt.load_state_dict(loaded_state_dict)

        use_tensor_rt = True

        self.test_nets = self.test_nets_before_tensorrt
        # Transfer the model to the appropriate device if needed
        self.test_nets.to(self.device)

        # self.get_logger().info(f"Model successfully loaded and ready for use.{self.test_nets}")
    
    def rotate_in_place(self, angle_degrees, angular_speed=0.5):
        # 각도를 라디안으로 변환
        angle_radians = np.deg2rad(angle_degrees)

        # 목표 각속도를 계산 (양수는 반시계 방향, 음수는 시계 방향)
        angular_velocity = angular_speed if angle_radians > 0 else -angular_speed

        # 회전에 필요한 시간을 계산 (거리 = 속도 * 시간 => 시간 = 거리 / 속도)
        rotation_time = abs(angle_radians / angular_velocity)

        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        twist_msg.linear.y = 0.0
        twist_msg.linear.z = 0.0
        twist_msg.angular.x = 0.0
        twist_msg.angular.y = 0.0
        twist_msg.angular.z = angular_velocity

        # 로봇에 회전 명령을 주기 위한 퍼블리셔 설정
        cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # 회전 명령 발행
        start_time = self.get_clock().now().seconds_nanoseconds()[0]
        while (self.get_clock().now().seconds_nanoseconds()[0] - start_time) < rotation_time:
            cmd_vel_publisher.publish(twist_msg)
            time.sleep(0.1)

        # 회전이 끝나면 정지 명령 발행
        twist_msg.angular.z = 0.0
        cmd_vel_publisher.publish(twist_msg)
        self.rotate_timer.cancel()
        self.get_logger().info(f"제자리에서 {angle_degrees}도 회전을 완료했습니다.")
    def inference_dummy_path_for_first_step(self):
        # 노드가 실행되고 처음 실행되는 inference 시간이 비이상적으로 큰 관계로 dummy path를 생성
        self.inference_dummy_path_for_first_step_timer.cancel()

        #dummy pose
        robot_poses = []
        for _ in range(self.semantic_map_input_dim):
            robot_poses.append((0,0))
        
        #dummy goal_pose
        goal_poses = []
        for _ in range(self.semantic_map_input_dim):
            goal_poses.append((0,0))
        #dummpy semantic map
        semantic_maps = []  
        for _ in range(self.semantic_map_input_dim):
            semantic_maps.append(np.zeros((96, 96, 3), dtype=np.uint8)
        )
        _ = self.inference_path(robot_poses, semantic_maps, goal_poses)
        self.get_logger().info("dummpy path 생성 완료.")

    def get_path_callback(self):
        if self.goal_pose:
            current_pose = self.robot_pose
            if not isinstance(self.robot_pose, PoseStamped):
                robot_pose_stamped = PoseStamped()
                robot_pose_stamped.header.frame_id = "map"
                robot_pose_stamped.header.stamp = self.get_clock().now().to_msg()
                robot_pose_stamped.pose = self.robot_pose
                current_pose = robot_pose_stamped
                
            
            # self.get_logger().info("경로 생성중")
            self.initial_path = self.navigator.getPath(current_pose, self.goal_pose, use_start=True)  # use_start=True 추가
            if self.initial_path is not None:
                self.get_logger().info("Global path generated.")

    def normalize_data(self, data, max_value):
        ndata  = data / max_value
        ndata = ndata * 2 - 1
        return ndata

    def unnormalize_data(self, ndata, max_value):
        ndata = (ndata + 1) / 2
        data = ndata * max_value
        return data
    
    def inference_path(self, nagent_poses, nimages, ngoal):
        '''
        Input : 10개의 위치, 이미지. 목적지 값(이건 같은 값 10개 일거임 
        Output : 현재 action(=위치)10+ pred_action10 의 20개 action
        '''
        B = 1
        
        # infer action
        with torch.no_grad():
            # get image features
            try:
                nimages = torch.tensor(nimages, dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)
                nagent_poses = torch.tensor(nagent_poses, dtype=torch.float32).to(self.device)
                ngoal = torch.tensor(ngoal, dtype=torch.float32).to(self.device)

                nagent_poses = self.normalize_data(nagent_poses, 96)
                ngoal = self.normalize_data(ngoal, 96)
    
                image_features = self.test_nets['vision_encoder'](nimages)
                # (2,512)
                # concat with low-dim observations

                obs_features = torch.cat([image_features, nagent_poses, ngoal], dim=-1)

                # reshape observation to (B,obs_horizon*obs_dim)
                obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

                # initialize action from Guassian noise
                noisy_action = torch.randn(
                    (B, self.pred_horizon, self.action_dim), device=self.device)
                naction = noisy_action
                # init scheduler
                self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

                for k in self.noise_scheduler.timesteps:
                    # predict noise
                    noise_pred = self.test_nets['noise_pred_net'](
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )
                    # inverse diffusion step (remove noise)
                    naction = self.noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample
                    
                # unnormalize action
                naction = naction.detach().to('cpu').numpy()
                # (B, pred_horizon, action_dim)
                naction = naction[0]
                
                #최종 출력된 path 출력 20사이즈
                #여기서 그냥 이후 15개의 위치 정보만 린턴
                return naction[self.semantic_map_input_dim:]
        
            except RuntimeError as e:
                self.get_logger().error(f"생성 과정 중에 문제가 발생했습니다.")
                return None

    def calc_goal_distance(self,robot_pose, goal_pose):
        distance_to_goal = np.sqrt((robot_pose.position.x - goal_pose.pose.position.x) ** 2 +
                                    (robot_pose.position.y - goal_pose.pose.position.y) ** 2)
        return distance_to_goal
    def calc_local_goal_distance(self, robot_pose, local_goal_pose):
        distance_to_goal = np.sqrt((robot_pose.position.x - local_goal_pose[0]) ** 2 +
                                    (robot_pose.position.y - local_goal_pose[1]) ** 2)
        return distance_to_goal
        
    def publish_global_costmap(self):
        try:
            self.costmap = self.navigator.getGlobalCostmap()
            
            self.width = self.costmap.metadata.size_x
            self.height = self.costmap.metadata.size_y
            self.resolution = self.costmap.metadata.resolution
            self.origin_x = self.costmap.metadata.origin.position.x
            self.origin_y = self.costmap.metadata.origin.position.y

            global width, height, resolution, origin_x, origin_y
    
            width = self.width
            height = self.height
            resolution = self.resolution
            origin_x = self.origin_x
            origin_y = self.origin_y

            self.crop_size = int(self.area / self.resolution)

            # 크롭 영역의 좌표 계산
            self.half_crop_size = self.crop_size // 2

            # costmap 데이터를 이미지로 변환
            self.costmap_np =  self.initialize_map_image(self.costmap)
            # 인플레이션된 영역을 줄이는 로직 추가
            self.costmap_np = self.reduce_inflation(self.costmap_np)

            cv2.imwrite("/home/kmg/nav2_swerve/src/swerve_drive_semantic_navigation/map_data/cost_map.png", self.costmap_np)
            
            global cost_map_image
            cost_map_image = self.costmap_np

            global map_image
            map_image = self.reduce_inflation(self.costmap_np, 40)
            # self.get_logger().info("초기 global costmap이 발행 완료되었습니다.")
            self.costmap_publish_timer.cancel()
            
        except Exception as e:
            self.get_logger().error(f"Global costmap 생성 실패: {e}")
            raise  # 예외를 발생시켜 프로그램이 종료되도록 합니다.
    def reduce_inflation(self, costmap_np, kernel_dim = 20):
        """
        코스트맵 이미지에서 검은색 영역을 침식(Erosion)하는 함수
        """
        # 침식 커널 정의 (3x3 커널을 사용하는 예시)
        kernel = np.ones((kernel_dim, kernel_dim), np.uint8)

        # 검은색 영역 침식 (흰색 영역이 침식되지 않도록 흑백 이미지를 반전시킨 후 침식)
        inverted_map = cv2.bitwise_not(costmap_np)
        eroded_map = cv2.erode(inverted_map, kernel, iterations=1)

        # 이미지를 원래대로 반전시켜 검은색 영역이 줄어든 결과를 얻음
        final_map = cv2.bitwise_not(eroded_map)

        return final_map

    def initialize_map_image(self, costmap):
        height = costmap.metadata.size_y
        width = costmap.metadata.size_x

        # Costmap 데이터를 2차원 배열로 변환
        costmap_data = np.array(costmap.data).reshape(height, width)
        map_image = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                if costmap_data[y, x] == -1:
                    map_image[y, x] = [128, 128, 128]  # 회색 (알 수 없음)
                elif costmap_data[y, x] == 0:
                    map_image[y, x] = [255, 255, 255]  # 흰색 (빈 공간)
                else:
                    map_image[y, x] = [0, 0, 0]  # 검은색 (장애물)
        return map_image

    def publish_initial_pose(self):
        initial_pose_msg = PoseWithCovarianceStamped()
        initial_pose_msg.header.stamp = self.get_clock().now().to_msg()
        initial_pose_msg.header.frame_id = 'map'
        # initial_pose_msg.pose.pose.position.x = 2.25
        initial_pose_msg.pose.pose.position.x = 0.0
        initial_pose_msg.pose.pose.position.y = 0.0
        initial_pose_msg.pose.pose.position.z = 0.0
        initial_pose_msg.pose.pose.orientation.x = 0.0
        initial_pose_msg.pose.pose.orientation.y = 0.0

        # initial_pose_msg.pose.pose.orientation.z = -0.0  # 90 degrees clockwise rotation
        # initial_pose_msg.pose.pose.orientation.w = 0.0   # 90 degrees clockwise rotation
        
        #crowd
        # initial_pose_msg.pose.pose.orientation.z = 0.02357536340486449
        # initial_pose_msg.pose.pose.orientation.w = 0.9997220624955361
        
        initial_pose_msg.pose.pose.orientation.z = -0.7071  # 90 degrees clockwise rotation
        initial_pose_msg.pose.pose.orientation.w = 0.7071   # 90 degrees clockwise rotation
        initial_pose_msg.pose.covariance = [0.0] * 36

        self.initial_pose_publisher.publish(initial_pose_msg)
        self.get_logger().info("초기 로봇 위치가 설정되었습니다.")
        self.initial_pose_timer.cancel()
    
    def pose_callback(self, msg):
        # self.get_logger().info(f"Received new robot pose: {msg}")
        self.robot_pose = msg.pose.pose
        global robot_current_pose
        robot_current_pose = self.robot_pose
    
    def laser_callback(self,msg):
        """
        라이다 센서 데이터를 받아 일정 거리 이내에 장애물이 있는지 확인합니다.
        """
        self.laser_scan = msg
        min_distance = 0.25  # 이 거리 이내에 장애물이 있을 경우 충돌 상태로 간주
        sensor_min_range = min(msg.ranges)
        # 라이다 센서의 최소 거리 값을 확인
        if  sensor_min_range >= 0.00 and  sensor_min_range <= min_distance and self.is_driving:
            self.detected_collision = True
            self.get_logger().info(f"충돌이 탐지되었습니다.")
            self.navigator.cancelTask()
        
    def set_goal_and_start_navigation(self):
        # 15초 후에 호출될 함수로 목표 지점 설정 및 주행 시작
        self.get_logger().info(f"15초 후에 설정된 목표 지점으로 주행을 시작합니다.")
        self.nav2_start_time = time.time()
        self.go_to_set_goal_timer.cancel()

        # 설정된 목표 지점 (여기서는 예시로 제공된 좌표)
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = "map"
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        
        global navigation_start_status 
        navigation_start_status = True  
        global start_time
        start_time = self.nav2_start_time
        # for just forward
        # goal_pose.pose = Pose(
        #     position=Point(x=-0.17435789108276367, y=10.21552562713623, z=0.0),
        #     orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        # )

        # for u shape test
        goal_pose.pose = Pose(
            position=Point(x=15.624313354492188, y=-0.3219001293182373, z=0.0),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        )

        # for crowd
        # goal_pose.pose = Pose(
        #     position=Point(x= -20.2978439331054 , y= 9.513629913330078, z=0.0),
        #     orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        # )

        self.goal_callback(goal_pose)
    
    def goal_callback(self, msg):
        self.get_logger().info(f"Received new goal: {msg}")
        # self.goal_pose = msg
        if not isinstance(msg, PoseStamped):
            goal_pose_stamped = PoseStamped()
            goal_pose_stamped.header.frame_id = "map"
            goal_pose_stamped.header.stamp = self.get_clock().now().to_msg()
            goal_pose_stamped.pose = msg.pose
            self.goal_pose = goal_pose_stamped
        else:
            self.goal_pose = msg

        self.get_path_callback()

        self.is_driving = True
        self.get_logger().info(f"is_driving set to True")

        # 새로운 목표 지점으로 이동 시작
        
        # self.navigator.cancelTask()

        # global plan 얻기위한 1초 딜레이.
        time.sleep(0.10)
        self.navigator.goToPose(self.goal_pose)
        
        # 재귀 함수의 종료 여부를 반환하는 부분
        result = self.navigate_to_goal_recursive()

        # 재귀 종료 시점에서만 전역 변수 업데이트
        global navigation_start_status 

        if result:
            navigation_start_status = False
            self.get_logger().info(f"목적지까지의 주행이 종료되었습니다. navigation_start_status = {navigation_start_status}")
        else :
            navigation_start_status = False
            self.get_logger().info(f"주행도중 충돌하였습니다. ")
         
        return result

    def navigate_to_goal_recursive(self):
        #남은 거리 체크
        self.distance_to_goal = self.calc_goal_distance(self.robot_pose, self.goal_pose)

        if self.distance_to_goal <= 0.45:
            self.get_logger().info("목적지에 도달하였습니다.")
            nav2_end_time = time.time()

            self.get_logger().info(f"최종 경로 주행 시간: {nav2_end_time - self.nav2_start_time} 초")
            self.is_driving = False
            self.navigator.cancelTask()
            return True  # 목표에 도달했음을 반환

        # 동적 장애물 회피 기동 시작 시기에 발동됨.
        if self.check_special_condition():
            self.global_path_while_one_episode = self.initial_path
            # 시나리오 내 에서의 동적 회피 수행.
            df = pd.DataFrame(self.episode_data)
            df.to_csv("/home/kmg/nav2_swerve/src/swerve_drive_semantic_navigation/map_data/episode_data.csv", index=False)
            
            self.trigger_custom_plan()

            if self.detected_collision:
                self.get_logger().info(f"에피소드 도중 충돌이 발생했습니다.")
                self.navigator.cancelTask()
                return False  # 충돌이 발생했음을 반환

            # 동적 회피 수행 후에는 다시 목적지에 가까워질때까지 주행.
            self.episode_data = [] 
            self.semantic_maps = []
            self.semantic_map_save_count = 0
            self.first_inference = True

            # self.need2inference_during_drive = False
            if self.calc_goal_distance(self.robot_pose, self.goal_pose) > 0.45:
                self.get_logger().info(f"에피소드 종료 다시 dwa로 주행합니다.")
                return self.goal_callback(self.goal_pose)
            else:
                self.get_logger().info(f"목적지 부근 도착으로 주행을 종료합니다. 남은 거리 : {self.distance_to_goal}")
                self.get_logger().info(f" 현위치 : {self.robot_pose.position.x}, {self.robot_pose.position.y} 목적지 :{self.goal_pose.pose.position.x}, {self.goal_pose.pose.position.y}")
                
                self.is_driving = False
                # global navigation_start_status 
                # navigation_start_status = False
                return True  # 목표지점에 도달했음을 반환

        time.sleep(0.1)
        return self.navigate_to_goal_recursive()  # 재귀 호출

    #처음 trigger 조건 
    def check_special_condition(self):
        # 만약 이미지가 semantic_map_input_dim(5)개가 다 찼다면
        if self.semantic_map_save_count == self.semantic_map_input_dim:
            call_diffusion_policy_status = True
        # 아니라면 (동적 장애물이 없는 경우, 혹은 기록하다가 동적 장애물이 없어 다시 abort한 경우)
        else :
            call_diffusion_policy_status = False

        return call_diffusion_policy_status

    def trigger_custom_plan(self):
        # self.get_logger().info("Custom plan에 따른 회피 기동 실행합니다.")
        self.custom_plan_in_progress = True
        false_inference_status = False
        # 기존 네비게이션 작업을 취소
        if self.first_inference:
            self.navigator.cancelTask()
            self.first_inference = False
        # Custom 경로 생성
        # inference time 측정 ->아마 inference 완료 전에 경로 주행이 끝나 있을듯.
        start_time = time.time()  # 시작 시간 측정
        self.during_create_custom_path = True
        custom_path, only_poses, generate_status = self.create_custom_path()
        
        self.during_create_custom_path = False
        end_time = time.time()  # 종료 시간 측정
        
        if not self.first_inference:
            # 기존 네비게이션 작업을 취소
            self.navigator.cancelTask()
        
        if generate_status:
            # 만약 경로가 특정 오류로 인해 생성되지 않은 경우             
            self.navigator.followPath(custom_path)
            self.get_logger().info(f"create_custom_path 실행 시간: {end_time - start_time} 초")
        else :
            self.get_logger().info(f"생성 실패로 다시 생성하기 위해 데이터를 축적합니다.")
            false_inference_status = True
        
        # closed_status, min_dist = self.check_path_closed_to_goal(only_poses)
        closed_status = False
        self.is_dynamic_obs_in_area_status = False
        self.episode_data = [] 
        self.semantic_maps = []
        self.semantic_map_save_count = 0
        self.accumulate_semantic_map = self.fixed_cost_map.copy()
        
        #custom plan 호출되어 있는 상태에서 10개의 semantic map이 생성될때까지
        # 에피소드가 5~10 에 대한 내용이 담길 경우 LOOP 탈출
        while self.custom_plan_in_progress and len(self.semantic_maps) < self.semantic_map_input_dim:
            # 충돌 확인
            if false_inference_status:
                self.get_logger().info(f"충돌 상태 : {self.detected_collision} 추가 횟수 카운트 : {self.semantic_map_save_count}, 시멘틱 이미지 수 : {len(self.semantic_maps)}, 로봇 위치 수:{self.episode_data}")

            # 한번이라도 탐지된 경우
            if self.dynamic_in_area_status:
                self.is_dynamic_obs_in_area_status = True

            if self.detected_collision:
                self.get_logger().info(f"충돌 발생. 주행에 실패했습니다.")
                closed_status = True
                self.navigator.cancelTask()
                break
            # count가 5가 될 때 이미지를 담아서 10이 될때 inference 시켜야 함.

            # -> 근데 솔직히 말해서 5 ~ 10 에서만 진행이 되게 하면 되는거 아닌가? 그러니까 5-10 상황만 저장을 하면 된다는 의미.
            if self.semantic_map_save_count >=0:
                # semantic 정보 저장.
                accumlate_image, semantic_map, robot_position_image = self.make_semantic_map(self.fixed_cost_map, 
                                                                    self.detected_dynamic_objects, 
                                                                    self.initial_path,
                                                                    self.laser_scan)
                # self.publish_semantic_map(accumlate_image) # 시각화 확인을 위한
                self.semantic_maps.append(semantic_map) # 입력값

                self.episode_data.append({
                    'x': robot_position_image[0],
                    'y' : robot_position_image[1]
                }) 
            
            self.semantic_map_save_count += 1
            
            # self.get_logger().info(f"Custom Plan에 따라 회피 기동 중입니다.")

            distance_to_goal = self.calc_local_goal_distance(self.robot_pose, self.original_local_goal)           
            # self.get_logger().info(f"남은 거리: {distance_to_goal}")

            # 만약 목적지에 일정 거리 내 도달하지 않았다면 계속해서 경로를 생성.
            # self.need2inference_during_drive = True
            
            if distance_to_goal <= 1.5:
                self.get_logger().info("로컬 목적지 부근에 도달하였습니다.")
                self.navigator.cancelTask()
                closed_status = True
                # self.need2inference_during_drive = False
                self.get_logger().info(f"is_driving set to False")
                break
            
            time.sleep(0.1)

        if not self.is_dynamic_obs_in_area_status:
            # 에피소드 중 동적 객체 탐지 안되었으면 카운트 1 증가
            self.undetected_dynamic_obs_during_generative_plan += 1
        else: 
            # 에피소드 중 동적 객체 탐지 안되었던 카운트 초기화
            self.undetected_dynamic_obs_during_generative_plan = 0


        self.get_logger().info(f"동적 객체 미탐지 누적횟수 : {self.undetected_dynamic_obs_during_generative_plan}")
        if self.undetected_dynamic_obs_during_generative_plan >=2 :
            # 에피소드 종료
            self.get_logger().info(f"에피소드 내에서 동적 객체가 2번의 경로 생성 동안 탐지되지 않아 diffusion를 통한 경로 생성을 종료합니다.")
            #탐지안되었던 카운트 초기화
            self.undetected_dynamic_obs_during_generative_plan = 0
            closed_status = True

        if closed_status: 
            self.custom_plan_in_progress = False
        else : 
            self.trigger_custom_plan()

        return 
        
    def check_path_closed_to_goal(self, only_poses):
        closed_goal_status = False
        first_dist_calc = True
        min_dist = 0
        for point in reversed(only_poses):

            # self.get_logger().info(f"포인트의 맵상 위치 : {point[0]}, {point[1]} ")

            # self.get_logger().info(f"로컬 목적지의 맵상 위치 : {self.transformed_initial_crop_goal[0]}, {self.transformed_initial_crop_goal[1]} ")
            distance_to_goal = np.sqrt((point[0] - self.original_local_goal[0]) ** 2 +
                                    (point[1] - self.original_local_goal[1]) ** 2)
            if first_dist_calc:
                min_dist = distance_to_goal

            # self.get_logger().info(f"거리 : {distance_to_goal} ")
            if distance_to_goal < min_dist:
                min_dist = distance_to_goal
            if distance_to_goal < 1.0:
                closed_goal_status = True
        return closed_goal_status, min_dist

    def create_custom_path(self):
        custom_path = Path()
        custom_path.header.frame_id = 'map'
        
        # self.get_logger().info(f"로컬 목적지 : {self.transformed_initial_crop_goal} ")
        # 여기서 diffusion policy의 입력 후 경로를 생성.
        crop_goal_for_input = []

        for _ in range(self.semantic_map_input_dim):
            crop_goal_for_input.append((self.transformed_initial_crop_goal[0], self.transformed_initial_crop_goal[1]))
        robot_pose_for_input = np.array([[entry['x'], entry['y']] for entry in self.episode_data])

        self.get_logger().info(f"로봇 포즈 시퀀스 : {robot_pose_for_input} ")
        path = self.inference_path(robot_pose_for_input, self.semantic_maps, crop_goal_for_input)

        if path is None:
            return None, None, False

        self.get_logger().info(f"경로 생성 완료 : {path} ")
        #경로의 각 지점들을 다시 맵좌표로 변환하는 과정이 필요.

        poses, only_poses = self.get_back_original_path(path)
        
        # self.get_logger().info(f"경로 복원 완료 : {poses} ")
        self.publish_path_markers(poses)
    
        custom_path.poses = poses
        
        return custom_path, only_poses, True

    def get_back_original_path(self,path):
        poses = []
        only_poses = []
        # 일단 리턴되는 포인트의 형식이 뭔지 알아야함.
        for point in path:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            unnorm_x = (point[0]*48 + 48)
            unnorm_y = (point[1]*48 + 48)

            cv2.circle(self.accumulate_semantic_map, (int(unnorm_x), int(unnorm_y)), 1, (255, 0, 0), -1)
            # resize 되었던 값 다시 돌리기
            original_x = unnorm_x / self.scale_factor
            original_y = unnorm_y / self.scale_factor

            rotated_x = self.cropped_map_size - 1 - original_x
            rotated_y = original_y

            # 90도 시계 방향 회전 (90도 반시계 방향 회전의 역)
            retransformed_x = self.cropped_map_size - 1 - rotated_y
            retransformed_y = rotated_x

            # 다시 원래 map에서의 위치로 크기 변환
            map_x = retransformed_x * self.resolution + self.origin_x + self.x_min * self.resolution
            map_y = retransformed_y * self.resolution + self.origin_y + self.y_min * self.resolution

            pose.pose.position.x = map_x
            pose.pose.position.y =map_y
            pose.pose.orientation.w = 1.0
            poses.append(pose)
            only_poses.append([map_x, map_y])

        cv2.imwrite(f"/home/kmg/nav2_swerve/src/swerve_drive_semantic_navigation/map_data/path_generated_{self.map_count}.png", self.accumulate_semantic_map)
        self.map_count += 1
        self.publish_semantic_map(self.accumulate_semantic_map)
        return poses, only_poses

    def publish_path_markers(self, poses):
        for i, pose in enumerate(poses):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "path_markers"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose = pose.pose
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            self.marker_publisher.publish(marker)

        # self.get_logger().info(f'Published {len(poses)} path markers.')

    def local_goal_marking(self):
        if not self.original_local_goal:

            # self.get_logger().info('로컬 목적 지점 지정 안됨.')
            return
        
        # 위치 각도 transformed 수정 안함 [ ]
        # 원본 수정 [ ] 
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "local_goal_marker"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # 설정할 점의 위치
        marker.pose.position.x = float(self.original_local_goal[0])
        marker.pose.position.y = float(self.original_local_goal[1])
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

    def get_dynamic_obs_pose(self, msg):
        # 동적 장애물들에 대한 위치 정보를 입력 받는 것.
        # self.get_logger().info('객체 정보 받음')
        self.detected_obs = msg
        dynamic_objects = []
        static_objects = []

        for marker in self.detected_obs.markers:
            position = marker.pose
            if marker.color.r == 1.0 and marker.color.g == 0.0 and marker.color.b == 0.0:
                dynamic_objects.append(position)
            elif marker.color.r == 0.0 and marker.color.g == 1.0 and marker.color.b == 0.0:
                static_objects.append(position)

        if not dynamic_objects and not static_objects:
            self.detected_status = True
            # self.get_logger().info("No objects detected.")
            return
        
        # 만약 탐지된 객체가 있는 경우 
        if self.is_driving :
            # 녹화중 아님, 동적 객체 탐지됨.
            if dynamic_objects: 
                if self.record_first_state:
                    _, _ = self.set_crop_center_position(self.robot_pose)
                    
                # 중심점을 기준으로 1.5 이내 동적 장애물이 있는지 확인.
                self.dynamic_in_area_status, self.detected_dynamic_objects = self.check_dynamic_obs_in_area(self.initial_crop_center_x, self.initial_crop_center_y, dynamic_objects)

            
    def update_semantic_map(self):
        if self.during_create_custom_path :
            # self.get_logger().info("커스텀 경로 생성중이므로 시멘틱 맵을 업데이트 하지 않습니다.")
            return
        if self.costmap == None or not self.is_driving:
            # self.get_logger().info("글로벌 코스트맵이 생성되지 않았거나, 현재 주행중이 아닙니다.")
            # self.get_logger().info(f"글로벌 코스트맵{type(self.costmap)}, 현재 주행중 {self.is_driving}")
            return
        if self.initial_path is None:
            # self.get_logger().info("Global path not generated.")
            return
        if self.semantic_map_save_count == self.semantic_map_input_dim:
            # self.get_logger().info("경로를 생성할 준비가 되었습니다.")
            return
        
        # 운전 중에 customplan에서 따로 맵을 수집하는 상황이 아닌 경우 ->평상시 주행 상황
        if not self.custom_plan_in_progress and self.is_driving:

            # 만약 동적 객체가 범위 내에 있는 경우
            if self.dynamic_in_area_status:
                
                self.semantic_map_save_count += 1
                self.undetectable_count_total = 0

                # self.get_logger().info(f"녹화 누적 횟수 {self.semantic_map_save_count}")
                # 처음 탐지되어 기록해야 할 상황인 경우 
                
                self.semantic_record_status = True

                accumlate_image, semantic_map, robot_position_image = self.make_semantic_map(self.fixed_cost_map, 
                                                                 self.detected_dynamic_objects, 
                                                                 self.initial_path,
                                                                 self.laser_scan)
            
                self.publish_semantic_map(accumlate_image) # 시각화 확인을 위한
                self.semantic_maps.append(semantic_map) # 입력값
                # self.get_logger().info('동적 객체가 있는 시멘틱 맵 완성')
                #로봇의 현 위치 정보를 기록
                self.episode_data.append({
                    'x': robot_position_image[0],
                    'y' : robot_position_image[1]
                })

            #동적 장애물 비탐지시
            else:
                # self.get_logger().info(f"비탐지 상태, 현재 녹화 시작 상태 {self.semantic_record_status}")
                if self.semantic_record_status:
                    self.semantic_map_save_count += 1
                    self.undetectable_count_total += 1

                    # self.get_logger().info(f"녹화 누적 횟수 {self.semantic_map_save_count}")
                    # self.get_logger().info(f"녹화 시작 후 비탐지 누적 횟수 {self.undetectable_count_total}")
                    
                    # 기록중이던 상황 -> 5회 이상인지 체크 
                    if self.undetectable_count_total == 5:
                    # abort
                        self.reset_semantic_map_seq()
                        # self.get_logger().info("5회 이상 장애물 탐지 불가로 녹화 초기화")
                        return
                    
                    accumlate_image, semantic_map, robot_position_image = self.make_semantic_map(self.fixed_cost_map, 
                                                                 self.detected_dynamic_objects, 
                                                                 self.initial_path,
                                                                 self.laser_scan)
             
                    self.publish_semantic_map(accumlate_image) # 시각화 확인을 위한
                    self.semantic_maps.append(semantic_map) # 입력값

                    # self.get_logger().info('동적 객체가 없는 시멘틱 맵 완성')
                    #로봇의 현 위치 정보를 기록
                    self.episode_data.append({
                        'x': robot_position_image[0],
                        'y' : robot_position_image[1]
                    })   

    def publish_semantic_map(self,image):
    # 이미지 메시지로 변환 및 발행
        bridge = CvBridge()
        image_msg = bridge.cv2_to_imgmsg(image, encoding="bgr8")
        image_msg.header.stamp = self.get_clock().now().to_msg()
        image_msg.header.frame_id = "map"
        self.semantic_map_publisher.publish(image_msg)  

    def check_dynamic_obs_in_area(self, center_x, center_y, dynamic_objects):
        dynamic_objects_in_area = []
        status = False
        # self.get_logger().info(f"center_x: {center_x}, center_y: {center_y}")

        for obj in dynamic_objects:
            distance = ((center_x - obj.position.x) ** 2 + (center_y - obj.position.y) ** 2) ** 0.5
            if distance < 2.0: # cropped map 중심점을 기준으로 2.0 이내 장애물이 존재하는 경우.
                dynamic_objects_in_area.append(obj)
                status = True
        return status, dynamic_objects_in_area

    
    def set_crop_center_position(self, current_pose):
        if self.record_first_state:
            # self.get_logger().info("crop center 설정")
            initial_robot_x = current_pose.position.x
            initial_robot_y = current_pose.position.y

            initial_orientation_q = [
                current_pose.orientation.x,
                current_pose.orientation.y,
                current_pose.orientation.z,
                current_pose.orientation.w
            ]
            (_, _, init_yaw) = tf_transformations.euler_from_quaternion(initial_orientation_q)

            center_offset = self.area / 3+ 0.2

            if -np.pi / 4 <= init_yaw < np.pi / 4:  # 앞쪽
                crop_center_x = initial_robot_x + center_offset
                crop_center_y = initial_robot_y
            elif np.pi / 4 <= init_yaw < 3 * np.pi / 4:  # 왼쪽
                crop_center_x = initial_robot_x
                crop_center_y = initial_robot_y + center_offset
            elif -3 * np.pi / 4 <= init_yaw < -np.pi / 4:  # 오른쪽
                crop_center_x = initial_robot_x
                crop_center_y = initial_robot_y - center_offset
            else:  # 뒤쪽
                crop_center_x = initial_robot_x - center_offset
                crop_center_y = initial_robot_y

            self.initial_crop_center_x = crop_center_x
            self.initial_crop_center_y = crop_center_y
            self.publish_crop_center_marker()
            self.publish_boundary_marker()

            # self.get_logger().info(f"크롭 영역 바운더리 발행 완료. crop_x:{self.initial_crop_center_x}, crop_y:{self.initial_crop_center_y}")

            # 설정된 크롭 포즈를 발행
            self.crop_pose = Pose()
            self.crop_pose.position.x = self.initial_crop_center_x
            self.crop_pose.position.y = self.initial_crop_center_y
            self.crop_pose.position.z = 0.0
            self.crop_pose.orientation.w = 1.0

            # 크롭 영역 계산
            crop_x_in_img = int((crop_center_x - self.origin_x) / self.resolution)
            crop_y_in_img = int((crop_center_y - self.origin_y) / self.resolution)

            self.x_min = crop_x_in_img - self.half_crop_size
            self.x_max = crop_x_in_img + self.half_crop_size
            self.y_min = crop_y_in_img - self.half_crop_size
            self.y_max = crop_y_in_img + self.half_crop_size

            x_min_map = max(0, self.x_min)
            x_max_map = min(self.width, self.x_max)
            y_min_map = max(0, self.y_min)
            y_max_map = min(self.height, self.y_max)

            # 3채널 인덱싱
            map_image = self.costmap_np[y_min_map:y_max_map, x_min_map:x_max_map, :]

            # costmap 크롭
            self.fixed_cost_map = map_image
            self.accumulate_semantic_map = map_image.copy()
            self.cropped_map_size = self.fixed_cost_map.shape[0]

            # 데이터 타입을 uint8로 변환
            self.publish_crop_semantic_map(self.fixed_cost_map)
            
            cv2.imwrite("/home/kmg/nav2_swerve/src/swerve_drive_semantic_navigation/map_data/initial_cropped_map.png", self.fixed_cost_map)

        return self.initial_crop_center_x, self.initial_crop_center_y
    
    def publish_crop_semantic_map(self, cropped_map):
        cropped_publish_map = cropped_map.astype(np.uint8)
        bridge = CvBridge()
        image_msg = bridge.cv2_to_imgmsg(cropped_publish_map, encoding="bgr8")
        image_msg.header.stamp = self.get_clock().now().to_msg()
        image_msg.header.frame_id = "map"
        self.semantic_map_publisher.publish(image_msg)
        # self.get_logger().info("crop semantic 맵 발행")

    def publish_crop_center_marker(self):
        # Marker 생성 및 퍼블리시
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "crop_center"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = self.initial_crop_center_x
        marker.pose.position.y = self.initial_crop_center_y
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        
        marker.color.g = 0.0
        if self.semantic_record_status:
            marker.color.r = 0.0
            marker.color.b = 1.0
        else:
            marker.color.r = 1.0
            marker.color.b = 0.0

        self.crop_center_marker_publisher.publish(marker)
        # self.get_logger().info("crop center 마커 발행")

    def reset_semantic_map_seq(self):
        self.episode_data = [] 
        self.semantic_maps = []
        self.semantic_record_status = False
        self.accumulate_semantic_map = None
        self.undetectable_count_total = 0
        self.semantic_map_save_count = 0
        self.record_first_state = True

    
    def make_semantic_map(self, cropped_map, dynamic_objects, initial_path, laser_scan):
        # 현재 로봇 위치 계산
        current_robot_x = int((self.robot_pose.position.x - self.origin_x) / self.resolution)
        current_robot_y = int((self.robot_pose.position.y - self.origin_y) / self.resolution)
    
        # 현재 로봇 위치를 이미지 좌표로 변환
        current_robot_x_on_image = current_robot_x - self.x_min
        current_robot_y_on_image = current_robot_y - self.y_min
        orientation_q = [self.robot_pose.orientation.x, self.robot_pose.orientation.y, self.robot_pose.orientation.z, self.robot_pose.orientation.w]
        (_, _, yaw) = tf_transformations.euler_from_quaternion(orientation_q)
        
        semantic_cropped_map = cropped_map.copy()

        # 레이저 스캔 데이터 처리
        angle_min = laser_scan.angle_min
        angle_increment = laser_scan.angle_increment
        ranges = laser_scan.ranges
        # 레이저 스캔 데이터를 포인트로 변환
        for i, r in enumerate(ranges):
            if r < laser_scan.range_max:
                angle = angle_min + i * angle_increment + yaw  
                scan_x = int(current_robot_x_on_image + (r * np.cos(angle)) / self.resolution)
                scan_y = int(current_robot_y_on_image + (r * np.sin(angle)) / self.resolution)

                cv2.circle(self.accumulate_semantic_map, (scan_x,scan_y), 1, (0,0,0), -1)
                cv2.circle(semantic_cropped_map, (scan_x,scan_y), 1, (0,0,0), -1)

        
        # 로봇의 현재 위치 그리기
        if 0 <= current_robot_x_on_image < cropped_map.shape[1] and 0 <= current_robot_y_on_image < cropped_map.shape[0]:
            # cv2.circle(self.accumulate_semantic_map, (current_robot_x_on_image, current_robot_y_on_image), int(0.3 / self.resolution), (0, 0, 255), -1)
            # cv2.circle(semantic_cropped_map, (current_robot_x_on_image, current_robot_y_on_image), int(0.3 / self.resolution), (0, 0, 255), -1)
            cv2.circle(self.accumulate_semantic_map, (current_robot_x_on_image, current_robot_y_on_image), 1, (0, 0, 255), -1)
            cv2.circle(semantic_cropped_map, (current_robot_x_on_image, current_robot_y_on_image), 1, (0, 0, 255), -1)

        # semantic map 내의 목적지 그리기
        if self.record_first_state:
            self.record_first_state = False
            # self.get_logger().info(f"local goal 설정")
            #여기서 crop goal 입력해 주고 
            goal_within_crop = False
            local_goal = None

            if self.goal_pose:
                # self.get_logger().info(f"goal 위치 resize 설정")
                goal_x = int((self.goal_pose.pose.position.x - self.origin_x) / self.resolution)
                goal_y = int((self.goal_pose.pose.position.y - self.origin_y) / self.resolution)
                goal_x_on_image = goal_x - self.x_min
                goal_y_on_image = goal_y - self.y_min

                if 0 <= goal_x_on_image < cropped_map.shape[1] and 0 <= goal_y_on_image < cropped_map.shape[0]:
                    cv2.circle(self.accumulate_semantic_map, (goal_x_on_image, goal_y_on_image), 1, (0, 255, 0), -1)
                    cv2.circle(semantic_cropped_map, (goal_x_on_image, goal_y_on_image), 1, (0, 255, 0), -1)
                    goal_within_crop = True

                    #여기서 crop goal 설정하는 로직 빠진듯.(original crop goal)

                    self.initial_crop_goal = (goal_x_on_image, goal_y_on_image)
                    transnformed_x, transnformed_y = self.transform_coordinates(goal_x_on_image, goal_y_on_image, self.cropped_map_size)
                    self.original_local_goal = (self.goal_pose.pose.position.x, self.goal_pose.pose.position.y)
                    local_goal = (transnformed_x, transnformed_y)

            if not goal_within_crop:

                # self.get_logger().info(f"경로가 잘려 있어 local goal 탐색")
                path_intersects = []
                cropped_path = []
            
                for pose in initial_path.poses:
                    # 경로의 포인트를 맵 좌표에서 이미지 좌표로 변환
                    x = int((pose.pose.position.x - self.origin_x) / self.resolution) - self.x_min
                    y = int((pose.pose.position.y - self.origin_y) / self.resolution) - self.y_min
                    cropped_path.append((x, y))

                for i in range(len(cropped_path) - 1):
                    x1, y1 = cropped_path[i]
                    x2, y2 = cropped_path[i + 1]       
                    if (0 <= x1 < cropped_map.shape[1] and 0 <= y1 < cropped_map.shape[0]) and \
                        not (0 <= x2 < cropped_map.shape[1] and 0 <= y2 < cropped_map.shape[0]):
                            # 경로의 끝점이 이미지 경계를 벗어날 경우 교차점을 저장
                            intersect = (x1, y1)
                            path_intersects.append(intersect)
                            # self.get_logger().info(f"{path_intersects} 이 탐색됨")

                # 경로가 크롭된 맵 경계에 닿는 지점 중 경로의 잘린 부분을 표시
                if path_intersects:
                    # self.get_logger().info(f"탐색된 local goal : {path_intersects}")
                    x_intersect, y_intersect = path_intersects[-1]
                    closest_intersect = (x_intersect, y_intersect)
                    self.initial_crop_goal = (x_intersect, y_intersect)
                    
                    self.original_local_goal= (x_intersect * self.resolution + self.origin_x + self.x_min * self.resolution, y_intersect * self.resolution + self.origin_y + self.y_min*self.resolution)
                    
                    transnformed_intersect_x, transnformed_intersect_y = self.transform_coordinates(x_intersect, y_intersect, self.cropped_map_size)
                    local_goal = (transnformed_intersect_x, transnformed_intersect_y)
                    # 이미지의 방향과 수치적인 좌표의 방향이 다른 것이므로 여기에는 transform된걸 그리면 안됨.
                    cv2.circle(self.accumulate_semantic_map, closest_intersect, 1, (0, 255, 0), -1)
                    cv2.circle(semantic_cropped_map, closest_intersect, 1, (0, 255, 0), -1)
                    # self.get_logger().info(f"path_intersects: {path_intersects} 이 탐색됨")

            # self.original_local_goal = local_goal
            # self.get_logger().info(f"local goal  : {local_goal}/")
            if local_goal is None:
                self.get_path_callback()
                return self.make_semantic_map(self.fixed_cost_map, 
                                              self.detected_dynamic_objects,
                                              self.initial_path,
                                              self.laser_scan)
                # resized_transform_goal_x = self.initial_crop_goal[0]
                # resized_transform_goal_y = self.initial_crop_goal[1]
            resized_transform_goal_x, resized_transform_goal_y = self.resize_positions(local_goal[0] , local_goal[1], self.cropped_map_size)
            # local goal은 굳이 10번 내내 리턴할 필요가 없으므로 
            self.transformed_initial_crop_goal = (resized_transform_goal_x, resized_transform_goal_y)      
        else: 
            cv2.circle(self.accumulate_semantic_map, self.initial_crop_goal, 1, (0, 255, 0), -1)
            cv2.circle(semantic_cropped_map, self.initial_crop_goal, 1, (0, 255, 0), -1)
        # 동적 객체 그리기
        if dynamic_objects:
            for obj in dynamic_objects:
                # self.get_logger().info(f'obj {obj}, {type(obj)}')

                # 좌표 변환 (중심점 기준 상대 좌표)
                obj_x = int((obj.position.x - self.origin_x) / self.resolution - self.x_min)
                obj_y = int((obj.position.y - self.origin_y) / self.resolution - self.y_min)
                cv2.circle(self.accumulate_semantic_map, (obj_x, obj_y), int(0.3 / self.resolution), (255, 255, 0), -1)
                # cv2.circle(semantic_cropped_map, (obj_x, obj_y), int(0.3 / self.resolution), (255, 255, 0), -1)
  
        # self.local_goal_marking(self.original_local_goal)
        transform_x , transform_y = self.transform_coordinates(current_robot_x_on_image, current_robot_y_on_image, self.cropped_map_size)
        resized_transform_x, resized_transform_y = self.resize_positions(transform_x, transform_y, self.cropped_map_size)
        robot_position_image = (resized_transform_x, resized_transform_y)

        # semantic 정보 그린거 rotate해주고
        # //self.accumulate_semantic_map = self.save_img_correct_side(self.accumulate_semantic_map)
        semantic_cropped_map = self.save_img_correct_side(semantic_cropped_map)
        # self.get_logger().info(f"시멘틱 맵 완성 count : {self.semantic_map_save_count}, 이미지 길이 : {len(self.semantic_maps)}")
        self.accumulate_semantic_map = semantic_cropped_map.copy()
        cv2.imwrite(f"/home/kmg/nav2_swerve/src/swerve_drive_semantic_navigation/map_data/semantic_map_total_{self.map_count}.png", self.accumulate_semantic_map)
        cv2.imwrite(f"/home/kmg/nav2_swerve/src/swerve_drive_semantic_navigation/map_data/semantic_map_{self.semantic_map_save_count}_ep{self.map_count}.png", semantic_cropped_map)
        
        return self.accumulate_semantic_map, semantic_cropped_map, robot_position_image

    def publish_boundary_marker(self):
        # 녹화시 녹화될 범위를 RVIZ에 표현.
        if self.initial_crop_center_x and self.initial_crop_center_y:
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "boundary"
                marker.id = 0
                marker.type = Marker.LINE_STRIP
                marker.action = Marker.ADD

                # Define the corners of the boundary square
                boundary  = 2.0 # 범위 4미터
                points = [
                    (self.initial_crop_center_x - boundary, self.initial_crop_center_y - boundary, 0.0),
                    (self.initial_crop_center_x - boundary, self.initial_crop_center_y + boundary, 0.0),
                    (self.initial_crop_center_x + boundary, self.initial_crop_center_y + boundary, 0.0),
                    (self.initial_crop_center_x + boundary, self.initial_crop_center_y - boundary, 0.0),
                    (self.initial_crop_center_x - boundary, self.initial_crop_center_y - boundary, 0.0)  # Close the loop
                ]

                for point in points:
                    p = Point()
                    p.x, p.y, p.z = point
                    marker.points.append(p)

                marker.scale.x = 0.1
                marker.color.a = 1.0
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.b = 0.0
                if self.semantic_record_status:
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    
                else :
                    marker.color.r = 1.0
                    marker.color.g = 1.0

                self.marker_publisher.publish(marker)
                # self.get_logger().info('레코드 바운더리 발행')

    def transform_coordinates(self,x, y, image_size):
        # 90도 반시계 방향 회전
        rotated_x = y
        rotated_y = image_size - 1 - x

        # 수직 뒤집기
        flipped_x = image_size -1 - rotated_x
        flipped_y = rotated_y

        return flipped_x, flipped_y
    
    def resize_positions(self, x,y,image_size):
        self.scale_factor = 96 / image_size
        transform_resized_x = int(x * self.scale_factor)
        transform_resized_y = int(y * self.scale_factor)
        return transform_resized_x, transform_resized_y
    
    def save_img_correct_side(self, image):
        rotated_map_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        fliped_map_image = cv2.flip(rotated_map_image, 1)
        # 이미지 리사이즈 --> diffusion 모델 input값이 96
        resized_image = cv2.resize(fliped_map_image, (96, 96))
        return resized_image
    
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
        original_map_in_trajectory = cv2.flip(map_image, 1)  # 좌우 반전 (flipCode=1)
        cost_map_in_trajectory = cv2.flip(cost_map_image, 1)  # 좌우 반전 (flipCode=1)

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
        self.record_robot_position_on_map()

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
    rclpy.init()
    test_dynamic_obs_avoidance = TestDynamicObsAvoidance()
    record_robot_position  = RecordRobotPosition()
    executor = MultiThreadedExecutor()
    executor.add_node(test_dynamic_obs_avoidance)
    executor.add_node(record_robot_position)
    try:
        test_dynamic_obs_avoidance.get_logger().info('Beginning client, shut down with CTRL-C')
        executor.spin()
    except KeyboardInterrupt:
        test_dynamic_obs_avoidance.get_logger().info('Keyboard interrupt, shutting down.\n')
    test_dynamic_obs_avoidance.destroy_node()
    rclpy.shutdown()


seed = 28.0 #31 까지
test_approach = 'redip_crowd'
save_log_dir = f'/home/kmg/nav2_swerve/src/swerve_drive_semantic_navigation/navigation_log/{test_approach}/'
navigation_start_status = False
robot_current_pose = None
#이동 거리 측정을 위함.
start_time = None

cost_map_image = None
map_image = None
origin_x = None
origin_y = None
width = None
height = None
resolution = None

if __name__ == "__main__":

    main()