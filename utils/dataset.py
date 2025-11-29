import csv
import math
import os
import sys

from pathlib import Path
from typing import List

import wandb
import datetime as dt
from tqdm import tqdm
from rich import print

import numpy as np
import open3d as o3d
import torch

import matplotlib.pyplot as plt


from torch.utils.data import Dataset

from dataloaders import dataset_factory
from utils.config import Config
from utils.camera import Camera


class SLAMDataset(Dataset):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.config = config

        self.silence = config.silence
        self.dtype = config.dtype
        self.device = config.device

        self.poses_ts = None # timestamp for each reference pose, also as np.array
        self.gt_poses = None # lidar poses
        self.calib = {"Tr": np.eye(4)} # as T_lidar<-camera
        
        if config.use_dataloader: 
            # Dataset factory -> returns specific dataloader
            self.loader = dataset_factory(
                dataloader=config.data_loader_name, # a specific dataset or data format
                data_path=Path(config.data_path),
                sequence=config.data_loader_seq,
                topic=config.data_loader_seq,
                cam_name=config.data_loader_seq,
            )

            config.end_frame = min(len(self.loader), config.end_frame)
            self.total_pc_count = int((config.end_frame - config.begin_frame) / config.step_frame)
            

            if hasattr(self.loader, 'gt_poses'):
                self.gt_poses = self.loader.gt_poses[config.begin_frame:config.end_frame:config.step_frame]
                self.gt_pose_provided = True
            else:
                self.gt_pose_provided = False
            if hasattr(self.loader, 'calibration'):
                self.calib["Tr"][:3, :4] = self.loader.calibration["Tr"].reshape(3, 4)
            if hasattr(self.loader, "K"):
                self.K = self.loader.K
            if hasattr(self.loader, "T_c_l"):
                self.T_c_l = self.loader.T_c_l
        


        # use pre-allocated numpy array
        self.odom_poses = None
        if config.track_on:
            self.odom_poses = np.broadcast_to(np.eye(4), (self.total_pc_count, 4, 4)).copy()

        self.pgo_poses = None
        if config.pgo_on:
            self.pgo_poses = np.broadcast_to(np.eye(4), (self.total_pc_count, 4, 4)).copy()

        self.travel_dist = np.zeros(self.total_pc_count) 
        self.time_table = []

        self.processed_frame: int = 0
        self.shift_ts: float = 0.0
        self.color_available: bool = False
        self.intensity_available: bool = False
        self.color_scale: float = 255.0

        # count the consecutive stop frame of the robot
        self.stop_count: int = 0
        self.stop_status = False

        self.static_mask = None

        # current frame's data
        self.cur_point_cloud_torch = None
        self.cur_point_ts_torch = None
        self.cur_features_camera_frame = None
        self.cur_pose_guess_torch = None
        self.cur_pose_torch = None

        # source data for registration
        self.cur_source_points = None
        self.cur_source_colors = None
        self.cur_source_normals = None

        # numpy
        self.last_pose_ref = np.eye(4)
        self.last_odom_transformation = np.eye(4)
        self.cur_pose_ref = np.eye(4)

        if self.config.kitti_correction_on:
            self.last_odom_transformation[0, 3] = (self.config.max_range * 1e-2)  # inital guess for booting on x aixs
            self.color_scale = 1.0
    

    def read_frame_with_loader(self, frame_id):

        frame_id_in_folder = self.config.begin_frame + frame_id * self.config.step_frame # has to be global index because of loader
        frame_data = self.loader[frame_id_in_folder]

        points = None

        if isinstance(frame_data, dict):
            dict_keys = list(frame_data.keys())
            if not self.silence:
                print("Available data source:", dict_keys)
            if "points" in dict_keys:
                points = frame_data["points"]
            if "img" in dict_keys:
                cur_img_rgb_np = frame_data["img"]  # 3 channel (rgb only) # uint8 [0, 255]
                cur_img_depth_np = frame_data["depth"]
                self.cur_cam = Camera(frame_id, cur_img_rgb_np, cur_img_depth_np, self.K, self.config.matcher_type, self.device)
            if "imus" in dict_keys: # TO ADD
                self.cur_frame_imus = frame_data["imus"] # imu not used currently. If used add it here and then use throught the code
        else: # no data found, for example None
            return
         
        self.cur_point_cloud_torch = torch.tensor(points, device=self.device, dtype=self.dtype)

    
    def preprocess_frame(self, frame_id):
        self.cur_cam.extract_features(self.config.max_keypoints)
        self.cur_cam.filter_depth(self.config.max_depth)
        self.cur_cam.keypoints_xyz_camera_frame()

    def initial_pose_guess(self, frame_id):
        if frame_id == 0:  # initialize the first frame, no tracking yet
            if self.gt_pose_provided:
                cur_pose_init_guess = self.gt_poses[frame_id]
            else:
                cur_pose_init_guess = np.eye(4)

        elif frame_id > 0:
            if not self.config.track_on and self.gt_pose_provided: # mapping (no tracking + gt_poses provided)
                cur_pose_init_guess = self.gt_poses[frame_id]
            else:
                if self.config.uniform_motion_on: 
                    # apply uniform motion model here
                    cur_pose_init_guess = (self.last_pose_ref @ self.last_odom_transformation)  # T_world<-cur = T_world<-last @ T_last<-cur
                else:  # static initial guess
                    cur_pose_init_guess = self.last_pose_ref
                
                # Case: No tracking + no gt_poses provided -> this will never happened here, bcs it is handled in main

        # pose initial guess tensor
        self.cur_pose_guess_torch = torch.tensor(
            cur_pose_init_guess, dtype=torch.float64, device=self.device
        )
    

    def update_poses(self, frame_id, cur_pose_torch: torch.tensor): 

        # need to be out of the computation graph, used for mapping
        self.cur_pose_torch = cur_pose_torch.detach()
        self.cur_pose_ref = self.cur_pose_torch.cpu().numpy()

        if frame_id == 0:
            self.last_odom_transformation = np.eye(4) # because pose[0] will be GT and last_odom is identity in the beginning (there is no pose before)
        else:
            self.last_odom_transformation = np.linalg.inv(self.last_pose_ref) @ self.cur_pose_ref  # T_last<-cur

        self.last_pose_ref = self.cur_pose_ref  # update for the next frame
        self.last_cam = self.cur_cam


        cur_frame_travel_dist = np.linalg.norm(self.last_odom_transformation[:3, 3])
        self.travel_dist[frame_id] = self.travel_dist[frame_id - 1] + cur_frame_travel_dist

        if self.config.track_on:
            if frame_id == 0:
                cur_odom_pose = self.cur_pose_ref
            else:
                cur_odom_pose = self.odom_poses[frame_id-1] @ self.last_odom_transformation  # T_world<-cur # same as self.cur_pose_ref ?

            self.odom_poses[frame_id] = cur_odom_pose
        
        if self.config.pgo_on:  # initialize the pgo pose
            self.pgo_poses[frame_id] = self.cur_pose_ref

