import sys
import math

import wandb
from rich import print
from tqdm import tqdm

import matplotlib.cm as cm
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import pypose as pp

from utils.config import Config
from utils.dataset import SLAMDataset
from utils.vhm import VoxelHashMap

from utils.tools import (
    get_time,
    transform_batch_torch,
    transform_torch,
)

class Mapper:
    def __init__(
        self,
        config: Config,
        dataset: SLAMDataset,
        vhm: VoxelHashMap,
        vhm_cpp
    ):
        self.config = config
        self.dataset = dataset
        self.vhm = vhm
        self.vhm_cpp = vhm_cpp

        self.silence = config.silence
        self.device = config.device
        self.dtype = config.dtype
        self.tran_dtype = config.tran_dtype

        self.used_poses = None
    

    def determine_mapping_poses(self, frame_id):
        #frame_id = self.dataset.processed_frame
        if self.config.pgo_on:
            self.used_poses = torch.tensor(
                self.dataset.pgo_poses[:frame_id+1],
                device=self.device,
                dtype=torch.float64,
            )
        elif self.config.track_on:
            self.used_poses = torch.tensor(
                self.dataset.odom_poses[:frame_id+1],
                device=self.device,
                dtype=torch.float64,
            )
        elif self.dataset.gt_pose_provided:  # for pure reconstruction with known pose
            self.used_poses = torch.tensor(
                self.dataset.gt_poses[:frame_id+1],
                device=self.device, 
                dtype=torch.float64
            )
    
    def process_frame(
        self,
        frame_id: int,
        keypoints_xyz_torch: torch.tensor, # 3D in local camera frame
        descriptors: torch.tensor,
        keypoints_rgb_torch: torch.tensor,
        cur_pose_torch: torch.tensor # camera pose
    ):
        frame_origin_torch = cur_pose_torch[:3, 3]

        update_points = transform_torch(keypoints_xyz_torch, cur_pose_torch) # transform from local lidar frame to global frame

        self.vhm.update(frame_id, update_points, descriptors, keypoints_rgb_torch)
        self.vhm.reset_local_map(frame_id, frame_origin_torch)

        self.vhm_cpp.update(frame_id, update_points.detach().cpu().numpy(), keypoints_rgb_torch.detach().cpu().numpy(), descriptors.detach().cpu().numpy())
        self.vhm_cpp.reset_local_map(frame_id, frame_origin_torch.detach().cpu().numpy())
        local_map_points, local_map_descriptors = self.vhm_cpp.get_local_map()
        local_map_points = torch.from_numpy(local_map_points).to(update_points)
        local_map_descriptors = torch.from_numpy(local_map_descriptors).to(descriptors)

        self.determine_mapping_poses(frame_id)