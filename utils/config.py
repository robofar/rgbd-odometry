import os

import yaml

import torch

from typing import Sequence



class Config:
    def __init__(self):

        ################# settings
        self.name: str = "rgbd_slam"
        self.run_name: str = ""         # name + timestamp
        self.output_root: str = "experiments"  # output root folder
        self.run_path: str = "" # output_root + run_name
        self.data_path: str = "/media/faris/1B6B-12E9/TUM/"  # input to the root of the data (root folder which then contains scans, imgs, poses, calibs, ...)

        self.use_dataloader: bool = True
        self.data_loader_name: str = ""
        self.data_loader_seq: str = ""

        self.first_frame_ref: bool = True  # if false, we directly use the world
        self.begin_frame: int = 0  # begin from this frame
        self.end_frame: int = 1000  # end at this frame
        self.step_frame: int = 1  # process every x frame
        self.stop_frame_thre: int = 20 # determine if the robot is stopped when there's almost no motion in a time peroid

        self.seed: int = 42 # random seed for the experiment
        self.num_workers: int = 12 # number of worker for the dataloader
        self.device: str = "cuda"  # use "cuda" or "cpu"
        self.gpu_id: str = "0"  # used GPU id
        self.silence: bool = True # print log in the terminal or not

        # kitti specific
        self.kitti_correction_on: bool = False # intrinsic vertical angle correction # issue 11
        self.correction_deg: float = 0.0

        ################# hash map data structure
        self.buffer_size: int = 20000 # 1_000_000

        self.voxel_size_m = 0.5
        self.num_points_per_voxel: int = 50

        self.descriptor_dim = 32 # ORB-32 ; SIFT-256

        # Correspondence search
        self.num_nei_cells = 1
        self.search_alpha = 1.0

        self.use_circular_buffer = True
        
        # Local Map
        self.temporal_local_map_on: bool = True
        self.local_map_travel_dist: bool = False
        self.local_map_radius = 5.0
        self.local_map_travel_dist_ratio = 0.2
        self.diff_ts_local = 30
        self.use_mid_ts = False


        ################# preprocess
        # rgbd camera depth filter
        self.max_depth: float = 5.0
        self.max_keypoints : int = 1000

        self.source_vox_down_m: float = 1.5 * self.voxel_size_m

        self.rand_downsample: bool = False  # apply random or voxel downsampling to input original point clcoud
        self.rand_down_r: float = 1.0 # the decimation ratio if using random downsampling (0-1)
        self.vox_down_m: float = 0.5 * self.voxel_size_m # the voxel size if using voxel downsampling (unit: m)

        ################# tracking
        self.track_on: bool = True
        self.uniform_motion_on: bool = True

        self.use_robust_kernel: bool = True

        self.reg_iter_n = 500 # KISS-ICP has 500
        self.reg_convergence_criterion = 0.001 # 0.0001 

        self.use_all_map_points : bool = True
        self.query_locally: bool = True

        self.matcher_type = "ORB"
        self.use_lowe_test : bool = True
        self.lowe_test_ratio : float = 0.7

        self.ransac_max_correspondence_distance : float = 0.2
        

        ################# pgo
        self.pgo_on: bool = False

        ################# rerun visualizer
        self.rerun_viz_on: bool = False
        self.world_axes_length: float = 2.0
        self.current_axes_length: float = 1.0
        self.point_radius: float = 0.02
        self.odometry_trajectory_color: Sequence[int] = (255, 165, 0)    # orange
        self.slam_trajectory_color: Sequence[int] = (255, 0, 0)    # red
        self.gt_trajectory_color: Sequence[int] = (0, 0, 255)    # blue

        ################# eval
        self.wandb_vis_on: bool = False # monitor the training on weight and bias or not
    

    def setup_dtype(self):
        self.dtype = torch.float32 # default torch tensor data type
        self.tran_dtype = torch.float64 # dtype used for all the transformation and poses
        self.idx_dtype = torch.int64
        self.descriptor_dtype = torch.uint8 # ORB

    def load(self, config_file):
        config_args = yaml.safe_load(open(os.path.abspath(config_file)))

        # common settings
        if "setting" in config_args:
            self.name = config_args["setting"].get("name", self.name)
            self.data_loader_name = config_args["setting"].get("data_loader_name", self.data_loader_name)
            self.data_loader_seq = config_args["setting"].get("data_loader_seq", self.data_loader_seq)

            self.end_frame = config_args["setting"].get("end_frame", self.end_frame)
        
        if "tracking" in config_args:
            self.track_on = config_args["tracking"].get("track_on", self.track_on)
            self.use_all_map_points = config_args["tracking"].get("use_all_map_points", self.use_all_map_points)