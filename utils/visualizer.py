import sys
import os

from typing import Optional, Union, Tuple, Sequence

import numpy as np
import torch

import rerun as rr

from utils.config import Config


class Visualizer:
    """
    ReRun visualizer.
    """

    def __init__(
        self,
        config: Config,
        app_id: str = "SFVO",
        spawn_viewer: bool = True
    ) -> None:
        self.config = config

        self.world_axes_length = self.config.world_axes_length
        self.current_axes_length = self.config.current_axes_length

        self.default_point_radius = self.config.point_radius

        self.odometry_trajectory_color = self.config.odometry_trajectory_color
        self.slam_trajectory_color = self.config.slam_trajectory_color
        self.gt_trajectory_color = self.config.gt_trajectory_color

        self._odometry_traj: list[np.ndarray] = []     # odometry positions
        self._slam_traj: list[np.ndarray] = []  # slam positions
        self._gt_traj: list[np.ndarray] = []  # ground-truth positions

        rr.init(application_id=app_id, spawn=spawn_viewer)
        if spawn_viewer:
            rr.spawn()
    

    ################ Poses as Frames
    def log_world_frame(self, init_pose):
        t = init_pose[:3,3]
        R = init_pose[:3,:3]
        colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)

        rr.log("poses/world", rr.Transform3D(translation=t, mat3x3=R))
        self._log_axes("poses/world/axes", self.world_axes_length, colors)

    def log_current_odometry_frame(self, odom_pose):
        t = odom_pose[:3,3]
        R = odom_pose[:3,:3]
        colors = np.array([[180,  80, 255], [0, 255, 255], [255, 0, 255]], dtype=np.uint8)

        rr.log("poses/odometry", rr.Transform3D(translation=t, mat3x3=R))
        self._log_axes("poses/odometry/axes", self.current_axes_length, colors)

    def log_current_slam_frame(self, slam_pose):
        pass

    def log_current_gt_frame(self, gt_pose):
        t = gt_pose[:3,3]
        R = gt_pose[:3,:3]
        colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)

        rr.log("poses/gt", rr.Transform3D(translation=t, mat3x3=R))
        self._log_axes("poses/gt/axes", self.current_axes_length, colors)
    


    ################ Positions as Points
    def log_odometry_positions(self, odom_poses):
        positions = odom_poses[:,:3,3]
        color = np.array([[255, 0, 0]], dtype=np.uint8)
        self._log_pointcloud("positions/odometry", positions, color, 0.01)
    
    def log_slam_positions(self, slam_poses):
        positions = slam_poses[:,:3,3]
        color = np.array([[0, 255, 0]], dtype=np.uint8)
        self._log_pointcloud("positions/slam", positions, color, 0.01)
    
    def log_gt_positions(self, gt_poses):
        positions = gt_poses[:,:3,3]
        color = np.array([[0, 0, 255]], dtype=np.uint8)
        self._log_pointcloud("positions/gt", positions, color, 0.01)
    


    ################ Trajectory as Line
    def log_odom_trajectory(self, odom_poses):
        node_prev = odom_poses[:-1,:3,3]
        node_cur = odom_poses[1:,:3,3]
        lines = np.stack([node_prev, node_cur], axis=1)
        color = np.array([0, 255, 0], dtype=np.uint8)
        colors = np.tile(np.array(color), (lines.shape[0], 1))
        
        rr.log("trajectory/odometry", rr.LineStrips3D(lines, colors=color))
    
    def log_slam_trajectory(self, slam_poses):
        node_prev = slam_poses[:-1,:3,3]
        node_cur = slam_poses[1:,:3,3]
        lines = np.stack([node_prev, node_cur], axis=1)
        color = np.array([0, 0, 255], dtype=np.uint8)
        colors = np.tile(np.array(color), (lines.shape[0], 1))
        
        rr.log("trajectory/slam", rr.LineStrips3D(lines, colors=color))
    
    def log_gt_trajectory(self, gt_poses):
        node_prev = gt_poses[:-1,:3,3]
        node_cur = gt_poses[1:,:3,3]
        lines = np.stack([node_prev, node_cur], axis=1)
        color = np.array([255, 0, 0], dtype=np.uint8)
        colors = np.tile(np.array(color), (lines.shape[0], 1))
        
        rr.log("trajectory/gt", rr.LineStrips3D(lines, colors=color))
    

    ################ Map
    def log_current_local_map(self, local_map, color=None):
        if isinstance(local_map, torch.Tensor):
            local_map = self._torch_to_numpy(local_map)
        if color is None:
            color = np.array([0,  0, 255], dtype=np.uint8)
        else:
            if isinstance(color, torch.Tensor):
                color = self._torch_to_numpy(color)
        self._log_pointcloud("world/current_local_map", local_map, color, 0.01)
    

    def log_frame_keypoints(self, frame_keypoints_xyz, color=None):
        if isinstance(frame_keypoints_xyz, torch.Tensor):
            frame_keypoints_xyz = self._torch_to_numpy(frame_keypoints_xyz)
        if color is None:
            color = np.array([0,  255, 0], dtype=np.uint8)
        else:
            if isinstance(color, torch.Tensor):
                color = self._torch_to_numpy(color)
        self._log_pointcloud("world/frame_keypoints", frame_keypoints_xyz, color, 0.01)
    

    def log_map_correspondences(self, map, color=None):
        if isinstance(map, torch.Tensor):
            map = self._torch_to_numpy(map)
        if color is None:
            color = np.array([255,  0, 0], dtype=np.uint8)
        else:
            if isinstance(color, torch.Tensor):
                color = self._torch_to_numpy(color)
        self._log_pointcloud("world/map_corr", map, color, 0.01)
    

    def log_query_correspondences(self, query, color=None):
        if isinstance(query, torch.Tensor):
            query = self._torch_to_numpy(query)
        if color is None:
            color = np.array([0,  0, 255], dtype=np.uint8)
        else:
            if isinstance(color, torch.Tensor):
                color = self._torch_to_numpy(color)
        self._log_pointcloud("world/query_corr", query, color, 0.01)
    

    def log_corr_lines(self, map_corr, query_corr):
        if isinstance(map_corr, torch.Tensor):
            map_corr = self._torch_to_numpy(map_corr)
        if isinstance(query_corr, torch.Tensor):
            query_corr = self._torch_to_numpy(query_corr)

        lines = np.stack([map_corr, query_corr], axis=1)
        color = np.array([0, 255, 0], dtype=np.uint8)
        colors = np.tile(np.array(color), (lines.shape[0], 1))
        
        rr.log("trajectory/corr_lines", rr.LineStrips3D(lines, colors=color))


    ################ RGB Image
    ################ Depth Image
    


    ################ helpers
    def _torch_to_numpy(self, tensor: torch.Tensor):
        ndarray = tensor.detach().cpu().numpy()
        return ndarray

    def _log_axes(self, path: str, length: float, colors: np.ndarray) -> None:
        """Draw an RGB axis triad (X red, Y green, Z blue) at the entity's transform."""
        origins = np.zeros((3, 3), dtype=np.float32)
        vectors = np.eye(3, dtype=np.float32) * length
        rr.log(path, rr.Arrows3D(origins=origins, vectors=vectors, colors=colors))
    

    def _log_pointcloud(
        self,
        path: str,
        points_xyz, # np or torch
        color = None, # np or torch
        radii: float = 0.1,
    ) -> None:
        
        if color is not None:
            rr.log(path, rr.Points3D(positions=points_xyz, colors=color, radii=radii))
        else:
            rr.log(path, rr.Points3D(positions=points_xyz, radii=radii))