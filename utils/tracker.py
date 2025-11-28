import numpy as np
import open3d as o3d
import torch
import sophuspy as sp

from rich import print
from tqdm import tqdm
import time
import math

from utils.config import Config
from utils.vhm import VoxelHashMap
from utils.visualizer import Visualizer
from utils.tools import skew_symmetric_batch


class Tracker:
    def __init__(
        self,
        config: Config,
        vhm: VoxelHashMap,
        viz: Visualizer
    ):
        self.config = config
        self.vhm = vhm
        self.viz = viz
        
        self.silence = config.silence
        self.device = config.device
        self.dtype = config.dtype
        self.tran_dtype = config.tran_dtype # use torch.float64 for all the transformations and poses
        self.idx_dtype = config.idx_dtype


    def reprojection_error_and_jacobians(self, keypoints_2D, keypoints_3D_world, K, T_w_c):
        N = keypoints_2D.shape[0]


        T_c_w = torch.linalg.inv(T_w_c) # (4,4)

        ones = torch.ones((N, 1), dtype=self.dtype, device=self.device)
        X_w_h = torch.cat([keypoints_3D_world, ones], dim=1) # (N,4)
        X_c_h = (T_c_w.to(X_w_h) @ X_w_h.T).T 
        X_c = X_c_h[:, :3] # (N,3)

        p_hom = (K @ X_c.T).T
        z_inv = 1.0 / (p_hom[:, 2] + 1e-12)
        u = p_hom[:, 0] * z_inv
        v = p_hom[:, 1] * z_inv
        p_img = torch.stack([u, v], dim=1) # (N,2)
        
        # Error (Prediction - Observation)
        error = p_img - keypoints_2D
        error = error.to(dtype=self.tran_dtype)

        # Jacobian
        J_hom = torch.zeros((N, 2, 3), dtype=self.tran_dtype, device=self.device)
        z2_inv = z_inv * z_inv
        
        J_hom[:, 0, 0] = z_inv
        J_hom[:, 1, 1] = z_inv
        J_hom[:, 0, 2] = -p_hom[:, 0] * z2_inv
        J_hom[:, 1, 2] = -p_hom[:, 1] * z2_inv


        J_icp = torch.zeros((N, 3, 6), dtype=self.tran_dtype, device=self.device)
        I3 = torch.eye(3, dtype=self.tran_dtype, device=self.device).unsqueeze(0).expand(N, -1, -1)
        
        J_icp[:, :, :3] = -1.0 * I3
        J_icp[:, :, 3:] = skew_symmetric_batch(keypoints_3D_world) # The C++ logic
        
        R = T_w_c[:3, :3]
        R_t = R.transpose(0, 1).unsqueeze(0).expand(N, -1, -1) # (N,3,3)
        K_expanded = K.unsqueeze(0).expand(N, -1, -1).to(dtype=self.tran_dtype) # (N,3,3)


        temp1 = torch.bmm(J_hom, K_expanded)
        temp2 = torch.bmm(temp1, R_t)
        J = torch.bmm(temp2, J_icp)

        return error, J

        



    def construct_normal_eq(self, e: torch.Tensor, J: torch.Tensor, w: torch.Tensor):
        """
        e: (N,2), J: (N,2,6), w: (N,1)
        H = J^T * J  (6,6)
        b = -J^T * e (6,)
        """

        sqrt_w = torch.sqrt(w)
        sqrt_w_2d = sqrt_w.repeat(1, 2)

        e_w = sqrt_w_2d * e
        J_w = sqrt_w.unsqueeze(-1) * J

        e_big = e_w.reshape(-1)         # (2N,)
        J_big = J_w.reshape(-1, 6)      # (2N, 6)

        H = J_big.T @ J_big           # (6,6)
        b = -J_big.T @ e_big         # (6,)

        return H, b


    def L1_norm_robust_kernel_weight(self, r):
        w = 1.0 / r
        w = w.unsqueeze(1)
        return w
    

    def kabsch_umeyama(self, ref_pts, query_pts):
        """
        Finds T such that query_pts ~= T * ref_pts
        """
        N = ref_pts.shape[0]
        
        ref_mean = torch.mean(ref_pts, dim=0, keepdim=True)
        query_mean = torch.mean(query_pts, dim=0, keepdim=True)
        
        ref_centered = ref_pts - ref_mean
        query_centered = query_pts - query_mean
        
        H = (ref_centered.transpose(0, 1) @ query_centered) / N # Covariance
        
        U, S, Vt = torch.linalg.svd(H)
        
        d = torch.det(U @ Vt)
        Mw = torch.eye(3, device=self.device, dtype=self.dtype)
        if d < 0:
            Mw[2, 2] = -1.0
            
        R = U @ Mw @ Vt
        
        t = query_mean.transpose(0, 1) - R @ ref_mean.transpose(0, 1)
        
        T = torch.eye(4, device=self.device, dtype=self.tran_dtype)
        T[:3, :3] = R
        T[:3, 3] = t.squeeze()
        
        return T
    

    def ransac_kabsch_umeyama(
        self,
        frame_keypoints_3d_camera,
        map_pts_3d_world, 
        initial_guess_T_wc,
        max_correspondence_threshold,
        min_points=3,
        probability=0.9999,
        inliers_ratio=0.1
    ):
        N = frame_keypoints_3d_camera.shape[0]
        

        ones = torch.ones((N, 1), device=self.device, dtype=self.dtype)
        frame_pts_cam_hom = torch.cat([frame_keypoints_3d_camera, ones], dim=1)
        
        frame_pts_world_guess = (initial_guess_T_wc.to(dtype=self.dtype) @ frame_pts_cam_hom.T).T[:, :3]
        
        if N < min_points:
            return initial_guess_T_wc, torch.empty(0, dtype=self.idx_dtype, device=self.device)

        best_inliers = torch.empty(0, dtype=self.idx_dtype, device=self.device)
        best_T_delta = torch.eye(4, dtype=self.tran_dtype, device=self.device) # Identity
        
        max_trials = int(math.ceil(math.log(1.0 - probability) / math.log(1.0 - inliers_ratio**min_points)))
        max_trials = min(max_trials, 1000)
        max_trials = max(max_trials, 1)
        
        indices = torch.arange(N, device=self.device)

        for _ in range(max_trials):
            sample_idx = indices[torch.randperm(N)[:min_points]]
            
            src = frame_pts_world_guess[sample_idx]
            dst = map_pts_3d_world[sample_idx]
            
            T_delta = self.kabsch_umeyama(src, dst)
            
            src_hom = torch.cat([frame_pts_world_guess, ones], dim=1)
            transformed = (T_delta.to(dtype=self.dtype) @ src_hom.T).T[:, :3]
            
            diff = transformed - map_pts_3d_world
            dist = torch.linalg.norm(diff, dim=1)
            
            current_inliers = indices[dist < max_correspondence_threshold]
            
            if len(current_inliers) > len(best_inliers):
                best_inliers = current_inliers
                best_T_delta = T_delta
                
                if len(best_inliers) > 0.9 * N: # good enough
                    break
        
        T_final = best_T_delta @ initial_guess_T_wc
        
        return T_final, best_inliers

    

    # LeastSquares PNP
    def GN(self, keypoints_2D, keypoints_3D_world, K, T_w_c, robust):
        error, J = self.reprojection_error_and_jacobians(
            keypoints_2D, keypoints_3D_world, K, T_w_c
        )
        
        # Kernel
        residual = torch.linalg.norm(error, dim=1) + 1e-9
        w = 1.0 / residual
        
        sqrt_w = torch.sqrt(w).unsqueeze(1).unsqueeze(2) # (N,1,1)
        J_weighted = J * sqrt_w # (N,2,6)
        e_weighted = error * torch.sqrt(w).unsqueeze(1) # (N,2)
        
        J_big = J_weighted.reshape(-1, 6)
        e_big = e_weighted.reshape(-1)
        
        H = J_big.T @ J_big
        b = -J_big.T @ e_big # Negative because b = -J^T * e
        
        try:
            dx = torch.linalg.solve(H, b)
        except:
            return None # singular matrix
            
        return dx
    
    


    def tracking(
        self,
        frame_id,
        cur_cam,
        init_pose,
        query_locally
    ):
        
        # Find 3D-2D correspondences only once (using init_pose)
        if self.config.use_all_map_points:
            neighb_idx, valid_neighbor_mask = self.vhm.full_nearest_neighbor_search(cur_cam, init_pose, query_locally)

            valid_2d = cur_cam.keypoints_xy[valid_neighbor_mask]
            valid_3d_cam = cur_cam.keypoints_xyz_cam[valid_neighbor_mask]

            if query_locally:
                valid_3d_world = self.vhm.local_map_points[neighb_idx[valid_neighbor_mask]]
            else:
                valid_3d_world = self.vhm.map_points[neighb_idx[valid_neighbor_mask]]
        else:
            global_neighb_idx, valid_global_neighbor_mask = self.vhm.nearest_neighbor_search(cur_cam, init_pose)
        
            if query_locally:
                local_neighb_idx = self.vhm.global2local[global_neighb_idx]
                valid_local_neighbor_mask = (local_neighb_idx != -1)

                valid_2d = cur_cam.keypoints_xy[valid_local_neighbor_mask]
                valid_3d_world = self.vhm.local_map_points[local_neighb_idx[valid_local_neighbor_mask]]
                valid_3d_cam = cur_cam.keypoints_xyz_cam[valid_local_neighbor_mask]
            else:
                valid_2d = cur_cam.keypoints_xy[valid_global_neighbor_mask]
                valid_3d_world = self.vhm.map_points[global_neighb_idx[valid_global_neighbor_mask]]
                valid_3d_cam = cur_cam.keypoints_xyz_cam[valid_global_neighbor_mask]
            
        
    
        T_ransac, inliers_idx = self.ransac_kabsch_umeyama(
            valid_3d_cam, # Points in current camera frame
            valid_3d_world, # Points in map
            init_pose,
            self.config.ransac_max_correspondence_distance
        )

        inlier_2d = valid_2d[inliers_idx]
        inlier_3d = valid_3d_world[inliers_idx]

        if inlier_2d.shape[0] < 4:
            return T_ransac
        
        T_optim = T_ransac

        # Gauss-Newton Optimization
        for i in range(self.config.reg_iter_n):
            
            delta_x = self.GN(inlier_2d, inlier_3d, cur_cam.K_torch, T_optim, self.config.use_robust_kernel)
            
            if delta_x is None:
                break
            

            delta_se3 = sp.SE3.exp(delta_x.detach().cpu().numpy())
            delta_matrix = torch.tensor(delta_se3.matrix(), dtype=self.tran_dtype, device=self.device)
            
            T_optim = delta_matrix @ T_optim

            if torch.linalg.norm(delta_x) < self.config.reg_convergence_criterion:
                break


        return T_optim
    