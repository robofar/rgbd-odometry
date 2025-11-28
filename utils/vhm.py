import sys
from rich import print
import torch
import torch.nn as nn
import numpy as np
import math

from utils.config import Config
from utils.visualizer import Visualizer

class VoxelHashMap(nn.Module):
    def __init__(self, config: Config, viz: Visualizer) -> None:
        super().__init__()

        self.config = config
        self.viz = viz
        self.silence = config.silence

        self.device = config.device
        self.dtype = config.dtype
        self.tran_dtype = config.tran_dtype
        self.idx_dtype = config.idx_dtype

        self.resolution = config.voxel_size_m
        self.buffer_size = config.buffer_size 
        self.k_per_voxel = self.config.num_points_per_voxel
        self.pruning_threshold = self.buffer_size * self.k_per_voxel * 2
        
        self.use_circular_buffer = config.use_circular_buffer

        # Hash Table
        self.primes = torch.tensor([73856093, 19349669, 83492791], dtype=self.idx_dtype, device=self.device)
        self.buffer_pt_index = torch.full((self.buffer_size, self.k_per_voxel), -1, dtype=self.idx_dtype, device=self.device)
        
        # Counter for Circular Buffer
        self.voxel_insertion_count = torch.zeros(self.buffer_size, dtype=self.idx_dtype, device=self.device)

        # Global Map
        self.map_points = torch.empty((0, 3), dtype=self.dtype, device=self.device)
        self.map_points_descriptor = torch.empty((0, self.config.descriptor_dim), dtype=self.config.descriptor_dtype, device=self.device)
        self.map_points_rgb = torch.empty((0, 3), dtype=self.dtype, device=self.device)
        self.point_ts_create = torch.empty((0), device=self.device, dtype=torch.int)
        self.point_ts_update = torch.empty((0), device=self.device, dtype=torch.int)

        # Local Map
        self.temporal_local_map_on = self.config.temporal_local_map_on 
        self.local_map_travel_dist = self.config.local_map_travel_dist
        self.diff_travel_dist_local = (self.config.local_map_radius * self.config.local_map_travel_dist_ratio) 
        self.diff_ts_local = self.config.diff_ts_local 
        self.local_map_radius = self.config.local_map_radius
        self.travel_dist = None  

        self.local_map_points = torch.empty((0, 3), dtype=self.dtype, device=self.device)
        self.local_map_points_descriptor = torch.empty((0, self.config.descriptor_dim), device=self.device)
        self.local_map_points_rgb = torch.empty((0, 3), dtype=self.dtype, device=self.device)
        
        self.local_mask = None
        self.global2local = None

        # Hamming Distance
        self.hamming_lut = torch.tensor(
            [bin(i).count("1") for i in range(256)],
            device=self.device,
            dtype=torch.uint8
        )

        self.set_search_neighborhood(num_nei_cells=config.num_nei_cells, search_alpha=config.search_alpha)
        self.to(self.device)


    def is_empty(self):
        return self.map_points.shape[0] == 0

    def count(self):
        return self.map_points.shape[0]

    def local_count(self):
        if self.local_map_points is not None:
            return self.local_map_points.shape[0]
        else:
            return 0

    def clean_map(self):
        """
        Removes points from map_points that are no longer referenced by buffer_pt_index.
        Re-indexes the buffer_pt_index to match the new compact map.
        """
        
        used_mask = (self.buffer_pt_index != -1)
        used_indices_flat = self.buffer_pt_index[used_mask] # 1D tensor of all active IDs

        unique_indices, inverse_indices = torch.unique(used_indices_flat, sorted=True, return_inverse=True)

        self.map_points = self.map_points[unique_indices]
        self.map_points_descriptor = self.map_points_descriptor[unique_indices]
        self.map_points_rgb = self.map_points_rgb[unique_indices]
        self.point_ts_create = self.point_ts_create[unique_indices]
        self.point_ts_update = self.point_ts_update[unique_indices]
        
        self.buffer_pt_index[used_mask] = inverse_indices
        

    def update(
        self,
        frame_id: int,
        points: torch.Tensor,
        descriptors: torch.Tensor,
        rgb: torch.Tensor
    ):
        
        
        grid_coords = torch.floor(points / self.resolution).to(self.primes) 
        hash_index  = torch.remainder((grid_coords * self.primes).sum(-1), self.buffer_size) 

        
        sort_idx = torch.argsort(hash_index)
        points_sorted = points[sort_idx]
        descriptors_sorted = descriptors[sort_idx]
        rgb_sorted = rgb[sort_idx]
        hash_index_sorted = hash_index[sort_idx]

        unique_hashes, counts = torch.unique_consecutive(hash_index_sorted, return_counts=True)
        
        
        cum_counts = torch.cumsum(counts, dim=0)
        starts = torch.zeros_like(cum_counts)
        starts[1:] = cum_counts[:-1]
        points_group_starts = starts.repeat_interleave(counts)
        full_indices = torch.arange(points.shape[0], device=self.device)
        rank_in_group = full_indices - points_group_starts


        if self.use_circular_buffer: # Circular Buffer: First-In, First-Out (Owerwrite Oldest)
            
            points_count_expanded = counts.repeat_interleave(counts)
            cutoff = points_count_expanded - self.k_per_voxel
            keep_mask = rank_in_group >= cutoff 
            
            valid_idx = keep_mask
            valid_hashes = hash_index_sorted[valid_idx]
            
            unique_hashes_final, counts_final = torch.unique_consecutive(valid_hashes, return_counts=True)
            base_insertion_counts = self.voxel_insertion_count[unique_hashes_final].repeat_interleave(counts_final)
            
            cum_counts_final = torch.cumsum(counts_final, dim=0)
            starts_final = torch.zeros_like(cum_counts_final)
            starts_final[1:] = cum_counts_final[:-1]
            relative_rank = torch.arange(valid_hashes.shape[0], device=self.device) - starts_final.repeat_interleave(counts_final)
            
            target_slots = (base_insertion_counts + relative_rank) % self.k_per_voxel
            
            self.voxel_insertion_count.index_add_(0, unique_hashes_final, counts_final)

        else: # Fixed Buffer: No new points are added once voxel is full
            
            current_buffer_rows = self.buffer_pt_index[unique_hashes] # (V, K)
            is_slot_empty = (current_buffer_rows == -1) # (V, K) boolean
            num_available_per_voxel = is_slot_empty.sum(dim=1) # (V,)
            
            num_to_take = torch.minimum(counts, num_available_per_voxel) # (V,)
            
            num_to_take_expanded = num_to_take.repeat_interleave(counts)
            keep_mask = rank_in_group < num_to_take_expanded
            
            valid_idx = keep_mask
            valid_hashes = hash_index_sorted[valid_idx]
            
            if valid_hashes.shape[0] == 0:
                return

            mask_voxels_with_adds = (num_to_take > 0)
            
            relevant_buffer_rows = current_buffer_rows[mask_voxels_with_adds]
            
            empty_row_idx, empty_col_idx = torch.nonzero(relevant_buffer_rows == -1, as_tuple=True)
            
            unique_empty_rows, count_empty_rows = torch.unique_consecutive(empty_row_idx, return_counts=True)
            
            cum_empty = torch.cumsum(count_empty_rows, dim=0)
            starts_empty = torch.zeros_like(cum_empty)
            starts_empty[1:] = cum_empty[:-1]
            starts_empty_expanded = starts_empty.repeat_interleave(count_empty_rows)
            rank_empty = torch.arange(empty_row_idx.shape[0], device=self.device) - starts_empty_expanded
            
            num_take_for_empties = num_to_take[mask_voxels_with_adds].repeat_interleave(count_empty_rows)
            
            use_slot_mask = rank_empty < num_take_for_empties
            
            final_col_indices = empty_col_idx[use_slot_mask]
            
            target_slots = final_col_indices
            
            if target_slots.shape[0] != valid_hashes.shape[0]:
                return



        valid_points = points_sorted[valid_idx]
        valid_descriptors = descriptors_sorted[valid_idx]
        valid_rgb = rgb_sorted[valid_idx]

        if valid_points.shape[0] == 0:
            return

        # Append to global list
        A = valid_points.shape[0]
        cur_pt_count = self.map_points.shape[0]
        new_ids = torch.arange(A, dtype=self.idx_dtype, device=self.device) + cur_pt_count
        new_ts = torch.full((A,), frame_id, dtype=torch.int, device=self.device)

        self.map_points = torch.cat((self.map_points, valid_points), dim=0)
        self.map_points_descriptor = torch.cat((self.map_points_descriptor, valid_descriptors), dim=0)
        self.map_points_rgb = torch.cat((self.map_points_rgb, valid_rgb), dim=0)
        self.point_ts_create = torch.cat((self.point_ts_create, new_ts), dim=0)
        self.point_ts_update = torch.cat((self.point_ts_update, new_ts), dim=0)

        # Update Buffer Index
        self.buffer_pt_index[valid_hashes, target_slots] = new_ids

        # Remove unused points
        if self.use_circular_buffer:
            self.clean_map()


    def reset_local_map(self, frame_id, sensor_position):
        if self.temporal_local_map_on: 
            point_ts_used = self.point_ts_create
            if self.local_map_travel_dist:
                delta_travel_dist = torch.abs(self.travel_dist[frame_id] - self.travel_dist[point_ts_used])
                time_mask = (delta_travel_dist < self.diff_travel_dist_local)
            else: 
                delta_t = torch.abs(frame_id - point_ts_used)
                time_mask = (delta_t < self.diff_ts_local) 
        else:
            time_mask = torch.ones(self.count(), dtype=torch.bool, device=self.device) 
        
        distance_to_sensor = self.map_points[time_mask] - sensor_position
        distance_to_sensor_squared = torch.sum(distance_to_sensor**2, dim=-1)
        distance_mask = (distance_to_sensor_squared < self.local_map_radius**2)

        time_mask_idx = torch.nonzero(time_mask).squeeze() 
        local_mask_idx = time_mask_idx[distance_mask] 

        local_mask = torch.full((time_mask.shape), False, dtype=torch.bool, device=self.device)
        local_mask[local_mask_idx] = True
        local_indices = torch.nonzero(local_mask).flatten()
        local_point_count = local_indices.size(0)

        self.local_map_points = self.map_points[local_mask]
        self.local_map_points_descriptor = self.map_points_descriptor[local_mask]
        self.local_map_points_rgb = self.map_points_rgb[local_mask]
        self.local_point_ts_create = self.point_ts_create[local_mask]
        self.local_point_ts_update = self.point_ts_update[local_mask]

        self.local_mask = local_mask

        local_mask_padded = torch.cat((local_mask, torch.tensor([False], device=self.device)))  
        global2local = torch.full_like(local_mask_padded, -1, dtype=torch.long)
        global2local[local_indices] = torch.arange(local_point_count, device=self.device)
        self.global2local = global2local



    def set_search_neighborhood(self, num_nei_cells: int = 1, search_alpha: float = 1.0):
        dx = torch.arange(
            -num_nei_cells,
            num_nei_cells + 1,
            device=self.primes.device,
            dtype=self.primes.dtype,
        )

        coords = torch.meshgrid(dx, dx, dx, indexing="ij")
        dx = torch.stack(coords, dim=-1).reshape(-1, 3) 

        dx2 = torch.sum(dx**2, dim=-1)
        self.neighbor_dx = dx[dx2 < (num_nei_cells + search_alpha) ** 2]  

        self.neighbor_K = self.neighbor_dx.shape[0]
        self.max_valid_dist2 = 3 * ((num_nei_cells + 1) * self.resolution) ** 2



    def lowe_test(self, dist1, dist2, ratio):
        valid_first = torch.isfinite(dist1)
        valid_second = torch.isfinite(dist2)
        ratio_mask = (dist1 < ratio * dist2)
        return valid_first & valid_second & ratio_mask
    

    def geometric_distance_test(self, query_points, map_points):
        diff = map_points - query_points
        dist2 = (diff ** 2).sum(dim=1)
        return dist2 < (self.max_valid_dist2)
    

    def nearest_neighbor_search(self, cur_cam, cur_cam_pose: torch.Tensor, use_neighb_voxels = True):
        query_features_cam_frame = cur_cam.keypoints_xyz_cam
        N = query_features_cam_frame.shape[0]
        ones = torch.ones((N, 1), dtype=self.dtype, device=self.device)
        query_features_cam_frame_hom = torch.cat([query_features_cam_frame, ones], dim=1)

        query_features_world_frame_hom = (cur_cam_pose @ query_features_cam_frame_hom.to(cur_cam_pose).T).T 
        query_features_world_frame = query_features_world_frame_hom[:, :3].to(query_features_cam_frame)

        grid_coords = (query_features_world_frame / self.resolution).floor().to(self.primes) 
        if use_neighb_voxels:
            neighb_cells = (grid_coords[..., None, :] + self.neighbor_dx)  
            hash_index = torch.remainder((neighb_cells * self.primes).sum(-1), self.buffer_size) 
        else:
            hash_index  = torch.remainder((grid_coords * self.primes).sum(-1), self.buffer_size) 
        
        query_voxels_points_indices = self.buffer_pt_index[hash_index] 
        query_voxels_points_indices = query_voxels_points_indices.reshape(query_voxels_points_indices.shape[0], -1)
        valid_indices_mask = (query_voxels_points_indices != -1)

        query_descriptors = cur_cam.descriptors.unsqueeze(1) 
        map_points_descriptors = self.map_points_descriptor[query_voxels_points_indices] 

        xor = query_descriptors ^ map_points_descriptors 
        ham_bits = self.hamming_lut[xor.long()]
        ham_dist = ham_bits.sum(dim=-1).to(torch.float32)
        ham_dist = ham_dist.masked_fill(~valid_indices_mask, torch.inf) 

        if not self.config.use_lowe_test:
            min_dist, argmin_k = ham_dist.min(dim=1)
            nn_indices = query_voxels_points_indices.gather(1, argmin_k.unsqueeze(1)).squeeze(1)
            valid_neighbor_descriptor = torch.isfinite(min_dist)
        else:
            best_dists, best_idx = torch.topk(ham_dist, k=2, dim=1, largest=False)
            knn_indices = query_voxels_points_indices.gather(1, best_idx)
            nn_indices = knn_indices[:, 0]
            valid_neighbor_descriptor = self.lowe_test(best_dists[:, 0], best_dists[:, 1], self.config.lowe_test_ratio)
        
        return nn_indices, valid_neighbor_descriptor
    


    def full_nearest_neighbor_search(self, cur_cam, cur_cam_pose: torch.Tensor, query_locally):
        query_features_cam_frame = cur_cam.keypoints_xyz_cam
        N = query_features_cam_frame.shape[0]

        ones = torch.ones((N, 1), dtype=self.dtype, device=self.device)
        query_features_cam_frame_hom = torch.cat([query_features_cam_frame, ones], dim=1)
        query_features_world_frame_hom = (cur_cam_pose @ query_features_cam_frame_hom.to(cur_cam_pose).T).T 
        query_features_world_frame = query_features_world_frame_hom[:, :3].to(query_features_cam_frame)

        query_descriptors_cpu = cur_cam.descriptors.to("cpu")   
        if query_locally:           
            map_descriptors_cpu = self.local_map_points_descriptor.to("cpu")   
        else:
            map_descriptors_cpu = self.map_points_descriptor.to("cpu")      
        
        hamming_lut_cpu = self.hamming_lut.to("cpu")    
        q = query_descriptors_cpu.unsqueeze(1)          
        m = map_descriptors_cpu.unsqueeze(0)           
        xor = q ^ m                                
        ham_bits = hamming_lut_cpu[xor.long()]    
        ham_dist = ham_bits.sum(dim=-1).to(torch.float32) 
        
        if not self.config.use_lowe_test:
            min_dist, argmin_k = ham_dist.min(dim=1)       
            nn_indices_cpu = argmin_k                          
            valid_neighbor_descriptor_cpu = torch.isfinite(min_dist)
        else:
            best_dists, best_idx = torch.topk(ham_dist, k=2, dim=1, largest=False)
            nn_indices_cpu = best_idx[:, 0]
            valid_neighbor_descriptor_cpu = self.lowe_test(best_dists[:, 0], best_dists[:, 1], self.config.lowe_test_ratio)
        
        nn_indices = nn_indices_cpu.to(device=self.device, dtype=self.idx_dtype)
        valid_neighbor = valid_neighbor_descriptor_cpu.to(device=self.device) 

        return nn_indices, valid_neighbor