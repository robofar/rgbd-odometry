import torch
import torch.nn as nn
import math

class VoxelHashMap(nn.Module):
    def __init__(self, config, viz) -> None:
        super().__init__()

        self.config = config
        self.viz = viz

        self.device = config.device
        self.dtype = config.dtype
        self.tran_dtype = config.tran_dtype
        self.idx_dtype = config.idx_dtype

        self.resolution = config.voxel_size_m
        self.buffer_size = config.buffer_size  # Number of hash map slots (V)
        self.k_per_voxel = self.config.num_points_per_voxel # Points per voxel slot (K)
        self.total_capacity = self.buffer_size * self.k_per_voxel

        self.temporal_local_map_on = self.config.temporal_local_map_on 
        self.local_map_travel_dist = self.config.local_map_travel_dist
        self.diff_travel_dist_local = (self.config.local_map_radius * self.config.local_map_travel_dist_ratio) 
        self.diff_ts_local = self.config.diff_ts_local 
        self.local_map_radius = self.config.local_map_radius
        self.travel_dist = None  

        self.primes = torch.tensor([73856093, 19349669, 83492791], dtype=self.idx_dtype, device=self.device)

        self.buffer_points = torch.empty((self.total_capacity, 3), dtype=self.dtype, device=self.device)
        self.buffer_descriptors = torch.empty((self.total_capacity, self.config.descriptor_dim), dtype=self.config.descriptor_dtype, device=self.device)
        self.buffer_rgb = torch.empty((self.total_capacity, 3), dtype=self.dtype, device=self.device)
        self.buffer_ts_create = torch.empty((self.total_capacity), device=self.device, dtype=self.idx_dtype)
        self.buffer_ts_update = torch.empty((self.total_capacity), device=self.device, dtype=self.idx_dtype)

        self.buffer_status = torch.full((self.buffer_size, self.k_per_voxel), -1, dtype=self.idx_dtype, device=self.device)
        self.voxel_insertion_count = torch.zeros(self.buffer_size, dtype=self.idx_dtype, device=self.device) # Total number of points ever added to specific voxel. Needed for circular buffer.

        self.current_valid_mask_flat = None

        # Hamming Distance
        self.hamming_lut = torch.tensor(
            [bin(i).count("1") for i in range(256)],
            device=self.device,
            dtype=torch.uint8
        )

        self.set_search_neighborhood(num_nei_cells=config.num_nei_cells, search_alpha=config.search_alpha)
        self.to(self.device)
    

    def _get_buffer_indices(self, voxel_indices, target_slots):
        return voxel_indices * self.k_per_voxel + target_slots

    def count(self):
        return (self.buffer_status != -1).sum().item()

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
    

    def update(
        self,
        frame_id: int,
        points: torch.Tensor,
        descriptors: torch.Tensor,
        rgb: torch.Tensor
    ):

        grid_coords = torch.floor(points / self.resolution).to(self.primes) 
        voxel_index  = torch.remainder((grid_coords * self.primes).sum(-1), self.buffer_size) # hash index (hash % num of voxels)

        # Sort by hash index
        sort_idx = torch.argsort(voxel_index)
        points_sorted = points[sort_idx]
        descriptors_sorted = descriptors[sort_idx]
        rgb_sorted = rgb[sort_idx]
        voxel_index_sorted = voxel_index[sort_idx]

        unique_voxel_indices, counts = torch.unique_consecutive(voxel_index_sorted, return_counts=True)
        
        # Calculate Index in Voxel of every feature point
        cum_counts = torch.cumsum(counts, dim=0)
        starts = torch.zeros_like(cum_counts)
        starts[1:] = cum_counts[:-1]
        points_voxel_starts = starts.repeat_interleave(counts)
        full_indices = torch.arange(points.shape[0], device=self.device)
        relative_index_in_voxel = full_indices - points_voxel_starts # starts from 0 (that's why is relative)

        # The point must survive the K-per-voxel cutoff for insertion
        points_count_expanded = counts.repeat_interleave(counts)
        cutoff = points_count_expanded - self.k_per_voxel # minimum index in voxel point must have (in case point count of voxel > K)
        keep_mask = relative_index_in_voxel >= cutoff 
        
        valid_voxel_indices = voxel_index_sorted[keep_mask]
        
        if valid_voxel_indices.shape[0] == 0:
            return

        # Repeat for kept points
        unique_voxel_indices_final, counts_final = torch.unique_consecutive(valid_voxel_indices, return_counts=True)
        
        cum_counts_final = torch.cumsum(counts_final, dim=0)
        starts_final = torch.zeros_like(cum_counts_final)
        starts_final[1:] = cum_counts_final[:-1]
        relative_index_in_voxel_final = torch.arange(valid_voxel_indices.shape[0], device=self.device) - starts_final.repeat_interleave(counts_final)

        ####################### circular buffer core ################
        base_insertion_counts = self.voxel_insertion_count[unique_voxel_indices_final].repeat_interleave(counts_final)
        global_index_in_voxel_final = (base_insertion_counts + relative_index_in_voxel_final) % self.k_per_voxel
        buffer_indices = self._get_buffer_indices(valid_voxel_indices, global_index_in_voxel_final) # flattened indices
        self.voxel_insertion_count.index_add_(0, unique_voxel_indices_final, counts_final)
        #############################################################

        # Get data for insertion
        valid_points = points_sorted[keep_mask]
        valid_descriptors = descriptors_sorted[keep_mask]
        valid_rgb = rgb_sorted[keep_mask]
        A = valid_points.shape[0]
        new_ts = torch.full((A,), frame_id, dtype=self.idx_dtype, device=self.device)

        # Insert/Overwrite data into the fixed-size buffers
        self.buffer_points[buffer_indices] = valid_points
        self.buffer_descriptors[buffer_indices] = valid_descriptors
        self.buffer_rgb[buffer_indices] = valid_rgb
        self.buffer_ts_create[buffer_indices] = new_ts
        self.buffer_ts_update[buffer_indices] = new_ts # New points are created and updated now

        # Update the status (mark as valid)
        # Note: Must use a view of buffer_status, not the flattened buffer_indices.
        self.buffer_status.view(-1)[buffer_indices] = frame_id
    


    ## --- Reset/Filter Function: Deletes Outdated/Distant Points---
    # The local map for retrieval is now simply the set of points where buffer_status != -1.
    def reset_local_map(self, frame_id, sensor_position):
        initial_valid_mask_flat = (self.buffer_status.flatten() != -1)
        valid_indices = torch.nonzero(initial_valid_mask_flat).flatten()
        
        
        # Temporal/Distance Check
        if self.temporal_local_map_on: 
            point_ts_create = self.buffer_ts_create.flatten()[valid_indices]
            if self.local_map_travel_dist:
                delta_travel_dist = torch.abs(self.travel_dist[frame_id] - self.travel_dist[point_ts_create])
                time_mask = (delta_travel_dist < self.diff_travel_dist_local)
            else: 
                delta_t = torch.abs(frame_id - point_ts_create)
                time_mask = (delta_t < self.diff_ts_local) 
        else:
            time_mask = torch.ones(self.count(), dtype=torch.bool, device=self.device) 
        
        # Distance Check (Spatially Local)
        valid_points = self.buffer_points.flatten(end_dim=-2)[valid_indices]
        distance_to_sensor = valid_points - sensor_position
        distance_to_sensor_squared = torch.sum(distance_to_sensor**2, dim=-1)
        distance_mask = (distance_to_sensor_squared < self.local_map_radius**2)

        # Combine Masks
        keep_mask = time_mask & distance_mask
        
        # 4. Invalidate (Delete) points that are not kept
        delete_indices_flat = valid_indices[~keep_mask]
        
        # Local map is still the same (ego-vehicle did not move)
        if delete_indices_flat.shape[0] > 0:
            self.buffer_status.view(-1)[delete_indices_flat] = -1

        self.current_valid_mask_flat = (self.buffer_status.flatten() != -1)
    

    ##########

    def lowe_test(self, dist1, dist2, ratio):
        valid_first = torch.isfinite(dist1)
        valid_second = torch.isfinite(dist2)
        ratio_mask = (dist1 < ratio * dist2)
        return valid_first & valid_second & ratio_mask

    def full_nearest_neighbor_search(self, cur_cam, cur_cam_pose: torch.Tensor):
        """
        Performs a full search over all currently valid map points (local map).
        """

        frame_descriptors = cur_cam.descriptors
        map_descriptors = self.buffer_descriptors[self.current_valid_mask_flat]

        query_descriptors_cpu = frame_descriptors.to("cpu")   
        map_descriptors_cpu = map_descriptors.to("cpu")      
        hamming_lut_cpu = self.hamming_lut.to("cpu")    
        q = query_descriptors_cpu.unsqueeze(1)          
        m = map_descriptors_cpu.unsqueeze(0)           
        xor = q ^ m                                
        ham_bits = hamming_lut_cpu[xor.long()]    
        ham_dist = ham_bits.sum(dim=-1).to(torch.float32) 
        
        # Get indices into the *valid* map points (0 to M-1)
        if not self.config.use_lowe_test:
            min_dist, argmin_k = ham_dist.min(dim=1)       
            nn_relative_idx_cpu = argmin_k                          
            valid_neighbor_descriptor_cpu = torch.isfinite(min_dist)
        else:
            best_dists, best_idx = torch.topk(ham_dist, k=2, dim=1, largest=False)
            nn_relative_idx_cpu = best_idx[:, 0]
            valid_neighbor_descriptor_cpu = self.lowe_test(best_dists[:, 0], best_dists[:, 1], self.config.lowe_test_ratio)
        
        nn_relative_idx = nn_relative_idx_cpu.to(device=self.device, dtype=self.idx_dtype)
        valid_neighbor = valid_neighbor_descriptor_cpu.to(device=self.device) 

        global_map_indices = torch.nonzero(self.current_valid_mask_flat).flatten()
        nn_absolute_idx = global_map_indices[nn_relative_idx]

        return nn_absolute_idx, valid_neighbor