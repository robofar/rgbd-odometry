import os
import sys
import time

import torch

from utils.tools import (
    setup_experiment,
    remove_gpu_cache,
    get_time
)

import wandb
from rich import print
from tqdm import tqdm

import dtyper as typer
from typing import Optional, Tuple

from utils.config import Config
from utils.dataset import SLAMDataset
from utils.vhm import VoxelHashMap
from utils.mapper import Mapper
from utils.visualizer import Visualizer
from utils.tracker import Tracker


app = typer.Typer(add_completion=False, rich_markup_mode="rich", context_settings={"help_option_names": ["-h", "--help"]})

@app.command()
def run_pings(config_path: str = typer.Argument(help='Path to *.yaml config file')):
    config = Config()
    config.load(config_path)

    argv = sys.argv
    run_path = setup_experiment(config, argv)

    print("[bold green]RGB-D SLAM starts[/bold green]")

    visualizer = None
    if config.rerun_viz_on:
        visualizer = Visualizer(config, app_id="SFVO", spawn_viewer=True)

    dataset = SLAMDataset(config=config)
    vhm = VoxelHashMap(config=config, viz=visualizer)
    mapper = Mapper(config, dataset, vhm) # right now nothing special, but we can extend it to whatever mapping we want
    tracker = Tracker(config, vhm, visualizer)

    for frame_id in tqdm(range(dataset.total_pc_count)):
        print(f"Frame: {frame_id}")
        remove_gpu_cache()

        # I. Load data ; Extract Features ; Guess initial pose (constant velocity model)
        T0 = get_time()
        if config.use_dataloader:
            dataset.read_frame_with_loader(frame_id)
        else:
            print("[bold red]For now, only dataloaders are supported[/bold red]")
            print("[bold red]Exiting...[/bold red]")
            sys.exit(0)
        

        dataset.preprocess_frame(frame_id)
        dataset.initial_pose_guess(frame_id)
        

        # II. Odometry
        T1 = get_time()
        if frame_id > 0:
            if config.track_on:
                print(f"Tracking...")
                cur_odom_cov = None # [TODO]: Return estimated odometry covariance from tracker
                cur_pose_torch_odom = tracker.tracking(frame_id, dataset.cur_cam, dataset.cur_pose_guess_torch, config.query_locally)
                dataset.update_poses(frame_id, cur_pose_torch_odom) # mapping for debug
            else: # incremental mapping with gt pose
                if dataset.gt_pose_provided:
                    print("Mapping...")
                    dataset.update_poses(frame_id, dataset.cur_pose_guess_torch) 
                else:
                    sys.exit("You are using the mapping mode, but no pose is provided.")
        else:
            dataset.update_poses(frame_id, dataset.cur_pose_guess_torch)


        #travel_dist = dataset.travel_dist[:frame_id + 1]
        #vhm.travel_dist = torch.tensor(travel_dist, device=config.device, dtype=config.dtype) # always update this. needed for local map setting

        # III. Loop detection and pgo
        T2 = get_time()
        if config.pgo_on:
            sys.exit("PGO not yet integrated...")

        # IV: Mapping
        T3 = get_time()
        # Update global map ; Reset local map ; Determine used poses for mapping
        mapper.process_frame(
            frame_id,
            dataset.cur_cam.keypoints_xyz_cam,
            dataset.cur_cam.descriptors,
            dataset.cur_cam.keypoints_rgb,
            dataset.cur_pose_torch
        )

        if visualizer is not None:
            if frame_id==0 and dataset.gt_pose_provided:
                visualizer.log_world_frame(dataset.gt_poses[frame_id])
            
            if dataset.odom_poses is not None:
                visualizer.log_current_odometry_frame(dataset.odom_poses[frame_id])
                visualizer.log_odometry_positions(dataset.odom_poses[:frame_id+1])
                visualizer.log_odom_trajectory(dataset.odom_poses[:frame_id+1])

            if dataset.gt_pose_provided:
                visualizer.log_current_gt_frame(dataset.gt_poses[frame_id])
                visualizer.log_gt_positions(dataset.gt_poses[:frame_id+1])
                visualizer.log_gt_trajectory(dataset.gt_poses[:frame_id+1])
            
            visualizer.log_current_local_map(vhm.buffer_points[vhm.current_valid_mask_flat], vhm.buffer_rgb[vhm.current_valid_mask_flat])
        
        print(f"Number of points in map: {vhm.count()}")
        print("---------------")




if __name__ == "__main__":
    app()