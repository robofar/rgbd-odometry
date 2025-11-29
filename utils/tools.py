import open3d as o3d
import os
from datetime import datetime
import torch
import multiprocessing
import warnings
import wandb
import shutil
import subprocess
import yaml
import numpy as np
import random
import getpass
import time

from utils.config import Config


# set up weight and bias
def setup_wandb():
    print("Weight & Bias logging option is on. Disable it by setting  wandb_vis_on: False  in the config file.")
    username = getpass.getuser()
    print(username)
    wandb_key_path = username + "_wandb.key"
    if not os.path.exists(wandb_key_path):
        wandb_key = input("[You need to firstly setup and login wandb] Please enter your wandb key (https://wandb.ai/authorize):")
        with open(wandb_key_path, "w") as fh:
            fh.write(wandb_key)
    else:
        print("wandb key already set")
    os.system('export WANDB_API_KEY=$(cat "' + wandb_key_path + '")')


def setup_experiment(config: Config, argv=None, debug_mode: bool = False):

    os.environ["NUMEXPR_MAX_THREADS"] = str(multiprocessing.cpu_count())
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # begining timestamp
    ts = "now"

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    warnings.filterwarnings("ignore", category=FutureWarning) 

    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print("No CUDA device available, use CPU instead")
        config.device = "cpu"
    else:
        torch.cuda.empty_cache()

    # this would make the processing slower, disabling it when you are not debugging
    # torch.autograd.set_detect_anomaly(True)

    # set the random seed for all
    seed_anything(config.seed)

    run_path = None

    if not debug_mode:

        run_name = config.name + "_" + ts  # modified to a name that is easier to index
        config.run_name = run_name

        run_path = os.path.join(config.output_root, run_name)

        access = 0o755
        os.makedirs(run_path, access, exist_ok=True)
        assert os.access(run_path, os.W_OK)
        if not config.silence:
            print(f"Start {run_path}")

        config.run_path = run_path

        map_path = os.path.join(run_path, "map")
        log_path = os.path.join(run_path, "log")
        meta_data_path = os.path.join(run_path, "meta")
        os.makedirs(map_path, access, exist_ok=True)
        os.makedirs(log_path, access, exist_ok=True)
        os.makedirs(meta_data_path, access, exist_ok=True)

        if config.wandb_vis_on:
            # set up wandb
            setup_wandb()
            wandb.init(
                project="RGBD-SLAM", config=vars(config), dir=run_path
            )  # your own worksapce
            wandb.run.name = run_name

        # write the full configs to yaml file
        config_dict = vars(config)
        config_out_path = os.path.join(meta_data_path, "config_all.yaml")
        with open(config_out_path, 'w') as file:
            yaml.dump(config_dict, file, default_flow_style=False)

    # set up dtypes, note that torch stuff cannot be write to yaml, so we set it up after write out the yaml for the whole config
    config.setup_dtype()
    torch.set_default_dtype(config.dtype)


def seed_anything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    o3d.utility.random.seed(seed)

def get_time():
    """
    :return: get timing statistics
    """
    cuda_available = torch.cuda.is_available()
    if cuda_available:  # issue #10
        torch.cuda.synchronize()
    return time.time()


def remove_gpu_cache():
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.empty_cache()


################

def voxel_down_sample_closest_to_voxel_center_torch(points: torch.tensor, voxel_size: float):
    """
        voxel based downsampling. Returns the indices of the points which are closest to the voxel centers.
    Args:
        points (torch.Tensor): [N,3] point coordinates
        voxel_size (float): grid resolution

    Returns:
        indices (torch.Tensor): [M] indices of the original point cloud, downsampled point cloud would be `points[indices]`

    Reference: Louis Wiesmann
    """
    _quantization = 1000  # if change to 1, then it would take the first (smallest) index lie in the voxel

    offset = torch.floor(points.min(dim=0)[0] / voxel_size).long()
    grid = torch.floor(points / voxel_size)
    center = (grid + 0.5) * voxel_size
    dist = ((points - center) ** 2).sum(dim=1) ** 0.5
    dist = (
        dist / dist.max() * (_quantization - 1)
    ).long()  # for speed up # [0-_quantization]

    grid = grid.long() - offset
    v_size = grid.max().ceil()
    grid_idx = grid[:, 0] + grid[:, 1] * v_size + grid[:, 2] * v_size * v_size

    unique, inverse = torch.unique(grid_idx, return_inverse=True)
    idx_d = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)

    offset = 10 ** len(str(idx_d.max().item()))

    idx_d = idx_d + dist.long() * offset

    idx = torch.empty(
        unique.shape, dtype=inverse.dtype, device=inverse.device
    ).scatter_reduce_(
        dim=0, index=inverse, src=idx_d, reduce="amin", include_self=False
    )
    # https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce_.html
    # This operation may behave nondeterministically when given tensors on
    # a CUDA device. consider to change a more stable implementation

    idx = idx % offset
    return idx



def voxel_down_sample_per_voxel(points: torch.Tensor, voxel_size: float, mode: str = "first") -> torch.Tensor:
    """
    Downsample points by selecting one representative per voxel.

    Args:
        points (torch.Tensor): (N,3)
        voxel_size (float): voxel edge length
        mode (str): "first" or "random"

    Returns:
        torch.Tensor: (M,) indices of selected points
    """

    device = points.device
    N = points.shape[0]

    # --- 1️⃣ Compute voxel coordinates
    grid = torch.floor(points / voxel_size)

    # --- 2️⃣ Shift to positive
    gmin = grid.min(dim=0).values
    grid = (grid - gmin).to(torch.long)

    # --- 3️⃣ Linear voxel ID
    dims = grid.max(dim=0).values + 1
    vx, vy = int(dims[0].item()), int(dims[1].item())
    lin = grid[:, 0] + grid[:, 1] * vx + grid[:, 2] * vx * vy  # (N,)

    # --- 4️⃣ Unique voxel IDs and mapping
    unique, inverse = torch.unique(lin, return_inverse=True)
    V = unique.shape[0]

    if mode == "first":
        idx = torch.arange(N, device=device)
        # Vectorized "first occurrence" per voxel (no Python loop)
        first_idx = torch.full((V,), N, dtype=torch.long, device=device)
        first_idx.scatter_reduce_(0, inverse, idx, reduce="amin", include_self=False)
        selected_idx = first_idx

    elif mode == "random":
        # Fully vectorized random pick per voxel (no Python loop)
        rnd = torch.rand(N, device=device)
        # compute random max per voxel
        max_rnd = torch.full((V,), -1.0, dtype=rnd.dtype, device=device)
        max_rnd.scatter_reduce_(0, inverse, rnd, reduce="amax", include_self=False)
        # keep points whose random value equals voxel's max
        mask = max_rnd[inverse] == rnd
        selected_idx = mask.nonzero(as_tuple=False).flatten()

    return selected_idx


################

def transform_torch(points: torch.tensor, transformation: torch.tensor):
    """
    Transform a batch of points by a transformation matrix
    Args:
        points: N,3 torch tensor, the coordinates of all N (axbxc) query points in the scaled
                kaolin coordinate system [-1,1] (torch.float32)
        transformation: 4,4 torch tensor, the transformation matrix (torch.float64)
    Returns:
        transformed_points: N,3 torch tensor, the transformed coordinates
    """
    # Add a homogeneous coordinate to each point in the point cloud
    points_homo = torch.cat([points, torch.ones(points.shape[0], 1).to(points)], dim=1)

    # Apply the transformation by matrix multiplication
    transformed_points_homo = torch.matmul(points_homo, transformation.to(points).T)

    # Remove the homogeneous coordinate from each transformed point
    transformed_points = transformed_points_homo[:, :3]

    return transformed_points


def transform_batch_torch(points: torch.tensor, transformation: torch.tensor):
    """
    Transform a batch of points by a batch of transformation matrices
    Args:
        points: N,3 torch tensor, the coordinates of all N (axbxc) query points in the scaled
                kaolin coordinate system [-1,1]
        transformation: N,4,4 torch tensor, the transformation matrices
    Returns:
        transformed_points: N,3 torch tensor, the transformed coordinates
    """

    # Extract rotation and translation components
    rotation = transformation[:, :3, :3].to(points)
    translation = transformation[:, :3, 3:].to(points)

    # Reshape points to match dimensions for batch matrix multiplication
    points = points.unsqueeze(-1)

    # Perform batch matrix multiplication using torch.bmm(), instead of memory hungry matmul
    transformed_points = torch.bmm(rotation, points) + translation

    # Squeeze to remove the last dimension
    transformed_points = transformed_points.squeeze(-1)

    return transformed_points



def skew_symmetric_batch(v):
    """
    Input: v (N, 3)
    Output: (N, 3, 3)
    """
    N = v.shape[0]
    zeros = torch.zeros(N, device=v.device, dtype=v.dtype)
    
    # v = [x, y, z]
    x = v[:, 0]
    y = v[:, 1]
    z = v[:, 2]
    
    # [[ 0, -z,  y],
    #  [ z,  0, -x],
    #  [-y,  x,  0]]
    
    skew = torch.stack([
            zeros, -z,      y,
            z,      zeros, -x,
        -y,      x,      zeros
    ], dim=1).reshape(N, 3, 3)
    return skew