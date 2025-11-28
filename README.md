# RGB-D Odometry

## 🚀 Pipeline Overview
<details>
<summary>Details (click to expand)</summary>
The pipeline processes incoming RGB-D frames sequentially to estimate the camera pose and incrementally build a (local) map of features.

### 1. Feature Extraction (Incoming Frame)

For every incoming RGB-D frame, ORB algorithm is used to detect keypoints and compute their descriptors. These features form the basis for frame-to-map data association in the next step.

### 2. Pose Estimation (Frame-to-Map Matching)

For all frames following the initial frame (`frame_id > 0`), the camera's current pose is estimated by matching the newly extracted frame features against the existing features in the local map.

The estimation process follows these steps:
  
- **Data Association**: Correspondences are established by comparing the ORB descriptors of the frame features with those of the local map features.

- **Data Association Verification (Lowe's Test & RANSAC)**: The initial correspondences are filtered using Lowe's ratio test. The remaining correspondences are further pruned using RANSAC. A Kabsch-Umeyama model fitting is performed within RANSAC iterations to robustly identify the inliers.

- **Pose Optimization (3D-2D Reprojection)**: The inlier correspondences are used to minimize the 3D-2D reprojection error, where camera pose is optimized. This minimization is performed using the weighted Least Squares (Gauss-Newton method).

- Robustness: To handle potential outliers and measurement noise during optimization, the reprojection error is weighted using a robust L1-norm kernel.

### 3. Map Management (Voxel Grid)
Once the optimized camera pose is obtained, the 2D features detected in the current frame are projected into the 3D world coordinate frame using their depth values and the estimated camera pose. The projected 3D features are then stored in a spatially partitioned map.

- **Voxel Grid Implementation**: The map is implemented as a Hash Table where the keys correspond to the 3D coordinates of the voxels.
- **Density Control**: Each voxel is limited to storing a maximum of $N_{max}$ number of features.
- **Feature Update Policy (FIFO)**: A First-In, First-Out (FIFO) replacement policy is implemented within each voxel. When a voxel is full, and the new feature projects to that voxel, the oldest feature is discarded, ensuring that the map always retains the newest and most relevant spatial information.

### 4. Local Map Reset Policy

- **Radius Threshold**: The local map is centered around the latest estimated camera pose. Features that fall inside a predefined spatial radius from this pose are part of this local map.

- **Time/Distance Threshold**: The removal of features may also be triggered based on elapsed time or total traveled distance since the features were last observed, ensuring stale features are pruned.

</details>

## 🛠️ Installation

### 0. Clone the repository
```
git clone https://github.com/robofar/rgbd-odometry.git
cd rgbd-odometry/
```

### 1. Set up conda environment
```
conda create -n rgbd_odometry python=3.10
conda activate rgbd_odometry
```

### 2. Install PyTorch
Depending on your NVIDIA driver version, different CUDA version are supported. For example, NVIDIA `535.183.01` driver version supports up to the CUDA `12.2` version. Therefore, you would install specific `PyTorch` version that is built on top of `<=12.2` CUDA version:
```
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 3. Install other dependencies
Other deps are given in `requirements.txt` (to be filtered soon) and can be installed via:
```
pip3 install -r requirements.txt
```

### 3. Run
Currently you can run the system on the [TUM RGB-D dataset](https://cvg.cit.tum.de/data/datasets/rgbd-dataset). ROS is to be supported.
```
python3 main.py ./config/tum.yaml
```

## 💡 Inspiration & References
This project and my interest in SLAM are heavily inspired by various researchers in this field.
- Cyrill Stachniss
- Yue Pan
- Xingguang Zhong
- Daniel Cremers
- Luca Carlone
- Giorgio Grisetti
- Davide Scaramuzza
- Tiziano Guadagnino