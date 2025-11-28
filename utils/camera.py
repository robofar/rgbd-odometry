import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Camera:
    def __init__(self, frame_id, rgb_image_np, depth_image_np, K, matcher_type, device = "cuda"):
        self.frame_id = frame_id

        self.device = device
        self.dtype = torch.float32
        self.tran_dtype = torch.float64
        self.idx_dtype = torch.int64
        self.descriptor_dtype = torch.uint8

        # RGB
        self.rgb_image_np = rgb_image_np  # (H,W,3), uint8
        self.rgb_image_torch = (torch.from_numpy(rgb_image_np).float().permute(2,0,1).to(device) / 255.0)
        self.rgb_image_torch = self.rgb_image_torch.clamp(0.0, 1.0)

        self.image_width = self.rgb_image_torch.shape[2]
        self.image_height = self.rgb_image_torch.shape[1]

        # Depth
        self.depth_image_np = depth_image_np
        self.depth_image_torch = (torch.from_numpy(depth_image_np).double().permute(2,0,1).squeeze(0).to(device))

        # Intrinsic
        self.K = K
        self.fx = K[0,0]
        self.fy = K[1,1]
        self.cx = K[0,2]
        self.cy = K[1,2]
        self.K_torch = torch.tensor(K, dtype=self.dtype, device=self.device)

        if matcher_type == "ORB":
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        elif matcher_type == "SIFT":
            self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        else:
            raise ValueError(f"Unknown matcher_type: {matcher_type}")

        # Feature detection
        self.keypoints = None # full keypoint info
        self.keypoints_xy = None # (x,y)
        self.keypoints_rgb = None
        self.descriptors = None
        self.keypoints_xyz_cam = None
    

    def extract_features(self, max_features, method="ORB"):
        gray = cv2.cvtColor(self.rgb_image_np, cv2.COLOR_RGB2GRAY)

        if method == "ORB":
            orb = cv2.ORB_create(nfeatures=max_features)
            kps, desc = orb.detectAndCompute(gray, None)

        elif method == "SIFT":
            sift = cv2.SIFT_create(nfeatures=max_features)
            kps, desc = sift.detectAndCompute(gray, None)

        else:
            raise ValueError(f"Unknown method: {method}")

        pts = np.array([kp.pt for kp in kps], dtype=np.float32)  # (M,2)
        self.keypoints_xy = torch.from_numpy(pts).to(self.device, self.dtype)
        self.descriptors = torch.from_numpy(desc).to(self.device) # dtype stays the same as in np

        xy_int = self.keypoints_xy.to(dtype=self.idx_dtype)
        x = xy_int[:, 0]
        y = xy_int[:, 1]
        self.keypoints_depth = self.depth_image_torch[y, x]
        self.keypoints_rgb = self.rgb_image_torch[:, y, x].T
    

    def filter_depth(self, d_max):
        mask = (
            torch.isfinite(self.keypoints_depth) &
            (self.keypoints_depth > 0) &
            (self.keypoints_depth <= d_max)
        )

        self.keypoints_xy = self.keypoints_xy[mask]
        self.keypoints_depth = self.keypoints_depth[mask]
        self.keypoints_rgb = self.keypoints_rgb[mask]
        self.descriptors = self.descriptors[mask]

    
    def keypoints_xyz_camera_frame(self):
        N = self.keypoints_xy.shape[0]
        ones = torch.ones((N, 1), dtype=self.dtype, device=self.device)
        uv_hom = torch.cat([self.keypoints_xy, ones], dim=1) # [u, v, 1]

        K_inv = torch.linalg.inv(self.K_torch).to(self.K_torch)
        rays = (K_inv @ uv_hom.T).T  # [x/z, y/z, 1]
        z = self.keypoints_depth.to(self.dtype).unsqueeze(1)
        xyz_cam = rays * z # [x, y, z]

        self.keypoints_xyz_cam = xyz_cam

