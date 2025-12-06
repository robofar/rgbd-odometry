#pragma once

#include "VoxelHashMap.h" // Includes common types (Vec3, DescType, VoxelHashMap, ConfigVHM)
#include <Eigen/Dense>
#include <algorithm>
#include <numeric>
#include <random>
#include <sophus/se3.hpp>
#include <vector>

using Mat4 = Eigen::Matrix4d;
using Mat3 = Eigen::Matrix3d;
using Vec3 = Eigen::Vector3d;
using Vec2 = Eigen::Vector2d;
using Mat6 = Eigen::Matrix<double, 6, 6>;
using Vec6 = Eigen::Matrix<double, 6, 1>;

struct ConfigTracker {
  int reg_iterations = 10;
  double min_grad_norm = 1e-4;
  double huber_k = 1.345;
  double lowe_test_ratio = 0.8;

  double ransac_max_correspondence_threshold = 0.1;
  int ransac_min_points = 3;
  double ransac_probability = 0.999;
  double ransac_inliers_ratio = 0.1;
};

class Tracker {
public:
  Tracker(const ConfigTracker &cfg);

  Sophus::SE3d tracking(double frame_id, const Sophus::SE3d &initial_T_wc,
                        const std::vector<Vec2> &kps_2d,
                        const std::vector<Vec3> &kps_3d_cam,
                        const std::vector<DescType> &descriptors, const Mat3 &K,
                        const std::vector<Vec3> &map_points_world,
                        const std::vector<DescType> &map_descriptors);

private:
  double huberWeight(double residual, double k);

  void findInlierCorrespondences(
      const std::vector<Vec2> &query_2d, const std::vector<Vec3> &query_3d_cam,
      const std::vector<DescType> &query_descs,
      const std::vector<Vec3> &map_points_world,
      const std::vector<DescType> &map_descs,
      std::vector<Vec3> &out_map_3d_world, // 3D Map Point (World Frame)
      std::vector<Vec3> &out_query_3d_cam, // 3D Query Point (Camera Frame)
      std::vector<Vec2> &out_query_2d      // 2D Query Keypoint (Image Plane)
  ) const;

  // Helper for RANSAC
  Mat4 KabschUmeyama(const Eigen::Ref<const Eigen::MatrixXd> &ref_pts,
                     const Eigen::Ref<const Eigen::MatrixXd> &query_pts) const;

  Mat4 ransacKabschUmeyama(const std::vector<Vec2> &frame_keypoints_2d,
                           const std::vector<Vec3> &frame_keypoints_3d_camera,
                           const std::vector<Vec3> &map_pts_3d_world,
                           const Mat4 &initial_guess_T_wc,
                           std::vector<Vec3> &out_inlier_map_pts_world,
                           std::vector<Vec3> &out_inlier_frame_pts_cam,
                           std::vector<Vec2> &out_inlier_frame_keypoints) const;

  void reprojectionErrorAndJacobian(
      const Vec3 &Pw, const Vec2 &kp, const Mat3 &K, const Mat4 &T_w_c,
      Vec2 &e,                       // 2x1 Residual vector
      Eigen::Matrix<double, 2, 6> &J // 2x6 Jacobian matrix
  ) const;

  Vec6 solveGaussNewton(
      const std::vector<Vec2> &inlier_kps_2d,
      const std::vector<Vec3> &inlier_kps_3d_world, const Mat3 &K,
      const Sophus::SE3d
          &T_w_c_sophus // Use Sophus directly for the matrix conversion
  ) const;

  ConfigTracker config;
};