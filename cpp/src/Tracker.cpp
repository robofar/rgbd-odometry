#include "Tracker.h"
#include <bit> // c++20
#include <cmath>
#include <limits>

Tracker::Tracker(const ConfigTracker &cfg) : config_(cfg) {}

double Tracker::huberWeight(double residual, double k) {
  if (residual < k) {
    return 1.0; // Quadratic region
  } else {
    return k / residual; // Linear region
  }
}

void Tracker::findInlierCorrespondences(
    const std::vector<Vec2> &query_2d, const std::vector<Vec3> &query_3d_cam,
    const std::vector<DescType> &query_descs,
    const std::vector<Vec3> &map_points_world,
    const std::vector<DescType> &map_descs, std::vector<Vec3> &out_map_3d_world,
    std::vector<Vec3> &out_query_3d_cam,
    std::vector<Vec2> &out_query_2d) const {

  out_map_3d_world.clear();
  out_query_3d_cam.clear();
  out_query_2d.clear();

  const size_t desc_size = query_descs[0].size();

  for (size_t i = 0; i < query_descs.size(); ++i) {
    int best_dist = std::numeric_limits<int>::max();
    int second_best_dist = std::numeric_limits<int>::max();
    int best_idx = -1;

    for (size_t j = 0; j < map_descs.size(); ++j) {
      int dist = 0;
      for (size_t k = 0; k < desc_size; ++k) {
        dist += std::popcount(query_descs[i][k] ^ map_descs[j][k]); // c++20
      }

      if (dist < best_dist) {
        second_best_dist = best_dist;
        best_dist = dist;
        best_idx = j;
      } else if (dist < second_best_dist) {
        second_best_dist = dist;
      }
    }

    // Lowe's Ratio Test (inliers)
    if (best_idx != -1 &&
        static_cast<double>(best_dist) <
            config.lowe_test_ratio * static_cast<double>(second_best_dist)) {

      out_map_3d_world.push_back(map_points_world[best_idx]);
      out_query_3d_cam.push_back(query_3d_cam[i]);
      out_query_2d.push_back(query_2d[i]);
    }
  }
}

Mat4 Tracker::KabschUmeyama(
    const Eigen::Ref<const Eigen::MatrixXd> &ref_pts,
    const Eigen::Ref<const Eigen::MatrixXd> &query_pts) const {

  int N = ref_pts.rows();
  if (N < 3)
    return Mat4::Identity(); // Needs at least 3 points

  Vec3 ref_mean = ref_pts.colwise().mean();
  Vec3 query_mean = query_pts.colwise().mean();

  Eigen::MatrixXd ref_centered = ref_pts.rowwise() - ref_mean.transpose();
  Eigen::MatrixXd query_centered = query_pts.rowwise() - query_mean.transpose();

  Mat3 H = (ref_centered.transpose() * query_centered) / (double)N;

  Eigen::JacobiSVD<Mat3> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Mat3 U = svd.matrixU();
  Mat3 V = svd.matrixV();

  Mat3 R = V * U.transpose();

  if (R.determinant() < 0) {
    Mat3 Mw = Mat3::Identity();
    Mw(2, 2) = -1.0;
    R = V * Mw * U.transpose();
  }

  Vec3 t = query_mean - R * ref_mean;

  Mat4 T = Mat4::Identity();
  T.block<3, 3>(0, 0) = R;
  T.block<3, 1>(0, 3) = t;

  return T;
}

Mat4 Tracker::ransacKabschUmeyama(
    const std::vector<Vec2> &frame_keypoints_2d,
    const std::vector<Vec3> &frame_keypoints_3d_camera,
    const std::vector<Vec3> &map_pts_3d_world, const Mat4 &initial_guess_T_wc,
    std::vector<Vec3> &out_inlier_map_pts_world,
    std::vector<Vec3> &out_inlier_frame_pts_cam,
    std::vector<Vec2> &out_inlier_frame_keypoints) const {

  out_inlier_frame_pts_cam.clear();
  out_inlier_map_pts_world.clear();
  out_inlier_frame_keypoints.clear();

  int N = frame_keypoints_3d_camera.size();
  const int MIN_POINTS = config.ransac_min_points; // e.g., 3

  if (N < MIN_POINTS) {
    return initial_guess_T_wc;
  }

  // Convert std::vector<Vec3> into Eigen::MatrixXd
  Eigen::MatrixXd frame_pts_cam_mat(N, 3);
  Eigen::MatrixXd map_pts_world_mat(N, 3);
  for (int i = 0; i < N; ++i) {
    frame_pts_cam_mat.row(i) = frame_keypoints_3d_camera[i].transpose();
    map_pts_world_mat.row(i) = map_pts_3d_world[i].transpose();
  }

  Eigen::MatrixXd frame_pts_world_guess =
      (initial_guess_T_wc *
       frame_pts_cam_mat.transpose().colwise().homogeneous())
          .transpose()
          .leftCols<3>(); // Get the 3D part (X, Y, Z)

  // RANSAC Setup
  std::random_device rd;
  std::mt19937 gen(rd());

  const double RANSAC_PROB = config.ransac_probability;    // e.g., 0.999
  const double INLIER_RATIO = config.ransac_inliers_ratio; // e.g., 0.1
  const double MAX_DIST = config.ransac_max_correspondence_distance;

  int max_trials =
      (int)std::ceil(std::log(1.0 - RANSAC_PROB) /
                     std::log(1.0 - std::pow(INLIER_RATIO, MIN_POINTS)));
  max_trials = std::min(max_trials, 1000);
  max_trials = std::max(max_trials, 1);

  std::vector<int> all_indices(N);
  std::iota(all_indices.begin(), all_indices.end(), 0); // [0, 1, 2, ..., N-1]

  std::vector<int> best_inliers; // Store indices temporarily
  Mat4 best_T_delta = Mat4::Identity();

  // RANSAC Loop
  for (int trial = 0; trial < max_trials; ++trial) {
    std::shuffle(all_indices.begin(), all_indices.end(), gen);
    Eigen::MatrixXd src_sample(MIN_POINTS, 3); // Source: World guess
    Eigen::MatrixXd dst_sample(MIN_POINTS, 3); // Destination: Map points

    for (int i = 0; i < MIN_POINTS; ++i) {
      int idx = all_indices[i];
      src_sample.row(i) = frame_pts_world_guess.row(idx);
      dst_sample.row(i) = map_pts_world_mat.row(idx);
    }

    Mat4 T_delta = KabschUmeyama(dst_sample, src_sample);

    Eigen::MatrixXd transformed =
        (T_delta * frame_pts_world_guess.transpose().colwise().homogeneous())
            .transpose()
            .leftCols<3>();

    std::vector<int> current_inliers;
    Eigen::MatrixXd diff = transformed - map_pts_world_mat;

    for (int i = 0; i < N; ++i) {
      // Check L2 norm (distance)
      if (diff.row(i).norm() < MAX_DIST) {
        current_inliers.push_back(i);
      }
    }

    // Model Selection
    if (current_inliers.size() > best_inliers.size()) {
      best_inliers = std::move(current_inliers);
      best_T_delta = T_delta;

      if (best_inliers.size() > 0.9 * N) { // Early exit
        break;
      }
    }
  }

  // T_final = best_T_delta * initial_guess_T_wc
  Mat4 T_final = best_T_delta * initial_guess_T_wc;

  // Filter and output the final inlier point sets
  for (int index : best_inliers) {
    out_inlier_frame_keypoints.push_back(frame_keypoints_2d[index]);
    out_inlier_frame_pts_cam.push_back(frame_keypoints_3d_camera[index]);
    out_inlier_map_pts_world.push_back(map_pts_3d_world[index]);
  }

  return T_final;
}

void Tracker::reprojectionErrorAndJacobian(
    const Vec3 &Pw, const Vec2 &kp, const Mat3 &K, const Mat4 &T_w_c, Vec2 &e,
    Eigen::Matrix<double, 2, 6> &J) const {

  Mat4 T_c_w = T_w_c.inverse();
  Mat3 R_t = T_w_c.block<3, 3>(0, 0).transpose(); // R_CW

  // Pc = T_c_w * Pw
  Vec3 Pc = T_c_w.block<3, 3>(0, 0) * Pw + T_c_w.block<3, 1>(0, 3);

  // p_hom = K * P_c
  Vec3 p_hom = K * Pc;
  Vec2 p_img;
  double Z = p_hom(2);
  double Z_inv = 1.0 / (Z + 1e-12);
  double Z2_inv = Z_inv * Z_inv;
  p_img(0) = p_hom(0) * Z_inv;
  p_img(1) = p_hom(1) * Z_inv;

  // Error 2x1
  e = p_img - kp;

  // 3. Jacobian (J = J_hom * K * R_t * J_se3): 2x6

  // J_hom (Projection Jacobian, 2x3)
  Eigen::Matrix<double, 2, 3> J_hom;
  J_hom << Z_inv, 0, -p_hom(0) * Z2_inv, 0, Z_inv, -p_hom(1) * Z2_inv;

  // J_se3 (3x6) - Derivative of P_w w.r.t Lie Algebra perturbation (xi)
  // J_se3 = [-I_3 | -[P_w]x]
  Eigen::Matrix<double, 3, 6> J_se3;
  J_se3.block<3, 3>(0, 0) = -Mat3::Identity(); // Translation part
  J_se3.block<3, 3>(0, 3) = Sophus::SO3d::hat(Pw);

  // Total Jacobian J (2x6)
  // J = J_hom * K * R_t * J_se3
  Eigen::Matrix<double, 2, 3> J_proj_kinematics = J_hom * K * R_t;
  J = J_proj_kinematics * J_se3;
}

Vec6 Tracker::solveGaussNewton(const std::vector<Vec2> &inlier_kps_2d,
                               const std::vector<Vec3> &inlier_kps_3d_world,
                               const Mat3 &K,
                               const Sophus::SE3d &T_w_c_sophus) const {

  int N = inlier_kps_2d.size();
  if (N < 4) { // Need at least 3-4 points for stable pose estimation
    return Vec6::Zero();
  }

  Mat4 T_w_c_mat = T_w_c_sophus.matrix();

  Vec2 e_i;
  Eigen::Matrix<double, 2, 6> J_i;

  Mat6 H = Mat6::Zero();
  Vec6 b = Vec6::Zero();

  for (int i = 0; i < N; ++i) {
    reprojectionErrorAndJacobian(inlier_kps_3d_world[i], inlier_kps_2d[i], K,
                                 T_w_c_mat, e_i, J_i);

    double residual = e_i.norm();
    double w_i = huberWeight(residual, config.huber_k);

    H += w_i * J_i.transpose() * J_i;
    b -= w_i * J_i.transpose() * e_i;
  }

  // H * dx = b (b is accumulated with -)
  dx = H.ldlt().solve(b);

  return dx;
}

Sophus::SE3d Tracker::tracking(double frame_id,
                               const Sophus::SE3d &initial_T_wc,
                               const std::vector<Vec2> &kps_2d,
                               const std::vector<Vec3> &kps_3d_cam,
                               const std::vector<DescType> &descriptors,
                               const Mat3 &K,
                               const std::vector<Vec3> &map_points_world,
                               const std::vector<DescType> &map_descriptors) {

  std::vector<Vec3> matched_map_3d_world;
  std::vector<Vec3> matched_query_3d_cam;
  std::vector<Vec2> matched_query_2d;

  findInlierCorrespondences(kps_2d, kps_3d_cam, descriptors, map_points_world,
                            map_descriptors, matched_map_3d_world,
                            matched_query_3d_cam, matched_query_2d);

  std::vector<Vec3> inlier_map_3d;
  std::vector<Vec3> inlier_query_3d_cam;
  std::vector<Vec2> inlier_query_2d;

  Mat4 T_ransac_mat =
      ransacKabschUmeyama(matched_query_2d, matched_query_3d_cam,
                          matched_map_3d_world, initial_T_wc.matrix(),
                          inlier_map_3d, inlier_query_3d_cam, inlier_query_2d);

  size_t num_inliers = inlier_query_2d.size();
  if (num_inliers < (size_t)config.ransac_min_points) {
    return Sophus::SE3d(T_ransac_mat);
  }

  // Non-linear Optimization (3D-2D Reprojection)
  Sophus::SE3d T_optim = Sophus::SE3d(T_ransac_mat);

  for (int iter = 0; iter < config.reg_iterations; ++iter) {
    Vec6 dx = solveGaussNewton(inlier_query_2d, inlier_map_3d, K, T_optim);
    Sophus::SE3d T_delta = Sophus::SE3d::exp(dx);
    T_optim = T_delta * T_optim;

    double dx_norm = dx.norm();
    if (dx_norm < config.min_grad_norm) {
      break;
    }
  }

  return T_optim;
}