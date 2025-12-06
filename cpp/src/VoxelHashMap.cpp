#include "VoxelHashMap.h"

VoxelHashMap::VoxelHashMap(const ConfigVHM &cfg) : config(cfg) {}

VoxelKey VoxelHashMap::pointToKey(const Vec3 &pt) const {
  return {static_cast<int>(std::floor(pt.x() / config.voxel_size)),
          static_cast<int>(std::floor(pt.y() / config.voxel_size)),
          static_cast<int>(std::floor(pt.z() / config.voxel_size))};
}

size_t VoxelHashMap::count() const {
  size_t total = 0;
  for (const auto &kv : grid) {
    total += kv.second.size();
  }
  return total;
}

void VoxelHashMap::update(
    int frame_id, const Eigen::Ref<const Eigen::MatrixXd> &points,
    const Eigen::Ref<const Eigen::MatrixXd> &rgb,
    const Eigen::Ref<const Eigen::Matrix<uint8_t, Eigen::Dynamic, 32>>
        &descriptors) {

  size_t N = points.rows();

  for (size_t i = 0; i < N; ++i) {
    MapPoint pt;
    pt.xyz = points.row(i).head<3>().transpose(); // Get 3D point (XYZ)
    pt.rgb = rgb.row(i).head<3>().transpose();    // Get RGB
                                                  //
    // desc_row is Eigen and pt.descriptor is std::array
    // pt.descriptor = descriptors.row(i).transpose(); // Get Descriptor vector
    const auto &desc_row = descriptors.row(i);
    std::copy(desc_row.begin(), desc_row.end(), pt.descriptor.begin());

    pt.ts_create = frame_id;
    pt.ts_update = frame_id;

    VoxelKey key = pointToKey(pt.xyz);
    auto &voxel = grid[key]; // access or insert voxel

    voxel.push_back(pt);

    // FIFO
    if (voxel.size() > config.max_points_per_voxel) {
      voxel.pop_front();
    }
  }
}

void VoxelHashMap::resetLocalMap(int frame_id, const Vec3 &sensor_pos) {

  double r2 = config.local_map_radius * config.local_map_radius;

  for (auto it = grid.begin(); it != grid.end();) {
    std::deque<MapPoint> &voxel = it->second;

    auto pt_it = voxel.begin();
    while (pt_it != voxel.end()) {
      double dist_sq = (pt_it->xyz - sensor_pos).squaredNorm();
      int time_diff = frame_id - pt_it->ts_create;

      bool too_far = dist_sq > r2;
      bool too_old = time_diff > config.max_time_diff;

      if (too_far || too_old) {
        pt_it = voxel.erase(pt_it);
      } else {
        ++pt_it;
      }
    }

    if (voxel.empty()) {
      it = grid.erase(it);
    } else {
      ++it;
    }
  }
}

MapData VoxelHashMap::getLocalMap() const {
  size_t total_points = 0;
  for (const auto &kv : this->grid) {
    total_points += kv.second.size();
  }

  Eigen::MatrixXd out_points(total_points, 3);
  Eigen::Matrix<uint8_t, Eigen::Dynamic, 32> out_descs(total_points, 32);

  size_t row_idx = 0;
  for (const auto &kv : this->grid) {
    for (const auto &pt : kv.second) {
      // Copy XYZ
      out_points.row(row_idx) = pt.xyz.transpose();

      // Copy Descriptor (must use map to handle std::array to Eigen row
      // efficiently)
      Eigen::Map<const Eigen::Matrix<uint8_t, 1, 32>> desc_map(
          pt.descriptor.data());
      out_descs.row(row_idx) = desc_map;

      row_idx++;
    }
  }

  return std::make_tuple(out_points, out_descs);
}