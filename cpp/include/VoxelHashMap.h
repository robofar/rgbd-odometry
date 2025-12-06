#pragma once

#include <Eigen/Core>
#include <array>
#include <cstdint>
#include <deque>
#include <unordered_map>
#include <vector>

using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
// using DescType = Eigen::VectorXd;
using DescType = std::array<uint8_t, 32>; // ORB features

// Will be filled from python Config class (while binding)
struct ConfigVHM {
  double voxel_size = 0.5;
  size_t max_points_per_voxel = 10;

  // Local Map
  double local_map_radius = 20.0;
  int max_time_diff = 30.0;
};

struct MapPoint {
  Vec3 xyz;
  Vec3 rgb;
  DescType descriptor;
  int ts_create = -1;
  int ts_update = -1;
};

struct VoxelKey {
  int x, y, z;

  bool operator==(const VoxelKey &other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

// Hash functor for VoxelKey so we can use it in std::unordered_map
struct VoxelKeyHash {
  std::size_t operator()(const VoxelKey &k) const noexcept {
    std::size_t h1 = std::hash<int>{}(k.x);
    std::size_t h2 = std::hash<int>{}(k.y);
    std::size_t h3 = std::hash<int>{}(k.z);

    // Simple hash combine (boost-style)
    std::size_t seed = h1;
    seed ^= h2 + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    seed ^= h3 + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    return seed;
  }
};

/*
struct VoxelKeyHash {
    std::size_t operator()(const VoxelKey& k) const {
        return ((std::hash<int>()(k.x) * 73856093) ^
                (std::hash<int>()(k.y) * 19349669) ^
                (std::hash<int>()(k.z) * 83492791));
    }
};
*/

using VoxelGrid =
    std::unordered_map<VoxelKey, std::deque<MapPoint>, VoxelKeyHash>;

using MapData =
    std::tuple<Eigen::MatrixXd, Eigen::Matrix<uint8_t, Eigen::Dynamic, 32>>;

class VoxelHashMap {
public:
  VoxelHashMap(const ConfigVHM &cfg);

  size_t count() const;
  void update(
      int frame_id,
      const Eigen::Ref<const Eigen::MatrixXd> &points, // N x 3 (XYZ in W frame)
      const Eigen::Ref<const Eigen::MatrixXd> &rgb,    // N x 3 (RGB)
      const Eigen::Ref<const Eigen::Matrix<uint8_t, Eigen::Dynamic, 32>>
          &descriptors // N x 32
  );
  void resetLocalMap(int frame_id, const Vec3 &sensor_pos);
  MapData getLocalMap() const;

private:
  ConfigVHM config;
  VoxelGrid grid;
  VoxelKey pointToKey(const Vec3 &pt) const;
};