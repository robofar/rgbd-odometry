// Required pybind11 headers for your system
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For std::tuple and std::vector

#include "VoxelHashMap.h"

namespace py = pybind11;

PYBIND11_MODULE(rgbd_odometry, m) {
  // 1. Bind ConfigVHM Structure
  py::class_<ConfigVHM>(m, "ConfigVHM")
      .def(py::init<>()) // Default constructor
      .def_readwrite("voxel_size", &ConfigVHM::voxel_size)
      .def_readwrite("max_points_per_voxel", &ConfigVHM::max_points_per_voxel)
      .def_readwrite("local_map_radius", &ConfigVHM::local_map_radius)
      .def_readwrite("max_time_diff", &ConfigVHM::max_time_diff);

  // 2. Bind VoxelHashMap Class
  py::class_<VoxelHashMap>(m, "VoxelHashMap")
      .def(py::init<const ConfigVHM &>(), py::arg("cfg"))
      .def("update", &VoxelHashMap::update, py::arg("frame_id"),
           py::arg("points"), py::arg("rgb"), py::arg("descriptors"))
      .def("reset_local_map", &VoxelHashMap::resetLocalMap, py::arg("frame_id"),
           py::arg("sensor_pos")) // Note: Vec3 will be interpreted as a 3x1
                                  // NumPy array
      .def("get_local_map", &VoxelHashMap::getLocalMap)
      .def("count", &VoxelHashMap::count);

  // Optional: Document the module
  m.doc() = "pybind11 module for RGB-D Odometry pipeline.";
}