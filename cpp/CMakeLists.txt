cmake_minimum_required(VERSION 3.14)
project(rgbd_odometry LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(third_party/Eigen)
add_subdirectory(third_party/pybind11)

add_library(vhm SHARED src/VoxelHashMap.cpp)
target_include_directories(vhm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(vhm PUBLIC Eigen3::Eigen)

pybind11_add_module(rgbd_odometry src/Bindings.cpp)
target_include_directories(rgbd_odometry
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(rgbd_odometry PRIVATE vhm Eigen3::Eigen)
