# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.24)
project(apriltag_gpu_consumer LANGUAGES CXX)
find_package(aruco3cuda CONFIG REQUIRED)

# Reuse the real CLI as an external consumer. The imported target must select
# C++17 and provide its installed headers, including the dictionary dependency.
add_executable(apriltag_consumer ../../apriltag/gpu/detect.cpp ../../examples/pgm.cpp)
target_include_directories(apriltag_consumer PRIVATE ../../examples)
target_link_libraries(apriltag_consumer PRIVATE aruco3cuda::apriltag_gpu)
set_target_properties(apriltag_consumer PROPERTIES CXX_STANDARD 14 CXX_EXTENSIONS OFF
  NO_SYSTEM_FROM_IMPORTED ON)
target_compile_options(apriltag_consumer PRIVATE -pedantic-errors)
