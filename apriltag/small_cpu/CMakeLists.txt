# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.24)
project(aruco3_apriltag_small_cpu LANGUAGES CXX)

find_package(OpenCV 4.14.0 EXACT CONFIG REQUIRED COMPONENTS core imgproc)

get_filename_component(repository_root "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
add_library(aruco3_apriltag_small_cpu STATIC
  detector.cpp
  video_detector.cpp
  motion_regions.cpp
  grid_alignment.cpp
  kalibr_grid_search.cpp
  isolated_ink.cpp
  noisy_ink.cpp
  noisy_print_search.cpp
  guide_average.cpp
  curved_edges.cpp
  dense_board_search.cpp
  entering_strip_search.cpp
  large_board_search.cpp
  automatic_search.cpp
  border_search.cpp
  coarse_mask.cpp
  component_candidates.cpp
  geometry.cpp
  small_tag_refinement.cpp
  "${repository_root}/cpu/cpu_grid_refine.cpp"
  "${repository_root}/src/dictionary/generated/dict_apriltag_36h11.cpp")
add_library(aruco3cuda::apriltag_small_cpu ALIAS aruco3_apriltag_small_cpu)
target_compile_features(aruco3_apriltag_small_cpu PUBLIC cxx_std_17)
target_include_directories(aruco3_apriltag_small_cpu
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE "${repository_root}/include" "${repository_root}/cpu")
target_link_libraries(aruco3_apriltag_small_cpu PUBLIC opencv_core PRIVATE opencv_imgproc)

if(PROJECT_IS_TOP_LEVEL)
  include(CTest)
endif()
if(BUILD_TESTING)
  add_executable(aruco3_apriltag_small_cpu_test detector_test.cpp)
  target_include_directories(aruco3_apriltag_small_cpu_test PRIVATE "${repository_root}/include" "${repository_root}/cpu")
  target_link_libraries(aruco3_apriltag_small_cpu_test PRIVATE aruco3_apriltag_small_cpu)
  add_test(NAME aruco3_apriltag_small_cpu COMMAND aruco3_apriltag_small_cpu_test)
  add_executable(aruco3_apriltag_isolated_ink_test isolated_ink_test.cpp)
  target_include_directories(aruco3_apriltag_isolated_ink_test PRIVATE "${repository_root}/cpu")
  target_link_libraries(aruco3_apriltag_isolated_ink_test PRIVATE aruco3_apriltag_small_cpu)
  add_test(NAME aruco3_apriltag_isolated_ink COMMAND aruco3_apriltag_isolated_ink_test)
  add_executable(aruco3_apriltag_motion_regions_test motion_regions_test.cpp)
  target_link_libraries(aruco3_apriltag_motion_regions_test PRIVATE aruco3_apriltag_small_cpu opencv_imgproc)
  add_test(NAME aruco3_apriltag_motion_regions COMMAND aruco3_apriltag_motion_regions_test)
endif()

option(ARUCO3_APRILTAG_SMALL_BENCHMARK "Build the host sequence benchmark" OFF)
if(ARUCO3_APRILTAG_SMALL_BENCHMARK)
  find_package(OpenCV 4.14.0 EXACT CONFIG REQUIRED COMPONENTS imgcodecs objdetect)
  add_executable(aruco3cuda_apriltag_opencv_sequence ../benchmark/opencv_sequence.cpp)
  target_compile_features(aruco3cuda_apriltag_opencv_sequence PRIVATE cxx_std_17)
  target_link_libraries(aruco3cuda_apriltag_opencv_sequence PRIVATE opencv_imgcodecs opencv_objdetect)
  add_executable(aruco3cuda_apriltag_small_sequence ../benchmark/small_sequence.cpp)
  target_link_libraries(aruco3cuda_apriltag_small_sequence
    PRIVATE aruco3_apriltag_small_cpu opencv_imgcodecs)
endif()
