# SPDX-License-Identifier: Apache-2.0
# Experimental CPU detector. No CUDA objects or runtime libraries are linked.
add_library(aruco3cuda_cpu STATIC
  cpu_detector.cpp
  cpu_grid_refine.cpp
  cpu_box_sample.cpp
  cpu_edge_regions.cpp
  cpu_component_candidates.cpp
  cpu_grouping.cpp
  cpu_contours.cpp
  cpu_sparse_pyramid.cpp
  cpu_threshold.cpp
  ../hybrid/cpu_candidates.cpp
  ../src/core/config.cpp
  ../src/core/status.cpp)
add_library(aruco3cuda::cpu ALIAS aruco3cuda_cpu)
target_compile_features(aruco3cuda_cpu PUBLIC cxx_std_17)
target_include_directories(aruco3cuda_cpu
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include
  PRIVATE ${PROJECT_SOURCE_DIR}/hybrid)
target_include_directories(aruco3cuda_cpu SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(aruco3cuda_cpu
  PUBLIC ${OpenCV_LIBS}
  PRIVATE aruco3cuda::warnings aruco3cuda::coverage)

add_executable(aruco3cuda_cpu_benchmark benchmark.cpp)
target_link_libraries(aruco3cuda_cpu_benchmark
  PRIVATE aruco3cuda::cpu aruco3cuda::util aruco3cuda::corpusgen
          aruco3cuda::warnings aruco3cuda::coverage)

add_executable(aruco3cuda_apriltag_contour_sequence ../apriltag/benchmark/contour_sequence.cpp)
target_link_libraries(aruco3cuda_apriltag_contour_sequence
  PRIVATE aruco3cuda::cpu aruco3cuda::warnings aruco3cuda::coverage)

add_executable(aruco3cuda_cpu_resources resource_probe.cpp)
target_link_libraries(aruco3cuda_cpu_resources
  PRIVATE aruco3cuda::cpu aruco3cuda::util aruco3cuda::warnings aruco3cuda::coverage)

find_package(Threads REQUIRED)
add_executable(aruco3cuda_cpu_video video_benchmark.cpp)
target_link_libraries(aruco3cuda_cpu_video
  PRIVATE aruco3cuda::cpu aruco3cuda::util Threads::Threads
          aruco3cuda::warnings aruco3cuda::coverage)

# Architecture tuning is explicit and never changes floating-point semantics.
option(ARUCO3CPU_NATIVE "Tune the experimental CPU library for the build host" OFF)
set(aruco3cpu_native_flag "")
if(ARUCO3CPU_NATIVE)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    set(aruco3cpu_native_flag "-mcpu=native")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i.86)$")
    set(aruco3cpu_native_flag "-march=native")
  else()
    message(FATAL_ERROR "ARUCO3CPU_NATIVE is not supported on ${CMAKE_SYSTEM_PROCESSOR}")
  endif()
  target_compile_options(aruco3cuda_cpu PRIVATE "${aruco3cpu_native_flag}")
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" aruco3cpu_build_type)
target_compile_definitions(aruco3cuda_cpu_benchmark PRIVATE
  ARUCO3CPU_BUILD_FLAGS="${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${aruco3cpu_build_type}} ${aruco3cpu_native_flag}")
