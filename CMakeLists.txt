# SPDX-License-Identifier: Apache-2.0
#
# Purpose:
#   Build foundation for ArUco3-CUDA. Defines the target architectures, language
#   standards, warnings, and tests.
#
# Target environment:
#   Following ADR-0002, the targets are Jetson Orin (sm_87) and DGX Spark GB10
#   (sm_121). The CUDA Toolkit is mounted into the development container from the
#   host.
cmake_minimum_required(VERSION 3.24)

# Default that follows the ADR-0002 decision. It must be set before project().
# When developing on a single machine, narrow it down with a preset in
# CMakePresets.json.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "87;121" CACHE STRING "Target CUDA architectures")
endif()

project(aruco3cuda
  VERSION 0.1.0
  DESCRIPTION "CUDA implementation of the ArUco3 detection strategy"
  LANGUAGES CXX CUDA)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# Translation units compiled by the host compiler also reference the CUDA
# runtime, so obtain the include path and libraries through CUDA::cudart.
find_package(CUDAToolkit REQUIRED)

include(Aruco3CudaOptions)
include(Aruco3CudaSanitizer)
include(Aruco3CudaInstall)

add_subdirectory(src)

if(ARUCO3CUDA_BUILD_REFERENCE)
  # OpenCV for the CPU baseline implementation. Resolve it at the top level so
  # the tests refer to the same configuration.
  find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs objdetect)
  message(STATUS "  OpenCV             : ${OpenCV_VERSION} (${OpenCV_DIR})")
  add_subdirectory(reference)
  add_subdirectory(hybrid)
  add_subdirectory(tools)
  add_subdirectory(bench)
endif()

if(ARUCO3CUDA_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

aruco3cuda_add_install_rules()
aruco3cuda_add_format_targets()
aruco3cuda_add_coverage_target()

message(STATUS "aruco3cuda ${PROJECT_VERSION}")
message(STATUS "  CUDA architectures : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  build tests        : ${ARUCO3CUDA_BUILD_TESTS}")
message(STATUS "  build reference    : ${ARUCO3CUDA_BUILD_REFERENCE}")
message(STATUS "  warnings as errors : ${ARUCO3CUDA_WARNINGS_AS_ERRORS}")
message(STATUS "  compute-sanitizer  : ${ARUCO3CUDA_ENABLE_COMPUTE_SANITIZER}")
message(STATUS "  coverage           : ${ARUCO3CUDA_ENABLE_COVERAGE}")
