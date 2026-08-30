# SPDX-License-Identifier: Apache-2.0
#
# util provides shared code that depends on neither CUDA nor OpenCV.
# It can be used from core, reference, and tools alike.
add_library(aruco3cuda_util
  sha256.cpp
  json_writer.cpp
  statistics.cpp)

add_library(aruco3cuda::util ALIAS aruco3cuda_util)

target_include_directories(aruco3cuda_util
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(aruco3cuda_util PRIVATE aruco3cuda::warnings aruco3cuda::coverage)

set_target_properties(aruco3cuda_util PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME aruco3cuda_util)
