# SPDX-License-Identifier: Apache-2.0
#
# util は CUDA にも OpenCV にも依存しない共通処理を提供する。
# core、reference、tools のいずれからも利用できる。
add_library(aruco3cuda_util
  sha256.cpp
  json_writer.cpp
  statistics.cpp)

add_library(aruco3cuda::util ALIAS aruco3cuda_util)

target_include_directories(aruco3cuda_util
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(aruco3cuda_util PRIVATE aruco3cuda::warnings)

set_target_properties(aruco3cuda_util PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME aruco3cuda_util)
