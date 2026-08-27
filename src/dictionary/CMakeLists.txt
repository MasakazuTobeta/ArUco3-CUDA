# SPDX-License-Identifier: Apache-2.0
#
# 定義済み Dictionary の packed 表現と照合処理。
# generated/ 以下は tools/dictgen が生成した source であり手で編集しない。
add_library(aruco3cuda_dictionary
  dictionary.cpp
  registry.cpp
  generated/dict_aruco_mip_36h12.cpp)

add_library(aruco3cuda::dictionary ALIAS aruco3cuda_dictionary)

target_include_directories(aruco3cuda_dictionary
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(aruco3cuda_dictionary PRIVATE aruco3cuda::warnings)

set_target_properties(aruco3cuda_dictionary PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME aruco3cuda_dictionary)
