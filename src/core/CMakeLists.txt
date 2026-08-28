# SPDX-License-Identifier: Apache-2.0
#
# core は CUDA による検出処理を担い、OpenCV 型へ依存しない。
# 現時点では build 基盤の疎通確認に必要な最小構成のみを含む。
add_library(aruco3cuda_core
  status.cpp
  types.cpp
  config.cpp
  workspace.cpp
  preprocess.cu
  threshold.cu
  scan.cu
  labeling.cu
  quad_extract.cu
  candidate_filter.cu
  candidate_group.cu
  cell_sample.cu
  cuda_check.cpp
  device_probe.cu)

add_library(aruco3cuda::core ALIAS aruco3cuda_core)

target_include_directories(aruco3cuda_core
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})

# CUDA::cudart は include path も伝播する。cuda_check.cpp のように
# host compiler で compile する翻訳単位でも CUDA runtime の header を使う。
target_link_libraries(aruco3cuda_core PRIVATE aruco3cuda::warnings aruco3cuda::coverage CUDA::cudart)

set_target_properties(aruco3cuda_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME aruco3cuda)

# cell_sample.cu は OpenCV の warpPerspective と丸めまで一致させる。積和を
# 融合すると 1 ULP ずれ、丸め境界で参照画素が 1 つずれる。この翻訳単位だけ
# 融合を止める。
set_source_files_properties(cell_sample.cu PROPERTIES COMPILE_OPTIONS "-fmad=false")
