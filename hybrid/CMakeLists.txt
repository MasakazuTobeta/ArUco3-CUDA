# SPDX-License-Identifier: Apache-2.0
#
# 案 C のハイブリッド経路。GPU で前処理と二値化を行い、候補抽出と decode を
# CPU で行う。GPU 側の段階を 1 つずつ置き換える際の基準であり、
# 案 A が期待どおりに動かない場合の fallback でもある。
add_library(aruco3cuda_hybrid STATIC hybrid_detector.cpp cpu_candidates.cpp device_image.cpp)
add_library(aruco3cuda::hybrid ALIAS aruco3cuda_hybrid)

target_include_directories(aruco3cuda_hybrid
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
  # core の内部 header を使う。前処理と二値化は公開 API ではない。
  PRIVATE ${PROJECT_SOURCE_DIR}/src/core)
target_link_libraries(aruco3cuda_hybrid
  PUBLIC aruco3cuda::core
  PRIVATE aruco3cuda::warnings aruco3cuda::coverage ${OpenCV_LIBS} CUDA::cudart)
target_include_directories(aruco3cuda_hybrid SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
