# SPDX-License-Identifier: Apache-2.0
#
# 測定 harness。測定条件と環境情報を結果と一体で残す。
# 実装があるのは CPU 経路と Hybrid 経路で、完全 GPU 経路は Phase 3 以降に追加する。
add_library(aruco3cuda_bench STATIC benchmark_harness.cpp)
add_library(aruco3cuda::bench ALIAS aruco3cuda_bench)

target_include_directories(aruco3cuda_bench PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(aruco3cuda_bench
  PUBLIC aruco3cuda::reference aruco3cuda::util aruco3cuda::core
  PRIVATE aruco3cuda::hybrid aruco3cuda::warnings aruco3cuda::coverage ${OpenCV_LIBS})
target_include_directories(aruco3cuda_bench SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})

add_executable(aruco3cuda_bench_cli main.cpp)
set_target_properties(aruco3cuda_bench_cli PROPERTIES OUTPUT_NAME aruco3cuda_bench)
target_link_libraries(aruco3cuda_bench_cli PRIVATE aruco3cuda::bench aruco3cuda::warnings aruco3cuda::coverage)
