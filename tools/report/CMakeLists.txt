# SPDX-License-Identifier: Apache-2.0
#
# 差分レポート tool。CPU 基準と評価対象の検出結果を突き合わせ、差異を
# 未検出・過検出・ID 不一致・rotation 不一致・四隅ずれへ分類する。
add_library(aruco3cuda_report STATIC report_diff.cpp)
add_library(aruco3cuda::report ALIAS aruco3cuda_report)

target_include_directories(aruco3cuda_report PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(aruco3cuda_report
  PUBLIC aruco3cuda::util
  PRIVATE aruco3cuda::warnings aruco3cuda::coverage)

add_executable(aruco3cuda_report_cli main.cpp)
set_target_properties(aruco3cuda_report_cli PROPERTIES OUTPUT_NAME aruco3cuda_report)
target_link_libraries(aruco3cuda_report_cli
  PRIVATE aruco3cuda::report aruco3cuda::reference aruco3cuda::hybrid aruco3cuda::core
          aruco3cuda::warnings aruco3cuda::coverage CUDA::cudart ${OpenCV_LIBS})
target_include_directories(aruco3cuda_report_cli SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
