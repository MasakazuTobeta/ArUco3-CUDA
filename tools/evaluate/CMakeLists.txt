# SPDX-License-Identifier: Apache-2.0
#
# 正確性評価 tool。合成 corpus の ground truth に対し CPU 基準、hybrid、CUDA の
# 3 経路を走らせ、precision と recall、四隅の RMSE を条件別に報告する。
add_library(aruco3cuda_evaluate STATIC accuracy.cpp)
add_library(aruco3cuda::evaluate ALIAS aruco3cuda_evaluate)

target_include_directories(aruco3cuda_evaluate PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(aruco3cuda_evaluate
  PUBLIC aruco3cuda::report
  PRIVATE aruco3cuda::warnings aruco3cuda::coverage)

add_executable(aruco3cuda_evaluate_cli main.cpp)
set_target_properties(aruco3cuda_evaluate_cli PROPERTIES OUTPUT_NAME aruco3cuda_evaluate)
target_link_libraries(aruco3cuda_evaluate_cli
  PRIVATE aruco3cuda::evaluate aruco3cuda::report aruco3cuda::corpusgen aruco3cuda::reference
          aruco3cuda::hybrid aruco3cuda::dictionary aruco3cuda::core aruco3cuda::warnings aruco3cuda::coverage
          CUDA::cudart ${OpenCV_LIBS})
target_include_directories(aruco3cuda_evaluate_cli SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
