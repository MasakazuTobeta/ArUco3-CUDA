# SPDX-License-Identifier: Apache-2.0
#
# Accuracy evaluation tool. Runs the three routes (CPU baseline, hybrid, CUDA)
# against the ground truth of the synthetic corpus and reports precision, recall,
# and four-corner RMSE broken down by condition.
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
