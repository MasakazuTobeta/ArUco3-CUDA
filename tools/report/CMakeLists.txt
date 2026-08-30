# SPDX-License-Identifier: Apache-2.0
#
# Difference report tool. Compares the detections of the CPU baseline against the
# route under evaluation and classifies each difference as a miss, a false positive,
# an ID mismatch, a rotation mismatch, or a corner displacement.
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
