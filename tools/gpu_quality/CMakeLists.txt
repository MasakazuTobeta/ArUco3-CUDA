# SPDX-License-Identifier: Apache-2.0
# Correctness audit across official OpenCV, optimized CPU and device detection.
add_executable(aruco3cuda_gpu_quality_audit audit.cpp)
target_link_libraries(aruco3cuda_gpu_quality_audit PRIVATE
  aruco3cuda::cpu aruco3cuda::cpu_quality aruco3cuda::core
  aruco3cuda::dictionary aruco3cuda::hybrid aruco3cuda::evaluate
  aruco3cuda::report aruco3cuda::util aruco3cuda::warnings
  aruco3cuda::coverage CUDA::cudart ${OpenCV_LIBS})
target_include_directories(aruco3cuda_gpu_quality_audit SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})

add_executable(aruco3cuda_cell_decode_probe cell_decode_probe.cpp)
target_include_directories(aruco3cuda_cell_decode_probe PRIVATE ${PROJECT_SOURCE_DIR}/src/core)
target_link_libraries(aruco3cuda_cell_decode_probe PRIVATE
  aruco3cuda::core aruco3cuda::util aruco3cuda::warnings)
