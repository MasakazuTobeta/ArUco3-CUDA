# SPDX-License-Identifier: Apache-2.0
#
# Hybrid path of option C. Preprocessing and thresholding run on the GPU while
# candidate extraction and decoding run on the CPU. It is the baseline used while
# replacing the GPU stages one at a time, and also the fallback should option A
# fail to work as expected.
add_library(aruco3cuda_hybrid STATIC hybrid_detector.cpp cpu_candidates.cpp device_image.cpp)
add_library(aruco3cuda::hybrid ALIAS aruco3cuda_hybrid)

target_include_directories(aruco3cuda_hybrid
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
  # Uses the internal headers of core. Preprocessing and thresholding are not
  # part of the public API.
  PRIVATE ${PROJECT_SOURCE_DIR}/src/core)
target_link_libraries(aruco3cuda_hybrid
  PUBLIC aruco3cuda::core
  PRIVATE aruco3cuda::warnings aruco3cuda::coverage ${OpenCV_LIBS} CUDA::cudart)
target_include_directories(aruco3cuda_hybrid SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
