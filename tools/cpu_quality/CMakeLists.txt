# SPDX-License-Identifier: Apache-2.0
add_library(aruco3cuda_cpu_quality STATIC quality_scenes.cpp)
add_library(aruco3cuda::cpu_quality ALIAS aruco3cuda_cpu_quality)
target_include_directories(aruco3cuda_cpu_quality PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(aruco3cuda_cpu_quality SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(aruco3cuda_cpu_quality
  PUBLIC ${OpenCV_LIBS} PRIVATE aruco3cuda::warnings aruco3cuda::coverage)

add_executable(aruco3cuda_cpu_quality_corpus main.cpp)
target_link_libraries(aruco3cuda_cpu_quality_corpus
  PRIVATE aruco3cuda::cpu_quality aruco3cuda::util aruco3cuda::warnings aruco3cuda::coverage)
