# SPDX-License-Identifier: Apache-2.0
#
# CPU baseline runner. Runs OpenCV ArUco detection with a fixed configuration
# and stores the results as JSON.
# Kept as a target independent of core so that no OpenCV dependency leaks into core.
add_library(aruco3cuda_reference STATIC reference_runner.cpp)
add_library(aruco3cuda::reference ALIAS aruco3cuda_reference)

target_include_directories(aruco3cuda_reference PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(aruco3cuda_reference
  PUBLIC aruco3cuda::util
  PRIVATE aruco3cuda::warnings aruco3cuda::coverage ${OpenCV_LIBS})
target_include_directories(aruco3cuda_reference SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})

add_executable(aruco3cuda_reference_runner main.cpp)
target_link_libraries(aruco3cuda_reference_runner
  PRIVATE aruco3cuda::reference aruco3cuda::warnings aruco3cuda::coverage)

