# SPDX-License-Identifier: Apache-2.0
#
# Build-time tool that generates packed codewords from OpenCV's predefined Dictionary.
# The generated files are committed to the repository, and their agreement with
# OpenCV is verified by tests.
add_executable(aruco3cuda_dictgen main.cpp)

target_link_libraries(aruco3cuda_dictgen
  PRIVATE aruco3cuda::dictionary aruco3cuda::util aruco3cuda::warnings aruco3cuda::coverage ${OpenCV_LIBS})
target_include_directories(aruco3cuda_dictgen SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
