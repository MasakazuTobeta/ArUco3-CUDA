# SPDX-License-Identifier: Apache-2.0
#
# OpenCV の定義済み Dictionary から packed codeword を生成する build 時 tool。
# 生成物は repository へ commit し、OpenCV との一致は test で検証する。
add_executable(aruco3cuda_dictgen main.cpp)

target_link_libraries(aruco3cuda_dictgen
  PRIVATE aruco3cuda::dictionary aruco3cuda::util aruco3cuda::warnings aruco3cuda::coverage ${OpenCV_LIBS})
target_include_directories(aruco3cuda_dictgen SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
