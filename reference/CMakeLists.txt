# SPDX-License-Identifier: Apache-2.0
#
# CPU 基準 runner。OpenCV の ArUco 検出を固定設定で実行し結果を JSON で保存する。
# core とは独立した target とし、core へ OpenCV 依存を持ち込まない。
add_library(aruco3cuda_reference STATIC reference_runner.cpp)
add_library(aruco3cuda::reference ALIAS aruco3cuda_reference)

target_include_directories(aruco3cuda_reference PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(aruco3cuda_reference
  PUBLIC aruco3cuda::util
  PRIVATE aruco3cuda::warnings ${OpenCV_LIBS})
target_include_directories(aruco3cuda_reference SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})

add_executable(aruco3cuda_reference_runner main.cpp)
target_link_libraries(aruco3cuda_reference_runner
  PRIVATE aruco3cuda::reference aruco3cuda::warnings)

