# SPDX-License-Identifier: Apache-2.0
#
# 目的:
#   ArUco3-CUDA の build 基盤。対象 architecture、言語標準、警告、テストを定義する。
#
# 対象環境:
#   ADR-0002 に従い、Jetson Orin (sm_87) と DGX Spark GB10 (sm_121) を対象とする。
#   CUDA Toolkit は開発 container へ host から mount される。
cmake_minimum_required(VERSION 3.24)

# ADR-0002 の決定に従う既定値。project() より前に設定する必要がある。
# 単一機で開発する場合は CMakePresets.json の preset で絞り込む。
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "87;121" CACHE STRING "対象 CUDA architecture")
endif()

project(aruco3cuda
  VERSION 0.1.0
  DESCRIPTION "ArUco3 検出戦略の CUDA 実装"
  LANGUAGES CXX CUDA)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# host compiler で compile する翻訳単位からも CUDA runtime を参照するため、
# include path と library を CUDA::cudart 経由で取得する。
find_package(CUDAToolkit REQUIRED)

include(Aruco3CudaOptions)
include(Aruco3CudaSanitizer)

add_subdirectory(src)

if(ARUCO3CUDA_BUILD_REFERENCE)
  # CPU 基準実装の OpenCV。test 側からも同じ設定を参照するため top level で解決する。
  find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs objdetect)
  message(STATUS "  OpenCV             : ${OpenCV_VERSION} (${OpenCV_DIR})")
  add_subdirectory(reference)
  add_subdirectory(tools)
  add_subdirectory(bench)
endif()

if(ARUCO3CUDA_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

aruco3cuda_add_format_targets()
aruco3cuda_add_coverage_target()

message(STATUS "aruco3cuda ${PROJECT_VERSION}")
message(STATUS "  CUDA architectures : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  build tests        : ${ARUCO3CUDA_BUILD_TESTS}")
message(STATUS "  build reference    : ${ARUCO3CUDA_BUILD_REFERENCE}")
message(STATUS "  warnings as errors : ${ARUCO3CUDA_WARNINGS_AS_ERRORS}")
message(STATUS "  compute-sanitizer  : ${ARUCO3CUDA_ENABLE_COMPUTE_SANITIZER}")
message(STATUS "  coverage           : ${ARUCO3CUDA_ENABLE_COVERAGE}")
