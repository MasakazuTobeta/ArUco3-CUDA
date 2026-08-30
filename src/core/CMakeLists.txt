# SPDX-License-Identifier: Apache-2.0
#
# core carries the CUDA detection work and does not depend on OpenCV types.
# It covers every detection stage: preprocessing, thresholding, candidate
# extraction, decoding, dictionary matching, and corner refinement.
add_library(aruco3cuda_core
  status.cpp
  types.cpp
  config.cpp
  workspace.cpp
  preprocess.cu
  threshold.cu
  scan.cu
  labeling.cu
  quad_extract.cu
  candidate_filter.cu
  candidate_group.cu
  cell_sample.cu
  cell_decode.cu
  dictionary_match.cu
  candidate_tree.cu
  detection_emit.cu
  corner_refine.cu
  detector.cpp
  cuda_check.cpp
  device_probe.cu)

add_library(aruco3cuda::core ALIAS aruco3cuda_core)

target_include_directories(aruco3cuda_core
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})

# CUDA::cudart propagates the include path as well. Translation units compiled
# by the host compiler, such as cuda_check.cpp, also use CUDA runtime headers.
# The public header detector.hpp includes cuda_runtime_api.h, so consumers need
# the include path too, which is why this is PUBLIC.
target_link_libraries(aruco3cuda_core
  PRIVATE aruco3cuda::warnings aruco3cuda::coverage
  PUBLIC CUDA::cudart)

set_target_properties(aruco3cuda_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME aruco3cuda)

# cell_sample.cu and cell_decode.cu must match OpenCV down to the rounding.
# Fusing multiply-add shifts the result by 1 ULP, which changes which pixel is
# sampled and how the variance threshold compares. Disable fusion in these
# translation units only.
set_source_files_properties(cell_sample.cu cell_decode.cu corner_refine.cu
  PROPERTIES COMPILE_OPTIONS "-fmad=false")
