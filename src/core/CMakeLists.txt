# SPDX-License-Identifier: Apache-2.0
#
# core carries the CUDA detection work and does not depend on OpenCV types.
# It covers every detection stage: preprocessing, thresholding, candidate
# extraction, decoding, dictionary matching, and corner refinement.
add_library(aruco3cuda_core
  status.cpp
  types.cpp
  config.cpp
  resize_profile.cpp
  build_profile.cpp
  workspace.cpp
  preprocess.cu
  threshold.cu
  scan.cu
  labeling.cu
  quad_extract.cu
  candidate_filter.cu
  contour_candidates.cu
  candidate_group.cu
  candidate_retry.cu
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

include(ArUco3CudaResizeProfile)
target_compile_definitions(aruco3cuda_core PRIVATE
  ARUCO3CUDA_RESIZE_PROFILE_OPENCV_ARM_HAL=${ARUCO3CUDA_RESIZE_PROFILE_OPENCV_ARM_HAL_VALUE}
  ARUCO3CUDA_RESIZE_ARM_SVE_BYTES=${ARUCO3CUDA_RESIZE_ARM_SVE_BYTES_VALUE}
  ARUCO3CUDA_CAROTENE_X_FMA=${ARUCO3CUDA_CAROTENE_X_FMA_VALUE})

include(ArUco3CudaStrategies)
target_compile_definitions(aruco3cuda_core PRIVATE
  ARUCO3CUDA_EXPERIMENT_CONTOUR_CANDIDATES=${ARUCO3CUDA_CONTOUR_STRATEGY_VALUE}
  ARUCO3CUDA_ENABLE_FUSED_THRESHOLD=${ARUCO3CUDA_THRESHOLD_STRATEGY_VALUE}
  ARUCO3CUDA_EXPERIMENT_DEMAND_PYRAMID=${ARUCO3CUDA_PYRAMID_STRATEGY_VALUE}
  ARUCO3CUDA_EXPERIMENT_OTSU_FIXED_POINT=${ARUCO3CUDA_OTSU_STRATEGY_VALUE}
  ARUCO3CUDA_EXPERIMENT_PERSISTENT_CANDIDATES=${ARUCO3CUDA_CANDIDATE_GRID_STRATEGY_VALUE}
  ARUCO3CUDA_OTSU_FIXED_POINT_MAX_OCCUPIED_BINS=${ARUCO3CUDA_OTSU_MAX_OCCUPIED_BINS}
  ARUCO3CUDA_CANDIDATE_GRID_BLOCKS=${ARUCO3CUDA_CANDIDATE_GRID_BLOCKS}
  ARUCO3CUDA_CONTOUR_STRATEGY_NAME="${ARUCO3CUDA_CONTOUR_STRATEGY}"
  ARUCO3CUDA_THRESHOLD_STRATEGY_NAME="${ARUCO3CUDA_THRESHOLD_STRATEGY}"
  ARUCO3CUDA_PYRAMID_STRATEGY_NAME="${ARUCO3CUDA_PYRAMID_STRATEGY}"
  ARUCO3CUDA_OTSU_STRATEGY_NAME="${ARUCO3CUDA_OTSU_STRATEGY}"
  ARUCO3CUDA_CANDIDATE_GRID_STRATEGY_NAME="${ARUCO3CUDA_CANDIDATE_GRID_STRATEGY}"
  ARUCO3CUDA_EXPERIMENT_PARALLEL_CONTOUR_COUNT=${ARUCO3CUDA_CONTOUR_COUNT_STRATEGY_VALUE}
  ARUCO3CUDA_CONTOUR_COUNT_STRATEGY_NAME="${ARUCO3CUDA_CONTOUR_COUNT_STRATEGY}"
  ARUCO3CUDA_EXPERIMENT_COOPERATIVE_RDP=${ARUCO3CUDA_CONTOUR_RDP_STRATEGY_VALUE}
  ARUCO3CUDA_CONTOUR_RDP_STRATEGY_NAME="${ARUCO3CUDA_CONTOUR_RDP_STRATEGY}"
  ARUCO3CUDA_EXPERIMENT_COOPERATIVE_MATERIALIZE=${ARUCO3CUDA_CONTOUR_MATERIALIZE_STRATEGY_VALUE}
  ARUCO3CUDA_CONTOUR_MATERIALIZE_STRATEGY_NAME="${ARUCO3CUDA_CONTOUR_MATERIALIZE_STRATEGY}"
  ARUCO3CUDA_EXPERIMENT_COMPACT_MATERIALIZE_EDGES=${ARUCO3CUDA_CONTOUR_MATERIALIZE_COMPACTION_STRATEGY_VALUE}
  ARUCO3CUDA_CONTOUR_MATERIALIZE_COMPACTION_STRATEGY_NAME="${ARUCO3CUDA_CONTOUR_MATERIALIZE_COMPACTION_STRATEGY}"
  ARUCO3CUDA_PRECOMPUTED_RESIZE_COEFFICIENTS=${ARUCO3CUDA_RESIZE_COEFFICIENT_STRATEGY_VALUE}
  ARUCO3CUDA_RESIZE_COEFFICIENT_STRATEGY_NAME="${ARUCO3CUDA_RESIZE_COEFFICIENT_STRATEGY}")

target_include_directories(aruco3cuda_core
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
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
# translation units only. Contour approximation also preserves the reference's
# separate operations at polygon split and cleanup boundaries. Candidate grouping
# likewise preserves the reference's separate float distance operations.
set_source_files_properties(cell_sample.cu cell_decode.cu corner_refine.cu contour_candidates.cu
  candidate_group.cu candidate_retry.cu
  PROPERTIES COMPILE_OPTIONS "-fmad=false")
