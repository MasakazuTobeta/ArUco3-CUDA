# SPDX-License-Identifier: Apache-2.0
#
# tools holds the helper executables that require OpenCV.
# core and dictionary never depend on tools.
add_subdirectory(dictgen)
add_subdirectory(corpusgen)
add_subdirectory(report)
add_subdirectory(evaluate)
add_subdirectory(contourcount)
if(ARUCO3CUDA_BUILD_CPU_EXPERIMENT)
  add_subdirectory(gpu_quality)
endif()
