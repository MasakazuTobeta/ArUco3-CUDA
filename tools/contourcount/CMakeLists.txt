# SPDX-License-Identifier: Apache-2.0
#
# Contour point counter. Implements the procedure the benchmark report uses to
# derive the quantity that sets the CPU/GPU boundary, so that the regression
# input can be regenerated from the corpus and the published coefficients can be
# checked rather than taken on trust.
add_executable(aruco3cuda_contourcount_cli main.cpp)
set_target_properties(aruco3cuda_contourcount_cli PROPERTIES OUTPUT_NAME aruco3cuda_contourcount)
target_link_libraries(aruco3cuda_contourcount_cli
  PRIVATE aruco3cuda::core aruco3cuda::warnings aruco3cuda::coverage ${OpenCV_LIBS})
target_include_directories(aruco3cuda_contourcount_cli SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
