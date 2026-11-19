# SPDX-License-Identifier: Apache-2.0
add_library(aruco3cuda_apriltag_gpu detector.cu)
target_include_directories(aruco3cuda_apriltag_gpu PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(aruco3cuda_apriltag_gpu PUBLIC aruco3cuda::dictionary CUDA::cudart)
add_library(aruco3cuda::apriltag_gpu ALIAS aruco3cuda_apriltag_gpu)

add_executable(aruco3cuda_apriltag_gpu_detect detect.cpp ../../examples/pgm.cpp)
target_include_directories(aruco3cuda_apriltag_gpu_detect PRIVATE ../../examples)
target_link_libraries(aruco3cuda_apriltag_gpu_detect PRIVATE aruco3cuda::apriltag_gpu)

if(ARUCO3CUDA_BUILD_TESTS)
  add_executable(aruco3cuda_apriltag_gpu_test test_detector.cpp)
  target_link_libraries(aruco3cuda_apriltag_gpu_test PRIVATE aruco3cuda::apriltag_gpu)
  add_test(NAME AprilTagGpu.synthetic COMMAND aruco3cuda_apriltag_gpu_test)
  set_tests_properties(AprilTagGpu.synthetic PROPERTIES SKIP_RETURN_CODE 77)
  find_package(Python3 QUIET COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    add_test(NAME AprilTagGpu.cli COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/test_cli.py" $<TARGET_FILE:aruco3cuda_apriltag_gpu_detect>)
    set_tests_properties(AprilTagGpu.cli PROPERTIES SKIP_RETURN_CODE 77)
  endif()
endif()
