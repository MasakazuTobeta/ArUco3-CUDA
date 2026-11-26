# SPDX-License-Identifier: Apache-2.0
add_executable(aruco3cuda_cpu_tests
  test_cpu_detector.cpp test_cpu_grid_refine.cpp test_cpu_threshold.cpp test_cpu_grouping.cpp
  test_cpu_accuracy_stress.cpp test_cpu_sparse_pyramid.cpp test_cpu_contours.cpp
  test_cpu_video_detector.cpp)
target_link_libraries(aruco3cuda_cpu_tests
  PRIVATE aruco3cuda::cpu aruco3cuda::corpusgen aruco3cuda::cpu_quality
    aruco3cuda::warnings aruco3cuda::coverage
    GTest::gtest GTest::gtest_main ${OpenCV_LIBS})
target_include_directories(aruco3cuda_cpu_tests SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
target_include_directories(aruco3cuda_cpu_tests PRIVATE ${PROJECT_SOURCE_DIR}/hybrid)
include(GoogleTest)
gtest_discover_tests(aruco3cuda_cpu_tests
  DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 600 LABELS cpu)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME cpu.measurement_tools
  COMMAND "${Python3_EXECUTABLE}" -m unittest discover
    -s "${CMAKE_CURRENT_SOURCE_DIR}" -p "test_*experiments.py" -v)
set_tests_properties(cpu.measurement_tools PROPERTIES TIMEOUT 120 LABELS "cpu;python")
