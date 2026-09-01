# SPDX-License-Identifier: Apache-2.0
find_package(GTest REQUIRED)

add_executable(aruco3cuda_unit_tests
  unit/test_status.cpp
  unit/test_device_probe.cpp
  unit/test_sha256.cpp
  unit/test_json_writer.cpp
  unit/test_dictionary.cpp
  unit/test_statistics.cpp
  unit/test_cuda_check.cpp
  unit/test_cuda_check_launch.cu
  unit/test_types.cpp
  unit/test_config.cpp
  unit/test_workspace.cpp)

target_link_libraries(aruco3cuda_unit_tests
  PRIVATE aruco3cuda::core aruco3cuda::util aruco3cuda::dictionary aruco3cuda::warnings aruco3cuda::coverage GTest::gtest GTest::gtest_main)

# test_cuda_check.cpp uses core's internal headers. CUDA error context only
# appears on failure paths, and those cannot be triggered deliberately through
# the public API. test_cuda_check_launch.cu is CUDA rather than C++ because the
# failure it provokes has to come from a real kernel launch; it inherits
# CMAKE_CUDA_ARCHITECTURES like every other translation unit here.
target_include_directories(aruco3cuda_unit_tests
  PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src/core)
target_link_libraries(aruco3cuda_unit_tests PRIVATE CUDA::cudart)

include(GoogleTest)
# Initializing the CUDA context is slow, so allow a generous discovery timeout.
gtest_discover_tests(aruco3cuda_unit_tests
  DISCOVERY_TIMEOUT 60
  PROPERTIES TIMEOUT 120)

aruco3cuda_add_sanitizer_tests(aruco3cuda_unit_tests)

if(ARUCO3CUDA_BUILD_REFERENCE)
  add_subdirectory(reference)
  add_subdirectory(cli)
endif()

# The Python tests need neither OpenCV nor the reference implementation, so they
# are registered independently of ARUCO3CUDA_BUILD_REFERENCE.
if(ARUCO3CUDA_BUILD_PYTHON)
  add_subdirectory(python)
endif()
