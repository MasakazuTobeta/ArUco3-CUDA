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
  unit/test_types.cpp
  unit/test_config.cpp
  unit/test_workspace.cpp)

target_link_libraries(aruco3cuda_unit_tests
  PRIVATE aruco3cuda::core aruco3cuda::util aruco3cuda::dictionary aruco3cuda::warnings aruco3cuda::coverage GTest::gtest GTest::gtest_main)

# test_cuda_check.cpp は core の内部 header を使う。CUDA のエラー文脈は
# 失敗経路でしか現れず、公開 API 経由では意図的に発生させられないため。
target_include_directories(aruco3cuda_unit_tests
  PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src/core)
target_link_libraries(aruco3cuda_unit_tests PRIVATE CUDA::cudart)

include(GoogleTest)
# CUDA context の初期化に時間がかかるため、探索時の timeout を広めに取る。
gtest_discover_tests(aruco3cuda_unit_tests
  DISCOVERY_TIMEOUT 60
  PROPERTIES TIMEOUT 120)

aruco3cuda_add_sanitizer_tests(aruco3cuda_unit_tests)

if(ARUCO3CUDA_BUILD_REFERENCE)
  add_subdirectory(reference)
  add_subdirectory(cli)
endif()
