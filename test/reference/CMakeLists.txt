# SPDX-License-Identifier: Apache-2.0
#
# CPU 基準 runner の検証。合成マーカー画像を test 内で生成し、
# 検出結果と ArUco3 の縮小率が期待どおりであることを確認する。
add_executable(aruco3cuda_reference_tests test_reference_runner.cpp)

target_link_libraries(aruco3cuda_reference_tests
  PRIVATE aruco3cuda::reference aruco3cuda::warnings GTest::gtest GTest::gtest_main ${OpenCV_LIBS})
target_include_directories(aruco3cuda_reference_tests SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})

gtest_discover_tests(aruco3cuda_reference_tests
  DISCOVERY_TIMEOUT 60
  PROPERTIES TIMEOUT 120)
