# SPDX-License-Identifier: Apache-2.0
#
# OpenCV を必要とする検証。CPU 基準 runner、Dictionary 適合性、corpus 生成器を
# 同じ実行 file へまとめる。
add_executable(aruco3cuda_reference_tests
  test_reference_runner.cpp
  test_dictionary_conformance.cpp
  test_corpus_generator.cpp)

target_link_libraries(aruco3cuda_reference_tests
  PRIVATE
    aruco3cuda::reference
    aruco3cuda::corpusgen
    aruco3cuda::dictionary
    aruco3cuda::warnings
    GTest::gtest
    GTest::gtest_main
    ${OpenCV_LIBS})
target_include_directories(aruco3cuda_reference_tests SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})

gtest_discover_tests(aruco3cuda_reference_tests
  DISCOVERY_TIMEOUT 60
  PROPERTIES TIMEOUT 300)
