# SPDX-License-Identifier: Apache-2.0
#
# OpenCV を必要とする検証。CPU 基準 runner、Dictionary 適合性、corpus 生成器を
# 同じ実行 file へまとめる。
add_executable(aruco3cuda_reference_tests
  test_reference_runner.cpp
  test_dictionary_conformance.cpp
  test_corpus_generator.cpp
  test_benchmark_harness.cpp
  test_preprocess.cpp
  test_threshold.cpp)

target_link_libraries(aruco3cuda_reference_tests
  PRIVATE
    aruco3cuda::reference
    aruco3cuda::corpusgen
    aruco3cuda::bench
    aruco3cuda::dictionary
    aruco3cuda::core
    aruco3cuda::warnings
    aruco3cuda::coverage
    GTest::gtest
    GTest::gtest_main
    ${OpenCV_LIBS})
# test_preprocess.cpp は core の内部 header を使う。前処理は公開 API では
# なく、段階ごとの検証には内部の型が必要になる。
target_include_directories(aruco3cuda_reference_tests PRIVATE ${PROJECT_SOURCE_DIR}/src/core)
target_link_libraries(aruco3cuda_reference_tests PRIVATE CUDA::cudart)
target_include_directories(aruco3cuda_reference_tests SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})

gtest_discover_tests(aruco3cuda_reference_tests
  DISCOVERY_TIMEOUT 60
  PROPERTIES TIMEOUT 300)
