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
  test_threshold.cpp
  test_labeling.cpp
  test_quad_extract.cpp
  test_candidate_filter.cpp
  test_candidate_group.cpp
  test_candidate_comparison.cpp
  test_cell_sample.cpp
  test_cell_decode.cpp
  test_dictionary_match.cpp
  test_candidate_tree.cpp
  test_detection_emit.cpp
  test_corner_refine.cpp
  test_detector.cpp
  test_detector_probe.cu
  test_hybrid_detector.cpp
  test_device_image.cpp
  test_report_diff.cpp)

target_link_libraries(aruco3cuda_reference_tests
  PRIVATE
    aruco3cuda::reference
    aruco3cuda::corpusgen
    aruco3cuda::bench
    aruco3cuda::dictionary
    aruco3cuda::core
    aruco3cuda::hybrid
    aruco3cuda::report
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

# 時間を測る test は反復が多く、ctest を並列実行すると GPU の取り合いで
# 実行時間が伸びる。値そのものは単独実行でしか意味を持たないが、正しさの
# 検査は並列でも成立させたいため timeout を広く取る。
gtest_discover_tests(aruco3cuda_reference_tests
  DISCOVERY_TIMEOUT 60
  PROPERTIES TIMEOUT 900)

# ハイブリッド経路は GPU kernel を実行するため、Compute Sanitizer の対象へ
# 含める。unit test だけでは前処理と二値化の kernel が検査されない。
aruco3cuda_add_sanitizer_tests(aruco3cuda_reference_tests)

# test_corner_refine.cpp は cv::cornerSubPix を逐語で写した oracle を持つ。
# GPU 側は -fmad=false で積和を融合しないため、host 側も融合を止めないと
# 比較にならない。compiler は既定で融合してよいことになっている。
set_source_files_properties(test_corner_refine.cpp
  PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
