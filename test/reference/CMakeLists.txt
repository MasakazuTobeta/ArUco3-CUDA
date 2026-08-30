# SPDX-License-Identifier: Apache-2.0
#
# Verification that requires OpenCV. The CPU baseline runner, dictionary
# conformance, and the corpus generator are collected into one executable.
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
  test_report_diff.cpp
  test_accuracy.cpp)

target_link_libraries(aruco3cuda_reference_tests
  PRIVATE
    aruco3cuda::reference
    aruco3cuda::corpusgen
    aruco3cuda::bench
    aruco3cuda::dictionary
    aruco3cuda::core
    aruco3cuda::hybrid
    aruco3cuda::report
    aruco3cuda::evaluate
    aruco3cuda::warnings
    aruco3cuda::coverage
    GTest::gtest
    GTest::gtest_main
    ${OpenCV_LIBS})
# test_preprocess.cpp uses the internal headers of core. Preprocessing is not
# part of the public API, and verifying it stage by stage needs the internal
# types.
target_include_directories(aruco3cuda_reference_tests PRIVATE ${PROJECT_SOURCE_DIR}/src/core)
target_link_libraries(aruco3cuda_reference_tests PRIVATE CUDA::cudart)
target_include_directories(aruco3cuda_reference_tests SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})

# Timing tests iterate a lot, and running ctest in parallel stretches their run
# time through GPU contention. The measured values are only meaningful in a
# standalone run, but the correctness checks should still hold when run in
# parallel, so the timeout is generous.
gtest_discover_tests(aruco3cuda_reference_tests
  DISCOVERY_TIMEOUT 60
  PROPERTIES TIMEOUT 900)

# The hybrid path runs GPU kernels, so include it in the Compute Sanitizer
# scope. The unit tests alone never exercise the preprocessing and thresholding
# kernels.
aruco3cuda_add_sanitizer_tests(aruco3cuda_reference_tests)

# test_corner_refine.cpp holds an oracle transcribed verbatim from
# cv::cornerSubPix. The GPU side does not fuse multiply-add because of
# -fmad=false, so unless the host side stops fusing as well the two are not
# comparable. Compilers are allowed to fuse by default.
set_source_files_properties(test_corner_refine.cpp
  PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
