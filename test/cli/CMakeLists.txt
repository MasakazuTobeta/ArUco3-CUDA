# SPDX-License-Identifier: Apache-2.0
#
# Automated CLI tests.
#
# Purpose:
#   The CLI is the first boundary a user touches, and mistakes in argument
#   parsing do not surface until run time. Nominal, error, and boundary cases
#   are run from ctest and judged by exit code and output.
#
# Note:
#   The executables run as separate processes, but gcov writes its counters out
#   at process exit, so this path is reflected in coverage as well.
set(kCliCorpusDir "${CMAKE_CURRENT_BINARY_DIR}/cli-corpus")

# Fixture for tests that need images. It doubles as the nominal case for corpusgen itself.
add_test(NAME cli.corpusgen.setup
  COMMAND $<TARGET_FILE:aruco3cuda_corpusgen_cli>
          --preset smoke --seed 20260827
          --output-dir "${kCliCorpusDir}"
          --manifest "${kCliCorpusDir}/manifest.json")
set_tests_properties(cli.corpusgen.setup PROPERTIES FIXTURES_SETUP cli_corpus TIMEOUT 300)

# Checks that take no arguments, or that need no input.
# Form: <test name> <target> <expected to succeed> <arguments...>
function(aruco3cuda_add_cli_test test_name target expect_success)
  add_test(NAME "${test_name}" COMMAND $<TARGET_FILE:${target}> ${ARGN})
  set_tests_properties("${test_name}" PROPERTIES TIMEOUT 300)
  if(NOT expect_success)
    set_tests_properties("${test_name}" PROPERTIES WILL_FAIL TRUE)
  endif()
endfunction()

# --- reference runner ---
aruco3cuda_add_cli_test(cli.reference.help aruco3cuda_reference_runner TRUE --help)
aruco3cuda_add_cli_test(cli.reference.list_dictionaries aruco3cuda_reference_runner TRUE --list-dictionaries)
aruco3cuda_add_cli_test(cli.reference.no_input aruco3cuda_reference_runner FALSE --omit-timing)
aruco3cuda_add_cli_test(cli.reference.unknown_option aruco3cuda_reference_runner FALSE --nonexistent-option)
aruco3cuda_add_cli_test(cli.reference.missing_value aruco3cuda_reference_runner FALSE --input)
aruco3cuda_add_cli_test(cli.reference.unknown_dictionary aruco3cuda_reference_runner FALSE
  --dictionary DICT_NOPE --input /nonexistent.png)
aruco3cuda_add_cli_test(cli.reference.invalid_integer aruco3cuda_reference_runner FALSE
  --threads abc --input /nonexistent.png)
aruco3cuda_add_cli_test(cli.reference.invalid_real aruco3cuda_reference_runner FALSE
  --min-marker-length-ratio xyz --input /nonexistent.png)
# Boundary: with ArUco3 enabled, OpenCV itself rejects the combination where both
# downscaling parameters are 0.
aruco3cuda_add_cli_test(cli.reference.aruco3_zero_parameters aruco3cuda_reference_runner FALSE
  --use-aruco3 1 --min-side-length-canonical 0 --min-marker-length-ratio 0
  --input /nonexistent.png)
aruco3cuda_add_cli_test(cli.reference.missing_image aruco3cuda_reference_runner FALSE
  --input /nonexistent/image.png)
aruco3cuda_add_cli_test(cli.reference.unwritable_output aruco3cuda_reference_runner FALSE
  --input /nonexistent/image.png --output /nonexistent-dir/out.json)

aruco3cuda_add_cli_test(cli.reference.invalid_use_aruco3 aruco3cuda_reference_runner FALSE
  --use-aruco3 yes --input /nonexistent.png)
aruco3cuda_add_cli_test(cli.reference.invalid_error_correction aruco3cuda_reference_runner FALSE
  --error-correction-rate abc --input /nonexistent.png)
aruco3cuda_add_cli_test(cli.reference.detector_options aruco3cuda_reference_runner FALSE
  --dictionary DICT_6X6_250 --threads 0 --use-aruco3 0
  --error-correction-rate 0.5 --adaptive-thresh-win-min 5
  --adaptive-thresh-win-max 21 --adaptive-thresh-win-step 8
  --input /nonexistent/image.png)

# With no output path given, the result is written to standard output.
add_test(NAME cli.reference.detect_to_stdout
  COMMAND $<TARGET_FILE:aruco3cuda_reference_runner>
          --input "${kCliCorpusDir}/clean_1280x720_n1_s128.png" --omit-timing)
set_tests_properties(cli.reference.detect_to_stdout PROPERTIES
  FIXTURES_REQUIRED cli_corpus TIMEOUT 300
  PASS_REGULAR_EXPRESSION "schema_version")

add_test(NAME cli.reference.detect
  COMMAND $<TARGET_FILE:aruco3cuda_reference_runner>
          --input "${kCliCorpusDir}/clean_1280x720_n1_s128.png"
          --omit-timing
          --output "${CMAKE_CURRENT_BINARY_DIR}/cli-reference.json")
set_tests_properties(cli.reference.detect PROPERTIES
  FIXTURES_REQUIRED cli_corpus TIMEOUT 300
  PASS_REGULAR_EXPRESSION "^$")

# --- corpusgen ---
aruco3cuda_add_cli_test(cli.corpusgen.help aruco3cuda_corpusgen_cli TRUE --help)
aruco3cuda_add_cli_test(cli.corpusgen.list_presets aruco3cuda_corpusgen_cli TRUE --list-presets)
aruco3cuda_add_cli_test(cli.corpusgen.unknown_preset aruco3cuda_corpusgen_cli FALSE
  --preset does_not_exist --output-dir "${CMAKE_CURRENT_BINARY_DIR}/cli-bad")
aruco3cuda_add_cli_test(cli.corpusgen.unknown_option aruco3cuda_corpusgen_cli FALSE --nonexistent-option x)
aruco3cuda_add_cli_test(cli.corpusgen.missing_value aruco3cuda_corpusgen_cli FALSE --seed)
aruco3cuda_add_cli_test(cli.corpusgen.invalid_seed aruco3cuda_corpusgen_cli FALSE --seed not_a_number)
aruco3cuda_add_cli_test(cli.corpusgen.unknown_dictionary aruco3cuda_corpusgen_cli FALSE
  --dictionary DICT_NOPE --output-dir "${CMAKE_CURRENT_BINARY_DIR}/cli-bad")

# --- dictgen ---
aruco3cuda_add_cli_test(cli.dictgen.help aruco3cuda_dictgen TRUE --help)
aruco3cuda_add_cli_test(cli.dictgen.list aruco3cuda_dictgen TRUE --list)
aruco3cuda_add_cli_test(cli.dictgen.no_action aruco3cuda_dictgen FALSE --dictionary DICT_ARUCO_MIP_36h12)
aruco3cuda_add_cli_test(cli.dictgen.unknown_option aruco3cuda_dictgen FALSE --nonexistent-option x)
aruco3cuda_add_cli_test(cli.dictgen.missing_value aruco3cuda_dictgen FALSE --dictionary)
aruco3cuda_add_cli_test(cli.dictgen.unknown_dictionary aruco3cuda_dictgen FALSE
  --dictionary DICT_NOPE --output "${CMAKE_CURRENT_BINARY_DIR}/cli-bad.cpp")
aruco3cuda_add_cli_test(cli.dictgen.check_missing_file aruco3cuda_dictgen FALSE
  --dictionary DICT_ARUCO_MIP_36h12 --check /nonexistent/table.cpp)
# Continuously confirm that the generated table stays consistent with OpenCV.
aruco3cuda_add_cli_test(cli.dictgen.check_generated aruco3cuda_dictgen TRUE
  --dictionary DICT_ARUCO_MIP_36h12
  --check "${PROJECT_SOURCE_DIR}/src/dictionary/generated/dict_aruco_mip_36h12.cpp")
# Also confirm that freshly generated content matches what is committed.
add_test(NAME cli.dictgen.generate_matches_committed
  COMMAND "${CMAKE_COMMAND}"
          -DDICTGEN=$<TARGET_FILE:aruco3cuda_dictgen>
          -DGENERATED=${CMAKE_CURRENT_BINARY_DIR}/cli-dictgen.cpp
          -DCOMMITTED=${PROJECT_SOURCE_DIR}/src/dictionary/generated/dict_aruco_mip_36h12.cpp
          -P "${CMAKE_CURRENT_SOURCE_DIR}/compare_generated.cmake")
set_tests_properties(cli.dictgen.generate_matches_committed PROPERTIES TIMEOUT 300)

# --- benchmark ---
aruco3cuda_add_cli_test(cli.bench.help aruco3cuda_bench_cli TRUE --help)
aruco3cuda_add_cli_test(cli.bench.no_input aruco3cuda_bench_cli FALSE --warmup 1)
aruco3cuda_add_cli_test(cli.bench.unknown_option aruco3cuda_bench_cli FALSE --nonexistent-option x)
aruco3cuda_add_cli_test(cli.bench.missing_value aruco3cuda_bench_cli FALSE --warmup)
aruco3cuda_add_cli_test(cli.bench.invalid_number aruco3cuda_bench_cli FALSE --warmup abc --input /x.png)
aruco3cuda_add_cli_test(cli.bench.unknown_route aruco3cuda_bench_cli FALSE --route CUDA-Nope --input /x.png)
# An invalid combination of route and memory kind must fail. CUDA-E2E is the
# host-input route and does not accept device-resident input (M-Device). Without
# this check, a run could measure under conditions other than the ones requested
# and nobody would notice.
aruco3cuda_add_cli_test(cli.bench.route_memory_mismatch aruco3cuda_bench_cli FALSE
  --route CUDA-E2E --memory-mode M-Device --warmup 0 --latency-iterations 1
  --throughput-frames 0 --input /x.png)
aruco3cuda_add_cli_test(cli.bench.missing_image aruco3cuda_bench_cli FALSE
  --warmup 0 --latency-iterations 1 --input /nonexistent/image.png)
# Boundary: zero latency iterations is invalid.
aruco3cuda_add_cli_test(cli.bench.zero_iterations aruco3cuda_bench_cli FALSE
  --warmup 0 --latency-iterations 0 --input /x.png)

# Route-name parsing. Routes other than the implemented CPU one must still be
# accepted as names.
aruco3cuda_add_cli_test(cli.bench.route_cpu aruco3cuda_bench_cli FALSE
  --route CPU --warmup 0 --latency-iterations 1 --input /x.png)
aruco3cuda_add_cli_test(cli.bench.route_resident aruco3cuda_bench_cli FALSE
  --route CUDA-Resident --warmup 0 --latency-iterations 1 --input /x.png)
aruco3cuda_add_cli_test(cli.bench.route_hybrid aruco3cuda_bench_cli FALSE
  --route Hybrid --warmup 0 --latency-iterations 1 --input /x.png)
# The path that passes detector settings. If value parsing is wrong, the actual
# measurement conditions disagree with the recorded ones.
aruco3cuda_add_cli_test(cli.bench.detector_options aruco3cuda_bench_cli FALSE
  --dictionary DICT_6X6_250 --threads 2 --use-aruco3 0
  --min-marker-length-ratio 0.1 --min-side-length-canonical 48
  --warmup 0 --latency-iterations 1 --input /nonexistent/image.png)

# CPU affinity. On machines that mix performance and efficiency cores, the
# measurements split into two clusters unless affinity is pinned.
aruco3cuda_add_cli_test(cli.bench.invalid_cpu_list aruco3cuda_bench_cli FALSE
  --cpu-list 0,abc --warmup 0 --latency-iterations 1 --input /x.png)
aruco3cuda_add_cli_test(cli.bench.empty_cpu_list aruco3cuda_bench_cli FALSE
  --cpu-list "" --warmup 0 --latency-iterations 1 --input /x.png)
aruco3cuda_add_cli_test(cli.bench.negative_cpu_list aruco3cuda_bench_cli FALSE
  --cpu-list -1 --warmup 0 --latency-iterations 1 --input /x.png)

add_test(NAME cli.bench.measure
  COMMAND $<TARGET_FILE:aruco3cuda_bench_cli>
          --input "${kCliCorpusDir}/clean_1280x720_n1_s128.png"
          --warmup 1 --latency-iterations 3 --throughput-frames 2 --save-samples --cpu-list 0
          --output "${CMAKE_CURRENT_BINARY_DIR}/cli-bench.jsonl")
set_tests_properties(cli.bench.measure PROPERTIES FIXTURES_REQUIRED cli_corpus TIMEOUT 300)

# --- diff report tool ---
aruco3cuda_add_cli_test(cli.report.help aruco3cuda_report_cli TRUE --help)
aruco3cuda_add_cli_test(cli.report.no_input aruco3cuda_report_cli FALSE --fail-on-diff)
aruco3cuda_add_cli_test(cli.report.unknown_option aruco3cuda_report_cli FALSE --nonexistent-option)
aruco3cuda_add_cli_test(cli.report.missing_value aruco3cuda_report_cli FALSE --input)
aruco3cuda_add_cli_test(cli.report.invalid_tolerance aruco3cuda_report_cli FALSE
  --corner-tolerance-px abc --input /nonexistent.png)
aruco3cuda_add_cli_test(cli.report.negative_tolerance aruco3cuda_report_cli FALSE
  --corner-tolerance-px -1 --input /nonexistent.png)
aruco3cuda_add_cli_test(cli.report.invalid_match_radius aruco3cuda_report_cli FALSE
  --match-radius-ratio 0 --input /nonexistent.png)
aruco3cuda_add_cli_test(cli.report.invalid_use_aruco3 aruco3cuda_report_cli FALSE
  --use-aruco3 yes --input /nonexistent.png)
aruco3cuda_add_cli_test(cli.report.missing_image aruco3cuda_report_cli FALSE
  --input /nonexistent/image.png)

# Nominal: confirm from ctest that results on the synthetic corpus match the CPU
# baseline. With --fail-on-diff, any difference fails the test.
add_test(NAME cli.report.matches_reference
  COMMAND $<TARGET_FILE:aruco3cuda_report_cli>
          --input "${kCliCorpusDir}/clean_1280x720_n1_s128.png"
          --input "${kCliCorpusDir}/clean_1280x720_n4_s128.png"
          --input "${kCliCorpusDir}/degraded_1280x720.png"
          --output "${CMAKE_CURRENT_BINARY_DIR}/cli-report.json"
          --fail-on-diff)
set_tests_properties(cli.report.matches_reference PROPERTIES
  FIXTURES_REQUIRED cli_corpus TIMEOUT 300
  PASS_REGULAR_EXPRESSION "No differences")

aruco3cuda_add_cli_test(cli.report.unwritable_output aruco3cuda_report_cli FALSE
  --input "${kCliCorpusDir}/clean_1280x720_n1_s128.png"
  --output /nonexistent-dir/report.json)
set_tests_properties(cli.report.unwritable_output PROPERTIES FIXTURES_REQUIRED cli_corpus)


# Continuously check the Doxygen requirements for the public API, so that a
# missing entry becomes visible as soon as a declaration is added.
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  add_test(NAME doc.doxygen_completeness
    COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tools/check_doxygen.py"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
  set_tests_properties(doc.doxygen_completeness PROPERTIES TIMEOUT 120)
endif()

# The aggregation script is checked as part of the CLI as well.
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  add_test(NAME cli.aggregate.summarize
    COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/bench/aggregate.py"
            "${CMAKE_CURRENT_BINARY_DIR}/cli-bench.jsonl")
  set_tests_properties(cli.aggregate.summarize PROPERTIES
    DEPENDS cli.bench.measure TIMEOUT 300
    PASS_REGULAR_EXPRESSION "=== Measurement results ===")
  add_test(NAME cli.aggregate.rejects_broken_input
    COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/bench/aggregate.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt")
  set_tests_properties(cli.aggregate.rejects_broken_input PROPERTIES WILL_FAIL TRUE TIMEOUT 300)
endif()
