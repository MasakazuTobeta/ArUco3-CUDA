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
# Every bundled table is cross-checked against OpenCV, and then regenerated and
# compared byte for byte with what is committed. Two checks rather than one: the
# first says the table agrees with OpenCV, the second says the committed file is
# exactly what the generator produces, so a hand edit is caught as well.
#
# The list is written out rather than globbed, so that adding a table to
# src/dictionary/generated without adding it here does not silently go
# unchecked.
set(kBundledDictionaries
  DICT_4X4_50
  DICT_4X4_100
  DICT_4X4_250
  DICT_4X4_1000
  DICT_5X5_50
  DICT_5X5_100
  DICT_5X5_250
  DICT_5X5_1000
  DICT_6X6_50
  DICT_6X6_100
  DICT_6X6_250
  DICT_6X6_1000
  DICT_7X7_50
  DICT_7X7_100
  DICT_7X7_250
  DICT_7X7_1000
  DICT_ARUCO_MIP_36h12)

foreach(kDictionary IN LISTS kBundledDictionaries)
  string(TOLOWER "${kDictionary}" kDictionaryFile)
  set(kCommitted "${PROJECT_SOURCE_DIR}/src/dictionary/generated/${kDictionaryFile}.cpp")

  aruco3cuda_add_cli_test("cli.dictgen.check_generated.${kDictionary}" aruco3cuda_dictgen TRUE
    --dictionary "${kDictionary}" --check "${kCommitted}")

  add_test(NAME "cli.dictgen.generate_matches_committed.${kDictionary}"
    COMMAND "${CMAKE_COMMAND}"
            -DDICTGEN=$<TARGET_FILE:aruco3cuda_dictgen>
            -DDICTIONARY=${kDictionary}
            -DGENERATED=${CMAKE_CURRENT_BINARY_DIR}/cli-dictgen-${kDictionary}.cpp
            -DCOMMITTED=${kCommitted}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/compare_generated.cmake")
  set_tests_properties("cli.dictgen.generate_matches_committed.${kDictionary}"
    PROPERTIES TIMEOUT 300)
endforeach()

# The count is asserted separately, so that removing a name from the list above
# without removing the table is caught too.
add_test(NAME cli.dictgen.bundled_count_matches_registry
  COMMAND $<TARGET_FILE:aruco3cuda_reference_runner> --list-dictionaries)
set_tests_properties(cli.dictgen.bundled_count_matches_registry PROPERTIES
  TIMEOUT 60 PASS_REGULAR_EXPRESSION "DICT_ARUCO_MIP_36h12")

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

# The pipeline against the CPU baseline at every marker size. The corpus for
# each is generated with that dictionary, because the markers themselves differ. Without these, a regression that only affects 4x4 or 7x7 would pass
# every other test: the rest of the suite runs on 6x6.
foreach(kPair "DICT_4X4_50" "DICT_5X5_1000" "DICT_7X7_1000")
  set(kDir "${CMAKE_CURRENT_BINARY_DIR}/cli-corpus-${kPair}")
  add_test(NAME "cli.corpusgen.setup.${kPair}"
    COMMAND $<TARGET_FILE:aruco3cuda_corpusgen_cli>
            --preset smoke --seed 20260827 --dictionary "${kPair}"
            --output-dir "${kDir}" --manifest "${kDir}/manifest.json")
  set_tests_properties("cli.corpusgen.setup.${kPair}" PROPERTIES
    FIXTURES_SETUP "cli_corpus_${kPair}" TIMEOUT 300)

  add_test(NAME "cli.report.matches_reference.${kPair}"
    COMMAND $<TARGET_FILE:aruco3cuda_report_cli>
            --dictionary "${kPair}" --fail-on-diff
            --input "${kDir}/clean_1280x720_n4_s128.png"
            --input "${kDir}/degraded_1280x720.png")
  set_tests_properties("cli.report.matches_reference.${kPair}" PROPERTIES
    FIXTURES_REQUIRED "cli_corpus_${kPair}" TIMEOUT 300
    PASS_REGULAR_EXPRESSION "No differences")
endforeach()

aruco3cuda_add_cli_test(cli.report.unwritable_output aruco3cuda_report_cli FALSE
  --input "${kCliCorpusDir}/clean_1280x720_n1_s128.png"
  --output /nonexistent-dir/report.json)
set_tests_properties(cli.report.unwritable_output PROPERTIES FIXTURES_REQUIRED cli_corpus)


# --- samples ---
#
# The samples are the first code a new user runs, so they are held to the same
# standard as the tools: nominal, error, and boundary cases from ctest. The
# round trip matters most. generate_marker renders from the packed dictionary
# and detect_image reads it back through the full pipeline, so a break in either
# the bit order, the border polarity, or the corner ordering shows up as a
# missing id rather than as a plausible-looking picture nobody checks.
if(TARGET aruco3cuda_example_generate_marker)
  # kExampleMarker has exactly one writer, the fixture below, and every other
  # test that passes --output writes somewhere of its own. The argument-handling
  # tests below expect a non-zero exit and never reach write_pgm, so pointing
  # them at the fixture's file happened to work - but ctest orders tests by the
  # previous run's timings, not by the fixture graph, and FIXTURES_REQUIRED only
  # puts the setup first. It does not stop anything else from writing the file
  # afterwards. Two of these tests do succeed, and one of them renders an 8x8
  # marker; when it landed between the setup and the detect tests, they read an
  # 8x8 image, found nothing, and failed. Keeping one writer per file is what
  # makes the ordering irrelevant.
  set(kExampleMarker "${CMAKE_CURRENT_BINARY_DIR}/example-marker.pgm")
  set(kExampleTinyMarker "${CMAKE_CURRENT_BINARY_DIR}/example-tiny-marker.pgm")
  set(kExampleAsciiPgm "${CMAKE_CURRENT_BINARY_DIR}/example-ascii.pgm")
  set(kExampleScratch "${CMAKE_CURRENT_BINARY_DIR}/example-scratch")

  # ASCII PGM must be rejected with a reason rather than misread. Written at
  # configure time because no tool in this repository produces a P2 file.
  file(WRITE "${kExampleAsciiPgm}" "P2\n2 2\n255\n0 0 0 0\n")

  aruco3cuda_add_cli_test(cli.example.generate_marker.help
    aruco3cuda_example_generate_marker TRUE --help)
  aruco3cuda_add_cli_test(cli.example.generate_marker.list_dictionaries
    aruco3cuda_example_generate_marker TRUE --list-dictionaries)
  aruco3cuda_add_cli_test(cli.example.generate_marker.no_output
    aruco3cuda_example_generate_marker FALSE --id 0)
  aruco3cuda_add_cli_test(cli.example.generate_marker.unknown_option
    aruco3cuda_example_generate_marker FALSE --nonexistent-option x)
  aruco3cuda_add_cli_test(cli.example.generate_marker.missing_value
    aruco3cuda_example_generate_marker FALSE --id)
  aruco3cuda_add_cli_test(cli.example.generate_marker.invalid_integer
    aruco3cuda_example_generate_marker FALSE --id abc
    --output "${kExampleScratch}-invalid-integer.pgm")
  aruco3cuda_add_cli_test(cli.example.generate_marker.unknown_dictionary
    aruco3cuda_example_generate_marker FALSE --dictionary DICT_NOPE
    --output "${kExampleScratch}-unknown-dictionary.pgm")
  aruco3cuda_add_cli_test(cli.example.generate_marker.unwritable_output
    aruco3cuda_example_generate_marker FALSE --output /nonexistent-dir/marker.pgm)
  # Boundary: DICT_ARUCO_MIP_36h12 holds 250 ids, so 249 is the last valid one.
  aruco3cuda_add_cli_test(cli.example.generate_marker.last_id
    aruco3cuda_example_generate_marker TRUE --id 249 --size 64
    --output "${kExampleScratch}-last-id.pgm")
  aruco3cuda_add_cli_test(cli.example.generate_marker.id_past_end
    aruco3cuda_example_generate_marker FALSE --id 250
    --output "${kExampleScratch}-id-past-end.pgm")
  # Boundary: the smallest renderable side is the cell count, 6 + 2 * 1 = 8.
  aruco3cuda_add_cli_test(cli.example.generate_marker.smallest_size
    aruco3cuda_example_generate_marker TRUE --size 8 --margin 0
    --output "${kExampleScratch}-smallest-size.pgm")
  aruco3cuda_add_cli_test(cli.example.generate_marker.size_below_cell_count
    aruco3cuda_example_generate_marker FALSE --size 7 --margin 0
    --output "${kExampleScratch}-size-below-cell-count.pgm")

  # Fixture: a marker large enough to be detected, and one deliberately below the
  # ArUco3 lower bound.
  add_test(NAME cli.example.marker.setup
    COMMAND $<TARGET_FILE:aruco3cuda_example_generate_marker>
            --id 42 --size 200 --margin 40 --output "${kExampleMarker}")
  set_tests_properties(cli.example.marker.setup PROPERTIES
    FIXTURES_SETUP example_marker TIMEOUT 60)
  add_test(NAME cli.example.tiny_marker.setup
    COMMAND $<TARGET_FILE:aruco3cuda_example_generate_marker>
            --id 42 --size 40 --margin 400 --output "${kExampleTinyMarker}")
  set_tests_properties(cli.example.tiny_marker.setup PROPERTIES
    FIXTURES_SETUP example_tiny_marker TIMEOUT 60)
endif()

if(TARGET aruco3cuda_example_detect_image)
  aruco3cuda_add_cli_test(cli.example.detect_image.help
    aruco3cuda_example_detect_image TRUE --help)
  aruco3cuda_add_cli_test(cli.example.detect_image.no_input
    aruco3cuda_example_detect_image FALSE --repeat 1)
  aruco3cuda_add_cli_test(cli.example.detect_image.unknown_option
    aruco3cuda_example_detect_image FALSE --nonexistent-option x)
  aruco3cuda_add_cli_test(cli.example.detect_image.missing_value
    aruco3cuda_example_detect_image FALSE --input)
  aruco3cuda_add_cli_test(cli.example.detect_image.missing_image
    aruco3cuda_example_detect_image FALSE --input /nonexistent/image.pgm)
  aruco3cuda_add_cli_test(cli.example.detect_image.ascii_pgm_rejected
    aruco3cuda_example_detect_image FALSE --input "${kExampleAsciiPgm}")
  # The exit code alone would also be produced by a missing file, so the reason
  # is asserted separately. Naming P2 is the whole point of rejecting it: a
  # reader who exported the wrong PGM variant needs to be told which one.
  add_test(NAME cli.example.detect_image.ascii_pgm_reason
    COMMAND $<TARGET_FILE:aruco3cuda_example_detect_image> --input "${kExampleAsciiPgm}")
  set_tests_properties(cli.example.detect_image.ascii_pgm_reason PROPERTIES
    TIMEOUT 60 PASS_REGULAR_EXPRESSION "ASCII PGM \\(P2\\) is not supported")

  # These two take a readable image on purpose. Pointing them at a bad path
  # would make them pass on the file error and never reach the option under
  # test.
  aruco3cuda_add_cli_test(cli.example.detect_image.invalid_repeat
    aruco3cuda_example_detect_image FALSE --repeat 0 --input "${kExampleMarker}")
  aruco3cuda_add_cli_test(cli.example.detect_image.unknown_dictionary
    aruco3cuda_example_detect_image FALSE --dictionary DICT_NOPE --input "${kExampleMarker}")
  set_tests_properties(cli.example.detect_image.invalid_repeat
                       cli.example.detect_image.unknown_dictionary
    PROPERTIES FIXTURES_REQUIRED example_marker)

  # Nominal: the id that went in comes back out.
  add_test(NAME cli.example.detect_image.round_trip
    COMMAND $<TARGET_FILE:aruco3cuda_example_detect_image> --input "${kExampleMarker}")
  # PASS_REGULAR_EXPRESSION passes when any one entry matches, so the count is
  # asserted the other way round: any count other than 1 fails the test.
  set_tests_properties(cli.example.detect_image.round_trip PROPERTIES
    FIXTURES_REQUIRED example_marker TIMEOUT 300
    PASS_REGULAR_EXPRESSION "id=42"
    FAIL_REGULAR_EXPRESSION "detections   : ([^1]|1[0-9])")

  # The workspace must stop allocating. If a per-frame cudaMalloc ever creeps
  # back in, allocation_count_ grows with --repeat and this fails.
  add_test(NAME cli.example.detect_image.repeat_allocates_once
    COMMAND $<TARGET_FILE:aruco3cuda_example_detect_image>
            --input "${kExampleMarker}" --repeat 8)
  set_tests_properties(cli.example.detect_image.repeat_allocates_once PROPERTIES
    FIXTURES_REQUIRED example_marker TIMEOUT 300
    PASS_REGULAR_EXPRESSION "workspace    : 1 allocation\\(s\\), 0 reallocation")

  # The default stream cannot be captured as a graph, so it takes the other
  # dispatch path. Both must produce the same id.
  add_test(NAME cli.example.detect_image.default_stream
    COMMAND $<TARGET_FILE:aruco3cuda_example_detect_image>
            --input "${kExampleMarker}" --default-stream)
  set_tests_properties(cli.example.detect_image.default_stream PROPERTIES
    FIXTURES_REQUIRED example_marker TIMEOUT 300
    PASS_REGULAR_EXPRESSION "id=42")

  add_test(NAME cli.example.detect_image.without_aruco3
    COMMAND $<TARGET_FILE:aruco3cuda_example_detect_image>
            --input "${kExampleMarker}" --no-aruco3)
  set_tests_properties(cli.example.detect_image.without_aruco3 PROPERTIES
    FIXTURES_REQUIRED example_marker TIMEOUT 300
    PASS_REGULAR_EXPRESSION "id=42")

  # Boundary: a marker below the ArUco3 lower bound is not found. This is the
  # most common reason a first run reports nothing, so it is pinned down rather
  # than left as a note in the documentation.
  add_test(NAME cli.example.detect_image.below_aruco3_lower_bound
    COMMAND $<TARGET_FILE:aruco3cuda_example_detect_image> --input "${kExampleTinyMarker}")
  set_tests_properties(cli.example.detect_image.below_aruco3_lower_bound PROPERTIES
    FIXTURES_REQUIRED example_tiny_marker TIMEOUT 300
    PASS_REGULAR_EXPRESSION "detections   : 0 ")

  # The same image with ArUco3 off has no lower bound, so it must be found.
  add_test(NAME cli.example.detect_image.small_marker_without_aruco3
    COMMAND $<TARGET_FILE:aruco3cuda_example_detect_image>
            --input "${kExampleTinyMarker}" --no-aruco3)
  set_tests_properties(cli.example.detect_image.small_marker_without_aruco3 PROPERTIES
    FIXTURES_REQUIRED example_tiny_marker TIMEOUT 300
    PASS_REGULAR_EXPRESSION "id=42")
endif()


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

  # SUPPORTED_SCHEMA_VERSIONS is a hand-maintained set, and both ways of getting
  # it wrong are silent. cli.aggregate.summarize only proves that the version the
  # harness emits right now is accepted, so it would stay green if the set were
  # narrowed to that one version and every published measurement file became
  # unreadable by its own aggregator. These two tests pin both ends: a version
  # that must be excluded, and the version the committed corpus is written in.
  add_test(NAME cli.aggregate.rejects_unsupported_schema
    COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/bench/aggregate.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/schema-version-3.jsonl")
  set_tests_properties(cli.aggregate.rejects_unsupported_schema PROPERTIES
    WILL_FAIL TRUE TIMEOUT 300)
  add_test(NAME cli.aggregate.reads_the_committed_corpus
    COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/bench/aggregate.py"
            "${PROJECT_SOURCE_DIR}/docs/measurements/2026-08-29-jetson-orin-memory-modes.jsonl")
  set_tests_properties(cli.aggregate.reads_the_committed_corpus PROPERTIES
    TIMEOUT 300 PASS_REGULAR_EXPRESSION "=== Measurement results ===")
endif()
