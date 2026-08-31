# SPDX-License-Identifier: Apache-2.0
#
# Tests for the ctypes Python package and the Python samples.
#
# Split by whether a device is needed. test_binding.py runs anywhere, so CI
# without a GPU runs it too; test_detect.py needs a device and runs only on the
# four physical machines. Filtering inside one file would hide which of the two
# a failure came from.
find_package(Python3 COMPONENTS Interpreter)
if(NOT Python3_Interpreter_FOUND)
  message(STATUS "  python tests       : skipped, no interpreter was found")
  return()
endif()

set(kPythonPath "${ARUCO3CUDA_PYTHON_PATH}")
set(kSampleDir "${PROJECT_SOURCE_DIR}/examples/python")

# The samples import both the package and their own pgm module, so the sample
# directory joins PYTHONPATH for the tests that run them.
set(kPackageEnvironment "PYTHONPATH=${kPythonPath}")
set(kSampleEnvironment "PYTHONPATH=${kPythonPath}:${kSampleDir}")

add_test(NAME python.binding
  COMMAND "${Python3_EXECUTABLE}" -m unittest discover -s "${CMAKE_CURRENT_SOURCE_DIR}"
          -p "test_binding.py" -v)
set_tests_properties(python.binding PROPERTIES
  ENVIRONMENT "${kPackageEnvironment}" TIMEOUT 120)

add_test(NAME python.detect
  COMMAND "${Python3_EXECUTABLE}" -m unittest discover -s "${CMAKE_CURRENT_SOURCE_DIR}"
          -p "test_detect.py" -v)
set_tests_properties(python.detect PROPERTIES
  ENVIRONMENT "${kPackageEnvironment}" TIMEOUT 300)

# --- the Python samples ---
#
# Form: <test name> <script> <expected to succeed> <arguments...>
function(aruco3cuda_add_python_sample_test test_name script expect_success)
  add_test(NAME "${test_name}"
    COMMAND "${Python3_EXECUTABLE}" "${kSampleDir}/${script}" ${ARGN})
  set_tests_properties("${test_name}" PROPERTIES
    ENVIRONMENT "${kSampleEnvironment}" TIMEOUT 300)
  if(NOT expect_success)
    set_tests_properties("${test_name}" PROPERTIES WILL_FAIL TRUE)
  endif()
endfunction()

# One writer per file, for the reason set out in test/cli/CMakeLists.txt.
# kPythonMarker is written only by the fixture below. None of these
# argument-handling tests has ever clobbered it, because each expects a non-zero
# exit and the sample validates before it writes - but that is a property of the
# sample, not of the test, and it is not what the fixture should depend on.
set(kPythonMarker "${CMAKE_CURRENT_BINARY_DIR}/python-marker.pgm")
set(kPythonAsciiPgm "${CMAKE_CURRENT_BINARY_DIR}/python-ascii.pgm")
set(kPythonScratch "${CMAKE_CURRENT_BINARY_DIR}/python-scratch")
file(WRITE "${kPythonAsciiPgm}" "P2\n2 2\n255\n0 0 0 0\n")

aruco3cuda_add_python_sample_test(python.example.generate_marker.help
  generate_marker.py TRUE --help)
aruco3cuda_add_python_sample_test(python.example.generate_marker.list_dictionaries
  generate_marker.py TRUE --list-dictionaries)
aruco3cuda_add_python_sample_test(python.example.generate_marker.no_output
  generate_marker.py FALSE --id 0)
aruco3cuda_add_python_sample_test(python.example.generate_marker.unknown_option
  generate_marker.py FALSE --nonexistent-option x)
aruco3cuda_add_python_sample_test(python.example.generate_marker.invalid_integer
  generate_marker.py FALSE --id abc --output "${kPythonScratch}-invalid-integer.pgm")
aruco3cuda_add_python_sample_test(python.example.generate_marker.unknown_dictionary
  generate_marker.py FALSE --dictionary DICT_NOPE --output "${kPythonScratch}-unknown-dictionary.pgm")
aruco3cuda_add_python_sample_test(python.example.generate_marker.id_past_end
  generate_marker.py FALSE --id 250 --output "${kPythonScratch}-id-past-end.pgm")
aruco3cuda_add_python_sample_test(python.example.generate_marker.size_below_cell_count
  generate_marker.py FALSE --size 7 --margin 0 --output "${kPythonScratch}-size-below-cell-count.pgm")
aruco3cuda_add_python_sample_test(python.example.generate_marker.unwritable_output
  generate_marker.py FALSE --output /nonexistent-dir/marker.pgm)

add_test(NAME python.example.marker.setup
  COMMAND "${Python3_EXECUTABLE}" "${kSampleDir}/generate_marker.py"
          --id 42 --size 200 --margin 40 --output "${kPythonMarker}")
set_tests_properties(python.example.marker.setup PROPERTIES
  ENVIRONMENT "${kSampleEnvironment}" FIXTURES_SETUP python_marker TIMEOUT 120)

aruco3cuda_add_python_sample_test(python.example.detect_image.help
  detect_image.py TRUE --help)
aruco3cuda_add_python_sample_test(python.example.detect_image.no_input
  detect_image.py FALSE --repeat 1)
aruco3cuda_add_python_sample_test(python.example.detect_image.unknown_option
  detect_image.py FALSE --nonexistent-option x)
aruco3cuda_add_python_sample_test(python.example.detect_image.missing_image
  detect_image.py FALSE --input /nonexistent/image.pgm)
aruco3cuda_add_python_sample_test(python.example.detect_image.ascii_pgm_rejected
  detect_image.py FALSE --input "${kPythonAsciiPgm}")
# The exit code alone would also come from a missing file, so the reason is
# asserted separately.
add_test(NAME python.example.detect_image.ascii_pgm_reason
  COMMAND "${Python3_EXECUTABLE}" "${kSampleDir}/detect_image.py"
          --input "${kPythonAsciiPgm}")
set_tests_properties(python.example.detect_image.ascii_pgm_reason PROPERTIES
  ENVIRONMENT "${kSampleEnvironment}" TIMEOUT 120
  PASS_REGULAR_EXPRESSION "ASCII PGM \\(P2\\) is not supported")

# These two take a readable image on purpose. Pointing them at a bad path would
# make them pass on the file error and never reach the option under test.
aruco3cuda_add_python_sample_test(python.example.detect_image.invalid_repeat
  detect_image.py FALSE --repeat 0 --input "${kPythonMarker}")
aruco3cuda_add_python_sample_test(python.example.detect_image.unknown_dictionary
  detect_image.py FALSE --dictionary DICT_NOPE --input "${kPythonMarker}")
set_tests_properties(python.example.detect_image.invalid_repeat
                     python.example.detect_image.unknown_dictionary
  PROPERTIES FIXTURES_REQUIRED python_marker)

# PASS_REGULAR_EXPRESSION passes when any one entry matches, so the count is
# asserted the other way round: any count other than 1 fails the test.
add_test(NAME python.example.detect_image.round_trip
  COMMAND "${Python3_EXECUTABLE}" "${kSampleDir}/detect_image.py"
          --input "${kPythonMarker}" --repeat 4)
set_tests_properties(python.example.detect_image.round_trip PROPERTIES
  ENVIRONMENT "${kSampleEnvironment}" FIXTURES_REQUIRED python_marker TIMEOUT 300
  PASS_REGULAR_EXPRESSION "id=42"
  FAIL_REGULAR_EXPRESSION "detections   : ([^1]|1[0-9])")

# The C++ and Python samples must render the same bytes. Only registered when
# the C++ samples are built, since one side of the comparison comes from there.
if(TARGET aruco3cuda_example_generate_marker)
  # Sizes chosen to cover both an exact multiple of the 8 cell grid and sizes
  # that are not, because that is where the two scalings could disagree.
  foreach(spec "42;200;40;1" "0;61;0;1" "249;96;8;2")
    list(GET spec 0 kId)
    list(GET spec 1 kSize)
    list(GET spec 2 kMargin)
    list(GET spec 3 kBorder)
    add_test(NAME "python.example.matches_cxx.id${kId}_size${kSize}_border${kBorder}"
      COMMAND "${CMAKE_COMMAND}"
              -DPYTHON=${Python3_EXECUTABLE}
              -DPYTHON_SAMPLE=${kSampleDir}/generate_marker.py
              -DCXX_SAMPLE=$<TARGET_FILE:aruco3cuda_example_generate_marker>
              -DMARKER_ID=${kId}
              -DMARKER_SIZE=${kSize}
              -DMARKER_MARGIN=${kMargin}
              -DBORDER_BITS=${kBorder}
              -DPYTHON_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cross-${kId}-${kSize}-${kBorder}-py.pgm
              -DCXX_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cross-${kId}-${kSize}-${kBorder}-cxx.pgm
              -P "${CMAKE_CURRENT_SOURCE_DIR}/compare_markers.cmake")
    set_tests_properties("python.example.matches_cxx.id${kId}_size${kSize}_border${kBorder}"
      PROPERTIES ENVIRONMENT "${kSampleEnvironment}" TIMEOUT 300)
  endforeach()
endif()
