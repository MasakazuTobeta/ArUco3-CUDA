# SPDX-License-Identifier: Apache-2.0
#
# Confirm that the Python and C++ samples render a marker to identical bytes.
#
# The two implement the same rule independently, one through the C++ dictionary
# API and one through the C ABI. If the bit order, the border polarity, or the
# nearest-neighbour scaling drifted on either side, the images would differ
# while both still looked like plausible markers, which is exactly the kind of
# difference an eye does not catch.
execute_process(
  COMMAND "${PYTHON}" "${PYTHON_SAMPLE}"
          --id "${MARKER_ID}" --size "${MARKER_SIZE}" --margin "${MARKER_MARGIN}"
          --border-bits "${BORDER_BITS}" --output "${PYTHON_OUTPUT}"
  RESULT_VARIABLE python_result
  OUTPUT_QUIET)
if(NOT python_result EQUAL 0)
  message(FATAL_ERROR "the Python sample failed to run: ${python_result}")
endif()

execute_process(
  COMMAND "${CXX_SAMPLE}"
          --id "${MARKER_ID}" --size "${MARKER_SIZE}" --margin "${MARKER_MARGIN}"
          --border-bits "${BORDER_BITS}" --output "${CXX_OUTPUT}"
  RESULT_VARIABLE cxx_result
  OUTPUT_QUIET)
if(NOT cxx_result EQUAL 0)
  message(FATAL_ERROR "the C++ sample failed to run: ${cxx_result}")
endif()

file(SHA256 "${PYTHON_OUTPUT}" python_hash)
file(SHA256 "${CXX_OUTPUT}" cxx_hash)
if(NOT python_hash STREQUAL cxx_hash)
  message(FATAL_ERROR
    "the two samples rendered different images for id=${MARKER_ID} "
    "size=${MARKER_SIZE} border=${BORDER_BITS}\n"
    "  python: ${python_hash}\n"
    "  c++   : ${cxx_hash}")
endif()
message(STATUS "both samples rendered the same image: ${python_hash}")
