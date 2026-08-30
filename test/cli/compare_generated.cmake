# SPDX-License-Identifier: Apache-2.0
#
# Confirm that what dictgen generates matches the committed generated file.
# A mismatch means either the OpenCV version changed or the generator's behavior did.
execute_process(
  COMMAND "${DICTGEN}" --dictionary DICT_ARUCO_MIP_36h12 --output "${GENERATED}"
  RESULT_VARIABLE generate_result
  OUTPUT_QUIET)
if(NOT generate_result EQUAL 0)
  message(FATAL_ERROR "dictgen failed to run: ${generate_result}")
endif()

file(SHA256 "${GENERATED}" generated_hash)
file(SHA256 "${COMMITTED}" committed_hash)
if(NOT generated_hash STREQUAL committed_hash)
  message(FATAL_ERROR
    "generated output does not match the committed content\n"
    "  generated: ${generated_hash}\n"
    "  committed: ${committed_hash}")
endif()
message(STATUS "generated output matches the committed content: ${committed_hash}")
