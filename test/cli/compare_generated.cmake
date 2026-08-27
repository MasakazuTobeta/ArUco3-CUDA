# SPDX-License-Identifier: Apache-2.0
#
# dictgen が生成した内容が commit 済みの生成物と一致することを確認する。
# 一致しなければ、OpenCV の version が変わったか生成器の挙動が変わっている。
execute_process(
  COMMAND "${DICTGEN}" --dictionary DICT_ARUCO_MIP_36h12 --output "${GENERATED}"
  RESULT_VARIABLE generate_result
  OUTPUT_QUIET)
if(NOT generate_result EQUAL 0)
  message(FATAL_ERROR "dictgen の実行に失敗した: ${generate_result}")
endif()

file(SHA256 "${GENERATED}" generated_hash)
file(SHA256 "${COMMITTED}" committed_hash)
if(NOT generated_hash STREQUAL committed_hash)
  message(FATAL_ERROR
    "生成物が commit 済みの内容と一致しない\n"
    "  生成: ${generated_hash}\n"
    "  commit 済み: ${committed_hash}")
endif()
message(STATUS "生成物は commit 済みの内容と一致する: ${committed_hash}")
