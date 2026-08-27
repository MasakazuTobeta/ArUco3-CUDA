# SPDX-License-Identifier: Apache-2.0
#
# 目的:
#   Compute Sanitizer を使用するテスト経路を用意する。
#   memcheck、racecheck、initcheck、synccheck を別々の ctest として登録し、
#   どの検査で失敗したかを切り分けられるようにする。
include_guard(GLOBAL)

option(ARUCO3CUDA_ENABLE_COMPUTE_SANITIZER "Compute Sanitizer の ctest を登録する" OFF)

set(ARUCO3CUDA_SANITIZER_TOOLS "memcheck;racecheck;initcheck;synccheck"
    CACHE STRING "登録する Compute Sanitizer の tool")

# 指定した test 実行 file を Compute Sanitizer 経由で実行する ctest を追加する。
#
# 引数:
#   target - 対象の test 実行 file target
#
# 備考:
#   sanitizer は実行速度が大きく低下するため、既定では無効とする。
#   CI では日次または PR で有効化する運用を想定する。
function(aruco3cuda_add_sanitizer_tests target)
  if(NOT ARUCO3CUDA_ENABLE_COMPUTE_SANITIZER)
    return()
  endif()
  find_program(kComputeSanitizer
    NAMES compute-sanitizer
    HINTS "$ENV{CUDA_HOME}/bin" "${CUDAToolkit_BIN_DIR}" /usr/local/cuda/bin)
  if(NOT kComputeSanitizer)
    message(WARNING "compute-sanitizer が見つからないため sanitizer test を登録しない")
    return()
  endif()
  # 意図的に CUDA API を失敗させる test は除外する。Compute Sanitizer は
  # 意図の有無に関わらず全ての API エラーを報告するため、
  # 意図した失敗が sanitizer の指摘として現れ、本物の問題を埋もれさせる。
  #
  # 時間を測る test も除外する。暖機と繰り返しで同じ経路を何百回も通るため、
  # sanitizer の下では実行時間が現実的でなくなる。同じ経路の正しさは
  # 対応する検証用の test が確かめており、繰り返しても新しい経路は通らない。
  #
  # 対象は suite 名で識別する。該当が無い target では filter は何も除外しない。
  set(kSanitizerGtestFilter "-*DeliberateError*.*:*Timing*.*")
  foreach(tool IN LISTS ARUCO3CUDA_SANITIZER_TOOLS)
    add_test(NAME "sanitizer.${tool}.${target}"
      COMMAND "${kComputeSanitizer}" --tool "${tool}" --error-exitcode 1
              "$<TARGET_FILE:${target}>" "--gtest_filter=${kSanitizerGtestFilter}")
    # sanitizer 経由では実行時間が延びるため timeout を個別に設定する。
    set_tests_properties("sanitizer.${tool}.${target}" PROPERTIES TIMEOUT 600)
  endforeach()
endfunction()
