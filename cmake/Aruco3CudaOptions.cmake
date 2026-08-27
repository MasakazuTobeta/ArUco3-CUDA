# SPDX-License-Identifier: Apache-2.0
#
# 目的:
#   project 全体の option、言語標準、警告設定を 1 箇所へ集約する。
#   しきい値や対象 architecture を source の固定値にせず、設定から上書きできるようにする。
include_guard(GLOBAL)

option(ARUCO3CUDA_BUILD_TESTS "自動テストを build する" ON)
option(ARUCO3CUDA_BUILD_REFERENCE "OpenCV を用いた CPU 基準 runner を build する" ON)
option(ARUCO3CUDA_WARNINGS_AS_ERRORS "警告を error として扱う" ON)
option(ARUCO3CUDA_ENABLE_CLANG_TIDY "C++ source へ clang-tidy を適用する" OFF)
option(ARUCO3CUDA_ENABLE_COVERAGE "C++ code の C0 / C1 カバレッジを測定する" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "build type" FORCE)
endif()

# 警告設定。CUDA には host compiler へ転送する形で適用し、
# nvcc が生成する code に起因する警告を避けるため C++ より範囲を狭める。
add_library(aruco3cuda_warnings INTERFACE)
add_library(aruco3cuda::warnings ALIAS aruco3cuda_warnings)

set(kCxxWarnings
  -Wall -Wextra -Wpedantic
  -Wshadow -Wnon-virtual-dtor -Wold-style-cast
  -Wcast-align -Wunused -Woverloaded-virtual
  -Wnull-dereference -Wdouble-promotion -Wformat=2)
set(kCudaHostWarnings -Wall -Wextra)

if(ARUCO3CUDA_WARNINGS_AS_ERRORS)
  list(APPEND kCxxWarnings -Werror)
  list(APPEND kCudaHostWarnings -Werror)
endif()

list(JOIN kCudaHostWarnings "," kCudaHostWarningsJoined)

target_compile_options(aruco3cuda_warnings INTERFACE
  "$<$<COMPILE_LANGUAGE:CXX>:${kCxxWarnings}>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${kCudaHostWarningsJoined}>")

# カバレッジ計測。CONTRIBUTING.md は C0 と C1 の 100% を目標と定める。
#
# 計測対象は host compiler が compile する C++ に限る。gcov は CUDA device code の
# 実行経路を計測できないため、device code は規約の定めどおり入力分割と境界値の
# test で経路を確認する。
if(ARUCO3CUDA_ENABLE_COVERAGE)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(FATAL_ERROR "カバレッジ計測は GCC または Clang でのみ対応する")
  endif()
  add_library(aruco3cuda_coverage INTERFACE)
  add_library(aruco3cuda::coverage ALIAS aruco3cuda_coverage)
  # 最適化を無効にしないと分岐が消え、C1 が実態と合わなくなる。
  target_compile_options(aruco3cuda_coverage INTERFACE
    "$<$<COMPILE_LANGUAGE:CXX>:--coverage;-O0;-g;-fno-inline>")
  target_link_options(aruco3cuda_coverage INTERFACE --coverage)
else()
  # option が無効でも target 名を解決できるようにし、各 target の記述を分岐させない。
  add_library(aruco3cuda_coverage INTERFACE)
  add_library(aruco3cuda::coverage ALIAS aruco3cuda_coverage)
endif()

if(ARUCO3CUDA_ENABLE_CLANG_TIDY)
  find_program(kClangTidy NAMES clang-tidy REQUIRED)
  # clang-tidy は .cu を解釈できないため C++ にのみ適用する。
  set(CMAKE_CXX_CLANG_TIDY "${kClangTidy}")
endif()

# カバレッジ報告の target。計測が無効な場合は作らない。
function(aruco3cuda_add_coverage_target)
  if(NOT ARUCO3CUDA_ENABLE_COVERAGE)
    return()
  endif()
  find_program(kGcovr NAMES gcovr)
  if(NOT kGcovr)
    message(WARNING "gcovr が見つからないため coverage-report target を作らない")
    return()
  endif()
  # 生成物と test 自身は計測対象から外す。test の網羅率は指標にならない。
  add_custom_target(coverage-report
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_BINARY_DIR}/coverage"
    COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure
    COMMAND "${kGcovr}"
            --root "${PROJECT_SOURCE_DIR}"
            --filter "${PROJECT_SOURCE_DIR}/src/"
            --filter "${PROJECT_SOURCE_DIR}/reference/"
            --filter "${PROJECT_SOURCE_DIR}/tools/"
            --filter "${PROJECT_SOURCE_DIR}/bench/"
            --exclude ".*/generated/.*"
            --exclude ".*/test/.*"
            --decisions
            --print-summary
            --html-details "${CMAKE_BINARY_DIR}/coverage/index.html"
            --json-summary "${CMAKE_BINARY_DIR}/coverage/summary.json"
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    COMMENT "ctest 実行後に C0 / C1 カバレッジを集計する")
endfunction()

# format 用の target。CI と手元で同じ判定を使う。
function(aruco3cuda_add_format_targets)
  find_program(kClangFormat NAMES clang-format)
  if(NOT kClangFormat)
    message(STATUS "  clang-format       : 見つからないため format target を作らない")
    return()
  endif()
  file(GLOB_RECURSE kFormatSources
    "${PROJECT_SOURCE_DIR}/include/*.hpp"
    "${PROJECT_SOURCE_DIR}/src/*.hpp" "${PROJECT_SOURCE_DIR}/src/*.cpp" "${PROJECT_SOURCE_DIR}/src/*.cu"
    "${PROJECT_SOURCE_DIR}/reference/*.hpp" "${PROJECT_SOURCE_DIR}/reference/*.cpp"
    "${PROJECT_SOURCE_DIR}/tools/*.hpp" "${PROJECT_SOURCE_DIR}/tools/*.cpp"
    "${PROJECT_SOURCE_DIR}/test/*.cpp" "${PROJECT_SOURCE_DIR}/test/*.cu")
  # 生成物は整形対象から外す。整形すると生成器の出力と byte 単位で一致しなくなり、
  # dictgen --check による再生成の検証が成立しなくなる。
  list(FILTER kFormatSources EXCLUDE REGEX "/generated/")
  add_custom_target(format-check
    COMMAND "${kClangFormat}" --dry-run --Werror ${kFormatSources}
    COMMENT "clang-format による整形差分の確認")
  add_custom_target(format-fix
    COMMAND "${kClangFormat}" -i ${kFormatSources}
    COMMENT "clang-format による整形の適用")
endfunction()
