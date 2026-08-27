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

if(ARUCO3CUDA_ENABLE_CLANG_TIDY)
  find_program(kClangTidy NAMES clang-tidy REQUIRED)
  # clang-tidy は .cu を解釈できないため C++ にのみ適用する。
  set(CMAKE_CXX_CLANG_TIDY "${kClangTidy}")
endif()

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
    "${PROJECT_SOURCE_DIR}/test/*.cpp" "${PROJECT_SOURCE_DIR}/test/*.cu")
  add_custom_target(format-check
    COMMAND "${kClangFormat}" --dry-run --Werror ${kFormatSources}
    COMMENT "clang-format による整形差分の確認")
  add_custom_target(format-fix
    COMMAND "${kClangFormat}" -i ${kFormatSources}
    COMMENT "clang-format による整形の適用")
endfunction()
