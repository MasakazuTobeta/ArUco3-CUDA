# SPDX-License-Identifier: Apache-2.0
#
# Purpose:
#   Collect the project-wide options, language standards, and warning settings in
#   one place. Thresholds and target architectures stay out of the source as hard
#   coded values so that they can be overridden from the configuration.
include_guard(GLOBAL)

option(ARUCO3CUDA_BUILD_TESTS "Build the automated tests" ON)
option(ARUCO3CUDA_BUILD_REFERENCE "Build the OpenCV based CPU baseline runner" ON)
option(ARUCO3CUDA_WARNINGS_AS_ERRORS "Treat warnings as errors" ON)
option(ARUCO3CUDA_ENABLE_CLANG_TIDY "Apply clang-tidy to the C++ sources" OFF)
option(ARUCO3CUDA_ENABLE_COVERAGE "Measure C0 / C1 coverage of the C++ code" OFF)

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

# Warning settings. For CUDA they are applied by forwarding them to the host
# compiler, and the set is narrower than for C++ so that warnings caused by the
# code nvcc generates are avoided.
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

# Coverage measurement. CONTRIBUTING.md sets 100% C0 and C1 as the goal.
#
# Measurement is limited to the C++ compiled by the host compiler. gcov cannot
# measure the execution paths of CUDA device code, so for device code the paths
# are confirmed, as the conventions require, by tests over input partitions and
# boundary values.
if(ARUCO3CUDA_ENABLE_COVERAGE)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(FATAL_ERROR "Coverage measurement is supported only with GCC or Clang")
  endif()
  add_library(aruco3cuda_coverage INTERFACE)
  add_library(aruco3cuda::coverage ALIAS aruco3cuda_coverage)
  # Without disabling optimization, branches disappear and C1 no longer matches
  # reality.
  target_compile_options(aruco3cuda_coverage INTERFACE
    "$<$<COMPILE_LANGUAGE:CXX>:--coverage;-O0;-g;-fno-inline>")
  target_link_options(aruco3cuda_coverage INTERFACE --coverage)
else()
  # Keep the target name resolvable even when the option is off, so that the
  # description of each target does not need a conditional branch.
  add_library(aruco3cuda_coverage INTERFACE)
  add_library(aruco3cuda::coverage ALIAS aruco3cuda_coverage)
endif()

if(ARUCO3CUDA_ENABLE_CLANG_TIDY)
  find_program(kClangTidy NAMES clang-tidy REQUIRED)
  # clang-tidy cannot parse .cu files, so apply it to C++ only.
  set(CMAKE_CXX_CLANG_TIDY "${kClangTidy}")
endif()

# Target for the coverage report. It is not created when measurement is off.
function(aruco3cuda_add_coverage_target)
  if(NOT ARUCO3CUDA_ENABLE_COVERAGE)
    return()
  endif()
  find_program(kGcovr NAMES gcovr)
  if(NOT kGcovr)
    message(WARNING "gcovr was not found, so the coverage-report target is not created")
    return()
  endif()
  # Exclude generated files and the tests themselves from measurement. The
  # coverage of the tests is not a meaningful metric.
  add_custom_target(coverage-report
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_BINARY_DIR}/coverage"
    COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure
    COMMAND "${kGcovr}"
            --root "${PROJECT_SOURCE_DIR}"
            --filter "${PROJECT_SOURCE_DIR}/src/"
            --filter "${PROJECT_SOURCE_DIR}/reference/"
            --filter "${PROJECT_SOURCE_DIR}/tools/"
            --filter "${PROJECT_SOURCE_DIR}/bench/"
            --filter "${PROJECT_SOURCE_DIR}/hybrid/"
            --exclude ".*/generated/.*"
            --exclude ".*/test/.*"
            --decisions
            --print-summary
            --html-details "${CMAKE_BINARY_DIR}/coverage/index.html"
            --json-summary "${CMAKE_BINARY_DIR}/coverage/summary.json"
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    COMMENT "Aggregate C0 / C1 coverage after running ctest")
endfunction()

# Targets for formatting. The decision is concentrated in tools/check-format.sh
# so that CI and local runs use the same file list and the same arguments.
# Duplicating the glob here would inevitably drift.
function(aruco3cuda_add_format_targets)
  find_program(kClangFormat NAMES clang-format)
  if(NOT kClangFormat)
    message(STATUS "  clang-format       : not found, format targets are not created")
    return()
  endif()
  add_custom_target(format-check
    COMMAND "${CMAKE_COMMAND}" -E env "CLANG_FORMAT=${kClangFormat}"
            "${PROJECT_SOURCE_DIR}/tools/check-format.sh"
    COMMENT "Check formatting differences with clang-format")
  add_custom_target(format-fix
    COMMAND "${CMAKE_COMMAND}" -E env "CLANG_FORMAT=${kClangFormat}"
            "${PROJECT_SOURCE_DIR}/tools/check-format.sh" --fix
    COMMENT "Apply formatting with clang-format")
endfunction()
