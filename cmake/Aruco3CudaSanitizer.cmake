# SPDX-License-Identifier: Apache-2.0
#
# Purpose:
#   Provide a test path that runs under Compute Sanitizer. memcheck, racecheck,
#   initcheck, and synccheck are registered as separate ctest entries so that it
#   is clear which check failed.
include_guard(GLOBAL)

option(ARUCO3CUDA_ENABLE_COMPUTE_SANITIZER "Register the Compute Sanitizer ctest entries" OFF)

set(ARUCO3CUDA_SANITIZER_TOOLS "memcheck;racecheck;initcheck;synccheck"
    CACHE STRING "Compute Sanitizer tools to register")

# Add ctest entries that run the given test executable through Compute Sanitizer.
#
# Arguments:
#   target - the target of the test executable to run
#
# Notes:
#   The sanitizer slows execution down considerably, so it is disabled by
#   default. In CI it is expected to be enabled on a daily schedule or per pull
#   request.
function(aruco3cuda_add_sanitizer_tests target)
  if(NOT ARUCO3CUDA_ENABLE_COMPUTE_SANITIZER)
    return()
  endif()
  find_program(kComputeSanitizer
    NAMES compute-sanitizer
    HINTS "$ENV{CUDA_HOME}/bin" "${CUDAToolkit_BIN_DIR}" /usr/local/cuda/bin)
  if(NOT kComputeSanitizer)
    message(WARNING "compute-sanitizer was not found, so the sanitizer tests are not registered")
    return()
  endif()
  # Exclude tests that fail CUDA API calls on purpose. Compute Sanitizer reports
  # every API error regardless of intent, so a deliberate failure shows up as a
  # sanitizer report and buries the real problems.
  #
  # Exclude timing tests as well. Warm-up and repetition walk the same path
  # hundreds of times, which makes the run time impractical under the sanitizer.
  # The correctness of that same path is already confirmed by the corresponding
  # verification tests, and repeating it does not reach any new path.
  #
  # The exclusions are identified by suite name. On a target with no match the
  # filter excludes nothing.
  set(kSanitizerGtestFilter "-*DeliberateError*.*:*Timing*.*")
  foreach(tool IN LISTS ARUCO3CUDA_SANITIZER_TOOLS)
    # Also look for resource leaks and swallowed CUDA API errors. The additional
    # cost is close to nothing.
    set(kExtraArgs --leak-check full --report-api-errors all)
    set(kTimeout 600)
    if(tool STREQUAL "racecheck")
      # By default racecheck accumulates analysis state until a block completes.
      # With kernels that use a lot of shared memory the analysis state hits its
      # limit and turns into a false report or a failure. Lowering the
      # synchronization granularity reduces the accumulation. This was required
      # on RTX Blackwell.
      list(APPEND kExtraArgs --force-synchronization-limit 1)
      # The fine-grained synchronization stretches the run time. Leave headroom
      # over the 565 s that was measured.
      set(kTimeout 1800)
    endif()
    add_test(NAME "sanitizer.${tool}.${target}"
      COMMAND "${kComputeSanitizer}" --tool "${tool}" --error-exitcode 1 ${kExtraArgs}
              "$<TARGET_FILE:${target}>" "--gtest_filter=${kSanitizerGtestFilter}")
    # Run time is longer under the sanitizer, so set the timeout individually.
    # Contending for the GPU stretches the time until it hits the timeout, so run
    # these serially.
    set_tests_properties("sanitizer.${tool}.${target}" PROPERTIES
      TIMEOUT ${kTimeout}
      RUN_SERIAL TRUE
      LABELS "sanitizer")
  endforeach()
endfunction()
