# SPDX-License-Identifier: Apache-2.0
#
# Purpose:
#   Make the project installable as a library. Without install rules, consumers
#   have no choice but to add_subdirectory the source tree, which is not a state
#   that deserves to be called a "library".
#
# Scope:
#   Only the three targets that do not depend on OpenCV (core, dictionary, util)
#   and the public headers. hybrid, reference, bench, and tools are instruments
#   for evaluation and measurement, not artifacts we ship to consumers, so they
#   are not installed.
function(aruco3cuda_add_install_rules)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)

  # Align the export names with the in-tree aliases. By default the NAMESPACE is
  # prepended to the target name, yielding aruco3cuda::aruco3cuda_core, which
  # contradicts the aruco3cuda::core shown in the documentation and makes the
  # installed library unusable through find_package.
  set_target_properties(aruco3cuda_core PROPERTIES EXPORT_NAME core)
  set_target_properties(aruco3cuda_dictionary PROPERTIES EXPORT_NAME dictionary)
  set_target_properties(aruco3cuda_util PROPERTIES EXPORT_NAME util)
  set_target_properties(aruco3cuda_warnings PROPERTIES EXPORT_NAME warnings)
  set_target_properties(aruco3cuda_coverage PROPERTIES EXPORT_NAME coverage)

  # warnings and coverage are INTERFACE targets and produce no artifacts, but
  # linking them PRIVATE into a STATIC library leaves them on the interface as
  # $<LINK_ONLY:>, so install(EXPORT) fails unless they are part of the export
  # set.
  install(TARGETS aruco3cuda_core aruco3cuda_dictionary aruco3cuda_util
                  aruco3cuda_warnings aruco3cuda_coverage
    EXPORT aruco3cudaTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")

  install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/aruco3cuda"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING PATTERN "*.hpp")

  install(EXPORT aruco3cudaTargets
    FILE aruco3cudaTargets.cmake
    NAMESPACE aruco3cuda::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/aruco3cuda")

  # Consumers must resolve the CUDA Toolkit as well. Without a find_dependency
  # call in the config file, find_package(aruco3cuda) succeeds but linking
  # fails.
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/aruco3cudaConfig.cmake.in"
[[@PACKAGE_INIT@
include(CMakeFindDependencyMacro)
find_dependency(CUDAToolkit)
include("${CMAKE_CURRENT_LIST_DIR}/aruco3cudaTargets.cmake")
check_required_components(aruco3cuda)
]])

  configure_package_config_file(
    "${CMAKE_CURRENT_BINARY_DIR}/aruco3cudaConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/aruco3cudaConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/aruco3cuda")

  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/aruco3cudaConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMinorVersion)

  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/aruco3cudaConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/aruco3cudaConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/aruco3cuda")

  install(FILES "${PROJECT_SOURCE_DIR}/LICENSE" "${PROJECT_SOURCE_DIR}/NOTICE"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/aruco3cuda")
endfunction()
