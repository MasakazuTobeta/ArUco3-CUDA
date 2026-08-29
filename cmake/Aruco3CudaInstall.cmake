# SPDX-License-Identifier: Apache-2.0
#
# 目的:
#   library として install できるようにする。install rule が無いと、利用者は
#   source tree を add_subdirectory するしかなく、「library」と呼べる状態に
#   ならない。
#
# 対象:
#   OpenCV へ依存しない 3 target (core、dictionary、util) と公開 header のみ。
#   hybrid、reference、bench、tools は評価と測定の道具であり、利用者へ配る
#   ものではないため install しない。
function(aruco3cuda_add_install_rules)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)

  # export 名を in-tree の alias と揃える。既定では NAMESPACE が target 名へ
  # 前置され aruco3cuda::aruco3cuda_core となり、文書が示す aruco3cuda::core と
  # 食い違う。install した library を find_package で使えなくなる。
  set_target_properties(aruco3cuda_core PROPERTIES EXPORT_NAME core)
  set_target_properties(aruco3cuda_dictionary PROPERTIES EXPORT_NAME dictionary)
  set_target_properties(aruco3cuda_util PROPERTIES EXPORT_NAME util)
  set_target_properties(aruco3cuda_warnings PROPERTIES EXPORT_NAME warnings)
  set_target_properties(aruco3cuda_coverage PROPERTIES EXPORT_NAME coverage)

  # warnings と coverage は INTERFACE target であり成果物を持たないが、
  # STATIC library へ PRIVATE link すると $<LINK_ONLY:> として interface へ
  # 残るため、export set に含めないと install(EXPORT) が通らない。
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

  # 利用者側でも CUDA Toolkit の解決が要る。config 側で find_dependency を
  # 呼ばないと、find_package(aruco3cuda) が通っても link で落ちる。
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
