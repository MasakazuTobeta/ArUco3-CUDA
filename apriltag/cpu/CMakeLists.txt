# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.24)
project(aruco3_apriltag_cpu LANGUAGES C CXX)
include(FetchContent)
include(GNUInstallDirs)
if(PROJECT_IS_TOP_LEVEL)
  include(CTest)
endif()

function(replace_apriltag_source_fragment source_var before after)
  string(REPLACE "${before}" "${after}" rewritten "${${source_var}}")
  if(rewritten STREQUAL "${${source_var}}")
    message(FATAL_ERROR "The pinned AprilTag source no longer contains: ${before}")
  endif()
  set("${source_var}" "${rewritten}" PARENT_SCOPE)
endfunction()

function(add_apriltag_dependency)
  set(BUILD_TESTING OFF)
  set(BUILD_EXAMPLES OFF)
  set(BUILD_PYTHON_WRAPPER OFF)
  set(BUILD_SHARED_LIBS OFF)
  FetchContent_Declare(apriltag3
    URL https://codeload.github.com/AprilRobotics/apriltag/tar.gz/efa633d9dc6f4ee33bb79b0592250f4aa71e90a8
    URL_HASH SHA256=dbf3288760ab07c757c192a94ab9b1e5c043bb6ab285d76b538acc04ca15f53e
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(apriltag3)
  # Install only the project-facing API. Upstream CMake defines its own install tree.
  set_property(DIRECTORY "${apriltag3_SOURCE_DIR}" PROPERTY EXCLUDE_FROM_ALL TRUE)
  set_target_properties(apriltag PROPERTIES C_VISIBILITY_PRESET hidden)

  # Use the same quad extractor with a compact union-find representation.
  # Keep the fetched source and any caller-provided offline source immutable.
  get_target_property(apriltag_sources apriltag SOURCES)
  foreach(upstream_name IN ITEMS apriltag_quad_thresh.c apriltag.c)
    list(FIND apriltag_sources "${upstream_name}" source_index)
    if(source_index LESS 0)
      message(FATAL_ERROR "The pinned AprilTag source list has changed")
    endif()
    set(upstream_source "${apriltag3_SOURCE_DIR}/${upstream_name}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${upstream_source}")
    file(READ "${upstream_source}" project_code)
    if(upstream_name STREQUAL "apriltag_quad_thresh.c")
      replace_apriltag_source_fragment(project_code
        "#include \"common/unionfind.h\"" "#include \"quad_union_find.h\"")
      replace_apriltag_source_fragment(project_code
        [=[unionfind_t *uf = unionfind_create(w * h);]=]
        [=[unionfind_t *uf = unionfind_create(w * h);
    if (uf == NULL)
        return NULL;]=])
      replace_apriltag_source_fragment(project_code
        [=[unionfind_t* uf = connected_components(td, threshim, w, h, ts);]=]
        [=[unionfind_t* uf = connected_components(td, threshim, w, h, ts);
    if (uf == NULL) {
        image_u8_destroy(threshim);
        return NULL;
    }]=])
      # Labeling has finished. Gradient workers only look up immutable roots.
      replace_apriltag_source_fragment(project_code
        "unionfind_get_representative(uf, y*w + x)" "unionfind_find_read_only(uf, y*w + x)")
      replace_apriltag_source_fragment(project_code
        "unionfind_get_representative(uf, (y + dy)*w + x + dx)"
        "unionfind_find_read_only(uf, (y + dy)*w + x + dx)")
      replace_apriltag_source_fragment(project_code
        "unionfind_get_set_size(uf, rep0)" "unionfind_root_size(uf, rep0)")
      replace_apriltag_source_fragment(project_code
        "unionfind_get_set_size(uf, rep1)" "unionfind_root_size(uf, rep1)")
    else()
      replace_apriltag_source_fragment(project_code
        [=[zarray_t *quads = apriltag_quad_thresh(td, quad_im);]=]
        [=[zarray_t *quads = apriltag_quad_thresh(td, quad_im);
    if (quads == NULL) {
        if (quad_im != im_orig)
            image_u8_destroy(quad_im);
        errno = ENOMEM;
        return NULL;
    }]=])
    endif()
    set(project_source "${CMAKE_CURRENT_BINARY_DIR}/${upstream_name}")
    file(CONFIGURE OUTPUT "${project_source}" CONTENT "${project_code}" @ONLY)
    list(REMOVE_AT apriltag_sources "${source_index}")
    list(INSERT apriltag_sources "${source_index}" "${project_source}")
  endforeach()
  set_property(TARGET apriltag PROPERTY SOURCES "${apriltag_sources}")
  target_include_directories(apriltag PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
endfunction()
add_apriltag_dependency()

add_library(aruco3_apriltag_cpu SHARED detector.cpp)
add_library(aruco3cuda::apriltag_cpu ALIAS aruco3_apriltag_cpu)
set_target_properties(aruco3_apriltag_cpu PROPERTIES EXPORT_NAME apriltag_cpu)
target_compile_features(aruco3_apriltag_cpu PUBLIC cxx_std_17)
target_include_directories(aruco3_apriltag_cpu PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_link_libraries(aruco3_apriltag_cpu PRIVATE apriltag)

if((PROJECT_IS_TOP_LEVEL AND BUILD_TESTING) OR ARUCO3CUDA_BUILD_TESTS)
  add_executable(apriltag_cpu_quad_union_find_test test_quad_union_find.c)
  target_compile_features(apriltag_cpu_quad_union_find_test PRIVATE c_std_99)
  add_test(NAME apriltag_cpu_quad_union_find COMMAND apriltag_cpu_quad_union_find_test)
endif()

option(ARUCO3_APRILTAG_CPU_COMPARE "Build the OpenCV correctness and timing comparison" OFF)
if(ARUCO3_APRILTAG_CPU_COMPARE)
  find_package(OpenCV 4.14.0 EXACT CONFIG REQUIRED COMPONENTS core imgproc objdetect imgcodecs)
  add_executable(apriltag_cpu_compare compare.cpp)
  target_link_libraries(apriltag_cpu_compare PRIVATE aruco3_apriltag_cpu ${OpenCV_LIBS})
  enable_testing()
  add_test(NAME apriltag_cpu_compare COMMAND apriltag_cpu_compare --check)
  add_test(NAME apriltag_cpu_compare.parallel COMMAND apriltag_cpu_compare --check --threads 4)
  find_package(Python3 QUIET COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    add_test(NAME apriltag_cpu_compare.observations COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/test_observe.py" $<TARGET_FILE:apriltag_cpu_compare>)
  endif()
endif()

if(PROJECT_IS_TOP_LEVEL)
  install(TARGETS aruco3_apriltag_cpu EXPORT AprilTagCpuTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../include/aruco3cuda/apriltag/cpu"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/aruco3cuda/apriltag")
  install(EXPORT AprilTagCpuTargets NAMESPACE aruco3cuda::
    FILE aruco3_apriltag_cpuConfig.cmake
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/aruco3_apriltag_cpu")
  install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/../../LICENSE" "${CMAKE_CURRENT_SOURCE_DIR}/../../NOTICE"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/aruco3_apriltag_cpu")
endif()
