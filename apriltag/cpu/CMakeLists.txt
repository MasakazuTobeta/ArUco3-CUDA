# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.24)
project(aruco3_apriltag_cpu LANGUAGES C CXX)
include(FetchContent)
include(GNUInstallDirs)
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
