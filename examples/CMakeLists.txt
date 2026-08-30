# SPDX-License-Identifier: Apache-2.0
#
# Purpose:
#   Build the runnable samples of the public API. This file is written to work
#   both ways round so that the second form, the one a real consumer uses, is
#   exercised rather than described.
#
# In tree, as part of this project:
#   cmake --preset native && cmake --build --preset native
#   The binaries land in <build dir>/examples/.
#
# Standalone, against an installed package:
#   cmake --install <build dir> --prefix /your/prefix
#   cmake -S examples -B build/examples -DCMAKE_PREFIX_PATH=/your/prefix
#   cmake --build build/examples
#   Nothing here refers to the source tree, so this is the same path a consumer
#   takes. If find_package or the exported target names break, this form fails
#   while the in-tree build keeps working, which is why CI runs both.
#
# Note:
#   Only CXX is enabled. The samples call the CUDA runtime but compile no device
#   code, so a consumer of this library does not need the CUDA language enabled
#   in their own project.
cmake_minimum_required(VERSION 3.24)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(aruco3cuda_examples LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  find_package(aruco3cuda REQUIRED)
  set(kBuiltInTree OFF)
else()
  set(kBuiltInTree ON)
endif()

# pgm.cpp is compiled into each sample rather than shared through a library. It
# is scaffolding for reading and writing files, not something a consumer is meant
# to take, and a reader should be able to see the whole of one sample by opening
# two files.
function(aruco3cuda_add_example target output_name source)
  add_executable(${target} ${source} pgm.cpp)
  set_target_properties(${target} PROPERTIES OUTPUT_NAME ${output_name})
  target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(${target} PRIVATE ${ARGN})
  if(kBuiltInTree)
    # In tree the samples are held to the same warnings as the rest of the code,
    # so sample code cannot rot into a warning nobody sees. These are internal
    # targets of this project, so a consumer copying this file does not want
    # them, and standalone builds leave them out.
    target_link_libraries(${target} PRIVATE aruco3cuda::warnings aruco3cuda::coverage)
  endif()
endfunction()

# Needs the dictionary only. It performs no detection, so it runs on a machine
# with no GPU.
aruco3cuda_add_example(aruco3cuda_example_generate_marker generate_marker
  generate_marker.cpp
  aruco3cuda::dictionary)

# Needs a GPU at run time. CUDA::cudart arrives through aruco3cuda::core, which
# links it PUBLIC because the public header includes cuda_runtime_api.h.
aruco3cuda_add_example(aruco3cuda_example_detect_image detect_image
  detect_image.cpp
  aruco3cuda::core aruco3cuda::dictionary)
