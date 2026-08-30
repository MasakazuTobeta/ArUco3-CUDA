# SPDX-License-Identifier: Apache-2.0
#
# Synthetic corpus generator. Produces images with four-corner ground truth from a fixed seed.
add_library(aruco3cuda_corpusgen STATIC corpus_generator.cpp)
add_library(aruco3cuda::corpusgen ALIAS aruco3cuda_corpusgen)

target_include_directories(aruco3cuda_corpusgen PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(aruco3cuda_corpusgen
  PUBLIC aruco3cuda::util
  PRIVATE aruco3cuda::warnings aruco3cuda::coverage ${OpenCV_LIBS})
target_include_directories(aruco3cuda_corpusgen SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})

add_executable(aruco3cuda_corpusgen_cli main.cpp)
set_target_properties(aruco3cuda_corpusgen_cli PROPERTIES OUTPUT_NAME aruco3cuda_corpusgen)
target_link_libraries(aruco3cuda_corpusgen_cli
  PRIVATE aruco3cuda::corpusgen aruco3cuda::warnings aruco3cuda::coverage)
