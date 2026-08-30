# SPDX-License-Identifier: Apache-2.0
#
# Packed representation of the predefined dictionaries and the matching code.
# Everything under generated/ is source produced by tools/dictgen and must not be
# edited by hand.
add_library(aruco3cuda_dictionary
  dictionary.cpp
  registry.cpp
  generated/dict_aruco_mip_36h12.cpp)

add_library(aruco3cuda::dictionary ALIAS aruco3cuda_dictionary)

target_include_directories(aruco3cuda_dictionary
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(aruco3cuda_dictionary PRIVATE aruco3cuda::warnings aruco3cuda::coverage)

set_target_properties(aruco3cuda_dictionary PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME aruco3cuda_dictionary)
