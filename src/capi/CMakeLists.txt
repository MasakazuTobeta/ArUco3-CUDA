# SPDX-License-Identifier: Apache-2.0
#
# The C ABI. Built as a shared library because that is what a foreign function
# interface can load; core, dictionary, and util stay static and are linked into
# it. Everything in the project is compiled position independent already, so
# there is no separate PIC build of the static libraries.
add_library(aruco3cuda_c SHARED aruco3cuda_c.cpp)
add_library(aruco3cuda::c ALIAS aruco3cuda_c)

target_include_directories(aruco3cuda_c
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

# PRIVATE, so that a consumer of the C ABI links one shared object and nothing
# else. The C header exposes no C++ type, so none of this has to propagate.
target_link_libraries(aruco3cuda_c
  PRIVATE aruco3cuda::core aruco3cuda::dictionary aruco3cuda::warnings aruco3cuda::coverage
          CUDA::cudart)

set_target_properties(aruco3cuda_c PROPERTIES
  OUTPUT_NAME aruco3cuda_c
  VERSION "${PROJECT_VERSION}"
  SOVERSION "${PROJECT_VERSION_MAJOR}"
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)

# CXX_VISIBILITY_PRESET only covers this target's own translation unit. The
# static libraries linked in were compiled without it, so their C++ symbols keep
# default visibility and are re-exported from the shared object unless the
# linker is told otherwise. Checked with nm -D: before this was added the object
# also exported the mangled aruco3cuda:: symbols of core and dictionary, down to
# Detector::Impl internals; afterwards it exports the 16 C functions and nothing
# else. That matters so a consumer cannot bind to an internal symbol that is
# free to change, and so that loading this alongside another copy of the same
# static libraries cannot cross-bind.
if(UNIX AND NOT APPLE)
  target_link_options(aruco3cuda_c PRIVATE "LINKER:--exclude-libs,ALL")
endif()
