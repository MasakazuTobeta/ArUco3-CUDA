# SPDX-License-Identifier: Apache-2.0
#
# Purpose:
#   Stage the Python package next to the shared object it loads, so that a build
#   tree is usable with nothing but PYTHONPATH:
#
#     export PYTHONPATH=<build dir>/python
#     python3 -c "import aruco3cuda; print(aruco3cuda.version())"
#
# Why a copy rather than a symlink to the source tree:
#   _binding.py looks for libaruco3cuda_c.so beside itself first. Keeping the
#   package and the shared object in one directory is what makes that work, and
#   it also means the staged tree is self-contained: it can be copied elsewhere
#   without dragging the source tree along.
set(kPythonPackageDir "${CMAKE_BINARY_DIR}/python/aruco3cuda")
set(kPythonSources
  "${CMAKE_CURRENT_SOURCE_DIR}/aruco3cuda/__init__.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/aruco3cuda/_binding.py")

# The stamp exists so the copy reruns when a source changes and not otherwise.
# A custom target alone would rerun on every build.
add_custom_command(
  OUTPUT "${kPythonPackageDir}/.stamp"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${kPythonPackageDir}"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${kPythonSources} "${kPythonPackageDir}"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different "$<TARGET_FILE:aruco3cuda_c>"
          "${kPythonPackageDir}/libaruco3cuda_c.so"
  COMMAND "${CMAKE_COMMAND}" -E touch "${kPythonPackageDir}/.stamp"
  DEPENDS ${kPythonSources} aruco3cuda_c
  COMMENT "Stage the Python package into ${kPythonPackageDir}"
  VERBATIM)

add_custom_target(aruco3cuda_python ALL DEPENDS "${kPythonPackageDir}/.stamp")

# Exported so that the tests and anything else needing it agree on one path
# instead of each rebuilding it from CMAKE_BINARY_DIR.
set(ARUCO3CUDA_PYTHON_PATH "${CMAKE_BINARY_DIR}/python" CACHE INTERNAL
    "Directory to put on PYTHONPATH to import the staged aruco3cuda package")
