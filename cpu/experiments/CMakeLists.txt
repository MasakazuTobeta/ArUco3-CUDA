# SPDX-License-Identifier: Apache-2.0
# Standalone experiment archive; never included by the production build.
cmake_minimum_required(VERSION 3.24)
project(aruco3cpu_experiments LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(OpenCV 4.14.0 EXACT REQUIRED COMPONENTS core imgproc imgcodecs objdetect)
get_filename_component(aruco3cpu_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

add_executable(aruco3cpu_dual_prune_probe
  dual_prune_probe.cpp
  "${aruco3cpu_root}/hybrid/cpu_candidates.cpp")
target_compile_features(aruco3cpu_dual_prune_probe PRIVATE cxx_std_17)
set_target_properties(aruco3cpu_dual_prune_probe PROPERTIES CXX_EXTENSIONS OFF)
target_include_directories(aruco3cpu_dual_prune_probe
  PRIVATE "${aruco3cpu_root}" "${aruco3cpu_root}/include")
target_include_directories(aruco3cpu_dual_prune_probe SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(aruco3cpu_dual_prune_probe PRIVATE ${OpenCV_LIBS})
target_compile_options(aruco3cpu_dual_prune_probe
  PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wold-style-cast -Werror)

add_executable(aruco3cpu_cache_stripe_probe
  cache_stripe_probe.cpp
  "${aruco3cpu_root}/cpu/cpu_threshold.cpp")
target_compile_features(aruco3cpu_cache_stripe_probe PRIVATE cxx_std_17)
set_target_properties(aruco3cpu_cache_stripe_probe PROPERTIES CXX_EXTENSIONS OFF)
target_include_directories(aruco3cpu_cache_stripe_probe PRIVATE "${aruco3cpu_root}/cpu")
target_include_directories(aruco3cpu_cache_stripe_probe SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(aruco3cpu_cache_stripe_probe PRIVATE ${OpenCV_LIBS})
target_compile_options(aruco3cpu_cache_stripe_probe
  PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wold-style-cast -Werror)

add_executable(aruco3cpu_count_first_probe count_first_probe.cpp count_first_contours.cpp)
target_compile_features(aruco3cpu_count_first_probe PRIVATE cxx_std_17)
target_include_directories(aruco3cpu_count_first_probe PRIVATE "${aruco3cpu_root}/include")
target_include_directories(aruco3cpu_count_first_probe SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(aruco3cpu_count_first_probe PRIVATE ${OpenCV_LIBS})
target_compile_options(aruco3cpu_count_first_probe
  PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wold-style-cast -Werror)

add_executable(aruco3cpu_generate_high_id_corpus
  generate_high_id_corpus.cpp
  "${aruco3cpu_root}/src/util/sha256.cpp"
  "${aruco3cpu_root}/src/util/json_writer.cpp")
target_compile_features(aruco3cpu_generate_high_id_corpus PRIVATE cxx_std_17)
target_include_directories(aruco3cpu_generate_high_id_corpus
  PRIVATE "${aruco3cpu_root}/include")
target_include_directories(aruco3cpu_generate_high_id_corpus SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(aruco3cpu_generate_high_id_corpus PRIVATE ${OpenCV_LIBS})
target_compile_options(aruco3cpu_generate_high_id_corpus
  PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wold-style-cast -Werror)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/generate_high_id_corpus.cpp")
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/generate_high_id_corpus.cpp" highid_source_sha256)
target_compile_definitions(aruco3cpu_generate_high_id_corpus PRIVATE
  ARUCO3CPU_HIGHID_SOURCE_SHA256="${highid_source_sha256}")
