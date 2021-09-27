# We require version CMake version 3.1 to prevent issues
# with dune_enable_all_packages and older CMake versions.
cmake_minimum_required(VERSION 3.1)
project(dune-nsimd CXX)

if(NOT (dune-common_DIR OR dune-common_ROOT OR
      "${CMAKE_PREFIX_PATH}" MATCHES ".*dune-common.*"))
    string(REPLACE  ${CMAKE_PROJECT_NAME} dune-common dune-common_DIR
      ${PROJECT_BINARY_DIR})
endif()

#find dune-common and set the module path
find_package(dune-common REQUIRED)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules"
  ${dune-common_MODULE_PATH})

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/scripts")
find_package(NSIMD COMPONENTS "${SIMD_EXT}" REQUIRED)

# Intel 
set(SSE2_COMPILE_FLAGS "-DSSE2 -msse2")
set(SSE42_COMPILE_FLAGS "-DSSE42 -msse4.2")
set(AVX_COMPILE_FLAGS "-DAVX -mavx")
set(AVX2_COMPILE_FLAGS "-DAVX2 -mavx2 -DFMA -mfma -DFP16 -mf16c")
set(AVX512_SKYLAKE_COMPILE_FLAGS "-DAVX512_SKYLAKE -mavx512f -mavx512dq -mavx512cd -mavx512bw -mavx512vl -mfma -DFP16 -mf16c")
set(AVX512_SKYLAKE_COMPILE_FLAGS "-DAVX512_KNL -mavx512f -mavx512pf -mavx512er -mavx512cd -mfma -DFP16 -mf16c")

# Arm
set(AARCH64_COMPILE_FLAGS "-DAARCH64 -mfp16-format=ieee -march=armv8.2-a+fp16")
set(NEON64_COMPILE_FLAGS "-DNEON64 -mfpu=neon")
set(NEON128_COMPILE_FLAGS "-DNEON128 -mfpu=neon")
set(SVE128_COMPILE_FLAGS "-DSVE128 -march=armv8.2-a+sve -msve-vector-bits=128")
set(SVE256_COMPILE_FLAGS "-DSVE256 -march=armv8.2-a+sve -msve-vector-bits=256")
set(SVE512_COMPILE_FLAGS "-DSVE512 -march=armv8.2-a+sve -msve-vector-bits=512")
set(SVE1024_COMPILE_FLAGS "-DSVE1024 -march=armv8.2-a+sve -msve-vector-bits=1024")
set(SVE2048_COMPILE_FLAGS "-DSVE2048 -march=armv8.2-a+sve -msve-vector-bits=2048")

set(SIMD_EXT_LIST "SSE2" "SSE42" "AVX" "AVX2" "AVX512_SKYLAKE" "AVX512_KNL" "AARCH64" "NEON64" "NEON128" "SVE128" "SVE256" "SVE512" "SVE1024" "SVE2048")
set(simd_index -1)
if (NOT ${SIMD_EXT} STREQUAL "")
  string(TOUPPER ${SIMD_EXT} simd_ext)
  list (FIND SIMD_EXT_LIST ${simd_ext} simd_index)
  if(${simd_index} GREATER -1)
    set (SIMD_COMPILE_FLAGS "${${simd_ext}_COMPILE_FLAGS}" )
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${${simd_ext}_COMPILE_FLAGS}" )
  endif()
endif()
message(STATUS "JE SUIS LAAAAAAAA : ${CMAKE_CXX_FLAGS} | ${SIMD_EXT} | ${simd_ext}")

#include the dune macros
include(DuneMacros)

# start a dune project with information from dune.module
dune_project()

dune_enable_all_packages()

add_subdirectory(dune)
add_subdirectory(doc)
add_subdirectory(cmake/modules)

# finalize the dune project, e.g. generating config.h etc.
finalize_dune_project(GENERATE_CONFIG_H_CMAKE)
