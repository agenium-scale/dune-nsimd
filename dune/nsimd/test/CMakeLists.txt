include(DuneCMakeCompat)
include(DuneInstance)

set(MAX_VECTOR_SIZE 512)

# Intel
set(SSE2_COMPILE_FLAGS "-DSSE2 -msse2")
set(AVX_COMPILE_FLAGS "-DAVX -mavx")
set(AVX2_COMPILE_FLAGS "-DAVX2 -mavx2 -DFMA -mfma -DFP16 -mf16c")
set(AVX512_COMPILE_FLAGS "-DAVX512_SKYLAKE -mavx512f -mavx512dq -mavx512cd -mavx512bw -mavx512vl")

# Arm 32 bits
set(NEON128_COMPILE_FLAGS "-DNEON128 -mfpu=neon")

# Arm 64 bits
set(AARCH64_COMPILE_FLAGS "-DAARCH64")
set(SVE128_COMPILE_FLAGS "-DSVE128 -march=armv8.2-a+sve -msve-vector-bits=128")
set(SVE256_COMPILE_FLAGS "-DSVE256 -march=armv8.2-a+sve -msve-vector-bits=256")
set(SVE512_COMPILE_FLAGS "-DSVE512 -march=armv8.2-a+sve -msve-vector-bits=512")

# List of architectures
set(LIST_X86 "sse2;avx;avx2;avx512")
set(LIST_AARCH64 "aarch64;sve128;sve256;sve512")

if ("${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
  set(LIST_SIMD_EXTS "${LIST_X86}")
  set(SIMD_EXT_FOR_VEC2D "${SSE2_COMPILE_FLAGS}")
  set(SIMD_EXT_FOR_VEC4D "${AVX2_COMPILE_FLAGS}")
  set(SIMD_EXT_FOR_VEC8D "${AVX512_COMPILE_FLAGS}")
else()
  set(LIST_SIMD_EXTS "${LIST_AARCH64}")
  set(SIMD_EXT_FOR_VEC2D "${AARCH64_COMPILE_FLAGS}")
  set(SIMD_EXT_FOR_VEC4D "${SVE256_COMPILE_FLAGS}")
  set(SIMD_EXT_FOR_VEC8D "${SVE512_COMPILE_FLAGS}")
endif()

######################################################################
#
# nsimdtest
#

dune_instance_begin(FILES nsimdtest.hh nsimdtest.cc)

foreach(VECTOR_SIZE IN ITEMS 128 256 512)
  # float and associated types
  math(EXPR LANES "${VECTOR_SIZE} / 32")
  foreach(TYPE IN ITEMS f i)
    dune_instance_add(ID "${LANES}${TYPE}")
    foreach(POINT IN ITEMS
        Type
        BinaryOpsScalarVector BinaryOpsVectorScalar
        BinaryOpsProxyVector BinaryOpsVectorProxy)
      dune_instance_add(TEMPLATE POINT ID "${POINT}_${LANES}${TYPE}"
        FILES nsimdtest_vector.cc nsimdtest_mask.cc)
    endforeach(POINT)
  endforeach(TYPE)

  # double and associated types
  math(EXPR LANES "${VECTOR_SIZE} / 64")
  foreach(TYPE IN ITEMS d q)
    dune_instance_add(ID "${POINT}_${LANES}${TYPE}")
    foreach(POINT IN ITEMS
        Type
        BinaryOpsScalarVector BinaryOpsVectorScalar
        BinaryOpsProxyVector BinaryOpsVectorProxy)
      dune_instance_add(TEMPLATE POINT ID "${POINT}_${LANES}${TYPE}"
        FILES nsimdtest_vector.cc nsimdtest_mask.cc)
    endforeach(POINT)
  endforeach(TYPE)
endforeach(VECTOR_SIZE)

dune_instance_end()

dune_list_filter(DUNE_INSTANCE_GENERATED INCLUDE REGEX [[\.cc$]])

foreach(iset IN ITEMS ${LIST_SIMD_EXTS})
  string(TOUPPER "${iset}" ISET)

  dune_add_test(NAME nsimdtest-${iset}
    SOURCES ${DUNE_INSTANCE_GENERATED}
    COMPILE_DEFINITIONS -DMAX_VECTOR_SIZE=${MAX_VECTOR_SIZE}
    COMPILE_FLAGS ${${ISET}_COMPILE_FLAGS}
    LINK_LIBRARIES dunecommon
  )
endforeach(iset)

######################################################################
#
# multirhstest
#

dune_instance_begin(FILES multirhstest.hh multirhstest.cc)

foreach(VECTOR_SIZE IN ITEMS 128 256 512)
  # float and associated types
  math(EXPR LANES "${VECTOR_SIZE} / 32")
  set(TYPE f)
  dune_instance_add(ID "${LANES}${TYPE}"
    FILES multirhstest_vector.cc)

  # double and associated types
  math(EXPR LANES "${VECTOR_SIZE} / 64")
  set(TYPE d)
  dune_instance_add(ID "${LANES}${TYPE}"
    FILES multirhstest_vector.cc)
endforeach(VECTOR_SIZE)

dune_instance_end()

dune_list_filter(DUNE_INSTANCE_GENERATED INCLUDE REGEX [[\.cc$]])

foreach(iset IN ITEMS ${LIST_SIMD_EXTS})
  string(TOUPPER "${iset}" ISET)

  dune_add_test(NAME multirhstest-${iset}
    SOURCES ${DUNE_INSTANCE_GENERATED}
    COMPILE_DEFINITIONS -DMAX_VECTOR_SIZE=${MAX_VECTOR_SIZE}
    COMPILE_FLAGS ${${ISET}_COMPILE_FLAGS}
    LINK_LIBRARIES dunecommon
    CMAKE_GUARD dune-istl_FOUND
  )
endforeach(iset)

######################################################################
#
# further tests
#

set(RUNS 10)

dune_add_test(NAME multirhsperftest-jacobi-scalar
  SOURCES multirhsperftest-jacobi-scalar.cc
  COMPILE_DEFINITIONS -DRUNS=${RUNS}
  LINK_LIBRARIES dunecommon
  CMAKE_GUARD dune-istl_FOUND
)

foreach(LANES IN ITEMS 2 4 8)
  dune_add_test(NAME multirhsperftest-jacobi-loop-${LANES}
    SOURCES multirhsperftest-jacobi-loop.cc
    COMPILE_DEFINITIONS -DLANES=${LANES} -DRUNS=${RUNS}
    LINK_LIBRARIES dunecommon
    CMAKE_GUARD dune-istl_FOUND
  )
endforeach(LANES)

dune_add_test(NAME multirhsperftest-jacobi-nsimd-2
  SOURCES multirhsperftest-jacobi-nsimd.cc
  COMPILE_DEFINITIONS -DMAX_VECTOR_SIZE=${MAX_VECTOR_SIZE}
  COMPILE_DEFINITIONS -DTYPE=Vec2d -DRUNS=${RUNS}
  COMPILE_FLAGS "${SIMD_EXT_FOR_VEC2D}"
  LINK_LIBRARIES dunecommon
  CMAKE_GUARD dune-istl_FOUND
)

dune_add_test(NAME multirhsperftest-jacobi-nsimd-4
  SOURCES multirhsperftest-jacobi-nsimd.cc
  COMPILE_DEFINITIONS -DMAX_VECTOR_SIZE=${MAX_VECTOR_SIZE}
  COMPILE_DEFINITIONS -DTYPE=Vec4d -DRUNS=${RUNS}
  COMPILE_FLAGS "${SIMD_EXT_FOR_VEC4D}"
  LINK_LIBRARIES dunecommon
  CMAKE_GUARD dune-istl_FOUND
)

dune_add_test(NAME multirhsperftest-jacobi-nsimd-8
  SOURCES multirhsperftest-jacobi-nsimd.cc
  COMPILE_DEFINITIONS -DMAX_VECTOR_SIZE=${MAX_VECTOR_SIZE}
  COMPILE_DEFINITIONS -DTYPE=Vec8d -DRUNS=${RUNS}
  COMPILE_FLAGS "${SIMD_EXT_FOR_VEC8D}"
  LINK_LIBRARIES dunecommon
  CMAKE_GUARD dune-istl_FOUND
)

set(RUNS 10)

dune_add_test(NAME multirhsperftest-AMG-scalar
  SOURCES multirhsperftest-AMG-scalar.cc
  COMPILE_DEFINITIONS -DRUNS=${RUNS}
  LINK_LIBRARIES dunecommon
  CMAKE_GUARD dune-istl_FOUND
)

foreach(LANES IN ITEMS 2 4 8)
  dune_add_test(NAME multirhsperftest-AMG-loop-${LANES}
    SOURCES multirhsperftest-AMG-loop.cc
    COMPILE_DEFINITIONS -DLANES=${LANES} -DRUNS=${RUNS}
    LINK_LIBRARIES dunecommon
    CMAKE_GUARD dune-istl_FOUND
  )
endforeach(LANES)

dune_add_test(NAME multirhsperftest-AMG-nsimd-2
  SOURCES multirhsperftest-AMG-nsimd.cc
  COMPILE_DEFINITIONS -DMAX_VECTOR_SIZE=${MAX_VECTOR_SIZE}
  COMPILE_DEFINITIONS -DTYPE=Vec2d -DRUNS=${RUNS}
  COMPILE_FLAGS "${SIMD_EXT_FOR_VEC2D}"
  LINK_LIBRARIES dunecommon
  CMAKE_GUARD dune-istl_FOUND
)

dune_add_test(NAME multirhsperftest-AMG-nsimd-4
  SOURCES multirhsperftest-AMG-nsimd.cc
  COMPILE_DEFINITIONS -DMAX_VECTOR_SIZE=${MAX_VECTOR_SIZE}
  COMPILE_DEFINITIONS -DTYPE=Vec4d -DRUNS=${RUNS}
  COMPILE_FLAGS "${SIMD_EXT_FOR_VEC4D}"
  LINK_LIBRARIES dunecommon
  CMAKE_GUARD dune-istl_FOUND
)

dune_add_test(NAME multirhsperftest-AMG-nsimd-8
  SOURCES multirhsperftest-AMG-nsimd.cc
  COMPILE_DEFINITIONS -DMAX_VECTOR_SIZE=${MAX_VECTOR_SIZE}
  COMPILE_DEFINITIONS -DTYPE=Vec8d -DRUNS=${RUNS}
  COMPILE_FLAGS "${SIMD_EXT_FOR_VEC8D}"
  LINK_LIBRARIES dunecommon
  CMAKE_GUARD dune-istl_FOUND
)
