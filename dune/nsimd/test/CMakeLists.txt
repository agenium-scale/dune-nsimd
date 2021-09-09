include(DuneCMakeCompat)
include(DuneInstance)

set(MAX_VECTOR_SIZE 512)

set(SSE2_COMPILE_FLAGS) # implicit in x86_64
set(AVX_COMPILE_FLAGS -mavx)
set(AVX2_COMPILE_FLAGS -mavx -mavx2 -mfma)

#set(AVX512_COMPILE_FLAGS -march=skylake-avx512)
# The above is not supported on Ubuntu 16.04 yet so use the below explicit flags
set(AVX512_COMPILE_FLAGS
  -mavx -mavx2 -mfma
  -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl)

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

foreach(iset IN ITEMS sse2 avx avx2 avx512)
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

foreach(iset IN ITEMS sse2 avx avx2 avx512)
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
  COMPILE_FLAGS -msse2
  LINK_LIBRARIES dunecommon
  CMAKE_GUARD dune-istl_FOUND
)

dune_add_test(NAME multirhsperftest-jacobi-nsimd-4
  SOURCES multirhsperftest-jacobi-nsimd.cc
  COMPILE_DEFINITIONS -DMAX_VECTOR_SIZE=${MAX_VECTOR_SIZE}
  COMPILE_DEFINITIONS -DTYPE=Vec4d -DRUNS=${RUNS}
  COMPILE_FLAGS -mavx2 -mfma
  LINK_LIBRARIES dunecommon
  CMAKE_GUARD dune-istl_FOUND
)

dune_add_test(NAME multirhsperftest-jacobi-nsimd-8
  SOURCES multirhsperftest-jacobi-nsimd.cc
  COMPILE_DEFINITIONS -DMAX_VECTOR_SIZE=${MAX_VECTOR_SIZE}
  COMPILE_DEFINITIONS -DTYPE=Vec8d -DRUNS=${RUNS}
  COMPILE_FLAGS -mavx512f -mfma
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
  COMPILE_FLAGS -msse2
  LINK_LIBRARIES dunecommon
  CMAKE_GUARD dune-istl_FOUND
)

dune_add_test(NAME multirhsperftest-AMG-nsimd-4
  SOURCES multirhsperftest-AMG-nsimd.cc
  COMPILE_DEFINITIONS -DMAX_VECTOR_SIZE=${MAX_VECTOR_SIZE}
  COMPILE_DEFINITIONS -DTYPE=Vec4d -DRUNS=${RUNS}
  COMPILE_FLAGS -mavx2 -mfma
  LINK_LIBRARIES dunecommon
  CMAKE_GUARD dune-istl_FOUND
)

dune_add_test(NAME multirhsperftest-AMG-nsimd-8
  SOURCES multirhsperftest-AMG-nsimd.cc
  COMPILE_DEFINITIONS -DMAX_VECTOR_SIZE=${MAX_VECTOR_SIZE}
  COMPILE_DEFINITIONS -DTYPE=Vec8d -DRUNS=${RUNS}
  COMPILE_FLAGS -mavx512f -mfma
  LINK_LIBRARIES dunecommon
  CMAKE_GUARD dune-istl_FOUND
)
