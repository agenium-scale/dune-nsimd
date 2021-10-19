#ifndef VECTOR_TYPES_H
#define VECTOR_TYPES_H

#include "instrset.h"  // Select supported instruction set

#if defined(NSIMD_SSE2)
    #define nsimd::sse2 SIMD128_EXT
#elif defined(NSIMD_SSE42)
    #define nsimd::sse2 SIMD128_EXT
#elif defined(NSIMD_SSE42)
    #define nsimd::sse42 SIMD128_EXT
#elif defined(NSIMD_AARCH64)
    #define nsimd::aarch64 SIMD128_EXT
#elif defined(NSIMD_SVE128)
    #define nsimd::sve128 SIMD128_EXT
#elif defined(NSIMD_NEON128)
    #define nsimd::neon128 SIMD128_EXT
#else 
    #define nsimd::cpu SIMD128_EXT
#endif

#if defined(NSIMD_AVX)
    #define nsimd::avx SIMD256_EXT
#elif defined(NSIMD_AVX2)
    #define nsimd::avx2 SIMD256_EXT
#elif defined(NSIMD_SVE256)
    #define nsimd::sve256 SIMD256_EXT
#endif

#if defined(NSIMD_AVX512_KNL)
    #define nsimd::avx512_knl SIMD512_EXT
#elif defined(NSIMD_AVX512_SKYLAKE)
    #define nsimd::avx512_skylake SIMD512_EXT
#elif defined(NSIMD_SVE512)
    #define nsimd::sve512 SIMD512_EXT
#endif

/*****************************************************************************
 * 
 *  Vector 128 bits
 * 
******************************************************************************/

#if INSTRSET >= 2
    
    typedef nsimd::pack<signed char, 1, SIMD128_EXT>    pack128_16i_t;
    typedef nsimd::pack<unsigned char, 1, SIMD128_EXT>  pack128_16ui_t;
    typedef nsimd::packl<signed char, 1, SIMD128_EXT>   packl128_16i_t;

    typedef nsimd::pack<signed short, 1, SIMD128_EXT>   pack128_8i_t;
    typedef nsimd::pack<unsigned short, 1, SIMD128_EXT> pack128_8ui_t;
    typedef nsimd::packl<signed short, 1, SIMD128_EXT>  packl128_8i_t;

    typedef nsimd::pack<signed int, 1, SIMD128_EXT>     pack128_4i_t;
    typedef nsimd::pack<unsigned int, 1, SIMD128_EXT>   pack128_4ui_t;
    typedef nsimd::packl<signed int, 1, SIMD128_EXT>    packl128_4i_t;
    
    typedef nsimd::pack<signed long, 1, SIMD128_EXT>    pack128_2i_t;
    typedef nsimd::pack<unsigned long, 1, SIMD128_EXT>  pack128_2ui_t;
    typedef nsimd::packl<signed long, 1, SIMD128_EXT>   packl128_2i_t;
#endif


/*****************************************************************************
 * 
 *  Vector 256 bits
 * 
******************************************************************************/

#if INSTRSET >= 7
    typedef nsimd::pack<signed char, 1, SIMD256_EXT>   pack256_32i_t;
    typedef nsimd::pack<unsigned char, 1, SIMD256_EXT> pack256_32ui_t;
    typedef nsimd::packl<signed char, 1, SIMD256_EXT>  packl256_32i_t;

    typedef nsimd::pack<signed short, 1, SIMD256_EXT>   pack256_16i_t;
    typedef nsimd::pack<unsigned short, 1, SIMD256_EXT> pack256_16ui_t;
    typedef nsimd::packl<signed short, 1, SIMD256_EXT>  packl256_16i_t;

    typedef nsimd::pack<signed int, 1, SIMD256_EXT>     pack256_8i_t;
    typedef nsimd::pack<unsigned int, 1, SIMD256_EXT>   pack256_8ui_t;
    typedef nsimd::packl<signed int, 1, SIMD256_EXT>    packl256_8i_t;
    
    typedef nsimd::pack<signed long, 1, SIMD256_EXT>    pack256_4i_t;
    typedef nsimd::pack<unsigned long, 1, SIMD256_EXT>  pack256_4ui_t;
    typedef nsimd::packl<signed long, 1, SIMD256_EXT>   packl256_4i_t;
#endif

/*****************************************************************************
 * 
 *  Vector 512 bits
 * 
******************************************************************************/

#if INSTRSET >= 9
    typedef nsimd::pack<signed char, 1, SIMD512_EXT>    pack512_64i_t;
    typedef nsimd::pack<unsigned char, 1, SIMD512_EXT>  pack512_64ui_t;
    typedef nsimd::packl<signed char, 1, SIMD512_EXT>   packl512_64i_t;

    typedef nsimd::pack<signed short, 1, SIMD512_EXT>   pack512_32i_t;
    typedef nsimd::pack<unsigned short, 1, SIMD512_EXT> pack512_32ui_t;
    typedef nsimd::packl<signed short, 1, SIMD512_EXT>  packl512_32i_t;

    typedef nsimd::pack<signed int, 1, SIMD512_EXT>     pack512_16i_t;
    typedef nsimd::pack<unsigned int, 1, SIMD512_EXT>   pack512_16ui_t;
    typedef nsimd::packl<signed int, 1, SIMD512_EXT>    packl512_16i_t;
    
    typedef nsimd::pack<signed long, 1, SIMD512_EXT>    pack512_8i_t;
    typedef nsimd::pack<unsigned long, 1, SIMD512_EXT>  pack512_8ui_t;
    typedef nsimd::packl<signed long, 1, SIMD512_EXT>   packl512_8i_t;
#endif


// -------------------------------------------------------------------------------



/*****************************************************************************
 * 
 *  Float typedef
 * 
******************************************************************************/

// PACK4F
#if INSTRSET >= 2
    typedef nsimd::pack<float, 1, SIMD128_EXT> pack4f_t;
    typedef nsimd::packl<float, 1, SIMD128_EXT> packl4f_t;
#endif

// PACK8F
#if INSTRSET >= 7
    typedef nsimd::pack<float, 1, SIMD256_EXT> pack8f_t;
    typedef nsimd::packl<float, 1, SIMD256_EXT> packl8f_t;
#endif

// PACK16F 
#if INSTRSET >= 9
    typedef nsimd::pack<float, 1, SIMD512_EXT> pack16f_t;
    typedef nsimd::packl<float, 1, SIMD512_EXT> packl16f_t;
#endif

/*****************************************************************************
 * 
 *  Double typedef
 * 
******************************************************************************/

// PACK2D
#if INSTRSET >= 2
    typedef nsimd::pack<double, 1, SIMD128_EXT> pack2d_t;
    typedef nsimd::packl<double, 1, SIMD128_EXT> packl2d_t;
#endif

// PACK4D
#if INSTRSET >= 7
    typedef nsimd::pack<double, 1, SIMD256_EXT> pack4d_t;
    typedef nsimd::packl<double, 1, SIMD256_EXT> packl4d_t;
#endif

// PACK8D 
#if INSTRSET >= 9
    typedef nsimd::pack<double, 1, SIMD512_EXT> pack8d_t;
    typedef nsimd::packl<double, 1, SIMD512_EXT> packl8d_t;
#endif


#endif // VECTOR_TYPES_H