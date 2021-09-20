/****************************  instrset.h   **********************************
* Author:        Agner Fog
* Date created:  2012-05-30
* Last modified: 2016-11-25
* Version:       1.25
* Project:       vector classes
* Description:
* Header file for various compiler-specific tasks and other common tasks to 
* vector class library:
* > selects the supported instruction set
* > defines integer types
* > defines compiler version macros
* > undefines certain macros that prevent function overloading
* > defines template class to represent compile-time integer constant
* > defines template for compile-time error messages
*
* (c) Copyright 2012-2016 GNU General Public License www.gnu.org/licenses
******************************************************************************/

#ifndef INSTRSET_H
#define INSTRSET_H 125

// Detect 64 bit mode
#if (defined(_M_AMD64) || defined(_M_X64) || defined(__amd64) ) && ! defined(__x86_64__)
#define __x86_64__ 1  // There are many different macros for this, decide on only one
#endif

// Find instruction set from compiler macros if INSTRSET not defined
// Note: Most of these macros are not defined in Microsoft compilers
#ifndef INSTRSET
#if defined ( NSIMD_AVX512_KNL ) || defined ( NSIMD_AVX512_SKYLAKE )
#define INSTRSET 9
#elif defined ( NSIMD_AVX2 )
#define INSTRSET 8
#elif defined ( NSIMD_AVX )
#define INSTRSET 7
#elif defined ( NSIMD_SSE42 )
#define INSTRSET 6
#elif defined ( NSIMD_SSE2 ) || defined ( __x86_64__ )
#define INSTRSET 2
#else 
#define INSTRSET 0
#endif // instruction set defines
#endif // INSTRSET

// Include the appropriate header file for intrinsic functions
#include <nsimd/cxx_adv_api.hpp>
#include <nsimd/cxx_adv_api_functions.hpp>
#include <nsimd/nsimd.h>

#if INSTRSET >= 8 && !defined(__FMA__)
// Assume that all processors that have AVX2 also have FMA3
#if defined (__GNUC__) && ! defined (__INTEL_COMPILER) && ! defined (__clang__)
// Prevent error message in g++ when using FMA intrinsics with avx2:
#pragma message "It is recommended to specify also option -mfma when using -mavx2 or higher"
#else
#define __FMA__  1
#endif
#endif

// Define integer types with known size
#if defined(__GNUC__) || defined(__clang__) || (defined(_MSC_VER) && _MSC_VER >= 1600)
  // Compilers supporting C99 or C++0x have stdint.h defining these integer types
  #include <stdint.h>
#elif defined(_MSC_VER)
  // Older Microsoft compilers have their own definitions
  typedef signed   __int8  int8_t;
  typedef unsigned __int8  uint8_t;
  typedef signed   __int16 int16_t;
  typedef unsigned __int16 uint16_t;
  typedef signed   __int32 int32_t;
  typedef unsigned __int32 uint32_t;
  typedef signed   __int64 int64_t;
  typedef unsigned __int64 uint64_t;
  #ifndef _INTPTR_T_DEFINED
    #define _INTPTR_T_DEFINED
    #ifdef  __x86_64__
      typedef int64_t intptr_t;
    #else
      typedef int32_t intptr_t;
    #endif
  #endif
#else
  // This works with most compilers
  typedef signed   char      int8_t;
  typedef unsigned char      uint8_t;
  typedef signed   short int int16_t;
  typedef unsigned short int uint16_t;
  typedef signed   int       int32_t;
  typedef unsigned int       uint32_t;
  typedef long long          int64_t;
  typedef unsigned long long uint64_t;
  #ifdef  __x86_64__
    typedef int64_t intptr_t;
  #else
    typedef int32_t intptr_t;
  #endif
#endif

#include <stdlib.h>                              // define abs(int)

#ifdef _MSC_VER                                  // Microsoft compiler or compatible Intel compiler
#include <intrin.h>                              // define _BitScanReverse(int), __cpuid(int[4],int), _xgetbv(int)
#endif // _MSC_VER

// functions in instrset_detect.cpp
#ifdef NSIMD_NAMESPACE
namespace NSIMD_NAMESPACE {
#endif
    int  instrset_detect(void);                      // tells which instruction sets are supported
    bool hasFMA3(void);                              // true if FMA3 instructions supported
    bool hasFMA4(void);                              // true if FMA4 instructions supported
    bool hasXOP(void);                               // true if XOP  instructions supported
    bool hasAVX512ER(void);                          // true if AVX512ER instructions supported
#ifdef NSIMD_NAMESPACE
}
#endif

// GCC version
#if defined(__GNUC__) && !defined (GCC_VERSION) && !defined (__clang__)
#define GCC_VERSION  ((__GNUC__) * 10000 + (__GNUC_MINOR__) * 100 + (__GNUC_PATCHLEVEL__))
#endif

// Clang version
#if defined (__clang__)
#define CLANG_VERSION  ((__clang_major__) * 10000 + (__clang_minor__) * 100 + (__clang_patchlevel__))
// Problem: The version number is not consistent across platforms
// http://llvm.org/bugs/show_bug.cgi?id=12643
// Apple bug 18746972
#endif

// Fix problem with non-overloadable macros named min and max in WinDef.h
#ifdef _MSC_VER
#if defined (_WINDEF_) && defined(min) && defined(max)
#undef min
#undef max
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#ifdef NSIMD_NAMESPACE
namespace NSIMD_NAMESPACE {
#endif
    // Template class to represent compile-time integer constant
    template <int32_t  n> class Const_int_t {};       // represent compile-time signed integer constant
    template <uint32_t n> class Const_uint_t {};      // represent compile-time unsigned integer constant
    #define const_int(n)  (Const_int_t <n>())         // n must be compile-time integer constant
    #define const_uint(n) (Const_uint_t<n>())         // n must be compile-time unsigned integer constant

    // Template for compile-time error messages
    template <bool> class Static_error_check {
    public:  Static_error_check() {};
    };
    template <> class Static_error_check<false> {     // generate compile-time error if false
    private: Static_error_check() {};
    };
#ifdef NSIMD_NAMESPACE
}
#endif 


#endif // INSTRSET_H
