/****************************  vectorf256.h   *******************************
* Author:        Agner Fog
* Date created:  2012-05-30
* Last modified: 2017-07-27
* Version:       1.30
* Project:       vector classes
* Description:
* Header file defining 256-bit floating point vector classes as interface
* to intrinsic functions in x86 microprocessors with AVX instruction set.
*
* Instructions:
* Use Gnu, Intel or Microsoft C++ compiler.
*
* The following vector classes are defined here:
* Vec4f     Vector of 4 single precision floating point numbers
* Vec4fb    Vector of 4 Booleans for use with Vec8f
* Vec8f     Vector of 8 single precision floating point numbers
* Vec8fb    Vector of 8 Booleans for use with Vec8f
* Vec16f     Vector of 16 single precision floating point numbers
* Vec16fb    Vector of 16 Booleans for use with Vec8f

* Vec2d     Vector of 4 double precision floating point numbers
* Vec2db    Vector of 4 Booleans for use with Vec4d
* Vec4d     Vector of 4 double precision floating point numbers
* Vec4db    Vector of 4 Booleans for use with Vec4d
* Vec8d     Vector of 4 double precision floating point numbers
* Vec8db    Vector of 4 Booleans for use with Vec4d
*
* Each vector object is represented internally in the CPU as a 256-bit register.
* This header file defines operators and functions for these vectors.
*
* For example:
* Vec4d a(1., 2., 3., 4.), b(5., 6., 7., 8.), c;
* c = a + b;     // now c contains (6., 8., 10., 12.)
*
* For detailed instructions, see Nsimd.pdf
*
* (c) Copyright 2012-2017 GNU General Public License http://www.gnu.org/licenses
*****************************************************************************/

// check combination of header files
#if defined (VECTORF_H)
#if    VECTORF_H != 2
#error Two different versions of vectorf.h included
#endif
#else
#define VECTORF_H  2

#ifdef NSIMD_NAMESPACE
namespace NSIMD_NAMESPACE {
#endif

/*****************************************************************************
 * 
 *  Float typedef
 * 
******************************************************************************/

// PACK4F
#if defined(NSIMD_SSE2)
typedef nsimd::pack<float, 1, nsimd::sse2> pack4f_t;
typedef nsimd::packl<float, 1, nsimd::sse2> packl4f_t;
#elif defined(NSIMD_SSE42)
typedef nsimd::pack<float, 1, nsimd::sse42> pack4f_t;
typedef nsimd::packl<float, 1, nsimd::sse42> packl4f_t;
#else
#error "No SIMD register that can store 4 float"
#endif

// PACK8F
#if defined(NSIMD_AVX)
typedef nsimd::pack<float, 1, nsimd::avx> pack8f_t;
typedef nsimd::packl<float, 1, nsimd::avx> packl8f_t;
#elif defined(NSIMD_AVX2)
typedef nsimd::pack<float, 1, nsimd::avx2> pack8f_t;
typedef nsimd::packl<float, 1, nsimd::avx2> packl8f_t;
#else
#error "No SIMD register that can store 8 float"
#endif

// PACK16F 
#if defined(NSIMD_AVX512_KNL)
typedef nsimd::pack<float, 1, nsimd::avx512_knl> pack16f_t;
typedef nsimd::packl<float, 1, nsimd::avx512_knl> packl16f_t;
#elif defined(NSIMD_AVX512_SKYLAKE)
typedef nsimd::pack<float, 1, nsimd::avx512_skylake> pack16f_t;
typedef nsimd::packl<float, 1, nsimd::avx512_skylake> packl16f_t;
#else
#error "No SIMD register that can store 16 float"
#endif

/*****************************************************************************
 * 
 *  Double typedef
 * 
******************************************************************************/

// PACK2D
#if defined(NSIMD_SSE2)
typedef nsimd::pack<double, 1, nsimd::sse2> pack2d_t;
typedef nsimd::packl<double, 1, nsimd::sse2> packl2d_t;
#elif defined(NSIMD_SSE42)
typedef nsimd::pack<double, 1, nsimd::sse42> pack2d_t;
typedef nsimd::packl<double, 1, nsimd::sse42> packl2d_t;
#else
#error "No SIMD register that can store 2 double"
#endif

// PACK4D
#if defined(NSIMD_AVX)
typedef nsimd::pack<double, 1, nsimd::avx> pack4d_t;
typedef nsimd::packl<double, 1, nsimd::avx> packl4d_t;
#elif defined(NSIMD_AVX2)
typedef nsimd::pack<double, 1, nsimd::avx2> pack4d_t;
typedef nsimd::packl<double, 1, nsimd::avx2> packl4d_t;
#else
#error "No SIMD register that can store 4 double"
#endif

// PACK8D 
#if defined(NSIMD_AVX512_KNL)
typedef nsimd::pack<double, 1, nsimd::avx512_knl> pack8d_t;
typedef nsimd::packl<double, 1, nsimd::avx512_knl> packl8d_t;
#elif defined(NSIMD_AVX512_SKYLAKE)
typedef nsimd::pack<double, 1, nsimd::avx512_skylake> pack8d_t;
typedef nsimd::packl<double, 1, nsimd::avx512_skylake> packl8d_t;
#else
#error "No SIMD register that can store 8 double"
#endif

/*****************************************************************************
*
*          select functions
*
*****************************************************************************/
// Select between two pack of float sources, element by element. Used in various functions 
// and operators. Corresponds to this pseudocode:
// for (int i = 0; i < size; i++) result[i] = s[i] ? a[i] : b[i];
// Each element in s must be either 0 (false) or 0xFFFFFFFF (true).
static inline pack4f_t selectf (pack4f_t const & s, pack4f_t const & a, pack4f_t const & b) {
    return nsimd::if_else1 (b, a, s);
}

#if defined(NSIMD_AVX) || defined(NSIMD_AVX2)
static inline pack8f_t selectf (pack8f_t const & s, pack8f_t const & a, pack8f_t const & b) {
    return nsimd::if_else1 (b, a, s);
}
#if defined(NSIMD_AVX512_KNL) || defined(NSIMD_AVX512_SKYLAKE)
static inline pack16f_t selectf (pack16f_t const & s, pack16f_t const & a, pack16f_t const & b) {
    return nsimd::if_else1 (b, a, s);
}
#endif
#endif

// Same, with two pack of double sources.
// and operators. Corresponds to this pseudocode:
// for (int i = 0; i < size; i++) result[i] = s[i] ? a[i] : b[i];
// Each element in s must be either 0 (false) or 0xFFFFFFFFFFFFFFFF (true). No other 
// values are allowed.
static inline pack2d_t selectd (pack2d_t const & s, pack2d_t const & a, pack2d_t const & b) {
    return nsimd::if_else1 (b, a, s);
}

#if defined(NSIMD_AVX) || defined(NSIMD_AVX2)
static inline pack4d_t selectd (pack4d_t const & s, pack4d_t const & a, pack4d_t const & b) {
    return nsimd::if_else1 (b, a, s);
}
#if defined(NSIMD_AVX512_KNL) || defined(NSIMD_AVX512_SKYLAKE)
static inline pack8d_t selectd (pack8d_t const & s, pack8d_t const & a, pack8d_t const & b) {
    return nsimd::if_else1 (b, a, s);
}
#endif
#endif


/*****************************************************************************
*
*          Generate compile-time constant vector
*
*****************************************************************************/
// Generate a constant vector of 8 integers stored in memory,
// load as pack8f_t
template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
static inline pack8f_t constant8f() {
    static const union {
        int     i[8];
        pack8f_t  ymm;
    } u = {{i0,i1,i2,i3,i4,i5,i6,i7}};
    return u.ymm;
}


/*****************************************************************************
*
*         Join two 128-bit vectors
*
*****************************************************************************/
#define set_m128r(lo,hi) _mm256_insertf128_ps(_mm256_castps128_ps256(lo),(hi),1)
    // _mm256_set_m128(hi,lo); // not defined in all versions of immintrin.h


/*****************************************************************************
*
*          Vec8fb: Vector of 8 Booleans for use with Vec8f
*
*****************************************************************************/

class Vec8fb {
protected:
    packl8f_t ymm; // Float vector
public:
    // Default constructor:
    Vec8fb() {
    }
    // Constructor to build from all elements:
    Vec8fb(bool b0, bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7) {
        bool data[8] = {-(int)b0, -(int)b1, -(int)b2, -(int)b3, -(int)b4, -(int)b5, -(int)b6, -(int)b7)};
        ymm = _mm256_castsi256_ps(_mm256_setr_epi32(-(int)b0, -(int)b1, -(int)b2, -(int)b3, -(int)b4, -(int)b5, -(int)b6, -(int)b7)); 
        //ymm = nsimd::reinterpret_i32_f32();
    }
    // Constructor to build from two Vec4fb:
    Vec8fb(Vec4fb const & a0, Vec4fb const & a1) {
        ymm = set_m128r(a0, a1);
    }
    // Constructor to convert from type __m256 used in intrinsics:
    Vec8fb(pack8f_t const & x) {
        ymm = x;
    }
    // Assignment operator to convert from type __m256 used in intrinsics:
    Vec8fb & operator = (packl8f_t const & x) {
        ymm = x;
        return *this;
    }
    // Constructor to broadcast the same value into all elements:
    Vec8fb(bool b) {
#if INSTRSET >= 8  // AVX2
        ymm = _mm256_castsi256_ps(_mm256_set1_epi32(-(int)b));
#else
        __m128 b1 = _mm_castsi128_ps(nsimd::set1<nsimd::pack<int>>(-(int)b));
        //ymm = _mm256_set_m128(b1,b1);
        ymm = set_m128r(b1,b1);
#endif
    }
    // Assignment operator to broadcast scalar value:
    Vec8fb & operator = (bool b) {
        *this = Vec8fb(b);
        return *this;
    }
private: // Prevent constructing from int, etc.
    Vec8fb(int b);
    Vec8fb & operator = (int x);
public:
    // Type cast operator to convert to __m256 used in intrinsics
    operator packl8f_t() const {
        return ymm;
    }
#if defined (VECTORI_H)
    // Constructor to convert from type Vec8ib used as Boolean for integer vectors
    Vec8fb(Vec8ib const & x) {
        ymm = _mm256_castsi256_ps(x);
    }
    // Assignment operator to convert from type Vec8ib used as Boolean for integer vectors
    Vec8fb & operator = (Vec8ib const & x) {
        ymm = _mm256_castsi256_ps(x);
        return *this;
    }
    // Type cast operator to convert to type Vec8ib used as Boolean for integer vectors
    operator Vec8ib() const {
        return _mm256_castps_si256(ymm);
    }
#endif // VECTORI_H
    // Member function to change a single element in vector
    // Note: This function is inefficient. Use load function if changing more than one element
    Vec8fb const & insert(uint32_t index, bool value) {
        static const int32_t maskl[16] = {0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0};
        packl8f_t mask  = nsimd::loadu((float const*)(maskl+8-(index & 7))); // mask with FFFFFFFF at index position
        if (value) {
            ymm = nsimd::orb(ymm,mask);
        }
        else {
            ymm = nsimd::andnotb(mask,ymm);
        }
        return *this;
    }
    // Member function extract a single element from vector
    bool extract(uint32_t index) const {
        union {
            float   f[8];
            int32_t i[8];
        } u;
        nsimd::storeu(u.f, ymm);
        return u.i[index & 7] != 0;
    }
    // Extract a single element. Operator [] can only read an element, not write.
    bool operator [] (uint32_t index) const {
        return extract(index);
    }
    // Member functions to split into two Vec4fb:
    Vec4fb get_low() const {
        return _mm256_castps256_ps128(ymm);
    }
    Vec4fb get_high() const {
        return _mm256_extractf128_ps(ymm,1);
    }
    static int size () {
        return 8;
    }
};


/*****************************************************************************
*
*          Operators for Vec8fb
*
*****************************************************************************/

// vector operator & : bitwise and
static inline Vec8fb operator & (Vec8fb const & a, Vec8fb const & b) {
    return nsimd::andb(a, b);
}
static inline Vec8fb operator && (Vec8fb const & a, Vec8fb const & b) {
    return a & b;
}

// vector operator &= : bitwise and
static inline Vec8fb & operator &= (Vec8fb & a, Vec8fb const & b) {
    a = a & b;
    return a;
}

// vector operator | : bitwise or
static inline Vec8fb operator | (Vec8fb const & a, Vec8fb const & b) {
    return nsimd::orb(a, b);
}
static inline Vec8fb operator || (Vec8fb const & a, Vec8fb const & b) {
    return a | b;
}

// vector operator |= : bitwise or
static inline Vec8fb & operator |= (Vec8fb & a, Vec8fb const & b) {
    a = a | b;
    return a;
}

// vector operator ^ : bitwise xor
static inline Vec8fb operator ^ (Vec8fb const & a, Vec8fb const & b) {
    return nsimd::xorb(a, b);
}

// vector operator ^= : bitwise xor
static inline Vec8fb & operator ^= (Vec8fb & a, Vec8fb const & b) {
    a = a ^ b;
    return a;
}

// vector operator ~ : bitwise not
static inline Vec8fb operator ~ (Vec8fb const & a) {
    return nsimd::xorb(a, constant8f<-1,-1,-1,-1,-1,-1,-1,-1>());
}

// vector operator ! : logical not
// (operator ! is less efficient than operator ~. Use only where not
// all bits in an element are the same)
static inline Vec8fb operator ! (Vec8fb const & a) {
    return Vec8fb( !Vec8ib(a));
}

// Functions for Vec8fb

// andnot: a & ~ b
static inline Vec8fb andnot(Vec8fb const & a, Vec8fb const & b) {
    return nsimd::andnotb(b, a);
}

/*****************************************************************************
*
*          Vec16fb: Vector of 16 Booleans for use with Vec16f
*
*****************************************************************************/
class Vec16fb : public Vec16b {
public:
    // Default constructor:
    Vec16fb () {
    }
    Vec16fb (Vec16b x) {
        m16 = x;
    }
    // Constructor to build from all elements:
    Vec16fb(bool x0, bool x1, bool x2, bool x3, bool x4, bool x5, bool x6, bool x7,
        bool x8, bool x9, bool x10, bool x11, bool x12, bool x13, bool x14, bool x15) :
        Vec16b(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15) {
    }
    // Constructor to convert from type __mmask16 used in intrinsics:
    Vec16fb (__mmask16 x) {
        m16 = x;
    }
    // Constructor to broadcast single value:
    Vec16fb(bool b) : Vec16b(b) {}
private: // Prevent constructing from int, etc.
    Vec16fb(int b);
public:
    // Constructor to make from two halves
    Vec16fb (Vec8fb const & x0, Vec8fb const & x1) {
        m16 = Vec16b(Vec8ib(x0), Vec8ib(x1));
    }
    // Assignment operator to convert from type __mmask16 used in intrinsics:
    Vec16fb & operator = (__mmask16 x) {
        m16 = x;
        return *this;
    }
    // Assignment operator to broadcast scalar value:
    Vec16fb & operator = (bool b) {
        m16 = Vec16b(b);
        return *this;
    }
private: // Prevent assigning int because of ambiguity
    Vec16fb & operator = (int x);
public:
};

// Define operators for Vec16fb

// vector operator & : bitwise and
static inline Vec16fb operator & (Vec16fb a, Vec16fb b) {
    return Vec16b(a) & Vec16b(b);
}
static inline Vec16fb operator && (Vec16fb a, Vec16fb b) {
    return a & b;
}

// vector operator | : bitwise or
static inline Vec16fb operator | (Vec16fb a, Vec16fb b) {
    return Vec16b(a) | Vec16b(b);
}
static inline Vec16fb operator || (Vec16fb a, Vec16fb b) {
    return a | b;
}

// vector operator ^ : bitwise xor
static inline Vec16fb operator ^ (Vec16fb a, Vec16fb b) {
    return Vec16b(a) ^ Vec16b(b);
}

// vector operator ~ : bitwise not
static inline Vec16fb operator ~ (Vec16fb a) {
    return ~Vec16b(a);
}

// vector operator ! : element not
static inline Vec16fb operator ! (Vec16fb a) {
    return ~a;
}

// vector operator &= : bitwise and
static inline Vec16fb & operator &= (Vec16fb & a, Vec16fb b) {
    a = a & b;
    return a;
}

// vector operator |= : bitwise or
static inline Vec16fb & operator |= (Vec16fb & a, Vec16fb b) {
    a = a | b;
    return a;
}

// vector operator ^= : bitwise xor
static inline Vec16fb & operator ^= (Vec16fb & a, Vec16fb b) {
    a = a ^ b;
    return a;
}


/*****************************************************************************
*
*          Vec8db: Vector of 8 Booleans for use with Vec8d
*
*****************************************************************************/

class Vec8db : public Vec8b {
public:
    // Default constructor:
    Vec8db () {
    }
    Vec8db (Vec16b x) {
        m16 = x;
    }
    // Constructor to build from all elements:
    Vec8db(bool x0, bool x1, bool x2, bool x3, bool x4, bool x5, bool x6, bool x7) :
        Vec8b(x0, x1, x2, x3, x4, x5, x6, x7) {
    }
    // Constructor to convert from type __mmask8 used in intrinsics:
    Vec8db (__mmask8 x) {
        m16 = x;
    }
    // Constructor to convert from type __mmask16 used in intrinsics:
    Vec8db (__mmask16 x) {
        m16 = x;
    }
    // Constructor to build from two halves
    Vec8db (Vec4db const & x0, Vec4db const & x1) {
        m16 = Vec8qb(Vec4qb(x0), Vec4qb(x1));
    }
    // Assignment operator to convert from type __mmask8 used in intrinsics:
    Vec8db & operator = (__mmask8 x) {
        m16 = (__mmask16)x;
        return *this;
    }
    // Assignment operator to convert from type __mmask16 used in intrinsics:
    Vec8db & operator = (__mmask16 x) {
        m16 = x;
        return *this;
    }
    // Constructor to broadcast single value:
    Vec8db(bool b) : Vec8b(b) {}
    // Assignment operator to broadcast scalar:
    Vec8db & operator = (bool b) {
        m16 = Vec8b(b);
        return *this;
    }
private: // Prevent constructing from int, etc.
    Vec8db(int b);
    Vec8db & operator = (int x);
public:
    static int size () {
        return 8;
    }
};

// Define operators for Vec8db

// vector operator & : bitwise and
static inline Vec8db operator & (Vec8db a, Vec8db b) {
    return Vec16b(a) & Vec16b(b);
}
static inline Vec8db operator && (Vec8db a, Vec8db b) {
    return a & b;
}

// vector operator | : bitwise or
static inline Vec8db operator | (Vec8db a, Vec8db b) {
    return Vec16b(a) | Vec16b(b);
}
static inline Vec8db operator || (Vec8db a, Vec8db b) {
    return a | b;
}

// vector operator ^ : bitwise xor
static inline Vec8db operator ^ (Vec8db a, Vec8db b) {
    return Vec16b(a) ^ Vec16b(b);
}

// vector operator ~ : bitwise not
static inline Vec8db operator ~ (Vec8db a) {
    return ~Vec16b(a);
}

// vector operator ! : element not
static inline Vec8db operator ! (Vec8db a) {
    return ~a;
}

// vector operator &= : bitwise and
static inline Vec8db & operator &= (Vec8db & a, Vec8db b) {
    a = a & b;
    return a;
}

// vector operator |= : bitwise or
static inline Vec8db & operator |= (Vec8db & a, Vec8db b) {
    a = a | b;
    return a;
}

// vector operator ^= : bitwise xor
static inline Vec8db & operator ^= (Vec8db & a, Vec8db b) {
    a = a ^ b;
    return a;
}



/*****************************************************************************
*
*          Vec16f: Vector of 16 single precision floating point values
*
*****************************************************************************/

class Vec16f {
protected:
    __m512 zmm; // Float vector
public:
    // Default constructor:
    Vec16f() {
    }
    // Constructor to broadcast the same value into all elements:
    Vec16f(float f) {
        zmm = _mm512_set1_ps(f);
    }
    // Constructor to build from all elements:
    Vec16f(float f0, float f1, float f2, float f3, float f4, float f5, float f6, float f7,
    float f8, float f9, float f10, float f11, float f12, float f13, float f14, float f15) {
        zmm = _mm512_setr_ps(f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15); 
    }
    // Constructor to build from two Vec8f:
    Vec16f(Vec8f const & a0, Vec8f const & a1) {
        zmm = _mm512_castpd_ps(_mm512_insertf64x4(_mm512_castps_pd(_mm512_castps256_ps512(a0)), _mm256_castps_pd(a1), 1));
    }
    // Constructor to convert from type __m512 used in intrinsics:
    Vec16f(__m512 const & x) {
        zmm = x;
    }
    // Assignment operator to convert from type __m512 used in intrinsics:
    Vec16f & operator = (__m512 const & x) {
        zmm = x;
        return *this;
    }
    // Type cast operator to convert to __m512 used in intrinsics
    operator __m512() const {
        return zmm;
    }
    // Member function to load from array (unaligned)
    Vec16f & load(float const * p) {
        zmm = _mm512_loadu_ps(p);
        return *this;
    }
    // Member function to load from array, aligned by 64
    // You may use load_a instead of load if you are certain that p points to an address
    // divisible by 64.
    Vec16f & load_a(float const * p) {
        zmm = _mm512_load_ps(p);
        return *this;
    }
    // Member function to store into array (unaligned)
    void store(float * p) const {
        _mm512_storeu_ps(p, zmm);
    }
    // Member function to store into array, aligned by 64
    // You may use store_a instead of store if you are certain that p points to an address
    // divisible by 64.
    void store_a(float * p) const {
        _mm512_store_ps(p, zmm);
    }
    // Partial load. Load n elements and set the rest to 0
    Vec16f & load_partial(int n, float const * p) {
        zmm = _mm512_maskz_loadu_ps(__mmask16((1 << n) - 1), p);
        return *this;
    }
    // Partial store. Store n elements
    void store_partial(int n, float * p) const {
        _mm512_mask_storeu_ps(p, __mmask16((1 << n) - 1), zmm);
    }
    // cut off vector to n elements. The last 8-n elements are set to zero
    Vec16f & cutoff(int n) {
        zmm = _mm512_maskz_mov_ps(__mmask16((1 << n) - 1), zmm);
        return *this;
    }
    // Member function to change a single element in vector
    Vec16f const & insert(uint32_t index, float value) {
        //zmm = _mm512_mask_set1_ps(zmm, __mmask16(1 << index), value);  // this intrinsic function does not exist (yet?)
        zmm = _mm512_castsi512_ps(_mm512_mask_set1_epi32(_mm512_castps_si512(zmm), __mmask16(1 << index), *(int32_t*)&value));  // ignore warning
        return *this;
    }
    // Member function extract a single element from vector
    float extract(uint32_t index) const {
        float a[16];
        store(a);
        return a[index & 15];
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    float operator [] (uint32_t index) const {
        return extract(index);
    }
    // Member functions to split into two Vec4f:
    Vec8f get_low() const {
        return _mm512_castps512_ps256(zmm);
    }
    Vec8f get_high() const {
        return _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(zmm),1));
    }
    static int size () {
        return 16;
    }
};


/*****************************************************************************
*
*          Operators for Vec16f
*
*****************************************************************************/

// vector operator + : add element by element
static inline Vec16f operator + (Vec16f const & a, Vec16f const & b) {
    return _mm512_add_ps(a, b);
}

// vector operator + : add vector and scalar
static inline Vec16f operator + (Vec16f const & a, float b) {
    return a + Vec16f(b);
}
static inline Vec16f operator + (float a, Vec16f const & b) {
    return Vec16f(a) + b;
}

// vector operator += : add
static inline Vec16f & operator += (Vec16f & a, Vec16f const & b) {
    a = a + b;
    return a;
}

// postfix operator ++
static inline Vec16f operator ++ (Vec16f & a, int) {
    Vec16f a0 = a;
    a = a + 1.0f;
    return a0;
}

// prefix operator ++
static inline Vec16f & operator ++ (Vec16f & a) {
    a = a + 1.0f;
    return a;
}

// vector operator - : subtract element by element
static inline Vec16f operator - (Vec16f const & a, Vec16f const & b) {
    return _mm512_sub_ps(a, b);
}

// vector operator - : subtract vector and scalar
static inline Vec16f operator - (Vec16f const & a, float b) {
    return a - Vec16f(b);
}
static inline Vec16f operator - (float a, Vec16f const & b) {
    return Vec16f(a) - b;
}

// vector operator - : unary minus
// Change sign bit, even for 0, INF and NAN
static inline Vec16f operator - (Vec16f const & a) {
    return _mm512_castsi512_ps(Vec16i(_mm512_castps_si512(a)) ^ 0x80000000);
}

// vector operator -= : subtract
static inline Vec16f & operator -= (Vec16f & a, Vec16f const & b) {
    a = a - b;
    return a;
}

// postfix operator --
static inline Vec16f operator -- (Vec16f & a, int) {
    Vec16f a0 = a;
    a = a - 1.0f;
    return a0;
}

// prefix operator --
static inline Vec16f & operator -- (Vec16f & a) {
    a = a - 1.0f;
    return a;
}

// vector operator * : multiply element by element
static inline Vec16f operator * (Vec16f const & a, Vec16f const & b) {
    return _mm512_mul_ps(a, b);
}

// vector operator * : multiply vector and scalar
static inline Vec16f operator * (Vec16f const & a, float b) {
    return a * Vec16f(b);
}
static inline Vec16f operator * (float a, Vec16f const & b) {
    return Vec16f(a) * b;
}

// vector operator *= : multiply
static inline Vec16f & operator *= (Vec16f & a, Vec16f const & b) {
    a = a * b;
    return a;
}

// vector operator / : divide all elements by same integer
static inline Vec16f operator / (Vec16f const & a, Vec16f const & b) {
    return _mm512_div_ps(a, b);
}

// vector operator / : divide vector and scalar
static inline Vec16f operator / (Vec16f const & a, float b) {
    return a / Vec16f(b);
}
static inline Vec16f operator / (float a, Vec16f const & b) {
    return Vec16f(a) / b;
}

// vector operator /= : divide
static inline Vec16f & operator /= (Vec16f & a, Vec16f const & b) {
    a = a / b;
    return a;
}

// vector operator == : returns true for elements for which a == b
static inline Vec16fb operator == (Vec16f const & a, Vec16f const & b) {
//    return _mm512_cmpeq_ps_mask(a, b);
    return _mm512_cmp_ps_mask(a, b, 0);
}

// vector operator != : returns true for elements for which a != b
static inline Vec16fb operator != (Vec16f const & a, Vec16f const & b) {
//    return _mm512_cmpneq_ps_mask(a, b);
    return _mm512_cmp_ps_mask(a, b, 4);
}

// vector operator < : returns true for elements for which a < b
static inline Vec16fb operator < (Vec16f const & a, Vec16f const & b) {
//    return _mm512_cmplt_ps_mask(a, b);
    return _mm512_cmp_ps_mask(a, b, 1);

}

// vector operator <= : returns true for elements for which a <= b
static inline Vec16fb operator <= (Vec16f const & a, Vec16f const & b) {
//    return _mm512_cmple_ps_mask(a, b);
    return _mm512_cmp_ps_mask(a, b, 2);
}

// vector operator > : returns true for elements for which a > b
static inline Vec16fb operator > (Vec16f const & a, Vec16f const & b) {
    return b < a;
}

// vector operator >= : returns true for elements for which a >= b
static inline Vec16fb operator >= (Vec16f const & a, Vec16f const & b) {
    return b <= a;
}

// Bitwise logical operators

// vector operator & : bitwise and
static inline Vec16f operator & (Vec16f const & a, Vec16f const & b) {
    return _mm512_castsi512_ps(Vec16i(_mm512_castps_si512(a)) & Vec16i(_mm512_castps_si512(b)));
}

// vector operator &= : bitwise and
static inline Vec16f & operator &= (Vec16f & a, Vec16f const & b) {
    a = a & b;
    return a;
}

// vector operator & : bitwise and of Vec16f and Vec16fb
static inline Vec16f operator & (Vec16f const & a, Vec16fb const & b) {
    return _mm512_maskz_mov_ps(b, a);
}
static inline Vec16f operator & (Vec16fb const & a, Vec16f const & b) {
    return b & a;
}

// vector operator | : bitwise or
static inline Vec16f operator | (Vec16f const & a, Vec16f const & b) {
    return _mm512_castsi512_ps(Vec16i(_mm512_castps_si512(a)) | Vec16i(_mm512_castps_si512(b)));
}

// vector operator |= : bitwise or
static inline Vec16f & operator |= (Vec16f & a, Vec16f const & b) {
    a = a | b;
    return a;
}

// vector operator ^ : bitwise xor
static inline Vec16f operator ^ (Vec16f const & a, Vec16f const & b) {
    return _mm512_castsi512_ps(Vec16i(_mm512_castps_si512(a)) ^ Vec16i(_mm512_castps_si512(b)));
}

// vector operator ^= : bitwise xor
static inline Vec16f & operator ^= (Vec16f & a, Vec16f const & b) {
    a = a ^ b;
    return a;
}

// vector operator ! : logical not. Returns Boolean vector
static inline Vec16fb operator ! (Vec16f const & a) {
    return a == Vec16f(0.0f);
}


/*****************************************************************************
*
*          Functions for Vec16f
*
*****************************************************************************/

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 8; i++) result[i] = s[i] ? a[i] : b[i];
// Each byte in s must be either 0 (false) or 0xFFFFFFFF (true). No other values are allowed.
static inline Vec16f select (Vec16fb const & s, Vec16f const & a, Vec16f const & b) {
    return _mm512_mask_mov_ps(b, s, a);
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
static inline Vec16f if_add (Vec16fb const & f, Vec16f const & a, Vec16f const & b) {
    return _mm512_mask_add_ps(a, f, a, b);
}

// Conditional multiply: For all vector elements i: result[i] = f[i] ? (a[i] * b[i]) : a[i]
static inline Vec16f if_mul (Vec16fb const & f, Vec16f const & a, Vec16f const & b) {
    return _mm512_mask_mul_ps(a, f, a, b);
}

// Horizontal add: Calculates the sum of all vector elements.
static inline float horizontal_add (Vec16f const & a) {
#if defined(__INTEL_COMPILER)
    return _mm512_reduce_add_ps(a);
#else
    return horizontal_add(a.get_low() + a.get_high());
#endif
}

// function max: a > b ? a : b
static inline Vec16f max(Vec16f const & a, Vec16f const & b) {
    return _mm512_max_ps(a,b);
}

// function min: a < b ? a : b
static inline Vec16f min(Vec16f const & a, Vec16f const & b) {
    return _mm512_min_ps(a,b);
}

// function abs: absolute value
// Removes sign bit, even for -0.0f, -INF and -NAN
static inline Vec16f abs(Vec16f const & a) {
    union {
        int32_t i;
        float   f;
    } u = {0x7FFFFFFF};
    return a & Vec16f(u.f);
}

// function sqrt: square root
static inline Vec16f sqrt(Vec16f const & a) {
    return _mm512_sqrt_ps(a);
}

// function square: a * a
static inline Vec16f square(Vec16f const & a) {
    return a * a;
}

// pow(Vec16f, int):
template <typename TT> static Vec16f pow(Vec16f const & a, TT const & n);

// Raise floating point numbers to integer power n
template <>
inline Vec16f pow<int>(Vec16f const & x0, int const & n) {
    return pow_template_i<Vec16f>(x0, n);
}

// allow conversion from unsigned int
template <>
inline Vec16f pow<uint32_t>(Vec16f const & x0, uint32_t const & n) {
    return pow_template_i<Vec16f>(x0, (int)n);
}


// Raise floating point numbers to integer power n, where n is a compile-time constant
template <int n>
static inline Vec16f pow_n(Vec16f const & a) {
    if (n < 0)    return Vec16f(1.0f) / pow_n<-n>(a);
    if (n == 0)   return Vec16f(1.0f);
    if (n >= 256) return pow(a, n);
    Vec16f x = a;                      // a^(2^i)
    Vec16f y;                          // accumulator
    const int lowest = n - (n & (n-1));// lowest set bit in n
    if (n & 1) y = x;
    if (n < 2) return y;
    x = x*x;                           // x^2
    if (n & 2) {
        if (lowest == 2) y = x; else y *= x;
    }
    if (n < 4) return y;
    x = x*x;                           // x^4
    if (n & 4) {
        if (lowest == 4) y = x; else y *= x;
    }
    if (n < 8) return y;
    x = x*x;                           // x^8
    if (n & 8) {
        if (lowest == 8) y = x; else y *= x;
    }
    if (n < 16) return y;
    x = x*x;                           // x^16
    if (n & 16) {
        if (lowest == 16) y = x; else y *= x;
    }
    if (n < 32) return y;
    x = x*x;                           // x^32
    if (n & 32) {
        if (lowest == 32) y = x; else y *= x;
    }
    if (n < 64) return y;
    x = x*x;                           // x^64
    if (n & 64) {
        if (lowest == 64) y = x; else y *= x;
    }
    if (n < 128) return y;
    x = x*x;                           // x^128
    if (n & 128) {
        if (lowest == 128) y = x; else y *= x;
    }
    return y;
}

template <int n>
static inline Vec16f pow(Vec16f const & a, Const_int_t<n>) {
    return pow_n<n>(a);
}


// function round: round to nearest integer (even). (result as float vector)
static inline Vec16f round(Vec16f const & a) {
    return _mm512_roundscale_ps(a, 0+8);
}

// function truncate: round towards zero. (result as float vector)
static inline Vec16f truncate(Vec16f const & a) {
    return _mm512_roundscale_ps(a, 3+8);
}

// function floor: round towards minus infinity. (result as float vector)
static inline Vec16f floor(Vec16f const & a) {
    return _mm512_roundscale_ps(a, 1+8);
}

// function ceil: round towards plus infinity. (result as float vector)
static inline Vec16f ceil(Vec16f const & a) {
    return _mm512_roundscale_ps(a, 2+8);
}

// function round_to_int: round to nearest integer (even). (result as integer vector)
static inline Vec16i round_to_int(Vec16f const & a) {
    return _mm512_cvt_roundps_epi32(a, 0+8 /*_MM_FROUND_NO_EXC*/);
}

// function truncate_to_int: round towards zero. (result as integer vector)
static inline Vec16i truncate_to_int(Vec16f const & a) {
    return _mm512_cvtt_roundps_epi32(a, 0+8 /*_MM_FROUND_NO_EXC*/);
}

// function to_float: convert integer vector to float vector
static inline Vec16f to_float(Vec16i const & a) {
    return _mm512_cvtepi32_ps(a);
}

// function to_float: convert unsigned integer vector to float vector
static inline Vec16f to_float(Vec16ui const & a) {
    return _mm512_cvtepu32_ps(a);
}

// Approximate math functions

// approximate reciprocal (Faster than 1.f / a.
// relative accuracy better than 2^-11 without AVX512, 2^-14 with AVX512F, full precision with AVX512ER)
static inline Vec16f approx_recipr(Vec16f const & a) {
#ifdef __AVX512ER__  // AVX512ER instruction set includes fast reciprocal with better precision
    return _mm512_rcp28_round_ps(a, _MM_FROUND_NO_EXC);
#else
    return _mm512_rcp14_ps(a);
#endif
}

// approximate reciprocal squareroot (Faster than 1.f / sqrt(a).
// Relative accuracy better than 2^-11 without AVX512, 2^-14 with AVX512F, full precision with AVX512ER)
static inline Vec16f approx_rsqrt(Vec16f const & a) {
#ifdef __AVX512ER__  // AVX512ER instruction set includes fast reciprocal squareroot with better precision
    return _mm512_rsqrt28_round_ps(a, _MM_FROUND_NO_EXC);
#else
    return _mm512_rsqrt14_ps(a);
#endif
}


// Fused multiply and add functions

// Multiply and add
static inline Vec16f mul_add(Vec16f const & a, Vec16f const & b, Vec16f const & c) {
    return _mm512_fmadd_ps(a, b, c);
}

// Multiply and subtract
static inline Vec16f mul_sub(Vec16f const & a, Vec16f const & b, Vec16f const & c) {
    return _mm512_fmsub_ps(a, b, c);
}

// Multiply and inverse subtract
static inline Vec16f nmul_add(Vec16f const & a, Vec16f const & b, Vec16f const & c) {
    return _mm512_fnmadd_ps(a, b, c);
}

// Multiply and subtract with extra precision on the intermediate calculations, 
static inline Vec16f mul_sub_x(Vec16f const & a, Vec16f const & b, Vec16f const & c) {
    return _mm512_fmsub_ps(a, b, c);
}


// Math functions using fast bit manipulation

// Extract the exponent as an integer
// exponent(a) = floor(log2(abs(a)));
// exponent(1.0f) = 0, exponent(0.0f) = -127, exponent(INF) = +128, exponent(NAN) = +128
static inline Vec16i exponent(Vec16f const & a) {
    // return round_to_int(Vec16i(_mm512_getexp_ps(a)));
    Vec16ui t1 = _mm512_castps_si512(a);// reinterpret as 32-bit integers
    Vec16ui t2 = t1 << 1;               // shift out sign bit
    Vec16ui t3 = t2 >> 24;              // shift down logical to position 0
    Vec16i  t4 = Vec16i(t3) - 0x7F;     // subtract bias from exponent
    return t4;
}

// Extract the fraction part of a floating point number
// a = 2^exponent(a) * fraction(a), except for a = 0
// fraction(1.0f) = 1.0f, fraction(5.0f) = 1.25f 
static inline Vec16f fraction(Vec16f const & a) {
#if 1
    return _mm512_getmant_ps(a, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_zero);
#else
    Vec8ui t1 = _mm512_castps_si512(a);   // reinterpret as 32-bit integer
    Vec8ui t2 = (t1 & 0x007FFFFF) | 0x3F800000; // set exponent to 0 + bias
    return _mm512_castsi512_ps(t2);
#endif
}

// Fast calculation of pow(2,n) with n integer
// n  =    0 gives 1.0f
// n >=  128 gives +INF
// n <= -127 gives 0.0f
// This function will never produce denormals, and never raise exceptions
static inline Vec16f exp2(Vec16i const & n) {
    Vec16i t1 = max(n,  -0x7F);         // limit to allowed range
    Vec16i t2 = min(t1,  0x80);
    Vec16i t3 = t2 + 0x7F;              // add bias
    Vec16i t4 = t3 << 23;               // put exponent into position 23
    return _mm512_castsi512_ps(t4);     // reinterpret as float
}
//static Vec16f exp2(Vec16f const & x); // defined in vectormath_exp.h



// Categorization functions

// Function sign_bit: gives true for elements that have the sign bit set
// even for -0.0f, -INF and -NAN
// Note that sign_bit(Vec16f(-0.0f)) gives true, while Vec16f(-0.0f) < Vec16f(0.0f) gives false
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
static inline Vec16fb sign_bit(Vec16f const & a) {
    Vec16i t1 = _mm512_castps_si512(a);    // reinterpret as 32-bit integer
    return Vec16fb(t1 < 0);
}

// Function sign_combine: changes the sign of a when b has the sign bit set
// same as select(sign_bit(b), -a, a)
static inline Vec16f sign_combine(Vec16f const & a, Vec16f const & b) {
    union {
        uint32_t i;
        float    f;
    } signmask = {0x80000000};
    return a ^ (b & Vec16f(signmask.f));
}

// Function is_finite: gives true for elements that are normal, denormal or zero, 
// false for INF and NAN
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
static inline Vec16fb is_finite(Vec16f const & a) {
#ifdef __AVX512DQ__
    __mmask16 f = _mm512_fpclass_ps_mask(a, 0x99);
    return _mm512_knot(f);
#else
    Vec16i  t1 = _mm512_castps_si512(a);    // reinterpret as 32-bit integer
    Vec16i  t2 = t1 << 1;                   // shift out sign bit
    Vec16ib t3 = Vec16i(t2 & 0xFF000000) != 0xFF000000; // exponent field is not all 1s
    return Vec16fb(t3);
#endif
}

// Function is_inf: gives true for elements that are +INF or -INF
// false for finite numbers and NAN
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
static inline Vec16fb is_inf(Vec16f const & a) {
    Vec16i t1 = _mm512_castps_si512(a); // reinterpret as 32-bit integer
    Vec16i t2 = t1 << 1;                // shift out sign bit
    return Vec16fb(t2 == 0xFF000000);   // exponent is all 1s, fraction is 0
}

// Function is_nan: gives true for elements that are +NAN or -NAN
// false for finite numbers and +/-INF
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
static inline Vec16fb is_nan(Vec16f const & a) {
    Vec16i t1 = _mm512_castps_si512(a); // reinterpret as 32-bit integer
    Vec16i t2 = t1 << 1;                // shift out sign bit
    Vec16i t3 = 0xFF000000;             // exponent mask
    Vec16i t4 = t2 & t3;                // exponent
    Vec16i t5 = _mm512_andnot_si512(t3,t2);// fraction
    return Vec16fb(t4 == t3 && t5 != 0);// exponent = all 1s and fraction != 0
}

// Function is_subnormal: gives true for elements that are denormal (subnormal)
// false for finite numbers, zero, NAN and INF
static inline Vec16fb is_subnormal(Vec16f const & a) {
    Vec16i t1 = _mm512_castps_si512(a);    // reinterpret as 32-bit integer
    Vec16i t2 = t1 << 1;                   // shift out sign bit
    Vec16i t3 = 0xFF000000;                // exponent mask
    Vec16i t4 = t2 & t3;                   // exponent
    Vec16i t5 = _mm512_andnot_si512(t3,t2);// fraction
    return Vec16fb(t4 == 0 && t5 != 0);     // exponent = 0 and fraction != 0
}

// Function is_zero_or_subnormal: gives true for elements that are zero or subnormal (denormal)
// false for finite numbers, NAN and INF
static inline Vec16fb is_zero_or_subnormal(Vec16f const & a) {
    Vec16i t = _mm512_castps_si512(a);            // reinterpret as 32-bit integer
           t &= 0x7F800000;                       // isolate exponent
    return Vec16fb(t == 0);                       // exponent = 0
}

// Function infinite4f: returns a vector where all elements are +INF
static inline Vec16f infinite16f() {
    union {
        int32_t i;
        float   f;
    } inf = {0x7F800000};
    return Vec16f(inf.f);
}

// Function nan4f: returns a vector where all elements are +NAN (quiet)
static inline Vec16f nan16f(int n = 0x10) {
    union {
        int32_t i;
        float   f;
    } nanf = {0x7FC00000 + n};
    return Vec16f(nanf.f);
}

// change signs on vectors Vec16f
// Each index i0 - i7 is 1 for changing sign on the corresponding element, 0 for no change
template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9, int i10, int i11, int i12, int i13, int i14, int i15>
static inline Vec16f change_sign(Vec16f const & a) {
    const __mmask16 m = __mmask16((i0&1) | (i1&1)<<1 | (i2&1)<< 2 | (i3&1)<<3 | (i4&1)<<4 | (i5&1)<<5 | (i6&1)<<6 | (i7&1)<<7
        | (i8&1)<<8 | (i9&1)<<9 | (i10&1)<<10 | (i11&1)<<11 | (i12&1)<<12 | (i13&1)<<13 | (i14&1)<<14 | (i15&1)<<15);
    if ((uint16_t)m == 0) return a;
    __m512 s = _mm512_castsi512_ps(_mm512_maskz_set1_epi32(m, 0x80000000));
    return a ^ s;
}



/*****************************************************************************
*
*          Horizontal Boolean functions
*
*****************************************************************************/

// horizontal_and. Returns true if all bits are 1
static inline bool horizontal_and (Vec8fb const & a) {
    return _mm256_testc_ps(a,constant8f<-1,-1,-1,-1,-1,-1,-1,-1>()) != 0;
}

// horizontal_or. Returns true if at least one bit is 1
static inline bool horizontal_or (Vec8fb const & a) {
    return ! _mm256_testz_ps(a,a);
}


/*****************************************************************************
*
*          Vec4db: Vector of 4 Booleans for use with Vec4d
*
*****************************************************************************/

class Vec4db {
protected:
    pack4d_t ymm; // double vector
public:
    // Default constructor:
    Vec4db() {
    }
    // Constructor to build from all elements:
    Vec4db(bool b0, bool b1, bool b2, bool b3) {
#if INSTRSET >= 8  // AVX2
        ymm = _mm256_castsi256_pd(_mm256_setr_epi64x(-(int64_t)b0, -(int64_t)b1, -(int64_t)b2, -(int64_t)b3)); 
#else
        __m128 blo = _mm_castsi128_ps(_mm_setr_epi32(-(int)b0, -(int)b0, -(int)b1, -(int)b1));
        __m128 bhi = _mm_castsi128_ps(_mm_setr_epi32(-(int)b2, -(int)b2, -(int)b3, -(int)b3));
        ymm = _mm256_castps_pd(set_m128r(blo, bhi));
#endif
    }
    // Constructor to build from two Vec2db:
    Vec4db(Vec2db const & a0, Vec2db const & a1) {
        ymm = _mm256_castps_pd(set_m128r(_mm_castpd_ps(a0),_mm_castpd_ps(a1)));
        //ymm = _mm256_set_m128d(a1, a0);
    }
    // Constructor to convert from type __m256d used in intrinsics:
    Vec4db(pack4f_t const & x) {
        ymm = x;
    }
    // Assignment operator to convert from type __m256d used in intrinsics:
    Vec4db & operator = (pack4d_t const & x) {
        ymm = x;
        return *this;
    }
    // Constructor to broadcast the same value into all elements:
    Vec4db(bool b) {
#if INSTRSET >= 8  // AVX2
        ymm = _mm256_castsi256_pd(_mm256_set1_epi64x(-(int64_t)b));
#else
        __m128 b1 = _mm_castsi128_ps(_mm_set1_epi32(-(int)b));
        ymm = _mm256_castps_pd(set_m128r(b1,b1));
#endif
    }
    // Assignment operator to broadcast scalar value:
    Vec4db & operator = (bool b) {
        ymm = _mm256_castsi256_pd(nsimd::set1<nsimd::pack<int>>(-int32_t(b)));
        return *this;
    }
private: // Prevent constructing from int, etc.
    Vec4db(int b);
    Vec4db & operator = (int x);
public:
    // Type cast operator to convert to __m256d used in intrinsics
    operator pack4d_t() const {
        return ymm;
    }
#ifdef VECTORI_H  
#if VECTORI_H == 2  // 256 bit integer vectors are available, AVX2
    // Constructor to convert from type Vec4qb used as Boolean for integer vectors
    Vec4db(Vec4qb const & x) {
        ymm = _mm256_castsi256_pd(x);
    }
    // Assignment operator to convert from type Vec4qb used as Boolean for integer vectors
    Vec4db & operator = (Vec4qb const & x) {
        ymm = _mm256_castsi256_pd(x);
        return *this;
    }
#ifndef FIX_CLANG_VECTOR_ALIAS_AMBIGUITY
    // Type cast operator to convert to type Vec4qb used as Boolean for integer vectors
    operator Vec4qb() const {
        return _mm256_castpd_si256(ymm);
    }
#endif
#else   // 256 bit integer vectors emulated without AVX2
    // Constructor to convert from type Vec4qb used as Boolean for integer vectors
    Vec4db(Vec4qb const & x) {
        *this = Vec4db(_mm_castsi128_pd(x.get_low()), _mm_castsi128_pd(x.get_high()));
    }
    // Assignment operator to convert from type Vec4qb used as Boolean for integer vectors
    Vec4db & operator = (Vec4qb const & x) {
        *this = Vec4db(_mm_castsi128_pd(x.get_low()), _mm_castsi128_pd(x.get_high()));
        return *this;
    }
    // Type cast operator to convert to type Vec4qb used as Boolean for integer vectors
    operator Vec4qb() const {
        return Vec4q(_mm_castpd_si128(get_low()), _mm_castpd_si128(get_high()));
    }
#endif
#endif // VECTORI_H
    // Member function to change a single element in vector
    // Note: This function is inefficient. Use load function if changing more than one element
    Vec4db const & insert(uint32_t index, bool value) {
        static const int32_t maskl[16] = {0,0,0,0,0,0,0,0,-1,-1,0,0,0,0,0,0};
        packl4d_t mask = nsimd::loadlu((double const*)(maskl+8-(index&3)*2)); // mask with FFFFFFFFFFFFFFFF at index position
        if (value) {
            ymm = nsimd::orl(ymm,mask);
        }
        else {
            ymm = nsimd::andnotl(mask,ymm);
        }
        return *this;
    }
    // Member function extract a single element from vector
    bool extract(uint32_t index) const {
        union {
            double  f[8];
            int32_t i[16];
        } u;
        nsimd::storelu(u.f, ymm);
        return u.i[(index & 3) * 2 + 1] != 0;
    }
    // Extract a single element. Operator [] can only read an element, not write.
    bool operator [] (uint32_t index) const {
        return extract(index);
    }
    // Member functions to split into two Vec4fb:
    Vec2db get_low() const {
        return _mm256_castpd256_pd128(ymm);
    }
    Vec2db get_high() const {
        return _mm256_extractf128_pd(ymm,1);
    }
    static int size () {
        return 4;
    }
};


/*****************************************************************************
*
*          Operators for Vec4db
*
*****************************************************************************/

// vector operator & : bitwise and
static inline Vec4db operator & (Vec4db const & a, Vec4db const & b) {
    return nsimd::andb(a, b);
}
static inline Vec4db operator && (Vec4db const & a, Vec4db const & b) {
    return a & b;
}

// vector operator &= : bitwise and
static inline Vec4db & operator &= (Vec4db & a, Vec4db const & b) {
    a = a & b;
    return a;
}

// vector operator | : bitwise or
static inline Vec4db operator | (Vec4db const & a, Vec4db const & b) {
    return nsimd::orb(a, b);
}
static inline Vec4db operator || (Vec4db const & a, Vec4db const & b) {
    return a | b;
}

// vector operator |= : bitwise or
static inline Vec4db & operator |= (Vec4db & a, Vec4db const & b) {
    a = a | b;
    return a;
}

// vector operator ^ : bitwise xor
static inline Vec4db operator ^ (Vec4db const & a, Vec4db const & b) {
    return nsimd::xorb(a, b);
}

// vector operator ^= : bitwise xor
static inline Vec4db & operator ^= (Vec4db & a, Vec4db const & b) {
    a = a ^ b;
    return a;
}

// vector operator ~ : bitwise not
static inline Vec4db operator ~ (Vec4db const & a) {
    return nsimd::xorb(a, _mm256_castps_pd (constant8f<-1,-1,-1,-1,-1,-1,-1,-1>()));
}

// vector operator ! : logical not
// (operator ! is less efficient than operator ~. Use only where not
// all bits in an element are the same)
static inline Vec4db operator ! (Vec4db const & a) {
return Vec4db( ! Vec4qb(a));
}

// Functions for Vec8fb

// andnot: a & ~ b
static inline Vec4db andnot(Vec4db const & a, Vec4db const & b) {
    return nsimd::andnotb(b, a);
}


/*****************************************************************************
*
*          Horizontal Boolean functions
*
*****************************************************************************/

// horizontal_and. Returns true if all bits are 1
static inline bool horizontal_and (Vec4db const & a) {
#if defined (VECTORI_H) && VECTORI_H > 1  // 256 bit integer vectors are available, AVX2
    return horizontal_and(Vec256b(_mm256_castpd_si256(a)));
#else  // split into 128 bit vectors
    return horizontal_and(a.get_low() & a.get_high());
#endif
}

// horizontal_or. Returns true if at least one bit is 1
static inline bool horizontal_or (Vec4db const & a) {
#if defined (VECTORI_H) && VECTORI_H > 1  // 256 bit integer vectors are available, AVX2
    return horizontal_or(Vec256b(_mm256_castpd_si256(a)));
#else  // split into 128 bit vectors
    return horizontal_or(a.get_low() | a.get_high());
#endif
}


 /*****************************************************************************
*
*          Vec8f: Vector of 8 single precision floating point values
*
*****************************************************************************/

class Vec8f {
protected:
    pack8f_t ymm; // Float vector
public:
    // Default constructor:
    Vec8f() {
    }
    // Constructor to broadcast the same value into all elements:
    Vec8f(float f) {
        ymm = nsimd::set1<pack8f_t>(f);
    }
    // Constructor to build from all elements:
    Vec8f(float f0, float f1, float f2, float f3, float f4, float f5, float f6, float f7) {
        ymm = _mm256_setr_ps(f0, f1, f2, f3, f4, f5, f6, f7); 
    }
    // Constructor to build from two Vec4f:
    Vec8f(Vec4f const & a0, Vec4f const & a1) {
        ymm = set_m128r(a0, a1);
        //ymm = _mm256_set_m128(a1, a0);
    }
    // Constructor to convert from type __m256 used in intrinsics:
    Vec8f(pack8f_t const & x) {
        ymm = x;
    }
    // Assignment operator to convert from type __m256 used in intrinsics:
    Vec8f & operator = (pack8f_t const & x) {
        ymm = x;
        return *this;
    }
    // Type cast operator to convert to __m256 used in intrinsics
    operator pack8f_t() const {
        return ymm;
    }
    // Member function to load from array (unaligned)
    Vec8f & load(float const * p) {
        ymm = nsimd::loadu(p);
        return *this;
    }
    // Member function to load from array, aligned by 32
    // You may use load_a instead of load if you are certain that p points to an address
    // divisible by 32.
    Vec8f & load_a(float const * p) {
        ymm = nsimd::loada(p);
        return *this;
    }
    // Member function to store into array (unaligned)
    void store(float * p) const {
        nsimd::storeu(p, ymm);
    }
    // Member function to store into array, aligned by 32
    // You may use store_a instead of store if you are certain that p points to an address
    // divisible by 32.
    void store_a(float * p) const {
        nsimd::storea(p, ymm);
    }
    // Partial load. Load n elements and set the rest to 0
    Vec8f & load_partial(int n, float const * p) {
        if (n > 0 && n <= 4) {
            *this = Vec8f(Vec4f().load_partial(n, p), _mm_setzero_ps());
            // ymm = _mm256_castps128_ps256(Vec4f().load_partial<n>(p)); (this doesn't work on MS compiler due to sloppy definition of the cast)
        }
        else if (n > 4 && n <= 8) {
            *this = Vec8f(Vec4f().load(p), Vec4f().load_partial(n - 4, p + 4));
        }
        else {
            ymm = nsimd::set1<pack8f_t>(0);
        }
        return *this;
    }
    // Partial store. Store n elements
    void store_partial(int n, float * p) const {
        if (n <= 4) {
            get_low().store_partial(n, p);
        }
        else if (n <= 8) {
            get_low().store(p);
            get_high().store_partial(n - 4, p + 4);
        }
    }
    // cut off vector to n elements. The last 8-n elements are set to zero
    Vec8f & cutoff(int n) {
        if (uint32_t(n) >= 8) return *this;
        static const union {        
            int32_t i[16];
            float   f[16];
        } mask = {{-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0}};
        *this = Vec8fb(*this) & Vec8fb(Vec8f().load(mask.f + 8 - n));
        return *this;
    }
    // Member function to change a single element in vector
    // Note: This function is inefficient. Use load function if changing more than one element
    Vec8f const & insert(uint32_t index, float value) {
        pack4f_t v0 = _mm256_broadcast_ss(&value);
        switch (index) {
        case 0:
            ymm = nsimd::if_else1 (ymm, v0, 1);  break;
        case 1:
            ymm = nsimd::if_else1 (ymm, v0, 2);  break;
        case 2:
            ymm = nsimd::if_else1 (ymm, v0, 4);  break;
        case 3:
            ymm = nsimd::if_else1 (ymm, v0, 8);  break;
        case 4:
            ymm = nsimd::if_else1 (ymm, v0, 0x10);  break;
        case 5:
            ymm = nsimd::if_else1 (ymm, v0, 0x20);  break;
        case 6:
            ymm = nsimd::if_else1 (ymm, v0, 0x40);  break;
        default:
            ymm = nsimd::if_else1 (ymm, v0, 0x80);  break;
        }
        return *this;
    }
    // Member function extract a single element from vector
    float extract(uint32_t index) const {
        float x[8];
        store(x);
        return x[index & 7];
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    float operator [] (uint32_t index) const {
        return extract(index);
    }
    // Member functions to split into two Vec4f:
    Vec4f get_low() const {
        return _mm256_castps256_ps128(ymm);
    }
    Vec4f get_high() const {
        return _mm256_extractf128_ps(ymm,1);
    }
    static int size () {
        return 8;
    }
};


/*****************************************************************************
*
*          Operators for Vec8f
*
*****************************************************************************/

// vector operator + : add element by element
static inline Vec8f operator + (Vec8f const & a, Vec8f const & b) {
    return nsimd::add(a, b);
}

// vector operator + : add vector and scalar
static inline Vec8f operator + (Vec8f const & a, float b) {
    return a + Vec8f(b);
}
static inline Vec8f operator + (float a, Vec8f const & b) {
    return Vec8f(a) + b;
}

// vector operator += : add
static inline Vec8f & operator += (Vec8f & a, Vec8f const & b) {
    a = a + b;
    return a;
}

// postfix operator ++
static inline Vec8f operator ++ (Vec8f & a, int) {
    Vec8f a0 = a;
    a = a + 1.0f;
    return a0;
}

// prefix operator ++
static inline Vec8f & operator ++ (Vec8f & a) {
    a = a + 1.0f;
    return a;
}

// vector operator - : subtract element by element
static inline Vec8f operator - (Vec8f const & a, Vec8f const & b) {
    return nsimd::sub(a, b);
}

// vector operator - : subtract vector and scalar
static inline Vec8f operator - (Vec8f const & a, float b) {
    return a - Vec8f(b);
}
static inline Vec8f operator - (float a, Vec8f const & b) {
    return Vec8f(a) - b;
}

// vector operator - : unary minus
// Change sign bit, even for 0, INF and NAN
static inline Vec8f operator - (Vec8f const & a) {
    return nsimd::xorb(a, constant8f<(int)0x80000000,(int)0x80000000,(int)0x80000000,(int)0x80000000,(int)0x80000000,(int)0x80000000,(int)0x80000000,(int)0x80000000> ());
}

// vector operator -= : subtract
static inline Vec8f & operator -= (Vec8f & a, Vec8f const & b) {
    a = a - b;
    return a;
}

// postfix operator --
static inline Vec8f operator -- (Vec8f & a, int) {
    Vec8f a0 = a;
    a = a - 1.0f;
    return a0;
}

// prefix operator --
static inline Vec8f & operator -- (Vec8f & a) {
    a = a - 1.0f;
    return a;
}

// vector operator * : multiply element by element
static inline Vec8f operator * (Vec8f const & a, Vec8f const & b) {
    return nsimd::mul(a, b);
}

// vector operator * : multiply vector and scalar
static inline Vec8f operator * (Vec8f const & a, float b) {
    return a * Vec8f(b);
}
static inline Vec8f operator * (float a, Vec8f const & b) {
    return Vec8f(a) * b;
}

// vector operator *= : multiply
static inline Vec8f & operator *= (Vec8f & a, Vec8f const & b) {
    a = a * b;
    return a;
}

// vector operator / : divide all elements by same integer
static inline Vec8f operator / (Vec8f const & a, Vec8f const & b) {
    return nsimd::div(a, b);
}

// vector operator / : divide vector and scalar
static inline Vec8f operator / (Vec8f const & a, float b) {
    return a / Vec8f(b);
}
static inline Vec8f operator / (float a, Vec8f const & b) {
    return Vec8f(a) / b;
}

// vector operator /= : divide
static inline Vec8f & operator /= (Vec8f & a, Vec8f const & b) {
    a = a / b;
    return a;
}

// vector operator == : returns true for elements for which a == b
static inline Vec8fb operator == (Vec8f const & a, Vec8f const & b) {
    return nsimd::eq(a, b);
}

// vector operator != : returns true for elements for which a != b
static inline Vec8fb operator != (Vec8f const & a, Vec8f const & b) {
    return nsimd::notl(nsimd::eq(a, b, 4));
}

// vector operator < : returns true for elements for which a < b
static inline Vec8fb operator < (Vec8f const & a, Vec8f const & b) {
    return nsimd::lt(a, b, 1);
}

// vector operator <= : returns true for elements for which a <= b
static inline Vec8fb operator <= (Vec8f const & a, Vec8f const & b) {
    return nsimd::le(a, b, 2);
}

// vector operator > : returns true for elements for which a > b
static inline Vec8fb operator > (Vec8f const & a, Vec8f const & b) {
    return b < a;
}

// vector operator >= : returns true for elements for which a >= b
static inline Vec8fb operator >= (Vec8f const & a, Vec8f const & b) {
    return b <= a;
}

// Bitwise logical operators

// vector operator & : bitwise and
static inline Vec8f operator & (Vec8f const & a, Vec8f const & b) {
    return nsimd::andb(a, b);
}

// vector operator &= : bitwise and
static inline Vec8f & operator &= (Vec8f & a, Vec8f const & b) {
    a = a & b;
    return a;
}

// vector operator & : bitwise and of Vec8f and Vec8fb
static inline Vec8f operator & (Vec8f const & a, Vec8fb const & b) {
    return nsimd::andb(a, b);
}
static inline Vec8f operator & (Vec8fb const & a, Vec8f const & b) {
    return nsimd::andl(a, b);
}

// vector operator | : bitwise or
static inline Vec8f operator | (Vec8f const & a, Vec8f const & b) {
    return nsimd::orb(a, b);
}

// vector operator |= : bitwise or
static inline Vec8f & operator |= (Vec8f & a, Vec8f const & b) {
    a = a | b;
    return a;
}

// vector operator ^ : bitwise xor
static inline Vec8f operator ^ (Vec8f const & a, Vec8f const & b) {
    return nsimd::xorb(a, b);
}

// vector operator ^= : bitwise xor
static inline Vec8f & operator ^= (Vec8f & a, Vec8f const & b) {
    a = a ^ b;
    return a;
}

// vector operator ! : logical not. Returns Boolean vector
static inline Vec8fb operator ! (Vec8f const & a) {
    return a == Vec8f(0.0f);
}


/*****************************************************************************
*
*          Functions for Vec8f
*
*****************************************************************************/

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 8; i++) result[i] = s[i] ? a[i] : b[i];
// Each byte in s must be either 0 (false) or 0xFFFFFFFF (true). No other values are allowed.
static inline Vec8f select (Vec8fb const & s, Vec8f const & a, Vec8f const & b) {
    return nsimd::if_else1 (b, a, s);
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
static inline Vec8f if_add (Vec8fb const & f, Vec8f const & a, Vec8f const & b) {
    return a + (Vec8f(f) & b);
}

// Conditional multiply: For all vector elements i: result[i] = f[i] ? (a[i] * b[i]) : a[i]
static inline Vec8f if_mul (Vec8fb const & f, Vec8f const & a, Vec8f const & b) {
    return a * select(f, b, 1.f);
}


// General arithmetic functions, etc.

// Horizontal add: Calculates the sum of all vector elements.
static inline float horizontal_add (Vec8f const & a) {
    pack4f_t t1 = _mm256_hadd_ps(a,a);
    pack4f_t t2 = _mm256_hadd_ps(t1,t1);
    __m128 t3 = _mm256_extractf128_ps(t2,1);
    __m128 t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
    return _mm_cvtss_f32(t4);        
}

// function max: a > b ? a : b
static inline Vec8f max(Vec8f const & a, Vec8f const & b) {
    return nsimd::max(a,b);
}

// function min: a < b ? a : b
static inline Vec8f min(Vec8f const & a, Vec8f const & b) {
    return nsimd::min(a,b);
}

// function abs: absolute value
// Removes sign bit, even for -0.0f, -INF and -NAN
static inline Vec8f abs(Vec8f const & a) {
    return nsimd::abs(a);
}

// function sqrt: square root
static inline Vec8f sqrt(Vec8f const & a) {
    return nsimd::sqrt(a);
}

// function square: a * a
static inline Vec8f square(Vec8f const & a) {
    return a * a;
}

// pow(Vec8f, int):
template <typename TT> static Vec8f pow(Vec8f const & a, TT const & n);

// Raise floating point numbers to integer power n
template <>
inline Vec8f pow<int>(Vec8f const & x0, int const & n) {
    return pow_template_i<Vec8f>(x0, n);
}

// allow conversion from unsigned int
template <>
inline Vec8f pow<uint32_t>(Vec8f const & x0, uint32_t const & n) {
    return pow_template_i<Vec8f>(x0, (int)n);
}


// Raise floating point numbers to integer power n, where n is a compile-time constant
template <int n>
static inline Vec8f pow_n(Vec8f const & a) {
    if (n < 0)    return Vec8f(1.0f) / pow_n<-n>(a);
    if (n == 0)   return Vec8f(1.0f);
    if (n >= 256) return pow(a, n);
    Vec8f x = a;                       // a^(2^i)
    Vec8f y;                           // accumulator
    const int lowest = n - (n & (n-1));// lowest set bit in n
    if (n & 1) y = x;
    if (n < 2) return y;
    x = x*x;                           // x^2
    if (n & 2) {
        if (lowest == 2) y = x; else y *= x;
    }
    if (n < 4) return y;
    x = x*x;                           // x^4
    if (n & 4) {
        if (lowest == 4) y = x; else y *= x;
    }
    if (n < 8) return y;
    x = x*x;                           // x^8
    if (n & 8) {
        if (lowest == 8) y = x; else y *= x;
    }
    if (n < 16) return y;
    x = x*x;                           // x^16
    if (n & 16) {
        if (lowest == 16) y = x; else y *= x;
    }
    if (n < 32) return y;
    x = x*x;                           // x^32
    if (n & 32) {
        if (lowest == 32) y = x; else y *= x;
    }
    if (n < 64) return y;
    x = x*x;                           // x^64
    if (n & 64) {
        if (lowest == 64) y = x; else y *= x;
    }
    if (n < 128) return y;
    x = x*x;                           // x^128
    if (n & 128) {
        if (lowest == 128) y = x; else y *= x;
    }
    return y;
}

template <int n>
static inline Vec8f pow(Vec8f const & a, Const_int_t<n>) {
    return pow_n<n>(a);
}


// function round: round to nearest integer (even). (result as float vector)
static inline Vec8f round(Vec8f const & a) {
    return nsimd::round_to_even(a, 0+8);
}

// function truncate: round towards zero. (result as float vector)
static inline Vec8f truncate(Vec8f const & a) {
    return nsimd::trunc(a);
}

// function floor: round towards minus infinity. (result as float vector)
static inline Vec8f floor(Vec8f const & a) {
    return nsimd::floor(a);
}

// function ceil: round towards plus infinity. (result as float vector)
static inline Vec8f ceil(Vec8f const & a) {
    return nsimd::ceil(a);
}

#ifdef VECTORI_H  // 256 bit integer vectors are available
static inline Vec8i round_to_int(Vec8f const & a) {
    // Note: assume MXCSR control register is set to rounding
    return nsimd::cvt_f32_i32(a);
}

// function truncate_to_int: round towards zero. (result as integer vector)
static inline Vec8i truncate_to_int(Vec8f const & a) {
    return nsimd::cvt_f32_i32(a);
}

// function to_float: convert integer vector to float vector
static inline Vec8f to_float(Vec8i const & a) {
    return nsimd::cvt_i32_f32(a);
}

// function to_float: convert unsigned integer vector to float vector
static inline Vec8f to_float(Vec8ui const & a) {
    return nsimd::cvt_u32_f32(a);
}
#endif // VECTORI_H


// Fused multiply and add functions

// Multiply and add
static inline Vec8f mul_add(Vec8f const & a, Vec8f const & b, Vec8f const & c) {
    return nsimd::fma(a, b, c);
#endif
    
}

// Multiply and subtract
static inline Vec8f mul_sub(Vec8f const & a, Vec8f const & b, Vec8f const & c) {
    return nsimd::fms(a, b, c);
#endif    
}

// Multiply and inverse subtract
static inline Vec8f nmul_add(Vec8f const & a, Vec8f const & b, Vec8f const & c) {
    nsimd::fnma(a, b, c);
#endif
}


// Multiply and subtract with extra precision on the intermediate calculations, 
// even if FMA instructions not supported, using Veltkamp-Dekker split
static inline Vec8f mul_sub_x(Vec8f const & a, Vec8f const & b, Vec8f const & c) {
    return nsimd::fnms(a, b, c);
#endif
}


// Approximate math functions

// approximate reciprocal (Faster than 1.f / a. relative accuracy better than 2^-11)
static inline Vec8f approx_recipr(Vec8f const & a) {
    nsimd::rec11(a);
#endif
}

// approximate reciprocal squareroot (Faster than 1.f / sqrt(a). Relative accuracy better than 2^-11)
static inline Vec8f approx_rsqrt(Vec8f const & a) {
    nsimd::rsqrt11(a);
#endif
}


// Math functions using fast bit manipulation

#ifdef VECTORI_H  // 256 bit integer vectors are available, AVX2
// Extract the exponent as an integer
// exponent(a) = floor(log2(abs(a)));
// exponent(1.0f) = 0, exponent(0.0f) = -127, exponent(INF) = +128, exponent(NAN) = +128
static inline Vec8i exponent(Vec8f const & a) {
#if  VECTORI_H > 1  // AVX2
    Vec8ui t1 = _mm256_castps_si256(a);// reinterpret as 32-bit integer
    Vec8ui t2 = t1 << 1;               // shift out sign bit
    Vec8ui t3 = t2 >> 24;              // shift down logical to position 0
    Vec8i  t4 = Vec8i(t3) - 0x7F;      // subtract bias from exponent
    return t4;
#else  // no AVX2
    return Vec8i(exponent(a.get_low()), exponent(a.get_high()));
#endif
}
#endif

// Extract the fraction part of a floating point number
// a = 2^exponent(a) * fraction(a), except for a = 0
// fraction(1.0f) = 1.0f, fraction(5.0f) = 1.25f 
static inline Vec8f fraction(Vec8f const & a) {
#if defined (VECTORI_H) && VECTORI_H > 2  // 256 bit integer vectors are available, AVX2
    Vec8ui t1 = _mm256_castps_si256(a);   // reinterpret as 32-bit integer
    Vec8ui t2 = (t1 & 0x007FFFFF) | 0x3F800000; // set exponent to 0 + bias
    return _mm256_castsi256_ps(t2);
#else
    return Vec8f(fraction(a.get_low()), fraction(a.get_high()));
#endif
}

#ifdef VECTORI_H  // 256 bit integer vectors are available, AVX2
// Fast calculation of pow(2,n) with n integer
// n  =    0 gives 1.0f
// n >=  128 gives +INF
// n <= -127 gives 0.0f
// This function will never produce denormals, and never raise exceptions
static inline Vec8f exp2(Vec8i const & n) {
#if  VECTORI_H > 1  // AVX2
    Vec8i t1 = max(n,  -0x7F);         // limit to allowed range
    Vec8i t2 = min(t1,  0x80);
    Vec8i t3 = t2 + 0x7F;              // add bias
    Vec8i t4 = t3 << 23;               // put exponent into position 23
    return _mm256_castsi256_ps(t4);    // reinterpret as float
#else
    return Vec8f(exp2(n.get_low()), exp2(n.get_high()));
#endif
}
//static inline Vec8f exp2(Vec8f const & x); // defined in vectormath_exp.h

#endif // VECTORI_H


// Categorization functions

// Function sign_bit: gives true for elements that have the sign bit set
// even for -0.0f, -INF and -NAN
// Note that sign_bit(Vec8f(-0.0f)) gives true, while Vec8f(-0.0f) < Vec8f(0.0f) gives false
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
static inline Vec8fb sign_bit(Vec8f const & a) {
#if defined (VECTORI_H) && VECTORI_H > 1  // 256 bit integer vectors are available, AVX2
    Vec8i t1 = _mm256_castps_si256(a);    // reinterpret as 32-bit integer
    Vec8i t2 = t1 >> 31;                  // extend sign bit
    return _mm256_castsi256_ps(t2);       // reinterpret as 32-bit Boolean
#else
    return Vec8fb(sign_bit(a.get_low()), sign_bit(a.get_high()));
#endif
}

// Function sign_combine: changes the sign of a when b has the sign bit set
// same as select(sign_bit(b), -a, a)
static inline Vec8f sign_combine(Vec8f const & a, Vec8f const & b) {
    Vec8f signmask = constant8f<(int)0x80000000,(int)0x80000000,(int)0x80000000,(int)0x80000000,(int)0x80000000,(int)0x80000000,(int)0x80000000,(int)0x80000000>();  // -0.0
    return a ^ (b & signmask);
}

// Function is_finite: gives true for elements that are normal, denormal or zero, 
// false for INF and NAN
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
static inline Vec8fb is_finite(Vec8f const & a) {
#if defined (VECTORI_H) && VECTORI_H > 1  // 256 bit integer vectors are available, AVX2
    Vec8i t1 = _mm256_castps_si256(a);    // reinterpret as 32-bit integer
    Vec8i t2 = t1 << 1;                // shift out sign bit
    Vec8ib t3 = Vec8i(t2 & 0xFF000000) != 0xFF000000; // exponent field is not all 1s
    return t3;
#else
    return Vec8fb(is_finite(a.get_low()), is_finite(a.get_high()));
#endif
}

// Function is_inf: gives true for elements that are +INF or -INF
// false for finite numbers and NAN
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
static inline Vec8fb is_inf(Vec8f const & a) {
#if defined (VECTORI_H) && VECTORI_H > 1  // 256 bit integer vectors are available, AVX2
    Vec8i t1 = _mm256_castps_si256(a); // reinterpret as 32-bit integer
    Vec8i t2 = t1 << 1;                // shift out sign bit
    return t2 == 0xFF000000;           // exponent is all 1s, fraction is 0
#else
    return Vec8fb(is_inf(a.get_low()), is_inf(a.get_high()));
#endif
}

// Function is_nan: gives true for elements that are +NAN or -NAN
// false for finite numbers and +/-INF
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
static inline Vec8fb is_nan(Vec8f const & a) {
#if defined (VECTORI_H) && VECTORI_H > 1  // 256 bit integer vectors are available, AVX2
    Vec8i t1 = _mm256_castps_si256(a); // reinterpret as 32-bit integer
    Vec8i t2 = t1 << 1;                // shift out sign bit
    Vec8i t3 = 0xFF000000;             // exponent mask
    Vec8i t4 = t2 & t3;                // exponent
    Vec8i t5 = _mm256_andnot_si256(t3,t2);// fraction
    return Vec8ib(t4 == t3 && t5 != 0);// exponent = all 1s and fraction != 0
#else
    return Vec8fb(is_nan(a.get_low()), is_nan(a.get_high()));
#endif
}

// Function is_subnormal: gives true for elements that are denormal (subnormal)
// false for finite numbers, zero, NAN and INF
static inline Vec8fb is_subnormal(Vec8f const & a) {
#if defined (VECTORI_H) && VECTORI_H > 1  // 256 bit integer vectors are available, AVX2
    Vec8i t1 = _mm256_castps_si256(a);    // reinterpret as 32-bit integer
    Vec8i t2 = t1 << 1;                   // shift out sign bit
    Vec8i t3 = 0xFF000000;                // exponent mask
    Vec8i t4 = t2 & t3;                   // exponent
    Vec8i t5 = _mm256_andnot_si256(t3,t2);// fraction
    return Vec8ib(t4 == 0 && t5 != 0);    // exponent = 0 and fraction != 0
#else
    return Vec8fb(is_subnormal(a.get_low()), is_subnormal(a.get_high()));
#endif
}

// Function is_zero_or_subnormal: gives true for elements that are zero or subnormal (denormal)
// false for finite numbers, NAN and INF
static inline Vec8fb is_zero_or_subnormal(Vec8f const & a) {
#if defined (VECTORI_H) && VECTORI_H > 1   // 256 bit integer vectors are available, AVX2
    Vec8i t = _mm256_castps_si256(a);            // reinterpret as 32-bit integer
          t &= 0x7F800000;                       // isolate exponent
    return t == 0;                               // exponent = 0
#else
    return Vec8fb(is_zero_or_subnormal(a.get_low()), is_zero_or_subnormal(a.get_high()));
#endif
}

// Function infinite4f: returns a vector where all elements are +INF
static inline Vec8f infinite8f() {
    return constant8f<0x7F800000,0x7F800000,0x7F800000,0x7F800000,0x7F800000,0x7F800000,0x7F800000,0x7F800000>();
}

// Function nan4f: returns a vector where all elements are +NAN (quiet)
static inline Vec8f nan8f(int n = 0x10) {
    return _mm256_castsi256_ps(_mm256_set1_epi32(0x7FC00000 + n));
}

// change signs on vectors Vec8f
// Each index i0 - i7 is 1 for changing sign on the corresponding element, 0 for no change
template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
static inline Vec8f change_sign(Vec8f const & a) {
    if ((i0 | i1 | i2 | i3 | i4 | i5 | i6 | i7) == 0) return a;
    __m256 mask = constant8f<i0 ? (int)0x80000000 : 0, i1 ? (int)0x80000000 : 0, i2 ? (int)0x80000000 : 0, i3 ? (int)0x80000000 : 0,
        i4 ? (int)0x80000000 : 0, i5 ? (int)0x80000000 : 0, i6 ? (int)0x80000000 : 0, i7 ? (int)0x80000000 : 0> ();
    return _mm256_xor_ps(a, mask);
}


/*****************************************************************************
*
*          Vec4d: Vector of 4 double precision floating point values
*
*****************************************************************************/

class Vec4d {
protected:
    pack4d_t ymm; // double vector
public:
    // Default constructor:
    Vec4d() {
    }
    // Constructor to broadcast the same value into all elements:
    Vec4d(double d) {
        ymm = nsimd::set<pack4d_t>(d);
    }
    // Constructor to build from all elements:
    Vec4d(double d0, double d1, double d2, double d3) {
        ymm = _mm256_setr_pd(d0, d1, d2, d3); 
    }
    // Constructor to build from two Vec2d:
    Vec4d(Vec2d const & a0, Vec2d const & a1) {
        ymm = _mm256_castps_pd(set_m128r(_mm_castpd_ps(a0), _mm_castpd_ps(a1)));
        //ymm = _mm256_set_m128d(a1, a0);
    }
    // Constructor to convert from type __m256d used in intrinsics:
    Vec4d(pack4d_t const & x) {
        ymm = x;
    }
    // Assignment operator to convert from type __m256d used in intrinsics:
    Vec4d & operator = (pack4d_t const & x) {
        ymm = x;
        return *this;
    }
    // Type cast operator to convert to __m256d used in intrinsics
    operator pack4d_t() const {
        return ymm;
    }
    // Member function to load from array (unaligned)
    Vec4d & load(double const * p) {
        ymm = nsimd::loadu(p);
        return *this;
    }
    // Member function to load from array, aligned by 32
    // You may use load_a instead of load if you are certain that p points to an address
    // divisible by 32
    Vec4d & load_a(double const * p) {
        ymm = nsimd::loada(p);
        return *this;
    }
    // Member function to store into array (unaligned)
    void store(double * p) const {
        nsimd::storeu(p, ymm);
    }
    // Member function to store into array, aligned by 32
    // You may use store_a instead of store if you are certain that p points to an address
    // divisible by 32
    void store_a(double * p) const {
        nsimd::storea(p, ymm);
    }
    // Partial load. Load n elements and set the rest to 0
    Vec4d & load_partial(int n, double const * p) {
        if (n > 0 && n <= 2) {
            *this = Vec4d(Vec2d().load_partial(n, p), _mm_setzero_pd());
        }
        else if (n > 2 && n <= 4) {
            *this = Vec4d(Vec2d().load(p), Vec2d().load_partial(n - 2, p + 2));
        }
        else {
            ymm = nsimd::set1<pack4d_t>(0);
        }
        return *this;
    }
    // Partial store. Store n elements
    void store_partial(int n, double * p) const {
        if (n <= 2) {
            get_low().store_partial(n, p);
        }
        else if (n <= 4) {
            get_low().store(p);
            get_high().store_partial(n - 2, p + 2);
        }
    }
    // cut off vector to n elements. The last 4-n elements are set to zero
    Vec4d & cutoff(int n) {
        ymm = _mm256_castps_pd(Vec8f(_mm256_castpd_ps(ymm)).cutoff(n*2));
        return *this;
    }
    // Member function to change a single element in vector
    // Note: This function is inefficient. Use load function if changing more than one element
    Vec4d const & insert(uint32_t index, double value) {
        pack4d_t v0 = _mm256_broadcast_sd(&value);
        switch (index) {
        case 0:
            ymm = nsimd::if_else1 (ymm, v0, 1);  break;
        case 1:
            ymm = nsimd::if_else1 (ymm, v0, 2);  break;
        case 2:
            ymm = nsimd::if_else1 (ymm, v0, 4);  break;
        default:
            ymm = nsimd::if_else1 (ymm, v0, 8);  break;
        }
        return *this;
    }
    // Member function extract a single element from vector
    double extract(uint32_t index) const {
        double x[4];
        store(x);
        return x[index & 3];
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    double operator [] (uint32_t index) const {
        return extract(index);
    }
    // Member functions to split into two Vec2d:
    Vec2d get_low() const {
        return _mm256_castpd256_pd128(ymm);
    }
    Vec2d get_high() const {
        return _mm256_extractf128_pd(ymm,1);
    }
    static int size () {
        return 4;
    }
};



/*****************************************************************************
*
*          Operators for Vec4d
*
*****************************************************************************/

// vector operator + : add element by element
static inline Vec4d operator + (Vec4d const & a, Vec4d const & b) {
    return nsimd::add(a, b);
}

// vector operator + : add vector and scalar
static inline Vec4d operator + (Vec4d const & a, double b) {
    return a + Vec4d(b);
}
static inline Vec4d operator + (double a, Vec4d const & b) {
    return Vec4d(a) + b;
}

// vector operator += : add
static inline Vec4d & operator += (Vec4d & a, Vec4d const & b) {
    a = a + b;
    return a;
}

// postfix operator ++
static inline Vec4d operator ++ (Vec4d & a, int) {
    Vec4d a0 = a;
    a = a + 1.0;
    return a0;
}

// prefix operator ++
static inline Vec4d & operator ++ (Vec4d & a) {
    a = a + 1.0;
    return a;
}

// vector operator - : subtract element by element
static inline Vec4d operator - (Vec4d const & a, Vec4d const & b) {
    return nsimd::sub(a, b);
}

// vector operator - : subtract vector and scalar
static inline Vec4d operator - (Vec4d const & a, double b) {
    return a - Vec4d(b);
}
static inline Vec4d operator - (double a, Vec4d const & b) {
    return Vec4d(a) - b;
}

// vector operator - : unary minus
// Change sign bit, even for 0, INF and NAN
static inline Vec4d operator - (Vec4d const & a) {
    return nsimd::xorb(a, nsimd::cvt_f32_f64(constant8f<0,(int)0x80000000,0,(int)0x80000000,0,(int)0x80000000,0,(int)0x80000000> ()));
}

// vector operator -= : subtract
static inline Vec4d & operator -= (Vec4d & a, Vec4d const & b) {
    a = a - b;
    return a;
}

// postfix operator --
static inline Vec4d operator -- (Vec4d & a, int) {
    Vec4d a0 = a;
    a = a - 1.0;
    return a0;
}

// prefix operator --
static inline Vec4d & operator -- (Vec4d & a) {
    a = a - 1.0;
    return a;
}

// vector operator * : multiply element by element
static inline Vec4d operator * (Vec4d const & a, Vec4d const & b) {
    return nsimd::mul(a, b);
}

// vector operator * : multiply vector and scalar
static inline Vec4d operator * (Vec4d const & a, double b) {
    return a * Vec4d(b);
}
static inline Vec4d operator * (double a, Vec4d const & b) {
    return Vec4d(a) * b;
}

// vector operator *= : multiply
static inline Vec4d & operator *= (Vec4d & a, Vec4d const & b) {
    a = a * b;
    return a;
}

// vector operator / : divide all elements by same integer
static inline Vec4d operator / (Vec4d const & a, Vec4d const & b) {
    return nsimd::div(a, b);
}

// vector operator / : divide vector and scalar
static inline Vec4d operator / (Vec4d const & a, double b) {
    return a / Vec4d(b);
}
static inline Vec4d operator / (double a, Vec4d const & b) {
    return Vec4d(a) / b;
}

// vector operator /= : divide
static inline Vec4d & operator /= (Vec4d & a, Vec4d const & b) {
    a = a / b;
    return a;
}

// vector operator == : returns true for elements for which a == b
static inline Vec4db operator == (Vec4d const & a, Vec4d const & b) {
    return nsimd::eq(a, b);
}

// vector operator != : returns true for elements for which a != b
static inline Vec4db operator != (Vec4d const & a, Vec4d const & b) {
    return nsimd::notl(nsimd::eq(a, b));
}

// vector operator < : returns true for elements for which a < b
static inline Vec4db operator < (Vec4d const & a, Vec4d const & b) {
    return nsimd::lt(a, b);
}

// vector operator <= : returns true for elements for which a <= b
static inline Vec4db operator <= (Vec4d const & a, Vec4d const & b) {
    return nsimd::le(a, b);
}

// vector operator > : returns true for elements for which a > b
static inline Vec4db operator > (Vec4d const & a, Vec4d const & b) {
    return b < a;
}

// vector operator >= : returns true for elements for which a >= b
static inline Vec4db operator >= (Vec4d const & a, Vec4d const & b) {
    return b <= a;
}

// Bitwise logical operators

// vector operator & : bitwise and
static inline Vec4d operator & (Vec4d const & a, Vec4d const & b) {
    return nsimd::andb(a, b);
}

// vector operator &= : bitwise and
static inline Vec4d & operator &= (Vec4d & a, Vec4d const & b) {
    a = a & b;
    return a;
}

// vector operator & : bitwise and of Vec4d and Vec4db
static inline Vec4d operator & (Vec4d const & a, Vec4db const & b) {
    return nsimd::andb(a, b);
}
static inline Vec4d operator & (Vec4db const & a, Vec4d const & b) {
    return nsimd::andb(a, b);
}

// vector operator | : bitwise or
static inline Vec4d operator | (Vec4d const & a, Vec4d const & b) {
    return nsimd::orb(a, b);
}

// vector operator |= : bitwise or
static inline Vec4d & operator |= (Vec4d & a, Vec4d const & b) {
    a = a | b;
    return a;
}

// vector operator ^ : bitwise xor
static inline Vec4d operator ^ (Vec4d const & a, Vec4d const & b) {
    return nsimd::xorb(a, b);
}

// vector operator ^= : bitwise xor
static inline Vec4d & operator ^= (Vec4d & a, Vec4d const & b) {
    a = a ^ b;
    return a;
}

// vector operator ! : logical not. Returns Boolean vector
static inline Vec4db operator ! (Vec4d const & a) {
    return a == Vec4d(0.0);
}


/*****************************************************************************
*
*          Functions for Vec4d
*
*****************************************************************************/

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 2; i++) result[i] = s[i] ? a[i] : b[i];
// Each byte in s must be either 0 (false) or 0xFFFFFFFFFFFFFFFF (true). 
// No other values are allowed.
static inline Vec4d select (Vec4db const & s, Vec4d const & a, Vec4d const & b) {
    return _mm256_blendv_pd(b, a, s);
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
static inline Vec4d if_add (Vec4db const & f, Vec4d const & a, Vec4d const & b) {
    return a + (Vec4d(f) & b);
}

// Conditional multiply: For all vector elements i: result[i] = f[i] ? (a[i] * b[i]) : a[i]
static inline Vec4d if_mul (Vec4db const & f, Vec4d const & a, Vec4d const & b) {
    return a * select(f, b, 1.);
}


// General arithmetic functions, etc.

// Horizontal add: Calculates the sum of all vector elements.
static inline double horizontal_add (Vec4d const & a) {
    pack4d_t t1 = _mm256_hadd_pd(a,a);
    __m128d t2 = _mm256_extractf128_pd(t1,1);
    __m128d t3 = _mm_add_sd(_mm256_castpd256_pd128(t1),t2);
    return _mm_cvtsd_f64(t3);        
}

// function max: a > b ? a : b
static inline Vec4d max(Vec4d const & a, Vec4d const & b) {
    return nsimd::max(a,b);
}

// function min: a < b ? a : b
static inline Vec4d min(Vec4d const & a, Vec4d const & b) {
    return nsimd::min(a,b);
}

// function abs: absolute value
// Removes sign bit, even for -0.0f, -INF and -NAN
static inline Vec4d abs(Vec4d const & a) {
    return nsimd::abs(a);
}

// function sqrt: square root
static inline Vec4d sqrt(Vec4d const & a) {
    return nsimd::sqrt(a);
}

// function square: a * a
static inline Vec4d square(Vec4d const & a) {
    return a * a;
}

// pow(Vec4d, int):
template <typename TT> static Vec4d pow(Vec4d const & a, TT const & n);

// Raise floating point numbers to integer power n
template <>
inline Vec4d pow<int>(Vec4d const & x0, int const & n) {
    return pow_template_i<Vec4d>(x0, n);
}

// allow conversion from unsigned int
template <>
inline Vec4d pow<uint32_t>(Vec4d const & x0, uint32_t const & n) {
    return pow_template_i<Vec4d>(x0, (int)n);
}


// Raise floating point numbers to integer power n, where n is a compile-time constant
template <int n>
static inline Vec4d pow_n(Vec4d const & a) {
    if (n < 0)    return Vec4d(1.0) / pow_n<-n>(a);
    if (n == 0)   return Vec4d(1.0);
    if (n >= 256) return pow(a, n);
    Vec4d x = a;                       // a^(2^i)
    Vec4d y;                           // accumulator
    const int lowest = n - (n & (n-1));// lowest set bit in n
    if (n & 1) y = x;
    if (n < 2) return y;
    x = x*x;                           // x^2
    if (n & 2) {
        if (lowest == 2) y = x; else y *= x;
    }
    if (n < 4) return y;
    x = x*x;                           // x^4
    if (n & 4) {
        if (lowest == 4) y = x; else y *= x;
    }
    if (n < 8) return y;
    x = x*x;                           // x^8
    if (n & 8) {
        if (lowest == 8) y = x; else y *= x;
    }
    if (n < 16) return y;
    x = x*x;                           // x^16
    if (n & 16) {
        if (lowest == 16) y = x; else y *= x;
    }
    if (n < 32) return y;
    x = x*x;                           // x^32
    if (n & 32) {
        if (lowest == 32) y = x; else y *= x;
    }
    if (n < 64) return y;
    x = x*x;                           // x^64
    if (n & 64) {
        if (lowest == 64) y = x; else y *= x;
    }
    if (n < 128) return y;
    x = x*x;                           // x^128
    if (n & 128) {
        if (lowest == 128) y = x; else y *= x;
    }
    return y;
}

template <int n>
static inline Vec4d pow(Vec4d const & a, Const_int_t<n>) {
    return pow_n<n>(a);
}


// function round: round to nearest integer (even). (result as double vector)
static inline Vec4d round(Vec4d const & a) {
    return nsimd::round_to_even(a);
}

// function truncate: round towards zero. (result as double vector)
static inline Vec4d truncate(Vec4d const & a) {
    return nsimd::trunc(a);
}

// function floor: round towards minus infinity. (result as double vector)
static inline Vec4d floor(Vec4d const & a) {
    return nsimd::floor(a);
}

// function ceil: round towards plus infinity. (result as double vector)
static inline Vec4d ceil(Vec4d const & a) {
    return nsimd::ceil(a);
}

// function round_to_int: round to nearest integer (even). (result as integer vector)
static inline Vec4i round_to_int(Vec4d const & a) {
    // Note: assume MXCSR control register is set to rounding
    return nsimd::cvt_f64_i32(a);
}

// function truncate_to_int: round towards zero. (result as integer vector)
static inline Vec4i truncate_to_int(Vec4d const & a) {
    return _mm256_cvttpd_epi32(a);
}

#ifdef VECTORI_H  // 256 bit integer vectors are available

// function truncate_to_int64: round towards zero. (inefficient)
static inline Vec4q truncate_to_int64(Vec4d const & a) {
#if defined (__AVX512DQ__) && defined (__AVX512VL__)
    //return _mm256_maskz_cvttpd_epi64( __mmask8(0xFF), a);
    return _mm256_cvttpd_epi64(a);
#else
    double aa[4];
    a.store(aa);
    return Vec4q(int64_t(aa[0]), int64_t(aa[1]), int64_t(aa[2]), int64_t(aa[3]));
#endif
}

// function truncate_to_int64_limited: round towards zero.
// result as 64-bit integer vector, but with limited range. Deprecated!
static inline Vec4q truncate_to_int64_limited(Vec4d const & a) {
#if defined (__AVX512DQ__) && defined (__AVX512VL__)
    return truncate_to_int64(a);
#elif VECTORI_H > 1
    // Note: assume MXCSR control register is set to rounding
    Vec2q   b = _mm256_cvttpd_epi32(a);                    // round to 32-bit integers
    __m256i c = permute4q<0,-256,1,-256>(Vec4q(b,b));      // get bits 64-127 to position 128-191
    __m256i s = _mm256_srai_epi32(c, 31);                  // sign extension bits
    return      _mm256_unpacklo_epi32(c, s);               // interleave with sign extensions
#else
    return Vec4q(truncate_to_int64_limited(a.get_low()), truncate_to_int64_limited(a.get_high()));
#endif
} 

// function round_to_int64: round to nearest or even. (inefficient)
static inline Vec4q round_to_int64(Vec4d const & a) {
#if defined (__AVX512DQ__) && defined (__AVX512VL__)
    return _mm256_cvtpd_epi64(a);
#else
    return truncate_to_int64(round(a));
#endif
}

// function round_to_int64_limited: round to nearest integer (even)
// result as 64-bit integer vector, but with limited range. Deprecated!
static inline Vec4q round_to_int64_limited(Vec4d const & a) {
#if defined (__AVX512DQ__) && defined (__AVX512VL__)
    return round_to_int64(a);
#elif VECTORI_H > 1
    // Note: assume MXCSR control register is set to rounding
    Vec2q   b = _mm256_cvtpd_epi32(a);                     // round to 32-bit integers
    __m256i c = permute4q<0,-256,1,-256>(Vec4q(b,b));      // get bits 64-127 to position 128-191
    __m256i s = _mm256_srai_epi32(c, 31);                  // sign extension bits
    return      _mm256_unpacklo_epi32(c, s);               // interleave with sign extensions
#else
    return Vec4q(round_to_int64_limited(a.get_low()), round_to_int64_limited(a.get_high()));
#endif
}

// function to_double: convert integer vector elements to double vector (inefficient)
static inline Vec4d to_double(Vec4q const & a) {
#if defined (__AVX512DQ__) && defined (__AVX512VL__)
        return _mm256_maskz_cvtepi64_pd( __mmask16(0xFF), a);
#else
        int64_t aa[4];
        a.store(aa);
        return Vec4d(double(aa[0]), double(aa[1]), double(aa[2]), double(aa[3]));
#endif
}

// function to_double_limited: convert integer vector elements to double vector
// limited to abs(x) < 2^31. Deprecated!
static inline Vec4d to_double_limited(Vec4q const & x) {
#if defined (__AVX512DQ__) && defined (__AVX512VL__)
    return to_double(x);
#else
    Vec8i compressed = permute8i<0,2,4,6,-256,-256,-256,-256>(Vec8i(x));
    return _mm256_cvtepi32_pd(compressed.get_low());  // AVX
#endif
}

#endif // VECTORI_H

// function to_double: convert integer vector to double vector
static inline Vec4d to_double(Vec4i const & a) {
    return nsimd::cvt_i32_d64(a);
}

// function compress: convert two Vec4d to one Vec8f
static inline Vec8f compress (Vec4d const & low, Vec4d const & high) {
    __m128 t1 = _mm256_cvtpd_ps(low);
    __m128 t2 = _mm256_cvtpd_ps(high);
    return Vec8f(t1, t2);
}

// Function extend_low : convert Vec8f vector elements 0 - 3 to Vec4d
static inline Vec4d extend_low(Vec8f const & a) {
    return _mm256_cvtps_pd(_mm256_castps256_ps128(a));
}

// Function extend_high : convert Vec8f vector elements 4 - 7 to Vec4d
static inline Vec4d extend_high (Vec8f const & a) {
    return _mm256_cvtps_pd(_mm256_extractf128_ps(a,1));
}

// Fused multiply and add functions

// Multiply and add
static inline Vec4d mul_add(Vec4d const & a, Vec4d const & b, Vec4d const & c) {
    return nsimd::fma(a, b, c);   
}

// Multiply and subtract
static inline Vec4d mul_sub(Vec4d const & a, Vec4d const & b, Vec4d const & c) {
    return nsimd::fms(a, b, c);    
}

// Multiply and inverse subtract
static inline Vec4d nmul_add(Vec4d const & a, Vec4d const & b, Vec4d const & c) {
    nsimd::fnma(a, b, c);
}

// Multiply and subtract with extra precision on the intermediate calculations, 
// even if FMA instructions not supported, using Veltkamp-Dekker split
static inline Vec4d mul_sub_x(Vec4d const & a, Vec4d const & b, Vec4d const & c) {
    nsimd::fnms(a, b, c);
}


// Math functions using fast bit manipulation

#ifdef VECTORI_H  // 256 bit integer vectors are available
// Extract the exponent as an integer
// exponent(a) = floor(log2(abs(a)));
// exponent(1.0) = 0, exponent(0.0) = -1023, exponent(INF) = +1024, exponent(NAN) = +1024
static inline Vec4q exponent(Vec4d const & a) {
#if VECTORI_H > 1  // AVX2
    Vec4uq t1 = _mm256_castpd_si256(a);// reinterpret as 64-bit integer
    Vec4uq t2 = t1 << 1;               // shift out sign bit
    Vec4uq t3 = t2 >> 53;              // shift down logical to position 0
    Vec4q  t4 = Vec4q(t3) - 0x3FF;     // subtract bias from exponent
    return t4;
#else
    return Vec4q(exponent(a.get_low()), exponent(a.get_high()));
#endif
}

// Extract the fraction part of a floating point number
// a = 2^exponent(a) * fraction(a), except for a = 0
// fraction(1.0) = 1.0, fraction(5.0) = 1.25 
static inline Vec4d fraction(Vec4d const & a) {
#if VECTORI_H > 1  // AVX2
    Vec4uq t1 = _mm256_castpd_si256(a);   // reinterpret as 64-bit integer
    Vec4uq t2 = Vec4uq((t1 & 0x000FFFFFFFFFFFFF) | 0x3FF0000000000000); // set exponent to 0 + bias
    return _mm256_castsi256_pd(t2);
#else
    return Vec4d(fraction(a.get_low()), fraction(a.get_high()));
#endif
}

// Fast calculation of pow(2,n) with n integer
// n  =     0 gives 1.0
// n >=  1024 gives +INF
// n <= -1023 gives 0.0
// This function will never produce denormals, and never raise exceptions
static inline Vec4d exp2(Vec4q const & n) {
#if VECTORI_H > 1  // AVX2
    Vec4q t1 = max(n,  -0x3FF);        // limit to allowed range
    Vec4q t2 = min(t1,  0x400);
    Vec4q t3 = t2 + 0x3FF;             // add bias
    Vec4q t4 = t3 << 52;               // put exponent into position 52
    return _mm256_castsi256_pd(t4);       // reinterpret as double
#else
    return Vec4d(exp2(n.get_low()), exp2(n.get_high()));
#endif
}
//static inline Vec4d exp2(Vec4d const & x); // defined in vectormath_exp.h
#endif


// Categorization functions

// Function sign_bit: gives true for elements that have the sign bit set
// even for -0.0, -INF and -NAN
// Note that sign_bit(Vec4d(-0.0)) gives true, while Vec4d(-0.0) < Vec4d(0.0) gives false
static inline Vec4db sign_bit(Vec4d const & a) {
#if defined (VECTORI_H) && VECTORI_H > 1  // 256 bit integer vectors are available, AVX2
    Vec4q t1 = _mm256_castpd_si256(a);    // reinterpret as 64-bit integer
    Vec4q t2 = t1 >> 63;               // extend sign bit
    return _mm256_castsi256_pd(t2);       // reinterpret as 64-bit Boolean
#else
    return Vec4db(sign_bit(a.get_low()),sign_bit(a.get_high()));
#endif
}

// Function sign_combine: changes the sign of a when b has the sign bit set
// same as select(sign_bit(b), -a, a)
static inline Vec4d sign_combine(Vec4d const & a, Vec4d const & b) {
    Vec4d signmask = _mm256_castps_pd(constant8f<0,(int)0x80000000,0,(int)0x80000000,0,(int)0x80000000,0,(int)0x80000000>());  // -0.0
    return a ^ (b & signmask);
}

// Function is_finite: gives true for elements that are normal, denormal or zero, 
// false for INF and NAN
static inline Vec4db is_finite(Vec4d const & a) {
#if defined (VECTORI_H) && VECTORI_H > 1  // 256 bit integer vectors are available, AVX2
    Vec4q t1 = _mm256_castpd_si256(a); // reinterpret as 64-bit integer
    Vec4q t2 = t1 << 1;                // shift out sign bit
    Vec4q t3 = 0xFFE0000000000000;     // exponent mask
    Vec4qb t4 = Vec4q(t2 & t3) != t3;  // exponent field is not all 1s
    return t4;
#else
    return Vec4db(is_finite(a.get_low()),is_finite(a.get_high()));
#endif
}

// Function is_inf: gives true for elements that are +INF or -INF
// false for finite numbers and NAN
static inline Vec4db is_inf(Vec4d const & a) {
#if defined (VECTORI_H) && VECTORI_H > 1  // 256 bit integer vectors are available, AVX2
    Vec4q t1 = _mm256_castpd_si256(a); // reinterpret as 64-bit integer
    Vec4q t2 = t1 << 1;                // shift out sign bit
    return t2 == 0xFFE0000000000000;   // exponent is all 1s, fraction is 0
#else
    return Vec4db(is_inf(a.get_low()),is_inf(a.get_high()));
#endif
}

// Function is_nan: gives true for elements that are +NAN or -NAN
// false for finite numbers and +/-INF
static inline Vec4db is_nan(Vec4d const & a) {
#if defined (VECTORI_H) && VECTORI_H > 1  // 256 bit integer vectors are available, AVX2
    Vec4q t1 = _mm256_castpd_si256(a); // reinterpret as 64-bit integer
    Vec4q t2 = t1 << 1;                // shift out sign bit
    Vec4q t3 = 0xFFE0000000000000;     // exponent mask
    Vec4q t4 = t2 & t3;                // exponent
    Vec4q t5 = _mm256_andnot_si256(t3,t2);// fraction
    return Vec4qb(t4 == t3 && t5 != 0);// exponent = all 1s and fraction != 0
#else
    return Vec4db(is_nan(a.get_low()),is_nan(a.get_high()));
#endif
}

// Function is_subnormal: gives true for elements that are denormal (subnormal)
// false for finite numbers, zero, NAN and INF
static inline Vec4db is_subnormal(Vec4d const & a) {
#if defined (VECTORI_H) && VECTORI_H > 1  // 256 bit integer vectors are available, AVX2
    Vec4q t1 = _mm256_castpd_si256(a); // reinterpret as 64-bit integer
    Vec4q t2 = t1 << 1;                // shift out sign bit
    Vec4q t3 = 0xFFE0000000000000;     // exponent mask
    Vec4q t4 = t2 & t3;                // exponent
    Vec4q t5 = _mm256_andnot_si256(t3,t2);// fraction
    return Vec4qb(t4 == 0 && t5 != 0); // exponent = 0 and fraction != 0
#else
    return Vec4db(is_subnormal(a.get_low()),is_subnormal(a.get_high()));
#endif
}

// Function is_zero_or_subnormal: gives true for elements that are zero or subnormal (denormal)
// false for finite numbers, NAN and INF
static inline Vec4db is_zero_or_subnormal(Vec4d const & a) {
#if defined (VECTORI_H) && VECTORI_H > 1  // 256 bit integer vectors are available, AVX2
    Vec4q t = _mm256_castpd_si256(a);     // reinterpret as 32-bit integer
          t &= 0x7FF0000000000000ll;   // isolate exponent
    return t == 0;                     // exponent = 0
#else
    return Vec4db(is_zero_or_subnormal(a.get_low()),is_zero_or_subnormal(a.get_high()));
#endif
}

// Function infinite2d: returns a vector where all elements are +INF
static inline Vec4d infinite4d() {
    return _mm256_castps_pd(constant8f<0,0x7FF00000,0,0x7FF00000,0,0x7FF00000,0,0x7FF00000>());
}

// Function nan4d: returns a vector where all elements are +NAN (quiet)
static inline Vec4d nan4d(int n = 0x10) {
#if defined (VECTORI_H) && VECTORI_H > 1  // 256 bit integer vectors are available, AVX2
    return _mm256_castsi256_pd(Vec4q(0x7FF8000000000000 + n));
#else
    return Vec4d(nan2d(n),nan2d(n));
#endif
}

// change signs on vectors Vec4d
// Each index i0 - i3 is 1 for changing sign on the corresponding element, 0 for no change
template <int i0, int i1, int i2, int i3>
static inline Vec4d change_sign(Vec4d const & a) {
    if ((i0 | i1 | i2 | i3) == 0) return a;
    __m256 mask = constant8f<0, i0 ? (int)0x80000000 : 0, 0, i1 ? (int)0x80000000 : 0, 0, i2 ? (int)0x80000000 : 0, 0, i3 ? (int)0x80000000 : 0> ();
    return _mm256_xor_pd(a, _mm256_castps_pd(mask));
}

/*****************************************************************************
*
*          Vec8d: Vector of 8 double precision floating point values
*
*****************************************************************************/

class Vec8d {
protected:
    __m512d zmm; // double vector
public:
    // Default constructor:
    Vec8d() {
    }
    // Constructor to broadcast the same value into all elements:
    Vec8d(double d) {
        zmm = _mm512_set1_pd(d);
    }
    // Constructor to build from all elements:
    Vec8d(double d0, double d1, double d2, double d3, double d4, double d5, double d6, double d7) {
        zmm = _mm512_setr_pd(d0, d1, d2, d3, d4, d5, d6, d7); 
    }
    // Constructor to build from two Vec4d:
    Vec8d(Vec4d const & a0, Vec4d const & a1) {
        zmm = _mm512_insertf64x4(_mm512_castpd256_pd512(a0), a1, 1);
    }
    // Constructor to convert from type __m512d used in intrinsics:
    Vec8d(__m512d const & x) {
        zmm = x;
    }
    // Assignment operator to convert from type __m512d used in intrinsics:
    Vec8d & operator = (__m512d const & x) {
        zmm = x;
        return *this;
    }
    // Type cast operator to convert to __m512d used in intrinsics
    operator __m512d() const {
        return zmm;
    }
    // Member function to load from array (unaligned)
    Vec8d & load(double const * p) {
        zmm = _mm512_loadu_pd(p);
        return *this;
    }
    // Member function to load from array, aligned by 64
    // You may use load_a instead of load if you are certain that p points to an address
    // divisible by 64
    Vec8d & load_a(double const * p) {
        zmm = _mm512_load_pd(p);
        return *this;
    }
    // Member function to store into array (unaligned)
    void store(double * p) const {
        _mm512_storeu_pd(p, zmm);
    }
    // Member function to store into array, aligned by 64
    // You may use store_a instead of store if you are certain that p points to an address
    // divisible by 64
    void store_a(double * p) const {
        _mm512_store_pd(p, zmm);
    }
    // Partial load. Load n elements and set the rest to 0
    Vec8d & load_partial(int n, double const * p) {
        zmm = _mm512_maskz_loadu_pd(__mmask16((1<<n)-1), p);
        return *this;
    }
    // Partial store. Store n elements
    void store_partial(int n, double * p) const {
        _mm512_mask_storeu_pd(p, __mmask16((1<<n)-1), zmm);
    }
    // cut off vector to n elements. The last 8-n elements are set to zero
    Vec8d & cutoff(int n) {
        zmm = _mm512_maskz_mov_pd(__mmask16((1<<n)-1), zmm);
        return *this;
    }
    // Member function to change a single element in vector
    // Note: This function is inefficient. Use load function if changing more than one element
    Vec8d const & insert(uint32_t index, double value) {
        //zmm = _mm512_mask_set1_pd(zmm, __mmask16(1 << index), value);  // this intrinsic function does not exist (yet?)
        zmm = _mm512_castsi512_pd(_mm512_mask_set1_epi64(_mm512_castpd_si512(zmm), __mmask16(1 << index), *(int64_t*)&value)); // ignore warning
        return *this;
    }
    // Member function extract a single element from vector
    double extract(uint32_t index) const {
        double a[8];
        store(a);
        return a[index & 7];        
    }

    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    double operator [] (uint32_t index) const {
        return extract(index);
    }
    // Member functions to split into two Vec4d:
    Vec4d get_low() const {
        return _mm512_castpd512_pd256(zmm);
    }
    Vec4d get_high() const {
        return _mm512_extractf64x4_pd(zmm,1);
    }
    static int size () {
        return 8;
    }
};



/*****************************************************************************
*
*          Operators for Vec8d
*
*****************************************************************************/

// vector operator + : add element by element
static inline Vec8d operator + (Vec8d const & a, Vec8d const & b) {
    return _mm512_add_pd(a, b);
}

// vector operator + : add vector and scalar
static inline Vec8d operator + (Vec8d const & a, double b) {
    return a + Vec8d(b);
}
static inline Vec8d operator + (double a, Vec8d const & b) {
    return Vec8d(a) + b;
}

// vector operator += : add
static inline Vec8d & operator += (Vec8d & a, Vec8d const & b) {
    a = a + b;
    return a;
}

// postfix operator ++
static inline Vec8d operator ++ (Vec8d & a, int) {
    Vec8d a0 = a;
    a = a + 1.0;
    return a0;
}

// prefix operator ++
static inline Vec8d & operator ++ (Vec8d & a) {
    a = a + 1.0;
    return a;
}

// vector operator - : subtract element by element
static inline Vec8d operator - (Vec8d const & a, Vec8d const & b) {
    return _mm512_sub_pd(a, b);
}

// vector operator - : subtract vector and scalar
static inline Vec8d operator - (Vec8d const & a, double b) {
    return a - Vec8d(b);
}
static inline Vec8d operator - (double a, Vec8d const & b) {
    return Vec8d(a) - b;
}

// vector operator - : unary minus
// Change sign bit, even for 0, INF and NAN
static inline Vec8d operator - (Vec8d const & a) {
    return _mm512_castsi512_pd(Vec8q(_mm512_castpd_si512(a)) ^ Vec8q(0x8000000000000000));
}

// vector operator -= : subtract
static inline Vec8d & operator -= (Vec8d & a, Vec8d const & b) {
    a = a - b;
    return a;
}

// postfix operator --
static inline Vec8d operator -- (Vec8d & a, int) {
    Vec8d a0 = a;
    a = a - 1.0;
    return a0;
}

// prefix operator --
static inline Vec8d & operator -- (Vec8d & a) {
    a = a - 1.0;
    return a;
}

// vector operator * : multiply element by element
static inline Vec8d operator * (Vec8d const & a, Vec8d const & b) {
    return _mm512_mul_pd(a, b);
}

// vector operator * : multiply vector and scalar
static inline Vec8d operator * (Vec8d const & a, double b) {
    return a * Vec8d(b);
}
static inline Vec8d operator * (double a, Vec8d const & b) {
    return Vec8d(a) * b;
}

// vector operator *= : multiply
static inline Vec8d & operator *= (Vec8d & a, Vec8d const & b) {
    a = a * b;
    return a;
}

// vector operator / : divide all elements by same integer
static inline Vec8d operator / (Vec8d const & a, Vec8d const & b) {
    return _mm512_div_pd(a, b);
}

// vector operator / : divide vector and scalar
static inline Vec8d operator / (Vec8d const & a, double b) {
    return a / Vec8d(b);
}
static inline Vec8d operator / (double a, Vec8d const & b) {
    return Vec8d(a) / b;
}

// vector operator /= : divide
static inline Vec8d & operator /= (Vec8d & a, Vec8d const & b) {
    a = a / b;
    return a;
}

// vector operator == : returns true for elements for which a == b
static inline Vec8db operator == (Vec8d const & a, Vec8d const & b) {
    return _mm512_cmp_pd_mask(a, b, 0);
}

// vector operator != : returns true for elements for which a != b
static inline Vec8db operator != (Vec8d const & a, Vec8d const & b) {
    return _mm512_cmp_pd_mask(a, b, 4);
}

// vector operator < : returns true for elements for which a < b
static inline Vec8db operator < (Vec8d const & a, Vec8d const & b) {
    return _mm512_cmp_pd_mask(a, b, 1);
}

// vector operator <= : returns true for elements for which a <= b
static inline Vec8db operator <= (Vec8d const & a, Vec8d const & b) {
    return _mm512_cmp_pd_mask(a, b, 2);
}

// vector operator > : returns true for elements for which a > b
static inline Vec8db operator > (Vec8d const & a, Vec8d const & b) {
    return b < a;
}

// vector operator >= : returns true for elements for which a >= b
static inline Vec8db operator >= (Vec8d const & a, Vec8d const & b) {
    return b <= a;
}

// Bitwise logical operators

// vector operator & : bitwise and
static inline Vec8d operator & (Vec8d const & a, Vec8d const & b) {
    return _mm512_castsi512_pd(Vec8q(_mm512_castpd_si512(a)) & Vec8q(_mm512_castpd_si512(b)));
}

// vector operator &= : bitwise and
static inline Vec8d & operator &= (Vec8d & a, Vec8d const & b) {
    a = a & b;
    return a;
}

// vector operator & : bitwise and of Vec8d and Vec8db
static inline Vec8d operator & (Vec8d const & a, Vec8db const & b) {
    return _mm512_maskz_mov_pd(b, a);
}

static inline Vec8d operator & (Vec8db const & a, Vec8d const & b) {
    return b & a;
}

// vector operator | : bitwise or
static inline Vec8d operator | (Vec8d const & a, Vec8d const & b) {
    return _mm512_castsi512_pd(Vec8q(_mm512_castpd_si512(a)) | Vec8q(_mm512_castpd_si512(b)));
}

// vector operator |= : bitwise or
static inline Vec8d & operator |= (Vec8d & a, Vec8d const & b) {
    a = a | b;
    return a;
}

// vector operator ^ : bitwise xor
static inline Vec8d operator ^ (Vec8d const & a, Vec8d const & b) {
    return _mm512_castsi512_pd(Vec8q(_mm512_castpd_si512(a)) ^ Vec8q(_mm512_castpd_si512(b)));
}

// vector operator ^= : bitwise xor
static inline Vec8d & operator ^= (Vec8d & a, Vec8d const & b) {
    a = a ^ b;
    return a;
}

// vector operator ! : logical not. Returns Boolean vector
static inline Vec8db operator ! (Vec8d const & a) {
    return a == Vec8d(0.0);
}


/*****************************************************************************
*
*          Functions for Vec8d
*
*****************************************************************************/

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 2; i++) result[i] = s[i] ? a[i] : b[i];
static inline Vec8d select (Vec8db const & s, Vec8d const & a, Vec8d const & b) {
    return _mm512_mask_mov_pd (b, s, a);
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
static inline Vec8d if_add (Vec8db const & f, Vec8d const & a, Vec8d const & b) {
    return _mm512_mask_add_pd(a, f, a, b);
}

// Conditional multiply: For all vector elements i: result[i] = f[i] ? (a[i] * b[i]) : a[i]
static inline Vec8d if_mul (Vec8db const & f, Vec8d const & a, Vec8d const & b) {
    return _mm512_mask_mul_pd(a, f, a, b);
}


// General arithmetic functions, etc.

// Horizontal add: Calculates the sum of all vector elements.
static inline double horizontal_add (Vec8d const & a) {
#if defined(__INTEL_COMPILER)
    return _mm512_reduce_add_pd(a);
#else
    return horizontal_add(a.get_low() + a.get_high());
#endif
}

// function max: a > b ? a : b
static inline Vec8d max(Vec8d const & a, Vec8d const & b) {
    return _mm512_max_pd(a,b);
}

// function min: a < b ? a : b
static inline Vec8d min(Vec8d const & a, Vec8d const & b) {
    return _mm512_min_pd(a,b);
}

// function abs: absolute value
// Removes sign bit, even for -0.0f, -INF and -NAN
static inline Vec8d abs(Vec8d const & a) {
    return _mm512_castsi512_pd(Vec8q(_mm512_castpd_si512(a)) & Vec8q(0x7FFFFFFFFFFFFFFF));
}

// function sqrt: square root
static inline Vec8d sqrt(Vec8d const & a) {
    return _mm512_sqrt_pd(a);
}

// function square: a * a
static inline Vec8d square(Vec8d const & a) {
    return a * a;
}

// pow(Vec8d, int):
template <typename TT> static Vec8d pow(Vec8d const & a, TT const & n);

// Raise floating point numbers to integer power n
template <>
inline Vec8d pow<int>(Vec8d const & x0, int const & n) {
    return pow_template_i<Vec8d>(x0, n);
}

// allow conversion from unsigned int
template <>
inline Vec8d pow<uint32_t>(Vec8d const & x0, uint32_t const & n) {
    return pow_template_i<Vec8d>(x0, (int)n);
}


// Raise floating point numbers to integer power n, where n is a compile-time constant
template <int n>
static inline Vec8d pow_n(Vec8d const & a) {
    if (n < 0)    return Vec8d(1.0) / pow_n<-n>(a);
    if (n == 0)   return Vec8d(1.0);
    if (n >= 256) return pow(a, n);
    Vec8d x = a;                       // a^(2^i)
    Vec8d y;                           // accumulator
    const int lowest = n - (n & (n-1));// lowest set bit in n
    if (n & 1) y = x;
    if (n < 2) return y;
    x = x*x;                           // x^2
    if (n & 2) {
        if (lowest == 2) y = x; else y *= x;
    }
    if (n < 4) return y;
    x = x*x;                           // x^4
    if (n & 4) {
        if (lowest == 4) y = x; else y *= x;
    }
    if (n < 8) return y;
    x = x*x;                           // x^8
    if (n & 8) {
        if (lowest == 8) y = x; else y *= x;
    }
    if (n < 16) return y;
    x = x*x;                           // x^16
    if (n & 16) {
        if (lowest == 16) y = x; else y *= x;
    }
    if (n < 32) return y;
    x = x*x;                           // x^32
    if (n & 32) {
        if (lowest == 32) y = x; else y *= x;
    }
    if (n < 64) return y;
    x = x*x;                           // x^64
    if (n & 64) {
        if (lowest == 64) y = x; else y *= x;
    }
    if (n < 128) return y;
    x = x*x;                           // x^128
    if (n & 128) {
        if (lowest == 128) y = x; else y *= x;
    }
    return y;
}

template <int n>
static inline Vec8d pow(Vec8d const & a, Const_int_t<n>) {
    return pow_n<n>(a);
}


// function round: round to nearest integer (even). (result as double vector)
static inline Vec8d round(Vec8d const & a) {
    return _mm512_roundscale_pd(a, 0);
}

// function truncate: round towards zero. (result as double vector)
static inline Vec8d truncate(Vec8d const & a) {
    return _mm512_roundscale_pd(a, 3);
}

// function floor: round towards minus infinity. (result as double vector)
static inline Vec8d floor(Vec8d const & a) {
    return _mm512_roundscale_pd(a, 1);
}

// function ceil: round towards plus infinity. (result as double vector)
static inline Vec8d ceil(Vec8d const & a) {
    return _mm512_roundscale_pd(a, 2);
}

// function round_to_int: round to nearest integer (even). (result as integer vector)
static inline Vec8i round_to_int(Vec8d const & a) {
    //return _mm512_cvtpd_epi32(a);
    return _mm512_cvt_roundpd_epi32(a, 0+8);
}

// function truncate_to_int: round towards zero. (result as integer vector)
static inline Vec8i truncate_to_int(Vec8d const & a) {
    return _mm512_cvttpd_epi32(a);
}

// function truncate_to_int64: round towards zero. (inefficient)
static inline Vec8q truncate_to_int64(Vec8d const & a) {
#ifdef __AVX512DQ__
    return _mm512_cvttpd_epi64(a);
#else
    double aa[8];
    a.store(aa);
    return Vec8q(int64_t(aa[0]), int64_t(aa[1]), int64_t(aa[2]), int64_t(aa[3]), int64_t(aa[4]), int64_t(aa[5]), int64_t(aa[6]), int64_t(aa[7]));
#endif
}

// function truncate_to_int64_limited: round towards zero.
// result as 64-bit integer vector, but with limited range. Deprecated!
static inline Vec8q truncate_to_int64_limited(Vec8d const & a) {
#ifdef __AVX512DQ__
    return truncate_to_int64(a);
#else
    // Note: assume MXCSR control register is set to rounding
    Vec4q   b = _mm512_cvttpd_epi32(a);                    // round to 32-bit integers
    __m512i c = permute8q<0,-256,1,-256,2,-256,3,-256>(Vec8q(b,b));      // get bits 64-127 to position 128-191, etc.
    __m512i s = _mm512_srai_epi32(c, 31);                  // sign extension bits
    return      _mm512_unpacklo_epi32(c, s);               // interleave with sign extensions
#endif
} 

// function round_to_int64: round to nearest or even. (inefficient)
static inline Vec8q round_to_int64(Vec8d const & a) {
#ifdef __AVX512DQ__
    return _mm512_cvtpd_epi64(a);
#else
    return truncate_to_int64(round(a));
#endif
}

// function round_to_int64_limited: round to nearest integer (even)
// result as 64-bit integer vector, but with limited range. Deprecated!
static inline Vec8q round_to_int64_limited(Vec8d const & a) {
#ifdef __AVX512DQ__
    return round_to_int64(a);
#else
    Vec4q   b = _mm512_cvt_roundpd_epi32(a, 0+8);     // round to 32-bit integers   
    __m512i c = permute8q<0,-256,1,-256,2,-256,3,-256>(Vec8q(b,b));  // get bits 64-127 to position 128-191, etc.
    __m512i s = _mm512_srai_epi32(c, 31);                            // sign extension bits
    return      _mm512_unpacklo_epi32(c, s);                         // interleave with sign extensions
#endif
}

// function to_double: convert integer vector elements to double vector (inefficient)
static inline Vec8d to_double(Vec8q const & a) {
#if defined (__AVX512DQ__)
    return _mm512_cvtepi64_pd(a);
#else
    int64_t aa[8];
    a.store(aa);
    return Vec8d(double(aa[0]), double(aa[1]), double(aa[2]), double(aa[3]), double(aa[4]), double(aa[5]), double(aa[6]), double(aa[7]));
#endif
}

// function to_double_limited: convert integer vector elements to double vector
// limited to abs(x) < 2^31. Deprecated!
static inline Vec8d to_double_limited(Vec8q const & x) {
#if defined (__AVX512DQ__)
    return to_double(x);
#else
    Vec16i compressed = permute16i<0,2,4,6,8,10,12,14,-256,-256,-256,-256,-256,-256,-256,-256>(Vec16i(x));
    return _mm512_cvtepi32_pd(compressed.get_low());
#endif
}

// function to_double: convert integer vector to double vector
static inline Vec8d to_double(Vec8i const & a) {
    return _mm512_cvtepi32_pd(a);
}

// function compress: convert two Vec8d to one Vec16f
static inline Vec16f compress (Vec8d const & low, Vec8d const & high) {
    __m256 t1 = _mm512_cvtpd_ps(low);
    __m256 t2 = _mm512_cvtpd_ps(high);
    return Vec16f(t1, t2);
}

// Function extend_low : convert Vec16f vector elements 0 - 3 to Vec8d
static inline Vec8d extend_low(Vec16f const & a) {
    return _mm512_cvtps_pd(_mm512_castps512_ps256(a));
}

// Function extend_high : convert Vec16f vector elements 4 - 7 to Vec8d
static inline Vec8d extend_high (Vec16f const & a) {
    return _mm512_cvtps_pd(a.get_high());
}


// Fused multiply and add functions

// Multiply and add
static inline Vec8d mul_add(Vec8d const & a, Vec8d const & b, Vec8d const & c) {
    return _mm512_fmadd_pd(a, b, c);
}

// Multiply and subtract
static inline Vec8d mul_sub(Vec8d const & a, Vec8d const & b, Vec8d const & c) {
    return _mm512_fmsub_pd(a, b, c);
}

// Multiply and inverse subtract
static inline Vec8d nmul_add(Vec8d const & a, Vec8d const & b, Vec8d const & c) {
    return _mm512_fnmadd_pd(a, b, c);
}

// Multiply and subtract with extra precision on the intermediate calculations, 
static inline Vec8d mul_sub_x(Vec8d const & a, Vec8d const & b, Vec8d const & c) {
    return _mm512_fmsub_pd(a, b, c);
}


// Math functions using fast bit manipulation

// Extract the exponent as an integer
// exponent(a) = floor(log2(abs(a)));
// exponent(1.0) = 0, exponent(0.0) = -1023, exponent(INF) = +1024, exponent(NAN) = +1024
static inline Vec8q exponent(Vec8d const & a) {
    Vec8uq t1 = _mm512_castpd_si512(a);// reinterpret as 64-bit integer
    Vec8uq t2 = t1 << 1;               // shift out sign bit
    Vec8uq t3 = t2 >> 53;              // shift down logical to position 0
    Vec8q  t4 = Vec8q(t3) - 0x3FF;     // subtract bias from exponent
    return t4;
}

// Extract the fraction part of a floating point number
// a = 2^exponent(a) * fraction(a), except for a = 0
// fraction(1.0) = 1.0, fraction(5.0) = 1.25 
static inline Vec8d fraction(Vec8d const & a) {
    return _mm512_getmant_pd(a, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_zero);
}

// Fast calculation of pow(2,n) with n integer
// n  =     0 gives 1.0
// n >=  1024 gives +INF
// n <= -1023 gives 0.0
// This function will never produce denormals, and never raise exceptions
static inline Vec8d exp2(Vec8q const & n) {
    Vec8q t1 = max(n,  -0x3FF);        // limit to allowed range
    Vec8q t2 = min(t1,  0x400);
    Vec8q t3 = t2 + 0x3FF;             // add bias
    Vec8q t4 = t3 << 52;               // put exponent into position 52
    return _mm512_castsi512_pd(t4);    // reinterpret as double
}
//static Vec8d exp2(Vec8d const & x); // defined in vectormath_exp.h


// Categorization functions

// Function sign_bit: gives true for elements that have the sign bit set
// even for -0.0, -INF and -NAN
// Note that sign_bit(Vec8d(-0.0)) gives true, while Vec8d(-0.0) < Vec8d(0.0) gives false
static inline Vec8db sign_bit(Vec8d const & a) {
    Vec8q t1 = _mm512_castpd_si512(a);    // reinterpret as 64-bit integer
    return Vec8db(t1 < 0);
}

// Function sign_combine: changes the sign of a when b has the sign bit set
// same as select(sign_bit(b), -a, a)
static inline Vec8d sign_combine(Vec8d const & a, Vec8d const & b) {
    union {
        uint64_t i;
        double f;
    } u = {0x8000000000000000};  // mask for sign bit
    return a ^ (b & Vec8d(u.f));
}

// Function is_finite: gives true for elements that are normal, denormal or zero, 
// false for INF and NAN
static inline Vec8db is_finite(Vec8d const & a) {
#ifdef __AVX512DQ__
    __mmask8 f = _mm512_fpclass_pd_mask(a, 0x99);
    return _mm512_knot(f);
#else
    Vec8q  t1 = _mm512_castpd_si512(a); // reinterpret as 64-bit integer
    Vec8q  t2 = t1 << 1;                // shift out sign bit
    Vec8q  t3 = 0xFFE0000000000000ll;   // exponent mask
    Vec8qb t4 = Vec8q(t2 & t3) != t3;   // exponent field is not all 1s
    return Vec8db(t4);
#endif
}

// Function is_inf: gives true for elements that are +INF or -INF
// false for finite numbers and NAN
static inline Vec8db is_inf(Vec8d const & a) {
    Vec8q t1 = _mm512_castpd_si512(a);           // reinterpret as 64-bit integer
    Vec8q t2 = t1 << 1;                          // shift out sign bit
    return Vec8db(t2 == 0xFFE0000000000000ll);   // exponent is all 1s, fraction is 0
}

// Function is_nan: gives true for elements that are +NAN or -NAN
// false for finite numbers and +/-INF
static inline Vec8db is_nan(Vec8d const & a) {
    Vec8q t1 = _mm512_castpd_si512(a); // reinterpret as 64-bit integer
    Vec8q t2 = t1 << 1;                // shift out sign bit
    Vec8q t3 = 0xFFE0000000000000ll;   // exponent mask
    Vec8q t4 = t2 & t3;                // exponent
    Vec8q t5 = _mm512_andnot_si512(t3,t2);// fraction
    return Vec8db(t4 == t3 && t5 != 0);// exponent = all 1s and fraction != 0
}

// Function is_subnormal: gives true for elements that are denormal (subnormal)
// false for finite numbers, zero, NAN and INF
static inline Vec8db is_subnormal(Vec8d const & a) {
    Vec8q t1 = _mm512_castpd_si512(a); // reinterpret as 64-bit integer
    Vec8q t2 = t1 << 1;                // shift out sign bit
    Vec8q t3 = 0xFFE0000000000000ll;   // exponent mask
    Vec8q t4 = t2 & t3;                // exponent
    Vec8q t5 = _mm512_andnot_si512(t3,t2);// fraction
    return Vec8db(t4 == 0 && t5 != 0); // exponent = 0 and fraction != 0
}

// Function is_zero_or_subnormal: gives true for elements that are zero or subnormal (denormal)
// false for finite numbers, NAN and INF
static inline Vec8db is_zero_or_subnormal(Vec8d const & a) {
    Vec8q t = _mm512_castpd_si512(a);            // reinterpret as 32-bit integer
          t &= 0x7FF0000000000000ll;             // isolate exponent
    return Vec8db(t == 0);                       // exponent = 0
}

// Function infinite2d: returns a vector where all elements are +INF
static inline Vec8d infinite8d() {
    union {
        uint64_t i;
        double f;
    } u = {0x7FF0000000000000};
    return Vec8d(u.f);
}

// Function nan8d: returns a vector where all elements are +NAN (quiet NAN)
static inline Vec8d nan8d(int n = 0x10) {
    union {
        uint64_t i;
        double f;
    } u = {0x7FF8000000000000 + uint64_t(n)};
    return Vec8d(u.f);
}

// change signs on vectors Vec8d
// Each index i0 - i3 is 1 for changing sign on the corresponding element, 0 for no change
template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
static inline Vec8d change_sign(Vec8d const & a) {
    const __mmask16 m = __mmask16((i0&1) | (i1&1)<<1 | (i2&1)<< 2 | (i3&1)<<3 | (i4&1)<<4 | (i5&1)<<5 | (i6&1)<<6 | (i7&1)<<7);
    if ((uint8_t)m == 0) return a;
    __m512d s = _mm512_castsi512_pd(_mm512_maskz_set1_epi64(m, 0x8000000000000000));
    return a ^ s;
}

/*****************************************************************************
*
*          Functions for reinterpretation between vector types
*
*****************************************************************************/

#if defined (VECTORI_H) && VECTORI_H >= 2
// AVX2 vectors defined


// ABI version 4 or later needed on Gcc for correct mangling of 256-bit intrinsic vectors.
// It is recommended to compile with -fabi-version=0 to get the latest abi version
#if !defined (GCC_VERSION) || (defined (__GXX_ABI_VERSION) && __GXX_ABI_VERSION >= 1004)  
static inline __m256i reinterpret_i (__m256i const & x) {
    return x;
}

static inline __m256i reinterpret_i (__m256  const & x) {
    return _mm256_castps_si256(x);
}

static inline __m256i reinterpret_i (pack4d_t const & x) {
    return _mm256_castpd_si256(x);
}

static inline pack4f_t  reinterpret_f (__m256i const & x) {
    return _mm256_castsi256_ps(x);
}

static inline pack4f_t  reinterpret_f (pack4f_t  const & x) {
    return x;
}

static inline pack4f_t  reinterpret_f (pack4d_t const & x) {
    return _mm256_castpd_ps(x);
}

static inline pack4d_t reinterpret_d (__m256i const & x) {
    return _mm256_castsi256_pd(x);
}

static inline pack4d_t reinterpret_d (pack4f_t  const & x) {
    return _mm256_castps_pd(x);
}

static inline pack4d_t reinterpret_d (pack4d_t const & x) {
    return x;
}

#else  // __GXX_ABI_VERSION < 1004

static inline __m256i reinterpret_i (Vec32c const & x) {
    return x;
}

static inline __m256i reinterpret_i (Vec16s const & x) {
    return x;
}

static inline __m256i reinterpret_i (Vec8i const & x) {
    return x;
}

static inline __m256i reinterpret_i (Vec4q const & x) {
    return x;
}

static inline __m256i reinterpret_i (Vec8f  const & x) {
    return _mm256_castps_si256(x);
}

static inline __m256i reinterpret_i (Vec4d const & x) {
    return _mm256_castpd_si256(x);
}

static inline pack4f_t  reinterpret_f (Vec32c const & x) {
    return _mm256_castsi256_ps(x);
}

static inline pack4f_t  reinterpret_f (Vec16s const & x) {
    return _mm256_castsi256_ps(x);
}

static inline pack4f_t  reinterpret_f (Vec8i const & x) {
    return _mm256_castsi256_ps(x);
}

static inline pack4f_t  reinterpret_f (Vec4q const & x) {
    return _mm256_castsi256_ps(x);
}

static inline pack4f_t  reinterpret_f (Vec8f  const & x) {
    return x;
}

static inline pack4f_t  reinterpret_f (Vec4d const & x) {
    return _mm256_castpd_ps(x);
}

static inline pack4d_t reinterpret_d (Vec32c const & x) {
    return _mm256_castsi256_pd(x);
}

static inline pack4d_t reinterpret_d (Vec16s const & x) {
    return _mm256_castsi256_pd(x);
}

static inline pack4d_t reinterpret_d (Vec8i const & x) {
    return _mm256_castsi256_pd(x);
}

static inline pack4d_t reinterpret_d (Vec4q const & x) {
    return _mm256_castsi256_pd(x);
}

static inline pack4d_t reinterpret_d (Vec8f  const & x) {
    return _mm256_castps_pd(x);
}

static inline pack4d_t reinterpret_d (Vec4d const & x) {
    return x;
}

#endif  // __GXX_ABI_VERSION

#else
// AVX2 emulated in vectori256e.h, AVX supported

// ABI version 4 or later needed on Gcc for correct mangling of 256-bit intrinsic vectors.
// It is recommended to compile with -fabi-version=0 to get the latest abi version
#if !defined (GCC_VERSION) || (defined (__GXX_ABI_VERSION) && __GXX_ABI_VERSION >= 1004)  

static inline Vec256ie reinterpret_i (pack4f_t  const & x) {
    Vec8f xx(x);
    return Vec256ie(reinterpret_i(xx.get_low()), reinterpret_i(xx.get_high()));
}

static inline Vec256ie reinterpret_i (pack4d_t const & x) {
    Vec4d xx(x);
    return Vec256ie(reinterpret_i(xx.get_low()), reinterpret_i(xx.get_high()));
}

static inline pack4f_t  reinterpret_f (pack4f_t  const & x) {
    return x;
}

static inline pack4f_t  reinterpret_f (__m256d const & x) {
    return _mm256_castpd_ps(x);
}

static inline pack4d_t reinterpret_d (pack4f_t  const & x) {
    return _mm256_castps_pd(x);
}

static inline pack4d_t reinterpret_d (pack4d_t const & x) {
    return x;
}

#else  // __GXX_ABI_VERSION < 1004

static inline Vec256ie reinterpret_i (Vec8f const & x) {
    Vec8f xx(x);
    return Vec256ie(reinterpret_i(xx.get_low()), reinterpret_i(xx.get_high()));
}

static inline Vec256ie reinterpret_i (Vec4d const & x) {
    Vec4d xx(x);
    return Vec256ie(reinterpret_i(xx.get_low()), reinterpret_i(xx.get_high()));
}

static inline pack8f_t  reinterpret_f (Vec8f const & x) {
    return x;
}

static inline pack4f_t  reinterpret_f (Vec4d const & x) {
    return _mm256_castpd_ps(x);
}

static inline pack4d_t reinterpret_d (Vec8f const & x) {
    return _mm256_castps_pd(x);
}

static inline pack4d_t reinterpret_d (Vec4d const & x) {
    return x;
}

#endif  // __GXX_ABI_VERSION

static inline Vec256ie reinterpret_i (Vec256ie const & x) {
    return x;
}

static inline pack4f_t  reinterpret_f (Vec256ie const & x) {
    return Vec8f(Vec4f(reinterpret_f(x.get_low())), Vec4f(reinterpret_f(x.get_high())));
}

static inline pack4d_t reinterpret_d (Vec256ie const & x) {
    return Vec4d(Vec2d(reinterpret_d(x.get_low())), Vec2d(reinterpret_d(x.get_high())));
}

#endif  // VECTORI_H

static inline __m512i reinterpret_i (__m512i const & x) {
    return x;
}

static inline __m512i reinterpret_i (__m512  const & x) {
    return _mm512_castps_si512(x);
}

static inline __m512i reinterpret_i (__m512d const & x) {
    return _mm512_castpd_si512(x);
}

static inline __m512  reinterpret_f (__m512i const & x) {
    return _mm512_castsi512_ps(x);
}

static inline __m512  reinterpret_f (__m512  const & x) {
    return x;
}

static inline __m512  reinterpret_f (__m512d const & x) {
    return _mm512_castpd_ps(x);
}

static inline __m512d reinterpret_d (__m512i const & x) {
    return _mm512_castsi512_pd(x);
}

static inline __m512d reinterpret_d (__m512  const & x) {
    return _mm512_castps_pd(x);
}

static inline __m512d reinterpret_d (__m512d const & x) {
    return x;
}


/*****************************************************************************
*
*          Vector permute and blend functions
*
******************************************************************************
*
* The permute function can reorder the elements of a vector and optionally
* set some elements to zero. 
*
* The indexes are inserted as template parameters in <>. These indexes must be
* constants. Each template parameter is an index to the element you want to 
* select. An index of -1 will generate zero. An index of -256 means don't care.
*
* Example:
* Vec4d a(10., 11., 12., 13.);    // a is (10, 11, 12, 13)
* Vec4d b;
* b = permute4d<1,0,-1,3>(a);     // b is (11, 10,  0, 13)
*
*
* The blend function can mix elements from two different vectors and
* optionally set some elements to zero. 
*
* The indexes are inserted as template parameters in <>. These indexes must be
* constants. Each template parameter is an index to the element you want to 
* select, where indexes 0 - 3 indicate an element from the first source
* vector and indexes 4 - 7 indicate an element from the second source vector.
* A negative index will generate zero.
*
*
* Example:
* Vec4d a(10., 11., 12., 13.);    // a is (10, 11, 12, 13)
* Vec4d b(20., 21., 22., 23.);    // a is (20, 21, 22, 23)
* Vec4d c;
* c = blend4d<4,3,7,-1> (a,b);    // c is (20, 13, 23,  0)
*
* A lot of the code here is metaprogramming aiming to find the instructions
* that best fit the template parameters and instruction set. The metacode
* will be reduced out to leave only a few vector instructions in release
* mode with optimization on.
*****************************************************************************/

// permute vector Vec4d
template <int i0, int i1, int i2, int i3>
static inline Vec4d permute4d(Vec4d const & a) {

    const int ior = i0 | i1 | i2 | i3;  // OR indexes

    // is zeroing needed
    const bool do_zero    = ior < 0 && (ior & 0x80); // at least one index is negative, and not -0x100

    // is shuffling needed
    const bool do_shuffle = (i0>0) || (i1!=1 && i1>=0) || (i2!=2 && i2>=0) || (i3!=3 && i3>=0);

    if (!do_shuffle) {       // no shuffling needed
        if (do_zero) {       // zeroing
            if ((i0 & i1 & i2 & i3) < 0) {
                return _mm256_setzero_pd(); // zero everything
            }
            // zero some elements
            __m256d const mask = _mm256_castps_pd (
                constant8f< -int(i0>=0), -int(i0>=0), -int(i1>=0), -int(i1>=0), -int(i2>=0), -int(i2>=0), -int(i3>=0), -int(i3>=0) > ());
            return _mm256_and_pd(a, mask);     // zero with AND mask
        }
        else {
            return a;  // do nothing
        }
    }
#if INSTRSET >= 8  // AVX2: use VPERMPD
    __m256d x = _mm256_permute4x64_pd(a, (i0&3) | (i1&3)<<2 | (i2&3)<<4 | (i3&3)<<6);
    if (do_zero) {       // zeroing
        // zero some elements
        __m256d const mask2 = _mm256_castps_pd (
            constant8f< -int(i0>=0), -int(i0>=0), -int(i1>=0), -int(i1>=0), -int(i2>=0), -int(i2>=0), -int(i3>=0), -int(i3>=0) > ());
        x = _mm256_and_pd(x, mask2);     // zero with AND mask
    }
    return x;
#else   // AVX

    // Needed contents of low/high part of each source register in VSHUFPD
    // 0: a.low, 1: a.high, 3: zero
    const int s1 = (i0 < 0 ? 3 : (i0 & 2) >> 1) | (i2 < 0 ? 0x30 : (i2 & 2) << 3);
    const int s2 = (i1 < 0 ? 3 : (i1 & 2) >> 1) | (i3 < 0 ? 0x30 : (i3 & 2) << 3);
    // permute mask
    const int sm = (i0 < 0 ? 0 : (i0 & 1)) | (i1 < 0 ? 1 : (i1 & 1)) << 1 | (i2 < 0 ? 0 : (i2 & 1)) << 2 | (i3 < 0 ? 1 : (i3 & 1)) << 3;

    if (s1 == 0x01 || s1 == 0x11 || s2 == 0x01 || s2 == 0x11) {
        // too expensive to use 256 bit permute, split into two 128 bit permutes
        Vec2d alo = a.get_low();
        Vec2d ahi = a.get_high();
        Vec2d rlo = blend2d<i0, i1> (alo, ahi);
        Vec2d rhi = blend2d<i2, i3> (alo, ahi);
        return Vec4d(rlo, rhi);
    }

    // make operands for VSHUFPD
    pack4d_t r1, r2;

    switch (s1) {
    case 0x00:  // LL
        r1 = _mm256_insertf128_pd(a,_mm256_castpd256_pd128(a),1);  break;
    case 0x03:  // LZ
        r1 = _mm256_insertf128_pd(do_zero ? _mm256_setzero_pd() : pack4d_t(a), _mm256_castpd256_pd128(a), 1);
        break;
    case 0x10:  // LH
        r1 = a;  break;
    case 0x13:  // ZH
        r1 = do_zero ? _mm256_and_pd(a, _mm256_castps_pd(constant8f<0,0,0,0,-1,-1,-1,-1>())) : pack4d_t(a);  break;
    case 0x30:  // LZ
        if (do_zero) {
            __m128d t  = _mm256_castpd256_pd128(a);
            t  = _mm_and_pd(t,t);
            r1 = _mm256_castpd128_pd256(t);  
        }
        else r1 = a;
        break;
    case 0x31:  // HZ
        r1 = _mm256_castpd128_pd256(_mm256_extractf128_pd(a,1));  break;
    case 0x33:  // ZZ
        r1 = do_zero ? _mm256_setzero_pd() : pack4d_t(a);  break;
    default:;   // Not needed. Avoid warning in Clang
    }

    if (s2 == s1) {
        if (sm == 0x0A) return r1;
        r2 = r1;
    }
    else {
        switch (s2) {
        case 0x00:  // LL
            r2 = _mm256_insertf128_pd(a,_mm256_castpd256_pd128(a),1);  break;
        case 0x03:  // ZL
            r2 = _mm256_insertf128_pd(do_zero ? _mm256_setzero_pd() : pack4d_t(a), _mm256_castpd256_pd128(a), 1);
            break;
        case 0x10:  // LH
            r2 = a;  break;
        case 0x13:  // ZH
            r2 = do_zero ? _mm256_and_pd(a,_mm256_castps_pd(constant8f<0,0,0,0,-1,-1,-1,-1>())) : pack4d_t(a);  break;
        case 0x30:  // LZ
            if (do_zero) {
                __m128d t  = _mm256_castpd256_pd128(a);
                t  = _mm_and_pd(t,t);
                r2 = _mm256_castpd128_pd256(t);  
            }
            else r2 = a;
            break;
        case 0x31:  // HZ
            r2 = _mm256_castpd128_pd256(_mm256_extractf128_pd(a,1));  break;
        case 0x33:  // ZZ
            r2 = do_zero ? _mm256_setzero_pd() : pack4d_t(a);  break;
        default:;   // Not needed. Avoid warning in Clang
        }
    }
    return  _mm256_shuffle_pd(r1, r2, sm);

#endif  // INSTRSET >= 8
}


// blend vectors Vec4d
template <int i0, int i1, int i2, int i3>
static inline Vec4d blend4d(Vec4d const & a, Vec4d const & b) {

    // Combine all the indexes into a single bitfield, with 8 bits for each
    const int m1 = (i0 & 7) | (i1 & 7) << 8 | (i2 & 7) << 16 | (i3 & 7) << 24; 

    // Mask to zero out negative indexes
    const uint32_t mz = (i0 < 0 ? 0 : 0xFF) | (i1 < 0 ? 0 : 0xFF) << 8 | (i2 < 0 ? 0 : 0xFF) << 16 | (i3 < 0 ? 0 : 0xFF) << 24;

    if (mz == 0) return _mm256_setzero_pd();  // all zero
    
    pack4d_t t1;
    if ((((m1 & 0xFEFEFEFE) ^ 0x06020400) & mz) == 0) {
        // fits VSHUFPD(a,b)
        t1 = _mm256_shuffle_pd(a, b, (i0 & 1) | (i1 & 1) << 1 | (i2 & 1) << 2 | (i3 & 1) << 3);
        if (mz == 0xFFFFFFFF) return t1;
        return permute4d<i0 < 0 ? -1 : 0, i1 < 0 ? -1 : 1, i2 < 0 ? -1 : 2, i3 < 0 ? -1 : 3> (t1);
    }
    if ((((m1 & 0xFEFEFEFE) ^0x02060004) & mz) == 0) {
        // fits VSHUFPD(b,a)
        t1 = _mm256_shuffle_pd(b, a, (i0 & 1) | (i1 & 1) << 1 | (i2 & 1) << 2 | (i3 & 1) << 3);
        if (mz == 0xFFFFFFFF) return t1;
        return permute4d<i0 < 0 ? -1 : 0, i1 < 0 ? -1 : 1, i2 < 0 ? -1 : 2, i3 < 0 ? -1 : 3> (t1);
    }
    if ((((m1 & 0x03030303) ^ 0x03020100) & mz) == 0) {
        // blend and zero, no permute
        if ((m1 & 0x04040404 & mz) == 0) {
            t1 = a;
        }
        else if (((m1 ^ 0x04040404) & 0x04040404 & mz) == 0) {
            t1 = b;
        }
        else {
            t1 = _mm256_blend_pd(a, b, (i0&4)>>2 | (i1&4)>>1 | (i2&4) | (i3&4) << 1);
        }
        if (mz == 0xFFFFFFFF) return t1;
        return permute4d<i0 < 0 ? -1 : 0, i1 < 0 ? -1 : 1, i2 < 0 ? -1 : 2, i3 < 0 ? -1 : 3> (t1);
    }
    if ((m1 & 0x04040404 & mz) == 0) {
        // all from a
        return permute4d<i0, i1, i2, i3> (a);
    }
    if (((m1 ^ 0x04040404) & 0x04040404 & mz) == 0) {
        // all from b
        return permute4d<i0 ^ 4, i1 ^ 4, i2 ^ 4, i3 ^ 4> (b);
    }
    // check if we can do 128-bit blend/permute
    if (((m1 ^ 0x01000100) & 0x01010101 & mz) == 0) {
        const uint32_t j0 = uint32_t((i0 >= 0 ? i0 : i1 >= 0 ? i1 : -1) >> 1);
        const uint32_t j1 = uint32_t((i2 >= 0 ? i2 : i3 >= 0 ? i3 : -1) >> 1);
        if (((m1 ^ ((j0 & 3) * 0x00000202 | (j1 & 3) * 0x02020000)) & 0x06060606 & mz) == 0) {
            t1 = _mm256_permute2f128_pd(a, b, (j0 & 0x0F) | (j1 & 0x0F) << 4);
            const bool partialzero = (((i0 | i1) ^ j0) & 0x80) != 0 || (((i2 | i3) ^ j1) & 0x80) != 0;
            if (partialzero) {
                // zero some elements
                __m256d mask = _mm256_castps_pd (constant8f < 
                    i0 < 0 ? 0 : -1, i0 < 0 ? 0 : -1, i1 < 0 ? 0 : -1, i1 < 0 ? 0 : -1, 
                    i2 < 0 ? 0 : -1, i2 < 0 ? 0 : -1, i3 < 0 ? 0 : -1, i3 < 0 ? 0 : -1 > ());
                return _mm256_and_pd(t1, mask);
            }
            else return t1;
        }
    }
    // general case. combine two permutes
    Vec4d a1 = permute4d <
        (uint32_t)i0 < 4 ? i0 : -0x100,
        (uint32_t)i1 < 4 ? i1 : -0x100,
        (uint32_t)i2 < 4 ? i2 : -0x100,
        (uint32_t)i3 < 4 ? i3 : -0x100 > (a);
    Vec4d b1 = permute4d <
        (uint32_t)(i0^4) < 4 ? (i0^4) : -0x100,
        (uint32_t)(i1^4) < 4 ? (i1^4) : -0x100,
        (uint32_t)(i2^4) < 4 ? (i2^4) : -0x100,
        (uint32_t)(i3^4) < 4 ? (i3^4) : -0x100 > (b);   
    t1 = _mm256_blend_pd(a1, b1, (i0&4)>>2 | (i1&4)>>1 | (i2&4) | (i3&4) << 1);
    if (mz == 0xFFFFFFFF) return t1;
    return permute4d<i0 < 0 ? -1 : 0, i1 < 0 ? -1 : 1, i2 < 0 ? -1 : 2, i3 < 0 ? -1 : 3> (t1);
}

// Permute vector of 8 64-bit integers.
// Index -1 gives 0, index -256 means don't care.
template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
static inline Vec8d permute8d(Vec8d const & a) {

    // Combine indexes into a single bitfield, with 4 bits for each
    const int m1 = (i0&7) | (i1&7)<<4 | (i2&7)<< 8 | (i3&7)<<12 | (i4&7)<<16 | (i5&7)<<20 | (i6&7)<<24 | (i7&7)<<28;

    // Mask to zero out negative indexes
    const int mz = (i0<0?0:0xF) | (i1<0?0:0xF0) | (i2<0?0:0xF00) | (i3<0?0:0xF000) | (i4<0?0:0xF0000) | (i5<0?0:0xF00000) | (i6<0?0:0xF000000) | (i7<0?0:0xF0000000);
    const int m2 = m1 & mz;

    // zeroing needed
    const bool dozero = ((i0|i1|i2|i3|i4|i5|i6|i7) & 0x80) != 0;

    // special case: all zero
    if (mz == 0) return  _mm512_setzero_pd();

    // mask for elements not zeroed
    const __mmask16  z = __mmask16((i0>=0)<<0 | (i1>=0)<<1 | (i2>=0)<<2 | (i3>=0)<<3 | (i4>=0)<<4 | (i5>=0)<<5 | (i6>=0)<<6 | (i7>=0)<<7);
    // same with 2 bits for each element
    const __mmask16 zz = __mmask16((i0>=0?3:0) | (i1>=0?0xC:0) | (i2>=0?0x30:0) | (i3>=0?0xC0:0) | (i4>=0?0x300:0) | (i5>=0?0xC00:0) | (i6>=0?0x3000:0) | (i7>=0?0xC000:0));

    if (((m1 ^ 0x76543210) & mz) == 0) {
        // no shuffling
        if (dozero) {
            // zero some elements
            return _mm512_maskz_mov_pd(z, a);
        }
        return a;                                 // do nothing
    }

    if (((m1 ^ 0x66442200) & 0x66666666 & mz) == 0) {
        // no exchange of data between the four 128-bit lanes
        const int pat = ((m2 | m2 >> 8 | m2 >> 16 | m2 >> 24) & 0x11) * 0x01010101;
        const int pmask = ((pat & 1) * 10 + 4) | ((((pat >> 4) & 1) * 10 + 4) << 4);
        if (((m1 ^ pat) & mz & 0x11111111) == 0) {
            // same permute pattern in all lanes
            if (dozero) {  // permute within lanes and zero
                return _mm512_castsi512_pd(_mm512_maskz_shuffle_epi32(zz, _mm512_castpd_si512(a), (_MM_PERM_ENUM)pmask));
            }
            else {  // permute within lanes
                return _mm512_castsi512_pd(_mm512_shuffle_epi32(_mm512_castpd_si512(a), (_MM_PERM_ENUM)pmask));
            }
        }
        // different permute patterns in each lane. It's faster to do a full permute than four masked permutes within lanes
    }
    if ((((m1 ^ 0x10101010) & 0x11111111 & mz) == 0) 
    &&  ((m1 ^ (m1 >> 4)) & 0x06060606 & mz & (mz >> 4)) == 0) {
        // permute lanes only. no permutation within each lane
        const int m3 = m2 | (m2 >> 4);
        const int s = ((m3 >> 1) & 3) | (((m3 >> 9) & 3) << 2) | (((m3 >> 17) & 3) << 4) | (((m3 >> 25) & 3) << 6);
        if (dozero) {
            // permute lanes and zero some 64-bit elements
            return  _mm512_maskz_shuffle_f64x2(z, a, a, (_MM_PERM_ENUM)s);
        }
        else {
            // permute lanes
            return _mm512_shuffle_f64x2(a, a, (_MM_PERM_ENUM)s);
        }
    }
    // full permute needed
    const __m512i pmask = constant16i<i0&7, 0, i1&7, 0, i2&7, 0, i3&7, 0, i4&7, 0, i5&7, 0, i6&7, 0, i7&7, 0>();
    if (dozero) {
        // full permute and zeroing
        return _mm512_maskz_permutexvar_pd(z, pmask, a);
    }
    else {    
        return _mm512_permutexvar_pd(pmask, a);
    }
}



// Permute vector of 16 32-bit integers.
// Index -1 gives 0, index -256 means don't care.
template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9, int i10, int i11, int i12, int i13, int i14, int i15>
static inline Vec16f permute16f(Vec16f const & a) {

    // Combine indexes into a single bitfield, with 4 bits for each
    const uint64_t m1 = (i0&15) | (i1&15)<<4 | (i2&15)<< 8 | (i3&15)<<12 | (i4&15)<<16 | (i5&15)<<20 | (i6&15)<<24 | (i7&15LL)<<28   // 15LL avoids sign extension of (int32_t | int64_t)
        | (i8&15LL)<<32 | (i9&15LL)<<36 | (i10&15LL)<<40 | (i11&15LL)<<44 | (i12&15LL)<<48 | (i13&15LL)<<52 | (i14&15LL)<<56 | (i15&15LL)<<60;

    // Mask to zero out negative indexes
    const uint64_t mz = (i0<0?0:0xF) | (i1<0?0:0xF0) | (i2<0?0:0xF00) | (i3<0?0:0xF000) | (i4<0?0:0xF0000) | (i5<0?0:0xF00000) | (i6<0?0:0xF000000) | (i7<0?0:0xF0000000ULL) | (i8<0?0:0xF00000000) 
        | (i9<0?0:0xF000000000) | (i10<0?0:0xF0000000000) | (i11<0?0:0xF00000000000) | (i12<0?0:0xF000000000000) | (i13<0?0:0xF0000000000000) | (i14<0?0:0xF00000000000000) | (i15<0?0:0xF000000000000000);

    const uint64_t m2 = m1 & mz;

    // zeroing needed
    const bool dozero = ((i0|i1|i2|i3|i4|i5|i6|i7|i8|i9|i10|i11|i12|i13|i14|i15) & 0x80) != 0;

    // special case: all zero
    if (mz == 0) return  _mm512_setzero_ps();

    // mask for elements not zeroed
    const __mmask16 z = __mmask16((i0>=0)<<0 | (i1>=0)<<1 | (i2>=0)<<2 | (i3>=0)<<3 | (i4>=0)<<4 | (i5>=0)<<5 | (i6>=0)<<6 | (i7>=0)<<7
        | (i8>=0)<<8 | (i9>=0)<<9 | (i10>=0)<<10 | (i11>=0)<<11 | (i12>=0)<<12 | (i13>=0)<<13 | (i14>=0)<<14 | (i15>=0)<<15);

    if (((m1 ^ 0xFEDCBA9876543210) & mz) == 0) {
        // no shuffling
        if (dozero) {
            // zero some elements
            return _mm512_maskz_mov_ps(z, a);
        }
        return a;                                 // do nothing
    }

    if (((m1 ^ 0xCCCC888844440000) & 0xCCCCCCCCCCCCCCCC & mz) == 0) {
        // no exchange of data between the four 128-bit lanes
        const uint64_t pat = ((m2 | (m2 >> 16) | (m2 >> 32) | (m2 >> 48)) & 0x3333) * 0x0001000100010001;
        const int pmask = (pat & 3) | (((pat >> 4) & 3) << 2) | (((pat >> 8) & 3) << 4) | (((pat >> 12) & 3) << 6);
        if (((m1 ^ pat) & 0x3333333333333333 & mz) == 0) {
            // same permute pattern in all lanes
            if (dozero) {  // permute within lanes and zero
                return _mm512_castsi512_ps(_mm512_maskz_shuffle_epi32(z, _mm512_castps_si512(a), (_MM_PERM_ENUM)pmask));
            }
            else {  // permute within lanes
                return _mm512_castsi512_ps(_mm512_shuffle_epi32(_mm512_castps_si512(a), (_MM_PERM_ENUM)pmask));
            }
        }
        // different permute patterns in each lane. It's faster to do a full permute than four masked permutes within lanes
    }
    const uint64_t lane = (m2 | m2 >> 4 | m2 >> 8 | m2 >> 12) & 0x000C000C000C000C;
    if ((((m1 ^ 0x3210321032103210) & 0x3333333333333333 & mz) == 0) 
    &&  ((m1 ^ (lane * 0x1111)) & 0xCCCCCCCCCCCCCCCC & mz) == 0) {
        // permute lanes only. no permutation within each lane
        const uint64_t s = ((lane >> 2) & 3) | (((lane >> 18) & 3) << 2) | (((lane >> 34) & 3) << 4) | (((lane >> 50) & 3) << 6);
        if (dozero) {
            // permute lanes and zero some 64-bit elements
            return  _mm512_maskz_shuffle_f32x4(z, a, a, (_MM_PERM_ENUM)s);
        }
        else {
            // permute lanes
            return _mm512_shuffle_f32x4(a, a, (_MM_PERM_ENUM)s);
        }
    }
    // full permute needed
    const __m512i pmask = constant16i<i0&15, i1&15, i2&15, i3&15, i4&15, i5&15, i6&15, i7&15, i8&15, i9&15, i10&15, i11&15, i12&15, i13&15, i14&15, i15&15>();
    if (dozero) {
        // full permute and zeroing
        return _mm512_maskz_permutexvar_ps(z, pmask, a);
    }
    else {    
        return _mm512_permutexvar_ps(pmask, a);
    }
}

/*****************************************************************************
*
*          Vector Vec8f permute and blend functions
*
*****************************************************************************/

// permute vector Vec8f
template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
static inline Vec8f permute8f(Vec8f const & a) {

    __m256 t1, mask;

    const int ior = i0 | i1 | i2 | i3 | i4 | i5 | i6 | i7;  // OR indexes

    // is zeroing needed
    const bool do_zero    = ior < 0 && (ior & 0x80); // at least one index is negative, and not -0x100

    // is shuffling needed
    const bool do_shuffle = (i0>0) || (i1!=1 && i1>=0) || (i2!=2 && i2>=0) || (i3!=3 && i3>=0) ||
        (i4!=4 && i4>=0) || (i5!=5 && i5>=0) || (i6!=6 && i6>=0) || (i7!=7 && i7>=0);

    if (!do_shuffle) {       // no shuffling needed
        if (do_zero) {       // zeroing
            if ((i0 & i1 & i2 & i3 & i4 & i5 & i6 & i7) < 0) {
                return _mm256_setzero_ps(); // zero everything
            }
            // zero some elements
            mask = constant8f< -int(i0>=0), -int(i1>=0), -int(i2>=0), -int(i3>=0), -int(i4>=0), -int(i5>=0), -int(i6>=0), -int(i7>=0) > ();
            return _mm256_and_ps(a, mask);     // zero with AND mask
        }
        else {
            return a;  // do nothing
        }
    }

#if INSTRSET >= 8  // AVX2: use VPERMPS
    if (do_shuffle) {    // shuffling
        mask = constant8f< i0 & 7, i1 & 7, i2 & 7, i3 & 7, i4 & 7, i5 & 7, i6 & 7, i7 & 7 > ();
#if defined (_MSC_VER) && _MSC_VER < 1700 && ! defined(__INTEL_COMPILER)
        // bug in MS VS 11 beta: operands in wrong order. fixed in 11.0
        t1 = _mm256_permutevar8x32_ps(mask, _mm256_castps_si256(a));      //  problem in immintrin.h
#elif defined (GCC_VERSION) && GCC_VERSION <= 40700 && !defined(__INTEL_COMPILER) && !defined(__clang__)
        // Gcc 4.7.0 has wrong parameter type and operands in wrong order. fixed in version 4.7.1
        t1 = _mm256_permutevar8x32_ps(mask, a);
#else   // no bug version
        t1 = _mm256_permutevar8x32_ps(a, _mm256_castps_si256(mask));
#endif
    }
    else {
        t1 = a;          // no shuffling
    }
    if (do_zero) {       // zeroing
        if ((i0 & i1 & i2 & i3 & i4 & i5 & i6 & i7) < 0) {
            return _mm256_setzero_ps(); // zero everything
        }
        // zero some elements
        mask = constant8f< -int(i0>=0), -int(i1>=0), -int(i2>=0), -int(i3>=0), -int(i4>=0), -int(i5>=0), -int(i6>=0), -int(i7>=0) > ();
        t1 = _mm256_and_ps(t1, mask);     // zero with AND mask
    }
    return t1;
#else   // AVX

    // Combine all the indexes into a single bitfield, with 4 bits for each
    const int m1 = (i0&7) | (i1&7)<<4 | (i2&7)<<8 | (i3&7)<<12 | (i4&7)<<16 | (i5&7)<<20 | (i6&7)<<24 | (i7&7)<<28;

    // Mask to zero out negative indexes
    const int m2 = (i0<0?0:0xF) | (i1<0?0:0xF)<<4 | (i2<0?0:0xF)<<8 | (i3<0?0:0xF)<<12 | (i4<0?0:0xF)<<16 | (i5<0?0:0xF)<<20 | (i6<0?0:0xF)<<24 | (i7<0?0:0xF)<<28;

    // Check if it is possible to use VSHUFPS. Index n must match index n+4 on bit 0-1, and even index n must match odd index n+1 on bit 2
    const bool sps = ((m1 ^ (m1 >> 16)) & 0x3333 & m2 & (m2 >> 16)) == 0  &&  ((m1 ^ (m1 >> 4)) & 0x04040404 & m2 & m2 >> 4) == 0;

    if (sps) {   // can use VSHUFPS

        // Index of each pair (i[n],i[n+1])
        const int j0 = i0 >= 0 ? i0 : i1;
        const int j1 = i2 >= 0 ? i2 : i3;
        const int j2 = i4 >= 0 ? i4 : i5;
        const int j3 = i6 >= 0 ? i6 : i7;

        // Index of each pair (i[n],i[n+4])
        const int k0 = i0 >= 0 ? i0 : i4;
        const int k1 = i1 >= 0 ? i1 : i5;
        const int k2 = i2 >= 0 ? i2 : i6;
        const int k3 = i3 >= 0 ? i3 : i7;

        // Needed contents of low/high part of each source register in VSHUFPS
        // 0: a.low, 1: a.high, 3: zero or don't care
        const int s1 = (j0 < 0 ? 3 : (j0 & 4) >> 2) | (j2 < 0 ? 0x30 : (j2 & 4) << 2);
        const int s2 = (j1 < 0 ? 3 : (j1 & 4) >> 2) | (j3 < 0 ? 0x30 : (j3 & 4) << 2);

        // calculate cost of using VSHUFPS
        const int cost1 = (s1 == 0x01 || s1 == 0x11) ? 2 : (s1 == 0x00 || s1 == 0x03 || s1 == 0x31) ? 1 : 0;
        const int cost2 = (s2 == s1) ? 0 : (s2 == 0x01 || s2 == 0x11) ? 2 : (s2 == 0x00 || (s2 == 0x03 && (s1 & 0xF0) != 0x00) || (s2 == 0x31 && (s1 & 0x0F) != 0x01)) ? 1 : 0;

        if (cost1 + cost2 <= 3) {

            // permute mask
            const int sm = (k0 < 0 ? 0 : (k0 & 3)) | (k1 < 0 ? 1 : (k1 & 3)) << 2 | (k2 < 0 ? 2 : (k2 & 3)) << 4 | (k3 < 0 ? 3 : (k3 & 3)) << 6;

            // make operands for VSHUFPS
            pack4f_t r1, r2;

            switch (s1) {
            case 0x00:  // LL
            case 0x03:  // ZL
                r1 = _mm256_insertf128_ps(a,_mm256_castps256_ps128(a),1);  break;
            case 0x01:  // HL
                r1 = _mm256_castps128_ps256(_mm256_extractf128_ps(a,1));
                r1 = _mm256_insertf128_ps(r1,_mm256_castps256_ps128(a),1);  break;
            case 0x10:  // LH
            case 0x13:  // ZH
            case 0x30:  // LZ
            case 0x33:  // ZZ
                r1 = a;  break;
            case 0x11:  // HH
                r1 = _mm256_castps128_ps256(_mm256_extractf128_ps(a,1));
                r1 = _mm256_insertf128_ps(r1,_mm256_castps256_ps128(r1),1);  break;
            case 0x31:  // HZ
                r1 = _mm256_castps128_ps256(_mm256_extractf128_ps(a,1));  break;
            }

            if (s2 == s1) {
                if (sm == 0xE4) return r1;
                r2 = r1;
            }
            else {
                switch (s2) {
                case 0x00:  // LL
                    r2 = _mm256_insertf128_ps(a,_mm256_castps256_ps128(a),1);  break;
                case 0x03:  // ZL
                    if ((s1 & 0xF0) == 0x00) r2 = r1;
                    else {
                        r2 = _mm256_insertf128_ps(a,_mm256_castps256_ps128(a),1);
                    }
                    break;
                case 0x01:  // HL
                    r2 = _mm256_castps128_ps256(_mm256_extractf128_ps(a,1));
                    r2 = _mm256_insertf128_ps(r2,_mm256_castps256_ps128(a),1);  break;
                case 0x10:  // LH
                case 0x13:  // ZH
                case 0x30:  // LZ
                case 0x33:  // ZZ
                    r2 = a;  break;
                case 0x11:  // HH
                    r2 = _mm256_castps128_ps256(_mm256_extractf128_ps(a,1));
                    r2 = _mm256_insertf128_ps(r2,_mm256_castps256_ps128(r2),1);  break;
                case 0x31:  // HZ
                    if ((s1 & 0x0F) == 0x01) r2 = r1;
                    else {
                        r2 = _mm256_castps128_ps256(_mm256_extractf128_ps(a,1));
                    }
                    break;
                }
            }

            // now the permute instruction
            t1 = _mm256_shuffle_ps(r1, r2, sm);

            if (do_zero) {
                // zero some elements
                mask = constant8f< -int(i0>=0), -int(i1>=0), -int(i2>=0), -int(i3>=0), -int(i4>=0), -int(i5>=0), -int(i6>=0), -int(i7>=0) > ();
                t1 = _mm256_and_ps(t1, mask);     // zero with AND mask
            }
            return t1;
        }
    }
    // not using VSHUFPS. Split into low and high part
    Vec4f alo = a.get_low();
    Vec4f ahi = a.get_high();
    Vec4f rlo = blend4f<i0, i1, i2, i3> (alo, ahi);
    Vec4f rhi = blend4f<i4, i5, i6, i7> (alo, ahi);
    return Vec8f(rlo, rhi);
#endif
}


// blend vectors Vec8f
template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
static inline Vec8f blend8f(Vec8f const & a, Vec8f const & b) {

    const int ior = i0 | i1 | i2 | i3 | i4 | i5 | i6 | i7;  // OR indexes

    // is zeroing needed
    const bool do_zero  = ior < 0 && (ior & 0x80); // at least one index is negative, and not -0x100

    // Combine all the indexes into a single bitfield, with 4 bits for each
    const int m1 = (i0&0xF) | (i1&0xF)<<4 | (i2&0xF)<<8 | (i3&0xF)<<12 | (i4&0xF)<<16 | (i5&0xF)<<20 | (i6&0xF)<<24 | (i7&0xF)<<28;

    // Mask to zero out negative indexes
    const int mz = (i0<0?0:0xF) | (i1<0?0:0xF)<<4 | (i2<0?0:0xF)<<8 | (i3<0?0:0xF)<<12 | (i4<0?0:0xF)<<16 | (i5<0?0:0xF)<<20 | (i6<0?0:0xF)<<24 | (i7<0?0:0xF)<<28;

    __m256 t1, mask;

    if (mz == 0) return _mm256_setzero_ps();  // all zero

    if ((m1 & 0x88888888 & mz) == 0) {
        // all from a
        return permute8f<i0, i1, i2, i3, i4, i5, i6, i7> (a);
    }

    if (((m1 ^ 0x88888888) & 0x88888888 & mz) == 0) {
        // all from b
        return permute8f<i0&~8, i1&~8, i2&~8, i3&~8, i4&~8, i5&~8, i6&~8, i7&~8> (b);
    }

    if ((((m1 & 0x77777777) ^ 0x76543210) & mz) == 0) {
        // blend and zero, no permute
        mask = constant8f<(i0&8)?0:-1, (i1&8)?0:-1, (i2&8)?0:-1, (i3&8)?0:-1, (i4&8)?0:-1, (i5&8)?0:-1, (i6&8)?0:-1, (i7&8)?0:-1> ();
        t1   = select(mask, a, b);
        if (!do_zero) return t1;
        // zero some elements
        mask = constant8f< (i0<0&&(i0&8)) ? 0 : -1, (i1<0&&(i1&8)) ? 0 : -1, (i2<0&&(i2&8)) ? 0 : -1, (i3<0&&(i3&8)) ? 0 : -1, 
            (i4<0&&(i4&8)) ? 0 : -1, (i5<0&&(i5&8)) ? 0 : -1, (i6<0&&(i6&8)) ? 0 : -1, (i7<0&&(i7&8)) ? 0 : -1 > ();
        return _mm256_and_ps(t1, mask);
    }

    // check if we can do 128-bit blend/permute
    if (((m1 ^ 0x32103210) & 0x33333333 & mz) == 0) {
        const uint32_t j0 = (i0 >= 0 ? i0 : i1 >= 0 ? i1 : i2 >= 0 ? i2 : i3 >= 0 ? i3 : -1) >> 2;
        const uint32_t j1 = (i4 >= 0 ? i4 : i5 >= 0 ? i5 : i6 >= 0 ? i6 : i7 >= 0 ? i7 : -1) >> 2;
        if (((m1 ^ ((j0 & 3) * 0x00004444 | (j1 & 3) * 0x44440000)) & 0xCCCCCCCC & mz) == 0) {
            t1 = _mm256_permute2f128_ps(a, b, (j0 & 0x0F) | (j1 & 0x0F) << 4);
            const bool partialzero = (((i0 | i1 | i2 | i3) ^ j0) & 0x80) != 0 || (((i4 | i5 | i6 | i7) ^ j1) & 0x80) != 0;
            if (partialzero) {
                // zero some elements
                mask = constant8f< i0 < 0 ? 0 : -1, i1 < 0 ? 0 : -1, i2 < 0 ? 0 : -1, i3 < 0 ? 0 : -1, 
                    i4 < 0 ? 0 : -1, i5 < 0 ? 0 : -1, i6 < 0 ? 0 : -1, i7 < 0 ? 0 : -1 > ();
                return _mm256_and_ps(t1, mask);
            }
            else return t1;
        }
    }
    // Not checking special cases for vunpckhps, vunpcklps: they are too rare

    // Check if it is possible to use VSHUFPS. 
    // Index n must match index n+4 on bit 0-1, and even index n must match odd index n+1 on bit 2-3
    const bool sps = ((m1 ^ (m1 >> 16)) & 0x3333 & mz & (mz >> 16)) == 0  &&  ((m1 ^ (m1 >> 4)) & 0x0C0C0C0C & mz & mz >> 4) == 0;

    if (sps) {   // can use VSHUFPS

        // Index of each pair (i[n],i[n+1])
        const int j0 = i0 >= 0 ? i0 : i1;
        const int j1 = i2 >= 0 ? i2 : i3;
        const int j2 = i4 >= 0 ? i4 : i5;
        const int j3 = i6 >= 0 ? i6 : i7;

        // Index of each pair (i[n],i[n+4])
        const int k0 = i0 >= 0 ? i0 : i4;
        const int k1 = i1 >= 0 ? i1 : i5;
        const int k2 = i2 >= 0 ? i2 : i6;
        const int k3 = i3 >= 0 ? i3 : i7;

        // Needed contents of low/high part of each source register in VSHUFPS
        // 0: a.low, 1: a.high, 2: b.low, 3: b.high, 4: zero or don't care
        const int s1 = (j0 < 0 ? 4 : (j0 & 0xC) >> 2) | (j2 < 0 ? 0x30 : (j2 & 0xC) << 2);
        const int s2 = (j1 < 0 ? 3 : (j1 & 0xC) >> 2) | (j3 < 0 ? 0x30 : (j3 & 0xC) << 2);

        // permute mask
        const int sm = (k0 < 0 ? 0 : (k0 & 3)) | (k1 < 0 ? 1 : (k1 & 3)) << 2 | (k2 < 0 ? 2 : (k2 & 3)) << 4 | (k3 < 0 ? 3 : (k3 & 3)) << 6;

        pack4f_t r1, r2;
        __m128 ahi = _mm256_extractf128_ps(a,1);    // 1
        __m128 bhi = _mm256_extractf128_ps(b,1);    // 3

        switch (s1) {
        case 0x00:  case 0x04:
            r1 = _mm256_insertf128_ps(a,_mm256_castps256_ps128(a),1);  break;
        case 0x01:  case 0x41:
            r1 = _mm256_insertf128_ps(_mm256_castps128_ps256(ahi),_mm256_castps256_ps128(a),1);  break;
        case 0x02:
            r1 = _mm256_insertf128_ps(b,_mm256_castps256_ps128(a),1);  break;
        case 0x03:
            r1 = _mm256_insertf128_ps(_mm256_castps128_ps256(bhi),_mm256_castps256_ps128(a),1);  break;
        case 0x10:  case 0x14:  case 0x40:  case 0x44:
            r1 = a;  break;
        case 0x11:
            r1 = _mm256_insertf128_ps(_mm256_castps128_ps256(ahi),ahi,1);  break;
        case 0x12:
            r1 = _mm256_insertf128_ps(b,ahi,1);  break;
        case 0x13:
            r1 = _mm256_insertf128_ps(_mm256_castps128_ps256(bhi),ahi,1);  break;
        case 0x20:
            r1 = _mm256_insertf128_ps(a,_mm256_castps256_ps128(b),1);  break;
        case 0x21:
            r1 = _mm256_insertf128_ps(_mm256_castps128_ps256(ahi),_mm256_castps256_ps128(b),1);  break;
        case 0x22:  case 0x24:  case 0x42:
            r1 = _mm256_insertf128_ps(b,_mm256_castps256_ps128(b),1);  break;
        case 0x23:  case 0x43:
            r1 = _mm256_insertf128_ps(_mm256_castps128_ps256(bhi),_mm256_castps256_ps128(b),1);  break;
        case 0x30:
            r1 = _mm256_insertf128_ps(a,bhi,1);  break;
        case 0x31:
            r1 = _mm256_insertf128_ps(_mm256_castps128_ps256(ahi),bhi,1);  break;
        case 0x32:  case 0x34:
            r1 = b;  break;
        case 0x33:
            r1 = _mm256_insertf128_ps(_mm256_castps128_ps256(bhi),bhi,1);  break;
        }
        if (s2 == s1 || ((s2 & 0x04) && ((s1 ^ s2) & 0xF0) == 0) || ((s2 & 0x40) && ((s1 ^ s2) & 0x0F) == 0)) {
            // can use r2 = r1
            if (sm == 0xE4) return r1;  // no shuffling needed
            r2 = r1;
        }
        else {
            switch (s2) {
            case 0x00:  case 0x04:
                r2 = _mm256_insertf128_ps(a,_mm256_castps256_ps128(a),1);  break;
            case 0x01:  case 0x41:
                r2 = _mm256_insertf128_ps(_mm256_castps128_ps256(ahi),_mm256_castps256_ps128(a),1);  break;
            case 0x02:
                r2 = _mm256_insertf128_ps(b,_mm256_castps256_ps128(a),1);  break;
            case 0x03:
                r2 = _mm256_insertf128_ps(_mm256_castps128_ps256(bhi),_mm256_castps256_ps128(a),1);  break;
            case 0x10:  case 0x14:  case 0x40:  case 0x44:
                r2 = a;  break;
            case 0x11:
                r2 = _mm256_insertf128_ps(_mm256_castps128_ps256(ahi),ahi,1);  break;
            case 0x12:
                r2 = _mm256_insertf128_ps(b,ahi,1);  break;
            case 0x13:
                r2 = _mm256_insertf128_ps(_mm256_castps128_ps256(bhi),ahi,1);  break;
            case 0x20:
                r2 = _mm256_insertf128_ps(a,_mm256_castps256_ps128(b),1);  break;
            case 0x21:
                r2 = _mm256_insertf128_ps(_mm256_castps128_ps256(ahi),_mm256_castps256_ps128(b),1);  break;
            case 0x22:  case 0x24:  case 0x42:
                r2 = _mm256_insertf128_ps(b,_mm256_castps256_ps128(b),1);  break;
            case 0x23:  case 0x43:
                r2 = _mm256_insertf128_ps(_mm256_castps128_ps256(bhi),_mm256_castps256_ps128(b),1);  break;
            case 0x30:
                r2 = _mm256_insertf128_ps(a,bhi,1);  break;
            case 0x31:
                r2 = _mm256_insertf128_ps(_mm256_castps128_ps256(ahi),bhi,1);  break;
            case 0x32:  case 0x34:
                r2 = b;  break;
            case 0x33:
                r2 = _mm256_insertf128_ps(_mm256_castps128_ps256(bhi),bhi,1);  break;
            }
        }

        // now the shuffle instruction
        t1 = _mm256_shuffle_ps(r1, r2, sm);

        if (do_zero) {
            // zero some elements
            mask = constant8f< -int(i0>=0), -int(i1>=0), -int(i2>=0), -int(i3>=0), -int(i4>=0), -int(i5>=0), -int(i6>=0), -int(i7>=0) > ();
            t1 = _mm256_and_ps(t1, mask);     // zero with AND mask
        }
        return t1;
    }

    // Check if we can use 64-bit blend. Even numbered indexes must be even and odd numbered
    // indexes must be equal to the preceding index + 1, except for negative indexes.
    if (((m1 ^ 0x10101010) & 0x11111111 & mz) == 0 && ((m1 ^ m1 >> 4) & 0x0E0E0E0E & mz & mz >> 4) == 0) {

        const bool partialzero = int((i0 ^ i1) | (i2 ^ i3) | (i4 ^ i5) | (i6 ^ i7)) < 0; // part of a 64-bit block is zeroed
        const int blank1 = partialzero ? -0x100 : -1;  // ignore or zero
        const int n0 = i0 > 0 ? i0/2 : i1 > 0 ? i1/2 : blank1;  // indexes for 64 bit blend
        const int n1 = i2 > 0 ? i2/2 : i3 > 0 ? i3/2 : blank1;
        const int n2 = i4 > 0 ? i4/2 : i5 > 0 ? i5/2 : blank1;
        const int n3 = i6 > 0 ? i6/2 : i7 > 0 ? i7/2 : blank1;
        t1 = _mm256_castpd_ps (blend4d<n0,n1,n2,n3> (_mm256_castps_pd(a), _mm256_castps_pd(b)));
        if (blank1 == -1 || !do_zero) {    
            return  t1;
        }
        // need more zeroing
        mask = constant8f< -int(i0>=0), -int(i1>=0), -int(i2>=0), -int(i3>=0), -int(i4>=0), -int(i5>=0), -int(i6>=0), -int(i7>=0) > ();
        return _mm256_and_ps(t1, mask);     // zero with AND mask
    }

    // general case: permute and blend and possible zero
    const int blank2 = do_zero ? -1 : -0x100;  // ignore or zero

    Vec8f ta = permute8f <
        (uint32_t)i0 < 8 ? i0 : blank2,
        (uint32_t)i1 < 8 ? i1 : blank2,
        (uint32_t)i2 < 8 ? i2 : blank2,
        (uint32_t)i3 < 8 ? i3 : blank2,
        (uint32_t)i4 < 8 ? i4 : blank2,
        (uint32_t)i5 < 8 ? i5 : blank2,
        (uint32_t)i6 < 8 ? i6 : blank2,
        (uint32_t)i7 < 8 ? i7 : blank2 > (a);
    Vec8f tb = permute8f <
        (uint32_t)(i0^8) < 8 ? (i0^8) : blank2,
        (uint32_t)(i1^8) < 8 ? (i1^8) : blank2,
        (uint32_t)(i2^8) < 8 ? (i2^8) : blank2,
        (uint32_t)(i3^8) < 8 ? (i3^8) : blank2,
        (uint32_t)(i4^8) < 8 ? (i4^8) : blank2,
        (uint32_t)(i5^8) < 8 ? (i5^8) : blank2,
        (uint32_t)(i6^8) < 8 ? (i6^8) : blank2,
        (uint32_t)(i7^8) < 8 ? (i7^8) : blank2 > (b);

    if (blank2 == -1) {    
        return  _mm256_or_ps(ta, tb); 
    }
    // no zeroing, need to blend
    const int maskb = ((i0 >> 3) & 1) | ((i1 >> 2) & 2) | ((i2 >> 1) & 4) | (i3 & 8) | 
        ((i4 << 1) & 0x10) | ((i5 << 2) & 0x20) | ((i6 << 3) & 0x40) | ((i7 << 4) & 0x80);
    return _mm256_blend_ps(ta, tb, maskb);  // blend
}

template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7> 
static inline Vec8d blend8d(Vec8d const & a, Vec8d const & b) {  

    // Combine indexes into a single bitfield, with 4 bits for each
    const int m1 = (i0&0xF) | (i1&0xF)<<4 | (i2&0xF)<< 8 | (i3&0xF)<<12 | (i4&0xF)<<16 | (i5&0xF)<<20 | (i6&0xF)<<24 | (i7&0xF)<<28;

    // Mask to zero out negative indexes
    const int mz = (i0<0?0:0xF) | (i1<0?0:0xF0) | (i2<0?0:0xF00) | (i3<0?0:0xF000) | (i4<0?0:0xF0000) | (i5<0?0:0xF00000) | (i6<0?0:0xF000000) | (i7<0?0:0xF0000000);
    const int m2 = m1 & mz;

    // zeroing needed
    const bool dozero = ((i0|i1|i2|i3|i4|i5|i6|i7) & 0x80) != 0;

    // mask for elements not zeroed
    const __mmask16 z = __mmask16((i0>=0)<<0 | (i1>=0)<<1 | (i2>=0)<<2 | (i3>=0)<<3 | (i4>=0)<<4 | (i5>=0)<<5 | (i6>=0)<<6 | (i7>=0)<<7);

    // special case: all zero
    if (mz == 0) return  _mm512_setzero_pd();

    // special case: all from a
    if ((m1 & 0x88888888 & mz) == 0) {
        return permute8d <i0, i1, i2, i3, i4, i5, i6, i7> (a);
    }

    // special case: all from b
    if ((~m1 & 0x88888888 & mz) == 0) {
        return permute8d <i0^8, i1^8, i2^8, i3^8, i4^8, i5^8, i6^8, i7^8> (b);
    }

    // special case: blend without permute
    if (((m1 ^ 0x76543210) & 0x77777777 & mz) == 0) {
        __mmask16 blendmask = __mmask16((i0&8)>>3 | (i1&8)>>2 | (i2&8)>>1 | (i3&8)>>0 | (i4&8)<<1 | (i5&8)<<2 | (i6&8)<<3 | (i7&8)<<4 );
        __m512d t = _mm512_mask_blend_pd(blendmask, a, b);
        if (dozero) {
            t = _mm512_maskz_mov_pd(z, t);
        }
        return t;
    }
    // special case: all data stay within their lane
    if (((m1 ^ 0x66442200) & 0x66666666 & mz) == 0) {

        // mask for elements from a and b
        const uint32_t mb = ((i0&8)?0xF:0) | ((i1&8)?0xF0:0) | ((i2&8)?0xF00:0) | ((i3&8)?0xF000:0) | ((i4&8)?0xF0000:0) | ((i5&8)?0xF00000:0) | ((i6&8)?0xF000000:0) | ((i7&8)?0xF0000000:0);
        const uint32_t mbz = mb & mz;     // mask for nonzero elements from b
        const uint32_t maz = ~mb & mz;    // mask for nonzero elements from a
        const uint32_t m1a = m1 & maz;
        const uint32_t m1b = m1 & mbz;
        const uint32_t pata = ((m1a | m1a >> 8 | m1a >> 16 | m1a >> 24) & 0xFF) * 0x01010101;  // permute pattern for elements from a
        const uint32_t patb = ((m1b | m1b >> 8 | m1b >> 16 | m1b >> 24) & 0xFF) * 0x01010101;  // permute pattern for elements from b

        if (((m1 ^ pata) & 0x11111111 & maz) == 0 && ((m1 ^ patb) & 0x11111111 & mbz) == 0) {
            // Same permute pattern in all lanes:
            // todo!: make special case for PSHUFD

            // This code generates two instructions instead of one, but we are avoiding the slow lane-crossing instruction,
            // and we are saving 64 bytes of data cache.
            // 1. Permute a, zero elements not from a (using _mm512_maskz_shuffle_epi32)
            __m512d ta = permute8d< (maz&0xF)?i0&7:-1, (maz&0xF0)?i1&7:-1, (maz&0xF00)?i2&7:-1, (maz&0xF000)?i3&7:-1, 
                (maz&0xF0000)?i4&7:-1, (maz&0xF00000)?i5&7:-1, (maz&0xF000000)?i6&7:-1, (maz&0xF0000000)?i7&7:-1> (a);
            // write mask for elements from b
            const __mmask16 sb = ((mbz&0xF)?3:0) | ((mbz&0xF0)?0xC:0) | ((mbz&0xF00)?0x30:0) | ((mbz&0xF000)?0xC0:0) | ((mbz&0xF0000)?0x300:0) | ((mbz&0xF00000)?0xC00:0) | ((mbz&0xF000000)?0x3000:0) | ((mbz&0xF0000000)?0xC000:0);
            // permute index for elements from b
            const int pi = ((patb & 1) * 10 + 4) | ((((patb >> 4) & 1) * 10 + 4) << 4);
            // 2. Permute elements from b and combine with elements from a through write mask
            return _mm512_castsi512_pd(_mm512_mask_shuffle_epi32(_mm512_castpd_si512(ta), sb, _mm512_castpd_si512(b), (_MM_PERM_ENUM)pi));
        }
        // not same permute pattern in all lanes. use full permute
    }
    // general case: full permute
    const __m512i pmask = constant16i<i0&0xF, 0, i1&0xF, 0, i2&0xF, 0, i3&0xF, 0, i4&0xF, 0, i5&0xF, 0, i6&0xF, 0, i7&0xF, 0>();
    if (dozero) {
        return _mm512_maskz_permutex2var_pd(z, a, pmask, b);
    }
    else {
        return _mm512_permutex2var_pd(a, pmask, b);
    }
}


template <int i0,  int i1,  int i2,  int i3,  int i4,  int i5,  int i6,  int i7, 
          int i8,  int i9,  int i10, int i11, int i12, int i13, int i14, int i15 > 
static inline Vec16f blend16f(Vec16f const & a, Vec16f const & b) {  

    // Combine indexes into a single bitfield, with 4 bits for each indicating shuffle, but not source
    const uint64_t m1 = (i0&0xF) | (i1&0xF)<<4 | (i2&0xF)<<8 | (i3&0xF)<<12 | (i4&0xF)<<16 | (i5&0xF)<<20 | (i6&0xF)<<24 | (i7&0xFLL)<<28
        | (i8&0xFLL)<<32 | (i9&0xFLL)<<36 | (i10&0xFLL)<<40 | (i11&0xFLL)<<44 | (i12&0xFLL)<<48 | (i13&0xFLL)<<52 | (i14&0xFLL)<<56 | (i15&0xFLL)<<60;

    // Mask to zero out negative indexes
    const uint64_t mz = (i0<0?0:0xF) | (i1<0?0:0xF0) | (i2<0?0:0xF00) | (i3<0?0:0xF000) | (i4<0?0:0xF0000) | (i5<0?0:0xF00000) | (i6<0?0:0xF000000) | (i7<0?0:0xF0000000ULL)
        | (i8<0?0:0xF00000000) | (i9<0?0:0xF000000000) | (i10<0?0:0xF0000000000) | (i11<0?0:0xF00000000000) | (i12<0?0:0xF000000000000) | (i13<0?0:0xF0000000000000) | (i14<0?0:0xF00000000000000) | (i15<0?0:0xF000000000000000);
    const uint64_t m2 = m1 & mz;

    // collect bit 4 of each index = select source
    const uint64_t ms = ((i0&16)?0xF:0) | ((i1&16)?0xF0:0) | ((i2&16)?0xF00:0) | ((i3&16)?0xF000:0) | ((i4&16)?0xF0000:0) | ((i5&16)?0xF00000:0) | ((i6&16)?0xF000000:0) | ((i7&16)?0xF0000000ULL:0)
        | ((i8&16)?0xF00000000:0) | ((i9&16)?0xF000000000:0) | ((i10&16)?0xF0000000000:0) | ((i11&16)?0xF00000000000:0) | ((i12&16)?0xF000000000000:0) | ((i13&16)?0xF0000000000000:0) | ((i14&16)?0xF00000000000000:0) | ((i15&16)?0xF000000000000000:0);

    // zeroing needed
    const bool dozero = ((i0|i1|i2|i3|i4|i5|i6|i7|i8|i9|i10|i11|i12|i13|i14|i15) & 0x80) != 0;

    // mask for elements not zeroed
    const __mmask16 z = __mmask16((i0>=0)<<0 | (i1>=0)<<1 | (i2>=0)<<2 | (i3>=0)<<3 | (i4>=0)<<4 | (i5>=0)<<5 | (i6>=0)<<6 | (i7>=0)<<7 
        | (i8>=0)<<8 | (i9>=0)<<9 | (i10>=0)<<10 | (i11>=0)<<11 | (i12>=0)<<12 | (i13>=0)<<13 | (i14>=0)<<14 | (i15>=0)<<15);

    // special case: all zero
    if (mz == 0) return  _mm512_setzero_ps();

    // special case: all from a
    if ((ms & mz) == 0) {
        return permute16f<i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15> (a);
    }

    // special case: all from b
    if ((~ms & mz) == 0) {
        return permute16f<i0^16,i1^16,i2^16,i3^16,i4^16,i5^16,i6^16,i7^16,i8^16,i9^16,i10^16,i11^16,i12^16,i13^16,i14^16,i15^16 > (b);
    }

    // special case: blend without permute
    if (((m1 ^ 0xFEDCBA9876543210) & mz) == 0) {
        __mmask16 blendmask = __mmask16((i0&16)>>4 | (i1&16)>>3 | (i2&16)>>2 | (i3&16)>>1 | (i4&16) | (i5&16)<<1 | (i6&16)<<2 | (i7&16)<<3
            | (i8&16)<<4 | (i9&16)<<5 | (i10&16)<<6 | (i11&16)<<7 | (i12&16)<<8 | (i13&16)<<9 | (i14&16)<<10 | (i15&16)<<11);
        __m512 t = _mm512_mask_blend_ps(blendmask, a, b);
        if (dozero) {
            t = _mm512_maskz_mov_ps(z, t);
        }
        return t;
    }

    // special case: all data stay within their lane
    if (((m1 ^ 0xCCCC888844440000) & 0xCCCCCCCCCCCCCCCC & mz) == 0) {

        // mask for elements from a and b
        const uint64_t mb  = ms;
        const uint64_t mbz = mb & mz;     // mask for nonzero elements from b
        const uint64_t maz = ~mb & mz;    // mask for nonzero elements from a
        const uint64_t m1a = m1 & maz;
        const uint64_t m1b = m1 & mbz;
        const uint64_t pata = ((m1a | m1a >> 16 | m1a >> 32 | m1a >> 48) & 0xFFFF) * 0x0001000100010001;  // permute pattern for elements from a
        const uint64_t patb = ((m1b | m1b >> 16 | m1b >> 32 | m1b >> 48) & 0xFFFF) * 0x0001000100010001;  // permute pattern for elements from b

        if (((m1 ^ pata) & 0x3333333333333333 & maz) == 0 && ((m1 ^ patb) & 0x3333333333333333 & mbz) == 0) {
            // Same permute pattern in all lanes:
            // todo!: special case for SHUFPS

            // This code generates two instructions instead of one, but we are avoiding the slow lane-crossing instruction,
            // and we are saving 64 bytes of data cache.
            // 1. Permute a, zero elements not from a (using _mm512_maskz_shuffle_epi32)
            __m512 ta = permute16f< (maz&0xF)?i0&15:-1, (maz&0xF0)?i1&15:-1, (maz&0xF00)?i2&15:-1, (maz&0xF000)?i3&15:-1, 
                (maz&0xF0000)?i4&15:-1, (maz&0xF00000)?i5&15:-1, (maz&0xF000000)?i6&15:-1, (maz&0xF0000000)?i7&15:-1,
                (maz&0xF00000000)?i8&15:-1, (maz&0xF000000000)?i9&15:-1, (maz&0xF0000000000)?i10&15:-1, (maz&0xF00000000000)?i11&15:-1, 
                (maz&0xF000000000000)?i12&15:-1, (maz&0xF0000000000000)?i13&15:-1, (maz&0xF00000000000000)?i14&15:-1, (maz&0xF000000000000000)?i15&15:-1> (a);
            // write mask for elements from b
            const __mmask16 sb = ((mbz&0xF)?1:0) | ((mbz&0xF0)?0x2:0) | ((mbz&0xF00)?0x4:0) | ((mbz&0xF000)?0x8:0) | ((mbz&0xF0000)?0x10:0) | ((mbz&0xF00000)?0x20:0) | ((mbz&0xF000000)?0x40:0) | ((mbz&0xF0000000)?0x80:0) 
                | ((mbz&0xF00000000)?0x100:0) | ((mbz&0xF000000000)?0x200:0) | ((mbz&0xF0000000000)?0x400:0) | ((mbz&0xF00000000000)?0x800:0) | ((mbz&0xF000000000000)?0x1000:0) | ((mbz&0xF0000000000000)?0x2000:0) | ((mbz&0xF00000000000000)?0x4000:0) | ((mbz&0xF000000000000000)?0x8000:0);
            // permute index for elements from b
            const int pi = (patb & 3) | (((patb >> 4) & 3) << 2) | (((patb >> 8) & 3) << 4) | (((patb >> 12) & 3) << 6);
            // 2. Permute elements from b and combine with elements from a through write mask
            return _mm512_castsi512_ps(_mm512_mask_shuffle_epi32(_mm512_castps_si512(ta), sb, _mm512_castps_si512(b), (_MM_PERM_ENUM)pi));
        }
        // not same permute pattern in all lanes. use full permute
    }

    // general case: full permute
    const __m512i pmask = constant16i<i0&0x1F, i1&0x1F, i2&0x1F, i3&0x1F, i4&0x1F, i5&0x1F, i6&0x1F, i7&0x1F, 
        i8&0x1F, i9&0x1F, i10&0x1F, i11&0x1F, i12&0x1F, i13&0x1F, i14&0x1F, i15&0x1F>();
    if (dozero) {
        return _mm512_maskz_permutex2var_ps(z, a, pmask, b);        
    }
    else {
        return _mm512_permutex2var_ps(a, pmask, b);
    }
}



/*****************************************************************************
*
*          Vector lookup functions
*
******************************************************************************
*
* These functions use vector elements as indexes into a table.
* The table is given as one or more vectors or as an array.
*
* This can be used for several purposes:
*  - table lookup
*  - permute or blend with variable indexes
*  - blend from more than two sources
*  - gather non-contiguous data
*
* An index out of range may produce any value - the actual value produced is
* implementation dependent and may be different for different instruction
* sets. An index out of range does not produce an error message or exception.
*
* Example:
* Vec4i a(2,0,0,3);               // index  a is (  2,   0,   0,   3)
* Vec4f b(1.0f,1.1f,1.2f,1.3f);   // table  b is (1.0, 1.1, 1.2, 1.3)
* Vec4f c;
* c = lookup4 (a,b);              // result c is (1.2, 1.0, 1.0, 1.3)
*
*****************************************************************************/

#ifdef VECTORI_H  // Vec8i and Vec4q must be defined

static inline Vec8f lookup8(Vec8i const & index, Vec8f const & table) {
#if INSTRSET >= 8 && VECTORI_H > 1 // AVX2
#if defined (_MSC_VER) && _MSC_VER < 1700 && ! defined(__INTEL_COMPILER)        
    // bug in MS VS 11 beta: operands in wrong order. fixed in 11.0
    return _mm256_permutevar8x32_ps(_mm256_castsi256_ps(index), _mm256_castps_si256(table)); 
#elif defined (GCC_VERSION) && GCC_VERSION <= 40700 && !defined(__INTEL_COMPILER) && !defined(__clang__)
        // Gcc 4.7.0 has wrong parameter type and operands in wrong order. fixed in version 4.7.1
    return _mm256_permutevar8x32_ps(_mm256_castsi256_ps(index), table);
#else
    // no bug version
    return _mm256_permutevar8x32_ps(table, index);
#endif

#else // AVX
    // swap low and high part of table
    pack4f_t  t1 = _mm256_castps128_ps256(_mm256_extractf128_ps(table, 1));
    pack4f_t  t2 = _mm256_insertf128_ps(t1, _mm256_castps256_ps128(table), 1);
    // join index parts
    __m256i index2 = _mm256_insertf128_si256(_mm256_castsi128_si256(index.get_low()), index.get_high(), 1);
    // permute within each 128-bit part
    pack4f_t  r0 = _mm256_permutevar_ps(table, index2);
    pack4f_t  r1 = _mm256_permutevar_ps(t2,    index2);
    // high index bit for blend
    __m128i k1 = _mm_slli_epi32(index.get_high() ^ 4, 29);
    __m128i k0 = _mm_slli_epi32(index.get_low(),      29);
    pack4f_t  kk = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_castsi128_ps(k0)), _mm_castsi128_ps(k1), 1);
    // blend the two permutes
    return _mm256_blendv_ps(r0, r1, kk);
#endif
}

template <int n>
static inline Vec8f lookup(Vec8i const & index, float const * table) {
    if (n <= 0) return 0;
    if (n <= 4) {
        Vec4f table1 = Vec4f().load(table);        
        return Vec8f(       
            lookup4 (index.get_low(),  table1),
            lookup4 (index.get_high(), table1));
    }
#if INSTRSET < 8  // not AVX2
    if (n <= 8) {
        return lookup8(index, Vec8f().load(table));
    }
#endif
    // Limit index
    Vec8ui index1;
    if ((n & (n-1)) == 0) {
        // n is a power of 2, make index modulo n
        index1 = Vec8ui(index) & (n-1);
    }
    else {
        // n is not a power of 2, limit to n-1
        index1 = min(Vec8ui(index), n-1);
    }
#if INSTRSET >= 8 && VECTORI_H > 1 // AVX2
    return _mm256_i32gather_ps(table, index1, 4);
#else // AVX
    return Vec8f(table[index1[0]],table[index1[1]],table[index1[2]],table[index1[3]],
    table[index1[4]],table[index1[5]],table[index1[6]],table[index1[7]]);
#endif
}

static inline Vec4d lookup4(Vec4q const & index, Vec4d const & table) {
#if INSTRSET >= 8 && VECTORI_H > 1 // AVX2
    // We can't use VPERMPD because it has constant indexes.
    // Convert the index to fit VPERMPS
    Vec8i index1 = permute8i<0,0,2,2,4,4,6,6> (Vec8i(index+index));
    Vec8i index2 = index1 + Vec8i(constant8i<0,1,0,1,0,1,0,1>());
#if defined (_MSC_VER) && _MSC_VER < 1700 && ! defined(__INTEL_COMPILER)        
    // bug in MS VS 11 beta: operands in wrong order. fixed in 11.0
    return _mm256_castps_pd(_mm256_permutevar8x32_ps(_mm256_castsi256_ps(index2), _mm256_castpd_si256(table))); 
#elif defined (GCC_VERSION) && GCC_VERSION <= 40700 && !defined(__INTEL_COMPILER) && !defined(__clang__)
        // Gcc 4.7.0 has wrong parameter type and operands in wrong order
    return _mm256_castps_pd(_mm256_permutevar8x32_ps(_mm256_castsi256_ps(index2), _mm256_castpd_ps(table)));
#else
    // no bug version
    return _mm256_castps_pd(_mm256_permutevar8x32_ps(_mm256_castpd_ps(table), index2));
#endif

#else // AVX
    // swap low and high part of table
    pack4d_t t1 = _mm256_castpd128_pd256(_mm256_extractf128_pd(table, 1));
    pack4d_t t2 = _mm256_insertf128_pd(t1, _mm256_castpd256_pd128(table), 1);
    // index << 1
    __m128i index2lo = index.get_low()  + index.get_low();
    __m128i index2hi = index.get_high() + index.get_high();
    // join index parts
    __m256i index3 = _mm256_insertf128_si256(_mm256_castsi128_si256(index2lo), index2hi, 1);
    // permute within each 128-bit part
    pack4d_t r0 = _mm256_permutevar_pd(table, index3);
    pack4d_t r1 = _mm256_permutevar_pd(t2,    index3);
    // high index bit for blend
    __m128i k1 = _mm_slli_epi64(index.get_high() ^ 2, 62);
    __m128i k0 = _mm_slli_epi64(index.get_low(),      62);
    pack4d_t kk = _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_castsi128_pd(k0)), _mm_castsi128_pd(k1), 1);
    // blend the two permutes
    return _mm256_blendv_pd(r0, r1, kk);
#endif
}

template <int n>
static inline Vec4d lookup(Vec4q const & index, double const * table) {
    if (n <= 0) return 0;
    if (n <= 2) {
        Vec2d table1 = Vec2d().load(table);        
        return Vec4d(       
            lookup2 (index.get_low(),  table1),
            lookup2 (index.get_high(), table1));
    }
#if INSTRSET < 8  // not AVX2
    if (n <= 4) {
        return lookup4(index, Vec4d().load(table));
    }
#endif
    // Limit index
    Vec8ui index1;
    if ((n & (n-1)) == 0) {
        // n is a power of 2, make index modulo n
        index1 = Vec8ui(index) & constant8i<n-1, 0, n-1, 0, n-1, 0, n-1, 0>();
    }
    else {
        // n is not a power of 2, limit to n-1
        index1 = min(Vec8ui(index), constant8i<n-1, 0, n-1, 0, n-1, 0, n-1, 0>() );
    }
#if INSTRSET >= 8 && VECTORI_H > 1 // AVX2
    return _mm256_i64gather_pd(table, index1, 8);
#else // AVX
    Vec4q index2 = Vec4q(index1);
    return Vec4d(table[index2[0]],table[index2[1]],table[index2[2]],table[index2[3]]);
#endif
}
#endif  // VECTORI_H

static inline Vec16f lookup16(Vec16i const & index, Vec16f const & table) {
    return _mm512_permutexvar_ps(index, table);
}

template <int n>
static inline Vec16f lookup(Vec16i const & index, float const * table) {
    if (n <= 0) return 0;
    if (n <= 16) {
        Vec16f table1 = Vec16f().load((float*)table);
        return lookup16(index, table1);
    }
    if (n <= 32) {
        Vec16f table1 = Vec16f().load((float*)table);
        Vec16f table2 = Vec16f().load((float*)table + 16);
        return _mm512_permutex2var_ps(table1, index, table2);
    }
    // n > 32. Limit index
    Vec16ui index1;
    if ((n & (n-1)) == 0) {
        // n is a power of 2, make index modulo n
        index1 = Vec16ui(index) & (n-1);
    }
    else {
        // n is not a power of 2, limit to n-1
        index1 = min(Vec16ui(index), uint32_t(n-1));
    }
    return _mm512_i32gather_ps(index1, (const float*)table, 4);
}


static inline Vec8d lookup8(Vec8q const & index, Vec8d const & table) {
    return _mm512_permutexvar_pd(index, table);
}

template <int n>
static inline Vec8d lookup(Vec8q const & index, double const * table) {
    if (n <= 0) return 0;
    if (n <= 8) {
        Vec8d table1 = Vec8d().load((double*)table);
        return lookup8(index, table1);
    }
    if (n <= 16) {
        Vec8d table1 = Vec8d().load((double*)table);
        Vec8d table2 = Vec8d().load((double*)table + 8);
        return _mm512_permutex2var_pd(table1, index, table2);
    }
    // n > 16. Limit index
    Vec8uq index1;
    if ((n & (n-1)) == 0) {
        // n is a power of 2, make index modulo n
        index1 = Vec8uq(index) & (n-1);
    }
    else {
        // n is not a power of 2, limit to n-1
        index1 = min(Vec8uq(index), uint32_t(n-1));
    }
    return _mm512_i64gather_pd(index1, (const double*)table, 8);
}

/*****************************************************************************
*
*          Gather functions with fixed indexes
*
*****************************************************************************/
// Load elements from array a with indices i0, i1, i2, i3, ..
template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
static inline Vec8f gather8f(void const * a) {
    return reinterpret_f(gather8i<i0, i1, i2, i3, i4, i5, i6, i7>(a));
}

// Load elements from array a with indices i0, i1, i2, i3
template <int i0, int i1, int i2, int i3>
static inline Vec4d gather4d(void const * a) {
    return reinterpret_d(gather4q<i0, i1, i2, i3>(a));
}

// Load elements from array a with indices i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15
template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, 
int i8, int i9, int i10, int i11, int i12, int i13, int i14, int i15>
static inline Vec16f gather16f(void const * a) {
    Static_error_check<(i0|i1|i2|i3|i4|i5|i6|i7|i8|i9|i10|i11|i12|i13|i14|i15)>=0> Negative_array_index;  // Error message if index is negative
    // find smallest and biggest index, using only compile-time constant expressions
    const int i01min   = i0  < i1  ? i0  : i1;
    const int i23min   = i2  < i3  ? i2  : i3;
    const int i45min   = i4  < i5  ? i4  : i5;
    const int i67min   = i6  < i7  ? i6  : i7;
    const int i89min   = i8  < i9  ? i8  : i9;
    const int i1011min = i10 < i11 ? i10 : i11;
    const int i1213min = i12 < i13 ? i12 : i13;
    const int i1415min = i14 < i15 ? i14 : i15;
    const int i0_3min   = i01min   < i23min    ? i01min   : i23min;
    const int i4_7min   = i45min   < i67min    ? i45min   : i67min;
    const int i8_11min  = i89min   < i1011min  ? i89min   : i1011min;
    const int i12_15min = i1213min < i1415min  ? i1213min : i1415min;
    const int i0_7min   = i0_3min  < i4_7min   ? i0_3min  : i4_7min;
    const int i8_15min  = i8_11min < i12_15min ? i8_11min : i12_15min;
    const int imin      = i0_7min  < i8_15min  ? i0_7min  : i8_15min;
    const int i01max   = i0  > i1  ? i0  : i1;
    const int i23max   = i2  > i3  ? i2  : i3;
    const int i45max   = i4  > i5  ? i4  : i5;
    const int i67max   = i6  > i7  ? i6  : i7;
    const int i89max   = i8  > i9  ? i8  : i9;
    const int i1011max = i10 > i11 ? i10 : i11;
    const int i1213max = i12 > i13 ? i12 : i13;
    const int i1415max = i14 > i15 ? i14 : i15;
    const int i0_3max   = i01max   > i23max    ? i01max   : i23max;
    const int i4_7max   = i45max   > i67max    ? i45max   : i67max;
    const int i8_11max  = i89max   > i1011max  ? i89max   : i1011max;
    const int i12_15max = i1213max > i1415max  ? i1213max : i1415max;
    const int i0_7max   = i0_3max  > i4_7max   ? i0_3max  : i4_7max;
    const int i8_15max  = i8_11max > i12_15max ? i8_11max : i12_15max;
    const int imax      = i0_7max  > i8_15max  ? i0_7max  : i8_15max;
    if (imax - imin <= 15) {
        // load one contiguous block and permute
        if (imax > 15) {
            // make sure we don't read past the end of the array
            Vec16f b = Vec16f().load((float const *)a + imax-15);
            return permute16f<i0-imax+15, i1-imax+15, i2-imax+15, i3-imax+15, i4-imax+15, i5-imax+15, i6-imax+15, i7-imax+15,
                i8-imax+15, i9-imax+15, i10-imax+15, i11-imax+15, i12-imax+15, i13-imax+15, i14-imax+15, i15-imax+15> (b);
        }
        else {
            Vec16f b = Vec16f().load((float const *)a + imin);
            return permute16f<i0-imin, i1-imin, i2-imin, i3-imin, i4-imin, i5-imin, i6-imin, i7-imin,
                i8-imin, i9-imin, i10-imin, i11-imin, i12-imin, i13-imin, i14-imin, i15-imin> (b);
        }
    }
    if ((i0<imin+16  || i0>imax-16)  && (i1<imin+16  || i1>imax-16)  && (i2<imin+16  || i2>imax-16)  && (i3<imin+16  || i3>imax-16)
    &&  (i4<imin+16  || i4>imax-16)  && (i5<imin+16  || i5>imax-16)  && (i6<imin+16  || i6>imax-16)  && (i7<imin+16  || i7>imax-16)    
    &&  (i8<imin+16  || i8>imax-16)  && (i9<imin+16  || i9>imax-16)  && (i10<imin+16 || i10>imax-16) && (i11<imin+16 || i11>imax-16)
    &&  (i12<imin+16 || i12>imax-16) && (i13<imin+16 || i13>imax-16) && (i14<imin+16 || i14>imax-16) && (i15<imin+16 || i15>imax-16) ) {
        // load two contiguous blocks and blend
        Vec16f b = Vec16f().load((float const *)a + imin);
        Vec16f c = Vec16f().load((float const *)a + imax-15);
        const int j0  = i0 <imin+16 ? i0 -imin : 31-imax+i0;
        const int j1  = i1 <imin+16 ? i1 -imin : 31-imax+i1;
        const int j2  = i2 <imin+16 ? i2 -imin : 31-imax+i2;
        const int j3  = i3 <imin+16 ? i3 -imin : 31-imax+i3;
        const int j4  = i4 <imin+16 ? i4 -imin : 31-imax+i4;
        const int j5  = i5 <imin+16 ? i5 -imin : 31-imax+i5;
        const int j6  = i6 <imin+16 ? i6 -imin : 31-imax+i6;
        const int j7  = i7 <imin+16 ? i7 -imin : 31-imax+i7;
        const int j8  = i8 <imin+16 ? i8 -imin : 31-imax+i8;
        const int j9  = i9 <imin+16 ? i9 -imin : 31-imax+i9;
        const int j10 = i10<imin+16 ? i10-imin : 31-imax+i10;
        const int j11 = i11<imin+16 ? i11-imin : 31-imax+i11;
        const int j12 = i12<imin+16 ? i12-imin : 31-imax+i12;
        const int j13 = i13<imin+16 ? i13-imin : 31-imax+i13;
        const int j14 = i14<imin+16 ? i14-imin : 31-imax+i14;
        const int j15 = i15<imin+16 ? i15-imin : 31-imax+i15;
        return blend16f<j0,j1,j2,j3,j4,j5,j6,j7,j8,j9,j10,j11,j12,j13,j14,j15>(b, c);
    }
    // use gather instruction
    return _mm512_i32gather_ps(Vec16i(i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15), (const float *)a, 4);
}


template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
static inline Vec8d gather8d(void const * a) {
    Static_error_check<(i0|i1|i2|i3|i4|i5|i6|i7)>=0> Negative_array_index;  // Error message if index is negative

    const int i01min = i0 < i1 ? i0 : i1;
    const int i23min = i2 < i3 ? i2 : i3;
    const int i45min = i4 < i5 ? i4 : i5;
    const int i67min = i6 < i7 ? i6 : i7;
    const int i0123min = i01min < i23min ? i01min : i23min;
    const int i4567min = i45min < i67min ? i45min : i67min;
    const int imin = i0123min < i4567min ? i0123min : i4567min;
    const int i01max = i0 > i1 ? i0 : i1;
    const int i23max = i2 > i3 ? i2 : i3;
    const int i45max = i4 > i5 ? i4 : i5;
    const int i67max = i6 > i7 ? i6 : i7;
    const int i0123max = i01max > i23max ? i01max : i23max;
    const int i4567max = i45max > i67max ? i45max : i67max;
    const int imax = i0123max > i4567max ? i0123max : i4567max;
    if (imax - imin <= 7) {
        // load one contiguous block and permute
        if (imax > 7) {
            // make sure we don't read past the end of the array
            Vec8d b = Vec8d().load((double const *)a + imax-7);
            return permute8d<i0-imax+7, i1-imax+7, i2-imax+7, i3-imax+7, i4-imax+7, i5-imax+7, i6-imax+7, i7-imax+7> (b);
        }
        else {
            Vec8d b = Vec8d().load((double const *)a + imin);
            return permute8d<i0-imin, i1-imin, i2-imin, i3-imin, i4-imin, i5-imin, i6-imin, i7-imin> (b);
        }
    }
    if ((i0<imin+8 || i0>imax-8) && (i1<imin+8 || i1>imax-8) && (i2<imin+8 || i2>imax-8) && (i3<imin+8 || i3>imax-8)
    &&  (i4<imin+8 || i4>imax-8) && (i5<imin+8 || i5>imax-8) && (i6<imin+8 || i6>imax-8) && (i7<imin+8 || i7>imax-8)) {
        // load two contiguous blocks and blend
        Vec8d b = Vec8d().load((double const *)a + imin);
        Vec8d c = Vec8d().load((double const *)a + imax-7);
        const int j0 = i0<imin+8 ? i0-imin : 15-imax+i0;
        const int j1 = i1<imin+8 ? i1-imin : 15-imax+i1;
        const int j2 = i2<imin+8 ? i2-imin : 15-imax+i2;
        const int j3 = i3<imin+8 ? i3-imin : 15-imax+i3;
        const int j4 = i4<imin+8 ? i4-imin : 15-imax+i4;
        const int j5 = i5<imin+8 ? i5-imin : 15-imax+i5;
        const int j6 = i6<imin+8 ? i6-imin : 15-imax+i6;
        const int j7 = i7<imin+8 ? i7-imin : 15-imax+i7;
        return blend8d<j0, j1, j2, j3, j4, j5, j6, j7>(b, c);
    }
    // use gather instruction
    return _mm512_i64gather_pd(Vec8q(i0,i1,i2,i3,i4,i5,i6,i7), (const double *)a, 8);
}


/*****************************************************************************
*
*          Vector scatter functions
*
******************************************************************************
*
* These functions write the elements of a vector to arbitrary positions in an
* array in memory. Each vector element is written to an array position 
* determined by an index. An element is not written if the corresponding
* index is out of range.
* The indexes can be specified as constant template parameters or as an
* integer vector.
* 
* The scatter functions are useful if the data are distributed in a sparce
* manner into the array. If the array is dense then it is more efficient
* to permute the data into the right positions and then write the whole
* permuted vector into the array.
*
* Example:
* Vec8d a(10,11,12,13,14,15,16,17);
* double b[16] = {0};
* scatter<0,2,14,10,1,-1,5,9>(a,b); 
* // Now, b = {10,14,11,0,0,16,0,0,0,17,13,0,0,0,12,0}
*
*****************************************************************************/

template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
static inline void scatter(Vec8f const & data, float * array) {
#if defined (__AVX512VL__)
    __m256i indx = constant8i<i0,i1,i2,i3,i4,i5,i6,i7>();
    __mmask16 mask = uint16_t(i0>=0 | (i1>=0)<<1 | (i2>=0)<<2 | (i3>=0)<<3| (i4>=0)<<4| (i5>=0)<<5| (i6>=0)<<6| (i7>=0)<<7);
    _mm256_mask_i32scatter_ps(array, mask, indx, data, 4);
#elif defined (__AVX512F__)
    __m512i indx = _mm512_castsi256_si512(constant8i<i0,i1,i2,i3,i4,i5,i6,i7>());
    __mmask16 mask = uint16_t(i0>=0 | (i1>=0)<<1 | (i2>=0)<<2 | (i3>=0)<<3| (i4>=0)<<4| (i5>=0)<<5| (i6>=0)<<6| (i7>=0)<<7);
    _mm512_mask_i32scatter_ps(array, mask, indx, _mm512_castps256_ps512(data), 4);
#else
    const int index[8] = {i0,i1,i2,i3,i4,i5,i6,i7};
    for (int i = 0; i < 8; i++) {
        if (index[i] >= 0) array[index[i]] = data[i];
    }
#endif
}

template <int i0, int i1, int i2, int i3>
static inline void scatter(Vec4d const & data, double * array) {
#if defined (__AVX512VL__)
    __m128i indx = constant4i<i0,i1,i2,i3>();
    __mmask16 mask = uint16_t(i0>=0 | (i1>=0)<<1 | (i2>=0)<<2 | (i3>=0)<<3);
    _mm256_mask_i32scatter_pd(array, mask, indx, data, 8);
#elif defined (__AVX512F__)
    __m256i indx = _mm256_castsi128_si256(constant4i<i0,i1,i2,i3>());
    __mmask16 mask = uint16_t(i0>=0 | (i1>=0)<<1 | (i2>=0)<<2 | (i3>=0)<<3);
    _mm512_mask_i32scatter_pd(array, mask, indx, _mm512_castpd256_pd512(data), 8);
#else
    const int index[4] = {i0,i1,i2,i3};
    for (int i = 0; i < 4; i++) {
        if (index[i] >= 0) array[index[i]] = data[i];
    }
#endif
}

static inline void scatter(Vec8i const & index, uint32_t limit, Vec8f const & data, float * array) {
#if defined (__AVX512VL__)
    __mmask16 mask = _mm256_cmplt_epu32_mask(index, Vec8ui(limit));
    _mm256_mask_i32scatter_ps(array, mask, index, data, 4);
#elif defined (__AVX512F__)
    // 16 bit mask. upper 8 bits are (0<0) = false
    __mmask16 mask = _mm512_cmplt_epu32_mask(_mm512_castsi256_si512(index), _mm512_castsi256_si512(Vec8ui(limit)));
    _mm512_mask_i32scatter_ps(array, mask, _mm512_castsi256_si512(index), _mm512_castps256_ps512(data), 4);
#else
    for (int i = 0; i < 8; i++) {
        if (uint32_t(index[i]) < limit) array[index[i]] = data[i];
    }
#endif
}

static inline void scatter(Vec4q const & index, uint32_t limit, Vec4d const & data, double * array) {
#if defined (__AVX512VL__)
    __mmask16 mask = _mm256_cmplt_epu64_mask(index, Vec4uq(uint64_t(limit)));
    _mm256_mask_i64scatter_pd(array, mask, index, data, 8);
#elif defined (__AVX512F__)
    // 16 bit mask. upper 8 bits are (0<0) = false
    __mmask16 mask = _mm512_cmplt_epu64_mask(_mm512_castsi256_si512(index), _mm512_castsi256_si512(Vec4uq(uint64_t(limit))));
    _mm512_mask_i64scatter_pd(array, mask, _mm512_castsi256_si512(index), _mm512_castpd256_pd512(data), 8);
#else
    for (int i = 0; i < 4; i++) {
        if (uint64_t(index[i]) < uint64_t(limit)) array[index[i]] = data[i];
    }
#endif
} 

static inline void scatter(Vec4i const & index, uint32_t limit, Vec4d const & data, double * array) {
#if defined (__AVX512VL__)
    __mmask16 mask = _mm_cmplt_epu32_mask(index, Vec4ui(limit));
    _mm256_mask_i32scatter_pd(array, mask, index, data, 8);
#elif defined (__AVX512F__)
    // 16 bit mask. upper 12 bits are (0<0) = false
    __mmask16 mask = _mm512_cmplt_epu32_mask(_mm512_castsi128_si512(index), _mm512_castsi128_si512(Vec4ui(limit)));
    _mm512_mask_i32scatter_pd(array, mask, _mm256_castsi128_si256(index), _mm512_castpd256_pd512(data), 8);
#else
    for (int i = 0; i < 4; i++) {
        if (uint32_t(index[i]) < limit) array[index[i]] = data[i];
    }
#endif
} 

template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7,
    int i8, int i9, int i10, int i11, int i12, int i13, int i14, int i15>
    static inline void scatter(Vec16f const & data, float * array) {
    __m512i indx = constant16i<i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15>();
    Vec16fb mask(i0>=0, i1>=0, i2>=0, i3>=0, i4>=0, i5>=0, i6>=0, i7>=0,
        i8>=0, i9>=0, i10>=0, i11>=0, i12>=0, i13>=0, i14>=0, i15>=0);
    _mm512_mask_i32scatter_ps(array, mask, indx, data, 4);
}

template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
static inline void scatter(Vec8d const & data, double * array) {
    __m256i indx = constant8i<i0,i1,i2,i3,i4,i5,i6,i7>();
    Vec8db mask(i0>=0, i1>=0, i2>=0, i3>=0, i4>=0, i5>=0, i6>=0, i7>=0);
    _mm512_mask_i32scatter_pd(array, mask, indx, data, 8);
}

static inline void scatter(Vec16i const & index, uint32_t limit, Vec16f const & data, float * array) {
    Vec16fb mask = Vec16ui(index) < limit;
    _mm512_mask_i32scatter_ps(array, mask, index, data, 4);
}

static inline void scatter(Vec8q const & index, uint32_t limit, Vec8d const & data, double * array) {
    Vec8db mask = Vec8uq(index) < uint64_t(limit);
    _mm512_mask_i64scatter_pd(array, mask, index, data, 8);
}

static inline void scatter(Vec8i const & index, uint32_t limit, Vec8d const & data, double * array) {
#if defined (__AVX512VL__)
    __mmask16 mask = _mm256_cmplt_epu32_mask(index, Vec8ui(limit));
#else
    __mmask16 mask = _mm512_cmplt_epu32_mask(_mm512_castsi256_si512(index), _mm512_castsi256_si512(Vec8ui(limit)));
#endif
    _mm512_mask_i32scatter_pd(array, mask, index, data, 8);
}

/*****************************************************************************
*
*          Horizontal scan functions
*
*****************************************************************************/

// Get index to the first element that is true. Return -1 if all are false
static inline int horizontal_find_first(Vec8fb const & x) {
    return horizontal_find_first(Vec8ib(x));
}

static inline int horizontal_find_first(Vec4db const & x) {
    return horizontal_find_first(Vec4qb(x));
}

// Count the number of elements that are true
static inline uint32_t horizontal_count(Vec8fb const & x) {
    return horizontal_count(Vec8ib(x));
}

static inline uint32_t horizontal_count(Vec4db const & x) {
    return horizontal_count(Vec4qb(x));
}

// Get index to the first element that is true. Return -1 if all are false
static inline int horizontal_find_first(Vec16fb const & x) {
    return horizontal_find_first(Vec16ib(x));
}

static inline int horizontal_find_first(Vec8db const & x) {
    return horizontal_find_first(Vec8qb(x));
}

// Count the number of elements that are true
static inline uint32_t horizontal_count(Vec16fb const & x) {
    return horizontal_count(Vec16ib(x));
}

static inline uint32_t horizontal_count(Vec8db const & x) {
    return horizontal_count(Vec8qb(x));
}

/*****************************************************************************
*
*          Boolean <-> bitfield conversion functions
*
*****************************************************************************/

// to_bits: convert boolean vector to integer bitfield
static inline uint8_t to_bits(Vec8fb const & x) {
    return to_bits(Vec8ib(x));
}

// to_Vec8fb: convert integer bitfield to boolean vector
static inline Vec8fb to_Vec8fb(uint8_t x) {
    return Vec8fb(to_Vec8ib(x));
}

// to_bits: convert boolean vector to integer bitfield
static inline uint8_t to_bits(Vec4db const & x) {
    return to_bits(Vec4qb(x));
}

// to_Vec4db: convert integer bitfield to boolean vector
static inline Vec4db to_Vec4db(uint8_t x) {
    return Vec4db(to_Vec4qb(x));
}

// to_bits: convert boolean vector to integer bitfield
static inline uint16_t to_bits(Vec16fb x) {
    return to_bits(Vec16ib(x));
}

// to_Vec16fb: convert integer bitfield to boolean vector
static inline Vec16fb to_Vec16fb(uint16_t x) {
    return Vec16fb(to_Vec16ib(x));
}

// to_bits: convert boolean vector to integer bitfield
static inline uint8_t to_bits(Vec8db x) {
    return to_bits(Vec8qb(x));
}

// to_Vec8db: convert integer bitfield to boolean vector
static inline Vec8db to_Vec8db(uint8_t x) {
    return Vec8db(to_Vec8qb(x));
}

#ifdef NSIMD_NAMESPACE
}
#endif

#endif // VECTORF_H
