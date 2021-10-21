/****************************  vectori128.h   *******************************
* Author:        Agner Fog
* Date created:  2012-05-30
* Last modified: 2017-05-02
* Version:       1.28
* Project:       vector classes
* Description:
* Header file defining integer vector classes as interface to intrinsic 
* functions in x86 microprocessors with SSE2 and later instruction sets
* up to AVX.
*
* Instructions:
* Use Gnu, Intel or Microsoft C++ compiler. Compile for the desired 
* instruction set, which must be at least SSE2. Specify the supported 
* instruction set by a command line define, e.g. __SSE4_1__ if the 
* compiler does not automatically do so.
*
* The following vector classes are defined here:
* Vec128b   Vector of 128  1-bit unsigned  integers or Booleans
* Vec16c    Vector of  16  8-bit signed    integers
* Vec16uc   Vector of  16  8-bit unsigned  integers
* Vec16cb   Vector of  16  Booleans for use with Vec16c and Vec16uc
* Vec8s     Vector of   8  16-bit signed   integers
* Vec8us    Vector of   8  16-bit unsigned integers
* Vec8sb    Vector of   8  Booleans for use with Vec8s and Vec8us
* Vec4i     Vector of   4  32-bit signed   integers
* Vec4ui    Vector of   4  32-bit unsigned integers
* Vec4ib    Vector of   4  Booleans for use with Vec4i and Vec4ui
* Vec2q     Vector of   2  64-bit signed   integers
* Vec2uq    Vector of   2  64-bit unsigned integers
* Vec2qb    Vector of   2  Booleans for use with Vec2q and Vec2uq
*
* Each vector object is represented internally in the CPU as a 128-bit register.
* This header file defines operators and functions for these vectors.
*
* For example:
* Vec4i a(1,2,3,4), b(5,6,7,8), c;
* c = a + b;     // now c contains (6,8,10,12)
*
* For detailed instructions, see Nsimd.pdf
*
* (c) Copyright 2012-2017 GNU General Public License http://www.gnu.org/licenses
*****************************************************************************/
#ifndef VECTORI128_H
#define VECTORI128_H

#include "instrset.h"  // Select supported instruction set
#include "vector_types.h"
#include "nsimd_common.h"

#if INSTRSET < 2   // SSE2 required
#error Please compile for the SSE2 instruction set or higher
#endif

#ifdef NSIMD_NAMESPACE
namespace NSIMD_NAMESPACE {
#endif

/*****************************************************************************
*
*          Vector of 128 1-bit unsigned integers or Booleans
*
*****************************************************************************/
class Vec128b {
public:
    // Default constructor:
    Vec128b() {
    }
    // Constructor to broadcast the same value into all elements
    // Removed because of undesired implicit conversions
    // Vec128b(int i) {
    //     xmm = _mm_set1_epi32(-(i & 1));}

    // Constructor to convert from type __m128i used in intrinsics:
    Vec128b(__m128i const & x) {
        xmm = x;
    }
    // Assignment operator to convert from type __m128i used in intrinsics:
    Vec128b & operator = (__m128i const & x) {
        xmm = x;
        return *this;
    }
    // Type cast operator to convert to __m128i used in intrinsics
    operator __m128i() const {
        return xmm;
    }
    // Member function to load from array (unaligned)
    Vec128b & load(void const * p) {
        xmm = _mm_loadu_si128((__m128i const*)p);
        return *this;
    }
    // Member function to load from array, aligned by 16
    // "load_a" is faster than "load" on older Intel processors (Pentium 4, Pentium M, Core 1,
    // Merom, Wolfdale) and Atom, but not on other processors from Intel, AMD or VIA.
    // You may use load_a instead of load if you are certain that p points to an address
    // divisible by 16.
    void load_a(void const * p) {
        xmm = _mm_load_si128((__m128i const*)p);
    }
    // Member function to store into array (unaligned)
    void store(void * p) const {
        _mm_storeu_si128((__m128i*)p, xmm);
    }
    // Member function to store into array, aligned by 16
    // "store_a" is faster than "store" on older Intel processors (Pentium 4, Pentium M, Core 1,
    // Merom, Wolfdale) and Atom, but not on other processors from Intel, AMD or VIA.
    // You may use store_a instead of store if you are certain that p points to an address
    // divisible by 16.
    void store_a(void * p) const {
        _mm_store_si128((__m128i*)p, xmm);
    }
    // Member function to change a single bit
    // Note: This function is inefficient. Use load function if changing more than one bit
    Vec128b const & set_bit(uint32_t index, int value) {
        static const union {
            uint64_t i[4];
            __m128i  x[2];
        } u = {{1,0,0,1}};                 // 2 vectors with bit 0 and 64 set, respectively
        int w = (index >> 6) & 1;          // qword index
        int bi = index & 0x3F;             // bit index within qword w
        __m128i mask = u.x[w];
        mask = _mm_sll_epi64(mask,_mm_cvtsi32_si128(bi)); // mask with bit number b set
        if (value & 1) {
            xmm = _mm_or_si128(mask,xmm);
        }
        else {
            xmm = _mm_andnot_si128(mask,xmm);
        }
        return *this;
    }
    // Member function to get a single bit
    // Note: This function is inefficient. Use store function if reading more than one bit
    int get_bit(uint32_t index) const {
        union {
            __m128i x;
            uint8_t i[16];
        } u;
        u.x = xmm; 
        int w = (index >> 3) & 0xF;            // byte index
        int bi = index & 7;                    // bit index within byte w
        return (u.i[w] >> bi) & 1;
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    bool operator [] (uint32_t index) const {
        return nsimd_common::get_bit(index) != 0;
    }
    static int size() {
        return 128;
    }
};


// Define operators for this class

// vector operator & : bitwise and
static inline Vec128b operator & (Vec128b const & a, Vec128b const & b) {
    return _mm_and_si128(a, b);
}
static inline Vec128b operator && (Vec128b const & a, Vec128b const & b) {
    return a & b;
}

// vector operator | : bitwise or
static inline Vec128b operator | (Vec128b const & a, Vec128b const & b) {
    return _mm_or_si128(a, b);
}
static inline Vec128b operator || (Vec128b const & a, Vec128b const & b) {
    return a | b;
}

// vector operator ^ : bitwise xor
static inline Vec128b operator ^ (Vec128b const & a, Vec128b const & b) {
    return _mm_xor_si128(a, b);
}

// vector operator ~ : bitwise not
static inline Vec128b operator ~ (Vec128b const & a) {
    return _mm_xor_si128(a, _mm_set1_epi32(-1));
}

// vector operator &= : bitwise and
static inline Vec128b & operator &= (Vec128b & a, Vec128b const & b) {
    a = a & b;
    return a;
}

// vector operator |= : bitwise or
static inline Vec128b & operator |= (Vec128b & a, Vec128b const & b) {
    a = a | b;
    return a;
}

// vector operator ^= : bitwise xor
static inline Vec128b & operator ^= (Vec128b & a, Vec128b const & b) {
    a = a ^ b;
    return a;
}

// Define functions for this class

// function andnot: a & ~ b
static inline Vec128b andnot (Vec128b const & a, Vec128b const & b) {
    return _mm_andnot_si128(b, a);
}


/*****************************************************************************
*
*          Generate compile-time constant vector
*
*****************************************************************************/
// Generate a constant vector of 4 integers stored in memory.
// Can be converted to any integer vector type
template <int32_t i0, int32_t i1, int32_t i2, int32_t i3>
static inline pack128_4i_t constant4i() {
    int data[4] = {i0, i1, i2, i3};
    pack128_4i_t res = nsimd::loadu<pack128_4i_t>(data);
    return res;
}

template <uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3>
static inline __m128i constant4ui() {
    return constant4i<int32_t(i0), int32_t(i1), int32_t(i2), int32_t(i3)>();
}

/*****************************************************************************
*
*          selectb function
*
*****************************************************************************/
// Select between two sources, byte by byte. Used in various functions and operators
// Corresponds to this pseudocode:
// for (int i = 0; i < 16; i++) result[i] = s[i] ? a[i] : b[i];
// Each byte in s must be either 0 (false) or 0xFF (true). No other values are allowed.
// The implementation depends on the instruction set: 
// If SSE4.1 is supported then only bit 7 in each byte of s is checked, 
// otherwise all bits in s are used.
static inline pack128_16i_t selectb (pack128_16i_t const & s, pack128_16i_t const & a, pack128_16i_t const & b) {
    return nsimd::if_else(nsimd::to_logical(b), a, s);
}



/*****************************************************************************
*
*          Horizontal Boolean functions
*
*****************************************************************************/

// horizontal_and. Returns true if all bits are 1
static inline bool horizontal_and (Vec128b const & a) {
    return nsimd::all(a);
}

// horizontal_or. Returns true if at least one bit is 1
static inline bool horizontal_or (Vec128b const & a) {
    return nsimd::any(a);
}



/*****************************************************************************
*
*          Vector of 16 8-bit signed integers
*
*****************************************************************************/

class Vec16c : public Vec128b {
protected:
    pack128_16i_t xmm; // Integer vector
public:
    // Default constructor:
    Vec16c() {
    }// Constructor to convert from type __m128i used in intrinsics:
    Vec128b(pack128_16i_t const & x) {
        xmm = x;
    }
    // Constructor to broadcast the same value into all elements:
    Vec16c(int i) {
        xmm = nsimd::set1<pack128_16i_t>((char)i);
    }
    // Constructor to build from all elements:
    Vec16c(int8_t i0, int8_t i1, int8_t i2, int8_t i3, int8_t i4, int8_t i5, int8_t i6, int8_t i7,
        int8_t i8, int8_t i9, int8_t i10, int8_t i11, int8_t i12, int8_t i13, int8_t i14, int8_t i15) {
        int8_t vec[16] = {i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15};
        xmm = nsimd::loada<pack128_16i_t>(vec);
    }
    // Constructor to convert from type __m128i used in intrinsics:
    Vec16c(pack128_16i_t const & x) {
        xmm = x;
    }
    // Assignment operator to convert from type __m128i used in intrinsics:
    Vec16c & operator = (pack128_16i_t const & x) {
        xmm = x;
        return *this;
    }
    // Type cast operator to convert to __m128i used in intrinsics
    operator pack128_16i_t() const {
        return xmm;
    }
    // Member function to load from array (unaligned)
    Vec16c & load(void const * p) {
        xmm = nsimd::loadu<pack128_16i_t>((pack128_16i_t const*)p);
        return *this;
    }
    // Member function to load from array (aligned)
    Vec16c & load_a(void const * p) {
        xmm = nsimd::loada<pack128_16i_t>((pack128_16i_t const*)p);
        return *this;
    }
    // Partial load. Load n elements and set the rest to 0
    Vec16c & load_partial(int n, void const * p) {
        xmm = nsimd_common::load_partial<pack128_16i_t, packl128_16i_t, char>(p, n)
        return *this;
    }
    // Partial store. Store n elements
    void store_partial(int n, void * p) const {
        nsimd_common::store_partial<pack128_16i_t, packl128_16i_t, char>(p, n, xmm);
    }
    // cut off vector to n elements. The last 16-n elements are set to zero
    Vec16c & cutoff(int n) {
        xmm = nsimd_common::cutoff<pack128_16i_t, packl128_16i_t>(xmm, n);
        return *this;
    }
    // Member function to change a single element in vector
    // Note: This function is inefficient. Use load function if changing more than one element
    Vec16c const & insert(uint32_t index, int8_t value) {
        xmm = nsimd_common::set_bit<pack128_16i_t, char>(index, value, xmm)
        return *this;
    }
    // Member function extract a single element from vector
    int8_t extract(uint32_t index) const {
        int8_t x[16];
        store(x);
        return x[index & 0x0F];
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    int8_t operator [] (uint32_t index) const {
        return extract(index);
    }
    static int size() {
        return 16;
    }
};

/*****************************************************************************
*
*          Vec16cb: Vector of 16 Booleans for use with Vec16c and Vec16uc
*
*****************************************************************************/

class Vec16cb : public Vec16c {
protected:
    packl128_16i_t xmm; // Integer vector
public:
    // Default constructor
    Vec16cb() {}
    // Constructor to build from all elements:
    Vec16cb(bool x0, bool x1, bool x2, bool x3, bool x4, bool x5, bool x6, bool x7,
        bool x8, bool x9, bool x10, bool x11, bool x12, bool x13, bool x14, bool x15) {     
        int8_t vec[16] = {-int8_t(x0), -int8_t(x1), -int8_t(x2), -int8_t(x3), -int8_t(x4), -int8_t(x5), -int8_t(x6), -int8_t(x7), 
            -int8_t(x8), -int8_t(x9), -int8_t(x10), -int8_t(x11), -int8_t(x12), -int8_t(x13), -int8_t(x14), -int8_t(x15)};
        xmm = nsimd::loadla<packl128_16i_t>(vec);
    }
    // Constructor to convert from type __m128i used in intrinsics:
    Vec16cb(packl128_16i_t const & x) {
        xmm = x;
    }
    // Assignment operator to convert from type __m128i used in intrinsics:
    Vec16cb & operator = (packl128_16i_t const & x) {
        xmm = x;
        return *this;
    }
    // Constructor to broadcast scalar value:
    Vec16cb(bool b) {
        xmm = nsimd::loadla<pack128_16i_t>(nsimd::set1l<packl128_16i_t>(-int8_t(b)));
    }
    // Assignment operator to broadcast scalar value:
    Vec16cb & operator = (bool b) {
        *this = Vec16cb(b);
        return *this;
    }
private: // Prevent constructing from int, etc.
    Vec16cb(int b);
    Vec16cb & operator = (int x);
public:
    Vec16cb & insert(int index, bool value) {
        xmm = nsimd_common::set_bit<packl128_16i_t, char>(index, -int8_t(value), xmm);
        return *this;
    }
    // Member function extract a single element from vector
    int8_t extract(uint32_t index) const {
        int8_t x[16];
        store(x);
        return x[index & 0x0F] != 0;
    }

    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    bool operator [] (uint32_t index) const {
        return extract(index);
    }
    // Type cast operator to convert to __m128i used in intrinsics
    operator packl128_16i_t() const {
        return xmm;
    }
};


/*****************************************************************************
*
*          Define operators for Vec16cb
*
*****************************************************************************/

// vector operator & : bitwise and
static inline Vec16cb operator & (Vec16cb const & a, Vec16cb const & b) {
    return nsimd::andl(a, b);
}
static inline Vec16cb operator && (Vec16cb const & a, Vec16cb const & b) {
    return a & b;
}
// vector operator &= : bitwise and
static inline Vec16cb & operator &= (Vec16cb & a, Vec16cb const & b) {
    a = a & b;
    return a;
}

// vector operator | : bitwise or
static inline Vec16cb operator | (Vec16cb const & a, Vec16cb const & b) {
    return nsimd::orl(a, b);
}
static inline Vec16cb operator || (Vec16cb const & a, Vec16cb const & b) {
    return a | b;
}
// vector operator |= : bitwise or
static inline Vec16cb & operator |= (Vec16cb & a, Vec16cb const & b) {
    a = a | b;
    return a;
}

// vector operator ^ : bitwise xor
static inline Vec16cb operator ^ (Vec16cb const & a, Vec16cb const & b) {
    return nsimd::xorl(a, b);
}
// vector operator ^= : bitwise xor
static inline Vec16cb & operator ^= (Vec16cb & a, Vec16cb const & b) {
    a = a ^ b;
    return a;
}

// vector operator ~ : bitwise not
static inline Vec16cb operator ~ (Vec16cb const & a) {
    return nsimd::notl(a);
}

// vector operator ! : element not
static inline Vec16cb operator ! (Vec16cb const & a) {
    return ~ a;
}

// vector function andnot
static inline Vec16cb andnot (Vec16cb const & a, Vec16cb const & b) {
    return nsimd::andnotl(a, b);
}

// Horizontal Boolean functions for Vec16cb

// horizontal_and. Returns true if all elements are true
static inline bool horizontal_and(Vec16cb const & a) {
    return nsimd::all(a);
}

// horizontal_or. Returns true if at least one element is true
static inline bool horizontal_or(Vec16cb const & a) {
    return nsimd::any(a);
} 


/*****************************************************************************
*
*          Define operators for Vec16c
*
*****************************************************************************/

// vector operator + : add element by element
static inline Vec16c operator + (Vec16c const & a, Vec16c const & b) {
    return nsimd::add(a, b);
}

// vector operator += : add
static inline Vec16c & operator += (Vec16c & a, Vec16c const & b) {
    a = a + b;
    return a;
}

// postfix operator ++
static inline Vec16c operator ++ (Vec16c & a, int) {
    Vec16c a0 = a;
    a = a + 1;
    return a0;
}

// prefix operator ++
static inline Vec16c & operator ++ (Vec16c & a) {
    a = a + 1;
    return a;
}

// vector operator - : subtract element by element
static inline Vec16c operator - (Vec16c const & a, Vec16c const & b) {
    return nsimd::sub(a, b);
}

// vector operator - : unary minus
static inline Vec16c operator - (Vec16c const & a) {
    return nsimd::sub(nsimd::set1<pack128_16i_t>(a), a);
}

// vector operator -= : add
static inline Vec16c & operator -= (Vec16c & a, Vec16c const & b) {
    a = a - b;
    return a;
}

// postfix operator --
static inline Vec16c operator -- (Vec16c & a, int) {
    Vec16c a0 = a;
    a = a - 1;
    return a0;
}

// prefix operator --
static inline Vec16c & operator -- (Vec16c & a) {
    a = a - 1;
    return a;
}

// vector operator * : multiply element by element
static inline Vec16c operator * (Vec16c const & a, Vec16c const & b) {
    return nsimd::mul(a, b);
}

// vector operator *= : multiply
static inline Vec16c & operator *= (Vec16c & a, Vec16c const & b) {
    a = a * b;
    return a;
}

// vector operator << : shift left all elements
static inline Vec16c operator << (Vec16c const & a, int b) {
    return nsimd::shl(a, b);
}

// vector operator <<= : shift left
static inline Vec16c & operator <<= (Vec16c & a, int b) {
    a = nsimd::shl(a, b);
    return a;
}

// vector operator >> : shift right arithmetic all elements
static inline Vec16c operator >> (Vec16c const & a, int b) {
    return nsimd::shr(a, b);
}

// vector operator >>= : shift right arithmetic
static inline Vec16c & operator >>= (Vec16c & a, int b) {
    a = nsimd::shr(a, b);
    return a;
}

// vector operator == : returns true for elements for which a == b
static inline Vec16cb operator == (Vec16c const & a, Vec16c const & b) {
    return nsimd::eq(a,b);
}

// vector operator != : returns true for elements for which a != b
static inline Vec16cb operator != (Vec16c const & a, Vec16c const & b) {
    return nsimd::ne(a,b);
}

// vector operator > : returns true for elements for which a > b (signed)
static inline Vec16cb operator > (Vec16c const & a, Vec16c const & b) {
    return nsimd::gt(a,b);
}

// vector operator < : returns true for elements for which a < b (signed)
static inline Vec16cb operator < (Vec16c const & a, Vec16c const & b) {
    return nsimd::lt(a, b);
}

// vector operator >= : returns true for elements for which a >= b (signed)
static inline Vec16cb operator >= (Vec16c const & a, Vec16c const & b) {
    return nsimd::ge(a,b);
}

// vector operator <= : returns true for elements for which a <= b (signed)
static inline Vec16cb operator <= (Vec16c const & a, Vec16c const & b) {
    return nsimd::le(a,b);
}

// vector operator & : bitwise and
static inline Vec16c operator & (Vec16c const & a, Vec16c const & b) {
    return nsimd::andb(a,b);
}
static inline Vec16c operator && (Vec16c const & a, Vec16c const & b) {
    return a & b;
}
// vector operator &= : bitwise and
static inline Vec16c & operator &= (Vec16c & a, Vec16c const & b) {
    a = nsimd::andb(a, b);
    return a;
}

// vector operator | : bitwise or
static inline Vec16c operator | (Vec16c const & a, Vec16c const & b) {
    return nsimd::orb(a, b);
}
static inline Vec16c operator || (Vec16c const & a, Vec16c const & b) {
    return a | b;
}
// vector operator |= : bitwise or
static inline Vec16c & operator |= (Vec16c & a, Vec16c const & b) {
    a = nsimd::orb(a, b);
    return a;
}

// vector operator ^ : bitwise xor
static inline Vec16c operator ^ (Vec16c const & a, Vec16c const & b) {
    return nsimd::xorb(a, b);
}
// vector operator ^= : bitwise xor
static inline Vec16c & operator ^= (Vec16c & a, Vec16c const & b) {
    a = a ^ b;
    return a;
}

// vector operator ~ : bitwise not
static inline Vec16c operator ~ (Vec16c const & a) {
    return nsimd::notb(a);
}

// vector operator ! : logical not, returns true for elements == 0
static inline Vec16cb operator ! (Vec16c const & a) {
    return nsimd::eq(a, nsimd::set1<pack128_16i_t>(char(0)));
}

// Functions for this class

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 16; i++) result[i] = s[i] ? a[i] : b[i];
// Each byte in s must be either 0 (false) or -1 (true). No other values are allowed.
static inline Vec16c select (Vec16cb const & s, Vec16c const & a, Vec16c const & b) {
    return selectb(s,a,b);
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
static inline Vec16c if_add (Vec16cb const & f, Vec16c const & a, Vec16c const & b) {
    return a + (Vec16c(f) & b);
}

// Horizontal add: Calculates the sum of all vector elements.
// Overflow will wrap around
static inline int32_t horizontal_add (Vec16c const & a) {
    return nsimd::addv(a);                                         // sign extend to 32 bits
}

// Horizontal add extended: Calculates the sum of all vector elements.
// Each element is sign-extended before addition to avoid overflow
static inline int32_t horizontal_add_x (Vec16c const & a) {
    return nsimd::addv(a);
}


// function add_saturated: add element by element, signed with saturation
static inline Vec16c add_saturated(Vec16c const & a, Vec16c const & b) {
    return nsimd::adds(a,b);
}

// function sub_saturated: subtract element by element, signed with saturation
static inline Vec16c sub_saturated(Vec16c const & a, Vec16c const & b) {
    return nsimd::subs(a,b);
}

// function max: a > b ? a : b
static inline Vec16c max(Vec16c const & a, Vec16c const & b) {
    return nsimd::max(a,b);
}

// function min: a < b ? a : b
static inline Vec16c min(Vec16c const & a, Vec16c const & b) {
    return nsimd::min(a,b);
}

// function abs: a >= 0 ? a : -a
static inline Vec16c abs(Vec16c const & a) {
    return nsimd::abs(a);
}

// function abs_saturated: same as abs, saturate if overflow
static inline Vec16c abs_saturated(Vec16c const & a) {
    pack128_16i_t absa   = nsimd::abs(pack);
    return nsimd::adds(absa, nsimd::set1<pack128_16i_t>(char(0)));
}

// function rotate_left: rotate each element left by b bits 
// Use negative count to rotate right
static inline Vec16c rotate_left(Vec16c const & aa, int ba) {
#ifdef __XOP__  // AMD XOP instruction set
    return _mm_rot_epi8(aa.native_register(),_mm_set1_epi8(ba.native_register()));
#else  // SSE2 instruction set
    __m128i a         = aa.native_register();
    __m128i b         = ba.native_register();
    __m128i bb        = _mm_cvtsi32_si128(b & 7);          // b modulo 8
    __m128i mbb       = _mm_cvtsi32_si128((8-b) & 7);      // 8-b modulo 8
    __m128i maskeven  = _mm_set1_epi32(0x00FF00FF);        // mask for even numbered bytes
    __m128i even      = _mm_and_si128(a,maskeven);         // even numbered bytes of a
    __m128i odd       = _mm_andnot_si128(maskeven,a);      // odd numbered bytes of a
    __m128i evenleft  = _mm_sll_epi16(even,bb);            // even bytes of a << b
    __m128i oddleft   = _mm_sll_epi16(odd,bb);             // odd  bytes of a << b
    __m128i evenright = _mm_srl_epi16(even,mbb);           // even bytes of a >> 8-b
    __m128i oddright  = _mm_srl_epi16(odd,mbb);            // odd  bytes of a >> 8-b
    __m128i evenrot   = _mm_or_si128(evenleft,evenright);  // even bytes of a rotated
    __m128i oddrot    = _mm_or_si128(oddleft,oddright);    // odd  bytes of a rotated
    __m128i allrot    = selectb(maskeven,evenrot,oddrot);  // all  bytes rotated
    return  allrot;
#endif
}


/*****************************************************************************
*
*          Vector of 16 8-bit unsigned integers
*
*****************************************************************************/

class Vec16uc : public Vec16c {
protected:
    pack128_16ui_t xmm; // Integer vector
public:
    // Default constructor:
    Vec16uc() {
    }
    // Constructor to broadcast the same value into all elements:
    Vec16uc(uint32_t i) {
        xmm = nsimd::set1<pack128_16ui_t>((char)i);
    }
    // Constructor to build from all elements:
    Vec16uc(uint8_t i0, uint8_t i1, uint8_t i2, uint8_t i3, uint8_t i4, uint8_t i5, uint8_t i6, uint8_t i7,
        uint8_t i8, uint8_t i9, uint8_t i10, uint8_t i11, uint8_t i12, uint8_t i13, uint8_t i14, uint8_t i15) {
        char data[16] = {i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15};
        xmm = nsimd::set1<pack128_16ui_t>(data);
    }
    // Constructor to convert from type __m128i used in intrinsics:
    Vec16uc(pack128_16ui_t const & x) {
        xmm = x;
    }
    // Assignment operator to convert from type __m128i used in intrinsics:
    Vec16uc & operator = (pack128_16ui_t const & x) {
        xmm = x;
        return *this;
    }
    // Member function to load from array (unaligned)
    Vec16uc & load(void const * p) {
        xmm = nsimd::loadu<pack128_16ui_t>((*char)p);
        return *this;
    }
    // Member function to load from array (aligned)
    Vec16uc & load_a(void const * p) {
        xmm = nsimd::loada<pack128_16ui_t>((*char)p);
        return *this;
    }
    // Member function to change a single element in vector
    // Note: This function is inefficient. Use load function if changing more than one element
    Vec16uc const & insert(uint32_t index, uint8_t value) {
        Vec16c::insert(index, value);
        return *this;
    }
    // Member function extract a single element from vector
    uint8_t extract(uint32_t index) const {
        return Vec16c::extract(index);
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    uint8_t operator [] (uint32_t index) const {
        return extract(index);
    }
};

// Define operators for this class

// vector operator << : shift left all elements
static inline Vec16uc operator << (Vec16uc const & a, uint32_t b) {
    return nsimd::shl(a, b);
}

// vector operator << : shift left all elements
static inline Vec16uc operator << (Vec16uc const & a, int32_t b) {
    return  nsimd::shl(a, (uint32_t)b);
}

// vector operator >> : shift right logical all elements
static inline Vec16uc operator >> (Vec16uc const & a, uint32_t b) {
    return nsimd::shr(a, b);
}

// vector operator >> : shift right logical all elements
static inline Vec16uc operator >> (Vec16uc const & a, int32_t b) {
    return nsimd::shr(a, (uint32_t)b);
}

// vector operator >>= : shift right logical
static inline Vec16uc & operator >>= (Vec16uc & a, int b) {
    a = a >> b;
    return a;
}

// vector operator >= : returns true for elements for which a >= b (unsigned)
static inline Vec16cb operator >= (Vec16uc const & a, Vec16uc const & b) {
    return nsimd::ge(a,b);
}

// vector operator <= : returns true for elements for which a <= b (unsigned)
static inline Vec16cb operator <= (Vec16uc const & a, Vec16uc const & b) {
    return nsimd::le(a,b);
}

// vector operator > : returns true for elements for which a > b (unsigned)
static inline Vec16cb operator > (Vec16uc const & a, Vec16uc const & b) {
    return nsimd::gt(a,b);
}

// vector operator < : returns true for elements for which a < b (unsigned)
static inline Vec16cb operator < (Vec16uc const & a, Vec16uc const & b) {
    return nsimd::lt(a,b);
}

// vector operator + : add
static inline Vec16uc operator + (Vec16uc const & a, Vec16uc const & b) {
    return nsimd::add(a,b);
}

// vector operator - : subtract
static inline Vec16uc operator - (Vec16uc const & a, Vec16uc const & b) {
    return nsimd::sub(a,b);
}

// vector operator * : multiply
static inline Vec16uc operator * (Vec16uc const & a, Vec16uc const & b) {
    return nsimd::mul(a,b);
}

// vector operator & : bitwise and
static inline Vec16uc operator & (Vec16uc const & a, Vec16uc const & b) {
    return nsimd::andb(a,b);
}
static inline Vec16uc operator && (Vec16uc const & a, Vec16uc const & b) {
    return a & b;
}

// vector operator | : bitwise or
static inline Vec16uc operator | (Vec16uc const & a, Vec16uc const & b) {
    return nsimd::orb(a,b);
}
static inline Vec16uc operator || (Vec16uc const & a, Vec16uc const & b) {
    return a | b;
}

// vector operator ^ : bitwise xor
static inline Vec16uc operator ^ (Vec16uc const & a, Vec16uc const & b) {
    return nsimd::xorb(a,b);
}

// vector operator ~ : bitwise not
static inline Vec16uc operator ~ (Vec16uc const & a) {
    return nsimd::notb(a);
}

// Functions for this class

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 16; i++) result[i] = s[i] ? a[i] : b[i];
// Each byte in s must be either 0 (false) or -1 (true). No other values are allowed.
// (s is signed)
static inline Vec16uc select (Vec16cb const & s, Vec16uc const & a, Vec16uc const & b) {
    return selectb(s,a,b);
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
static inline Vec16uc if_add (Vec16cb const & f, Vec16uc const & a, Vec16uc const & b) {
    return a + (Vec16uc(f) & b);
}

// Horizontal add: Calculates the sum of all vector elements.
// Overflow will wrap around
// (Note: horizontal_add_x(Vec16uc) is slightly faster)
static inline uint32_t horizontal_add (Vec16uc const & a) {
    return nsimd::addv(a);
}

// Horizontal add extended: Calculates the sum of all vector elements.
// Each element is zero-extended before addition to avoid overflow
static inline uint32_t horizontal_add_x (Vec16uc const & a) {
    return nsimd::addv(a);
}

// function add_saturated: add element by element, unsigned with saturation
static inline Vec16uc add_saturated(Vec16uc const & a, Vec16uc const & b) {
    return nsimd::adds(a, b);
}

// function sub_saturated: subtract element by element, unsigned with saturation
static inline Vec16uc sub_saturated(Vec16uc const & a, Vec16uc const & b) {
    return nsimd::subs(a, b);
}

// function max: a > b ? a : b
static inline Vec16uc max(Vec16uc const & a, Vec16uc const & b) {
    return nsimd::max(a,b);
}

// function min: a < b ? a : b
static inline Vec16uc min(Vec16uc const & a, Vec16uc const & b) {
    return nsimd::min(a,b);
}


    
/*****************************************************************************
*
*          Vector of 8 16-bit signed integers
*
*****************************************************************************/

class Vec8s : public Vec128b {
protected:
    pack128_8i_t xmm; // Integer vector
public:
    // Default constructor:
    Vec8s() {
    }
    // Constructor to broadcast the same value into all elements:
    Vec8s(int i) {
        xmm = nsimd::set1<pack128_8i_t>((int16_t)i);
    }
    // Constructor to build from all elements:
    Vec8s(int16_t i0, int16_t i1, int16_t i2, int16_t i3, int16_t i4, int16_t i5, int16_t i6, int16_t i7) {
        int16_t data[8] = {i0, i1, i2, i3, i4, i5, i6, i7};
        xmm = nsimd::set1<pack128_8i_t>(data);
    }
    // Constructor to convert from type __m128i used in intrinsics:
    Vec8s(pack128_8i_t const & x) {
        xmm = x;
    }
    // Assignment operator to convert from type __m128i used in intrinsics:
    Vec8s & operator = (pack128_8i_t const & x) {
        xmm = x;
        return *this;
    }
    // Type cast operator to convert to __m128i used in intrinsics
    operator pack128_8i_t() const {
        return xmm;
    }
    // Member function to load from array (unaligned)
    Vec8s & load(void const * p) {
        xmm = nsimd::loadu<pack128_8i_t>((short*)p);
        return *this;
    }
    // Member function to load from array (aligned)
    Vec8s & load_a(void const * p) {
        xmm = nsimd::loada<pack128_8i_t>((short*)p);
        return *this;
    }
    // Partial load. Load n elements and set the rest to 0
    Vec8s & load_partial(int n, void const * p) {
        xmm = nsimd_common::load_partial<pack128_8i_t, packl128_8i_t, short>(p, n)
        return *this;
    }
    // Partial store. Store n elements
    void store_partial(int n, void * p) const {
        nsimd_common::store_partial<pack128_8i_t, packl128_8i_t, short>(p, n, xmm);
    }

    // cut off vector to n elements. The last 8-n elements are set to zero
    Vec8s & cutoff(int n) {
        xmm = nsimd_common::cutoff<pack128_8i_t, packl128_8i_t>(xmm, n);
        return *this;
    }
    // Member function to change a single element in vector
    // Note: This function is inefficient. Use load function if changing more than one element
    Vec8s const & insert(uint32_t index, int16_t value) {
        xmm = nsimd_common::set_bit<pack128_8i_t, short>(index, value, xmm)
        return *this;
    }
    // Member function extract a single element from vector
    // Note: This function is inefficient. Use store function if extracting more than one element
    int16_t extract(uint32_t index) const {
        return nsimd_common::get_bit<pack128_8i_t, short>(index, xmm);
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    int16_t operator [] (uint32_t index) const {
        return extract(index);
    }
    static int size() {
        return 8;
    }
};

/*****************************************************************************
*
*          Vec8sb: Vector of 8 Booleans for use with Vec8s and Vec8us
*
*****************************************************************************/

class Vec8sb : public Vec8s {
protected:
    packl128_8i_t xmm; // Integer vector
public:
    // Constructor to build from all elements:
    Vec8sb(bool x0, bool x1, bool x2, bool x3, bool x4, bool x5, bool x6, bool x7) {
        int16_t vec[8] = {-int16_t(x0), -int16_t(x1), -int16_t(x2), -int16_t(x3), -int16_t(x4), -int16_t(x5), -int16_t(x6), -int16_t(x7)};
        xmm = nsimd::loadla<packl128_8i_t>(vec);
    }
    // Default constructor:
    Vec8sb() {
    }
    // Constructor to convert from type __m128i used in intrinsics:
    Vec8sb(packl128_8i_t const & x) {
        xmm = x;
    }
    // Assignment operator to convert from type __m128i used in intrinsics:
    Vec8sb & operator = (packl128_8i_t const & x) {
        xmm = x;
        return *this;
    }
    // Constructor to broadcast scalar value:
    Vec8sb(bool b) : Vec8s(-int16_t(b)) {
        xmm = nsimd::set1<packl128_8i_t>(-int16_t(b));
    }
    // Assignment operator to broadcast scalar value:
    Vec8sb & operator = (bool b) {
        *this = Vec8sb(b);
        return *this;
    }
private: // Prevent constructing from int, etc.
    Vec8sb(int b);
    Vec8sb & operator = (int x);
public:
    Vec8sb & insert (int index, bool a) {
        xmm = nsimd_common::set_bit<packl128_8i_t, short>(index, -int16_t(value), xmm);
        return *this;
    }
    // Member function extract a single element from vector
    // Note: This function is inefficient. Use store function if extracting more than one element
    bool extract(uint32_t index) const {
        return nsimd_common::get_bit<packl128_8i_t, short>(index, xmm);
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    bool operator [] (uint32_t index) const {
        return extract(index);
    }

    // Type cast operator to convert
    operator packl128_8i_t() const {
        return xmm;
    }
};


/*****************************************************************************
*
*          Define operators for Vec8sb
*
*****************************************************************************/

// vector operator & : bitwise and
static inline Vec8sb operator & (Vec8sb const & a, Vec8sb const & b) {
    return nsimd::andl(a,b);
}
static inline Vec8sb operator && (Vec8sb const & a, Vec8sb const & b) {
    return a & b;
}
// vector operator &= : bitwise and
static inline Vec8sb & operator &= (Vec8sb & a, Vec8sb const & b) {
    a = nsimd::andb(a,b);
    return a;
}

// vector operator | : bitwise or
static inline Vec8sb operator | (Vec8sb const & a, Vec8sb const & b) {
    return nsimd::orl(a,b);
}
static inline Vec8sb operator || (Vec8sb const & a, Vec8sb const & b) {
    return a | b;
}
// vector operator |= : bitwise or
static inline Vec8sb & operator |= (Vec8sb & a, Vec8sb const & b) {
    a = nsimd::orl(a,b);
    return a;
}

// vector operator ^ : bitwise xor
static inline Vec8sb operator ^ (Vec8sb const & a, Vec8sb const & b) {
    return nsimd::xorl(a,b);
}
// vector operator ^= : bitwise xor
static inline Vec8sb & operator ^= (Vec8sb & a, Vec8sb const & b) {
    a = a ^ b;
    return a;
}

// vector operator ~ : bitwise not
static inline Vec8sb operator ~ (Vec8sb const & a) {
    return nsimd::notl(a);
}

// vector operator ! : element not
static inline Vec8sb operator ! (Vec8sb const & a) {
    return ~ a;
}

// vector function andnot
static inline Vec8sb andnot (Vec8sb const & a, Vec8sb const & b) {
    return nsimd::andnotl(a,b);
}

// Horizontal Boolean functions for Vec8sb

// horizontal_and. Returns true if all elements are true
static inline bool horizontal_and(Vec8sb const & a) {
    return nsimd::all(a);
}

// horizontal_or. Returns true if at least one element is true
static inline bool horizontal_or(Vec8sb const & a) {
    return nsimd::any(a);
}


/*****************************************************************************
*
*         operators for Vec8s
*
*****************************************************************************/

// vector operator + : add element by element
static inline Vec8s operator + (Vec8s const & a, Vec8s const & b) {
    return nsimd::add(a,b);
}

// vector operator += : add
static inline Vec8s & operator += (Vec8s & a, Vec8s const & b) {
    a = nsimd::add(a,b);
    return a;
}

// postfix operator ++
static inline Vec8s operator ++ (Vec8s & a, int) {
    Vec8s a0 = a;
    a = a + 1;
    return a0;
}

// prefix operator ++
static inline Vec8s & operator ++ (Vec8s & a) {
    a = a + 1;
    return a;
}

// vector operator - : subtract element by element
static inline Vec8s operator - (Vec8s const & a, Vec8s const & b) {
    return nsimd::sub(a, b);
}

// vector operator - : unary minus
static inline Vec8s operator - (Vec8s const & a) {
    return nsimd::sub(nsimd::set1<pack128_8i_t>(short(0)), a);
}

// vector operator -= : subtract
static inline Vec8s & operator -= (Vec8s & a, Vec8s const & b) {
    a = a - b;
    return a;
}

// postfix operator --
static inline Vec8s operator -- (Vec8s & a, int) {
    Vec8s a0 = a;
    a = a - 1;
    return a0;
}

// prefix operator --
static inline Vec8s & operator -- (Vec8s & a) {
    a = a - 1;
    return a;
}

// vector operator * : multiply element by element
static inline Vec8s operator * (Vec8s const & a, Vec8s const & b) {
    return nsimd::mul(a, b);
}

// vector operator *= : multiply
static inline Vec8s & operator *= (Vec8s & a, Vec8s const & b) {
    a = a * b;
    return a;
}

// vector operator / : divide all elements by same integer
// See bottom of file


// vector operator << : shift left
static inline Vec8s operator << (Vec8s const & a, int b) {
    return nsimd::shl(a,b);
}

// vector operator <<= : shift left
static inline Vec8s & operator <<= (Vec8s & a, int b) {
    a = nsimd::shl(a,b);
    return a;
}

// vector operator >> : shift right arithmetic
static inline Vec8s operator >> (Vec8s const & a, int b) {
    return nsimd::shr(a,b);
}

// vector operator >>= : shift right arithmetic
static inline Vec8s & operator >>= (Vec8s & a, int b) {
    a = nsimd::shr(a,b);
    return a;
}

// vector operator == : returns true for elements for which a == b
static inline Vec8sb operator == (Vec8s const & a, Vec8s const & b) {
    return nsimd::eq(a,b);
}

// vector operator != : returns true for elements for which a != b
static inline Vec8sb operator != (Vec8s const & a, Vec8s const & b) {
    return nsimd::ne(a,b);
}

// vector operator > : returns true for elements for which a > b
static inline Vec8sb operator > (Vec8s const & a, Vec8s const & b) {
    return nsimd::gt(a, b);
}

// vector operator < : returns true for elements for which a < b
static inline Vec8sb operator < (Vec8s const & a, Vec8s const & b) {
    return nsimd::lt(a,b);
}

// vector operator >= : returns true for elements for which a >= b (signed)
static inline Vec8sb operator >= (Vec8s const & a, Vec8s const & b) {
    return nsimd::ge(a,b);
}

// vector operator <= : returns true for elements for which a <= b (signed)
static inline Vec8sb operator <= (Vec8s const & a, Vec8s const & b) {
    return nsimd::le(a,b);
}

// vector operator & : bitwise and
static inline Vec8s operator & (Vec8s const & a, Vec8s const & b) {
    return nsimd::andb(a,b);
}
static inline Vec8s operator && (Vec8s const & a, Vec8s const & b) {
    return a & b;
}
// vector operator &= : bitwise and
static inline Vec8s & operator &= (Vec8s & a, Vec8s const & b) {
    a = a & b;
    return a;
}

// vector operator | : bitwise or
static inline Vec8s operator | (Vec8s const & a, Vec8s const & b) {
    return nsimd::orb(a,b);
}
static inline Vec8s operator || (Vec8s const & a, Vec8s const & b) {
    return a | b;
}
// vector operator |= : bitwise or
static inline Vec8s & operator |= (Vec8s & a, Vec8s const & b) {
    a = a | b;
    return a;
}

// vector operator ^ : bitwise xor
static inline Vec8s operator ^ (Vec8s const & a, Vec8s const & b) {
    return nsimd::xorb(a,b);
}
// vector operator ^= : bitwise xor
static inline Vec8s & operator ^= (Vec8s & a, Vec8s const & b) {
    a = a ^ b;
    return a;
}

// vector operator ~ : bitwise not
static inline Vec8s operator ~ (Vec8s const & a) {
    return Vec8s( ~ Vec128b(a));
}

// vector operator ! : logical not, returns true for elements == 0
static inline Vec8s operator ! (Vec8s const & a) {
    return nsimd::notb(a);
}

// Functions for this class

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 8; i++) result[i] = s[i] ? a[i] : b[i];
// Each byte in s must be either 0 (false) or -1 (true). No other values are allowed.
// (s is signed)
static inline Vec8s select (Vec8sb const & s, Vec8s const & a, Vec8s const & b) {
    return selectb(s,a,b);
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
static inline Vec8s if_add (Vec8sb const & f, Vec8s const & a, Vec8s const & b) {
    return a + (Vec8s(f) & b);
}

// Horizontal add: Calculates the sum of all vector elements.
// Overflow will wrap around
static inline int32_t horizontal_add (Vec8s const & a) {
    return nsimd::addv(a);
}

// Horizontal add extended: Calculates the sum of all vector elements.
// Elements are sign extended before adding to avoid overflow
static inline int32_t horizontal_add_x (Vec8s const & a) {
    return nsimd::addv(a);
}

// function add_saturated: add element by element, signed with saturation
static inline Vec8s add_saturated(Vec8s const & a, Vec8s const & b) {
    return nsimd::adds(a, b);
}

// function sub_saturated: subtract element by element, signed with saturation
static inline Vec8s sub_saturated(Vec8s const & a, Vec8s const & b) {
    return nsimd::subs(a, b);
}

// function max: a > b ? a : b
static inline Vec8s max(Vec8s const & a, Vec8s const & b) {
    return nsimd::max(a,b);
}

// function min: a < b ? a : b
static inline Vec8s min(Vec8s const & a, Vec8s const & b) {
    return nsimd::min(a,b);
}

// function abs: a >= 0 ? a : -a
static inline Vec8s abs(Vec8s const & a) {
    return nsimd::abs(a);
}

// function abs_saturated: same as abs, saturate if overflow
static inline Vec8s abs_saturated(Vec8s const & a) {
    return nsimd::adds(nsimd::abs(a), nsimd::set1<pack128_8i_t>(short(0)));
}

// function rotate_left all elements
// Use negative count to rotate right
static inline Vec8s rotate_left(Vec8s const & a, int b) {
#ifdef __XOP__  // AMD XOP instruction set
    return _mm_rot_epi16(a,_mm_set1_epi16(b));
#else  // SSE2 instruction set
    __m128i left  = _mm_sll_epi16(a,_mm_cvtsi32_si128(b & 0x0F));      // a << b 
    __m128i right = _mm_srl_epi16(a,_mm_cvtsi32_si128((16-b) & 0x0F)); // a >> (16 - b)
    __m128i rot   = _mm_or_si128(left,right);                          // or
    return  rot;
#endif
}


/*****************************************************************************
*
*          Vector of 8 16-bit unsigned integers
*
*****************************************************************************/

class Vec8us : public Vec8s {
protected:
    pack128_8ui_t xmm; // Integer vector
public:
    // Default constructor:
    Vec8us() {
    }
    // Constructor to broadcast the same value into all elements:
    Vec8us(uint32_t i) {
        xmm = nsimd::set1<pack128_8ui_t>((uint16_t)i);
    }
    // Constructor to build from all elements:
    Vec8us(uint16_t i0, uint16_t i1, uint16_t i2, uint16_t i3, uint16_t i4, uint16_t i5, uint16_t i6, uint16_t i7) {
        uint16_t[8] data = {i0, i1, i2, i3, i4, i5, i6, i7}
        xmm = nsimd::loadu<pack128_8ui_t>(data);
    }
    // Constructor to convert from type __m128i used in intrinsics:
    Vec8us(pack128_8ui_t const & x) {
        xmm = x;
    }
    // Assignment operator to convert from type __m128i used in intrinsics:
    Vec8us & operator = (pack128_8ui_t const & x) {
        xmm = x;
        return *this;
    }
    // Member function to load from array (unaligned)
    Vec8us & load(void const * p) {
        xmm = nsimd::loadu<pack128_8ui_t>((uint16_t const*)p);
        return *this;
    }
    // Member function to load from array (aligned)
    Vec8us & load_a(void const * p) {
        xmm = nsimd::loada<pack128_8ui_t>((uint16_t const*)p);
        return *this;
    }
    // Member function to change a single element in vector
    // Note: This function is inefficient. Use load function if changing more than one element
    Vec8us const & insert(uint32_t index, uint16_t value) {
        xmm = nsimd_common::set_bit<pack128_8ui_t, short>(index, value, xmm)
        return *this;
    }
    // Member function extract a single element from vector
    uint16_t extract(uint32_t index) const {
        return nsimd_common::get_bit<pack128_8ui_t, short>(index, xmm);
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    uint16_t operator [] (uint32_t index) const {
        return extract(index);
    }

    // Type cast operator to convert
    operator packl128_8i_t() const {
        return xmm;
    }
};

// Define operators for this class

// vector operator + : add
static inline Vec8us operator + (Vec8us const & a, Vec8us const & b) {
    return nsimd::add(a,b);
}

// vector operator - : subtract
static inline Vec8us operator - (Vec8us const & a, Vec8us const & b) {
    return nsimd::sub(a,b);
}

// vector operator * : multiply
static inline Vec8us operator * (Vec8us const & a, Vec8us const & b) {
    return nsimd::mul(a,b);
}

// vector operator / : divide
// See bottom of file

// vector operator >> : shift right logical all elements
static inline Vec8us operator >> (Vec8us const & a, uint32_t b) {
    return nsimd::shl(a, b);
}

// vector operator >> : shift right logical all elements
static inline Vec8us operator >> (Vec8us const & a, int32_t b) {
    return a >> (uint32_t)b;
}

// vector operator >>= : shift right logical
static inline Vec8us & operator >>= (Vec8us & a, int b) {
    a = a >> b;
    return a;
}

// vector operator << : shift left all elements
static inline Vec8us operator << (Vec8us const & a, uint32_t b) {
    return nsimd::shr(a, b); 
}

// vector operator << : shift left all elements
static inline Vec8us operator << (Vec8us const & a, int32_t b) {
    return a << (uint32_t)b;
}

// vector operator >= : returns true for elements for which a >= b (unsigned)
static inline Vec8sb operator >= (Vec8us const & a, Vec8us const & b) {
    return nsimd::ge(a,b);
}

// vector operator <= : returns true for elements for which a <= b (unsigned)
static inline Vec8sb operator <= (Vec8us const & a, Vec8us const & b) {
    return b >= a;
}

// vector operator > : returns true for elements for which a > b (unsigned)
static inline Vec8sb operator > (Vec8us const & a, Vec8us const & b) {
    return nsimd::gt(a,b);
}

// vector operator < : returns true for elements for which a < b (unsigned)
static inline Vec8sb operator < (Vec8us const & a, Vec8us const & b) {
    return b > a;
}

// vector operator & : bitwise and
static inline Vec8us operator & (Vec8us const & a, Vec8us const & b) {
    return nsimd::andb(a, b);
}
static inline Vec8us operator && (Vec8us const & a, Vec8us const & b) {
    return a & b;
}

// vector operator | : bitwise or
static inline Vec8us operator | (Vec8us const & a, Vec8us const & b) {
    return nsimd::orb(a,b);
}
static inline Vec8us operator || (Vec8us const & a, Vec8us const & b) {
    return a | b;
}

// vector operator ^ : bitwise xor
static inline Vec8us operator ^ (Vec8us const & a, Vec8us const & b) {
    return nsimd::xorb(a,b);
}

// vector operator ~ : bitwise not
static inline Vec8us operator ~ (Vec8us const & a) {
    return nsimd::notb(a);
}

// Functions for this class

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 8; i++) result[i] = s[i] ? a[i] : b[i];
// Each word in s must be either 0 (false) or -1 (true). No other values are allowed.
// (s is signed)
static inline Vec8us select (Vec8sb const & s, Vec8us const & a, Vec8us const & b) {
    return selectb(s,a,b);
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
static inline Vec8us if_add (Vec8sb const & f, Vec8us const & a, Vec8us const & b) {
    return a + (Vec8us(f) & b);
}

// Horizontal add: Calculates the sum of all vector elements.
// Overflow will wrap around
static inline uint32_t horizontal_add (Vec8us const & a) {
    return nsimd::addv(a);
}

// Horizontal add extended: Calculates the sum of all vector elements.
// Each element is zero-extended before addition to avoid overflow
static inline uint32_t horizontal_add_x (Vec8us const & a) {
    return nsimd::addv(a);
}

// function add_saturated: add element by element, unsigned with saturation
static inline Vec8us add_saturated(Vec8us const & a, Vec8us const & b) {
    return nsimd::adds(a, b);
}

// function sub_saturated: subtract element by element, unsigned with saturation
static inline Vec8us sub_saturated(Vec8us const & a, Vec8us const & b) {
    return nsimd::subs(a, b);
}

// function max: a > b ? a : b
static inline Vec8us max(Vec8us const & a, Vec8us const & b) {
    return nsimd::max(a, b);
}

// function min: a < b ? a : b
static inline Vec8us min(Vec8us const & a, Vec8us const & b) {
    return nsimd::min(a, b);
}



/*****************************************************************************
*
*          Vector of 4 32-bit signed integers
*
*****************************************************************************/

class Vec4i : public Vec128b {
protected:
    pack128_4i_t xmm; // Integer vector
public:
    // Default constructor:
    Vec4i() {
    }
    // Constructor to broadcast the same value into all elements:
    Vec4i(int i) {
        xmm = nsimd::set1<pack128_4i_t>(i);
    }
    // Constructor to build from all elements:
    Vec4i(int32_t i0, int32_t i1, int32_t i2, int32_t i3) {
        int32_t data[4] = {i0, i1, i2, i3}
        xmm = nsimd::loadu<pack128_4i_t>(data);
    }
    // Constructor to convert from type __m128i used in intrinsics:
    Vec4i(pack128_4i_t const & x) {
        xmm = x;
    }
    // Assignment operator to convert from type __m128i used in intrinsics:
    Vec4i & operator = (pack128_4i_t const & x) {
        xmm = x;
        return *this;
    }
    // Type cast operator to convert to __m128i used in intrinsics
    operator pack128_4i_t() const {
        return xmm;
    }
    // Member function to load from array (unaligned)
    Vec4i & load(void const * p) {
        xmm = nsimd::loadu<pack128_4i_t>((int32_t const *)p);
        return *this;
    }
    // Member function to load from array (aligned)
    Vec4i & load_a(void const * p) {
        xmm = nsimd::loada<pack128_4i_t>((int32_t const *)p);
        return *this;
    }

    // Partial load. Load n elements and set the rest to 0
    Vec4i & load_partial(int n, void const * p) {
        xmm = nsimd_common::load_partial<pack128_4i_t, packl128_4i_t, int>(p, n)
        return *this;
    }
    // Partial store. Store n elements
    void store_partial(int n, void * p) const {
        nsimd_common::store_partial<pack128_4i_t, packl128_4i_t, int>(p, n, xmm);
    }
    // cut off vector to n elements. The last 4-n elements are set to zero
    Vec4i & cutoff(int n) {
        xmm = nsimd_common::cutoff<pack128_4i_t, packl128_4i_t>(xmm, n);
        return *this;
    }
    // Member function to change a single element in vector
    // Note: This function is inefficient. Use load function if changing more than one element
    Vec4i const & insert(uint32_t index, int32_t value) {
        xmm = nsimd_common::set_bit<pack128_4i_t, int>(index, value, xmm)
        return *this;
    }
    // Member function extract a single element from vector
    int32_t extract(uint32_t index) const {
        int32_t x[4];
        store(x);
        return x[index & 3];
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    int32_t operator [] (uint32_t index) const {
        return extract(index);
    }
    static int size() {
        return 4;
    }
};


/*****************************************************************************
*
*          Vec4ib: Vector of 4 Booleans for use with Vec4i and Vec4ui
*
*****************************************************************************/
class Vec4ib : public Vec4i {
protected:
    packl128_4i_t xmm; // Integer vector
public:
    // Default constructor:
    Vec4ib() {
    }
    // Constructor to build from all elements:
    Vec4ib(bool x0, bool x1, bool x2, bool x3) {
        int32_t vec[4] = {-int32_t(x0), -int32_t(x1), -int32_t(x2), -int32_t(x3)};
        xmm = nsimd::loadla<packl128_8i_t>(vec);
    }
    // Constructor to convert from type __m128i used in intrinsics:
    Vec4ib(packl128_8i_t const & x) {
        xmm = x;
    }
    // Assignment operator to convert from type __m128i used in intrinsics:
    Vec4ib & operator = (packl128_8i_t const & x) {
        xmm = x;
        return *this;
    }
    // Constructor to broadcast scalar value:
    Vec4ib(bool b) {
        xmm = nsimd::set1<packl128_4i_t>(-int32_t(b));
    }
    // Assignment operator to broadcast scalar value:
    Vec4ib & operator = (bool b) {
        *this = Vec4ib(b);
        return *this;
    }
private: // Prevent constructing from int, etc.
    Vec4ib(int b);
    Vec4ib & operator = (int x);
public:
    Vec4ib & insert (int index, bool a) {
        xmm = nsimd_common::set_bit<packl128_4i_t, int>(index, -int32_t(value), xmm);
        return *this;
    }    
    // Member function extract a single element from vector
    bool extract(uint32_t index) const {
        return nsimd_common::get_bit<packl128_4i_t, int>(index, xmm) != 0;
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    bool operator [] (uint32_t index) const {
        return extract(index);
    }

    // Type cast operator to convert
    operator packl128_4i_t() const {
        return xmm;
    }
};


/*****************************************************************************
*
*          Define operators for Vec4ib
*
*****************************************************************************/

// vector operator & : bitwise and
static inline Vec4ib operator & (Vec4ib const & a, Vec4ib const & b) {
    return nsimd::andl(a, b);
}
static inline Vec4ib operator && (Vec4ib const & a, Vec4ib const & b) {
    return a & b;
}
// vector operator &= : bitwise and
static inline Vec4ib & operator &= (Vec4ib & a, Vec4ib const & b) {
    a = a & b;
    return a;
}

// vector operator | : bitwise or
static inline Vec4ib operator | (Vec4ib const & a, Vec4ib const & b) {
    return nsimd::orl(a,b);
}
static inline Vec4ib operator || (Vec4ib const & a, Vec4ib const & b) {
    return a | b;
}
// vector operator |= : bitwise or
static inline Vec4ib & operator |= (Vec4ib & a, Vec4ib const & b) {
    a = a | b;
    return a;
}

// vector operator ^ : bitwise xor
static inline Vec4ib operator ^ (Vec4ib const & a, Vec4ib const & b) {
    return nsimd::xorl(a,b);
}
// vector operator ^= : bitwise xor
static inline Vec4ib & operator ^= (Vec4ib & a, Vec4ib const & b) {
    a = a ^ b;
    return a;
}

// vector operator ~ : bitwise not
static inline Vec4ib operator ~ (Vec4ib const & a) {
    return nsimd::notl(a);
}

// vector operator ! : element not
static inline Vec4ib operator ! (Vec4ib const & a) {
    return ~ a;
}

// vector function andnot
static inline Vec4ib andnot (Vec4ib const & a, Vec4ib const & b) {
    return nsimd::andnotl(a,b);
}

// Horizontal Boolean functions for Vec4ib

// horizontal_and. Returns true if all elements are true
static inline bool horizontal_and(Vec4ib const & a) {
    return nsimd::all(a);
}

// horizontal_or. Returns true if at least one element is true
static inline bool horizontal_or(Vec4ib const & a) {
    return nsimd::any(a);
}


/*****************************************************************************
*
*          Operators for Vec4i
*
*****************************************************************************/

// vector operator + : add element by element
static inline Vec4i operator + (Vec4i const & a, Vec4i const & b) {
    return nsimd::add0(a, b);
}

// vector operator += : add
static inline Vec4i & operator += (Vec4i & a, Vec4i const & b) {
    a = a + b;
    return a;
}

// postfix operator ++
static inline Vec4i operator ++ (Vec4i & a, int) {
    Vec4i a0 = a;
    a = a + 1;
    return a0;
}

// prefix operator ++
static inline Vec4i & operator ++ (Vec4i & a) {
    a = a + 1;
    return a;
}

// vector operator - : subtract element by element
static inline Vec4i operator - (Vec4i const & a, Vec4i const & b) {
    return nsimd::sub(a, b);
}

// vector operator - : unary minus
static inline Vec4i operator - (Vec4i const & a) {
    return nsimd::sub(nsimd::set1<pack128_4i_t>(0), a);
}

// vector operator -= : subtract
static inline Vec4i & operator -= (Vec4i & a, Vec4i const & b) {
    a = a - b;
    return a;
}

// postfix operator --
static inline Vec4i operator -- (Vec4i & a, int) {
    Vec4i a0 = a;
    a = a - 1;
    return a0;
}

// prefix operator --
static inline Vec4i & operator -- (Vec4i & a) {
    a = a - 1;
    return a;
}

// vector operator * : multiply element by element
static inline Vec4i operator * (Vec4i const & a, Vec4i const & b) {
    return nsimd::mul(a, b);
}

// vector operator *= : multiply
static inline Vec4i & operator *= (Vec4i & a, Vec4i const & b) {
    a = a * b;
    return a;
}

// vector operator / : divide all elements by same integer
// See bottom of file


// vector operator << : shift left
static inline Vec4i operator << (Vec4i const & a, int32_t b) {
    return nsimd::shl(a, b);
}

// vector operator <<= : shift left
static inline Vec4i & operator <<= (Vec4i & a, int32_t b) {
    a = a << b;
    return a;
}

// vector operator >> : shift right arithmetic
static inline Vec4i operator >> (Vec4i const & a, int32_t b) {
    return nsimd::shr(a, b);
}

// vector operator >>= : shift right arithmetic
static inline Vec4i & operator >>= (Vec4i & a, int32_t b) {
    a = a >> b;
    return a;
}

// vector operator == : returns true for elements for which a == b
static inline Vec4ib operator == (Vec4i const & a, Vec4i const & b) {
    return nsimd::eq(a, b);
}

// vector operator != : returns true for elements for which a != b
static inline Vec4ib operator != (Vec4i const & a, Vec4i const & b) {
    return nsimd::ne(a, b);
}
  
// vector operator > : returns true for elements for which a > b
static inline Vec4ib operator > (Vec4i const & a, Vec4i const & b) {
    return nsimd::gt(a, b);
}

// vector operator < : returns true for elements for which a < b
static inline Vec4ib operator < (Vec4i const & a, Vec4i const & b) {
    return b > a;
}

// vector operator >= : returns true for elements for which a >= b (signed)
static inline Vec4ib operator >= (Vec4i const & a, Vec4i const & b) {
    return nsimd::ge(a, b);
}

// vector operator <= : returns true for elements for which a <= b (signed)
static inline Vec4ib operator <= (Vec4i const & a, Vec4i const & b) {
    return b >= a;
}

// vector operator & : bitwise and
static inline Vec4i operator & (Vec4i const & a, Vec4i const & b) {
    return nsimd::andb(a, b);
}
static inline Vec4i operator && (Vec4i const & a, Vec4i const & b) {
    return a & b;
}
// vector operator &= : bitwise and
static inline Vec4i & operator &= (Vec4i & a, Vec4i const & b) {
    a = a & b;
    return a;
}

// vector operator | : bitwise or
static inline Vec4i operator | (Vec4i const & a, Vec4i const & b) {
    return nsimd::orb;
}
static inline Vec4i operator || (Vec4i const & a, Vec4i const & b) {
    return a | b;
}
// vector operator |= : bitwise and
static inline Vec4i & operator |= (Vec4i & a, Vec4i const & b) {
    a = a | b;
    return a;
}

// vector operator ^ : bitwise xor
static inline Vec4i operator ^ (Vec4i const & a, Vec4i const & b) {
    return nsimd::xorb(a, b);
}
// vector operator ^= : bitwise and
static inline Vec4i & operator ^= (Vec4i & a, Vec4i const & b) {
    a = a ^ b;
    return a;
}

// vector operator ~ : bitwise not
static inline Vec4i operator ~ (Vec4i const & a) {
    return nsimd::notb(a);
}

// vector operator ! : returns true for elements == 0
static inline Vec4ib operator ! (Vec4i const & a) {
    return nsimd::eq(a, nsimd::set1<pack128_4i_t>(0));
}

// Functions for this class

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 4; i++) result[i] = s[i] ? a[i] : b[i];
// Each byte in s must be either 0 (false) or -1 (true). No other values are allowed.
// (s is signed)
static inline Vec4i select (Vec4ib const & s, Vec4i const & a, Vec4i const & b) {
    return selectb(s,a,b);
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
static inline Vec4i if_add (Vec4ib const & f, Vec4i const & a, Vec4i const & b) {
    return a + (Vec4i(f) & b);
}

// Horizontal add: Calculates the sum of all vector elements.
// Overflow will wrap around
static inline int32_t horizontal_add (Vec4i const & a) {
    return nsimd::addv(a);
}

// Horizontal add extended: Calculates the sum of all vector elements.
// Elements are sign extended before adding to avoid overflow
static inline int64_t horizontal_add_x (Vec4i const & a) {
    return nsimd::addv(a);
}

// function add_saturated: add element by element, signed with saturation
static inline Vec4i add_saturated(Vec4i const & a, Vec4i const & b) {
    return nsimd::adds(a, b);
}

// function sub_saturated: subtract element by element, signed with saturation
static inline Vec4i sub_saturated(Vec4i const & a, Vec4i const & b) {
    return nsimd::subs(a, b);
}

// function max: a > b ? a : b
static inline Vec4i max(Vec4i const & a, Vec4i const & b) {
    return nsimd::max(a, b);
}

// function min: a < b ? a : b
static inline Vec4i min(Vec4i const & a, Vec4i const & b) {
    return nsimd::min(a, b);
}

// function abs: a >= 0 ? a : -a
static inline Vec4i abs(Vec4i const & a) {
    return nsimd::abs(a);
}

// function abs_saturated: same as abs, saturate if overflow
static inline Vec4i abs_saturated(Vec4i const & a) {
    pack128_4i_t absa   = nsimd::abs(pack);
    return nsimd::adds(absa, nsimd::set1<pack128_4i_t>(int(0)));
}

// function rotate_left all elements
// Use negative count to rotate right
static inline Vec4i rotate_left(Vec4i const & a, int b) {
#ifdef __AVX512VL__
    return _mm_rolv_epi32(a, _mm_set1_epi32(b));
#elif defined __XOP__  // AMD XOP instruction set
    return _mm_rot_epi32(a,_mm_set1_epi32(b));
#else  // SSE2 instruction set
    __m128i left  = _mm_sll_epi32(a,_mm_cvtsi32_si128(b & 0x1F));      // a << b 
    __m128i right = _mm_srl_epi32(a,_mm_cvtsi32_si128((32-b) & 0x1F)); // a >> (32 - b)
    __m128i rot   = _mm_or_si128(left,right);                          // or
    return  rot;
#endif
}


/*****************************************************************************
*
*          Vector of 4 32-bit unsigned integers
*
*****************************************************************************/

class Vec4ui : public Vec4i {
protected:
    pack128_4ui_t xmm; // Integer vector
public:
    // Default constructor:
    Vec4ui() {
    }
    // Constructor to broadcast the same value into all elements:
    Vec4ui(uint32_t i) {
        xmm = nsimd::set1<pack128_4ui_t>(i);
    }
    // Constructor to build from all elements:
    Vec4ui(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3) {
        int32_t data[4] = {i0, i1, i2, i3};
        xmm = nsimd::loadu<pack128_4ui_t>(data);
    }
    // Constructor to convert from type __m128i used in intrinsics:
    Vec4ui(pack128_4ui_t const & x) {
        xmm = x;
    }
    // Assignment operator to convert from type __m128i used in intrinsics:
    Vec4ui & operator = (pack128_4ui_t const & x) {
        xmm = x;
        return *this;
    }
    // Member function to load from array (unaligned)
    Vec4ui & load(void const * p) {
        xmm = nsimd::loadu<pack128_4ui_t>((uint32_t const*)p);
        return *this;
    }
    // Member function to load from array (aligned)
    Vec4ui & load_a(void const * p) {
        xmm = nsimd::loada<pack128_4ui_t>((uint32_t const*)p);
        return *this;
    }
    // Member function to change a single element in vector
    // Note: This function is inefficient. Use load function if changing more than one element
    Vec4ui const & insert(uint32_t index, uint32_t value) {
        xmm = nsimd_common::set_bit<pack128_4i_t, int>(index, -uint32_t(value), xmm);
        return *this;
    }
    // Member function extract a single element from vector
    uint32_t extract(uint32_t index) const {
        return nsimd_common::get_bit<pack128_4i_t, int>(index, xmm);
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    uint32_t operator [] (uint32_t index) const {
        return extract(index);
    }
};

// Define operators for this class

// vector operator + : add
static inline Vec4ui operator + (Vec4ui const & a, Vec4ui const & b) {
    return nsimd::add(a, b);
}

// vector operator - : subtract
static inline Vec4ui operator - (Vec4ui const & a, Vec4ui const & b) {
    return nsimd::sub(a, b);
}

// vector operator * : multiply
static inline Vec4ui operator * (Vec4ui const & a, Vec4ui const & b) {
    return nsimd::mul(a, b);
}

// vector operator / : divide
// See bottom of file

// vector operator >> : shift right logical all elements
static inline Vec4ui operator >> (Vec4ui const & a, uint32_t b) {
    return nsimd::shr(a, b); 
}

// vector operator >> : shift right logical all elements
static inline Vec4ui operator >> (Vec4ui const & a, int32_t b) {
    return a >> (uint32_t)b;
}

// vector operator >>= : shift right logical
static inline Vec4ui & operator >>= (Vec4ui & a, int b) {
    a = a >> b;
    return a;
}

// vector operator << : shift left all elements
static inline Vec4ui operator << (Vec4ui const & a, uint32_t b) {
    return nsimd::shl(a, b); 
}

// vector operator << : shift left all elements
static inline Vec4ui operator << (Vec4ui const & a, int32_t b) {
    return nsimd::shl(a, b);
}

// vector operator > : returns true for elements for which a > b (unsigned)
static inline Vec4ib operator > (Vec4ui const & a, Vec4ui const & b) {
    return nsimd::gt(a, b);
}

// vector operator < : returns true for elements for which a < b (unsigned)
static inline Vec4ib operator < (Vec4ui const & a, Vec4ui const & b) {
    return b > a;
}

// vector operator >= : returns true for elements for which a >= b (unsigned)
static inline Vec4ib operator >= (Vec4ui const & a, Vec4ui const & b) {
    return nsimd::ge(a, b);
}

// vector operator <= : returns true for elements for which a <= b (unsigned)
static inline Vec4ib operator <= (Vec4ui const & a, Vec4ui const & b) {
    return b >= a;
}

// vector operator & : bitwise and
static inline Vec4ui operator & (Vec4ui const & a, Vec4ui const & b) {
    return nsimd::andb(a, b);
}
static inline Vec4ui operator && (Vec4ui const & a, Vec4ui const & b) {
    return a & b;
}

// vector operator | : bitwise or
static inline Vec4ui operator | (Vec4ui const & a, Vec4ui const & b) {
    return nsimd::orb(a, b);
}
static inline Vec4ui operator || (Vec4ui const & a, Vec4ui const & b) {
    return a | b;
}

// vector operator ^ : bitwise xor
static inline Vec4ui operator ^ (Vec4ui const & a, Vec4ui const & b) {
    return nsimd::xorb(a, b);
}

// vector operator ~ : bitwise not
static inline Vec4ui operator ~ (Vec4ui const & a) {
    return nsimd::notb(a);
}

// Functions for this class

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 8; i++) result[i] = s[i] ? a[i] : b[i];
// Each word in s must be either 0 (false) or -1 (true). No other values are allowed.
// (s is signed)
static inline Vec4ui select (Vec4ib const & s, Vec4ui const & a, Vec4ui const & b) {
    return selectb(s,a,b);
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
static inline Vec4ui if_add (Vec4ib const & f, Vec4ui const & a, Vec4ui const & b) {
    return a + (Vec4ui(f) & b);
}

// Horizontal add: Calculates the sum of all vector elements.
// Overflow will wrap around
static inline uint32_t horizontal_add (Vec4ui const & a) {
    return nsimd::addv(a);
}

// Horizontal add extended: Calculates the sum of all vector elements.
// Elements are zero extended before adding to avoid overflow
static inline uint64_t horizontal_add_x (Vec4ui const & a) {
    return nsimd::addv(a);
}

// function add_saturated: add element by element, unsigned with saturation
static inline Vec4ui add_saturated(Vec4ui const & a, Vec4ui const & b) {
    return nsimd::adds(a, b);    
}

// function sub_saturated: subtract element by element, unsigned with saturation
static inline Vec4ui sub_saturated(Vec4ui const & a, Vec4ui const & b) {
   return nsimd::adds(a, b); 
}

// function max: a > b ? a : b
static inline Vec4ui max(Vec4ui const & a, Vec4ui const & b) {
    return nsimd::max(a, b);
}

// function min: a < b ? a : b
static inline Vec4ui min(Vec4ui const & a, Vec4ui const & b) {
    return nsimd::min(a, b);
}


/*****************************************************************************
*
*          Vector of 2 64-bit signed integers
*
*****************************************************************************/

class Vec2q : public Vec128b {
protected:
    pack128_2i_t xmm; // Integer vector
public:
    // Default constructor:
    Vec2q() {
    }
    // Constructor to broadcast the same value into all elements:
    Vec2q(int64_t i) {
        xmm = nsimd::set1<pack128_2i_t>(i);
    }
    // Constructor to build from all elements:
    Vec2q(int64_t i0, int64_t i1) {
        int64_t data[2] = {i0, i1};
        xmm = nsimd::loadu<pack128_2i_t>(data);
    }
    // Constructor to convert from type __m128i used in intrinsics:
    Vec2q(pack128_2i_t const & x) {
        xmm = x;
    }
    // Assignment operator to convert from type __m128i used in intrinsics:
    Vec2q & operator = (pack128_2i_t const & x) {
        xmm = x;
        return *this;
    }
    // Type cast operator to convert to __m128i used in intrinsics
    operator pack128_2i_t() const {
        return xmm;
    }
    // Member function to load from array (unaligned)
    Vec2q & load(void const * p) {
        xmm = nsimd::loadu<pack128_2i_t>((int64_t const*)p);
        return *this;
    }
    // Member function to load from array (aligned)
    Vec2q & load_a(void const * p) {
        xmm = nsimd::loada<pack128_2i_t>((int64_t const*)p);
        return *this;
    }
    // Partial load. Load n elements and set the rest to 0
    Vec2q & load_partial(int n, void const * p) {
        xmm = nsimd_common::load_partial<pack128_2i_t, packl128_2i_t, long>(p, n)
        return *this;
    }
    // Partial store. Store n elements
    void store_partial(int n, void * p) const {
        nsimd_common::store_partial<pack128_2i_t, packl128_2i_t, long>(p, n, xmm);
    }
    // cut off vector to n elements. The last 2-n elements are set to zero
    Vec2q & cutoff(int n) {
        xmm = nsimd_common::cutoff<pack128_2i_t, packl128_4i_t>(xmm, n);
        return *this;
    }
    // Member function to change a single element in vector
    // Note: This function is inefficient. Use load function if changing more than one element
    Vec2q const & insert(uint32_t index, int64_t value) {
        xmm = nsimd_common::set_bit<pack128_2i_t, int64_t>(index, value, xmm)
        return *this;
    }
    // Member function extract a single element from vector
    int64_t extract(uint32_t index) const {
        int64_t x[2];
        store(x);
        return x[index & 1];
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    int64_t operator [] (uint32_t index) const {
        return extract(index);
    }
    static int size() {
        return 2;
    }
};

/*****************************************************************************
*
*          Vec2qb: Vector of 2 Booleans for use with Vec2q and Vec2uq
*
*****************************************************************************/
// Definition will be different for the AVX512 instruction set
class Vec2qb : public Vec2q {
protected:
    packl128_2i_t xmm; // Integer vector
public:
    // Default constructor:
    Vec2qb() {
    }
    // Constructor to build from all elements:
    Vec2qb(bool x0, bool x1) {
        int64_t vec[2] = {-int64_t(x0), -int64_t(x1)};
        xmm = nsimd::loadla<packl128_2i_t>(vec);
    }
    // Constructor to convert from type __m128i used in intrinsics:
    Vec2qb(packl128_2i_t const & x) {
        xmm = x;
    }
    // Assignment operator to convert from type __m128i used in intrinsics:
    Vec2qb & operator = (packl128_2i_t const & x) {
        xmm = x;
        return *this;
    }
    // Constructor to broadcast scalar value:
    Vec2qb(bool b) {
        xmm = nsimd::set1<packl128_2i_t>(-int64_t(b));
    }
    // Assignment operator to broadcast scalar value:
    Vec2qb & operator = (bool b) {
        *this = Vec2qb(b);
        return *this;
    }
private: // Prevent constructing from int, etc.
    Vec2qb(int b);
    Vec2qb & operator = (int x);
public:
    Vec2qb & insert (int index, bool a) {
        xmm = nsimd_common::set_bit<packl128_2i_t, int64_t>(index, -int64_t(a), xmm);
        return *this;
    }    
    // Member function extract a single element from vector
    bool extract(uint32_t index) const {
        return nsimd_common::get_bit<packl128_2i_t, int64_t>(index, xmm) != 0;
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    bool operator [] (uint32_t index) const {
        return extract(index);
    }
    // Type cast operator to convert
    operator packl128_2i_t() const {
        return xmm;
    }
};


/*****************************************************************************
*
*          Define operators for Vec2qb
*
*****************************************************************************/

// vector operator & : bitwise and
static inline Vec2qb operator & (Vec2qb const & a, Vec2qb const & b) {
    return nsimd::andl(a, b);
}
static inline Vec2qb operator && (Vec2qb const & a, Vec2qb const & b) {
    return a & b;
}
// vector operator &= : bitwise and
static inline Vec2qb & operator &= (Vec2qb & a, Vec2qb const & b) {
    a = a & b;
    return a;
}

// vector operator | : bitwise or
static inline Vec2qb operator | (Vec2qb const & a, Vec2qb const & b) {
    return nsimd::orl(a, b);
}
static inline Vec2qb operator || (Vec2qb const & a, Vec2qb const & b) {
    return a | b;
}
// vector operator |= : bitwise or
static inline Vec2qb & operator |= (Vec2qb & a, Vec2qb const & b) {
    a = a | b;
    return a;
}

// vector operator ^ : bitwise xor
static inline Vec2qb operator ^ (Vec2qb const & a, Vec2qb const & b) {
    return nsimd::xorl(a, b);
}
// vector operator ^= : bitwise xor
static inline Vec2qb & operator ^= (Vec2qb & a, Vec2qb const & b) {
    a = a ^ b;
    return a;
}

// vector operator ~ : bitwise not
static inline Vec2qb operator ~ (Vec2qb const & a) {
    return nsimd::notl(a);
}

// vector operator ! : element not
static inline Vec2qb operator ! (Vec2qb const & a) {
    return ~ a;
}

// vector function andnot
static inline Vec2qb andnot (Vec2qb const & a, Vec2qb const & b) {
    return nsimd:andnotl(a, b);
}

// Horizontal Boolean functions for Vec2qb

// horizontal_and. Returns true if all elements are true
static inline bool horizontal_and(Vec2qb const & a) {
    return nsimd::all(a);
}

// horizontal_or. Returns true if at least one element is true
static inline bool horizontal_or(Vec2qb const & a) {
    return nsimd::any(a);
} 


/*****************************************************************************
*
*          Operators for Vec2q
*
*****************************************************************************/

// vector operator + : add element by element
static inline Vec2q operator + (Vec2q const & a, Vec2q const & b) {
    return nsimd::add(a, b);
}

// vector operator += : add
static inline Vec2q & operator += (Vec2q & a, Vec2q const & b) {
    a = a + b;
    return a;
}

// postfix operator ++
static inline Vec2q operator ++ (Vec2q & a, int) {
    Vec2q a0 = a;
    a = a + 1;
    return a0;
}

// prefix operator ++
static inline Vec2q & operator ++ (Vec2q & a) {
    a = a + 1;
    return a;
}

// vector operator - : subtract element by element
static inline Vec2q operator - (Vec2q const & a, Vec2q const & b) {
    return nsimd::sub(a, b);
}

// vector operator - : unary minus
static inline Vec2q operator - (Vec2q const & a) {
    return nsimd::sub(nsimd::set1<pack128_2i_t>(0.0), a);
}

// vector operator -= : subtract
static inline Vec2q & operator -= (Vec2q & a, Vec2q const & b) {
    a = a - b;
    return a;
}

// postfix operator --
static inline Vec2q operator -- (Vec2q & a, int) {
    Vec2q a0 = a;
    a = a - 1;
    return a0;
}

// prefix operator --
static inline Vec2q & operator -- (Vec2q & a) {
    a = a - 1;
    return a;
}

// vector operator * : multiply element by element
static inline Vec2q operator * (Vec2q const & a, Vec2q const & b) {
    return nsimd::mul(a, b);
}

// vector operator *= : multiply
static inline Vec2q & operator *= (Vec2q & a, Vec2q const & b) {
    a = a * b;
    return a;
}

// vector operator << : shift left
static inline Vec2q operator << (Vec2q const & a, int32_t b) {
    return nsimd::shl(a, b);
}

// vector operator <<= : shift left
static inline Vec2q & operator <<= (Vec2q & a, int32_t b) {
    a = a << b;
    return a;
}

// vector operator >> : shift right arithmetic
static inline Vec2q operator >> (Vec2q const & a, int32_t b) {
    return nsimd::shr(a, b);
}

// vector operator >>= : shift right arithmetic
static inline Vec2q & operator >>= (Vec2q & a, int32_t b) {
    a = a >> b;
    return a;
}

// vector operator == : returns true for elements for which a == b
static inline Vec2qb operator == (Vec2q const & a, Vec2q const & b) {
    return nsimd::eq(a, b);
}

// vector operator != : returns true for elements for which a != b
static inline Vec2qb operator != (Vec2q const & a, Vec2q const & b) {
    return nsimd::ne(a, b);
}
  
// vector operator < : returns true for elements for which a < b
static inline Vec2qb operator < (Vec2q const & a, Vec2q const & b) {
    return nsimd::lt(a, b);
}

// vector operator > : returns true for elements for which a > b
static inline Vec2qb operator > (Vec2q const & a, Vec2q const & b) {
    return b < a;
}

// vector operator >= : returns true for elements for which a >= b (signed)
static inline Vec2qb operator >= (Vec2q const & a, Vec2q const & b) {
    return nsimd::ge(a, b);
}

// vector operator <= : returns true for elements for which a <= b (signed)
static inline Vec2qb operator <= (Vec2q const & a, Vec2q const & b) {
    return b >= a;
}

// vector operator & : bitwise and
static inline Vec2q operator & (Vec2q const & a, Vec2q const & b) {
    return nsimd::andb(a, b);
}
static inline Vec2q operator && (Vec2q const & a, Vec2q const & b) {
    return a & b;
}
// vector operator &= : bitwise and
static inline Vec2q & operator &= (Vec2q & a, Vec2q const & b) {
    a = a & b;
    return a;
}

// vector operator | : bitwise or
static inline Vec2q operator | (Vec2q const & a, Vec2q const & b) {
    return nsimd::orb(a, b);
}
static inline Vec2q operator || (Vec2q const & a, Vec2q const & b) {
    return a | b;
}
// vector operator |= : bitwise or
static inline Vec2q & operator |= (Vec2q & a, Vec2q const & b) {
    a = a | b;
    return a;
}

// vector operator ^ : bitwise xor
static inline Vec2q operator ^ (Vec2q const & a, Vec2q const & b) {
    return nsimd::xorb(a, b);
}
// vector operator ^= : bitwise xor
static inline Vec2q & operator ^= (Vec2q & a, Vec2q const & b) {
    a = a ^ b;
    return a;
}

// vector operator ~ : bitwise not
static inline Vec2q operator ~ (Vec2q const & a) {
    return nsimd::notb(a);
}

// vector operator ! : logical not, returns true for elements == 0
static inline Vec2qb operator ! (Vec2q const & a) {
    return nsimd::eq(a, nsimd::set1<pack128_2i_t>(0.0));
}

// Functions for this class

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 8; i++) result[i] = s[i] ? a[i] : b[i];
// Each byte in s must be either 0 (false) or -1 (true). No other values are allowed.
// (s is signed)
static inline Vec2q select (Vec2qb const & s, Vec2q const & a, Vec2q const & b) {
    return selectb(s,a,b);
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
static inline Vec2q if_add (Vec2qb const & f, Vec2q const & a, Vec2q const & b) {
    return a + (Vec2q(f) & b);
}

// Horizontal add: Calculates the sum of all vector elements.
// Overflow will wrap around
static inline int64_t horizontal_add (Vec2q const & a) {
    return nsimd::addv(a);
}

// function max: a > b ? a : b
static inline Vec2q max(Vec2q const & a, Vec2q const & b) {
    return nsimd::max(a, b);
}

// function min: a < b ? a : b
static inline Vec2q min(Vec2q const & a, Vec2q const & b) {
    return nsimd::min(a, b);
}

// function abs: a >= 0 ? a : -a
static inline Vec2q abs(Vec2q const & a) {
    return nsimd::abs(a);
}

// function abs_saturated: same as abs, saturate if overflow
static inline Vec2q abs_saturated(Vec2q const & a) {
    pack128_2i_t absa   = nsimd::abs(pack);
    return nsimd::adds(absa, nsimd::set1<pack128_2i_t>(0.0));
}

// function rotate_left all elements
// Use negative count to rotate right
static inline Vec2q rotate_left(Vec2q const & a, int b) {
#ifdef __AVX512VL__
    return _mm_rolv_epi64(a, _mm_set1_epi64x(int64_t(b)));
#elif defined __XOP__  // AMD XOP instruction set
    return (Vec2q)_mm_rot_epi64(a,Vec2q(b));
#else  // SSE2 instruction set
    __m128i left  = _mm_sll_epi64(a,_mm_cvtsi32_si128(b & 0x3F));      // a << b 
    __m128i right = _mm_srl_epi64(a,_mm_cvtsi32_si128((64-b) & 0x3F)); // a >> (64 - b)
    __m128i rot   = _mm_or_si128(left,right);                          // or
    return  (Vec2q)rot;
#endif
}


/*****************************************************************************
*
*          Vector of 2 64-bit unsigned integers
*
*****************************************************************************/

class Vec2uq : public Vec2q {
protected:
    pack128_2ui_t xmm; // Integer vector
public:
    // Default constructor:
    Vec2uq() {
    }
    // Constructor to broadcast the same value into all elements:
    Vec2uq(uint64_t i) {
        xmm = nsimd::set1<pack128_2ui_t>(i);
    }
    // Constructor to build from all elements:
    Vec2uq(uint64_t i0, uint64_t i1) {
        uint64_t data[2] = {i0, i1};
        xmm = nsimd::loadu<pack128_2ui_t>(data);
    }
    // Constructor to convert from type __m128i used in intrinsics:
    Vec2uq(pack128_2ui_t const & x) {
        xmm = x;
    }
    // Assignment operator to convert from type __m128i used in intrinsics:
    Vec2uq & operator = (pack128_2ui_t const & x) {
        xmm = x;
        return *this;
    }
    // Member function to load from array (unaligned)
    Vec2uq & load(void const * p) {
        xmm = nsimd::loadu<pack128_2ui_t>((uint64_t const*)p);
        return *this;
    }
    // Member function to load from array (aligned)
    Vec2uq & load_a(void const * p) {
        xmm = nsimd::loada<pack128_2ui_t>((uint64_t const*)p);
        return *this;
    }
    // Member function to change a single element in vector
    // Note: This function is inefficient. Use load function if changing more than one element
    Vec2uq const & insert(uint32_t index, uint64_t value) {
        xmm = nsimd_common::set_bit<pack128_2ui_t, uint64_t>(index, -value, xmm);
        return *this;
    }
    // Member function extract a single element from vector
    uint64_t extract(uint32_t index) const {
        return nsimd_common::get_bit<pack128_2ui_t, uint64_t>(index, xmm);
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    uint64_t operator [] (uint32_t index) const {
        return extract(index);
    }
};

// Define operators for this class

// vector operator + : add
static inline Vec2uq operator + (Vec2uq const & a, Vec2uq const & b) {
    return nsimd::add(a, b);
}

// vector operator - : subtract
static inline Vec2uq operator - (Vec2uq const & a, Vec2uq const & b) {
    return nsimd::sub(a, b);
}

// vector operator * : multiply element by element
static inline Vec2uq operator * (Vec2uq const & a, Vec2uq const & b) {
    return nsimd::mul(a, b);
}

// vector operator >> : shift right logical all elements
static inline Vec2uq operator >> (Vec2uq const & a, uint32_t b) {
    return nsimd::shr(a, b); 
}

// vector operator >> : shift right logical all elements
static inline Vec2uq operator >> (Vec2uq const & a, int32_t b) {
    return a >> (uint32_t)b;
}

// vector operator >>= : shift right logical
static inline Vec2uq & operator >>= (Vec2uq & a, int b) {
    a = a >> b;
    return a;
}

// vector operator << : shift left all elements
static inline Vec2uq operator << (Vec2uq const & a, uint32_t b) {
    return nsimd::shl(a, b);
}

// vector operator << : shift left all elements
static inline Vec2uq operator << (Vec2uq const & a, int32_t b) {
    return a << (uint32_t)b;
}

// vector operator > : returns true for elements for which a > b (unsigned)
static inline Vec2qb operator > (Vec2uq const & a, Vec2uq const & b) {
    return nsimd::gt(a, b);
}

// vector operator < : returns true for elements for which a < b (unsigned)
static inline Vec2qb operator < (Vec2uq const & a, Vec2uq const & b) {
    return nsimd::lt(a, b);
}

// vector operator >= : returns true for elements for which a >= b (unsigned)
static inline Vec2qb operator >= (Vec2uq const & a, Vec2uq const & b) {
    return nsimd::ge(a, b);
}

// vector operator <= : returns true for elements for which a <= b (unsigned)
static inline Vec2qb operator <= (Vec2uq const & a, Vec2uq const & b) {
    return nsimd::le(a, b);
}

// vector operator & : bitwise and
static inline Vec2uq operator & (Vec2uq const & a, Vec2uq const & b) {
    return nsimd::andb(a, b);
}
static inline Vec2uq operator && (Vec2uq const & a, Vec2uq const & b) {
    return a & b;
}

// vector operator | : bitwise or
static inline Vec2uq operator | (Vec2uq const & a, Vec2uq const & b) {
    return nsimd::orb(a, b);
}
static inline Vec2uq operator || (Vec2uq const & a, Vec2uq const & b) {
    return a | b;
}

// vector operator ^ : bitwise xor
static inline Vec2uq operator ^ (Vec2uq const & a, Vec2uq const & b) {
    return nsimd::xorb(a, b);
}

// vector operator ~ : bitwise not
static inline Vec2uq operator ~ (Vec2uq const & a) {
    return nsimd::orb(a);
}


// Functions for this class

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 2; i++) result[i] = s[i] ? a[i] : b[i];
// Each word in s must be either 0 (false) or -1 (true). No other values are allowed.
// (s is signed)
static inline Vec2uq select (Vec2qb const & s, Vec2uq const & a, Vec2uq const & b) {
    return selectb(s,a,b);
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
static inline Vec2uq if_add (Vec2qb const & f, Vec2uq const & a, Vec2uq const & b) {
    return a + (Vec2uq(f) & b);
}

// Horizontal add: Calculates the sum of all vector elements.
// Overflow will wrap around
static inline uint64_t horizontal_add (Vec2uq const & a) {
    return nsimd::addv(a);
}

// function max: a > b ? a : b
static inline Vec2uq max(Vec2uq const & a, Vec2uq const & b) {
    return nsimd::max(a, b);
}

// function min: a < b ? a : b
static inline Vec2uq min(Vec2uq const & a, Vec2uq const & b) {
    return nsimd::min(a, b);
}


/*****************************************************************************
*
*          Vector permute functions
*
******************************************************************************
*
* These permute functions can reorder the elements of a vector and optionally
* set some elements to zero. 
*
* The indexes are inserted as template parameters in <>. These indexes must be
* constants. Each template parameter is an index to the element you want to 
* select. A negative index will generate zero. an index of -256 means don't care.
*
* Example:
* Vec4i a(10,11,12,13);         // a is (10,11,12,13)
* Vec4i b, c;
* b = permute4i<0,0,2,2>(a);    // b is (10,10,12,12)
* c = permute4i<3,2,-1,-1>(a);  // c is (13,12, 0, 0)
*
* The permute functions for vectors of 8-bit integers are inefficient if 
* the SSSE3 instruction set or later is not enabled.
*
* A lot of the code here is metaprogramming aiming to find the instructions
* that best fit the template parameters and instruction set. The metacode
* will be reduced out to leave only a few vector instructions in release
* mode with optimization on.
*****************************************************************************/

template <int i0, int i1>
static inline Vec2q permute2q(Vec2q const & a) {
    if (i0 == 0) {
        if (i1 == 0) {       // 0,0
            return _mm_unpacklo_epi64(a, a);
        }
        else if (i1 == 1 || i1 == -0x100) {  // 0,1
            return a;
        }
        else {               // 0,-1
            // return _mm_mov_epi64(a); // doesn't work with MS VS 2008
            return _mm_and_si128(a, constant4i<-1,-1,0,0>());
        }
    }
    else if (i0 == 1) {
        if (i1 == 0) {       // 1,0
            return _mm_shuffle_epi32(a, 0x4E);
        }
        else if (i1 == 1) {  // 1,1
            return _mm_unpackhi_epi64(a, a);
        }
        else {               // 1,-1
            return _mm_srli_si128(a, 8);
        }
    }
    else { // i0 < 0
        if (i1 == 0) {       // -1,0
            return _mm_slli_si128(a, 8);
        }
        else if (i1 == 1) {  // -1,1
            if (i0 == -0x100) return a;
            return _mm_and_si128(a, constant4i<0,0,-1,-1>());
        }
        else {               // -1,-1
            return _mm_setzero_si128();
        }
    }
}

template <int i0, int i1>
static inline Vec2uq permute2uq(Vec2uq const & a) {
    return Vec2uq (permute2q <i0, i1> ((__m128i)a));
}

// permute vector Vec4i
template <int i0, int i1, int i2, int i3>
static inline Vec4i permute4i(Vec4i const & a) {

    // Combine all the indexes into a single bitfield, with 4 bits for each
    const uint32_t m1 = (i0&3) | (i1&3)<<4 | (i2&3)<<8 | (i3&3)<<12; 

    // Mask to zero out negative indexes
    const uint32_t mz = (i0<0?0:0xF) | (i1<0?0:0xF)<<4 | (i2<0?0:0xF)<<8 | (i3<0?0:0xF)<<12;

    // Mask indicating required zeroing of all indexes, with 4 bits for each, 0 for index = -1, 0xF for index >= 0 or -256
    const uint32_t ssz = ((i0 & 0x80) ? 0 : 0xF) | ((i1 & 0x80) ? 0 : 0xF) << 4 | ((i2 & 0x80) ? 0 : 0xF) << 8 | ((i3 & 0x80) ? 0 : 0xF) << 12;

    // Mask indicating 0 for don't care, 0xF for non-negative value of required zeroing
    const uint32_t md = mz | ~ ssz;

    // Test if permutation needed
    const bool do_shuffle = ((m1 ^ 0x00003210) & mz) != 0;

    // is zeroing needed
    const bool do_zero    = (ssz != 0xFFFF);

    if (mz == 0) {
        return _mm_setzero_si128();    // special case: all zero or don't care
    }
    // Test if we can do with 64-bit permute only
    if ((m1 & 0x0101 & mz) == 0        // even indexes are even or negative
    && (~m1 & 0x1010 & mz) == 0        // odd  indexes are odd  or negative
    && ((m1 ^ ((m1 + 0x0101) << 4)) & 0xF0F0 & mz & (mz << 4)) == 0  // odd index == preceding even index +1 or at least one of them negative
    && ((mz ^ (mz << 4)) & 0xF0F0 & md & md << 4) == 0) {      // each pair of indexes are both negative or both positive or one of them don't care
        const int j0 = i0 >= 0 ? i0 / 2 : (i0 & 0x80) ? i0 : i1 >= 0 ? i1/2 : i1;
        const int j1 = i2 >= 0 ? i2 / 2 : (i2 & 0x80) ? i2 : i3 >= 0 ? i3/2 : i3;
        return Vec4i(permute2q<j0, j1> (Vec2q(a)));    // 64 bit permute
    }
#if  INSTRSET >= 4  // SSSE3
    if (do_shuffle && do_zero) {
        // With SSSE3 we can do both with the PSHUFB instruction
        const int j0 = (i0 & 3) << 2;
        const int j1 = (i1 & 3) << 2;
        const int j2 = (i2 & 3) << 2;
        const int j3 = (i3 & 3) << 2;
        __m128i mask1 = constant4i <
            i0 < 0 ? -1 : j0 | (j0+1)<<8 | (j0+2)<<16 | (j0+3) << 24,
            i1 < 0 ? -1 : j1 | (j1+1)<<8 | (j1+2)<<16 | (j1+3) << 24,
            i2 < 0 ? -1 : j2 | (j2+1)<<8 | (j2+2)<<16 | (j2+3) << 24,
            i3 < 0 ? -1 : j3 | (j3+1)<<8 | (j3+2)<<16 | (j3+3) << 24 > ();
        return _mm_shuffle_epi8(a,mask1);
    }
#endif
    __m128i t1;

    if (do_shuffle) {  // permute
        t1 = _mm_shuffle_epi32(a, (i0&3) | (i1&3)<<2 | (i2&3)<<4 | (i3&3)<<6);
    }
    else {
        t1 = a;
    }
    if (do_zero) {     // set some elements to zero
        __m128i mask2 = constant4i< -int(i0>=0), -int(i1>=0), -int(i2>=0), -int(i3>=0) >();
        t1 = _mm_and_si128(t1,mask2);
    }
    return t1;
}

template <int i0, int i1, int i2, int i3>
static inline Vec4ui permute4ui(Vec4ui const & a) {
    return Vec4ui (permute4i <i0,i1,i2,i3> (a));
}

template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
static inline Vec8s permute8s(Vec8s const & a) {
    if ((i0 & i1 & i2 & i3 & i4 & i5 & i6 & i7) < 0) {
        return _mm_setzero_si128();  // special case: all zero
    }
#if  INSTRSET >= 4  // SSSE3

    // special case: rotate
    if (i0>=0 && i0 < 8 && i1==((i0+1)&7) && i2==((i0+2)&7) && i3==((i0+3)&7) && i4==((i0+4)&7) && i5==((i0+5)&7) && i6==((i0+6)&7) && i7==((i0+7)&7)) {
        if (i0 == 0) return a;  // do nothing
        return _mm_alignr_epi8(a, a, (i0 & 7) * 2);
    }    
    
    // General case: Use PSHUFB
    const int j0 = i0 < 0 ? 0xFFFF : ( (i0 & 7) * 2 | ((i0 & 7) * 2 + 1) << 8 );
    const int j1 = i1 < 0 ? 0xFFFF : ( (i1 & 7) * 2 | ((i1 & 7) * 2 + 1) << 8 );
    const int j2 = i2 < 0 ? 0xFFFF : ( (i2 & 7) * 2 | ((i2 & 7) * 2 + 1) << 8 );
    const int j3 = i3 < 0 ? 0xFFFF : ( (i3 & 7) * 2 | ((i3 & 7) * 2 + 1) << 8 );
    const int j4 = i4 < 0 ? 0xFFFF : ( (i4 & 7) * 2 | ((i4 & 7) * 2 + 1) << 8 );
    const int j5 = i5 < 0 ? 0xFFFF : ( (i5 & 7) * 2 | ((i5 & 7) * 2 + 1) << 8 );
    const int j6 = i6 < 0 ? 0xFFFF : ( (i6 & 7) * 2 | ((i6 & 7) * 2 + 1) << 8 );
    const int j7 = i7 < 0 ? 0xFFFF : ( (i7 & 7) * 2 | ((i7 & 7) * 2 + 1) << 8 );
    __m128i mask = constant4i < j0 | j1 << 16, j2 | j3 << 16, j4 | j5 << 16, j6 | j7 << 16 > ();
    return _mm_shuffle_epi8(a,mask);

#else   // SSE2 has no simple solution. Find the optimal permute method.
    // Without proper metaprogramming features, we have to use constant expressions 
    // and if-statements to make sure these calculations are resolved at compile time.
    // All this should produce at most 8 instructions in the final code, depending
    // on the template parameters.

    // Temporary vectors
    __m128i t1, t2, t3, t4, t5, t6, t7;

    // Combine all the indexes into a single bitfield, with 4 bits for each
    const int m1 = (i0&7) | (i1&7)<<4 | (i2&7)<<8 | (i3&7)<<12 
        | (i4&7)<<16 | (i5&7)<<20 | (i6&7)<<24 | (i7&7)<<28; 

    // Mask to zero out negative indexes
    const int m2 = (i0<0?0:0xF) | (i1<0?0:0xF)<<4 | (i2<0?0:0xF)<<8 | (i3<0?0:0xF)<<12
        | (i4<0?0:0xF)<<16 | (i5<0?0:0xF)<<20 | (i6<0?0:0xF)<<24 | (i7<0?0:0xF)<<28;

    // Test if we can do without permute
    const bool case0 = ((m1 ^ 0x76543210) & m2) == 0; // all indexes point to their own place or negative

    // Test if we can do with 32-bit permute only
    const bool case1 = 
        (m1 & 0x01010101 & m2) == 0        // even indexes are even or negative
        && (~m1 & 0x10101010 & m2) == 0    // odd  indexes are odd  or negative
        && ((m1 ^ ((m1 + 0x01010101) << 4)) & 0xF0F0F0F0 & m2 & (m2 << 4)) == 0; // odd index == preceding even index +1 or at least one of them negative

    // Test if we can do with 16-bit permute only
    const bool case2 = 
        (((m1 & 0x44444444) ^ 0x44440000) & m2) == 0;  // indexes 0-3 point to lower 64 bits, 1-7 to higher 64 bits, or negative

    if (case0) {
        // no permute needed
        t7 = a;
    }
    else if (case1) {
        // 32 bit permute only
        const int j0 = i0 >= 0 ? i0/2 : i1 >= 0 ? i1/2 : 0;
        const int j1 = i2 >= 0 ? i2/2 : i3 >= 0 ? i3/2 : 0;
        const int j2 = i4 >= 0 ? i4/2 : i5 >= 0 ? i5/2 : 0;
        const int j3 = i6 >= 0 ? i6/2 : i7 >= 0 ? i7/2 : 0;
        t7 = _mm_shuffle_epi32(a, (j0&3) | (j1&3)<<2 | (j2&3)<<4 | (j3&3)<<6 );
    }
    else if (case2) {
        // 16 bit permute only
        const int j0 = i0 >= 0 ? i0&3 : 0;
        const int j1 = i1 >= 0 ? i1&3 : 1;
        const int j2 = i2 >= 0 ? i2&3 : 2;
        const int j3 = i3 >= 0 ? i3&3 : 3;
        const int j4 = i4 >= 0 ? i4&3 : 0;
        const int j5 = i5 >= 0 ? i5&3 : 1;
        const int j6 = i6 >= 0 ? i6&3 : 2;
        const int j7 = i7 >= 0 ? i7&3 : 3;
        if (j0!=0 || j1!=1 || j2!=2 || j3!=3) {            
            t1 = _mm_shufflelo_epi16(a, j0 | j1<<2 | j2<<4 | j3<<6);
        }
        else t1 = a;
        if (j4!=0 || j5!=1 || j6!=2 || j7!=3) {            
            t7 = _mm_shufflehi_epi16(t1, j4 | j5<<2 | j6<<4 | j7<<6);
        }
        else t7 = t1;
    }
    else {
        // Need at least two permute steps

        // Index to where each dword of a is needed
        const int nn = (m1 & 0x66666666) | 0x88888888; // indicate which dwords are needed
        const int n0 = ((((uint32_t)(nn ^ 0x00000000) - 0x22222222) & 0x88888888) ^ 0x88888888) & m2;
        const int n1 = ((((uint32_t)(nn ^ 0x22222222) - 0x22222222) & 0x88888888) ^ 0x88888888) & m2;
        const int n2 = ((((uint32_t)(nn ^ 0x44444444) - 0x22222222) & 0x88888888) ^ 0x88888888) & m2;
        const int n3 = ((((uint32_t)(nn ^ 0x66666666) - 0x22222222) & 0x88888888) ^ 0x88888888) & m2;
        // indicate which dwords are needed in low half
        const int l0 = (n0 & 0xFFFF) != 0;
        const int l1 = (n1 & 0xFFFF) != 0;
        const int l2 = (n2 & 0xFFFF) != 0;
        const int l3 = (n3 & 0xFFFF) != 0;
        // indicate which dwords are needed in high half
        const int h0 = (n0 & 0xFFFF0000) != 0;
        const int h1 = (n1 & 0xFFFF0000) != 0;
        const int h2 = (n2 & 0xFFFF0000) != 0;
        const int h3 = (n3 & 0xFFFF0000) != 0;

        // Test if we can do with two permute steps
        const bool case3 = l0 + l1 + l2 + l3 <= 2  &&  h0 + h1 + h2 + h3 <= 2;

        if (case3) {
            // one 32-bit permute followed by one 16-bit permute in each half.
            // Find permute indices for 32-bit permute
            const int j0 = l0 ? 0 : l1 ? 1 : l2 ? 2 : 3;
            const int j1 = l3 ? 3 : l2 ? 2 : l1 ? 1 : 0;
            const int j2 = h0 ? 0 : h1 ? 1 : h2 ? 2 : 3;
            const int j3 = h3 ? 3 : h2 ? 2 : h1 ? 1 : 0;

            // Find permute indices for low 16-bit permute
            const int r0 = i0 < 0 ? 0 : (i0>>1 == j0 ? 0 : 2) + (i0 & 1);
            const int r1 = i1 < 0 ? 1 : (i1>>1 == j0 ? 0 : 2) + (i1 & 1);
            const int r2 = i2 < 0 ? 2 : (i2>>1 == j1 ? 2 : 0) + (i2 & 1);
            const int r3 = i3 < 0 ? 3 : (i3>>1 == j1 ? 2 : 0) + (i3 & 1);

            // Find permute indices for high 16-bit permute
            const int s0 = i4 < 0 ? 0 : (i4>>1 == j2 ? 0 : 2) + (i4 & 1);
            const int s1 = i5 < 0 ? 1 : (i5>>1 == j2 ? 0 : 2) + (i5 & 1);
            const int s2 = i6 < 0 ? 2 : (i6>>1 == j3 ? 2 : 0) + (i6 & 1);
            const int s3 = i7 < 0 ? 3 : (i7>>1 == j3 ? 2 : 0) + (i7 & 1);

            // 32-bit permute
            t1 = _mm_shuffle_epi32 (a, j0 | j1<<2 | j2<<4 | j3<<6);
            // 16-bit permutes
            if (r0!=0 || r1!=1 || r2!=2 || r3!=3) {  // 16 bit permute of low  half
                t2 = _mm_shufflelo_epi16(t1, r0 | r1<<2 | r2<<4 | r3<<6);
            }
            else t2 = t1;
            if (s0!=0 || s1!=1 || s2!=2 || s3!=3) {  // 16 bit permute of high half                
                t7 = _mm_shufflehi_epi16(t2, s0 | s1<<2 | s2<<4 | s3<<6);
            }
            else t7 = t2;
        }
        else {
            // Worst case. We need two sets of 16-bit permutes
            t1 = _mm_shuffle_epi32(a, 0x4E);  // swap low and high 64-bits

            // Find permute indices for low 16-bit permute from swapped t1
            const int r0 = i0 < 4 ? 0 : i0 & 3;
            const int r1 = i1 < 4 ? 1 : i1 & 3;
            const int r2 = i2 < 4 ? 2 : i2 & 3;
            const int r3 = i3 < 4 ? 3 : i3 & 3;
            // Find permute indices for high 16-bit permute from swapped t1
            const int s0 = i4 < 0 || i4 >= 4 ? 0 : i4 & 3;
            const int s1 = i5 < 0 || i5 >= 4 ? 1 : i5 & 3;
            const int s2 = i6 < 0 || i6 >= 4 ? 2 : i6 & 3;
            const int s3 = i7 < 0 || i7 >= 4 ? 3 : i7 & 3;
            // Find permute indices for low 16-bit permute from direct a
            const int u0 = i0 < 0 || i0 >= 4 ? 0 : i0 & 3;
            const int u1 = i1 < 0 || i1 >= 4 ? 1 : i1 & 3;
            const int u2 = i2 < 0 || i2 >= 4 ? 2 : i2 & 3;
            const int u3 = i3 < 0 || i3 >= 4 ? 3 : i3 & 3;
            // Find permute indices for high 16-bit permute from direct a
            const int v0 = i4 < 4 ? 0 : i4 & 3;
            const int v1 = i5 < 4 ? 1 : i5 & 3;
            const int v2 = i6 < 4 ? 2 : i6 & 3;
            const int v3 = i7 < 4 ? 3 : i7 & 3;

            // 16-bit permutes
            if (r0!=0 || r1!=1 || r2!=2 || r3!=3) {  // 16 bit permute of low  half
                t2 = _mm_shufflelo_epi16(t1, r0 | r1<<2 | r2<<4 | r3<<6);
            }
            else t2 = t1;
            if (u0!=0 || u1!=1 || u2!=2 || u3!=3) {  // 16 bit permute of low  half
                t3 = _mm_shufflelo_epi16(a, u0 | u1<<2 | u2<<4 | u3<<6);
            }
            else t3 = a;
            if (s0!=0 || s1!=1 || s2!=2 || s3!=3) {  // 16 bit permute of low  half
                t4 = _mm_shufflehi_epi16(t2, s0 | s1<<2 | s2<<4 | s3<<6);
            }
            else t4 = t2;
            if (v0!=0 || v1!=1 || v2!=2 || v3!=3) {  // 16 bit permute of low  half
                t5 = _mm_shufflehi_epi16(t3, v0 | v1<<2 | v2<<4 | v3<<6);
            }
            else t5 = t3;
            // merge data from t4 and t5
            t6  = constant4i <
                ((i0 & 4) ? 0xFFFF : 0) | ((i1 & 4) ? 0xFFFF0000 : 0),
                ((i2 & 4) ? 0xFFFF : 0) | ((i3 & 4) ? 0xFFFF0000 : 0),
                ((i4 & 4) ? 0 : 0xFFFF) | ((i5 & 4) ? 0 : 0xFFFF0000),
                ((i6 & 4) ? 0 : 0xFFFF) | ((i7 & 4) ? 0 : 0xFFFF0000) > ();
            t7 = selectb(t6,t4,t5);  // select between permuted data t4 and t5
        }
    }
    // Set any elements to zero if required
    if (m2 != -1 && ((i0 | i1 | i2 | i3 | i4 | i5 | i6 | i7) & 0x80)) {
        // some elements need to be set to 0
        __m128i mask = constant4i <
            (i0 < 0 ? 0xFFFF0000 : -1) & (i1 < 0 ? 0x0000FFFF : -1),
            (i2 < 0 ? 0xFFFF0000 : -1) & (i3 < 0 ? 0x0000FFFF : -1),
            (i4 < 0 ? 0xFFFF0000 : -1) & (i5 < 0 ? 0x0000FFFF : -1),
            (i6 < 0 ? 0xFFFF0000 : -1) & (i7 < 0 ? 0x0000FFFF : -1) > ();
        return  _mm_and_si128(t7,mask);
    }
    else {
        return  t7;
    }
#endif
}

template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
static inline Vec8us permute8us(Vec8us const & a) {
    return Vec8us (permute8s <i0,i1,i2,i3,i4,i5,i6,i7> (a));
}


template <int i0, int i1, int i2,  int i3,  int i4,  int i5,  int i6,  int i7, 
          int i8, int i9, int i10, int i11, int i12, int i13, int i14, int i15 > 
static inline Vec16c permute16c(Vec16c const & a) {

    __m128i temp;

    // Combine all even indexes into a single bitfield, with 4 bits for each
    const uint32_t me = (i0&15) | (i2&15)<<4 | (i4&15)<<8 | (i6&15)<<12 
        | (i8&15)<<16 | (i10&15)<<20 | (i12&15)<<24 | (i14&15)<<28; 

    // Combine all odd indexes into a single bitfield, with 4 bits for each
    const uint32_t mo = (i1&15) | (i3&15)<<4 | (i5&15)<<8 | (i7&15)<<12 
        | (i9&15)<<16 | (i11&15)<<20 | (i13&15)<<24 | (i15&15)<<28; 

    // Mask indicating sign of all even indexes, with 4 bits for each, 0 for negative, 0xF for non-negative
    const uint32_t se = (i0<0?0:0xF) | (i2<0?0:0xF)<<4 | (i4<0?0:0xF)<<8 | (i6<0?0:0xF)<<12
        | (i8<0?0:0xF)<<16 | (i10<0?0:0xF)<<20 | (i12<0?0:0xF)<<24 | (i14<0?0:0xF)<<28;

    // Mask indicating sign of all odd indexes, with 4 bits for each, 0 for negative, 0xF for non-negative
    const uint32_t so = (i1<0?0:0xF) | (i3<0?0:0xF)<<4 | (i5<0?0:0xF)<<8 | (i7<0?0:0xF)<<12
        | (i9<0?0:0xF)<<16 | (i11<0?0:0xF)<<20 | (i13<0?0:0xF)<<24 | (i15<0?0:0xF)<<28;

    // Mask indicating sign of all indexes, with 2 bits for each, 0 for negative (means set to zero or don't care), 0x3 for non-negative
    const uint32_t ss = (se & 0x33333333) | (so & 0xCCCCCCCC);

    // Mask indicating required zeroing of all indexes, with 2 bits for each, 0 for index = -1, 3 for index >= 0 or -256
    const uint32_t ssz = ((i0&0x80)?0:3) | ((i1 &0x80)?0:3)<< 2 | ((i2 &0x80)?0:3)<< 4 | ((i3 &0x80)?0:3)<< 6 | 
                    ((i4 &0x80)?0:3)<< 8 | ((i5 &0x80)?0:3)<<10 | ((i6 &0x80)?0:3)<<12 | ((i7 &0x80)?0:3)<<14 | 
                    ((i8 &0x80)?0:3)<<16 | ((i9 &0x80)?0:3)<<18 | ((i10&0x80)?0:3)<<20 | ((i11&0x80)?0:3)<<22 | 
                    ((i12&0x80)?0:3)<<24 | ((i13&0x80)?0:3)<<26 | ((i14&0x80)?0:3)<<28 | ((i15&0x80)?0:3)<<30 ;

    // These indexes are used only to avoid bogus compiler warnings in false branches
    const int I0  = i0  > 0 ? (i0  & 0xF) : 0;
    const int I15 = i15 > 0 ? (i15 & 0xF) : 0;

    // special case: all zero
    if (ss == 0) {
        return _mm_setzero_si128();  
    }

    // remember if extra zeroing is needed
    bool do_and_zero = (ssz != 0xFFFFFFFFu);

    // check for special shortcut cases
    int shortcut = 0;

    // check if any permutation
    if (((me ^ 0xECA86420) & se) == 0 && ((mo ^ 0xFDB97531) & so) == 0) {
        shortcut = 1;
    }
    // check if we can use punpcklbw
    else if (((me ^ 0x76543210) & se) == 0 && ((mo ^ 0x76543210) & so) == 0) {
        shortcut = 2;
    }
    // check if we can use punpckhbw
    else if (((me ^ 0xFEDCBA98) & se) == 0 && ((mo ^ 0xFEDCBA98) & so) == 0) {
        shortcut = 3;
    }

    #if defined (_MSC_VER) && ! defined(__INTEL_COMPILER)
    #pragma warning(disable: 4307)  // disable MS warning C4307: '+' : integral constant overflow
    #endif

    // check if we can use byte shift right
    else if (i0 > 0 && ((me ^ (uint32_t(I0)*0x11111111u + 0xECA86420u)) & se) == 0 && 
    ((mo ^ (uint32_t(I0)*0x11111111u + 0xFDB97531u)) & so) == 0) {
        shortcut = 4;
        do_and_zero = ((0xFFFFFFFFu >> 2*I0) & ~ ssz) != 0;
    }
    // check if we can use byte shift left
    else if (i15 >= 0 && i15 < 15 &&         
    ((mo ^ (uint32_t(I15*0x11111111u) - (0x02468ACEu & so))) & so) == 0 && 
    ((me ^ (uint32_t(I15*0x11111111u) - (0x13579BDFu & se))) & se) == 0) {
        shortcut = 5;
        do_and_zero = ((0xFFFFFFFFu << 2*(15-I15)) & ~ ssz) != 0;
    }

#if  INSTRSET >= 4  // SSSE3 (PSHUFB available only under SSSE3)

    // special case: rotate
    if (i0>0 && i0 < 16    && i1==((i0+1)&15) && i2 ==((i0+2 )&15) && i3 ==((i0+3 )&15) && i4 ==((i0+4 )&15) && i5 ==((i0+5 )&15) && i6 ==((i0+6 )&15) && i7 ==((i0+7 )&15) 
    && i8==((i0+8)&15) && i9==((i0+9)&15) && i10==((i0+10)&15) && i11==((i0+11)&15) && i12==((i0+12)&15) && i13==((i0+13)&15) && i14==((i0+14)&15) && i15==((i0+15)&15)) {
        temp = _mm_alignr_epi8(a, a, i0 & 15);
        shortcut = -1;
    }
    if (shortcut == 0 || do_and_zero) {
        // general case: use PSHUFB
        __m128i mask = constant4i< 
            (i0  & 0xFF) | (i1  & 0xFF) << 8 | (i2  & 0xFF) << 16 | (i3  & 0xFF) << 24 ,
            (i4  & 0xFF) | (i5  & 0xFF) << 8 | (i6  & 0xFF) << 16 | (i7  & 0xFF) << 24 ,
            (i8  & 0xFF) | (i9  & 0xFF) << 8 | (i10 & 0xFF) << 16 | (i11 & 0xFF) << 24 ,
            (i12 & 0xFF) | (i13 & 0xFF) << 8 | (i14 & 0xFF) << 16 | (i15 & 0xFF) << 24 > ();
        temp = _mm_shuffle_epi8(a,mask);
        shortcut = -1;
        do_and_zero = false;
    }

#endif

    // Check if we can use 16-bit permute. Even numbered indexes must be even and odd numbered
    // indexes must be equal to the preceding index + 1, except for negative indexes.
    if (shortcut == 0 && (me & 0x11111111 & se) == 0 && ((mo ^ 0x11111111) & 0x11111111 & so) == 0 && ((me ^ mo) & 0xEEEEEEEE & se & so) == 0) {
        temp = permute8s <
            i0  >= 0 ? i0 /2 : i1  >= 0 ? i1 /2 : (i0  | i1 ),
            i2  >= 0 ? i2 /2 : i3  >= 0 ? i3 /2 : (i2  | i3 ),
            i4  >= 0 ? i4 /2 : i5  >= 0 ? i5 /2 : (i4  | i5 ),
            i6  >= 0 ? i6 /2 : i7  >= 0 ? i7 /2 : (i6  | i7 ),
            i8  >= 0 ? i8 /2 : i9  >= 0 ? i9 /2 : (i8  | i9 ),
            i10 >= 0 ? i10/2 : i11 >= 0 ? i11/2 : (i10 | i11),
            i12 >= 0 ? i12/2 : i13 >= 0 ? i13/2 : (i12 | i13),
            i14 >= 0 ? i14/2 : i15 >= 0 ? i15/2 : (i14 | i15) > (Vec8s(a));
        shortcut = 100;
        do_and_zero = (se != so && ssz != 0xFFFFFFFFu);
    }
  
    // Check if we can use 16-bit permute with bytes swapped. Even numbered indexes must be odd and odd 
    // numbered indexes must be equal to the preceding index - 1, except for negative indexes.
    // (this case occurs when reversing byte order)
    if (shortcut == 0 && ((me ^ 0x11111111) & 0x11111111 & se) == 0 && (mo & 0x11111111 & so) == 0 && ((me ^ mo) & 0xEEEEEEEE & se & so) == 0) {
        Vec16c swapped = Vec16c(rotate_left(Vec8s(a), 8)); // swap odd and even bytes
        temp = permute8s <
            i0  >= 0 ? i0 /2 : i1  >= 0 ? i1 /2 : (i0  | i1 ),
            i2  >= 0 ? i2 /2 : i3  >= 0 ? i3 /2 : (i2  | i3 ),
            i4  >= 0 ? i4 /2 : i5  >= 0 ? i5 /2 : (i4  | i5 ),
            i6  >= 0 ? i6 /2 : i7  >= 0 ? i7 /2 : (i6  | i7 ),
            i8  >= 0 ? i8 /2 : i9  >= 0 ? i9 /2 : (i8  | i9 ),
            i10 >= 0 ? i10/2 : i11 >= 0 ? i11/2 : (i10 | i11),
            i12 >= 0 ? i12/2 : i13 >= 0 ? i13/2 : (i12 | i13),
            i14 >= 0 ? i14/2 : i15 >= 0 ? i15/2 : (i14 | i15) > (Vec8s(swapped));
        shortcut = 101;
        do_and_zero = (se != so && ssz != 0xFFFFFFFFu);
    }

    // all shortcuts end here
    if (shortcut) {
        switch (shortcut) {
        case 1:
            temp = a;  break;
        case 2:
            temp = _mm_unpacklo_epi8(a,a);  break;
        case 3:
            temp = _mm_unpackhi_epi8(a,a);  break;
        case 4:
            temp = _mm_srli_si128(a, I0);  break;
        case 5:
            temp = _mm_slli_si128(a, 15-I15);  break;
        default:
            break;  // result is already in temp
        }
        if (do_and_zero) {
            // additional zeroing needed
            __m128i maskz = constant4i < 
                (i0  < 0 ? 0 : 0xFF) | (i1  < 0 ? 0 : 0xFF00) | (i2  < 0 ? 0 : 0xFF0000) | (i3  < 0 ? 0 : 0xFF000000) ,
                (i4  < 0 ? 0 : 0xFF) | (i5  < 0 ? 0 : 0xFF00) | (i6  < 0 ? 0 : 0xFF0000) | (i7  < 0 ? 0 : 0xFF000000) ,
                (i8  < 0 ? 0 : 0xFF) | (i9  < 0 ? 0 : 0xFF00) | (i10 < 0 ? 0 : 0xFF0000) | (i11 < 0 ? 0 : 0xFF000000) ,
                (i12 < 0 ? 0 : 0xFF) | (i13 < 0 ? 0 : 0xFF00) | (i14 < 0 ? 0 : 0xFF0000) | (i15 < 0 ? 0 : 0xFF000000) > ();
            temp = _mm_and_si128(temp, maskz);
        }
        return temp;
    }

    // complicated cases: use 16-bit permute up to four times
    const bool e2e = (~me & 0x11111111 & se) != 0;  // even bytes of source to even bytes of destination
    const bool e2o = (~mo & 0x11111111 & so) != 0;  // even bytes of source to odd  bytes of destination
    const bool o2e = (me  & 0x11111111 & se) != 0;  // odd  bytes of source to even bytes of destination
    const bool o2o = (mo  & 0x11111111 & so) != 0;  // odd  bytes of source to odd  bytes of destination
    
    Vec16c swapped, te2e, te2o, to2e, to2o, combeven, combodd;

    if (e2o || o2e) swapped = rotate_left(Vec8s(a), 8); // swap odd and even bytes

    // even-to-even bytes
    if (e2e) te2e = permute8s <(i0&1)?-1:i0/2, (i2&1)?-1:i2/2, (i4&1)?-1:i4/2, (i6&1)?-1:i6/2,
        (i8&1)?-1:i8/2, (i10&1)?-1:i10/2, (i12&1)?-1:i12/2, (i14&1)?-1:i14/2> (Vec8s(a));                 
    // odd-to-even bytes
    if (o2e) to2e = permute8s <(i0&1)?i0/2:-1, (i2&1)?i2/2:-1, (i4&1)?i4/2:-1, (i6&1)?i6/2:-1,
        (i8&1)?i8/2:-1, (i10&1)?i10/2:-1, (i12&1)?i12/2:-1, (i14&1)?i14/2:-1> (Vec8s(swapped));
    // even-to-odd bytes
    if (e2o) te2o = permute8s <(i1&1)?-1:i1/2, (i3&1)?-1:i3/2, (i5&1)?-1:i5/2, (i7&1)?-1:i7/2, 
        (i9&1)?-1:i9/2, (i11&1)?-1:i11/2, (i13&1)?-1:i13/2, (i15&1)?-1:i15/2> (Vec8s(swapped));
    // odd-to-odd bytes
    if (o2o) to2o = permute8s <(i1&1)?i1/2:-1, (i3&1)?i3/2:-1, (i5&1)?i5/2:-1, (i7&1)?i7/2:-1,
        (i9&1)?i9/2:-1, (i11&1)?i11/2:-1, (i13&1)?i13/2:-1, (i15&1)?i15/2:-1> (Vec8s(a));

    if (e2e && o2e) combeven = te2e | to2e;
    else if (e2e)   combeven = te2e;
    else if (o2e)   combeven = to2e;
    else            combeven = _mm_setzero_si128();

    if (e2o && o2o) combodd  = te2o | to2o;
    else if (e2o)   combodd  = te2o;
    else if (o2o)   combodd  = to2o;
    else            combodd  = _mm_setzero_si128();

    __m128i maske = constant4i <     // mask used even bytes
        (i0  < 0 ? 0 : 0xFF) | (i2  < 0 ? 0 : 0xFF0000),
        (i4  < 0 ? 0 : 0xFF) | (i6  < 0 ? 0 : 0xFF0000),
        (i8  < 0 ? 0 : 0xFF) | (i10 < 0 ? 0 : 0xFF0000),
        (i12 < 0 ? 0 : 0xFF) | (i14 < 0 ? 0 : 0xFF0000) > ();
    __m128i masko = constant4i <     // mask used odd bytes
        (i1  < 0 ? 0 : 0xFF00) | (i3  < 0 ? 0 : 0xFF000000),
        (i5  < 0 ? 0 : 0xFF00) | (i7  < 0 ? 0 : 0xFF000000),
        (i9  < 0 ? 0 : 0xFF00) | (i11 < 0 ? 0 : 0xFF000000),
        (i13 < 0 ? 0 : 0xFF00) | (i15 < 0 ? 0 : 0xFF000000) > ();

    return  _mm_or_si128(            // combine even and odd bytes
        _mm_and_si128(combeven, maske),
        _mm_and_si128(combodd, masko));
}

template <int i0, int i1, int i2,  int i3,  int i4,  int i5,  int i6,  int i7, 
          int i8, int i9, int i10, int i11, int i12, int i13, int i14, int i15 > 
static inline Vec16uc permute16uc(Vec16uc const & a) {
    return Vec16uc (permute16c <i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15> (a));
}


/*****************************************************************************
*
*          Vector blend functions
*
******************************************************************************
*
* These blend functions can mix elements from two different vectors and
* optionally set some elements to zero. 
*
* The indexes are inserted as template parameters in <>. These indexes must be
* constants. Each template parameter is an index to the element you want to 
* select, where higher indexes indicate an element from the second source
* vector. For example, if each vector has 4 elements, then indexes 0 - 3
* will select an element from the first vector and indexes 4 - 7 will select 
* an element from the second vector. A negative index will generate zero.
*
* The blend functions for vectors of 8-bit integers are inefficient if 
* the SSSE3 instruction set or later is not enabled.
*
* Example:
* Vec4i a(100,101,102,103);         // a is (100, 101, 102, 103)
* Vec4i b(200,201,202,203);         // b is (200, 201, 202, 203)
* Vec4i c;
* c = blend4i<1,4,-1,7> (a,b);      // c is (101, 200,   0, 203)
*
* A lot of the code here is metaprogramming aiming to find the instructions
* that best fit the template parameters and instruction set. The metacode
* will be reduced out to leave only a few vector instructions in release
* mode with optimization on.
*****************************************************************************/

template <int i0, int i1, int i2,  int i3,  int i4,  int i5,  int i6,  int i7, 
          int i8, int i9, int i10, int i11, int i12, int i13, int i14, int i15 > 
static inline Vec16c blend16c(Vec16c const & a, Vec16c const & b) {

    // Combine bit 0-3 of all even indexes into a single bitfield, with 4 bits for each
    const int me = (i0&15) | (i2&15)<<4 | (i4&15)<<8 | (i6&15)<<12 
        | (i8&15)<<16 | (i10&15)<<20 | (i12&15)<<24 | (i14&15)<<28; 

    // Combine bit 0-3 of all odd indexes into a single bitfield, with 4 bits for each
    const int mo = (i1&15) | (i3&15)<<4 | (i5&15)<<8 | (i7&15)<<12 
        | (i9&15)<<16 | (i11&15)<<20 | (i13&15)<<24 | (i15&15)<<28; 

    // Mask indicating sign of all even indexes, with 4 bits for each, 0 for negative, 0xF for non-negative
    const int se = (i0<0?0:0xF) | (i2<0?0:0xF)<<4 | (i4<0?0:0xF)<<8 | (i6<0?0:0xF)<<12
        | (i8<0?0:0xF)<<16 | (i10<0?0:0xF)<<20 | (i12<0?0:0xF)<<24 | (i14<0?0:0xF)<<28;

    // Mask indicating sign of all odd indexes, with 4 bits for each, 0 for negative, 0xF for non-negative
    const int so = (i1<0?0:0xF) | (i3<0?0:0xF)<<4 | (i5<0?0:0xF)<<8 | (i7<0?0:0xF)<<12
        | (i9<0?0:0xF)<<16 | (i11<0?0:0xF)<<20 | (i13<0?0:0xF)<<24 | (i15<0?0:0xF)<<28;

    // Combine bit 4 of all even indexes into a single bitfield, with 4 bits for each
    const int ne = (i0&16)>>4 | (i2&16) | (i4&16)<<4 | (i6&16)<<8 
        | (i8&16)<<12 | (i10&16)<<16 | (i12&16)<<20 | (i14&16)<<24; 

    // Combine bit 4 of all odd indexes into a single bitfield, with 4 bits for each
    const int no = (i1&16)>>4 | (i3&16) | (i5&16)<<4 | (i7&16)<<8
        | (i9&16)<<12 | (i11&16)<<16 | (i13&16)<<20 | (i15&16)<<24; 

    // Check if zeroing needed
    const bool do_zero = ((i0|i1|i2|i3|i4|i5|i6|i7|i8|i9|i10|i11|i12|i13|i14|i15) & 0x80) != 0; // needs zeroing

    // no elements from b
    if (((ne & se) | (no & so)) == 0) {
        return permute16c <i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15> (a);
    }

    // no elements from a
    if ((((ne^0x11111111) & se) | ((no^0x11111111) & so)) == 0) {
        return permute16c <i0^16, i1^16, i2^16, i3^16, i4^16, i5^16, i6^16, i7^16, i8^16, i9^16, i10^16, i11^16, i12^16, i13^16, i14^16, i15^16> (b);
    }
    __m128i t;

    // check if we can use punpcklbw
    if (((me ^ 0x76543210) & se) == 0 && ((mo ^ 0x76543210) & so) == 0) {
        if ((ne & se) == 0 && ((no ^ 0x11111111) & so) == 0) {        
            t = _mm_unpacklo_epi8(a,b);
        }
        if ((no & so) == 0 && ((ne ^ 0x11111111) & se) == 0) {        
            t = _mm_unpacklo_epi8(b,a);
        }
        if (do_zero) {
            // additional zeroing needed
            __m128i maskz = constant4i < 
                (i0  < 0 ? 0 : 0xFF) | (i1  < 0 ? 0 : 0xFF00) | (i2  < 0 ? 0 : 0xFF0000) | (i3  < 0 ? 0 : 0xFF000000) ,
                (i4  < 0 ? 0 : 0xFF) | (i5  < 0 ? 0 : 0xFF00) | (i6  < 0 ? 0 : 0xFF0000) | (i7  < 0 ? 0 : 0xFF000000) ,
                (i8  < 0 ? 0 : 0xFF) | (i9  < 0 ? 0 : 0xFF00) | (i10 < 0 ? 0 : 0xFF0000) | (i11 < 0 ? 0 : 0xFF000000) ,
                (i12 < 0 ? 0 : 0xFF) | (i13 < 0 ? 0 : 0xFF00) | (i14 < 0 ? 0 : 0xFF0000) | (i15 < 0 ? 0 : 0xFF000000) > ();
            t = _mm_and_si128(t, maskz);
        }
        return t;
    }

    // check if we can use punpckhbw
    if (((me ^ 0xFEDCBA98) & se) == 0 && ((mo ^ 0xFEDCBA98) & so) == 0) {
        if ((ne & se) == 0 && ((no ^ 0x11111111) & so) == 0) {        
            t = _mm_unpackhi_epi8(a,b);
        }
        if ((no & so) == 0 && ((ne ^ 0x11111111) & se) == 0) {        
            t = _mm_unpackhi_epi8(b,a);
        }
        if (do_zero) {
            // additional zeroing needed
            __m128i maskz = constant4i < 
                (i0  < 0 ? 0 : 0xFF) | (i1  < 0 ? 0 : 0xFF00) | (i2  < 0 ? 0 : 0xFF0000) | (i3  < 0 ? 0 : 0xFF000000) ,
                (i4  < 0 ? 0 : 0xFF) | (i5  < 0 ? 0 : 0xFF00) | (i6  < 0 ? 0 : 0xFF0000) | (i7  < 0 ? 0 : 0xFF000000) ,
                (i8  < 0 ? 0 : 0xFF) | (i9  < 0 ? 0 : 0xFF00) | (i10 < 0 ? 0 : 0xFF0000) | (i11 < 0 ? 0 : 0xFF000000) ,
                (i12 < 0 ? 0 : 0xFF) | (i13 < 0 ? 0 : 0xFF00) | (i14 < 0 ? 0 : 0xFF0000) | (i15 < 0 ? 0 : 0xFF000000) > ();
            t = _mm_and_si128(t, maskz);
        }
        return t;
    }
    
#if  INSTRSET >= 4  // SSSE3
    // special case: shift left
    if (i0 > 0 && i0 < 16 && i1==i0+1 && i2==i0+2 && i3==i0+3 && i4==i0+4 && i5==i0+5 && i6==i0+6 && i7==i0+7 && 
        i8==i0+8 && i9==i0+9 && i10==i0+10 && i11==i0+11 && i12==i0+12 && i13==i0+13 && i14==i0+14 && i15==i0+15) {
        return _mm_alignr_epi8(b, a, (i0 & 15));
    }

    // special case: shift right
    if (i0 > 15 && i0 < 32 && i1==((i0+1)&31) && i2 ==((i0+2 )&31) && i3 ==((i0+3 )&31) && i4 ==((i0+4 )&31) && i5 ==((i0+5 )&31) && i6 ==((i0+6 )&31) && i7 ==((i0+7 )&31) && 
        i8==((i0+8 )&31)   && i9==((i0+9)&31) && i10==((i0+10)&31) && i11==((i0+11)&31) && i12==((i0+12)&31) && i13==((i0+13)&31) && i14==((i0+14)&31) && i15==((i0+15)&31)) {
        return _mm_alignr_epi8(a, b, (i0 & 15));
    }
#endif

#if INSTRSET >= 5   // SSE4.1 supported
    // special case: blend without permute
    if (((me ^ 0xECA86420) & se) == 0 && ((mo ^ 0xFDB97531) & so) == 0) {
        __m128i maskbl = constant4i<
            ((i0 & 16) ? 0xFF : 0) | ((i1 & 16) ? 0xFF00 : 0) | ((i2 & 16) ? 0xFF0000 : 0) | ((i3 & 16) ? 0xFF000000 : 0) ,
            ((i4 & 16) ? 0xFF : 0) | ((i5 & 16) ? 0xFF00 : 0) | ((i6 & 16) ? 0xFF0000 : 0) | ((i7 & 16) ? 0xFF000000 : 0) ,
            ((i8 & 16) ? 0xFF : 0) | ((i9 & 16) ? 0xFF00 : 0) | ((i10& 16) ? 0xFF0000 : 0) | ((i11& 16) ? 0xFF000000 : 0) ,
            ((i12& 16) ? 0xFF : 0) | ((i13& 16) ? 0xFF00 : 0) | ((i14& 16) ? 0xFF0000 : 0) | ((i15& 16) ? 0xFF000000 : 0) > ();
        t = _mm_blendv_epi8(a, b, maskbl);
        if (do_zero) {
            // additional zeroing needed
            __m128i maskz = constant4i < 
                (i0  < 0 ? 0 : 0xFF) | (i1  < 0 ? 0 : 0xFF00) | (i2  < 0 ? 0 : 0xFF0000) | (i3  < 0 ? 0 : 0xFF000000) ,
                (i4  < 0 ? 0 : 0xFF) | (i5  < 0 ? 0 : 0xFF00) | (i6  < 0 ? 0 : 0xFF0000) | (i7  < 0 ? 0 : 0xFF000000) ,
                (i8  < 0 ? 0 : 0xFF) | (i9  < 0 ? 0 : 0xFF00) | (i10 < 0 ? 0 : 0xFF0000) | (i11 < 0 ? 0 : 0xFF000000) ,
                (i12 < 0 ? 0 : 0xFF) | (i13 < 0 ? 0 : 0xFF00) | (i14 < 0 ? 0 : 0xFF0000) | (i15 < 0 ? 0 : 0xFF000000) > ();
            t = _mm_and_si128(t, maskz);
        }
        return t;
    }
#endif // SSE4.1

#if defined ( __XOP__ )    // Use AMD XOP instruction VPPERM
    __m128i mask = constant4i<
        (i0 <0 ? 0x80 : (i0 &31)) | (i1 <0 ? 0x80 : (i1 &31)) << 8 | (i2 <0 ? 0x80 : (i2 &31)) << 16 | (i3 <0 ? 0x80 : (i3 &31)) << 24,
        (i4 <0 ? 0x80 : (i4 &31)) | (i5 <0 ? 0x80 : (i5 &31)) << 8 | (i6 <0 ? 0x80 : (i6 &31)) << 16 | (i7 <0 ? 0x80 : (i7 &31)) << 24,
        (i8 <0 ? 0x80 : (i8 &31)) | (i9 <0 ? 0x80 : (i9 &31)) << 8 | (i10<0 ? 0x80 : (i10&31)) << 16 | (i11<0 ? 0x80 : (i11&31)) << 24,
        (i12<0 ? 0x80 : (i12&31)) | (i13<0 ? 0x80 : (i13&31)) << 8 | (i14<0 ? 0x80 : (i14&31)) << 16 | (i15<0 ? 0x80 : (i15&31)) << 24 > ();
    return _mm_perm_epi8(a, b, mask);

#elif  INSTRSET >= 4  // SSSE3
   
    // general case. Use PSHUFB
    __m128i maska = constant4i<
        ((i0 & 0x90) ? 0xFF : (i0 &15)) | ((i1 & 0x90) ? 0xFF : (i1 &15)) << 8 | ((i2 & 0x90) ? 0xFF : (i2 &15)) << 16 | ((i3 & 0x90) ? 0xFF : (i3 &15)) << 24,
        ((i4 & 0x90) ? 0xFF : (i4 &15)) | ((i5 & 0x90) ? 0xFF : (i5 &15)) << 8 | ((i6 & 0x90) ? 0xFF : (i6 &15)) << 16 | ((i7 & 0x90) ? 0xFF : (i7 &15)) << 24,
        ((i8 & 0x90) ? 0xFF : (i8 &15)) | ((i9 & 0x90) ? 0xFF : (i9 &15)) << 8 | ((i10& 0x90) ? 0xFF : (i10&15)) << 16 | ((i11& 0x90) ? 0xFF : (i11&15)) << 24,
        ((i12& 0x90) ? 0xFF : (i12&15)) | ((i13& 0x90) ? 0xFF : (i13&15)) << 8 | ((i14& 0x90) ? 0xFF : (i14&15)) << 16 | ((i15& 0x90) ? 0xFF : (i15&15)) << 24 > ();
    __m128i maskb = constant4i<
        (((i0^0x10) & 0x90) ? 0xFF : (i0 &15)) | (((i1^0x10) & 0x90) ? 0xFF : (i1 &15)) << 8 | (((i2^0x10) & 0x90) ? 0xFF : (i2 &15)) << 16 | (((i3^0x10) & 0x90) ? 0xFF : (i3 &15)) << 24,
        (((i4^0x10) & 0x90) ? 0xFF : (i4 &15)) | (((i5^0x10) & 0x90) ? 0xFF : (i5 &15)) << 8 | (((i6^0x10) & 0x90) ? 0xFF : (i6 &15)) << 16 | (((i7^0x10) & 0x90) ? 0xFF : (i7 &15)) << 24,
        (((i8^0x10) & 0x90) ? 0xFF : (i8 &15)) | (((i9^0x10) & 0x90) ? 0xFF : (i9 &15)) << 8 | (((i10^0x10)& 0x90) ? 0xFF : (i10&15)) << 16 | (((i11^0x10)& 0x90) ? 0xFF : (i11&15)) << 24,
        (((i12^0x10)& 0x90) ? 0xFF : (i12&15)) | (((i13^0x10)& 0x90) ? 0xFF : (i13&15)) << 8 | (((i14^0x10)& 0x90) ? 0xFF : (i14&15)) << 16 | (((i15^0x10)& 0x90) ? 0xFF : (i15&15)) << 24 > ();
    __m128i a1 = _mm_shuffle_epi8(a,maska);
    __m128i b1 = _mm_shuffle_epi8(b,maskb);
    return       _mm_or_si128(a1,b1);

#else                 // SSE2
    // combine two permutes
    __m128i a1 = permute16c <
        (uint32_t)i0  < 16 ? i0  : -1,
        (uint32_t)i1  < 16 ? i1  : -1,
        (uint32_t)i2  < 16 ? i2  : -1,
        (uint32_t)i3  < 16 ? i3  : -1,
        (uint32_t)i4  < 16 ? i4  : -1,
        (uint32_t)i5  < 16 ? i5  : -1,
        (uint32_t)i6  < 16 ? i6  : -1,
        (uint32_t)i7  < 16 ? i7  : -1,
        (uint32_t)i8  < 16 ? i8  : -1,
        (uint32_t)i9  < 16 ? i9  : -1,
        (uint32_t)i10 < 16 ? i10 : -1,
        (uint32_t)i11 < 16 ? i11 : -1,
        (uint32_t)i12 < 16 ? i12 : -1,
        (uint32_t)i13 < 16 ? i13 : -1,
        (uint32_t)i14 < 16 ? i14 : -1,
        (uint32_t)i15 < 16 ? i15 : -1 > (a);
    __m128i b1 = permute16c <
        (uint32_t)(i0 ^16) < 16 ? (i0 ^16) : -1,
        (uint32_t)(i1 ^16) < 16 ? (i1 ^16) : -1,
        (uint32_t)(i2 ^16) < 16 ? (i2 ^16) : -1,
        (uint32_t)(i3 ^16) < 16 ? (i3 ^16) : -1,
        (uint32_t)(i4 ^16) < 16 ? (i4 ^16) : -1,
        (uint32_t)(i5 ^16) < 16 ? (i5 ^16) : -1,
        (uint32_t)(i6 ^16) < 16 ? (i6 ^16) : -1,
        (uint32_t)(i7 ^16) < 16 ? (i7 ^16) : -1,        
        (uint32_t)(i8 ^16) < 16 ? (i8 ^16) : -1,
        (uint32_t)(i9 ^16) < 16 ? (i9 ^16) : -1,
        (uint32_t)(i10^16) < 16 ? (i10^16) : -1,
        (uint32_t)(i11^16) < 16 ? (i11^16) : -1,
        (uint32_t)(i12^16) < 16 ? (i12^16) : -1,
        (uint32_t)(i13^16) < 16 ? (i13^16) : -1,
        (uint32_t)(i14^16) < 16 ? (i14^16) : -1,
        (uint32_t)(i15^16) < 16 ? (i15^16) : -1 > (b);
    return   _mm_or_si128(a1,b1);

#endif
}

template <int i0, int i1, int i2,  int i3,  int i4,  int i5,  int i6,  int i7, 
          int i8, int i9, int i10, int i11, int i12, int i13, int i14, int i15 > 
static inline Vec16uc blend16uc(Vec16uc const & a, Vec16uc const & b) {
    return Vec16uc( blend16c<i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15> (a,b));
}


template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
static inline Vec8s blend8s(Vec8s const & a, Vec8s const & b) {

    // Combine all the indexes into a single bitfield, with 4 bits for each
    const int m1 = (i0&0xF) | (i1&0xF)<<4 | (i2&0xF)<<8 | (i3&0xF)<<12 
        | (i4&0xF)<<16 | (i5&0xF)<<20 | (i6&0xF)<<24 | (i7&0xF)<<28; 

    // Mask to zero out negative indexes
    const int mz = (i0<0?0:0xF) | (i1<0?0:0xF)<<4 | (i2<0?0:0xF)<<8 | (i3<0?0:0xF)<<12
        | (i4<0?0:0xF)<<16 | (i5<0?0:0xF)<<20 | (i6<0?0:0xF)<<24 | (i7<0?0:0xF)<<28;

    // Some elements must be set to zero
    const bool do_zero = (mz != -1) && ((i0 | i1 | i2 | i3 | i4 | i5 | i6 | i7) & 0x80) != 0;

    // temp contains temporary result, some zeroing needs to be done
    bool zeroing_pending = false;

    // partially finished result
    __m128i temp;

    if ((m1 & 0x88888888 & mz) == 0) {
        // no elements from b
        return permute8s <i0, i1, i2, i3, i4, i5, i6, i7> (a);
    }

    if (((m1^0x88888888) & 0x88888888 & mz) == 0) {
        // no elements from a
        return permute8s <i0&~8, i1&~8, i2&~8, i3&~8, i4&~8, i5&~8, i6&~8, i7&~8> (b);
    }

    // special case: PUNPCKLWD 
    if (((m1 ^ 0xB3A29180) & mz) == 0) {
        temp = _mm_unpacklo_epi16(a, b);
        if (do_zero) zeroing_pending = true; else return temp;
    }
    if (((m1 ^ 0x3B2A1908) & mz) == 0) {
        temp = _mm_unpacklo_epi16(b, a);
        if (do_zero) zeroing_pending = true; else return temp;
    }
    // special case: PUNPCKHWD 
    if (((m1 ^ 0xF7E6D5C4) & mz) == 0) {
        temp = _mm_unpackhi_epi16(a, b);
        if (do_zero) zeroing_pending = true; else return temp;
    }
    if (((m1 ^ 0x7F6E5D4C) & mz) == 0) {
        temp = _mm_unpackhi_epi16(b, a);
        if (do_zero) zeroing_pending = true; else return temp;
    }

#if  INSTRSET >= 4  // SSSE3
    // special case: shift left
    if (i0 > 0 && i0 < 8 && ((m1 ^ ((i0 & 7) * 0x11111111u + 0x76543210u)) & mz) == 0) {
        temp = _mm_alignr_epi8(b, a, (i0 & 7) * 2);
        if (do_zero) zeroing_pending = true; else return temp;
    }

    // special case: shift right
    if (i0 > 8 && i0 < 16 && ((m1 ^ 0x88888888 ^ ((i0 & 7) * 0x11111111u + 0x76543210u)) & mz) == 0) {
        temp = _mm_alignr_epi8(a, b, (i0 & 7) * 2);
        if (do_zero) zeroing_pending = true; else return temp;
    }
#endif // SSSE3

#if INSTRSET >= 5   // SSE4.1 supported
    // special case: blending without permuting
    if ((((m1 & ~0x88888888) ^ 0x76543210) & mz) == 0) {
        temp = _mm_blend_epi16(a, b, (i0>>3&1) | (i1>>3&1)<<1 | (i2>>3&1)<<2 | (i3>>3&1)<<3 
            | (i4>>3&1)<<4 | (i5>>3&1)<<5 | (i6>>3&1)<<6 | (i7>>3&1)<<7);
        if (do_zero) zeroing_pending = true; else return temp;
    }
#endif // SSE4.1

    if (zeroing_pending) {
        // additional zeroing of temp needed
        __m128i maskz = constant4i < 
            (i0 < 0 ? 0 : 0xFFFF) | (i1 < 0 ? 0 : 0xFFFF0000) ,
            (i2 < 0 ? 0 : 0xFFFF) | (i3 < 0 ? 0 : 0xFFFF0000) ,
            (i4 < 0 ? 0 : 0xFFFF) | (i5 < 0 ? 0 : 0xFFFF0000) ,
            (i6 < 0 ? 0 : 0xFFFF) | (i7 < 0 ? 0 : 0xFFFF0000) > ();
        return _mm_and_si128(temp, maskz);
    }        

    // general case
#ifdef __XOP__     // Use AMD XOP instruction PPERM
    __m128i mask = constant4i <
        (i0 < 0 ? 0x8080 : (i0*2 & 31) | ((i0*2 & 31)+1)<<8) | (i1 < 0 ? 0x80800000 : ((i1*2 & 31)<<16) | ((i1*2 & 31)+1)<<24),
        (i2 < 0 ? 0x8080 : (i2*2 & 31) | ((i2*2 & 31)+1)<<8) | (i3 < 0 ? 0x80800000 : ((i3*2 & 31)<<16) | ((i3*2 & 31)+1)<<24),
        (i4 < 0 ? 0x8080 : (i4*2 & 31) | ((i4*2 & 31)+1)<<8) | (i5 < 0 ? 0x80800000 : ((i5*2 & 31)<<16) | ((i5*2 & 31)+1)<<24),
        (i6 < 0 ? 0x8080 : (i6*2 & 31) | ((i6*2 & 31)+1)<<8) | (i7 < 0 ? 0x80800000 : ((i7*2 & 31)<<16) | ((i7*2 & 31)+1)<<24) > ();
    return _mm_perm_epi8(a, b, mask);
#else  
    // combine two permutes
    __m128i a1 = permute8s <
        (uint32_t)i0 < 8 ? i0 : -1,
        (uint32_t)i1 < 8 ? i1 : -1,
        (uint32_t)i2 < 8 ? i2 : -1,
        (uint32_t)i3 < 8 ? i3 : -1,
        (uint32_t)i4 < 8 ? i4 : -1,
        (uint32_t)i5 < 8 ? i5 : -1,
        (uint32_t)i6 < 8 ? i6 : -1,
        (uint32_t)i7 < 8 ? i7 : -1 > (a);
    __m128i b1 = permute8s <
        (uint32_t)(i0^8) < 8 ? (i0^8) : -1,
        (uint32_t)(i1^8) < 8 ? (i1^8) : -1,
        (uint32_t)(i2^8) < 8 ? (i2^8) : -1,
        (uint32_t)(i3^8) < 8 ? (i3^8) : -1,
        (uint32_t)(i4^8) < 8 ? (i4^8) : -1,
        (uint32_t)(i5^8) < 8 ? (i5^8) : -1,
        (uint32_t)(i6^8) < 8 ? (i6^8) : -1,
        (uint32_t)(i7^8) < 8 ? (i7^8) : -1 > (b);
    return   _mm_or_si128(a1,b1);

#endif
}

template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
static inline Vec8us blend8us(Vec8us const & a, Vec8us const & b) {
    return Vec8us(blend8s<i0,i1,i2,i3,i4,i5,i6,i7> (a,b));
}

template <int i0, int i1, int i2, int i3>
static inline Vec4i blend4i(Vec4i const & a, Vec4i const & b) {

    // Combine all the indexes into a single bitfield, with 8 bits for each
    const int m1 = (i0 & 7) | (i1 & 7) << 8 | (i2 & 7) << 16 | (i3 & 7) << 24; 

    // Mask to zero out negative indexes
    const int mz = (i0 < 0 ? 0 : 0xFF) | (i1 < 0 ? 0 : 0xFF) << 8 | (i2 < 0 ? 0 : 0xFF) << 16 | (i3 < 0 ? 0 : 0xFF) << 24;

    // Some elements must be set to zero
    const bool do_zero = (mz != -1) && ((i0 | i1 | i2 | i3) & 0x80) != 0;

    // temp contains temporary result, some zeroing needs to be done
    bool zeroing_pending = false;

    // partially finished result
    __m128i temp;
#if defined (_MSC_VER) || defined (__clang__)
    temp = a;  // avoid spurious warning message for temp unused
#endif

    // special case: no elements from b
    if ((m1 & 0x04040404 & mz) == 0) {
        return permute4i<i0,i1,i2,i3>(a);
    }

    // special case: no elements from a
    if (((m1^0x04040404) & 0x04040404 & mz) == 0) {
        return permute4i<i0&~4, i1&~4, i2&~4, i3&~4>(b);
    }

    // special case: PUNPCKLDQ
    if (((m1 ^ 0x05010400) & mz) == 0) {
        temp = _mm_unpacklo_epi32(a, b);
        if (do_zero) zeroing_pending = true; else return temp;
    }
    if (((m1 ^ 0x01050004) & mz) == 0) {
        temp = _mm_unpacklo_epi32(b, a);
        if (do_zero) zeroing_pending = true; else return temp;
    }

    // special case: PUNPCKHDQ 
    if (((m1 ^ 0x07030602) & mz) == 0) {
        temp = _mm_unpackhi_epi32(a, b);
        if (do_zero) zeroing_pending = true; else return temp;
    }
    if (((m1 ^ 0x03070206) & mz) == 0) {
        temp = _mm_unpackhi_epi32(b, a);
        if (do_zero) zeroing_pending = true; else return temp;
    }

#if  INSTRSET >= 4  // SSSE3
    // special case: shift left
    if (i0 > 0 && i0 < 4 && ((m1 ^ ((i0 & 3) * 0x01010101u + 0x03020100u)) & mz) == 0) {
        temp = _mm_alignr_epi8(b, a, (i0 & 3) * 4);
        if (do_zero) zeroing_pending = true; else return temp;
    }

    // special case: shift right
    if (i0 > 4 && i0 < 8 && ((m1 ^ 0x04040404 ^ ((i0 & 3) * 0x01010101u + 0x03020100u)) & mz) == 0) {
        temp = _mm_alignr_epi8(a, b, (i0 & 3) * 4);
        if (do_zero) zeroing_pending = true; else return temp;
    }
#endif // SSSE3

#if INSTRSET >= 5   // SSE4.1 supported
    if ((((m1 & ~0x04040404) ^ 0x03020100) & mz) == 0) {
        // blending without permuting
        temp = _mm_blend_epi16(a, b, ((i0>>2)&1)*3 | ((((i1>>2)&1)*3)<<2) | ((((i2>>2)&1)*3)<<4) | ((((i3>>2)&1)*3)<<6));
        if (do_zero) zeroing_pending = true; else return temp;
    }
#endif // SSE4.1

    if (zeroing_pending) {
        // additional zeroing of temp needed
        __m128i maskz = constant4i < (i0 < 0 ? 0 : -1), (i1 < 0 ? 0 : -1), (i2 < 0 ? 0 : -1), (i3 < 0 ? 0 : -1) > ();
        return _mm_and_si128(temp, maskz);
    }        

    // general case
#ifdef __XOP__     // Use AMD XOP instruction PPERM
    __m128i mask = constant4i <
        i0 < 0 ? 0x80808080 : (i0*4 & 31) + (((i0*4 & 31) + 1) << 8) + (((i0*4 & 31) + 2) << 16) + (((i0*4 & 31) + 3) << 24),
        i1 < 0 ? 0x80808080 : (i1*4 & 31) + (((i1*4 & 31) + 1) << 8) + (((i1*4 & 31) + 2) << 16) + (((i1*4 & 31) + 3) << 24),
        i2 < 0 ? 0x80808080 : (i2*4 & 31) + (((i2*4 & 31) + 1) << 8) + (((i2*4 & 31) + 2) << 16) + (((i2*4 & 31) + 3) << 24),
        i3 < 0 ? 0x80808080 : (i3*4 & 31) + (((i3*4 & 31) + 1) << 8) + (((i3*4 & 31) + 2) << 16) + (((i3*4 & 31) + 3) << 24) > ();
    return _mm_perm_epi8(a, b, mask);

#else  // combine two permutes
    __m128i a1 = permute4i <
        (uint32_t)i0 < 4 ? i0 : -1,
        (uint32_t)i1 < 4 ? i1 : -1,
        (uint32_t)i2 < 4 ? i2 : -1,
        (uint32_t)i3 < 4 ? i3 : -1  > (a);
    __m128i b1 = permute4i <
        (uint32_t)(i0^4) < 4 ? (i0^4) : -1,
        (uint32_t)(i1^4) < 4 ? (i1^4) : -1,
        (uint32_t)(i2^4) < 4 ? (i2^4) : -1,
        (uint32_t)(i3^4) < 4 ? (i3^4) : -1  > (b);
    return  _mm_or_si128(a1,b1);
#endif
}

template <int i0, int i1, int i2, int i3>
static inline Vec4ui blend4ui(Vec4ui const & a, Vec4ui const & b) {
    return Vec4ui (blend4i<i0,i1,i2,i3> (a,b));
}

template <int i0, int i1>
static inline Vec2q blend2q(Vec2q const & a, Vec2q const & b) {

    // Combine all the indexes into a single bitfield, with 8 bits for each
    const int m1 = (i0&3) | (i1&3)<<8; 

    // Mask to zero out negative indexes
    const int mz = (i0 < 0 ? 0 : 0xFF) | (i1 < 0 ? 0 : 0xFF) << 8;

    // no elements from b
    if ((m1 & 0x0202 & mz) == 0) {
        return permute2q <i0, i1> (a);
    }
    // no elements from a
    if (((m1^0x0202) & 0x0202 & mz) == 0) {
        return permute2q <i0 & ~2, i1 & ~2> (b);
    }
    // (all cases where one index is -1 or -256 would go to the above cases)

    // special case: PUNPCKLQDQ 
    if (i0 == 0 && i1 == 2) {
        return _mm_unpacklo_epi64(a, b);
    }
    if (i0 == 2 && i1 == 0) {
        return _mm_unpacklo_epi64(b, a);
    }
    // special case: PUNPCKHQDQ 
    if (i0 == 1 && i1 == 3) {
        return _mm_unpackhi_epi64(a, b);
    }
    if (i0 == 3 && i1 == 1) {
        return _mm_unpackhi_epi64(b, a);
    }

#if  INSTRSET >= 4  // SSSE3
    // special case: shift left
    if (i0 == 1 && i1 == 2) {
        return _mm_alignr_epi8(b, a, 8);
    }
    // special case: shift right
    if (i0 == 3 && i1 == 0) {
        return _mm_alignr_epi8(a, b, 8);
    }
#endif // SSSE3

#if INSTRSET >= 5   // SSE4.1 supported
    if (((m1 & ~0x0202) ^ 0x0100) == 0 && mz == 0xFFFF) {
        // blending without permuting
        return _mm_blend_epi16(a, b, (i0>>1 & 1) * 0xF | ((i1>>1 & 1) * 0xF) << 4 );
    }
#endif // SSE4.1

    // general case. combine two permutes 
    // (all cases are caught by the above special cases if SSE4.1 or higher is supported)
    __m128i a1, b1;
    a1 = permute2q <(uint32_t)i0 < 2 ? i0 : -1, (uint32_t)i1 < 2 ? i1 : -1 > (a);
    b1 = permute2q <(uint32_t)(i0^2) < 2 ? (i0^2) : -1, (uint32_t)(i1^2) < 2 ? (i1^2) : -1 > (b);
    return  _mm_or_si128(a1,b1);
}

template <int i0, int i1>
static inline Vec2uq blend2uq(Vec2uq const & a, Vec2uq const & b) {
    return Vec2uq (blend2q <i0, i1> ((__m128i)a, (__m128i)b));
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
* Vec4i a(2,0,0,3);           // index a is (  2,   0,   0,   3)
* Vec4i b(100,101,102,103);   // table b is (100, 101, 102, 103)
* Vec4i c;
* c = lookup4 (a,b);          // c is (102, 100, 100, 103)
*
*****************************************************************************/

static inline Vec16c lookup16(Vec16c const & index, Vec16c const & table) {
#if INSTRSET >= 5  // SSSE3
    return _mm_shuffle_epi8(table, index);
#else
    uint8_t ii[16];
    int8_t  tt[16], rr[16];
    table.store(tt);  index.store(ii);
    for (int j = 0; j < 16; j++) rr[j] = tt[ii[j] & 0x0F];
    return Vec16c().load(rr);
#endif
}

static inline Vec16c lookup32(Vec16c const & index, Vec16c const & table0, Vec16c const & table1) {
#ifdef __XOP__  // AMD XOP instruction set. Use VPPERM
    return _mm_perm_epi8(table0, table1, index);
#elif INSTRSET >= 5  // SSSE3
    Vec16c r0 = _mm_shuffle_epi8(table0, index + 0x70);           // make negative index for values >= 16
    Vec16c r1 = _mm_shuffle_epi8(table1, (index ^ 0x10) + 0x70);  // make negative index for values <  16
    return r0 | r1;
#else
    uint8_t ii[16];
    int8_t  tt[16], rr[16];
    table0.store(tt);  table1.store(tt+16);  index.store(ii);
    for (int j = 0; j < 16; j++) rr[j] = tt[ii[j] & 0x1F];
    return Vec16c().load(rr);
#endif
}

template <int n>
static inline Vec16c lookup(Vec16c const & index, void const * table) {
    if (n <=  0) return 0;
    if (n <= 16) return lookup16(index, Vec16c().load(table));
    if (n <= 32) return lookup32(index, Vec16c().load(table), Vec16c().load((int8_t*)table + 16));
    // n > 32. Limit index
    Vec16uc index1;
    if ((n & (n-1)) == 0) {
        // n is a power of 2, make index modulo n
        index1 = Vec16uc(index) & uint8_t(n-1);
    }
    else {
        // n is not a power of 2, limit to n-1
        index1 = min(Vec16uc(index), uint8_t(n-1));
    }
    uint8_t ii[16];  index1.store(ii);
    int8_t  rr[16];
    for (int j = 0; j < 16; j++) {
        rr[j] = ((int8_t*)table)[ii[j]];
    }
    return Vec16c().load(rr);
}

static inline Vec8s lookup8(Vec8s const & index, Vec8s const & table) {
#if INSTRSET >= 5  // SSSE3
    return _mm_shuffle_epi8(table, index * 0x202 + 0x100);
#else
    int16_t ii[8], tt[8], rr[8];
    table.store(tt);  index.store(ii);
    for (int j = 0; j < 8; j++) rr[j] = tt[ii[j] & 0x07];
    return Vec8s().load(rr);
#endif
}

static inline Vec8s lookup16(Vec8s const & index, Vec8s const & table0, Vec8s const & table1) {
#ifdef __XOP__  // AMD XOP instruction set. Use VPPERM
    return _mm_perm_epi8(table0, table1, index * 0x202 + 0x100);
#elif INSTRSET >= 5  // SSSE3
    Vec8s r0 = _mm_shuffle_epi8(table0, Vec16c(index * 0x202) + Vec16c(Vec8s(0x7170)));
    Vec8s r1 = _mm_shuffle_epi8(table1, Vec16c(index * 0x202 ^ 0x1010) + Vec16c(Vec8s(0x7170)));
    return r0 | r1;
#else
    int16_t ii[16], tt[32], rr[16];
    table0.store(tt);  table1.store(tt+8);  index.store(ii);
    for (int j = 0; j < 16; j++) rr[j] = tt[ii[j] & 0x1F];
    return Vec8s().load(rr);
#endif
}

template <int n>
static inline Vec8s lookup(Vec8s const & index, void const * table) {
    if (n <=  0) return 0;
    if (n <=  8) return lookup8 (index, Vec8s().load(table));
    if (n <= 16) return lookup16(index, Vec8s().load(table), Vec8s().load((int16_t*)table + 8));
    // n > 16. Limit index
    Vec8us index1;
    if ((n & (n-1)) == 0) {
        // n is a power of 2, make index modulo n
        index1 = Vec8us(index) & (n-1);
    }
    else {
        // n is not a power of 2, limit to n-1
        index1 = min(Vec8us(index), n-1);
    }
#if INSTRSET >= 8 // AVX2. Use VPERMD
    Vec8s t1 = _mm_i32gather_epi32((const int *)table, __m128i((Vec4i(index1)) & (Vec4i(0x0000FFFF))), 2);  // even positions
    Vec8s t2 = _mm_i32gather_epi32((const int *)table, _mm_srli_epi32(index1, 16) , 2);  // odd  positions
    return blend8s<0,8,2,10,4,12,6,14>(t1, t2);
#else
    uint16_t ii[8];  index1.store(ii);
    return Vec8s(((int16_t*)table)[ii[0]], ((int16_t*)table)[ii[1]], ((int16_t*)table)[ii[2]], ((int16_t*)table)[ii[3]],
                 ((int16_t*)table)[ii[4]], ((int16_t*)table)[ii[5]], ((int16_t*)table)[ii[6]], ((int16_t*)table)[ii[7]]);
#endif
}


static inline Vec4i lookup4(Vec4i const & index, Vec4i const & table) {
#if INSTRSET >= 5  // SSSE3
    return _mm_shuffle_epi8(table, index * 0x04040404 + 0x03020100);
#else
    return Vec4i(table[index[0]],table[index[1]],table[index[2]],table[index[3]]);
#endif
}

static inline Vec4i lookup8(Vec4i const & index, Vec4i const & table0, Vec4i const & table1) {
    // return Vec4i(lookup16(Vec8s(index * 0x20002 + 0x10000), Vec8s(table0), Vec8s(table1)));
#ifdef __XOP__  // AMD XOP instruction set. Use VPPERM
    return _mm_perm_epi8(table0, table1, index * 0x04040404 + 0x03020100);
#elif INSTRSET >= 8 // AVX2. Use VPERMD
    __m256i table01 = _mm256_inserti128_si256(_mm256_castsi128_si256(table0), table1, 1); // join tables into 256 bit vector

#if defined (_MSC_VER) && _MSC_VER < 1700 && ! defined(__INTEL_COMPILER)
    // bug in MS VS 11 beta: operands in wrong order
    return _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_castsi128_si256(index), table01));
#elif defined (GCC_VERSION) && GCC_VERSION <= 40700 && !defined(__INTEL_COMPILER) && !defined(__clang__)
    // Gcc 4.7.0 also has operands in wrong order
    return _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_castsi128_si256(index), table01));
#else
    return _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(table01, _mm256_castsi128_si256(index)));
#endif // bug

#elif INSTRSET >= 4  // SSSE3
    Vec4i r0 = _mm_shuffle_epi8(table0, Vec16c(index * 0x04040404) + Vec16c(Vec4i(0x73727170)));
    Vec4i r1 = _mm_shuffle_epi8(table1, Vec16c(index * 0x04040404 ^ 0x10101010) + Vec16c(Vec4i(0x73727170)));
    return r0 | r1;
#else    // SSE2
    int32_t ii[4], tt[8], rr[4];
    table0.store(tt);  table1.store(tt+4);  index.store(ii);
    for (int j = 0; j < 4; j++) rr[j] = tt[ii[j] & 0x07];
    return Vec4i().load(rr);
#endif
}

static inline Vec4i lookup16(Vec4i const & index, Vec4i const & table0, Vec4i const & table1, Vec4i const & table2, Vec4i const & table3) {
#if INSTRSET >= 8 // AVX2. Use VPERMD
    __m256i table01 = _mm256_inserti128_si256(_mm256_castsi128_si256(table0), table1, 1); // join tables into 256 bit vector
    __m256i table23 = _mm256_inserti128_si256(_mm256_castsi128_si256(table2), table3, 1); // join tables into 256 bit vector
#if defined (_MSC_VER) && _MSC_VER < 1700 && ! defined(__INTEL_COMPILER)
    // bug in MS VS 11 beta: operands in wrong order
    __m128i r0 = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_castsi128_si256(index    ), table01));
    __m128i r1 = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_castsi128_si256(index ^ 8), table23));
#elif defined (GCC_VERSION) && GCC_VERSION <= 40700 && !defined(__INTEL_COMPILER) && !defined(__clang__)
    // Gcc 4.7.0 also has operands in wrong order
    __m128i r0 = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_castsi128_si256(index    ), table01));
    __m128i r1 = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_castsi128_si256(index ^ 8), table23));
#else
    __m128i r0 = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(table01, _mm256_castsi128_si256(index)));
    __m128i r1 = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(table23, _mm256_castsi128_si256(index ^ 8)));
#endif // bug
    return _mm_blendv_epi8(r0, r1, index > 8);

#elif defined (__XOP__)  // AMD XOP instruction set. Use VPPERM
    Vec4i r0 = _mm_perm_epi8(table0, table1, ((index    ) * 0x04040404u + 0x63626160u) & 0X9F9F9F9Fu);
    Vec4i r1 = _mm_perm_epi8(table2, table3, ((index ^ 8) * 0x04040404u + 0x63626160u) & 0X9F9F9F9Fu);
    return r0 | r1;

#elif INSTRSET >= 5  // SSSE3
    Vec16c aa = Vec16c(Vec4i(0x73727170));
    Vec4i r0 = _mm_shuffle_epi8(table0, Vec16c((index     ) * 0x04040404) + aa);
    Vec4i r1 = _mm_shuffle_epi8(table1, Vec16c((index ^  4) * 0x04040404) + aa);
    Vec4i r2 = _mm_shuffle_epi8(table2, Vec16c((index ^  8) * 0x04040404) + aa);
    Vec4i r3 = _mm_shuffle_epi8(table3, Vec16c((index ^ 12) * 0x04040404) + aa);
    return (r0 | r1) | (r2 | r3);

#else    // SSE2
    int32_t ii[4], tt[16], rr[4];
    table0.store(tt);  table1.store(tt+4);  table2.store(tt+8);  table3.store(tt+12);
    index.store(ii);
    for (int j = 0; j < 4; j++) rr[j] = tt[ii[j] & 0x0F];
    return Vec4i().load(rr);
#endif
}

template <int n>
static inline Vec4i lookup(Vec4i const & index, void const * table) {
    if (n <= 0) return 0;
    if (n <= 4) return lookup4(index, Vec4i().load(table));
    if (n <= 8) return lookup8(index, Vec4i().load(table), Vec4i().load((int32_t*)table + 4));
    // n > 8. Limit index
    Vec4ui index1;
    if ((n & (n-1)) == 0) {
        // n is a power of 2, make index modulo n
        index1 = Vec4ui(index) & (n-1);
    }
    else {
        // n is not a power of 2, limit to n-1
        index1 = min(Vec4ui(index), n-1);
    }
#if INSTRSET >= 8 // AVX2. Use VPERMD
    return _mm_i32gather_epi32((const int *)table, index1, 4);
#else
    uint32_t ii[4];  index1.store(ii);
    return Vec4i(((int32_t*)table)[ii[0]], ((int32_t*)table)[ii[1]], ((int32_t*)table)[ii[2]], ((int32_t*)table)[ii[3]]);
#endif
}


static inline Vec2q lookup2(Vec2q const & index, Vec2q const & table) {
#if INSTRSET >= 5  // SSSE3
    return _mm_shuffle_epi8(table, index * 0x0808080808080808ll + 0x0706050403020100ll);
#else
    int64_t ii[2], tt[2];
    table.store(tt);  index.store(ii);
    return Vec2q(tt[int(ii[0])], tt[int(ii[1])]);
#endif
}

template <int n>
static inline Vec2q lookup(Vec2q const & index, void const * table) {
    if (n <= 0) return 0;
    // n > 0. Limit index
    Vec2uq index1;
    if ((n & (n-1)) == 0) {
        // n is a power of 2, make index modulo n
        index1 = Vec2uq(index) & (n-1);
    }
    else {
        // n is not a power of 2, limit to n-1.
        // There is no 64-bit min instruction, but we can use the 32-bit unsigned min,
        // since n is a 32-bit integer
        index1 = Vec2uq(min(Vec2uq(index), constant4i<n-1, 0, n-1, 0>()));
    }
    uint32_t ii[4];  index1.store(ii);  // use only lower 32 bits of each index
    int64_t const * tt = (int64_t const *)table;
    return Vec2q(tt[ii[0]], tt[ii[2]]);
}


/*****************************************************************************
*
*          Other permutations with variable indexes
*
*****************************************************************************/

// Function shift_bytes_up: shift whole vector left by b bytes.
// You may use a permute function instead if b is a compile-time constant
static inline Vec16c shift_bytes_up(Vec16c const & a, int b) {
    if ((uint32_t)b > 15) return _mm_setzero_si128();
#if INSTRSET >= 4    // SSSE3
    static const char mask[32] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};    
    return Vec16c(_mm_shuffle_epi8(a, Vec16c().load(mask+16-b)));
#else
    Vec2uq a1 = Vec2uq(a);
    if (b < 8) {    
        a1 = (a1 << (b*8)) | (permute2uq<-1,0>(a1) >> (64 - (b*8)));
    }
    else {
        a1 = permute2uq<-1,0>(a1) << ((b-8)*8);
    }
    return Vec16c(a1);
#endif
}

// Function shift_bytes_down: shift whole vector right by b bytes
// You may use a permute function instead if b is a compile-time constant
static inline Vec16c shift_bytes_down(Vec16c const & a, int b) {
    if ((uint32_t)b > 15) return _mm_setzero_si128();
#if INSTRSET >= 4    // SSSE3
    static const char mask[32] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
    return Vec16c(_mm_shuffle_epi8(a, Vec16c().load(mask+b)));
#else
    Vec2uq a1 = Vec2uq(a);
    if (b < 8) {    
        a1 = (a1 >> (b*8)) | (permute2uq<1,-1>(a1) << (64 - (b*8)));
    }
    else {
        a1 = permute2uq<1,-1>(a1) >> ((b-8)*8); 
    }
    return Vec16c(a1);
#endif
}

/*****************************************************************************
*
*          Gather functions with fixed indexes
*
*****************************************************************************/
// Load elements from array a with indices i0, i1, i2, i3
template <int i0, int i1, int i2, int i3>
static inline Vec4i gather4i(void const * a) {
    Static_error_check<(i0|i1|i2|i3)>=0> Negative_array_index;  // Error message if index is negative
    const int i01min = i0 < i1 ? i0 : i1;
    const int i23min = i2 < i3 ? i2 : i3;
    const int imin   = i01min < i23min ? i01min : i23min;
    const int i01max = i0 > i1 ? i0 : i1;
    const int i23max = i2 > i3 ? i2 : i3;
    const int imax   = i01max > i23max ? i01max : i23max;
    if (imax - imin <= 3) {
        // load one contiguous block and permute
        if (imax > 3) {
            // make sure we don't read past the end of the array
            Vec4i b = Vec4i().load((int32_t const *)a + imax-3);
            return permute4i<i0-imax+3, i1-imax+3, i2-imax+3, i3-imax+3>(b);
        }
        else {
            Vec4i b = Vec4i().load((int32_t const *)a + imin);
            return permute4i<i0-imin, i1-imin, i2-imin, i3-imin>(b);
        }
    }
    if ((i0<imin+4 || i0>imax-4) && (i1<imin+4 || i1>imax-4) && (i2<imin+4 || i2>imax-4) && (i3<imin+4 || i3>imax-4)) {
        // load two contiguous blocks and blend
        Vec4i b = Vec4i().load((int32_t const *)a + imin);
        Vec4i c = Vec4i().load((int32_t const *)a + imax-3);
        const int j0 = i0<imin+4 ? i0-imin : 7-imax+i0;
        const int j1 = i1<imin+4 ? i1-imin : 7-imax+i1;
        const int j2 = i2<imin+4 ? i2-imin : 7-imax+i2;
        const int j3 = i3<imin+4 ? i3-imin : 7-imax+i3;
        return blend4i<j0, j1, j2, j3>(b, c);
    }
    // use AVX2 gather if available
#if INSTRSET >= 8
    return _mm_i32gather_epi32((const int *)a, Vec4i(i0,i1,i2,i3), 4);
#else
    return lookup<imax+1>(Vec4i(i0,i1,i2,i3), a);
#endif
}

// Load elements from array a with indices i0, i1
template <int i0, int i1>
static inline Vec2q gather2q(void const * a) {
    Static_error_check<(i0|i1)>=0> Negative_array_index;  // Error message if index is negative
    const int imin = i0 < i1 ? i0 : i1;
    const int imax = i0 > i1 ? i0 : i1;
    if (imax - imin <= 1) {
        // load one contiguous block and permute
        if (imax > 1) {
            // make sure we don't read past the end of the array
            Vec2q b = Vec2q().load((int64_t const *)a + imax-1);
            return permute2q<i0-imax+1, i1-imax+1>(b);
        }
        else {
            Vec2q b = Vec2q().load((int64_t const *)a + imin);
            return permute2q<i0-imin, i1-imin>(b);
        }
    }
    return Vec2q(((int64_t*)a)[i0], ((int64_t*)a)[i1]);
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
* Vec8q a(10,11,12,13,14,15,16,17);
* int64_t b[16] = {0};
* scatter<0,2,14,10,1,-1,5,9>(a,b); 
* // Now, b = {10,14,11,0,0,16,0,0,0,17,13,0,0,0,12,0}
*
*****************************************************************************/

template <int i0, int i1, int i2, int i3>
static inline void scatter(Vec4i const & data, void * array) {
#if defined (__AVX512VL__)
    __m128i indx = constant4i<i0,i1,i2,i3>();
    __mmask16 mask = uint16_t(i0>=0 | (i1>=0)<<1 | (i2>=0)<<2 | (i3>=0)<<3);
    _mm_mask_i32scatter_epi32((int*)array, mask, indx, data, 4);
#else
    int32_t* arr = (int32_t*)array;
    const int index[4] = {i0,i1,i2,i3};
    for (int i = 0; i < 4; i++) {
        if (index[i] >= 0) arr[index[i]] = data[i];
    }
#endif
}

template <int i0, int i1>
static inline void scatter(Vec2q const & data, void * array) {
    int64_t* arr = (int64_t*)array;
    if (i0 >= 0) arr[i0] = data[0];
    if (i1 >= 0) arr[i1] = data[1];
}

static inline void scatter(Vec4i const & index, uint32_t limit, Vec4i const & data, void * array) {
#if defined (__AVX512VL__)
    __mmask16 mask = _mm_cmplt_epu32_mask(index, Vec4ui(limit));
    _mm_mask_i32scatter_epi32((int*)array, mask, index, data, 4);
#else
    int32_t* arr = (int32_t*)array;
    for (int i = 0; i < 4; i++) {
        if (uint32_t(index[i]) < limit) arr[index[i]] = data[i];
    }
#endif
}

static inline void scatter(Vec2q const & index, uint32_t limit, Vec2q const & data, void * array) {
    int64_t* arr = (int64_t*)array;
    if (uint64_t(index[0]) < uint64_t(limit)) arr[index[0]] = data[0];
    if (uint64_t(index[1]) < uint64_t(limit)) arr[index[1]] = data[1];
} 

static inline void scatter(Vec4i const & index, uint32_t limit, Vec2q const & data, void * array) {
    int64_t* arr = (int64_t*)array;
    if (uint32_t(index[0]) < limit) arr[index[0]] = data[0];
    if (uint32_t(index[1]) < limit) arr[index[1]] = data[1];
} 

/*****************************************************************************
*
*          Functions for conversion between integer sizes
*
*****************************************************************************/

// Extend 8-bit integers to 16-bit integers, signed and unsigned

// Function extend_low : extends the low 8 elements to 16 bits with sign extension
static inline Vec8s extend_low (Vec16c const & a) {
    __m128i sign = _mm_cmpgt_epi8(_mm_setzero_si128(),a);  // 0 > a
    return         _mm_unpacklo_epi8(a,sign);              // interleave with sign extensions
}

// Function extend_high : extends the high 8 elements to 16 bits with sign extension
static inline Vec8s extend_high (Vec16c const & a) {
    __m128i sign = _mm_cmpgt_epi8(_mm_setzero_si128(),a);  // 0 > a
    return         _mm_unpackhi_epi8(a,sign);              // interleave with sign extensions
}

// Function extend_low : extends the low 8 elements to 16 bits with zero extension
static inline Vec8us extend_low (Vec16uc const & a) {
    return    _mm_unpacklo_epi8(a,_mm_setzero_si128());    // interleave with zero extensions
}

// Function extend_high : extends the high 8 elements to 16 bits with zero extension
static inline Vec8us extend_high (Vec16uc const & a) {
    return    _mm_unpackhi_epi8(a,_mm_setzero_si128());    // interleave with zero extensions
}

// Extend 16-bit integers to 32-bit integers, signed and unsigned

// Function extend_low : extends the low 4 elements to 32 bits with sign extension
static inline Vec4i extend_low (Vec8s const & a) {
    __m128i sign = _mm_srai_epi16(a,15);                   // sign bit
    return         _mm_unpacklo_epi16(a,sign);             // interleave with sign extensions
}

// Function extend_high : extends the high 4 elements to 32 bits with sign extension
static inline Vec4i extend_high (Vec8s const & a) {
    __m128i sign = _mm_srai_epi16(a,15);                   // sign bit
    return         _mm_unpackhi_epi16(a,sign);             // interleave with sign extensions
}

// Function extend_low : extends the low 4 elements to 32 bits with zero extension
static inline Vec4ui extend_low (Vec8us const & a) {
    return    _mm_unpacklo_epi16(a,_mm_setzero_si128());   // interleave with zero extensions
}

// Function extend_high : extends the high 4 elements to 32 bits with zero extension
static inline Vec4ui extend_high (Vec8us const & a) {
    return    _mm_unpackhi_epi16(a,_mm_setzero_si128());   // interleave with zero extensions
}

// Extend 32-bit integers to 64-bit integers, signed and unsigned

// Function extend_low : extends the low 2 elements to 64 bits with sign extension
static inline Vec2q extend_low (Vec4i const & a) {
    __m128i sign = _mm_srai_epi32(a,31);                   // sign bit
    return         _mm_unpacklo_epi32(a,sign);             // interleave with sign extensions
}

// Function extend_high : extends the high 2 elements to 64 bits with sign extension
static inline Vec2q extend_high (Vec4i const & a) {
    __m128i sign = _mm_srai_epi32(a,31);                   // sign bit
    return         _mm_unpackhi_epi32(a,sign);             // interleave with sign extensions
}

// Function extend_low : extends the low 2 elements to 64 bits with zero extension
static inline Vec2uq extend_low (Vec4ui const & a) {
    return    _mm_unpacklo_epi32(a,_mm_setzero_si128());   // interleave with zero extensions
}

// Function extend_high : extends the high 2 elements to 64 bits with zero extension
static inline Vec2uq extend_high (Vec4ui const & a) {
    return    _mm_unpackhi_epi32(a,_mm_setzero_si128());   // interleave with zero extensions
}

// Compress 16-bit integers to 8-bit integers, signed and unsigned, with and without saturation

// Function compress : packs two vectors of 16-bit integers into one vector of 8-bit integers
// Overflow wraps around
static inline Vec16c compress (Vec8s const & low, Vec8s const & high) {
    return nsimd_common::compress16(low, high);
}

// Function compress : packs two vectors of 16-bit integers into one vector of 8-bit integers
// Signed, with saturation
static inline Vec16c compress_saturated (Vec8s const & low, Vec8s const & high) {
    return nsimd_common::compress16(low, high, true);
}

// Function compress : packs two vectors of 16-bit integers to one vector of 8-bit integers
// Unsigned, overflow wraps around
static inline Vec16uc compress (Vec8us const & low, Vec8us const & high) {
    return  nsimd_common::compress16(low, high);
}

// Function compress : packs two vectors of 16-bit integers into one vector of 8-bit integers
// Unsigned, with saturation
static inline Vec16uc compress_saturated (Vec8us const & low, Vec8us const & high) {
    return nsimd_common::compress16(low, high, true);
}

// Compress 32-bit integers to 16-bit integers, signed and unsigned, with and without saturation

// Function compress : packs two vectors of 32-bit integers into one vector of 16-bit integers
// Overflow wraps around
static inline Vec8s compress (Vec4i const & low, Vec4i const & high) {
    return nsimd_common::compress8(low, high);
}

// Function compress : packs two vectors of 32-bit integers into one vector of 16-bit integers
// Signed with saturation
static inline Vec8s compress_saturated (Vec4i const & low, Vec4i const & high) {
    return nsimd_common::compress8<pack128_8i_t, int16_t, pack128_4i_t, int32_t>(low, high, true);
}

// Function compress : packs two vectors of 32-bit integers into one vector of 16-bit integers
// Overflow wraps around
static inline Vec8us compress (Vec4ui const & low, Vec4ui const & high) {
    return nsimd_common::compress8<pack128_8ui_t, uint16_t, pack128_4ui_t, uint32_t>(low, high);
}

// Function compress : packs two vectors of 32-bit integers into one vector of 16-bit integers
// Unsigned, with saturation
static inline Vec8us compress_saturated (Vec4ui const & low, Vec4ui const & high) {
    return nsimd_common::compress8<pack128_8ui_t, uint16_t, pack128_4ui_t, uint32_t>(low, high, true);
}

// Compress 64-bit integers to 32-bit integers, signed and unsigned, with and without saturation

// Function compress : packs two vectors of 64-bit integers into one vector of 32-bit integers
// Overflow wraps around
static inline Vec4i compress (Vec2q const & low, Vec2q const & high) {
    return nsimd_common::compress4<pack128_4i_t, int32_t, pack128_2i_t, int64_t>(low, high);
}

// Function compress : packs two vectors of 64-bit integers into one vector of 32-bit integers
// Signed, with saturation
// This function is very inefficient unless the SSE4.2 instruction set is supported
static inline Vec4i compress_saturated (Vec2q const & low, Vec2q const & high) {
    return nsimd_common::compress4<pack128_4i_t, int32_t, pack128_2i_t, int64_t>(low, high, true);
}

// Function compress : packs two vectors of 32-bit integers into one vector of 16-bit integers
// Overflow wraps around
static inline Vec4ui compress (Vec2uq const & low, Vec2uq const & high) {
    return nsimd_common::compress4<pack128_4ui_t, uint32_t, pack128_2ui_t, uint64_t>(low, high);
}

// Function compress : packs two vectors of 64-bit integers into one vector of 32-bit integers
// Unsigned, with saturation
static inline Vec4ui compress_saturated (Vec2uq const & low, Vec2uq const & high) {
    return nsimd_common::compress4<pack128_4ui_t, uint32_t, pack128_2ui_t, uint64_t>(low, high, true);
}

/*****************************************************************************
*
*          Helper functions for division and bit scan
*
*****************************************************************************/

// Define popcount function. Gives sum of bits
static inline uint32_t vml_popcnt (uint32_t a) {	
    return nsimd_popcnt32_(a);
}


// Define bit-scan-forward function. Gives index to lowest set bit
#if defined (__GNUC__) || defined(__clang__)
static inline uint32_t bit_scan_forward (uint32_t a) __attribute__ ((pure));
static inline uint32_t bit_scan_forward (uint32_t a) {	
    uint32_t r;
    __asm("bsfl %1, %0" : "=r"(r) : "r"(a) : );
    return r;
}
#else
static inline uint32_t bit_scan_forward (uint32_t a) {	
    unsigned long r;
    _BitScanForward(&r, a);                      // defined in intrin.h for MS and Intel compilers
    return r;
}
#endif


// Define bit-scan-reverse function. Gives index to highest set bit = floor(log2(a))
#if defined (__GNUC__) || defined(__clang__)
static inline uint32_t bit_scan_reverse (uint32_t a) __attribute__ ((pure));
static inline uint32_t bit_scan_reverse (uint32_t a) {	
    uint32_t r;
    __asm("bsrl %1, %0" : "=r"(r) : "r"(a) : );
    return r;
}
#else
static inline uint32_t bit_scan_reverse (uint32_t a) {	
    unsigned long r;
    _BitScanReverse(&r, a);                      // defined in intrin.h for MS and Intel compilers
    return r;
}
#endif

// Same function, for compile-time constants.
// We need template metaprogramming for calculating this function at compile time.
// This may take a long time to compile because of the template recursion.
// Todo: replace this with a constexpr function when C++14 becomes available
template <uint32_t n> 
struct BitScanR {
    enum {val = (
        n >= 0x10 ? 4 + (BitScanR<(n>>4)>::val) :
        n  <    2 ? 0 :
        n  <    4 ? 1 :
        n  <    8 ? 2 : 3 )                       };
};
template <> struct BitScanR<0> {enum {val = 0};};          // Avoid infinite template recursion

#define bit_scan_reverse_const(n)  (BitScanR<n>::val)      // n must be a valid compile-time constant


/*****************************************************************************
*
*          Integer division operators
*
******************************************************************************
*/

// vector operator / : divide each element by divisor

// vector of 4 32-bit signed integers
static inline Vec4i operator / (Vec4i const & a, Vec4i const & d) {
    return nsimd::div(a, d);
}

// vector of 4 32-bit unsigned integers
static inline Vec4ui operator / (Vec4ui const & a, Vec4ui const & d) {
    return nsimd::div(a, d);
}

// vector of 8 16-bit signed integers
static inline Vec8s operator / (Vec8s const & a, Vec8s const & d) {
    return nsimd::div(a, d);
}

// vector of 8 16-bit unsigned integers
static inline Vec8us operator / (Vec8us const & a, Vec8us const & d) {
    return nsimd::div(a, d);
}
 
// vector of 16 8-bit signed integers
static inline Vec16c operator / (Vec16c const & a, Vec16c const & d) {
    return nsimd::div(a, d);
}

// vector of 16 8-bit unsigned integers
static inline Vec16uc operator / (Vec16uc const & a, Vec16uc const & d) {
    return nsimd::div(a, d);
}

// vector operator /= : divide
static inline Vec8s & operator /= (Vec8s & a, Vec8s const & d) {
    a = a / d;
    return a;
}

// vector operator /= : divide
static inline Vec8us & operator /= (Vec8us & a, Vec8us const & d) {
    a = a / d;
    return a;
}

// vector operator /= : divide
static inline Vec4i & operator /= (Vec4i & a, Vec4i const & d) {
    a = a / d;
    return a;
}

// vector operator /= : divide
static inline Vec4ui & operator /= (Vec4ui & a, Vec4ui const & d) {
    a = a / d;
    return a;
}

// vector operator /= : divide
static inline Vec16c & operator /= (Vec16c & a, Vec16c const & d) {
    a = a / d;
    return a;
}

// vector operator /= : divide
static inline Vec16uc & operator /= (Vec16uc & a, Vec16uc const & d) {
    a = a / d;
    return a;
}

/*****************************************************************************
*
*          Integer division 2: divisor is a compile-time constant
*
*****************************************************************************/

// Divide Vec4i by compile-time constant
template <int32_t d>
static inline Vec4i divide_by_i(Vec4i const & x) {
    Static_error_check<(d!=0)> Dividing_by_zero;                     // Error message if dividing by zero
    if (d ==  1) return  x;
    if (d == -1) return -x;
    if (uint32_t(d) == 0x80000000u) return Vec4i(x == Vec4i(0x80000000)) & 1; // prevent overflow when changing sign
    pack128_4i_t div = nsimd::set1<pack128_4i_t>(d);
    return nsimd::div(x, div); 
}

// define Vec4i a / const_int(d)
template <int32_t d>
static inline Vec4i operator / (Vec4i const & a, Const_int_t<d>) {
    return divide_by_i<d>(a);
}

// define Vec4i a / const_uint(d)
template <uint32_t d>
static inline Vec4i operator / (Vec4i const & a, Const_uint_t<d>) {
    Static_error_check< (d<0x80000000u) > Error_overflow_dividing_signed_by_unsigned; // Error: dividing signed by overflowing unsigned
    return divide_by_i<int32_t(d)>(a);                               // signed divide
}

// vector operator /= : divide
template <int32_t d>
static inline Vec4i & operator /= (Vec4i & a, Const_int_t<d> b) {
    a = a / b;
    return a;
}

// vector operator /= : divide
template <uint32_t d>
static inline Vec4i & operator /= (Vec4i & a, Const_uint_t<d> b) {
    a = a / b;
    return a;
}


// Divide Vec4ui by compile-time constant
template <uint32_t d>
static inline Vec4ui divide_by_ui(Vec4ui const & x) {
    Static_error_check<(d!=0)> Dividing_by_zero;                     // Error message if dividing by zero
    if (d == 1) return x;                                            // divide by 18_16i_t>(d0);
    pack128_4ui_t div = nsimd::set1<pack128_4ui_t>(d);
    return nsimd::div(x, div);                                       
}

// define Vec4ui a / const_uint(d)
template <uint32_t d>
static inline Vec4ui operator / (Vec4ui const & a, Const_uint_t<d>) {
    return divide_by_ui<d>(a);
}

// define Vec4ui a / const_int(d)
template <int32_t d>
static inline Vec4ui operator / (Vec4ui const & a, Const_int_t<d>) {
    Static_error_check< (d>=0) > Error_dividing_unsigned_by_negative;// Error: dividing unsigned by negative is ambiguous
    return divide_by_ui<d>(a);                                       // unsigned divide
}

// vector operator /= : divide
template <uint32_t d>
static inline Vec4ui & operator /= (Vec4ui & a, Const_uint_t<d> b) {
    a = a / b;
    return a;
}

// vector operator /= : divide
template <int32_t d>
static inline Vec4ui & operator /= (Vec4ui & a, Const_int_t<d> b) {
    a = a / b;
    return a;
}


// Divide Vec8s by compile-time constant 
template <int d>
static inline Vec8s divide_by_i(Vec8s const & x) {
    const int16_t d0 = int16_t(d);                                   // truncate d to 16 bits
    Static_error_check<(d0 != 0)> Dividing_by_zero;                  // Error message if dividing by zero
    if (d0 ==  1) return  x;                                         // divide by  1
    if (d0 == -1) return -x;                                         // divide by -1
    if (uint16_t(d0) == 0x8000u) return Vec8s(x == Vec8s(0x8000)) & 1;// prevent overflow when changing sign
    pack128_8i_t div = nsimd::set1<pack128_8i_t>(d0);
    return nsimd::div(x, div);
}

// define Vec8s a / const_int(d)
template <int d>
static inline Vec8s operator / (Vec8s const & a, Const_int_t<d>) {
    return divide_by_i<d>(a);
}

// define Vec8s a / const_uint(d)
template <uint32_t d>
static inline Vec8s operator / (Vec8s const & a, Const_uint_t<d>) {
    Static_error_check< (d<0x8000u) > Error_overflow_dividing_signed_by_unsigned; // Error: dividing signed by overflowing unsigned
    return divide_by_i<int(d)>(a);                                   // signed divide
}

// vector operator /= : divide
template <int32_t d>
static inline Vec8s & operator /= (Vec8s & a, Const_int_t<d> b) {
    a = a / b;
    return a;
}

// vector operator /= : divide
template <uint32_t d>
static inline Vec8s & operator /= (Vec8s & a, Const_uint_t<d> b) {
    a = a / b;
    return a;
}


// Divide Vec8us by compile-time constant
template <uint32_t d>
static inline Vec8us divide_by_ui(Vec8us const & x) {
    const uint16_t d0 = uint16_t(d);                                 // truncate d to 16 bits
    Static_error_check<(d0 != 0)> Dividing_by_zero;                  // Error message if dividing by zero
    if (d0 == 1) return x;                                           // divide by 1
    pack128_8ui_t div = nsimd::set1<pack128_8ui_t>(d0);
    return nsimd::div(x, div);
}

// define Vec8us a / const_uint(d)
template <uint32_t d>
static inline Vec8us operator / (Vec8us const & a, Const_uint_t<d>) {
    return divide_by_ui<d>(a);
}

// define Vec8us a / const_int(d)
template <int d>
static inline Vec8us operator / (Vec8us const & a, Const_int_t<d>) {
    Static_error_check< (d>=0) > Error_dividing_unsigned_by_negative;// Error: dividing unsigned by negative is ambiguous
    return divide_by_ui<d>(a);                                       // unsigned divide
}

// vector operator /= : divide
template <uint32_t d>
static inline Vec8us & operator /= (Vec8us & a, Const_uint_t<d> b) {
    a = a / b;
    return a;
}

// vector operator /= : divide
template <int32_t d>
static inline Vec8us & operator /= (Vec8us & a, Const_int_t<d> b) {
    a = a / b;
    return a;
}


// define Vec16c a / const_int(d)
template <int d>
static inline Vec16c operator / (Vec16c const & a, Const_int_t<d>) {
    const uint8_t d0 = uint8_t(d);                                 // truncate d to 16 bits
    Static_error_check<(d0 != 0)> Dividing_by_zero;                  // Error message if dividing by zero
    if (d0 == 1) return x;                                           // divide by 1
    pack128_16i_t div = nsimd::set1<pack128_16i_t>(d0);
    return nsimd::div(x, div);
}

// define Vec16c a / const_uint(d)
template <uint32_t d>
static inline Vec16c operator / (Vec16c const & a, Const_uint_t<d>) {
    Static_error_check< (uint8_t(d)<0x80u) > Error_overflow_dividing_signed_by_unsigned; // Error: dividing signed by overflowing unsigned
    return a / Const_int_t<d>();                              // signed divide
}

// vector operator /= : divide
template <int32_t d>
static inline Vec16c & operator /= (Vec16c & a, Const_int_t<d> b) {
    a = a / b;
    return a;
}
// vector operator /= : divide
template <uint32_t d>
static inline Vec16c & operator /= (Vec16c & a, Const_uint_t<d> b) {
    a = a / b;
    return a;
}

// define Vec16uc a / const_uint(d)
template <uint32_t d>
static inline Vec16uc operator / (Vec16uc const & a, Const_uint_t<d>) {
    const uint8_t d0 = uint8_t(d);                                 // truncate d to 16 bits
    Static_error_check<(d0 != 0)> Dividing_by_zero;                  // Error message if dividing by zero
    if (d0 == 1) return x;                                           // divide by 1
    pack128_16ui_t div = nsimd::set1<pack128_16ui_t>(d0);
    return nsimd::div(x, div);
}

// define Vec16uc a / const_int(d)
template <int d>
static inline Vec16uc operator / (Vec16uc const & a, Const_int_t<d>) {
    Static_error_check< (int8_t(d)>=0) > Error_dividing_unsigned_by_negative;// Error: dividing unsigned by negative is ambiguous
    return a / Const_uint_t<d>();                         // unsigned divide
}

// vector operator /= : divide
template <uint32_t d>
static inline Vec16uc & operator /= (Vec16uc & a, Const_uint_t<d> b) {
    a = a / b;
    return a;
}

// vector operator /= : divide
template <int32_t d>
static inline Vec16uc & operator /= (Vec16uc & a, Const_int_t<d> b) {
    a = a / b;
    return a;
}

/*****************************************************************************
*
*          Horizontal scan functions
*
*****************************************************************************/

// Get index to the first element that is true. Return -1 if all are false
static inline int horizontal_find_first(Vec16cb const & x) {
    return nsimd_common::horizontal_find_first(x);
}

static inline int horizontal_find_first(Vec8sb const & x) {
    return nsimd_common::horizontal_find_first(x);
}

static inline int horizontal_find_first(Vec4ib const & x) {
    return nsimd_common::horizontal_find_first(x);
}

static inline int horizontal_find_first(Vec2qb const & x) {
    return nsimd_common::horizontal_find_first(x);
}

// Count the number of elements that are true
static inline uint32_t horizontal_count(Vec16cb const & x) {
    return nsind::nbtrue(x);
}

static inline uint32_t horizontal_count(Vec8sb const & x) {
    return nsind::nbtrue(x);
}

static inline uint32_t horizontal_count(Vec4ib const & x) {
    return nsind::nbtrue(x);
}

static inline uint32_t horizontal_count(Vec2qb const & x) {
    return nsind::nbtrue(x);
}


/*****************************************************************************
*
*          Boolean <-> bitfield conversion functions
*
*****************************************************************************/

// to_bits: convert boolean vector to integer bitfield
static inline uint16_t to_bits(Vec16cb const & x) {
    return (uint16_t)_mm_movemask_epi8(x);
}

// to_Vec16bc: convert integer bitfield to boolean vector
static inline Vec16cb to_Vec16cb(uint16_t x) {
    static const uint32_t table[16] = {  // lookup-table
        0x00000000, 0x000000FF, 0x0000FF00, 0x0000FFFF, 
        0x00FF0000, 0x00FF00FF, 0x00FFFF00, 0x00FFFFFF, 
        0xFF000000, 0xFF0000FF, 0xFF00FF00, 0xFF00FFFF, 
        0xFFFF0000, 0xFFFF00FF, 0xFFFFFF00, 0xFFFFFFFF}; 
    uint32_t a0 = table[x       & 0xF];
    uint32_t a1 = table[(x>>4)  & 0xF];
    uint32_t a2 = table[(x>>8)  & 0xF];
    uint32_t a3 = table[(x>>12) & 0xF];
    return Vec16cb(Vec16c(Vec4ui(a0, a1, a2, a3)));
}

// to_bits: convert boolean vector to integer bitfield
static inline uint8_t to_bits(Vec8sb const & x) {
    __m128i a = _mm_packs_epi16(x, x);  // 16-bit words to bytes
    return (uint8_t)_mm_movemask_epi8(a);
}

// to_Vec8sb: convert integer bitfield to boolean vector
static inline Vec8sb to_Vec8sb(uint8_t x) {
    static const uint32_t table[16] = {  // lookup-table
        0x00000000, 0x000000FF, 0x0000FF00, 0x0000FFFF, 
        0x00FF0000, 0x00FF00FF, 0x00FFFF00, 0x00FFFFFF, 
        0xFF000000, 0xFF0000FF, 0xFF00FF00, 0xFF00FFFF, 
        0xFFFF0000, 0xFFFF00FF, 0xFFFFFF00, 0xFFFFFFFF}; 
    uint32_t a0 = table[x       & 0xF];
    uint32_t a1 = table[(x>>4)  & 0xF];
    Vec4ui   b  = Vec4ui(a0, a1, a0, a1);
    return _mm_unpacklo_epi8(b, b);  // duplicate bytes to 16-bit words
}

#if INSTRSET < 9 || MAX_VECTOR_SIZE < 512
// These functions are defined in Vectori512.h if AVX512 instruction set is used

// to_bits: convert boolean vector to integer bitfield
static inline uint8_t to_bits(Vec4ib const & x) {
    __m128i a = _mm_packs_epi32(x, x);  // 32-bit dwords to 16-bit words
    __m128i b = _mm_packs_epi16(a, a);  // 16-bit words to bytes
    return _mm_movemask_epi8(b) & 0xF;
}

// to_Vec4ib: convert integer bitfield to boolean vector
static inline Vec4ib to_Vec4ib(uint8_t x) {
    static const uint32_t table[16] = {    // lookup-table
        0x00000000, 0x000000FF, 0x0000FF00, 0x0000FFFF, 
        0x00FF0000, 0x00FF00FF, 0x00FFFF00, 0x00FFFFFF, 
        0xFF000000, 0xFF0000FF, 0xFF00FF00, 0xFF00FFFF, 
        0xFFFF0000, 0xFFFF00FF, 0xFFFFFF00, 0xFFFFFFFF}; 
    uint32_t a = table[x & 0xF];           // 4 bytes
    __m128i b = _mm_cvtsi32_si128(a);      // transfer to vector register
    __m128i c = _mm_unpacklo_epi8(b, b);   // duplicate bytes to 16-bit words
    __m128i d = _mm_unpacklo_epi16(c, c);  // duplicate 16-bit words to 32-bit dwords
    return d;
}

// to_bits: convert boolean vector to integer bitfield
static inline uint8_t to_bits(Vec2qb const & x) {
    uint32_t a = _mm_movemask_epi8(x);
    return (a & 1) | ((a >> 7) & 2);
}

// to_Vec2qb: convert integer bitfield to boolean vector
static inline Vec2qb to_Vec2qb(uint8_t x) {
    return Vec2qb(Vec2q(-(x&1), -((x>>1)&1)));
}

#else  // function prototypes here only

// to_bits: convert boolean vector to integer bitfield
static inline uint8_t to_bits(Vec4ib x);

// to_Vec4ib: convert integer bitfield to boolean vector
static inline Vec4ib to_Vec4ib(uint8_t x);

// to_bits: convert boolean vector to integer bitfield
static inline uint8_t to_bits(Vec2qb x);

// to_Vec2qb: convert integer bitfield to boolean vector
static inline Vec2qb to_Vec2qb(uint8_t x);

#endif  // INSTRSET < 9 || MAX_VECTOR_SIZE < 512

#ifdef NSIMD_NAMESPACE
}
#endif

#endif // VECTORI128_H
