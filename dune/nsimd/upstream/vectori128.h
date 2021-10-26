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
template <typename PackT, typename T>
class Vec128b {
protected:
    PackT xmm; // Integer vector
public:
    // Default constructor:
    Vec128b() {
    }
    // Constructor to broadcast the same value into all elements
    Vec128b(T i) {
        xmm = nsimd::set1<PackT>(i);
    }
    // Constructor to convert from type __m128i used in intrinsics:
    Vec128b(PackT const & x) {
        xmm = x;
    }
    // Assignment operator to convert from type __m128i used in intrinsics:
    Vec128b & operator = (PackT const & x) {
        xmm = x;
        return *this;
    }
    // Type cast operator to convert to __m128i used in intrinsics
    operator PackT() const {
        return xmm;
    }
    // Member function to load from array (unaligned)
    Vec128b & load(void const * p) {
        xmm = nsimd::loadu<PackT>((T const*)p);
        return *this;
    }
    // Member function to load from array, aligned by 16
    // "load_a" is faster than "load" on older Intel processors (Pentium 4, Pentium M, Core 1,
    // Merom, Wolfdale) and Atom, but not on other processors from Intel, AMD or VIA.
    // You may use load_a instead of load if you are certain that p points to an address
    // divisible by 16.
    Vec128b & load_a(void const * p) {
        xmm = nsimd::loada<PackT>((T const*)p);
        return *this;
    }
    // Member function to store into array (unaligned)
    void store(void * p) const {
        nsimd::storeu<T>((T*)p, xmm);
    }
    // Member function to store into array, aligned by 16
    // "store_a" is faster than "store" on older Intel processors (Pentium 4, Pentium M, Core 1,
    // Merom, Wolfdale) and Atom, but not on other processors from Intel, AMD or VIA.
    // You may use store_a instead of store if you are certain that p points to an address
    // divisible by 16.
    void store_a(void * p) const {
       nsimd::storea<T>((T*)p, xmm);
    }
    // Member function to change a single bit
    // Note: This function is inefficient. Use load function if changing more than one bit
    Vec128b const & set_bit(uint32_t index, int value) {
        xmm = nsimd_common::set_bit<PackT, T>(index, value, xmm);
        return *this;
    }
    // Member function to get a single bit
    // Note: This function is inefficient. Use store function if reading more than one bit
    int get_bit(uint32_t index) const {
        return nsimd_common::get_bit<PackT, T>(index, xmm);
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    bool operator [] (uint32_t index) const {
        return get_bit(index) != 0;
    }
    static int size() {
        return 128;
    }
};


// Define operators for this class

// vector operator & : bitwise and
template <typename PackT, typename T>
static inline Vec128b<PackT, T> operator & (Vec128b<PackT, T> const & a, Vec128b<PackT, T> const & b) {
    return nsimd::andb(a, b);
}
template <typename PackT, typename T>
static inline Vec128b<PackT, T> operator && (Vec128b<PackT, T> const & a, Vec128b<PackT, T> const & b) {
    return a & b;
}

// vector operator | : bitwise or
template <typename PackT, typename T>
static inline Vec128b<PackT, T> operator | (Vec128b<PackT, T> const & a, Vec128b<PackT, T> const & b) {
    return nsimd::orb(a, b);
}
template <typename PackT, typename T>
static inline Vec128b<PackT, T> operator || (Vec128b<PackT, T> const & a, Vec128b<PackT, T> const & b) {
    return a | b;
}

// vector operator ^ : bitwise xor
template <typename PackT, typename T>
static inline Vec128b<PackT, T> operator ^ (Vec128b<PackT, T> const & a, Vec128b<PackT, T> const & b) {
    return nsimd::xorb(a, b);
}

// vector operator ~ : bitwise not
template <typename PackT, typename T>
static inline Vec128b<PackT, T> operator ~ (Vec128b<PackT, T> const & a) {
    return nsimd::notb(a);
}

// vector operator &= : bitwise and
template <typename PackT, typename T>
static inline Vec128b<PackT, T> & operator &= (Vec128b<PackT, T> & a, Vec128b<PackT, T> const & b) {
    a = nsimd::andb(a, b);
    return a;
}

// vector operator |= : bitwise or
template <typename PackT, typename T>
static inline Vec128b<PackT, T> & operator |= (Vec128b<PackT, T> & a, Vec128b<PackT, T> const & b) {
    a = nsimd::orb(a, b);
    return a;
}

// vector operator ^= : bitwise xor
template <typename PackT, typename T>
static inline Vec128b<PackT, T> & operator ^= (Vec128b<PackT, T> & a, Vec128b<PackT, T> const & b) {
    a = nsimd::xorb(a, b);
    return a;
}

// Define functions for this class

// function andnot: a & ~ b
template <typename PackT, typename T>
static inline Vec128b<PackT, T> andnot (Vec128b<PackT, T> const & a, Vec128b<PackT, T> const & b) {
    return nsimd::andnotb(b, a);
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
    int32_t data[4] = {i0, i1, i2, i3};
    pack128_4i_t res = nsimd::loadu<pack128_4i_t>(data);
    return res;
}

template <uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3>
static inline pack128_4ui_t constant4ui() {
    uint32_t data[4] = {i0, i1, i2, i3};
    pack128_4ui_t res = nsimd::loadu<pack128_4ui_t>(data);
    return res;
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
template <typename PackT, typename T>
static inline bool horizontal_and (Vec128b<PackT, T> const & a) {
    return nsimd::all(nsimd::to_logical(a));
}

// horizontal_or. Returns true if at least one bit is 1
template <typename PackT, typename T>
static inline bool horizontal_or (Vec128b<PackT, T> const & a) {
    return nsimd::any(nsimd::to_logical(a));
}

template <typename PackT, typename PacklT, typename T>
class Vec128 : public Vec128b<PackT, T> {
using Vec128b<PackT, T>::xmm;
public:
    // Default constructor:
    Vec128() {
    }// Constructor to convert from type __m128i used in intrinsics:
    Vec128(PackT const & x) {
        xmm = x;
    }
    // Constructor to broadcast the same value into all elements:
    Vec128(int i) {
        xmm = nsimd::set1<PackT>((T)i);
    }
    // Assignment operator to convert from type __m128i used in intrinsics:
    Vec128 & operator = (PackT const & x) {
        xmm = x;
        return *this;
    }
    // Type cast operator to convert to __m128i used in intrinsics
    operator PackT() const {
        return xmm;
    }
    // Member function to load from array (unaligned)
    Vec128 & load(void const * p) {
        xmm = nsimd::loadu<PackT>((T const*)p);
        return *this;
    }
    // Member function to load from array (aligned)
    Vec128 & load_a(void const * p) {
        xmm = nsimd::loada<PackT>((T const*)p);
        return *this;
    }
    // Partial load. Load n elements and set the rest to 0
    Vec128 & load_partial(int n, void const * p) {
        xmm = nsimd_common::load_partial<PackT, PacklT, T>(p, n);
        return *this;
    }
    // Partial store. Store n elements
    void store_partial(int n, void * p) const {
        nsimd_common::store_partial<PackT, PacklT, T>(p, n, xmm);
    }
    // cut off vector to n elements. The last 16-n elements are set to zero
    Vec128 & cutoff(int n) {
        xmm = nsimd_common::cutoff<PackT, PacklT, T>(xmm, n);
        return *this;
    }
    // Member function to change a single element in vector
    // Note: This function is inefficient. Use load function if changing more than one element
    Vec128 const & insert(uint32_t index, int8_t value) {
        xmm = nsimd_common::set_bit<PackT, char>(index, value, xmm);
        return *this;
    }
    // Member function extract a single element from vector
    T extract(uint32_t index) const {
        int8_t x[16];
        this->store(x);
        return x[index & 0x0F];
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    T operator [] (uint32_t index) const {
        return extract(index);
    }
};



/*****************************************************************************
*
*          Vector of 16 8-bit signed integers
*
*****************************************************************************/

class Vec16c : public Vec128<pack128_16i_t, packl128_16i_t, int8_t> {
public:
    // Default constructor:
    Vec16c() {
    }
    // Constructor to build from all elements:
    Vec16c(int8_t i0, int8_t i1, int8_t i2, int8_t i3, int8_t i4, int8_t i5, int8_t i6, int8_t i7,
        int8_t i8, int8_t i9, int8_t i10, int8_t i11, int8_t i12, int8_t i13, int8_t i14, int8_t i15) {
        int8_t vec[16] = {i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15};
        xmm = nsimd::loadu<pack128_16i_t>(vec);
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
    static int size() {
        return 16;
    }
};

/*****************************************************************************
*
*          Vec16cb: Vector of 16 Booleans for use with Vec16c and Vec16uc
*
*****************************************************************************/

class Vec16cb : public Vec128<packl128_16i_t, packl128_16i_t, int8_t> {
public:
    using Vec128<packl128_16i_t, packl128_16i_t, int8_t>::xmm;
    // Default constructor
    Vec16cb() {}
    // Constructor to build from all elements:
    Vec16cb(bool x0, bool x1, bool x2, bool x3, bool x4, bool x5, bool x6, bool x7,
        bool x8, bool x9, bool x10, bool x11, bool x12, bool x13, bool x14, bool x15) {     
        int8_t vec[16] = {-(int)((int8_t)x0), -(int)((int8_t)x1), -(int)((int8_t)x2), -(int)((int8_t)x3), -(int)((int8_t)x4), -(int)((int8_t)x5), -(int)((int8_t)x6), -(int)((int8_t)x7), 
            -(int)((int8_t)x8), -(int)((int8_t)x9), -(int)((int8_t)x10), -(int)((int8_t)x11), -(int)((int8_t)x12), -(int)((int8_t)x13), -(int)((int8_t)x14), -(int)((int8_t)x15)};
        this->xmm = nsimd::loadla<packl128_16i_t>(vec);
    }
    // Assignment operator to convert from type __m128i used in intrinsics:
    Vec16cb & operator = (packl128_16i_t const & x) {
        this->xmm = x;
        return *this;
    }
    // Constructor to broadcast scalar value:
    Vec16cb(bool b) {
        this->xmm = nsimd::set1l<packl128_16i_t>(-int8_t(b));
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
    // Member function extract a single element from vector
    int8_t extract(uint32_t index) const {
        int8_t x[16];
        store(x);
        return x[index & 0x0F] != 0;
    }
    // Type cast operator to convert to __m128i used in intrinsics
    operator packl128_16i_t() const {
        return this->xmm;
    }
    
    static int size() {
        return 16;
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
    return nsimd_common::rotate_left<pack128_16i_t>(a, b);
}


/*****************************************************************************
*
*          Vector of 16 8-bit unsigned integers
*
*****************************************************************************/

class Vec16uc : public Vec128<pack128_16ui_t, packl128_16ui_t, uint8_t> {
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

class Vec8s : public Vec128<pack128_8i_t, packl128_i_t, int16_t> {
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
    static int size() {
        return 8;
    }
};

/*****************************************************************************
*
*          Vec8sb: Vector of 8 Booleans for use with Vec8s and Vec8us
*
*****************************************************************************/

class Vec8sb : public Vec128<packl128_8i_t, packl128_8i_t, uint8_t> {
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
    Vec8sb(bool b) {
        xmm = nsimd::set1l<packl128_8i_t>(-int16_t(b));
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
    // Member function extract a single element from vector
    // Note: This function is inefficient. Use store function if extracting more than one element
    bool extract(uint32_t index) const {
        return nsimd_common::get_bit<packl128_8i_t, int16_t>(index, xmm);
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
    return nsimd_common::rotate_left<pack128_8i_t>(a, b);
}


/*****************************************************************************
*
*          Vector of 8 16-bit unsigned integers
*
*****************************************************************************/

class Vec8us : public Vec128<pack128_8ui_t, packl128_8ui_t, uint16_t> {
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
    // Type cast operator to convert
    operator pack128_8ui_t() const {
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

class Vec4i : public Vec128<pack128_4i_t, packl128_4i_t, int32_t> {
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
    static int size() {
        return 4;
    }
};


/*****************************************************************************
*
*          Vec4ib: Vector of 4 Booleans for use with Vec4i and Vec4ui
*
*****************************************************************************/
class Vec4ib : public Vec128<packl128_4i_t, packl128_4i_t, int32_t> {
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
    // Member function extract a single element from vector
    bool extract(uint32_t index) const {
        return nsimd_common::get_bit<packl128_4i_t, int>(index, xmm) != 0;
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
    return nsimd_common::rotate_left<pack128_4i_t>(a, b);
}


/*****************************************************************************
*
*          Vector of 4 32-bit unsigned integers
*
*****************************************************************************/

class Vec4ui : public Vec128<pack128_4ui_t, packl128_4ui_t, uint32_t> {
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

    // Type cast operator to convert
    operator packl128_4i_t() const {
        return xmm;
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

class Vec2q : public Vec128b<pack128_2i_t, int64_t>  {
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
        xmm = nsimd_common::load_partial<pack128_2i_t, packl128_2i_t, int64_t>(p, n);
        return *this;
    }
    // Partial store. Store n elements
    void store_partial(int n, void * p) const {
        nsimd_common::store_partial<pack128_2i_t, packl128_2i_t, int64_t>(p, n, xmm);
    }
    // cut off vector to n elements. The last 2-n elements are set to zero
    Vec2q & cutoff(int n) {
        xmm = nsimd_common::cutoff<pack128_2i_t, packl128_4i_t, int64_t>(xmm, n);
        return *this;
    }
    // Member function to change a single element in vector
    // Note: This function is inefficient. Use load function if changing more than one element
    Vec2q const & insert(uint32_t index, int64_t value) {
        xmm = nsimd_common::set_bit<pack128_2i_t, int64_t>(index, value, xmm);
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
class Vec2qb : public Vec128<packl128_2i_t, packl128_2i_t, int64_t> {
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
    // Member function extract a single element from vector
    bool extract(uint32_t index) const {
        return nsimd_common::get_bit<packl128_2i_t, int64_t>(index, xmm) != 0;
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
    return nsimd_common::rotate_left<pack128_2i_t>(a, b);
}


/*****************************************************************************
*
*          Vector of 2 64-bit unsigned integers
*
*****************************************************************************/

class Vec2uq : public Vec128<pack128_2ui_t, packl128_2ui_t, uint64_t> {
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
    // Type cast operator to convert
    operator packl128_2i_t() const {
        return xmm;
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
    int64_t idx[2] = {int64_t(i0), int64_t(i1)};
    pack128_2i_t index = nsimd::loadu<pack128_2i_t>(idx);
    return nsimd_common::lookup2<pack128_2i_t,int64_t>(index, a);
}

template <int i0, int i1>
static inline Vec2uq permute2uq(Vec2uq const & a) {
    uint64_t idx[2] = {uint64_t(i0), uint64_t(i1)};
    pack128_2ui_t index = nsimd::loadu<pack128_2ui_t>(idx);
    return nsimd_common::lookup2<pack128_2ui_t,uint64_t>(index, a);
}

// permute vector Vec4i
template <int i0, int i1, int i2, int i3>
static inline Vec4i permute4i(Vec4i const & a) {
    int32_t idx[4] = {int32_t(i0), int32_t(i1), int32_t(i2), int32_t(i3)};
    pack128_4i_t index = nsimd::loadu<pack128_4i_t>(idx);
    return nsimd_common::lookup4<pack128_4i_t,int32_t>(index, a);
}

template <int i0, int i1, int i2, int i3>
static inline Vec4ui permute4ui(Vec4ui const & a) {
    uint32_t idx[4] = {uint32_t(i0), uint32_t(i1), uint32_t(i2), int32_t(i3)};
    pack128_4ui_t index = nsimd::loadu<pack128_4ui_t>(idx);
    return nsimd_common::lookup4<pack128_4ui_t,uint32_t>(index, a);
}

template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
static inline Vec8s permute8s(Vec8s const & a) {
    int16_t idx[8] = {int16_t(i0), int16_t(i1), int16_t(i2), int16_t(i3), int16_t(i4), int16_t(i5), int16_t(i6), int16_t(i7)};
    pack128_8i_t index = nsimd::loadu<pack128_8i_t>(idx);
    return nsimd_common::lookup8<pack128_8i_t,int16_t>(index, a);
}

template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
static inline Vec8us permute8us(Vec8us const & a) {
    uint16_t idx[8] = {uint16_t(i0), uint16_t(i1), uint16_t(i2), uint16_t(i3), uint16_t(i4), uint16_t(i5), uint16_t(i6), uint16_t(i7)};
    pack128_8ui_t index = nsimd::loadu<pack128_8ui_t>(idx);
    return nsimd_common::lookup8<pack128_8ui_t,uint16_t>(index, a);
}


template <int i0, int i1, int i2,  int i3,  int i4,  int i5,  int i6,  int i7, 
          int i8, int i9, int i10, int i11, int i12, int i13, int i14, int i15 > 
static inline Vec16c permute16c(Vec16c const & a) {

    int8_t idx[16] = {
        int8_t(i0), int8_t(i1), int8_t(i2), int8_t(i3), int8_t(i4), int8_t(i5), int8_t(i6), int8_t(i7),
        int8_t(i8), int8_t(i9), int8_t(i10), int8_t(i11), int8_t(i12), int8_t(i13), int8_t(i14), int8_t(i15)
    };
    pack128_16i_t index = nsimd::loadu<pack128_16i_t>(idx);
    return nsimd_common::lookup16<pack128_16i_t,int8_t>(index, a);
}

template <int i0, int i1, int i2,  int i3,  int i4,  int i5,  int i6,  int i7, 
          int i8, int i9, int i10, int i11, int i12, int i13, int i14, int i15 > 
static inline Vec16uc permute16uc(Vec16uc const & a) {
    uint8_t idx[16] = {
        uint8_t(i0), uint8_t(i1), uint8_t(i2), uint8_t(i3), uint8_t(i4), uint8_t(i5), uint8_t(i6), uint8_t(i7),
        uint8_t(i8), uint8_t(i9), uint8_t(i10), uint8_t(i11), uint8_t(i12), uint8_t(i13), uint8_t(i14), uint8_t(i15)
    };
    pack128_16ui_t index = nsimd::loadu<pack128_16ui_t>(idx);
    return nsimd_common::lookup16<pack128_16ui_t,uint8_t>(index, a);
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
    return nsimd_common::blend16<i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15, pack128_16i_t,int8_t>(a, b);
}

template <int i0, int i1, int i2,  int i3,  int i4,  int i5,  int i6,  int i7, 
          int i8, int i9, int i10, int i11, int i12, int i13, int i14, int i15 > 
static inline Vec16uc blend16uc(Vec16uc const & a, Vec16uc const & b) {
    return nsimd_common::blend16<i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15, pack128_16ui_t,uint8_t>(a, b);
}


template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
static inline Vec8s blend8s(Vec8s const & a, Vec8s const & b) {
    return nsimd_common::blend8<i0,i1,i2,i3,i4,i5,i6,i7, pack128_8i_t,int16_t>(a, b);
}

template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
static inline Vec8us blend8us(Vec8us const & a, Vec8us const & b) {
    return nsimd_common::blend8<i0,i1,i2,i3,i4,i5,i6,i7, pack128_8ui_t,uint16_t>(a, b);
}

template <int i0, int i1, int i2, int i3>
static inline Vec4i blend4i(Vec4i const & a, Vec4i const & b) {
    return nsimd_common::blend4<i0,i1,i2,i3, pack128_4i_t,int32_t>(a, b);
}

template <int i0, int i1, int i2, int i3>
static inline Vec4ui blend4ui(Vec4ui const & a, Vec4ui const & b) {
    return nsimd_common::blend4<i0,i1,i2,i3, pack128_4ui_t,uint32_t>(a, b);
}

template <int i0, int i1>
static inline Vec2q blend2q(Vec2q const & a, Vec2q const & b) {
    return nsimd_common::blend2<i0,i1, pack128_2i_t,int64_t>(a, b);
}

template <int i0, int i1>
static inline Vec2uq blend2uq(Vec2uq const & a, Vec2uq const & b) {
    return nsimd_common::blend2<i0,i1, pack128_2ui_t,uint64_t>(a, b);
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
    return nsimd_common::lookup16<pack128_16i_t, int8_t>(index, table);
}

static inline Vec16c lookup32(Vec16c const & index, Vec16c const & table0, Vec16c const & table1) {
    return nsimd_common::lookup32<pack128_16i_t, int8_t>(index, table0, table1);
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
    return nsimd_common::lookup8<pack128_8i_t, int16_t>(index, table);
}

static inline Vec8s lookup16(Vec8s const & index, Vec8s const & table0, Vec8s const & table1) {
    int16_t ii[16], tt[32], rr[16];
    table0.store(tt);  table1.store(tt+8);  index.store(ii);
    for (int j = 0; j < 16; j++) rr[j] = tt[ii[j] & 0x1F];
    return Vec8s().load(rr);
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
    uint16_t ii[8];  index1.store(ii);
    return Vec8s(((int16_t*)table)[ii[0]], ((int16_t*)table)[ii[1]], ((int16_t*)table)[ii[2]], ((int16_t*)table)[ii[3]],
                 ((int16_t*)table)[ii[4]], ((int16_t*)table)[ii[5]], ((int16_t*)table)[ii[6]], ((int16_t*)table)[ii[7]]);
}


static inline Vec4i lookup4(Vec4i const & index, Vec4i const & table) {
    return nsimd_common::lookup4<pack128_4i_t, int32_t>(index, table);
}

static inline Vec4i lookup8(Vec4i const & index, Vec4i const & table0, Vec4i const & table1) {
    int32_t ii[4], tt[8], rr[4];
    table0.store(tt);  table1.store(tt+4);  index.store(ii);
    for (int j = 0; j < 4; j++) rr[j] = tt[ii[j] & 0x07];
    return Vec4i().load(rr);
}

static inline Vec4i lookup16(Vec4i const & index, Vec4i const & table0, Vec4i const & table1, Vec4i const & table2, Vec4i const & table3) {
    int32_t ii[4], tt[16], rr[4];
    table0.store(tt);  table1.store(tt+4);  table2.store(tt+8);  table3.store(tt+12);
    index.store(ii);
    for (int j = 0; j < 4; j++) rr[j] = tt[ii[j] & 0x0F];
    return Vec4i().load(rr);
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
    uint32_t ii[4];  index1.store(ii);
    return Vec4i(((int32_t*)table)[ii[0]], ((int32_t*)table)[ii[1]], ((int32_t*)table)[ii[2]], ((int32_t*)table)[ii[3]]);
}


static inline Vec2q lookup2(Vec2q const & index, Vec2q const & table) {
    return nsimd_common::lookup2<pack128_2i_t, int64_t>(index, table);
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
    return nsimd_common::shift_bytes_up16(a, b);
}

// Function shift_bytes_down: shift whole vector right by b bytes
// You may use a permute function instead if b is a compile-time constant
static inline Vec16c shift_bytes_down(Vec16c const & a, int b) {
    return nsimd_common::shift_bytes_down16(a, b);
}

/*****************************************************************************
*
*          Gather functions with fixed indexes
*
*****************************************************************************/
// Load elements from array a with indices i0, i1, i2, i3
template <int i0, int i1, int i2, int i3>
static inline Vec4i gather4i(void const * a) {
    return nsimd_common::gather4<i0,i1,i2,i3,pack128_4i_t, int32_t>(a);
}

// Load elements from array a with indices i0, i1
template <int i0, int i1>
static inline Vec2q gather2q(void const * a) {
    return nsimd_common::gather2<i0,i1,pack128_2i_t, int64_t>(a);
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
    nsimd_common::scatter4<i0,i1,i2,i3,pack128_4i_t,int32_t>(data, array);
}

template <int i0, int i1>
static inline void scatter(Vec2q const & data, void * array) {
    nsimd_common::scatter2<i0,i1,pack128_2i_t,int64_t>(data, array);
}

static inline void scatter(Vec4i const & index, uint32_t limit, Vec4i const & data, void * array) {
    nsimd_common::scatter<pack128_4i_t,int32_t>(index, limit, data, array);
}

static inline void scatter(Vec2q const & index, uint32_t limit, Vec2q const & data, void * array) {
    nsimd_common::scatter<pack128_2i_t,int64_t>(index, limit, data, array);
} 

static inline void scatter(Vec4i const & index, uint32_t limit, Vec2q const & data, void * array) {
    nsimd_common::scatter<pack128_2i_t,int64_t>(index, limit, data, array);
} 

/*****************************************************************************
*
*          Functions for conversion between integer sizes
*
*****************************************************************************/

// Extend 8-bit integers to 16-bit integers, signed and unsigned

// Function extend_low : extends the low 8 elements to 16 bits with sign extension
static inline Vec8s extend_low (Vec16c const & a) {
    return nsimd_common::extend_low<Vec16c, Vec8s>(a);
}

// Function extend_high : extends the high 8 elements to 16 bits with sign extension
static inline Vec8s extend_high (Vec16c const & a) {
    return nsimd_common::extend_high<Vec16c, Vec8s>(a);
}

// Function extend_low : extends the low 8 elements to 16 bits with zero extension
static inline Vec8us extend_low (Vec16uc const & a) {
    return nsimd_common::extend_low<Vec16uc, Vec8us>(a);
}

// Function extend_high : extends the high 8 elements to 16 bits with zero extension
static inline Vec8us extend_high (Vec16uc const & a) {
    return nsimd_common::extend_high<Vec16uc, Vec8us>(a);
}

// Extend 16-bit integers to 32-bit integers, signed and unsigned

// Function extend_low : extends the low 4 elements to 32 bits with sign extension
static inline Vec4i extend_low (Vec8s const & a) {
    return nsimd_common::extend_low<Vec8s, Vec4i>(a);
}

// Function extend_high : extends the high 4 elements to 32 bits with sign extension
static inline Vec4i extend_high (Vec8s const & a) {
    return nsimd_common::extend_high<Vec8s, Vec4i>(a);
}

// Function extend_low : extends the low 4 elements to 32 bits with zero extension
static inline Vec4ui extend_low (Vec8us const & a) {
    return nsimd_common::extend_low<Vec8us, Vec4ui>(a);
}

// Function extend_high : extends the high 4 elements to 32 bits with zero extension
static inline Vec4ui extend_high (Vec8us const & a) {
    return nsimd_common::extend_high<Vec8us, Vec4ui>(a);
}

// Extend 32-bit integers to 64-bit integers, signed and unsigned

// Function extend_low : extends the low 2 elements to 64 bits with sign extension
static inline Vec2q extend_low (Vec4i const & a) {
    return nsimd_common::extend_low<Vec4i, Vec2q>(a);
}

// Function extend_high : extends the high 2 elements to 64 bits with sign extension
static inline Vec2q extend_high (Vec4i const & a) {
    return nsimd_common::extend_high<Vec4i, Vec2q>(a);
}

// Function extend_low : extends the low 2 elements to 64 bits with zero extension
static inline Vec2uq extend_low (Vec4ui const & a) {
    return nsimd_common::extend_low<Vec4ui, Vec2uq>(a);
}

// Function extend_high : extends the high 2 elements to 64 bits with zero extension
static inline Vec2uq extend_high (Vec4ui const & a) {
    return nsimd_common::extend_high<Vec4ui, Vec2uq>(a);
}

// Compress 16-bit integers to 8-bit integers, signed and unsigned, with and without saturation

// Function compress : packs two vectors of 16-bit integers into one vector of 8-bit integers
// Overflow wraps around
static inline Vec16c compress (Vec8s const & low, Vec8s const & high) {
    return nsimd_common::compress16<pack128_16i_t, int8_t, pack128_8i_t, int16_t>(low, high);
}

// Function compress : packs two vectors of 16-bit integers into one vector of 8-bit integers
// Signed, with saturation
static inline Vec16c compress_saturated (Vec8s const & low, Vec8s const & high) {
    return nsimd_common::compress16<pack128_16i_t, int8_t, pack128_8i_t, int16_t>(low, high, true);
}

// Function compress : packs two vectors of 16-bit integers to one vector of 8-bit integers
// Unsigned, overflow wraps around
static inline Vec16uc compress (Vec8us const & low, Vec8us const & high) {
    return  nsimd_common::compress16<pack128_16i_t, int8_t, pack128_8i_t, int16_t>(low, high);
}

// Function compress : packs two vectors of 16-bit integers into one vector of 8-bit integers
// Unsigned, with saturation
static inline Vec16uc compress_saturated (Vec8us const & low, Vec8us const & high) {
    return nsimd_common::compress16<pack128_16i_t, int8_t, pack128_8i_t, int16_t>(low, high, true);
}

// Compress 32-bit integers to 16-bit integers, signed and unsigned, with and without saturation

// Function compress : packs two vectors of 32-bit integers into one vector of 16-bit integers
// Overflow wraps around
static inline Vec8s compress (Vec4i const & low, Vec4i const & high) {
    return nsimd_common::compress8<pack128_8i_t, int16_t, pack128_4i_t, int32_t>(low, high);
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
//static inline uint16_t to_bits(Vec16cb const & x) {
//    return (uint16_t)_mm_movemask_epi8(x);
//}
//
//// to_Vec16bc: convert integer bitfield to boolean vector
//static inline Vec16cb to_Vec16cb(uint16_t x) {
//    static const uint32_t table[16] = {  // lookup-table
//        0x00000000, 0x000000FF, 0x0000FF00, 0x0000FFFF, 
//        0x00FF0000, 0x00FF00FF, 0x00FFFF00, 0x00FFFFFF, 
//        0xFF000000, 0xFF0000FF, 0xFF00FF00, 0xFF00FFFF, 
//        0xFFFF0000, 0xFFFF00FF, 0xFFFFFF00, 0xFFFFFFFF}; 
//    uint32_t a0 = table[x       & 0xF];
//    uint32_t a1 = table[(x>>4)  & 0xF];
//    uint32_t a2 = table[(x>>8)  & 0xF];
//    uint32_t a3 = table[(x>>12) & 0xF];
//    return Vec16cb(Vec16c(Vec4ui(a0, a1, a2, a3)));
//}
//
//// to_bits: convert boolean vector to integer bitfield
//static inline uint8_t to_bits(Vec8sb const & x) {
//    __m128i a = _mm_packs_epi16(x, x);  // 16-bit words to bytes
//    return (uint8_t)_mm_movemask_epi8(a);
//}
//
//// to_Vec8sb: convert integer bitfield to boolean vector
//static inline Vec8sb to_Vec8sb(uint8_t x) {
//    static const uint32_t table[16] = {  // lookup-table
//        0x00000000, 0x000000FF, 0x0000FF00, 0x0000FFFF, 
//        0x00FF0000, 0x00FF00FF, 0x00FFFF00, 0x00FFFFFF, 
//        0xFF000000, 0xFF0000FF, 0xFF00FF00, 0xFF00FFFF, 
//        0xFFFF0000, 0xFFFF00FF, 0xFFFFFF00, 0xFFFFFFFF}; 
//    uint32_t a0 = table[x       & 0xF];
//    uint32_t a1 = table[(x>>4)  & 0xF];
//    Vec4ui   b  = Vec4ui(a0, a1, a0, a1);
//    return _mm_unpacklo_epi8(b, b);  // duplicate bytes to 16-bit words
//}
//
//#if INSTRSET < 9 || MAX_VECTOR_SIZE < 512
//// These functions are defined in Vectori512.h if AVX512 instruction set is used
//
//// to_bits: convert boolean vector to integer bitfield
//static inline uint8_t to_bits(Vec4ib const & x) {
//    __m128i a = _mm_packs_epi32(x, x);  // 32-bit dwords to 16-bit words
//    __m128i b = _mm_packs_epi16(a, a);  // 16-bit words to bytes
//    return _mm_movemask_epi8(b) & 0xF;
//}
//
//// to_Vec4ib: convert integer bitfield to boolean vector
//static inline Vec4ib to_Vec4ib(uint8_t x) {
//    static const uint32_t table[16] = {    // lookup-table
//        0x00000000, 0x000000FF, 0x0000FF00, 0x0000FFFF, 
//        0x00FF0000, 0x00FF00FF, 0x00FFFF00, 0x00FFFFFF, 
//        0xFF000000, 0xFF0000FF, 0xFF00FF00, 0xFF00FFFF, 
//        0xFFFF0000, 0xFFFF00FF, 0xFFFFFF00, 0xFFFFFFFF}; 
//    uint32_t a = table[x & 0xF];           // 4 bytes
//    __m128i b = _mm_cvtsi32_si128(a);      // transfer to vector register
//    __m128i c = _mm_unpacklo_epi8(b, b);   // duplicate bytes to 16-bit words
//    __m128i d = _mm_unpacklo_epi16(c, c);  // duplicate 16-bit words to 32-bit dwords
//    return d;
//}
//
//// to_bits: convert boolean vector to integer bitfield
//static inline uint8_t to_bits(Vec2qb const & x) {
//    uint32_t a = _mm_movemask_epi8(x);
//    return (a & 1) | ((a >> 7) & 2);
//}
//
//// to_Vec2qb: convert integer bitfield to boolean vector
//static inline Vec2qb to_Vec2qb(uint8_t x) {
//    return Vec2qb(Vec2q(-(x&1), -((x>>1)&1)));
//}
//
//#else  // function prototypes here only
//
//// to_bits: convert boolean vector to integer bitfield
//static inline uint8_t to_bits(Vec4ib x);
//
//// to_Vec4ib: convert integer bitfield to boolean vector
//static inline Vec4ib to_Vec4ib(uint8_t x);
//
//// to_bits: convert boolean vector to integer bitfield
//static inline uint8_t to_bits(Vec2qb x);
//
//// to_Vec2qb: convert integer bitfield to boolean vector
//static inline Vec2qb to_Vec2qb(uint8_t x);

//#endif  // INSTRSET < 9 || MAX_VECTOR_SIZE < 512

#ifdef NSIMD_NAMESPACE
}
#endif

#endif // VECTORI128_H
