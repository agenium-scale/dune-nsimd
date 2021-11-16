/****************************  vectorf128.h   *******************************
* Author:        Agner Fog
* Date created:  2012-05-30
* Last modified: 2017-05-10
* Version:       1.29
* Project:       vector classes
* Description:
* Header file defining floating point vector classes as interface to 
* intrinsic functions in x86 microprocessors with SSE2 and later instruction
* sets up to AVX.
*
* Instructions:
* Use Gnu, Intel or Microsoft C++ compiler. Compile for the desired 
* instruction set, which must be at least SSE2. Specify the supported 
* instruction set by a command line define, e.g. __SSE4_1__ if the 
* compiler does not automatically do so.
*
* The following vector classes are defined here:
* Vec4f     Vector of 4 single precision floating point numbers
* Vec4fb    Vector of 4 Booleans for use with Vec4f
* Vec2d     Vector of 2 double precision floating point numbers
* Vec2db    Vector of 2 Booleans for use with Vec2d
*
* Each vector object is represented internally in the CPU as a 128-bit register.
* This header file defines operators and functions for these vectors.
*
* For example:
* Vec2d a(1.0, 2.0), b(3.0, 4.0), c;
* c = a + b;     // now c contains (4.0, 6.0)
*
* For detailed instructions, see Nsimd.pdf
*
* (c) Copyright 2012-2017 GNU General Public License http://www.gnu.org/licenses
*****************************************************************************/
#ifndef VECTORF128_H
#define VECTORF128_H

#if defined _MSC_VER && _MSC_VER >= 1800
// solve problem with ambiguous overloading of pow function in Microsoft math.h:
// make sure math.h is included first rather than last
#include <math.h>
#endif 

#include "vectori128.h"  // Define integer vectors
#include "nsimd_common.h"

#ifdef NSIMD_NAMESPACE
namespace NSIMD_NAMESPACE {
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
static inline pack4f_t selectf (packl4f_t const & s, pack4f_t const & a, pack4f_t const & b) {
    return nsimd::if_else1(s, a, b);
}

// Same, with two pack of double sources.
// and operators. Corresponds to this pseudocode:
// for (int i = 0; i < size; i++) result[i] = s[i] ? a[i] : b[i];
// Each element in s must be either 0 (false) or 0xFFFFFFFFFFFFFFFF (true). No other 
// values are allowed.
static inline pack2d_t selectd (packl2d_t const & s, pack2d_t const & a, pack2d_t const & b) {
    return nsimd::if_else1(s, a, b);
}


/*****************************************************************************
*
*          Vec4fb: Vector of 4 Booleans for use with Vec4f
*
*****************************************************************************/

class Vec4fb {
protected:
    packl4f_t xmm; // Float vector
public:
    // Default constructor:
    Vec4fb() {
    }
    // Constructor to build from all elements:
    Vec4fb(bool b0, bool b1, bool b2, bool b3) {
        int f[4] = {-(int)b0, -(int)b1, -(int)b2, -(int)b3};
        xmm = nsimd::reinterpretl<packl4f_t>(nsimd::loadla<packl128_4i_t>(f)); 
    }
    // Constructor to convert from type __m128 used in intrinsics:
    Vec4fb(packl4f_t const & x) {
        xmm = x;
    }
    // Assignment operator to convert from type __m128 used in intrinsics:
    Vec4fb & operator = (packl4f_t const & x) {
        xmm = x;
        return *this;
    }
    // Constructor to broadcast scalar value:
    Vec4fb(bool b) {
        xmm = nsimd::reinterpretl<packl4f_t>(nsimd::set1l<packl128_4i_t>(-int32_t(b))); 
    }
    // Assignment operator to broadcast scalar value:
    Vec4fb & operator = (bool b) {
        *this = Vec4fb(b);
        return *this;
    }
private: // Prevent constructing from int, etc.
    Vec4fb(int b);
    Vec4fb & operator = (int x);
public:
    // Constructor to convert from type Vec4ib used as Boolean for integer vectors
    Vec4fb(Vec4ib const & x) {
        xmm = nsimd::reinterpretl<packl4f_t>(packl128_4i_t(x)); 
    }
    // Assignment operator to convert from type Vec4ib used as Boolean for integer vectors
    Vec4fb & operator = (Vec4ib const & x) {
        xmm = nsimd::reinterpretl<packl4f_t>(packl128_4i_t(x)); 
        return *this;
    }
    // Type cast operator to convert to __m128 used in intrinsics
    operator packl4f_t() const {
        return xmm;
    }

    // Member function to change a single element in vector
    // Note: This function is inefficient. Use load function if changing more than one element
    Vec4fb const & insert(uint32_t index, bool value) {
        static const int32_t maskl[8] = {0,0,0,0,-1,0,0,0};
        packl4f_t mask  = nsimd::loadlu<packl128_4i_t>((float const*)(maskl+4-(index & 3))); // mask with FFFFFFFF at index position
        if (value) {
            xmm = nsimd::orl(xmm,mask);
        }
        else {
            xmm = nsimd::andnotl(mask,xmm);
        }
        return *this;
    }
    // Member function extract a single element from vector
    bool extract(uint32_t index) const {
        //return Vec4ib(*this).extract(index);
        return Vec4ib(nsimd::reinterpretl<packl128_4i_t>(xmm)).extract(index);
    }
    // Extract a single element. Operator [] can only read an element, not write.
    bool operator [] (uint32_t index) const {
        return extract(index);
    }
    static int size() {
        return 4;
    }
};


/*****************************************************************************
*
*          Operators for Vec4fb
*
*****************************************************************************/

// vector operator & : bitwise and
static inline Vec4fb operator & (Vec4fb const & a, Vec4fb const & b) {
    return nsimd::andl(packl4f_t(a), packl4f_t(b));
}
static inline Vec4fb operator && (Vec4fb const & a, Vec4fb const & b) {
    return a & b;
}

// vector operator &= : bitwise and
static inline Vec4fb & operator &= (Vec4fb & a, Vec4fb const & b) {
    a = a & b;
    return a;
}

// vector operator | : bitwise or
static inline Vec4fb operator | (Vec4fb const & a, Vec4fb const & b) {
    return nsimd::orl(packl4f_t(a), packl4f_t(b));
}
static inline Vec4fb operator || (Vec4fb const & a, Vec4fb const & b) {
    return a | b;
}

// vector operator |= : bitwise or
static inline Vec4fb & operator |= (Vec4fb & a, Vec4fb const & b) {
    a = a | b;
    return a;
}

// vector operator ^ : bitwise xor
static inline Vec4fb operator ^ (Vec4fb const & a, Vec4fb const & b) {
    return nsimd::xorl(packl4f_t(a), packl4f_t(b));
}

// vector operator ^= : bitwise xor
static inline Vec4fb & operator ^= (Vec4fb & a, Vec4fb const & b) {
    a = a ^ b;
    return a;
}

// vector operator ~ : bitwise not
static inline Vec4fb operator ~ (Vec4fb const & a) {
    return nsimd::notl(packl4f_t(a));
}

// vector operator ! : logical not
// (operator ! is less efficient than operator ~. Use only where not
// all bits in an element are the same)
static inline Vec4fb operator ! (Vec4fb const & a) {
    return nsimd::notl(packl4f_t(a));
}

// Functions for Vec4fb

// andnot: a & ~ b
static inline Vec4fb andnot(Vec4fb const & a, Vec4fb const & b) {
    return nsimd::andnotl(packl4f_t(b), packl4f_t(a));
}


/*****************************************************************************
*
*          Horizontal Boolean functions
*
*****************************************************************************/

// horizontal_and. Returns true if all bits are 1
static inline bool horizontal_and (Vec4fb const & a) {
    return nsimd::all(packl4f_t(a));
}

// horizontal_or. Returns true if at least one bit is 1
static inline bool horizontal_or (Vec4fb const & a) {
    return nsimd::any(packl4f_t(a));
}


/*****************************************************************************
*
*          Vec2db: Vector of 2 Booleans for use with Vec2d
*
*****************************************************************************/

class Vec2db {
protected:
    packl2d_t xmm; // Double vector
public:
    // Default constructor:
    Vec2db() {
    }
    // Constructor to broadcast the same value into all elements:
    // Constructor to build from all elements:
    Vec2db(bool b0, bool b1) {
        int f[4] = {-(int)b0, -(int)b0, -(int)b1, -(int)b1};
        xmm = nsimd::reinterpretl<packl2d_t>(nsimd::loadla<packl128_4i_t>(f)); 
    }
    // Constructor to convert from type __m128d used in intrinsics:
    Vec2db(packl2d_t const & x) {
        xmm = x;
    }
    // Assignment operator to convert from type __m128d used in intrinsics:
    Vec2db & operator = (packl2d_t const & x) {
        xmm = x;
        return *this;
    }
    // Constructor to broadcast scalar value:
    Vec2db(bool b) {
        packl2d_t set = nsimd::set1<nsimd::packl<int>>(-int32_t(b));
        xmm = nsimd::reinterpretl<int, double>(set);
    }
    // Assignment operator to broadcast scalar value:
    Vec2db & operator = (bool b) {
        *this = Vec2db(b);
        return *this;
    }
private: // Prevent constructing from int, etc.
    Vec2db(int b);
    Vec2db & operator = (int x);
public:
    // Constructor to convert from type Vec2qb used as Boolean for integer vectors
    Vec2db(Vec2qb const & x) {
        xmm = nsimd::reinterpretl<packl2d_t>(packl128_2i_t(x));
    }
    // Assignment operator to convert from type Vec2qb used as Boolean for integer vectors
    Vec2db & operator = (Vec2qb const & x) {
        xmm = nsimd::reinterpretl<packl2d_t>(packl128_2i_t(x));
        return *this;
    }
    // Type cast operator to convert to __m128d used in intrinsics
    operator packl2d_t() const {
        return xmm;
    }
    // Member function to change a single element in vector
    // Note: This function is inefficient. Use load function if changing more than one element
    Vec2db const & insert(uint32_t index, bool value) {
        static const int32_t maskl[8] = {0,0,0,0,-1,-1,0,0};
        packl4f_t mask  = nsimd::loadlu<packl4f_t>((float const*)(maskl+4-(index&1)*2)); // mask with FFFFFFFFFFFFFFFF at index position
        if (value) {
            xmm = nsimd::orl(xmm,nsimd::reinterpretl<packl2d_t>(mask));
        }
        else {
            xmm = nsimd::andnotl(nsimd::reinterpretl<packl2d_t>(mask),xmm);
        }
        return *this;
    }
    // Member function extract a single element from vector
    bool extract(uint32_t index) const {
        return Vec2qb(*this).extract(index);
    }
    // Extract a single element. Operator [] can only read an element, not write.
    bool operator [] (uint32_t index) const {
        return extract(index);
    }
    static int size() {
        return 2;
    }
};


/*****************************************************************************
*
*          Operators for Vec2db
*
*****************************************************************************/

// vector operator & : bitwise and
static inline Vec2db operator & (Vec2db const & a, Vec2db const & b) {
    return nsimd::andl(packl2d_t(a), packl2d_t(b));
}
static inline Vec2db operator && (Vec2db const & a, Vec2db const & b) {
    return a & b;
}

// vector operator &= : bitwise and
static inline Vec2db & operator &= (Vec2db & a, Vec2db const & b) {
    a = a & b;
    return a;
}

// vector operator | : bitwise or
static inline Vec2db operator | (Vec2db const & a, Vec2db const & b) {
    return nsimd::orl(packl2d_t(a), packl2d_t(b));
}
static inline Vec2db operator || (Vec2db const & a, Vec2db const & b) {
    return a | b;
}

// vector operator |= : bitwise or
static inline Vec2db & operator |= (Vec2db & a, Vec2db const & b) {
    a = a | b;
    return a;
}

// vector operator ^ : bitwise xor
static inline Vec2db operator ^ (Vec2db const & a, Vec2db const & b) {
    return nsimd::xorl(packl2d_t(a), packl2d_t(b));
}

// vector operator ^= : bitwise xor
static inline Vec2db & operator ^= (Vec2db & a, Vec2db const & b) {
    a = a ^ b;
    return a;
}

// vector operator ~ : bitwise not
static inline Vec2db operator ~ (Vec2db const & a) {
    return nsimd::notl(packl2d_t(a));
}

// vector operator ! : logical not
// (operator ! is less efficient than operator ~. Use only where not
// all bits in an element are the same)
static inline Vec2db operator ! (Vec2db const & a) {
    return Vec2db (! Vec2qb(a));
}

// Functions for Vec2db

// andnot: a & ~ b
static inline Vec2db andnot(Vec2db const & a, Vec2db const & b) {
    return nsimd::andnotl(packl2d_t(b), packl2d_t(a));
}


/*****************************************************************************
*
*          Horizontal Boolean functions
*
*****************************************************************************/

// horizontal_and. Returns true if all bits are 1
static inline bool horizontal_and (Vec2db const & a) {
    return nsimd::all(packl2d_t(a));
}

// horizontal_or. Returns true if at least one bit is 1
static inline bool horizontal_or (Vec2db const & a) {
    return nsimd::any(packl2d_t(a));
}



/*****************************************************************************
*
*          Vec4f: Vector of 4 single precision floating point values
*
*****************************************************************************/

class Vec4f {
protected:
    pack4f_t xmm; // Float vector
public:
    // Default constructor:
    Vec4f() {
    }
    // Constructor to broadcast the same value into all elements:
    Vec4f(float f) {
        xmm = nsimd::set1<pack4f_t>(f);
    }
    // Constructor to build from all elements:
    Vec4f(float f0, float f1, float f2, float f3) {
        float f[4] = {f0, f1, f2, f3};
        xmm = nsimd::loada<pack4f_t>(f);
    }
    // Constructor to convert from type __m128 used in intrinsics:
    Vec4f(pack4f_t const & x) {
        xmm = x;
    }
    // Assignment operator to convert from type __m128 used in intrinsics:
    Vec4f & operator = (pack4f_t const & x) {
        xmm = x;
        return *this;
    }
    // Type cast operator to convert to __m128 used in intrinsics
    operator pack4f_t() const {
        return xmm;
    }
    // Member function to load from array (unaligned)
    Vec4f & load(float const * p) {
        xmm = nsimd::loadu<pack4f_t>(p);
        return *this;
    }
    // Member function to load from array, aligned by 16
    // "load_a" is faster than "load" on older Intel processors (Pentium 4, Pentium M, Core 1,
    // Merom, Wolfdale) and Atom, but not on other processors from Intel, AMD or VIA.
    // You may use load_a instead of load if you are certain that p points to an address
    // divisible by 16.
    Vec4f & load_a(float const * p) {
        xmm = nsimd::loada<pack4f_t>(p);
        return *this;
    }
    // Member function to store into array (unaligned)
    void store(float * p) const {
        nsimd::storeu(p, xmm);
    }
    // Member function to store into array, aligned by 16
    // "store_a" is faster than "store" on older Intel processors (Pentium 4, Pentium M, Core 1,
    // Merom, Wolfdale) and Atom, but not on other processors from Intel, AMD or VIA.
    // You may use store_a instead of store if you are certain that p points to an address
    // divisible by 16.
    void store_a(float * p) const {
        nsimd::storea(p, xmm);
    }
    // Partial load. Load n elements and set the rest to 0
    Vec4f & load_partial(int n, float const * p) {
        xmm = nsimd_common::load_partial<pack4f_t, packl4f_t, float>(p, n);
        return *this;
    }
    // Partial store. Store n elements
    void store_partial(int n, float * p) const {
        nsimd_common::store_partial<pack4f_t, packl4f_t, float>(p, n, xmm);
    }
    // cut off vector to n elements. The last 4-n elements are set to zero
    Vec4f & cutoff(int n) {
        xmm = nsimd_common::cutoff<pack4f_t, packl4f_t, float>(xmm, n);
        return *this;
    }
    // Member function to change a single element in vector
    // Note: This function is inefficient. Use load function if changing more than one element
    Vec4f const & insert(uint32_t index, float value) {
        xmm = nsimd_common::set_bit(index, value, xmm);
        return *this;
    };
    // Member function extract a single element from vector
    float extract(uint32_t index) const {
        float x[4];
        store(x);
        return x[index & 3];
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    float operator [] (uint32_t index) const {
        return extract(index);
    }
    static int size() {
        return 4;
    }
};


/*****************************************************************************
*
*          Operators for Vec4f
*
*****************************************************************************/

// vector operator + : add element by element
static inline Vec4f operator + (Vec4f const & a, Vec4f const & b) {
    return nsimd::add(pack4f_t(a), pack4f_t(b));
}

// vector operator + : add vector and scalar
static inline Vec4f operator + (Vec4f const & a, float b) {
    return a + Vec4f(b);
}
static inline Vec4f operator + (float a, Vec4f const & b) {
    return Vec4f(a) + b;
}

// vector operator += : add
static inline Vec4f & operator += (Vec4f & a, Vec4f const & b) {
    a = a + b;
    return a;
}

// postfix operator ++
static inline Vec4f operator ++ (Vec4f & a, int) {
    Vec4f a0 = a;
    a = a + 1.0f;
    return a0;
}

// prefix operator ++
static inline Vec4f & operator ++ (Vec4f & a) {
    a = a + 1.0f;
    return a;
}

// vector operator - : subtract element by element
static inline Vec4f operator - (Vec4f const & a, Vec4f const & b) {
    return nsimd::sub(pack4f_t(a), pack4f_t(b));
}

// vector operator - : subtract vector and scalar
static inline Vec4f operator - (Vec4f const & a, float b) {
    return a - Vec4f(b);
}
static inline Vec4f operator - (float a, Vec4f const & b) {
    return Vec4f(a) - b;
}

// vector operator - : unary minus
// Change sign bit, even for 0, INF and NAN
static inline Vec4f operator - (Vec4f const & a) {
    return nsimd::xorb(pack4f_t(a), nsimd::reinterpret<pack4f_t>(nsimd::set1<pack128_4i_t>(0x80000000)));
}

// vector operator -= : subtract
static inline Vec4f & operator -= (Vec4f & a, Vec4f const & b) {
    a = a - b;
    return a;
}

// postfix operator --
static inline Vec4f operator -- (Vec4f & a, int) {
    Vec4f a0 = a;
    a = a - 1.0f;
    return a0;
}

// prefix operator --
static inline Vec4f & operator -- (Vec4f & a) {
    a = a - 1.0f;
    return a;
}

// vector operator * : multiply element by element
static inline Vec4f operator * (Vec4f const & a, Vec4f const & b) {
    return nsimd::mul(pack4f_t(a), pack4f_t(b));
}

// vector operator * : multiply vector and scalar
static inline Vec4f operator * (Vec4f const & a, float b) {
    return a * Vec4f(b);
}
static inline Vec4f operator * (float a, Vec4f const & b) {
    return Vec4f(a) * b;
}

// vector operator *= : multiply
static inline Vec4f & operator *= (Vec4f & a, Vec4f const & b) {
    a = a * b;
    return a;
}

// vector operator / : divide all elements by same integer
static inline Vec4f operator / (Vec4f const & a, Vec4f const & b) {
    return nsimd::div(pack4f_t(a), pack4f_t(b));
}

// vector operator / : divide vector and scalar
static inline Vec4f operator / (Vec4f const & a, float b) {
    return a / Vec4f(b);
}
static inline Vec4f operator / (float a, Vec4f const & b) {
    return Vec4f(a) / b;
}

// vector operator /= : divide
static inline Vec4f & operator /= (Vec4f & a, Vec4f const & b) {
    a = a / b;
    return a;
}

// vector operator == : returns true for elements for which a == b
static inline Vec4fb operator == (Vec4f const & a, Vec4f const & b) {
    return nsimd::eq(pack4f_t(a), pack4f_t(b));
}

// vector operator != : returns true for elements for which a != b
static inline Vec4fb operator != (Vec4f const & a, Vec4f const & b) {
    return nsimd::ne(pack4f_t(a), pack4f_t(b));
}

// vector operator < : returns true for elements for which a < b
static inline Vec4fb operator < (Vec4f const & a, Vec4f const & b) {
    return nsimd::lt(pack4f_t(a), pack4f_t(b));
}

// vector operator <= : returns true for elements for which a <= b
static inline Vec4fb operator <= (Vec4f const & a, Vec4f const & b) {
    return nsimd::le(pack4f_t(a), pack4f_t(b));
}

// vector operator > : returns true for elements for which a > b
static inline Vec4fb operator > (Vec4f const & a, Vec4f const & b) {
    return nsimd::gt(pack4f_t(a), pack4f_t(b));
}

// vector operator >= : returns true for elements for which a >= b
static inline Vec4fb operator >= (Vec4f const & a, Vec4f const & b) {
    return nsimd::ge(pack4f_t(a),pack4f_t(b));
}

// Bitwise logical operators

// vector operator & : bitwise and
static inline Vec4f operator & (Vec4f const & a, Vec4f const & b) {
    return nsimd::andb(pack4f_t(a), pack4f_t(b));
}

// vector operator &= : bitwise and
static inline Vec4f & operator &= (Vec4f & a, Vec4f const & b) {
    a = a & b;
    return a;
}

// vector operator & : bitwise and of Vec4f and Vec4fb
static inline Vec4f operator & (Vec4f const & a, Vec4fb const & b) {
    return nsimd::andb(pack4f_t(a), nsimd::to_mask(packl4f_t(b)));
}
static inline Vec4f operator & (Vec4fb const & a, Vec4f const & b) {
    return nsimd::andb(pack4f_t(a), pack4f_t(b));
}

// vector operator | : bitwise or
static inline Vec4f operator | (Vec4f const & a, Vec4f const & b) {
    return nsimd::orb(pack4f_t(a), pack4f_t(b));
}

// vector operator |= : bitwise or
static inline Vec4f & operator |= (Vec4f & a, Vec4f const & b) {
    a = a | b;
    return a;
}

// vector operator ^ : bitwise xor
static inline Vec4f operator ^ (Vec4f const & a, Vec4f const & b) {
    return nsimd::xorb(pack4f_t(a), pack4f_t(b));
}

// vector operator ^= : bitwise xor
static inline Vec4f & operator ^= (Vec4f & a, Vec4f const & b) {
    a = a ^ b;
    return a;
}

// vector operator ! : logical not. Returns Boolean vector
static inline Vec4fb operator ! (Vec4f const & a) {
    return a == Vec4f(0.0f);
}


/*****************************************************************************
*
*          Functions for Vec4f
*
*****************************************************************************/

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 4; i++) result[i] = s[i] ? a[i] : b[i];
// Each byte in s must be either 0 (false) or 0xFFFFFFFF (true). No other values are allowed.
static inline Vec4f select (Vec4fb const & s, Vec4f const & a, Vec4f const & b) {
    return selectf(s,a,b);
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
static inline Vec4f if_add (Vec4fb const & f, Vec4f const & a, Vec4f const & b) {
    return a + (Vec4f(f) & b);
}

// Conditional multiply: For all vector elements i: result[i] = f[i] ? (a[i] * b[i]) : a[i]
static inline Vec4f if_mul (Vec4fb const & f, Vec4f const & a, Vec4f const & b) {
    return a * select(f, b, 1.f);
}


// General arithmetic functions, etc.

// Horizontal add: Calculates the sum of all vector elements.
static inline float horizontal_add (Vec4f const & a) {
    return nsimd_common::horizontal_add(a);
}

// function max: a > b ? a : b
static inline Vec4f max(Vec4f const & a, Vec4f const & b) {
    return nsimd::max(pack4f_t(a),pack4f_t(b));
}

// function min: a < b ? a : b
static inline Vec4f min(Vec4f const & a, Vec4f const & b) {
    return nsimd::min(pack4f_t(a),pack4f_t(b));
}

// function abs: absolute value
// Removes sign bit, even for -0.0f, -INF and -NAN
static inline Vec4f abs(Vec4f const & a) {
    return nsimd::abs(pack4f_t(a));
}

// function sqrt: square root
static inline Vec4f sqrt(Vec4f const & a) {
    return nsimd::sqrt(pack4f_t(a));
}

// function square: a * a
static inline Vec4f square(Vec4f const & a) {
    return a * a;
}

// pow(vector,int) function template
template <typename VTYPE>
static inline VTYPE pow_template_i(VTYPE const & x0, int n) {
    VTYPE x = x0;                      // a^(2^i)
    VTYPE y(1.0f);                     // accumulator
    if (n >= 0) {                      // make sure n is not negative
        while (true) {                 // loop for each bit in n
            if (n & 1) y *= x;         // multiply if bit = 1
            n >>= 1;                   // get next bit of n
            if (n == 0) return y;      // finished
            x *= x;                    // x = a^2, a^4, a^8, etc.
        }
    }
    else {                             // n < 0
        return VTYPE(1.0f)/pow_template_i<VTYPE>(x0,-n);  // reciprocal
    }
}

// pow(Vec4f, int):
// The purpose of this template is to prevent implicit conversion of a float
// exponent to int when calling pow(vector, float) and vectormath_exp.h is
// not included

template <typename TT> static Vec4f pow(Vec4f const & a, TT const & n);

// Raise floating point numbers to integer power n
template <>
inline Vec4f pow<int>(Vec4f const & x0, int const & n) {
    return pow_template_i<Vec4f>(x0, n);
}

// allow conversion from unsigned int
template <>
inline Vec4f pow<uint32_t>(Vec4f const & x0, uint32_t const & n) {
    return pow_template_i<Vec4f>(x0, (int)n);
}

// Raise floating point numbers to integer power n, where n is a compile-time constant
template <int n>
static inline Vec4f pow_n(Vec4f const & a) {
    if (n < 0)    return Vec4f(1.0f) / pow_n<-n>(a);
    if (n == 0)   return Vec4f(1.0f);
    if (n >= 256) return pow(a, n);
    Vec4f x = a;                       // a^(2^i)
    Vec4f y;                           // accumulator
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

// implement as function pow(vector, const_int)
template <int n>
static inline Vec4f pow(Vec4f const & a, Const_int_t<n>) {
    return pow_n<n>(a);
}

// implement the same as macro pow_const(vector, int)
#define pow_const(x,n) pow_n<n>(x)

// function round: round to nearest integer (even). (result as float vector)
static inline Vec4f round(Vec4f const & a) {
    nsimd::round_to_even(pack4f_t(a));
}

// function truncate: round towards zero. (result as float vector)
static inline Vec4f truncate(Vec4f const & a) {
    return nsimd::trunc(pack4f_t(a));
}

// function floor: round towards minus infinity. (result as float vector)
static inline Vec4f floor(Vec4f const & a) {
    return nsimd::floor(pack4f_t(a));
}

// function ceil: round towards plus infinity. (result as float vector)
static inline Vec4f ceil(Vec4f const & a) {
    return nsimd::ceil(pack4f_t(a));
}

// function round_to_int: round to nearest integer (even). (result as integer vector)
static inline Vec4i round_to_int(Vec4f const & a) {
    return nsimd::cvt<pack128_4i_t>(pack4f_t(a));
}

// function truncate_to_int: round towards zero. (result as integer vector)
static inline Vec4i truncate_to_int(Vec4f const & a) {
    return nsimd::cvt<pack128_4i_t>(nsimd::trunc(pack4f_t(a)));
}

// function to_float: convert integer vector to float vector
static inline Vec4f to_float(Vec4i const & a) {
    return nsimd::cvt<pack4f_t>(pack128_4i_t(a));
}

// function to_float: convert unsigned integer vector to float vector
static inline Vec4f to_float(Vec4ui const & a) {
    nsimd::cvt<pack4f_t>(pack128_4ui_t(a));
}


// Approximate math functions

// approximate reciprocal (Faster than 1.f / a. relative accuracy better than 2^-11)
static inline Vec4f approx_recipr(Vec4f const & a) {
    return nsimd::rec11(pack4f_t(a));
}

// approximate reciprocal squareroot (Faster than 1.f / sqrt(a). Relative accuracy better than 2^-11)
static inline Vec4f approx_rsqrt(Vec4f const & a) {
    return nsimd::rsqrt11(pack4f_t(a));
}

// Fused multiply and add functions

// Multiply and add
static inline Vec4f mul_add(Vec4f const & a, Vec4f const & b, Vec4f const & c) {
    return nsimd::fma(pack4f_t(a), pack4f_t(b), pack4f_t(c));
}

// Multiply and subtract
static inline Vec4f mul_sub(Vec4f const & a, Vec4f const & b, Vec4f const & c) {
    return nsimd::fms(pack4f_t(a), pack4f_t(b), pack4f_t(c));
}

// Multiply and inverse subtract
static inline Vec4f nmul_add(Vec4f const & a, Vec4f const & b, Vec4f const & c) {
    return nsimd::fnma(pack4f_t(a), pack4f_t(b), pack4f_t(c));
}


// Multiply and subtract with extra precision on the intermediate calculations, 
// even if FMA instructions not supported, using Veltkamp-Dekker split
static inline Vec4f mul_sub_x(Vec4f const & a, Vec4f const & b, Vec4f const & c) {
    return nsimd::fnms(pack4f_t(a), pack4f_t(b), pack4f_t(c));
}

// Math functions using fast bit manipulation

// Extract the exponent as an integer
// exponent(a) = floor(log2(abs(a)));
// exponent(1.0f) = 0, exponent(0.0f) = -127, exponent(INF) = +128, exponent(NAN) = +128
static inline Vec4i exponent(Vec4f const & a) {
    Vec4ui t1 = nsimd::reinterpret<pack128_4ui_t>(pack4f_t(a));   // reinterpret as 32-bit integer
    Vec4ui t2 = t1 << 1;               // shift out sign bit
    Vec4ui t3 = t2 >> 24;              // shift down logical to position 0
    Vec4i  t4 = Vec4i(t3) - 0x7F;      // subtract bias from exponent
    return t4;
}

// Extract the fraction part of a floating point number
// a = 2^exponent(a) * fraction(a), except for a = 0
// fraction(1.0f) = 1.0f, fraction(5.0f) = 1.25f
// NOTE: The name fraction clashes with an ENUM in MAC XCode CarbonCore script.h !
static inline Vec4f fraction(Vec4f const & a) {
    Vec4ui t1 = nsimd::reinterpret<pack128_4ui_t>(pack4f_t(a));   // reinterpret as 32-bit integer
    Vec4ui t2 = Vec4ui((t1 & 0x007FFFFF) | 0x3F800000); // set exponent to 0 + bias
    return nsimd::reinterpret<pack4f_t>(pack128_4ui_t(t2));;
}

// Fast calculation of pow(2,n) with n integer
// n  =    0 gives 1.0f
// n >=  128 gives +INF
// n <= -127 gives 0.0f
// This function will never produce denormals, and never raise exceptions
static inline Vec4f exp2(Vec4i const & n) {
    return nsimd::exp2_u35(nsimd::reinterpret<pack4f_t>(pack128_4i_t(n)));
}

//static Vec4f exp2(Vec4f const & x); // defined in vectormath_exp.h


// Control word manipulaton
// ------------------------
// The MXCSR control word has the following bits:
//  0:    Invalid Operation Flag
//  1:    Denormal Flag (=subnormal)
//  2:    Divide-by-Zero Flag
//  3:    Overflow Flag
//  4:    Underflow Flag
//  5:    Precision Flag
//  6:    Denormals Are Zeros (=subnormals)
//  7:    Invalid Operation Mask
//  8:    Denormal Operation Mask (=subnormal)
//  9:    Divide-by-Zero Mask
// 10:    Overflow Mask
// 11:    Underflow Mask
// 12:    Precision Mask
// 13-14: Rounding control
//        00: round to nearest or even
//        01: round down towards -infinity
//        10: round up   towards +infinity
//        11: round towards zero (truncate)
// 15: Flush to Zero

// Function get_control_word:
// Read the MXCSR control word
static inline uint32_t get_control_word() {
    return _mm_getcsr();
}

// Function set_control_word:
// Write the MXCSR control word
static inline void set_control_word(uint32_t w) {
    _mm_setcsr(w);
}

// Function no_subnormals:
// Set "Denormals Are Zeros" and "Flush to Zero" mode to avoid the extremely
// time-consuming denormals in case of underflow
static inline void no_subnormals() {
    uint32_t t1 = get_control_word();
    t1 |= (1 << 6) | (1 << 15);     // set bit 6 and 15 in MXCSR
    set_control_word(t1);
}

// Function reset_control_word:
// Set the MXCSR control word to the default value 0x1F80.
// This will mask floating point exceptions, set rounding mode to nearest (or even),
// and allow denormals.
static inline void reset_control_word() {
    set_control_word(0x1F80);
}


// Categorization functions

// Function sign_bit: gives true for elements that have the sign bit set
// even for -0.0f, -INF and -NAN
// Note that sign_bit(Vec4f(-0.0f)) gives true, while Vec4f(-0.0f) < Vec4f(0.0f) gives false
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
static inline Vec4fb sign_bit(Vec4f const & a) {
    Vec4i t1 = nsimd::reinterpret<pack128_4i_t>(pack4f_t(a));    // reinterpret as 32-bit integer
    Vec4i t2 = t1 >> 31;               // extend sign bit
    return nsimd::reinterpretl<packl4f_t>(nsimd::to_logical(pack128_4i_t(t2)));       // reinterpret as 32-bit Boolean
}

// Function sign_combine: changes the sign of a when b has the sign bit set
// same as select(sign_bit(b), -a, a)
static inline Vec4f sign_combine(Vec4f const & a, Vec4f const & b) {
    Vec4f signmask = nsimd::reinterpret<pack4f_t>(constant4ui<0x80000000,0x80000000,0x80000000,0x80000000>());  // -0.0
    return a ^ (b & signmask);
}

// Function is_finite: gives true for elements that are normal, denormal or zero, 
// false for INF and NAN
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
static inline Vec4fb is_finite(Vec4f const & a) {
    Vec4i t1 = nsimd::reinterpret<pack128_4i_t>(pack4f_t(a));    // reinterpret as 32-bit integer
    Vec4i t2 = t1 << 1;                // shift out sign bit
    return Vec4ib(Vec4i(t2 & 0xFF000000) != 0xFF000000); // exponent field is not all 1s
}

// Function is_inf: gives true for elements that are +INF or -INF
// false for finite numbers and NAN
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
static inline Vec4fb is_inf(Vec4f const & a) {
    return nsimd::isinf((pack4f_t)a);
}

// Function is_nan: gives true for elements that are +NAN or -NAN
// false for finite numbers and +/-INF
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
static inline Vec4fb is_nan(Vec4f const & a) {
    return nsimd::isnan((pack4f_t)a);
}

// Function is_subnormal: gives true for elements that are denormal (subnormal)
// false for finite numbers, zero, NAN and INF
static inline Vec4fb is_subnormal(Vec4f const & a) {
    return nsimd::isnormal((pack4f_t)a);
}

// Function is_zero_or_subnormal: gives true for elements that are zero or subnormal (denormal)
// false for finite numbers, NAN and INF
static inline Vec4fb is_zero_or_subnormal(Vec4f const & a) {
    Vec4i t = nsimd::reinterpret<pack128_4i_t>(pack4f_t(a));     // reinterpret as 32-bit integer
          t &= 0x7F800000;             // isolate exponent
    return t == 0;                     // exponent = 0
}

// Function infinite4f: returns a vector where all elements are +INF
static inline Vec4f infinite4f() {
    return nsimd::reinterpret<pack4f_t>(nsimd::set1<pack128_4i_t>(0x7F800000));
}

// Function nan4f: returns a vector where all elements are NAN (quiet)
static inline Vec4f nan4f(int n = 0x10) {
    return nsimd::reinterpret<pack4f_t>(nsimd::set1<pack128_4i_t>(0x7FC00000 + n));
}

/*****************************************************************************
*
*          Vector Vec4f permute and blend functions
*
******************************************************************************
*
* The permute function can reorder the elements of a vector and optionally
* set some elements to zero. 
*
* The indexes are inserted as template parameters in <>. These indexes must be
* constants. Each template parameter is an index to the element you want to 
* select. A negative index will generate zero.
*
* Example:
* Vec4f a(10.f,11.f,12.f,13.f);        // a is (10,11,12,13)
* Vec4f b, c;
* b = permute4f<0,0,2,2>(a);           // b is (10,10,12,12)
* c = permute4f<3,2,-1,-1>(a);         // c is (13,12, 0, 0)
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
* Vec4f a(10.f,11.f,12.f,13.f);        // a is (10, 11, 12, 13)
* Vec4f b(20.f,21.f,22.f,23.f);        // b is (20, 21, 22, 23)
* Vec4f c;
* c = blend4f<1,4,-1,7> (a,b);         // c is (11, 20,  0, 23)
*
* Don't worry about the complicated code for these functions. Most of the
* code is resolved at compile time to generate only a few instructions.
*****************************************************************************/

// permute vector Vec4f
template <int i0, int i1, int i2, int i3>
static inline Vec4f permute4f(Vec4f const & a) {
    float idx[4] = {float(i0), float(i1), float(i2), float(i3)};
    pack4f_t index = nsimd::loadu<pack4f_t>(idx);
    return nsimd_common::lookup4<pack4f_t,float>(index, a);
}


// blend vectors Vec4f
template <int i0, int i1, int i2, int i3>
static inline Vec4f blend4f(Vec4f const & a, Vec4f const & b) {

    // Combine all the indexes into a single bitfield, with 8 bits for each
    const int m1 = (i0&7) | (i1&7)<<8 | (i2&7)<<16 | (i3&7)<<24; 

    // Mask to zero out negative indexes
    const int m2 = (i0<0?0:0xFF) | (i1<0?0:0xFF)<<8 | (i2<0?0:0xFF)<<16 | (i3<0?0:0xFF)<<24;

    if ((m1 & 0x04040404 & m2) == 0) {
        // no elements from b
        return permute4f<i0,i1,i2,i3>(a);
    }
    if (((m1^0x04040404) & 0x04040404 & m2) == 0) {
        // no elements from a
        return permute4f<i0&~4, i1&~4, i2&~4, i3&~4>(b);
    }
    if (((m1 & ~0x04040404) ^ 0x03020100) == 0 && m2 == -1) {
        // selecting without shuffling or zeroing
        __m128i sel = constant4i <i0 & 4 ? 0 : -1, i1 & 4 ? 0 : -1, i2 & 4 ? 0 : -1, i3 & 4 ? 0 : -1> ();
        return selectf(_mm_castsi128_ps(sel), a, b);
    }
#ifdef __XOP__     // Use AMD XOP instruction PPERM
    __m128i maska = constant4i <
        i0 < 0 ? 0x80808080 : (i0*4 & 31) + (((i0*4 & 31) + 1) << 8) + (((i0*4 & 31) + 2) << 16) + (((i0*4 & 31) + 3) << 24),
        i1 < 0 ? 0x80808080 : (i1*4 & 31) + (((i1*4 & 31) + 1) << 8) + (((i1*4 & 31) + 2) << 16) + (((i1*4 & 31) + 3) << 24),
        i2 < 0 ? 0x80808080 : (i2*4 & 31) + (((i2*4 & 31) + 1) << 8) + (((i2*4 & 31) + 2) << 16) + (((i2*4 & 31) + 3) << 24),
        i3 < 0 ? 0x80808080 : (i3*4 & 31) + (((i3*4 & 31) + 1) << 8) + (((i3*4 & 31) + 2) << 16) + (((i3*4 & 31) + 3) << 24) > ();
    return _mm_castsi128_ps(_mm_perm_epi8(_mm_castps_si128(a), _mm_castps_si128(b), maska));
#else
    if ((((m1 & ~0x04040404) ^ 0x03020100) & m2) == 0) {
        // selecting and zeroing, not shuffling
        __m128i sel1  = constant4i <i0 & 4 ? 0 : -1, i1 & 4 ? 0 : -1, i2 & 4 ? 0 : -1, i3 & 4 ? 0 : -1> ();
        __m128i mask1 = constant4i< -int(i0>=0), -int(i1>=0), -int(i2>=0), -int(i3>=0) >();
        __m128 t1 = selectf(_mm_castsi128_ps(sel1), a, b);   // select
        return  _mm_and_ps(t1, _mm_castsi128_ps(mask1));     // zero
    }
    // special cases unpckhps, unpcklps, shufps
    Vec4f t;
    if (((m1 ^ 0x05010400) & m2) == 0) {
        t = _mm_unpacklo_ps(a, b);
        goto DOZERO;
    }
    if (((m1 ^ 0x01050004) & m2) == 0) {
        t = _mm_unpacklo_ps(b, a);
        goto DOZERO;
    }
    if (((m1 ^ 0x07030602) & m2) == 0) {
        t = _mm_unpackhi_ps(a, b);
        goto DOZERO;
    }
    if (((m1 ^ 0x03070206) & m2) == 0) {
        t = _mm_unpackhi_ps(b, a);
        goto DOZERO;
    }    
    // first two elements from a, last two from b
    if (((m1^0x04040000) & 0x04040404 & m2) == 0) {
        t = _mm_shuffle_ps(a, b, (i0&3) + ((i1&3)<<2) + ((i2&3)<<4) + ((i3&3)<<6));
        goto DOZERO;
    } 
    // first two elements from b, last two from a
    if (((m1^0x00000404) & 0x04040404 & m2) == 0) {
        t = _mm_shuffle_ps(b, a, (i0&3) + ((i1&3)<<2) + ((i2&3)<<4) + ((i3&3)<<6));
        goto DOZERO;
    }
    {   // general case. combine two permutes
        __m128 a1 = permute4f <
            (uint32_t)i0 < 4 ? i0 : -1,
            (uint32_t)i1 < 4 ? i1 : -1,
            (uint32_t)i2 < 4 ? i2 : -1,
            (uint32_t)i3 < 4 ? i3 : -1  > (a);
        __m128 b1 = permute4f <
            (uint32_t)(i0^4) < 4 ? (i0^4) : -1,
            (uint32_t)(i1^4) < 4 ? (i1^4) : -1,
            (uint32_t)(i2^4) < 4 ? (i2^4) : -1,
            (uint32_t)(i3^4) < 4 ? (i3^4) : -1  > (b);
        return  _mm_or_ps(a1,b1);
    }
DOZERO:
    if ((i0|i1|i2|i3) & 0x80) {
        // zero some elements
        __m128i mask1 = constant4i< -int(i0>=0), -int(i1>=0), -int(i2>=0), -int(i3>=0) >();
        t = _mm_and_ps(t,_mm_castsi128_ps(mask1));     // zero with AND mask
    }        
    return t;

#endif // __XOP__
}

// change signs on vectors Vec4f
// Each index i0 - i3 is 1 for changing sign on the corresponding element, 0 for no change
template <int i0, int i1, int i2, int i3>
static inline Vec4f change_sign(Vec4f const & a) {
    if ((i0 | i1 | i2 | i3) == 0) return a;
    __m128i mask = constant4ui<i0 ? 0x80000000 : 0, i1 ? 0x80000000 : 0, i2 ? 0x80000000 : 0, i3 ? 0x80000000 : 0>();
    return  _mm_xor_ps(a, _mm_castsi128_ps(mask));     // flip sign bits
}


/*****************************************************************************
*
*          Vec2d: Vector of 2 double precision floating point values
*
*****************************************************************************/

class Vec2d {
protected:
    pack2d_t xmm; // double vector
public:
    // Default constructor:
    Vec2d() {
    }
    // Constructor to broadcast the same value into all elements:
    Vec2d(double d) {
        xmm = nsimd::set1<pack2d_t>(d);
    }
    // Constructor to build from all elements:
    Vec2d(double d0, double d1) {
        double data[2] = {d0, d1};
        xmm = nsimd::loadu<pack2d_t>(data); 
    }
    // Constructor to convert from type pack2d_t used in intrinsics:
    Vec2d(pack2d_t const & x) {
        xmm = x;
    }
    // Assignment operator to convert from type pack2d_t used in intrinsics:
    Vec2d & operator = (pack2d_t const & x) {
        xmm = x;
        return *this;
    }
    // Type cast operator to convert to __m128d used in intrinsics
    operator pack2d_t() const {
        return xmm;
    }
    // Member function to load from array (unaligned)
    Vec2d & load(double const * p) {
        xmm = nsimd::loadu<pack2d_t>(p);
        return *this;
    }
    // Member function to load from array, aligned by 16
    // "load_a" is faster than "load" on older Intel processors (Pentium 4, Pentium M, Core 1,
    // Merom, Wolfdale) and Atom, but not on other processors from Intel, AMD or VIA.
    // You may use load_a instead of load if you are certain that p points to an address
    // divisible by 16.
    Vec2d const & load_a(double const * p) {
        xmm = nsimd::loada<pack2d_t>(p);
        return *this;
    }
    // Member function to store into array (unaligned)
    void store(double * p) const {
        nsimd::storeu(p, xmm);
    }
    // Member function to store into array, aligned by 16
    // "store_a" is faster than "store" on older Intel processors (Pentium 4, Pentium M, Core 1,
    // Merom, Wolfdale) and Atom, but not on other processors from Intel, AMD or VIA.
    // You may use store_a instead of store if you are certain that p points to an address
    // divisible by 16.
    void store_a(double * p) const {
        nsimd::storea(p, xmm);
    }
    // Partial load. Load n elements and set the rest to 0
    Vec2d & load_partial(int n, double const * p) {
        xmm = nsimd_common::load_partial<pack2d_t, packl2d_t, double>(p, n);
        return *this;
    }
    // Partial store. Store n elements
    void store_partial(int n, double * p) const {
        nsimd_common::store_partial<pack2d_t, packl2d_t, double>(p, n, xmm);
    }
    // cut off vector to n elements. The last 4-n elements are set to zero
    Vec2d & cutoff(int n) {
        xmm = nsimd_common::cutoff<pack2d_t, packl2d_t, double>(xmm, n);
        return *this;
    }
    // Member function to change a single element in vector
    // Note: This function is inefficient. Use load function if changing more than one element
    Vec2d const & insert(uint32_t index, double value) {
        xmm = nsimd_common::set_bit<pack2d_t, double>(index, value, xmm);
        return *this;
    };
    // Member function extract a single element from vector
    double extract(uint32_t index) const {
        double x[2];
        store(x);
        return x[index & 1];
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    double operator [] (uint32_t index) const {
        return extract(index);
    }
    static int size() {
        return 2;
    }
};


/*****************************************************************************
*
*          Operators for Vec2d
*
*****************************************************************************/

// vector operator + : add element by element
static inline Vec2d operator + (Vec2d const & a, Vec2d const & b) {
    return nsimd::add(pack2d_t(a), pack2d_t(b));
}

// vector operator + : add vector and scalar
static inline Vec2d operator + (Vec2d const & a, double b) {
    return a + Vec2d(b);
}
static inline Vec2d operator + (double a, Vec2d const & b) {
    return Vec2d(a) + b;
}

// vector operator += : add
static inline Vec2d & operator += (Vec2d & a, Vec2d const & b) {
    a = a + b;
    return a;
}

// postfix operator ++
static inline Vec2d operator ++ (Vec2d & a, int) {
    Vec2d a0 = a;
    a = a + 1.0;
    return a0;
}

// prefix operator ++
static inline Vec2d & operator ++ (Vec2d & a) {
    a = a + 1.0;
    return a;
}

// vector operator - : subtract element by element
static inline Vec2d operator - (Vec2d const & a, Vec2d const & b) {
    return nsimd::sub(pack2d_t(a), pack2d_t(b));
}

// vector operator - : subtract vector and scalar
static inline Vec2d operator - (Vec2d const & a, double b) {
    return a - Vec2d(b);
}
static inline Vec2d operator - (double a, Vec2d const & b) {
    return Vec2d(a) - b;
}

// vector operator - : unary minus
// Change sign bit, even for 0, INF and NAN
static inline Vec2d operator - (Vec2d const & a) {
    return nsimd::xorb(pack2d_t(a), nsimd::set1<pack2d_t>(0.0));
}

// vector operator -= : subtract
static inline Vec2d & operator -= (Vec2d & a, Vec2d const & b) {
    a = a - b;
    return a;
}

// postfix operator --
static inline Vec2d operator -- (Vec2d & a, int) {
    Vec2d a0 = a;
    a = a - 1.0;
    return a0;
}

// prefix operator --
static inline Vec2d & operator -- (Vec2d & a) {
    a = a - 1.0;
    return a;
}

// vector operator * : multiply element by element
static inline Vec2d operator * (Vec2d const & a, Vec2d const & b) {
    return nsimd::mul(pack2d_t(a), pack2d_t(b));
}

// vector operator * : multiply vector and scalar
static inline Vec2d operator * (Vec2d const & a, double b) {
    return a * Vec2d(b);
}
static inline Vec2d operator * (double a, Vec2d const & b) {
    return Vec2d(a) * b;
}

// vector operator *= : multiply
static inline Vec2d & operator *= (Vec2d & a, Vec2d const & b) {
    a = a * b;
    return a;
}

// vector operator / : divide all elements by same integer
static inline Vec2d operator / (Vec2d const & a, Vec2d const & b) {
    return nsimd::div(pack2d_t(a), pack2d_t(b));
}

// vector operator / : divide vector and scalar
static inline Vec2d operator / (Vec2d const & a, double b) {
    return a / Vec2d(b);
}
static inline Vec2d operator / (double a, Vec2d const & b) {
    return Vec2d(a) / b;
}

// vector operator /= : divide
static inline Vec2d & operator /= (Vec2d & a, Vec2d const & b) {
    a = a / b;
    return a;
}

// vector operator == : returns true for elements for which a == b
static inline Vec2db operator == (Vec2d const & a, Vec2d const & b) {
    return nsimd::eq(pack2d_t(a), pack2d_t(b));
}

// vector operator != : returns true for elements for which a != b
static inline Vec2db operator != (Vec2d const & a, Vec2d const & b) {
    return nsimd::ne(pack2d_t(a), pack2d_t(b));
}

// vector operator < : returns true for elements for which a < b
static inline Vec2db operator < (Vec2d const & a, Vec2d const & b) {
    return nsimd::lt(pack2d_t(a), pack2d_t(b));
}

// vector operator <= : returns true for elements for which a <= b
static inline Vec2db operator <= (Vec2d const & a, Vec2d const & b) {
    return nsimd::le(pack2d_t(a), pack2d_t(b));
}

// vector operator > : returns true for elements for which a > b
static inline Vec2db operator > (Vec2d const & a, Vec2d const & b) {
    return b < a;
}

// vector operator >= : returns true for elements for which a >= b
static inline Vec2db operator >= (Vec2d const & a, Vec2d const & b) {
    return b <= a;
}

// Bitwise logical operators

// vector operator & : bitwise and
static inline Vec2d operator & (Vec2d const & a, Vec2d const & b) {
    return nsimd::andb(pack2d_t(a), pack2d_t(b));
}

// vector operator &= : bitwise and
static inline Vec2d & operator &= (Vec2d & a, Vec2d const & b) {
    a = a & b;
    return a;
}

// vector operator & : bitwise and of Vec2d and Vec2db
static inline Vec2d operator & (Vec2d const & a, Vec2db const & b) {
    return nsimd::andb(pack2d_t(a), pack2d_t(b));
}
static inline Vec2d operator & (Vec2db const & a, Vec2d const & b) {
    return nsimd::andb(pack2d_t(a), pack2d_t(b));
}

// vector operator | : bitwise or
static inline Vec2d operator | (Vec2d const & a, Vec2d const & b) {
    return nsimd::orb(pack2d_t(a), pack2d_t(b));
}

// vector operator |= : bitwise or
static inline Vec2d & operator |= (Vec2d & a, Vec2d const & b) {
    a = a | b;
    return a;
}

// vector operator ^ : bitwise xor
static inline Vec2d operator ^ (Vec2d const & a, Vec2d const & b) {
    return nsimd::xorb(pack2d_t(a), pack2d_t(b));
}

// vector operator ^= : bitwise xor
static inline Vec2d & operator ^= (Vec2d & a, Vec2d const & b) {
    a = a ^ b;
    return a;
}

// vector operator ! : logical not. Returns Boolean vector
static inline Vec2db operator ! (Vec2d const & a) {
    return a == Vec2d(0.0);
}


/*****************************************************************************
*
*          Functions for Vec2d
*
*****************************************************************************/

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 2; i++) result[i] = s[i] ? a[i] : b[i];
// Each byte in s must be either 0 (false) or 0xFFFFFFFFFFFFFFFF (true). 
// No other values are allowed.
static inline Vec2d select (Vec2db const & s, Vec2d const & a, Vec2d const & b) {
    return selectd(s,a,b);
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
static inline Vec2d if_add (Vec2db const & f, Vec2d const & a, Vec2d const & b) {
    return a + (Vec2d(f) & b);
}

// Conditional multiply: For all vector elements i: result[i] = f[i] ? (a[i] * b[i]) : a[i]
static inline Vec2d if_mul (Vec2db const & f, Vec2d const & a, Vec2d const & b) {
    return a * select(f, b, 1.);
}


// General arithmetic functions, etc.

// Horizontal add: Calculates the sum of all vector elements.
static inline double horizontal_add (Vec2d const & a) {
    return nsimd::addv((pack2d_t)a);
}

// function max: a > b ? a : b
static inline Vec2d max(Vec2d const & a, Vec2d const & b) {
    return nsimd::max((pack2d_t)a,(pack2d_t)b);
}

// function min: a < b ? a : b
static inline Vec2d min(Vec2d const & a, Vec2d const & b) {
    return nsimd::min((pack2d_t)a,(pack2d_t)b);
}

// function abs: absolute value
// Removes sign bit, even for -0.0f, -INF and -NAN
static inline Vec2d abs(Vec2d const & a) {
    return nsimd::abs((pack2d_t)a);
}

// function sqrt: square root
static inline Vec2d sqrt(Vec2d const & a) {
    return nsimd::sqrt((pack2d_t)a);
}

// function square: a * a
static inline Vec2d square(Vec2d const & a) {
    return nsimd::min((pack2d_t)a,(pack2d_t)a);
}

// pow(Vec2d, int):
// The purpose of this template is to prevent implicit conversion of a float
// exponent to int when calling pow(vector, float) and vectormath_exp.h is
// not included

template <typename TT> static Vec2d pow(Vec2d const & a, TT const & n);

// Raise floating point numbers to integer power n
template <>
inline Vec2d pow<int>(Vec2d const & x0, int const & n) {
    return pow_template_i<Vec2d>(x0, n);
}

// allow conversion from unsigned int
template <>
inline Vec2d pow<uint32_t>(Vec2d const & x0, uint32_t const & n) {
    return pow_template_i<Vec2d>(x0, (int)n);
}


// Raise floating point numbers to integer power n, where n is a compile-time constant
template <int n>
static inline Vec2d pow_n(Vec2d const & a) {
    if (n < 0)    return Vec2d(1.0) / pow_n<-n>(a);
    if (n == 0)   return Vec2d(1.0);
    if (n >= 256) return pow(a, n);
    Vec2d x = a;                       // a^(2^i)
    Vec2d y;                           // accumulator
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
static inline Vec2d pow(Vec2d const & a, Const_int_t<n>) {
    return pow_n<n>(a);
}


// avoid unsafe optimization in function round
static inline Vec2d round(Vec2d const & a)
{
    return nsimd::round_to_even((pack2d_t)a);
}

// function truncate: round towards zero. (result as double vector)
static inline Vec2d truncate(Vec2d const & a) {
    return nsimd::trunc((pack2d_t)a);
}

// function floor: round towards minus infinity. (result as double vector)
// (note: may fail on MS Visual Studio 2008, works in later versions)
static inline Vec2d floor(Vec2d const & a) {
    return nsimd::floor((pack2d_t)a);
}

// function ceil: round towards plus infinity. (result as double vector)
static inline Vec2d ceil(Vec2d const & a) {
    return nsimd::ceil((pack2d_t)a);
}

// function truncate_to_int: round towards zero.
static inline Vec4i truncate_to_int(Vec2d const & a, Vec2d const & b) {
    Vec4i t1 = Vec4i(nsimd::cvt<pack128_8i_t>((pack2d_t)a));
    Vec4i t2 = Vec4i(nsimd::cvt<pack128_8i_t>((pack2d_t)b));
    return blend4i<0,1,4,5> (t1, t2);
}

// function round_to_int: round to nearest integer (even).
// result as 32-bit integer vector
static inline Vec4i round_to_int(Vec2d const & a, Vec2d const & b) {
    // Note: assume MXCSR control register is set to rounding
    Vec4i t1 = Vec4i(nsimd::cvt<pack128_8i_t>((pack2d_t)a));
    Vec4i t2 = Vec4i(nsimd::cvt<pack128_8i_t>((pack2d_t)b));
    return blend4i<0,1,4,5> (t1, t2);
}
// function round_to_int: round to nearest integer (even).
// result as 32-bit integer vector. Upper two values of result are 0
static inline Vec4i round_to_int(Vec2d const & a) {
    Vec4i t1 = Vec4i(nsimd::cvt<pack128_8i_t>((pack2d_t)a));
    return t1;
}

// function truncate_to_int64: round towards zero. (inefficient)
static inline Vec2q truncate_to_int64(Vec2d const & a) {
    double aa[2];
    a.store(aa);
    return Vec2q(int64_t(aa[0]), int64_t(aa[1]));
}

// function truncate_to_int64_limited: round towards zero. (inefficient)
// result as 64-bit integer vector, but with limited range. Deprecated!
static inline Vec2q truncate_to_int64_limited(Vec2d const & a) {
    // Note: assume MXCSR control register is set to rounding
    Vec4i t1 = Vec4i(nsimd::cvt<pack128_8i_t>((pack2d_t)a));
    return extend_low(t1);
}

// function round_to_int64: round to nearest or even. (inefficient)
static inline Vec2q round_to_int64(Vec2d const & a) {
    return Vec2q(nsimd::cvt<pack128_4i_t>((pack2d_t)a));
}

// function round_to_int: round to nearest integer (even)
// result as 64-bit integer vector, but with limited range. Deprecated!
static inline Vec2q round_to_int64_limited(Vec2d const & a) {
    return round_to_int64(a);
#endif
}

// function to_double: convert integer vector elements to double vector (inefficient)
static inline Vec2d to_double(Vec2q const & a) {
    return Vec2d(nsimd::cvt<pack2d_t>((pack128_4i_t)a));
}

// function to_double_limited: convert integer vector elements to double vector
// limited to abs(x) < 2^31. Deprecated!
static inline Vec2d to_double_limited(Vec2q const & x) {
    return to_double(x);
}

// function to_double_low: convert integer vector elements [0] and [1] to double vector
static inline Vec2d to_double_low(Vec4i const & a) {
    return Vec2d(nsimd::cvt<pack2d_t>((pack128_4i_t)a));
}

// function to_double_high: convert integer vector elements [2] and [3] to double vector
static inline Vec2d to_double_high(Vec4i const & a) {
    return to_double_low(nsimd::shr((pack128_4i_t)a,8));
}

// function compress: convert two Vec2d to one Vec4f
static inline Vec4f compress (Vec2d const & low, Vec2d const & high) {
    Vec4f t1 = Vec4f(nsimd::cvt<pack4f_t>((pack2d_t)low));
    Vec4f t2 = Vec4f(nsimd::cvt<pack4f_t>((pack2d_t)high));
    return blend4f<0,1,4,5> (t1, t2);
}

// Function extend_low : convert Vec4f vector elements [0] and [1] to Vec2d
static inline Vec2d extend_low (Vec4f const & a) {
    return Vec2d(nsimd::cvt<pack2d_t>((pack4f_t)a));
}

// Function extend_high : convert Vec4f vector elements [2] and [3] to Vec2d
static inline Vec2d extend_high (Vec4f const & a) {
    return _mm_cvtps_pd(_mm_movehl_ps(a,a));
}


// Fused multiply and add functions

// Multiply and add
static inline Vec2d mul_add(Vec2d const & a, Vec2d const & b, Vec2d const & c) {
    return nsimd::fma((pack2d_t)a, (pack2d_t)b, (pack2d_t)c);
}

// Multiply and subtract
static inline Vec2d mul_sub(Vec2d const & a, Vec2d const & b, Vec2d const & c) {
    return nsimd::fms((pack2d_t)a, (pack2d_t)b, (pack2d_t)c);
}

// Multiply and inverse subtract
static inline Vec2d nmul_add(Vec2d const & a, Vec2d const & b, Vec2d const & c) {
return nsimd::fnma((pack2d_t)a, (pack2d_t)b, (pack2d_t)c);
}


// Multiply and subtract with extra precision on the intermediate calculations, 
// even if FMA instructions not supported, using Veltkamp-Dekker split
static inline Vec2d mul_sub_x(Vec2d const & a, Vec2d const & b, Vec2d const & c) {
    return nsimd::fms((pack2d_t)a, (pack2d_t)b, (pack2d_t)c);
}


// Math functions using fast bit manipulation

// Extract the exponent as an integer
// exponent(a) = floor(log2(abs(a)));
// exponent(1.0) = 0, exponent(0.0) = -1023, exponent(INF) = +1024, exponent(NAN) = +1024
static inline Vec2q exponent(Vec2d const & a) {
    Vec2uq t1 = _mm_castpd_si128(a);   // reinterpret as 64-bit integer
    Vec2uq t2 = t1 << 1;               // shift out sign bit
    Vec2uq t3 = t2 >> 53;              // shift down logical to position 0
    Vec2q  t4 = Vec2q(t3) - 0x3FF;     // subtract bias from exponent
    return t4;
}

// Extract the fraction part of a floating point number
// a = 2^exponent(a) * fraction(a), except for a = 0
// fraction(1.0) = 1.0, fraction(5.0) = 1.25
// NOTE: The name fraction clashes with an ENUM in MAC XCode CarbonCore script.h !
static inline Vec2d fraction(Vec2d const & a) {
    Vec2uq t1 = _mm_castpd_si128(a);   // reinterpret as 64-bit integer
    Vec2uq t2 = Vec2uq((t1 & 0x000FFFFFFFFFFFFFll) | 0x3FF0000000000000ll); // set exponent to 0 + bias
    return _mm_castsi128_pd(t2);
}

// Fast calculation of pow(2,n) with n integer
// n  =     0 gives 1.0
// n >=  1024 gives +INF
// n <= -1023 gives 0.0
// This function will never produce denormals, and never raise exceptions
static inline Vec2d exp2(Vec2q const & n) {
    Vec2q t1 = max(n,  -0x3FF);        // limit to allowed range
    Vec2q t2 = min(t1,  0x400);
    Vec2q t3 = t2 + 0x3FF;             // add bias
    Vec2q t4 = t3 << 52;               // put exponent into position 52
    return _mm_castsi128_pd(t4);       // reinterpret as double
}
//static Vec2d exp2(Vec2d const & x); // defined in vectormath_exp.h


// Categorization functions

// Function sign_bit: gives true for elements that have the sign bit set
// even for -0.0, -INF and -NAN
// Note that sign_bit(Vec2d(-0.0)) gives true, while Vec2d(-0.0) < Vec2d(0.0) gives false
static inline Vec2db sign_bit(Vec2d const & a) {
    Vec2q t1 = _mm_castpd_si128(a);    // reinterpret as 64-bit integer
    Vec2q t2 = t1 >> 63;               // extend sign bit
    return _mm_castsi128_pd(t2);       // reinterpret as 64-bit Boolean
}

// Function sign_combine: changes the sign of a when b has the sign bit set
// same as select(sign_bit(b), -a, a)
static inline Vec2d sign_combine(Vec2d const & a, Vec2d const & b) {
    Vec2d signmask = _mm_castsi128_pd(constant4ui<0,0x80000000,0,0x80000000>());  // -0.0
    return a ^ (b & signmask);
}

// Function is_finite: gives true for elements that are normal, denormal or zero, 
// false for INF and NAN
static inline Vec2db is_finite(Vec2d const & a) {
    Vec2q t1 = _mm_castpd_si128(a);    // reinterpret as integer
    Vec2q t2 = t1 << 1;                // shift out sign bit
    Vec2q t3 = 0xFFE0000000000000ll;   // exponent mask
    Vec2qb t4 = Vec2q(t2 & t3) != t3;  // exponent field is not all 1s
    return t4;
}

// Function is_inf: gives true for elements that are +INF or -INF
// false for finite numbers and NAN
static inline Vec2db is_inf(Vec2d const & a) {
    Vec2q t1 = _mm_castpd_si128(a);    // reinterpret as integer
    Vec2q t2 = t1 << 1;                // shift out sign bit
    return t2 == 0xFFE0000000000000ll; // exponent is all 1s, fraction is 0
}

// Function is_nan: gives true for elements that are +NAN or -NAN
// false for finite numbers and +/-INF
static inline Vec2db is_nan(Vec2d const & a) {
    Vec2q t1 = _mm_castpd_si128(a);    // reinterpret as integer
    Vec2q t2 = t1 << 1;                // shift out sign bit
    Vec2q t3 = 0xFFE0000000000000ll;   // exponent mask
    Vec2q t4 = t2 & t3;                // exponent
    Vec2q t5 = _mm_andnot_si128(t3,t2);// fraction
    return Vec2qb((t4==t3) & (t5!=0)); // exponent = all 1s and fraction != 0
}

// Function is_subnormal: gives true for elements that are subnormal (denormal)
// false for finite numbers, zero, NAN and INF
static inline Vec2db is_subnormal(Vec2d const & a) {
    Vec2q t1 = _mm_castpd_si128(a);    // reinterpret as 32-bit integer
    Vec2q t2 = t1 << 1;                // shift out sign bit
    Vec2q t3 = 0xFFE0000000000000ll;   // exponent mask
    Vec2q t4 = t2 & t3;                // exponent
    Vec2q t5 = _mm_andnot_si128(t3,t2);// fraction
    return Vec2qb((t4==0) & (t5!=0));  // exponent = 0 and fraction != 0
}

// Function is_zero_or_subnormal: gives true for elements that are zero or subnormal (denormal)
// false for finite numbers, NAN and INF
static inline Vec2db is_zero_or_subnormal(Vec2d const & a) {
    Vec2q t = _mm_castpd_si128(a);     // reinterpret as 32-bit integer
          t &= 0x7FF0000000000000ll;   // isolate exponent
    return t == 0;                     // exponent = 0
}

// Function infinite2d: returns a vector where all elements are +INF
static inline Vec2d infinite2d() {
    return _mm_castsi128_pd(_mm_setr_epi32(0,0x7FF00000,0,0x7FF00000));
}

// Function nan2d: returns a vector where all elements are +NAN (quiet)
static inline Vec2d nan2d(int n = 0x10) {
    return _mm_castsi128_pd(_mm_setr_epi32(n, 0x7FF80000, n, 0x7FF80000));
}


/*****************************************************************************
*
*          Functions for reinterpretation between vector types
*
*****************************************************************************/

static inline __m128i reinterpret_i (__m128i const & x) {
    return x;
}

static inline __m128i reinterpret_i (__m128  const & x) {
    return _mm_castps_si128(x);
}

static inline __m128i reinterpret_i (__m128d const & x) {
    return _mm_castpd_si128(x);
}

static inline __m128  reinterpret_f (__m128i const & x) {
    return _mm_castsi128_ps(x);
}

static inline __m128  reinterpret_f (__m128  const & x) {
    return x;
}

static inline __m128  reinterpret_f (__m128d const & x) {
    return _mm_castpd_ps(x);
}

static inline __m128d reinterpret_d (__m128i const & x) {
    return _mm_castsi128_pd(x);
}

static inline __m128d reinterpret_d (__m128  const & x) {
    return _mm_castps_pd(x);
}

static inline __m128d reinterpret_d (__m128d const & x) {
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
* Vec2d a(10., 11.);              // a is (10, 11)
* Vec2d b, c;
* b = permute2d<1,1>(a);          // b is (11, 11)
* c = permute2d<-1,0>(a);         // c is ( 0, 10)
*
*
* The blend function can mix elements from two different vectors and
* optionally set some elements to zero. 
*
* The indexes are inserted as template parameters in <>. These indexes must be
* constants. Each template parameter is an index to the element you want to 
* select, where indexes 0 - 1 indicate an element from the first source
* vector and indexes 2 - 3 indicate an element from the second source vector.
* An index of -1 will generate zero.
*
*
* Example:
* Vec2d a(10., 11.);              // a is (10, 11)
* Vec2d b(20., 21.);              // b is (20, 21)
* Vec2d c;
* c = blend2d<0,3> (a,b);         // c is (10, 21)
*
* A lot of the code here is metaprogramming aiming to find the instructions
* that best fit the template parameters and instruction set. The metacode
* will be reduced out to leave only a few vector instructions in release
* mode with optimization on.
*****************************************************************************/

// permute vector Vec2d
template <int i0, int i1>
static inline Vec2d permute2d(Vec2d const & a) {
    // is shuffling needed
    const bool do_shuffle = (i0 > 0) || (i1 != 1 && i1 >= 0);
    // is zeroing needed
    const bool do_zero    = ((i0 | i1) < 0 && (i0 | i1) & 0x80);

    if (do_zero && !do_shuffle) {                          // zeroing, not shuffling
        if ((i0 & i1) < 0) return _mm_setzero_pd();        // zero everything
        // zero some elements
        __m128i mask1 = constant4i< -int(i0>=0), -int(i0>=0), -int(i1>=0), -int(i1>=0) >();
        return  _mm_and_pd(a,_mm_castsi128_pd(mask1));     // zero with AND mask
    }
    else if (do_shuffle && !do_zero) {                     // shuffling, not zeroing        
        return _mm_shuffle_pd(a, a, (i0&1) | (i1&1)<<1);
    }
    else if (do_shuffle && do_zero) {                      // shuffling and zeroing        
        // both shuffle and zero
        if (i0 < 0 && i1 >= 0) {                           // zero low half, shuffle high half
            return _mm_shuffle_pd(_mm_setzero_pd(), a, (i1 & 1) << 1);
        }
        if (i0 >= 0 && i1 < 0) {                           // shuffle low half, zero high half
            return _mm_shuffle_pd(a, _mm_setzero_pd(), i0 & 1);
        }
    }
    return a;                                              // trivial case: do nothing
}


// blend vectors Vec2d
template <int i0, int i1>
static inline Vec2d blend2d(Vec2d const & a, Vec2d const & b) {

    // Combine all the indexes into a single bitfield, with 8 bits for each
    const int m1 = (i0 & 3) | (i1 & 3) << 8; 

    // Mask to zero out negative indexes
    const int m2 = (i0 < 0 ? 0 : 0xFF) | (i1 < 0 ? 0 : 0xFF) << 8;

    if ((m1 & 0x0202 & m2) == 0) {
        // no elements from b, only elements from a and possibly zero
        return permute2d <i0, i1> (a);
    }
    if (((m1^0x0202) & 0x0202 & m2) == 0) {
        // no elements from a, only elements from b and possibly zero
        return permute2d <i0 & ~2, i1 & ~2> (b);
    }
    // selecting from both a and b without zeroing
    if ((i0 & 2) == 0) { // first element from a, second element from b
        return _mm_shuffle_pd(a, b, (i0 & 1) | (i1 & 1) << 1);
    }
    else {         // first element from b, second element from a
        return _mm_shuffle_pd(b, a, (i0 & 1) | (i1 & 1) << 1);
    }
}

// change signs on vectors Vec4f
// Each index i0 - i1 is 1 for changing sign on the corresponding element, 0 for no change
template <int i0, int i1>
static inline Vec2d change_sign(Vec2d const & a) {
    if ((i0 | i1) == 0) return a;
    __m128i mask = constant4ui<0, i0 ? 0x80000000 : 0, 0, i1 ? 0x80000000 : 0> ();
    return  _mm_xor_pd(a, _mm_castsi128_pd(mask));     // flip sign bits
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

static inline Vec4f lookup4(Vec4i const & index, Vec4f const & table) {
#if INSTRSET >= 7  // AVX
    return _mm_permutevar_ps(table, index);
#else
    int32_t ii[4];
    float   tt[6];
    table.store(tt);  (index & 3).store(ii);
    __m128 r01 = _mm_loadh_pi(_mm_load_ss(&tt[ii[0]]), (const __m64 *)&tt[ii[1]]);
    __m128 r23 = _mm_loadh_pi(_mm_load_ss(&tt[ii[2]]), (const __m64 *)&tt[ii[3]]);
    return _mm_shuffle_ps(r01, r23, 0x88);
#endif
}

static inline Vec4f lookup8(Vec4i const & index, Vec4f const & table0, Vec4f const & table1) {
#if INSTRSET >= 8  // AVX2
    __m256 tt = _mm256_insertf128_ps(_mm256_castps128_ps256(table0), table1, 1); // combine tables

#if defined (_MSC_VER) && _MSC_VER < 1700 && ! defined(__INTEL_COMPILER)        
    // bug in MS VS 11 beta: operands in wrong order
    __m128 r = _mm256_castps256_ps128(_mm256_permutevar8x32_ps(_mm256_castsi256_ps(_mm256_castsi128_si256(index)), _mm256_castps_si256(tt))); 
    r = _mm_and_ps(r,r); // fix another bug in VS 11 beta (would store r as 256 bits aligned by 16)
#elif defined (GCC_VERSION) && GCC_VERSION <= 40700 && !defined(__INTEL_COMPILER) && !defined(__clang__)
    // Gcc 4.7.0 has wrong parameter type and operands in wrong order
    __m128 r = _mm256_castps256_ps128(_mm256_permutevar8x32_ps(_mm256_castsi256_ps(_mm256_castsi128_si256(index)), tt)); 
#else
    // no bug version
    __m128 r = _mm256_castps256_ps128(_mm256_permutevar8x32_ps(tt, _mm256_castsi128_si256(index)));
#endif
    return r;

#elif INSTRSET >= 7  // AVX 
    __m128  r0 = _mm_permutevar_ps(table0, index);
    __m128  r1 = _mm_permutevar_ps(table1, index);
    __m128i i4 = _mm_slli_epi32(index, 29);
    return _mm_blendv_ps(r0, r1, _mm_castsi128_ps(i4));

#elif INSTRSET >= 5  // SSE4.1
    Vec4f   r0 = lookup4(index, table0);
    Vec4f   r1 = lookup4(index, table1);
    __m128i i4 = _mm_slli_epi32(index, 29);
    return _mm_blendv_ps(r0, r1, _mm_castsi128_ps(i4));

#else               // SSE2
    Vec4f   r0 = lookup4(index, table0);
    Vec4f   r1 = lookup4(index, table1);
    __m128i i4 = _mm_srai_epi32(_mm_slli_epi32(index, 29), 31);
    return selectf(_mm_castsi128_ps(i4), r1, r0);
#endif
}

template <int n>
static inline Vec4f lookup(Vec4i const & index, float const * table) {
    if (n <= 0) return 0.0f;
    if (n <= 4) return lookup4(index, Vec4f().load(table));
    if (n <= 8) {
#if INSTRSET >= 8  // AVX2
        __m256 tt = _mm256_loadu_ps(table);
#if defined (_MSC_VER) && _MSC_VER < 1700 && ! defined(__INTEL_COMPILER)        
        // bug in MS VS 11 beta: operands in wrong order
        __m128 r = _mm256_castps256_ps128(_mm256_permutevar8x32_ps(_mm256_castsi256_ps(_mm256_castsi128_si256(index)), _mm256_castps_si256(tt)));
        r = _mm_and_ps(r,r); // fix another bug in VS 11 beta (would store r as 256 bits aligned by 16)
#elif defined (GCC_VERSION) && GCC_VERSION <= 40700 && !defined(__INTEL_COMPILER) && !defined(__clang__)
        // Gcc 4.7.0 has wrong parameter type and operands in wrong order
        __m128 r = _mm256_castps256_ps128(_mm256_permutevar8x32_ps(_mm256_castsi256_ps(_mm256_castsi128_si256(index)), tt));
#else
        // no bug version
        __m128 r = _mm256_castps256_ps128(_mm256_permutevar8x32_ps(tt, _mm256_castsi128_si256(index)));
#endif
        return r;
#else   // not AVX2
        return lookup8(index, Vec4f().load(table), Vec4f().load(table+4));
#endif  // INSTRSET
    }
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
#if INSTRSET >= 8  // AVX2
    return _mm_i32gather_ps(table, index1, 4);
#else
    uint32_t ii[4];  index1.store(ii);
    return Vec4f(table[ii[0]], table[ii[1]], table[ii[2]], table[ii[3]]);
#endif
}

static inline Vec2d lookup2(Vec2q const & index, Vec2d const & table) {
#if INSTRSET >= 7  // AVX
    return _mm_permutevar_pd(table, index + index);
#else
    int32_t ii[4];
    double  tt[2];
    table.store(tt);  (index & 1).store(ii);
    return Vec2d(tt[ii[0]], tt[ii[2]]);
#endif
}

static inline Vec2d lookup4(Vec2q const & index, Vec2d const & table0, Vec2d const & table1) {
#if INSTRSET >= 7  // AVX
    Vec2q index2 = index + index;          // index << 1
    __m128d r0 = _mm_permutevar_pd(table0, index2);
    __m128d r1 = _mm_permutevar_pd(table1, index2);
    __m128i i4 = _mm_slli_epi64(index, 62);
    return _mm_blendv_pd(r0, r1, _mm_castsi128_pd(i4));
#else
    int32_t ii[4];
    double  tt[4];
    table0.store(tt);  table1.store(tt + 2);  
    (index & 3).store(ii);
    return Vec2d(tt[ii[0]], tt[ii[2]]);
#endif
}

template <int n>
static inline Vec2d lookup(Vec2q const & index, double const * table) {
    if (n <= 0) return 0.0;
    if (n <= 2) return lookup2(index, Vec2d().load(table));
#if INSTRSET < 8  // not AVX2
    if (n <= 4) return lookup4(index, Vec2d().load(table), Vec2d().load(table + 2));
#endif
    // Limit index
    Vec2uq index1;
    if ((n & (n-1)) == 0) {
        // n is a power of 2, make index modulo n
        index1 = Vec2uq(index) & (n-1);
    }
    else {
        // n is not a power of 2, limit to n-1
        index1 = min(Vec2uq(index), n-1);
    }
#if INSTRSET >= 8  // AVX2
    return _mm_i64gather_pd(table, index1, 8);
#else
    uint32_t ii[4];  index1.store(ii);
    return Vec2d(table[ii[0]], table[ii[2]]);
#endif
}


/*****************************************************************************
*
*          Gather functions with fixed indexes
*
*****************************************************************************/
// Load elements from array a with indices i0, i1, i2, i3
template <int i0, int i1, int i2, int i3>
static inline Vec4f gather4f(void const * a) {
    return reinterpret_f(gather4i<i0, i1, i2, i3>(a));
}

// Load elements from array a with indices i0, i1
template <int i0, int i1>
static inline Vec2d gather2d(void const * a) {
    return reinterpret_d(gather2q<i0, i1>(a));
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

template <int i0, int i1, int i2, int i3>
static inline void scatter(Vec4f const & data, float * array) {
#if defined (__AVX512VL__)
    __m128i indx = constant4i<i0,i1,i2,i3>();
    __mmask16 mask = uint16_t(i0>=0 | (i1>=0)<<1 | (i2>=0)<<2 | (i3>=0)<<3);
    _mm_mask_i32scatter_ps(array, mask, indx, data, 4);
#else
    const int index[4] = {i0,i1,i2,i3};
    for (int i = 0; i < 4; i++) {
        if (index[i] >= 0) array[index[i]] = data[i];
    }
#endif
}

template <int i0, int i1>
static inline void scatter(Vec2d const & data, double * array) {
    if (i0 >= 0) array[i0] = data[0];
    if (i1 >= 0) array[i1] = data[1];
}

static inline void scatter(Vec4i const & index, uint32_t limit, Vec4f const & data, float * array) {
#if defined (__AVX512VL__)
    __mmask16 mask = _mm_cmplt_epu32_mask(index, Vec4ui(limit));
    _mm_mask_i32scatter_ps(array, mask, index, data, 4);
#else
    for (int i = 0; i < 4; i++) {
        if (uint32_t(index[i]) < limit) array[index[i]] = data[i];
    }
#endif
}

static inline void scatter(Vec2q const & index, uint32_t limit, Vec2d const & data, double * array) {
    if (uint64_t(index[0]) < uint64_t(limit)) array[index[0]] = data[0];
    if (uint64_t(index[1]) < uint64_t(limit)) array[index[1]] = data[1];
}

static inline void scatter(Vec4i const & index, uint32_t limit, Vec2d const & data, double * array) {
    if (uint32_t(index[0]) < limit) array[index[0]] = data[0];
    if (uint32_t(index[1]) < limit) array[index[1]] = data[1];
}

/*****************************************************************************
*
*          Horizontal scan functions
*
*****************************************************************************/

// Get index to the first element that is true. Return -1 if all are false
static inline int horizontal_find_first(Vec4fb const & x) {
    return horizontal_find_first(Vec4ib(x));
}

static inline int horizontal_find_first(Vec2db const & x) {
    return horizontal_find_first(Vec2qb(x));
}

// Count the number of elements that are true
static inline uint32_t horizontal_count(Vec4fb const & x) {
    return horizontal_count(Vec4ib(x));
}

static inline uint32_t horizontal_count(Vec2db const & x) {
    return horizontal_count(Vec2qb(x));
}

/*****************************************************************************
*
*          Boolean <-> bitfield conversion functions
*
*****************************************************************************/

// to_bits: convert boolean vector to integer bitfield
//static inline uint8_t to_bits(Vec4fb const & x) {
//    return to_bits(Vec4ib(x));
//}
//
//// to_Vec4fb: convert integer bitfield to boolean vector
//static inline Vec4fb to_Vec4fb(uint8_t x) {
//    return Vec4fb(to_Vec4ib(x));
//}
//
//// to_bits: convert boolean vector to integer bitfield
//static inline uint8_t to_bits(Vec2db const & x) {
//    return to_bits(Vec2qb(x));
//}
//
//// to_Vec2db: convert integer bitfield to boolean vector
//static inline Vec2db to_Vec2db(uint8_t x) {
//    return Vec2db(to_Vec2qb(x));
//}

#ifdef NSIMD_NAMESPACE
}
#endif

#endif // VECTORF128_H
