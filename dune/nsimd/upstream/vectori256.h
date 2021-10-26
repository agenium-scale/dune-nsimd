/****************************  vectori256.h   *******************************
* Author:        Agner Fog
* Date created:  2012-05-30
* Last modified: 2017-02-19
* Version:       1.27
* Project:       vector classes
* Description:
* Header file defining integer vector classes as interface to intrinsic 
* functions in x86 microprocessors with AVX2 and later instruction sets.
*
* Instructions:
* Use Gnu, Intel or Microsoft C++ compiler. Compile for the desired 
* instruction set, which must be at least AVX2. 
*
* The following vector classes are defined here:
* Vec256b   Vector of 256  1-bit unsigned  integers or Booleans
* Vec32c    Vector of  32  8-bit signed    integers
* Vec32uc   Vector of  32  8-bit unsigned  integers
* Vec32cb   Vector of  32  Booleans for use with Vec32c and Vec32uc
* Vec16s    Vector of  16  16-bit signed   integers
* Vec16us   Vector of  16  16-bit unsigned integers
* Vec16sb   Vector of  16  Booleans for use with Vec16s and Vec16us
* Vec8i     Vector of   8  32-bit signed   integers
* Vec8ui    Vector of   8  32-bit unsigned integers
* Vec8ib    Vector of   8  Booleans for use with Vec8i and Vec8ui
* Vec4q     Vector of   4  64-bit signed   integers
* Vec4uq    Vector of   4  64-bit unsigned integers
* Vec4qb    Vector of   4  Booleans for use with Vec4q and Vec4uq
*
* Each vector object is represented internally in the CPU as a 256-bit register.
* This header file defines operators and functions for these vectors.
*
* For example:
* Vec8i a(1,2,3,4,5,6,7,8), b(9,10,11,12,13,14,15,16), c;
* c = a + b;     // now c contains (10,12,14,16,18,20,22,24)
*
* For detailed instructions, see Nsimd.pdf
*
* (c) Copyright 2012-2017 GNU General Public License http://www.gnu.org/licenses
*****************************************************************************/

// check combination of header files
#if defined (VECTORI256_H)
#if    VECTORI256_H != 2
#error Two different versions of vectori256.h included
#endif
#else
#define VECTORI256_H  2

#ifdef VECTORF256_H
#error Please put header file vectori256.h before vectorf256.h
#endif


#if INSTRSET < 8   // AVX2 required
#error Wrong instruction set for vectori256.h, AVX2 required or use vectori256e.h
#endif

#include "vectori128.h"
#include "vector_types.h"
#include "nsimd_common.h"

#ifdef NSIMD_NAMESPACE
namespace NSIMD_NAMESPACE {
#endif

/*****************************************************************************
*
*         Join two 128-bit vectors
*
*****************************************************************************/
#define set_m128ir(lo,hi) _mm256_inserti128_si256(_mm256_castsi128_si256(lo),(hi),1)


/*****************************************************************************
*
*          Vector of 256 1-bit unsigned integers or Booleans
*
*****************************************************************************/
template <typename PackT, typename PacklT, template T>
class Vec256b {
protected:
    PackT ymm; // Integer vector
public:
    // Default constructor:
    Vec256b() {
    }
    // Constructor to broadcast the same value into all elements
    Vec256b(T i) {
        ymm = nsimd::set1<PackT>(i);
    }
    // Constructor to broadcast the same value into all elements
    Vec256b(int i) {
        ymm = nsimd::set1<PackT>(T(i));
    }

    // Constructor to build from two Vec128b:
    Vec256b(Vec128b<PackT, PacklT,T> const & a0, Vec128b<PackT, PacklT,T> const & a1) {
        ymm = set_m128ir(a0, a1);
    }
    // Constructor to convert from type PackT used in intrinsics:
    Vec256b(PackT const & x) {
        ymm = x;
    }
    // Assignment operator to convert from type PackT used in intrinsics:
    Vec256b & operator = (PackT const & x) {
        ymm = x;
        return *this;
    }
    // Type cast operator to convert to PackT used in intrinsics
    operator PackT() const {
        return ymm;
    }
    // Member function to load from array (unaligned)
    Vec256b & load(void const * p) {
        ymm = nsimd::loadu<PackT>((T const*)p);
        return *this;
    }
    // Member function to load from array, aligned by 32
    // You may use load_a instead of load if you are certain that p points to an address
    // divisible by 32, but there is hardly any speed advantage of load_a on modern processors
    Vec256b & load_a(void const * p) {
        ymm = nsimd::loada<PackT>((T const*)p);
        return *this;
    }
    // Partial load. Load n elements and set the rest to 0
    Vec256b & load_partial(int n, void const * p) {
        ymm = nsimd_common::load_partial<PackT, PacklT, T>(p, n)
        return *this;
    }
    // Partial store. Store n elements
    void store_partial(int n, void * p) const {
        nsimd_common::store_partial<PackT, PacklT, T>(p, n, ymm);
    }
    // cut off vector to n elements. The last 32-n elements are set to zero
    Vec256b & cutoff(int n) {
        ymm = nsimd_common::cutoff<PackT, PacklT, T>(ymm, n);
        return *this;
    }
    // Member function to store into array (unaligned)
    void store(void * p) const {
        nsimd::storeu<T>((T*)p, ymm);
    }
    // Member function to store into array, aligned by 32
    // You may use store_a instead of store if you are certain that p points to an address
    // divisible by 32, but there is hardly any speed advantage of load_a on modern processors
    void store_a(void * p) const {
        nsimd::storea<T>((T*)p, ymm);
    }
    // Member function to change a single bit
    // Note: This function is inefficient. Use load function if changing more than one bit
    Vec256b const & set_bit(uint32_t index, int value) {
        ymm = nsimd_common::set_bit<PackT, T>(index, value, ymm)
        return *this;
    }
    // Member function to get a single bit
    // Note: This function is inefficient. Use store function if reading more than one bit
    int get_bit(uint32_t index) const {
        return nsimd_common::get_bit<PackT, T>(index, ymm);
    }
    // Member function to change a single element in vector
    // Note: This function is inefficient. Use load function if changing more than one element
    Vec256b const & insert(uint32_t index, T value) {
        ymm = nsimd_common::set_bit<PackT, T>(index, value, ymm)
        return *this;
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    bool operator [] (uint32_t index) const {
        return get_bit(index) != 0;
    }
    // Member functions to split into two Vec128b:
    Vec128b<PackT, PacklT,T> get_low() const {
        return nsimd_common::get_low<>(ymm);
    }
    Vec128b<PackT, PacklT,T> get_high() const {
        return nsimd_common::get_high<>(ymm);
    }
    virtual static int size() {
        return 256;
    }
};


// Define operators for this class

// vector operator & : bitwise and
template <typename PackT, typename PacklT, template T>
static inline Vec256b<PackT,PacklT,T> operator & (Vec256b<PackT,PacklT,T> const & a, Vec256b<PackT,PacklT,T> const & b) {
    return nsimd::andb(a, b);
}
template <typename PackT, typename PacklT, template T>
static inline Vec256b<PackT,PacklT,T> operator && (Vec256b<PackT,PacklT,T> const & a, Vec256b<PackT,PacklT,T> const & b) {
    return a & b;
}

// vector operator | : bitwise or
template <typename PackT, typename PacklT, template T>
static inline Vec256b<PackT,PacklT,T> operator | (Vec256b<PackT,PacklT,T> const & a, Vec256b<PackT,PacklT,T> const & b) {
    return nsimd::orb(a, b);
}
template <typename PackT, typename PacklT, template T>
static inline Vec256b<PackT,PacklT,T> operator || (Vec256b<PackT,PacklT,T> const & a, Vec256b<PackT,PacklT,T> const & b) {
    return a | b;
}

// vector operator ^ : bitwise xor
template <typename PackT, typename PacklT, template T>
static inline Vec256b<PackT,PacklT,T> operator ^ (Vec256b<PackT,PacklT,T> const & a, Vec256b<PackT,PacklT,T> const & b) {
    return nsimd::xorb(a, b);
}

// vector operator ~ : bitwise not
template <typename PackT, typename PacklT, template T>
static inline Vec256b<PackT,PacklT,T> operator ~ (Vec256b<PackT,PacklT,T> const & a) {
    return nsimd::notb(a);
}

// vector operator &= : bitwise and
template <typename PackT, typename PacklT, template T>
static inline Vec256b<PackT,PacklT,T> & operator &= (Vec256b<PackT,PacklT,T> & a, Vec256b<PackT,PacklT,T> const & b) {
    a = nsimd::andb(a,b);
    return a;
}

// vector operator |= : bitwise or
template <typename PackT, typename PacklT, template T>
static inline Vec256b<PackT,PacklT,T> & operator |= (Vec256b<PackT,PacklT,T> & a, Vec256b<PackT,PacklT,T> const & b) {
    a = nsimd::orb(a,b);
    return a;
}

// vector operator ^= : bitwise xor
template <typename PackT, typename PacklT, template T>
static inline Vec256b<PackT,PacklT,T> & operator ^= (Vec256b<PackT,PacklT,T> & a, Vec256b<PackT,PacklT,T> const & b) {
    a = nsimd::xorb(a,b);
    return a;
}

// Define functions for this class

// function andnot: a & ~ b
template <typename PackT, typename PacklT, template T>
static inline Vec256b<PackT,PacklT,T> andnot (Vec256b<PackT,PacklT,T> const & a, Vec256b<PackT,PacklT,T> const & b) {
    return nsimd::andnotb(b, a);
}


/*****************************************************************************
*
*          Generate compile-time constant vector
*
*****************************************************************************/
// Generate a constant vector of 8 integers stored in memory.
// Can be converted to any integer vector type
template <int32_t i0, int32_t i1, int32_t i2, int32_t i3, int32_t i4, int32_t i5, int32_t i6, int32_t i7>
static inline pack256_8i_t constant8i() {
    int32_t data[8] = {i0, i1, i2, i3, i4, i5, i6, i7};
    pack256_8i_t res = nsimd::loadu<pack256_8i_t>(data);
    return res;
}

template <uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3, uint32_t i4, uint32_t i5, uint32_t i6, uint32_t i7>
static inline pack256_8ui_t constant8ui() {
    uint32_t data[8] = {i0, i1, i2, i3, i4, i5, i6, i7};
    pack256_8ui_t res = nsimd::loadu<pack256_8ui_t>(data);
    return res;
}

/*****************************************************************************
*
*          selectb function
*
*****************************************************************************/
// Select between two sources, byte by byte. Used in various functions and operators
// Corresponds to this pseudocode:
// for (int i = 0; i < 32; i++) result[i] = s[i] ? a[i] : b[i];
// Each byte in s must be either 0 (false) or 0xFF (true). No other values are allowed.
// Only bit 7 in each byte of s is checked, 
static inline pack256_32i_t selectb (pack256_32i_t const & s, pack256_32i_t const & a, pack256_32i_t const & b) {
    return return nsimd::if_else(nsimd::to_logical(b), a, s);
}



/*****************************************************************************
*
*          Horizontal Boolean functions
*
*****************************************************************************/

// horizontal_and. Returns true if all bits are 1
template <typename PackT, typename PacklT, template T>
static inline bool horizontal_and (Vec256b<PackT,PacklT,T> const & a) {
    return nsimd::all(nsimd::to_logical(a));
}

// horizontal_or. Returns true if at least one bit is 1
template <typename PackT, typename PacklT, template T>
static inline bool horizontal_or (Vec256b<PackT,PacklT,T> const & a) {
    return nsimd::any(nsimd::to_logical(a));
}



/*****************************************************************************
*
*          Vector of 32 8-bit signed integers
*
*****************************************************************************/

class Vec32c : public Vec256b<pack256_32i_t, packl256_32i_t, int8_t> {
public:
    // Default constructor:
    Vec32c(){
    }
    // Constructor to build from all elements:
    Vec32c(int8_t i0, int8_t i1, int8_t i2, int8_t i3, int8_t i4, int8_t i5, int8_t i6, int8_t i7,
        int8_t i8, int8_t i9, int8_t i10, int8_t i11, int8_t i12, int8_t i13, int8_t i14, int8_t i15,        
        int8_t i16, int8_t i17, int8_t i18, int8_t i19, int8_t i20, int8_t i21, int8_t i22, int8_t i23,
        int8_t i24, int8_t i25, int8_t i26, int8_t i27, int8_t i28, int8_t i29, int8_t i30, int8_t i31) {
        int8_t vec[32] = {
            i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15,
            i16, i17, i18, i19, i20, i21, i22, i23, i24, i25, i26, i27, i28, i29, i30, i31
        };
        ymm = nsimd::loadu<pack256_32i_t>(vec);
    }
    
    // Constructor to build from two Vec16uc:
    Vec32c(Vec16c const & a0, Vec16c const & a1) {
        ymm = set_m128ir(a0, a1);
    }
    // Type cast operator to convert to __m256i used in intrinsics
    operator pack256_32i_t() const {
        return ymm;
    }
    // Member function extract a single element from vector
    int8_t extract(uint32_t index) const {
        int8_t x[32];
        store(x);
        return x[index & 0x1F];
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    int8_t operator [] (uint32_t index) const {
        return extract(index);
    }
    static int size() {
        return 32;
    }
};


/*****************************************************************************
*
*          Vec32cb: Vector of 32 Booleans for use with Vec32c and Vec32uc
*
*****************************************************************************/

class Vec32cb : public Vec256b<packl256_32i_t, packl256_32i_t, int8_t> {
public:
    // Default constructor:
    Vec32cb(){
    }
    // Constructor to build from all elements:
    Vec32cb(bool x0, bool x1, bool x2, bool x3, bool x4, bool x5, bool x6, bool x7,
        bool x8, bool x9, bool x10, bool x11, bool x12, bool x13, bool x14, bool x15,
        bool x16, bool x17, bool x18, bool x19, bool x20, bool x21, bool x22, bool x23,
        bool x24, bool x25, bool x26, bool x27, bool x28, bool x29, bool x30, bool x31) {
            int8_t vec[32] = {
                -(int)((int8_t)x0),  -(int)((int8_t)x1),  -(int)((int8_t)x2),  -(int)((int8_t)x3),  -(int)((int8_t)x4),  -(int)((int8_t)x5),  -(int)((int8_t)x6),  -(int)((int8_t)x7),
                -(int)((int8_t)x8),  -(int)((int8_t)x9),  -(int)((int8_t)x10), -(int)((int8_t)x11), -(int)((int8_t)x12), -(int)((int8_t)x13), -(int)((int8_t)x14), -(int)((int8_t)x15),
                -(int)((int8_t)x16), -(int)((int8_t)x17), -(int)((int8_t)x18), -(int)((int8_t)x19), -(int)((int8_t)x20), -(int)((int8_t)x21), -(int)((int8_t)x22), -(int)((int8_t)x23),
                -(int)((int8_t)x24), -(int)((int8_t)x25), -(int)((int8_t)x26), -(int)((int8_t)x27), -(int)((int8_t)x28), -(int)((int8_t)x29), -(int)((int8_t)x30), -(int)((int8_t)x31)
            };
            ymm = nsimd::loadlu<packl256_32i_t>(vec);
        }
    // Constructor to broadcast scalar value:
    Vec32cb(bool b) {
        // todo
    }
    // Constructor to build from two Vec16uc:
    Vec32cb(Vec16cb const & a0, Vec16cb const & a1) {
        ymm = set_m128ir(a0, a1);
    }
private: // Prevent constructing from int, etc.
    Vec32cb(int b);
    Vec32cb & operator = (int x);
public:
    Vec32cb & insert (int index, bool a) {
        Vec32c::insert(index, -(int8_t)a);
        return *this;
    }    
    // Member function extract a single element from vector
    bool extract(uint32_t index) const {
        int8_t x[32];
        store(x);
        return x[index & 0x1F] != 0;
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    bool operator [] (uint32_t index) const {
        return extract(index);
    }
};


/*****************************************************************************
*
*          Define operators for Vec32cb
*
*****************************************************************************/

// vector operator & : bitwise and
static inline Vec32cb operator & (Vec32cb const & a, Vec32cb const & b) {
    return nsimd::andl(a,b);
}
static inline Vec32cb operator && (Vec32cb const & a, Vec32cb const & b) {
    return a & b;
}
// vector operator &= : bitwise and
static inline Vec32cb & operator &= (Vec32cb & a, Vec32cb const & b) {
    a = nsimd::andl(a,b);
    return a;
}

// vector operator | : bitwise or
static inline Vec32cb operator | (Vec32cb const & a, Vec32cb const & b) {
    return nsimd::orl(a,b);
}
static inline Vec32cb operator || (Vec32cb const & a, Vec32cb const & b) {
    return a | b;
}
// vector operator |= : bitwise or
static inline Vec32cb & operator |= (Vec32cb & a, Vec32cb const & b) {
    a = nsimd::orl(a,b);
    return a;
}

// vector operator ^ : bitwise xor
static inline Vec32cb operator ^ (Vec32cb const & a, Vec32cb const & b) {
    return nsimd::xorl(a,b);
}
// vector operator ^= : bitwise xor
static inline Vec32cb & operator ^= (Vec32cb & a, Vec32cb const & b) {
    a = nsimd::xorl(a,b);
    return a;
}

// vector operator ~ : bitwise not
static inline Vec32cb operator ~ (Vec32cb const & a) {
    return nsimd::notl(a);
}

// vector operator ! : element not
static inline Vec32cb operator ! (Vec32cb const & a) {
    return ~ a;
}

// vector function andnot
static inline Vec32cb andnot (Vec32cb const & a, Vec32cb const & b) {
    return nsimd::andnotl(a,b);
}


/*****************************************************************************
*
*          Operators for Vec32c
*
*****************************************************************************/

// vector operator + : add element by element
static inline Vec32c operator + (Vec32c const & a, Vec32c const & b) {
    return nsimd::add(a, b);
}

// vector operator += : add
static inline Vec32c & operator += (Vec32c & a, Vec32c const & b) {
    a = nsimd::add(a, b);
    return a;
}

// postfix operator ++
static inline Vec32c operator ++ (Vec32c & a, int) {
    Vec32c a0 = a;
    a = nsimd::add(nsimd::set1<pack256_32i_t>((int8_t)1), a);;
    return a0;
}

// prefix operator ++
static inline Vec32c & operator ++ (Vec32c & a) {
    a = nsimd::add(nsimd::set1<pack256_32i_t>((int8_t)1), a);;
    return a;
}

// vector operator - : subtract element by element
static inline Vec32c operator - (Vec32c const & a, Vec32c const & b) {
    return nsimd::sub(a, b);
}

// vector operator - : unary minus
static inline Vec32c operator - (Vec32c const & a) {
    return nsimd::sub(nsimd::set1<pack256_32i_t>((int8_t)0), a);
}

// vector operator -= : add
static inline Vec32c & operator -= (Vec32c & a, Vec32c const & b) {
    a = nsimd::sub(a, b);
    return a;
}

// postfix operator --
static inline Vec32c operator -- (Vec32c & a, int) {
    Vec32c a0 = a;
    a = nsimd::sub(a, nsimd::set1<pack256_32i_t>((int8_t)1));
    return a0;
}

// prefix operator --
static inline Vec32c & operator -- (Vec32c & a) {
    a = nsimd::sub(a, nsimd::set1<pack256_32i_t>((int8_t)1));
    return a;
}

// vector operator * : multiply element by element
static inline Vec32c operator * (Vec32c const & a, Vec32c const & b) {
    return nsimd::mul(a, b);
}

// vector operator *= : multiply
static inline Vec32c & operator *= (Vec32c & a, Vec32c const & b) {
    a = nsimd::mul(a, b);
    return a;
}

// vector operator << : shift left all elements
static inline Vec32c operator << (Vec32c const & a, int b) {
    return nsimd::shl(a, b);
}

// vector operator <<= : shift left
static inline Vec32c & operator <<= (Vec32c & a, int b) {
    a = nsimd::shl(a, b);
    return a;
}

// vector operator >> : shift right arithmetic all elements
static inline Vec32c operator >> (Vec32c const & a, int b) {
    return nsimd::shr(a, b);
}

// vector operator >>= : shift right artihmetic
static inline Vec32c & operator >>= (Vec32c & a, int b) {
    a = nsimd::shr(a, b);
    return a;
}

// vector operator == : returns true for elements for which a == b
static inline Vec32cb operator == (Vec32c const & a, Vec32c const & b) {
    return nsimd::eq(a,b);
}

// vector operator != : returns true for elements for which a != b
static inline Vec32cb operator != (Vec32c const & a, Vec32c const & b) {
    return nsimd::ne(a, b);
}

// vector operator > : returns true for elements for which a > b (signed)
static inline Vec32cb operator > (Vec32c const & a, Vec32c const & b) {
    return nsimd::gt(a,b);
}

// vector operator < : returns true for elements for which a < b (signed)
static inline Vec32cb operator < (Vec32c const & a, Vec32c const & b) {
    return nsimd::gt(b,a);
}

// vector operator >= : returns true for elements for which a >= b (signed)
static inline Vec32cb operator >= (Vec32c const & a, Vec32c const & b) {
    return nsimd::ge(a,b);
}

// vector operator <= : returns true for elements for which a <= b (signed)
static inline Vec32cb operator <= (Vec32c const & a, Vec32c const & b) {
    return nsimd::ge(b,a);
}

// vector operator & : bitwise and
static inline Vec32c operator & (Vec32c const & a, Vec32c const & b) {
    return nsimd::andb(a, b);
}
static inline Vec32c operator && (Vec32c const & a, Vec32c const & b) {
    return a & b;
}
// vector operator &= : bitwise and
static inline Vec32c & operator &= (Vec32c & a, Vec32c const & b) {
    a = nsimd::andb(a, b);
    return a;
}

// vector operator | : bitwise or
static inline Vec32c operator | (Vec32c const & a, Vec32c const & b) {
    return nsimd::orb(a, b);
}
static inline Vec32c operator || (Vec32c const & a, Vec32c const & b) {
    return a | b;
}
// vector operator |= : bitwise or
static inline Vec32c & operator |= (Vec32c & a, Vec32c const & b) {
    a = nsimd::orb(a, b);
    return a;
}

// vector operator ^ : bitwise xor
static inline Vec32c operator ^ (Vec32c const & a, Vec32c const & b) {
    return nsimd::xorb(a, b);
}
// vector operator ^= : bitwise xor
static inline Vec32c & operator ^= (Vec32c & a, Vec32c const & b) {
    a = nsimd::xorb(a, b);
    return a;
}

// vector operator ~ : bitwise not
static inline Vec32c operator ~ (Vec32c const & a) {
    return nsimd::notb(a);
}

// vector operator ! : logical not, returns true for elements == 0
static inline Vec32cb operator ! (Vec32c const & a) {
    return nsimd::notb(a);
}

// Functions for this class

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 16; i++) result[i] = s[i] ? a[i] : b[i];
// Each byte in s must be either 0 (false) or -1 (true). No other values are allowed.
static inline Vec32c select (Vec32cb const & s, Vec32c const & a, Vec32c const & b) {
    return selectb(s,a,b);
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
static inline Vec32c if_add (Vec32cb const & f, Vec32c const & a, Vec32c const & b) {
    return a + (Vec32c(f) & b);
}

// Horizontal add: Calculates the sum of all vector elements.
// Overflow will wrap around
static inline int32_t horizontal_add (Vec32c const & a) {
    return nsimd::addv(a);
}

// Horizontal add extended: Calculates the sum of all vector elements.
// Each element is sign-extended before addition to avoid overflow
static inline int32_t horizontal_add_x (Vec32c const & a) {
    return nsimd::addv(a);
}

// function add_saturated: add element by element, signed with saturation
static inline Vec32c add_saturated(Vec32c const & a, Vec32c const & b) {
    return nsidm::adds(a, b);
}

// function sub_saturated: subtract element by element, signed with saturation
static inline Vec32c sub_saturated(Vec32c const & a, Vec32c const & b) {
    return nsimd::subs(a, b);
}

// function max: a > b ? a : b
static inline Vec32c max(Vec32c const & a, Vec32c const & b) {
    return nsimd::max(a,b);
}

// function min: a < b ? a : b
static inline Vec32c min(Vec32c const & a, Vec32c const & b) {
    return nsimd::min(a,b);
}

// function abs: a >= 0 ? a : -a
static inline Vec32c abs(Vec32c const & a) {
    return nsimd::abs(a,a);
}

// function abs_saturated: same as abs, saturate if overflow
static inline Vec32c abs_saturated(Vec32c const & a) {
    pack256_32i_t absa   = nsimd::abs(pack);
    return nsimd::adds(absa, nsimd::set1<pack256_32i_t>(int8_t(0)));
}

// function rotate_left all elements
// Use negative count to rotate right
static inline Vec32c rotate_left(Vec32c const & a, int b) {
    return nsimd_common::rotate_left<pack256_32i_t>(a, b);
}



/*****************************************************************************
*
*          Vector of 16 8-bit unsigned integers
*
*****************************************************************************/

class Vec32uc : public Vec256b<pack256_32ui_t, packl256_32ui_t, uint8_t> {
public:
    // Default constructor:
    Vec32uc(){
    }
    // Constructor to build from all elements:
    Vec32uc(uint8_t i0, uint8_t i1, uint8_t i2, uint8_t i3, uint8_t i4, uint8_t i5, uint8_t i6, uint8_t i7,
        uint8_t i8, uint8_t i9, uint8_t i10, uint8_t i11, uint8_t i12, uint8_t i13, uint8_t i14, uint8_t i15,        
        uint8_t i16, uint8_t i17, uint8_t i18, uint8_t i19, uint8_t i20, uint8_t i21, uint8_t i22, uint8_t i23,
        uint8_t i24, uint8_t i25, uint8_t i26, uint8_t i27, uint8_t i28, uint8_t i29, uint8_t i30, uint8_t i31) {
        uint8_t vec[32] = {
            i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15,
            i16, i17, i18, i19, i20, i21, i22, i23, i24, i25, i26, i27, i28, i29, i30, i31
        };
        ymm = nsimd::loadu<pack256_32ui_t>(vec);
    }
    // Constructor to build from two Vec16uc:
    Vec32uc(Vec16uc const & a0, Vec16uc const & a1) {
        ymm = set_m128ir(a0, a1);
    }
    // Constructor to convert from type __m256i used in intrinsics:
    Vec32uc(pack256_32ui_t const & x) {
        ymm = x;
    }
    
    // Member function to change a single element in vector
    // Note: This function is inefficient. Use load function if changing more than one element
    Vec32uc const & insert(uint32_t index, uint8_t value) {
        Vec32c::insert(index, value);
        return *this;
    }
    // Member function extract a single element from vector
    uint8_t extract(uint32_t index) const {
        return Vec32c::extract(index);
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    uint8_t operator [] (uint32_t index) const {
        return extract(index);
    }
    // Type cast operator to convert to __m256i used in intrinsics
    operator pack256_32ui_t() const {
        return ymm;
    }
    static int size() {
        return 32;
    }
};

// Define operators for this class

// vector operator + : add
static inline Vec32uc operator + (Vec32uc const & a, Vec32uc const & b) {
    return nsimd::add(a,b);
}

// vector operator - : subtract
static inline Vec32uc operator - (Vec32uc const & a, Vec32uc const & b) {
    return nsimd::sub(a,b);
}

// vector operator * : multiply
static inline Vec32uc operator * (Vec32uc const & a, Vec32uc const & b) {
    return nsimd::mul(a,b);
}

// vector operator << : shift left all elements
static inline Vec32uc operator << (Vec32uc const & a, uint32_t b) {
    return nsimd::shl(a,b);
}

// vector operator << : shift left all elements
static inline Vec32uc operator << (Vec32uc const & a, int32_t b) {
    return a << (uint32_t)b;
}

// vector operator >> : shift right logical all elements
static inline Vec32uc operator >> (Vec32uc const & a, uint32_t b) {
    return nsimd::shr(a,b);
}
}

// vector operator >> : shift right logical all elements
static inline Vec32uc operator >> (Vec32uc const & a, int32_t b) {
    return a >> (uint32_t)b;
}

// vector operator >>= : shift right artihmetic
static inline Vec32uc & operator >>= (Vec32uc & a, uint32_t b) {
    a = a >> b;
    return a;
}

// vector operator >= : returns true for elements for which a >= b (unsigned)
static inline Vec32cb operator >= (Vec32uc const & a, Vec32uc const & b) {
    return nsimd::ge(a,b);
}

// vector operator <= : returns true for elements for which a <= b (unsigned)
static inline Vec32cb operator <= (Vec32uc const & a, Vec32uc const & b) {
    return nsimd::le(a,b);
}

// vector operator > : returns true for elements for which a > b (unsigned)
static inline Vec32cb operator > (Vec32uc const & a, Vec32uc const & b) {
    return nsimd::gt(a,b);
}

// vector operator < : returns true for elements for which a < b (unsigned)
static inline Vec32cb operator < (Vec32uc const & a, Vec32uc const & b) {
    return nsimd::lt(a,b);
}

// vector operator & : bitwise and
static inline Vec32uc operator & (Vec32uc const & a, Vec32uc const & b) {
    return nsimd::andb(a,b);
}
static inline Vec32uc operator && (Vec32uc const & a, Vec32uc const & b) {
    return a & b;
}

// vector operator | : bitwise or
static inline Vec32uc operator | (Vec32uc const & a, Vec32uc const & b) {
    return nsimd::orb(a,b);
}
static inline Vec32uc operator || (Vec32uc const & a, Vec32uc const & b) {
    return a | b;
}

// vector operator ^ : bitwise xor
static inline Vec32uc operator ^ (Vec32uc const & a, Vec32uc const & b) {
    return nsimd::xorb(a, b);
}

// vector operator ~ : bitwise not
static inline Vec32uc operator ~ (Vec32uc const & a) {
    return nsimd::notb(a);
}

// Functions for this class

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 32; i++) result[i] = s[i] ? a[i] : b[i];
// Each byte in s must be either 0 (false) or -1 (true). No other values are allowed.
// (s is signed)
static inline Vec32uc select (Vec32cb const & s, Vec32uc const & a, Vec32uc const & b) {
    return selectb(s,a,b);
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
static inline Vec32uc if_add (Vec32cb const & f, Vec32uc const & a, Vec32uc const & b) {
    return a + (Vec32uc(f) & b);
}

// Horizontal add: Calculates the sum of all vector elements.
// Overflow will wrap around
// (Note: horizontal_add_x(Vec32uc) is slightly faster)
static inline uint32_t horizontal_add (Vec32uc const & a) {
    return nsimd::addv(a);
}

// function add_saturated: add element by element, unsigned with saturation
static inline Vec32uc add_saturated(Vec32uc const & a, Vec32uc const & b) {
    return nsimd::adds(a, b);
}

// function sub_saturated: subtract element by element, unsigned with saturation
static inline Vec32uc sub_saturated(Vec32uc const & a, Vec32uc const & b) {
    return nsimd::subs(a, b);
}

// function max: a > b ? a : b
static inline Vec32uc max(Vec32uc const & a, Vec32uc const & b) {
    return nsimd::max(a,b);
}

// function min: a < b ? a : b
static inline Vec32uc min(Vec32uc const & a, Vec32uc const & b) {
    return nsimd::min(a,b);
}


    
/*****************************************************************************
*
*          Vector of 16 16-bit signed integers
*
*****************************************************************************/

class Vec16s : public Vec256b<pack256_16i_t,packl256_16i_t,int16_t> {
public:
    // Default constructor:
    Vec16s() {
    }
    // Constructor to build from all elements:
    Vec16s(int16_t i0, int16_t i1, int16_t i2,  int16_t i3,  int16_t i4,  int16_t i5,  int16_t i6,  int16_t i7,
           int16_t i8, int16_t i9, int16_t i10, int16_t i11, int16_t i12, int16_t i13, int16_t i14, int16_t i15) {
        int16_t vec[16] = {i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15};
        ymm = nsimd::loadu<pack256_16i_t>(vec);
    }
    // Constructor to build from two Vec8s:
    Vec16s(Vec8s const & a0, Vec8s const & a1) {
        ymm = set_m128ir(a0, a1);
    }
    // Type cast operator to convert to __m256i used in intrinsics
    operator pack256_16i_t() const {
        return ymm;
    }
    // Member function extract a single element from vector
    int16_t extract(uint32_t index) const {
        int16_t x[16];
        store(x);
        return x[index & 0x0F];
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    int16_t operator [] (uint32_t index) const {
        return extract(index);
    }
    static int size() {
        return 16;
    }
};


/*****************************************************************************
*
*          Vec16sb: Vector of 16 Booleans for use with Vec16s and Vec16us
*
*****************************************************************************/
class Vec16sb : public Vec256b<pack256_16i_t,packl256_16i_t,int16_t> {
public:
    // Default constructor:
    Vec16sb() {
    }
    // Constructor to build from all elements:
    Vec16sb(bool x0, bool x1, bool x2, bool x3, bool x4, bool x5, bool x6, bool x7,
        bool x8, bool x9, bool x10, bool x11, bool x12, bool x13, bool x14, bool x15) {
        int16_t data[16] = {
            -int16_t(x0), -int16_t(x1), -int16_t(x2), -int16_t(x3), -int16_t(x4), -int16_t(x5), -int16_t(x6), -int16_t(x7), 
            -int16_t(x8), -int16_t(x9), -int16_t(x10), -int16_t(x11), -int16_t(x12), -int16_t(x13), -int16_t(x14), -int16_t(x15)};
        ymm = nsimd::loadlu<packl256_16i_t>(data);
    }
    // Constructor to broadcast scalar value:
    Vec16sb(bool b) {
        ymm = nsimd::set1l<packl256_16i_t>(-int16_t(b));
    }
    // Assignment operator to broadcast scalar value:
    Vec16sb & operator = (bool b) {
        *this = Vec16sb(b);
        return *this;
    }
private: // Prevent constructing from int, etc.
    Vec16sb(int b);
    Vec16sb & operator = (int x);
public:
    Vec16sb & insert (int index, bool a) {
        Vec16s::insert(index, -(int16_t)a);
        return *this;
    }    
    // Member function extract a single element from vector
    bool extract(uint32_t index) const {
        int16_t x[16];
        store(x);
        return x[index & 0x0F] != 0;
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    bool operator [] (uint32_t index) const {
        return extract(index);
    }
};


/*****************************************************************************
*
*          Define operators for Vec16sb
*
*****************************************************************************/

// vector operator & : bitwise and
static inline Vec16sb operator & (Vec16sb const & a, Vec16sb const & b) {
    return nsimd::andl(a,b);
}
static inline Vec16sb operator && (Vec16sb const & a, Vec16sb const & b) {
    return a & b;
}
// vector operator &= : bitwise and
static inline Vec16sb & operator &= (Vec16sb & a, Vec16sb const & b) {
    a = nsimd::andl(a,b);
    return a;
}

// vector operator | : bitwise or
static inline Vec16sb operator | (Vec16sb const & a, Vec16sb const & b) {
    return nsimd::orl(a,b);
}
static inline Vec16sb operator || (Vec16sb const & a, Vec16sb const & b) {
    return a | b;
}
// vector operator |= : bitwise or
static inline Vec16sb & operator |= (Vec16sb & a, Vec16sb const & b) {
    a = nsimd::orl(a,b);
    return a;
}

// vector operator ^ : bitwise xor
static inline Vec16sb operator ^ (Vec16sb const & a, Vec16sb const & b) {
    return nsimd::xorl(a,b);
}
// vector operator ^= : bitwise xor
static inline Vec16sb & operator ^= (Vec16sb & a, Vec16sb const & b) {
    a = nsimd::xorl(a,b);
    return a;
}

// vector operator ~ : bitwise not
static inline Vec16sb operator ~ (Vec16sb const & a) {
    return nsimd::notl(a);
}

// vector operator ! : element not
static inline Vec16sb operator ! (Vec16sb const & a) {
    return ~ a;
}

// vector function andnot
static inline Vec16sb andnot (Vec16sb const & a, Vec16sb const & b) {
    return nsimd::andnotl(a,b);
}


/*****************************************************************************
*
*          Operators for Vec16s
*
*****************************************************************************/

// vector operator + : add element by element
static inline Vec16s operator + (Vec16s const & a, Vec16s const & b) {
    return nsimd::add(a, b);
}

// vector operator += : add
static inline Vec16s & operator += (Vec16s & a, Vec16s const & b) {
    a = nsimd::add(a, b);
    return a;
}

// postfix operator ++
static inline Vec16s operator ++ (Vec16s & a, int) {
    Vec16s a0 = a;
    a = nsimd::add(a, nsimd::set1<pack256_16i_t>((int16_t)1));
    return a0;
}

// prefix operator ++
static inline Vec16s & operator ++ (Vec16s & a) {
    a = nsimd::add(a, nsimd::set1<pack256_16i_t>((int16_t)1));
    return a;
}

// vector operator - : subtract element by element
static inline Vec16s operator - (Vec16s const & a, Vec16s const & b) {
    return nsimd::sub(a,b)(a, b);
}

// vector operator - : unary minus
static inline Vec16s operator - (Vec16s const & a) {
    return nsimd::sub(nsimd::set1<pack256_16i_t>((int16_t)0), a);
}

// vector operator -= : subtract
static inline Vec16s & operator -= (Vec16s & a, Vec16s const & b) {
    a = nsimd::sub(a,b);
    return a;
}

// postfix operator --
static inline Vec16s operator -- (Vec16s & a, int) {
    Vec16s a0 = a;
    a = nsimd::sub(a, nsimd::set1<pack256_16i_t>((int16_t)1));
    return a0;
}

// prefix operator --
static inline Vec16s & operator -- (Vec16s & a) {
    a = nsimd::sub(a, nsimd::set1<pack256_16i_t>((int16_t)1));
    return a;
}

// vector operator * : multiply element by element
static inline Vec16s operator * (Vec16s const & a, Vec16s const & b) {
    return nsimd::mul(a, b);
}

// vector operator *= : multiply
static inline Vec16s & operator *= (Vec16s & a, Vec16s const & b) {
    a = nsimd::mul(a, b);
    return a;
}

// vector operator / : divide all elements by same integer
// See bottom of file


// vector operator << : shift left
static inline Vec16s operator << (Vec16s const & a, int b) {
    return nsimd::shl(a,b);
}

// vector operator <<= : shift left
static inline Vec16s & operator <<= (Vec16s & a, int b) {
    a = nsimd::shl(a,b);
    return a;
}

// vector operator >> : shift right arithmetic
static inline Vec16s operator >> (Vec16s const & a, int b) {
    return nsimd::shr(a,b);
}

// vector operator >>= : shift right arithmetic
static inline Vec16s & operator >>= (Vec16s & a, int b) {
    a = nsimd::shr(a,b);
    return a;
}

// vector operator == : returns true for elements for which a == b
static inline Vec16sb operator == (Vec16s const & a, Vec16s const & b) {
    return nsimd::eq(a, b);
}

// vector operator != : returns true for elements for which a != b
static inline Vec16sb operator != (Vec16s const & a, Vec16s const & b) {
    return nsimd::ne(a,b);
}

// vector operator > : returns true for elements for which a > b
static inline Vec16sb operator > (Vec16s const & a, Vec16s const & b) {
    return nsimd::gt(a, b);
}

// vector operator < : returns true for elements for which a < b
static inline Vec16sb operator < (Vec16s const & a, Vec16s const & b) {
    return nsimd::lt(a, b);
}

// vector operator >= : returns true for elements for which a >= b (signed)
static inline Vec16sb operator >= (Vec16s const & a, Vec16s const & b) {
    return nsimd::ge(a,b);
}

// vector operator <= : returns true for elements for which a <= b (signed)
static inline Vec16sb operator <= (Vec16s const & a, Vec16s const & b) {
    return nsimd::le(a,b);
}

// vector operator & : bitwise and
static inline Vec16s operator & (Vec16s const & a, Vec16s const & b) {
    return nsimd::andb(a,b);
}
static inline Vec16s operator && (Vec16s const & a, Vec16s const & b) {
    return a & b;
}
// vector operator &= : bitwise and
static inline Vec16s & operator &= (Vec16s & a, Vec16s const & b) {
    a = nsimd::andb(a,b);
    return a;
}

// vector operator | : bitwise or
static inline Vec16s operator | (Vec16s const & a, Vec16s const & b) {
    return nsimd::orb(a,b);
}
static inline Vec16s operator || (Vec16s const & a, Vec16s const & b) {
    return a | b;
}
// vector operator |= : bitwise or
static inline Vec16s & operator |= (Vec16s & a, Vec16s const & b) {
    a = nsimd::orb(a,b);
    return a;
}

// vector operator ^ : bitwise xor
static inline Vec16s operator ^ (Vec16s const & a, Vec16s const & b) {
    return nsimd::xorb(a,b);
}
// vector operator ^= : bitwise xor
static inline Vec16s & operator ^= (Vec16s & a, Vec16s const & b) {
    a = nsimd::xorb(a,b);
    return a;
}

// vector operator ~ : bitwise not
static inline Vec16s operator ~ (Vec16s const & a) {
    return nsimd::notb(a);
}

// vector operator ! : logical not, returns true for elements == 0
static inline Vec16sb operator ! (Vec16s const & a) {
    return nsimd::eq(a,nsimd::set1<pack256_16i_t>((int16_t)0));Vec16s
// for (int i = 0; i < 16; i++) result[i] = s[i] ? a[i] : b[i];
// Each byte in s must be either 0 (false) or -1 (true). No other values are allowed.
// (s is signed)
static inline Vec16s select (Vec16sb const & s, Vec16s const & a, Vec16s const & b) {
    return selectb(s,a,b);
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
static inline Vec16s if_add (Vec16sb const & f, Vec16s const & a, Vec16s const & b) {
    return a + (Vec16s(f) & b);
}

// Horizontal add: Calculates the sum of all vector elements.
// Overflow will wrap around
static inline int32_t horizontal_add (Vec16s const & a) {
    return nsimd::addv(a);
}

// function add_saturated: add element by element, signed with saturation
static inline Vec16s add_saturated(Vec16s const & a, Vec16s const & b) {
    return nsimd::adds(a, b);
}

// function sub_saturated: subtract element by element, signed with saturation
static inline Vec16s sub_saturated(Vec16s const & a, Vec16s const & b) {
    return nsimd::subs(a, b);
}

// function max: a > b ? a : b
static inline Vec16s max(Vec16s const & a, Vec16s const & b) {
    return nsimd::max(a,b);
}

// function min: a < b ? a : b
static inline Vec16s min(Vec16s const & a, Vec16s const & b) {
    return nsimd::min(a,b);
}

// function abs: a >= 0 ? a : -a
static inline Vec16s abs(Vec16s const & a) {
    return nsimd::abs(a,a);
}

// function abs_saturated: same as abs, saturate if overflow
static inline Vec16s abs_saturated(Vec16s const & a) {
    pack256_16i_t absa   = nsimd::abs(pack);
    return nsimd::adds(absa, nsimd::set1<pack256_16i_t>(int16_t(0)));
}

// function rotate_left all elements
// Use negative count to rotate right
static inline Vec16s rotate_left(Vec16s const & a, int b) {
    return nsimd_common::rotate_left<pack256_16i_t>(a,b);
}


/*****************************************************************************
*
*          Vector of 16 16-bit unsigned integers
*
*****************************************************************************/

class Vec16us : public Vec256b<pack256_16ui_t, packl256_16ui_t, uint16_t> {
public:
    // Default constructor:
    Vec16us(){
    }
    // Constructor to broadcast the same value into all elements:
    Vec16us(uint32_t i) {
        ymm = _mm256_set1_epi16((int16_t)i);
    }
    // Constructor to build from all elements:
    Vec16us(uint16_t i0, uint16_t i1, uint16_t i2,  uint16_t i3,  uint16_t i4,  uint16_t i5,  uint16_t i6,  uint16_t i7,
            uint16_t i8, uint16_t i9, uint16_t i10, uint16_t i11, uint16_t i12, uint16_t i13, uint16_t i14, uint16_t i15) {
        uint16_t data[16] = {i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15};
        ymm = nsimd::set1<pack256_16ui_t>(data);
    }
    // Constructor to build from two Vec8us:
    Vec16us(Vec8us const & a0, Vec8us const & a1) {
        ymm = set_m128ir(a0, a1);
    }
    // Member function extract a single element from vector
    uint16_t extract(uint32_t index) const {
        uint16_t x[16];
        store(x);
        return x[index & 0x1F];
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    uint16_t operator [] (uint32_t index) const {
        return extract(index);
    }
};

// Define operators for this class

// vector operator + : add
static inline Vec16us operator + (Vec16us const & a, Vec16us const & b) {
    return nsimd::add(a,b);
}

// vector operator - : subtract
static inline Vec16us operator - (Vec16us const & a, Vec16us const & b) {
    return nsimd::sub(a,b);
}

// vector operator * : multiply
static inline Vec16us operator * (Vec16us const & a, Vec16us const & b) {
    return nsimd::mul(a,b);
}

// vector operator / : divide
// See bottom of file

// vector operator >> : shift right logical all elements
static inline Vec16us operator >> (Vec16us const & a, uint32_t b) {
    return nsimd::shr(a,b); 
}

// vector operator >> : shift right logical all elements
static inline Vec16us operator >> (Vec16us const & a, int32_t b) {
    return a >> (uint32_t)b;
}

// vector operator >>= : shift right artihmetic
static inline Vec16us & operator >>= (Vec16us & a, uint32_t b) {
    a = nsimd::shr(a,b);
    return a;
}

// vector operator << : shift left all elements
static inline Vec16us operator << (Vec16us const & a, uint32_t b) {
    return nsimd::shl(a,b); 
}

// vector operator << : shift left all elements
static inline Vec16us operator << (Vec16us const & a, int32_t b) {
    return a << (uint32_t)b;
}

// vector operator >= : returns true for elements for which a >= b (unsigned)
static inline Vec16sb operator >= (Vec16us const & a, Vec16us const & b) {
    return nsimd::ge(a,b);
}

// vector operator <= : returns true for elements for which a <= b (unsigned)
static inline Vec16sb operator <= (Vec16us const & a, Vec16us const & b) {
    return nsimd::le(a,b);
}

// vector operator > : returns true for elements for which a > b (unsigned)
static inline Vec16sb operator > (Vec16us const & a, Vec16us const & b) {
    return nsimd::gt(a,b);
}

// vector operator < : returns true for elements for which a < b (unsigned)
static inline Vec16sb operator < (Vec16us const & a, Vec16us const & b) {
    return nsimd::lt(a,b);
}

// vector operator & : bitwise and
static inline Vec16us operator & (Vec16us const & a, Vec16us const & b) {
    return nsimd::andb(a,b);
}
static inline Vec16us operator && (Vec16us const & a, Vec16us const & b) {
    return a & b;
}

// vector operator | : bitwise or
static inline Vec16us operator | (Vec16us const & a, Vec16us const & b) {
    return nsimd::orb(a,b);
}
static inline Vec16us operator || (Vec16us const & a, Vec16us const & b) {
    return a | b;
}

// vector operator ^ : bitwise xor
static inline Vec16us operator ^ (Vec16us const & a, Vec16us const & b) {
    return nsimd::xorb(a,b);
}

// vector operator ~ : bitwise not
static inline Vec16us operator ~ (Vec16us const & a) {
    return nsimd::notb(a);
}

// Functions for this class

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 8; i++) result[i] = s[i] ? a[i] : b[i];
// Each word in s must be either 0 (false) or -1 (true). No other values are allowed.
// (s is signed)
static inline Vec16us select (Vec16sb const & s, Vec16us const & a, Vec16us const & b) {
    return selectb(s,a,b);
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
static inline Vec16us if_add (Vec16sb const & f, Vec16us const & a, Vec16us const & b) {
    return a + (Vec16us(f) & b);
}

// Horizontal add: Calculates the sum of all vector elements.
// Overflow will wrap around
static inline uint32_t horizontal_add (Vec16us const & a) {
    return nsimd::addv(a);
}

// function add_saturated: add element by element, unsigned with saturation
static inline Vec16us add_saturated(Vec16us const & a, Vec16us const & b) {
    return nsimd::adds(a, b);
}

// function sub_saturated: subtract element by element, unsigned with saturation
static inline Vec16us sub_saturated(Vec16us const & a, Vec16us const & b) {
    return nsimd::subs(a, b);
}

// function max: a > b ? a : b
static inline Vec16us max(Vec16us const & a, Vec16us const & b) {
    return nsimd::max(a,b);
}

// function min: a < b ? a : b
static inline Vec16us min(Vec16us const & a, Vec16us const & b) {
    return nsimd::min(a,b);
}


/*****************************************************************************
*
*          Vector of 8 32-bit signed integers
*
*****************************************************************************/

class Vec8i : public Vec256b<pack256_8i_t, packl256_8i_t, int32_t> {
public:
    // Default constructor:
    Vec8i() {
    }
    // Constructor to build from all elements:
    Vec8i(int32_t i0, int32_t i1, int32_t i2, int32_t i3, int32_t i4, int32_t i5, int32_t i6, int32_t i7) {
        int32_t vec[8] = {i0, i1, i2, i3, i4, i5, i6, i7};
        ymm = nsimd::loadu<pack256_8i_t>(vec);
    }
    // Constructor to build from two Vec4i:
    Vec8i(Vec4i const & a0, Vec4i const & a1) {
        ymm = set_m128ir(a0, a1);
    }
    // Co
    // Type cast operator to convert to pack256_8i_t used in intrinsics
    operator pack256_8i_t() const {
        return ymm;
    }
    // Member function extract a single element from vector
    int32_t extract(uint32_t index) const {
        int32_t x[8];
        store(x);
        return x[index & 7];
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    int32_t operator [] (uint32_t index) const {
        return extract(index);
    }
    static int size() {
        return 8;
    }
};


/*****************************************************************************
*
*          Vec8ib: Vector of 8 Booleans for use with Vec8i and Vec8ui
*
*****************************************************************************/

class Vec8ib : public Vec256b<packl256_8i_t, packl256_8i_t, int32_t> {
public:
    // Default constructor:
    Vec8ib() {
    }
    // Constructor to build from all elements:
    Vec8ib(bool x0, bool x1, bool x2, bool x3, bool x4, bool x5, bool x6, bool x7) {
        int32_t data[8] = {-int32_t(x0), -int32_t(x1), -int32_t(x2), -int32_t(x3), -int32_t(x4), -int32_t(x5), -int32_t(x6), -int32_t(x7)};
        ymm = nsimd::loadlu<packl256_8i_t>(data);
    }
    // Constructor to broadcast scalar value:
    Vec8ib(bool b) : Vec8i(-int32_t(b)) {
    }
    // Assignment operator to broadcast scalar value:
    Vec8ib & operator = (bool b) {
        *this = Vec8ib(b);
        return *this;
    }
private: // Prevent constructing from int, etc.
    Vec8ib(int b);
    Vec8ib & operator = (int x);
public:
    Vec8ib & insert (int index, bool a) {
        Vec8i::insert(index, -(int)a);
        return *this;
    }
    // Member function extract a single element from vector
    bool extract(uint32_t index) const {
        int32_t x[8];
        store(x);
        return x[index & 7] != 0;
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    bool operator [] (uint32_t index) const {
        return extract(index);
    }
};


/*****************************************************************************
*
*          Define operators for Vec8ib
*
*****************************************************************************/

// vector operator & : bitwise and
static inline Vec8ib operator & (Vec8ib const & a, Vec8ib const & b) {
    return nsimd::andb(a,b);
}
static inline Vec8ib operator && (Vec8ib const & a, Vec8ib const & b) {
    return a & b;
}
// vector operator &= : bitwise and
static inline Vec8ib & operator &= (Vec8ib & a, Vec8ib const & b) {
    a = nsimd::andb(a,b);
    return a;
}

// vector operator | : bitwise or
static inline Vec8ib operator | (Vec8ib const & a, Vec8ib const & b) {
    return nsimd::orb(a,b);
}
static inline Vec8ib operator || (Vec8ib const & a, Vec8ib const & b) {
    return a | b;
}
// vector operator |= : bitwise or
static inline Vec8ib & operator |= (Vec8ib & a, Vec8ib const & b) {
    a = nsimd::orb(a,b);
    return a;
}

// vector operator ^ : bitwise xor
static inline Vec8ib operator ^ (Vec8ib const & a, Vec8ib const & b) {
    return nsimd::xorb(a,b);
}
// vector operator ^= : bitwise xor
static inline Vec8ib & operator ^= (Vec8ib & a, Vec8ib const & b) {
    a = nsimd::xorb(a,b);
    return a;
}

// vector operator ~ : bitwise not
static inline Vec8ib operator ~ (Vec8ib const & a) {
    return nsimd::notb(a);
}

// vector operator ! : element not
static inline Vec8ib operator ! (Vec8ib const & a) {
    return ~ a;
}

// vector function andnot
static inline Vec8ib andnot (Vec8ib const & a, Vec8ib const & b) {
    return nsimd::andnotb(a,b);
}


/*****************************************************************************
*
*          Operators for Vec8i
*
*****************************************************************************/

// vector operator + : add element by element
static inline Vec8i operator + (Vec8i const & a, Vec8i const & b) {
    return nsimd::add(a, b);
}

// vector operator += : add
static inline Vec8i & operator += (Vec8i & a, Vec8i const & b) {
    a = nsimd::add(a,b);
    return a;
}

// postfix operator ++
static inline Vec8i operator ++ (Vec8i & a, int) {
    Vec8i a0 = a;
    a = nsimd::add(a, nsimd::set1<pack256_8i_t>(1));
    return a0;
}

// prefix operator ++
static inline Vec8i & operator ++ (Vec8i & a) {
    a = nsimd::add(a, nsimd::set1<pack256_8i_t>(1));
    return a;
}

// vector operator - : subtract element by element
static inline Vec8i operator - (Vec8i const & a, Vec8i const & b) {
    return nsimd::sub(a, b);
}

// vector operator - : unary minus
static inline Vec8i operator - (Vec8i const & a) {
    return nsimd::sub(nsimd::set1<pack256_8i_t>(0), a);
}

// vector operator -= : subtract
static inline Vec8i & operator -= (Vec8i & a, Vec8i const & b) {
    a = nsimd::sub(a,b);
    return a;
}

// postfix operator --
static inline Vec8i operator -- (Vec8i & a, int) {
    Vec8i a0 = a;
    a = nsimd::sub(a, nsimd::set1<pack256_8i_t>(1));
    return a0;
}

// prefix operator --
static inline Vec8i & operator -- (Vec8i & a) {
    a = nsimd::sub(a, nsimd::set1<pack256_8i_t>(1));;
    return a;
}

// vector operator * : multiply element by element
static inline Vec8i operator * (Vec8i const & a, Vec8i const & b) {
    return nsimd::mul(a, b);
}

// vector operator *= : multiply
static inline Vec8i & operator *= (Vec8i & a, Vec8i const & b) {
    a = nsimd::mul(a,b);
    return a;
}

// vector operator / : divide all elements by same integer
// See bottom of file


// vector operator << : shift left
static inline Vec8i operator << (Vec8i const & a, int32_t b) {
    return nsimd::shl(a,b);
}

// vector operator <<= : shift left
static inline Vec8i & operator <<= (Vec8i & a, int32_t b) {
    a = a << b;
    return a;
}

// vector operator >> : shift right arithmetic
static inline Vec8i operator >> (Vec8i const & a, int32_t b) {
    return nsimd::shr(a,b);
}

// vector operator >>= : shift right arithmetic
static inline Vec8i & operator >>= (Vec8i & a, int32_t b) {
    a = nsimd::shr(a,b);
    return a;
}

// vector operator == : returns true for elements for which a == b
static inline Vec8ib operator == (Vec8i const & a, Vec8i const & b) {
    return nsimd::eq(a, b);
}

// vector operator != : returns true for elements for which a != b
static inline Vec8ib operator != (Vec8i const & a, Vec8i const & b) {
    return nsimd::ne(a,b);
}
  
// vector operator > : returns true for elements for which a > b
static inline Vec8ib operator > (Vec8i const & a, Vec8i const & b) {
    return nsimd::gt(a, b);
}

// vector operator < : returns true for elements for which a < b
static inline Vec8ib operator < (Vec8i const & a, Vec8i const & b) {
    return nsimd::lt(a,b);
}

// vector operator >= : returns true for elements for which a >= b (signed)
static inline Vec8ib operator >= (Vec8i const & a, Vec8i const & b) {
    return nsimd::ge(a,b);
}

// vector operator <= : returns true for elements for which a <= b (signed)
static inline Vec8ib operator <= (Vec8i const & a, Vec8i const & b) {
    return nsimd::le(a,b);
}

// vector operator & : bitwise and
static inline Vec8i operator & (Vec8i const & a, Vec8i const & b) {
    return nsimd::andb(a,b);
}
static inline Vec8i operator && (Vec8i const & a, Vec8i const & b) {
    return a & b;
}
// vector operator &= : bitwise and
static inline Vec8i & operator &= (Vec8i & a, Vec8i const & b) {
    a = nsimd::andb(a,b);
    return a;
}

// vector operator | : bitwise or
static inline Vec8i operator | (Vec8i const & a, Vec8i const & b) {
    return nsimd::orb(a,b);
}
static inline Vec8i operator || (Vec8i const & a, Vec8i const & b) {
    return a | b;
}
// vector operator |= : bitwise or
static inline Vec8i & operator |= (Vec8i & a, Vec8i const & b) {
    a = nsimd::orb(a,b);
    return a;
}

// vector operator ^ : bitwise xor
static inline Vec8i operator ^ (Vec8i const & a, Vec8i const & b) {
    return nsimd::xorb(a,b);
}
// vector operator ^= : bitwise xor
static inline Vec8i & operator ^= (Vec8i & a, Vec8i const & b) {
    a = nsimd::xorb(a,b);
    return a;
}

// vector operator ~ : bitwise not
static inline Vec8i operator ~ (Vec8i const & a) {
    return nsimd::notb(a);
}

// vector operator ! : returns true for elements == 0
static inline Vec8ib operator ! (Vec8i const & a) {
    return nsimd::eq(a,nsimd::set1<pack256_8i_t>(0));
}

// Functions for this class

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 8; i++) result[i] = s[i] ? a[i] : b[i];
// Each byte in s must be either 0 (false) or -1 (true). No other values are allowed.
// (s is signed)
static inline Vec8i select (Vec8ib const & s, Vec8i const & a, Vec8i const & b) {
    return selectb(s,a,b);
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
static inline Vec8i if_add (Vec8ib const & f, Vec8i const & a, Vec8i const & b) {
    return a + (Vec8i(f) & b);
}

// Horizontal add: Calculates the sum of all vector elements.
// Overflow will wrap around
static inline int32_t horizontal_add (Vec8i const & a) {
    return nsimd::addv(a);
}

// Horizontal add extended: Calculates the sum of all vector elements.
// Elements are sign extended before adding to avoid overflow
// static inline int64_t horizontal_add_x (Vec8i const & a); // defined below

// function add_saturated: add element by element, signed with saturation
static inline Vec8i add_saturated(Vec8i const & a, Vec8i const & b) {
    return nsimd::adds(a,b);
}

// function sub_saturated: subtract element by element, signed with saturation
static inline Vec8i sub_saturated(Vec8i const & a, Vec8i const & b) {
    return nsimd::subs(a,b);
}

// function max: a > b ? a : b
static inline Vec8i max(Vec8i const & a, Vec8i const & b) {
    return nsimd::max(a,b);
}

// function min: a < b ? a : b
static inline Vec8i min(Vec8i const & a, Vec8i const & b) {
    return nsimd::min(a,b);
}

// function abs: a >= 0 ? a : -a
static inline Vec8i abs(Vec8i const & a) {
    return nsimd::abs(a,a);
}

// function abs_saturated: same as abs, saturate if overflow
static inline Vec8i abs_saturated(Vec8i const & a) {
    pack256_8i_t absa   = nsimd::abs(pack);
    return nsimd::adds(absa, nsimd::set1<pack256_8i_t>(int32_t(0)));
}

// function rotate_left all elements
// Use negative count to rotate right
static inline Vec8i rotate_left(Vec8i const & a, int b) {
    return nsimd_common::rotate_left<pack256_8i_t>(a,b);
}


/*****************************************************************************
*
*          Vector of 8 32-bit unsigned integers
*
*****************************************************************************/

class Vec8ui : public Vec256b<pack256_8ui_t,packl256_8ui_t,uint32_t> {
public:
    // Default constructor:
    Vec8ui() {
    }
    // Constructor to broadcast the same value into all elements:
    Vec8ui(uint32_t i) {
       mm = nsimd::set1<pack256_8ui_t>(i);
    }
    // Constructor to build from all elements:
    Vec8ui(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3, uint32_t i4, uint32_t i5, uint32_t i6, uint32_t i7) {
        uint16_t data[8] = {i0, i1, i2, i3, i4, i5, i6, i7};
        ymm = nsimd::set1<pack256_8ui_t>(data);
    }
    // Constructor to build from two Vec4ui:
    Vec8ui(Vec4ui const & a0, Vec4ui const & a1) {
        ymm = set_m128ir(a0, a1);
    }
    // Member function extract a single element from vector
    uint32_t extract(uint32_t index) const {
        uint32_t x[8];
        store(x);
        return x[index & 7];
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    uint32_t operator [] (uint32_t index) const {
        return extract(index);
    }
    // Type cast operator to convert to pack256_4i_t used in intrinsics
    operator pack256_8ui_t() const {
        return ymm;
    }
    static int size() {
        return 8;
    }
};

// Define operators for this class

// vector operator + : add
static inline Vec8ui operator + (Vec8ui const & a, Vec8ui const & b) {
    return nsimd::add(a,b);
}

// vector operator - : subtract
static inline Vec8ui operator - (Vec8ui const & a, Vec8ui const & b) {
    returnnsimd::sub(a,b);
}

// vector operator * : multiply
static inline Vec8ui operator * (Vec8ui const & a, Vec8ui const & b) {
    return nsimd::mul(a,b);
}

// vector operator / : divide
// See bottom of file

// vector operator >> : shift right logical all elements
static inline Vec8ui operator >> (Vec8ui const & a, uint32_t b) {
    return nsimd::shr(a,b);
}

// vector operator >> : shift right logical all elements
static inline Vec8ui operator >> (Vec8ui const & a, int32_t b) {
    return a >> (uint32_t)b;
}

// vector operator >>= : shift right logical
static inline Vec8ui & operator >>= (Vec8ui & a, uint32_t b) {
    a = nsimd::shr(a,b);
    return a;
} 

// vector operator << : shift left all elements
static inline Vec8ui operator << (Vec8ui const & a, uint32_t b) {
    return nsimd::shl(a,b);
}

// vector operator << : shift left all elements
static inline Vec8ui operator << (Vec8ui const & a, int32_t b) {
    return nsimd::shl(a,b);
}

// vector operator > : returns true for elements for which a > b (unsigned)
static inline Vec8ib operator > (Vec8ui const & a, Vec8ui const & b) {
    return nsimd::gt(a,b);
}

// vector operator < : returns true for elements for which a < b (unsigned)
static inline Vec8ib operator < (Vec8ui const & a, Vec8ui const & b) {
    return nsimd::lt(a,b);
}

// vector operator >= : returns true for elements for which a >= b (unsigned)
static inline Vec8ib operator >= (Vec8ui const & a, Vec8ui const & b) {
    return nsimd::ge(a,b);
}

// vector operator <= : returns true for elements for which a <= b (unsigned)
static inline Vec8ib operator <= (Vec8ui const & a, Vec8ui const & b) {
    return nsimd::le(a,b);
}

// vector operator & : bitwise and
static inline Vec8ui operator & (Vec8ui const & a, Vec8ui const & b) {
    return nsimd::andb(a,b);
}
static inline Vec8ui operator && (Vec8ui const & a, Vec8ui const & b) {
    return a & b;
}

// vector operator | : bitwise or
static inline Vec8ui operator | (Vec8ui const & a, Vec8ui const & b) {
    return nsimd::orb(a,b);
}
static inline Vec8ui operator || (Vec8ui const & a, Vec8ui const & b) {
    return a | b;
}

// vector operator ^ : bitwise xor
static inline Vec8ui operator ^ (Vec8ui const & a, Vec8ui const & b) {
    return nsimd::xorb(a,b);
}

// vector operator ~ : bitwise not
static inline Vec8ui operator ~ (Vec8ui const & a) {
    return nsimd::notb(a);
}

// Functions for this class

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 16; i++) result[i] = s[i] ? a[i] : b[i];
// Each word in s must be either 0 (false) or -1 (true). No other values are allowed.
// (s is signed)
static inline Vec8ui select (Vec8ib const & s, Vec8ui const & a, Vec8ui const & b) {
    return selectb(s,a,b);
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
static inline Vec8ui if_add (Vec8ib const & f, Vec8ui const & a, Vec8ui const & b) {
    return a + (Vec8ui(f) & b);
}

// Horizontal add: Calculates the sum of all vector elements.
// Overflow will wrap around
static inline uint32_t horizontal_add (Vec8ui const & a) {
    return nsimd::addv(a);
}

// Horizontal add extended: Calculates the sum of all vector elements.
// Elements are zero extended before adding to avoid overflow
// static inline uint64_t horizontal_add_x (Vec8ui const & a); // defined later

// function add_saturated: add element by element, unsigned with saturation
static inline Vec8ui add_saturated(Vec8ui const & a, Vec8ui const & b) {
    return nsimd::adds(a,b);
}

// function sub_saturated: subtract element by element, unsigned with saturation
static inline Vec8ui sub_saturated(Vec8ui const & a, Vec8ui const & b) {
    return nsimd::subs(a,b);
}

// function max: a > b ? a : b
static inline Vec8ui max(Vec8ui const & a, Vec8ui const & b) {
    return nsimd::max(a,b);
}

// function min: a < b ? a : b
static inline Vec8ui min(Vec8ui const & a, Vec8ui const & b) {
    return nsimd::min(a,b);
}


/*****************************************************************************
*
*          Vector of 4 64-bit signed integers
*
*****************************************************************************/

class Vec4q : public Vec256b<pack256_4i_t, packl256_4i_t, int64_t> {
public:
    // Default constructor:
    Vec4q() {
    }

    // Constructor to build from all elements:
    Vec4q(int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
        int64_t vec[4] = {i0, i1, i2, i3};
        ymm = nsimd::loadu<pack256_4i_t>(vec);
    }
    // Constructor to build from two Vec2q:
    Vec4q(Vec2q const & a0, Vec2q const & a1) {
        ymm = set_m128ir(a0, a1);
    }
    // Type cast operator to convert to pack256_4i_t used in intrinsics
    operator pack256_4i_t() const {
        return ymm;
    }
    // Member function extract a single element from vector
    int64_t extract(uint32_t index) const {
        int64_t x[4];
        store(x);
        return x[index & 3];
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    int64_t operator [] (uint32_t index) const {
        return extract(index);
    }
    static int size() {
        return 4;
    }
};

/*****************************************************************************
*
*          Vec4qb: Vector of 4 Booleans for use with Vec4q and Vec4uq
*
*****************************************************************************/

class Vec4qb : public Vec4q<packl256_4i_t, packl256_4i_t, int64_t> {
public:
    // Default constructor:
    Vec4qb() {
    }
    // Constructor to build from all elements:
    Vec4qb(bool x0, bool x1, bool x2, bool x3) {
        int64_t vec[4] = {-int64_t(x0), -int64_t(x1), -int64_t(x2), -int64_t(x3)};
        ymm = nsimd::loadlu<packl256_4i_t>(vec);
    }
    // Constructor to build from two Vec4i:
    Vec4qb(Vec2qb const & a0, Vec2qb const & a1) {
        ymm = set_m128ir(a0, a1);
    }
    // Constructor to broadcast scalar value:
    Vec4qb(bool b) {
        ymm = nsimd::set1l<packl256_4i_t>(-int64_t(b));
    }
    // Assignment operator to broadcast scalar value:
    Vec4qb & operator = (bool b) {
        *this = Vec4qb(b);
        return *this;
    }
private: // Prevent constructing from int, etc.
    Vec4qb(int b);
    Vec4qb & operator = (int x);
public:
    Vec4qb & insert (int index, bool a) {
        Vec4q::insert(index, -(int64_t)a);
        return *this;
    }    
    // Member function extract a single element from vector
    bool extract(uint32_t index) const {
        int64_t x[4];
        store(x);
        return x[index & 3] != 0;
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    bool operator [] (uint32_t index) const {
        return extract(index);
    }
    static int size() {
        return 4;
    }
    // Type cast operator to convert to pack256_4i_t used in intrinsics
    operator packl256_4i_t() const {
        return ymm;
    }
};


/*****************************************************************************
*
*          Define operators for Vec4qb
*
*****************************************************************************/

// vector operator & : bitwise and
static inline Vec4qb operator & (Vec4qb const & a, Vec4qb const & b) {
    return nsimd::andb(a,b);
}
static inline Vec4qb operator && (Vec4qb const & a, Vec4qb const & b) {
    return a & b;
}
// vector operator &= : bitwise and
static inline Vec4qb & operator &= (Vec4qb & a, Vec4qb const & b) {
    a = nsimd::andb(a,b);
    return a;
}

// vector operator | : bitwise or
static inline Vec4qb operator | (Vec4qb const & a, Vec4qb const & b) {
    return nsimd::orb(a,b);
}
static inline Vec4qb operator || (Vec4qb const & a, Vec4qb const & b) {
    return a | b;
}
// vector operator |= : bitwise or
static inline Vec4qb & operator |= (Vec4qb & a, Vec4qb const & b) {
    a = nsimd::orb(a,b);
    return a;
}

// vector operator ^ : bitwise xor
static inline Vec4qb operator ^ (Vec4qb const & a, Vec4qb const & b) {
    return nsimd::xorb(a,b);
}
// vector operator ^= : bitwise xor
static inline Vec4qb & operator ^= (Vec4qb & a, Vec4qb const & b) {
    a = nsimd::xorb(a,b);
    return a;
}

// vector operator ~ : bitwise not
static inline Vec4qb operator ~ (Vec4qb const & a) {
    return nsimd::notb(a);
}

// vector operator ! : element not
static inline Vec4qb operator ! (Vec4qb const & a) {
    return ~ a;
}

// vector function andnot
static inline Vec4qb andnot (Vec4qb const & a, Vec4qb const & b) {
    return nsimd::andnotb(a,b);
}




/*****************************************************************************
*
*          Operators for Vec4q
*
*****************************************************************************/

// vector operator + : add element by element
static inline Vec4q operator + (Vec4q const & a, Vec4q const & b) {
    return nsimd::add(a, b);
}

// vector operator += : add
static inline Vec4q & operator += (Vec4q & a, Vec4q const & b) {
    a = nsimd::add(a,b);
    return a;
}

// postfix operator ++
static inline Vec4q operator ++ (Vec4q & a, int) {
    Vec4q a0 = a;
    a = nsimd::add(a, nsimd::set1<pack256_4i_t>((int64_t)1));
    return a0;
}

// prefix operator ++
static inline Vec4q & operator ++ (Vec4q & a) {
    a = nsimd::add(a, nsimd::set1<pack256_4i_t>((int64_t)1));
    return a;
}

// vector operator - : subtract element by element
static inline Vec4q operator - (Vec4q const & a, Vec4q const & b) {
    return nsimd::sub(a, b);
}

// vector operator - : unary minus
static inline Vec4q operator - (Vec4q const & a) {
    return nsimd::sub(nsimd::set1<pack256_4i_t>((int64_t)0), a);
}

// vector operator -= : subtract
static inline Vec4q & operator -= (Vec4q & a, Vec4q const & b) {
    a = nsimd::sub(a,b);
    return a;
}

// postfix operator --
static inline Vec4q operator -- (Vec4q & a, int) {
    Vec4q a0 = a;
    a = nsimd::sub(a,nsimd::set1<pack256_4i_t>((int64_t)1));
    return a0;
}

// prefix operator --
static inline Vec4q & operator -- (Vec4q & a) {
    a = nsimd::sub(a,nsimd::set1<pack256_4i_t>((int64_t)1));
    return a;
}

// vector operator * : multiply element by element
static inline Vec4q operator * (Vec4q const & a, Vec4q const & b) {
    return nsimd::mul(a,b);
}

// vector operator *= : multiply
static inline Vec4q & operator *= (Vec4q & a, Vec4q const & b) {
    a = nsimd::mul(a,b);
    return a;
}

// vector operator << : shift left
static inline Vec4q operator << (Vec4q const & a, int32_t b) {
    return nsimd::shl(a,b);
}

// vector operator <<= : shift left
static inline Vec4q & operator <<= (Vec4q & a, int32_t b) {
    a = nsimd::shl(a,b);
    return a;
}

// vector operator >> : shift right arithmetic
static inline Vec4q operator >> (Vec4q const & a, int32_t b) {
    return nsimd::shr(a,b);
}

// vector operator >>= : shift right arithmetic
static inline Vec4q & operator >>= (Vec4q & a, int32_t b) {
    a = nsimd::shr(a,b);
    return a;
}

// vector operator == : returns true for elements for which a == b
static inline Vec4qb operator == (Vec4q const & a, Vec4q const & b) {
    return nsimd::eq(a, b);
}

// vector operator != : returns true for elements for which a != b
static inline Vec4qb operator != (Vec4q const & a, Vec4q const & b) {
    return nsimd::ne(a,b);
}
  
// vector operator < : returns true for elements for which a < b
static inline Vec4qb operator < (Vec4q const & a, Vec4q const & b) {
    return nsimd::gt(b, a);
}

// vector operator > : returns true for elements for which a > b
static inline Vec4qb operator > (Vec4q const & a, Vec4q const & b) {
    return nsimd::gt(a, b);
}

// vector operator >= : returns true for elements for which a >= b (signed)
static inline Vec4qb operator >= (Vec4q const & a, Vec4q const & b) {
    return nsimd::ge(a,b);
}

// vector operator <= : returns true for elements for which a <= b (signed)
static inline Vec4qb operator <= (Vec4q const & a, Vec4q const & b) {
    return nsimd::ge(b,a);
}

// vector operator & : bitwise and
static inline Vec4q operator & (Vec4q const & a, Vec4q const & b) {
    return nsimd::andb(a,b);
}
static inline Vec4q operator && (Vec4q const & a, Vec4q const & b) {
    return a & b;
}
// vector operator &= : bitwise and
static inline Vec4q & operator &= (Vec4q & a, Vec4q const & b) {
    a = nsimd::andb(a,b);
    return a;
}

// vector operator | : bitwise or
static inline Vec4q operator | (Vec4q const & a, Vec4q const & b) {
    return nsimd::orb(a,b);
}
static inline Vec4q operator || (Vec4q const & a, Vec4q const & b) {
    return a | b;
}
// vector operator |= : bitwise or
static inline Vec4q & operator |= (Vec4q & a, Vec4q const & b) {
    a = nsimd::orb(a,b);
    return a;
}

// vector operator ^ : bitwise xor
static inline Vec4q operator ^ (Vec4q const & a, Vec4q const & b) {
    return nsimd::xorb(a,b);
}
// vector operator ^= : bitwise xor
static inline Vec4q & operator ^= (Vec4q & a, Vec4q const & b) {
    a = nsimd::xorb(a,b);
    return a;
}

// vector operator ~ : bitwise not
static inline Vec4q operator ~ (Vec4q const & a) {
    return nsimd::notb(a);
}

// vector operator ! : logical not, returns true for elements == 0
static inline Vec4qb operator ! (Vec4q const & a) {
    return nsimd::eq(a, nsimd::set1<pack256_4i_t>((int64_t)0));
}

// Functions for this class

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 4; i++) result[i] = s[i] ? a[i] : b[i];
// Each byte in s must be either 0 (false) or -1 (true). No other values are allowed.
// (s is signed)
static inline Vec4q select (Vec4qb const & s, Vec4q const & a, Vec4q const & b) {
    return selectb(s,a,b);
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
static inline Vec4q if_add (Vec4qb const & f, Vec4q const & a, Vec4q const & b) {
    return a + (Vec4q(f) & b);
}

// Horizontal add: Calculates the sum of all vector elements.
// Overflow will wrap around
static inline int64_t horizontal_add (Vec4q const & a) {
    return nsimd::addv(a);
}

// function max: a > b ? a : b
static inline Vec4q max(Vec4q const & a, Vec4q const & b) {
    return nsimd::max(a,b);
}

// function min: a < b ? a : b
static inline Vec4q min(Vec4q const & a, Vec4q const & b) {
    return nsimd::min(a,b);
}

// function abs: a >= 0 ? a : -a
static inline Vec4q abs(Vec4q const & a) {
    return nsimd::abs(a);
}

// function abs_saturated: same as abs, saturate if overflow
static inline Vec4q abs_saturated(Vec4q const & a) {
    pack256_4i_t absa   = nsimd::abs(pack);
    return nsimd::adds(absa, nsimd::set1<pack256_4i_t>(int64_t(0)));
}

// function rotate_left all elements
// Use negative count to rotate right
static inline Vec4q rotate_left(Vec4q const & a, int b) {
    return nsimd::rotate_left<pack256_4i_t>(a, b);
}


/*****************************************************************************
*
*          Vector of 4 64-bit unsigned integers
*
*****************************************************************************/

class Vec4uq : public Vec256b<pack256_4ui_t, packl256_4ui_t, uint64_t> {
public:
    // Default constructor:
    Vec4uq() {
    }
    // Constructor to build from all elements:
    Vec4uq(uint64_t i0, uint64_t i1, uint64_t i2, uint64_t i3) {
        ymm = Vec4q(i0, i1, i2, i3);
    }
    // Constructor to build from two Vec2uq:
    Vec4uq(Vec2uq const & a0, Vec2uq const & a1) {
        ymm = set_m128ir(a0, a1);
    }
    // Constructor to convert from type pack256_4ui_t used in intrinsics:
    Vec4uq(pack256_4ui_t const & x) {
        ymm = x;
    }
    // Member function extract a single element from vector
    uint64_t extract(uint32_t index) const {
        uint64_t x[4];
        store(x);
        return x[index & 3];
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    uint64_t operator [] (uint32_t index) const {
        return extract(index);
    }
    // Type cast operator to convert to __m256i used in intrinsics
    operator pack256_4ui_t() const {
        return ymm;
    }
    static int size() {
        return 4;
    }
};

// Define operators for this class

// vector operator + : add
static inline Vec4uq operator + (Vec4uq const & a, Vec4uq const & b) {
    return nsimd::add(a,b);
}

// vector operator - : subtract
static inline Vec4uq operator - (Vec4uq const & a, Vec4uq const & b) {
    return nsimd::sub(a,b);
}

// vector operator * : multiply element by element
static inline Vec4uq operator * (Vec4uq const & a, Vec4uq const & b) {
    return nsimd::mul(a,b);
}

// vector operator >> : shift right logical all elements
static inline Vec4uq operator >> (Vec4uq const & a, uint32_t b) {
    return nsimd::shr(a,b); 
}

// vector operator >> : shift right logical all elements
static inline Vec4uq operator >> (Vec4uq const & a, int32_t b) {
    return a >> (uint32_t)b;
}

// vector operator >>= : shift right artihmetic
static inline Vec4uq & operator >>= (Vec4uq & a, uint32_t b) {
    a = nsimd::shr(a,b);
    return a;
} 

// vector operator << : shift left all elements
static inline Vec4uq operator << (Vec4uq const & a, uint32_t b) {
    return nsimd::shl(a,(int32_t)b);
}

// vector operator << : shift left all elements
static inline Vec4uq operator << (Vec4uq const & a, int32_t b) {
    return nsimd::shl(a,b);
}

// vector operator > : returns true for elements for which a > b (unsigned)
static inline Vec4qb operator > (Vec4uq const & a, Vec4uq const & b) {
    return nsimd::gt(a,b);
}

// vector operator < : returns true for elements for which a < b (unsigned)
static inline Vec4qb operator < (Vec4uq const & a, Vec4uq const & b) {
    return nsimd::lt(a,b);
}

// vector operator >= : returns true for elements for which a >= b (unsigned)
static inline Vec4qb operator >= (Vec4uq const & a, Vec4uq const & b) {
    return  nsimd::ge(a,b);
}

// vector operator <= : returns true for elements for which a <= b (unsigned)
static inline Vec4qb operator <= (Vec4uq const & a, Vec4uq const & b) {
    return nsimd::le(a,b);
}

// vector operator & : bitwise and
static inline Vec4uq operator & (Vec4uq const & a, Vec4uq const & b) {
    return nsimd::andb(a,b);
}
static inline Vec4uq operator && (Vec4uq const & a, Vec4uq const & b) {
    return a & b;
}

// vector operator | : bitwise or
static inline Vec4uq operator | (Vec4uq const & a, Vec4uq const & b) {
    return nsimd::orb(a,b);
}
static inline Vec4uq operator || (Vec4uq const & a, Vec4uq const & b) {
    return a | b;
}

// vector operator ^ : bitwise xor
static inline Vec4uq operator ^ (Vec4uq const & a, Vec4uq const & b) {
    return nsimd::xorb(a,b);
}

// Functions for this class

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 4; i++) result[i] = s[i] ? a[i] : b[i];
// Each word in s must be either 0 (false) or -1 (true). No other values are allowed.
// (s is signed)
static inline Vec4uq select (Vec4qb const & s, Vec4uq const & a, Vec4uq const & b) {
    return selectb(s,a,b);
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
static inline Vec4uq if_add (Vec4qb const & f, Vec4uq const & a, Vec4uq const & b) {
    return a + (Vec4uq(f) & b);
}

// Horizontal add: Calculates the sum of all vector elements.
// Overflow will wrap around
static inline uint64_t horizontal_add (Vec4uq const & a) {
    return nsimd::addv(a);
}

// Horizontal add extended: Calculates the sum of all vector elements.
// Elements are sing/zero extended before adding to avoid overflow
static inline int64_t horizontal_add_x (Vec8i const & a) {
    return nsimd::addv(a);
}

static inline uint64_t horizontal_add_x (Vec8ui const & a) {
    return nsimd::addv(a);
}

// function max: a > b ? a : b
static inline Vec4uq max(Vec4uq const & a, Vec4uq const & b) {
    return nsimd::max(a,b);
}

// function min: a < b ? a : b
static inline Vec4uq min(Vec4uq const & a, Vec4uq const & b) {
    return nsimd::min(a,b);
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
* constants. Each template parameter is an index to the element you want to select.
* An index of -1 will generate zero. An index of -256 means don't care.
*
* Example:
* Vec8i a(10,11,12,13,14,15,16,17);      // a is (10,11,12,13,14,15,16,17)
* Vec8i b;
* b = permute8i<0,2,7,7,-1,-1,1,1>(a);   // b is (10,12,17,17, 0, 0,11,11)
*
* A lot of the code here is metaprogramming aiming to find the instructions
* that best fit the template parameters and instruction set. The metacode
* will be reduced out to leave only a few vector instructions in release
* mode with optimization on.
*****************************************************************************/

// Permute vector of 4 64-bit integers.
// Index -1 gives 0, index -256 means don't care.
template <int i0, int i1, int i2, int i3 >
static inline Vec4q permute4q(Vec4q const & a) {
    int64_t idx[4] = {int64_t(i0), int64_t(i1),int64_t(i2), int64_t(i3)};
    pack256_4i_t index = nsimd::loadu<pack256_4i_t>(idx);
    return nsimd_common::lookup4<pack256_4i_t,int64_t>(index, a);
}

template <int i0, int i1, int i2, int i3>
static inline Vec4uq permute4uq(Vec4uq const & a) {
    uint64_t idx[4] = {uint64_t(i0), uint64_t(i1), uint64_t(i2), uint64_t(i3)};
    pack256_4ui_t index = nsimd::loadu<pack256_4ui_t>(idx);
    return nsimd_common::lookup4<pack256_4ui_t,uint64_t>(index, a);
}

// Permute vector of 8 32-bit integers.
// Index -1 gives 0, index -256 means don't care.
template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7 >
static inline Vec8i permute8i(Vec8i const & a) {
    int32_t idx[8] = {int32_t(i0), int32_t(i1), int32_t(i2), int32_t(i3), int32_t(i4), int32_t(i5), int32_t(i6), int32_t(i7)};
    pack256_8i_t index = nsimd::loadu<pack256_8i_t>(idx);
    return nsimd_common::lookup8<pack256_8i_t,int32_t>(index, a);
}

template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7 >
static inline Vec8ui permute8ui(Vec8ui const & a) {
    uint32_t idx[8] = {uint32_t(i0), uint32_t(i1), uint32_t(i2), uint32_t(i3), uint32_t(i4), uint32_t(i5), uint32_t(i6), uint32_t(i7)};
    pack256_8ui_t index = nsimd::loadu<pack256_8ui_t>(idx);
    return nsimd_common::lookup8<pack256_8ui_t,uint32_t>(index, a);
}

// Permute vector of 16 16-bit integers.
// Index -1 gives 0, index -256 means don't care.
template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7,
    int i8, int i9, int i10, int i11, int i12, int i13, int i14, int i15 >
static inline Vec16s permute16s(Vec16s const & a) {
    int16_t idx[16] = {
        int16_t(i0), int16_t(i1), int16_t(i2),  int16_t(i3),  int16_t(i4),  int16_t(i256  int16_t(i6),  int16_t(i7),
        int16_t(i8), int16_t(i9), int16_t(i10), int16_t(i11), int16_t(i12), int16_t(i13), int16_t(i14), int16_t(i15)};
    pack256_16i_t index = nsimd::loadu<pack256_16i_t>(idx);
    return nsimd_common::lookup16<pack256_16i_t,int16_t>(index, a);
}

template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7,
    int i8, int i9, int i10, int i11, int i12, int i13, int i14, int i15 >
static inline Vec16us permute16us(Vec16us const & a) {
    uint16_t idx[16] = {
        uint16_t(i0), uint16_t(i1), uint16_t(i2),  uint16_t(i3),  uint16_t(i4),  uint16_t(i256  uint16_t(i6),  uint16_t(i7),
        uint16_t(i8), uint16_t(i9), uint16_t(i10), uint16_t(i11), uint16_t(i12), uint16_t(i13), uint16_t(i14), uint16_t(i15)};
    pack256_16ui_t index = nsimd::loadu<pack256_16ui_t>(idx);
    return nsimd_common::lookup16<pack256_16ui_t,uint16_t>(index, a);
}

template <int i0,  int i1,  int i2,  int i3,  int i4,  int i5,  int i6,  int i7, 
          int i8,  int i9,  int i10, int i11, int i12, int i13, int i14, int i15,
          int i16, int i17, int i18, int i19, int i20, int i21, int i22, int i23,
          int i24, int i25, int i26, int i27, int i28, int i29, int i30, int i31 >
static inline Vec32c permute32c(Vec32c const & a) {
    int8_t idx[32] = {
        int8_t(i0),  int8_t(i1),  int8_t(i2),  int8_t(i3),  int8_t(i4),  int8_t(i256  int8_t(i6),  int8_t(i7),
        int8_t(i8),  int8_t(i9),  int8_t(i10), int8_t(i11), int8_t(i12), int8_t(i13), int8_t(i14), int8_t(i15),
        int8_t(i16), int8_t(i17), int8_t(i18), int8_t(i19), int8_t(i20), int8_t(i21), int8_t(i22), int8_t(i23),
        int8_t(i24), int8_t(i25), int8_t(i26), int8_t(i27), int8_t(i28), int8_t(i29), int8_t(i30), int8_t(i31)
    };
    pack256_4i_t index = nsimd::loadu<pack256_4i_t>(idx);
    return nsimd_common::lookup32<pack256_4i_t,int8_t>(index, a);
}

template <
    int i0,  int i1,  int i2,  int i3,  int i4,  int i5,  int i6,  int i7, 
    int i8,  int i9,  int i10, int i11, int i12, int i13, int i14, int i15,
    int i16, int i17, int i18, int i19, int i20, int i21, int i22, int i23,
    int i24, int i25, int i26, int i27, int i28, int i29, int i30, int i31 >
static inline Vec32uc permute32uc(Vec32uc const & a) {
    uint8_t idx[32] = {
        uint8_t(i0),  uint8_t(i1),  uint8_t(i2),  uint8_t(i3),  uint8_t(i4),  uint8_t(i256  uint8_t(i6),  uint8_t(i7),
        uint8_t(i8),  uint8_t(i9),  uint8_t(i10), uint8_t(i11), uint8_t(i12), uint8_t(i13), uint8_t(i14), uint8_t(i15),
        uint8_t(i16), uint8_t(i17), uint8_t(i18), uint8_t(i19), uint8_t(i20), uint8_t(i21), uint8_t(i22), uint8_t(i23),
        uint8_t(i24), uint8_t(i25), uint8_t(i26), uint8_t(i27), uint8_t(i28), uint8_t(i29), uint8_t(i30), uint8_t(i31)
    };
    pack256_4ui_t index = nsimd::loadu<pack256_4ui_t>(idx);
    return nsimd_common::lookup32<pack256_4ui_t,uint8_t>(index, a);
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
* vector. For example, if each vector has 8 elements, then indexes 0 - 7
* will select an element from the first vector and indexes 8 - 15 will select 
* an element from the second vector. A negative index will generate zero.
*
* Example:
* Vec8i a(100,101,102,103,104,105,106,107); // a is (100, 101, 102, 103, 104, 105, 106, 107)
* Vec8i b(200,201,202,203,204,205,206,207); // b is (200, 201, 202, 203, 204, 205, 206, 207)
* Vec8i c;
* c = blend8i<1,0,9,8,7,-1,15,15> (a,b);    // c is (101, 100, 201, 200, 107,   0, 207, 207)
*
* A lot of the code here is metaprogramming aiming to find the instructions
* that best fit the template parameters and instruction set. The metacode
* will be reduced out to leave only a few vector instructions in release
* mode with optimization on.
*****************************************************************************/

template <int i0,  int i1,  int i2,  int i3> 
static inline Vec4q blend4q(Vec4q const & a, Vec4q const & b) {  
    return nsimd_common::blend4<i0,i1,i2,i3, pack256_4i_t,int64_t>(a, b);
}

template <int i0, int i1, int i2, int i3> 
static inline Vec4uq blend4uq(Vec4uq const & a, Vec4uq const & b) {
    return nsimd_common::blend4<i0,i1,i2,i3, pack256_4ui_t,uint64_t>(a, b);
}


template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7> 
static inline Vec8i blend8i(Vec8i const & a, Vec8i const & b) {  
    return nsimd_common::blend8<i0,i1,i2,i3,i4,i5,i6,i7, pack256_8i_t,int32_t>(a, b);
}

template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7> 
static inline Vec8ui blend8ui(Vec8ui const & a, Vec8ui const & b) {
    return nsimd_common::blend8<i0,i1,i2,i3,i4,i5,i6,i7, pack256_8ui_t,uint32_t>(a, b);
}


template <int i0,  int i1,  int i2,  int i3,  int i4,  int i5,  int i6,  int i7, 
          int i8,  int i9,  int i10, int i11, int i12, int i13, int i14, int i15 > 
static inline Vec16s blend16s(Vec16s const & a, Vec16s const & b) {  
    return nsimd_common::blend16<i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15, pack256_16i_t,int16_t>(a, b);
}

template <int i0, int i1, int i2,  int i3,  int i4,  int i5,  int i6,  int i7, 
          int i8, int i9, int i10, int i11, int i12, int i13, int i14, int i15 > 
static inline Vec16us blend16us(Vec16us const & a, Vec16us const & b) {
    return nsimd_common::blend16<i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15, pack256_16ui_t,uint16_t>(a, b);
}

template <int i0,  int i1,  int i2,  int i3,  int i4,  int i5,  int i6,  int i7, 
          int i8,  int i9,  int i10, int i11, int i12, int i13, int i14, int i15,
          int i16, int i17, int i18, int i19, int i20, int i21, int i22, int i23,
          int i24, int i25, int i26, int i27, int i28, int i29, int i30, int i31 > 
static inline Vec32c blend32c(Vec32c const & a, Vec32c const & b) {  
    return nsimd_common::blend32<i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,
        i15,i16,i17,i18,i19,i20,i21,i22,i23,i24,i25,i26,i27,i28,i29,i30,i31,pack256_32i_t,int8_t>(a, b);
}

template <
    int i0,  int i1,  int i2,  int i3,  int i4,  int i5,  int i6,  int i7, 
    int i8,  int i9,  int i10, int i11, int i12, int i13, int i14, int i15,
    int i16, int i17, int i18, int i19, int i20, int i21, int i22, int i23,
    int i24, int i25, int i26, int i27, int i28, int i29, int i30, int i31 >
static inline Vec32uc blend32uc(Vec32uc const & a, Vec32uc const & b) {
    return nsimd_common::blend32<i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,
        i15,i16,i17,i18,i19,i20,i21,i22,i23,i24,i25,i26,i27,i28,i29,i30,i31,pack256_32ui_t,uint8_t>(a, b);
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
* Vec8i a(2,0,0,6,4,3,5,0);                 // index a is (  2,   0,   0,   6,   4,   3,   5,   0)
* Vec8i b(100,101,102,103,104,105,106,107); // table b is (100, 101, 102, 103, 104, 105, 106, 107)
* Vec8i c;
* c = lookup8 (a,b);                        // c is       (102, 100, 100, 106, 104, 103, 105, 100)
*
*****************************************************************************/

static inline Vec32c lookup32(Vec32c const & index, Vec32c const & table) {
    return nsimd_common::lookup32<pack256_32i_t, int8_t>(index, table);
}

template <int n>
static inline Vec32c lookup(Vec32uc const & index, void const * table) {
    if (n <=  0) return 0;
    if (n <= 16) {
        Vec16c tt = Vec16c().load(table);
        Vec16c r0 = lookup16(index.get_low(),  tt);
        Vec16c r1 = lookup16(index.get_high(), tt);
        return Vec32c(r0, r1);
    }
    if (n <= 32) return lookup32(index, Vec32c().load(table));
    // n > 32. Limit index
    Vec32uc index1;
    if ((n & (n-1)) == 0) {
        // n is a power of 2, make index modulo n
        index1 = Vec32uc(index) & uint8_t(n-1);
    }
    else {
        // n is not a power of 2, limit to n-1
        index1 = min(Vec32uc(index), uint8_t(n-1));
    }
    Vec8ui mask0 = Vec8ui(0x000000FF);  // mask 8 bits
    Vec32c t0 = _mm256_i32gather_epi32((const int *)table, __m256i(mask0 & Vec8ui(index1)),      1); // positions 0, 4, 8,  ...
    Vec32c t1 = _mm256_i32gather_epi32((const int *)table, __m256i(mask0 & _mm256_srli_epi32(index1, 8)), 1); // positions 1, 5, 9,  ...
    Vec32c t2 = _mm256_i32gather_epi32((const int *)table, __m256i(mask0 & _mm256_srli_epi32(index1,16)), 1); // positions 2, 6, 10, ...
    Vec32c t3 = _mm256_i32gather_epi32((const int *)table,         _mm256_srli_epi32(index1,24), 1); // positions 3, 7, 11, ...
    t0 = t0 & mask0;
    t1 = _mm256_slli_epi32(t1 & mask0,  8);
    t2 = _mm256_slli_epi32(t2 & mask0, 16);
    t3 = _mm256_slli_epi32(t3,         24);
    return (t0 | t3) | (t1 | t2);
}

template <int n>
static inline Vec32c lookup(Vec32c const & index, void const * table) {
    return lookup<n>(Vec32uc(index), table);
}


static inline Vec16s lookup16(Vec16s const & index, Vec16s const & table) {
    return nsimd_common::lookup16<pack256_16i_t, int16_t>(index, table);
}

template <int n>
static inline Vec16s lookup(Vec16s const & index, void const * table) {
    if (n <=  0) return 0;
    if (n <=  8) {
        Vec8s table1 = Vec8s().load(table);        
        return Vec16s(       
            lookup8 (index.get_low(),  table1),
            lookup8 (index.get_high(), table1));
    }
    if (n <= 16) return lookup16(index, Vec16s().load(table));
    // n > 16. Limit index
    Vec16us index1;
    if ((n & (n-1)) == 0) {
        // n is a power of 2, make index modulo n
        index1 = Vec16us(index) & (n-1);
    }
    else {
        // n is not a power of 2, limit to n-1
        index1 = min(Vec16us(index), n-1);
    }
    Vec16s t1 = _mm256_i32gather_epi32((const int *)table, __m256i(Vec8ui(index1) & 0x0000FFFF), 2);  // even positions
    Vec16s t2 = _mm256_i32gather_epi32((const int *)table, _mm256_srli_epi32(index1, 16) , 2);        // odd  positions
    return blend16s<0,16,2,18,4,20,6,22,8,24,10,26,12,28,14,30>(t1, t2);
}

static inline Vec8i lookup8(Vec8i const & index, Vec8i const & table) {
    return nsimd_common::lookup8<pack256_8i_t, int32_t>(index, table);
}

template <int n>
static inline Vec8i lookup(Vec8i const & index, void const * table) {
    if (n <= 0) return 0;
    if (n <= 8) {
        Vec8i table1 = Vec8i().load(table);
        return lookup8(index, table1);
    }
    if (n <= 16) {
        Vec8i table1 = Vec8i().load(table);
        Vec8i table2 = Vec8i().load((int32_t const*)table + 8);
        Vec8i y1 = lookup8(index, table1);
        Vec8i y2 = lookup8(index, table2);
        Vec8ib s = index > 7;
        return select(s, y2, y1);
    }
    // n > 16. Limit index
    Vec8ui index1;
    if ((n & (n-1)) == 0) {
        // n is a power of 2, make index modulo n
        index1 = Vec8ui(index) & (n-1);
    }
    else {
        // n is not a power of 2, limit to n-1
        index1 = min(Vec8ui(index), n-1);
    }
    return nsimd_common::lookup8<pack256_8i_t, int32_t>(index1, table);
}

static inline Vec4q lookup4(Vec4q const & index, Vec4q const & table) {
    return nsimd_common::lookup4<pack256_4i_t, int64_t>(index, table);
}

template <int n>
static inline Vec4q lookup(Vec4q const & index, int64_t const * table) {
    if (n <= 0) return 0;
    // n > 0. Limit index
    Vec4uq index1;
    if ((n & (n-1)) == 0) {
        // n is a power of 2, make index modulo n
        index1 = Vec4uq(index) & (n-1);
    }
    else {
        // n is not a power of 2, limit to n-1.
        // There is no 64-bit min instruction, but we can use the 32-bit unsigned min,
        // since n is a 32-bit integer
        index1 = Vec4uq(min(Vec8ui(index), constant8i<n-1, 0, n-1, 0, n-1, 0, n-1, 0>()));
    }
    return nsimd_common::lookup4<pack256_4i_t, int64_t>(index1, table);
}


/*****************************************************************************
*
*          Other permutations with variable indexes
*
*****************************************************************************/

// Function shift_bytes_up: shift whole vector left by b bytes.
// You may use a permute function instead if b is a compile-time constant
static inline Vec32c shift_bytes_up(Vec32c const & a, int b) {
    if (b < 16) {    
        return Vec32c(shift_bytes_up(a.get_low(),b), shift_bytes_up(a.get_high(),b) | shift_bytes_down(a.get_low(),16-b));
    }
    else {
        return Vec32c(Vec16c(0), shift_bytes_up(a.get_high(),b-16));
    }
}

// Function shift_bytes_down: shift whole vector right by b bytes
// You may use a permute function instead if b is a compile-time constant
static inline Vec32c shift_bytes_down(Vec32c const & a, int b) {
    if (b < 16) {    
        return Vec32c(shift_bytes_down(a.get_low(),b) | shift_bytes_up(a.get_high(),16-b), shift_bytes_down(a.get_high(),b));
    }
    else {
        return Vec32c(shift_bytes_down(a.get_high(),b-16), Vec16c(0));
    }
}

/*****************************************************************************
*
*          Gather functions with fixed indexes
*
*****************************************************************************/
// Load elements from array a with indices i0, i1, i2, i3, i4, i5, i6, i7
template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
static inline Vec8i gather8i(void const * a) {
    return nsimd_common::gather8<i0,i1,i2,i3,i4,i5,i6,i7,pack256_8i_t, int32_t>(a);
}

template <int i0, int i1, int i2, int i3>
static inline Vec4q gather4q(void const * a) {
    return nsimd_common::gather4<i0,i1,i2,i3,pack256_4i_t, int64_t>(a);
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

template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
static inline void scatter(Vec8i const & data, void * array) {
    nsimd_common::scatter8<i0,i1,i2,i3,i4,i5,i6,i7,pack256_8i_t,int32_t>(data, array);
}

template <int i0, int i1, int i2, int i3>
static inline void scatter(Vec4q const & data, void * array) {
    nsimd_common::scatter4<i0,i1,i2,i3,pack256_4i_t,int64_t>(data, array);
}

static inline void scatter(Vec8i const & index, uint32_t limit, Vec8i const & data, void * array) {
    nsimd_common::scatter<pack256_8i_t,int32_t>(index, limit, data, array);
} 

static inline void scatter(Vec4q const & index, uint32_t limit, Vec4q const & data, void * array) {
    nsimd_common::scatter<pack256_4i_t,int64_t>(index, limit, data, array);
} 

// TODO reinterpret
static inline void scatter(Vec4i const & index, uint32_t limit, Vec4q const & data, void * array) {
    nsimd_common::scatter<pack256_4i_t,int64_t>(index, limit, data, array);
} 

/*****************************************************************************
*
*          Functions for conversion between integer sizes
*
*****************************************************************************/

// Extend 8-bit integers to 16-bit integers, signed and unsigned

// Function extend_low : extends the low 16 elements to 16 bits with sign extension
static inline Vec16s extend_low (Vec32c const & a) {
    return nsimd_common::extend_low<Vec32c, Vec16s>(a);
}

// Function extend_high : extends the high 16 elements to 16 bits with sign extension
static inline Vec16s extend_high (Vec32c const & a) {
    return nsimd_common::extend_high<Vec32c, Vec16s>(a);
}

// Function extend_low : extends the low 16 elements to 16 bits with zero extension
static inline Vec16us extend_low (Vec32uc const & a) {
    return nsimd_common::extend_low<Vec32uc, Vec16us>(a);
}

// Function extend_high : extends the high 19 elements to 16 bits with zero extension
static inline Vec16us extend_high (Vec32uc const & a) {
    return nsimd_common::extend_high<Vec32uc, Vec16us>(a);
}

// Extend 16-bit integers to 32-bit integers, signed and unsigned

// Function extend_low : extends the low 8 elements to 32 bits with sign extension
static inline Vec8i extend_low (Vec16s const & a) {
    return nsimd_common::extend_low<Vec16s, Vec8i>(a);
}

// Function extend_high : extends the high 8 elements to 32 bits with sign extension
static inline Vec8i extend_high (Vec16s const & a) {
    return nsimd_common::extend_high<Vec16s, Vec8i>(a);
}

// Function extend_low : extends the low 8 elements to 32 bits with zero extension
static inline Vec8ui extend_low (Vec16us const & a) {
    return nsimd_common::extend_low<Vec16us, Vec8ui>(a);
}

// Function extend_high : extends the high 8 elements to 32 bits with zero extension
static inline Vec8ui extend_high (Vec16us const & a) {
    return nsimd_common::extend_high<Vec16us, Vec8ui>(a);
}

// Extend 32-bit integers to 64-bit integers, signed and unsigned

// Function extend_low : extends the low 4 elements to 64 bits with sign extension
static inline Vec4q extend_low (Vec8i const & a) {
    return nsimd_common::extend_low<Vec8i, Vec4q>(a);
}

// Function extend_high : extends the high 4 elements to 64 bits with sign extension
static inline Vec4q extend_high (Vec8i const & a) {
    return nsimd_common::extend_high<Vec8i, Vec4q>(a);
}

// Function extend_low : extends the low 4 elements to 64 bits with zero extension
static inline Vec4uq extend_low (Vec8ui const & a) {
    return nsimd_common::extend_low<Vec8ui, Vec4uq>(a);
}

// Function extend_high : extends the high 4 elements to 64 bits with zero extension
static inline Vec4uq extend_high (Vec8ui const & a) {
    return nsimd_common::extend_high<Vec8ui, Vec4uq>(a);
}

// Compress 16-bit integers to 8-bit integers, signed and unsigned, with and without saturation

// Function compress : packs two vectors of 16-bit integers into one vector of 8-bit integers
// Overflow wraps around
static inline Vec32c compress (Vec16s const & low, Vec16s const & high) {
    return nsimd_common::compress32<pack256_32i_t, int8_t, pack256_16i_t, int16_t>(low, high, false);
}

// Function compress : packs two vectors of 16-bit integers into one vector of 8-bit integers
// Signed, with saturation
static inline Vec32c compress_saturated (Vec16s const & low, Vec16s const & high) {
    return nsimd_common::compress32<pack256_32i_t, int8_t, pack256_16i_t, int16_t>(low, high, true);
}

// Function compress : packs two vectors of 16-bit integers to one vector of 8-bit integers
// Unsigned, overflow wraps around
static inline Vec32uc compress (Vec16us const & low, Vec16us const & high) {
    return nsimd_common::compress32<pack256_32ui_t, uint8_t, pack256_16ui_t, uint16_t>(low, high, false);
}

// Function compress : packs two vectors of 16-bit integers into one vector of 8-bit integers
// Unsigned, with saturation
static inline Vec32uc compress_saturated (Vec16us const & low, Vec16us const & high) {
    return nsimd_common::compress32<pack256_32ui_t, uint8_t, pack256_16ui_t, uint16_t>(low, high, true);
}

// Compress 32-bit integers to 16-bit integers, signed and unsigned, with and without saturation

// Function compress : packs two vectors of 32-bit integers into one vector of 16-bit integers
// Overflow wraps around
static inline Vec16s compress (Vec8i const & low, Vec8i const & high) {
    return nsimd_common::compress16<pack256_16i_t, int16_t, pack256_8i_t, int32_t>(low, high, false);
}

// Function compress : packs two vectors of 32-bit integers into one vector of 16-bit integers
// Signed with saturation
static inline Vec16s compress_saturated (Vec8i const & low, Vec8i const & high) {
    return nsimd_common::compress16<pack256_16i_t, int16_t, pack256_8i_t, int32_t>(low, high, false);
}

// Function compress : packs two vectors of 32-bit integers into one vector of 16-bit integers
// Overflow wraps around
static inline Vec16us compress (Vec8ui const & low, Vec8ui const & high) {
    return nsimd_common::compress16<pack256_16ui_t, uint16_t, pack256_8ui_t, uint32_t>(low, high, false);
}

// Function compress : packs two vectors of 32-bit integers into one vector of 16-bit integers
// Unsigned, with saturation
static inline Vec16us compress_saturated (Vec8ui const & low, Vec8ui const & high) {
    return nsimd_common::compress16<pack256_16ui_t, uint16_t, pack256_8ui_t, uint32_t>(low, high, true);
}

// Compress 64-bit integers to 32-bit integers, signed and unsigned, with and without saturation

// Function compress : packs two vectors of 64-bit integers into one vector of 32-bit integers
// Overflow wraps around
static inline Vec8i compress (Vec4q const & low, Vec4q const & high) {  
    return nsimd_common::compress8<pack256_8i_t, int32_t, pack256_4i_t, int64_t>(low, high, false);
}

// Function compress : packs two vectors of 64-bit integers into one vector of 32-bit integers
// Signed, with saturation
static inline Vec8i compress_saturated (Vec4q const & a, Vec4q const & b) {
    return nsimd_common::compress8<pack256_8i_t, int32_t, pack256_4i_t, int64_t>(low, high, true);
}

// Function compress : packs two vectors of 32-bit integers into one vector of 16-bit integers
// Overflow wraps around
static inline Vec8ui compress (Vec4uq const & low, Vec4uq const & high) {
    return nsimd_common::compress8<pack256_8ui_t, uint32_t, pack256_4ui_t, uint64_t>(low, high, false);
}

// Function compress : packs two vectors of 64-bit integers into one vector of 32-bit integers
// Unsigned, with saturation
static inline Vec8ui compress_saturated (Vec4uq const & low, Vec4uq const & high) {
    return nsimd_common::compress8<pack256_8ui_t, uint32_t, pack256_4ui_t, uint64_t>(low, high, true);
}


/*****************************************************************************
*
*          Integer division operators
*
*          Please see the file vectori128.h for explanation.
*
*****************************************************************************/

// vector operator / : divide each element by divisor

// vector of 8 32-bit signed integers
static inline Vec8i operator / (Vec8i const & a, Vec8i const & d) {
    return nsimd::div(a,d);
}

// vector of 8 32-bit unsigned integers
static inline Vec8ui operator / (Vec8ui const & a, Vec8ui const & d) {
    return nsimd::div(a,d);
}

// vector of 16 16-bit signed integers
static inline Vec16s operator / (Vec16s const & a, Vec16s const & d) {
    return nsimd::div(a,d);
}

// vector of 16 16-bit unsigned integers
static inline Vec16us operator / (Vec16us const & a, Vec16us const & d) {
    return nsimd::div(a,d); 
}

// vector of 32 8-bit signed integers
static inline Vec32c operator / (Vec32c const & a, Vec32c const & d) {
    return nsimd::div(a,d);
}

// vector of 32 8-bit unsigned integers
static inline Vec32uc operator / (Vec32uc const & a, Vec32uc const & d) {
    return nsimd::div(a,d);
}

// vector operator /= : divide
static inline Vec8i & operator /= (Vec8i & a, Vec8i const & d) {
    a = a / d;
    return a;
}

// vector operator /= : divide
static inline Vec8ui & operator /= (Vec8ui & a, Vec8ui const & d) {
    a = a / d;
    return a;
}

// vector operator /= : divide
static inline Vec16s & operator /= (Vec16s & a, Vec16s const & d) {
    a = a / d;
    return a;
}


// vector operator /= : divide
static inline Vec16us & operator /= (Vec16us & a, Vec16us const & d) {
    a = a / d;
    return a;

}

// vector operator /= : divide
static inline Vec32c & operator /= (Vec32c & a, Vec32c const & d) {
    a = a / d;
    return a;
}

// vector operator /= : divide
static inline Vec32uc & operator /= (Vec32uc & a, Vec32uc const & d) {
    a = a / d;
    return a;
}


/*****************************************************************************
*
*          Integer division 2: divisor is a compile-time constant
*
*****************************************************************************/

// Divide Vec8i by compile-time constant
template <int32_t d>
static inline Vec8i divide_by_i(Vec8i const & x) {
    Static_error_check<(d!=0)> Dividing_by_zero;                     // Error message if dividing by zero
    if (d ==  1) return  x;
    if (d == -1) return -x;
    if (uint32_t(d) == 0x80000000u) return Vec4i(x == Vec4i(0x80000000)) & 1; // prevent overflow when changing sign
    pack256_8i_t div = nsimd::set1<pack256_8i_t>(d);
    return nsimd::div(x, div);
}

// define Vec8i a / const_int(d)
template <int32_t d>
static inline Vec8i operator / (Vec8i const & a, Const_int_t<d>) {
    return divide_by_i<d>(a);
}

// define Vec8i a / const_uint(d)
template <uint32_t d>
static inline Vec8i operator / (Vec8i const & a, Const_uint_t<d>) {
    Static_error_check< (d<0x80000000u) > Error_overflow_dividing_signed_by_unsigned; // Error: dividing signed by overflowing unsigned
    return divide_by_i<int32_t(d)>(a);                               // signed divide
}

// vector operator /= : divide
template <int32_t d>
static inline Vec8i & operator /= (Vec8i & a, Const_int_t<d> b) {
    a = a / b;
    return a;
}

// vector operator /= : divide
template <uint32_t d>
static inline Vec8i & operator /= (Vec8i & a, Const_uint_t<d> b) {
    a = a / b;
    return a;
}


// Divide Vec8ui by compile-time constant
template <uint32_t d>
static inline Vec8ui divide_by_ui(Vec8ui const & x) {
    Static_error_check<(d!=0)> Dividing_by_zero;                     // Error message if dividing by zero
    if (d == 1) return x;                                            // divide by 18_16i_t>(d0);
    pack256_8ui_t div = nsimd::set1<pack256_8ui_t>(d);
    return nsimd::div(x, div);
}

// define Vec8ui a / const_uint(d)
template <uint32_t d>
static inline Vec8ui operator / (Vec8ui const & a, Const_uint_t<d>) {
    return divide_by_ui<d>(a);
}

// define Vec8ui a / const_int(d)
template <int32_t d>
static inline Vec8ui operator / (Vec8ui const & a, Const_int_t<d>) {
    Static_error_check< (d>=0) > Error_dividing_unsigned_by_negative;// Error: dividing unsigned by negative is ambiguous
    return divide_by_ui<d>(a);                                       // unsigned divide
}

// vector operator /= : divide
template <uint32_t d>
static inline Vec8ui & operator /= (Vec8ui & a, Const_uint_t<d> b) {
    a = a / b;
    return a;
}

// vector operator /= : divide
template <int32_t d>
static inline Vec8ui & operator /= (Vec8ui & a, Const_int_t<d> b) {
    a = a / b;
    return a;
}


// Divide Vec16s by compile-time constant 
template <int d>
static inline Vec16s divide_by_i(Vec16s const & x) {
    const int16_t d0 = int16_t(d);                                   // truncate d to 16 bits
    Static_error_check<(d0 != 0)> Dividing_by_zero;                  // Error message if dividing by zero
    if (d0 ==  1) return  x;                                         // divide by  1
    if (d0 == -1) return -x;                                         // divide by -1
    if (uint16_t(d0) == 0x8000u) return Vec8s(x == Vec8s(0x8000)) & 1;// prevent overflow when changing sign
    pack256_16i_t div = nsimd::set1<pack256_16i_t>(d0);
    return nsimd::div(x, div);
}

// define Vec16s a / const_int(d)
template <int d>
static inline Vec16s operator / (Vec16s const & a, Const_int_t<d>) {
    return divide_by_i<d>(a);
}

// define Vec16s a / const_uint(d)
template <uint32_t d>
static inline Vec16s operator / (Vec16s const & a, Const_uint_t<d>) {
    Static_error_check< (d<0x8000u) > Error_overflow_dividing_signed_by_unsigned; // Error: dividing signed by overflowing unsigned
    return divide_by_i<int(d)>(a);                                   // signed divide
}

// vector operator /= : divide
template <int32_t d>
static inline Vec16s & operator /= (Vec16s & a, Const_int_t<d> b) {
    a = a / b;
    return a;
}

// vector operator /= : divide
template <uint32_t d>
static inline Vec16s & operator /= (Vec16s & a, Const_uint_t<d> b) {
    a = a / b;
    return a;
}


// Divide Vec16us by compile-time constant
template <uint32_t d>
static inline Vec16us divide_by_ui(Vec16us const & x) {
    const uint16_t d0 = uint16_t(d);                                 // truncate d to 16 bits
    Static_error_check<(d0 != 0)> Dividing_by_zero;                  // Error message if dividing by zero
    if (d0 == 1) return x;                                           // divide by 1
    pack256_16ui_t div = nsimd::set1<pack256_16ui_t>(d0);
    return nsimd::div(x, div);
}

// define Vec16us a / const_uint(d)
template <uint32_t d>
static inline Vec16us operator / (Vec16us const & a, Const_uint_t<d>) {
    return divide_by_ui<d>(a);
}

// define Vec16us a / const_int(d)
template <int d>
static inline Vec16us operator / (Vec16us const & a, Const_int_t<d>) {
    Static_error_check< (d>=0) > Error_dividing_unsigned_by_negative;// Error: dividing unsigned by negative is ambiguous
    return divide_by_ui<d>(a);                                       // unsigned divide
}

// vector operator /= : divide
template <uint32_t d>
static inline Vec16us & operator /= (Vec16us & a, Const_uint_t<d> b) {
    a = a / b;
    return a;
}

// vector operator /= : divide
template <int32_t d>
static inline Vec16us & operator /= (Vec16us & a, Const_int_t<d> b) {
    a = a / b;
    return a;
}


// define Vec32c a / const_int(d)
template <int d>
static inline Vec32c operator / (Vec32c const & a, Const_int_t<d>) {
    const int8_t d0 = int8_t(d);                                 // truncate d to 32 bits
    Static_error_check<(d0 != 0)> Dividing_by_zero;                  // Error message if dividing by zero
    if (d0 == 1) return x;                                           // divide by 1
    pack256_8i_t div = nsimd::set1<pack256_8i_t>(d0);
    return nsimd::div(x, div);
}

// define Vec32c a / const_uint(d)
template <uint32_t d>
static inline Vec32c operator / (Vec32c const & a, Const_uint_t<d>) {
    Static_error_check< (uint8_t(d)<0x80u) > Error_overflow_dividing_signed_by_unsigned; // Error: dividing signed by overflowing unsigned
    return a / Const_int_t<d>();                                     // signed divide
}

// vector operator /= : divide
template <int32_t d>
static inline Vec32c & operator /= (Vec32c & a, Const_int_t<d> b) {
    a = a / b;
    return a;
}
// vector operator /= : divide
template <uint32_t d>
static inline Vec32c & operator /= (Vec32c & a, Const_uint_t<d> b) {
    a = a / b;
    return a;
}

// define Vec32uc a / const_uint(d)
template <uint32_t d>
static inline Vec32uc operator / (Vec32uc const & a, Const_uint_t<d>) {
    const uint8_t d0 = uint8_t(d);                                 // truncate d to 32 bits
    Static_error_check<(d0 != 0)> Dividing_by_zero;                  // Error message if dividing by zero
    if (d0 == 1) return x;                                           // divide by 1
    pack256_8ui_t div = nsimd::set1<pack256_8ui_t>(d0);
    return nsimd::div(x, div);
}

// define Vec32uc a / const_int(d)
template <int d>
static inline Vec32uc operator / (Vec32uc const & a, Const_int_t<d>) {
    Static_error_check< (int8_t(d)>=0) > Error_dividing_unsigned_by_negative;// Error: dividing unsigned by negative is ambiguous
    return a / Const_uint_t<d>();                                    // unsigned divide
}

// vector operator /= : divide
template <uint32_t d>
static inline Vec32uc & operator /= (Vec32uc & a, Const_uint_t<d> b) {
    a = a / b;
    return a;
}

// vector operator /= : divide
template <int32_t d>
static inline Vec32uc & operator /= (Vec32uc & a, Const_int_t<d> b) {
    a = a / b;
    return a;
}

/*****************************************************************************
*
*          Horizontal scan functions
*
*****************************************************************************/

// Get index to the first element that is true. Return -1 if all are false
static inline int horizontal_find_first(Vec32cb const & x) {
    return nsimd_common::horizontal_fond_first(x);
}

static inline int horizontal_find_first(Vec16sb const & x) {
    return nsimd_common::horizontal_fond_first(x);
}

static inline int horizontal_find_first(Vec8ib const & x) {
    return nsimd_common::horizontal_fond_first(x);
}

static inline int horizontal_find_first(Vec4qb const & x) {
    return nsimd_common::horizontal_fond_first(x);
}

// Count the number of elements that are true
static inline uint32_t horizontal_count(Vec32cb const & x) {
    return nsimd::nbtrue(x);
}

static inline uint32_t horizontal_count(Vec16sb const & x) {
    return nsimd::nbtrue(x);
}

static inline uint32_t horizontal_count(Vec8ib const & x) {
    return nsimd::nbtrue(x);
}

static inline uint32_t horizontal_count(Vec4qb const & x) {
    return nsimd::nbtrue(x);
}

/*****************************************************************************
*
*          Boolean <-> bitfield conversion functions
*
*****************************************************************************/

//// to_bits: convert boolean vector to integer bitfield
//static inline uint32_t to_bits(Vec32cb const & x) {
//    return (uint32_t)_mm256_movemask_epi8(x);
//}
//
//// to_Vec16c: convert integer bitfield to boolean vector
//static inline Vec32cb to_Vec32cb(uint32_t x) {
//    return Vec32cb(Vec32c(to_Vec16cb(uint16_t(x)), to_Vec16cb(uint16_t(x>>16))));
//}
//
//// to_bits: convert boolean vector to integer bitfield
//static inline uint16_t to_bits(Vec16sb const & x) {
//    __m128i a = _mm_packs_epi16(x.get_low(), x.get_high());  // 16-bit words to bytes
//    return (uint16_t)_mm_movemask_epi8(a);
//}
//
//// to_Vec16sb: convert integer bitfield to boolean vector
//static inline Vec16sb to_Vec16sb(uint16_t x) {
//    return Vec16sb(Vec16s(to_Vec8sb(uint8_t(x)), to_Vec8sb(uint8_t(x>>8))));
//}
//
//#if INSTRSET < 9 || MAX_VECTOR_SIZE < 512
//// These functions are defined in Vectori512.h if AVX512 instruction set is used
//
//// to_bits: convert boolean vector to integer bitfield
//static inline uint8_t to_bits(Vec8ib const & x) {
//    __m128i a = _mm_packs_epi32(x.get_low(), x.get_high());  // 32-bit dwords to 16-bit words
//    __m128i b = _mm_packs_epi16(a, a);  // 16-bit words to bytes
//    return (uint8_t)_mm_movemask_epi8(b);
//}
//
//// to_Vec8ib: convert integer bitfield to boolean vector
//static inline Vec8ib to_Vec8ib(uint8_t x) {
//    return Vec8ib(Vec8i(to_Vec4ib(x), to_Vec4ib(x>>4)));
//}
//
//// to_bits: convert boolean vector to integer bitfield
//static inline uint8_t to_bits(Vec4qb const & x) {
//    uint32_t a = _mm256_movemask_epi8(x);
//    return ((a & 1) | ((a >> 7) & 2)) | (((a >> 14) & 4) | ((a >> 21) & 8));
//}
//
//// to_Vec4qb: convert integer bitfield to boolean vector
//static inline Vec4qb to_Vec4qb(uint8_t x) {
//    return  Vec4qb(Vec4q(-(x&1), -((x>>1)&1), -((x>>2)&1), -((x>>3)&1)));
//}
//
//#else  // function prototypes here only
//
//// to_bits: convert boolean vector to integer bitfield
//static inline uint8_t to_bits(Vec8ib x);
//
//// to_Vec8ib: convert integer bitfield to boolean vector
//static inline Vec8ib to_Vec8ib(uint8_t x);
//
//// to_bits: convert boolean vector to integer bitfield
//static inline uint8_t to_bits(Vec4qb x);
//
//// to_Vec4qb: convert integer bitfield to boolean vector
//static inline Vec4qb to_Vec4qb(uint8_t x);

#endif  // INSTRSET < 9 || MAX_VECTOR_SIZE < 512

#ifdef NSIMD_NAMESPACE
}
#endif

#endif // VECTORI256_H
