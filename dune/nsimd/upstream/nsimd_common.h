#ifndef NSIMD_COMMON_H
#define NSIMD_COMMON_H

#include "instrset.h"  // Select supported instruction set
#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>

namespace nsimd_common {
//------------------------ SATURATION -------------------------
//  32-bit saturation : return saturated signed 32-bit value
static int32_t saturate32(int64_t num)
{
	int32_t result = (int32_t)num;
	if (num > 0x07fffffff) {
		result = 0x07fffffff;
	} else if (num < -0x80000000) {
		result = -0x07fffffff;
	}
	return result;
}

//  32-bit saturation : return saturated unsigned 32-bit value
static uint32_t saturateU32(uint64_t num)
{
	uint32_t result = (uint32_t)num;
	if (num > 0x0ffffffff) {
		result = 0x0ffffffff;
	}
	return result;
}

//  16-bit saturation : return saturated signed 16-bit value
static int16_t saturate16(int32_t num)
{
	int16_t result = (int16_t)num;
	if (num > 0x07fff) {
		result = 0x7fff;
	} else if (num < -0x8000) {
		result = -0x7fff;
	}
	return result;
}

//  16-bit saturation : return saturated unsigned 16-bit value
static uint16_t saturateU16(uint32_t num)
{
	uint16_t result = (uint16_t)num;
	if (num > 0x0ffff) {
		result = 0xffff;
	}
	return result;
}

//  8-bit saturation : return saturated signed 8-bit value
static int8_t saturate8(int16_t num)
{
	uint8_t result = (int8_t)num;
	if (num > 0x07f) {
		result = 0x07f;
	} else if (num < -0x80) {
		result = -0x07f;
	}
	return result;
}

//  8-bit saturation : return saturated unsigned 8-bit value
static uint8_t saturateU8(uint16_t num)
{
	uint8_t result = (uint8_t)num;
	if (num > 0x0ff) {
		result = 0x0ff;
	} 
	return result;
}

//------------------------ LOAD_PARTIAL -------------------------
template <typename PackT, typename PacklT, typename T>
PackT load_partial(const T *v, int n)
{
    PackT r = nsimd::set1<PackT>(T(0));
    PackT mask = nsimd::mask_for_loop_tail<PacklT >(0, n);
    r = nsimd::masko_loadu1<T>(mask, v, r);
    return r;
}

//------------------------ STORE_PARTIAL -------------------------
template <typename PackT, typename PacklT, typename T>
void store_partial(T* v, int n, PackT p)
{
    PacklT mask = nsimd::mask_for_loop_tail<PacklT>(0, n);
    nsimd::mask_storeu1<T>(mask, v, p);
}

//------------------------ CUTOFF -------------------------
template <typename PackT, typename PacklT>
PackT cutoff(PackT& v, int n)
{
    unsigned int len = nsimd::len(v);
    if ((unsigned int)n >= len)
    {
        return v;
    }
    nsimd::pack<T> res = nsimd::set1<PackT >(T(0));
    nsimd::packl<T> mask = nsimd::mask_for_loop_tail<PacklT >(0, n);
    T* r = (T *)malloc(sizeof(T) * len);
    nsimd::storeu<PackT>(r, v);
    res = nsimd::masko_loadu1<PackT>(mask, r, res);
    return res;
}

//------------------------ SET_BIT -------------------------
template <typename PackT, typename T>
PackT set_bit(unsigned int index, T value, PackT& pack)
{
    unsigned int len = nsimd::len(pack);
    if (index >= len) {
        return pack;
    }
    T buf[len];
    nsimd::storeu(buf, pack);
    buf[index] = value;
    nsimd::pack<T> res = nsimd::loadu<PackT >(buf);
    return res;
}

//------------------------ GET_BIT -------------------------
template <typename PackT, typename T>
int get_bit(unsigned int index, PackT& pack)
{
    unsigned int len = nsimd::len(pack);
    if (index >= len) {
        return -1;
    }
    T buf[len];
    nsimd::storeu(buf, pack);
    return buf[index];
}

//------------------------ ABS SATURATED -------------------------
template <typename T>
nsimd::pack<T> abs_saturated(nsimd::pack<T>& pack)
{
    nsimd::pack<T> absa   = nsimd::abs(pack);
    return nsimd::adds(absa, nsimd::set1<nsimd::pack<T>>(T(0)));
}

//------------------------ HORIZONTALL ADD -------------------------
template <typename T>
float horizontal_add(nsimd::pack<T>& pack)
{
    return nsimd::addv(pack);
}

template <typename T>
float horizontal_add_x(nsimd::pack<T>& pack)
{
    return nsimd::addv(pack);
}

//------------------------ SHUFFLE -------------------------
template <typename T>
nsimd::pack<T> lookup32(nsimd::pack<T> const & index, nsimd::pack<T> const & table)
{
  int max_len = 32;
  T idx[max_len];
  T buf[max_len];
  T res[max_len];
  nsimd::storeu(idx, index);
  nsimd::storeu(buf, table);
  res[0] = (idx[0] >= 0 && idx[0] < max_len) ? buf[idx[0]] : T(0);
  res[1] = (idx[1] >= 0 && idx[1] < max_len) ? buf[idx[1]] : T(0);
  res[2] = (idx[2] >= 0 && idx[2] < max_len) ? buf[idx[2]] : T(0);
  res[3] = (idx[3] >= 0 && idx[3] < max_len) ? buf[idx[3]] : T(0);
  res[4] = (idx[4] >= 0 && idx[4] < max_len) ? buf[idx[4]] : T(0);
  res[5] = (idx[5] >= 0 && idx[5] < max_len) ? buf[idx[5]] : T(0);
  res[6] = (idx[6] >= 0 && idx[6] < max_len) ? buf[idx[6]] : T(0);
  res[7] = (idx[7] >= 0 && idx[7] < max_len) ? buf[idx[7]] : T(0);
  res[8] = (idx[8] >= 0 && idx[8] < max_len) ? buf[idx[8]] : T(0);
  res[9] = (idx[9] >= 0 && idx[9] < max_len) ? buf[idx[9]] : T(0);
  res[10] = (idx[10] >= 0 && idx[10] < max_len) ? buf[idx[10]] : T(0);
  res[11] = (idx[11] >= 0 && idx[11] < max_len) ? buf[idx[11]] : T(0);
  res[12] = (idx[12] >= 0 && idx[12] < max_len) ? buf[idx[12]] : T(0);
  res[13] = (idx[13] >= 0 && idx[13] < max_len) ? buf[idx[13]] : T(0);
  res[14] = (idx[14] >= 0 && idx[14] < max_len) ? buf[idx[14]] : T(0);
  res[15] = (idx[15] >= 0 && idx[15] < max_len) ? buf[idx[15]] : T(0);

  res[16] = (idx[16] >= 0 && idx[16] < max_len) ? buf[idx[16]] : T(0);
  res[17] = (idx[17] >= 0 && idx[17] < max_len) ? buf[idx[17]] : T(0);
  res[18] = (idx[18] >= 0 && idx[18] < max_len) ? buf[idx[18]] : T(0);
  res[19] = (idx[19] >= 0 && idx[19] < max_len) ? buf[idx[19]] : T(0);
  res[20] = (idx[20] >= 0 && idx[20] < max_len) ? buf[idx[20]] : T(0);
  res[21] = (idx[21] >= 0 && idx[21] < max_len) ? buf[idx[21]] : T(0);
  res[22] = (idx[22] >= 0 && idx[22] < max_len) ? buf[idx[22]] : T(0);
  res[23] = (idx[23] >= 0 && idx[23] < max_len) ? buf[idx[23]] : T(0);
  res[24] = (idx[24] >= 0 && idx[24] < max_len) ? buf[idx[24]] : T(0);
  res[25] = (idx[25] >= 0 && idx[25] < max_len) ? buf[idx[25]] : T(0);
  res[26] = (idx[26] >= 0 && idx[26] < max_len) ? buf[idx[26]] : T(0);
  res[27] = (idx[27] >= 0 && idx[27] < max_len) ? buf[idx[27]] : T(0);
  res[28] = (idx[28] >= 0 && idx[28] < max_len) ? buf[idx[28]] : T(0);
  res[29] = (idx[29] >= 0 && idx[29] < max_len) ? buf[idx[29]] : T(0);
  res[30] = (idx[30] >= 0 && idx[30] < max_len) ? buf[idx[30]] : T(0);
  res[31] = (idx[31] >= 0 && idx[31] < max_len) ? buf[idx[31]] : T(0);
  return nsimd::loadu<nsimd::pack<T> >(res);
}

template <typename T>
nsimd::pack<T> lookup16(nsimd::pack<T> const & index, nsimd::pack<T> const & table)
{
  int max_len = 16;
  T idx[max_len];
  T buf[max_len];
  T res[max_len];
  nsimd::storeu(idx, index);
  nsimd::storeu(buf, table);
  res[0] = (idx[0] >= 0 && idx[0] < max_len) ? buf[idx[0]] : T(0);
  res[1] = (idx[1] >= 0 && idx[1] < max_len) ? buf[idx[1]] : T(0);
  res[2] = (idx[2] >= 0 && idx[2] < max_len) ? buf[idx[2]] : T(0);
  res[3] = (idx[3] >= 0 && idx[3] < max_len) ? buf[idx[3]] : T(0);
  res[4] = (idx[4] >= 0 && idx[4] < max_len) ? buf[idx[4]] : T(0);
  res[5] = (idx[5] >= 0 && idx[5] < max_len) ? buf[idx[5]] : T(0);
  res[6] = (idx[6] >= 0 && idx[6] < max_len) ? buf[idx[6]] : T(0);
  res[7] = (idx[7] >= 0 && idx[7] < max_len) ? buf[idx[7]] : T(0);
  res[8] = (idx[8] >= 0 && idx[8] < max_len) ? buf[idx[8]] : T(0);
  res[9] = (idx[9] >= 0 && idx[9] < max_len) ? buf[idx[9]] : T(0);
  res[10] = (idx[10] >= 0 && idx[10] < max_len) ? buf[idx[10]] : T(0);
  res[11] = (idx[11] >= 0 && idx[11] < max_len) ? buf[idx[11]] : T(0);
  res[12] = (idx[12] >= 0 && idx[12] < max_len) ? buf[idx[12]] : T(0);
  res[13] = (idx[13] >= 0 && idx[13] < max_len) ? buf[idx[13]] : T(0);
  res[14] = (idx[14] >= 0 && idx[14] < max_len) ? buf[idx[14]] : T(0);
  res[15] = (idx[15] >= 0 && idx[15] < max_len) ? buf[idx[15]] : T(0);
  return nsimd::loadu<nsimd::pack<T> >(res);
}

template <typename T>
nsimd::pack<T> lookup8(nsimd::pack<T> const & index, nsimd::pack<T> const & table)
{
  int max_len = 8;
  T idx[max_len];
  T buf[max_len];
  T res[max_len];
  nsimd::storeu(idx, index);
  nsimd::storeu(buf, table);
  res[0] = (idx[0] >= 0 && idx[0] < max_len) ? buf[idx[0]] : T(0);
  res[1] = (idx[1] >= 0 && idx[1] < max_len) ? buf[idx[1]] : T(0);
  res[2] = (idx[2] >= 0 && idx[2] < max_len) ? buf[idx[2]] : T(0);
  res[3] = (idx[3] >= 0 && idx[3] < max_len) ? buf[idx[3]] : T(0);
  res[4] = (idx[4] >= 0 && idx[4] < max_len) ? buf[idx[4]] : T(0);
  res[5] = (idx[5] >= 0 && idx[5] < max_len) ? buf[idx[5]] : T(0);
  res[6] = (idx[6] >= 0 && idx[6] < max_len) ? buf[idx[6]] : T(0);
  res[7] = (idx[7] >= 0 && idx[7] < max_len) ? buf[idx[7]] : T(0);
  return nsimd::loadu<nsimd::pack<T> >(res);
}

template <typename T>
nsimd::pack<T> lookup4(nsimd::pack<T> const & index, nsimd::pack<T> const & table)
{
  int max_len = 4;
  T idx[max_len];
  T buf[max_len];
  T res[max_len];
  nsimd::storeu(idx, index);
  nsimd::storeu(buf, table);
  res[0] = (idx[0] >= 0 && idx[0] < max_len) ? buf[idx[0]] : T(0);
  res[1] = (idx[1] >= 0 && idx[1] < max_len) ? buf[idx[1]] : T(0);
  res[2] = (idx[2] >= 0 && idx[2] < max_len) ? buf[idx[2]] : T(0);
  res[3] = (idx[3] >= 0 && idx[3] < max_len) ? buf[idx[3]] : T(0);
  return nsimd::loadu<nsimd::pack<T> >(res);
}

template <typename T>
nsimd::pack<T> lookup2(nsimd::pack<T> const & index, nsimd::pack<T> const & table)
{
  int max_len = 2;
  T idx[max_len];
  T buf[max_len];
  T res[max_len];
  nsimd::storeu(idx, index);
  nsimd::storeu(buf, table);
  res[0] = (idx[0] >= 0 && idx[0] < max_len) ? buf[idx[0]] : T(0);
  res[1] = (idx[1] >= 0 && idx[1] < max_len) ? buf[idx[1]] : T(0);
  return nsimd::loadu<nsimd::pack<T> >(res);
}

//------------------------ SHIFT BYTES UP -------------------------
template <typename T>
nsimd::pack<T> shift_bytes_up64(nsimd::pack<T> const & a, int b)
{
  int max_len = 64;
    if (b > max_len) {
        return nsimd::set1<nsimd::pack<T> >(T(0));
    }
    static const T mask[max_len*2] = {
      -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
      -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,
      32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61
    };
    nsimd::pack<T> index = nsimd::loadu<nsimd::pack<T> >(mask+max_len-b);
    return lookup8<T>(index, a);
}

template <typename T>
nsimd::pack<T> shift_bytes_up32(nsimd::pack<T> const & a, int b)
{
  int max_len = 32;
    if (b > max_len) {
        return nsimd::set1<nsimd::pack<T> >(T(0));
    }
    static const T mask[max_len*2] = {
      -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
    };
    nsimd::pack<T> index = nsimd::loadu<nsimd::pack<T> >(mask+max_len-b);
    return lookup32<T>(index, a);
}

template <typename T>
nsimd::pack<T> shift_bytes_up16(nsimd::pack<T> const & a, int b)
{
  int max_len = 16;
    if (b > max_len) {
        return nsimd::set1<nsimd::pack<T> >(T(0));
    }
    static const T mask[max_len*2] = {
      -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    };
    nsimd::pack<T> index = nsimd::loadu<nsimd::pack<T> >(mask+max_len-b);
    return lookup16<T>(index, a);
}

template <typename T>
nsimd::pack<T> shift_bytes_up8(nsimd::pack<T> const & a, int b)
{
  int max_len = 8;
    if (b > max_len) {
        return nsimd::set1<nsimd::pack<T> >(T(0));
    }
    static const T mask[max_len*2] = {-1,-1,-1,-1,-1,-1,-1,-1, 0,1,2,3,4,5,6,7};
    nsimd::pack<T> index = nsimd::loadu<nsimd::pack<T> >(mask+max_len-b);
    return lookup8<T>(index, a);
}

template <typename T>
nsimd::pack<T> shift_bytes_up4(nsimd::pack<T> const & a, int b)
{
  int max_len = 4;
    if (b > max_len) {
        return nsimd::set1<nsimd::pack<T> >(T(0));
    }
    static const T mask[max_len*2] = {-1,-1,-1,-1,0,1,2,3};
    nsimd::pack<T> index = nsimd::loadu<nsimd::pack<T> >(mask+max_len-b);
    return lookup4<T>(index, a);
}

//------------------------ SHIFT BYTES DOWN -------------------------
template <typename T>
nsimd::pack<T> shift_bytes_down8(nsimd::pack<T> const & a, int b)
{
  int max_len = 2;
    if (b > max_len) {
        return nsimd::set1<nsimd::pack<T> >(T(0));
    }
    static const T mask[max_len*2] = {0,1,2,3,4,5,6,7,-1,-1,-1,-1,-1,-1,-1,-1};
    nsimd::pack<T> index = nsimd::loadu<nsimd::pack<T> >(mask+b);
    return lookup8<T>(index, a);
}

//------------------------ SCATTER -------------------------
template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, typename T>
static inline void scatter8(nsimd::pack<T> const & data, void * array)
{
    T a[8] = {i0, i1, i2, i3, i4, i5, i6, i7}; 
    nsimd::pack<T> const a2 = nsimd::loadu<nsimd::pack<T> >(a);
    nsimd::scatter((T*)array, a2, data);
}

template <typename T>
void scatter8(nsimd::pack<T> const & index, int limit, nsimd::pack<T> const & data, void * array) {
    int max_len = 8;
    T buf[max_len];
    T idx[max_len];
    nsimd::storeu(buf, data);
    nsimd::storeu(idx, index);
    for (int i = 0; i < max_len; i++) {
        if (idx[0] < limit) {
            array[index[i]] = buf[i];
        }
    }
}

//------------------------ GATHER -------------------------
template <int i0, int i1, typename T>
static inline nsimd::pack<T> gather2(void * array)
{
    T buf[2] = {i0, i1};
    nsimd::pack<T> a = nsimd::loadu<nsimd::pack<T> >((T*)buf);
    return nsimd::gather((int*)array, a);
}

template <int i0, int i1, int i2, int i3, typename T>
static inline nsimd::pack<T> gather4(void * array)
{
    T buf[4] = {i0, i1, i2, i3};
    nsimd::pack<T> a = nsimd::loadu<nsimd::pack<T> >((T*)buf);
    return nsimd::gather((int*)array, a);
}

template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, typename T>
static inline nsimd::pack<T> gather8(void * array)
{
    T buf[8] = {i0, i1, i2, i3, i4, i5, i6, i7};
    nsimd::pack<T> a = nsimd::loadu<nsimd::pack<T> >((T*)buf);
    return nsimd::gather((int*)array, a);
}

//------------------------ BLEND -------------------------
template <int i0, int i1, typename PackT, typename T>
PackT blend2(PackT const &v1, PackT const &v2) {
    int max_len = 2;
    T buf[2 * max_len];
    T res[max_len];
    nsimd::storeu(buf, v1);
    nsimd::storeu(buf + nsimd::len(v1), v2);
    res[0] = (i0 >= 0 && i0 < 2*max_len ? buf[i0] : T(0));
    res[1] = (i1 >= 0 && i1 < 2*max_len ? buf[i1] : T(0));
    return nsimd::loadu<PackT>(res);
}

template <int i0, int i1, int i2, int i3, typename PackT, typename T>
PackT blend4(PackT const &v1, PackT const &v2) {
    int max_len = 4;
    T buf[2 * max_len];
    T res[max_len];
    nsimd::storeu(buf, v1);
    nsimd::storeu(buf + nsimd::len(v1), v2);
    res[0] = (i0 >= 0 && i0 < 2*max_len ? buf[i0] : T(0));
    res[1] = (i1 >= 0 && i1 < 2*max_len ? buf[i1] : T(0));
    res[2] = (i2 >= 0 && i2 < 2*max_len ? buf[i2] : T(0));
    res[3] = (i3 >= 0 && i3 < 2*max_len ? buf[i3] : T(0));
    return nsimd::loadu<PackT>(res);
}

template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, typename PackT, typename T>
PackT blend8(PackT const &v1, PackT const &v2) {
    int max_len = 8;
    T buf[2 * max_len];
    T res[max_len];
    nsimd::storeu(buf, v1);
    nsimd::storeu(buf + nsimd::len(v1), v2);
    res[0] = (i0 >= 0 && i0 < 2*max_len ? buf[i0] : T(0));
    res[1] = (i1 >= 0 && i1 < 2*max_len ? buf[i1] : T(0));
    res[2] = (i2 >= 0 && i2 < 2*max_len ? buf[i2] : T(0));
    res[3] = (i3 >= 0 && i3 < 2*max_len ? buf[i3] : T(0));
    res[4] = (i4 >= 0 && i4 < 2*max_len ? buf[i4] : T(0));
    res[5] = (i5 >= 0 && i5 < 2*max_len ? buf[i5] : T(0));
    res[6] = (i6 >= 0 && i6 < 2*max_len ? buf[i6] : T(0));
    res[7] = (i7 >= 0 && i7 < 2*max_len ? buf[i7] : T(0));
    return nsimd::loadu<PackT>(res);
}

template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7,
    int i8, int i9, int i10, int i11, int i12, int i13, int i14, int i15, typename PackT, typename T>
PackT blend16(PackT const &v1, PackT const &v2) {
    int max_len = 16;
    T buf[2 * max_len];
    T res[max_len];
    nsimd::storeu(buf, v1);
    nsimd::storeu(buf + nsimd::len(v1), v2);
    res[0] = (i0 >= 0 && i0 < 2*max_len ? buf[i0] : T(0));
    res[1] = (i1 >= 0 && i1 < 2*max_len ? buf[i1] : T(0));
    res[2] = (i2 >= 0 && i2 < 2*max_len ? buf[i2] : T(0));
    res[3] = (i3 >= 0 && i3 < 2*max_len ? buf[i3] : T(0));
    res[4] = (i4 >= 0 && i4 < 2*max_len ? buf[i4] : T(0));
    res[5] = (i5 >= 0 && i5 < 2*max_len ? buf[i5] : T(0));
    res[6] = (i6 >= 0 && i6 < 2*max_len ? buf[i6] : T(0));
    res[7] = (i7 >= 0 && i7 < 2*max_len ? buf[i7] : T(0));

    res[8] = (i8 >= 0 && i8 < 2*max_len ? buf[i8] : T(0));
    res[9] = (i9 >= 0 && i9 < 2*max_len ? buf[i9] : T(0));
    res[10] = (i10 >= 0 && i10 < 2*max_len ? buf[i10] : T(0));
    res[11] = (i11 >= 0 && i11 < 2*max_len ? buf[i11] : T(0));
    res[12] = (i12 >= 0 && i12 < 2*max_len ? buf[i12] : T(0));
    res[13] = (i13 >= 0 && i13 < 2*max_len ? buf[i13] : T(0));
    res[14] = (i14 >= 0 && i14 < 2*max_len ? buf[i14] : T(0));
    res[15] = (i15 >= 0 && i15 < 2*max_len ? buf[i15] : T(0));
    return nsimd::loadu<PackT>(res);
}

//-------------------------- GET_HIGH/LOW ------------------------------
// len(PACK_T) = len(PACK_U)/2
template <typename PACK_T, typename PACK_U, typename T, typename U>
static inline PACK_T get_high(PACK_U const & a) {
    unsigned int len_U = nsimd::len(a);
    U buf[len_U];
    nsimd::storeu(buf, a);
    return nsimd::loadu<PACK_T>(buf+(len_U/2));
}

// len(PACK_T) = len(PACK_U)/2
template <typename PACK_T, typename PACK_U, typename T, typename U>
static inline PACK_T get_low(PACK_U const & a) {
    unsigned int len_U = nsimd::len(a);
    U buf[len_U];
    nsimd::storeu(buf, a);
    return nsimd::loadu<PACK_T>(buf);
}

//-------------------------- HORIZONTAL_COUNT ------------------------------
// Count the number of elements that are true
template <typename T>
static inline uint32_t horizontal_count(T const & x) {
    return nsimd::nbtrue(x);
}

//-------------------------- HORIZONTAL_FIND_FIRST ------------------------------
// Get the index of the first element set to true
template <typename T>
static inline uint32_t horizontal_find_first(T const & x) {
    return nsimd::nbtrue(x);
}

//------------------------ COMPRESS -------------------------
template <typename PackT, typename T, typename PackU, typename U>
packT compress4(PackU const & low, PackU const & high, bool is_saturated = false)
{
    U low1[2];
    U high1[2];
    T res[4];
    nsimd::storeu(low1, low);
    nsimd::storeu(high1, high);
    res[0] = saturate8(low1[0]);
    res[1] = saturate8(low1[1]);
    res[2] = saturate8(high1[0]);
    res[3] = saturate8(high1[1]);
    return nsimd::loadu<PackT>(res);
}

template <typename PackT, typename T, typename PackU, typename U>
PackT compress8(PackU const & low, PackU const & high, bool is_saturated = false)
{
    U low1[4];
    U high1[4];
    T res[8];
    nsimd::storeu(low1, low);
    nsimd::storeu(high1, high);
    res[0] = saturate16(low1[0]);
    res[1] = saturate16(low1[1]);
    res[2] = saturate16(low1[2]);
    res[3] = saturate16(low1[3]);
    res[4] = saturate16(high1[0]);
    res[5] = saturate16(high1[1]);
    res[6] = saturate16(high1[2]);
    res[7] = saturate16(high1[3]);
    return nsimd::loadu<PackT>(res);
}

template <typename PackT, typename T, typename PackU, typename U>
PackT compress16(PackU const & low, PackU const & high, bool is_saturated = false)
{
    U low1[8];
    U high1[8];
    T res[16];
    nsimd::storeu(low1, low);
    nsimd::storeu(high1, high);
    res[0] = saturate32(low1[0]);
    res[1] = saturate32(low1[1]);
    res[2] = saturate32(low1[2]);
    res[3] = saturate32(low1[3]);
    res[4] = saturate32(low1[4]);
    res[5] = saturate32(low1[5]);
    res[6] = saturate32(low1[6]);
    res[7] = saturate32(low1[7]);

    res[8] = saturate32(high1[0]);
    res[9] = saturate32(high1[1]);
    res[10] = saturate32(high1[2]);
    res[11] = saturate32(high1[3]);
    res[12] = saturate32(high1[4]);
    res[13] = saturate32(high1[5]);
    res[14] = saturate32(high1[6]);
    res[15] = saturate32(high1[7]);
    return nsimd::loadu<PackT>(res);
}

}      // nsimd_common
#endif // NSIMD_COMMON_H