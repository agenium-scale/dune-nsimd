#if defined(_M_AMD64) || defined(_M_X64) || defined(__amd64) || \
    defined(__x86_64__)
#define NSIMD_SSE2
#else
#define NSIMD_AARCH64
#endif
#include <nsimd/nsimd-all.hpp>

#include <iostream>
#include <dune/nsimd/nsimd.hh>

int main() {

  Vec4i Vec(1,2,3,4);
  assert(Dune::Simd::lane(2,Vec)==3);
  assert(Vec[2]==3);
  assert(Vec.extract(2)==3);

  Dune::Simd::lane(2,Vec) += 3;
  assert(Dune::Simd::lane(2,Vec)==6);
  assert(Vec[2]==6);
  assert(Vec.extract(2)==6);

  Vec4i Vec2 = Vec+2;
  assert(Dune::Simd::lane(0,Vec2)==3);
  assert(Vec2[0]==3);
  assert(Vec2.extract(0)==3);

  Dune::Simd::lane(3,Vec2) = 10;
  assert(Dune::Simd::lane(3,Vec2)==10);
  assert(Vec2[3]==10);
  assert(Vec2.extract(3)==10);

}
