#include <iostream>
#include <dune/vectorclass/vectorclass.hh>

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
