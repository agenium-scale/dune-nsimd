#include "config.h"


#include <dune/common/simd/loop.hh>

#include "multirhsperftest.hh"

int main(int argc, char ** argv)
{
  runInformation times;

  times = runAMGCGTest<Dune::LoopSIMD<double, LANES>>(RUNS);

  printRunInformation(times);

  return 0;
}
