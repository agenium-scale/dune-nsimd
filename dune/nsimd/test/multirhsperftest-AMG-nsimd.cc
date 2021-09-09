#include "config.h"

#include <dune/nsimd/nsimd.hh>

#include "multirhsperftest.hh"

int main(int argc, char ** argv)
{
  runInformation times;

  times = runAMGCGTest< TYPE >(RUNS);

  printRunInformation(times);

  return 0;
}
