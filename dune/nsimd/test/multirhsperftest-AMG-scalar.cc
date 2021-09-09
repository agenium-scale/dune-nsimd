#include "config.h"

#include "multirhsperftest.hh"

int main(int argc, char ** argv)
{
  runInformation times;

  times = runAMGCGTest<double>(RUNS);

  printRunInformation(times);

  return 0;
}
