#ifndef DUNE_VECTORCLASS_TEST_MULTIRHSPERFTEST
#define DUNE_VECTORCLASS_TEST_MULTIRHSPERFTEST

#include "config.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <dune/common/classname.hh>
#include <dune/common/simd/simd.hh>

#include <dune/istl/test/multirhstest.hh>

struct runInformation {
  std::vector<double> timestamps;
  int runs;
  std::string className;
};

template<class T>
runInformation runJacobiCGTest(unsigned int Runs) {

  runInformation out;
  out.className = Dune::className<T>();
  out.runs = Runs;

  // define Types
  typedef typename Dune::Simd::Scalar<T> MT;
  typedef Dune::FieldVector<T,1> VB;
  typedef Dune::FieldMatrix<MT,1,1> MB;
  typedef Dune::AlignedAllocator<VB> AllocV;
  typedef Dune::BlockVector<VB,AllocV> Vector;
  typedef Dune::BCRSMatrix<MB> Matrix;

  // size
  unsigned int size = 100;
  unsigned int N = size*size;

  // make a compressed row matrix with five point stencil
  Matrix A;
  setupLaplacian(A,size);

  typedef Dune::MatrixAdapter<Matrix,Vector,Vector> Operator;
  Operator op(A);        // make linear operator from A

  Dune::SeqJac<Matrix,Vector,Vector> jac(A,1,0.1);

  using VectorType = decltype(detectVectorType(op));

  double reduction = 1e-1;
  int verb = 1;
  Dune::CGSolver<VectorType> cg(op,jac,reduction,8000,verb);

  std::vector<double> measuredTime = run_test("Jacobi","CG",op,cg,N,Runs);
  out.timestamps = measuredTime;

  return out;
}

template<class T>
runInformation runAMGCGTest(unsigned int Runs) {

  runInformation out;
  out.className = Dune::className<T>();
  out.runs = Runs;

  // define Types
  typedef typename Dune::Simd::Scalar<T> MT;
  typedef Dune::FieldVector<T,1> VB;
  typedef Dune::FieldMatrix<MT,1,1> MB;
  typedef Dune::AlignedAllocator<VB> AllocV;
  typedef Dune::BlockVector<VB,AllocV> Vector;
  typedef Dune::BCRSMatrix<MB> Matrix;

  // size
  unsigned int size = 100;
  unsigned int N = size*size;

  // make a compressed row matrix with five point stencil
  Matrix A;
  setupLaplacian(A,size);

  typedef Dune::MatrixAdapter<Matrix,Vector,Vector> Operator;
  Operator op(A);        // make linear operator from A

  // AMG
  typedef Dune::Amg::RowSum Norm;
  typedef Dune::Amg::CoarsenCriterion<Dune::Amg::UnSymmetricCriterion<Matrix,Norm> >
          Criterion;
  typedef Dune::SeqSSOR<Matrix,Vector,Vector> Smoother;
  typedef typename Dune::Amg::SmootherTraits<Smoother>::Arguments SmootherArgs;
  SmootherArgs smootherArgs;
  smootherArgs.iterations = 1;
  smootherArgs.relaxationFactor = 1;
  unsigned int coarsenTarget = 1000;
  unsigned int maxLevel = 10;
  Criterion criterion(15,coarsenTarget);
  criterion.setDefaultValuesIsotropic(2);
  criterion.setAlpha(.67);
  criterion.setBeta(1.0e-4);
  criterion.setMaxLevel(maxLevel);
  criterion.setSkipIsolated(false);
  criterion.setNoPreSmoothSteps(1);
  criterion.setNoPostSmoothSteps(1);
  Dune::SeqScalarProduct<Vector> sp;
  typedef Dune::Amg::AMG<Operator,Vector,Smoother,Dune::Amg::SequentialInformation> AMG;
  Smoother smoother(A,1,1);
  AMG amg(op, criterion, smootherArgs);

  //CG-Solver
  using VectorType = decltype(detectVectorType(op));
  double reduction = 1e-1;
  int verb = 1;
  Dune::CGSolver<VectorType> cg(op,amg,reduction,8000,verb);

  std::vector<double> measuredTime = run_test("Jacobi","CG",op,cg,N,Runs);
  out.timestamps = measuredTime;

  return out;
}

void printRunInformation(runInformation info, bool verbose = false){
  std::cout << "========================================" << std::endl;
  std::cout << "type: " << info.className << std::endl;

  if(verbose){
    for(int i=0; i<info.runs; ++i)
      std::cout << "Run " << i+1 << " took " << info.timestamps[i] << std::endl;
    std::cout << "----------------------------------------" << std::endl;
  }

  if(info.runs > 1){
    info.timestamps.erase(info.timestamps.begin());
    --info.runs;

    double totalTime = 0.0;
    for(auto t : info.timestamps)
      totalTime += t;
    std::cout << "total time(" << info.runs << " runs): " << totalTime << std::endl;

    double median = totalTime/(double)info.runs;
    std::cout << "median: " << median << std::endl;

    double variance = 0.0;
    for(auto t : info.timestamps)
      variance += (t-median)*(t-median);
    variance /= (double)info.runs;
    std::cout << "variance: " << variance << std::endl;
  }
  std::cout << "========================================" << std::endl;
}

#endif
