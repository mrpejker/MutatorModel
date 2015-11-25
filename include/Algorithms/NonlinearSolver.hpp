#ifndef MutatorModel_NonlinearSolver_hpp
#define MutatorModel_NonlinearSolver_hpp

#include <Alglib/optimization.h>

namespace utility {

  using namespace alglib;

  template< typename _Functor >
  class NonlinearSolver {
    _Functor& target_function_;
  public:
    using Functor = _Functor;
    NonlinearSolver( Functor& f ) : target_function_(f) {}

    static void target_fvec(const real_1d_array &x, real_1d_array &fvec, void* ptr) {
      static_cast<Functor*>(ptr)->operator()(x, fvec);
    };

    static void target_fjac(const real_1d_array &x, real_1d_array &fvec, real_2d_array &fjac, void *ptr) {
      static_cast<Functor*>(ptr)->operator()(x, fvec);
      static_cast<Functor*>(ptr)->df(x, fjac);
    };

    void solve( alglib::real_1d_array& x ) {
      double epsg = 1e-14;
      double epsf = 0;
      double epsx = 0;
      ae_int_t maxits = 100;
      minlmstate state;
      minlmreport rep;

      void *ptr = static_cast<void*>(&target_function_);
      minlmcreatevj(target_function_.inputs, x, state);
      minlmsetcond(state, epsg, epsf, epsx, maxits);
      alglib::minlmoptimize(state, 
        NonlinearSolver::target_fvec, 
        NonlinearSolver::target_fjac, 
        nullptr,
        ptr
        );
      minlmresults(state, x, rep);

      printf("%d\n", int(rep.terminationtype)); // EXPECTED: 4
      //printf("%s\n", x.tostring(2).c_str());    // EXPECTED: [-3,+3]    std::cout << S  << std::endl;
    };
    
  };

};

#endif