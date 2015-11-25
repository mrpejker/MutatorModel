#ifndef MutatorModel_NonlinearSolver_hpp
#define MutatorModel_NonlinearSolver_hpp

#include <unsupported/Eigen/NonLinearOptimization>

#include <Concepts/VectorFunction.hpp>

using namespace Eigen;
using namespace concepts;

namespace utility {

  template< typename _Scalar >
  struct NonlinearSolver {
    using Scalar = _Scalar;
    using VectorFunctionType = typename VectorFunction<_Scalar>;
    using InputType = typename VectorFunction<_Scalar>::InputType;
    using ResultStatusType = LevenbergMarquardtSpace::Status;

    template< typename T >
    ResultStatusType solve(T& vf, typename T::InputType& x) {
      LevenbergMarquardt<T, Scalar> lm(vf);
      lm.parameters.maxfev = 2000;
      //lm.parameters.xtol = 1.0e-14;
      //lm.parameters.ftol = 1.0e-14;
      //lm.parameters.gtol = 1.0e-14;
      //lm.parameters.epsfcn = 0.0;
      //lm.parameters.factor = 10.0;     
      
      LevenbergMarquardtSpace::Status status{ };
      auto xRef = x;
      //VectorXd& xRef = static_cast<VectorXd&>(x);

      for (int outer_iteration = 0; outer_iteration < 100; outer_iteration++) {

        //Optimisation cycle
        status = lm.minimizeInit(xRef);

        //Stop criteria
        if (status == LevenbergMarquardtSpace::RelativeErrorTooSmall) { break; };

        if (status == LevenbergMarquardtSpace::ImproperInputParameters)
          return status;
        do {
          status = lm.minimizeOneStep(xRef);
          std::cout << "Iter : " << lm.iter << " Solution : " << xRef.transpose() << std::endl;
        } while (status == LevenbergMarquardtSpace::Running);       

        //Outer iteration action
        for (auto i = 0; i < xRef.size(); i++) {
          if (xRef(i) > 1.0) xRef(i) = 1.0;
          if (xRef(i) < 0.0) xRef(i) = 0.0;
        };
        double norm = xRef.sum();
        xRef /= norm;        

        //Debug info
        std::cout << "Outer Iteration : " << outer_iteration << std::endl;
        std::cout << "Result : " << static_cast<VectorXd>(x).transpose() << std::endl;
        std::cout << "Status : " << status << std::endl;
        std::cout << "Iterations : " << lm.iter << std::endl;

        //Stop criteria
        const double feps = 1e-15;
        if ((status == LevenbergMarquardtSpace::RelativeErrorTooSmall) && (std::abs(norm - 1.0) < feps)) break;
      };

      //ResultStatusType ret = lm.minimize(static_cast<VectorXd&>(x));

      return status;
    };
    
  };

}

#endif