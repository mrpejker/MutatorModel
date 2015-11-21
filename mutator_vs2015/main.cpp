#include <iostream>
#include <fstream>

#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

#include <Algorithms/NumericQuadrature.hpp>
#include <Algorithms/SpecialFunctions.hpp>

#include <Algorithms/NonlinearSolver.hpp>
#include <Equations/MutatorEquations.hpp>

int main(int argc, char *argv[]) {
   
  try {
    using Equations = MutatorEquations<1>;
    using StateVectorType = Equations::StateVectorType;
    using FitnessVectorType = Equations::FitnessVectorType;
    using SourceTerm = Equations::SourceTerm;

    using NonlinearSolver = utility::NonlinearSolver<double>;
    NonlinearSolver nsolver;
    NonlinearSolver::InputType x(1, 1);
    x << 0.12;
    ShiftedLegendre<2> f{ };
    NonlinearSolver::ResultStatusType status = nsolver.solve(f, x);
    std::cout << " x = " << x << std::endl;
    
    //using Solver = DiscontinuousGalerkin<Equations, 1>;

    //Specify equations parameters
    FitnessVectorType fitness;
    fitness <<
      1.0, 0.0, // wild type
      1.0, 0.0; // mutator type
    std::cout << " fitness = " << std::endl;
    std::cout << fitness << std::endl;
    Equations equations(fitness, 
      1.0, //
      50.0, //
      1.0, //
      0.0 // 
      ); 

    //Specify initial conditions
    StateVectorType u{ };    
    u.setZero();
    //u.P(0) = 1.0;
    //u.Q << 0.52, 0.48;    
    std::cout << " P = " << std::endl;
    std::cout << u.P << std::endl;
    std::cout << " Q = " << std::endl;
    std::cout << u.Q << std::endl;

    std::cout << " S(u_0) = " << std::endl;
    std::cout << equations.Source_term(u) << std::endl;

    std::cout << " dS(u_0)/du = " << std::endl;
    std::cout << equations.Source_term.df(u) << std::endl;

    //Nonlinear solver
    //Eigen::NumericalDiff<VectorFunction<double>> numdiff(equations.Source_term);
    /*NonlinearSolver::ResultStatusType status = nsolver.solve(equations.Source_term, u);

    std::cout << " S(u_0) = " << std::endl;
    std::cout << equations.Source_term(u) << std::endl;

    std::cout << " dS(u_0)/du = " << std::endl;
    std::cout << equations.Source_term.df(u) << std::endl;*/
    
    //Solver solver(equations); // Instantiate solver algorithm object
    //    
    ////Solve
    //double max_time = 100;
    //double time_step = 1.0e-1;
    //auto result = solver.solve(0.0, 0.0, time_step, u);    

    // Output result

    std::cout << "press [ENTER] to continue " << std::endl;
    std::cin.get();
  }
  catch (std::exception e) {
    std::cout << e.what() << std::endl;
  }
  catch (...) {
    std::cout << "Fatal error." << std::endl;
    std::terminate();
  }
  return 0;
};
