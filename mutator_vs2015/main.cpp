#include <iostream>
#include <fstream>

#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

#include <Algorithms/NumericQuadrature.hpp>

#include <Algorithms/NonlinearSolver.hpp>
#include <Algorithms/DiscontinuousGalerkinPredictor.hpp>
#include <Equations/MutatorEquations.hpp>

int main(int argc, char *argv[]) {
   
  try {
    const int N{ 1 };    
    using Equations = MutatorEquations<N>;
    using StateVectorType = Equations::StateVectorType;
    using FitnessVectorType = Equations::FitnessVectorType;
    using SourceTerm = Equations::SourceTerm;        

    //Specify equations parameters
    double J{ 1.0 };
    FitnessVectorType fitness{ };
    fitness.setZero();
    fitness(0) = J; // wild type
    fitness(N + 1) = J; // mutator type
    std::cout << " fitness = " << fitness.transpose() << std::endl;
    
    double a{ 1.0 };
    double mu{ 0.5 };
    Equations equations(fitness, 
      1.0, // mu_1
      mu, // mu_2
      a, // alpha_1
      0.0 // alpha_2
      ); 

    //Specify initial conditions
    StateVectorType u{ };    
    //u.setRandom(); 
    u.setZero();
    u.P(0) = 0.5;
    u.Q(0) = 0.5;
    //u.Q << 0.52, 0.48;    
    for (auto i = 0; i < u.size(); i++) {
      if (u(i) > 1.0) u(i) = 1.0;
      if (u(i) < 0.0) u(i) = 0.0;
    };
    double norm = u.sum(); u /= norm;
    
    std::cout << " P = " << u.P.transpose() << std::endl;
    std::cout << " Q = " << u.Q.transpose() << std::endl;
    std::cout << " S(u_0) = " << equations.Source_term(u).transpose() << std::endl;
    /*std::cout << " dS(u_0)/du = " << std::endl;
    std::cout << equations.Source_term.df(u) << std::endl;*/

    //Steady state nonlinear solver
    using SteadyStateSolver = utility::NonlinearSolver<double>;
    SteadyStateSolver steady_solver{ };    
    SteadyStateSolver::ResultStatusType status = steady_solver.solve(equations.Source_term, u);

    std::cout << " P = " << u.P.transpose() << std::endl;
    std::cout << " Q = " << u.Q.transpose() << std::endl;
    std::cout << " S(u_0) = " << equations.Source_term(u).transpose() << std::endl;
    /*std::cout << " dS(u_0)/du = " << std::endl;
    std::cout << equations.Source_term.df(u) << std::endl;*/
    
    // Time accurate solution solver algorithm
    using Predictor = DiscontinuousGalerkinPredictor<Equations, 1>;
    using Solution = Predictor::TimeExpandedSolution;
    Predictor DG_predictor(equations);
    
    double max_time = 100;
    double time_step = 1.0e-1;

    //Current state information   
    int current_iteration{ 0 };
    double current_time{ 0.0 };
    StateVectorType current_state { };
    current_state.setZero();
    current_state.P(0) = 1.0;
    
    std::ofstream ofs("result.dat");
    Solution solution{ current_state };
    for (; current_time <= max_time;) {
      //advance solution
      DG_predictor.solve( time_step, solution );

      //update state      
      current_state = solution.sample_solution(1.0); 
      current_time += time_step;
      current_iteration++;

      //iteration output
      std::cout << "Iteration " << current_iteration << "summary" << std::endl;
      std::cout << " P = " << current_state.P.transpose() << std::endl;
      std::cout << " Q = " << current_state.Q.transpose() << std::endl;
    };

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
