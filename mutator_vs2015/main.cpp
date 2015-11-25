#include <iostream>
#include <fstream>

#include <Algorithms/NumericQuadrature.hpp>
#include <Algorithms/NonlinearSolver.hpp>
#include <Algorithms/DiscontinuousGalerkinPredictor.hpp>

#include <Equations/MutatorEquations.hpp>

int main(int argc, char *argv[]) {
   
  try {
    const int N{ 1 };    
    using Equations = MutatorEquations<N>;
    using StateVectorType = Equations::SolutionType;    
    using SourceTerm = Equations::SourceTerm;        

    //Specify equations parameters
    Equations::Parameters params{ };
    double J{ 1.0 };
    double a{ 1.0 };
    double mu{ 0.5 };
    params.fitness_function_ = [&] (double x) { return (x == 0) ? std::pair<double, double>(J, J) : std::pair<double, double>(0, 0); };       
    params.mutation_rate_P = 1.0;
    params.mutation_rate_Q = mu;
    params.mutator_gene_transition_rate_P_to_Q = a;
    params.mutator_gene_transition_rate_Q_to_P = 0;    
    Equations equations(params);

    //Specify initial conditions
    StateVectorType u{ };    
    u.setZero();
    u.P()(0) = 1.0;
   /* u.setZero();
    u.P(0) = 0.5;
    u.Q(0) = 0.5; 
    for (auto i = 0; i < u.size(); i++) {
      if (u(i) > 1.0) u(i) = 1.0;
      if (u(i) < 0.0) u(i) = 0.0;
    };
    double norm = u.sum(); u /= norm;*/

    //Check source term validity
    SourceTerm source = equations.Source_term;
    SourceTerm::ValueType S;
    source(u, S);
    std::cout << S  << std::endl;

    //Steady state computation
    utility::NonlinearSolver<double> nsolver{ };   
    VectorXd x = u;
    nsolver.solve(source, x);
    
    // Time accurate solution solver algorithm
    using Predictor = DiscontinuousGalerkinPredictor<Equations, 3>;
    using Solution = Predictor::TimeExpandedSolution;
    Predictor DG_predictor(equations);
    
    double max_time = 0.0;
    double time_step = 1.0e-1;

    //Current state information   
    int current_iteration{ 0 };
    double current_time{ 0.0 };
    StateVectorType current_state { u };

    std::ofstream ofs("result.dat");
    Solution solution{ current_state };    

    ////Output initial conditions and source term
    //source(solution.sample_solution(0.0), S);
    //std::cout << S.transpose() << std::endl;

    for (; current_time <= max_time;) {
      //advance solution
      DG_predictor.solve( time_step, solution );

      //update state      
      current_state = solution.sample_solution(1.0, DG_predictor.basis());
      current_time += time_step;
      current_iteration++;

      //iteration output
      std::cout << "Iteration " << current_iteration << "summary" << std::endl;
      //std::cout << " state = " << current_state.transpose() << std::endl;
      std::cout << " P = " << current_state.P().transpose() << std::endl;
      std::cout << " Q = " << current_state.Q().transpose() << std::endl;
    };

    // Output result
    std::cout << "press [ENTER] to continue " << std::endl;
    std::cin.get();
  }
  catch (std::exception e) {
    std::cout << e.what() << std::endl;
    std::cout << "press [ENTER] to continue " << std::endl;
    std::cin.get();
  }
  catch (...) {
    std::cout << "Fatal error." << std::endl;
    std::terminate();
  }
  return 0;
};
