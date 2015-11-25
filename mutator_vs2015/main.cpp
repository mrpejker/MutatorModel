#include <iostream>
#include <fstream>
#include <cmath>

#include <Algorithms/NumericQuadrature.hpp>
#include <Algorithms/NonlinearSolver.hpp>
//#include <Algorithms/DiscontinuousGalerkinPredictor.hpp>

#include <Equations/MutatorEquations.hpp>


int main(int argc, char *argv[]) {
   
  try {
    std::cout << std::scientific;
    const int N{ 1000 };
    using Equations = MutatorEquations<N>;
    using Solution = Equations::SolutionType;    
    using SourceTerm = Equations::SourceTerm<false>;        

    //Specify equations parameters
    Equations::Parameters params{ };
    double J{ 1.0 };
    double a{ 1.0e-5 };
    double mu{ 10 };
    //params.fitness_function_ = [&] (double x) { return (x == 0) ? std::pair<double, double>(J, J) : std::pair<double, double>(0, 0); };
    //params.fitness_function_ = [&] (double x) { return std::pair<double, double>(3*x*x/2, 3*x*x/2); };
    params.fitness_function_ = [&] (double x) { return std::pair<double, double>(0.3*x, 0.3*x); };
    params.mutation_rate_P = 1.0;
    params.mutation_rate_Q = mu;
    params.mutator_gene_transition_rate_P_to_Q = a;
    params.mutator_gene_transition_rate_Q_to_P = 0.0;


    Equations equations(params);

    //Specify initial conditions
    Solution u{ };
    u.P(0) = 0.5;
    u.Q(0) = 0.5;

    //Check source term validity
    SourceTerm& source = equations.Source_term;
    real_1d_array x;
    x.setlength(Equations::NumberOfVariables);
    x.setcontent(Equations::NumberOfVariables, u.data());
    real_1d_array dxdt;
    dxdt.setlength(Equations::NumberOfVariables);

    //Steady state computation
    //utility::NonlinearSolver<SourceTerm> nsolver( source );
    //nsolver.solve(x);
    //std::cout << "Solution x = " << x.tostring(2) << std::endl;
    
    // Time accurate solution solver algorithm
    //using Predictor = DiscontinuousGalerkinPredictor<Equations, 1>;
    //using TimeExpandedSolution = Predictor::TimeExpandedSolution;
    //Predictor DG_predictor(equations);
    
    int max_iter = 100000;
    double max_time = 10000.0;
    double time_step = 1.0e-2;

    //Current state information   
    int current_iteration{ 0 };
    double current_time{ 0.0 };
    Solution current_state { u };

    //std::ofstream ofs("result.dat");
    //TimeExpandedSolution solution{ current_state };

    //////Output initial conditions and source term
    ////source(solution.sample_solution(0.0), S);
    ////std::cout << S.transpose() << std::endl;

    std::ofstream ofs("result.dat");
    ofs << R"(VARIABLES = "iteration", "R", "q", "s" )" << std::endl;

    for (; current_time <= max_time;) {
      //  //advance solution
      //  DG_predictor.solve( time_step, solution );
      
      source(x, dxdt);
      for (int i = 0; i < Equations::NumberOfVariables; i++) {
        x[i] += dxdt[i] * time_step;
      };
      
      //update state      
      //  current_state = solution.sample_solution(1.0, DG_predictor.basis());
      auto prev_state{ current_state };
      for (int i = 0; i < Equations::NumberOfVariables; i++) {
        current_state[i] = x[i];
      };
      current_time += time_step;
      current_iteration++;

      //obtain characteristics of interest
      double R = 0;
      double s = 0;
      double s1 = 0;
      double s2 = 0;
      double sumP = 0;
      double sumQ = 0;
      for (int i = 0; i < Equations::NumberOfGenes; i++) {
        double x = 1.0 - (2.0*i) / N;
        s1 += current_state.P(i) * x;
        s2 += current_state.Q(i) * x;
        s += current_state.P(i) * x + current_state.Q(i) * x;
        R += (current_state.P(i) * params.fitness_function_.f(i) + current_state.Q(i) * params.fitness_function_.g(i));
        sumP += current_state.P(i);
        sumQ += current_state.Q(i);
      };
      s1 /= sumP;
      s2 /= sumQ;
      s /= (sumP + sumQ);
      R /= (sumP + sumQ);
      double q = sumQ / (sumP + sumQ);

      //iteration output
      std::cout << "Iteration " << current_iteration << " summary" << std::endl;
      std::cout << " R = " << R << std::endl;
      std::cout << " s = " << s << std::endl;
      std::cout << " s1 = " << s1 << std::endl;
      std::cout << " s2 = " << s2 << std::endl;
      std::cout << " q = " << q << std::endl;
      ofs << current_iteration << " " << R << " " << q << " " << s << " " << std::endl;

      //std::cout << "x = " << Map<RowVectorXd>(current_state.data(), 1, current_state.NumberOfVariables) << std::endl;
      //std::cout << " P = " << current_state.P().transpose() << std::endl;
      //std::cout << " Q = " << current_state.Q().transpose() << std::endl;

      // Check stop critiria
      if (current_iteration >= max_iter) {
        std::cout << "Maximum number of iterations reached." << std::endl;
        break;
      };

      // Check convergence criteria
      auto dx { 0.0 };
      for (int i = 0; i < Equations::NumberOfVariables; i++) {
        dx += std::abs(current_state[i] - prev_state[i]);
      };
      if (dx < 1e-14) {
        std::cout << "Convergence reached." << std::endl;
        break;
      };
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
