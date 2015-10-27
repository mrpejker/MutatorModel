#include <iostream>
#include <fstream>
#include "Eigen/Dense"
#include "LocalDiscontinuousGalerkin.hpp"
#include "MutatorEquations.hpp"

int main(int argc, char *argv[]) {
  using Equations = MutatorEquations<1>;
  using Solver = LocalDiscontinuousGalerkin<Equations, 1>;
  using StateVector = Solver::StateVariables;

  try {
    Solver solver{}; // Instantiate solver algorithm object
    StateVector u_init{}; // Initial conditions
    u_init.P = { 1.0, 0.0 };
    u_init.Q = { 0.0, 0.0 };

    //Solve
    double max_time = 100;
    double time_step = 1.0e-1;
    std::ofstream ofs("result.dat");

    for (double time = 0.0; time <= max_time; time += time_step) {
      //solve
      StateVector u = solver.solve(time_step, u_init);

      //normilize
      u.normilize();

      //output
      std::cout << u.P[0] << " " << u.P[1] << std::endl;

      //next step
      u_init = u;
    };
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