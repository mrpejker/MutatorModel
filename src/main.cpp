#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <equations/MutatorEquation/MutatorEquation.hpp>

enum class SolutionType {
  Mixed,
  Mutator,
  NonSelective
};

struct SolutionResult {
  double s;
  double p;
  double q;
  double R;
  SolutionType type;
};

template< int N, typename F >
SolutionResult solve(double mu, double alpha, F fitness, double tolerance) {
  int m = 2 * (N + 1);
  using Equations = MutatorEquation<N>;
  auto A = Equations(fitness, mu, alpha);
  Eigen::VectorXd x(m), S(m);
  S.setZero();
  x.setZero();
  x(0) = 0.01;
  x(N + 1) = 0.99;
  //std::cout << "initital conditions:" << std::endl;
  //std::cout << "x = " << x.transpose() << std::endl;
  SolutionResult status;

  { // Solve equation for steady state
    std::ofstream ofs("history.dat");
    ofs << R"(VARIABLES = "time", "avgFitness", "sumQ" )" << std::endl;
    ofs << 0.0 << " " << x.block<N + 1, 1>(0, 0).sum() << " " << x.block<N + 1, 1>(N + 1, 0).sum() << std::endl;

    // Solver and logging parameters
    double timestamp_time = 1.0;
    double next_timestep_time = timestamp_time;
    double dt = 1e-2;
    double max_time = 10000.0;

    // Calculation loop
    for (double time = 0.0; time <= max_time; time += dt) {
      // Update solution
      A(x, S);
      x += dt * S;
      x /= x.lpNorm<1>();

      // Compute descriptive statistics
      auto residual_norm = S.norm();
      std::cout << "time = " << time << ", residual norm = " << residual_norm << std::endl;           

      // Compute average fitness and surplus for every type
      double R = 0.0;
      double RP = 0.0;
      double RQ = 0.0;
      for (auto i = 0; i <= N; i++) {
        R += x(i) * A.fitness()(i);
        R += x(i + N + 1)*A.fitness()(i + N + 1);
        RP += x(i) * A.fitness()(i);
        RQ += x(i + N + 1)*A.fitness()(i + N + 1);
      };

      if (R > 1e4) { 
        std::cout << x << std::endl;
        continue;
      };
      
      double s = 0.0;
      double sP = 0.0;
      double sQ = 0.0;
      for (auto i = 0; i <= N; i++) {
        double pi = x.block<N + 1, 1>(0, 0).row(i).value();
        double qi = x.block<N + 1, 1>(N + 1, 0).row(i).value();
        s += pi * i + qi * i;
        sP += pi * i;
        sQ += qi * i;
      };

      // Update status
      status.R = R;
      status.s = (1.0 - 2.0 * s / N);
      status.p = x.block<N + 1, 1>(0, 0).sum();
      status.q = x.block<N + 1, 1>(N + 1, 0).sum();

      std::cout << "R = " << R << ", s = " << status.s << ",p = " << status.p << ", q = " << status.q << " " << std::endl;

      // Stop criteria
      if (residual_norm < tolerance) {
        std::cout << "tolerance = " << tolerance << " reached, calculation stoped" << std::endl;
        if (status.q + tolerance > 1.0) {
          status.type = SolutionType::Mutator;
        }
        else {
          status.type = SolutionType::Mixed;
        }
        break;
      };

      // Non-selective phase detection
      if (status.s < tolerance) {
        std::cout << "non-selective phase reached, calculation stoped" << std::endl;
        status.type = SolutionType::NonSelective;
        break;
      }
      
    };
  }
  
  { // Output resulting distributions            

    std::ofstream ofs("dist.dat");
    ofs << R"(VARIABLES = "mutations", "P", "Q" )" << std::endl;
    for (auto i = 0; i <= N; i++) {
      double pi = x.block<N + 1, 1>(0, 0).row(i).value();
      double qi = x.block<N + 1, 1>(N + 1, 0).row(i).value();
      ofs << i << " " << pi << " " << qi << std::endl;
    }
  }  
  return std::move(status);
};


int main(int argc, char *argv[])
{
  {
    const int L = 400;
    double mu = 4.0; //1.5
    double alpha = 1; //0.1      
    //auto fitness = [] (double x) { return (x + 1); };
    //auto fitness = [] (double x) { return x > 0.8 ? 5*x*x : 0.0; };
    //auto fitness = [] (double x) { return 5*x*x; };    
    //auto fitness = [] (double x) { return x == 1.0 ? 10.0 : 0.0; };

    if (argc != 7) {
      std::cout << "usage: <alpha> <mu> <a0> <a1> <a2> <a3>" << std::endl;
      std::cout << "where fitness function f(x) = a0 + a1*x + a2*x^2 + a3*x^3" << std::endl;
      return 0;
    };
    
    alpha = std::stod(argv[1]);
    mu = std::stod(argv[2]);
    double a0 = std::stod(argv[3]);
    double a1 = std::stod(argv[4]);
    double a2 = std::stod(argv[5]);
    double a3 = std::stod(argv[6]);
    auto fitness = [=] (double x) { return a0 + a1*x + a2*x*x + a3*x*x*x; };

    std::cout << "parameters : " << std::endl;
    std::cout << "L = " << L << std::endl;
    std::cout << "alpha = " << alpha << std::endl;
    std::cout << "mu    = " << mu << std::endl;

    auto status = solve<L>(mu, alpha, fitness, 1e-10);

    std::cout << "parameters : " << std::endl;
    std::cout << "L = " << L << std::endl;
    std::cout << "alpha = " << alpha << std::endl;
    std::cout << "mu    = " << mu << std::endl;    
    std::cout << "result : " << std::endl;
    std::cout << "R = " << status.R << ", s = " << status.s << ", p = " << status.p << ", q = " << status.q << std::endl;   
  }

}