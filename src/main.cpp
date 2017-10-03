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
  double sP;
  double sQ;
  double p;
  double q;
  double R;
  SolutionType type;
};

template< int N, typename F, typename G>
SolutionResult solve(double mu, double alpha, F fitness_f, G fitness_g, double tolerance) {
  int m = 2 * (N + 1);
  using Equations = MutatorEquation<N>;
  auto System = Equations(fitness_f, fitness_g, mu, alpha);
  Eigen::VectorXd x(m), S(m);
  S.setZero();
  x.setZero();
  x(0) = 0.1;
  x(1) = 0.2;
  x(2) = 0.7;
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
      System(x, S);
      x += dt * S;
      x /= x.lpNorm<1>();

      // Compute descriptive statistics
      auto residual_norm = S.norm();
  //    std::cout << "time = " << time << ", residual norm = " << residual_norm << std::endl;           

      // Compute average fitness and surplus for every type
      double R = 0.0;
      double RP = 0.0;
      double RQ = 0.0;
      for (auto i = 0; i <= N; i++) {
        R += x(i) * System.fitness()(i);
        R += x(i + N + 1)*System.fitness()(i + N + 1);
        RP += x(i) * System.fitness()(i);
        RQ += x(i + N + 1)*System.fitness()(i + N + 1);
      };

      if (R > 1e4) { 
        //std::cout << x << std::endl;
        continue;
      };
      
      double s = 0.0;
      double sP = 0.0;
      double sQ = 0.0;
      for (auto i = 0; i <= N; i++) {
        double pi = x.block<N + 1, 1>(0, 0).row(i).value();
        double qi = x.block<N + 1, 1>(N + 1, 0).row(i).value();
        s += pi * i + qi * i;// av gene state 
        sP += pi * i;// av gene state for normal subpopulation
        sQ += qi * i;// av gene state for mutstor subpopulation
      };

      // Update status
      status.R = R;
      status.s = (1.0 - 2.0 * s / N);
	  status.sP = (1.0 - 2.0 * sP / N);
	  status.sQ = (1.0 - 2.0 * sQ / N);

      status.p = x.block<N + 1, 1>(0, 0).sum();
      status.q = x.block<N + 1, 1>(N + 1, 0).sum();

    //  std::cout << "R = " << R << ", s = " << s << ",p = " << status.p << ", q = " << status.q << " " << std::endl;

      // Stop criteria
      if (residual_norm < tolerance) {
       // std::cout << "tolerance = " << tolerance << " reached, calculation stoped" << std::endl;
        if (status.q + tolerance > 1.0) {
          status.type = SolutionType::Mutator;
	  std::cout << "mutator phase";
        }
        else {
          status.type = SolutionType::Mixed;

	  std::cout << "mixed phase";
		}
		//std::cout << "x = " << x.transpose() << std::endl;
        break;
      };

      // Non-selective phase detection
      if (s < tolerance) {
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


int main()
{
	const int L = 50;
    double mu = 0.1; //1.5
    double alpha = 0.001; //0.1 
	double k = 5;
	double err = 0.001; 

	auto fitness_f = [] (double x) { return 1.2*(x + 1); };
	auto fitness_g = [](double x) { return (x + 1); };

	//auto fitness_f = [](double x) {return 1.5*(x + 1)*(x + 1); };
	//auto fitness_g = [](double x) {return (x + 1)*(x + 1);};
    //auto fitness = [] (double x) { return x > 0.8 ? 5*x*x : 0.0; };
	//for ( doub_fle k = 1; k <= 5; k += 0.5) {
	//auto fitness_f = [&k](double x) { return k*(x*x*x + 1); };
	//auto fitness_f = [](double x) { return 1.2*(x*x*x + 1); };
	//auto fitness_g = [](double x) { return (x*x*x + 1); };

	//auto fitness = [] (double x) { return x == 1.0 ? 10.0 : 0.0; };
	
	//std::ofstream ofs("phases.dat");
	

   std::ofstream ofs("theory.dat", std::ofstream::app);
	ofs << "*******************************************" << std::endl;
	ofs << "Parameters:" << "L =" << L << std::endl;
	ofs << "mu=" << mu << std::endl;
//	ofs << R"(VARIABLES = "mu", "R_mut", "R_mx", "R_num", "s", "s1", "s2")" << std::endl;
    ofs << R"(VARIABLES = "mu", "R_mut", "R_mx", "R_num", "q")" << std::endl;

	//std::cout << "parameters : " << std::endl;
    //std::cout << "alpha = " << alpha << std::endl;
    //std::cout << "mu    = " << mu << std::endl;

	/*for (double alpha = 0.001; alpha <= 3; alpha += 0.05)
		for (double mu = 0.001; mu <= 1.0; mu += 0.05) {
			double R_mt = -1e50;
			double R_mx = -1e50;
			for (double x = 1.0; x >= -1.0; x -= 1e-2) {
				double R1 = fitness_g(x) + mu * (std::sqrt(1 - x * x) - 1);
				if (R1 > R_mt) R_mt = R1;
				double R2 = fitness_f(x) - alpha + 0.01*(std::sqrt(1 - x * x) - 1);
				if (R2 > R_mx) R_mx = R2;
			};
			if (abs(R_mt - R_mx) < err) {
				ofs << alpha <<  mu << " " << R_mt << " " << R_mx << std::endl;
			}
		}
		*/
		
		
	for (double mu = 0.001; mu <= 1.0; mu += 0.05) {
		double R_mt = -1e50;
		double R_mx = -1e50;
		for (double x = 1.0; x >= -1.0; x -= 1e-2) {
			double R1 = fitness_g(x) + mu * (std::sqrt(1 - x * x) - 1);
			if (R1 > R_mt) R_mt = R1;
			double R2 = fitness_f(x) - alpha + 0.01*(std::sqrt(1 - x * x) - 1);
			if (R2 > R_mx) R_mx = R2;
		};
		//ofs << mu << " ";

		auto status = solve<L>(mu, alpha, fitness_f, fitness_g, 1e-10);

		ofs << mu << " " << R_mt << " " << R_mx << " " << status.R << " " << status.p << std::endl;



		//std::cout << "R_mt  = " << R_mt << std::endl;
		//std::cout << "R_mx  = " << R_mx << std::endl;

		//auto status = solve<L>(mu, alpha, fitness_f, fitness_g, 1e-10);



		//double R_mt = fitness_g(x) + mu * (std::sqrt(1 - x * x) - 1);
		//double R_mx = fitness_f(x) - alpha + std::sqrt(1 - x * x) - 1;
		//std::cout << "parameters : " << std::endl;
		//std::cout << "L = " << L << std::endl;
		//std::cout << "alpha = " << alpha << std::endl;
		//std::cout << "mu    = " << mu << std::endl;
		//std::cout << "k    = " << k << std::endl;
		
		//std::cout << "R_mt  = " << R_mt << std::endl;
		//std::cout << "R_mx  = " << R_mx << std::endl;
		std::cout << "result : " << std::endl;
		std::cout << "R = " << status.R << ", s = " << status.s << ", p = " << status.p << ", q = " << status.q << std::endl;
		//ofs << " " << status.R << " ";
		//ofs << " " << status.s <<  " " << status.sP << " " << status.sQ <<  std::endl;

	}

 // // Phases in parametric space
	//const int N = 100;
	//std::ofstream oph("phases.dat");
	//auto fitness = [=] (int i) {
	//  double x = 1.0 * i / N;
	//  double f = (1 - x)*(1 - x);
	//  return f;
	//};
	//int n_mu = 15;
	//double mu_min = 0.01;
	//double mu_max = 2.0;
	//int n_alpha = 20;
	//double alpha_min = 0.01;
	//double alpha_max = 1.5;
	////std::vector<SolutionType>types
	//for (auto i_alpha = n_alpha; i_alpha >= 0; i_alpha--) {
	//  auto alpha = alpha_min + i_alpha * (alpha_max - alpha_min) / n_alpha;
	//  //std::cout << alpha << " : ";
	//    for (auto i_mu = 0; i_mu <= n_mu; i_mu++) {
	//    auto mu = mu_min + i_mu * (mu_max - mu_min) / n_mu;
	//
	//    // solve
	//    auto status = solve<100>(mu, alpha, fitness, 1e-10);
	//    //std::cout << "mu = " << mu << ", alpha = " << alpha;
	//    //std::cout << ", phase = ";
	//    switch (status.type) {
	//    case SolutionType::NonSelective: { std::cout << "NS "; break; };
	//    case SolutionType::Mutator: { std::cout << "MT "; break; };
	//    case SolutionType::Mixed: { std::cout << "MX "; break; };
	//    }
	//    //std::cout << std::endl;
	//  }
	//  std::cout << std::endl;
	//
	//}

 // }

  /*
  using Equations = MutatorEquation<n>;
  using EquationsNumericalDiff = Eigen::NumericalDiff<Equations>;
  EquationsNumericalDiff Af;
  Af(x, fvec);
  std::cout << fvec;
  Eigen::LevenbergMarquardt<EquationsNumericalDiff> lm(Af);
  Eigen::LevenbergMarquardtSpace::Status status = lm.minimize(x);
  std::cout << "status: " << status << std::endl;
  std::cout << "x that minimizes the function: " << std::endl << x << std::endl;  */
	}