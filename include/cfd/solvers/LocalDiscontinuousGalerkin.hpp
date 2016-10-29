#ifndef MutatorModel_LocalDiscontinuousGalerkin_hpp
#define MutatorModel_LocalDiscontinuousGalerkin_hpp

#include <cfd/numerics/NumericQuadrature.hpp>

template < typename Equations_, int TimeOrder_ = 2 >
class LocalDiscontinuousGalerkin {
private:
  using Equations = Equations_;

  //! Number of variables
  constexpr static size_t num_variables_ = Equations::Representation_space_dimensions;
  constexpr static size_t num_variables_expanded_ = num_variables_ * (TimeOrder_ + 1);
  constexpr static int mono_index(int i, int p) { return i + p * (TimeOrder_+1); };
public:
  //! Associated types
  using StateVariables = typename Equations::StateVariables;
  using SourceTerm = typename Equations::SourceTerm;
private:
  //! Internal state representation
  Eigen::Matrix<double, num_variables_, 1 > current_state_;
  Eigen::Matrix<double, num_variables_expanded_, 1 > current_state_expansion_;

  //! Source term
  SourceTerm source_{};

  //! Basis functions
  double legendre_(int n, double x) {
    if (n == 0) return 1;
    if (n == 1) return x;
    if (n >= 2) return ((2.0 * n + 1.0)*x*legendre_(n - 1, x) - n*legendre_(n - 2, x)) / (n + 1.0);
  };

  double shifted_legendre_(int n, double x) {    
    if (n == 0) return 1;
    if (n == 1) return 2*x - 1;
    if (n == 2) return 6*x*x - 6*x + 1;
    if (n == 3) return 20 * std::pow(x, 3) - 30 * std::pow(x, 2) + 12 * x - 1;
    throw std::exception("Unkown order of shifted Legendre polynomial");
    return 0;
  };

  //! Basis functions derivative
  double legendre_derivative_(int n, double x) {
    return (legendre_(n-1, x) - x*legendre_(n, x)) * n / (1.0 - x*x);
  };

  double shifted_legendre_derivative_(int n, double x) {
    if (n == 0) return 0;
    if (n == 1) return 2;
    if (n == 2) return 12*x - 6;
    if (n == 3) return 60*std::pow(x,2) - 60 * x + 12;
    throw std::exception("Unkown order of shifted Legendre polynomial derivative");
    return 0;
  };

  //! Numerical intergrator object with required order of accuracy
  NumericQuadrature<TimeOrder_ + 1> integrate_;

  //! Constant matrices
  Eigen::Matrix<double, TimeOrder_ + 1, TimeOrder_ + 1 > M_;
  Eigen::Matrix<double, TimeOrder_ + 1, TimeOrder_ + 1 > K_;
  Eigen::Matrix<double, TimeOrder_ + 1, TimeOrder_ + 1 > F0_;
  Eigen::Matrix<double, TimeOrder_ + 1, TimeOrder_ + 1 > F1_;
  Eigen::Matrix<double, num_variables_, num_variables_ > E_;
  Eigen::Matrix<double, num_variables_expanded_, num_variables_expanded_ > Y_;

private:
  //Target function
  //void target_function_(const real_1d_array &x, real_1d_array &f, void *ptr)
  //{
  //  fi[0] = 10 * pow(x[0] + 3, 2);
  //  fi[1] = pow(x[1] - 3, 2);
  //}

public:
  //! Initialize solver
  LocalDiscontinuousGalerkin() {
    //Compute M matrix
    for (int i = 0; i <= TimeOrder_; i++)
      for (int j = 0; j <= TimeOrder_; j++) {
        if (i != j) { // kroneker
          M_(i, j) = 0.0;
          continue;
        };

        // integrate
        double res = integrate_([&] (double tau) { return shifted_legendre_(i, tau) * shifted_legendre_(j, tau); }, 0.0, 1.0);
        M_(i, j) = res;
      };

    //Compute K matrix
    for (int i = 0; i <= TimeOrder_; i++)
      for (int j = 0; j <= TimeOrder_; j++) {
        // integrate
        double res = integrate_([&](double tau) { return legendre_derivative_(i, tau) * shifted_legendre_(j, tau); }, 0.0, 1.0);        
        K_(i, j) = res;
      };

    //Compute F0 matrix
    for (int i = 0; i <= TimeOrder_; i++)
      for (int j = 0; j <= TimeOrder_; j++) {
        F0_(i, j) = shifted_legendre_(i, 0.0) * shifted_legendre_(j, 0.0);
      };

    //Compute F1 matrix
    for (int i = 0; i <= TimeOrder_; i++)
      for (int j = 0; j <= TimeOrder_; j++) {
        F1_(i, j) = shifted_legendre_(i, 1.0) * shifted_legendre_(j, 1.0);
      };

    //Debug output
    std::cout << "M matrix : " << std::endl;
    std::cout << M_ << std::endl;
    std::cout << "K matrix : " << std::endl;
    std::cout << K_ << std::endl;
    std::cout << "F0 matrix : " << std::endl;
    std::cout << F0_ << std::endl;
    std::cout << "F1 matrix : " << std::endl;
    std::cout << F1_ << std::endl;
  };

  //! Main function that integrates underlying equation in time
  StateVariables solve(double dt, const StateVariables& initialConditions) {

    //Copy initial conditions
    for (int i = 0; i < num_variables_; i++) current_state_[i] = initialConditions[i];

    //Get linear matrix from source
    auto E = source_.linear_matrix(initialConditions);
    std::cout << " E matrix : " << std::endl;
    std::cout << E << std::endl;
    
    //Transition to reference time element
    auto EStar = dt * E;

    //Compute matrix Y and right hand side
    for (int i = 0; i <= TimeOrder_; i++) {
      for (int p = 0; p < num_variables_; p++) {
        int mono_index_i = mono_index(i, p);

        //Y matrix elements
        for (int j = 0; j <= TimeOrder_; j++) {
          for (int q = 0; q < num_variables_; q++) {
            int mono_index_j = mono_index(j, q);
            double delta_pq = (q == p) ? 1 : 0;
            double a = delta_pq * (F1_(i, j) - K_(i, j));
            a += -EStar(p, q) * M_(i, j);
            Y_(mono_index_i, mono_index_j) = a;
          };
        };

        //right hand side
        current_state_expansion_[mono_index_i] = shifted_legendre_(i, 0.0) * initialConditions[p];
      };
    }; // matrix computation
    std::cout << "Y matrix : " << std::endl;
    std::cout << Y_ << std::endl;

    //Solve system	
    //Eigen::JacobiSVD<decltype(Y_)> svd(Y_, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd Y(Y_);
    decltype(current_state_expansion_) x = Y.fullPivLu().solve(current_state_expansion_);

    //Check rank deficiency case
    //if (svd.rank() < ndims) throw std::runtime_error{ "Could not solve for gradient due to rank deficiency" };

    //Return result
    StateVariables result;

    for (int p = 0; p < num_variables_; p++) {
      result[p] = 0;
      for (int i = 0; i <= TimeOrder_; i++) {
        result[p] += shifted_legendre_(i, 1.0) * x[mono_index(i, p)];
      };
    };

    return result;
  };

};

#endif