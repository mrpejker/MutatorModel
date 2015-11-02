#ifndef MutatorModel_DiscontinuousGalerkin_hpp
#define MutatorModel_DiscontinuousGalerkin_hpp

#include <fstream>
#include <NumericQuadrature.hpp>
#include <SpecialFunctions.hpp>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>
#include <unsupported/Eigen/KroneckerProduct>

template < typename _Equations, int _TimeOrder >
class DiscontinuousGalerkin {
private:
  using Equations = _Equations;
public:
  //! Associated types
  using StateVectorType = typename Equations::StateVectorType;  
  using SourceTerm = typename Equations::SourceTerm;
  constexpr static int NumberOfVariables{ Equations::NumberOfVariables };
private:
  //! Convenience typedefs
  constexpr static int num_variables_expanded_{ NumberOfVariables * _TimeOrder };
  using StateVectorExpanded_t = Eigen::Matrix< double, _TimeOrder, NumberOfVariables >;
  using Matrix_t = Eigen::Matrix<double, _TimeOrder, _TimeOrder >;

  //! Internal state representation   
  int current_iteration_{ 0 };
  double current_time_{ 0.0 };
  StateVectorType current_state_{ };
  StateVectorExpanded_t current_state_expansion_{ };

  //! Numerical intergrator object with required order of accuracy
  NumericQuadrature<_TimeOrder + 1> integrate_{ };

  //! Equations object
  Equations equations_;

private:
  struct TargetImplicitIdentity : Functor<double>
  {  
  private:
    //! Equations object
    const Equations& equations_;

    //! Helper function that samples solution at given point in time
    StateVectorType sample_solution_(const InputType &x, double tau) const {
      Eigen::Map< const StateVectorExpanded_t > solution( x.data() );
      StateVectorType state{ }; state.setZero();      
      for (int i = 0; i < _TimeOrder; i++) {        
        state += solution.row(i) * shifted_legendre(i, tau);
      };
      return state;
    };
  public:
    
    //! Constant matrices that depend solely on choice of orthogonal basis functions\  
    Matrix_t M; //! Mass matrix
    Matrix_t KTime; //! Temporal stiffness matrix
    Matrix_t FluxTime0; //! Time direction flux matrix at t = 0.0
    Matrix_t FluxTime1; //! Time direction flux matrix at t = 1.0

    //! Constructor
    TargetImplicitIdentity(const Equations& equations) : Functor<double>(num_variables_expanded_, num_variables_expanded_),
      equations_{ equations }
    {};        

    //! Jacobian matrix
    int df(const InputType &x, JacobianType &jac) const
    {                
      auto u = sample_solution_(x, 1.0);
      jac = equations_.Source_term.df(u);
      return 0;
    };
     
    //! Relation that must hold for solution
    int operator()(const InputType &x, ValueType &f) const
    { 
      auto u = sample_solution_(x, 1.0);
      f = equations_.Source_term(u);
      return 0;
    }
  } identity_;

  //! Main function that integrates underlying equation in time and advances one step
  StateVectorExpanded_t step_(double dt) {
    StateVectorExpanded_t result{ current_state_expansion_ };

    //Solve nonlinear system    
    Eigen::NumericalDiff<TargetImplicitIdentity> numDiff(identity_);
    Eigen::NumericalDiff<TargetImplicitIdentity>::JacobianType jac;
    /*numDiff.df(current_state_expansion_, jac);
    std::cout << "Numerical jacobian" << std::endl;
    std::cout << jac << std::endl;*/
    Eigen::LevenbergMarquardt<decltype(numDiff), double> lm(numDiff);
    //Eigen::LevenbergMarquardt<decltype(identity_), double> lm(identity_);
    lm.parameters.maxfev = 2000;
    lm.parameters.xtol = 1.0e-10;
    std::cout << lm.parameters.maxfev << std::endl;
        
    Eigen::VectorXd x(num_variables_expanded_); x << result.transpose();
    int ret = lm.minimize( x );
    result << x.transpose();

    std::cout << "Optimisation result summary : " << std::endl;
    std::cout << "state = (" << result << ")" << std::endl;    
    std::cout << "iterations : " << lm.iter << std::endl;

    TargetImplicitIdentity::ValueType fvalue( num_variables_expanded_ );
    identity_(result, fvalue);
    std::cout << "function value : " << fvalue.transpose() << std::endl;

    std::cout << "status : " << ret << std::endl;

    return result;
  };

public:

  //! Initialize solver
  DiscontinuousGalerkin(const Equations& equations) : equations_(equations),
    identity_{ equations }
  {
    //Compute M matrix
    for (int i = 0; i < _TimeOrder; i++)
      for (int j = 0; j < _TimeOrder; j++) {
        // Compute matrix elements
        identity_.M(i, j) = integrate_([&] (double tau) { return shifted_legendre(i, tau) * shifted_legendre(j, tau); }, 0.0, 1.0);
        identity_.KTime(i, j) = integrate_([&] (double tau) { return shifted_legendre_derivative(i, tau) * shifted_legendre(j, tau); }, 0.0, 1.0);
        identity_.FluxTime0(i, j) = shifted_legendre(i, 0.0) * shifted_legendre(j, 0.0);
        identity_.FluxTime1(i, j) = shifted_legendre(i, 1.0) * shifted_legendre(j, 1.0);
      };

    //Debug output
    std::cout << "M matrix : " << std::endl;
    std::cout << identity_.M << std::endl;
    std::cout << "KTime matrix : " << std::endl;
    std::cout << identity_.KTime << std::endl;
    std::cout << "FluxTime0 matrix : " << std::endl;
    std::cout << identity_.FluxTime0 << std::endl;
    std::cout << "FluxTime1 matrix : " << std::endl;
    std::cout << identity_.FluxTime1 << std::endl;
    
  };

public:

  //! Main function that integrates underlying equation in time
  StateVectorType solve(double time_start, double time_end, double time_step, const StateVectorType& initialConditions) {
    StateVectorType u{ initialConditions }; 
    std::ofstream ofs("result.dat");

    //Set initial conditions
    current_iteration_ = 0;
    current_time_ = time_start;
    current_state_ = initialConditions;
    current_state_expansion_.setZero();
    current_state_expansion_.row(0) = current_state_;
   
    StateVectorExpanded_t solution;
    for (; current_time_ <= time_end;) {
      //advance solution
      solution = step_(time_step);            

      //update state      
      current_state_ = u;
      current_state_expansion_ = solution;
      current_time_ += time_step;
      current_iteration_++;

      //iteration output
      std::cout << "Iteration : " << current_iteration_ << std::endl;
      std::cout << " P " << std::endl;
      std::cout << current_state_.P << std::endl;
      std::cout << " Q " << std::endl;
      std::cout << current_state_.Q << std::endl;
    };

    return current_state_;
  }; 
  
};

#endif