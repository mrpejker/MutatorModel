#ifndef MutatorModel_DiscontinuousGalerkin_hpp
#define MutatorModel_DiscontinuousGalerkin_hpp

#include <fstream>
#include <Algorithms/NumericQuadrature.hpp>
#include <OrthogonalBasis/ShiftedLegendreBasis.hpp>

#include <unsupported/Eigen/NumericalDiff>
#include <unsupported/Eigen/KroneckerProduct>

template < typename _Equations, int _TimeOrder >
class DiscontinuousGalerkinPredictor {

public:
  //! Associated types
  using Equations = _Equations;
  using StateVectorType = typename Equations::StateVectorType;  
  using SourceTerm = typename Equations::SourceTerm;

  //! Compile time constants
  constexpr static int TimeOrder{ _TimeOrder };
  constexpr static int NumberOfVariables{ Equations::NumberOfVariables };
private:
  //! Equations object
  Equations equations_;

  //! Basis functions
  ShiftedLegendreBasis< TimeOrder > basis_functions_{ };

  //! Numerical intergrator object with required order of accuracy
  NumericQuadrature<TimeOrder + 1> integrate_{ };

  //! Convenience typedefs
  constexpr static int num_variables_expanded_{ NumberOfVariables * TimeOrder };
  using StateVector_t = Eigen::Matrix<double, 1, NumberOfVariables >;
  using StateVectorExpanded_t = Eigen::Matrix<double, TimeOrder, NumberOfVariables >;
  using Matrix_t = Eigen::Matrix<double, TimeOrder, TimeOrder >;
public:

  //! Class that represents time solution
  class TimeExpandedSolution {
    //! Basis functions reference
    ShiftedLegendreBasis< TimeOrder > basis_functions_{ };

    //! Time extension interval
    double time_size_{ 1.0 };

    //! Coefficients of decomposition
    Eigen::Matrix<double, TimeOrder, NumberOfVariables > coefficients_{ };

  public:
    //! Get size
    int size() const { return coefficients_.SizeAtCompileTime };

    //! Get flattened coefficients vector
    Eigen::VectorXd vector() { 
      return Eigen::Map<Eigen::VectorXd>(coefficients_.data(), coefficients_.SizeAtCompileTime); 
    };

    //! Helper function that samples solution at given point in dimensionless time
    StateVectorType sample_solution(double tau) const {      
      StateVectorType state{ }; state.setZero();  
      for (int i = 0; i < _TimeOrder; i++) {
        state += coefficients_.row(i) * basis_functions_[i](tau);
      };
      return state;
    };

    //! Construct solution from initial conditions
    TimeExpandedSolution(const StateVectorType& x) {
      coefficients_.setZero();
      coefficients_.row(0) = x;
    };
  };

private:
  struct TargetImplicitIdentity : VectorFunction<double>
  {  
  private:
    //! Equations object
    const Equations& equations_;    
  public:
    //! Constant matrices that depend solely on choice of orthogonal basis functions\  
    Matrix_t M; //! Mass matrix
    Matrix_t KTime; //! Temporal stiffness matrix
    Matrix_t FluxTime0; //! Time direction flux matrix at t = 0.0
    Matrix_t FluxTime1; //! Time direction flux matrix at t = 1.0

    //! Constructor
    TargetImplicitIdentity(const Equations& equations) : VectorFunction<double>(num_variables_expanded_, num_variables_expanded_),
      equations_{ equations }
    {};  

    //! Jacobian matrix
    int df(const InputType &x, JacobianType &jac) const
    {                
      auto u = sample_solution(x, 1.0);
      jac = equations_.Source_term.df(u);
      return 0;
    };
     
    //! Relation that must hold for solution
    int operator()(const InputType &x, ValueType &f) const
    {       
      auto u = static_cast<StateVectorExpandedType>(x);
      f = equations_.Source_term(u.sample_solution(1.0));
      return 0;
    }
  } identity_;

public:

  //! Initialize solver
  DiscontinuousGalerkinPredictor(const Equations& equations) : equations_(equations),
    identity_{ equations }
  {
    //Instantiate basis functions
    for (int i = 0; i <= TimeOrder; i++) basis_functions_[i] = shifted_legendre(i);

    //Compute M matrix
    for (int i = 0; i < _TimeOrder; i++)
      for (int j = 0; j < _TimeOrder; j++) {
        // Compute matrix elements
        //identity_.M(i, j) = integrate_([&] (double tau) { return shifted_legendre(i, tau) * shifted_legendre(j, tau); }, 0.0, 1.0);
        //identity_.KTime(i, j) = integrate_([&] (double tau) { return shifted_legendre_derivative(i, tau) * shifted_legendre(j, tau); }, 0.0, 1.0);
        //identity_.FluxTime0(i, j) = shifted_legendre(i)(0.0) * shifted_legendre(j)(0.0);
        //identity_.FluxTime1(i, j) = shifted_legendre(i)(1.0) * shifted_legendre(j)(1.0);
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
  TimeExpandedSolution solve(double time_step, const TimeExpandedSolution& solution) {
    

    return solution;
  }; 
  
};

#endif