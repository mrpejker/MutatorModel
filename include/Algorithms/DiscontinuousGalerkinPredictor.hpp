#ifndef MutatorModel_DiscontinuousGalerkin_hpp
#define MutatorModel_DiscontinuousGalerkin_hpp

#include <fstream>
#include <Algorithms/NumericQuadrature.hpp>
#include <BasisFunctions/ShiftedLegendreBasis.hpp>

#include <unsupported/Eigen/NumericalDiff>
#include <unsupported/Eigen/KroneckerProduct>

using namespace Eigen;

template < typename _Equations, int _TimeOrder >
class DiscontinuousGalerkinPredictor {
public:

  //! Compile time constants
  constexpr static int TimeOrder{ _TimeOrder };
  constexpr static int NumberOfVariables { _Equations::NumberOfVariables };
  constexpr static int NumberOfVariablesExpanded{ NumberOfVariables * TimeOrder };

  //! Associated types
  using Equations = _Equations;
  using Solution = typename Equations::SolutionType;
  using SourceTerm = typename Equations::SourceTerm;
  using BasisFunctions = typename ShiftedLegendreBasis< TimeOrder >;
private:
  //! Equations object
  Equations equations_;

  //! Basis functions
  BasisFunctions basis_{ };

  //! Numerical intergrator object with required order of accuracy
  NumericQuadrature<TimeOrder + 1> integrate_{ };      
public:

  const BasisFunctions& basis() const { return basis_; };

  //! Class that represents time solution
  class TimeExpandedSolution : public Matrix<double, TimeOrder, NumberOfVariables > {
    using Base = typename Matrix<double, TimeOrder, NumberOfVariables >;   

    //! Time extension interval
    double time_size_{ 1.0 };
  public:          
    //! Helper function that samples solution at given point in dimensionless time
    Solution sample_solution(double tau, const BasisFunctions& basis) const {
      Solution state{ }; state.setZero();  
      for (int i = 0; i < _TimeOrder; i++) {
        state += row(i) * basis.functions[i](tau);
      };
      return state;
    };

    //! Construct solution from initial conditions
    TimeExpandedSolution(const Solution& x) {
      setZero();
      row(0) = x;
    };

    //! Copy constructor
    //TimeExpandedSolution(const TimeExpandedSolution& other) : coefficients_(other.coeff()) { };
    //TimeExpandedSolution(TimeExpandedSolution&&) = delete;

    //! Default constructor
    TimeExpandedSolution(void) : Base() {}

    //! This constructor allows you to construct TimeExpandedSolution from Eigen expressions
    template<typename OtherDerived>
    TimeExpandedSolution(const Eigen::MatrixBase<OtherDerived>& other)
      : Base(other)
    { }

    // This method allows you to assign Eigen expressions to MutatorEquationsSolution
    template<typename OtherDerived>
    TimeExpandedSolution & operator= (const Eigen::MatrixBase <OtherDerived>& other)
    {
      this->Base::operator=(other);
      return *this;
    }
  };

private:
  struct TargetImplicitIdentity  {  
  private:
    //! Equations object
    const Equations& equations_;
    const BasisFunctions& basis_;
  public:
    //! Time step
    double dt;

    //! Solution at time \tau = 0
    Solution initialCondtions;

    //! Constant matrices that depend solely on choice of orthogonal basis functions    
    using Matrix_t = Eigen::Matrix<double, TimeOrder, TimeOrder >;    
    Matrix_t M;
    Matrix_t KTime; //! Temporal stiffness matrix
    Matrix_t FluxTime0; //! Time direction flux matrix at t = 0.0
    Matrix_t FluxTime1; //! Time direction flux matrix at t = 1.0

    //! Constructor
    TargetImplicitIdentity(const Equations& equations, const BasisFunctions& basis) : 
      concepts::VectorFunction<double, NumberOfVariablesExpanded, NumberOfVariablesExpanded>(),
      equations_{ equations }, 
      basis_{ basis }
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
      const TimeExpandedSolution& u = Map<TimeExpandedSolution>(x.data(), TimeOrder, NumberOfVariables);
      for (int i = 0; i < TimeOrder; i++) {
        for (int p = 0; p < NumberOfVariables; p++) {

        };
      };
      f = equations_.Source_term(u.sample_solution(1.0, basis_));
      return 0;
    }
  } identity_;

public:

  //! Initialize solver
  DiscontinuousGalerkinPredictor(const Equations& equations) : equations_(equations),
    identity_{ equations, basis_ }
  {
    //Compute M matrix
    for (int i = 0; i < _TimeOrder; i++)
      for (int j = 0; j < _TimeOrder; j++) {
        // Compute matrix elements
        identity_.M(i, j) = integrate_([&] (double tau) { return basis_.functions[i](tau) * basis_.functions[j](tau); }, 0.0, 1.0);
        identity_.KTime(i, j) = integrate_([&] (double tau) { return basis_.functions[i].df(tau) * basis_.functions[j](tau); }, 0.0, 1.0);
        identity_.FluxTime0(i, j) = basis_.functions[i](0.0) * basis_.functions[j](0.0);
        identity_.FluxTime1(i, j) = basis_.functions[i](1.0) * basis_.functions[j](1.0);
      };

    //Debug output
    /*std::cout << "M matrix : " << std::endl;
    std::cout << identity_.M << std::endl;
    std::cout << "KTime matrix : " << std::endl;
    std::cout << identity_.KTime << std::endl;
    std::cout << "FluxTime0 matrix : " << std::endl;
    std::cout << identity_.FluxTime0 << std::endl;
    std::cout << "FluxTime1 matrix : " << std::endl;
    std::cout << identity_.FluxTime1 << std::endl; */
  };

public:


  //! Main function that integrates underlying equation in time
  TimeExpandedSolution solve(double time_step, const TimeExpandedSolution& expanded_solution) {
    identity_.dt = time_step;
    identity_.initialCondtions = expanded_solution.sample_solution(0.0, basis_);

    //LevenbergMarquardt<TargetImplicitIdentity, double> lm(identity_);
    //lm.parameters.maxfev = 2000;
    ////lm.parameters.xtol = 1.0e-14;
    ////lm.parameters.ftol = 1.0e-14;
    ////lm.parameters.gtol = 1.0e-14;
    ////lm.parameters.epsfcn = 0.0;
    ////lm.parameters.factor = 10.0;     
    //    
    //LevenbergMarquardtSpace::Status status{ };
    //Map<TargetImplicitIdentity::InputType> xRef(expanded_solution.coeff(),
    //  NumberOfVariablesExpanded,1);
    //status = lm.minimizeInit(xRef);

    //if (status == LevenbergMarquardtSpace::ImproperInputParameters)
    //  return status;
    //do {
    //  status = lm.minimizeOneStep(xRef);
    //  std::cout << "Iter : " << lm.iter << " Solution : " << xRef.transpose() << std::endl;
    //} while (status == LevenbergMarquardtSpace::Running);

    return expanded_solution;
  }; 
  
};

#endif