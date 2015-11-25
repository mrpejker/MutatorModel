#ifndef MutatorModel_MutatorEquations_hpp
#define MutatorModel_MutatorEquations_hpp

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Alglib/optimization.h>

using namespace Eigen;
using namespace alglib;

//! Solution representation
template< size_t _GenomeLenght > // number of ordinary genes
class MutatorEquationsSolution
{
  std::array<double, 2 * (_GenomeLenght + 1) > solution_;
public:
  constexpr static int NumberOfVariables{ 2 * (_GenomeLenght + 1) }; //! Export number of variables
  constexpr static int NumberOfGenes{ (_GenomeLenght + 1) }; //! Export total number of genes

  // Data raw pointer
  double* data() { return solution_.data(); }

  // Default constructor
  MutatorEquationsSolution(void) : solution_() {}

  // Semantics
  double& operator[](int i) { return solution_[i]; }

  using Distribution_t = typename Matrix<double, NumberOfGenes, 1>;
  inline auto P() { return Map<Distribution_t>(solution_.data(), NumberOfGenes, 1); }
  inline auto Q() { return Map<Distribution_t>(solution_.data() + NumberOfGenes, NumberOfGenes, 1); }
  inline auto& P(int i) { return solution_[i]; }
  inline auto& Q(int i) { return solution_[i + NumberOfGenes]; }
};

//! Fitness function type
template< size_t _GenomeLenght > // number of ordinary genes
class FitnessFunction {
  std::array<double, 2 * (_GenomeLenght + 1) > fitness_;
public:
  constexpr static int NumberOfVariables{ 2 * (_GenomeLenght + 1) }; //! Export number of variables
  constexpr static int NumberOfGenes{ (_GenomeLenght + 1) };         //! Export total number of genes  
                                                          
  //! Default constructor
  FitnessFunction(void) : fitness_() {};
  
  //! Construct from function
  template <typename F>
  FitnessFunction(F fitness) {
    for (int i = 0; i < NumberOfGenes; i++) {
      auto x = 1.0 - (2.0*i) / _GenomeLenght;
      std::pair<double, double> res = fitness(x);
      fitness_[i] = std::move(res.first);
      fitness_[i + NumberOfGenes] = std::move(res.second);
    };
  };

  // Semantics
  using Distribution_t = typename Matrix<double, NumberOfGenes, 1>;
  inline auto f() { return Map<Distribution_t>(fitness_.data(), NumberOfGenes, 1); }
  inline auto g() { return Map<Distribution_t>(fitness_.data() + NumberOfGenes, NumberOfGenes, 1); }
  inline auto& f(int i) { return fitness_[i]; }
  inline auto& g(int i) { return fitness_[i + NumberOfGenes]; }
};

//! Define class for system state representation as an element of vector space
template< size_t _GenomeLenght > // number of ordinary genes
class MutatorEquations {
public:
  //! Compile time parameters
  constexpr static int NumberOfVariables{ 2 * (_GenomeLenght + 1) }; //!< Export number of variables
  constexpr static int NumberOfGenes{ (_GenomeLenght + 1) }; //!< Export total number of genes
                                                
  //! Assosciated types
  using SolutionType = typename MutatorEquationsSolution<_GenomeLenght>; //!<
  using FitnessFunctionType = typename FitnessFunction<_GenomeLenght>; //!<

  //! Parameters of model  
  struct Parameters {
    double mutation_rate_P{ 1.0 };   //!< Mutation rate for ordinary genes of wild type genome
    double mutation_rate_Q{ 100.0 }; //!< Mutation rate for ordinary genes of mutator type genome
    double mutator_gene_transition_rate_P_to_Q{ 1.0 }; //!< Forward mutation rate for mutator-gene
    double mutator_gene_transition_rate_Q_to_P{ 0.0 }; //!< Backward mutation rate for mutator-gene
    FitnessFunctionType fitness_function_{ }; //! Specified fitness function
  };
private:
  Parameters parameters_{ };
public:
  //! Constructor with specified mutation rates
  MutatorEquations(Parameters& parameters) :
    parameters_(parameters),
    Source_term(parameters_)
  { };

  //! Accessors
  auto mutation_rate_P() const { return parameters_.mutation_rate_P };
  auto mutation_rate_Q() const { return parameters_.mutation_rate_Q };
  auto mutator_gene_transition_rate_P_to_Q() const { return parameters_.mutator_gene_transition_rate_P_to_Q };
  auto mutator_gene_transition_rate_Q_to_P() const { return parameters_.mutator_gene_transition_rate_Q_to_P };

  //! Compute average fitness of solution
  double average_fitness(const SolutionType& solution, const FitnessFunctionType& fitness) const {
    auto avg = 0;
    for (int i = 0; i < NumberOfGenes; i++) {
      avg += fitness.f(i) * solution.P(i);
      avg += fitness.g(i) * solution.Q(i);
    };
    return avg;
  };

  //! Define algebraic source term  
  template< bool _IsPenalized = false >
  struct SourceTerm
  {
    constexpr static int inputs{ NumberOfVariables };
    constexpr static int outputs{ NumberOfVariables };
  private:
    Parameters& parameters_{ };
  public:
    //! Constructor
    SourceTerm(Parameters& parameters) :
      parameters_{ parameters }
      { };

    //! Function that maps system state into source term
    void operator()(const real_1d_array &x, real_1d_array &fvec) const {
      //! Compute average fittness
      double R = 0;
      for (int i = 0; i < NumberOfGenes; i++) {
        R += parameters_.fitness_function_.f(i)*x[i];
        R += parameters_.fitness_function_.g(i)*x[i + NumberOfGenes];
      };

      //! Compute source term
      for (int i = 0; i < NumberOfGenes; i++) {
        fvec[i] = 0;
        fvec[i + NumberOfGenes] = 0;

        //! Compute mutation process
        if ((i + 1) < NumberOfGenes) {
          fvec[i] += x[i + 1] * parameters_.mutation_rate_P * (i + 1) / _GenomeLenght;
          fvec[i + NumberOfGenes] += x[i + NumberOfGenes + 1] * parameters_.mutation_rate_Q * (i + 1) / _GenomeLenght;
        };
        if ((i - 1) >= 0) {
          fvec[i] += x[i - 1] * parameters_.mutation_rate_P * (_GenomeLenght - i + 1) / _GenomeLenght;
          fvec[i + NumberOfGenes] += x[i + NumberOfGenes - 1] * parameters_.mutation_rate_Q * (_GenomeLenght - i + 1) / _GenomeLenght;
        };

        //! Compute mutator gene switching process
        fvec[i] += x[i + NumberOfGenes] * parameters_.mutator_gene_transition_rate_Q_to_P;
        fvec[i + NumberOfGenes] += x[i] * parameters_.mutator_gene_transition_rate_P_to_Q;

        //! Compute replication process
        fvec[i] += x[i] * (parameters_.fitness_function_.f(i) -
          (parameters_.mutation_rate_P + parameters_.mutator_gene_transition_rate_P_to_Q));
        fvec[i + NumberOfGenes] += x[i + NumberOfGenes] * (parameters_.fitness_function_.g(i) -
          (parameters_.mutation_rate_Q + parameters_.mutator_gene_transition_rate_Q_to_P));

        //! Compute selection and normalization term
        fvec[i] -= x[i] * R;
        fvec[i + NumberOfGenes] -= x[i + NumberOfGenes] * R;

        //! Penalize
        if (_IsPenalized) {
          if (x[i] > 1.0) fvec[i] += std::exp(x[i] - 1.0) - 1.0;
          if (x[i] < 0.0) fvec[i] += std::exp(-x[i]) - 1.0;
          if (x[i + NumberOfGenes] > 1.0) fvec[i + NumberOfGenes] += std::exp(x[i + NumberOfGenes] - 1.0) - 1.0;
          if (x[i + NumberOfGenes] < 0.0) fvec[i + NumberOfGenes] += std::exp(-x[i + NumberOfGenes]) - 1.0;
        };
      };

      return;
    }; // operator()

    //! Function that returns Jacobian matrix components
    void df(const real_1d_array &x, real_2d_array &jacobian) const {

      for (int i = 0; i < NumberOfGenes; i++) {
        //R 
        for (int j = 0; j < NumberOfGenes; j++) {
          jacobian[i][j] = -2 * parameters_.fitness_function_.f(j) * x[j];
          jacobian[i + NumberOfGenes][j] = -2 * parameters_.fitness_function_.f(j) * x[j];
          jacobian[i][j + NumberOfGenes] = -2 * parameters_.fitness_function_.g(j) * x[j];
          jacobian[i + NumberOfGenes][j + NumberOfGenes] = -2 * parameters_.fitness_function_.g(j) * x[j];
        };

        //A_1
        jacobian[i][i + NumberOfGenes] += parameters_.mutator_gene_transition_rate_Q_to_P;
        jacobian[i + NumberOfGenes][i] += parameters_.mutator_gene_transition_rate_P_to_Q;

        //A_2
        jacobian[i][i] += parameters_.fitness_function_.f(i) - (parameters_.mutation_rate_P + parameters_.mutator_gene_transition_rate_P_to_Q);
        jacobian[i + NumberOfGenes][i + NumberOfGenes] += parameters_.fitness_function_.g(i) - (parameters_.mutation_rate_Q +
          parameters_.mutator_gene_transition_rate_Q_to_P);

        //A_3
        if ((i + 1) < NumberOfGenes) {
          jacobian[i][i + 1] += parameters_.mutation_rate_P * (i + 1) / _GenomeLenght;
          jacobian[i + NumberOfGenes][i + 1 + NumberOfGenes] += parameters_.mutation_rate_Q * (i + 1) / _GenomeLenght;
        };
        if ((i - 1) >= 0) {
          jacobian[i][i - 1] += parameters_.mutation_rate_P * (_GenomeLenght - i + 1) / _GenomeLenght;
          jacobian[i + NumberOfGenes][i - 1 + NumberOfGenes] += parameters_.mutation_rate_Q * (_GenomeLenght - i + 1) / _GenomeLenght;
        };

        //! Penalize
        if (_IsPenalized) {
          if (x[i] > 1.0) jacobian[i][i] += std::exp(x[i] - 1.0);
          if (x[i] < 0.0) jacobian[i][i] += std::exp(-x[i]);
          if (x[i + NumberOfGenes] > 1.0) jacobian[i + NumberOfGenes][i + NumberOfGenes] += std::exp(x[i + NumberOfGenes] - 1.0);
          if (x[i + NumberOfGenes] < 0.0) jacobian[i + NumberOfGenes][i + NumberOfGenes] += std::exp(-x[i + NumberOfGenes]);
        };
      };

      //std::cout << "Debug jacobian : " << jacobian.tostring(2) << std::endl;
      return;
    }; // Jacobian

  };

  //! Source term object
  SourceTerm<false> Source_term;

}; // class MutatorEquations

#endif