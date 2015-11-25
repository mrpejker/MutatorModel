#ifndef MutatorModel_MutatorEquations_hpp
#define MutatorModel_MutatorEquations_hpp

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Concepts/VectorFunction.hpp>

using namespace Eigen;

//! Solution representation
template< size_t _GenomeLenght > // number of ordinary genes
struct MutatorEquationsSolution : public Matrix< double, 2 * (_GenomeLenght + 1), 1>
{
  constexpr static int NumberOfVariables{ 2 * (_GenomeLenght + 1) }; //! Export number of variables
  constexpr static int NumberOfGenes{ (_GenomeLenght + 1) }; //! Export total number of genes
  using Base = typename Matrix< double, NumberOfVariables, 1>;
  
  // Default constructor
  MutatorEquationsSolution(void) : Base() {}

  // This constructor allows you to construct MutatorEquationsSolution from Eigen expressions
  template<typename OtherDerived>
  MutatorEquationsSolution(const Eigen::MatrixBase<OtherDerived>& other)
    : Base(other)
  { }

  // This method allows you to assign Eigen expressions to MutatorEquationsSolution
  template<typename OtherDerived>
  MutatorEquationsSolution & operator= (const Eigen::MatrixBase <OtherDerived>& other)
  {
    this->Base::operator=(other);
    return *this;
  }

  inline auto P() { return head<NumberOfGenes>(); }
  inline auto P(int j) { return head<NumberOfGenes>(j); }
  inline auto Q() { return tail<NumberOfGenes>(); }
  inline auto Q(int i) { return tail<NumberOfGenes>(i); }
};

//! Fitness function type
template< size_t _GenomeLenght > // number of ordinary genes
struct FitnessFunction {
  constexpr static int NumberOfVariables{ 2 * (_GenomeLenght + 1) }; //! Export number of variables
  constexpr static int NumberOfGenes{ (_GenomeLenght + 1) }; //! Export total number of genes  
private:
  using Base = typename Matrix< double, NumberOfVariables, 1>;
  Base rep_{ };
public: 
  //Accessors
  inline auto f() { return rep_.head<NumberOfGenes>; };
  inline auto f(int i) { return rep_.head<NumberOfGenes>(i); };
  inline auto g() { return rep_.tail<NumberOfGenes>; };
  inline auto g(int i) { return rep_.tail<NumberOfGenes>(i); };

  //! Default constructor
  FitnessFunction(void) : rep_() {};
  
  //! Construct from function
  template <typename F>
  FitnessFunction(F fitness) {
    for (int i = 0; i < NumberOfGenes; i++) {
      auto x = (1.0*i) / _GenomeLenght;
      std::pair<double, double> res = fitness(x);
      rep_(i) = std::move(res.first);
      rep_(i + NumberOfGenes) = std::move(res.second);
    };
  };

  //Functor behaviour
  Map<VectorXd> operator()() { return Map<VectorXd>(rep_.data(), rep_.cols() * rep_.rows()); }
  
};

//! Define class for system state representation as an element of vector space
template< size_t _GenomeLenght > // number of ordinary genes
class MutatorEquations {  

public:
  //Compile time parameters
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
  MutatorEquations(const Parameters& parameters) :
    parameters_(parameters),
    Source_term(parameters)
  { };

  //Accessors
  auto mutation_rate_P() const { return parameters_.mutation_rate_P };
  auto mutation_rate_Q() const { return parameters_.mutation_rate_Q };
  auto mutator_gene_transition_rate_P_to_Q() const { return parameters_.mutator_gene_transition_rate_P_to_Q };
  auto mutator_gene_transition_rate_Q_to_P() const { return parameters_.mutator_gene_transition_rate_Q_to_P };

  //! Define algebraic source term  
  struct SourceTerm : concepts::VectorFunction< double >
  {
  private:      
    const Parameters& parameters_{ };
    InputType fitness_{ };
  public:   
    //! Constructor
    SourceTerm(const Parameters& parameters) : concepts::VectorFunction<double>(NumberOfVariables, NumberOfVariables),
      parameters_{ parameters }   
    {  
      fitness_ = parameters.fitness_function_();
    };              
 
    //! Function that maps system state into source term
    int operator()(const InputType &x, ValueType& fvec) const {
      SolutionType state = x;
      SolutionType source{ };

      //! Compute average fittness
      double R = fitness_.dot(state);
    
      //! Compute source term
      for (int i = 0; i < NumberOfGenes; i++) {
        source.P()(i) = 0;
        source.Q()(i) = 0;

        //! Compute mutation process
        if ((i + 1) < NumberOfGenes) {
          source.P()(i) = source.P()(i) + state.P()(i + 1) * parameters_.mutation_rate_P * (i + 1) / _GenomeLenght;
          source.Q()(i) = source.Q()(i) + state.Q()(i + 1) * parameters_.mutation_rate_Q * (i + 1) / _GenomeLenght;
        };
        if ((i - 1) >= 0) {
          source.P()(i) += state.P()(i - 1) * parameters_.mutation_rate_P * (_GenomeLenght - i + 1) / _GenomeLenght;
          source.Q()(i) += state.Q()(i - 1) * parameters_.mutation_rate_Q * (_GenomeLenght - i + 1) / _GenomeLenght;
        };

        //! Compute mutator gene switching process
        source.P()(i) += state.Q()(i) * parameters_.mutator_gene_transition_rate_Q_to_P;
        source.Q()(i) += state.P()(i) * parameters_.mutator_gene_transition_rate_P_to_Q;

        //! Compute replication process
        source.P()(i) += state.P()(i) * (fitness_(i) - 
          (parameters_.mutation_rate_P + parameters_.mutator_gene_transition_rate_P_to_Q));
        source.Q()(i) += state.Q()(i) * (fitness_(i + NumberOfGenes) -
          (parameters_.mutation_rate_Q + parameters_.mutator_gene_transition_rate_Q_to_P));

        //! Compute selection and normalization term
        source.P()(i) -= state.P()(i) * R;
        source.Q()(i) -= state.Q()(i) * R;

        ////! Penalize
        //if (state.P(i) > 1.0) source.P(i) += std::exp(state.P(i) - 1.0) - 1.0;
        //if (state.P(i) < 0.0) source.P(i) += std::exp(-state.P(i)) - 1.0;
        //if (state.Q(i) > 1.0) source.Q(i) += std::exp(state.Q(i) - 1.0) - 1.0;
        //if (state.Q(i) < 0.0) source.Q(i) += std::exp(-state.Q(i)) - 1.0;
      };
       
      fvec = source;
      return 0; //!< Pass source term outside
    };

    //! Function that returns Jacobian matrix components
    int df(const InputType &x, JacobianType& jacobian) { 
      auto state = static_cast<const SolutionType&>(x);

      for (int i = 0; i < NumberOfGenes; i++) {
        //R 
        for (int j = 0; j < NumberOfGenes; j++) {
          jacobian(i, j) = -2 * fitness_(i) * state.P()(j);
          jacobian(i + NumberOfGenes, j) = -2 * fitness_(j) * state.P()(j);
          jacobian(i, j + NumberOfGenes) = -2 * fitness_(j + NumberOfGenes) * state.Q()(j);
          jacobian(i + NumberOfGenes, j + NumberOfVariables) = -2 * fitness_(j + NumberOfGenes) * state.Q()(j);
        };

        //A_1
        jacobian(i, i + NumberOfGenes) += parameters_.mutator_gene_transition_rate_Q_to_P;
        jacobian(i + NumberOfGenes, i) += parameters_.mutator_gene_transition_rate_P_to_Q;

        //A_2
        jacobian(i, i) += fitness_(i) - (parameters_.mutation_rate_P + parameters_.mutator_gene_transition_rate_P_to_Q);
        jacobian(i + NumberOfGenes, i + NumberOfGenes) += fitness_(i + NumberOfGenes) - (parameters_.mutation_rate_Q + 
          parameters_.mutator_gene_transition_rate_Q_to_P);

        //A_3
        if ((i + 1) < NumberOfGenes) {
          jacobian(i, i + 1) += parameters_.mutation_rate_P * (i + 1) / _GenomeLenght;
          jacobian(i + NumberOfGenes, i + 1 + NumberOfGenes) += parameters_.mutation_rate_Q * (i + 1) / _GenomeLenght;
        };
        if ((i - 1) >= 0) {
          jacobian(i, i - 1) += parameters_.mutation_rate_P * (_GenomeLenght - i + 1) / _GenomeLenght;
          jacobian(i + NumberOfGenes, i - 1 + NumberOfGenes) += parameters_.mutation_rate_Q * (_GenomeLenght - i + 1) / _GenomeLenght;
        };

        //! Penalize
       /* if (state.P()(i) > 1.0) jacobian(i, i) += std::exp(state.P()(i) - 1.0);
        if (state.P()(i) < 0.0) jacobian(i, i) += std::exp(-state.P()(i));
        if (state.Q()(i) > 1.0) jacobian(i + NumberOfGenes, i + NumberOfGenes) += std::exp(state.Q()(i) - 1.0);
        if (state.Q()(i) < 0.0) jacobian(i + NumberOfGenes, i + NumberOfGenes) += std::exp(-state.Q()(i));*/
      };

      return 0;
    }; // Jacobian 

  } Source_term;

}; // class MutatorEquations

#endif