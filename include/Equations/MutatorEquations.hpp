#ifndef MutatorModel_MutatorEquations_hpp
#define MutatorModel_MutatorEquations_hpp

#include <Eigen/Dense>
#include <Algorithms/SpecialFunctions.hpp>
#include <Concepts/VectorFunction.hpp>

template< size_t _GenomeLenght > // number of ordinary genes
class MutatorEquations {
private:
  //! Constant parameters of model
  constexpr static int num_variables_{ 2 * (_GenomeLenght + 1) };  //!< Total number of variables
  double mutation_rate_P_{ 1.0 };   //!< Mutation rate for ordinary genes of wild type genome
  double mutation_rate_Q_{ 100.0 }; //!< Mutation rate for ordinary genes of mutator type genome
  double mutator_gene_transition_rate_P_to_Q_{ 1.0 }; //!< Forward mutation rate for mutator-gene
  double mutator_gene_transition_rate_Q_to_P_{ 0.0 }; //!< Backward mutation rate for mutator-gene
   
  //! Convenience typedefs
  using StateVector_t = Eigen::Matrix< double, -1, 1, 0, -1, 1>;
  using Distribution_t = Eigen::Matrix< double, (_GenomeLenght + 1), 1>;
public: 
  //! Define class for system state representation as an element of vector space
  struct StateVectorType : StateVector_t
  {
    using PlainObjectType = StateVector_t;

    Eigen::Map< Distribution_t > P{ this->data() };
    Eigen::Map< Distribution_t > Q{ this->data() + (_GenomeLenght + 1) };  

    StateVectorType() : StateVector_t(num_variables_, 1) {};
    StateVectorType(const StateVector_t& val) : StateVector_t(val) {};

    operator StateVector_t&() { return static_cast<StateVector_t&>(this); }
    operator StateVector_t() { return static_cast<StateVector_t>(*this); }; 

    void normilize() {
      static_cast<StateVector_t>(this).normalize();
    };
  };

  //! Define class for fitness function representation as an element of vector space
  struct FitnessVectorType : StateVector_t
  {
    Eigen::Map< Distribution_t > f{ this->data() };
    Eigen::Map< Distribution_t > g{ this->data() + (_GenomeLenght + 1) };
    
    FitnessVectorType() : StateVector_t(num_variables_, 1) {};
  };

  //! Export number of equations and variables
  constexpr static int NumberOfVariables = num_variables_;

private:
  //! Fitness function
  FitnessVectorType fitness_{ };
  
public:
  //! Define algebraic source term  
  struct SourceTerm : concepts::VectorFunction< double >
  {
  private:
    constexpr static int n_vars_{ (_GenomeLenght + 1) };
    double mutation_rate_P_;   //!< Mutation rate for ordinary genes of wild type genome
    double mutation_rate_Q_; //!< Mutation rate for ordinary genes of mutator type genome
    double mutator_gene_transition_rate_P_to_Q_; //!< Forward mutation rate for mutator-gene
    double mutator_gene_transition_rate_Q_to_P_; //!< Backward mutation rate for mutator-gene

    //! Fitness function
    FitnessVectorType fitness_{ };
  public:   
    //! Constructor
    SourceTerm(const FitnessVectorType& fitness, double mu1, double mu2, double alpha1, double alpha2) : 
      concepts::VectorFunction<double>(num_variables_, num_variables_),
      fitness_{ fitness },
      mutation_rate_P_{ mu1 },
      mutation_rate_Q_{ mu2 },
      mutator_gene_transition_rate_P_to_Q_{ alpha1 },
      mutator_gene_transition_rate_Q_to_P_{ alpha2 }    
    {};  

    //! Function that returns Jacobian matrix components
    JacobianType df(const StateVectorType& state) const {
      JacobianType jacobian(num_variables_, num_variables_);
      for (int i = 0; i < n_vars_; i++) {
        //R 
        for (int j = 0; j < n_vars_; j++) {
          jacobian(i, j) = -2 * fitness_.f(i) * state.P(j);
          jacobian(i + n_vars_, j) = -2 * fitness_.f(j) * state.P(j);
          jacobian(i, j + n_vars_) = -2 * fitness_.g(j) * state.Q(j);
          jacobian(i + n_vars_, j + n_vars_) = -2 * fitness_.g(j) * state.Q(j);
        };

        //A_1
        jacobian(i, i + n_vars_) += mutator_gene_transition_rate_Q_to_P_;
        jacobian(i + n_vars_, i) += mutator_gene_transition_rate_P_to_Q_;

        //A_2
        jacobian(i, i) += fitness_.f(i) - (mutation_rate_P_ + mutator_gene_transition_rate_P_to_Q_);
        jacobian(i + n_vars_, i + n_vars_) += fitness_.g(i) - (mutation_rate_Q_ + mutator_gene_transition_rate_Q_to_P_);

        //A_3
        if ((i + 1) < n_vars_) {
          jacobian(i, i + 1) += mutation_rate_P_ * (i + 1) / _GenomeLenght;
          jacobian(i + n_vars_, i + 1 + n_vars_) += mutation_rate_Q_ * (i + 1) / _GenomeLenght;
        };
        if ((i - 1) >= 0) {
          jacobian(i, i - 1) += mutation_rate_P_ * (_GenomeLenght - i + 1) / _GenomeLenght;
          jacobian(i + n_vars_, i - 1 + n_vars_) += mutation_rate_Q_ * (_GenomeLenght - i + 1) / _GenomeLenght;
        };

        //! Penalize
       /* if (state.P(i) > 1.0) jacobian(i, i) += std::exp(state.P(i) - 1.0);
        if (state.P(i) < 0.0) jacobian(i, i) += std::exp(-state.P(i));
        if (state.Q(i) > 1.0) jacobian(i + n_vars_, i + n_vars_) += std::exp(state.Q(i) - 1.0);
        if (state.Q(i) < 0.0) jacobian(i + n_vars_, i + n_vars_) += std::exp(-state.Q(i));*/
      };

      return jacobian;
    };
 
    //! Function that maps system state into source term
    StateVectorType operator()(const StateVectorType& state) const {
      StateVectorType source{ }; //Allocate storage for resulting source term
      source.setZero();

      //! Compute average fittness
      double R = fitness_.dot(state);      
    
      //! Compute source term
      for (int i = 0; i <= _GenomeLenght; i++) {
        //! Compute mutation process
        if ((i + 1) < n_vars_) {
          source.P(i) += state.P(i + 1) * mutation_rate_P_ * (i + 1) / _GenomeLenght;
          source.Q(i) += state.Q(i + 1) * mutation_rate_Q_ * (i + 1) / _GenomeLenght;
        };
        if ((i - 1) >= 0) {
          source.P(i) += state.P(i - 1) * mutation_rate_P_ * (_GenomeLenght - i + 1) / _GenomeLenght;
          source.Q(i) += state.Q(i - 1) * mutation_rate_Q_ * (_GenomeLenght - i + 1) / _GenomeLenght;
        };

        //! Compute mutator gene switching process
        source.P(i) += state.Q(i) * mutator_gene_transition_rate_Q_to_P_;
        source.Q(i) += state.P(i) * mutator_gene_transition_rate_P_to_Q_;

        //! Compute replication process
        source.P(i) += state.P(i) * (fitness_.f(i) - (mutation_rate_P_ + mutator_gene_transition_rate_P_to_Q_));
        source.Q(i) += state.Q(i) * (fitness_.g(i) - (mutation_rate_Q_ + mutator_gene_transition_rate_Q_to_P_));

        //! Compute selection and normalization term
        source.P(i) -= state.P(i) * R;
        source.Q(i) -= state.Q(i) * R;

        //! Penalize
       /* if (state.P(i) > 1.0) source(i) += std::exp(state.P(i) - 1.0) - 1.0;
        if (state.P(i) < 0.0) source(i) += std::exp(-state.P(i)) - 1.0;
        if (state.Q(i) > 1.0) source(i) += std::exp(state.Q(i) - 1.0) - 1.0;
        if (state.Q(i) < 0.0) source(i) += std::exp(-state.Q(i)) - 1.0;*/
      };
       
      return source; //!< Pass source term outside
    };

    int operator()(const InputType &x, ValueType& fvec) {
      fvec = operator()(x);
      //std::cout << fvec << std::endl;
      return 0;
    };

    int df(const InputType &x, JacobianType& fjac) {
      fjac = df(x);
      //std::cout << fjac << std::endl;
      return 0;
    };

  } Source_term;

  //! Constructor with specified mutation rates
  MutatorEquations(const FitnessVectorType& fitness, double mu1, double mu2, double alpha1, double alpha2) :
    fitness_{ fitness },
    mutation_rate_P_{ mu1 },
    mutation_rate_Q_{ mu2 },
    mutator_gene_transition_rate_P_to_Q_{ alpha1 },
    mutator_gene_transition_rate_Q_to_P_{ alpha2 },
    Source_term(fitness, mu1, mu2, alpha1, alpha2)  
  {
  };

  //! Accessors
  auto mutation_rate_P() const { return mutation_rate_P_ };
  auto mutation_rate_Q() const { return mutation_rate_Q_ };
  auto mutator_gene_transition_rate_P_to_Q() const { return mutator_gene_transition_rate_P_to_Q_ };
  auto mutator_gene_transition_rate_Q_to_P() const { return mutator_gene_transition_rate_Q_to_P_ };

};

#endif