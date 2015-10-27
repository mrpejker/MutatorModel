#ifndef MutatorModel_MutatorEquations_hpp
#define MutatorModel_MutatorEquations_hpp

#include <array>
#include <Eigen/Dense>

template< 
  size_t GenomeLenght_ // number of ordinary genes
>
class MutatorEquations {
private:
  //! Distribution type
  using Distribution_t = std::array<double, GenomeLenght_ + 1>;

  //! Representation space dimensions
  constexpr static size_t rep_space_dims_{ 2 * (GenomeLenght_ + 1) };

public:
  constexpr static size_t Representation_space_dimensions { rep_space_dims_ };

  //! Define class for system state representation 
  struct StateVariables
  {
    //! Representation space dimensions
    constexpr static size_t size_ { 2 * (GenomeLenght_ + 1) };
    inline size_t size() const { return size_ };

    Distribution_t P{ 0.0 }; //Wild type relative frequencies
    Distribution_t Q{ 0.0 }; //Mutator type relative frequencies

    //! Default constructor constructs population of solely wild type with zero mutations
    StateVariables() {
      P[0] = 1.0;
      Q[0] = 0.0;
    };

    //! Normilize
    void normilize() {
      double sum{ 0.0 };
      for (auto p : P) sum += p;
      for (auto q : Q) sum += q;
      for (auto& p : P) p /= sum;
      for (auto& q : Q) q /= sum;
    };

    //! Zero based uniform index
    inline const double& operator[](size_t idx) const {
      if ((idx < 0) || ( idx >= Representation_space_dimensions)) throw std::out_of_range("index out of range");
      if (idx < Representation_space_dimensions / 2) return P[idx];
      return Q[idx - Representation_space_dimensions / 2];
    };

    //! Zero based uniform index
    inline double& operator[](size_t idx) {
      if ((idx < 0) || (idx >= Representation_space_dimensions)) throw std::out_of_range("index out of range");
      if (idx < Representation_space_dimensions / 2) return P[idx];
      return Q[idx - Representation_space_dimensions / 2];
    };
  };
private:
  
  //! Parameters of model
  constexpr static double mutation_rate_P_{ 1.0 };   // mu 1
  constexpr static double mutation_rate_Q_{ 100.0 }; // mu 2
  constexpr static double mutator_gene_transition_rate_P_to_Q_{ 1.0 }; // alpha 1
  constexpr static double mutator_gene_transition_rate_Q_to_P_{ 0.0 }; // alpha 2

public:
  //! Define functor class for right hand side as algebraic source term
  struct SourceTerm
  {
  private:
    //! Number of variables
    constexpr static size_t n_vars_ { (GenomeLenght_ + 1) };

    //! Matrix type of linear matrix
    using Matrix_t = Eigen::Matrix<double, 2 * n_vars_, 2 * n_vars_>;

    //! Fitness function
    Distribution_t f_{ 0.0 }; //Wild type relative frequencies
    Distribution_t g_{ 0.0 }; //Mutator type relative frequencies
  public:
    //! Default single peak function
    SourceTerm() {
      f_[0] = 1.0;
    };

    //! Function that returns linear matrix components
    Matrix_t linear_matrix(const StateVariables& state) {
      Matrix_t mat;
      for (int i = 0; i < n_vars_; i++) {
        //R
        for (int j = 0; j < n_vars_; j++) {
          mat(i, j) = 0; // -f_[j];
          mat(i + n_vars_, j) = 0; // -f_[j];
          mat(i, j + n_vars_) = 0; // -g_[j];
          mat(i + n_vars_, j + n_vars_) = 0;// -g_[j];
        };

        //A_1
        mat(i, i + n_vars_) += mutator_gene_transition_rate_P_to_Q_;
        mat(i + n_vars_, i) += mutator_gene_transition_rate_Q_to_P_;

        //A_2
        mat(i, i) += f_[i] - (mutation_rate_P_ + mutator_gene_transition_rate_P_to_Q_);
        mat(i + n_vars_, i + n_vars_) += g_[i] - (mutation_rate_Q_ + mutator_gene_transition_rate_Q_to_P_);

        //A_3
        if ((i + 1) < n_vars_) {
          mat(i, i + 1) += mutation_rate_P_ * (i + 1) / GenomeLenght_;
          mat(i + n_vars_, i + 1 + n_vars_) += mutation_rate_Q_ * (i + 1) / GenomeLenght_;
        };
        if ((i - 1) >= 0) {
          mat(i, i - 1) += mutation_rate_P_ * (GenomeLenght_ - i + 1) / GenomeLenght_;
          mat(i + n_vars_, i - 1 + n_vars_) += mutation_rate_Q_ * (GenomeLenght_ - i + 1) / GenomeLenght_;
        };

      };

      return mat;
    };
    
    //! Function that maps system state into source term
    StateVariables operator()(const StateVariables& state) {
      StateVariables source; //Allocate storage for resulting source term

      //! Compute average fittness
      double R = 0.0;
      for (auto i = 0; i <= GenomeLenght_; i++) {
        R += f_[i] * state.P[i] + g_[i] * state.Q[i];
      };

      //! Compute source term
      for (int i = 0; i <= GenomeLenght_; i++) {
        //! Compute mutation process
        if ((i + 1) < n_vars_) {
          source.P[i] += state.P[i + 1] * mutation_rate_P_ * (i + 1) / GenomeLenght_;
          source.Q[i] += state.Q[i + 1] * mutation_rate_Q_ * (i + 1) / GenomeLenght_;
        };
        if ((i - 1) >= 0) {
          source.P[i] += state.P[i - 1] * mutation_rate_P_ * (GenomeLenght_ - i + 1) / GenomeLenght_;
          source.Q[i] += state.Q[i - 1] * mutation_rate_Q_ * (GenomeLenght_ - i + 1) / GenomeLenght_;
        };

        //! Compute mutator gene switching process
        source.P[i] += state.Q[i] * mutator_gene_transition_rate_P_to_Q_;
        source.Q[i] += state.P[i] * mutator_gene_transition_rate_Q_to_P_;

        //! Compute replication process
        source.P[i] += state.P[i] * (f_[i] - (mutation_rate_P_ + mutator_gene_transition_rate_P_to_Q_));
        source.Q[i] += state.Q[i] * (g_[i] - (mutation_rate_Q_ + mutator_gene_transition_rate_Q_to_P_));

        //! Compute selection and normalization term
        source.P[i] -= state.P[i] * R;
        source.Q[i] -= state.Q[i] * R;
      };
 
      //source.normilize();
      return source; //!< Pass source term outside
    };

  };

};

#endif