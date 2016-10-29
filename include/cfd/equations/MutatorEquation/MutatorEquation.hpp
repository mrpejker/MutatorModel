#ifndef RustyFlow_cfd_equations_MutatorEquation_hpp
#define RustyFlow_cfd_equations_MutatorEquation_hpp

#include <array>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>

// Generic functor
template<typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
struct Functor
{
  typedef _Scalar Scalar;
  enum {
    InputsAtCompileTime = NX,
    ValuesAtCompileTime = NY
  };
  typedef Eigen::Matrix<Scalar, InputsAtCompileTime, 1> InputType;
  typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
  typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;

  int m_inputs, m_values;

  Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
  Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

  int inputs() const { return m_inputs; }
  int values() const { return m_values; }

};


template<
  size_t GenomeLenght_ // number of ordinary genes
>
class MutatorEquation;

using Eigen::SparseMatrix;
namespace Eigen {
namespace internal {
  // MutatorEquation looks-like a SparseMatrix, so let's inherits its traits:
  template<size_t GenomeLenght_>
  struct traits<MutatorEquation<GenomeLenght_>> : public Eigen::internal::traits<Eigen::SparseMatrix<double> > {
  };
}
}

template< 
  size_t GenomeLenght_ // number of ordinary genes
>
class MutatorEquation : public Eigen::EigenBase<MutatorEquation<GenomeLenght_>>, public Functor<double> {
private: // Internal representation
  constexpr static int n_vars_{ 2 * (GenomeLenght_ + 1) };
  constexpr static int n_positions_{ (GenomeLenght_ + 1) };
  constexpr static int rows_{ n_vars_ };
  constexpr static int cols_{ n_vars_ };

  //! Distribution type
  using Distribution_t = std::array<double, GenomeLenght_ + 1>;
public:
  // Required typedefs, constants, and method:
  typedef double Scalar;
  typedef double RealScalar;
  typedef int StorageIndex;
  enum {
    ColsAtCompileTime = Eigen::Dynamic,
    MaxColsAtCompileTime = Eigen::Dynamic,
    IsRowMajor = false
  };

  Eigen::Index rows() const { return rows_; }
  Eigen::Index cols() const { return cols_; }

  int inputs() const { return n_vars_; } // Number of variables
  int values() const { return n_vars_; } // Number of values  

  template<typename Rhs>
  Eigen::Product<MutatorEquation, Rhs, Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs> &x) const {
    return Eigen::Product<MutatorEquation, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
  }

private:
  //! Parameters of model
  double mutation_rate_P_{ 1.0 }; // mu 1
  double mutation_rate_Q_{ 2.0 }; // mu 2
  double mutator_gene_transition_rate_P_to_Q_{ 1.0 }; // alpha 1
  double mutator_gene_transition_rate_Q_to_P_{ 0.0 }; // alpha 2
  Eigen::VectorXd fitness_{ };
  std::vector<double> f_{};
  std::vector<double> g_{};
public:
  //! Constructor with default values
  MutatorEquation() = default;

  //! Constructor with parameters set
  template< typename F >
  MutatorEquation(F f, double mu, double alpha, double target_surplus = 1.0) :
    mutation_rate_Q_{ mu }, 
    mutator_gene_transition_rate_P_to_Q_{ alpha }
  {    
    double surplus = 0.0;
    fitness_.resize(2 * (GenomeLenght_ + 1), 1);
    f_.resize(GenomeLenght_ + 1);
    g_.resize(GenomeLenght_ + 1);
    for (int i = 0; i < GenomeLenght_ + 1; i++) { 
      double val_x = 1.0 * GenomeLenght_ + 1 - 2 * i;
      val_x /= GenomeLenght_ + 1.0;
      double val = f(val_x);
      surplus += val;
      f_[i] = val;
      g_[i] = val;
      fitness_(i) = val;
      fitness_(i + GenomeLenght_ + 1) = val;            
    };
    
    //for (int i = 0; i < GenomeLenght_ + 1; i++) {
    //  f_[i] *= (target_surplus / surplus);
    //  g_[i] *= (target_surplus / surplus);
    //};
    //fitness_ *= (target_surplus / surplus);
  }  

  //Functor behaviour
  int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
  {     
    //Compute average fittness
    double R = 0.0;
    for (auto i = 0; i <= GenomeLenght_; i++) {
      int ip = i;
      int iq = i + GenomeLenght_ + 1;
      R += f_[i] * x(ip) + g_[i] * x(iq);
      fvec(ip) = 0.0;
      fvec(iq) = 0.0;      
    };

    //Compute source term
    for (int i = 0; i <= GenomeLenght_; i++) {
      int ip = i;
      int iq = i + GenomeLenght_ + 1;
      //Compute mutation process
      if ((i + 1) <= GenomeLenght_) {
        fvec(ip) += x(ip+1) * mutation_rate_P_ * (i + 1) / GenomeLenght_;
        fvec(iq) += x(iq+1) * mutation_rate_Q_ * (i + 1) / GenomeLenght_;
      };
      if ((i - 1) >= 0) {
        fvec(ip) += x(ip - 1) * mutation_rate_P_ * (GenomeLenght_ - i + 1) / GenomeLenght_;
        fvec(iq) += x(iq - 1) * mutation_rate_Q_ * (GenomeLenght_ - i + 1) / GenomeLenght_;
      };

      //std::cout << fvec;

      //! Compute mutator gene switching process
      fvec(ip) += x(iq) * mutator_gene_transition_rate_Q_to_P_;
      fvec(iq) += x(ip) * mutator_gene_transition_rate_P_to_Q_;

      //! Compute replication process
      fvec(ip) += x(ip) * (f_[i] - (mutation_rate_P_ + mutator_gene_transition_rate_P_to_Q_));
      fvec(iq) += x(iq) * (g_[i] - (mutation_rate_Q_ + mutator_gene_transition_rate_Q_to_P_));

      //! Compute selection and normalization term
      fvec(ip) -= x(ip) * R;
      fvec(iq) -= x(iq) * R;

      //std::cout << fvec;
    };

    // Boundary conditions
    for (auto i = 0; i <= GenomeLenght_; i++) {
      int ip = i;
      int iq = i + GenomeLenght_ + 1;            
      if (x(ip) < 0) fvec(ip) = x(ip)*x(ip);
      if (x(ip) > 1.0) fvec(ip) = (x(ip)-1.0)*(x(ip)-1.0);
      if (x(iq) < 0) fvec(iq) = x(iq)*x(iq);
      if (x(iq) > 1.0) fvec(iq) = (x(iq) - 1.0)*(x(iq) - 1.0);
    };

    //std::cout << fvec.sum() << std::endl;
    return 0;
  } // operator()

  //API
  template <typename Derived>
  Eigen::VectorXd source_term(const Eigen::MatrixBase<Derived>& x) const {
    Eigen::VectorXd source(x.rows());
    this->operator()(x, source);
    return std::move(source);
  };

  Eigen::VectorXd fitness() const {
    return fitness_;
  };

};

// Equation specification and jacobian
// Implementation of MatrixReplacement * Eigen::DenseVector though a specialization of internal::generic_product_impl:
namespace Eigen {
namespace internal {
template<typename Rhs, size_t N>
struct generic_product_impl<MutatorEquation<N>, Rhs, SparseShape, DenseShape, GemvProduct> // GEMV stands for matrix-vector
  : generic_product_impl_base<MutatorEquation<N>, Rhs, generic_product_impl<MutatorEquation<N>, Rhs> >
{
  typedef typename Product<MutatorEquation<N>, Rhs>::Scalar Scalar;
  template<typename Dest>
  static void scaleAndAddTo(Dest& dst, const MutatorEquation<N>& lhs, const Rhs& rhs, const Scalar& alpha)
  {
    // This method should implement "dst += alpha * lhs * rhs" inplace,
    // however, for iterative solvers, alpha is always equal to 1, so let's not bother about it.
    assert(alpha == Scalar(1) && "scaling is not implemented");
    // Here we could simply call dst.noalias() += lhs.my_matrix() * rhs,
    // but let's do something fancier (and less efficient):
    //for (Index i = 0; i<lhs.cols(); ++i)
    //  dst += rhs(i) * lhs.my_matrix().col(i);
    dst = lhs.source_term(rhs);
  }
};
}
}



#endif