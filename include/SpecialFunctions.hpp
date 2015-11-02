#ifndef MutatorModel_SpecialFunctions_hpp
#define MutatorModel_SpecialFunctions_hpp

#include <exception>
#include <cmath>

// Generic functor 
template<typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
struct Functor
{
private:
  int m_inputs, m_values;
public:
  using Scalar = _Scalar;
  enum {
    InputsAtCompileTime = NX,
    ValuesAtCompileTime = NY
  };
  using InputType = Eigen::Matrix<Scalar, InputsAtCompileTime, 1>;
  using ValueType = Eigen::Matrix<Scalar, ValuesAtCompileTime, 1>;
  using JacobianType = Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime>;
  
  Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
  Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

  int inputs() const { return m_inputs; }
  int values() const { return m_values; }
};

//Legendre polynomials
template< int _Order = -1 >
struct ShiftedLegendre : Functor< double, 1, 1 > {
  double df(int n, const double& x) {
    if (n == 0) return 0;
    if (n == 1) return 2;
    if (n == 2) return 12 * x - 6;
    if (n == 3) return 60 * std::pow(x, 2) - 60 * x + 12;
    throw std::exception("Unkown order of shifted Legendre polynomial derivative");
    return 0;
  };

  double operator()(int n, const double& x) {
    if (n == 0) return 1;
    if (n == 1) return 2 * x - 1;
    if (n == 2) return 6 * x*x - 6 * x + 1;
    if (n == 3) return 20 * std::pow(x, 3) - 30 * std::pow(x, 2) + 12 * x - 1;
    throw std::exception("Unkown order of shifted Legendre polynomial");
    return 0;
  };
private:
  double operator()(const double& x) {
    return 1.0;
  };
};

template<>
struct ShiftedLegendre<0> : Functor< double, 1, 1 > {
  double df(const double& x) {
    return 0;
  };
  double operator()(const double& x) {
    return 1.0;
  };
};

template<>
struct ShiftedLegendre<1> : Functor< double, 1, 1 > {  
  double df(const double& x) {
    return 2;
  };
  double operator()(const double& x) {    
    return 2*x - 1.0;
  };
};

template<>
struct ShiftedLegendre<2> : Functor< double, 1, 1 >{
  double df(const double& x) {
    return 12*x - 6;
  };

  double operator()(const double& x) {    
    return 6*x*x -6*x + 1;
  };
};

double shifted_legendre(int n, double x) {
  if (n == 0) return 1;
  if (n == 1) return 2 * x - 1;
  if (n == 2) return 6 * x*x - 6 * x + 1;
  if (n == 3) return 20 * std::pow(x, 3) - 30 * std::pow(x, 2) + 12 * x - 1;
  throw std::exception("Unkown order of shifted Legendre polynomial");
  return 0;
};

double shifted_legendre_derivative(int n, double x) {
  if (n == 0) return 0;
  if (n == 1) return 2;
  if (n == 2) return 12 * x - 6;
  if (n == 3) return 60 * std::pow(x, 2) - 60 * x + 12;
  throw std::exception("Unkown order of shifted Legendre polynomial derivative");
  return 0;
};

#endif