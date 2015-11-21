#ifndef MutatorModel_SpecialFunctions_hpp
#define MutatorModel_SpecialFunctions_hpp

#include <exception>
#include <cmath>

#include <Concepts\VectorFunction.hpp>

using namespace concepts;

//Legendre polynomials
template< int _Order = -1 >
struct ShiftedLegendre : VectorFunction< double, 1, 1 > {
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
struct ShiftedLegendre<0> : VectorFunction< double, 1, 1 > {
  double df(const double& x) {
    return 0;
  };
  double operator()(const double& x) {
    return 1.0;
  };
};

template<>
struct ShiftedLegendre<1> : VectorFunction< double, 1, 1 > {
  double df(const double& x) {
    return 2;
  };
  double operator()(const double& x) {    
    return 2*x - 1.0;
  };
};

template<>
struct ShiftedLegendre<2> : VectorFunction< double > {
  //! Constructor
  ShiftedLegendre() : VectorFunction< double >(1, 1) {};

  double df(const double& x) {
    return 12*x - 6;
  };

  double operator()(const double& x) {    
    return 6*x*x - 6*x + 1;
  };

  int df(const InputType &x, JacobianType& fjac) {
    fjac(0,0) = df(x(0));
    return 0;
  };

  int operator()(const InputType &x, ValueType& fvec) {
    fvec(0) = this->operator()(x(0));
    return 0;
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