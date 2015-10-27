#include <array>

namespace numerics_internal {
  struct NumericQuadrature_base {

  };
};


//! Compute Gauss integral
template< int Order = 0>
struct NumericQuadrature {
private:
  //Should never be compiled
  template<class F>
  double operator()(F f, double a, double b) {
    return (b - a) * f((a + b) / 2);
  };
};

template<>
struct NumericQuadrature< 1 > {
private:
  template<class F>
  double operator()(F f, double a, double b) {
    return (b - a) * f((a + b) / 2);
  };
};

template<>
struct NumericQuadrature< 2 > {
  std::array<double, 2> e{ -0.5773502691896257645091488, 0.5773502691896257645091488 };
  std::array<double, 2> w{ 1.0, 1.0 };

  template<class F>
  double operator()(F f, double a, double b) {
    double sum = 0;
    double A = (b - a) / 2.0;
    double mid = (b + a) / 2.0;
    for (size_t i = 0; i < 2; i++) sum += w[i] * f(mid + e[i] * A);
    return A * sum;
  };
};

template<>
struct NumericQuadrature< 3 > {
  std::array<double, 3> e{ -0.7745966692414833770358531,
    0.0,
    0.7745966692414833770358531 };
  std::array<double, 3> w{ 0.5555555555555555555555556, 
    0.8888888888888888888888889, 
    0.5555555555555555555555556 };

  template<class F>
  double operator()(F f, double a, double b) {
    double sum = 0;
    double A = (b - a) / 2.0;
    double mid = (b + a) / 2.0;
    for (size_t i = 0; i < 3; i++) sum += w[i] * f(mid + e[i] * A);
    return A * sum;
  };
};

template<>
struct NumericQuadrature< 4 > {
  std::array<double, 4> e{ -0.8611363115940526,
    -0.3399810435848563,
    0.3399810435848563,
    0.8611363115940526 };
  std::array<double, 4> w{ 0.3478548451374538,
    0.6521451548625461,
    0.6521451548625461,
    0.3478548451374538 };

  template<class F>
  double operator()(F f, double a, double b) {
    double sum = 0;
    double A = (b - a) / 2.0;
    double mid = (b + a) / 2.0;
    for (size_t i = 0; i < 4; i++) sum += w[i] * f(mid + e[i] * A);
    return A * sum;
  };
};