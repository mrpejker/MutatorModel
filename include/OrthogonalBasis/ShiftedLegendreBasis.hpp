#ifndef MutatorModel_ShiftedLegendreBasis_hpp
#define MutatorModel_ShiftedLegendreBasis_hpp

#include <exception>
#include <cmath>

#include <Concepts\VectorFunction.hpp>

using namespace concepts;

namespace impl {
  struct ShiftedLegendreBase {
    virtual double df(const double& x) = 0;
    virtual double operator()(const double& x) = 0;
  };
} //namespace internal

//Legendre polynomials
template< int _Order = -1 >
struct ShiftedLegendre {
private:
  std::vector<double> w
public:
  ShiftedLegendre(int order) {
  };
};

template<>
struct ShiftedLegendre<0> {
  double df(const double& x) { return 0; };
  double operator()(const double& x) { return 1.0; };
};

template<>
struct ShiftedLegendre<1> {
  double df(const double& x) { return 2; };
  double operator()(const double& x) { return 2 * x - 1.0; };
};

template<>
struct ShiftedLegendre<2> {
  double df(const double& x) { return 12 * x - 6; };
  double operator()(const double& x) { return 6 * x*x - 6 * x + 1; };
};

//Shifted Legendre polynomials basis
template< int _Size >
struct ShiftedLegendreBasis {
  //! Get number of functions in basis
  inline int size() { return _Size; };

  //! Indexing operator that returns basis functions
  std::function<double(double)> operator[](int order) const {
    switch (order) {
      case 0: return static_cast<std::function<double(double)>>(ShiftedLegendre<0>());
      case 1: return static_cast<std::function<double(double)>>(ShiftedLegendre<1>());
      case 2: return static_cast<std::function<double(double)>>(ShiftedLegendre<2>());
      default: {
        throw std::exception{ };
      };
    };
  };

private:
  //Delete explicitly some constructors
  ShiftedLegendreBasis() = delete;
  ShiftedLegendreBasis(const ShiftedLegendreBasis&) = delete;
  ShiftedLegendreBasis(ShiftedLegendreBasis&&) = delete;
public: 

};

//template<int N, int... Rest>
//struct ShiftedLegendreBasis_impl {
//  static constexpr auto& value = ShiftedLegendreBasis_impl<N - 1, N, Rest...>::value;
//};
//
//template<int... Rest>
//struct ShiftedLegendreBasis_impl<0, Rest...> {
//  static constexpr VectorFunction<double,1,1> value[] = { ShiftedLegendre<0>, Rest... };
//};
//
//template<int... Rest>
//constexpr VectorFunction<double,1,1> ShiftedLegendreBasis_impl<0, Rest...>::value[];
//
//template<int N>
//struct ShiftedLegendreBasis {
//  static_assert(N >= 0, "N must be at least 0");
//
//  static constexpr auto& value = ShiftedLegendreBasis_impl<N>::value;
//
//  ShiftedLegendreBasis() = delete;
//  ShiftedLegendreBasis(const ShiftedLegendreBasis&) = delete;
//  ShiftedLegendreBasis(ShiftedLegendreBasis&&) = delete;
//};


#endif