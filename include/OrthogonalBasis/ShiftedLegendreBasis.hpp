#ifndef MutatorModel_ShiftedLegendreBasis_hpp
#define MutatorModel_ShiftedLegendreBasis_hpp

#include <exception>
#include <cmath>

#include <Eigen/Core>

//Legendre polynomials
template< typename T, int _Order = -1 >
struct ShiftedLegendre {
private:
  int order_;
  std::vector<double> coef_; // polynomial coefficients
public:
  ShiftedLegendre(int order) : order_(order) {    
    switch (order) {
    case 0: { coef_ = { 1.0 }; break; }
    case 1: { coef_ = { 2.0, -1.0 }; break; }
    case 2: { coef_ = { 6.0, -6.0, 1.0 }; break; }
    default: {
      throw std::exception("Unsupported order of shifted Legendre polynomial");
    };
    };
  };

  ShiftedLegendre() : ShiftedLegendre(0) {};

  double df(const T& x) const { 
    T result{ 0 };
    for (int i = 0; i < order_; i++) { result = result * x + coef_[i] * (order_ - i); };
    return result;
  };
  
  double operator()(const T& x) const {
    T result{ 0 };
    for (int i = 0; i <= order_; i++) { result = result * x + coef_[i]; };
    return result;   
  };
};

//Shifted Legendre polynomials basis
template< int _Size >
struct ShiftedLegendreBasis {
  //! Get number of functions in basis
  inline int size() { return _Size; };

  //! Basis functions intself
  std::array<ShiftedLegendre<double>, _Size> functions{ };

  //! Constructor
  ShiftedLegendreBasis() {
    for (int i = 0; i < size(); i++) {
      functions[i] = ShiftedLegendre<double>(i);
    };
  };

private:
  //Delete explicitly some constructors
  ShiftedLegendreBasis(const ShiftedLegendreBasis&) = delete;
  ShiftedLegendreBasis(ShiftedLegendreBasis&&) = delete;
};

#endif