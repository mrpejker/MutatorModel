#ifndef MutatorModel_VectorFunction_hpp
#define MutatorModel_VectorFunction_hpp

#include <Eigen/Dense>

using namespace Eigen;

namespace concepts {
 
template <typename _Scalar, int NX = Dynamic, int NY = Dynamic>
struct VectorFunction
{  
  using Scalar = _Scalar;
  enum {
    InputsAtCompileTime = NX,
    ValuesAtCompileTime = NY
  };
  using InputType = Matrix<Scalar, InputsAtCompileTime, 1>;
  using ValueType = Matrix<Scalar, ValuesAtCompileTime, 1>;
  using JacobianType = Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime>;
  using QRSolver = ColPivHouseholderQR<JacobianType>;
  const int m_inputs, m_values;

  VectorFunction() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
  VectorFunction(int inputs, int values) : m_inputs(inputs), m_values(values) {}

  int inputs() const { return m_inputs; }
  int values() const { return m_values; }  

  //int operator()(const InputType &x, ValueType& fvec) { }
  // should be defined in derived classes

  //int df(const InputType &x, JacobianType& fjac) { }
  // should be defined in derived classes
};

} // namespace concepts

#endif