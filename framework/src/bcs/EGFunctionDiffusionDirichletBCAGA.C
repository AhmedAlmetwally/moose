#include "EGFunctionDiffusionDirichletBCAGA.h"

// MOOSE includes
#include "Function.h"
#include "MooseVariableFE.h"

#include "libmesh/numeric_vector.h"
#include "libmesh/utility.h"

// C++ includes
#include <cmath>

registerMooseObject("MooseApp", EGFunctionDiffusionDirichletBCAGA);

InputParameters
EGFunctionDiffusionDirichletBCAGA::validParams()
{
  InputParameters params = IntegratedBC::validParams();
  params.addClassDescription(
      "Diffusion Dirichlet boundary condition for discontinuous Gelerkin method.");
  params.addParam<Real>("value", 0.0, "The value the variable should have on the boundary");
  params.addRequiredParam<FunctionName>("function", "The forcing function.");
  params.addRequiredParam<Real>("epsilon", "Epsilon");
  params.addParam<MaterialPropertyName>("sigma", "Sigma");
  params.addParam<MaterialPropertyName>(
      "diff", 1, "The diffusion (or thermal conductivity or viscosity) coefficient.");
  params.addRequiredCoupledVar("v", "The governing variable that controls diffusion of u.");
  return params;
}

EGFunctionDiffusionDirichletBCAGA::EGFunctionDiffusionDirichletBCAGA(const InputParameters & parameters)
  : IntegratedBC(parameters),
    _func(getFunction("function")),
    _epsilon(getParam<Real>("epsilon")),
    _sigma(getMaterialProperty<Real>("sigma")),
    _diff(getMaterialProperty<Real>("diff")),
    _v_var(dynamic_cast<MooseVariable &>(*getVar("v", 0))),
    _v(coupledValue("v")),
    _grad_v(coupledGradient("v")),
    _v_id(coupled("v"))
{
}

Real
EGFunctionDiffusionDirichletBCAGA::computeQpResidual()
{
  const unsigned int elem_b_order = _var.order();
  double h_elem =
      _current_elem_volume / _current_side_volume * 1. / Utility::pow<2>(elem_b_order);

  if (elem_b_order == 0)
  {
      h_elem = _current_elem->hmin();;
  }

  Real fn = _func.value(_t, _q_point[_qp]);
  Real r = 0;
  r -= (_diff[_qp] * (_grad_v[_qp]+ _grad_u[_qp])  * _normals[_qp] * _test[_i][_qp]);
  r += _epsilon * ((_v[_qp]+_u[_qp]) - fn) * _diff[_qp] * _grad_test[_i][_qp] * _normals[_qp];
  r += _sigma[_qp] *_diff[_qp] / h_elem * ((_v[_qp]+_u[_qp]) - fn) * _test[_i][_qp];

  return r;
}

Real EGFunctionDiffusionDirichletBCAGA::computeQpJacobian()
{
  const unsigned int elem_b_order = _var.order();
  double h_elem =
      _current_elem_volume / _current_side_volume * 1. / Utility::pow<2>(elem_b_order);

  if (elem_b_order == 0)
  {
      h_elem = _current_elem->hmin();;
  }

  Real r = 0;
    r -= (_diff[_qp] * _grad_phi[_j][_qp] * _normals[_qp] * _test[_i][_qp]);
    r += _epsilon * _phi[_j][_qp] * _diff[_qp] * _grad_test[_i][_qp] * _normals[_qp];
    r += _sigma[_qp] *_diff[_qp] / h_elem * _phi[_j][_qp] * _test[_i][_qp];

  return r;
}
Real
EGFunctionDiffusionDirichletBCAGA::computeQpOffDiagJacobian(unsigned int jvar)
{
  const unsigned int elem_b_order = _var.order();
  double h_elem =
      _current_elem_volume / _current_side_volume * 1. / Utility::pow<2>(elem_b_order);

  if (elem_b_order == 0)
  {
      h_elem = _current_elem->hmin();;
  }

  Real r = 0;
  if (jvar == _v_id)
  {
    r -= (_diff[_qp] * _grad_phi[_j][_qp] * _normals[_qp] * _test[_i][_qp]);
    r += _epsilon * _phi[_j][_qp] * _diff[_qp] * _grad_test[_i][_qp] * _normals[_qp];
    r += _sigma[_qp] *_diff[_qp] / h_elem * _phi[_j][_qp] * _test[_i][_qp];
  }
  return r;
}
