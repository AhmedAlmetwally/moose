#include "EGCGDiffusionAGA.h"

// MOOSE includes
#include "MooseVariableFE.h"

#include "libmesh/utility.h"

registerMooseObject("MooseApp", EGCGDiffusionAGA);

InputParameters
EGCGDiffusionAGA::validParams()
{
  InputParameters params = DGKernel::validParams();
  params.addClassDescription("Computes residual contribution for the diffusion operator using "
                             "discontinous Galerkin method.");
  // See header file for sigma and epsilon
  params.addParam<MaterialPropertyName>("sigma", "sigma");
  params.addRequiredParam<Real>("epsilon", "epsilon");
  params.addParam<MaterialPropertyName>(
      "diff", 1., "The diffusion (or thermal conductivity or viscosity) coefficient.");
  params.addRequiredCoupledVar("v", "The governing variable that controls diffusion of u.");
  return params;
}

EGCGDiffusionAGA::EGCGDiffusionAGA(const InputParameters & parameters)
  : DGKernel(parameters),
    _epsilon(getParam<Real>("epsilon")),
    _sigma(getMaterialProperty<Real>("sigma")),
    _sigma_neighbor(getNeighborMaterialProperty<Real>("sigma")),
    _diff(getMaterialProperty<Real>("diff")),
    _diff_neighbor(getNeighborMaterialProperty<Real>("diff")),
    _v_var(dynamic_cast<MooseVariable &>(*getVar("v", 0))),
    _v(coupledValue("v")),
    _v_neighbor(coupledNeighborValue("v")),
    _grad_v(coupledGradient("v")),
    _grad_v_neighbor(coupledNeighborGradient("v")),
    _v_id(coupled("v"))
{
}

Real
EGCGDiffusionAGA::computeQpResidual(Moose::DGResidualType type)
{
  Real r = 0;

  const unsigned int elem_b_order = _var.order();
  double h_elem =
      _current_elem_volume / _current_side_volume * 1. / Utility::pow<2>(elem_b_order);

  if (elem_b_order == 0)
  {
      h_elem = _current_elem->hmin();;
  }

  switch (type)
  {
    case Moose::Element:
      r += _epsilon * 0.5 * (_v[_qp] - _v_neighbor[_qp]) * _diff[_qp] * _grad_test[_i][_qp] * _normals[_qp];
      // std::cout << "Moose::Element _grad_test[_i][_qp] "<<_grad_test[_i][_qp]<<"  _grad_test_neighbor[_i][_qp]"<<_grad_test_neighbor[_i][_qp] <<" _v[_qp] - _v_neighbor[_qp] "<<_v[_qp] - _v_neighbor[_qp]<<std::endl;
      break;

    case Moose::Neighbor:
      r += _epsilon * 0.5 * (_v[_qp] - _v_neighbor[_qp]) * _diff_neighbor[_qp] * _grad_test_neighbor[_i][_qp] * _normals[_qp];
       // std::cout << "Moose::Neighbor _grad_test[_i][_qp]"<< _grad_test[_i][_qp]<<"_grad_test_neighbor[_i][_qp]"<<_grad_test_neighbor[_i][_qp]<<" _v[_qp] - _v_neighbor[_qp]"<<_v[_qp] - _v_neighbor[_qp]<<std::endl;
      break;
  }

  return r;
}
Real EGCGDiffusionAGA::computeQpJacobian(Moose::DGJacobianType /*type*/) { return 0; }
Real
EGCGDiffusionAGA::computeQpOffDiagJacobian(Moose::DGJacobianType type, unsigned int jvar)
{
  Real r = 0;

  const unsigned int elem_b_order = _var.order();
  double h_elem =
      _current_elem_volume / _current_side_volume * 1. / Utility::pow<2>(elem_b_order);

  if (elem_b_order == 0)
  {
      h_elem = _current_elem->hmin();;
  }

  if (jvar == _v_id)
  {
    switch (type)
    {
      case Moose::ElementElement:
        r += _epsilon * 0.5 * _phi[_j][_qp] * _diff[_qp] * _grad_test[_i][_qp] * _normals[_qp];
        break;

      case Moose::ElementNeighbor:
         r += _epsilon * 0.5 * -_phi_neighbor[_j][_qp] * _diff[_qp] * _grad_test[_i][_qp] * _normals[_qp];
        break;

      case Moose::NeighborElement:
        r += _epsilon * 0.5 * _phi[_j][_qp] * _diff_neighbor[_qp] * _grad_test_neighbor[_i][_qp] * _normals[_qp];
        break;

      case Moose::NeighborNeighbor:
        r += _epsilon * 0.5 * -_phi_neighbor[_j][_qp] * _diff_neighbor[_qp] * _grad_test_neighbor[_i][_qp] * _normals[_qp];
        break;
    }
  }

  return r;
}
