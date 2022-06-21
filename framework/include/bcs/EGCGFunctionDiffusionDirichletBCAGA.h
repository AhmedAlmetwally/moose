#pragma once

#include "IntegratedBC.h"

// Forward Declarations
class EGCGFunctionDiffusionDirichletBCAGA;

/**
 * Implements a simple BC for DG
 *
 * BC derived from diffusion problem that can handle:
 * \f$ { \nabla u \cdot n_e} [v] + \epsilon { \nabla v \cdot n_e } [u] + (\frac{\sigma}{|e|} \cdot
 * [u][v]) \f$
 *
 * \f$ [a] = [ a_1 - a_2 ] \f$
 * \f$ {a} = 0.5 * (a_1 + a_2) \f$
 */
class EGCGFunctionDiffusionDirichletBCAGA : public IntegratedBC
{
public:
  /**
   * Factory constructor, takes parameters so that all derived classes can be built using the same
   * constructor.
   */
  static InputParameters validParams();

  EGCGFunctionDiffusionDirichletBCAGA(const InputParameters & parameters);

protected:
  virtual Real computeQpResidual() override;
  virtual Real computeQpJacobian() override;
  virtual Real computeQpOffDiagJacobian(unsigned int jvar) override;

private:
  const Function & _func;

  Real _epsilon;
  const MaterialProperty<Real> & _sigma;
  const MaterialProperty<Real> & _diff;
  MooseVariable & _v_var;
  const VariableValue & _v;
  const VariableGradient & _grad_v;
  unsigned int _v_id;
};
