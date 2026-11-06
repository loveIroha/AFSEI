"""Optional IB/FE numerical stabilization, separate from physical E and nu.

Vadala-Roth et al. (2020), doi:10.1016/j.cma.2020.112978, equations
(21),(24),(25): dev(sigma_extra) + kappa*ln(J)/J I. The input extra
stress may include active stress. This is a stress projection, NOT the
Flory energy modification. Incompressibility is still imposed by MAC.
"""
from math import isfinite
import torch
from ..mechanics import determinant3


def validate_volumetric_stabilization(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {'formulation', 'bulk_modulus_dyn_cm2'}:
        raise ValueError('volumetric_stabilization requires formulation and bulk_modulus_dyn_cm2')
    if value['formulation'] != 'deviatoric-logj':
        raise ValueError('volumetric_stabilization formulation must be deviatoric-logj')
    bulk = value['bulk_modulus_dyn_cm2']
    if isinstance(bulk, bool) or not isinstance(bulk, (int, float)) or not isfinite(bulk) or bulk < 0:
        raise ValueError('numerical stabilization bulk modulus must be finite and nonnegative')
    return dict(formulation='deviatoric-logj', bulk_modulus_dyn_cm2=float(bulk))


def logarithmic_volume_energy(F, bulk):
    """Numerical energy U=bulk/2 ln(J)^2, defined only on J>0."""
    return .5*bulk*torch.log(determinant3(F)).square()


def logarithmic_volume_pk1(F, bulk):
    return bulk*torch.log(determinant3(F))[..., None, None]*torch.linalg.inv(F).transpose(-1, -2)


def deviatoric_logj_pk1(F, original_pk1, bulk):
    """Preserve deviatoric Cauchy stress; replace its isotropic component.

    No clamp, hard volume constraint, pressure-BC change or state mutation.
    The existing solid validity checks reject J<=0. The existing automatic
    tangent differentiates BOTH the projection and the volume stress.
    """
    invT = torch.linalg.inv(F).transpose(-1, -2)
    hydro = (original_pk1*F).sum(dim=(-2, -1))/3
    coefficient = bulk*torch.log(determinant3(F))-hydro
    return original_pk1+coefficient[..., None, None]*invT
