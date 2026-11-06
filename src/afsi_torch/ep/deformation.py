"""Santiago 2018 Eq.14: spatial conductivity from material conductivity.

F[e,i,I] = dx_i/dX_I. G may be per cell or per quadrature point.
The equivalent reference weak diffusion is J F^-1 Gd F^-T = G.
Cm and Sv are kept constant; current-volume mass/source weights contain J.
No stretch-activated ionic current is introduced.
"""
import torch


def push_forward_conductivity(F,G):
    J=torch.linalg.det(F)
    if not torch.isfinite(F).all() or not torch.isfinite(G).all() or (J<=0).any():
        raise ValueError('finite orientation-preserving deformation required')
    if G.ndim==4:
        return (F[:,None]@G@F.transpose(-1,-2)[:,None])/J[:,None,None,None]
    return (F@G@F.transpose(-1,-2))/J[:,None,None]
