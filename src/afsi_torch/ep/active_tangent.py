"""Exact P1 Land force derivative, with the source's clipped PK1 multiplier.

Only the execution of the active tangent changes. The force, history update,
passive tangent, MAC residual and stopping criteria remain the same.
Inputs use stable trial buffers; no history is published by this kernel.
"""
import torch


# PyTorch releases differ in the selected subgradient at clamp equality.
# Probe once on CPU; this boolean becomes a compile-time constant on GPU.
with torch.enable_grad():
    _probe = torch.tensor(1., dtype=torch.float64, requires_grad=True)
    _CLAMP_EQUALITY_LIVE = bool(torch.autograd.grad(_probe.clamp_min(1.), _probe)[0])
del _probe


def above(value, bound):
    return value >= bound if _CLAMP_EQUALITY_LIVE else value > bound


def below(value, bound):
    return value <= bound if _CLAMP_EQUALITY_LIVE else value < bound


def active_local_jacobian(nodes, gradients, weights, fiber, previous_F,
                          fields, anchor, dt, use_land, ramp):
    """Return [cell, node, component, node, component] force Jacobians.

    With v=Ff and s_a=grad(N_a).f, the element force is -s_a A(v.v) v.
    Its derivative is -s_a s_b [A I + 2(dA/d(v.v)) v outer v].
    A includes the quadrature integral of Ta and the extra active PK1 stretch
    factor. Masks follow the installed PyTorch clamp convention at equality.
    """
    F = torch.einsum('eai,eaJ->eiJ', nodes, gradients)
    v = torch.einsum('eij,ej->ei', F, fiber)
    norm2 = v.square().sum(-1)
    stretch = norm2.clamp_min(1e-16).sqrt()[:, None]
    previous = torch.einsum('eij,ej->ei', previous_F, fiber).square().sum(-1).clamp_min(1e-16).sqrt()[:, None]
    lc = stretch.clamp_max(1.2)
    raw_h = 1+2.3*(lc+lc.clamp_max(.87)-1.87)
    h = raw_h.clamp_min(0)
    dh = 2.3*(1+below(lc, .87).to(nodes.dtype))*below(stretch, 1.2)*above(raw_h, 0)
    xs, xw, zs, zw = fields.unbind(-1)
    a_s, a_w = 40.14*dt, 405.86*dt
    b_s = 10*(-torch.expm1(-a_s)/a_s)
    b_w = 10*(-torch.expm1(-a_w)/a_w)
    zs = zs*torch.exp(-a_s)+b_s*(stretch-previous)
    zw = zw*torch.exp(-a_w)+b_w*(stretch-previous)
    binding = xs*(zs+1)+xw*zw
    raw_ta = 4.8e6*h*binding
    ta = torch.where(use_land > .5, raw_ta.clamp_min(0), anchor*ramp)
    dta = 4.8e6*(dh*binding+h*(xs*b_s+xw*b_w))*above(raw_ta, 0)
    dta = torch.where(use_land > .5, dta, 0.)
    integrated = (weights*ta).sum(-1)
    # The Land stretch and the extra PK1 factor have DIFFERENT cutoffs.
    d_integrated = (weights*dta).sum(-1)*above(norm2, 1e-16)/(2*stretch[:, 0])
    clipped = norm2.clamp_min(1.).sqrt()
    multiplier = 1+4.9*(clipped-1)
    # Multiply the mask LAST so the scalar 4.9 retains the mesh precision.
    # bool_tensor*4.9 first would create a default-dtype (float32) temporary.
    d_multiplier = (4.9/(2*clipped))*above(norm2, 1.)
    coefficient = multiplier*integrated
    derivative = multiplier*d_integrated+d_multiplier*integrated
    matrix = (coefficient[:, None, None]*torch.eye(3, device=nodes.device, dtype=nodes.dtype)
              +2*derivative[:, None, None]*v[:, :, None]*v[:, None, :])
    projected = torch.einsum('eaJ,eJ->ea', gradients, fiber)
    return -torch.einsum('ea,eb,eik->eaibk', projected, projected, matrix)
