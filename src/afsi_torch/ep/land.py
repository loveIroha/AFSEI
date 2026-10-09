"""Source-aligned Land quadrature loads and their deformation derivatives.

All scalar states are P1 fields; mechanical fiber directions are DG0.
History is supplied from an accepted step and is read-only in force/JVP calls.
"""
from dataclasses import dataclass
from math import isfinite
import torch
from ..p1 import prepare_p1


def distortion_step(old_z, delta_lambda, dt_s, rate_s, gain=10.0):
    if not isfinite(dt_s) or dt_s <= 0 or not isfinite(rate_s) or rate_s <= 0:
        raise ValueError('positive finite step in seconds and decay rate required')
    step = rate_s*dt_s
    # Stable for small dt; retains autograd through the stretch increment.
    decay = old_z.new_tensor(-step)
    return old_z*decay.exp()+gain*delta_lambda*(-decay.expm1()/step)


def land_tension(stretch, previous_stretch, xs, xw, zs_old, zw_old, dt_s):
    lc = stretch.clamp_max(1.2)
    h = (1+2.3*(lc+lc.clamp_max(0.87)-1.87)).clamp_min(0)
    zs = distortion_step(zs_old, stretch-previous_stretch, dt_s, 40.14)
    zw = distortion_step(zw_old, stretch-previous_stretch, dt_s, 405.86)
    return (h*(1.2e6/0.25)*(xs*(zs+1)+xw*zw)).clamp_min(0)


def active_pk1(F, fiber, tension_cgs):
    """Nominal active PK1 with the source's clipped extra stretch multiplier."""
    Ff = torch.einsum('...ij,...j->...i', F, fiber)
    stretch = Ff.square().sum(-1).clamp_min(1.0).sqrt()
    coefficient = tension_cgs*(1+4.9*(stretch-1))
    return coefficient[..., None, None]*Ff[..., :, None]*fiber[..., None, :]


def land_ode_step(state, calcium_um, stretch, zs, zw, dt_ms):
    """Readable PyTorch reference of the source Euler Land update, not the hot loop."""
    if not isfinite(dt_ms) or dt_ms <= 0:
        raise ValueError('positive finite ODE step in milliseconds required')
    xs, xw, ct, tm = state.clamp_min(0).unbind(-1)
    cat50 = (0.535-2.4*(stretch.clamp_max(1.2)-1)).clamp_min(0.1)
    gammasu = 0.0085*torch.maximum(zs.clamp_min(0), (-1-zs).clamp_min(0))
    gammawu = 0.615*zw.abs()
    xu = (1-tm-xs-xw).clamp_min(0)
    kb = 0.04*0.35**2.4/(1-0.25-0.5*(1-0.25))
    # Safe branch arguments keep zeros from producing inf in eager PyTorch.
    ct_safe = ct.clamp_min(1e-12)
    phalf = torch.where(ct > 1e-12, ct_safe.pow(1.2), 0)
    nhalf = torch.where(ct > 1e-12, ct_safe.pow(-1.2).clamp_max(100), 100)
    derivatives = torch.stack((0.012*xw-xs*gammasu-0.018*xs,
        0.182*xu-0.012*xw-xw*gammawu-0.170*xw,
        0.1*(-ct+(calcium_um/cat50).square()*(1-ct)),
        kb*nhalf*xu-0.04*phalf*tm), -1)
    result = (state+dt_ms*derivatives).clamp_min(0)
    return torch.cat((result[..., :3], result[..., 3:].clamp_max(1)), -1)


@dataclass(frozen=True)
class LandQuadratureLoad:
    """Immutable trial load; explicitly construct a new load after acceptance.

    mode='land' evaluates Ta from current geometry; mode='frozen' applies an
    accepted closure Ta[E,Q] times ramp_factor, preserving spatial variation.
    Closure snapshots must be taken BEFORE publishing strain history.
    """
    geometry: object
    fiber: torch.Tensor
    previous_F: torch.Tensor
    fields: torch.Tensor
    dt_s: float
    mode: str = 'land'
    anchor: torch.Tensor | None = None
    ramp_factor: float = 0.0

    @classmethod
    def prepare(cls, X, cells, fiber, previous_x, xs, xw, zs_old, zw_old, dt_s,
                *, anchor=None, ramp_factor=0.0):
        if not isfinite(dt_s) or dt_s <= 0:
            raise ValueError('positive mechanical dt in seconds required')
        g = prepare_p1(X, cells, degree=5)
        for value, shape in ((fiber, (len(cells), 3)), (previous_x, X.shape),
                             *((v, (len(X),)) for v in (xs, xw, zs_old, zw_old))):
            if (value.shape != shape or value.device != X.device or value.dtype != X.dtype
                    or not torch.isfinite(value).all()):
                raise ValueError('Land fields must be finite with mesh shape/device/dtype')
        if (torch.linalg.vector_norm(fiber, dim=-1) <= 0).any():
            raise ValueError('nonzero mechanical fiber required')
        if not isfinite(ramp_factor) or not 0 <= ramp_factor <= 1:
            raise ValueError('closure ramp factor must be in [0,1]')
        if anchor is not None and (anchor.shape != g.weights.shape or anchor.device != X.device
                or anchor.dtype != X.dtype or not torch.isfinite(anchor).all() or (anchor < 0).any()):
            raise ValueError('closure anchor must be a nonnegative quadrature field')
        gradients = g.gradients[:, 0]
        prev_F = torch.einsum('eai,eaJ->eiJ', previous_x[g.cells], gradients)
        nodal = torch.stack((xs, xw, zs_old, zw_old), -1)
        fields = torch.einsum('qa,eak->eqk', g.values, nodal[g.cells])
        return cls(g, fiber.detach().clone(), prev_F.detach().clone(), fields.detach().clone(),
                   dt_s, 'land' if anchor is None else 'frozen',
                   None if anchor is None else anchor.detach().clone(), ramp_factor)

    def _element_tension(self, F, fiber, previous_F, fields, anchor):
        if self.mode == 'frozen':
            return anchor*self.ramp_factor
        stretch = torch.einsum('ij,j->i', F, fiber).square().sum().clamp_min(1e-16).sqrt()
        previous = torch.einsum('ij,j->i', previous_F, fiber).square().sum().clamp_min(1e-16).sqrt()
        xs, xw, zs, zw = fields.unbind(-1)
        return land_tension(stretch, previous, xs, xw, zs, zw, self.dt_s)

    def _element_force(self, nodes, gradients, weights, fiber, previous_F, fields, anchor):
        F = nodes.T @ gradients
        ta = self._element_tension(F, fiber, previous_F, fields, anchor)
        P = active_pk1(F[None], fiber[None], ta)
        return -torch.einsum('q,qiJ,aJ->ai', weights, P, gradients)

    def _args(self, x):
        g = self.geometry
        anchor = torch.zeros_like(g.weights) if self.anchor is None else self.anchor
        return (x[g.cells], g.gradients[:, 0], g.weights, self.fiber,
                self.previous_F, self.fields, anchor)

    def tension(self, x):
        nodes, gradients, _, fiber, previous_F, fields, anchor = self._args(x)
        F = torch.einsum('eai,eaJ->eiJ', nodes, gradients)
        return torch.vmap(self._element_tension)(F, fiber, previous_F, fields, anchor)

    def closure_snapshot(self, accepted_x):
        return self.tension(accepted_x).detach().clone()

    def force(self, x):
        """AFSI force convention: minus the weak PK1 divergence."""
        local = torch.vmap(self._element_force)(*self._args(x))
        return torch.zeros_like(x).index_add(0, self.geometry.cells.reshape(-1), local.reshape(-1, 3))

    def force_jvp(self, x, direction):
        return torch.func.jvp(self.force, (x,), (direction,))[1]

    def local_jacobian(self, x, chunk_size=1024):
        """Element Jacobians for addition to the existing sparse solid tangent."""
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError('positive integer chunk size required')
        args = self._args(x)
        kernel = torch.vmap(torch.func.jacrev(self._element_force, argnums=0))
        return torch.cat([kernel(*(v[i:i+chunk_size] for v in args))
                          for i in range(0, len(self.geometry.cells), chunk_size)])
