"""Two pressure-open ends for the existing 2D MAC predictor/projection.

The open-end convention matches AFSEI's vector-Laplacian open-box operator:
prescribed pressure and homogeneous normal velocity derivative. This is the
natural pseudo-traction (mu grad(u)-p I)n, NOT a claim of exact symmetric
Cauchy-traction discretization. Top/bottom outer box boundaries remain no-slip.
Pressure data belong to the physical momentum boundary condition; the pressure
Poisson equation consequently has Dirichlet data at both ends, not Neumann data.
"""
from math import isfinite
import torch
from .flow import ChannelFlow, ChannelResult
from .grid import divergence
from .multigrid import ChannelMultigrid
from ..mac.execution import tensor_kernel


def pressure_gradient(p, spacing, inlet=0., outlet=0.):
    hx, hy = spacing
    gx = torch.cat((2*(p[:1]-inlet)/hx, torch.diff(p, dim=0)/hx,
                    2*(outlet-p[-1:])/hx), dim=0)
    gy = torch.cat((torch.zeros_like(p[:, :1]), torch.diff(p, dim=1)/hy,
                    torch.zeros_like(p[:, :1])), dim=1)
    return gx, gy


def wall_boundary(velocity):
    u, v = velocity
    v = torch.cat((torch.zeros_like(v[:, :1]), v[:, 1:-1], torch.zeros_like(v[:, :1])), 1)
    return u, v


def open_laplacian(velocity, spacing):
    u, v = velocity
    hx, hy = spacing
    ul, ur = torch.cat((u[1:2], u[:-1]), 0), torch.cat((u[1:], u[-2:-1]), 0)
    ub, ut = torch.cat((-u[:, :1], u[:, :-1]), 1), torch.cat((u[:, 1:], -u[:, -1:]), 1)
    vl, vr = torch.cat((v[:1], v[:-1]), 0), torch.cat((v[1:], v[-1:]), 0)
    vb, vt = torch.cat((-v[:, :1], v[:, :-1]), 1), torch.cat((v[:, 1:], -v[:, -1:]), 1)
    return (ul-2*u+ur)/hx**2+(ub-2*u+ut)/hy**2, (vl-2*v+vr)/hx**2+(vb-2*v+vt)/hy**2


def open_convection(velocity, spacing):
    """Same centered conservative fluxes; half control volumes at open u faces."""
    u, v = velocity
    hx, hy = spacing
    ux = .5*(u[1:]+u[:-1])
    du = torch.cat((2*(ux[:1].square()-u[:1].square())/hx,
                    torch.diff(ux.square(), dim=0)/hx,
                    2*(u[-1:].square()-ux[-1:].square())/hx), 0)
    vp = torch.cat((v[:1], v, v[-1:]), 0)
    ve = .5*(vp[1:]+vp[:-1])
    up = torch.cat((-u[:, :1], u, -u[:, -1:]), 1)
    ue = .5*(up[:, 1:]+up[:, :-1])
    du = du+torch.diff(ve*ue, dim=1)/hy
    vy = .5*(v[:, 1:]+v[:, :-1])
    dv = torch.cat((torch.zeros_like(v[:, :1]), torch.diff(vy.square(), dim=1)/hy,
                    torch.zeros_like(v[:, :1])), 1)+torch.diff(ve*ue, dim=0)/hx
    return du, dv


class PressureChannelFlow(ChannelFlow):
    def __init__(self, grid, *, dt, rho, mu, inlet_pressure, outlet_pressure,
                 device='cpu', fused=True, options=None, pressure_backend='reference'):
        if any(not isfinite(a) or a <= 0 for a in (dt, rho, mu)):
            raise ValueError('positive finite dt/rho/mu required')
        if not all(isfinite(a) for a in (inlet_pressure, outlet_pressure)):
            raise ValueError('finite pressures required')
        self.grid, self.dt, self.rho, self.mu = grid, dt, rho, mu
        self.inlet_pressure, self.outlet_pressure = inlet_pressure, outlet_pressure
        self.viscous_number = dt*mu/rho*sum(h**-2 for h in grid.spacing)
        if self.viscous_number > .25:
            raise ValueError('explicit viscous step too large')
        self.pressure_solver = ChannelMultigrid(grid, device=device, options=options,
            fused=fused, backend=pressure_backend, pressure_inlet=True)
        self.optimized = True
        self._predict = tensor_kernel(self._predict, device) if fused else self._predict
        self._correct = tensor_kernel(self._correct, device) if fused else self._correct

    def _predict(self, velocity, density):
        vel = wall_boundary(velocity)
        adv, diffusion = open_convection(vel, self.grid.spacing), open_laplacian(vel, self.grid.spacing)
        return wall_boundary(tuple(u+self.dt*(-a+self.mu/self.rho*l+f/self.rho)
                                  for u, a, l, f in zip(vel, adv, diffusion, density)))

    def _correct(self, tentative, p):
        grad = pressure_gradient(p, self.grid.spacing, self.inlet_pressure, self.outlet_pressure)
        return tuple(u-self.dt/self.rho*g for u, g in zip(tentative, grad))

    @torch.no_grad()
    def project(self, tentative, initial=None):
        self.grid.check_velocity(tentative)
        rhs = -self.rho/self.dt*divergence(tentative, self.grid.spacing)
        rhs = rhs.clone()
        rhs[0] += 2*self.inlet_pressure/self.grid.spacing[0]**2
        rhs[-1] += 2*self.outlet_pressure/self.grid.spacing[0]**2
        p, info = self.pressure_solver.solve(rhs, initial)
        return ChannelResult(self._correct(tentative, p), p, dict(pressure=info))

    @torch.no_grad()
    def step(self, velocity, density, *, time, pressure_initial=None):
        self.grid.check_velocity(velocity)
        self.grid.check_velocity(density)
        if not isfinite(time) or any(not torch.isfinite(v).all() for v in (*velocity, *density)):
            raise ValueError('finite time, velocity and force required')
        speed = torch.stack(tuple(v.abs().max() for v in velocity))
        cfl = (self.dt*speed/speed.new_tensor(self.grid.spacing)).sum().item()
        centered = (self.dt*speed.square().sum()/(self.mu/self.rho)).item()
        if cfl > .25 or centered > 1:
            raise ValueError('centered explicit advection stability guard exceeded; reduce dt')
        result = self.project(self._predict(velocity, density), pressure_initial)
        return ChannelResult(result.velocity, result.pressure,
            dict(result.diagnostics, courant=cfl, centered_number=centered,
                 viscous_number=self.viscous_number, inlet_time_s=time))

    def power_weights(self, component, like):
        weights = torch.full(self.grid.face_shape(component), self.grid.volume,
                             device=like.device, dtype=like.dtype)
        if component == 0:
            weights[0] *= .5
            weights[-1] *= .5
        return weights
