"""Physical MAC cut-plane pressure data, outside the immersed ventricular wall.

Uses the existing BE viscous predictor and masked MAC pressure projection.
Pressure values are prescribed at physical half-cell box faces, never as an
internal IB pressure jump. This retains the original projection boundary
convention (pressure Dirichlet / predictor velocity Neumann); it is not an
unsplit symmetric-stress traction discretization.
"""
from dataclasses import dataclass
import torch
from .sealed_pressure import gradient, divergence
from ..ep.windkessel import flow_driven_trial, MMHG_TO_DYN_CM2 as C


@dataclass(frozen=True)
class CutPlane:
    axis: int
    side: int
    aperture: torch.Tensor
    name: str

    def validate(self, grid):
        if self.axis not in (0, 1, 2) or self.side not in (0, -1):
            raise ValueError('cut plane requires an axis and a physical low/high face')
        if self.aperture.shape != tuple(n for a, n in enumerate(grid.shape) if a != self.axis):
            raise ValueError('cut-plane aperture shape mismatch')
        if self.aperture.dtype != torch.bool or not self.aperture.any():
            raise ValueError('nonempty boolean aperture required')


class CutPlaneFlow:
    """Affine boundary lift plus an implicit scalar, flow-driven RCR Schur solve.

Only pressure/velocity responses are cached. Old Pc is immutable during all
calls in one mechanical step; callers publish last_circuit after acceptance.
response() is the homogeneous derivative of advance(), including RCR impedance.
"""
    def __getattr__(self, name):
        return getattr(self.base, name)

    @torch.no_grad()
    def __init__(self, base, planes):
        self.base = base
        self.planes = tuple(planes)
        self.domain = base.domain
        self.units = []
        self.mode = 'pressure'
        self.pressures = self.domain.masks[0].new_zeros(len(planes))
        self.outlet = None
        self.last_circuit = None
        self.last_boundary = None
        for plane in self.planes:
            plane.validate(self.grid)
            if not torch.all(self.domain.masks[plane.axis].select(plane.axis, plane.side)[plane.aperture] == 1):
                raise ValueError('a pressure plane must consist of open physical MAC faces')
            lift = list(self.grid.zeros(device=plane.aperture.device))
            sign = -1. if plane.side == 0 else 1.
            lift[plane.axis].select(plane.axis, plane.side).copy_(
                sign * 2 * C / self.grid.spacing[plane.axis] * plane.aperture.to(lift[plane.axis].dtype))
            rhs = divergence(lift, self.grid.spacing)
            p, _ = self.pressure_solver.solve(rhs)
            g = gradient(p, self.grid.spacing, self.domain.masks)
            u = tuple(-self.dt / self.rho * (a + b) for a, b in zip(g, lift))
            self.units.append((u, p))
        self.conductance = (torch.stack([self.flux(u) for u, _ in self.units], 1)
                            if self.units else self.pressures.new_zeros((0,0)))

    def flux(self, velocity):
        values = []
        for plane in self.planes:
            sign = -1. if plane.side == 0 else 1.
            area = self.grid.volume / self.grid.spacing[plane.axis]
            values.append(sign * area * (velocity[plane.axis].select(
                plane.axis, plane.side) * plane.aperture).sum())
        return torch.stack(values) if values else self.pressures.new_zeros((0,))

    def prescribe(self, pressures_mmhg):
        values = self.pressures.new_tensor(pressures_mmhg)
        if values.shape != self.pressures.shape or not torch.isfinite(values).all():
            raise ValueError('one finite pressure per cut plane required')
        self.mode = 'pressure'
        self.pressures = values
        self.outlet = None

    def rcr(self, old_pc_mmhg, parameters, *, outlet=0, other_pressures_mmhg=None):
        if not 0 <= outlet < len(self.planes):
            raise ValueError('RCR outlet index outside cut planes')
        values = [0.] * len(self.planes) if other_pressures_mmhg is None else other_pressures_mmhg
        self.prescribe(values)
        self.mode = 'rcr'
        self.old_pc = old_pc_mmhg
        self.parameters = parameters
        self.outlet = outlet
        trial = flow_driven_trial(old_pc_mmhg, 0., self.dt, parameters)
        self.intercept = trial.outlet_mmhg
        self.impedance = trial.impedance_mmhg_s_per_ml

    def _combine(self, velocity, pressure, values):
        u = tuple(v.clone() for v in velocity)
        p = pressure.clone() if pressure is not None else None
        for value, (unit_u, unit_p) in zip(values, self.units):
            u = tuple(a + value * b for a, b in zip(u, unit_u))
            if p is not None:
                p = p + value * unit_p
        return u, p

    def _boundary_values(self, velocity, *, derivative):
        values = torch.zeros_like(self.pressures) if derivative else self.pressures.clone()
        if self.mode == 'rcr':
            j = self.outlet
            values[j] = 0.
            q0 = self.flux(velocity)[j] + self.conductance[j] @ values
            intercept = 0. if derivative else self.intercept
            values[j] = (intercept + self.impedance * q0) / (
                1. - self.impedance * self.conductance[j, j])
        return values

    @torch.no_grad()
    def advance(self, departure, density, pressure_initial=None):
        u0, p0, info = self.base.advance(departure, density, pressure_initial)
        values = self._boundary_values(u0, derivative=False)
        u, p = self._combine(u0, p0, values)
        q = self.flux(u)
        circuit = None
        if self.mode == 'rcr':
            circuit = flow_driven_trial(self.old_pc, q[self.outlet], self.dt, self.parameters)
        self.last_circuit = circuit
        self.last_boundary = dict(pressures_mmhg=values, outward_flux_ml_s=q,
            power_into_fluid_erg_s=-C * (values * q).sum(),
            rcr_gap_mmhg=(values[self.outlet] - circuit.outlet_mmhg if circuit else values.new_zeros(())),
            application='physical MAC cut faces; no internal pressure sheet',
            discretization='original BE predictor / Dirichlet-pressure projection')
        info['cut_boundary'] = self.last_boundary
        return u, p, info

    @torch.no_grad()
    def response(self, density):
        u0 = self.base.response(density)
        values = self._boundary_values(u0, derivative=True)
        return self._combine(u0, None, values)[0]

    @torch.no_grad()
    def solve_rhs(self, rhs, pressure_initial=None):
        return self.advance(rhs, tuple(torch.zeros_like(u) for u in rhs), pressure_initial)
