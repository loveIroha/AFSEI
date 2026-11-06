"""Unmodified quadrature IB pair with the AFSEI basal port extension.

Spatial pair: upstream AFSI_GPU/Griffith--Luo unified FE weak form.
Time: existing AFSEI BE--BE, frozen accepted interaction geometry.
Ports are an AFSEI extension, not a claim about the 2017 paper's valves.
No pressure-mode correction or imposed cavity-volume identity is applied.
"""
from dataclasses import dataclass
from time import perf_counter
import torch
from .mac_cap_coupled import closed_rows
from .mac_balanced_dynamic import BalancedDynamicProblem
from .mac_compact_cap import add_cap_lift, cap_reaction
from .windkessel import MMHG_TO_DYN_CM2 as C
from ..nonlinear import normalized_linear_action


@dataclass(frozen=True)
class StandardPortStencil:
    original: object
    rows: object
    potential: torch.Tensor
    gradient: tuple
    report: dict


class StandardPortTransfer:
    """Adapter only: S and J are the untouched upstream FE/IB operators."""
    def __init__(self, endpoint, base):
        self.endpoint, self.base = endpoint, base

    def __getattr__(self, name):
        return getattr(self.base, name)

    def prepare(self, x):
        started = perf_counter()
        original = self.base.prepare(x)
        rows = closed_rows(self.endpoint, x)
        # Scalar zero broadcasts in the existing residual, without allocating
        # four redundant grid-sized zero arrays or adding a hydrostatic mode.
        zero = x.new_zeros(())
        return StandardPortStencil(original, rows, zero, (zero,)*3,
                                   dict(setup_seconds=perf_counter()-started))

    def reset_warm_start(self):
        self.base.reset_warm_start()

    def spread(self, force, stencil):
        return self.base.spread(force, stencil.original)

    def interpolate(self, velocity, stencil, *, check_finite=True):
        return self.base.interpolate(velocity, stencil.original)

    def relative_parts(self, force, cap_pressure, stencil, *, rows=None, check_finite=True):
        density, info = self.spread(force, stencil)
        rows = stencil.rows if rows is None else rows
        return (add_cap_lift(rows, -cap_pressure, density, check_finite=check_finite),
                force.new_zeros(()), info)


class StandardPortProblem(BalancedDynamicProblem):
    """Same port laws and BE mechanics, without modifying the IB pair."""
    def pressure_mode(self, x):
        return self.stencil

    def linearization(self, z, *, reuse_tangent=False):
        self.accepted(z)
        d, rows = self.driver, self.rows
        x = self.unpack(z)[1]
        if d.assembler is None:
            d.assembler = d.model.tangent_factory(d.owner.balanced_tangent_chunk_size)
        reusable=(reuse_tangent and self._tangent is not None and
                  self._tangent_open==(self.last_open,self.last_inlet_open) and
                  (x-self._tangent_x).abs().max().item()<.05*min(d.grid.spacing))
        if not reusable:
            self._tangent = d.assembler.assemble(x, self.time)
            self._tangent_x = x.clone()
            self._tangent_open = (self.last_open,self.last_inlet_open)
            self.tangent_assemblies += 1
        tangent = self._tangent
        self.linear_open = self.last_open
        self.linear_inlet_open = self.last_inlet_open

        def action(v):
            self.jacobian_actions += 1
            dy = v[:self.mechanical_size].reshape_as(x)
            dbp = self.cap_scale*v[self.mechanical_size:]
            df = torch.sparse.mm(tangent, dy.reshape(-1, 1)).reshape_as(x)
            if self.control.wall_pressure_mmhg:
                df=df+self.load.action(x,self.control.wall_pressure_mmhg,dy)
            df = df+cap_reaction(rows, dbp, check_finite=False)
            d.transfer.reset_warm_start()
            density, _, _ = d.transfer.relative_parts(df, dbp, self.stencil, rows=rows,
                                                       check_finite=False)
            du = d.flow.response(density)
            dw, _ = d.transfer.interpolate(du, self.stencil)
            dq = rows.flow(du)-rows.material_flow(dw)
            lower = dq/rows.area
            if self.control.mode == 'rcr' and self.linear_open:
                lower[-1] = self.impedance*dq[-1]-dbp[-1]/C
            elif self.control.mode == 'filling' and self.linear_inlet_open:
                lower[-1] = -dbp[-1]/C
            return torch.cat(((dy-d.flow.dt*dw).flatten(), self.cap_row_units(lower)))
        return normalized_linear_action(action)
