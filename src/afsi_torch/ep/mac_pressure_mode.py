"""Experimental work-adjoint pressure-mode transfer on the original MAC box.

This changes a numerical coupling operator, never the source PK1/active load.
It is explicit, isolated and unavailable to ordinary run/checkpoint drivers.
The gradient target comes from a unit CLOSED boundary pressure, not a fitted
pressure offset or the final preload's residual force.
"""
from dataclasses import dataclass
from time import perf_counter
import torch
from .mac_preload import MACEndocardialPressure
from .mac_cap_coupled import closed_rows
from .mac_mass_budget import ClosedCavityMassAudit
from .mac_pressure_compatibility import surface_pressure_density, weighted_norm
from .windkessel import MMHG_TO_DYN_CM2 as C
from ..mac.grid import divergence
from ..mac.open_boundary import gradient, face_weights, negative_laplacian


@dataclass(frozen=True)
class PressureModeStencil:
    original: object
    mode: torch.Tensor
    dual: torch.Tensor
    correction: tuple
    gradient: tuple
    target: tuple
    potential: torch.Tensor
    force_potential: torch.Tensor
    rows: object
    report: dict


class PressureModeTransfer:
    """T=S+b a^T, I=J+a b^T W; identical frozen geometry both ways.

    t=-F_source_endo(1mmHg), a=t/(t:t), b=H G q+Bcap^T*1mmHg/W-S t.
    The compact pressure q makes H G q a discrete gradient. Thus the unit
    closed mode has an exact original-BE response u=0,p=q. T/I are adjoints.
    Other force directions orthogonal to a are unchanged. This is one global
    mode correction; it does not establish pointwise wall impermeability.
    """
    def __init__(self, endpoint, *, level=2):
        if type(level) is not int or not 0 <= level <= 4:
            raise ValueError('pressure-mode surface level in 0..4 required')
        self.base = endpoint.transfer
        if isinstance(self.base, PressureModeTransfer):
            raise ValueError('pressure-mode transfer cannot be wrapped twice')
        self.endpoint, self.level = endpoint, level

    def __getattr__(self, name):
        return getattr(self.base, name)

    @torch.no_grad()
    def prepare(self, x):
        if x.is_cuda: torch.cuda.synchronize(x.device)
        started = perf_counter()
        d = self.endpoint
        original = self.base.prepare(x)
        rows = closed_rows(d, x)
        t = -MACEndocardialPressure(d.model).force(x, 1.)
        if not torch.isfinite(t).all() or t.square().sum() <= 0:
            raise ValueError('finite nonzero source pressure mode required')
        dual = t/t.square().sum()
        audit = ClosedCavityMassAudit(x, d.model.endo_faces, d.grid)
        wall, cap = audit.triangles(x)
        fw = surface_pressure_density(audit, wall, 1., level=self.level, kernel='bs4-bs3')
        fc = surface_pressure_density(audit, cap, 1., level=self.level+2, kernel='bs4-bs3')
        density = tuple(a+b for a, b in zip(fw, fc))
        # Only the known hydrostatic CLOSED surface mode is projected. General
        # material/active forces still use the full original fluid response.
        q, info = d.flow.pressure_solver.solve(-divergence(density, d.grid.spacing))
        # The exact regularized cavity pressure has compact support. Remove
        # quadrature-induced harmonic tails OUTSIDE that geometric support,
        # so velocity-Neumann / pressure-Dirichlet boundary commutators cannot
        # enter the hydrostatic mode. This is explicit and reported, never a
        # fitted pressure shift/normalization or a reference-force subtraction.
        bounds = x[torch.unique(d.model.endo_faces)]
        low, high = bounds.amin(0), bounds.amax(0)
        support = torch.ones_like(q, dtype=torch.bool)
        for axis in range(3):
            coord = (d.grid.origin[axis]+(.5+torch.arange(d.grid.shape[axis],
                     device=x.device, dtype=x.dtype))*d.grid.spacing[axis])
            shape = [1]*3; shape[axis] = len(coord)
            keep = (coord >= low[axis]-2*d.grid.spacing[axis]) & (coord <= high[axis]+2*d.grid.spacing[axis])
            support &= keep.reshape(shape)
        removed = q.masked_fill(support, 0).abs().max().item()
        q = q.masked_fill(~support, 0)
        if any(q.select(axis, side).abs().max() > 0
               for axis in range(3) for side in (0, 1, q.shape[axis]-2, q.shape[axis]-1)):
            raise ValueError('compact hydrostatic pressure reaches outer boundary; enlarge MAC box')
        # phi=H_pressure*q; compact support gives G phi=H_velocity G q.
        phi = q+d.flow.dt*d.flow.mu/d.flow.rho*negative_laplacian(q, d.grid.spacing)
        g = gradient(phi, d.grid.spacing)
        lift = rows.lift(torch.full_like(rows.area, C))
        self.base.reset_warm_start()
        st, _ = self.base.spread(t, original)
        correction = tuple(a+b-c for a, b, c in zip(g, lift, st))
        # Rigid-field consistency is essential: a pressure fix must not
        # silently change linear/angular momentum or rigid interpolation.
        net = torch.stack([(b*face_weights(d.grid, c, b)).sum()
                           for c, b in enumerate(correction)])
        torque = x.new_zeros(3)
        center = x.mean(0)
        for c, b in enumerate(correction):
            r = d.grid.coordinates(c, device=x.device, dtype=x.dtype)-center
            load = torch.zeros_like(r); load[..., c] = b*face_weights(d.grid, c, b)
            torque += torch.linalg.cross(r, load).sum((0, 1, 2))
        residual = weighted_norm(d.grid, tuple(a-b for a, b in zip(density, gradient(q, d.grid.spacing))))
        if x.is_cuda: torch.cuda.synchronize(x.device)
        report = dict(setup_seconds=perf_counter()-started, surface_level=self.level,
            hydrostatic_hodge_residual=info,
            discarded_unit_surface_non_gradient_weighted_l2=residual.item(),
            correction_weighted_l2=weighted_norm(d.grid, correction).item(),
            correction_net_force_dyn=net.tolist(), correction_torque_dyn_cm=torque.tolist(),
            correction_net_force_norm_dyn=net.norm().item(), correction_torque_norm_dyn_cm=torque.norm().item(),
            removed_quadrature_tail_max_mmhg=removed/C,
            compact_pressure_support='endo bounds plus two MAC cells',
            hydrostatic_response_evaluated_by_exact_BE_superposition=True,
            solid_force_or_preload_residual_used_to_build_mode=False,
            pressure_field_normalized_or_shifted=False,
            extra_pressure_solves_during_residual=0, corrected_global_modes=1)
        self.base.reset_warm_start()
        target = tuple(a+b for a, b in zip(g, lift))
        return PressureModeStencil(original, t, dual, correction, g, target, q, phi, rows, report)

    def reset_warm_start(self):
        self.base.reset_warm_start()

    @torch.no_grad()
    def spread(self, force, stencil):
        amplitude = (stencil.dual*force).sum()
        # Algebraically identical to S f+b(a:f), but avoid cancellation
        # between two separately computed large FE-mass/IB responses.
        density, info = self.base.spread(force-amplitude*stencil.mode, stencil.original)
        return tuple(f+amplitude*b for f, b in zip(density, stencil.target)), info

    @torch.no_grad()
    def relative_density(self, force, cap_pressure, stencil):
        """T force-B^T cap_pressure/W with the pressure cancellation explicit."""
        amplitude = (stencil.dual*force).sum()
        remainder, info = self.base.spread(force-amplitude*stencil.mode, stencil.original)
        cap = stencil.rows.lift(amplitude*C-cap_pressure)
        return tuple(f+amplitude*g+c for f, g, c in zip(remainder, stencil.gradient, cap)), info

    @torch.no_grad()
    def relative_parts(self, force, cap_pressure, stencil):
        """Return remainder and analytic pressure amplitude, for exact BE addition."""
        amplitude = (stencil.dual*force).sum()
        remainder, info = self.base.spread(force-amplitude*stencil.mode, stencil.original)
        cap = stencil.rows.lift(amplitude*C-cap_pressure)
        return tuple(f+c for f, c in zip(remainder, cap)), amplitude, info

    @torch.no_grad()
    def interpolate(self, velocity, stencil):
        nodal, info = self.base.interpolate(velocity, stencil.original)
        power = sum((u*b*face_weights(self.grid, c, b)).sum()
                    for c, (u, b) in enumerate(zip(velocity, stencil.target)))
        power -= (stencil.mode*nodal).sum()
        return nodal+stencil.dual*power, info
