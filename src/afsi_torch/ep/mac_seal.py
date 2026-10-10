"""Local weak basal seal projection on the original open-box MAC grid.

This component constrains patch-averaged relative normal velocity, together
with MAC divergence. It is not a coupled solid step or an anatomical valve.
The supplied material velocity is prescribed; the returned reaction must be
included in a future coupled residual, not silently discarded there.
"""
from dataclasses import dataclass
from math import isfinite, prod
import torch
from .mac_ports import _stencil
from ..mac.grid import divergence
from ..mac.open_boundary import gradient, face_weights
from ..mac.dirichlet_mg import DirichletMultigrid
from ..mac.spectral_dirichlet import SpectralDirichletPressure
from ..fluid.solvers import pcg, SolverOptions


@dataclass(frozen=True)
class LocalCapRows:
    prepared: object
    selected: torch.Tensor
    patches: torch.Tensor
    area: torch.Tensor
    components: tuple
    diagonal: torch.Tensor

    @property
    def count(self):
        return len(self.area)

    def flow(self, velocity):
        self.prepared.cap.grid.check_velocity(velocity)
        like = self.area
        if any(u.device != like.device or u.dtype != like.dtype for u in velocity):
            raise ValueError('cap and velocity must share dtype/device')
        out = like.new_zeros(self.count)
        for u, (row, index, coefficient) in zip(velocity, self.components):
            out.index_add_(0, row, coefficient*u.reshape(-1)[index])
        return out

    def lift(self, multiplier):
        """Positive W^-1 B^T multiplier; supported away from box faces."""
        if (multiplier.shape != self.area.shape or multiplier.dtype != self.area.dtype
                or multiplier.device != self.area.device or not torch.isfinite(multiplier).all()):
            raise ValueError('finite matching patch multipliers required')
        grid = self.prepared.cap.grid
        out = grid.zeros(device=self.area.device, dtype=self.area.dtype)
        for u, (row, index, coefficient) in zip(out, self.components):
            u.reshape(-1).index_add_(0, index, coefficient*multiplier[row]/grid.volume)
        return out

    def material_flow(self, nodal_velocity):
        p = self.prepared
        values = (p.cap.material_points(nodal_velocity)[self.selected]
                  *p.area_vectors[self.selected]).sum(-1)
        return self.area.new_zeros(self.count).index_add_(0, self.patches, values)

    def reaction(self, pressure):
        # This is the exact material adjoint of the selected cap flux rows.
        if (pressure.shape != self.area.shape or pressure.dtype != self.area.dtype
                or pressure.device != self.area.device or not torch.isfinite(pressure).all()):
            raise ValueError('finite matching patch pressures required')
        p, cap = self.prepared, self.prepared.cap
        force = p.area_vectors[self.selected]*pressure[self.patches, None]
        b = cap.barycentric[self.selected]
        edges = cap.rim[cap.parent[self.selected]]
        out = torch.zeros_like(cap.X)
        out.index_add_(0, edges[:, 0], force*b[:, :1])
        out.index_add_(0, edges[:, 1], force*b[:, 1:2])
        central = (force*b[:, 2:]).sum(0)/len(cap.rim)
        out.index_add_(0, cap.rim[:, 0], central.expand(len(cap.rim), -1))
        return out

    def diagnostics(self, velocity, nodal_velocity):
        """Distinguish enforced patch averages from unresolved point slip."""
        p = self.prepared
        points = p.points[self.selected]
        values = torch.stack(tuple((u.reshape(-1)[ids]*weights).sum(-1)
            for u, (ids, weights) in zip(velocity, _stencil(p.cap.grid, points))), -1)
        areas = p.area_vectors[self.selected]
        relative = values-p.cap.material_points(nodal_velocity)[self.selected]
        point_flux = (relative*areas).sum(-1)
        point_area = torch.linalg.vector_norm(areas, dim=-1)
        slip = point_flux/point_area
        patch_flux = self.flow(velocity)-self.material_flow(nodal_velocity)
        return dict(net_leak_ml_s=patch_flux.sum().item(),
            max_patch_average_normal_slip_cm_s=(patch_flux/self.area).abs().max().item(),
            max_patch_relative_flux_ml_s=patch_flux.abs().max().item(),
            pointwise_absolute_leak_ml_s=point_flux.abs().sum().item(),
            pointwise_normal_slip_rms_cm_s=torch.sqrt((point_area*slip.square()).sum()/point_area.sum()).item(),
            pointwise_normal_slip_max_cm_s=slip.abs().max().item())


@torch.no_grad()
def prepare_local_cap(prepared, *, patch_width_cells=3., minimum_area_fraction=.15,
                      regions=(0,)):
    """Material quadrature partition into MAC-scale local pressure patches.

    Tiny edge fragments are assigned to the nearest substantial patch. Every
    selected integration point is covered exactly once. Default excludes
    MV/AV; regions=(0,1,2) closes the complete idealized cap during iso phases.
    Geometry is frozen during a linear solve, never flattened or re-meshed.
    """
    if (not isfinite(patch_width_cells) or patch_width_cells < 2.
            or not isfinite(minimum_area_fraction) or not 0 < minimum_area_fraction < .5):
        raise ValueError('patch width >=2 cells and minimum area fraction in (0,.5) required')
    if not regions or len(set(regions)) != len(regions) or any(r not in (0,1,2) for r in regions):
        raise ValueError('nonempty distinct cap regions 0/1/2 required')
    cap, grid = prepared.cap, prepared.cap.grid
    mask = torch.zeros_like(cap.labels, dtype=torch.bool)
    for region in regions: mask |= cap.labels == region
    selected = torch.nonzero(mask).flatten()
    points, areas = prepared.points[selected], prepared.area_vectors[selected]
    width = patch_width_cells*max(grid.spacing)
    uv = (points-points.mean(0))@cap.frame[:2].T
    bins = torch.floor((uv-uv.amin(0))/width).to(torch.int64)
    _, inverse = torch.unique(bins, dim=0, return_inverse=True)
    n = int(inverse.max().item())+1
    point_area = torch.linalg.vector_norm(areas, dim=-1)
    bin_area = points.new_zeros(n).index_add_(0, inverse, point_area)
    centers = points.new_zeros((n, 2)).index_add_(0, inverse, uv*point_area[:, None])/bin_area[:, None]
    substantial = torch.nonzero(bin_area >= minimum_area_fraction*width**2).flatten()
    if len(substantial) == 0:
        raise ValueError('patch width too large for the non-port cap')
    # Merge small fragments only; substantial bins retain separate equations.
    assignment = torch.cdist(centers, centers[substantial]).argmin(-1)
    assignment[substantial] = torch.arange(len(substantial), device=points.device)
    patches = assignment[inverse]
    count = len(substantial)
    area = points.new_zeros(count).index_add_(0, patches, point_area)
    components = []
    diagonal = points.new_zeros(count)
    row = patches[:, None].expand(-1, 64).reshape(-1)
    for c, (ids, weights) in enumerate(_stencil(grid, points)):
        values = (weights*areas[:, c, None]).reshape(-1)
        sparse = torch.sparse_coo_tensor(torch.stack((row, ids.reshape(-1))), values,
            (count, prod(grid.face_shape(c))), device=points.device,
            check_invariants=False).coalesce()
        index, coefficient = sparse.indices(), sparse.values()
        keep = coefficient != 0
        r, i, a = index[0, keep], index[1, keep], coefficient[keep]
        components.append((r, i, a))
        diagonal.index_add_(0, r, a.square()/grid.volume)
    if (diagonal <= 0).any():
        raise ValueError('empty or degenerate local cap constraint')
    return LocalCapRows(prepared, selected, patches, area, tuple(components), diagonal)


class MACCapProjection:
    """Minimum kinetic correction subject to divergence and local cap flux.

    Uses the same Dirichlet pressure operator via multigrid or DST-II. Schur actions
    contain pressure solves, but no viscosity/convection/solid/EP advances.
    All arrays, sparse assembly and reductions remain on the selected device.
    """
    def __init__(self, rows, pressure_solver, *, options=None):
        self.rows, self.grid, self.pressure_solver = rows, rows.prepared.cap.grid, pressure_solver
        if (not isinstance(pressure_solver, (DirichletMultigrid,SpectralDirichletPressure))
                or tuple(pressure_solver.shapes[0]) != self.grid.shape
                or tuple(pressure_solver.spacings[0]) != self.grid.spacing):
            raise ValueError('pressure solver and cap MAC grids differ')
        self.options = options or SolverOptions(rtol=1e-9, atol=1e-11, max_iterations=400,
                                                recompute_every=30, check_every=1)
        self.pressure_calls = self.pressure_cycles = 0
        self.weights = tuple(face_weights(self.grid, c, rows.area) for c in range(3))

    def divergence_project(self, velocity):
        rhs = -divergence(velocity, self.grid.spacing)
        scale = torch.linalg.vector_norm(rhs).item()
        if scale == 0:
            return tuple(u.clone() for u in velocity), torch.zeros_like(rhs)
        # Homogeneous pressure actions avoid an absolute-tolerance-dependent
        # Schur operator, and always start from zero (no history dependence).
        phi, info = self.pressure_solver.solve(rhs/scale, initial=None)
        phi = phi*scale
        self.pressure_calls += 1
        self.pressure_cycles += info['cycles']
        return tuple(u-g for u, g in zip(velocity, gradient(phi, self.grid.spacing))), phi

    def action(self, multiplier):
        correction, _ = self.divergence_project(self.rows.lift(multiplier))
        return self.rows.flow(correction)

    @torch.no_grad()
    def solve(self, velocity, nodal_velocity, *, dt, density=1.):
        if any(not isfinite(v) or v <= 0 for v in (dt, density)):
            raise ValueError('positive finite dt and density required')
        self.pressure_calls = self.pressure_cycles = 0
        # Do not assume an arbitrary input is exactly solenoidal.
        base, base_phi = self.divergence_project(velocity)
        target = self.rows.material_flow(nodal_velocity)
        rhs = self.rows.flow(base)-target
        multiplier, info = pcg(self.action, rhs, self.rows.diagonal, options=self.options)
        correction, phi = self.divergence_project(self.rows.lift(multiplier))
        result = tuple(u-v for u, v in zip(base, correction))
        pressure = density/dt*multiplier
        reaction = self.rows.reaction(pressure)
        # Pressure increment includes the initial divergence correction.
        pressure_increment = density/dt*(base_phi-phi)
        residual = self.rows.flow(result)-target
        tolerance = max(self.options.atol, self.options.rtol*torch.linalg.vector_norm(rhs).item())
        true_norm = torch.linalg.vector_norm(residual).item()
        if true_norm > 2*tolerance:
            raise RuntimeError(f'cap true flux residual {true_norm:g} > {2*tolerance:g}')
        before = .5*density*sum((u.square()*w).sum() for u, w in zip(velocity, self.weights))
        base_energy = .5*density*sum((u.square()*w).sum() for u, w in zip(base, self.weights))
        after = .5*density*sum((u.square()*w).sum() for u, w in zip(result, self.weights))
        cap_loss = .5*density*sum((u.square()*w).sum() for u, w in zip(correction, self.weights))
        divergence_loss = .5*density*sum(((u-v).square()*w).sum()
                                         for u, v, w in zip(velocity, base, self.weights))
        loss = cap_loss+divergence_loss
        material_power = (reaction*nodal_velocity).sum()
        fluid_power = -sum((u*v*w).sum() for u, v, w in
                          zip(result, self.rows.lift(pressure), self.weights))
        energy_error = after-before+loss+dt*material_power
        report = dict(patches=self.rows.count, iterations=info.iterations,
            pressure_calls=self.pressure_calls, pressure_cycles=self.pressure_cycles,
            constraint_residual_norm_ml_s=true_norm, constraint_tolerance_ml_s=tolerance,
            max_divergence_before_s_inv=divergence(velocity, self.grid.spacing).abs().max().item(),
            max_divergence_after_s_inv=divergence(result, self.grid.spacing).abs().max().item(),
            cap_fluid_power_erg_s=fluid_power.item(), cap_material_power_erg_s=material_power.item(),
            cap_total_power_erg_s=(fluid_power+material_power).item(),
            kinetic_energy_before_erg=before.item(), kinetic_energy_after_erg=after.item(),
            kinetic_energy_solenoidal_base_erg=base_energy.item(),
            divergence_projection_loss_erg=divergence_loss.item(),
            cap_projection_loss_erg=cap_loss.item(),
            projection_loss_erg=loss.item(), energy_balance_error_erg=energy_error.item(),
            energy_balance_relative_error=(energy_error.abs()/(before.abs()+after.abs()+loss.abs()
                +dt*material_power.abs()).clamp_min(1e-30)).item(),
            prescribed_material_motion=True, coupled_solid_advanced=False)
        return result, pressure_increment, pressure, reaction, report
