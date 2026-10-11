"""Offline hydrostatic load compatibility on the original BE/MAC box.

Surface load comparisons are NOT a replacement for the source nodal PK1,
paired basal reactions, or a new preload/checkpoint. The composite kernel is
an isolated candidate: production FETransfer and cap operators are unchanged.
"""
from math import isfinite
from time import perf_counter
import torch
from ..ib import peskin4
from ..mac.grid import divergence
from ..mac.open_boundary import gradient, velocity_laplacian, face_weights
from .mac_mass_budget import ClosedCavityMassAudit
from .mac_pressure_audit import cavity_masks, pressure_statistics
from .mac_preload import MACEndocardialPressure
from .windkessel import MMHG_TO_DYN_CM2 as C


def bs3(r):
    """Centered quadratic cardinal B-spline, support [-1.5,1.5]."""
    a = r.abs()
    return torch.where(a < .5, .75-a.square(), .5*(1.5-a).clamp_min(0).square())


def bs4(r):
    """Centered cubic cardinal B-spline, support [-2,2]."""
    a = r.abs()
    return torch.where(a < 1., 2/3-a.square()+.5*a**3, (2-a).clamp_min(0)**3/6)


def surface_stencil(grid, points, *, kernel):
    if kernel not in ('peskin4', 'bs4-bs3'):
        raise ValueError('unknown diagnostic surface kernel')
    if (points.ndim != 2 or points.shape[1] != 3 or points.dtype != torch.float64
            or not torch.isfinite(points).all()):
        raise ValueError('finite float64 surface points required')
    h = points.new_tensor(grid.spacing)
    box = (points-points.new_tensor(grid.origin))/h
    if (box < 2).any() or (box >= box.new_tensor(grid.shape)-2).any():
        raise ValueError('surface kernel support reaches the outer box')
    result = []
    for c in range(3):
        scaled = (points-points.new_tensor(grid.face_origin(c)))/h
        nodes, phi = [], []
        for axis in range(3):
            count = 4 if kernel == 'peskin4' or axis == c else 3
            radius = count/2
            index = torch.floor(scaled[:, axis]-radius+1).to(torch.int64)[:, None]
            index = index+torch.arange(count, device=points.device)
            r = scaled[:, axis, None]-index.to(points.dtype)
            basis = peskin4 if kernel == 'peskin4' else (bs4 if axis == c else bs3)
            nodes.append(index); phi.append(basis(r))
        shape = grid.face_shape(c)
        ids = ((nodes[0][:, :, None, None]*shape[1]+nodes[1][:, None, :, None])
               *shape[2]+nodes[2][:, None, None, :]).reshape(len(points), -1)
        weights = (phi[0][:, :, None, None]*phi[1][:, None, :, None]
                   *phi[2][:, None, None, :]).reshape_as(ids)
        result.append((ids, weights))
    return result


def surface_trace(grid, velocity, points, *, kernel):
    grid.check_velocity(velocity)
    if any(u.dtype != points.dtype or u.device != points.device for u in velocity):
        raise ValueError('surface and velocity dtype/device must agree')
    return torch.stack([(u.flatten()[ids]*w).sum(-1)
        for u, (ids, w) in zip(velocity, surface_stencil(grid, points, kernel=kernel))], -1)


@torch.no_grad()
def surface_pressure_density(audit, triangles, pressure_mmhg, *, level, kernel):
    """Spread -p n (blood-outward n) in bounded-memory chunks.

    This is a boundary-force diagnostic only. In particular, cap fluid load is
    not silently added to the solid source endocardial pressure covector.
    """
    if (not isfinite(pressure_mmhg) or type(level) is not int or not 0 <= level <= 6
            or triangles.ndim != 3 or triangles.shape[1:] != (3, 3)
            or triangles.dtype != audit.X.dtype or triangles.device != audit.X.device
            or not torch.isfinite(triangles).all()):
        raise ValueError('matching finite triangles, pressure and level in 0..6 required')
    bary = audit.rule(level); q = len(bary)
    batch = max(1, audit.chunk_points//q)
    result = audit.grid.zeros(device=triangles.device, dtype=triangles.dtype)
    for start in range(0, len(triangles), batch):
        nodes = triangles[start:start+batch]
        points = torch.einsum('qa,fai->fqi', bary, nodes).reshape(-1, 3)
        force = (-pressure_mmhg*C*audit.area(nodes)/q)[:, None].expand(-1, q, -1).reshape(-1, 3)
        for c, (ids, weights) in enumerate(surface_stencil(audit.grid, points, kernel=kernel)):
            result[c].flatten().index_add_(0, ids.flatten(),
                (weights*force[:, c, None]/audit.grid.volume).flatten())
    return result


def weighted_norm(grid, field):
    return sum((v.square()*face_weights(grid, c, v)).sum()
               for c, v in enumerate(field)).sqrt()


def compatible_be_density(flow, pressure):
    """H G p: manufactured hydrostatic control for THIS split BE projection.

    It is not a pressure gauge shift or a proposed physiological forcing.
    H=I-nu*dt*L, so the existing predictor returns dt/rho*G p exactly.
    """
    g = gradient(pressure, flow.grid.spacing)
    return tuple(v-flow.dt*flow.mu/flow.rho*velocity_laplacian(v, c, flow.grid.spacing)
                 for c, v in enumerate(g))


@torch.no_grad()
def load_response(endpoint, x, density, core, *, expected_pressure=None):
    grid, flow = endpoint.grid, endpoint.flow
    grid.check_velocity(density)
    # Orthogonal MAC Hodge decomposition with the unchanged zero-Dirichlet
    # pressure operator. No free constant is introduced.
    potential, info = flow.pressure_solver.solve(-divergence(density, grid.spacing))
    g = gradient(potential, grid.spacing)
    remainder = tuple(f-v for f, v in zip(density, g))
    u, p, fluid = flow.solve_rhs(tuple(flow.dt/flow.rho*f for f in density))
    endpoint.transfer.reset_warm_start()
    nodal, _ = endpoint.transfer.interpolate(u, endpoint.transfer.prepare(x))
    net = torch.stack([(f*face_weights(grid, c, f)).sum() for c, f in enumerate(density)])
    r = dict(mac_core_pressure=pressure_statistics(p, core),
        max_velocity_cm_s=max(v.abs().max().item() for v in u),
        original_ib_displacement_response_l2_cm=(flow.dt*nodal).norm().item(),
        divergence_max_s_inv=divergence(u, grid.spacing).abs().max().item(),
        density_weighted_l2=weighted_norm(grid, density).item(),
        non_gradient_density_weighted_l2=weighted_norm(grid, remainder).item(),
        non_gradient_density_fraction=(weighted_norm(grid, remainder)/
            weighted_norm(grid, density).clamp_min(1e-30)).item(),
        total_fluid_force_dyn=net.tolist(), hodge_pressure_residual=info,
        flow=fluid)
    if expected_pressure is not None:
        r['full_pressure_error_max_mmhg'] = (p-expected_pressure).abs().max().item()/C
    return r


@torch.no_grad()
def audit_preload_loads(driver, state, *, levels=(2, 3), guard_cells=2):
    """Static geometry/force audit; never march, re-preload or publish state."""
    if (state.fsi.step != 0 or state.fsi.time != driver.config.start_time_s
            or state.force_load.use_land or getattr(state, 'cap_pressure', None) is not None
            or driver.config.mechanics_profile != 'source'
            or driver.config.prescribed_endocardial_pressure_mmhg <= 0
            or any(v.abs().max() != 0 for v in state.fsi.velocity)
            or state.fsi.pressure.abs().max() != 0
            or len(levels) < 2 or tuple(sorted(set(levels))) != tuple(levels)
            or any(type(k) is not int or not 0 <= k <= 4 for k in levels)):
        raise ValueError('unadvanced source pressure preload at rest and increasing levels 0..4 required')
    driver.buffer.stage(state.force_load)
    e = driver._fsi_stepper(driver.maximum_ticks)
    x = state.fsi.x
    audit = ClosedCavityMassAudit(x, driver.model.endo_faces, e.grid)
    wall, cap = audit.triangles(x)
    inside, core = cavity_masks(x, driver.model.endo_faces, e.grid, guard_cells=guard_cells)
    pressure = driver.config.prescribed_endocardial_pressure_mmhg
    traction = MACEndocardialPressure(driver.model).force(x, pressure)
    material = driver.model.force(x, state.fsi.time)-traction
    stencil = e.transfer.prepare(x)
    e.transfer.reset_warm_start()
    material_density = e.transfer.spread(material, stencil)[0]
    e.transfer.reset_warm_start()
    endo_density = e.transfer.spread(-traction, stencil)[0]
    results = {}
    def measure(name, density, expected=None):
        if x.is_cuda: torch.cuda.synchronize(x.device)
        started = perf_counter()
        results[name] = load_response(e, x, density, core, expected_pressure=expected)
        if x.is_cuda: torch.cuda.synchronize(x.device)
        results[name]['response_seconds'] = perf_counter()-started
    measure('source_material_without_prescribed_traction', material_density)
    for level in levels:
        cap_density = surface_pressure_density(audit, cap, pressure, level=level+2, kernel='peskin4')
        measure(f'source_endo_volume_transfer_plus_fluid_cap_level_{level}',
            tuple(a+b for a, b in zip(endo_density, cap_density)))
        for kernel in ('peskin4', 'bs4-bs3'):
            w = surface_pressure_density(audit, wall, pressure, level=level, kernel=kernel)
            c = surface_pressure_density(audit, cap, pressure, level=level+2, kernel=kernel)
            measure(f'closed_surface_{kernel}_level_{level}', tuple(a+b for a, b in zip(w, c)))
    target = inside.to(x.dtype)*(pressure*C)
    measure('manufactured_discrete_hydrostatic_control', compatible_be_density(e.flow, target), target)
    changes = {}
    for kernel in ('peskin4', 'bs4-bs3'):
        last, prior = (results[f'closed_surface_{kernel}_level_{k}'] for k in levels[-2:][::-1])
        changes[kernel] = dict(
            velocity_max_change_cm_s=abs(last['max_velocity_cm_s']-prior['max_velocity_cm_s']),
            core_pressure_mean_change_mmhg=abs(last['mac_core_pressure']['mean_mmhg']-
                                               prior['mac_core_pressure']['mean_mmhg']),
            non_gradient_fraction_change=abs(last['non_gradient_density_fraction']-
                                             prior['non_gradient_density_fraction']))
    manufactured = results['manufactured_discrete_hydrostatic_control']
    return dict(status='completed', scope='offline hydrostatic load compatibility; original Cartesian BE/MAC',
        grid_shape=e.grid.shape, grid_spacing_cm=e.grid.spacing, dt_s=e.flow.dt,
        source_pressure_mmhg=pressure, results=results, level_changes=changes,
        manufactured_control_passed=(manufactured['max_velocity_cm_s'] < 1e-6
            and manufactured['full_pressure_error_max_mmhg'] < .05),
        source_force_balance_l2_dyn=driver.model.force(x, state.fsi.time).norm().item(),
        static_pressure_preload_implemented=False, production_handoff_enabled=False,
        production_transfer_kernel_changed=False, candidate_kernel_scope='offline closed-surface load only',
        cap_fluid_load_has_paired_material_reaction=False,
        isolated_loads_are_complete_fsi_equilibria=False, pressure_reference_shift_applied=False,
        geometry_modified=False, ep_advanced=False, land_history_advanced=False,
        state_advanced=False, full_fsi_outer_iterations_added=False,
        new_fluid_volume_mesh_added=False, full_cardiac_cycle_implemented=False)
