"""Offline pressure/load and open-port interface audit on the original MAC box.

No time advance, new fluid mesh, boundary correction or pressure gauge shift.
Core masks are only diagnostic selections of EXISTING pressure cells. Surface
Peskin slip is distinguished from the original volume-L2 IB motion equation.
"""
from math import isfinite
import torch
import torch.nn.functional as F
from .mac_mass_budget import ClosedCavityMassAudit, peskin_trace
from .mac_ports import MACBasalPorts
from .mac_preload import MACEndocardialPressure
from .mac_cap_coupled import closed_rows
from .mac_rcr_coupled import RCRState, outlet_rows
from .windkessel import MMHG_TO_DYN_CM2 as C
from ..circulation.geometry import vertical_inside
from ..mac.open_boundary import gradient


@torch.no_grad()
def cavity_masks(x, faces, grid, *, guard_cells=2):
    """Ray classify cell centres; remove a Chebyshev band for bulk statistics."""
    if type(guard_cells) is not int or not 0 <= guard_cells <= 4:
        raise ValueError('diagnostic guard_cells must be an integer in 0..4')
    audit = ClosedCavityMassAudit(x, faces, grid)
    center = x[audit.rim[:, 0]].mean(0)
    vertices = torch.cat((x, center[None]))
    cap = torch.cat((audit.rim, torch.full((len(audit.rim), 1), len(x),
                    device=x.device, dtype=torch.int64)), 1)
    surface = torch.cat((faces[:, [0, 2, 1]], cap))
    points = grid.coordinates(device=x.device, dtype=x.dtype)
    bounds = x[torch.unique(faces)]
    low, high = bounds.amin(0), bounds.amax(0)
    ix = torch.nonzero((points[:, 0, 0, 0] >= low[0]) & (points[:, 0, 0, 0] <= high[0])).flatten()
    iy = torch.nonzero((points[0, :, 0, 1] >= low[1]) & (points[0, :, 0, 1] <= high[1])).flatten()
    iz = torch.nonzero((points[0, 0, :, 2] >= low[2]) & (points[0, 0, :, 2] <= high[2])).flatten()
    if min(len(ix), len(iy), len(iz)) == 0:
        raise ValueError('cavity has no diagnostic MAC centres; refine the grid')
    xy = points[ix[:, None], iy[None, :], 0, :2].reshape(-1, 2)
    inside = vertical_inside(vertices, surface, xy, points[0, 0, iz, 2])
    mask = torch.zeros(grid.shape, device=x.device, dtype=torch.bool)
    mask[ix[:, None, None], iy[None, :, None], iz[None, None, :]] = inside.reshape(len(ix), len(iy), len(iz))
    # Pad with outside, so a cavity crossing the box cannot gain false core cells.
    outside = (~mask).to(x.dtype)[None, None]
    if guard_cells:
        outside = F.max_pool3d(F.pad(outside, (guard_cells,)*6, value=1.),
                              2*guard_cells+1, stride=1)
    core = outside[0, 0] == 0
    if not core.any():
        raise ValueError('no cavity core after diagnostic guard; refine or reduce guard_cells')
    return mask, core


def pressure_statistics(pressure, mask):
    if (pressure.shape != mask.shape or mask.dtype != torch.bool or pressure.device != mask.device
            or pressure.dtype != torch.float64 or not mask.any() or not torch.isfinite(pressure).all()):
        raise ValueError('finite float64 pressure and nonempty matching boolean mask required')
    values = pressure[mask]/C
    return dict(cell_count=len(values), mean_mmhg=values.mean().item(),
                minimum_mmhg=values.min().item(), maximum_mmhg=values.max().item(),
                standard_deviation_mmhg=values.std(correction=0).item())


def cap_regions(prepared, velocity, material_velocity):
    """Report each labelled region; open AV throughflow is never wall leakage."""
    area = prepared.area_vectors
    u = peskin_trace(prepared.cap.grid, velocity, prepared.points)
    w = prepared.cap.material_points(material_velocity)
    relative = ((u-w)*area).sum(-1)
    fluid = (u*area).sum(-1)
    motion = (w*area).sum(-1)
    rows = []
    for label, name in enumerate(('non_port_cap', 'mv', 'av')):
        keep = prepared.cap.labels == label
        a = area[keep].norm(dim=-1)
        v = relative[keep]
        rows.append(dict(region=name, area_cm2=a.sum().item(),
            fluid_flux_ml_s=fluid[keep].sum().item(), material_flux_ml_s=motion[keep].sum().item(),
            relative_flux_ml_s=v.sum().item(), absolute_relative_flux_ml_s=v.abs().sum().item(),
            normal_slip_rms_cm_s=((v.square()/a).sum()/a.sum()).sqrt().item()))
    return rows


@torch.no_grad()
def forcing_pressure_decomposition(driver, state, endpoint, core):
    """Linear force-only responses; saved advection history is NOT reconstructed.

    Separate forcing pieces and compare their sum to one independent combined
    response. The actual saved pressure minus that sum is an explicitly labelled
    time/advection remainder, not evidence of a full endpoint reconstruction.
    """
    x = state.fsi.x
    load = MACEndocardialPressure(driver.model)
    control = getattr(state, 'cavity_pressure_mmhg', driver.config.prescribed_endocardial_pressure_mmhg)
    active = driver.buffer.force(x)
    preload = load.force(x, driver.config.prescribed_endocardial_pressure_mmhg)
    total_model = driver.model.force(x, state.fsi.force_time)
    forces = dict(passive_and_basal=total_model-active-preload,
                  ep_land_active=active, source_wall_control=load.force(x, control))
    rows = None
    if getattr(state, 'cap_pressure', None) is not None:
        rows = (outlet_rows(endpoint, state.cap_geometry)
                if isinstance(state, RCRState) and state.boundary_history == 'rcr'
                else closed_rows(endpoint, state.cap_geometry))
        forces['cap_material_reaction'] = rows.reaction(state.cap_pressure)
    expected = sum(forces.values())
    force_error = (expected-state.fsi.force).abs().max().item()
    if not torch.allclose(expected, state.fsi.force, rtol=1e-9, atol=1e-7):
        raise AssertionError('saved force does not equal audited source/active/cap pieces')
    endpoint.transfer.reset_warm_start()
    stencil = endpoint.transfer.prepare(state.force_load.previous_x)
    densities = {}
    for name, force in forces.items():
        endpoint.transfer.reset_warm_start()
        densities[name] = endpoint.transfer.spread(force, stencil)[0]
    if rows is not None:
        densities['cap_fluid_traction'] = tuple(-v for v in rows.lift(state.cap_pressure))
    pressure_sum = torch.zeros_like(state.fsi.pressure)
    velocity_sum = endpoint.grid.zeros(device=x.device, dtype=x.dtype)
    component_reports = {}
    for name, density in densities.items():
        u, p, _ = endpoint.flow.solve_rhs(tuple(endpoint.flow.dt/endpoint.flow.rho*f for f in density))
        component_reports[name] = pressure_statistics(p, core)
        pressure_sum += p
        velocity_sum = tuple(a+b for a, b in zip(velocity_sum, u))
    combined = tuple(sum(density[c] for density in densities.values()) for c in range(3))
    u, p, _ = endpoint.flow.solve_rhs(tuple(endpoint.flow.dt/endpoint.flow.rho*f for f in combined))
    pressure_error = (pressure_sum-p).norm()/p.norm().clamp_min(1.)
    velocity_error = sum((a-b).square().sum() for a, b in zip(velocity_sum, u)).sqrt()
    velocity_scale = sum(a.square().sum() for a in u).sqrt().clamp_min(1.)
    if pressure_error > 1e-7 or velocity_error/velocity_scale > 1e-7:
        raise AssertionError('force-only pressure/velocity superposition failed')
    w = (x-state.force_load.previous_x)/state.force_load.dt_s
    return dict(components=component_reports, combined_force_only=pressure_statistics(p, core),
        saved_time_advection_remainder=pressure_statistics(state.fsi.pressure-pressure_sum, core),
        old_velocity_or_departure_reconstructed=False, saved_force_max_error_dyn=force_error,
        pressure_superposition_relative_error=pressure_error.item(),
        velocity_superposition_relative_error=(velocity_error/velocity_scale).item(),
        source_wall_control_power_erg_s=(forces['source_wall_control']*w).sum().item(),
        additional_offline_fluid_responses=len(densities)+1)


@torch.no_grad()
def audit_pressure_interface(driver, state, *, levels=(1, 2, 3), guard_cells=2, decompose=True):
    if (state.fsi.step < 1 or not isfinite(state.force_load.dt_s) or state.force_load.dt_s <= 0
            or len(levels) < 2 or tuple(sorted(set(levels))) != tuple(levels)
            or any(type(k) is not int or not 0 <= k <= 4 for k in levels)):
        raise ValueError('accepted motion history and >=2 increasing levels in 0..4 required')
    if not hasattr(state, 'cavity_pressure_mmhg'):
        # Plain preloaded MAC checkpoints are supported; their control is prescribed.
        if driver.config.prescribed_endocardial_pressure_mmhg == 0:
            raise ValueError('source-pressure preload/control checkpoint required')
    driver.buffer.stage(state.force_load)
    dt = state.force_load.dt_s
    endpoint = driver._fsi_stepper(round(dt/driver.protocol.dt_ep_s))
    grid = endpoint.grid
    x, old = state.fsi.x, state.force_load.previous_x
    audit = ClosedCavityMassAudit(driver.mesh.X, driver.model.endo_faces, grid)
    motion = (x-old)/dt
    wall_now = audit.triangles(x)[0]
    wall_old = audit.triangles(old)[0]
    wall_motion = audit.triangles(motion)[0]
    wall_levels = []
    for level in levels:
        current, _ = audit.region(wall_now, wall_motion, state.fsi.velocity, level)
        frozen, _ = audit.region(wall_old, wall_motion, state.fsi.velocity, level)
        wall_levels.append(dict(level=level, current=current, frozen=frozen))
    ports = MACBasalPorts(driver.mesh.X, driver.model.endo_faces, grid,
                         ports=getattr(driver, 'port_geometry', None))
    cap_old = cap_regions(ports.prepare(old), state.fsi.velocity, motion)
    cap_now = cap_regions(ports.prepare(x), state.fsi.velocity, motion)
    last, prior = wall_levels[-1], wall_levels[-2]
    mask, core = cavity_masks(x, driver.model.endo_faces, grid, guard_cells=guard_cells)
    pressure = pressure_statistics(state.fsi.pressure, core)
    control = getattr(state, 'cavity_pressure_mmhg', driver.config.prescribed_endocardial_pressure_mmhg)
    endpoint.transfer.reset_warm_start()
    projected, _ = endpoint.transfer.interpolate(state.fsi.velocity, endpoint.transfer.prepare(old))
    load = MACEndocardialPressure(driver.model)
    volume_rate = ((load.volume(x)-load.volume(old))/dt).item()
    av_pressure = (state.cap_pressure[-1].item()/C if isinstance(state, RCRState)
                   and state.boundary_history == 'rcr' and state.cap_pressure is not None else None)
    opened = isinstance(state, RCRState) and state.outlet_open and state.boundary_history == 'rcr'
    current_wall = last['current']
    current_cap_flux = sum(r['relative_flux_ml_s'] for r in cap_now)
    allowed_outlet = cap_now[2]['relative_flux_ml_s'] if opened else 0.
    closed_cap_absolute = sum(r['absolute_relative_flux_ml_s'] for r in cap_now[:2])
    if not opened:
        closed_cap_absolute += cap_now[2]['absolute_relative_flux_ml_s']
    uniform_gradient = gradient(torch.full_like(state.fsi.pressure, C), grid.spacing)
    signed_change = abs(current_wall['relative_flux_ml_s']-prior['current']['relative_flux_ml_s'])
    absolute_change = abs(current_wall['absolute_relative_flux_ml_s']-prior['current']['absolute_relative_flux_ml_s'])
    report = dict(scope='offline source-control/MAC-pressure and open-port interface audit',
        grid_shape=grid.shape, grid_spacing_cm=grid.spacing, time_s=state.fsi.time,
        step=state.fsi.step, dt_s=dt, source_wall_control_mmhg=control,
        mac_cavity_core_pressure=pressure, diagnostic_guard_cells=guard_cells,
        diagnostic_inside_cell_count=mask.sum().item(),
        control_minus_core_mean_mmhg=control-pressure['mean_mmhg'],
        av_constraint_multiplier_mmhg=av_pressure,
        av_multiplier_minus_core_mean_mmhg=None if av_pressure is None else av_pressure-pressure['mean_mmhg'],
        outer_box_pressure_boundary_mmhg=0.,
        uniform_1mmhg_shift_gradient_max_dyn_cm3=max(v.abs().max().item() for v in uniform_gradient),
        uniform_pressure_shift_is_null_mode=False,
        outlet_open=opened, anatomical_valves_present=False,
        cap_frozen_geometry=cap_old, cap_current_geometry=cap_now,
        frozen_actual_relative_av_flux_ml_s=cap_old[2]['relative_flux_ml_s'],
        current_actual_relative_av_flux_ml_s=cap_now[2]['relative_flux_ml_s'],
        finite_volume_rate_ml_s=volume_rate,
        finite_volume_plus_frozen_av_flux_ml_s=volume_rate+cap_old[2]['relative_flux_ml_s'],
        wall_levels=wall_levels,
        wall_signed_quadrature_change_ml_s=signed_change,
        wall_absolute_quadrature_change_ml_s=absolute_change,
        wall_quadrature_engineering_tolerance_ml_s=.001,
        wall_quadrature_converged_to_engineering_target=max(signed_change, absolute_change) < .001,
        cap_uses_original_operator_quadrature=True,
        cap_surface_subdivision_level=ports.metadata['subdivision_level'],
        full_surface_signed_relative_flux_ml_s=current_wall['relative_flux_ml_s']+current_cap_flux,
        closed_surface_signed_slip_ml_s=current_wall['relative_flux_ml_s']+current_cap_flux-allowed_outlet,
        closed_surface_absolute_slip_ml_s=current_wall['absolute_relative_flux_ml_s']+closed_cap_absolute,
        open_av_throughflow_excluded_from_closed_surface_slip=opened,
        original_ib_motion_residual_l2_cm=(dt*(projected-motion).norm()).item(),
        pressure_identification_established=False, pointwise_impermeability_established=False,
        full_cardiac_cycle_implemented=False, simulation_state_advanced=False,
        production_step_overhead_added=False)
    if decompose:
        report['forcing_pressure_decomposition'] = forcing_pressure_decomposition(driver, state, endpoint, core)
    return report
