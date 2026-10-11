"""Pressure-traction-free CLOSED-cap trial in the original implicit MAC solve.

This is a static handoff validation, never a production/checkpoint converter.
Displacement and paired cap pressures are solved together. No wall pressure
unknown, prescribed endocardial traction, outer FSI loop or pressure shift is
used. Failed static trials must not replace source preload/history states.
"""
from dataclasses import dataclass, replace
from math import isfinite
from time import perf_counter
import torch
from .mac_preload import MACEndocardialPressure
from .mac_isovolumic import IsovolumicAFSEIStepper
from .mac_cap_coupled import CapIsovolumicProblem, closed_rows
from .mac_pressure_audit import cavity_masks, pressure_statistics
from .mac_mass_budget import ClosedCavityMassAudit
from ..mac.paper_coupling import BEProblem, PaperState
from ..mac.open_boundary import face_weights
from ..mac.grid import divergence
from ..nonlinear import newton, normalized_linear_action, NonlinearFailure
from .windkessel import MMHG_TO_DYN_CM2 as C


@dataclass(frozen=True)
class HandoffTargets:
    # Same static velocity/displacement limits used by source MAC preload.
    displacement_l2_cm: float = 1e-9
    velocity_max_cm_s: float = 1e-6
    pressure_error_mmhg: float = .05
    volume_error_ml: float = 1e-3
    closed_cap_flux_norm_ml_s: float = 1e-5
    work_relative_error: float = 1e-7

    def __post_init__(self):
        if any(not isfinite(v) or v <= 0 for v in self.__dict__.values()):
            raise ValueError('finite positive static handoff targets required')


class TractionFreeCapProblem(CapIsovolumicProblem):
    """Use original BE response and cached cap border, without a wall controller."""
    def __init__(self, endpoint, state, stencil, departure):
        if endpoint.owner.config.prescribed_endocardial_pressure_mmhg != 0:
            raise ValueError('handoff trial must have zero prescribed wall traction')
        BEProblem.__init__(self, endpoint, state, stencil, departure)
        self.load = MACEndocardialPressure(endpoint.model)
        self.rows = closed_rows(endpoint, state.x)
        self.mechanical_size = state.x.numel()
        self.cap_scale = endpoint.flow.rho*self.rows.area/(
            endpoint.flow.dt**2*self.rows.diagonal)
        self.last_z = self.last_residual = self.last_cap_pressure = None
        self.cache_hits = 0
        self.setup_seconds = 0.
        self.setup_pressure_calls = 0
        self.preconditioner_rebuilt = False
        self._prepare_preconditioner()

    @property
    def extra_unknowns(self):
        return self.rows.count

    def unpack(self, z):
        y = z[:self.mechanical_size].reshape_as(self.state.x)
        return y, self.state.x+y, 0.

    def cap_pressure(self, z):
        return self.cap_scale*z[self.mechanical_size:]

    def validate(self, z):
        if z.shape != (self.mechanical_size+self.rows.count,) or not torch.isfinite(z).all():
            raise ValueError('finite matching displacement/cap-pressure trial required')
        self.driver.solid.validate(self.unpack(z)[1])

    def residual(self, z):
        self.validate(z)
        d, rows = self.driver, self.rows
        y, x, _ = self.unpack(z)
        bp = self.cap_pressure(z)
        force = d.solid.force(x, self.time)+rows.reaction(bp)
        d.transfer.reset_warm_start()
        density, spread = d.transfer.spread(force, self.stencil)
        density = tuple(a-b for a, b in zip(density, rows.lift(bp)))
        velocity, pressure, flow = d.flow.advance(self.departure, density, self.pressure_initial)
        nodal, interpolation = d.transfer.interpolate(velocity, self.stencil)
        residual = torch.cat(((y-d.flow.dt*nodal).flatten(),
            (d.flow.dt*rows.flow(velocity)-rows.material_flow(y))/rows.area))
        if not torch.isfinite(residual).all():
            raise FloatingPointError('nonfinite traction-free cap residual')
        self.pressure_initial = pressure
        self.last_z, self.last_residual = z.clone(), residual.clone()
        self.last_cap_pressure = bp.clone()
        self.last_data = x, 0., self.load.volume(x), force, velocity, pressure, flow, nodal, density, spread, interpolation
        self.evaluations += 1
        return residual

    def linearization(self, z):
        self.accepted(z)
        d, rows = self.driver, self.rows
        x = self.unpack(z)[1]
        if d.assembler is None:
            d.assembler = d.model.tangent_factory()
        tangent = d.assembler.assemble(x, self.time)
        self.tangent_assemblies += 1
        def action(v):
            self.jacobian_actions += 1
            dy = v[:self.mechanical_size].reshape_as(x)
            df = torch.sparse.mm(tangent, dy.reshape(-1, 1)).reshape_as(x)
            du, dw = self.sensitivity(df, self.cap_scale*v[self.mechanical_size:])
            cap = (d.flow.dt*rows.flow(du)-rows.material_flow(dy))/rows.area
            return torch.cat(((dy-d.flow.dt*dw).flatten(), cap))
        return normalized_linear_action(action)

    def preconditioner(self, z):
        # Same existing kinetic cap columns, without the artificial wall-pressure border.
        lu, pivots = torch.linalg.lu_factor(self.cap_h)
        def apply(v):
            self.preconditioner_applications += 1
            y = v[:self.mechanical_size]
            lower = v[self.mechanical_size:]+self.rows.material_flow(y.reshape_as(self.state.x))/self.rows.area
            q = torch.linalg.lu_solve(lu, pivots, lower[:, None]).flatten()
            return torch.cat((y-self.cap_b@q, q))
        return apply


@torch.no_grad()
def traction_free_step(endpoint, old, *, problem_class=TractionFreeCapProblem):
    """One discarded implicit mechanical/fluid trial; EP/ODE are not advanced."""
    before = endpoint.flow.calls
    stencil = endpoint.transfer.prepare(old.x)
    problem = problem_class(endpoint, old, stencil, endpoint.flow.advect(old.velocity))
    z = old.x.new_zeros(old.x.numel()+problem.extra_unknowns)
    r0 = problem.residual(z)
    settings = endpoint.config.nonlinear
    tolerance = max(settings.atol, settings.rtol*r0[:old.x.numel()].norm().item())
    result = newton(problem.residual, z, validate=problem.validate,
        linearization_factory=problem.linearization, preconditioner_factory=problem.preconditioner,
        options=replace(settings, rtol=0., atol=tolerance))
    residual = problem.accepted(result.x)
    if residual.norm().item() > tolerance:
        raise NonlinearFailure('traction-free true residual exceeds original target', result)
    x, _, volume, force, u, p, _, nodal, *_ = problem.last_data
    endpoint.check_support(x)
    courant = endpoint.flow.dt*sum(v.abs().max()/h for v, h in zip(u, endpoint.grid.spacing)).item()
    if not isfinite(courant) or courant > endpoint.config.max_courant:
        raise ValueError('traction-free trial exceeds original Courant limit')
    bp, rows = problem.last_cap_pressure, problem.rows
    material = (x-old.x)/endpoint.flow.dt
    rates = rows.flow(u)-rows.material_flow(material)
    projected_rates = rows.flow(u)-rows.material_flow(nodal)
    fluid_power = -sum((a*b*face_weights(endpoint.grid, c, x)).sum()
        for c, (a, b) in enumerate(zip(u, rows.lift(bp))))
    solid_power = (rows.reaction(bp)*nodal).sum()
    expected = -(bp*projected_rates).sum()
    work_error = abs((fluid_power+solid_power-expected).item())
    work_scale = max(abs(fluid_power.item())+abs(solid_power.item())+abs(expected.item()), 1.)
    report = dict(nonlinear_iterations=result.iterations, nonlinear_residual_norm=residual.norm().item(),
        nonlinear_tolerance=tolerance, fluid_solves=endpoint.flow.calls-before,
        jacobian_actions=problem.jacobian_actions,
        preconditioner_rebuilt=problem.preconditioner_rebuilt,
        preconditioner_setup_seconds=problem.setup_seconds,
        closed_cap_flux_norm_ml_s=rates.norm().item(),
        original_ib_motion_residual_l2_cm=(x-old.x-endpoint.flow.dt*nodal).norm().item(),
        pressure_min_mmhg=p.min().item()/C, pressure_max_mmhg=p.max().item()/C,
        cap_pressure_min_mmhg=bp.min().item()/C, cap_pressure_max_mmhg=bp.max().item()/C,
        cap_reaction_l2_dyn=rows.reaction(bp).norm().item(),
        cap_fluid_power_erg_s=fluid_power.item(), cap_material_power_erg_s=solid_power.item(),
        cap_total_power_erg_s=(fluid_power+solid_power).item(),
        cap_work_adjoint_relative_error=work_error/work_scale,
        cap_closed_power_relative_error=abs((fluid_power+solid_power).item())/work_scale,
        cap_projected_flux_norm_ml_s=projected_rates.norm().item(),
        cap_pressure_flux_power_bound_erg_s=(bp.norm()*projected_rates.norm()).item(),
        divergence_max_s_inv=divergence(u, endpoint.grid.spacing).abs().max().item(),
        max_velocity_cm_s=max(v.abs().max().item() for v in u),
        displacement_l2_cm=(x-old.x).norm().item(),
        volume_ml=volume.item(), courant=courant,
        retained_prescribed_wall_traction_mmhg=0., wall_pressure_unknowns=0,
        global_volume_constraint_added=False, full_fsi_outer_iterations_added=False,
        paired_cap_reaction_applied=True)
    new_time = old.time+endpoint.flow.dt
    trial = PaperState(old.step+1, new_time, x, u, p, force, new_time,
                       pressure_time=new_time, previous_x=old.x.clone(), previous_dt=endpoint.flow.dt)
    return trial, report


@torch.no_grad()
def validate_static_handoff(driver, state, *, steps=1, targets=None, diagnostic_guard_cells=2):
    """Try an actual paired source-free handoff, return evidence without publishing.

    Require an unadvanced source preload. Keep all Land/EP/strain arrays and
    geometry in the supplied checkpoint. An active-cycle state is not a static
    preload and cannot be substituted silently.
    """
    targets = targets or HandoffTargets()
    if (type(steps) is not int or not 1 <= steps <= 20 or driver.config.mechanics_profile != 'source'
            or driver.config.prescribed_endocardial_pressure_mmhg <= 0
            or state.fsi.step != 0 or state.force_load.use_land
            or state.fsi.time != driver.config.start_time_s
            or getattr(state, 'cap_pressure', None) is not None
            or any(v.abs().max() != 0 for v in state.fsi.velocity)
            or state.fsi.pressure.abs().max() != 0):
        raise ValueError('unadvanced source prescribed-pressure preload at rest required')
    driver.buffer.stage(state.force_load)
    # Independent original prescribed-pressure trial: distinguish a failed
    # source preload from a failed change of pressure-support formulation.
    base_endpoint = driver._fsi_stepper(driver.maximum_ticks)
    baseline, base_info = base_endpoint.physical_step(state.fsi,
        state.fsi.time+base_endpoint.flow.dt, True)
    base_displacement = (baseline.x-state.fsi.x).norm().item()
    base_speed = max(v.abs().max().item() for v in baseline.velocity)
    if (base_displacement > targets.displacement_l2_cm or base_speed > targets.velocity_max_cm_s):
        raise ValueError('original source preload fails existing static equilibrium targets')
    # Fresh model/config: remove the load from both force and exact tangent.
    free = IsovolumicAFSEIStepper(driver.mesh,
        replace(driver.config, prescribed_endocardial_pressure_mmhg=0.))
    free.closed_cap = True
    free.cap_patch_width_cells = 3.
    free.buffer.stage(state.force_load)
    e = free._fsi_stepper(free.maximum_ticks)
    x0 = state.fsi.x
    initial_force = free.model.force(x0, state.fsi.time)
    baseline_force = driver.model.force(x0, state.fsi.time)
    traction = MACEndocardialPressure(driver.model).force(x0, driver.config.prescribed_endocardial_pressure_mmhg)
    if not torch.allclose(initial_force+traction, baseline_force, rtol=1e-9, atol=1e-7):
        raise AssertionError('wall traction removed inconsistently')
    _, core = cavity_masks(x0, driver.model.endo_faces, e.grid, guard_cells=diagnostic_guard_cells)
    surface = ClosedCavityMassAudit(driver.mesh.X, driver.model.endo_faces, e.grid)
    volume0 = MACEndocardialPressure(free.model).volume(x0).item()
    trial = replace(state.fsi, force=initial_force)
    rows = []
    for _ in range(steps):
        torch.cuda.synchronize(x0.device); started = perf_counter()
        trial, r = traction_free_step(e, trial)
        torch.cuda.synchronize(x0.device)
        r['implicit_trial_seconds'] = perf_counter()-started
        r['mac_core_pressure'] = pressure_statistics(trial.pressure, core)
        r['displacement_from_source_l2_cm'] = (trial.x-x0).norm().item()
        r['volume_error_from_source_ml'] = r['volume_ml']-volume0
        r['trial_time_s'] = trial.time
        rows.append(r)
    wall, _ = surface.region(surface.triangles(trial.x)[0],
        surface.triangles((trial.x-trial.previous_x)/e.flow.dt)[0], trial.velocity, 3)
    failed = []
    for field, limit, label in (
        ('displacement_from_source_l2_cm', targets.displacement_l2_cm, 'static_displacement'),
        ('max_velocity_cm_s', targets.velocity_max_cm_s, 'static_velocity'),
        ('volume_error_from_source_ml', targets.volume_error_ml, 'source_closed_volume'),
        ('closed_cap_flux_norm_ml_s', targets.closed_cap_flux_norm_ml_s, 'closed_cap_flux'),
        ('cap_work_adjoint_relative_error', targets.work_relative_error, 'cap_work_adjoint'),
        ('cap_closed_power_relative_error', targets.work_relative_error, 'closed_cap_power')):
        if any(abs(r[field]) > limit for r in rows): failed.append(label)
    if any(abs(r['mac_core_pressure']['mean_mmhg']-driver.config.prescribed_endocardial_pressure_mmhg)
           > targets.pressure_error_mmhg for r in rows):
        failed.append('preload_pressure')
    return dict(status='completed', scope='discarded static source-free paired MAC handoff validation',
        static_handoff_validated=not failed, failed_targets=failed, targets=targets.__dict__,
        baseline_static_displacement_l2_cm=base_displacement,
        baseline_static_velocity_max_cm_s=base_speed,
        baseline_nonlinear_residual_norm=base_info['nonlinear']['residual_norm'],
        source_pressure_mmhg=driver.config.prescribed_endocardial_pressure_mmhg,
        original_source_force_l2_dyn=baseline_force.norm().item(),
        source_free_initial_material_force_l2_dyn=initial_force.norm().item(),
        source_reference_volume_ml=volume0, rows=rows, final_wall_surface=wall,
        ep_advanced=False, land_history_advanced=False, source_geometry_modified=False,
        source_state_advanced=False, candidate_published=False,
        production_handoff_enabled=False, full_cardiac_cycle_implemented=False,
        pressure_reference_shift_applied=False, anatomical_valves_present=False,
        local_wall_sealing_established=False)
