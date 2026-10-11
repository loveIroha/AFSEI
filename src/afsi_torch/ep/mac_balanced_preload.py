"""Source material preload including the paired moving-cap pressure reaction.

Validation-only pressure-mode handoff. Source endocardial load stays separate
from the explicit cap reaction; the prescribed preload load is removed from
both force and tangent in discarded implicit FSI trials. Land strain history
is recomputed from the reference mesh, never imported from the old preload.
"""
from dataclasses import replace
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from time import perf_counter
import torch
from .mac_preload import MACEndocardialPressure, _check_driver
from .mac_preload import mesh_binding
from .checkpoint import digest, MESH, INTEGER
from .mac_transport import closed_volume_covectors
from .mac_isovolumic import IsovolumicAFSEIStepper
from .mac_pressure_mode import PressureModeTransfer
from .mac_pressure_handoff import traction_free_step, TractionFreeCapProblem, HandoffTargets
from .mac_pressure_audit import cavity_masks, pressure_statistics
from .driver import StrainHistory
from .windkessel import MMHG_TO_DYN_CM2 as C
from ..two_way.preload import (PassivePreloader, PreloadConfig, SourceLoadingHistory,
                              PreloadResult, PreloadStrainHistory)
from .buffered_load import ActiveLoadState
from ..nonlinear import normalized_linear_action, NonlinearFailure


class EndocardialAndCapPressure(MACEndocardialPressure):
    """Original source endocardial force PLUS explicit uniform cap reaction."""
    def cap_covector(self, x):
        return closed_volume_covectors(x, self.faces, self.rim)[1]

    def force(self, x, pressure_mmhg):
        return super().force(x, pressure_mmhg)+pressure_mmhg*C*self.cap_covector(x)

    def cap_action(self, x, v):
        edge, de = x[self.rim], v[self.rim]
        center, dc = x[self.rim[:, 0]].mean(0), v[self.rim[:, 0]].mean(0)
        e1, e2 = edge[:, 1]-edge[:, 0], center-edge[:, 0]
        if v.ndim == 3: e1, e2 = e1[..., None], e2[..., None]
        da = .5*(torch.linalg.cross(de[:, 1]-de[:, 0], e2, dim=1)
                 +torch.linalg.cross(e1, dc-de[:, 0], dim=1))
        width = v.shape[1:]
        out = v.new_zeros(v.shape).index_add(0, self.rim.flatten(),
            (da[:, None].expand(-1, 2, *width)/3).reshape(-1, *width))
        return out.index_add(0, self.rim[:, 0],
            (da.sum(0)/(3*len(self.rim))).expand(len(self.rim), *width))

    def action(self, x, pressure_mmhg, v):
        return super().action(x, pressure_mmhg, v)+pressure_mmhg*C*self.cap_action(x, v)


class BalancedCapProblem(TractionFreeCapProblem):
    """Same joint equations, stably evaluating the common pressure mode."""
    def cap_row_units(self, value, *, inverse=False):
        # Uniform pressure work is proportional to the area-weighted sum of
        # normal relative velocities. Resolve that sum more tightly than the
        # local roundoff floor using an invertible rank-one ROW scaling.
        # No constraint or unknown is added. Precondition with its inverse.
        weight = self.rows.area/self.rows.area.norm()
        coefficient = -15/16 if inverse else 15.
        return value+coefficient*weight*(weight*value).sum()

    def sensitivity(self, force, pressure=None):
        d = self.driver
        pressure = torch.zeros_like(self.rows.area) if pressure is None else pressure
        force = force+self.rows.reaction(pressure)
        d.transfer.reset_warm_start()
        density, _, _ = d.transfer.relative_parts(force, pressure, self.stencil)
        velocity = d.flow.response(density)
        nodal, _ = d.transfer.interpolate(velocity, self.stencil)
        return velocity, nodal

    def residual(self, z):
        self.validate(z)
        d, rows = self.driver, self.rows
        y, x, _ = self.unpack(z); bp = self.cap_pressure(z)
        force = d.solid.force(x, self.time)+rows.reaction(bp)
        d.transfer.reset_warm_start()
        remainder, amplitude, spread = d.transfer.relative_parts(force, bp, self.stencil)
        known_pressure = amplitude*self.stencil.potential
        velocity, pressure, flow = d.flow.advance(self.departure, remainder,
                                                self.pressure_initial-known_pressure)
        pressure = pressure+known_pressure
        # Retain the full forcing for independent work/equation audits.
        density = tuple(f+amplitude*g for f, g in zip(remainder, self.stencil.gradient))
        nodal, interpolation = d.transfer.interpolate(velocity, self.stencil)
        # Elementary row operation using y=dt*I(u): enforce the same closed
        # relative flux with I(u) explicitly, in normal-velocity units. This
        # resolves pressure work without forcing all displacement components
        # below the floating-point floor. The physical dy-based flux is still
        # checked independently after the solve.
        residual = torch.cat(((y-d.flow.dt*nodal).flatten(),
            self.cap_row_units((rows.flow(velocity)-rows.material_flow(nodal))/rows.area)))
        if not torch.isfinite(residual).all():
            raise FloatingPointError('nonfinite balanced cap residual')
        self.pressure_initial = pressure
        self.last_z, self.last_residual = z.clone(), residual.clone()
        self.last_cap_pressure = bp.clone()
        self.last_data = x, 0., self.load.volume(x), force, velocity, pressure, flow, nodal, density, spread, interpolation
        self.evaluations += 1
        return residual

    def linearization(self, z):
        self.accepted(z)
        d, rows = self.driver, self.rows; x = self.unpack(z)[1]
        if d.assembler is None: d.assembler = d.model.tangent_factory()
        tangent = d.assembler.assemble(x, self.time); self.tangent_assemblies += 1
        def action(v):
            self.jacobian_actions += 1
            dy = v[:self.mechanical_size].reshape_as(x)
            df = torch.sparse.mm(tangent, dy.reshape(-1, 1)).reshape_as(x)
            du, dw = self.sensitivity(df, self.cap_scale*v[self.mechanical_size:])
            cap = self.cap_row_units((rows.flow(du)-rows.material_flow(dw))/rows.area)
            return torch.cat(((dy-d.flow.dt*dw).flatten(), cap))
        return normalized_linear_action(action)

    def preconditioner(self, z):
        lu, pivots = torch.linalg.lu_factor(self.cap_h)
        def apply(v):
            self.preconditioner_applications += 1
            q = torch.linalg.lu_solve(lu, pivots,
                (self.driver.flow.dt*self.cap_row_units(v[self.mechanical_size:], inverse=True))[:, None]).flatten()
            return torch.cat((v[:self.mechanical_size]-self.cap_b@q, q))
        return apply


@torch.no_grad()
def solve(driver, settings=None, *, progress=None):
    _check_driver(driver)
    settings = settings or PreloadConfig(nominal_steps=100, minimum_step_mmhg=.005,
                                         force_atol_dyn=1e-5, force_rtol=1e-10)
    if settings.nominal_steps < 1:
        raise ValueError('nominal source strain loading endpoints required')
    history = SourceLoadingHistory(driver, settings.target_mmhg,
                                   driver.config.start_time_s, settings.nominal_steps)
    endpoint = driver._fsi_stepper(driver.maximum_ticks)
    loader = PassivePreloader(driver, settings, pressure_load=EndocardialAndCapPressure(driver.model),
        geometry_validator=lambda old, x: endpoint.check_support(x))
    result = loader.solve(progress, observe_state=history.observe)
    return result, history.final()


@torch.no_grad()
def validate(driver, result, history, *, steps=3, level=2, guard_cells=2):
    """Fresh preload/history plus solved cap pressures; no EP/circulation march.

    Returns a reviewable static initialization and discarded trial evidence.
    This driver has no production checkpoint/EP-cycle support yet.
    """
    _check_driver(driver)
    if (type(steps) is not int or not 1 <= steps <= 20
            or history.pressure_mmhg != result.pressure_mmhg
            or not torch.equal(history.x, result.x)):
        raise ValueError('matching recomputed preload history and 1..20 trials required')
    # Static pressure-work verification resolves velocities much smaller than
    # a production displacement step. Tighten this discarded trial; do not
    # relax targets or change the user's production configuration.
    original = driver.config.fsi.nonlinear
    # On the 128^3 real mesh, follower-force cancellation reaches a roughly
    # 1.3e-11 combined-residual floor. Physical targets below are independent
    # and unchanged; demanding 1e-12 here only causes futile backtracking.
    strict = replace(original, rtol=min(original.rtol, 1e-12), atol=min(original.atol, 3e-11),
        linear=replace(original.linear, atol=min(original.linear.atol, 1e-15)))
    config = replace(driver.config, fsi=replace(driver.config.fsi, nonlinear=strict))
    free = IsovolumicAFSEIStepper(driver.mesh, config)
    free.closed_cap = True; free.cap_patch_width_cells = 3.
    state = free.initialize(x=result.x)
    strain = StrainHistory(history.stretch.clone(), history.zs.clone(), history.zw.clone())
    if not torch.allclose(state.strain.stretch, strain.stretch, rtol=1e-11, atol=1e-11):
        raise ValueError('new preload stretch/history mismatch')
    load = replace(state.force_load, use_land=False, zs_old=strain.zs.clone(), zw_old=strain.zw.clone())
    free.buffer.stage(load)
    material = free.model.force(result.x, state.fsi.time)
    state = replace(state, strain=strain, force_load=load, fsi=replace(state.fsi, force=material))
    e = free._fsi_stepper(free.maximum_ticks)
    e.transfer = PressureModeTransfer(e, level=level)
    stencil = e.transfer.prepare(result.x)
    bp = torch.full_like(stencil.rows.area, result.pressure_mmhg*C)
    cap_reaction = stencil.rows.reaction(bp)
    explicit_cap = result.pressure_mmhg*C*EndocardialAndCapPressure(free.model).cap_covector(result.x)
    cap_error = (cap_reaction-explicit_cap).norm().item()
    equilibrium = material+MACEndocardialPressure(free.model).force(result.x, result.pressure_mmhg)+cap_reaction
    if cap_error > 1e-7 or equilibrium.norm().item() > result.force_tolerance_dyn:
        raise ValueError('preload cap reaction differs from the actual coupled reaction')
    e.transfer.reset_warm_start()
    remainder, amplitude, _ = e.transfer.relative_parts(material+cap_reaction, bp, stencil)
    u, p, _ = e.flow.solve_rhs(tuple(e.flow.dt/e.flow.rho*f for f in remainder))
    p = p+amplitude*stencil.potential
    # p is SOLVED from material and actual cap reactions; not set to target.
    initial = replace(state, fsi=replace(state.fsi, force=material+cap_reaction, pressure=p))
    _, core = cavity_masks(result.x, free.model.endo_faces, e.grid, guard_cells=guard_cells)
    trial = initial.fsi; records = []
    for _ in range(steps):
        if result.x.is_cuda: torch.cuda.synchronize(result.x.device)
        started = perf_counter()
        try:
            trial, r = traction_free_step(e, trial, problem_class=BalancedCapProblem)
        except NonlinearFailure as exc:
            exc.add_note(f"Static trial: residual={exc.result.residual_norm:g}, "
                         f"tolerance={exc.result.tolerance:g}; "
                         f"history={[row['residual_norm'] for row in exc.result.history]}")
            raise
        if result.x.is_cuda: torch.cuda.synchronize(result.x.device)
        r.update(trial_seconds=perf_counter()-started,
            pressure_core=pressure_statistics(trial.pressure, core),
            displacement_from_preload_l2_cm=(trial.x-result.x).norm().item(),
            volume_change_from_preload_ml=r['volume_ml']-result.volume_ml)
        records.append(r)
    targets = HandoffTargets()
    failed = []
    for key, limit, name in (
        ('displacement_from_preload_l2_cm', targets.displacement_l2_cm, 'static_displacement'),
        ('max_velocity_cm_s', targets.velocity_max_cm_s, 'static_velocity'),
        ('volume_change_from_preload_ml', targets.volume_error_ml, 'closed_volume'),
        ('closed_cap_flux_norm_ml_s', targets.closed_cap_flux_norm_ml_s, 'closed_cap_flux'),
        ('cap_work_adjoint_relative_error', targets.work_relative_error, 'cap_work_adjoint'),
        ('cap_closed_power_relative_error', targets.work_relative_error, 'closed_cap_power')):
        if any(abs(r[key]) > limit for r in records): failed.append(name)
    if any(abs(r['pressure_core']['mean_mmhg']-result.pressure_mmhg) > targets.pressure_error_mmhg
           for r in records): failed.append('physical_pressure')
    if max(v.abs().max().item() for v in u) > targets.velocity_max_cm_s:
        failed.append('initial_force_only_velocity')
    if stencil.report['correction_net_force_norm_dyn'] > 1e-6:
        failed.append('rigid_translation_consistency')
    if stencil.report['correction_torque_norm_dyn_cm'] > 1e-6:
        failed.append('rigid_rotation_consistency')
    report = dict(status='completed', scope='pressure-mode corrected closed-cap static preload validation',
        static_targets_passed=not failed, failed_targets=failed, targets=targets.__dict__,
        preload_pressure_mmhg=result.pressure_mmhg, preload_volume_ml=result.volume_ml,
        reference_volume_ml=result.reference_volume_ml, preload_force_residual_dyn=equilibrium.norm().item(),
        preload_force_tolerance_dyn=result.force_tolerance_dyn, uniform_cap_reaction_error_dyn=cap_error,
        loading_history_steps=history.steps, loading_history_duration_s=history.duration_s,
        land_strain_history_recomputed=True, reused_old_loading_history=False,
        initial_pressure_core=pressure_statistics(p, core),
        initial_force_only_velocity_max_cm_s=max(v.abs().max().item() for v in u),
        initial_zs_max_abs=strain.zs.abs().max().item(), initial_zw_max_abs=strain.zw.abs().max().item(),
        transfer_mode=stencil.report, trials=records,
        static_trial_nonlinear_rtol=strict.rtol, static_trial_nonlinear_atol_cm=strict.atol,
        production_nonlinear_tolerance_changed=False,
        static_cap_residual_units='cm/s; motion-equivalent row, area-mean mode scaled by 16',
        static_cap_area_mean_row_scale=16,
        prescribed_wall_pressure_during_fsi_mmhg=0., paired_cap_material_reactions_solved=True,
        production_driver_enabled=False, spatial_convergence_established=False,
        pointwise_wall_impermeability_established=False, ep_advanced=False, circulation_advanced=False,
        full_fsi_outer_iterations_added=False, new_fluid_volume_mesh_added=False,
        source_pk1_or_active_unloading_changed=False, full_cardiac_cycle_implemented=False)
    return free, initial, bp, report


def save_result(path, driver, state, cap_pressure, report, loading_history):
    """Atomic, checksummed validation artifact; ordinary checkpoint.load rejects it.

    Only a statically validated unadvanced initialization can be published.
    This is deliberately a preload RESULT, not an enabled production driver.
    """
    initial_tick = round(driver.config.start_time_s/driver.protocol.dt_ep_s)
    if (report.get('static_targets_passed') is not True or state.tick != initial_tick or state.fsi.step != 0
            or state.fsi.time != driver.config.start_time_s
            or state.force_load.use_land or driver.config.prescribed_endocardial_pressure_mmhg != 0
            or any(v.abs().max() != 0 for v in state.fsi.velocity)):
        raise ValueError('only a validated unadvanced pressure-supported preload result can be saved')
    e = driver._fsi_stepper(driver.maximum_ticks)
    if not isinstance(e.transfer, PressureModeTransfer):
        raise ValueError('pressure-mode execution must accompany a preload result')
    array = lambda v: v.detach().cpu().numpy()
    data = {name: array(getattr(driver.mesh, name)) for name in MESH}
    data.update({name: array(getattr(state.fsi, name)) for name in ('x', 'pressure', 'force')})
    data.update({name: array(getattr(state.tissue, name)) for name in ('voltage_v', 'gpb', 'land')})
    data.update({name: array(getattr(state.strain, name)) for name in ('stretch', 'zs', 'zw')})
    data['cap_pressure'] = array(cap_pressure)
    metadata = dict(producer='afsei-mac-pressure-mode-preload-result', schema=1,
        completed=True, units='cm-g-s; V; microA/cm2; GPB-ms-mV',
        mesh_binding_sha256=mesh_binding(driver.mesh), config=asdict(driver.config),
        time_s=state.fsi.time, tick=state.tick, pressure_mode_level=e.transfer.level,
        report=dict(report, candidate_result_written=True), loading_history=loading_history,
        production_checkpoint=False, production_driver_enabled=False,
        land_strain_history_recomputed=True)
    metadata['sha256'] = digest(metadata, data)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): raise FileExistsError('preload result already exists')
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, metadata=json.dumps(metadata, allow_nan=False), **data)
    temporary.replace(path)


def read_result(path):
    """Verify an artifact for review/import; does not enable a cycle driver."""
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    try:
        metadata = json.loads(str(data.pop('metadata')))
        sha = metadata.pop('sha256')
        if (metadata.get('producer') != 'afsei-mac-pressure-mode-preload-result'
                or metadata.get('schema') != 1 or metadata.get('completed') is not True
                or metadata.get('report', {}).get('static_targets_passed') is not True
                or metadata.get('production_checkpoint') is not False
                or digest(metadata, data) != sha):
            raise ValueError('invalid pressure-mode preload result/schema/checksum')
        required = {*MESH, 'x', 'pressure', 'force', 'voltage_v', 'gpb', 'land',
                    'stretch', 'zs', 'zw', 'cap_pressure'}
        if set(data) != required or any(a.dtype != (np.int64 if name in INTEGER else np.float64)
               or not np.isfinite(a).all() for name, a in data.items()):
            raise ValueError('invalid pressure-mode preload result arrays')
        return metadata, data
    except (KeyError, TypeError) as exc:
        raise ValueError('malformed pressure-mode preload result') from exc


def save_loading_result(path, driver, result, history):
    """Persist completed static loading before the separate handoff can fail.

    This artifact has NO MAC/EP continuation state or static acceptance claim.
    It can only skip the same reference loading path in this validation tool.
    """
    data = {name: getattr(driver.mesh, name).detach().cpu().numpy() for name in MESH}
    data.update({name: getattr(history, name).detach().cpu().numpy()
                 for name in ('x', 'stretch', 'zs', 'zw')})
    metadata = dict(producer='afsei-mac-balanced-loading-path', schema=1,
        mesh_binding_sha256=mesh_binding(driver.mesh), production_checkpoint=False,
        report=dict(preload_pressure_mmhg=result.pressure_mmhg,
            preload_force_tolerance_dyn=result.force_tolerance_dyn,
            loading_history_steps=history.steps, loading_history_duration_s=history.duration_s),
        loading_history=result.history, rejected=result.rejected)
    metadata['sha256'] = digest(metadata, data)
    path = Path(path)
    if path.exists(): raise FileExistsError('loading result already exists')
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, metadata=json.dumps(metadata, allow_nan=False), **data)
    temporary.replace(path)


def _read_loading_result(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    metadata = json.loads(str(data.pop('metadata')))
    if metadata.get('producer') == 'afsei-mac-pressure-mode-preload-result':
        return read_result(path)
    sha = metadata.pop('sha256', None)
    if (metadata.get('producer') != 'afsei-mac-balanced-loading-path'
            or metadata.get('schema') != 1 or metadata.get('production_checkpoint') is not False
            or digest(metadata, data) != sha or set(data) != {*MESH, 'x', 'stretch', 'zs', 'zw'}
            or any(a.dtype != (np.int64 if name in INTEGER else np.float64)
                   or not np.isfinite(a).all() for name, a in data.items())):
        raise ValueError('invalid balanced loading path/schema/checksum')
    return metadata, data


@torch.no_grad()
def reuse_validated_loading(driver, path):
    """Validate/reuse THIS completed loading path, never the old solid-only path.

    Revalidate mesh, passive/cap force and final stretch. No EP/cycle driver is
    restored or enabled. A completed loading-only result does not need prior
    handoff acceptance; the caller MUST perform that separate validation.
    A changed material model fails the current force test.
    """
    _check_driver(driver)
    metadata, data = _read_loading_result(path)
    report = metadata['report']
    if metadata['mesh_binding_sha256'] != mesh_binding(driver.mesh):
        raise ValueError('balanced preload reference mesh/fields mismatch')
    for name in MESH:
        if not np.array_equal(data[name], getattr(driver.mesh, name).cpu().numpy()):
            raise ValueError('balanced preload mesh/boundary mismatch')
    tensor = lambda name: torch.as_tensor(data[name], device=driver.mesh.X.device)
    x = tensor('x'); driver.model.validate(x)
    zero = torch.zeros_like(driver.mesh.X[:, 0])
    driver.buffer.stage(ActiveLoadState(driver.mesh.X, zero, zero, zero, zero,
                                      driver.config.fsi.time.dt, False))
    p = report['preload_pressure_mmhg']
    load = EndocardialAndCapPressure(driver.model)
    residual = (driver.model.force(x, driver.config.start_time_s)+load.force(x, p)).norm().item()
    if residual > report['preload_force_tolerance_dyn']:
        raise ValueError('balanced preload fails current passive/cap equilibrium')
    F = driver.model.element_gradient(x)
    cell = torch.linalg.vector_norm(torch.einsum('eij,ej->ei', F, driver.mesh.fiber), dim=-1)
    stretch = driver.ep.project_cell_field(cell)
    if not torch.allclose(stretch, tensor('stretch'), rtol=1e-11, atol=1e-11):
        raise ValueError('balanced preload stretch/history mismatch')
    entries = metadata['loading_history']
    endpoints = [row for row in entries if row.get('physical_loading_endpoint')]
    steps, duration = report['loading_history_steps'], report['loading_history_duration_s']
    if len(endpoints) != steps or any(abs(row['pressure_mmhg']-p*(i+1)/steps) > 1e-10
            or abs(row['physical_loading_time_s']-duration*(i+1)/steps) > 1e-10
            for i, row in enumerate(endpoints)):
        raise ValueError('balanced preload loading endpoints inconsistent')
    h = PreloadStrainHistory(x, stretch, tensor('zs'), tensor('zw'), p, duration, steps)
    for name in ('zs', 'zw'):
        if abs(getattr(h, name).abs().max().item()-endpoints[-1][name+'_max_abs']) > 1e-10:
            raise ValueError('balanced preload distortion history mismatch')
    result = PreloadResult(x, p, residual, report['preload_force_tolerance_dyn'],
        load.volume(x).item(), load.volume(driver.mesh.X).item(), entries, metadata.get('rejected', []))
    return result, h
