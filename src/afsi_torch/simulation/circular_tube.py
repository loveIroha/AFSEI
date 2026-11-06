"""Electrically activated 3D circular tube, using existing AFSEI solvers.

One first-order EP/ODE-to-FSI split per time step. Mechanics and flow are solved
implicitly by the unmodified BEIBStepper; EP diffusion uses the configured CN
nodal FEM operator. There are no porous domains, valves or prescribed motion.
"""
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
import json
import math
import xml.etree.ElementTree as ET
import numpy as np
import torch
from .circular_tube_config import resolve_config
from .circular_tube_restart import restart_plan
from ..solids.circular_tube import generate_circular_tube, section_areas
from ..ep.circular_tube import CircularTubeSolid
from ..ep.planar_tube import land_resting_equilibrium
from ..ep.land import distortion_step, land_tension
from ..ep.monodomain import Monodomain
from ..ep.tissue import GPBLand, initial_state
from ..mac.paper_coupling import BEIBStepper
from ..mac.grid import divergence
from ..mac.output import MACWriter
from ..cycle_checkpoint import atomic_json


def _lumen_flow(grid, velocity, x, ring):
    """Approximate +z flow through a moving section's projected polygon.

    Interpolate between two MAC z-face planes; use cell-center inclusion in
    the x-y polygon. This is an output diagnostic, never a BC or constraint.
    """
    p = x[ring]
    xf = grid.origin[0]+grid.spacing[0]*(torch.arange(grid.shape[0], device=x.device, dtype=x.dtype)+.5)
    yf = grid.origin[1]+grid.spacing[1]*(torch.arange(grid.shape[1], device=x.device, dtype=x.dtype)+.5)
    qx, qy = torch.meshgrid(xf, yf, indexing='ij')
    a, b = p[:, :2], p.roll(-1, dims=0)[:, :2]
    dy = b[:, 1]-a[:, 1]
    safe_dy = torch.where(dy.abs() > 1e-30, dy, torch.ones_like(dy))
    intercept = a[:, 0]+(qy[..., None]-a[:, 1])*(b[:, 0]-a[:, 0])/safe_dy
    crosses = ((a[:, 1] > qy[..., None]) != (b[:, 1] > qy[..., None])) & (qx[..., None] < intercept)
    inside = (crosses.sum(-1) % 2) == 1
    location = float((p[:, 2].mean()-grid.origin[2])/grid.spacing[2])
    lo = math.floor(location)
    if not 0 <= lo < grid.shape[2]:
        raise ValueError('moving tube section left the MAC box')
    alpha = location-lo
    uz = (1-alpha)*velocity[2][:, :, lo]+alpha*velocity[2][:, :, lo+1]
    return float((uz*inside).sum()*grid.spacing[0]*grid.spacing[1])


def _ep_collection(folder, frames):
    root = ET.Element('VTKFile', type='Collection', version='0.1', byte_order='LittleEndian')
    collection = ET.SubElement(root, 'Collection')
    for time, name in frames:
        ET.SubElement(collection, 'DataSet', timestep=repr(time), group='', part='0', file=name)
    temporary = folder/'ep.pvd.tmp'
    ET.ElementTree(root).write(temporary, encoding='utf-8', xml_declaration=True)
    temporary.replace(folder/'ep.pvd')


@torch.no_grad()
def run(config, output, *, device='cuda', passive=False, no_stimulus=False, resume=None, resume_refine_time=False):
    if resume_refine_time and resume is None:
        raise ValueError("time-refinement restart requires a source checkpoint")
    config = json.loads(json.dumps(config))
    if passive:
        config['active']['tension_scale'] = 0.
    if no_stimulus:
        config['stimulus']['amplitude_ua_per_cm2'] = 0.
    geometry, fsi, parameters, numerics = resolve_config(config)
    if torch.device(device).type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('the existing GPB/Land CUDA kernels require a CUDA device')
    device = torch.device(device)
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('choose a new or empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out/'config.resolved.json', config)
    atomic_json(out/'fsi.resolved.json', asdict(fsi))
    setup_start = perf_counter()
    mesh = generate_circular_tube(geometry, device=device)
    original_mesh = mesh
    solid_factor = config.get('solid_refinement_factor', 1)
    field_transfer = None
    if solid_factor != 1:
        from ..solids.refined_circular_tube import refine_circular_tube
        mesh, field_transfer = refine_circular_tube(mesh, geometry, factor=solid_factor)
    layered = 'layers' in config['solid']
    if layered:
        from ..ep.layered_circular_tube import LayeredCircularTubeSolid
    solid_type = LayeredCircularTubeSolid if layered else CircularTubeSolid
    solid = solid_type(mesh, geometry,
        **{k:v for k,v in config['solid'].items() if k != 'end_condition'},
        **{k:v for k,v in config['active'].items() if k not in ('mechanism', 'rest_reference')},
        dt_s=fsi.time.dt,
        **(dict(end_support=config['solid']['end_condition'], mechanism=config['active']['mechanism']) if not layered else {}))
    if field_transfer is not None:
        solid.field_transfer = field_transfer
    ep_nodes = solid.ep_nodes if layered else torch.arange(len(original_mesh.X),device=mesh.X.device)
    ep_cells = solid.ep_cells if layered else original_mesh.cells
    ep_cell_ids = solid.ep_cell_ids if layered else torch.arange(len(original_mesh.cells),device=mesh.X.device)*solid_factor**3
    ep_X = mesh.X[ep_nodes]
    # Conductivity is axial; muscle stress is circumferential. These are two
    # explicit, different fields rather than an implicit reuse of mesh.fiber.
    radial = mesh.X[:, :2]-mesh.X.new_tensor(geometry.center_xy_cm)
    radial = radial/torch.linalg.vector_norm(radial, dim=-1, keepdim=True)
    sheet = torch.cat((radial, torch.zeros_like(radial[:, :1])), -1)
    axial = mesh.X.new_tensor((0., 0., 1.)).expand_as(mesh.X)
    ep = Monodomain(ep_X, ep_cells, axial[ep_nodes], sheet[ep_nodes], parameters=parameters, numerics=numerics)
    ep_deformation_feedback=config['ep'].get('deformation_feedback','none')
    ep_geometry_updates=0
    def ep_geometric_stretch(x):
        if field_transfer is None:
            return solid.nodal_stretch(ep, x)
        # Diagnostic restriction only: equal reference volumes within each parent.
        coarse_stretch = solid.cell_stretch(x).reshape(-1, solid_factor**3).mean(1)
        return ep.project_cell_field(coarse_stretch)
    history, ep_frames = [], []
    state = driver = tissue = zs = zw = stretch = activation = None
    completed_ep_steps = 0
    completed_ode_substeps = 0
    restart_info = None
    predictor_scale = 1.
    previous_dt_s = fsi.time.dt
    setup_seconds = elapsed_seconds = 0.
    report = None
    budget = None
    initial_step = 0

    def save_state(name):
        if state is None:
            return
        array = lambda v: v.detach().cpu().numpy()
        temporary = out/(name+'.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, step=state.step, time_s=state.time,
                x=array(state.x), reference=array(mesh.X), cells=array(mesh.cells),
                inner_rings=array(mesh.inner_rings), inner_faces=array(mesh.inner_faces), end_faces=array(mesh.end_faces),
                previous_x=array(state.previous_x if state.previous_x is not None else state.x),
                u=array(state.velocity[0]), v=array(state.velocity[1]), w=array(state.velocity[2]),
                pressure=array(state.pressure), force=array(state.force),
                force_time_s=state.force_time, voltage_v=array(tissue.voltage_v),
                gpb=array(tissue.gpb), land=array(tissue.land), zs=array(zs), zw=array(zw),
                stretch=array(stretch), activation_time_s=array(activation),
                resting_tension_dyn_cm2=array(resting_tension), ep_steps=completed_ep_steps,
                fsi_steps=completed_ep_steps, ode_substeps=completed_ode_substeps, previous_dt_s=previous_dt_s,
                ep_node_ids=array(ep_nodes), ep_solid_cell_ids=array(ep_cell_ids),
                **(budget.checkpoint() if budget is not None else {}),
                **(driver.flow.checkpoint() if 'circulation' in config else {}))
        temporary.replace(out/name)

    def sample(info=None):
        info = info or {}
        nonlinear = info.get('nonlinear', {})
        div = divergence(state.velocity, driver.grid.spacing)
        return dict(step=state.step, time_s=state.time, **solid.diagnostics(state.x),
            inlet_section_flow_ml_s=_lumen_flow(driver.grid, state.velocity, state.x, mesh.inner_rings[0]),
            mid_section_flow_ml_s=_lumen_flow(driver.grid, state.velocity, state.x, mesh.inner_rings[len(mesh.inner_rings)//2]),
            outlet_section_flow_ml_s=_lumen_flow(driver.grid, state.velocity, state.x, mesh.inner_rings[-1]),
            voltage_min_mv=float(tissue.voltage_v.min()*1000), voltage_max_mv=float(tissue.voltage_v.max()*1000),
            ep_last_staged_min_J=float(ep.volume_ratio.min()),
            ep_last_staged_max_J=float(ep.volume_ratio.max()),
            calcium_min_um=float(tissue.gpb[:, 37].min()*1000), calcium_max_um=float(tissue.gpb[:, 37].max()*1000),
            land_input_stretch_min=float(stretch.min()), land_input_stretch_max=float(stretch.max()),
            land_zs_min=float(zs.min()), land_zs_max=float(zs.max()),
            land_zw_min=float(zw.min()), land_zw_max=float(zw.max()),
            activated_fraction=float(torch.isfinite(activation).double().mean()),
            max_velocity_cm_s=max(float(u.abs().max()) for u in state.velocity),
            max_divergence_s_inv=float(div.abs().max()),
            nonlinear_iterations=nonlinear.get('iterations', 0),
            nonlinear_residual=nonlinear.get('residual_norm', 0.),
            nonlinear_tolerance=nonlinear.get('tolerance', 0.),
            fluid_solves=nonlinear.get('fluid_solves', 0),
            pressure_cycles=nonlinear.get('pressure_cycles', 0),
            gmres_iterations=nonlinear.get('gmres_iterations', 0),
            courant=info.get('courant', 0.),
            **(driver.flow.diagnostics() if 'circulation' in config else {}),
            **(domain.diagnostics() if domain is not None else {}),
            **(budget.diagnostics(state.velocity) if budget is not None else {}))

    def write_frame():
        if writer is None:
            return
        writer.write(state)
        name = f'ep_{state.step:06d}.vtu'
        filename = out/'vtk'/name
        temporary = filename.with_suffix('.vtu.tmp')
        array = lambda v: v.detach().cpu().numpy()
        writer.meshio.write(temporary, writer.meshio.Mesh(array(state.x[ep_nodes]), [('tetra', array(ep_cells))],
            point_data=dict(voltage_mV=array(tissue.voltage_v)*1000,
                calcium_uM=array(tissue.gpb[:, 37])*1000,
                circumferential_stretch=array(ep_geometric_stretch(state.x)),
                land_input_stretch=array(stretch),
                activation_time_s=array(torch.where(torch.isfinite(activation), activation,
                                                    torch.full_like(activation, -1.)))),
            cell_data=dict(mechanical_fiber_reference=[array(mesh.fiber[ep_cell_ids])])),
            file_format='vtu', binary=True, compression='zlib')
        temporary.replace(filename)
        ep_frames.append((state.time, name))
        _ep_collection(out/'vtk', ep_frames)

    writer = None
    domain = None
    try:
        ep.prepare(config['time']['ep_dt_s'])
        ode = GPBLand(config['time']['ep_dt_s'], config['time']['ode_dt_ms'],
                      method=config['ep'].get('ode_method', 'sdirk'))
        tissue = initial_state(ep_X[:, 0])
        stretch = solid.activation_stretch(torch.ones_like(tissue.voltage_v))
        zs, zw = torch.zeros_like(stretch), torch.zeros_like(stretch)
        tissue = replace(tissue, land=land_resting_equilibrium(tissue.gpb[:, 37]*1000, stretch))
        resting_tension = land_tension(stretch, stretch, tissue.land[:, 0], tissue.land[:, 1], zs, zw, fsi.time.dt)
        solid.set_reference(resting_tension)
        solid.stage(mesh.X, tissue, zs, zw)
        driver = BEIBStepper(solid, fsi, device)
        if 'hydraulic_domain' in config:
            from ..mac.sealed_ports import SealedDomain
            from ..mac.sealed_flow import SealedBEFlow
            domain = SealedDomain(driver.grid, geometry, config['hydraulic_domain'], mesh.X, dimension=3)
            domain.bind_wall(mesh)
            domain.validate_wall(mesh.X)
            if fsi.output.write_vtk: domain.write_geometry(out/'vtk')
            driver.flow = SealedBEFlow(driver.flow, domain,
                fused=fsi.execution.execution_backend == 'fused', graphs=fsi.execution.pressure_backend == 'graph',
                helmholtz_solver=config['fsi']['flow'].get('helmholtz_solver','jacobi'))
            atomic_json(out/'hydraulic_domain.json', domain.summary())
        if 'circulation' in config:
            from ..circulation.rc_outlet import RCBoxFlow, parse_rc
            driver.flow = RCBoxFlow(driver.flow, parse_rc(config['circulation'], dimension=3), mesh.X)
        state = driver.initialize(mesh.X)
        if domain is not None:
            from .tube_budget import TubeBudget
            budget = TubeBudget(mesh, domain)
        activation = torch.full_like(stretch, float('inf'))
        if resume is not None:
            source=Path(resume)
            old=json.loads((source.parent/'config.resolved.json').read_text())
            if domain is not None or 'circulation' in config:
                raise ValueError('this resume adapter is for the open single tube only')
            with np.load(source) as a:
                tensor=lambda name:torch.as_tensor(a[name].copy(),device=device,dtype=mesh.X.dtype)
                if not np.array_equal(a['reference'],mesh.X.cpu().numpy()) or not np.array_equal(a['cells'],mesh.cells.cpu().numpy()):
                    raise ValueError('resume mesh mismatch')
                restart_info=restart_plan(old,config,a,refine_time=resume_refine_time)
                initial_step=restart_info['solver_step_index'];clock=restart_info['clock_s']
                predictor_scale=restart_info['predictor_scale']
                previous_dt_s=restart_info['previous_dt_s']
                completed_ode_substeps=restart_info['accepted_ode_substeps']
                state=replace(state,step=initial_step,time=clock,x=tensor('x'),previous_x=tensor('previous_x'),
                    velocity=tuple(tensor(n) for n in ('u','v','w')),pressure=tensor('pressure'),
                    force=tensor('force'),force_time=float(a['force_time_s']))
                tissue=replace(tissue,voltage_v=tensor('voltage_v'),gpb=tensor('gpb'),land=tensor('land'))
                zs,zw,stretch,activation=(tensor(n) for n in ('zs','zw','stretch','activation_time_s'))
                resting_tension=tensor('resting_tension_dyn_cm2');solid.set_reference(resting_tension)
                completed_ep_steps=restart_info['accepted_ep_steps']
                solid.stage(state.x,tissue,zs,zw)
        stim = config['stimulus']
        stimulus_mask = (ep_X[:, 2] <= geometry.axial_start_cm+stim['width_cm']+1e-12).to(mesh.X.dtype)
        if not stimulus_mask.count_nonzero():
            raise ValueError('stimulus contains no EP nodes')
        writer = MACWriter(out/'vtk', solid, driver.grid,
            pressure_description=('physical projection pressure; sealed end reservoirs; central exterior p=0; impermeable fixed septa/collars' if domain is not None else 'physical projection pressure; high-z full outer box face RC; other faces p=0; tube mouths remain interior'
                if 'circulation' in config else ('Eulerian pressure multiplier pi; in pure fluid pi is physical pressure; stabilized wall mean physical pressure is pi-kappa*ln(J)/J; outer box p=0' if getattr(solid, 'volumetric_stabilization', None) is not None else 'physical projection pressure; p=0 on every outer box face; original 3D BE MAC'))) if fsi.output.write_vtk else None
        history.append(sample()); write_frame()
        torch.cuda.synchronize(device)
        setup_seconds = perf_counter()-setup_start
        started = perf_counter()
        steps = round(fsi.time.end_time/fsi.time.dt)
        for k in range(initial_step,steps):
            if ep_deformation_feedback=='santiago2018':
                ep.update_deformation(state.x[ep_nodes])
                ep_geometry_updates+=1
            overlap = max(0., min(state.time+fsi.time.dt, stim['start_s']+stim['duration_s'])-max(state.time, stim['start_s']))
            stimulus = stimulus_mask*(stim['amplitude_ua_per_cm2']*overlap/fsi.time.dt)
            trial_tissue, _, _ = ode.step(ep, tissue, stretch, zs, zw, state.time, stimulus)
            solid.stage(state.x, trial_tissue, zs, zw)
            diagnostics = (k+1) % fsi.output.log_every == 0 or k+1 == steps
            # Rescale only the first displacement predictor after a dt change.
            # Keep the checkpoint's physical previous_x intact until acceptance.
            step_state = state
            if k == initial_step and predictor_scale != 1. and state.previous_x is not None:
                step_state = replace(state, previous_x=state.x-predictor_scale*(state.x-state.previous_x))
            trial_state, info = driver.step(step_state, diagnostics=diagnostics)
            trial_stretch = (solid.activation_stretch(torch.ones_like(stretch)) if field_transfer is not None
                             else solid.activation_stretch(solid.nodal_stretch(ep, trial_state.x)))
            trial_zs = distortion_step(zs, trial_stretch-stretch, fsi.time.dt, 40.14)
            trial_zw = distortion_step(zw, trial_stretch-stretch, fsi.time.dt, 405.86)
            crossed = (tissue.voltage_v < -.02) & (trial_tissue.voltage_v >= -.02) & torch.isinf(activation)
            fraction = (-.02-tissue.voltage_v)/(trial_tissue.voltage_v-tissue.voltage_v).clamp_min(1e-30)
            # Publish only after BOTH EP and the implicit fluid/solid solve succeed.
            activation = torch.where(crossed, state.time+fsi.time.dt*fraction, activation)
            if domain is not None:
                domain.validate_wall(trial_state.x)
            if 'circulation' in config:
                driver.flow.commit(trial_state.velocity, step=trial_state.step)
            if budget is not None:
                budget.accepted(state.x, trial_state.x, trial_state.velocity, fsi.time.dt)
            tissue, state = trial_tissue, trial_state
            stretch, zs, zw = trial_stretch, trial_zs, trial_zw
            completed_ep_steps += 1
            completed_ode_substeps += ode.substeps
            previous_dt_s = fsi.time.dt
            if diagnostics:
                row = sample(info); history.append(row)
                atomic_json(out/'history.json', history)
                print(f"Circular tube step={state.step}, t={state.time*1000:.3f} ms, "
                      f"V={row['lumen_volume_ml']:.6g} ml, r/r0={row['minimum_radius_ratio']:.5f}, "
                      f"Vm_max={row['voltage_max_mv']:.2f} mV, "
                      f"dTa_max={row['maximum_applied_tension_dyn_cm2']:.5g} dyn/cm2", flush=True)
            if state.step % fsi.output.output_every == 0 or k+1 == steps:
                write_frame()
            if state.step % fsi.output.checkpoint_every == 0:
                save_state('checkpoint.npz')
        torch.cuda.synchronize(device)
        elapsed_seconds = perf_counter()-started
        report = dict(status='completed', accepted_steps=completed_ep_steps, solver_step_index=state.step, time_s=state.time,
            ep_steps=completed_ep_steps, ode_substeps=completed_ode_substeps, restart=restart_info,
            scope='simplified 3D electrically activated immersed circular tube; no valves or porous media',
            gpu=torch.cuda.get_device_name(device), solid_nodes=len(mesh.X), solid_cells=len(mesh.cells),
            fluid_shape=driver.grid.shape, setup_seconds=setup_seconds, elapsed_seconds=elapsed_seconds,
            accepted_steps_this_run=state.step-initial_step,
            resumed_from=str(resume) if resume is not None else None,
            mean_seconds_per_step_including_output=elapsed_seconds/max(state.step-initial_step,1),
            three_field_time_integration=True, fluid_solid_solver='unmodified BEIBStepper',
            spatial_coupling=config['fsi']['spatial_coupling'],
            fsi_time_scheme=config['fsi']['time_scheme'],
            viscosity_linear_solver=config['fsi']['flow'].get('helmholtz_solver','jacobi'),
            complete_griffith_luo_2017_time_scheme=False,
            cnab_vs_bebe_performance_measured=False,
            ep_discretization=numerics.discretization, ep_backend=numerics.backend,
            ep_time_scheme=numerics.time_scheme,
            ep_deformation_feedback=ep_deformation_feedback,
            ep_geometry_updates_this_run=ep_geometry_updates,
            ep_last_staged_min_J=float(ep.volume_ratio.min()),
            ep_last_staged_max_J=float(ep.volume_ratio.max()),
            ode_method=config['ep'].get('ode_method', 'sdirk'),
            ode_dt_ms=config['time']['ode_dt_ms'],
            coupling_time_scheme='first-order sequential EP/ODE then fully implicit BE fluid/solid',
            active_model=('Land isometric paired fixed-y reference-surface traction; no active PK1'
                if config['active']['mechanism']=='paired-transverse-land-traction' else
                'existing Land nominal PK1; circumferential DG0 fibers; positive resting-equilibrium increment'),
            muscle_stretch_feedback=solid.stretch_feedback,
            land_reference_ratio=solid.land_reference_ratio,
            mechanics_to_land_feedback=solid.stretch_feedback=='circumferential',
            ep_conduction=('Santiago Eq14 push-forward; exact reference diffusion pullback; current-volume mass/source; insulated surfaces'
                if ep_deformation_feedback=='santiago2018' else 'reference-mesh axial conductivity; insulated wall surfaces'),
            end_condition=config['solid']['end_condition'],
            material_layers=config['solid'].get('layers'),
            ep_nodes=len(ep_nodes), ep_cells=len(ep_cells), solid_refinement_factor=solid_factor,
            ep_to_solid_transfer='exact nested coarse P1 prolongation' if field_transfer is not None else 'same mesh',
            reference_geometry_changed_by_refinement=False,
            jelly_electrically_excitable=False if layered else None,
            fluid_boundary=config['fluid']['boundary'], exterior_fluid_present=True,
            porous_media=False, valves=False, contact_model=False, circulation='circulation' in config,
            rc_outlet_scope=config.get('circulation', {}).get('outlet_scope'),
            physical_tube_mouth_boundary_aligned=False,
            hydraulic_path_isolated=domain is not None,
            hydraulic_domain=domain.summary() if domain is not None else None,
            conservation_budget=budget.summary() if budget is not None else dict(enabled=False),
            rc_parameters_cgs=(dict(resistance_dyn_s_per_cm5=driver.flow.R, compliance_cm5_per_dyn=driver.flow.C)
                if 'circulation' in config else None),
            literature_replication=False, performance_comparison_performed=False,
            final=history[-1], output=writer.summary() if writer else dict(enabled=False))
        atomic_json(out/'report.json', report)
    except Exception as exc:
        atomic_json(out/'report.json', dict(status='failed', accepted_steps=completed_ep_steps, solver_step_index=state.step if state else 0,
            time_s=state.time if state else 0., ep_steps=completed_ep_steps,
            ode_substeps=completed_ode_substeps, restart=restart_info, error=str(exc)))
        raise
    finally:
        ep.close()
        atomic_json(out/'history.json', history)
        save_state('final_state.npz')
    return report
