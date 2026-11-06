"""Configured planar EP-GPB-Land / Neo-Hookean / MAC-MG demonstration."""
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import json
import math
import numpy as np
import torch
from ..solids.planar_tube import TubeGeometry, generate_tube, NeoHookeanTube
from ..ep.planar_tube import PairedHoopLoad, ActiveTubeSolid, land_resting_equilibrium
from ..ep.monodomain import Monodomain, MonodomainParameters
from ..ep.numerics import MonodomainNumerics
from ..ep.tissue import GPBLand, initial_state
from ..ep.land import distortion_step, land_tension
from ..fluid.solvers import SolverOptions
from ..mac2d.grid import ChannelGrid, divergence
from ..mac2d.pressure_channel import PressureChannelFlow
from ..mac2d.multigrid import MGOptions
from ..mac2d.transfer import TriangleTransfer
from ..mac2d.coupling import ValveStepper
from ..mac2d.output import fields, collection
from ..cycle_checkpoint import atomic_json


def validate_config(c):
    required = {'demo','geometry','fluid','solid','ep','stimulus','time','solver','output','active'}
    if not required <= set(c) or set(c)-required-{'execution', 'circulation', 'hydraulic_domain'} or c['demo'] != 'active-tube-2d':
        raise ValueError(f'configuration requires exactly {sorted(required)}')
    execution = c.get('execution', {})
    if set(execution)-{'profile'} or execution.get('profile','optimized') not in ('optimized','reference'):
        raise ValueError('execution.profile must be optimized or reference')
    if c['active']['mechanism'] != 'paired-hoop-land':
        raise ValueError('this demo explicitly implements the paired-hoop Land reduction')
    if c['active'].get('stretch_model','lumen-gap') not in ('lumen-gap','plane-strain-isometric'):
        raise ValueError('active.stretch_model must be lumen-gap or plane-strain-isometric')
    if c['solid'].get('end_condition','spring-approximation') not in ('spring-approximation','fixed-reference-ends','fixed-sealing-collars'):
        raise ValueError('unsupported solid.end_condition')
    if c['solid'].get('end_condition')=='fixed-sealing-collars' and 'hydraulic_domain' not in c:
        raise ValueError('fixed-sealing-collars requires the matching fixed hydraulic geometry')
    rc = c['fluid']['end_boundary'] == 'zero-inlet-pressure-rc-outlet'
    if rc:
        from ..circulation.rc_outlet import parse_rc
        parse_rc(c.get('circulation'), dimension=2)
        if c['fluid']['inlet_pressure_dyn_per_cm2'] != 0 or c['fluid']['outlet_pressure_dyn_per_cm2'] != 0:
            raise ValueError('RC config requires zero reference end pressures; outlet_pressure is not an extra load')
    elif c['fluid']['end_boundary'] != 'pressure-neumann-velocity' or 'circulation' in c:
        raise ValueError('unsupported boundary condition; select fixed pressures or explicit RC')
    expected_y = 'reference-pressure-vent-between-septa' if 'hydraulic_domain' in c else 'no-slip'
    if c['fluid']['outer_y_boundary'] != expected_y:
        raise ValueError('outer y boundary must be '+expected_y)
    if 'hydraulic_domain' in c:
        from ..mac.sealed_ports import validate_hydraulic
        validate_hydraulic(c['hydraulic_domain'])
        if not rc: raise ValueError('sealed reservoirs require the RC case')
    if c['ep']['numerics']['discretization'] not in ('nodal_p1','consistent_p1'):
        raise ValueError('active-tube requires nodal_p1 or consistent_p1 FEM')
    if c['ep'].get('ode_method','sdirk') not in ('sdirk','explicit_euler'):
        raise ValueError('ep.ode_method must be sdirk or explicit_euler')
    if c['ep'].get('deformation_feedback','none') not in ('none','santiago2018'):
        raise ValueError('ep.deformation_feedback must be none or santiago2018')
    dt, dep, end = c['time']['fsi_dt_s'], c['time']['ep_dt_s'], c['time']['end_time_s']
    if any(not math.isfinite(v) or v <= 0 for v in (dt,dep,end)):
        raise ValueError('positive finite time parameters required')
    for v in (dep/dt,end/dt):
        if round(v) < 1 or abs(round(v)-v) > 1e-8:
            raise ValueError('EP dt and duration must be integer multiples of FSI dt')
    if end < c['stimulus']['start_s'] or c['stimulus']['duration_s'] <= 0 or c['stimulus']['width_cm'] <= 0:
        raise ValueError('stimulus must have positive support and start before the run ends')
    if type(c['time'].get('stop_on_no_contact_limit',False)) is not bool:
        raise ValueError('time.stop_on_no_contact_limit must be boolean')
    if c['output']['log_every'] < 1 or c['output']['field_every'] < 1:
        raise ValueError('positive output intervals required')


@torch.no_grad()
def run(config, output, *, device='cuda', passive=False, no_stimulus=False, resume=None):
    validate_config(config)
    if not torch.cuda.is_available() or torch.device(device).type != 'cuda':
        raise RuntimeError('GPB/Land uses the existing CUDA kernel; this demo requires CUDA')
    config = json.loads(json.dumps(config))
    optimized = config.get('execution',{}).get('profile','optimized') == 'optimized'
    if not optimized:
        config['solver'].update(pressure_backend='reference',mass_backend='pcg',ib_backend='reference')
    if passive:
        config['active']['tension_scale'] = 0.
    if no_stimulus:
        config['stimulus']['amplitude_ua_per_cm2'] = 0.
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('choose an empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'config.resolved.json',config)
    setup_start = perf_counter()
    geometry = TubeGeometry(**config['geometry'])
    mesh = generate_tube(geometry,device=device)
    end_condition=config['solid'].get('end_condition','spring-approximation')
    passive_solid = NeoHookeanTube(mesh,fused=optimized,
        **{k:v for k,v in config['solid'].items() if k!='end_condition'})
    active_cfg = {k:v for k,v in config['active'].items() if k != 'mechanism'}
    active = PairedHoopLoad(mesh,geometry,fused=optimized,**active_cfg)
    solid = ActiveTubeSolid(passive_solid,active)
    fc,tc,sc = config['fluid'],config['time'],config['solver']
    grid = ChannelGrid(tuple(fc['shape']),tuple(fc['lengths_cm']))
    flow = PressureChannelFlow(grid,dt=tc['fsi_dt_s'],rho=fc['density_g_per_cm3'],mu=fc['viscosity_dyn_s_per_cm2'],
        inlet_pressure=fc['inlet_pressure_dyn_per_cm2'],outlet_pressure=fc['outlet_pressure_dyn_per_cm2'],
        device=device,fused=optimized,options=MGOptions(**sc['pressure']),pressure_backend=sc['pressure_backend'])
    domain = None
    if 'hydraulic_domain' in config:
        from ..mac.sealed_ports import SealedDomain
        from ..mac.sealed_flow import SealedChannelFlow
        domain = SealedDomain(grid, geometry, config['hydraulic_domain'], mesh.X, dimension=2,
            depth=config['circulation']['extrusion_depth_cm'])
        domain.bind_wall(mesh)
        domain.validate_wall(mesh.X)
        if config['output'].get('write_vtk', True): domain.write_geometry(out/'vtk')
        flow = SealedChannelFlow(flow, domain, fused=optimized, graphs=optimized and sc['pressure_backend'] == 'graph')
        atomic_json(out/'hydraulic_domain.json', domain.summary())
    if 'circulation' in config:
        from ..circulation.rc_outlet import RCChannelFlow, parse_rc
        flow = RCChannelFlow(flow, parse_rc(config['circulation'], dimension=2), mesh.X,
            extrusion_depth_cm=config['circulation']['extrusion_depth_cm'])
    fixed_nodes=torch.unique(mesh.roots) if end_condition=='fixed-reference-ends' else None
    if end_condition=='fixed-sealing-collars':
        width=config['hydraulic_domain']['collar_cells']*grid.spacing[0]
        z=mesh.X[:,0]-geometry.inlet_buffer_cm
        fixed_nodes=torch.where((z<=width+1e-12)|(z>=geometry.length_cm-width-1e-12))[0]
    transfer_class=TriangleTransfer
    transfer_options={}
    if fixed_nodes is not None:
        from ..mac2d.fixed_transfer import FixedEndTriangleTransfer
        transfer_class=FixedEndTriangleTransfer
        transfer_options['fixed_nodes']=fixed_nodes
    transfer = transfer_class(grid,solid.geometry,mass_backend=sc['mass_backend'],warm_start=True,
        fused=optimized,optimized=optimized,options=SolverOptions(**sc['mass']),ib_backend=sc['ib_backend'],**transfer_options)
    driver = ValveStepper(flow,transfer,solid,optimized=optimized,fused=optimized)
    state = driver.initialize()
    budget = None
    if domain is not None:
        from .tube_budget import TubeBudget
        budget = TubeBudget(mesh, domain, lower=active.lower, upper=active.upper)
    n = mesh.vertex_count
    X,cells = mesh.X[:n],mesh.cells[:,:3]
    fiber,sheet = X.new_tensor([1.,0.]).expand_as(X),X.new_tensor([0.,1.]).expand_as(X)
    numerical = dict(config['ep']['numerics'])
    numerical['solver'] = SolverOptions(**numerical['solver'])
    ep = Monodomain(X,cells,fiber,sheet,parameters=MonodomainParameters(**config['ep']['parameters']),
                     numerics=MonodomainNumerics(**numerical))
    ep.prepare(tc['ep_dt_s'])
    ep_deformation_feedback=config['ep'].get('deformation_feedback','none')
    ep_geometry_updates=0
    tissue = initial_state(X[:,0])
    ode = GPBLand(tc['ep_dt_s'],tc['ode_dt_ms'],method=config['ep'].get('ode_method','sdirk'))
    previous_stretch = torch.ones_like(tissue.voltage_v)
    zs,zw = torch.zeros_like(previous_stretch),torch.zeros_like(previous_stretch)
    tissue=replace(tissue,land=land_resting_equilibrium(tissue.gpb[:,37]*1000,previous_stretch))
    resting_tension=land_tension(previous_stretch,previous_stretch,tissue.land[:,0],tissue.land[:,1],zs,zw,tc['ep_dt_s'])
    active.set_reference(resting_tension)
    active.set_history(tissue,previous_stretch,zs,zw,tc['ep_dt_s'])
    stim = config['stimulus']
    stimulus_mask = (X[:,0] <= geometry.inlet_buffer_cm+stim['width_cm']).to(X.dtype)
    if stimulus_mask.count_nonzero() == 0:
        raise ValueError('no wall nodes in the stimulus region')
    stride = round(tc['ep_dt_s']/tc['fsi_dt_s'])
    steps = round(tc['end_time_s']/tc['fsi_dt_s'])
    history,frames,snapshots = [],[],[]
    activation = torch.full_like(previous_stretch,float('inf'))
    max_div,max_power_error = 0.,0.
    ep_steps = 0
    prior_steps=0
    resume_info=None
    if resume is not None:
        resume=Path(resume)
        source_config=json.loads((resume.parent/'config.resolved.json').read_text())
        source_report=json.loads((resume.parent/'report.json').read_text())
        def physical_config(value):
            value=json.loads(json.dumps(value));value.pop('output',None)
            for key in ('fsi_dt_s','end_time_s','stop_on_no_contact_limit'):
                value['time'].pop(key,None)
            return value
        if physical_config(source_config)!=physical_config(config):
            raise ValueError('resume must preserve model, load, EP cadence, geometry and boundary conditions')
        old_dt=source_config['time']['fsi_dt_s']
        if tc['fsi_dt_s']>old_dt:
            raise ValueError('resume may preserve or reduce the FSI step')
        with np.load(resume) as saved:
            get=lambda name:torch.from_numpy(saved[name].copy()).to(mesh.X)
            clock=float(saved['time_s']);mapped_step=round(clock/tc['fsi_dt_s'])
            if abs(mapped_step*tc['fsi_dt_s']-clock)>1e-12 or mapped_step>=steps:
                raise ValueError('resume clock must align with the new FSI grid and precede end time')
            if 'voltage_v' in saved:
                voltage=get('voltage_v')
            else:
                with np.load(resume.parent/'snapshots.npz') as frames_saved:
                    if abs(float(frames_saved['times'][-1])-clock)>1e-12:
                        raise ValueError('legacy resume voltage/geometry snapshot is not at checkpoint time')
                    voltage=torch.from_numpy(frames_saved['voltage'][-1].copy()).to(mesh.X)/1000
            tissue=replace(tissue,voltage_v=voltage,gpb=get('gpb'),land=get('land'))
            previous_stretch,zs,zw=(get(k) for k in ('previous_stretch','zs','zw'))
            old_step=int(saved['step'])
            if 'ep_steps' in saved:
                ep_steps=int(saved['ep_steps'])
            else:
                old_stride=round(tc['ep_dt_s']/old_dt)
                ep_steps=(old_step+old_stride-1)//old_stride
                if source_report['status']=='failed' and old_step%old_stride==0:
                    ep_steps+=1
            state=replace(state,step=mapped_step,time=clock,x=get('x'),
                velocity=(get('u'),get('v')),pressure=get('pressure'),force=get('force'))
            activation_history_exact='activation_time_s' in saved
            if activation_history_exact:
                activation=get('activation_time_s')
            else:
                # Only an upper bound is available for legacy checkpoints.
                # This diagnostic never enters a PDE, ODE or force evaluation.
                activation=torch.where(voltage>=-.02,torch.full_like(voltage,clock),activation)
            if 'circulation' in config:
                flow.pressure=get('rc_pressure_dyn_cm2')
                flow.committed_steps=(int(saved['tube_budget_steps']) if 'tube_budget_steps' in saved
                    else int(saved['rc_committed_steps']))
                flow.integrated_outlet_flow=get('rc_integrated_outlet_ml')
                flow.last_accepted=dict(pressure_dyn_cm2=flow.pressure.clone(),
                    flow_ml_s=flow.flux(state.velocity),capacitor_energy_erg=.5*flow.C*flow.pressure.square())
            if budget is not None:
                budget.cumulative.copy_(get('tube_budget_cumulative_ml'))
                budget.steps=int(saved['tube_budget_steps'])
                prior_steps=budget.steps
            else:prior_steps=old_step
        active.set_history(tissue,previous_stretch,zs,zw,tc['ep_dt_s'])
        solid.validate(state.x)
        if domain is not None:domain.validate_wall(state.x);domain.accepted(state.velocity)
        history=json.loads((resume.parent/'history.json').read_text())
        max_div=max(row['max_divergence_s_inv'] for row in history)
        max_power_error=max(row['transfer_power_relative_error'] for row in history)
        resume_info=dict(source=str(resume),source_time_s=clock,source_dt_s=old_dt,
            resumed_time_grid_step=mapped_step,prior_accepted_steps=prior_steps,
            ep_advanced_until_s=ep_steps*tc['ep_dt_s'],EP_stage_preserved_without_repeated_ODE=True,
            activation_times_exact=activation_history_exact)
    run_start_step=state.step
    termination_reason='end-time'

    def sample(info=None):
        gap = active.stretch_sections(state.x)*geometry.lumen_height_cm
        u,v = state.velocity
        # A mid-tube lumen flux diagnostic, distinct from total box/end flow.
        mid = geometry.axial_elements//2
        lo,hi = state.x[active.lower[mid]],state.x[active.upper[mid]]
        i = int(round(float(.5*(lo[0]+hi[0]))/grid.spacing[0]))
        if not 0 <= i <= grid.shape[0]:
            raise ValueError('lumen diagnostic left the MAC box')
        y = grid.coordinates(0,device=device)[0,:,1]
        overlap = (torch.minimum(y+grid.spacing[1]/2,hi[1])-torch.maximum(y-grid.spacing[1]/2,lo[1])).clamp_min(0)
        stretch = active.nodal_stretch(state.x)
        ta = land_tension(stretch,previous_stretch,tissue.land[:,0],tissue.land[:,1],zs,zw,tc['ep_dt_s'])
        return dict(step=state.step,time_s=state.time,
            step_diagnostics_available=info is not None,
            minimum_gap_cm=float(gap.min()),mid_gap_cm=float(gap[mid]),
            mean_gap_cm=float(gap.mean()),
            lower_mid_displacement_cm=float(state.x[active.lower[mid],1]-mesh.X[active.lower[mid],1]),
            upper_mid_displacement_cm=float(state.x[active.upper[mid],1]-mesh.X[active.upper[mid],1]),
            lumen_mid_flow_cm2_s=float((u[i]*overlap).sum()),
            box_inlet_flow_cm2_s=float(u[0].sum()*grid.spacing[1]),
            box_outlet_flow_cm2_s=float(u[-1].sum()*grid.spacing[1]),
            voltage_min_mv=float(tissue.voltage_v.min()*1000),voltage_max_mv=float(tissue.voltage_v.max()*1000),
            ep_last_staged_min_J=float(ep.volume_ratio.min()),
            ep_last_staged_max_J=float(ep.volume_ratio.max()),
            tension_max_dyn_cm2=float(ta.max()),max_velocity_cm_s=max(float(a.abs().max()) for a in state.velocity),
            applied_tension_increment_max_dyn_cm2=float((ta-resting_tension).clamp_min(0).max())*active.scale,
            **active.diagnostics(state.x),
            fixed_end_max_displacement_cm=(float((state.x[fixed_nodes]-mesh.X[fixed_nodes]).abs().max())
                if fixed_nodes is not None else None),
            max_divergence_s_inv=float(divergence(state.velocity,grid.spacing).abs().max()),
            activated_fraction=float(torch.isfinite(activation).double().mean()),
            pressure_cycles=(info or {}).get('flow',{}).get('pressure',{}).get('cycles',0),
            transfer_power_relative_error=(info or {}).get('power_relative_error',0.),
            transfer_power_abs_error_erg_s=(info or {}).get('power_abs_error',0.),
            transfer_solid_power_erg_s=(info or {}).get('solid_power',0.),
            transfer_fluid_power_erg_s=(info or {}).get('fluid_power',0.),
            **(flow.diagnostics() if 'circulation' in config else {}),
            **(domain.diagnostics() if domain is not None else {}),
            **(budget.diagnostics(state.velocity) if budget is not None else {}))

    def write_frame():
        if config['output'].get('write_vtk',True):
            frame = fields(out/'vtk',solid,state,grid,fluid=True)
            import meshio
            xyz = np.column_stack((state.x[:n].cpu().numpy(),np.zeros(n)))
            meshio.write_points_cells(out/'vtk'/f'ep_{state.step:06d}.vtu',xyz,[('triangle',cells.cpu().numpy())],
                point_data={'voltage_mV':tissue.voltage_v.cpu().numpy()*1000,
                            'calcium_uM':tissue.gpb[:,37].cpu().numpy()*1000,
                            'muscle_stretch':active.nodal_stretch(state.x).cpu().numpy()},binary=True)
            frames.append(frame)
            # Publish collections while the long experiment is still running.
            collection(out/'vtk',frames)
            entries=[f'<DataSet timestep="{f["time"]:.16g}" group="" part="0" file="ep_{f["step"]:06d}.vtu"/>' for f in frames]
            (out/'vtk'/'ep.pvd').write_text('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" byte_order="LittleEndian"><Collection>'+''.join(entries)+'</Collection></VTKFile>')
        load = active.load_fields(state.x)
        snapshots.append(dict(time_s=state.time,x=state.x.cpu().numpy(),
            voltage=tissue.voltage_v.cpu().numpy()*1000,u=state.velocity[0].cpu().numpy(),
            v=state.velocity[1].cpu().numpy(),p=state.pressure.cpu().numpy(),
            active_load_points=load['points'].cpu().numpy(),
            active_traction_dyn_cm2=load['traction'].cpu().numpy(),
            active_muscle_stretch=load['muscle_stretch'].cpu().numpy()))
        if config['output'].get('wave_sections',False):
            # Small live file, reference sections; no extra FSI/EP solve.
            ref=mesh.X[active.lower,0].cpu().numpy()-geometry.inlet_buffer_cm
            section_voltage=.5*(tissue.voltage_v[active.lower]+tissue.voltage_v[active.upper])*1000
            sections.append(dict(time_s=state.time,voltage_mv=section_voltage.cpu().numpy(),
                calcium_um=(.5*(tissue.gpb[active.lower,37]+tissue.gpb[active.upper,37])*1000).cpu().numpy(),
                gap_cm=(active.stretch_sections(state.x)*geometry.lumen_height_cm).cpu().numpy(),
                traction_dyn_cm2=load['traction'].mean(1).cpu().numpy()))
            temporary=out/'wave_sections.working.npz'
            np.savez_compressed(temporary,x_cm=ref,edge_x_cm=.5*(ref[:-1]+ref[1:]),
                **{key:np.stack([row[key] for row in sections]) for key in sections[0]})
            temporary.replace(out/'wave_sections.npz')

    sections=[]
    if not history:history.append(sample())
    write_frame()
    torch.cuda.synchronize(); setup_seconds=perf_counter()-setup_start
    start=perf_counter()
    steady_start=None
    compute_seconds=0.
    warmed_compute_seconds=0.
    block_start=start
    try:
        for k in range(state.step,steps):
            if state.time+1e-12 >= ep_steps*tc['ep_dt_s']:
                if ep_deformation_feedback=='santiago2018':
                    # P1 EP uses the shared solid vertices; higher-order solid
                    # nodes are excluded. Never feed a rejected trial geometry.
                    ep.update_deformation(state.x[:n])
                    ep_geometry_updates+=1
                stretch = active.nodal_stretch(state.x)
                zs = distortion_step(zs,stretch-previous_stretch,tc['ep_dt_s'],40.14)
                zw = distortion_step(zw,stretch-previous_stretch,tc['ep_dt_s'],405.86)
                previous_stretch = stretch.clone()
                # Integrate stimulus overlap rather than rounding a pulse edge.
                overlap = max(0.,min(state.time+tc['ep_dt_s'],stim['start_s']+stim['duration_s'])-max(state.time,stim['start_s']))
                current_stim = stimulus_mask*(stim['amplitude_ua_per_cm2']*overlap/tc['ep_dt_s'])
                old_v=tissue.voltage_v
                tissue,_,_ = ode.step(ep,tissue,stretch,zs,zw,state.time,current_stim)
                crossed=(old_v<-.02)&(tissue.voltage_v>=-.02)&torch.isinf(activation)
                fraction=(-.02-old_v)/(tissue.voltage_v-old_v).clamp_min(1e-30)
                activation=torch.where(crossed,state.time+tc['ep_dt_s']*fraction,activation)
                ep_steps += 1
                active.set_history(tissue,stretch,zs,zw,tc['ep_dt_s'])
                # Refresh the original lagged force using the new EP load at x_n.
                state=replace(state,force=solid.force(state.x))
            diagnostics=(k+1)%config['output']['log_every']==0
            trial_state,info=driver.step(state,diagnostics=diagnostics)
            if active.stretch_sections(trial_state.x).min() < .2:
                if config['time'].get('stop_on_no_contact_limit',False):
                    termination_reason='next trial crosses original 20%-gap no-contact limit'
                    break
                raise ValueError('gap below the no-contact demo limit; no contact model is enabled')
            if domain is not None:
                domain.validate_wall(trial_state.x)
            if 'circulation' in config:
                flow.commit(trial_state.velocity, step=flow.committed_steps+1)
            if budget is not None:
                budget.accepted(state.x, trial_state.x, trial_state.velocity, tc['fsi_dt_s'])
            state = trial_state
            field_due=(k+1)%config['output']['field_every']==0 or k+1==steps
            boundary=diagnostics or field_due or k==run_start_step+9
            if boundary:
                torch.cuda.synchronize()
                block_end=perf_counter()
                compute_seconds+=block_end-block_start
                if steady_start is not None:
                    warmed_compute_seconds+=block_end-block_start
            if k == run_start_step+9:
                steady_start=block_end
            if diagnostics or k+1==steps:
                row=sample(info); history.append(row)
                max_div=max(max_div,row['max_divergence_s_inv'])
                max_power_error=max(max_power_error,info.get('power_relative_error',0.))
                print(f"Tube step={state.step}, t={state.time*1000:.3f} ms, gap={row['minimum_gap_cm']:.6f} cm, "
                      f"Vm_max={row['voltage_max_mv']:.2f} mV, dTa={row['applied_tension_increment_max_dyn_cm2']:.3g}, "
                      f"traction_mean={row['active_traction_mean_dyn_cm2']:.3g} dyn/cm2, "
                      f"Qmid={row['lumen_mid_flow_cm2_s']:.5g} cm2/s",flush=True)
                atomic_json(out/'history.json',history)
            if field_due:
                write_frame()
            if boundary:
                torch.cuda.synchronize()
                block_start=perf_counter()
        torch.cuda.synchronize()
        finished=perf_counter()
        if state.step!=steps:
            # A protective stop can occur between output blocks; account for
            # its accepted steps and the rejected stopping trial as real cost.
            compute_seconds+=finished-block_start
            if steady_start is not None:warmed_compute_seconds+=finished-block_start
        elapsed=finished-start
        if history[-1]['time_s']!=state.time:history.append(sample())
        advanced_steps=state.step-run_start_step
        report=dict(status='completed' if state.step==steps else 'stopped-no-contact-limit',
            termination_reason=termination_reason,planned_end_time_s=tc['end_time_s'],resume=resume_info,
            scope='2D paired-hoop EP-GPB-Land / Neo-Hookean / original explicit 2D MAC-MG AFSI',
            accepted_steps=prior_steps+advanced_steps,time_grid_step=state.step,
            accepted_steps_this_run=advanced_steps,ep_steps=ep_steps,time_s=state.time,gpu=torch.cuda.get_device_name(),
            setup_seconds=setup_seconds,elapsed_seconds=elapsed,
            compute_seconds=compute_seconds,warmed_compute_seconds=warmed_compute_seconds,
            warmed_compute_steps=max(0,advanced_steps-10),execution_profile='optimized' if optimized else 'reference',
            timing_scope='synchronized step blocks including acceptance checks; excludes sample/field/disk output; warmup first 10 physical steps',
            seconds_per_step_excluding_first_10=warmed_compute_seconds/(advanced_steps-10) if steady_start and advanced_steps>10 else None,
            ep_nodes=n,solid_nodes=len(mesh.X),solid_cells=len(mesh.cells),fluid_shape=grid.shape,
            ep_discretization=ep.numerics.discretization,ep_time_scheme=ep.numerics.time_scheme,
            ep_deformation_feedback=ep_deformation_feedback,ep_geometry_updates=ep_geometry_updates,
            ep_last_staged_min_J=float(ep.volume_ratio.min()),
            ep_last_staged_max_J=float(ep.volume_ratio.max()),
            ode_method=ode.method,ode_dt_ms=ode.dt_ode_ms,
            max_sampled_divergence_s_inv=max_div,max_sampled_transfer_power_relative_error=max_power_error,
            final=history[-1],three_field_time_integration=True,
            mechanics=('plane strain compressible Neo-Hookean; fixed reference end support'
                if fixed_nodes is not None else 'plane strain compressible Neo-Hookean; no prescribed displacement'),
            solid_end_condition=end_condition,
            active_model='paired hoop reduction; not 3D active PK1',
            active_stretch_model=active.stretch_model,
            active_end_inactive_cm=active.end_inactive_cm,
            mechanics_to_land_feedback=('isometric omitted hoop direction; no gap-to-muscle feedback'
                if active.stretch_model=='plane-strain-isometric' else 'legacy lumen-gap surrogate'),
            active_reference='positive increment above equilibrium Land tension at reference stretch and initial resting Ca',
            resting_tension_dyn_cm2=float(resting_tension.max()),
            exterior_fluid_present=True,lumen_flux='mid-section MAC overlap quadrature; approximate moving-wall diagnostic',
            boundary=('zero inlet reference pressure / implicit RC on full box outlet / velocity normal derivatives zero'
                if 'circulation' in config else 'prescribed end pressures + homogeneous normal velocity derivatives; vector-Laplacian pseudo-traction'),
            porous_media=False, circulation='circulation' in config,
            rc_outlet_scope=config.get('circulation', {}).get('outlet_scope'),
            literature_replication=False, physical_tube_mouth_boundary_aligned=False,
            hydraulic_path_isolated=domain is not None,
            hydraulic_domain=domain.summary() if domain is not None else None,
            conservation_budget=budget.summary() if budget is not None else dict(enabled=False),
            rc_parameters_cgs=(dict(resistance_dyn_s_per_cm5=flow.R, compliance_cm5_per_dyn=flow.C)
                if 'circulation' in config else None),
            fluid_solid_scheme='unchanged mac2d.ValveStepper explicit-lagged',passive_control=passive,
            no_stimulus_control=no_stimulus)
        atomic_json(out/'report.json',report)
    except Exception as exc:
        speed=torch.stack([v.abs().max() for v in state.velocity])
        atomic_json(out/'report.json',dict(status='failed',accepted_steps=state.step,time_s=state.time,
            error=str(exc),elapsed_seconds=perf_counter()-start,setup_seconds=setup_seconds,
            active_stretch_model=active.stretch_model,
            last_accepted_velocity_max_cm_s=float(speed.max()),
            last_accepted_courant=float((flow.dt*speed/speed.new_tensor(grid.spacing)).sum()),
            last_accepted_centered_number=float(flow.dt*speed.square().sum()/(flow.mu/flow.rho))))
        raise
    finally:
        # Keep the last accepted geometry/velocity available even when a
        # failure occurs between scheduled samples. EP/load staging may
        # already have advanced for the attempted explicit step.
        if snapshots[-1]['time_s'] != state.time:
            write_frame()
        ep.close()
        atomic_json(out/'history.json',history)
        if frames:
            collection(out/'vtk',frames)
        entries=[f'<DataSet timestep="{f["time"]:.16g}" group="" part="0" file="ep_{f["step"]:06d}.vtu"/>' for f in frames]
        if frames:
            (out/'vtk'/'ep.pvd').write_text('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n<Collection>\n'+
                '\n'.join(entries)+'\n</Collection>\n</VTKFile>\n',encoding='utf-8')
        np.savez_compressed(out/'snapshots.npz',reference=mesh.X.cpu().numpy(),cells=mesh.cells.cpu().numpy(),
            lower=active.lower.cpu().numpy(),upper=active.upper.cpu().numpy(),vertex_count=n,
            times=np.array([s['time_s'] for s in snapshots]),
            **{name:np.stack([s[name] for s in snapshots]) for name in
               ('x','voltage','u','v','p','active_load_points','active_traction_dyn_cm2','active_muscle_stretch')})
        np.savez_compressed(out/'final_state.npz',gpb=tissue.gpb.cpu().numpy(),land=tissue.land.cpu().numpy(),
            zs=zs.cpu().numpy(),zw=zw.cpu().numpy(),force=state.force.cpu().numpy(),
            x=state.x.cpu().numpy(),u=state.velocity[0].cpu().numpy(),v=state.velocity[1].cpu().numpy(),
            pressure=state.pressure.cpu().numpy(),time_s=state.time,step=state.step,
            previous_stretch=previous_stretch.cpu().numpy(),
            voltage_v=tissue.voltage_v.cpu().numpy(),ep_steps=ep_steps,
            activation_time_s=activation.cpu().numpy(),
            **(budget.checkpoint() if budget is not None else {}),
            **(flow.checkpoint() if 'circulation' in config else {}))
    return report
