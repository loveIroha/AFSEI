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
    if set(c) != required or c['demo'] != 'active-tube-2d':
        raise ValueError(f'configuration requires exactly {sorted(required)}')
    if c['active']['mechanism'] != 'paired-hoop-land':
        raise ValueError('this demo explicitly implements the paired-hoop Land reduction')
    if c['fluid']['end_boundary'] != 'pressure-neumann-velocity' or c['fluid']['outer_y_boundary'] != 'no-slip':
        raise ValueError('unsupported boundary condition; do not silently substitute a velocity inlet')
    if c['ep']['numerics']['discretization'] != 'nodal_p1':
        raise ValueError('the active-tube demo requires nodal FEM')
    dt, dep, end = c['time']['fsi_dt_s'], c['time']['ep_dt_s'], c['time']['end_time_s']
    if any(not math.isfinite(v) or v <= 0 for v in (dt,dep,end)):
        raise ValueError('positive finite time parameters required')
    for v in (dep/dt,end/dt):
        if round(v) < 1 or abs(round(v)-v) > 1e-8:
            raise ValueError('EP dt and duration must be integer multiples of FSI dt')
    if end < c['stimulus']['start_s'] or c['stimulus']['duration_s'] <= 0 or c['stimulus']['width_cm'] <= 0:
        raise ValueError('stimulus must have positive support and start before the run ends')
    if c['output']['log_every'] < 1 or c['output']['field_every'] < 1:
        raise ValueError('positive output intervals required')


@torch.no_grad()
def run(config, output, *, device='cuda', passive=False, no_stimulus=False):
    validate_config(config)
    if not torch.cuda.is_available() or torch.device(device).type != 'cuda':
        raise RuntimeError('GPB/Land uses the existing CUDA kernel; this demo requires CUDA')
    config = json.loads(json.dumps(config))
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
    passive_solid = NeoHookeanTube(mesh,**config['solid'])
    active_cfg = {k:v for k,v in config['active'].items() if k != 'mechanism'}
    active = PairedHoopLoad(mesh,geometry,**active_cfg)
    solid = ActiveTubeSolid(passive_solid,active)
    fc,tc,sc = config['fluid'],config['time'],config['solver']
    grid = ChannelGrid(tuple(fc['shape']),tuple(fc['lengths_cm']))
    flow = PressureChannelFlow(grid,dt=tc['fsi_dt_s'],rho=fc['density_g_per_cm3'],mu=fc['viscosity_dyn_s_per_cm2'],
        inlet_pressure=fc['inlet_pressure_dyn_per_cm2'],outlet_pressure=fc['outlet_pressure_dyn_per_cm2'],
        device=device,options=MGOptions(**sc['pressure']),pressure_backend=sc['pressure_backend'])
    transfer = TriangleTransfer(grid,solid.geometry,mass_backend=sc['mass_backend'],warm_start=True,
        fused=True,optimized=True,options=SolverOptions(**sc['mass']),ib_backend=sc['ib_backend'])
    driver = ValveStepper(flow,transfer,solid,optimized=True,fused=True)
    state = driver.initialize()
    n = mesh.vertex_count
    X,cells = mesh.X[:n],mesh.cells[:,:3]
    fiber,sheet = X.new_tensor([1.,0.]).expand_as(X),X.new_tensor([0.,1.]).expand_as(X)
    numerical = dict(config['ep']['numerics'])
    numerical['solver'] = SolverOptions(**numerical['solver'])
    ep = Monodomain(X,cells,fiber,sheet,parameters=MonodomainParameters(**config['ep']['parameters']),
                     numerics=MonodomainNumerics(**numerical))
    ep.prepare(tc['ep_dt_s'])
    tissue = initial_state(X[:,0])
    ode = GPBLand(tc['ep_dt_s'],tc['ode_dt_ms'])
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
            minimum_gap_cm=float(gap.min()),mid_gap_cm=float(gap[mid]),
            mean_gap_cm=float(gap.mean()),
            lower_mid_displacement_cm=float(state.x[active.lower[mid],1]-mesh.X[active.lower[mid],1]),
            upper_mid_displacement_cm=float(state.x[active.upper[mid],1]-mesh.X[active.upper[mid],1]),
            lumen_mid_flow_cm2_s=float((u[i]*overlap).sum()),
            box_inlet_flow_cm2_s=float(u[0].sum()*grid.spacing[1]),
            box_outlet_flow_cm2_s=float(u[-1].sum()*grid.spacing[1]),
            voltage_min_mv=float(tissue.voltage_v.min()*1000),voltage_max_mv=float(tissue.voltage_v.max()*1000),
            tension_max_dyn_cm2=float(ta.max()),max_velocity_cm_s=max(float(a.abs().max()) for a in state.velocity),
            applied_tension_increment_max_dyn_cm2=float((ta-resting_tension).clamp_min(0).max())*active.scale,
            max_divergence_s_inv=float(divergence(state.velocity,grid.spacing).abs().max()),
            activated_fraction=float(torch.isfinite(activation).double().mean()),
            pressure_cycles=(info or {}).get('flow',{}).get('pressure',{}).get('cycles',0))

    def write_frame():
        frame = fields(out/'vtk',solid,state,grid,fluid=True)
        import meshio
        xyz = np.column_stack((state.x[:n].cpu().numpy(),np.zeros(n)))
        meshio.write_points_cells(out/'vtk'/f'ep_{state.step:06d}.vtu',xyz,[('triangle',cells.cpu().numpy())],
            point_data={'voltage_mV':tissue.voltage_v.cpu().numpy()*1000,
                        'calcium_uM':tissue.gpb[:,37].cpu().numpy()*1000},binary=True)
        frames.append(frame)
        snapshots.append(dict(time_s=state.time,x=state.x.cpu().numpy(),
            voltage=tissue.voltage_v.cpu().numpy()*1000,u=state.velocity[0].cpu().numpy(),
            v=state.velocity[1].cpu().numpy(),p=state.pressure.cpu().numpy()))

    history.append(sample()); write_frame()
    torch.cuda.synchronize(); setup_seconds=perf_counter()-setup_start
    start=perf_counter()
    steady_start=None
    try:
        for k in range(steps):
            if k % stride == 0:
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
            state,info=driver.step(state,diagnostics=diagnostics)
            if active.stretch_sections(state.x).min() < .2:
                raise ValueError('gap below the no-contact demo limit; no contact model is enabled')
            if k == 9:
                torch.cuda.synchronize(); steady_start=perf_counter()
            if diagnostics or k+1==steps:
                row=sample(info); history.append(row)
                max_div=max(max_div,row['max_divergence_s_inv'])
                max_power_error=max(max_power_error,info.get('power_relative_error',0.))
                print(f"Tube step={state.step}, t={state.time*1000:.3f} ms, gap={row['minimum_gap_cm']:.6f} cm, "
                      f"Vm_max={row['voltage_max_mv']:.2f} mV, dTa={row['applied_tension_increment_max_dyn_cm2']:.3g}, "
                      f"Qmid={row['lumen_mid_flow_cm2_s']:.5g} cm2/s",flush=True)
                atomic_json(out/'history.json',history)
            if (k+1)%config['output']['field_every']==0 or k+1==steps:
                write_frame()
        torch.cuda.synchronize()
        elapsed=perf_counter()-start
        report=dict(status='completed',scope='2D paired-hoop EP-GPB-Land / Neo-Hookean / original explicit 2D MAC-MG AFSI',
            accepted_steps=state.step,ep_steps=ep_steps,time_s=state.time,gpu=torch.cuda.get_device_name(),
            setup_seconds=setup_seconds,elapsed_seconds=elapsed,
            seconds_per_step_excluding_first_10=(perf_counter()-steady_start)/(steps-10) if steady_start and steps>10 else None,
            ep_nodes=n,solid_nodes=len(mesh.X),solid_cells=len(mesh.cells),fluid_shape=grid.shape,
            max_sampled_divergence_s_inv=max_div,max_sampled_transfer_power_relative_error=max_power_error,
            final=history[-1],three_field_time_integration=True,
            mechanics='plane strain compressible Neo-Hookean; no prescribed displacement',
            active_model='paired hoop reduction; not 3D active PK1',
            active_reference='positive increment above equilibrium Land tension at reference stretch and initial resting Ca',
            resting_tension_dyn_cm2=float(resting_tension.max()),
            exterior_fluid_present=True,lumen_flux='mid-section MAC overlap quadrature; approximate moving-wall diagnostic',
            boundary='prescribed end pressures + homogeneous normal velocity derivatives; vector-Laplacian pseudo-traction',
            fluid_solid_scheme='unchanged mac2d.ValveStepper explicit-lagged',passive_control=passive,
            no_stimulus_control=no_stimulus)
        atomic_json(out/'report.json',report)
    except Exception as exc:
        atomic_json(out/'report.json',dict(status='failed',accepted_steps=state.step,time_s=state.time,error=str(exc)))
        raise
    finally:
        ep.close()
        atomic_json(out/'history.json',history)
        collection(out/'vtk',frames)
        entries=[f'<DataSet timestep="{f["time"]:.16g}" group="" part="0" file="ep_{f["step"]:06d}.vtu"/>' for f in frames]
        (out/'vtk'/'ep.pvd').write_text('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n<Collection>\n'+
            '\n'.join(entries)+'\n</Collection>\n</VTKFile>\n',encoding='utf-8')
        np.savez_compressed(out/'snapshots.npz',reference=mesh.X.cpu().numpy(),cells=mesh.cells.cpu().numpy(),
            lower=active.lower.cpu().numpy(),upper=active.upper.cpu().numpy(),vertex_count=n,
            times=np.array([s['time_s'] for s in snapshots]),
            **{name:np.stack([s[name] for s in snapshots]) for name in ('x','voltage','u','v','p')})
    return report
