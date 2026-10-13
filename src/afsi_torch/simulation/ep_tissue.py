"""Fixed-tissue experiment orchestration; calls existing EP and GPB solvers.

No LV mesh, circulation, fluid solve, mechanical solve or duplicated PDE/ODE.
"""
from dataclasses import asdict
from pathlib import Path
import os
import time
import xml.etree.ElementTree as ET
import numpy as np
import torch
from ..cycle_checkpoint import atomic_json
from ..ep import Monodomain, GPBLand, initial_state
from ..ep.experiment import EPTissueConfig
from ..geometry.tissue_slab import structured_tissue_slab


def stimulus_mask(X, config):
    """Closed spatial box, with tolerance only for floating coordinate rounding."""
    return ((X>=X.new_tensor(config.minimum_cm)-1e-12)&
            (X<=X.new_tensor(config.maximum_cm)+1e-12)).all(dim=1)


def stimulus_field(X, config):
    """Construct the prescribed current directly in the mesh floating precision."""
    return X.new_zeros(len(X)).masked_fill(stimulus_mask(X,config),config.amplitude_ua_per_cm2)


def pulse_active(step, dt_s, config):
    """Half-open pulse [start, start+duration); no floating endpoint drift."""
    start=round(config.start_s/dt_s)
    stop=round((config.start_s+config.duration_s)/dt_s)
    return start<=step<stop


def first_activation(previous, current, activation, time_s, dt_s, threshold_v):
    """First upward crossing, linearly interpolated between accepted EP states."""
    crossing=torch.isnan(activation)&(previous<threshold_v)&(current>=threshold_v)
    denominator=torch.where(crossing,current-previous,torch.ones_like(current))
    interpolated=time_s+dt_s*(threshold_v-previous)/denominator
    return torch.where(crossing,interpolated,activation)


def observation_indices(mesh, domain, settings):
    shape=tuple(n+1 for n in mesh.shape)
    def ijk(p):
        return tuple(min(n,max(0,round((v-o)/h))) for v,o,h,n in
                     zip(p,domain.origin_cm,mesh.spacing_cm,mesh.shape))
    def index(p):
        a,b,c=ijk(p)
        return (a*shape[1]+b)*shape[2]+c
    probes=np.array([index(p) for p in settings.probes_cm],dtype=np.int64)
    grid=np.arange(len(mesh.X)).reshape(shape)
    line_selection=list(ijk(settings.line_anchor_cm)); line_selection[settings.line_axis]=slice(None)
    line=grid[tuple(line_selection)].copy()
    slice_index=ijk(tuple(settings.slice_coordinate_cm if a==settings.slice_axis else o
                          for a,o in enumerate(domain.origin_cm)))[settings.slice_axis]
    return probes,line,slice_index


def _write_fields(path, X, cells, voltage, calcium, activation, mask):
    import meshio
    meshio.write_points_cells(path,X,[('tetra',cells)],point_data={
        'voltage_mV':voltage*1000,'calcium_uM':calcium*1000,
        'activation_time_ms':activation*1000,'activated':np.isfinite(activation).astype(np.uint8),
        'stimulus_region':mask.astype(np.uint8)})


def _plots(folder, X, shape, observation, slice_index, line, activation, times, volts, calcium,
           probes, snapshot_steps, snapshots, dt):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    grid=X.reshape((*shape,3)); lat=activation.reshape(shape)
    sel=[slice(None)]*3; sel[observation.slice_axis]=slice_index; sel=tuple(sel)
    axes=[a for a in range(3) if a!=observation.slice_axis]
    p=grid[sel]; values=np.ma.masked_invalid(lat[sel]*1000)
    extent=(p[...,axes[0]].min(),p[...,axes[0]].max(),p[...,axes[1]].min(),p[...,axes[1]].max())
    fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    im=ax[0].imshow(values.T,origin='lower',extent=extent,aspect='equal',cmap='turbo')
    if values.count()>0 and float(values.max())>float(values.min()):
        ax[0].contour(p[...,axes[0]],p[...,axes[1]],values,levels=10,colors='k',linewidths=.4)
    fig.colorbar(im,ax=ax[0],label='Activation time (ms)')
    ax[0].set(xlabel=f'{"xyz"[axes[0]]} (cm)',ylabel=f'{"xyz"[axes[1]]} (cm)',
              title=f'First upward crossing of {observation.activation_threshold_mv:g} mV')
    ax[1].plot(X[line,observation.line_axis],activation[line]*1000,lw=2)
    ax[1].set(xlabel=f'{"xyz"[observation.line_axis]} (cm)',ylabel='Activation time (ms)',
              title='Activation along the selected tissue edge')
    ax[1].grid(alpha=.25)
    fig.savefig(folder/'activation.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(2,1,figsize=(9,6),sharex=True,layout='constrained')
    for j,node in enumerate(probes):
        label='('+', '.join(f'{v:g}' for v in X[node])+') cm'
        ax[0].plot(times*1000,volts[:,j]*1000,label=label)
        ax[1].plot(times*1000,calcium[:,j]*1000)
    ax[0].set(ylabel='Voltage (mV)',title='GPB at fixed tissue probes')
    ax[0].legend(fontsize=8,ncol=2)
    ax[1].set(xlabel='Time (ms)',ylabel='Cytosolic calcium (uM)')
    for a in ax: a.grid(alpha=.25)
    fig.savefig(folder/'voltage_traces.png',dpi=180); plt.close(fig)
    # Preserve early wavefront snapshots and the final state, with one voltage scale.
    ids=np.unique([*range(min(5,len(snapshots))),len(snapshots)-1])
    rows=(len(ids)+2)//3
    fig,ax=plt.subplots(rows,3,figsize=(12,3.2*rows),squeeze=False,layout='constrained')
    vmin=min(v.min()*1000 for v in snapshots); vmax=max(v.max()*1000 for v in snapshots)
    for a,i in zip(ax.flat,ids):
        image=snapshots[i].reshape(shape)[sel].T*1000
        im=a.imshow(image,origin='lower',extent=extent,aspect='equal',cmap='turbo',vmin=vmin,vmax=vmax)
        a.set(title=f't = {snapshot_steps[i]*dt*1000:g} ms',
              xlabel=f'{"xyz"[axes[0]]} (cm)',ylabel=f'{"xyz"[axes[1]]} (cm)')
    for a in list(ax.flat)[len(ids):]: a.set_visible(False)
    fig.colorbar(im,ax=[a for a in ax.flat if a.get_visible()],label='Voltage (mV)',shrink=.8)
    fig.savefig(folder/'voltage_snapshots.png',dpi=180); plt.close(fig)


@torch.no_grad()
def run(config: EPTissueConfig, output, *, device='cuda'):
    if not isinstance(config,EPTissueConfig):
        raise ValueError('typed EPTissueConfig required')
    device=torch.device(device)
    if device.type!='cuda' or not torch.cuda.is_available():
        raise RuntimeError('the existing GPB ODE solver requires an actual CUDA device')
    if device.index is None:
        device=torch.device('cuda',torch.cuda.current_device())
    torch.cuda.set_device(device)
    if config.output.write_vtk:
        import meshio  # fail before integration if optional export dependencies are missing
    folder=Path(output)
    if folder.exists() and any(folder.iterdir()):
        raise ValueError('output directory must be new or empty; existing results are not overwritten')
    folder.mkdir(parents=True,exist_ok=True)
    if config.output.write_plots:
        os.environ.setdefault('MPLCONFIGDIR',str(folder.resolve()/'.matplotlib'))
        import matplotlib
    atomic_json(folder/'config_resolved.json',dict(demo='ep-slab',**asdict(config)))
    started=time.perf_counter()
    ep=None
    try:
        mesh=structured_tissue_slab(config.domain.lengths_cm,config.domain.spacing_cm,
                                    config.domain.origin_cm,device=device)
        X,cells=mesh.X,mesh.cells
        f=X.new_tensor(config.domain.fiber).expand(len(X),-1)
        s=X.new_tensor(config.domain.sheet).expand(len(X),-1)
        ep=Monodomain(X,cells,f,s,parameters=config.parameters,numerics=config.ep_numerics)
        dt=config.time.dt_ep_s
        ode=GPBLand(dt,config.time.dt_ode_ms)
        ep.prepare(dt)
        state=initial_state(X[:,0])
        zero=torch.zeros_like(state.voltage_v); stretch=torch.ones_like(zero)
        mask=stimulus_mask(X,config.stimulus)
        if not mask.any(): raise ValueError('mesh contains no stimulated nodes')
        stimulus=stimulus_field(X,config.stimulus)
        activation=torch.full_like(zero,float('nan'))
        probes,line,slice_index=observation_indices(mesh,config.domain,config.observations)
        probe_gpu=torch.as_tensor(probes,device=device)
        trace_steps=[0]; voltage_trace=[state.voltage_v[probe_gpu].clone()]
        calcium_trace=[state.gpb[probe_gpu,37].clone()]
        steps=round(config.time.end_time_s/dt)
        requested={round(t/dt) for t in config.output.snapshot_times_s if t<=config.time.end_time_s+1e-12}
        requested.update((0,steps))
        snapshot_steps=[]; snapshots=[]
        Xcpu=X.cpu().numpy(); Ccpu=cells.cpu().numpy(); Mcpu=mask.cpu().numpy()
        np.savez_compressed(folder/'mesh.npz',coordinates_cm=Xcpu,cells=Ccpu,
                            shape=np.array(mesh.shape),stimulus_region=Mcpu)
        vtk_frames=[]
        def observe(step):
            if step in requested:
                snapshot_steps.append(step); snapshots.append(state.voltage_v.cpu().numpy())
            if config.output.write_vtk and (step%config.output.vtk_every==0 or step==steps):
                name=f'fields_{step:06d}.vtu'
                _write_fields(folder/name,Xcpu,Ccpu,state.voltage_v.cpu().numpy(),
                              state.gpb[:,37].cpu().numpy(),activation.cpu().numpy(),Mcpu)
                vtk_frames.append((step*dt,name))
        observe(0)
        torch.cuda.synchronize(device)
        setup_seconds=time.perf_counter()-started
        integration_started=time.perf_counter(); export_seconds=0.
        maximum_ratio=0.; iterations=0
        for step in range(steps):
            previous=state.voltage_v
            state,_,info=ode.step(ep,state,stretch,zero,zero,step*dt,
                stimulus if pulse_active(step,dt,config.stimulus) else zero,advance_land=False)
            activation=first_activation(previous,state.voltage_v,activation,step*dt,dt,
                                         config.observations.activation_threshold_mv/1000)
            iterations+=info.iterations
            maximum_ratio=max(maximum_ratio,info.residual_norm/info.tolerance)
            accepted=step+1
            if accepted%config.output.trace_every==0 or accepted==steps:
                trace_steps.append(accepted); voltage_trace.append(state.voltage_v[probe_gpu].clone())
                calcium_trace.append(state.gpb[probe_gpu,37].clone())
            if accepted in requested or (config.output.write_vtk and
                    (accepted%config.output.vtk_every==0 or accepted==steps)):
                t0=time.perf_counter(); observe(accepted); export_seconds+=time.perf_counter()-t0
            if accepted%config.output.log_every==0 or accepted==steps:
                lo,hi,count=torch.stack((state.voltage_v.min()*1000,state.voltage_v.max()*1000,
                                        torch.isfinite(activation).sum().to(dtype=X.dtype))).tolist()
                print(f'EP slab step={accepted}/{steps}, t={accepted*dt*1000:.3f} ms, '
                      f'V=[{lo:.3f},{hi:.3f}] mV, activated={int(count)}/{len(X)}',flush=True)
        torch.cuda.synchronize(device)
        integration_seconds=time.perf_counter()-integration_started-export_seconds
        lat=activation.cpu().numpy(); v=state.voltage_v.cpu().numpy(); ca=state.gpb[:,37].cpu().numpy()
        times=np.array(trace_steps)*dt
        voltages=torch.stack(voltage_trace).cpu().numpy(); calcium=torch.stack(calcium_trace).cpu().numpy()
        np.savez_compressed(folder/'final_state.npz',voltage_v=v,gpb=state.gpb.cpu().numpy(),
                            activation_time_s=lat,calcium_mm=ca)
        np.savez_compressed(folder/'voltage_snapshots.npz',time_s=np.array(snapshot_steps)*dt,
                            voltage_v=np.stack(snapshots))
        # Small readable probe table: time, V and calcium for each resolved probe.
        columns=['time_ms']+[f'probe_{i}_voltage_mv' for i in range(len(probes))]+[
            f'probe_{i}_calcium_um' for i in range(len(probes))]
        np.savetxt(folder/'probes.csv',np.column_stack((times*1000,voltages*1000,calcium*1000)),
                   delimiter=',',header=','.join(columns),comments='')
        np.savetxt(folder/'activation_line.csv',np.column_stack((Xcpu[line],lat[line]*1000)),
                   delimiter=',',header='x_cm,y_cm,z_cm,activation_ms',comments='')
        if config.output.write_vtk:
            _write_fields(folder/'activation.vtu',Xcpu,Ccpu,v,ca,lat,Mcpu)
            root=ET.Element('VTKFile',type='Collection',version='0.1',byte_order='LittleEndian')
            collection=ET.SubElement(root,'Collection')
            for t,name in vtk_frames:
                ET.SubElement(collection,'DataSet',timestep=str(t),group='',part='0',file=name)
            ET.ElementTree(root).write(folder/'series.pvd',encoding='utf-8',xml_declaration=True)
        if config.output.write_plots:
            _plots(folder,Xcpu,tuple(n+1 for n in mesh.shape),config.observations,slice_index,line,
                   lat,times,voltages,calcium,probes,snapshot_steps,snapshots,dt)
        def finite_value(a):
            return float(a) if np.isfinite(a) else None
        backend=ep._systems[dt][2]
        report=dict(status='completed',case_name=config.case_name,scope=config.reference.comparison_scope,
            accepted_steps=steps,time_s=steps*dt,node_count=len(X),cell_count=len(cells),
            brick_shape=mesh.shape,brick_spacing_cm=mesh.spacing_cm,
            exact_volume_cm3=float(np.prod(config.domain.lengths_cm)),
            assembled_volume_cm3=float(ep.mass_diagonal.sum().item()),
            stimulated_nodes=int(Mcpu.sum()),stimulus_geometric_volume_cm3=float(np.prod(
                np.array(config.stimulus.maximum_cm)-config.stimulus.minimum_cm)),
            stimulus_nodal_quadrature_volume_cm3=float(ep.mass_diagonal[mask].sum().item()),
            stimulus_membrane_ua_per_cm2=config.stimulus.amplitude_ua_per_cm2,
            stimulus_volume_ua_per_cm3=config.parameters.surface_to_volume_per_cm*config.stimulus.amplitude_ua_per_cm2,
            ep_discretization=ep.numerics.discretization,ep_backend=ep.numerics.backend,
            linear_solver_max_residual_to_tolerance=maximum_ratio,
            linear_solver_mean_iterations=iterations/steps,amg_setup_count=backend.setup_count if backend else 0,
            voltage_range_mv=[float(v.min()*1000),float(v.max()*1000)],
            calcium_range_um=[float(ca.min()*1000),float(ca.max()*1000)],
            activated_nodes=int(np.isfinite(lat).sum()),activated_fraction=float(np.isfinite(lat).mean()),
            full_activation_observed=bool(np.isfinite(lat).all()),
            last_activation_ms=finite_value(np.nanmax(lat)*1000) if np.isfinite(lat).any() else None,
            activation_threshold_mv=config.observations.activation_threshold_mv,
            probes=[dict(requested_cm=p,resolved_cm=Xcpu[node].tolist(),node=int(node),
                         activation_ms=finite_value(lat[node]*1000)) for p,node in zip(config.observations.probes_cm,probes)],
            slice_resolved_cm=float(grid_coordinate(mesh,config.domain,config.observations.slice_axis,slice_index)),
            gpu=torch.cuda.get_device_name(device),dtype='float64',device=str(device),
            dt_ep_ms=dt*1000,dt_ode_ms=ode.dt_ode_ms,ode_substeps=steps*ode.substeps,
            setup_seconds=setup_seconds,integration_seconds=integration_seconds,
            mean_seconds_per_ep_step=integration_seconds/steps,field_export_seconds=export_seconds,
            elapsed_seconds=time.perf_counter()-started,
            diffusion_cm2_s=[g/(config.parameters.surface_to_volume_per_cm*config.parameters.capacitance_uf_per_cm2)
                             for g in config.parameters.conductivity_us_per_cm],
            fixed_geometry=True,solid_motion_advanced=False,fluid_advanced=False,land_advanced=False,
            strict_paper_model_reproduction=False,reference=asdict(config.reference))
        atomic_json(folder/'report.json',report)
        return report
    except Exception as error:
        atomic_json(folder/'failure.json',dict(status='failed',error_type=type(error).__name__,error=str(error)))
        raise
    finally:
        if ep is not None: ep.close()


def grid_coordinate(mesh,domain,axis,index):
    return domain.origin_cm[axis]+mesh.spacing_cm[axis]*index
