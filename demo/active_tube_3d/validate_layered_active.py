"""Short active-load verification using original EP, FE, MAC and IB solvers.

Electrical preparation holds the saved geometry fixed. The subsequent probe
advances all three fields from that prepared state, with an independent FSI
clock. This is deliberately NOT a physiological restart/full-cycle solution.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from afsi_torch.simulation.circular_tube_config import resolve_config
from afsi_torch.solids.circular_tube import generate_circular_tube
from afsi_torch.ep.layered_circular_tube import LayeredCircularTubeSolid
from afsi_torch.ep.monodomain import Monodomain
from afsi_torch.ep.tissue import TissueState,GPBLand
from afsi_torch.ep.land import distortion_step
from afsi_torch.mac.paper_coupling import BEIBStepper
from afsi_torch.mac.sealed_ports import SealedDomain
from afsi_torch.mac.sealed_flow import SealedBEFlow
from afsi_torch.circulation.rc_outlet import RCBoxFlow,parse_rc
from afsi_torch.simulation.tube_budget import TubeBudget
from afsi_torch.simulation.circular_tube import _ep_collection
from afsi_torch.mac.grid import divergence
from afsi_torch.mac.output import MACWriter
from afsi_torch.cycle_checkpoint import atomic_json


def atomic_npz(path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as f:np.savez_compressed(f,**arrays)
    temporary.replace(path)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--passive',action='store_true')
    p.add_argument('--preparation-max-ms',type=float,default=40.)
    p.add_argument('--target-tension-kpa',type=float,default=1.)
    p.add_argument('--probe-ms',type=float,default=.25)
    p.add_argument('--ramp-steps',type=int,default=4,
        help='probe-only linear release envelope; zero reproduces abrupt loading')
    p.add_argument('--viscous-solver',choices=['jacobi','pcg'],default='jacobi')
    p.add_argument('--resume-probe',type=Path,help='resume an accepted verification state, preserving both clocks and RC')
    args=p.parse_args()
    c=json.loads(args.config.read_text())
    geometry,fsi,params,numerics=resolve_config(c)
    if 'layers' not in c['solid'] or 'hydraulic_domain' not in c:raise ValueError('sealed two-layer config required')
    if any(not np.isfinite(v) or v<=0 for v in (args.preparation_max_ms,args.target_tension_kpa,args.probe_ms)):
        raise ValueError('positive finite preparation, tension and probe controls required')
    steps=round(args.probe_ms/(1000*fsi.time.dt))
    if steps<1 or abs(steps*fsi.time.dt*1000-args.probe_ms)>1e-10:raise ValueError('whole probe steps required')
    if args.ramp_steps<0 or args.ramp_steps>steps:raise ValueError('ramp must fit inside the probe')
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('choose an empty output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    signature=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
    array=lambda v:v.detach().cpu().numpy()
    mesh=generate_circular_tube(geometry,device='cuda')
    solid=LayeredCircularTubeSolid(mesh,geometry,
        **{k:v for k,v in c['solid'].items() if k!='end_condition'},
        **{k:v for k,v in c['active'].items() if k not in ('mechanism','rest_reference')},dt_s=fsi.time.dt)
    X=mesh.X[solid.ep_nodes]
    radial=X[:,:2]-X.new_tensor(geometry.center_xy_cm);radial/=radial.norm(dim=-1,keepdim=True)
    sheet=torch.cat((radial,radial.new_zeros((len(X),1))),1)
    axial=X.new_tensor((0.,0.,1.)).expand_as(X)
    ep=Monodomain(X,solid.ep_cells,axial,sheet,parameters=params,numerics=numerics)
    ep.prepare(c['time']['ep_dt_s']);ode=GPBLand(c['time']['ep_dt_s'],c['time']['ode_dt_ms'])
    stim=c['stimulus'];mask=(X[:,2]<=geometry.axial_start_cm+stim['width_cm']+1e-12).to(X.dtype)
    pulse=lambda t:mask*(stim['amplitude_ua_per_cm2']*max(0.,min(t+fsi.time.dt,stim['start_s']+stim['duration_s'])-max(t,stim['start_s']))/fsi.time.dt)
    with np.load(args.source) as source:
        if not np.array_equal(source['reference'],array(mesh.X)) or not np.array_equal(source['ep_node_ids'],array(solid.ep_nodes)):
            raise ValueError('source mesh or EP domain mismatch')
        convert=lambda key:torch.from_numpy(source[key].copy()).to(mesh.X)
        fixed_x=convert('x');reference_tension=convert('resting_tension_dyn_cm2')
        source_time=float(source['time_s'])
        tissue=TissueState(convert('voltage_v'),convert('gpb'),convert('land'))
        stretch,zs,zw=(convert(k) for k in ('stretch','zs','zw'))
    solid.set_reference(reference_tension)
    try:
        if args.prepared.exists():
            with np.load(args.prepared) as cache:
                if str(cache['signature'])!=signature:raise ValueError('prepared-state config mismatch')
                convert=lambda key:torch.from_numpy(cache[key].copy()).to(mesh.X)
                tissue=TissueState(convert('voltage_v'),convert('gpb'),convert('land'))
                stretch,zs,zw=(convert(k) for k in ('stretch','zs','zw'))
                electrical_time=float(cache['electrical_time_s'])
            preparation=json.loads(args.prepared.with_suffix('.json').read_text())
        else:
            tstep=round(source_time/fsi.time.dt);limit=round(args.preparation_max_ms/(1000*fsi.time.dt))
            if limit<=tstep:raise ValueError('preparation ends before source')
            preparation_rows=[];started=perf_counter()
            while tstep<limit:
                t=tstep*fsi.time.dt
                tissue,_,_=ode.step(ep,tissue,stretch,zs,zw,t,pulse(t))
                zs=distortion_step(zs,torch.zeros_like(stretch),fsi.time.dt,40.14)
                zw=distortion_step(zw,torch.zeros_like(stretch),fsi.time.dt,405.86)
                tstep+=1
                if tstep%100==0 or tstep==limit:
                    solid.stage(fixed_x,tissue,zs,zw)
                    tension=float(solid.applied_tension(fixed_x).max())/10000
                    row=dict(electrical_time_ms=tstep*fsi.time.dt*1000,tension_kpa=tension,
                        voltage_min_mv=float(tissue.voltage_v.min()*1000),voltage_max_mv=float(tissue.voltage_v.max()*1000))
                    preparation_rows.append(row);print('Preparation '+json.dumps(row),flush=True)
                    if tension>=args.target_tension_kpa:break
            electrical_time=tstep*fsi.time.dt
            preparation=dict(scope='fixed-geometry EP/GPB/Land preparation; no FSI advancement',source=str(args.source),
                source_time_s=source_time,electrical_time_s=electrical_time,elapsed_seconds=perf_counter()-started,
                target_tension_kpa=args.target_tension_kpa,target_reached=preparation_rows[-1]['tension_kpa']>=args.target_tension_kpa,
                history=preparation_rows,signature=signature)
            atomic_npz(args.prepared,signature=signature,electrical_time_s=electrical_time,
                voltage_v=array(tissue.voltage_v),gpb=array(tissue.gpb),land=array(tissue.land),
                stretch=array(stretch),zs=array(zs),zw=array(zw))
            atomic_json(args.prepared.with_suffix('.json'),preparation)
        atomic_json(args.output/'preparation.json',preparation)
        if args.prepare_only:
            atomic_json(args.output/'report.json',dict(status='completed',scope='electrical preparation only',preparation=preparation));return
        if not preparation['target_reached']:raise ValueError('prepared tension did not reach the requested test threshold')
        nominal_scale=0. if args.passive else c['active']['tension_scale']
        # Mutable scalar buffer avoids recompiling material/tangent kernels for
        # each envelope value. This exists only in this verification driver.
        solid.scale=mesh.X.new_tensor(0. if args.ramp_steps else nominal_scale)
        actual=json.loads(json.dumps(c));actual['active']['tension_scale']=nominal_scale
        actual['fsi']['flow']['helmholtz_solver']=args.viscous_solver
        actual['time']['end_time_s']=steps*fsi.time.dt
        atomic_json(args.output/'config.resolved.json',actual)
        atomic_json(args.output/'probe.resolved.json',dict(source=str(args.source),prepared=str(args.prepared),
            fixed_geometry_preparation=True,physiological_restart=False,
            electrical_start_s=electrical_time,mechanical_start_s=0.,probe_duration_ms=args.probe_ms,
            steps=steps,passive_control=args.passive,dt_s=fsi.time.dt,
            probe_only_load_ramp_steps=args.ramp_steps,production_load_law_changed=False))
        solid.stage(fixed_x,tissue,zs,zw)
        setup_started=perf_counter()
        driver=BEIBStepper(solid,fsi,'cuda')
        domain=SealedDomain(driver.grid,geometry,c['hydraulic_domain'],mesh.X,dimension=3)
        domain.bind_wall(mesh);domain.validate_wall(fixed_x)
        driver.flow=RCBoxFlow(SealedBEFlow(driver.flow,domain,fused=True,graphs=True,
            helmholtz_solver=args.viscous_solver),parse_rc(c['circulation'],dimension=3),mesh.X)
        state=driver.initialize(fixed_x)
        budget=TubeBudget(mesh,domain,initial_x=fixed_x)
        prior_history=[]
        resumed_steps=0
        if args.resume_probe:
            controls=json.loads((args.resume_probe.parent/'probe.resolved.json').read_text())
            if controls['probe_only_load_ramp_steps']!=args.ramp_steps or controls['passive_control']!=args.passive:
                raise ValueError('resume must preserve the verification loading envelope')
            with np.load(args.resume_probe) as old:
                get=lambda key:torch.from_numpy(old[key].copy()).to(mesh.X)
                if not np.array_equal(old['reference'],array(mesh.X)):raise ValueError('resume mesh differs')
                resumed_steps=int(old['step']);clock=float(old['probe_time_s'])
                if abs(float(old['electrical_time_s'])-clock-electrical_time)>1e-12:
                    raise ValueError('resume electrical clock differs from the preparation')
                state=replace(state,step=resumed_steps,time=clock,x=get('x'),previous_x=get('previous_x'),
                    velocity=tuple(get(k) for k in ('u','v','w')),pressure=get('pressure'),force_time=clock,pressure_time=clock)
                tissue=TissueState(get('voltage_v'),get('gpb'),get('land'))
                stretch,zs,zw=(get(k) for k in ('stretch','zs','zw'))
                driver.flow.pressure=get('rc_pressure_dyn_cm2')
                driver.flow.committed_steps=int(old['rc_committed_steps'])
                driver.flow.integrated_outlet_flow=get('rc_integrated_outlet_ml')
                budget.cumulative.copy_(get('tube_budget_cumulative_ml'));budget.steps=int(old['tube_budget_steps'])
            domain.validate_wall(state.x);domain.accepted(state.velocity)
            solid.scale.fill_(nominal_scale*min(1.,resumed_steps/args.ramp_steps) if args.ramp_steps else nominal_scale)
            solid.stage(state.x,tissue,zs,zw)
            prior_history=json.loads((args.resume_probe.parent/'history.json').read_text())
            if prior_history[-1]['step']!=resumed_steps:raise ValueError('resume history does not match the accepted state')
            for row in prior_history:row.setdefault('viscous_solver','jacobi')
            driver.flow.last_accepted=dict(pressure_dyn_cm2=driver.flow.pressure.clone(),
                flow_ml_s=driver.flow.flux(state.velocity),capacitor_energy_erg=.5*driver.flow.C*driver.flow.pressure.square())
        writer=MACWriter(args.output/'vtk',solid,driver.grid,pressure_description='active-load component verification; independently prepared EP clock; not physiological restart')
        frames=[]
        def write_frame():
            writer.write(state)
            name=f'ep_{state.step:06d}.vtu'
            writer.meshio.write(args.output/'vtk'/name,writer.meshio.Mesh(array(state.x[solid.ep_nodes]),
                [('tetra',array(solid.ep_cells))],point_data=dict(voltage_mV=array(tissue.voltage_v)*1000,
                calcium_uM=array(tissue.gpb[:,37])*1000,circumferential_stretch=array(stretch))),binary=True)
            frames.append((state.time,name));_ep_collection(args.output/'vtk',frames)
        domain.write_geometry(args.output/'vtk');write_frame()
        torch.cuda.synchronize();setup_seconds=perf_counter()-setup_started
        history=list(prior_history);x_history=[array(state.x)];x_steps=[state.step]
        initial_volume=float(budget.initial_moving_volume)
        pcg_before=0
        def sample(info=None,step_seconds=0.):
            info=info or {};n=info.get('nonlinear',{})
            return dict(step=state.step,probe_time_ms=state.time*1000,electrical_time_ms=(electrical_time+state.time)*1000,
                applied_scale=float(solid.scale),
                **solid.diagnostics(state.x),**driver.flow.diagnostics(),**domain.diagnostics(),**budget.diagnostics(state.velocity),
                max_divergence_s_inv=float(divergence(state.velocity,driver.grid.spacing).abs().max()),
                voltage_min_mv=float(tissue.voltage_v.min()*1000),voltage_max_mv=float(tissue.voltage_v.max()*1000),
                nonlinear_iterations=n.get('iterations',0),gmres_iterations=n.get('gmres_iterations',0),
                fluid_solves=n.get('fluid_solves',0),helmholtz_sweeps=n.get('helmholtz_sweeps',0),
                nonlinear_residual=n.get('residual_norm',0.),nonlinear_tolerance=n.get('tolerance',0.),step_seconds=step_seconds)
        if not history:history.append(sample())
        started=perf_counter()
        try:
            for k in range(state.step,steps):
                print(f'Probe starting step={k+1}',flush=True);step_start=perf_counter()
                pcg_before=driver.flow.helmholtz_pcg_iterations
                clock=electrical_time+state.time
                trial_tissue,_,_=ode.step(ep,tissue,stretch,zs,zw,clock,pulse(clock))
                solid.scale.fill_(nominal_scale*min(1.,(k+1)/args.ramp_steps) if args.ramp_steps else nominal_scale)
                solid.stage(state.x,trial_tissue,zs,zw)
                trial,info=driver.step(state,diagnostics=True)
                new_stretch=solid.nodal_stretch(ep,trial.x)
                new_zs=distortion_step(zs,new_stretch-stretch,fsi.time.dt,40.14)
                new_zw=distortion_step(zw,new_stretch-stretch,fsi.time.dt,405.86)
                domain.validate_wall(trial.x);driver.flow.commit(trial.velocity,step=trial.step)
                budget.accepted(state.x,trial.x,trial.velocity,fsi.time.dt)
                tissue,state,stretch,zs,zw=trial_tissue,trial,new_stretch,new_zs,new_zw
                torch.cuda.synchronize();seconds=perf_counter()-step_start
                row=sample(info,seconds)
                row['helmholtz_pcg_iterations']=driver.flow.helmholtz_pcg_iterations-pcg_before
                row['viscous_solver']=args.viscous_solver
                history.append(row);x_history.append(array(state.x));x_steps.append(state.step)
                atomic_json(args.output/'history.json',history)
                print('Probe '+json.dumps(row),flush=True)
            write_frame()
            report=dict(status='completed',scope='short three-field active-load verification after fixed-geometry electrical preparation',
                physiological_restart=False,passive_control=args.passive,three_fields_advanced_during_probe=True,
                viscous_solver=args.viscous_solver,
                probe_only_load_ramp_steps=args.ramp_steps,production_load_law_changed=False,
                prepared=str(args.prepared),electrical_start_s=electrical_time,probe_duration_ms=state.time*1000,
                resumed_from=str(args.resume_probe) if args.resume_probe else None,resumed_steps=resumed_steps,
                accepted_steps=state.step,setup_seconds=setup_seconds,elapsed_seconds=perf_counter()-started,
                initial_lumen_ml=initial_volume,relative_volume_change=(history[-1]['lumen_volume_ml']-initial_volume)/initial_volume,
                final=history[-1],output=writer.summary())
            atomic_json(args.output/'report.json',report)
        except Exception as exc:
            report=dict(status='failed',accepted_steps=state.step,error=str(exc),elapsed_seconds=perf_counter()-started)
            if hasattr(exc,'result'):report['nonlinear_result']=str(exc.result)
            atomic_json(args.output/'report.json',report);raise
        finally:
            atomic_json(args.output/'history.json',history)
            atomic_npz(args.output/'final_state.npz',x=array(state.x),previous_x=array(state.previous_x if state.previous_x is not None else state.x),
                reference=array(mesh.X),cells=array(mesh.cells),u=array(state.velocity[0]),v=array(state.velocity[1]),w=array(state.velocity[2]),
                pressure=array(state.pressure),gpb=array(tissue.gpb),land=array(tissue.land),voltage_v=array(tissue.voltage_v),
                stretch=array(stretch),zs=array(zs),zw=array(zw),step=state.step,probe_time_s=state.time,
                electrical_time_s=electrical_time+state.time,**budget.checkpoint(),**driver.flow.checkpoint())
            atomic_npz(args.output/'coordinates_history.npz',steps=np.array(x_steps),x=np.stack(x_history))
    finally:ep.close()


if __name__=='__main__':main()
