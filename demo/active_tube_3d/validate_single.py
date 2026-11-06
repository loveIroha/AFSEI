"""Single-layer test using AFSEI's existing EP and coupled solvers.

EP mode keeps the exact stimulus time and computes on a fixed reference wall.
Coupled controls use the same geometry, fluid, material and solver tolerances.
No preactivation, probe-only force ramp or prescribed displacement is used.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter
import xml.etree.ElementTree as ET
import numpy as np
import torch
from afsi_torch.simulation.circular_tube_config import resolve_config
from afsi_torch.simulation.tube_validation import propagation_metrics,contraction_metrics
from afsi_torch.solids.circular_tube import generate_circular_tube
from afsi_torch.cycle_checkpoint import atomic_json


def array(v):return v.detach().cpu().numpy()


@torch.no_grad()
def fixed_ep(c,out,device):
    from afsi_torch.ep.monodomain import Monodomain
    from afsi_torch.ep.tissue import GPBLand,initial_state
    from afsi_torch.ep.planar_tube import land_resting_equilibrium
    from afsi_torch.ep.land import land_tension
    from afsi_torch.ep.circular_tube import CircularTubeSolid
    g,_,params,numerics=resolve_config(c)
    if out.exists() and any(out.iterdir()):raise ValueError('choose an empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'config.resolved.json',c)
    mesh=generate_circular_tube(g,device=device);X=mesh.X
    axial=X.new_tensor((0.,0.,1.)).expand_as(X)
    radial=X[:,:2]-X.new_tensor(g.center_xy_cm);radial/=radial.norm(dim=-1,keepdim=True)
    sheet=torch.cat((radial,radial.new_zeros((len(radial),1))),1)
    dt=c['time']['ep_dt_s'];steps=round(c['time']['end_time_s']/dt)
    ep=Monodomain(X,mesh.cells,axial,sheet,parameters=params,numerics=numerics)
    history=[];frames=[];activation=X.new_full((len(X),),float('inf'))
    try:
        ep.prepare(dt);ode=GPBLand(dt,c['time']['ode_dt_ms']);tissue=initial_state(X[:,0])
        one=torch.ones_like(tissue.voltage_v);zero=torch.zeros_like(one)
        tissue=replace(tissue,land=land_resting_equilibrium(tissue.gpb[:,37]*1000,one))
        ref=land_tension(one,one,tissue.land[:,0],tissue.land[:,1],zero,zero,dt)
        solid=CircularTubeSolid(mesh,g,
            **{k:v for k,v in c['solid'].items() if k!='end_condition'},
            **{k:v for k,v in c['active'].items() if k not in ('mechanism','rest_reference')},dt_s=dt,
            end_support=c['solid']['end_condition'])
        solid.set_reference(ref)
        stim=c['stimulus'];mask=(X[:,2]<=g.axial_start_cm+stim['width_cm']+1e-12).to(X.dtype)
        def sample(k):
            solid.stage(X,tissue,zero,zero);ta=solid.applied_tension(X)
            history.append(dict(time_s=k*dt,voltage_min_mv=float(tissue.voltage_v.min()*1000),
                voltage_max_mv=float(tissue.voltage_v.max()*1000),
                activated_fraction=float(torch.isfinite(activation).double().mean()),
                maximum_applied_tension_dyn_cm2=float(ta.max())))
        def frame(k):
            if not c['output']['write_vtk']:return
            import meshio
            folder=out/'vtk';folder.mkdir(exist_ok=True);name=f'ep_{k:06d}.vtu'
            meshio.write(folder/name,meshio.Mesh(array(X),[('tetra',array(mesh.cells))],
                point_data={'voltage_mV':array(tissue.voltage_v)*1000,
                    'activation_time_s':array(torch.where(torch.isfinite(activation),activation,-torch.ones_like(activation)))},
                cell_data={'applied_tension_dyn_cm2':[array((solid.applied_tension(X)*solid.average_weights).sum(1))]}))
            frames.append((k*dt,name))
        sample(0);frame(0);torch.cuda.synchronize();started=perf_counter()
        for k in range(steps):
            t=k*dt;overlap=max(0.,min(t+dt,stim['start_s']+stim['duration_s'])-max(t,stim['start_s']))
            trial,_,_=ode.step(ep,tissue,one,zero,zero,t,mask*(stim['amplitude_ua_per_cm2']*overlap/dt))
            crossed=(tissue.voltage_v<-.02)&(trial.voltage_v>=-.02)&torch.isinf(activation)
            fraction=(-.02-tissue.voltage_v)/(trial.voltage_v-tissue.voltage_v).clamp_min(1e-30)
            activation=torch.where(crossed,t+dt*fraction,activation);tissue=trial
            if (k+1)%c['output']['log_every']==0 or k+1==steps:sample(k+1)
            if (k+1)%c['output']['field_every']==0 or k+1==steps:frame(k+1)
        torch.cuda.synchronize();elapsed=perf_counter()-started
        np.savez_compressed(out/'final_state.npz',reference=array(X),x=array(X),cells=array(mesh.cells),
            voltage_v=array(tissue.voltage_v),activation_time_s=array(activation),gpb=array(tissue.gpb),land=array(tissue.land))
        atomic_json(out/'history.json',history)
        if frames:
            root=ET.Element('VTKFile',type='Collection',version='0.1',byte_order='LittleEndian');col=ET.SubElement(root,'Collection')
            for time,name in frames:ET.SubElement(col,'DataSet',timestep=repr(time),group='',part='0',file=name)
            ET.ElementTree(root).write(out/'vtk/ep.pvd',encoding='utf-8',xml_declaration=True)
        report=dict(status='completed',scope='fixed-reference EP/GPB/Land; no mechanics or fluid advance',
            elapsed_seconds=elapsed,ep_steps=steps,time_s=steps*dt,final=history[-1])
        atomic_json(out/'report.json',report)
    finally:ep.close()


def analyze(c,out,mode):
    g,_,_,_=resolve_config(c);m=generate_circular_tube(g)
    with np.load(out/'final_state.npz') as a:
        metrics=propagation_metrics(a['reference'],a['activation_time_s'],g.axial_start_cm,g.length_cm,c['stimulus']['width_cm'])
        mechanics=contraction_metrics(a['reference'],a['x'],array(m.inner_rings),g.center_xy_cm)
    history=json.loads((out/'history.json').read_text());report=json.loads((out/'report.json').read_text())
    result=dict(mode=mode,status=report['status'],propagation=metrics,mechanics=mechanics,
        elapsed_seconds=report['elapsed_seconds'],full_cycle_validated=False,
        literature_replication=False,convergence_validated=False,
        peak_sampled_applied_tension_kpa=max(r['maximum_applied_tension_dyn_cm2'] for r in history)/1e4)
    atomic_json(out/'validation.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(14,4))
    rows=metrics['sections'];axs[0].plot([r['z_cm'] for r in rows],[r['activation_mean_ms'] for r in rows],'.-')
    axs[0].set(xlabel='Axial reference position (cm)',ylabel='Activation time (ms)',title='Threshold: -20 mV')
    axs[1].plot([r['time_s']*1000 for r in history],[r['maximum_applied_tension_dyn_cm2']/1e4 for r in history])
    axs[1].set(xlabel='Time (ms)',ylabel='Maximum applied tension (kPa)')
    if mode=='ep':axs[2].text(.5,.5,'Fixed wall: no mechanical advance',ha='center',transform=axs[2].transAxes)
    else:axs[2].plot([r['time_s']*1000 for r in history],[100*(1-r['minimum_radius_ratio']) for r in history])
    axs[2].set(xlabel='Time (ms)',ylabel='Minimum-radius reduction (%)')
    for ax in axs:ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/'validation.png',dpi=160);plt.close(fig)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['ep','coupled','no-stimulus','passive'],default='coupled')
    p.add_argument('--end-ms',type=float);p.add_argument('--device',default='cuda:0')
    p.add_argument('--resume',type=Path)
    a=p.parse_args();c=json.loads(a.config.read_text())
    if 'layers' in c['solid']:raise ValueError('single-layer validation only')
    if a.end_ms is not None:c['time']['end_time_s']=a.end_ms/1000
    if a.mode=='no-stimulus':c['stimulus']['amplitude_ua_per_cm2']=0.
    if a.mode=='passive':c['active']['tension_scale']=0.
    if a.mode=='ep':
        if a.resume is not None:raise ValueError('EP-only resume is not implemented')
        fixed_ep(c,a.output,a.device)
    else:
        from afsi_torch.simulation.circular_tube import run
        run(c,a.output,device=a.device,resume=a.resume)
    print(json.dumps(analyze(c,a.output,a.mode),indent=2))


if __name__=='__main__':main()
