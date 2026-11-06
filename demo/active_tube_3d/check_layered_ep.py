"""EP-only excitation check on the myocardial layer; no FSI advancement."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter
import torch
from afsi_torch.simulation.circular_tube_config import resolve_config
from afsi_torch.solids.circular_tube import generate_circular_tube
from afsi_torch.ep.layered_circular_tube import myocardium_ep_mesh
from afsi_torch.ep.monodomain import Monodomain
from afsi_torch.ep.tissue import GPBLand,initial_state
from afsi_torch.ep.planar_tube import land_resting_equilibrium
from afsi_torch.ep.land import land_tension
from afsi_torch.cycle_checkpoint import atomic_json


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--duration-ms',type=float,default=2.)
    a=p.parse_args();c=json.loads(a.config.read_text());g,_,params,numerics=resolve_config(c)
    if a.output.exists() and any(a.output.iterdir()):raise ValueError('choose an empty output directory')
    a.output.mkdir(parents=True,exist_ok=True)
    m=generate_circular_tube(g,device='cuda');nodes,_,cells=myocardium_ep_mesh(m);X=m.X[nodes]
    fiber=X.new_tensor((0.,0.,1.)).expand_as(X)
    radial=X[:,:2]-X.new_tensor(g.center_xy_cm);radial=radial/radial.norm(dim=-1,keepdim=True)
    sheet=torch.cat((radial,radial.new_zeros((len(radial),1))),1)
    dt=c['time']['ep_dt_s'];steps=round(a.duration_ms/(1000*dt))
    if steps<1 or abs(steps*dt*1000-a.duration_ms)>1e-10:raise ValueError('duration must contain whole EP steps')
    ep=Monodomain(X,cells,fiber,sheet,parameters=params,numerics=numerics);ep.prepare(dt)
    ode=GPBLand(dt,c['time']['ode_dt_ms']);tissue=initial_state(X[:,0]);one=torch.ones_like(tissue.voltage_v);zero=torch.zeros_like(one)
    tissue=replace(tissue,land=land_resting_equilibrium(tissue.gpb[:,37]*1000,one))
    stim=c['stimulus'];mask=(X[:,2]<=g.axial_start_cm+stim['width_cm']+1e-12).to(X.dtype)
    tetra=X[cells];vol=torch.linalg.det((tetra[:,1:]-tetra[:,:1]).transpose(-1,-2)).abs()/6
    fraction=(mask[cells].mean(1)*vol).sum()/vol.sum()
    active=torch.zeros_like(mask,dtype=torch.bool);peak=tissue.voltage_v.clone();minimum=peak.clone()
    # Translate the first pulse to time zero for this short verification only.
    torch.cuda.synchronize();start=perf_counter()
    try:
        for k in range(steps):
            t=k*dt;overlap=max(0.,min(t+dt,stim['duration_s'])-max(t,0.))
            tissue,_,_=ode.step(ep,tissue,one,zero,zero,t,mask*(stim['amplitude_ua_per_cm2']*overlap/dt))
            active|=tissue.voltage_v>=-.02
            peak=torch.maximum(peak,tissue.voltage_v);minimum=torch.minimum(minimum,tissue.voltage_v)
        torch.cuda.synchronize()
        report=dict(status='completed',scope='fixed myocardial EP/GPB/Land; pulse shifted to zero; no fluid or solid advance',
            duration_ms=a.duration_ms,steps=steps,nodes=len(nodes),cells=len(cells),
            stimulus_amplitude_ua_per_cm2=stim['amplitude_ua_per_cm2'],volume_weighted_stimulated_fraction=float(fraction),
            volume_mean_stimulus_ua_per_cm2=float(fraction*stim['amplitude_ua_per_cm2']),
            ever_activated_fraction=float(active.double().mean()),voltage_peak_mv=float(peak.max()*1000),
            voltage_min_mv=float(minimum.min()*1000),final_voltage_min_mv=float(tissue.voltage_v.min()*1000),
            final_voltage_max_mv=float(tissue.voltage_v.max()*1000),elapsed_seconds=perf_counter()-start,
            finite=bool(torch.isfinite(tissue.gpb).all() & torch.isfinite(tissue.land).all()))
        atomic_json(a.output/'report.json',report);atomic_json(a.output/'config.input.json',c)
        print(json.dumps(report,indent=2))
    finally:ep.close()


if __name__=='__main__':main()
