"""Production EP/GPB/Land history on the original 3936-node tube mesh.

Mechanics-independent isometric-driver branch only. The tape is shared input
for an explicitly scoped FSI benchmark; replay does not constitute external EP.
"""
from dataclasses import replace
from pathlib import Path
import argparse,json,struct,time,hashlib
import numpy as np,torch
from afsi_torch.simulation.circular_tube_config import resolve_config
from afsi_torch.solids.circular_tube import generate_circular_tube
from afsi_torch.ep.monodomain import Monodomain
from afsi_torch.ep.tissue import GPBLand,initial_state
from afsi_torch.ep.planar_tube import land_resting_equilibrium
from afsi_torch.ep.land import land_tension

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();c=json.loads(args.config.read_text());g,fsi,par,num=resolve_config(c)
    if c['active']['stretch_feedback']!='isometric-driver':raise ValueError('history replay requires mechanics-independent Land inputs')
    out=args.output;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    mesh=generate_circular_tube(g,device='cuda:0');X=mesh.X;n=len(X);dt=fsi.time.dt;steps=round(.3/dt)
    radial=X[:,:2]-X.new_tensor(g.center_xy_cm);radial/=radial.norm(dim=1,keepdim=True)
    sheet=torch.column_stack((radial,torch.zeros(n,device=X.device,dtype=X.dtype)))
    axial=X.new_tensor([0.,0.,1.]).expand_as(X);ep=Monodomain(X,mesh.cells,axial,sheet,parameters=par,numerics=num);ep.prepare(dt)
    ode=GPBLand(dt,c['time']['ode_dt_ms']);tissue=initial_state(X[:,0]);one=torch.ones_like(tissue.voltage_v);zero=torch.zeros_like(one)
    tissue=replace(tissue,land=land_resting_equilibrium(tissue.gpb[:,37]*1000,one))
    rest=land_tension(one,one,tissue.land[:,0],tissue.land[:,1],zero,zero,dt)
    history=torch.empty((steps+1,n),device=X.device,dtype=X.dtype);history[0]=tissue.land[:,0]
    stim=c['stimulus'];mask=(X[:,2]<=g.axial_start_cm+stim['width_cm']+1e-12).to(X.dtype)
    torch.cuda.synchronize();started=time.perf_counter()
    for k in range(steps):
        t=k*dt;overlap=max(0.,min(t+dt,stim['start_s']+stim['duration_s'])-max(t,stim['start_s']))
        tissue,_,_=ode.step(ep,tissue,one,zero,zero,t,mask*(stim['amplitude_ua_per_cm2']*overlap/dt))
        history[k+1]=tissue.land[:,0]
    torch.cuda.synchronize();seconds=time.perf_counter()-started
    data=history.cpu().numpy();path=out/'ep_land_xs.bin'
    with path.open('wb') as f:
        f.write(struct.pack('<QQd',n,steps+1,dt));f.write(np.asarray(rest.cpu(),dtype='<f8').tobytes());f.write(np.asarray(data,dtype='<f8').tobytes())
    np.savez_compressed(out/'final_electrical_state.npz',voltage_v=tissue.voltage_v.cpu().numpy(),gpb=tissue.gpb.cpu().numpy(),land=tissue.land.cpu().numpy())
    # Full electrical state must agree with an existing unchanged-drive baseline.
    baseline=Path('/mnt/large2/gjh/AFSEI/results/volumetric_stabilization_20261025/from_rest_300ms/solid1/continued_300ms/final_state.npz')
    a=np.load(baseline);errors={key:float(np.max(np.abs(value-a[key]))) for key,value in (
        ('voltage_v',tissue.voltage_v.cpu().numpy()),('gpb',tissue.gpb.cpu().numpy()),('land',tissue.land.cpu().numpy()))}
    passed=errors['voltage_v']<1e-9 and errors['gpb']<1e-8 and errors['land']<1e-9
    report=dict(status='completed',time_s=.3,nodes=n,steps=steps,compute_seconds=seconds,
        backend='existing AFSEI production GPU EP/ODE solvers',mechanics_independent=True,full_electrical_baseline_errors=errors,
        baseline_state_agreement_passed=passed,tape_sha256=hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),
        external_replay_scope='FSI with the same EP-derived loading, not an independent external three-field solver')
    (out/'report.json').write_text(json.dumps(report,indent=2));ep.close();print(json.dumps(report,indent=2))
    if not passed:raise RuntimeError('electrical tape differs from the production baseline; do not use for matched forcing')

if __name__=='__main__':main()
