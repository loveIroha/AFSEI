"""Source-UFL formula / GPU assembly at identical geometry, state and pressure.

This is a matched-load constitutive/weak-force comparison, not a new PV run.
No NS solve, EP integration, Ta scaling or accepted-state change is performed.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace
import hashlib,json
import numpy as np
import torch
from afsi_torch.ep import mac_balanced_checkpoint as restart
from afsi_torch.ep.mac_preload import MACEndocardialPressure
from afsi_torch.ep.active_diagnostics import active_metrics
from afsi_torch.ep.windkessel import MMHG_TO_DYN_CM2 as C


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('configuration',type=Path)
    cfg=json.loads(p.parse_args().configuration.read_text());out=Path(cfg['output'])
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    path=Path(cfg['checkpoint']);before=hashlib.sha256(path.read_bytes()).hexdigest()
    d,s,_=restart.load(path,cardiac=True);load=d.buffer
    x=s.fsi.x;g=load.geometry
    arrays=lambda value:value.detach().cpu().numpy()
    F=arrays(d.model.element_gradient(x));old=arrays(load.previous_F)
    fiber=arrays(d.mesh.fiber);fields=arrays(load.fields);weights=arrays(g.weights)
    gradient=arrays(g.gradients[:,0]);cells=arrays(g.cells)
    ff=np.einsum('eij,ej->ei',F,fiber);of=np.einsum('eij,ej->ei',old,fiber)
    lam=np.linalg.norm(ff,axis=1)[:,None];previous=np.linalg.norm(of,axis=1)[:,None]
    xs,xw,zs,zw=fields.transpose(2,0,1);dt=s.force_load.dt_s
    lc=np.minimum(lam,1.2);h=np.maximum(0,1+2.3*(lc+np.minimum(lc,.87)-1.87))
    zs=zs*np.exp(-40.14*dt)+10*(lam-previous)*(-np.expm1(-40.14*dt)/(40.14*dt))
    zw=zw*np.exp(-405.86*dt)+10*(lam-previous)*(-np.expm1(-405.86*dt)/(405.86*dt))
    ta=np.maximum(0,4.8e6*h*(xs*(zs+1)+xw*zw))
    if not s.force_load.use_land:
        ta=arrays(load.anchor)*s.force_load.ramp_factor
    coefficient=(weights*ta*(1+4.9*(np.maximum(lam,1)-1))).sum(-1)
    directional=np.einsum('eaJ,eJ->ea',gradient,fiber)
    local=-np.einsum('e,ei,ea->eai',coefficient,ff,directional)
    source=np.zeros_like(arrays(x));np.add.at(source,cells.reshape(-1),local.reshape(-1,3))
    actual=arrays(load.force(x));norm=lambda f:float(np.linalg.norm(f))
    active_error=norm(source-actual)/max(norm(source),1.)
    ta_error=float(np.max(abs(ta-arrays(load.tension(x)))))
    pressure=MACEndocardialPressure(d.model)
    coords=arrays(x);faces=arrays(pressure.faces);tri=coords[faces]
    unit=-C*.5*np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])/3
    source_pressure=np.zeros_like(coords)
    for k in range(3):np.add.at(source_pressure,faces[:,k],unit)
    velocity=arrays((x-s.force_load.previous_x)/dt)
    comparisons=[]
    for value in cfg.get('pressures_mmhg',[8.,100.]):
        reference=source+value*source_pressure
        gpu=actual+arrays(pressure.force(x,value))
        comparisons.append(dict(pressure_mmhg=value,weak_force_relative_error=norm(gpu-reference)/max(norm(reference),1.),
            source_total_force_norm_dyn=norm(reference),afsei_total_force_norm_dyn=norm(gpu),
            source_power_w=float(np.sum(reference*velocity))/1e7,
            afsei_power_w=float(np.sum(gpu*velocity))/1e7))
    passed=active_error<1e-10 and ta_error<1e-6 and all(c['weak_force_relative_error']<1e-10 for c in comparisons)
    report=dict(passed=passed,scope='same geometry/internal fields/pressure; active and pressure weak force',
        time_ms=s.fsi.time*1000,source_commit='5bdde73c6be10cefadb005a0fcec5fdb6521f336',
        active_force_relative_error=active_error,ta_max_absolute_error_dyn_cm2=ta_error,
        active_metrics=active_metrics(d.model,load,x),comparisons=comparisons,
        physical_time_advanced=False,new_fluid_solves=0,tension_scale_changed=False,
        checkpoint_sha256_before=before,checkpoint_sha256_after=hashlib.sha256(path.read_bytes()).hexdigest(),
        pump_performance_equivalence_proved=False)
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
