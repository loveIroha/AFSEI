"""Read-only basal kinematics and pressure-transfer audit; no time advancement."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from afsi_torch.ep import mac_balanced_checkpoint as restart
from afsi_torch.ep.gao_tracts import prepare_cut_flow
from afsi_torch.ep.mac_transport import closed_volume_gradient
from afsi_torch.ep.mac_preload import MACEndocardialPressure
from afsi_torch.ep.mac_mass_budget import peskin_trace
from afsi_torch.mac.sealed_pressure import divergence,gradient
from afsi_torch.ep.windkessel import MMHG_TO_DYN_CM2 as C


def native(v):
    if isinstance(v,torch.Tensor):return v.item() if v.numel()==1 else v.tolist()
    if isinstance(v,dict):return {k:native(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [native(x) for x in v]
    return v


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('configuration',type=Path)
    args=p.parse_args();cfg=json.loads(args.configuration.read_text())
    out=Path(cfg['output']);out.mkdir(exist_ok=False,parents=True)
    torch.set_num_threads(4)
    started=perf_counter()
    d,s,_=restart.load(cfg['resume'],cardiac=True)
    e=d._fsi_stepper(d.maximum_ticks);x=s.fsi.x
    load=MACEndocardialPressure(e.model)
    rim=torch.unique(load.rim);axis=2
    dt=d.maximum_ticks*d.protocol.dt_ep_s
    with np.load(cfg['geometry_anchor'],allow_pickle=False) as a:
        d.gao_tract_anchor=torch.as_tensor(a['x'],device=x.device,dtype=x.dtype)
    report=dict(source=cfg['resume'],time_ms=s.fsi.time*1000,
                advances_time=False,changes_support=False,beta=d.config.fsi.beta,
                rim_nodes=len(rim),dt_s=dt,connections={})

    def describe_velocity(v,storage):
        return dict(rim_mean_axial_cm_s=v[rim,axis].mean(),
            rim_min_axial_cm_s=v[rim,axis].min(),rim_max_axial_cm_s=v[rim,axis].max(),
            rim_velocity_norm_cm_s=v[rim].norm(),
            lv_volume_rate_ml_s=(closed_volume_gradient(x,load.faces,load.rim)*v).sum(),
            tract_volume_rate_ml_s=(storage.discrete(x,x)*v).sum())

    # Evaluate the original transfer on exactly the same saved geometry.
    old_stencil=e.transfer.prepare(x)
    old_interp,_=e.transfer.interpolate(s.fsi.velocity,old_stencil,check_finite=False)
    d.gao_cut_boundary=True
    prepare_cut_flow(e,s.fsi,s);f=e.flow;storage=f.domain.storage
    report['tract_geometry']=f.domain.metadata
    report['saved_geometric_velocity']=describe_velocity((x-s.fsi.previous_x)/dt,storage)
    report['old_interpolation']=describe_velocity(old_interp,storage)
    report['old_stencil']=old_stencil.report
    report['rim_displacement_from_anchor_cm']=(x[rim]-d.gao_tract_anchor[rim]).mean(0)
    report['tract_volume_ml']=storage.volume(x)

    # Compare both traces on the migrated velocity; only fixed-wall masks
    # and its documented divergence projection are applied here.
    masked=f.domain.mask(s.fsi.velocity)
    q,info=f.pressure_solver.solve(-f.rho/f.dt*divergence(masked,e.grid.spacing))
    u=tuple(a-f.dt/f.rho*b for a,b in zip(masked,gradient(q,e.grid.spacing,f.domain.masks)))
    stencil=e.transfer.prepare(x)
    base,_=e.transfer.base.interpolate(u,stencil.original)
    full,_=e.transfer.interpolate(u,stencil,check_finite=False)
    old_same,_=e.transfer.interpolate(u,old_stencil,check_finite=False)
    for name,v in [('base_on_migrated_velocity',base),('old_on_migrated_velocity',old_same),
                   ('new_on_migrated_velocity',full),('new_global_correction',full-base)]:
        report['connections'][name]=describe_velocity(v,storage)
    report['new_stencil']=stencil.report
    report['global_dual_rim_square_fraction']=(stencil.dual[rim].square().sum()/stencil.dual.square().sum())
    report['new_pressure_mode_net_force_dyn']=stencil.mode.sum(0)

    # Algebraic probe only: constants violate fixed-wall normal masks,
    # so this is not a physical boundary-admissible pass/fail test.
    rigid=[]
    for a in range(3):
        velocity=tuple(torch.full_like(v,float(c==a)) for c,v in enumerate(s.fsi.velocity))
        v,_=e.transfer.interpolate(velocity,stencil,check_finite=False)
        expected=torch.zeros_like(x);expected[:,a]=1.
        rigid.append(dict(axis=a,max_nodal_error=(v-expected).abs().max(),
            rim_mean_error=(v-expected)[rim].mean(0)))
    report['new_constant_velocity_reproduction']=rigid
    report['constant_velocity_probe_boundary_admissible']=False

    # Independent geometry check along the saved original trajectory.
    report['saved_trajectory']=[]
    for path in cfg.get('trajectory',[]):
        with np.load(path,allow_pickle=False) as a:
            y=torch.as_tensor(a['x'],device=x.device,dtype=x.dtype)
            yp=torch.as_tensor(a['previous_x'],device=x.device,dtype=x.dtype)
            meta=json.loads(str(a['metadata']))
        report['saved_trajectory'].append(dict(path=path,
            volume_ml=float(load.volume(y)),tract_volume_ml=storage.volume(y),
            tract_rate_ml_s=(storage.volume(y)-storage.volume(yp))/dt,
            rim_mean_axial_displacement_cm=float((y[rim]-d.gao_tract_anchor[rim])[:,axis].mean())))
    np.savez_compressed(out/'frozen_fields.npz',x=x.cpu().numpy(),
        old_nodal_velocity=old_interp.cpu().numpy(),new_base_velocity=base.cpu().numpy(),
        new_nodal_velocity=full.cpu().numpy(),rim=rim.cpu().numpy(),
        **{f'fluid_velocity_{c}':v.cpu().numpy() for c,v in enumerate(u)})
    report['elapsed_seconds']=perf_counter()-started
    (out/'report.json').write_text(json.dumps(native(report),indent=2,allow_nan=False)+'\n')
    print(json.dumps(native(report),indent=2,allow_nan=False))


if __name__=='__main__':main()
