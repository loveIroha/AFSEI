"""Compute the full source-duration passive loading history for fixed LV tracts."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from afsi_torch.ep import mac_balanced_checkpoint as restart
from afsi_torch.ep.gao_tracts import LVTractDomain
from afsi_torch.ep.tract_storage import TractStorage
from afsi_torch.ep.tract_preload import TractPressureLoad
from afsi_torch.ep.mac_rcr_coupled import outlet_rows
from afsi_torch.ep.mac_preload import mesh_binding
from afsi_torch.ep.checkpoint import digest
from afsi_torch.two_way.preload import PassivePreloader,PreloadConfig,SourceLoadingHistory


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('configuration',type=Path)
    cfg=json.loads(p.parse_args().configuration.read_text())
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    d,_,_=restart.load(cfg['reference_checkpoint'],cardiac=True)
    e=d._fsi_stepper(d.maximum_ticks)
    with np.load(cfg['geometry_anchor'],allow_pickle=False) as a:
        anchor=torch.as_tensor(a['geometry_anchor'] if 'geometry_anchor' in a else a['x'],device=d.mesh.X.device)
    cap=outlet_rows(e,anchor).prepared.cap
    domain=LVTractDomain(e.grid,cap,anchor,mode='filling',**cfg.get('cut_geometry',{}))
    if cfg.get('tract_geometry_gpu',False):
        from afsi_torch.ep.tract_storage_gpu import GPUTractStorage
        domain.storage=GPUTractStorage(cap,domain)
    else:domain.storage=TractStorage(cap,domain)
    domain.validate_wall(cap,d.mesh.X);domain.storage.volume(d.mesh.X)
    ramp_s=cfg.get('ramp_s',d.config.start_time_s)
    history=SourceLoadingHistory(d,8.,d.config.start_time_s,100,ramp_s=ramp_s)
    settings=PreloadConfig(nominal_steps=history.ramp_steps)
    report=dict(configuration=cfg,status='running',loading_steps=100,duration_s=d.config.start_time_s,
        ep_ode_advanced=False,circulation_advanced=False,geometry=domain.metadata,records=[])
    started=perf_counter()
    def observe(entry,x):
        history.observe(entry,x)
        report['records'].append(entry)
        if history.accepted%10==0:
            (out/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(dict(loading_samples=history.accepted,pressure_mmhg=entry['pressure_mmhg'],
                force_residual_dyn=entry['residual_norm_dyn'],elapsed_seconds=perf_counter()-started)),flush=True)
    def validate(old,x):
        e.check_support(x);domain.validate_wall(cap,x);domain.storage.volume(x)
    loader=PassivePreloader(d,settings,pressure_load=TractPressureLoad(d.model,domain.storage),
                            geometry_validator=validate)
    try:
        result=loader.solve(observe_state=observe)
        while history.accepted<history.steps:
            entry=dict(pressure_mmhg=settings.target_mmhg,residual_norm_dyn=result.force_residual_norm_dyn,
                tolerance_dyn=result.force_tolerance_dyn,volume_ml=result.volume_ml,newton_iterations=0)
            history.observe_held_equilibrium(entry)
            report['records'].append(entry)
        h=history.final()
        data={k:getattr(h,k).cpu().numpy() for k in ('x','stretch','zs','zw')}
        data['geometry_anchor']=anchor.cpu().numpy()
        meta=dict(producer='afsei-fixed-tract-loading-history',schema=1,completed=True,
            units='cm-g-s; pressure mmHg; time s',mesh_binding_sha256=mesh_binding(d.mesh),
            config=asdict(d.config),pressure_mmhg=h.pressure_mmhg,duration_s=h.duration_s,
            steps=h.steps,force_residual_dyn=result.force_residual_norm_dyn,
            force_tolerance_dyn=result.force_tolerance_dyn,ep_ode_advanced=False,
            geometry=domain.metadata,loading_records=report['records'])
        meta['loading_schedule']=dict(kind='linear-ramp-then-hold',ramp_s=ramp_s,
            hold_s=d.config.start_time_s-ramp_s,total_s=d.config.start_time_s,
            target_mmhg=settings.target_mmhg,nominal_samples=100)
        meta['sha256']=digest(meta,data)
        with (out/'preload_history.npz').open('wb') as stream:
            np.savez_compressed(stream,metadata=json.dumps(meta,allow_nan=False),**data)
        report.update(status='passed',elapsed_seconds=perf_counter()-started,
            force_residual_dyn=result.force_residual_norm_dyn,force_tolerance_dyn=result.force_tolerance_dyn,
            lv_volume_ml=float(loader.pressure.closed.volume(result.x)),
            total_blood_volume_ml=result.volume_ml,loading_samples=history.accepted,
            zs_max_abs=float(h.zs.abs().max()),zw_max_abs=float(h.zw.abs().max()))
    except Exception as exc:
        report.update(status='failed',failure=type(exc).__name__+': '+str(exc),
                      loading_samples=history.accepted,elapsed_seconds=perf_counter()-started)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        raise
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)


if __name__=='__main__':main()
