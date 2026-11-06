"""Discarded physical-cut-plane LV trials; never overwrite a cycle checkpoint."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter
import traceback
import hashlib
import numpy as np
import torch
from afsi_torch.ep import mac_balanced_checkpoint as restart
from afsi_torch.ep.mac_balanced_dynamic import BalancedAFSEIStepper
from afsi_torch.ep.mac_preload import MACEndocardialPressure
from afsi_torch.ep.gao_tracts import prepare_cut_flow
from afsi_torch.mac.sealed_pressure import divergence,gradient
from afsi_torch.mac.open_boundary import face_weights
from afsi_torch.ep.windkessel import MMHG_TO_DYN_CM2 as C


def native(value):
    if isinstance(value,torch.Tensor):
        return value.item() if value.numel()==1 else value.tolist()
    if isinstance(value,dict):return {k:native(v) for k,v in value.items() if not k.startswith('_')}
    if isinstance(value,(tuple,list)):return [native(v) for v in value]
    return value


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('configuration',type=Path)
    args=parser.parse_args()
    cfg=json.loads(args.configuration.read_text())
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    (out/'configuration.json').write_text(json.dumps(cfg,indent=2)+'\n')
    torch.set_num_threads(4)
    d,s,_=restart.load(cfg['resume'],cardiac=True)
    if s.mode not in ('filling','rcr') or d.spatial_coupling!='legacy-balanced':
        raise ValueError('this trial requires an open legacy LV snapshot')
    connection=cfg.get('connection','physical-cut')
    if connection not in ('physical-cut','legacy-sheet'):
        raise ValueError('connection must be physical-cut or legacy-sheet')
    d.gao_cut_boundary=connection=='physical-cut'
    # A failed acceptance audit is diagnostic evidence, not a reason to
    # repeat the same migration at another dt before inspecting its budget.
    d.config=replace(d.config,max_retries=0)
    if cfg.get('geometry_anchor'):
        with np.load(cfg['geometry_anchor'],allow_pickle=False) as archive:
            d.gao_tract_anchor=torch.as_tensor(archive['x'],device=s.fsi.x.device,dtype=s.fsi.x.dtype)
    e=d._fsi_stepper(d.maximum_ticks)
    started=perf_counter()
    report=dict(scope='short real-LV connection comparison; not a full cycle',
        connection=connection,
        source_sha256=hashlib.sha256(Path(cfg['resume']).read_bytes()).hexdigest(),
        source=cfg['resume'],geometry_anchor=cfg.get('geometry_anchor',cfg['resume']),
        source_time_ms=s.fsi.time*1000,mode=s.mode,
        initial_volume_ml=float(MACEndocardialPressure(e.model).volume(s.fsi.x)),
        dt_ms=d.maximum_ticks*d.protocol.dt_ep_s*1000,
        model=dict(ep='unchanged',gpb_land='unchanged',solid='unchanged',time='BE-BE',
            boundary=('physical MAC box faces at ends of added idealized fixed tracts'
                if d.gao_cut_boundary else 'original internal basal pressure sheet'),
            old_internal_pressure_sheet=('zero at the open basal connection'
                if d.gao_cut_boundary else 'original implementation')),
        accepted_steps=0,records=[])
    try:
        if d.gao_cut_boundary:
            prepare_cut_flow(e,s.fsi,s)
        f=e.flow
        if d.gao_cut_boundary:
            report['tract_geometry']=f.domain.metadata
        # New rigid partitions require an explicit, documented velocity
        # projection. Preserve x, tissue, strain and all material history.
        if d.gao_cut_boundary:
            old=s.fsi.velocity
            masked=f.domain.mask(old)
            correction,info=f.pressure_solver.solve(-f.rho/f.dt*divergence(masked,e.grid.spacing))
            g=gradient(correction,e.grid.spacing,f.domain.masks)
            u=tuple(v-f.dt/f.rho*w for v,w in zip(masked,g))
            energy=lambda values:.5*f.rho*sum((v*v*face_weights(e.grid,c,s.fsi.x)).sum() for c,v in enumerate(values))
            report['migration']=dict(kinetic_before_erg=energy(old).item(),kinetic_after_erg=energy(u).item(),
                max_divergence_s_inv=divergence(u,e.grid.spacing).abs().max().item(),
                geometry_tissue_strain_history_preserved=True,pressure_reference_shifted=False)
            from afsi_torch.ep.gao_tracts import cut_rows
            fresh_rows=cut_rows(e,s.cap_geometry)
            bp=torch.full_like(fresh_rows.area,s.cap_pressure.mean());bp[-2:]=0.
            s=replace(s,cap_pressure=bp,ep=replace(s.ep,fsi=replace(s.fsi,velocity=u)))
            s=replace(s,ep=replace(s.ep,fsi=replace(s.fsi,force=d.force_for_state(s))))
        else:
            report['migration']=dict(applied=False)
        for i in range(cfg.get('steps',3)):
            tick=s.tick+d.maximum_ticks
            # Hold the named boundary phase for this short diagnostic only;
            # tissue and original FSI drivers advance normally.
            step_started=perf_counter()
            s,info=BalancedAFSEIStepper.step(d,s,target_tick=tick,diagnostics=True)
            record=dict(time_ms=s.fsi.time*1000,balanced=info['balanced'],
                step_seconds=perf_counter()-step_started,
                nonlinear=info.get('nonlinear'),flow=info.get('flow'))
            report['records'].append(native(record));report['accepted_steps']+=1
            (out/'progress.json').write_text(json.dumps(native(report),indent=2,allow_nan=False)+'\n')
            print(json.dumps(native(record),allow_nan=False),flush=True)
        report['status']='passed'
    except Exception as exc:
        report.update(status='failed',failure=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        problem=getattr(e,'_last_cut_problem',None)
        if problem is not None and problem.last_data is not None:
            x,_,v,force,u,p,*_=problem.last_data
            rows=problem.rows
            actual=rows.flow(u)-rows.material_flow((x-problem.state.x)/e.flow.dt)
            boundary=e.flow.last_boundary
            report['failed_trial']=native(dict(volume_ml=v,basal_flux_ml_s=actual[-2:].sum(),
                finite_step_mass_error_ml_s=(v-problem.load.volume(problem.state.x))/e.flow.dt+actual[-1],
                cut=boundary,internal_sheet_mmhg=problem.last_cap_pressure[-1]/C,
                residual_norm=problem.last_residual.norm(),pressure_min_mmhg=p.min()/C,
                pressure_max_mmhg=p.max()/C))
            if hasattr(problem,'cut_storage'):
                storage=(problem.cut_storage.volume(x)-problem.cut_storage.volume(problem.state.x))/e.flow.dt
                report['failed_trial'].update(tract_volume_rate_ml_s=storage,
                    cavity_plus_tract_mass_error_ml_s=float((v-problem.load.volume(problem.state.x))/e.flow.dt)
                        +storage+float(boundary['outward_flux_ml_s'].sum()))
            report['failed_trial']['tract_budget']=native(getattr(e,'_cut_budget',None))
            report['attempts']=getattr(exc,'attempts',None)
            np.savez_compressed(out/'discarded_trial.npz',old_x=problem.state.x.cpu().numpy(),
                trial_x=x.cpu().numpy(),pressure=p.cpu().numpy(),
                **{f'velocity_{c}':v.cpu().numpy() for c,v in enumerate(u)})
        print(report['failure'],flush=True)
    finally:
        report['elapsed_seconds']=perf_counter()-started
        (out/'report.json').write_text(json.dumps(native(report),indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(status=report['status'],accepted_steps=report['accepted_steps'],
            elapsed_seconds=report['elapsed_seconds'],output=str(out))),flush=True)
    if report['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
