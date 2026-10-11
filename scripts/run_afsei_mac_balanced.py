"""Active pressure-balanced EP/MAC and actual-flow RCR interval validation."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter
import torch
from afsi_torch.ep import checkpoint
from afsi_torch.ep import mac_balanced_checkpoint as restart
from afsi_torch.ep.mac_balanced_dynamic import BalancedAFSEIStepper
from afsi_torch.ep.protocol import ticks


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--preload',type=Path)
    group.add_argument('--resume',type=Path)
    p.add_argument('--reference-checkpoint',type=Path,help='source production mesh/config, not old preload geometry')
    p.add_argument('--mode',choices=('closed','rcr'))
    p.add_argument('--duration-ms',type=float,default=2.)
    p.add_argument('--probe-pc-mmhg',type=float,help='explicit circuit condition probe; not a physiological beat')
    p.add_argument('--level',type=int,default=3)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    if not torch.cuda.is_available() or a.duration_ms <= 0: p.error('actual CUDA and positive duration required')
    inputs = [v for v in (a.preload,a.resume,a.reference_checkpoint) if v]
    shas = {str(v):hashlib.sha256(v.read_bytes()).hexdigest() for v in inputs}
    if a.resume:
        d,state,progress = restart.load(a.resume)
        if a.reference_checkpoint: p.error('resume restores production configuration')
        if a.mode and a.mode != state.mode:
            if a.mode != 'rcr' or state.mode != 'closed': p.error('only closed to RCR connection is supported here')
            e = d._fsi_stepper(d.maximum_ticks)
            from afsi_torch.ep.mac_rcr_coupled import outlet_rows
            rows = outlet_rows(e,state.cap_geometry)
            bp = torch.full_like(rows.area,state.cap_pressure.mean())
            state = replace(state,mode='rcr',cap_pressure=bp)
            state = replace(state,ep=replace(state.ep,fsi=replace(state.fsi,force=d.force_for_state(state))))
    else:
        if not a.reference_checkpoint: p.error('reference checkpoint supplies original production tolerances')
        old,source,_ = checkpoint.load(a.reference_checkpoint)
        d = BalancedAFSEIStepper(old.mesh,replace(old.config,prescribed_endocardial_pressure_mmhg=0.),level=a.level)
        del old,source
        state = d.initialize_preload(a.preload,mode=a.mode or 'closed')
        progress = dict(preload_source=str(a.preload),preload_sha256=shas[str(a.preload)],ep_steps=0,ode_substeps=0)
    if a.probe_pc_mmhg is not None:
        if state.mode != 'rcr': p.error('circuit probe requires RCR')
        state = replace(state,arterial_pc_mmhg=a.probe_pc_mmhg)
        progress['circuit_condition_probe'] = a.probe_pc_mmhg
    a.output.mkdir(parents=True,exist_ok=False)
    target = state.tick+ticks(a.duration_ms/1000,d.protocol.dt_ep_s)
    d.profile = True; rows = []; initial_time = state.fsi.time
    initial_rejected = state.rejected_attempts; started = perf_counter(); failure = None
    try:
        while state.tick < target:
            torch.cuda.synchronize(); begin = perf_counter()
            state,info = d.step(state,target_tick=target)
            torch.cuda.synchronize(); seconds = perf_counter()-begin
            ta = d.buffer.tension(state.fsi.x)
            rows.append(dict(step=state.fsi.step,time_s=state.fsi.time,wall_seconds=seconds,
                balanced=info['balanced'],nonlinear=info['nonlinear'],timings=info['timings'],
                ep=info['ep'],max_velocity_cm_s=info['max_velocity_cm_s'],
                active_tension_max_cgs=ta.max().item(),rejected_attempts=info['rejected_attempts']))
            progress['ep_steps'] += info['ep']['steps']; progress['ode_substeps'] += info['ep']['ode_substeps']
            if len(rows)%50 == 0: restart.save(a.output/'checkpoint.npz',d,state,progress)
            if len(rows)%10 == 0 or state.tick == target:
                r = info['balanced']
                print(f"Balanced t={state.fsi.time*1000:.3f}ms Ta={ta.max().item():.5g} "
                      f"P={r['pressure_core']['mean_mmhg']:.5g} Q={r['measured_relative_q_ml_s']:.5g} "
                      f"mass={r['mass_error_ml_s']:.3g} open={r['outlet_open']} step={seconds:.3f}s",flush=True)
    except BaseException as exc:
        failure = repr(exc)
        raise
    finally:
        restart.save(a.output/'checkpoint.npz',d,state,progress)
        unchanged = all(shas[str(v)] == hashlib.sha256(v.read_bytes()).hexdigest() for v in inputs)
        limits = dict(mass_error_ml_s=1e-5,closed_flux_norm_ml_s=1e-5,boundary_work_relative_error=1e-7,
                      transfer_power_relative_error=1e-7)
        if state.mode=='rcr':
            limits.update(rcr_balance_ml_s=1e-8,rcr_energy_relative_error=1e-8)
        failed = [k for k,limit in limits.items() if any(abs(r['balanced'][k]) > limit for r in rows)]
        report = dict(status='completed' if state.tick==target and failure is None else 'incomplete',failure=failure,
            scope=__doc__,mode=state.mode,accepted_steps=len(rows),reached_time_s=state.fsi.time,
            duration_s=state.fsi.time-initial_time,elapsed_seconds=perf_counter()-started,
            rejected_attempts=state.rejected_attempts-initial_rejected,rows=rows,targets=limits,
            validation_targets_passed=bool(rows) and not failed and unchanged and state.tick==target and failure is None,
            failed_targets=failed,input_sha256=shas,input_unchanged=unchanged,
            circuit_condition_probe=progress.get('circuit_condition_probe'),
            gpu=torch.cuda.get_device_name(),fluid_shape=d.config.fsi.fluid.shape,
            mean_seconds_after_first=sum(r['wall_seconds'] for r in rows[1:])/len(rows[1:]) if len(rows)>1 else None,
            full_ep_fsi_step_timing=True,io_in_step_timing=False,
            volume_transfer='integrated-p1-gradient-v1',
            ep_advanced=bool(rows),windkessel_actual_flow_connected=state.mode=='rcr' and bool(rows),
            full_cardiac_phase_controller=False,pointwise_wall_impermeability_established=False)
        (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2),flush=True)
    if not report['validation_targets_passed']: raise SystemExit('dynamic validation targets failed')


if __name__=='__main__': main()
