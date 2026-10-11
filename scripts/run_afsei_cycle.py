"""Run complete cardiac cycles with source EP/Land and implicit Cartesian MAC FSI."""
import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter
import torch
from afsi_torch.ep import checkpoint
from afsi_torch.ep import mac_balanced_checkpoint as restart
from afsi_torch.ep.mac_cycle import CardiacStepper,PHASES
from afsi_torch.ep.protocol import ticks


def publish_json(path,data):
    temporary=path.with_suffix(path.suffix+'.working')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--preload',type=Path); g.add_argument('--resume',type=Path)
    p.add_argument('--reference-checkpoint',type=Path)
    p.add_argument('--cycles',type=int,help='fresh run only; default 2, period 800 ms')
    p.add_argument('--duration-ms',type=float,help='optional short interval, not complete-cycle validation')
    p.add_argument('--checkpoint-every',type=int,default=100)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not torch.cuda.is_available() or a.checkpoint_every<1: p.error('CUDA and positive checkpoint interval required')
    if a.resume:
        if a.cycles is not None or a.reference_checkpoint: p.error('resume restores cycle count and configuration')
        d,state,progress=restart.load(a.resume,cardiac=True)
    else:
        if not a.reference_checkpoint: p.error('reference checkpoint is required with preload')
        cycles=a.cycles if a.cycles is not None else 2
        if cycles<1: p.error('positive cycle count required')
        old,source,_=checkpoint.load(a.reference_checkpoint)
        cfg=replace(old.config,prescribed_endocardial_pressure_mmhg=0.,
            protocol=replace(old.protocol,pulse_starts_s=tuple(.5+.8*i for i in range(cycles))))
        d=CardiacStepper(old.mesh,cfg,cycles=cycles)
        del old,source
        state=d.initialize_preload(a.preload)
        progress=dict(preload_source=str(a.preload),ep_steps=0,ode_substeps=0,
                      phase_seen={},max_mass_error_ml_s=0.,max_cumulative_mass_error_ml=0.)
    if state.tick>=d.end_tick: p.error('checkpoint has already reached the requested cycle end')
    target=d.end_tick
    if a.duration_ms is not None:
        if a.duration_ms<=0: p.error('positive duration required')
        target=min(target,state.tick+ticks(a.duration_ms/1000,d.protocol.dt_ep_s))
    a.output.mkdir(parents=True,exist_ok=False)
    d.profile=True; start=perf_counter(); step_seconds=0.; count=0; failure=None
    # Persist the initial accepted state before the first expensive step.
    restart.save(a.output/'checkpoint.npz',d,state,progress)
    columns=('time_ms','beat','phase','volume_ml','pressure_core_mmhg','pc_mmhg',
             'q_in_ml_s','q_out_ml_s','tension_max_cgs','mass_error_ml_s','step_seconds')
    with (a.output/'steps.jsonl').open('w',buffering=1) as stream,(a.output/'pv.csv').open('w',newline='',buffering=1) as pv:
        writer=csv.DictWriter(pv,fieldnames=columns); writer.writeheader()
        try:
            while state.tick<target:
                torch.cuda.synchronize(); begin=perf_counter()
                state,info=d.step(state,target_tick=target)
                torch.cuda.synchronize(); seconds=perf_counter()-begin
                step_seconds+=seconds; count+=1
                r=info['balanced']; c=info['cardiac']; ta=d.buffer.tension(state.fsi.x).max().item()
                progress['ep_steps']+=info['ep']['steps']; progress['ode_substeps']+=info['ep']['ode_substeps']
                seen=progress['phase_seen'].setdefault(str(state.beat),[])
                if state.phase not in seen: seen.append(state.phase)
                progress['max_mass_error_ml_s']=max(progress['max_mass_error_ml_s'],abs(r['mass_error_ml_s']))
                progress['max_cumulative_mass_error_ml']=max(progress['max_cumulative_mass_error_ml'],abs(c['cumulative_mass_error_ml']))
                row=dict(time_ms=state.fsi.time*1000,beat=state.beat,phase=state.phase,volume_ml=r['volume_ml'],
                    pressure_core_mmhg=r['pressure_core']['mean_mmhg'],pc_mmhg=state.arterial_pc_mmhg,
                    q_in_ml_s=r['inlet_flow_ml_s'],q_out_ml_s=r['measured_relative_q_ml_s'],
                    tension_max_cgs=ta,mass_error_ml_s=r['mass_error_ml_s'],step_seconds=seconds)
                writer.writerow(row)
                stream.write(json.dumps(dict(**row,cardiac=c,nonlinear=info['nonlinear'],
                    balanced=r,ep=info['ep'],timings=info['timings'],retries=info['rejected_attempts']),allow_nan=False)+'\n')
                if count%a.checkpoint_every==0 or c['events']:
                    restart.save(a.output/'checkpoint.npz',d,state,progress)
                if count%10==0 or c['events'] or state.tick==target:
                    publish_json(a.output/'progress.json',dict(status='running',**row,progress=progress,
                        reached_tick=state.tick,target_tick=target,mean_step_seconds=step_seconds/count))
                    print(f"AFSEI beat={state.beat} {state.phase} t={row['time_ms']:.3f}ms "
                          f"V={r['volume_ml']:.6f} P={row['pressure_core_mmhg']:.5f} "
                          f"Qin={r['inlet_flow_ml_s']:.5g} Qout={r['measured_relative_q_ml_s']:.5g} "
                          f"step={seconds:.3f}s events={c['events']}",flush=True)
        except BaseException as exc:
            failure=repr(exc)
            raise
        finally:
            restart.save(a.output/'checkpoint.npz',d,state,progress)
            all_phases=all(set(PHASES)<=set(progress['phase_seen'].get(str(i),[])) for i in range(1,d.cycles+1))
            completed=state.tick==target and failure is None
            full_pass=completed and state.tick==d.end_tick and all_phases and state.phase=='FILLING'
            report=dict(status='completed' if completed else 'incomplete',failure=failure,
                accepted_steps_this_run=count,reached_time_s=state.fsi.time,end_time_s=d.end_tick*d.protocol.dt_ep_s,
                elapsed_seconds=perf_counter()-start,mean_step_seconds=step_seconds/count if count else None,
                cycles=d.cycles,phase=state.phase,beat=state.beat,progress=progress,
                full_cardiac_phase_controller=True,full_cycle_validation_passed=full_pass,
                short_interval_requested=a.duration_ms is not None,
                pointwise_wall_impermeability_established=False,
                fluid_grid='original Cartesian MAC',checkpoint=str(a.output/'checkpoint.npz'))
            publish_json(a.output/'report.json',report)
            publish_json(a.output/'progress.json',report)
    print(json.dumps(report,indent=2),flush=True)
    if a.duration_ms is None and not full_pass: raise SystemExit('time reached but complete cardiac phase validation failed')


if __name__=='__main__': main()
