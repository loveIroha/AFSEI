"""Short reproducible GPU benchmark from a fixed accepted MAC checkpoint."""
import argparse
import cProfile
import json
from pathlib import Path
import pstats
from time import perf_counter
import torch
from afsi_torch.ep import mac_balanced_checkpoint as restart


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cardiac',action='store_true')
    p.add_argument('--steps',type=int,default=5)
    p.add_argument('--warmup',type=int,default=1)
    p.add_argument('--profile',action='store_true')
    p.add_argument('--mass-check-every',type=int,choices=(4,8,16))
    p.add_argument('--tangent-chunk-size',type=int)
    p.add_argument('--helmholtz-initial-sweeps',type=int)
    p.add_argument('--reuse-step-tangent',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--step-ticks',type=int,choices=(1,2),
                   help='Bound each step to one or two EP ticks (0.05 or 0.1 ms).')
    a=p.parse_args()
    if a.steps<1 or a.warmup<0: p.error('invalid step counts')
    a.output.mkdir(parents=True,exist_ok=False)
    d,state,progress=restart.load(a.input,cardiac=a.cardiac); d.profile=True
    if a.reuse_step_tangent is not None: d.reuse_step_tangent=a.reuse_step_tangent
    if a.tangent_chunk_size is not None:
        if a.tangent_chunk_size<1: p.error('positive tangent chunk size required')
        d.balanced_tangent_chunk_size=a.tangent_chunk_size; d._steppers.clear()
    if a.helmholtz_initial_sweeps is not None:
        d.helmholtz_initial_sweeps=a.helmholtz_initial_sweeps; d._steppers.clear()
    if a.mass_check_every is not None:
        d.mass_graph_check_every=a.mass_check_every
        d._steppers.clear()
    rows=[]
    for k in range(a.warmup+a.steps):
        torch.cuda.synchronize(); start=perf_counter()
        target=None if a.step_ticks is None else state.tick+a.step_ticks
        state,info=d.step(state,target_tick=target)
        torch.cuda.synchronize(); seconds=perf_counter()-start
        row=dict(warmup=k<a.warmup,time_s=state.fsi.time,seconds=seconds,
                 nonlinear=info['nonlinear'],timings=info['timings'],balanced=info['balanced'],
                 courant=info['courant'],max_velocity_cm_s=info['max_velocity_cm_s'],
                 retries=info['rejected_attempts'])
        rows.append(row)
        for key,source in (('ep_steps','steps'),('ode_substeps','ode_substeps')):
            progress[key]=progress.get(key,0)+info['ep'][source]
        if a.cardiac:
            seen=progress.setdefault('phase_seen',{}).setdefault(str(state.beat),[])
            if state.phase not in seen: seen.append(state.phase)
            progress['max_mass_error_ml_s']=max(progress.get('max_mass_error_ml_s',0.),abs(info['balanced']['mass_error_ml_s']))
            progress['max_cumulative_mass_error_ml']=max(progress.get('max_cumulative_mass_error_ml',0.),abs(info['cardiac']['cumulative_mass_error_ml']))
        print(json.dumps({k:v for k,v in row.items() if k not in ('nonlinear','balanced','timings')}) ,flush=True)
    restart.save(a.output/'checkpoint.npz',d,state,progress)
    measured=rows[a.warmup:]
    report=dict(input=str(a.input),steps=a.steps,warmup=a.warmup,rows=rows,
        mean_seconds=sum(r['seconds'] for r in measured)/len(measured),
        gpu=torch.cuda.get_device_name(),fluid_shape=d.config.fsi.fluid.shape,
        spacing_cm=d._fsi_stepper(d.maximum_ticks).grid.spacing,
        dt_s=d.config.fsi.time.dt if a.step_ticks is None else a.step_ticks*d.protocol.dt_ep_s,
        mass_graph_check_every=d.mass_graph_check_every,
        tangent_chunk_size=d.balanced_tangent_chunk_size,
        helmholtz_initial_sweeps=d.helmholtz_initial_sweeps,reuse_step_tangent=d.reuse_step_tangent,
        compact_cap_operators=d.compact_cap_operators,reached_time_s=state.fsi.time)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    if a.profile:
        profiler=cProfile.Profile(); profiler.enable()
        target=None if a.step_ticks is None else state.tick+a.step_ticks
        d.step(state,target_tick=target); torch.cuda.synchronize(); profiler.disable()
        with (a.output/'profile.txt').open('w') as stream:
            pstats.Stats(profiler,stream=stream).strip_dirs().sort_stats('cumtime').print_stats(55)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2),flush=True)


if __name__=='__main__': main()
