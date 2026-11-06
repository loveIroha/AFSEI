"""Run one standard JSON job with low-frequency external GPU observations.

The monitor never synchronizes a solver tensor or changes solver options.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def publish(path, data):
    tmp = path.with_suffix(path.suffix+'.working')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def query(command):
    r = subprocess.run(command, text=True, capture_output=True, timeout=10)
    return dict(returncode=r.returncode, output=r.stdout.strip(), error=r.stderr.strip())


def family(pid):
    rows = subprocess.run(['ps','-eo','pid=,ppid='],text=True,capture_output=True,check=True).stdout.splitlines()
    pairs = [tuple(map(int,row.split())) for row in rows]
    selected={pid}
    while True:
        expanded=selected|{child for child,parent in pairs if parent in selected}
        if expanded==selected:return sorted(selected)
        selected=expanded


def summarize(output):
    import numpy as np
    report=json.loads((output/'report.json').read_text())
    stage_seconds={}; rejections=0; accepted_seconds=[]; phases={}
    with (output/'steps.jsonl').open() as f:
        for line in f:
            r=json.loads(line)
            accepted_seconds.append(r['step_seconds'])
            rejections+=len(r['retries'])
            for key,value in r['timings'].items():
                if key.endswith('_seconds') and isinstance(value,(int,float)):
                    stage_seconds[key]=stage_seconds.get(key,0.)+value
            phase=r['cardiac']['solved_phase']
            entry=phases.setdefault(phase,dict(accepted_steps=0,wall_seconds=0.))
            entry['accepted_steps']+=1;entry['wall_seconds']+=r['step_seconds']
    data=list(csv.DictReader((output/'pv.csv').open()))
    configuration=json.loads((output/'configuration.json').read_text())
    starts=configuration['config']['protocol']['pulse_starts_s']
    end=report['end_time_s']; beat_stats=[]; traces=[]
    for index,start in enumerate(starts):
        stop=starts[index+1] if index+1<len(starts) else end
        rows=[r for r in data if start*1000-1e-7<=float(r['time_ms'])<=stop*1000+1e-7]
        if not rows:continue
        t=np.array([float(r['time_ms']) for r in rows])
        v=np.array([float(r['volume_ml']) for r in rows])
        p=np.array([float(r['pressure_core_mmhg']) for r in rows])
        edv=float(v[0]);esv=float(v.min())
        beat_stats.append(dict(beat=index+1,start_ms=start*1000,requested_end_ms=stop*1000,
            reached_ms=float(t[-1]),edv_ml=edv,esv_ml=esv,stroke_volume_ml=edv-esv,
            ef_percent=100*(edv-esv)/edv,peak_core_pressure_mmhg=float(p.max()),end_volume_ml=float(v[-1])))
        traces.append((index+1,t,v,p))
    result=dict(solver_status=report['status'],solver_elapsed_seconds=report['elapsed_seconds'],
        full_cycle_driver_validation=report['full_cycle_validation_passed'],
        physiological_or_periodic_convergence_established=False,
        accepted_steps=len(accepted_seconds),rejected_attempts=rejections,
        accepted_step_stage_seconds=stage_seconds,phase_step_wall_seconds=phases,
        mean_step_seconds=float(np.mean(accepted_seconds)) if accepted_seconds else None,
        median_step_seconds=float(np.median(accepted_seconds)) if accepted_seconds else None,
        p95_step_seconds=float(np.quantile(accepted_seconds,.95)) if accepted_seconds else None,
        beats=beat_stats,pressure_definition='cavity-core mean fluid pressure',
        ep_mesh=configuration['ep_mesh'],cycle_timing=configuration['cardiac_execution'])
    publish(output/'performance_results.json',result)
    if traces:
        os.environ.setdefault('MPLCONFIGDIR','/tmp/afsei_mpl_20261023')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(15,4.8),constrained_layout=True)
        for (index,t,v,p),stats in zip(traces,beat_stats):
            label=f"Beat {index}: EF {stats['ef_percent']:.1f}%"
            axes[0].plot(v,p,label=label)
            axes[1].plot(t,v,label=f'Beat {index}')
            axes[2].plot(t,p,label=f'Beat {index}')
        for ax,xlabel,ylabel in zip(axes,['Cavity volume (mL)','Time (ms)','Time (ms)'],
                ['Core pressure (mmHg)','Cavity volume (mL)','Core pressure (mmHg)']):
            ax.set(xlabel=xlabel,ylabel=ylabel);ax.grid(alpha=.25);ax.legend()
        fig.suptitle('Coarse EP, two 900 ms cycles; recorded trajectories without artificial closure')
        for suffix in ('png','pdf'):fig.savefig(output/f'pv_and_histories.{suffix}',dpi=180)
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('configuration',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    config=json.loads(args.configuration.read_text());gpu=config['physical_gpu']
    output=root/config['output']
    if output.exists():raise FileExistsError('output already exists: '+str(output))
    logs=root/'results/logs';logs.mkdir(exist_ok=True)
    monitor=logs/(output.name+'.gpu.jsonl')
    summary=logs/(output.name+'.launcher.json')
    command=[sys.executable,'-u',str(root/'scripts/run_afsei_job.py'),str(args.configuration.resolve())]
    started=time.perf_counter();child=None
    with monitor.open('x',buffering=1) as stream:
        child=subprocess.Popen(command,cwd=root)
        while True:
            entry=dict(utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started,
                solver_returncode=child.poll())
            try:
                entry['own_process_family']=family(child.pid)
                entry['gpu']=query(['nvidia-smi',f'--id={gpu}',
                    '--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm',
                    '--format=csv,noheader,nounits'])
                entry['processes']=query(['nvidia-smi',f'--id={gpu}',
                    '--query-compute-apps=pid,used_gpu_memory,gpu_uuid','--format=csv,noheader,nounits'])
            except Exception as exc:entry['monitor_error']=repr(exc)
            stream.write(json.dumps(entry)+'\n')
            if child.poll() is not None:break
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:pass
    elapsed=time.perf_counter()-started
    record=dict(returncode=child.returncode,total_launcher_wall_seconds=elapsed,
        monitor=str(monitor),gpu=gpu,command=command,output=str(output))
    publish(summary,record)
    if (output/'report.json').exists():
        try:summarize(output)
        except Exception as exc:
            record['postprocess_error']=repr(exc);publish(summary,record)
            print('postprocess error:',repr(exc),flush=True)
    print(json.dumps(record,indent=2),flush=True)
    raise SystemExit(child.returncode)


if __name__=='__main__':main()
