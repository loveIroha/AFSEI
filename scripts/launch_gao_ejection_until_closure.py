"""One-off continuation to first accepted AV closure, then CPU analysis."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT=Path('/mnt/large2/gjh/AFSEI')
CONFIG=Path('configs/gao2017/connection_pair_20261026/ejection_until_closure_20261028.json')
REQUEST=json.loads((ROOT/CONFIG).read_text())
CASE=ROOT/REQUEST['output']
JOB=CASE.parent/(CASE.name+'_job.json')
LOG=CASE.parent/(CASE.name+'.log')


def publish(path,value):
    tmp=path.with_suffix(path.suffix+'.working')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def inspect():
    source=ROOT/REQUEST['resume']
    with np.load(source,allow_pickle=False) as data:
        meta=json.loads(str(data['metadata']));nodes=len(data['X'])
    previous=json.loads((source.parent/'report.json').read_text())
    if previous['status']!='passed' or not previous.get('advanced_restart_roundtrip'):
        raise ValueError('input interval has not passed its accepted-endpoint restart check')
    if abs(meta['time']*1000-577.9)>1e-7 or previous['end_tick']!=meta['tick']:
        raise ValueError('input checkpoint time/tick differs from the validated endpoint')
    if REQUEST['stop_after_event']!={'name':'OUTLET_CLOSURE_TA_SNAPSHOT','advance_ms':0.} or REQUEST['duration_ms']!=400.:
        raise ValueError('launcher restricted to first closure with the reviewed 400 ms upper bound')
    if meta['cardiac']['phase']!='EJECTION':raise ValueError('continuation must start in accepted ejection')
    protocol=meta['config']['protocol'] if 'protocol' in meta['config'] else None
    # Record checkpoint metadata, not assumptions from old default protocols.
    summary=dict(source=str(source),source_time_ms=meta['time']*1000,source_tick=meta['tick'],
        cardiac=meta['cardiac'],windkessel=meta['balanced']['windkessel'],
        configuration=str(CONFIG),request=REQUEST,config_keys=list(meta['config']),
        cardiac_execution=meta.get('cardiac_execution'),protocol=protocol,
        coarse_nodes=nodes,fluid=meta['config']['fsi']['fluid'],
        fsi_time=meta['config']['fsi']['time'])
    return summary


def work():
    status=CASE.parent/(CASE.name+'_completion.json')
    simulation=subprocess.Popen([sys.executable,'-u','scripts/run_afsei_cut_cycle.py',str(CONFIG)],cwd=ROOT)
    publish(CASE.parent/(CASE.name+'_runtime.json'),dict(worker_pid=os.getpid(),simulation_pid=simulation.pid,
        started_at_utc=datetime.now(timezone.utc).isoformat()))
    returncode=simulation.wait()
    result=dict(simulation_returncode=returncode,configuration=str(CONFIG),output=str(CASE),
        finished_at_utc=datetime.now(timezone.utc).isoformat())
    report_path=CASE/'report.json'
    if report_path.exists():
        final=json.loads(report_path.read_text())
        result.update(simulation_status=final['status'],accepted_steps=final['accepted_steps'],
            event_window=final.get('event_window'),failure=final.get('failure'))
    if (CASE/'steps.jsonl').exists() and (CASE/'steps.jsonl').stat().st_size:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',MPLCONFIGDIR='/tmp/afsei-mpl-gao-ejection-closure')
        with (CASE/'postprocess.log').open('xb') as log:
            analysis=subprocess.run([sys.executable,'scripts/analyze_gao_ejection_interval.py',str(CASE)],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        result['analysis_returncode']=analysis.returncode
        if analysis.returncode==0:
            audit=json.loads((CASE/'interval_analysis.json').read_text())
            result.update(interval_and_diagnostics_passed=audit['interval_and_diagnostics_passed'],
                first_outlet_closure=audit['first_outlet_closure'])
    result['status']=('completed' if result.get('interval_and_diagnostics_passed') is True else 'needs_review')
    publish(status,result);print(json.dumps(result,indent=2),flush=True)
    return returncode


def launch(gpu):
    source=inspect()
    if any(p.exists() for p in (CASE,JOB,LOG)):
        raise SystemExit('Existing experiment found; inspect it instead of launching twice.')
    active=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=gpu_uuid',
        '--format=csv,noheader'],text=True).strip()
    if active:raise SystemExit('Requested GPU occupied; existing jobs are left unchanged.')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',
        CUDA_HOME='/mnt/large2/gjh/conda-envs/afsei',CONDA_PREFIX='/mnt/large2/gjh/conda-envs/afsei',
        TORCH_CUDA_ARCH_LIST='8.9',MPLCONFIGDIR='/tmp/afsei-mpl-gao-ejection-closure')
    with JOB.open('x') as record:
        with LOG.open('xb') as stream:
            process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker'],
                cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        data=dict(pid=process.pid,gpu=gpu,configuration=str(CONFIG),output=str(CASE),log=str(LOG),
            started_at_utc=datetime.now(timezone.utc).isoformat(),
            stop='first accepted outlet closure; 400 ms duration upper bound',
            automatic_cpu_analysis=True,source_time_ms=source['source_time_ms'],
            inherited_cardio=source['cardiac'],windkessel=source['windkessel'])
        record.write(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data,indent=2))


def status():
    result=dict(output=str(CASE),job_exists=JOB.exists())
    if JOB.exists():result.update(json.loads(JOB.read_text()))
    runtime=CASE.parent/(CASE.name+'_runtime.json')
    if runtime.exists():result['runtime']=json.loads(runtime.read_text())
    completion=CASE.parent/(CASE.name+'_completion.json')
    if completion.exists():result['completion']=json.loads(completion.read_text())
    steps=CASE/'steps.jsonl'
    if steps.exists() and steps.stat().st_size:
        with steps.open('rb') as f:
            size=f.seek(0,2);width=min(size,1048576);f.seek(size-width)
            lines=f.read().splitlines()
        for line in reversed(lines):
            try:row=json.loads(line)
            except json.JSONDecodeError:continue
            b=row['balanced']
            result['latest_accepted']=dict(time_ms=row['time_ms'],phase=row['phase'],
                volume_ml=b['volume_ml'],pressure_core_mmhg=b['pressure_core']['mean_mmhg'],
                pressure_cut_mmhg=b['boundary_pressure_mmhg'],
                qout_ml_s=b.get('physical_outlet_flow_ml_s',0.),
                accepted_dt_s=row['accepted_dt_s'],step_seconds=row['step_seconds'],
                mass_error_ml_s=b['mass_error_ml_s'],events=row['cardiac']['events'],
                rejected_attempts_in_latest_step=len(row.get('rejected_attempts',[])))
            break
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,default=0);parser.add_argument('--inspect',action='store_true')
    parser.add_argument('--worker',action='store_true');parser.add_argument('--status',action='store_true')
    args=parser.parse_args()
    # Status is a read-only path; it never inspects/restarts an old checkpoint.
    if args.worker:raise SystemExit(work())
    if args.status:status()
    elif args.inspect:print(json.dumps(inspect(),indent=2))
    else:launch(args.gpu)
