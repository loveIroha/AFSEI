"""Generate the corrected initial history, then run two full corrected cycles."""
import argparse
from datetime import datetime,timezone
import json,os,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PRELOAD=Path('configs/gao2017/source_clock_20261029/preload.json')
CONFIG=Path('configs/gao2017/source_clock_20261029/two_cycles.json')
PREFIX=ROOT/'results/logs/afsei_source_timing_two_cycles_20261029'


def publish(path,data):
    temporary=Path(str(path)+'.working')
    temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(path)


def work(gpu,preload=PRELOAD,configuration=CONFIG,prefix=PREFIX):
    started=datetime.now(timezone.utc).isoformat()
    result=dict(started_at_utc=started,physical_gpu=gpu,stages=[])
    with Path(str(prefix)+'.gpu.jsonl').open('x',buffering=1) as telemetry:
        for stage,script,case_configuration in [('preload','scripts/prepare_gao_cut_loading.py',preload),
                                           ('full-two-cycles','scripts/run_afsei_cut_cycle.py',configuration)]:
            child=subprocess.Popen([sys.executable,'-u',script,str(case_configuration)],cwd=ROOT,stdin=subprocess.DEVNULL)
            publish(Path(str(prefix)+'.runtime.json'),dict(supervisor_pid=os.getpid(),simulation_pid=child.pid,
                stage=stage,physical_gpu=gpu,started_at_utc=started))
            while True:
                try:
                    sample=subprocess.check_output(['nvidia-smi','-i',str(gpu),
                        '--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw',
                        '--format=csv,noheader,nounits'],text=True,timeout=10).strip()
                    telemetry.write(json.dumps(dict(time_utc=datetime.now(timezone.utc).isoformat(),stage=stage,sample=sample))+'\n')
                except Exception as exc:telemetry.write(json.dumps(dict(stage=stage,error=str(exc)))+'\n')
                try:code=child.wait(timeout=60);break
                except subprocess.TimeoutExpired:pass
            request=json.loads((ROOT/case_configuration).read_text());case=ROOT/request['output']
            report=json.loads((case/'report.json').read_text()) if (case/'report.json').exists() else {}
            result['stages'].append(dict(stage=stage,returncode=code,output=str(case),status=report.get('status'),
                accepted_steps=report.get('accepted_steps'),loading_samples=report.get('loading_samples'),
                elapsed_seconds=report.get('elapsed_seconds'),failure=report.get('failure')))
            if code or report.get('status')!='passed':
                result['status']='needs_review';break
            if stage=='preload':
                import numpy as np
                with np.load(case/'preload_history.npz',allow_pickle=False) as a:meta=json.loads(str(a['metadata']))
                entries=[r for r in meta['loading_records'] if r.get('physical_loading_endpoint')]
                if len(entries)!=100:raise ValueError('initial history is incomplete')
                for i,r in enumerate(entries,1):
                    if abs(r['pressure_mmhg']-8*min(1.,i/40))>1e-10 or abs(r['physical_loading_time_s']-.005*i)>1e-10:
                        raise ValueError('initial loading does not match 200ms ramp / 300ms hold')
                result['preload_schedule_verified']=True
            else:
                result.update(status='completed',full_cycle_completed=report.get('full_cycle_completed'),
                    final_schedule_ms=report.get('final_schedule_ms'),end_time_ms=report.get('end_time_ms'),
                    periodic_state_verified=report.get('periodic_state_verified'))
    result['finished_at_utc']=datetime.now(timezone.utc).isoformat()
    publish(Path(str(prefix)+'.completion.json'),result)
    print(json.dumps(result,indent=2),flush=True)


def launch(gpu,preload=PRELOAD,configuration=CONFIG,prefix=PREFIX):
    if gpu not in (0,1):raise ValueError('physical GPU must be 0 or 1')
    prefix=ROOT/prefix;prefix.parent.mkdir(parents=True,exist_ok=True)
    if any(Path(str(prefix)+suffix).exists() for suffix in ('.launch.json','.log','.runtime.json')):
        raise SystemExit('Existing task found; inspect it rather than launching twice')
    for c in (preload,configuration):
        if (ROOT/json.loads((ROOT/c).read_text())['output']).exists():raise SystemExit('Output directory already exists')
    active=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if active:raise SystemExit('Requested GPU occupied')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',
        CUDA_HOME='/mnt/large2/gjh/conda-envs/afsei',CONDA_PREFIX='/mnt/large2/gjh/conda-envs/afsei',
        TORCH_CUDA_ARCH_LIST='8.9',OMP_NUM_THREADS='4',MPLCONFIGDIR='/tmp/afsei-mpl',PYTHONUNBUFFERED='1')
    with Path(str(prefix)+'.launch.json').open('x') as record:
        with Path(str(prefix)+'.log').open('xb') as log:
            worker=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker','--gpu',str(gpu),
                '--preload-configuration',str(preload),'--configuration',str(configuration),'--prefix',str(prefix)],
                cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        data=dict(supervisor_pid=worker.pid,physical_gpu=gpu,configuration=str(ROOT/configuration),
            preload_configuration=str(ROOT/preload),log=str(prefix)+'.log',
            started_at_utc=datetime.now(timezone.utc).isoformat(),duration_override=None,
            scope='200ms initial ramp + 300ms hold, then two full source-wall cycles')
        record.write(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--preload-configuration',type=Path,default=PRELOAD)
    parser.add_argument('--configuration',type=Path,default=CONFIG)
    parser.add_argument('--prefix',type=Path,default=PREFIX)
    parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.worker:
        try:work(args.gpu,args.preload_configuration,args.configuration,ROOT/args.prefix)
        except Exception as exc:
            publish(Path(str(ROOT/args.prefix)+'.completion.json'),dict(status='needs_review',
                failure=type(exc).__name__+': '+str(exc),finished_at_utc=datetime.now(timezone.utc).isoformat()))
            raise
    else:launch(args.gpu,args.preload_configuration,args.configuration,args.prefix)
