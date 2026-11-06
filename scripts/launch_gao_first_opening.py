"""Start the bounded first-AV-opening experiment on an idle GPU."""
import argparse,json,os,subprocess,sys
from pathlib import Path

root=Path('/mnt/large2/gjh/AFSEI')
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('configuration',nargs='?',default='configs/gao2017/connection_pair_20261026/first_opening_20261026.json')
parser.add_argument('--gpu',type=int,default=0)
args=parser.parse_args();config=args.configuration
request=json.loads((root/config).read_text());output=root/request['output']
if 'stop_after_event' not in request or request.get('duration_ms',0)<=0:
    raise SystemExit('launcher requires a bounded accepted-event experiment')
log=output.parent/(output.name+'.log')
job=output.parent/(output.name+'_job.json')
if output.exists() or log.exists() or job.exists():raise SystemExit('experiment already exists; inspect it instead of relaunching')
gpus=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).strip()
if gpus:raise SystemExit('a GPU compute process is already present; inspect before launch')
env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),OMP_NUM_THREADS='4',
    CUDA_HOME='/mnt/large2/gjh/conda-envs/afsei',CONDA_PREFIX='/mnt/large2/gjh/conda-envs/afsei',
    TORCH_CUDA_ARCH_LIST='8.9',MPLCONFIGDIR='/tmp/afsei-mpl-gao-opening')
with log.open('xb') as stream:
    process=subprocess.Popen([sys.executable,'-u','scripts/run_afsei_cut_cycle.py',config],
        cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=process.pid,gpu=args.gpu,configuration=config,output=str(output),log=str(log),
    purpose='bounded real accepted-event experiment; no full cycles',event_window=request['stop_after_event'],duration_ms=request['duration_ms'])
job.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
