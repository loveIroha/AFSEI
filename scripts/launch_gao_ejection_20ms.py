"""Start the configured 20 ms continuation exactly once on an idle GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


root=Path('/mnt/large2/gjh/AFSEI')
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--gpu',type=int,default=0)
args=parser.parse_args()
config='configs/gao2017/connection_pair_20261026/ejection_20ms_20261028.json'
request=json.loads((root/config).read_text())
if request['duration_ms']!=20. or request.get('stop_after_event'):
    raise SystemExit('launcher is limited to the existing 20 ms interval')
output=root/request['output'];log=output.parent/(output.name+'.log');job=output.parent/(output.name+'_job.json')
if any(p.exists() for p in (output,log,job)):
    raise SystemExit('experiment already exists; inspect instead of relaunching')
if not (root/request['resume']).is_file():raise SystemExit('validated input checkpoint missing')
active=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),
    '--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).strip()
if active:raise SystemExit('GPU is occupied; leave existing jobs unchanged')
env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),OMP_NUM_THREADS='4',
    CUDA_HOME='/mnt/large2/gjh/conda-envs/afsei',CONDA_PREFIX='/mnt/large2/gjh/conda-envs/afsei',
    TORCH_CUDA_ARCH_LIST='8.9',MPLCONFIGDIR='/tmp/afsei-mpl-gao-ejection')
with log.open('xb') as stream:
    process=subprocess.Popen([sys.executable,'-u','scripts/run_afsei_cut_cycle.py',config],cwd=root,
        env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=process.pid,gpu=args.gpu,configuration=config,output=str(output),log=str(log),
    purpose='557.9 to 577.9 ms: actual ejection and sparse read-only momentum budgets',duration_ms=20.)
job.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
