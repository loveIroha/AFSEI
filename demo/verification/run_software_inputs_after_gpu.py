"""Reuse GPU0 after its grid case; prepare shared drive and matched pilot."""
from pathlib import Path
import json,os,subprocess,sys,time
from run_accuracy_queue import gpu_idle
repo=Path('/mnt/large2/gjh/AFSEI');root=repo/'results/accuracy_performance_20261025'
status=root/'software_gpu_inputs_status.json'
def save(value):
    p=status.with_suffix('.tmp');p.write_text(json.dumps(value,indent=2));p.replace(status)
started=time.monotonic()
try:
    while True:
        qp=root/'queue_gpu0.json';q=json.loads(qp.read_text()) if qp.exists() else {}
        if q.get('stage') in ('completed','failed') and gpu_idle(0):break
        save(dict(stage='waiting-for-GPU0-grid-case',grid_stage=q.get('stage'),formal_performance_established=False))
        if time.monotonic()-started>24*3600:raise TimeoutError('GPU preparation queue timed out')
        time.sleep(30)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',AFSEI_GPU='0',CUDA_HOME=sys.prefix,
        CONDA_PREFIX=sys.prefix,TORCH_CUDA_ARCH_LIST='8.9',MAX_JOBS='2',OMP_NUM_THREADS='2',
        MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PATH=str(Path(sys.prefix)/'bin')+os.pathsep+env.get('PATH',''),
        TORCH_EXTENSIONS_DIR=str(repo/'.cache/torch_extensions'),TRITON_CACHE_DIR=str(repo/'.cache/triton'),
        TORCHINDUCTOR_CACHE_DIR=str(repo/'.cache/torchinductor'),MPLCONFIGDIR=str(root/'mpl-cache'))
    cfg='demo/verification/configs/accuracy_performance_20261025/solid3_mac64.json'
    jobs=[('electrical-history',['demo/verification/export_ep_drive_tape.py','--config',cfg,'--output',str(root/'ep_drive_tape')]),
          ('AFSEI-constant-active-pilot',['demo/verification/run_constant_active_pilot.py','--config',cfg,
            '--input',str(root/'ibamr_inputs/constant_60ms_active.npz'),'--output',str(root/'afsei_constant_active_pilot')])]
    for label,args in jobs:
        save(dict(stage=label,formal_performance_established=False))
        with (root/(label+'.log')).open('w') as log:
            subprocess.run([sys.executable,'-u',*args],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    save(dict(stage='completed',EP_history_checked=True,AFSEI_pilot_completed=True,formal_performance_established=False))
except Exception as e:
    save(dict(stage='failed',error=repr(e),settings_relaxed=False,formal_performance_established=False));raise
