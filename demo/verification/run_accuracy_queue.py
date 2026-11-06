"""Bounded experiment queue; only designated GPUs and no automatic retries."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time

def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def gpu_idle(index):
    uuid=subprocess.check_output(['nvidia-smi','-i',str(index),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    return not any(line.split(',')[0].strip()==uuid for line in rows.splitlines())

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',required=True,type=int)
    p.add_argument('--cases',required=True,nargs='+');p.add_argument('--root',required=True,type=Path)
    args=p.parse_args();args.root.mkdir(parents=True,exist_ok=True)
    status=args.root/f'queue_gpu{args.gpu}.json';records=[]
    for name in args.cases:
        out=args.root/name
        if out.exists() and any(out.iterdir()):
            raise ValueError(f'case already exists; inspect before resuming: {out}')
        waited=time.monotonic()
        while not gpu_idle(args.gpu):
            write(status,dict(stage='waiting-for-idle-GPU',gpu=args.gpu,next_case=name,finished=records))
            if time.monotonic()-waited>24*3600:raise TimeoutError('GPU remained occupied for 24 hours')
            time.sleep(30)
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(args.gpu),AFSEI_GPU=str(args.gpu),
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MPLCONFIGDIR=str(args.root/'mpl-cache'),
            CUDA_HOME=sys.prefix,CONDA_PREFIX=sys.prefix,TORCH_CUDA_ARCH_LIST='8.9',MAX_JOBS='2',
            TORCH_EXTENSIONS_DIR=str(Path.cwd()/'.cache/torch_extensions'),
            TRITON_CACHE_DIR=str(Path.cwd()/'.cache/triton'),
            TORCHINDUCTOR_CACHE_DIR=str(Path.cwd()/'.cache/torchinductor'),
            PYTHONPATH=str(Path.cwd()/'src')+(os.pathsep+os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else ''),
            PATH=str(Path(sys.prefix)/'bin')+os.pathsep+os.environ.get('PATH',''))
        cfg=Path('demo/verification/configs/accuracy_performance_20261025')/(name+'.json')
        command=[sys.executable,'-u','demo/verification/run_accuracy_case.py','--config',str(cfg),'--output',str(out)]
        started=time.time()
        with (args.root/(name+'.log')).open('w') as log:
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            write(status,dict(stage='running',gpu=args.gpu,case=name,pid=process.pid,finished=records))
            code=process.wait()
        record=dict(case=name,returncode=code,wall_seconds=time.time()-started);records.append(record)
        if code:
            write(status,dict(stage='failed',gpu=args.gpu,finished=records,next_cases_not_started=True));return 1
    write(status,dict(stage='completed',gpu=args.gpu,finished=records));return 0

if __name__=='__main__':sys.exit(main())
