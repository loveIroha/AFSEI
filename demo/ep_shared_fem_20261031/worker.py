from pathlib import Path
import sys,json,os,subprocess,hashlib,time
ROOT=Path('/mnt/large2/gjh/AFSEI');OUT=ROOT/'results/ep_shared_fem_20261031'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',CUDA_HOME='/mnt/large2/gjh/conda-envs/afsei',TORCH_CUDA_ARCH_LIST='8.9',MAX_JOBS='4',MPLCONFIGDIR=str(ROOT/'.cache/matplotlib'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
runner=ROOT/'demo/ep_afsei_nodal_opencarp_matched/run_afsei_matched.py'
hashes={n:hashlib.sha256((ROOT/'src/afsi_torch/ep'/n).read_bytes()).hexdigest() for n in ('graph_pcg.py','gpu_kernels.py','monodomain.py','jacobi.py')}
(OUT/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
try:
    for job in json.loads((OUT/'jobs.json').read_text()):
        path=OUT/job['name'];assert not path.exists()
        with (OUT/(job['name']+'.log')).open('w') as log:
            child=subprocess.Popen([sys.executable,'-u',str(runner),'--config',job['config']],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            (OUT/'progress.json').write_text(json.dumps(dict(status='running',job=job['name'],pid=child.pid,physical_gpu=0),indent=2))
            code=child.wait()
        if code:raise RuntimeError(f'{job["name"]} exit {code}')
        r=json.loads((path/'report.json').read_text());assert r['max_cell_guard']==0 and r['max_true_residual_ratio']<=1 and r['activated_fraction']==1
    (OUT/'progress.json').write_text(json.dumps(dict(status='simulations_completed'),indent=2))
except Exception as e:
    (OUT/'progress.json').write_text(json.dumps(dict(status='failed',error=str(e)),indent=2));raise
