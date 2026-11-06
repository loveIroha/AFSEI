from pathlib import Path
import os,sys,json,subprocess,argparse
R=Path('/mnt/large2/gjh/AFSEI');OUT=R/'results/tube_cg_euler_wave_20261031';base=R/'demo/active_tube_2d'
ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['pilot','full']);a=ap.parse_args()
assert json.loads((OUT/'euler_validation.json').read_text())['passed']
output=OUT/a.stage;assert not output.exists()
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',CUDA_HOME='/mnt/large2/gjh/conda-envs/afsei',TORCH_CUDA_ARCH_LIST='8.9',MAX_JOBS='4',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MPLCONFIGDIR=str(R/'.cache/matplotlib'))
cmd=[sys.executable,'-u',str(base/'run.py'),'--config',str(base/'configs/tube_cg_euler_wave_gpu.json'),'--output',str(output)]
if a.stage=='pilot':cmd+=['--end-ms','100']
with (OUT/(a.stage+'.log')).open('w') as log:
    child=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
record=dict(pid=child.pid,stage=a.stage,output=str(output),physical_gpu=1,shared_gpu=True,exclusive_performance_claim=False)
(OUT/(a.stage+'_launch.json')).write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
