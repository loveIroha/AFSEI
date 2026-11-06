from pathlib import Path
import os,json,time,subprocess
ROOT=Path('/mnt/large2/gjh/AFSEI');pair=ROOT/'results/ep_euler_fem_pair_20261031';base=ROOT/'demo/ep_afsei_nodal_opencarp_matched'
while True:
    try:status=json.loads((pair/'pair_progress.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):time.sleep(10);continue
    if status['status']=='failed':raise RuntimeError(status)
    if status['status']=='completed':break
    time.sleep(10)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',CUDA_HOME='/mnt/large2/gjh/conda-envs/afsei',TORCH_CUDA_ARCH_LIST='8.9',MAX_JOBS='4',
    MPLCONFIGDIR=str(ROOT/'.cache/matplotlib'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
python='/mnt/large2/gjh/conda-envs/afsei/bin/python'
for script in ['profile_common_pde.py','analyze_pair.py']:subprocess.run([python,str(base/script)],env=env,check=True)
for method in ['torchcor_cg','nodal_afsei']:
    subprocess.run([python,str(base/'export_afsei_matched_vti.py'),'--source',str(pair/method)],env=env,check=True)
(pair/'postprocessing_completed.json').write_text(json.dumps(dict(status='completed'),indent=2));print('PAIR_POSTPROCESSING_COMPLETED',flush=True)
