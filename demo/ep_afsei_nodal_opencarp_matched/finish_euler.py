from pathlib import Path
import json,os,time,subprocess
base=Path('/mnt/large2/gjh/AFSEI/demo/ep_afsei_nodal_opencarp_matched')
launch=json.loads((base/'euler_launch_20261031.json').read_text());out=Path(launch['output'])
while not (out/'report.json').exists():os.kill(launch['pid'],0);time.sleep(10)
env=os.environ.copy();env.update(MPLCONFIGDIR='/mnt/large2/gjh/AFSEI/.cache/matplotlib',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
    CUDA_VISIBLE_DEVICES='1',CUDA_HOME='/mnt/large2/gjh/conda-envs/afsei',TORCH_CUDA_ARCH_LIST='8.9',MAX_JOBS='4')
python='/mnt/large2/gjh/conda-envs/afsei/bin/python'
for script in ['analyze_euler.py','check_calcium_replay.py']:
    subprocess.run([python,str(base/script)],env=env,check=True)
subprocess.run([python,str(base/'export_afsei_matched_vti.py'),'--source',str(out)],env=env,check=True)
(out/'postprocessing_completed.json').write_text(json.dumps(dict(status='completed'),indent=2))
print('EULER_POSTPROCESSING_COMPLETED',flush=True)
