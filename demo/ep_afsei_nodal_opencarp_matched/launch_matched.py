from pathlib import Path
import os,subprocess,json
root=Path('/mnt/large2/gjh/AFSEI/demo/ep_afsei_nodal_opencarp_matched')
smoke=Path('/mnt/large2/gjh/AFSEI/results/ep_afsei_nodal_opencarp_matched_20261030_smoke')
r=json.loads((smoke/'report.json').read_text())
assert r['status']=='completed' and r['max_cell_guard']==0 and r['max_true_residual_ratio']<=1
target=Path(json.loads((root/'afsei_matched_full.json').read_text())['output'])
assert not target.exists(),'Refusing to overwrite an existing experiment'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',CUDA_HOME='/mnt/large2/gjh/conda-envs/afsei',
    TORCH_CUDA_ARCH_LIST='8.9',MAX_JOBS='4',MPLCONFIGDIR='/mnt/large2/gjh/AFSEI/.cache/matplotlib')
with (root/'full.log').open('w') as log:
    p=subprocess.Popen(['/mnt/large2/gjh/conda-envs/afsei/bin/python','-u','run_afsei_matched.py','--config','afsei_matched_full.json'],
        cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
(root/'launch.json').write_text(json.dumps(dict(pid=p.pid,output=str(target),physical_gpu=1),indent=2))
print((root/'launch.json').read_text())
