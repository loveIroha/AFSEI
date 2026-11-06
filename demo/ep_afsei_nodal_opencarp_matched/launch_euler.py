from pathlib import Path
import json,os,subprocess
root=Path('/mnt/large2/gjh/AFSEI/demo/ep_afsei_nodal_opencarp_matched')
assert json.loads((root/'euler_validation_20261031.json').read_text())['passed']
cfg=json.loads((root/'euler1us_full.json').read_text());assert not Path(cfg['output']).exists(),'Refusing to overwrite output'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',CUDA_HOME='/mnt/large2/gjh/conda-envs/afsei',
    TORCH_CUDA_ARCH_LIST='8.9',MAX_JOBS='4',MPLCONFIGDIR='/mnt/large2/gjh/AFSEI/.cache/matplotlib')
with (root/'euler1us_full_20261031.log').open('w') as log:
    p=subprocess.Popen(['/mnt/large2/gjh/conda-envs/afsei/bin/python','-u','run_afsei_matched.py','--config','euler1us_full.json'],
        cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
report=dict(pid=p.pid,physical_gpu=1,configuration=str(root/'euler1us_full.json'),output=cfg['output'])
(root/'euler_launch_20261031.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
