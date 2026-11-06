"""Launch two bounded near-preload probes on the two idle GPUs."""
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path('/mnt/large2/gjh/AFSEI')
out=root/'results/gao_preload_compatibility_20261026'
out.mkdir(exist_ok=False)
jobs=[]
for gpu,name in ((0,'legacy'),(1,'physical_cut')):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',
        CUDA_HOME='/mnt/large2/gjh/conda-envs/afsei',CONDA_PREFIX='/mnt/large2/gjh/conda-envs/afsei',
        TORCH_CUDA_ARCH_LIST='8.9',MPLCONFIGDIR='/tmp/afsei-mpl-gao-preload')
    config=f'configs/gao2017/connection_pair_20261026/preload_{name}.json'
    with (out/f'{name}.log').open('xb') as stream:
        process=subprocess.Popen([sys.executable,'-u','scripts/check_gao_preload_compatibility.py',config],
            cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    jobs.append(dict(case=name,gpu=gpu,pid=process.pid,configuration=config))
(out/'jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
print(json.dumps(jobs))
