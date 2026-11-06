"""Launch two bounded diagnostic jobs on idle GPUs; no production cycles."""
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path('/mnt/large2/gjh/AFSEI')
out=root/'results/gao_connection_pair_20261026'
out.mkdir(exist_ok=False)
jobs=[]
for gpu,name in ((0,'legacy'),(1,'physical_cut')):
    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
        CUDA_HOME='/mnt/large2/gjh/conda-envs/afsei',
        CONDA_PREFIX='/mnt/large2/gjh/conda-envs/afsei', TORCH_CUDA_ARCH_LIST='8.9',
        MPLCONFIGDIR='/tmp/afsei-mpl-gao-pair')
    config=f'configs/gao2017/connection_pair_20261026/{name}.json'
    with (out/f'{name}.log').open('xb') as stream:
        p=subprocess.Popen([sys.executable,'-u','scripts/check_afsei_cut_boundary.py',config],
            cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
            start_new_session=True)
    jobs.append(dict(name=name,gpu=gpu,pid=p.pid,configuration=config))
(out/'jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
print(json.dumps(jobs))
