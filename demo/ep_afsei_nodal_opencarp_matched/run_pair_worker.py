from pathlib import Path
import json,subprocess,os,time
ROOT=Path('/mnt/large2/gjh/AFSEI');base=ROOT/'demo/ep_afsei_nodal_opencarp_matched'
pair=ROOT/'results/ep_euler_fem_pair_20261031'
assert json.loads((pair/'preflight.json').read_text())['passed']
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',CUDA_HOME='/mnt/large2/gjh/conda-envs/afsei',
    TORCH_CUDA_ARCH_LIST='8.9',MAX_JOBS='4',MPLCONFIGDIR=str(ROOT/'.cache/matplotlib'))
python='/mnt/large2/gjh/conda-envs/afsei/bin/python'
for method in ['torchcor_cg','nodal_afsei']:
    config=base/(method+'.json');settings=json.loads(config.read_text())
    assert not Path(settings['output']).exists(),'Refusing to overwrite an existing experiment'
    with (pair/(method+'.log')).open('w') as log:
        p=subprocess.Popen([python,'-u',str(base/'run_afsei_matched.py'),'--config',str(config)],
            cwd=base,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        status=dict(status='running',method=method,pid=p.pid,physical_gpu=0,output=settings['output'])
        (pair/'pair_progress.json').write_text(json.dumps(status,indent=2))
        code=p.wait()
    if code:
        status.update(status='failed',returncode=code);(pair/'pair_progress.json').write_text(json.dumps(status,indent=2))
        raise RuntimeError(f'{method} exited {code}; see {pair}/{method}.log')
    assert json.loads((Path(settings['output'])/'report.json').read_text())['status']=='completed'
(pair/'pair_progress.json').write_text(json.dumps(dict(status='completed',physical_gpu=0),indent=2))
print('PAIR_SIMULATIONS_COMPLETED',flush=True)
