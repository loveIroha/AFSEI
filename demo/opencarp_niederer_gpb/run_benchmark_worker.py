"""Launch/track the official benchmark derivative without changing solver code."""
from pathlib import Path
import json,os,re,shutil,subprocess,sys,time,traceback

runtime=Path('/mnt/large2/gjh/opencarp-runtime/v20.0')
demo=Path('/mnt/large2/gjh/AFSEI/demo/opencarp_niederer_gpb')
out=Path('/mnt/large2/gjh/AFSEI/results/opencarp_gpb_h0p0625_20261030')
env=os.environ.copy()
env.update(PATH=str(runtime/'squashfs-root/usr/bin')+':'+env['PATH'],
    CARPUTILS_SETTINGS=str(runtime/'settings.yaml'),MPLCONFIGDIR=str(runtime/'matplotlib-cache'),
    OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
cmd=['/mnt/large2/gjh/venvs/opencarp/bin/python','-u',str(demo/'run.py'),
    '--dx','62.5','--np','16','--ID',str(out/'simulation_v2'),
    '--overwrite-behaviour','error','--plot',str(out/'diagonal_activation.png')]
if '--launch' in sys.argv:
    if out.exists():
        previous=json.loads((out/'run_status.json').read_text())
        if '--retry' not in sys.argv or previous.get('status')!='failed':
            raise SystemExit('Refusing to overwrite existing benchmark output: '+str(out))
        if (out/'attempt1_run.log').exists():raise SystemExit('Retry archive already exists')
        for name in ('run.log','run_status.json','worker.log','launch.json'):
            if (out/name).exists():shutil.copy2(out/name,out/('attempt1_'+name))
    else:out.mkdir(parents=True)
    for filename in ('experiment.json','run.py','nversion.par','original_run.py','original_nversion.par'):
        shutil.copy2(demo/filename,out/filename)
    with (out/'worker.log').open('w') as log:
        p=subprocess.Popen([sys.executable,'-u',__file__],stdout=log,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,start_new_session=True,env=env)
    (out/'launch.json').write_text(json.dumps({'pid':p.pid,'command':cmd,'cwd':str(demo),'mpi_ranks':16},indent=2))
    print(json.dumps({'pid':p.pid,'output':str(out)},indent=2));raise SystemExit(0)

started=time.time()
def status(**kwargs):
    (out/'run_status.json').write_text(json.dumps(dict(pid=os.getpid(),elapsed_s=time.time()-started,**kwargs),indent=2))
status(status='running',command=cmd)
try:
    with (out/'run.log').open('w',buffering=1) as log:
        process=subprocess.Popen(cmd,cwd=demo,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        status(status='running',command=cmd,process_pid=process.pid)
        for line in process.stdout:
            # Some official binaries embed CI repository URLs in build metadata.
            line=re.sub(r'https://[^\s/@]+:[^\s/@]+@git\.opencarp\.org/', 'https://git.opencarp.org/',line)
            log.write(line)
        rc=process.wait()
    status(status='completed' if rc==0 else 'failed',returncode=rc,command=cmd)
    if rc:raise SystemExit(rc)
except Exception as exc:
    status(status='failed',error=repr(exc),command=cmd)
    traceback.print_exc();raise
