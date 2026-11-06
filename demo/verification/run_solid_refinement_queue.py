"""Launch only on unoccupied GPUs; never interrupts an existing process."""
from pathlib import Path
import subprocess,json,time,os
from datetime import datetime,timezone
R=Path('/mnt/large2/gjh/AFSEI');O=R/'results/solid_space_campaign_20261025'
def write(stage,**data):
    p=O/'queue_status.json';q=p.with_suffix('.tmp')
    q.write_text(json.dumps(dict(stage=stage,updated_utc=datetime.now(timezone.utc).isoformat(),**data),indent=2));q.replace(p)
def available():
    q=subprocess.run(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],capture_output=True,text=True,check=True)
    mapping={int(s.split(',')[0]):s.split(',')[1].strip() for s in q.stdout.splitlines()}
    q=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],capture_output=True,text=True,check=True)
    occupied=set(q.stdout.splitlines())
    return [i for i in [0,1] if mapping[i] not in occupied]
def main():
    pending=['solid2_mac32','solid2_mac64'];running={};finished={}
    deadline=time.monotonic()+24*3600
    while pending or running:
        for name,p in list(running.items()):
            code=p.poll()
            if code is not None:finished[name]=code;running.pop(name)
        free=available()
        while pending and free:
            gpu=free.pop(0);name=pending.pop(0)
            log=(O/(name+'.log')).open('w');env=os.environ.copy()
            env['AFSEI_GPU']=str(gpu);env['CUDA_VISIBLE_DEVICES']=str(gpu)
            env['MPLCONFIGDIR']='/tmp/afsei-solid-refine-'+str(gpu)
            p=subprocess.Popen(['/mnt/large2/gjh/conda-envs/afsei/bin/python','-u',
               str(R/'demo/verification/run_solid_refinement_case.py'),'--config',str(O/(name+'.json')),
               '--output',str(O/name)],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
            log.close();running[name]=p
            (O/(name+'_launch.json')).write_text(json.dumps(dict(pid=p.pid,gpu=gpu),indent=2))
        write('running' if running else 'waiting-for-free-gpu',pending=pending,
              running={k:v.pid for k,v in running.items()},finished=finished)
        if pending and time.monotonic()>deadline:
            write('resource-timeout',pending=pending,finished=finished);return
        if pending or running:time.sleep(30)
    if all(code==0 for code in finished.values()):
        from analyze_solid_refinement import main as analyze
        analyze();write('completed',finished=finished)
    else:write('case-failed-no-retry',finished=finished)
if __name__=='__main__':
    try:main()
    except Exception as exc:write('queue-failed',error=str(exc));raise
