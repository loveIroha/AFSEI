"""Wait for owned spatial jobs, summarize them, then run a fair 2D GPU pair.

Never kills a process, changes physical settings, relaxes tolerances, or retries
a failed simulation. The performance stage waits for unoccupied GPUs.
"""
from pathlib import Path
import json,subprocess,time,os,sys
from datetime import datetime,timezone

ROOT=Path('/mnt/large2/gjh/AFSEI')
OUT=ROOT/'results/reliability_campaign_20261025'

def status(stage,**values):
    data=dict(stage=stage,updated_utc=datetime.now(timezone.utc).isoformat(),**values)
    p=OUT/'followup_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)
    print(json.dumps(data),flush=True)

def running(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False

def main():
    jobs=[('mac48','mac48_launch.json'),('mac64_budget16m','mac64_budget16m_launch.json')]
    status('waiting-spatial-results')
    while any(running(json.loads((OUT/f).read_text())['pid']) for _,f in jobs):time.sleep(30)
    reports={name:json.loads((OUT/name/'report.json').read_text()) if (OUT/name/'report.json').exists()
             else {'status':'interrupted-without-report'} for name,_ in jobs}
    status('summarizing-spatial',simulation_status={k:v['status'] for k,v in reports.items()})
    if all((OUT/name/'history.json').exists() for name,_ in jobs):
        from compare_spatial import main as summarize
        summarize()
    status('waiting-exclusive-performance-resources')
    deadline=time.monotonic()+24*3600
    while True:
        q=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True)
        if q.returncode==0 and not q.stdout.strip():break
        if time.monotonic()>deadline:
            status('performance-not-started-resource-timeout');return
        time.sleep(30)
    source=ROOT/'demo/active_tube_2d/configs/tube_plane_strain_check_gpu.json'
    c=json.loads(source.read_text());c['time']['end_time_s']=.045
    # Existing stable 45 ms configuration, same equations/mesh/dt/tolerances.
    assert c['time']['fsi_dt_s']==1e-5
    config=OUT/'performance_2d.json';config.write_text(json.dumps(c,indent=2))
    env=os.environ.copy();env['AFSEI_GPU']='0';env['CUDA_VISIBLE_DEVICES']='0';env['MPLCONFIGDIR']='/tmp/afsei-campaign-benchmark'
    status('running-matched-performance',gpu=0,repeats=2,window_s=[0,.045],
           scope='AFSEI optimized versus its original GPU implementation; not external-code speedup')
    with (OUT/'performance_2d.log').open('w') as log:
        p=subprocess.run([sys.executable,str(ROOT/'demo/active_tube_2d/benchmark.py'),'--config',str(config),
                          '--output',str(OUT/'performance_2d'),'--repeats','2','--end-ms','45'],
                         cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if p.returncode:
        status('performance-failed-no-speedup-claim',returncode=p.returncode);return
    bench=json.loads((OUT/'performance_2d/benchmark.json').read_text())
    status('scheduled-stages-completed',performance_fields_passed=bench['passed'],
           implementation_speedup=bench['median_speedup'],
           spatial_status={k:v['status'] for k,v in reports.items()},
           full_coupled_external_reference_complete=False,
           solid_mesh_convergence_complete=False)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        status('followup-failed',error=str(exc));raise
