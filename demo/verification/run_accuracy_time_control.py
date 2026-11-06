"""Only if spatial criteria pass, test a fresh-rest 20us full trajectory.

Do not spend a second long trajectory on a spatial configuration already known
to fail the declared accuracy criteria. No tolerance or model is adjusted.
"""
from pathlib import Path
import json,os,subprocess,sys,time
from run_accuracy_queue import gpu_idle
root=Path('/mnt/large2/gjh/AFSEI/results/accuracy_performance_20261025')
status=root/'time_control_status.json'
def save(value):
    p=status.with_suffix('.tmp');p.write_text(json.dumps(value,indent=2));p.replace(status)
started=time.monotonic()
try:
    while not (root/'analysis/report.json').exists():
        save(dict(stage='waiting-for-spatial-results',full_space_time_convergence_established=False))
        if time.monotonic()-started>24*3600:raise TimeoutError('spatial report not ready')
        time.sleep(30)
    report=json.loads((root/'analysis/report.json').read_text())
    if not report['fixed_dt_spatial_acceptance_passed']:
        save(dict(stage='not-started-spatial-criteria-not-passed',full_space_time_convergence_established=False,
            reason='Review spatial errors before another long run; current numerical model and tolerances retained.'))
        sys.exit(0)
    while not gpu_idle(0):
        if time.monotonic()-started>24*3600:raise TimeoutError('GPU0 remained busy')
        time.sleep(30)
    c=json.loads((root/'solid3_mac128/config.json').read_text())
    c['time'].update(fsi_dt_s=2e-5,ep_dt_s=2e-5)
    c['output'].update(log_every=250,field_every=500,checkpoint_every=1000)
    cfg=Path('demo/verification/configs/accuracy_performance_20261025/solid3_mac128_dt20us.json')
    cfg.write_text(json.dumps(c,indent=2))
    save(dict(stage='running-from-rest-time-control',dt_s=2e-5,end_time_s=.3,full_space_time_convergence_established=False))
    subprocess.run([sys.executable,'-u','demo/verification/run_accuracy_queue.py','--gpu','0','--root',str(root/'time_control'),
        '--cases','solid3_mac128_dt20us'],check=True)
    subprocess.run([sys.executable,'-u','demo/verification/analyze_accuracy_time_control.py'],check=True)
    save(dict(stage='completed',report=str(root/'time_control/comparison.json')))
except Exception as e:
    save(dict(stage='failed',error=repr(e),parameters_relaxed=False,full_space_time_convergence_established=False));raise
