"""One-off CPU analysis after the existing 20 ms simulation exits.

This never launches/restarts a simulation and creates no recurring task.
"""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/mnt/large2/gjh/AFSEI')
CASE=ROOT/'results/gao_cut_cycle_validation_20261026/ejection_20ms_20261028'


def publish(data):
    target=CASE/'postprocess_result.json';temporary=target.with_suffix('.json.working')
    data['time_utc']=datetime.now(timezone.utc).isoformat()
    temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(target)


def watch():
    simulation=json.loads((CASE.parent/(CASE.name+'_job.json')).read_text())
    deadline=time.monotonic()+7200
    while not (CASE/'report.json').exists():
        try:os.kill(simulation['pid'],0)
        except ProcessLookupError:
            publish(dict(status='unable_to_analyze',reason='simulation exited without a final report'))
            return 1
        if time.monotonic()>deadline:
            publish(dict(status='not_finished',reason='two-hour observation limit; simulation left untouched'))
            return 1
        time.sleep(15)
    env=dict(os.environ,MPLCONFIGDIR='/tmp/afsei-mpl-gao-ejection')
    result=subprocess.run([sys.executable,'scripts/analyze_gao_ejection_interval.py',str(CASE)],cwd=ROOT,env=env)
    simulation_status=json.loads((CASE/'report.json').read_text())['status']
    passed=False
    if result.returncode==0:
        passed=json.loads((CASE/'interval_analysis.json').read_text())['interval_and_diagnostics_passed']
    publish(dict(status='completed' if result.returncode==0 else 'analysis_failed',
        simulation_status=simulation_status,interval_and_diagnostics_passed=passed,
        analysis_returncode=result.returncode))
    return result.returncode


def start():
    job=CASE/'postprocess_job.json'
    if job.exists():print(job.read_text());return
    if not (CASE.parent/(CASE.name+'_job.json')).is_file():
        raise SystemExit('existing simulation job record missing')
    with job.open('x') as record:
        with (CASE/'postprocess.log').open('xb') as stream:
            process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--watch'],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        data=dict(pid=process.pid,purpose='one-off CPU analysis of the existing simulation; no new FSI run',
                  output=str(CASE),log=str(CASE/'postprocess.log'))
        record.write(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data,indent=2))


if __name__=='__main__':
    if sys.argv[1:]==['--watch']:raise SystemExit(watch())
    if sys.argv[1:]:raise SystemExit('only --watch or no arguments supported')
    start()
