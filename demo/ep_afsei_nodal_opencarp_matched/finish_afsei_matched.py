from pathlib import Path
import subprocess,time,os,json
here=Path(__file__).resolve().parent
root=Path('/mnt/large2/gjh/AFSEI/results/ep_afsei_nodal_opencarp_matched_20261030')
launch=json.loads((here/'launch.json').read_text())
while not (root/'report.json').exists():
    os.kill(launch['pid'],0)
    time.sleep(10)
env=os.environ.copy();env['MPLCONFIGDIR']='/mnt/large2/gjh/AFSEI/.cache/matplotlib'
Path(env['MPLCONFIGDIR']).mkdir(parents=True,exist_ok=True)
python='/mnt/large2/gjh/conda-envs/afsei/bin/python'
for script in ['analyze_afsei_matched.py','export_afsei_matched_vti.py']:
    subprocess.run([python,str(here/script)],env=env,check=True)
(root/'postprocessing_completed.json').write_text(json.dumps(dict(status='completed'),indent=2))
print('POSTPROCESSING_COMPLETED',flush=True)
